# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for reference temperature/conductivity source selection + QC (#131 B1).

Covers the plausibility QC, the auto/explicit/constant resolution chain, the
NetCDF load path, measured salinity, the L1 adapter, the perturb T_source
threading, and the CLI parse -> merge -> kwargs flow.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import netCDF4
import numpy as np
import pytest

from odas_tpw.rsi.helpers import (
    load_channels,
    reference_temperature_qc,
    resolve_conductivity_channel,
    resolve_measured_salinity,
    resolve_temperature_channel,
    temperature_candidates,
)

REPO_ROOT = Path(__file__).parent.parent

RAILED_T = 58.46  # CasperWest corpus-wide railed T1 value [degC]
DEAD_T = -17.09  # dead-sensor value [degC]


# ---------------------------------------------------------------------------
# reference_temperature_qc
# ---------------------------------------------------------------------------


class TestReferenceTemperatureQC:
    def test_normal_ocean_passes(self):
        rng = np.random.default_rng(0)
        T = 12.0 + 2.0 * rng.standard_normal(500)
        assert reference_temperature_qc(T) is None

    def test_warm_pool_passes(self):
        rng = np.random.default_rng(1)
        T = 31.0 + 0.5 * rng.standard_normal(500)
        assert reference_temperature_qc(T) is None

    def test_polar_passes(self):
        """-1.9 degC (polar surface water) is above TEMP_QC_MIN=-3."""
        rng = np.random.default_rng(2)
        T = -1.9 + 0.05 * rng.standard_normal(500)
        assert reference_temperature_qc(T) is None

    def test_railed_58_46_fails(self):
        T = np.full(500, RAILED_T)
        reason = reference_temperature_qc(T)
        assert reason is not None
        assert "58.5" in reason and "outside plausible" in reason

    def test_dead_negative_fails(self):
        T = np.full(500, DEAD_T)
        reason = reference_temperature_qc(T)
        assert reason is not None
        assert "outside plausible" in reason

    def test_drift_with_in_range_median_fails(self):
        """A 27->58 degC drift over the last fifth of the record: the median
        stays in range but >10% of samples are outside — must still fail
        (the drift case)."""
        T = np.concatenate([np.full(400, 27.0), np.linspace(41.0, 58.0, 100)])
        assert -3.0 <= np.median(T) <= 40.0
        reason = reference_temperature_qc(T)
        assert reason is not None
        assert "% of samples outside" in reason

    def test_mostly_nan_fails(self):
        rng = np.random.default_rng(3)
        T = 12.0 + rng.standard_normal(500)
        T[: int(0.7 * T.size)] = np.nan
        reason = reference_temperature_qc(T)
        assert reason is not None
        assert "non-finite" in reason

    def test_constant_in_range_fails(self):
        T = np.full(500, 16.0)
        reason = reference_temperature_qc(T)
        assert reason is not None
        assert "constant" in reason

    def test_constant_below_n_min_passes(self):
        """Short records (n < n_min) skip the constant check."""
        T = np.full(50, 16.0)
        assert reference_temperature_qc(T) is None

    def test_in_water_only_when_pressure_given(self):
        """17% deck samples at -20 degC (polar winter) must not fail a
        healthy sensor when pressure identifies them as out of water."""
        n = 1000
        n_deck = 170
        T = np.concatenate([np.full(n_deck, -20.0), np.full(n - n_deck, 10.0)])
        T[n_deck:] += 0.01 * np.random.default_rng(4).standard_normal(n - n_deck)
        P = np.concatenate([np.full(n_deck, 0.1), np.linspace(1.0, 50.0, n - n_deck)])
        assert reference_temperature_qc(T) is not None  # full channel fails
        assert reference_temperature_qc(T, pressure=P) is None  # in-water passes

    def test_full_channel_fallback_when_never_in_water(self):
        """A bench recording (P ~ 0.3 dbar throughout) falls back to the
        full channel instead of QC-ing an empty subset."""
        rng = np.random.default_rng(5)
        T = 16.0 + 0.01 * rng.standard_normal(500)
        P = np.full(500, 0.3)
        assert reference_temperature_qc(T, pressure=P) is None

    def test_empty_fails(self):
        assert reference_temperature_qc(np.array([])) is not None


# ---------------------------------------------------------------------------
# resolve_temperature_channel / temperature_candidates
# ---------------------------------------------------------------------------


def _mk_channels(n_slow=500, *, railed_t1=False, with_t2=True, bare_t=False, jac_t=True):
    rng = np.random.default_rng(7)
    ch = {
        "P": np.linspace(2.0, 50.0, n_slow),
        "sh1": rng.standard_normal(n_slow * 8),  # fast — must be ignored
        "T1_dT1": rng.standard_normal(n_slow * 8),  # fast — must be ignored
    }
    ch["T1"] = (
        np.full(n_slow, RAILED_T) if railed_t1 else 20.0 + rng.standard_normal(n_slow)
    )
    if with_t2:
        ch["T2"] = 19.0 + rng.standard_normal(n_slow)
    if bare_t:
        ch["T"] = 18.0 + rng.standard_normal(n_slow)
    if jac_t:
        ch["JAC_T"] = 21.0 + rng.standard_normal(n_slow)
    return ch


class TestResolveTemperatureChannel:
    def test_auto_picks_healthy_t1_first(self):
        ch = _mk_channels()
        name, reason = resolve_temperature_channel(ch, 500, "auto", pressure=ch["P"])
        assert name == "T1"
        assert reason is None

    def test_auto_falls_to_t2_on_railed_t1(self):
        ch = _mk_channels(railed_t1=True)
        with pytest.warns(UserWarning, match="skipping reference temperature channel T1"):
            name, reason = resolve_temperature_channel(ch, 500, "auto", pressure=ch["P"])
        assert name == "T2"
        assert reason is None

    def test_candidate_order_numbered_then_bare_then_jac(self):
        ch = _mk_channels(bare_t=True)
        assert temperature_candidates(ch, 500) == ["T1", "T2", "T", "JAC_T"]

    def test_bare_t_only_is_used(self):
        """A per-profile store with only a slow bare T resolves to it."""
        ch = _mk_channels(with_t2=False, bare_t=True, jac_t=False)
        del ch["T1"]
        name, reason = resolve_temperature_channel(ch, 500, "auto", pressure=ch["P"])
        assert name == "T"
        assert reason is None

    def test_all_fail_raises_with_reasons(self):
        ch = _mk_channels(railed_t1=True, with_t2=False, jac_t=False)
        with (
            pytest.warns(UserWarning),
            pytest.raises(ValueError, match="no plausible reference temperature") as exc,
        ):
            resolve_temperature_channel(ch, 500, "auto", pressure=ch["P"])
        # Error must list the candidate, its reason, and the remedy
        assert "T1" in str(exc.value)
        assert "outside plausible" in str(exc.value)
        assert "--temperature" in str(exc.value)

    def test_no_candidates_raises(self):
        ch = {"P": np.linspace(2, 50, 500)}
        with pytest.raises(ValueError, match="no slow temperature channel"):
            resolve_temperature_channel(ch, 500, "auto")

    def test_explicit_missing_raises(self):
        ch = _mk_channels()
        with pytest.raises(ValueError, match="not found"):
            resolve_temperature_channel(ch, 500, "T9")

    def test_explicit_fast_length_raises(self):
        ch = _mk_channels()
        with pytest.raises(ValueError, match="not a slow channel"):
            resolve_temperature_channel(ch, 500, "T1_dT1")

    def test_explicit_qc_fail_proceeds_with_warning(self):
        ch = _mk_channels(railed_t1=True)
        with pytest.warns(UserWarning, match="fails plausibility QC"):
            name, reason = resolve_temperature_channel(ch, 500, "T1", pressure=ch["P"])
        assert name == "T1"
        assert reason is not None

    def test_explicit_hotel_style_channel(self):
        """An arbitrary slow channel name (e.g. a hotel temperature) works."""
        ch = _mk_channels()
        ch["sbe_temperature"] = np.full(500, 15.0) + 0.1 * np.arange(500) / 500
        name, reason = resolve_temperature_channel(ch, 500, "sbe_temperature")
        assert name == "sbe_temperature"
        assert reason is None


class TestResolveConductivityChannel:
    def test_auto_finds_jac_c(self):
        ch = {"JAC_C": np.full(100, 5.0)}
        assert resolve_conductivity_channel(ch, 100, "auto") == "JAC_C"

    def test_auto_absent_returns_none(self):
        assert resolve_conductivity_channel({}, 100, "auto") is None

    def test_auto_fast_length_jac_c_returns_none(self):
        ch = {"JAC_C": np.full(800, 5.0)}
        assert resolve_conductivity_channel(ch, 100, "auto") is None

    def test_explicit_missing_raises(self):
        with pytest.raises(ValueError, match="not found"):
            resolve_conductivity_channel({}, 100, "hotel_C")

    def test_explicit_wrong_length_raises(self):
        ch = {"hotel_C": np.full(800, 5.0)}
        with pytest.raises(ValueError, match="not a slow channel"):
            resolve_conductivity_channel(ch, 100, "hotel_C")


# ---------------------------------------------------------------------------
# Synthetic per-profile NetCDF builder
# ---------------------------------------------------------------------------


def _write_nc(
    path: Path,
    *,
    n_fast=1024,
    n_slow=128,
    t1_railed=False,
    with_t2=False,
    with_bare_t=False,
    with_jac=False,
) -> Path:
    """Minimal NC for load_channels/_compute_epsilon (railed/healthy variants)."""
    fs_fast = 512.0
    fs_slow = 64.0
    rng = np.random.default_rng(0)

    ds = netCDF4.Dataset(str(path), "w", format="NETCDF4")
    try:
        ds.createDimension("time_fast", n_fast)
        ds.createDimension("time_slow", n_slow)
        ds.fs_fast = fs_fast
        ds.fs_slow = fs_slow

        tf = ds.createVariable("t_fast", "f8", ("time_fast",))
        tf[:] = np.arange(n_fast) / fs_fast
        ts = ds.createVariable("t_slow", "f8", ("time_slow",))
        ts[:] = np.arange(n_slow) / fs_slow

        p = ds.createVariable("P", "f8", ("time_slow",))
        p[:] = np.linspace(5.0, 50.0, n_slow)
        t1 = ds.createVariable("T1", "f8", ("time_slow",))
        t1[:] = np.full(n_slow, RAILED_T) if t1_railed else np.linspace(20.0, 5.0, n_slow)
        if with_t2:
            t2 = ds.createVariable("T2", "f8", ("time_slow",))
            t2[:] = np.linspace(19.5, 5.5, n_slow)
        if with_bare_t:
            t = ds.createVariable("T", "f8", ("time_slow",))
            t[:] = np.linspace(18.0, 6.0, n_slow)
        if with_jac:
            jt = ds.createVariable("JAC_T", "f8", ("time_slow",))
            jt[:] = np.linspace(20.2, 5.2, n_slow)
            jc = ds.createVariable("JAC_C", "f8", ("time_slow",))
            jc[:] = np.linspace(4.8, 3.9, n_slow)

        sh1 = ds.createVariable("sh1", "f8", ("time_fast",))
        sh1[:] = rng.standard_normal(n_fast) * 0.05
        ax = ds.createVariable("Ax", "f8", ("time_fast",))
        ax[:] = rng.standard_normal(n_fast) * 0.005
    finally:
        ds.close()
    return path


class TestLoadChannelsNc:
    def test_railed_t1_falls_to_t2_with_metadata(self, tmp_path):
        nc = _write_nc(tmp_path / "railed.nc", t1_railed=True, with_t2=True)
        with pytest.warns(UserWarning, match="skipping reference temperature channel T1"):
            data = load_channels(nc)
        assert data["metadata"]["temperature_source"] == "T2"
        assert data["metadata"]["temperature_qc"] == "pass"
        np.testing.assert_allclose(data["T"], np.linspace(19.5, 5.5, 128))

    def test_healthy_t1_matches_legacy_selection(self, tmp_path):
        nc = _write_nc(tmp_path / "healthy.nc", with_t2=True)
        data = load_channels(nc)
        assert data["metadata"]["temperature_source"] == "T1"
        np.testing.assert_array_equal(data["T"], np.linspace(20.0, 5.0, 128))

    def test_bare_t_after_numbered(self, tmp_path):
        """Healthy T2 wins over bare T (numbered before bare in the chain)."""
        nc = _write_nc(tmp_path / "bare.nc", t1_railed=True, with_t2=True, with_bare_t=True)
        with pytest.warns(UserWarning):
            data = load_channels(nc)
        assert data["metadata"]["temperature_source"] == "T2"

    def test_conductivity_auto_and_metadata(self, tmp_path):
        nc = _write_nc(tmp_path / "jac.nc", with_jac=True)
        data = load_channels(nc)
        assert data["metadata"]["conductivity_source"] == "JAC_C"
        assert "C" in data and "JAC_T" in data

    def test_constant_reference_temperature(self, tmp_path):
        nc = _write_nc(tmp_path / "const.nc", t1_railed=True)
        data = load_channels(nc, temperature_name=12.0)
        assert data["metadata"]["temperature_source"] == "constant:12"
        assert data["metadata"]["temperature_qc"] == "pass"
        np.testing.assert_array_equal(data["T"], np.full(128, 12.0))

    def test_constant_out_of_range_warns_and_records_qc(self, tmp_path):
        """An implausible constant reference (e.g. 99 degC) must not claim
        temperature_qc='pass' — the escape hatch is explicit, but the
        provenance stays honest and a warning fires."""
        nc = _write_nc(tmp_path / "const99.nc", t1_railed=True)
        with pytest.warns(UserWarning, match="outside plausible ocean range"):
            data = load_channels(nc, temperature_name=99.0)
        assert data["metadata"]["temperature_source"] == "constant:99"
        assert "outside plausible" in data["metadata"]["temperature_qc"]

    def test_explicit_conductivity_missing_raises(self, tmp_path):
        nc = _write_nc(tmp_path / "noc.nc")
        with pytest.raises(ValueError, match="conductivity channel"):
            load_channels(nc, conductivity_name="hotel_C")


# ---------------------------------------------------------------------------
# Measured salinity
# ---------------------------------------------------------------------------


class TestMeasuredSalinity:
    def _data(self, *, jac_t_railed=False, drop_c=False, drop_jac_t=False):
        n = 200
        P = np.linspace(2.0, 60.0, n)
        T1 = np.linspace(20.0, 6.0, n)
        JAC_T = np.linspace(20.1, 6.1, n)
        if jac_t_railed:
            JAC_T = np.full(n, RAILED_T)
        C = np.linspace(4.9, 3.8, n)
        data = {
            "P": P,
            "T": T1,
            "metadata": {"source": "synthetic", "temperature_source": "T1"},
        }
        if not drop_c:
            data["C"] = C
            data["metadata"]["conductivity_source"] = "JAC_C"
        if not drop_jac_t:
            data["JAC_T"] = JAC_T
        return data

    def test_matches_direct_gsw(self):
        import gsw

        data = self._data()
        sal = resolve_measured_salinity(data)
        expected = gsw.SP_from_C(data["C"], data["JAC_T"], data["P"])
        np.testing.assert_allclose(sal, expected)
        assert data["metadata"]["salinity_pair_temperature"] == "JAC_T"

    def test_missing_conductivity_warns_and_returns_none(self):
        data = self._data(drop_c=True)
        with pytest.warns(UserWarning, match="no conductivity channel"):
            assert resolve_measured_salinity(data) is None

    def test_railed_jac_t_pairs_with_reference(self):
        import gsw

        data = self._data(jac_t_railed=True)
        with pytest.warns(UserWarning, match="pairing JAC_C with the reference"):
            sal = resolve_measured_salinity(data)
        expected = gsw.SP_from_C(data["C"], data["T"], data["P"])
        np.testing.assert_allclose(sal, expected)
        assert data["metadata"]["salinity_pair_temperature"] == "T1"

    def test_abandoned_measured_salinity_records_no_pair(self):
        """All-NaN conductivity abandons measured salinity (visc35 fallback);
        the product must NOT carry a salinity_pair_temperature claiming a
        pairing that never produced a usable salinity."""
        data = self._data()
        data["C"] = np.full_like(data["C"], np.nan)
        with pytest.warns(UserWarning, match="entirely non-finite"):
            sal = resolve_measured_salinity(data)
        assert sal is None
        assert "salinity_pair_temperature" not in data["metadata"]

    def test_nan_gaps_median_filled(self):
        data = self._data()
        data["C"] = np.asarray(data["C"], dtype=np.float64).copy()
        data["C"][:10] = np.nan
        with pytest.warns(UserWarning, match="non-finite sample"):
            sal = resolve_measured_salinity(data)
        assert np.all(np.isfinite(sal))

    def test_compute_epsilon_measured_salinity(self, tmp_path):
        """salinity='measured' end-to-end: runs, and records the pairing."""
        from odas_tpw.rsi.dissipation import _compute_epsilon

        nc = _write_nc(
            tmp_path / "meas.nc", n_fast=8192, n_slow=1024, with_jac=True
        )
        results = _compute_epsilon(nc, goodman=False, fft_length=512, salinity="measured")
        assert results
        ds = results[0]
        assert ds.attrs["conductivity_source"] == "JAC_C"
        assert ds.attrs["salinity_pair_temperature"] == "JAC_T"
        assert np.all(np.isfinite(ds["nu"].values))

    def test_compute_epsilon_measured_without_c_falls_back(self, tmp_path):
        """No conductivity: warn + visc35 path (nu == visc35(T_mean))."""
        from odas_tpw.rsi.dissipation import _compute_epsilon
        from odas_tpw.scor160.ocean import visc35

        nc = _write_nc(tmp_path / "noc.nc", n_fast=8192, n_slow=1024)
        with pytest.warns(UserWarning, match="no conductivity channel"):
            results = _compute_epsilon(
                nc, goodman=False, fft_length=512, salinity="measured"
            )
        assert results
        ds = results[0]
        np.testing.assert_allclose(ds["nu"].values, visc35(ds["T_mean"].values))

    def test_compute_epsilon_rejects_other_strings(self, tmp_path):
        from odas_tpw.rsi.dissipation import _compute_epsilon

        nc = _write_nc(tmp_path / "hotel.nc")
        with pytest.raises(ValueError, match="not resolved at this layer"):
            _compute_epsilon(nc, goodman=False, salinity="hotel:salinity")

    def test_compute_chi_rejects_other_strings(self, tmp_path):
        from odas_tpw.rsi.chi_io import _compute_chi

        nc = _write_nc(tmp_path / "hotel_chi.nc")
        with pytest.raises(ValueError, match="not resolved at this layer"):
            _compute_chi(nc, salinity="hotel:salinity")


# ---------------------------------------------------------------------------
# End-to-end epsilon: railed T1 -> T2 drives viscosity (the B1 physics)
# ---------------------------------------------------------------------------


class TestComputeEpsilonReferenceTemperature:
    def test_railed_t1_healthy_t2_nu_from_t2(self, tmp_path):
        from odas_tpw.rsi.dissipation import _compute_epsilon
        from odas_tpw.scor160.ocean import visc35

        nc = _write_nc(
            tmp_path / "railed.nc",
            n_fast=8192,
            n_slow=1024,
            t1_railed=True,
            with_t2=True,
        )
        with pytest.warns(UserWarning, match="skipping reference temperature channel T1"):
            results = _compute_epsilon(nc, goodman=False, fft_length=512)
        assert results
        for ds in results:
            assert ds.attrs["temperature_source"] == "T2"
            T_mean = ds["T_mean"].values
            # T2's range, not the railed 58.46
            assert np.all(T_mean < 30.0)
            np.testing.assert_allclose(ds["nu"].values, visc35(T_mean))

    def test_healthy_auto_bit_identical_to_explicit_t1(self, tmp_path):
        from odas_tpw.rsi.dissipation import _compute_epsilon

        nc = _write_nc(tmp_path / "healthy.nc", n_fast=8192, n_slow=1024, with_t2=True)
        res_auto = _compute_epsilon(nc, goodman=False, fft_length=512)
        res_t1 = _compute_epsilon(nc, goodman=False, fft_length=512, temperature="T1")
        assert len(res_auto) == len(res_t1) > 0
        for a, b in zip(res_auto, res_t1):
            np.testing.assert_array_equal(a["epsilon"].values, b["epsilon"].values)
            np.testing.assert_array_equal(a["nu"].values, b["nu"].values)
            np.testing.assert_array_equal(a["T_mean"].values, b["T_mean"].values)

    def test_constant_reference_temperature(self, tmp_path):
        from odas_tpw.rsi.dissipation import _compute_epsilon
        from odas_tpw.scor160.ocean import visc35

        nc = _write_nc(tmp_path / "const.nc", n_fast=8192, n_slow=1024, t1_railed=True)
        results = _compute_epsilon(nc, goodman=False, fft_length=512, temperature=12.0)
        assert results
        for ds in results:
            assert ds.attrs["temperature_source"] == "constant:12"
            np.testing.assert_allclose(ds["T_mean"].values, 12.0)
            np.testing.assert_allclose(ds["nu"].values, visc35(np.full(1, 12.0))[0])


# ---------------------------------------------------------------------------
# Adapter: railed T1 + healthy T2 -> L1.temp from T2
# ---------------------------------------------------------------------------


class _FakePFile:
    """Minimal PFile stand-in for pfile_to_l1data."""

    def __init__(self, *, railed_t1=False, n_slow=400, fs_fast=512.0, fs_slow=64.0):
        ratio = round(fs_fast / fs_slow)
        n_fast = n_slow * ratio
        rng = np.random.default_rng(11)
        self.fs_fast = fs_fast
        self.fs_slow = fs_slow
        self.t_slow = np.arange(n_slow) / fs_slow
        self.t_fast = np.arange(n_fast) / fs_fast
        self.start_time = datetime(2025, 1, 15)
        self.channels = {
            "P": np.linspace(5.0, 50.0, n_slow),
            "T1": np.full(n_slow, RAILED_T)
            if railed_t1
            else np.linspace(20.0, 5.0, n_slow),
            "T2": np.linspace(19.0, 6.0, n_slow),
            "sh1": rng.standard_normal(n_fast) * 0.1,
            "Ax": rng.standard_normal(n_fast) * 0.01,
        }
        self.channel_info = {n: {"type": "x"} for n in self.channels}
        self._fast_channels = {"sh1", "Ax"}
        self.config = {
            "instrument_info": {"vehicle": "VMP"},
            "channels": [{"name": n} for n in self.channels],
        }


class TestAdapterReferenceTemperature:
    def test_railed_t1_uses_t2(self):
        from odas_tpw.rsi.adapter import pfile_to_l1data

        pf = _FakePFile(railed_t1=True)
        with pytest.warns(UserWarning, match="skipping reference temperature channel T1"):
            l1 = pfile_to_l1data(pf, speed=0.5)
        # T2 spans 19 -> 6, not the railed 58.46
        assert l1.temp.max() < 30.0
        np.testing.assert_allclose(l1.temp[0], 19.0, atol=0.1)

    def test_healthy_t1_unchanged(self):
        from odas_tpw.rsi.adapter import pfile_to_l1data

        pf = _FakePFile()
        l1 = pfile_to_l1data(pf, speed=0.5)
        np.testing.assert_allclose(l1.temp[0], 20.0, atol=0.1)

    def test_constant_temperature(self):
        from odas_tpw.rsi.adapter import pfile_to_l1data

        pf = _FakePFile(railed_t1=True)
        l1 = pfile_to_l1data(pf, speed=0.5, temperature=9.5)
        np.testing.assert_allclose(l1.temp, 9.5)


# ---------------------------------------------------------------------------
# perturb: epsilon.T_source threads into the loaders/computations
# ---------------------------------------------------------------------------


class TestPerturbTSourceThreading:
    @patch("odas_tpw.rsi.dissipation._compute_epsilon", return_value=[])
    @patch("odas_tpw.rsi.chi_io._load_therm_channels")
    @patch("odas_tpw.rsi.profile.extract_profiles")
    @patch("odas_tpw.rsi.profile.get_profiles", return_value=[(0, 99)])
    @patch("odas_tpw.rsi.profile._smooth_fall_rate", return_value=np.zeros(100))
    @patch("odas_tpw.rsi.p_file.PFile")
    def test_t_source_reaches_loader_and_epsilon(
        self,
        mock_pfile_cls,
        mock_smooth,
        mock_get_prof,
        mock_extract,
        mock_load_therm,
        mock_eps,
        tmp_path,
    ):
        from odas_tpw.perturb.pipeline import process_file

        mock_pf = MagicMock()
        mock_pf.channels = {"P": np.linspace(0, 50, 100), "T1": np.zeros(100)}
        mock_pf.t_slow = np.linspace(0, 50, 100)
        mock_pf.fs_slow = 64.0
        mock_pf.fs_fast = 512.0
        mock_pf.config = {"instrument_info": {"vehicle": "vmp"}}
        mock_pfile_cls.return_value = mock_pf

        prof_path = str(tmp_path / "fake_prof.nc")
        mock_extract.return_value = ([Path(prof_path)], [{}])
        mock_load_therm.return_value = {"marker": True}

        config = {
            "files": {"output_root": str(tmp_path)},
            "profiles": {},
            "epsilon": {"T_source": "T2"},
            "chi": {"enable": True},
            "ctd": {"enable": False},
            "ct": {"align": False},
            "fp07": {"calibrate": False},
            "stratification": {"enable": False},
        }
        output_dirs = {
            "profiles": tmp_path / "profiles",
            "diss": tmp_path / "diss",
            "chi": tmp_path / "chi",
        }
        for d in output_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        process_file(tmp_path / "test.p", config, None, output_dirs)

        # The pre-load for chi must carry the configured temperature source
        mock_load_therm.assert_called_once()
        assert mock_load_therm.call_args.kwargs["temperature_name"] == "T2"

        # _compute_epsilon gets temperature=T_source; the raw config key is
        # stripped from the kwargs splat
        mock_eps.assert_called_once()
        eps_kwargs = mock_eps.call_args.kwargs
        assert eps_kwargs["temperature"] == "T2"
        assert "T_source" not in eps_kwargs
        assert "T1_norm" not in eps_kwargs and "T2_norm" not in eps_kwargs

    def test_t1_norm_rejected_by_validate_config(self):
        from odas_tpw.perturb.config import validate_config

        with pytest.raises(ValueError, match="T1_norm"):
            validate_config({"epsilon": {"T1_norm": 1.0}})

    def test_example_config_still_validates(self):
        from odas_tpw.perturb.config import load_config

        cfg_path = REPO_ROOT / "examples" / "arcterx_2025_interior" / "perturb.yaml"
        config = load_config(cfg_path)
        assert config["epsilon"].get("T_source") is None
        assert "T1_norm" not in config["epsilon"]

    def test_template_has_no_norm_keys(self, tmp_path):
        from odas_tpw.perturb.config import generate_template, load_config

        p = generate_template(tmp_path / "perturb.yaml")
        text = p.read_text()
        assert "T1_norm" not in text and "T2_norm" not in text
        assert "T_source" in text
        load_config(p)  # template must validate


# ---------------------------------------------------------------------------
# CLI: parse -> _merge_for_section -> kwargs (a dropped mapping fails here)
# ---------------------------------------------------------------------------


def _parse(subparser_adder, argv):
    from odas_tpw.rsi import cli

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default=None)
    sub = parser.add_subparsers(dest="command")
    getattr(cli, subparser_adder)(sub)
    return parser.parse_args(argv)


class TestCliTemperatureFlags:
    def test_eps_flags_reach_epsilon_kwargs(self):
        from odas_tpw.rsi.cli import _merge_for_section

        args = _parse(
            "_add_eps_parser",
            [
                "eps",
                "f.p",
                "-o",
                "out",
                "--temperature",
                "T2",
                "--conductivity",
                "JAC_C",
                "--salinity",
                "measured",
            ],
        )
        merged = _merge_for_section(args, "epsilon")
        assert merged["temperature"] == "T2"
        assert merged["conductivity"] == "JAC_C"
        assert merged["salinity"] == "measured"

    def test_eps_numeric_temperature_and_salinity(self):
        from odas_tpw.rsi.cli import _merge_for_section

        args = _parse(
            "_add_eps_parser",
            ["eps", "f.p", "-o", "out", "--temperature", "12.5", "--salinity", "34.5"],
        )
        merged = _merge_for_section(args, "epsilon")
        assert merged["temperature"] == 12.5
        assert merged["salinity"] == 34.5

    def test_eps_defaults_are_auto(self):
        from odas_tpw.rsi.cli import _merge_for_section

        args = _parse("_add_eps_parser", ["eps", "f.p", "-o", "out"])
        merged = _merge_for_section(args, "epsilon")
        assert merged["temperature"] == "auto"
        assert merged["conductivity"] == "auto"

    def test_chi_flags_reach_chi_kwargs(self):
        from odas_tpw.rsi.cli import _merge_for_section

        args = _parse(
            "_add_chi_parser",
            [
                "chi",
                "f.p",
                "-o",
                "out",
                "--temperature",
                "JAC_T",
                "--conductivity",
                "JAC_C",
                "--salinity",
                "measured",
            ],
        )
        merged = _merge_for_section(args, "chi")
        assert merged["temperature"] == "JAC_T"
        assert merged["conductivity"] == "JAC_C"
        assert merged["salinity"] == "measured"

    def test_pipeline_flags_reach_pipeline_sections(self):
        from odas_tpw.rsi.cli import _merge_for_section

        args = _parse(
            "_add_pipeline_parser",
            ["pipeline", "f.p", "-o", "out", "--temperature", "T2", "--conductivity", "JAC_C"],
        )
        for section in ("epsilon_pipeline", "chi_pipeline"):
            merged = _merge_for_section(args, section)
            assert merged["temperature"] == "T2", section
            assert merged["conductivity"] == "JAC_C", section

    def test_salinity_rejects_junk(self):
        with pytest.raises(SystemExit):
            _parse("_add_eps_parser", ["eps", "f.p", "-o", "out", "--salinity", "salty"])

    def test_pipeline_salinity_measured_maps_to_none(self, tmp_path, monkeypatch, capsys):
        """--salinity measured never reaches run_pipeline as a string."""
        import odas_tpw.rsi.pipeline as pipeline_mod
        from odas_tpw.rsi.cli import _cmd_pipeline

        captured: dict = {}

        def fake_run_pipeline(files, out, **kwargs):
            captured.update(kwargs)
            return out

        monkeypatch.setattr(pipeline_mod, "run_pipeline", fake_run_pipeline)

        p_file = tmp_path / "x.p"
        p_file.write_bytes(b"")
        args = _parse(
            "_add_pipeline_parser",
            [
                "pipeline",
                str(p_file),
                "-o",
                str(tmp_path / "out"),
                "--salinity",
                "measured",
                "--temperature",
                "T2",
            ],
        )
        _cmd_pipeline(args)
        assert "salinity" not in captured  # None values are stripped
        assert captured["temperature"] == "T2"
        assert "automatic" in capsys.readouterr().out
