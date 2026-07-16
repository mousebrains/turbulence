# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""Cross-probe consistency diagnostics (issue #131): metric, tiers, and wiring.

Covers the shared processing helper (median ln-ratio, n_windows, significance
z, two-tier warning) plus its integration into the rsi diss and chi dataset
builds (attrs `probe_ratio_*` / `chi_probe_ratio_*`) and the m9 chi dof_spec
Goodman DOF loss.
"""

import logging
import math
import types
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from odas_tpw.processing.probe_consistency import (
    N_MIN_PRACTICAL,
    N_MIN_STATISTICAL,
    PROBE_RATIO_MAX,
    Z_WARN,
    annotate_probe_consistency,
    lueck_ln_sigma,
    probe_pair_stats,
)

LOGGER_NAME = "odas_tpw.processing.probe_consistency"

TEST_DATA_DIR = Path(__file__).parent / "data"
PROFILE_FILE = TEST_DATA_DIR / "SN479_0006.p"


def _pair(e1, e2, sigma=0.8):
    """(values, sigma_ln) for a two-probe stack with constant sigma_ln."""
    values = np.vstack([e1, e2])
    return values, np.full_like(values, sigma)


# ---------------------------------------------------------------------------
# lueck_ln_sigma
# ---------------------------------------------------------------------------


def test_lueck_ln_sigma_matches_formula():
    eps = np.array([[1e-9, 1e-7], [1e-8, 1e-6]])
    nu = np.array([1.3e-6, 1.0e-6])
    speed = np.array([0.7, 0.6])
    diss_length, fs = 4096.0, 512.0
    got = lueck_ln_sigma(eps, nu, speed, diss_length, fs)
    L_K = (nu[np.newaxis, :] ** 3 / eps) ** 0.25
    L_hat = (speed * diss_length / fs)[np.newaxis, :] / L_K
    expected = np.sqrt(5.5 / (1.0 + (L_hat / 4.0) ** (7.0 / 9.0)))
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_lueck_ln_sigma_var_resolved_widens_sigma():
    eps = np.full((1, 3), 1e-8)
    nu = np.full(3, 1.3e-6)
    speed = np.full(3, 0.7)
    base = lueck_ln_sigma(eps, nu, speed, 4096, 512.0)
    trunc = lueck_ln_sigma(eps, nu, speed, 4096, 512.0, var_resolved=np.full((1, 3), 0.3))
    # Truncation (V_f < 1) shrinks L_hat, so sigma_ln must grow.
    assert np.all(trunc > base)


def test_lueck_ln_sigma_nan_epsilon_gives_nan():
    eps = np.array([[np.nan, 1e-8]])
    out = lueck_ln_sigma(eps, np.full(2, 1.3e-6), np.full(2, 0.7), 4096, 512.0)
    assert math.isnan(out[0, 0]) and math.isfinite(out[0, 1])


# ---------------------------------------------------------------------------
# probe_pair_stats
# ---------------------------------------------------------------------------


def test_pair_stats_exact_ratio_and_z():
    n = 50
    e2 = np.full(n, 1e-9)
    values, sigma = _pair(100.0 * e2, e2, sigma=0.8)
    (s,) = probe_pair_stats(values, sigma, ["sh1", "sh2"])
    assert s.pair == "sh1/sh2"
    assert s.median_ratio == pytest.approx(100.0)
    assert s.n_windows == n
    se_med = 1.2533 * math.sqrt(2) * 0.8 / math.sqrt(0.7 * n)
    assert s.z == pytest.approx(math.log(100.0) / se_med, rel=1e-6)


def test_pair_stats_only_common_finite_windows():
    e1 = np.array([1e-9, np.nan, 2e-9, 1e-9, -1e-9])
    e2 = np.array([1e-9, 1e-9, np.nan, 2e-9, 1e-9])
    values, sigma = _pair(e1, e2)
    (s,) = probe_pair_stats(values, sigma, ["sh1", "sh2"])
    # windows 0 and 3 survive (1 & 2 have a NaN; 4 is non-positive)
    assert s.n_windows == 2


def test_pair_stats_no_common_windows():
    values, sigma = _pair(np.array([np.nan, 1e-9]), np.array([1e-9, np.nan]))
    (s,) = probe_pair_stats(values, sigma, ["sh1", "sh2"])
    assert s.n_windows == 0
    assert math.isnan(s.median_ratio) and math.isnan(s.z)


def test_pair_stats_three_probes_all_pairs():
    vals = np.vstack([np.full(30, 1e-9), np.full(30, 2e-9), np.full(30, 4e-9)])
    stats = probe_pair_stats(vals, np.full_like(vals, 0.8), ["sh1", "sh2", "sh3"])
    assert [s.pair for s in stats] == ["sh1/sh2", "sh1/sh3", "sh2/sh3"]
    assert stats[1].median_ratio == pytest.approx(0.25)


def test_pair_stats_single_probe_empty():
    assert probe_pair_stats(np.full((1, 10), 1e-9), np.full((1, 10), 0.8), ["sh1"]) == []


def test_pair_stats_accepts_1d_sigma():
    n = 25
    values, _ = _pair(np.full(n, 3e-9), np.full(n, 1e-9))
    (s,) = probe_pair_stats(values, np.full(n, 0.5), ["sh1", "sh2"])
    assert math.isfinite(s.z)


# ---------------------------------------------------------------------------
# annotate_probe_consistency: attrs + two-tier warnings
# ---------------------------------------------------------------------------


def test_annotate_pathological_ratio_warns_and_writes_attrs(caplog):
    """e_1 = 100 x e_2 (CAS_080-class) must warn and carry the attrs."""
    n = 50
    e2 = np.full(n, 1e-9)
    values, sigma = _pair(100.0 * e2, e2, sigma=0.8)
    ds = xr.Dataset()
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        stats = annotate_probe_consistency(
            ds, values, sigma, ["sh1", "sh2"], quantity="epsilon", context="CAS_080.P profile 1"
        )
    assert len(stats) == 1
    assert ds.attrs["probe_ratio_pairs"] == "sh1/sh2"
    assert ds.attrs["probe_ratio_median"] == [pytest.approx(100.0)]
    assert ds.attrs["n_ratio_windows"] == [n]
    assert ds.attrs["probe_ratio_z"][0] > Z_WARN
    assert "binning" in ds.attrs["probe_ratio_comment"]  # F3: non-survival documented
    warned = [r for r in caplog.records if "persistent inter-probe" in r.message]
    assert len(warned) == 1
    assert "CAS_080.P profile 1" in warned[0].message
    assert "#131" in warned[0].message


def test_annotate_healthy_pair_silent_with_plausible_z(caplog):
    """Statistically consistent probes at realistic sigma_ln: attrs, no warning."""
    rng = np.random.default_rng(7)
    n = 60
    sigma = 0.8
    e2 = 1e-9 * np.exp(rng.standard_normal(n) * sigma)
    e1 = e2 * np.exp(rng.standard_normal(n) * sigma)  # same distribution, no offset
    values, sig = _pair(e1, e2, sigma=sigma)
    ds = xr.Dataset()
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        (s,) = annotate_probe_consistency(ds, values, sig, ["sh1", "sh2"])
    assert not [r for r in caplog.records if "persistent inter-probe" in r.message]
    assert math.isfinite(s.z) and s.z < Z_WARN
    assert 1.0 / PROBE_RATIO_MAX < s.median_ratio < PROBE_RATIO_MAX
    assert "probe_ratio_median" in ds.attrs  # attrs are ALWAYS written


def test_annotate_small_n_no_statistical_tier(caplog):
    """N < 20: attrs written, but a significant-yet-moderate offset stays silent."""
    n = 15
    assert n < N_MIN_STATISTICAL
    e2 = np.full(n, 1e-9)
    # ratio 1.5 is inside the practical floor; tiny sigma makes z >> 3.
    values, sigma = _pair(1.5 * e2, e2, sigma=0.01)
    ds = xr.Dataset()
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        (s,) = annotate_probe_consistency(ds, values, sigma, ["sh1", "sh2"])
    assert s.z > Z_WARN  # would trip the statistical tier at N >= 20
    assert not [r for r in caplog.records if "persistent inter-probe" in r.message]
    assert ds.attrs["n_ratio_windows"] == [n]


def test_annotate_practical_floor_fires_below_statistical_n(caplog):
    """A calibration-scale ratio warns from N >= 10 even without N >= 20."""
    n = 10
    assert N_MIN_PRACTICAL <= n < N_MIN_STATISTICAL
    e2 = np.full(n, 1e-9)
    values, sigma = _pair(5.0 * e2, e2, sigma=5.0)  # huge sigma: z is small
    ds = xr.Dataset()
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        annotate_probe_consistency(ds, values, sigma, ["sh1", "sh2"])
    warned = [r for r in caplog.records if "persistent inter-probe" in r.message]
    assert len(warned) == 1
    assert "practical floor" in warned[0].message


def test_annotate_below_practical_n_silent(caplog):
    n = 5
    e2 = np.full(n, 1e-9)
    values, sigma = _pair(5.0 * e2, e2, sigma=5.0)
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        annotate_probe_consistency(xr.Dataset(), values, sigma, ["sh1", "sh2"])
    assert not [r for r in caplog.records if "persistent inter-probe" in r.message]


def test_annotate_low_ratio_direction_also_trips_floor(caplog):
    """The practical floor is two-sided: median ratio 1/5 warns like 5."""
    n = 30
    e2 = np.full(n, 1e-9)
    values, sigma = _pair(e2 / 5.0, e2, sigma=5.0)
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        annotate_probe_consistency(xr.Dataset(), values, sigma, ["sh1", "sh2"])
    assert [r for r in caplog.records if "practical floor" in r.message]


def test_annotate_single_probe_writes_nothing():
    ds = xr.Dataset()
    stats = annotate_probe_consistency(
        ds, np.full((1, 10), 1e-9), np.full((1, 10), 0.8), ["sh1"]
    )
    assert stats == []
    assert "probe_ratio_pairs" not in ds.attrs


def test_annotate_chi_prefix():
    n = 30
    e2 = np.full(n, 1e-8)
    values, sigma = _pair(2.0 * e2, e2)
    ds = xr.Dataset()
    annotate_probe_consistency(
        ds, values, sigma, ["T1_dT1", "T2_dT2"], quantity="chi", attr_prefix="chi_"
    )
    assert ds.attrs["chi_probe_ratio_pairs"] == "T1_dT1/T2_dT2"
    assert "chi_probe_ratio_median" in ds.attrs
    assert "probe_ratio_median" not in ds.attrs


# ---------------------------------------------------------------------------
# Integration: rsi diss / chi dataset builds carry the attrs (real fixture)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not PROFILE_FILE.exists(), reason="SN479 fixture missing")
def test_diss_build_carries_probe_ratio_attrs(tmp_path):
    from odas_tpw.rsi.dissipation import _compute_epsilon

    results = _compute_epsilon(PROFILE_FILE, fft_length=256, goodman=True)
    assert results
    ds = results[0]
    assert ds.attrs["probe_ratio_pairs"] == "sh1/sh2"
    assert len(ds.attrs["probe_ratio_median"]) == 1
    assert ds.attrs["n_ratio_windows"][0] > 0
    assert np.isfinite(ds.attrs["probe_ratio_z"][0])

    # The list attrs must survive NetCDF serialization (write_profile_results).
    out = tmp_path / "diss.nc"
    ds.to_netcdf(out)
    back = xr.open_dataset(out)
    try:
        assert float(np.atleast_1d(back.attrs["probe_ratio_median"])[0]) == pytest.approx(
            float(ds.attrs["probe_ratio_median"][0])
        )
    finally:
        back.close()


@pytest.fixture(scope="module")
def chi_ds():
    if not PROFILE_FILE.exists():
        pytest.skip("SN479 fixture missing")
    from odas_tpw.rsi.chi_io import _compute_chi

    results = _compute_chi(PROFILE_FILE, fft_length=1024, goodman=True)
    assert results
    return results[0]


@pytest.mark.skipif(not PROFILE_FILE.exists(), reason="SN479 fixture missing")
class TestChiBuildIntegration:
    def test_chi_build_carries_prefixed_attrs(self, chi_ds):
        assert chi_ds.attrs["chi_probe_ratio_pairs"] == "T1_dT1/T2_dT2"
        assert len(chi_ds.attrs["chi_probe_ratio_median"]) == 1
        assert chi_ds.attrs["chi_n_ratio_windows"][0] > 0

    def test_chi_dof_spec_subtracts_goodman_vib(self, chi_ds):
        # m9: SN479 has 2 accelerometers; with goodman the chi dof_spec loses
        # one FFT segment per removed vibration signal (Lueck 2022b), matching
        # the epsilon convention in dissipation.py.
        from odas_tpw.rsi.dissipation import DOF_NUTTALL

        num_ffts = 2 * 4 - 1  # diss_length = 4 * fft_length, 50% overlap
        assert chi_ds.attrs["dof_spec"] == pytest.approx(DOF_NUTTALL * (num_ffts - 2))

    def test_chi_dof_spec_no_goodman(self):
        from odas_tpw.rsi.chi_io import _compute_chi
        from odas_tpw.rsi.dissipation import DOF_NUTTALL

        results = _compute_chi(PROFILE_FILE, fft_length=1024, goodman=False)
        assert results
        num_ffts = 2 * 4 - 1
        assert results[0].attrs["dof_spec"] == pytest.approx(DOF_NUTTALL * num_ffts)


# ---------------------------------------------------------------------------
# m9 unit level: _build_chi_ds_from_pipeline dof_spec
# ---------------------------------------------------------------------------


def _l3(n_wave=8, n_spec=1):
    from odas_tpw.chi.l3_chi import L3ChiData

    kcyc = np.tile(np.linspace(1.0, 100.0, n_wave)[:, None], (1, n_spec))
    return L3ChiData(
        time=np.arange(n_spec, dtype=float),
        pres=np.zeros(n_spec),
        temp=np.full(n_spec, 20.0),
        pspd_rel=np.full(n_spec, 1.0),
        section_number=np.ones(n_spec, dtype=int),
        nu=np.full(n_spec, 1.3e-6),
        kappa_T=np.full(n_spec, 1.4e-7),
        kcyc=kcyc,
        freq=np.linspace(1.0, 98.0, n_wave),
        gradt_spec=np.zeros((1, n_wave, n_spec)),
        noise_spec=np.zeros((1, n_wave, n_spec)),
        H2=np.ones((n_spec, n_wave)),
        tau0=np.full(n_spec, 0.007),
    )


def _build_chi_ds(n_v=None):
    from odas_tpw.chi.batchelor import batchelor_grad
    from odas_tpw.rsi.chi_io import _build_chi_ds_from_pipeline

    one = np.full((1, 1), 1e-7)
    l4 = types.SimpleNamespace(
        chi=one,
        epsilon_T=np.full((1, 1), 1e-9),
        kB=np.full((1, 1), 50.0),
        K_max=np.full((1, 1), 40.0),
        fom=np.ones((1, 1)),
        K_max_ratio=np.full((1, 1), 0.8),
        var_resolved=np.full((1, 1), 0.9),
        method="fit",
    )
    kwargs = dict(
        fft_length=256,
        diss_length=1024,
        overlap=512,
        fs_fast=512.0,
        fp07_model="single_pole",
        spectrum_model="batchelor",
        fit_method="iterative",
        f_AA=98.0,
        grad_func=batchelor_grad,
        chi_method="fit",
    )
    if n_v is not None:
        kwargs["n_v"] = n_v
    return _build_chi_ds_from_pipeline(_l3(), l4, ["T1_dT1"], **kwargs)


def test_chi_dof_spec_builder_values():
    from odas_tpw.rsi.dissipation import DOF_NUTTALL

    num_ffts = 2 * (1024 // 256) - 1  # 7
    assert _build_chi_ds(n_v=0).attrs["dof_spec"] == pytest.approx(DOF_NUTTALL * num_ffts)
    assert _build_chi_ds(n_v=2).attrs["dof_spec"] == pytest.approx(DOF_NUTTALL * (num_ffts - 2))
    # Floor at one segment, mirroring dissipation.py.
    assert _build_chi_ds(n_v=99).attrs["dof_spec"] == pytest.approx(DOF_NUTTALL * 1)
    # Back-compat default (n_v omitted) is the no-Goodman value.
    assert _build_chi_ds().attrs["dof_spec"] == pytest.approx(DOF_NUTTALL * num_ffts)
