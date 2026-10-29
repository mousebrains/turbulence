# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""The chi noise model takes the bead coefficients the data were converted with (#245).

With perturb's ``fp07.calibrate: true`` the FP07 DATA are converted with the
per-file in-situ fit. The noise model converts its counts to kelvin with
``T_0``/``beta_1``/``beta_2`` -- the bead's R(T), not the electronics -- so it
must use the same fit. It used the embedded configuration's, which on SUNRISE
2021 Pelican SN194 after the T2 swap (t_0 262.376, beta_1 708.07, beta_2 1e3)
put the T2 noise model several times too high.

Exercised end to end on the SN479 fixture through ``process_file``; the fit is
spied on, not re-run, so the expected coefficients are exactly the ones the
pipeline applied.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import ClassVar

import netCDF4 as nc
import numpy as np
import pytest
import xarray as xr

from odas_tpw.chi.fp07 import gradT_noise
from odas_tpw.fp07cal.logr import coeffs_to_config
from odas_tpw.rsi.chi_io import (
    BEAD_COEFFS_SOURCE_ATTR,
    IN_SITU_COEFFS_KEY,
    THERM_CAL_ATTR_KEYS,
    _extract_therm_cal,
    _load_therm_channels,
    _therm_gradient_config,
)

DATA = Path(__file__).parent / "data"
P_FILE = DATA / "SN479_0006.p"

pytestmark = pytest.mark.skipif(not P_FILE.exists(), reason="Test data not available")

# SUNRISE 2021 Pelican SN194 T2 after the bead swap (SR1P_518 on): the
# embedded coefficients the campaign config calls garbage.
GARBAGE_T2 = {"t_0": "262.376", "beta_1": "708.07", "beta_2": "1.0e3"}

# SN479's embedded T1 stanza (the electronics plus a nominal bead).
_CFG = {
    "name": "T1", "e_b": "0.68283", "b": "0.99861", "g": "6.0", "beta_1": "3143.55",
    "beta_2": "2.5e5", "adc_fs": "4.096", "adc_bits": "16", "t_0": "289.301",
}


def _run(tmp_path: Path, p_file: Path, fp07: dict, monkeypatch=None, edit=None):
    """process_file with chi on; returns (result, dirs, the fit's own result)."""
    from odas_tpw.perturb import fp07_cal
    from odas_tpw.perturb.gps import GPSFixed
    from odas_tpw.perturb.pipeline import process_file

    captured: dict = {}
    if monkeypatch is not None:
        real = fp07_cal.fp07_calibrate

        def spy(*args, **kwargs):
            res = real(*args, **kwargs)
            if edit is not None:
                edit(res)
            captured.update(res)
            return res

        monkeypatch.setattr(fp07_cal, "fp07_calibrate", spy)
    dirs = {k: tmp_path / k for k in ("profiles", "diss", "chi")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = process_file(
            p_file, {"chi": {"enable": True}, "fp07": fp07}, GPSFixed(15.0, 145.0), dirs
        )
    assert not res.get("errors"), res.get("errors")
    return res, dirs, captured


def _var_attrs(path: Path, vname: str) -> dict:
    ds = nc.Dataset(str(path))
    try:
        var = ds.variables[vname]
        return {k: var.getncattr(k) for k in var.ncattrs()}
    finally:
        ds.close()


def _embedded_cal(vname: str) -> dict[str, float]:
    from odas_tpw.rsi.p_file import PFile

    return _therm_gradient_config(PFile(P_FILE).config, vname)[1]


class TestExtractThermCal:
    """The fit replaces the bead's R(T) whole; the electronics stay embedded."""

    def test_no_fit_is_the_embedded_configuration(self):
        assert _extract_therm_cal(_CFG, None) == _extract_therm_cal(_CFG)
        assert _extract_therm_cal(_CFG)["beta_2"] == 2.5e5

    def test_order_1_leaves_no_quadratic_term(self):
        cal = _extract_therm_cal(_CFG, {"t_0": 290.0, "beta_1": 3080.0})
        assert cal["T_0"] == 290.0 and cal["beta_1"] == 3080.0
        assert "beta_2" not in cal  # the embedded 2.5e5 must not survive
        for k in ("e_b", "b", "gain", "adc_fs", "adc_bits"):
            assert cal[k] == _extract_therm_cal(_CFG)[k]

    def test_order_2_replaces_beta_2(self):
        cal = _extract_therm_cal(_CFG, {"t_0": 290.0, "beta_1": 3080.0, "beta_2": 3.7e5})
        assert (cal["T_0"], cal["beta_1"], cal["beta_2"]) == (290.0, 3080.0, 3.7e5)

    def test_order_3_carries_beta_3(self):
        """The noise model inverts a fitted curve exactly, so it needs all of it."""
        fit = {"t_0": 290.0, "beta_1": 3080.0, "beta_2": 2.7e5, "beta_3": 6.4e5}
        cal = _extract_therm_cal(_CFG, fit)
        assert (cal["beta_2"], cal["beta_3"]) == (2.7e5, 6.4e5)
        assert set(cal) <= set(THERM_CAL_ATTR_KEYS)

    def test_an_embedded_beta_3_is_never_read(self):
        """Without a fit (calibrate: false, fp07-cal patched input) nothing changes."""
        cal = _extract_therm_cal(_CFG | {"beta_3": "6.4e5"})
        assert "beta_3" not in cal
        assert cal == _extract_therm_cal(_CFG)

    def test_order_2_adds_beta_2_where_the_embedded_had_none(self):
        cfg = {k: v for k, v in _CFG.items() if k != "beta_2"}
        cal = _extract_therm_cal(cfg, {"t_0": 290.0, "beta_1": 3080.0, "beta_2": 3.7e5})
        assert cal["beta_2"] == 3.7e5

    def test_fit_overrides_the_legacy_beta_key(self):
        cfg = {k: v for k, v in _CFG.items() if k != "beta_1"} | {"beta": "3400"}
        cal = _extract_therm_cal(cfg, {"t_0": 290.0, "beta_1": 3080.0})
        assert cal["beta_1"] == 3080.0 and "beta" not in cal


@pytest.mark.parametrize("order", [1, 2, 3])
def test_fit_order_maps_to_the_gradient_attrs(tmp_path, monkeypatch, order):
    """Each order's fitted coefficients, exactly, on T*_dT*; no stale beta_2."""
    res, dirs, fit = _run(tmp_path, P_FILE, {"calibrate": True, "order": order}, monkeypatch)
    assert res["fp07"]["fitted"] == ["T1", "T2"]
    prof = sorted(dirs["profiles"].glob("*.nc"))[0]
    tag = f"in-situ (Steinhart-Hart order {order} vs JAC_T)"
    for base in ("T1", "T2"):
        attrs = _var_attrs(prof, f"{base}_d{base}")
        want = coeffs_to_config(fit["coefficients"][base])
        assert attrs["T_0"] == want["t_0"]
        assert attrs["beta_1"] == want["beta_1"]
        if order == 1:
            assert "beta_2" not in attrs  # the embedded 2.5e5 is gone
        else:
            assert attrs["beta_2"] == want["beta_2"]
        if order == 3:
            assert attrs["beta_3"] == want["beta_3"]
        else:
            assert "beta_3" not in attrs
        assert attrs[BEAD_COEFFS_SOURCE_ATTR] == tag
        # The electronics are still the embedded configuration's.
        embedded = _embedded_cal(f"{base}_d{base}")
        for k in ("e_b", "b", "gain", "adc_fs", "adc_bits"):
            assert attrs[k] == embedded[k]
    chi = sorted(dirs["chi"].glob("*.nc"))[0]
    with xr.open_dataset(chi) as ds:
        assert ds.attrs[f"{BEAD_COEFFS_SOURCE_ATTR}_T1_dT1"] == tag
        assert ds.attrs[f"{BEAD_COEFFS_SOURCE_ATTR}_T2_dT2"] == tag


def test_refused_channel_keeps_the_embedded_coefficients(tmp_path, monkeypatch):
    """A channel the fit refused keeps its data AND its noise model embedded."""

    def refuse_t2(res):
        for key in ("channels", "coefficients", "lags", "info"):
            res[key].pop("T2", None)
        res["fast_channels"].pop("T2_dT2", None)
        res["factory"]["T2"] = "refused for the test"

    res, dirs, fit = _run(tmp_path, P_FILE, {"calibrate": True}, monkeypatch, refuse_t2)
    assert res["fp07"]["fitted"] == ["T1"]
    prof = sorted(dirs["profiles"].glob("*.nc"))[0]
    t2 = _var_attrs(prof, "T2_dT2")
    assert "calibration" not in t2
    assert BEAD_COEFFS_SOURCE_ATTR not in t2
    assert {k: t2[k] for k in THERM_CAL_ATTR_KEYS if k in t2} == _embedded_cal("T2_dT2")
    t1 = _var_attrs(prof, "T1_dT1")
    assert t1["beta_1"] == coeffs_to_config(fit["coefficients"]["T1"])["beta_1"]
    with xr.open_dataset(sorted(dirs["chi"].glob("*.nc"))[0]) as ds:
        assert f"{BEAD_COEFFS_SOURCE_ATTR}_T1_dT1" in ds.attrs
        assert f"{BEAD_COEFFS_SOURCE_ATTR}_T2_dT2" not in ds.attrs


def test_unrepresentable_fit_applies_nothing(tmp_path, monkeypatch):
    """A coefficient with no configuration form raises BEFORE anything is applied,
    so no channel is left with fitted data under embedded noise coefficients."""

    def zero_t2(res):
        res["coefficients"]["T2"] = np.array([3.4e-3, 0.0, 1e-6])

    res, dirs, _ = _run(tmp_path, P_FILE, {"calibrate": True}, monkeypatch, zero_t2)
    assert res["fp07"]["fitted"] == []
    assert "calibration raised" in res["fp07"]["factory"]["all"]
    prof = sorted(dirs["profiles"].glob("*.nc"))[0]
    for vname in ("T1_dT1", "T2_dT2"):
        attrs = _var_attrs(prof, vname)
        assert "calibration" not in attrs and BEAD_COEFFS_SOURCE_ATTR not in attrs


class TestCalibrateFalseUnchanged:
    """``calibrate: false`` (and fp07-cal patched input, run so) writes what it did."""

    # The SN479 attrs as written before #245, in their written order: a golden.
    GOLDEN: ClassVar[dict[str, list[tuple[str, object]]]] = {
        "T1_dT1": [
            ("units", "degree_Celsius"), ("sensor_type", "therm"), ("long_name", "T1_dT1"),
            ("diff_gain", 0.912), ("e_b", 0.68283), ("b", 0.99861), ("beta_1", 3143.55),
            ("beta_2", 2.5e5), ("adc_fs", 4.096), ("adc_bits", 16.0), ("gain", 6.0),
            ("T_0", 289.301),
        ],
        "T2_dT2": [
            ("units", "degree_Celsius"), ("sensor_type", "therm"), ("long_name", "T2_dT2"),
            ("diff_gain", 0.92), ("e_b", 0.68281), ("b", 0.99927), ("beta_1", 3143.55),
            ("beta_2", 2.5e5), ("adc_fs", 4.096), ("adc_bits", 16.0), ("gain", 6.0),
            ("T_0", 289.301),
        ],
    }

    def test_attrs_are_the_golden_set_in_order(self, tmp_path):
        res, dirs, _ = _run(tmp_path, P_FILE, {"calibrate": False})
        assert "fp07" not in res
        prof = sorted(dirs["profiles"].glob("*.nc"))[0]
        for vname, golden in self.GOLDEN.items():
            assert list(_var_attrs(prof, vname).items()) == golden, vname
        with xr.open_dataset(sorted(dirs["chi"].glob("*.nc"))[0]) as ds:
            assert not [k for k in ds.attrs if k.startswith(BEAD_COEFFS_SOURCE_ATTR)]

    def test_the_p_route_is_unchanged(self):
        from odas_tpw.rsi.p_file import PFile

        pf = PFile(P_FILE)
        d = _load_therm_channels(pf)
        assert d["therm_cal_source"] == [None, None]
        assert d["therm_cal"] == [_embedded_cal("T1_dT1"), _embedded_cal("T2_dT2")]


class TestNoiseSpectrum:
    """The chi product's noise spectrum is noise_thermchannel on the FITTED bead."""

    def test_spec_noise_is_built_from_the_fit(self, tmp_path, monkeypatch):
        _, dirs, fit = _run(tmp_path, P_FILE, {"calibrate": True}, monkeypatch)
        prof = sorted(dirs["profiles"].glob("*.nc"))[0]
        with xr.open_dataset(prof) as p:
            fs = float(p.attrs["fs_fast"])
        with xr.open_dataset(sorted(dirs["chi"].glob("*.nc"))[0]) as ds:
            assert ds["spec_noise"].dims == ("probe", "freq", "time")
            n_win = ds.sizes["time"]
            for ci, (base, dg) in enumerate((("T1", 0.912), ("T2", 0.92))):
                embedded = _embedded_cal(f"{base}_d{base}")
                c = coeffs_to_config(fit["coefficients"][base])  # order 2
                fitted = {
                    **embedded, "T_0": c["t_0"], "beta_1": c["beta_1"], "beta_2": c["beta_2"],
                    "exact_inversion": True,  # a fit's curve is inverted exactly
                }
                # SN479's nominal coefficients sit near its fit in the noise
                # (median 1.05x on T1, 1.11x on T2, though T2's t_0 is ~8 K
                # out: (1+R)^2/R is flat at R=1), so this pins the source, not
                # the size; TestSunriseGarbageEmbedded pins a case where it matters.
                assert fitted != embedded
                for w in (0, n_win // 2, n_win - 1):
                    F = ds["F"].values[:, w]
                    T = float(ds["T_mean"].values[w])
                    W = float(ds["speed"].values[w])
                    want, _ = gradT_noise(F, T, W, fs=fs, diff_gain=dg, **fitted)
                    got = ds["spec_noise"].values[ci, :, w]
                    ok = want > 0
                    np.testing.assert_allclose(got[ok], want[ok], rtol=1e-6)

    def test_p_route_and_profile_route_agree(self, tmp_path, monkeypatch):
        """The PFile branch (pf.channel_info) and the NetCDF attrs give one therm_cal."""
        from odas_tpw.rsi.p_file import PFile
        from odas_tpw.rsi.profile import extract_profiles

        pf = PFile(P_FILE)
        fit = {"t_0": 290.1, "beta_1": 3075.0}  # an order-1 fit on T2 only
        pf.channel_info["T2"][IN_SITU_COEFFS_KEY] = fit
        pf.channel_info["T2"]["calibration"] = "in-situ (Steinhart-Hart order 1 vs JAC_T)"
        prof = extract_profiles(pf, tmp_path)[0]
        d_p = _load_therm_channels(pf)
        d_nc = _load_therm_channels(prof)
        assert d_nc["therm_cal"] == d_p["therm_cal"]
        assert d_nc["therm_cal_source"] == d_p["therm_cal_source"] == [
            None, "in-situ (Steinhart-Hart order 1 vs JAC_T)",
        ]
        assert d_p["therm_cal"][1]["beta_1"] == 3075.0
        assert "beta_2" not in d_p["therm_cal"][1]
        assert d_p["therm_cal"][0] == _embedded_cal("T1_dT1")


@pytest.fixture()
def garbage_p(tmp_path) -> Path:
    """The fixture with SUNRISE 2021 Pelican's garbage T2 coefficients embedded."""
    from odas_tpw.rsi import config_patch as cp

    out = tmp_path / "garbage" / P_FILE.name
    out.parent.mkdir()
    spec = cp.EditSpec(note="SUNRISE 2021 Pelican T2", author="test",
                       channels={"T2": dict(GARBAGE_T2)})
    text, changes = cp.edit_config_text(
        cp.read_config_text(P_FILE), spec, when="2026-10-29 12:00:00"
    )
    assert changes
    cp.write_patched_pfile(P_FILE, out, text)
    return out


class TestSunriseGarbageEmbedded:
    """SUNRISE-style regression: garbage embedded T2 coefficients, fitted data.

    The fit depends only on the raw counts, the bridge and the reference, so a
    copy of the fixture whose T2 stanza carries SUNRISE 2021 Pelican's garbage
    must give exactly the original's T2 attrs and noise spectrum. Before #245 it
    gave a noise model on the garbage, several times too high.
    """

    def test_noise_model_matches_the_fitted_calibration(self, tmp_path, garbage_p):
        _, d_ref, _ = _run(tmp_path / "ref", P_FILE, {"calibrate": True})
        _, d_bad, _ = _run(tmp_path / "bad", garbage_p, {"calibrate": True})
        prof_ref = sorted(d_ref["profiles"].glob("*.nc"))[0]
        prof_bad = sorted(d_bad["profiles"].glob("*.nc"))[0]
        bad_attrs = _var_attrs(prof_bad, "T2_dT2")
        for k in ("T_0", "beta_1", "beta_2"):
            assert bad_attrs[k] == _var_attrs(prof_ref, "T2_dT2")[k], k
        assert bad_attrs["beta_1"] > 2500.0  # not the embedded 708.07
        with xr.open_dataset(sorted(d_ref["chi"].glob("*.nc"))[0]) as ref, \
                xr.open_dataset(sorted(d_bad["chi"].glob("*.nc"))[0]) as bad:
            np.testing.assert_array_equal(bad["spec_noise"].values, ref["spec_noise"].values)
            np.testing.assert_array_equal(bad["chi"].values, ref["chi"].values)
            # What the garbage would have given: noise_thermchannel on the
            # embedded coefficients, against the fixed spectrum, mid-profile.
            garbage = {**_embedded_cal("T2_dT2"),
                       "T_0": 262.376, "beta_1": 708.07, "beta_2": 1.0e3}
            w = bad.sizes["time"] // 2
            F = bad["F"].values[:, w]
            with xr.open_dataset(prof_bad) as p:
                fs = float(p.attrs["fs_fast"])
            old, _ = gradT_noise(F, float(bad["T_mean"].values[w]), float(bad["speed"].values[w]),
                                 fs=fs, diff_gain=0.92, **garbage)
            new = bad["spec_noise"].values[1, :, w]
            ok = new > 0
            assert np.median(old[ok] / new[ok]) > 3.0


def test_stale_profile_file_warns(tmp_path, monkeypatch, caplog):
    """A per-profile file with in-situ data and no source attr predates #245."""
    _, dirs, _ = _run(tmp_path, P_FILE, {"calibrate": True}, monkeypatch)
    prof = sorted(dirs["profiles"].glob("*.nc"))[0]
    with caplog.at_level(logging.WARNING, logger="odas_tpw.rsi.chi_io"):
        _load_therm_channels(prof)
    assert not [r for r in caplog.records if "#245" in r.message]
    ds = nc.Dataset(str(prof), "r+")
    try:
        ds.variables["T2_dT2"].delncattr(BEAD_COEFFS_SOURCE_ATTR)
    finally:
        ds.close()
    with caplog.at_level(logging.WARNING, logger="odas_tpw.rsi.chi_io"):
        d = _load_therm_channels(prof)
    msgs = [r.message for r in caplog.records if "#245" in r.message]
    assert len(msgs) == 1 and "T2_dT2" in msgs[0] and "T1_dT1" not in msgs[0]
    assert d["therm_cal_source"][1] is None


# Per-file perturb fits (order 2 over a narrow temperature range) whose ODAS
# order-1 inversion was far off, with the median L = ln(R/R_0) of the file's own
# raw counts: the data's branch, which the NTC root must find.
#   (t_0, beta_1, beta_2, L_data)
CURVED_FITS = {
    "husker FSA_011 T1": (164.99497451604446, 115.30492419609796, 145.1874435295229, -0.6135),
    "husker FSA_011 T2": (211.00146087877468, 334.65067740466907, 634.7499786499253, -0.8680),
    "CASPER 2015 CAS_035 T1": (412.44534219123767, -216.0380347368681, -181.23061637192924,
                               -0.4565),
    "CASPER 2015 CAS_035 T2": (402.21471049748925, -242.5999867714686, -214.14860825124876,
                               -0.4832),
    "CASPER 2015 CAS_024 T1": (511.85674243200094, -146.20766746580335, -123.31209918244686,
                               -0.4443),
    "Taiwan 2017 CAS_006 T1": (292.25478525732586, -12268.527041024807, -2489.567993152531,
                               -0.5216),
}
_BRIDGE = {"a": 0.0, "b": 1.0, "g": 6.0, "e_b": 0.68, "adc_fs": 4.096, "adc_bits": 16}
_ETA = (_BRIDGE["b"] / 2) * 2 ** _BRIDGE["adc_bits"] * _BRIDGE["g"] * _BRIDGE["e_b"] / (
    _BRIDGE["adc_fs"]
)


def _fd_noise(F, coeffs, L_data, fs=512.0, dg=0.94):
    """(T_C, expected PSD): the noise model with convert_therm's own dT/dcount.

    The temperature is the curve's at the data's L; the scale factor is a
    central finite difference of convert_therm in counts there, and R = e^L.
    """
    from odas_tpw.chi.fp07 import _noise_f_intermediates, _unpack_noise_config
    from odas_tpw.rsi.channels import convert_therm

    t_0, b1, b2 = coeffs
    T_K = 1.0 / (1 / t_0 + L_data / b1 + L_data**2 / b2)
    Z = (1 - np.exp(L_data)) / (1 + np.exp(L_data))
    N = Z * _ETA
    params = {**_BRIDGE, "t_0": t_0, "beta_1": b1, "beta_2": b2}
    h = 0.5
    dTdN = (convert_therm(np.array([N + h]), params)[0][0]
            - convert_therm(np.array([N - h]), params)[0][0]) / (2 * h)
    p = _unpack_noise_config(None, gain=6.0, e_b=0.68, b=1.0, beta_1=b1, T_0=t_0, beta_2=b2)
    base_f, jg, cf, _ = _noise_f_intermediates(np.asarray(F, float), fs, dg, p)
    phi = 4 * p["K_B"] * np.exp(L_data) * p["R_0"] * p["T_K"]
    return T_K - 273.15, (base_f + jg * phi) * cf * dTdN**2


class TestExactInversion:
    """A fitted curve is inverted exactly: noise matches convert_therm's dT/dcount."""

    F = np.linspace(2.0, 100.0, 50)

    @pytest.mark.parametrize("case", [*sorted(CURVED_FITS), "nominal at 2 C"])
    def test_curved_fit_matches_a_finite_difference(self, case):
        from odas_tpw.chi.fp07 import noise_thermchannel

        # The nominal curve at L = 0.55 (2.1 C), where the ODAS form is 0.5% off.
        t_0, b1, b2, L_data = CURVED_FITS.get(case, (289.301, 3143.55, 2.5e5, 0.55))
        T_C, want = _fd_noise(self.F, (t_0, b1, b2), L_data)
        got = noise_thermchannel(self.F, T_C, T_0=t_0, beta_1=b1, beta_2=b2,
                                 exact_inversion=True)
        np.testing.assert_allclose(got, want, rtol=1e-3)

    def test_the_odas_form_was_far_off_there(self):
        """The defect: FSA_011 T1's first-order inversion, ~300x in power."""
        from odas_tpw.chi.fp07 import noise_thermchannel

        t_0, b1, b2, L_data = CURVED_FITS["husker FSA_011 T1"]
        T_C, want = _fd_noise(self.F, (t_0, b1, b2), L_data)
        old = noise_thermchannel(self.F, T_C, T_0=t_0, beta_1=b1, beta_2=b2)
        assert np.median(old / want) > 100

    def test_the_root_is_the_datas_branch(self):
        """CAS_024 T1: the root nearest the order-1 estimate is 0.04 in L off the
        data (-0.403); the NTC root (d(1/T)/dL > 0) is on them."""
        from odas_tpw.chi.fp07 import bead_inversion

        t_0, b1, b2, L_data = CURVED_FITS["CASPER 2015 CAS_024 T1"]
        T_C = 1.0 / (1 / t_0 + L_data / b1 + L_data**2 / b2) - 273.15
        L, slope, solved = bead_inversion(T_C, t_0, b1, b2)
        assert solved[0] and slope[0] > 0
        assert L[0] == pytest.approx(L_data, abs=1e-9)

    def test_batch_matches_scalar(self):
        from odas_tpw.chi.fp07 import noise_thermchannel, noise_thermchannel_batch

        t_0, b1, b2, _ = CURVED_FITS["CASPER 2015 CAS_035 T1"]
        T = np.array([21.57, 21.69, 22.11])
        kw = {"T_0": t_0, "beta_1": b1, "beta_2": b2, "exact_inversion": True}
        batch = noise_thermchannel_batch(self.F, T, **kw)
        for i, t in enumerate(T):
            np.testing.assert_allclose(batch[i], noise_thermchannel(self.F, t, **kw), rtol=1e-12)

    def test_order_3_uses_beta_3(self):
        """A cubic term moves the exact noise model and is solved with the rest."""
        from odas_tpw.chi.fp07 import bead_inversion, noise_thermchannel

        kw = {"T_0": 289.98, "beta_1": 3078.6, "beta_2": 2.7e5}
        with3 = noise_thermchannel(self.F, 8.0, beta_3=-2.0e3, exact_inversion=True, **kw)
        without = noise_thermchannel(self.F, 8.0, exact_inversion=True, **kw)
        assert not np.allclose(with3, without, rtol=1e-3)
        L, slope, _ = bead_inversion(8.0, 289.98, 3078.6, 2.7e5, -2.0e3)
        a = (1 / 289.98, 1 / 3078.6, 1 / 2.7e5, 1 / -2.0e3)
        assert sum(ai * L[0] ** i for i, ai in enumerate(a)) == pytest.approx(1 / 281.15)
        assert slope[0] == pytest.approx(a[1] + 2 * a[2] * L[0] + 3 * a[3] * L[0] ** 2)

    def test_order_1_is_the_odas_form_bit_for_bit(self):
        from odas_tpw.chi.fp07 import noise_thermchannel, noise_thermchannel_batch

        kw = {"T_0": 283.2, "beta_1": 3086.0}
        assert np.array_equal(noise_thermchannel(self.F, 28.1, exact_inversion=True, **kw),
                              noise_thermchannel(self.F, 28.1, **kw))
        T = np.array([27.5, 28.1])
        assert np.array_equal(noise_thermchannel_batch(self.F, T, exact_inversion=True, **kw),
                              noise_thermchannel_batch(self.F, T, **kw))

    def test_nominal_curve_agrees_with_the_odas_form(self):
        """Where the first-order form is valid (beta_2 = 2.5e5) the two agree to
        ~0.5% in power (at 2 C; the exact one is the finite difference's)."""
        from odas_tpw.chi.fp07 import noise_thermchannel

        kw = {"T_0": 289.301, "beta_1": 3143.55, "beta_2": 2.5e5}
        for T in (2.0, 15.0, 28.0):
            np.testing.assert_allclose(
                noise_thermchannel(self.F, T, exact_inversion=True, **kw),
                noise_thermchannel(self.F, T, **kw), rtol=1e-2)

    def test_no_real_root_falls_back_with_a_warning(self):
        """FSA_011 T1's parabola cannot reach 28 C (its turning point is ~27.08 C)."""
        from odas_tpw.chi.fp07 import bead_inversion, noise_thermchannel
        from odas_tpw.rsi.chi_io import _count_unsolved_inversions

        t_0, b1, b2, _ = CURVED_FITS["husker FSA_011 T1"]
        _, _, solved = bead_inversion(np.array([26.9, 28.0]), t_0, b1, b2)
        assert solved.tolist() == [True, False]
        with pytest.warns(UserWarning, match="no real root"):
            got = noise_thermchannel(self.F, 28.0, T_0=t_0, beta_1=b1, beta_2=b2,
                                     exact_inversion=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            odas = noise_thermchannel(self.F, 28.0, T_0=t_0, beta_1=b1, beta_2=b2)
        np.testing.assert_allclose(got, odas, rtol=1e-12)
        cal = {"T_0": t_0, "beta_1": b1, "beta_2": b2, "exact_inversion": True}
        assert _count_unsolved_inversions([26.9, 28.0, np.nan], cal) == 1
        assert _count_unsolved_inversions([26.9, 28.0], {**cal, "exact_inversion": False}) == 0
