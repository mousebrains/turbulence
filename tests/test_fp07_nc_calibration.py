# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""FP07 factory calibration through the per-profile NetCDF path (#131 m8).

``extract_profiles`` embeds ``diff_gain`` + the base-thermistor calibration
(exactly ``_extract_therm_cal``'s output, including ``b``) as float attrs on
every T*_dT* variable; the NetCDF branch of ``_load_therm_channels`` reads
the same keys back, so the perturb chi path (which loads per-profile NetCDFs)
uses the instrument's real coefficients instead of the generic 0.94/defaults.

Real-fixture ground truth (SN479): T1 has b=0.99861 and T1_dT1 has
diff_gain=0.912 — the historical hard-coded 0.94/1.0 fallbacks are provably
wrong there.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from odas_tpw.rsi.chi_io import (
    THERM_CAL_ATTR_KEYS,
    _compute_chi,
    _load_therm_channels,
)
from odas_tpw.rsi.profile import extract_profiles

TEST_DATA_DIR = Path(__file__).parent / "data"
P_FILE = TEST_DATA_DIR / "SN479_0006.p"

pytestmark = pytest.mark.skipif(not P_FILE.exists(), reason="Test data not available")


@pytest.fixture(scope="module")
def prof_nc(tmp_path_factory) -> Path:
    """Per-profile NetCDF freshly extracted from the SN479 fixture."""
    out = tmp_path_factory.mktemp("m8_profiles")
    paths = extract_profiles(P_FILE, out)
    assert len(paths) == 1
    return paths[0]


@pytest.fixture()
def attrless_nc(tmp_path) -> Path:
    """Per-profile NetCDF with the m8 attrs stripped (simulates a pre-m8 file)."""
    import netCDF4 as nc

    paths = extract_profiles(P_FILE, tmp_path)
    ds = nc.Dataset(str(paths[0]), "r+")
    try:
        for vname in ("T1_dT1", "T2_dT2"):
            var = ds.variables[vname]
            for key in ("diff_gain", *THERM_CAL_ATTR_KEYS):
                if key in var.ncattrs():
                    var.delncattr(key)
    finally:
        ds.close()
    return paths[0]


class TestAttrWrite:
    """extract_profiles writes exactly _extract_therm_cal's output + diff_gain."""

    def test_attrs_match_p_config_exactly(self, prof_nc):
        import netCDF4 as nc

        from odas_tpw.rsi.chi_io import _therm_gradient_config
        from odas_tpw.rsi.p_file import PFile

        pf = PFile(P_FILE)
        ds = nc.Dataset(str(prof_nc))
        try:
            for vname in ("T1_dT1", "T2_dT2"):
                var = ds.variables[vname]
                dg_expected, cal_expected = _therm_gradient_config(pf.config, vname)
                assert float(var.getncattr("diff_gain")) == dg_expected
                written = {
                    k: float(var.getncattr(k))
                    for k in THERM_CAL_ATTR_KEYS
                    if k in var.ncattrs()
                }
                # exact keys AND exact values — single source of truth (F7)
                assert written == cal_expected
        finally:
            ds.close()

    def test_real_sn479_values_not_the_hardcodes(self, prof_nc):
        """SN479 ground truth: the 0.94 / b=1.0 hardcodes are wrong here."""
        import netCDF4 as nc

        ds = nc.Dataset(str(prof_nc))
        try:
            var = ds.variables["T1_dT1"]
            assert float(var.getncattr("diff_gain")) == 0.912
            assert float(var.getncattr("b")) == 0.99861  # eta term consumes 'b'
        finally:
            ds.close()


class TestRoundTrip:
    """.p-branch and NC-branch loaders agree exactly on the coefficients."""

    def test_diff_gains_and_therm_cal_exact_equality(self, prof_nc):
        d_p = _load_therm_channels(P_FILE)
        d_nc = _load_therm_channels(prof_nc)
        assert d_nc["diff_gains"] == d_p["diff_gains"]  # exact, no tolerance
        assert d_nc["therm_cal"] == d_p["therm_cal"]  # exact, no tolerance
        assert d_p["diff_gains"] == [0.912, 0.92]
        assert d_p["therm_cal"][0]["b"] == 0.99861

    def test_attrless_nc_falls_back_with_logged_warning(self, attrless_nc, caplog):
        with caplog.at_level(logging.WARNING, logger="odas_tpw.rsi.chi_io"):
            d = _load_therm_channels(attrless_nc)
        assert d["diff_gains"] == [0.94, 0.94]
        assert d["therm_cal"] == [{}, {}]
        msgs = [r.message for r in caplog.records if "FP07 calibration" in r.message]
        assert len(msgs) == 1  # once per file, not per channel
        assert attrless_nc.name in msgs[0]
        assert "T1_dT1" in msgs[0] and "T2_dT2" in msgs[0]

    def test_new_nc_no_warning(self, prof_nc, caplog):
        with caplog.at_level(logging.WARNING, logger="odas_tpw.rsi.chi_io"):
            _load_therm_channels(prof_nc)
        assert not [r for r in caplog.records if "FP07 calibration" in r.message]


class TestChiParity:
    """chi(.p) == chi(new per-profile nc) within tolerance on matched windows.

    Compared with tolerance only AFTER aligning window timestamps: the two
    routes have five known benign divergence sources — f4 channel storage in
    the NetCDF, profile re-detection edges, recomputed speed, W1a
    full-file-vs-trimmed QC context, and perturb's fp07 in-situ recalibration
    of the DATA. On this fixture the attrs carry the FACTORY coefficients
    (the .p config string); perturb's fp07 in-situ calibration may rewrite
    the channel DATA before extract_profiles runs — the attrs still describe
    the electronics (noise floor, bilinear gain), and factory coefficients
    beat the generic defaults either way.

    Discrimination check (measured on this fixture, fft_length=512):
    with the m8 attrs max |log10 ratio| ~ 6e-4; with the generic defaults
    (pre-m8 files) max ~ 1.5e-2, p95 ~ 8e-3. The 5e-3 max / 1e-3 median
    bounds pass the former and fail the latter.
    """

    def test_chi_parity(self, prof_nc):
        res_p = _compute_chi(P_FILE, fft_length=512)
        res_nc = _compute_chi(prof_nc, fft_length=512)
        assert len(res_p) == 1 and len(res_nc) == 1
        t_p = np.asarray(res_p[0]["t"].values, dtype=float)
        t_nc = np.asarray(res_nc[0]["t"].values, dtype=float)
        # Align window timestamps (both are seconds since the same file
        # start_time); re-detection may shift edge windows, so match on
        # rounded times rather than assuming identical grids.
        _, i_p, i_nc = np.intersect1d(
            np.round(t_p, 6), np.round(t_nc, 6), return_indices=True
        )
        assert i_p.size >= 0.8 * min(t_p.size, t_nc.size)
        chi_p = res_p[0]["chi"].values[:, i_p]
        chi_nc = res_nc[0]["chi"].values[:, i_nc]
        ok = np.isfinite(chi_p) & np.isfinite(chi_nc) & (chi_p > 0) & (chi_nc > 0)
        assert ok.sum() >= 20  # enough windows for the statistics to mean much
        ratio = np.abs(np.log10(chi_nc[ok] / chi_p[ok]))
        assert np.median(ratio) < 1e-3
        assert ratio.max() < 5e-3
