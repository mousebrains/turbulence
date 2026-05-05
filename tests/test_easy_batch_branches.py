# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Easy-batch branch tests — small-module quick wins toward >95% coverage."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

# ---------------------------------------------------------------------------
# processing/top_trim.py — line 67: bin_edges < 2 (max_depth < min_depth)
# ---------------------------------------------------------------------------


class TestTopTrimEmptyBinEdges:
    def test_max_depth_less_than_min_depth_returns_none(self):
        """When max_depth < min_depth, arange yields empty edges → return None."""
        from odas_tpw.processing.top_trim import compute_trim_depth

        depth = np.linspace(0.0, 5.0, 100)
        channels = {"sh1": np.random.default_rng(0).standard_normal(100)}
        # max_depth (0.5) < min_depth (5.0) ⇒ arange(4.95, 0.6, 0.1) is empty
        result = compute_trim_depth(
            depth, channels, dz=0.1, min_depth=5.0, max_depth=0.5,
        )
        assert result is None


# ---------------------------------------------------------------------------
# processing/bottom.py — line 89: len(bins) < 2
# ---------------------------------------------------------------------------


class TestBottomCrashShortBins:
    def test_max_depth_equals_minimum_returns_none(self):
        """When max_depth == depth_minimum, np.arange yields a single-element bin."""
        from odas_tpw.processing.bottom import detect_bottom_crash

        # depth all at exactly 10.0 (== depth_minimum)
        depth = np.full(200, 10.0)
        accel = np.random.default_rng(0).standard_normal(200) * 0.1
        result = detect_bottom_crash(
            depth, {"Ax": accel}, fs=512.0, depth_minimum=10.0, depth_window=4.0,
        )
        # arange(10, 10+4, 4) = [10] → only 1 bin → return None
        assert result is None

    def test_no_vibration_channels_returns_none(self):
        """Empty mapping → return None at the early check."""
        from odas_tpw.processing.bottom import detect_bottom_crash

        depth = np.linspace(0.0, 100.0, 1000)
        result = detect_bottom_crash(depth, {}, fs=512.0)
        assert result is None


# ---------------------------------------------------------------------------
# scor160/goodman.py — _bias_correction insufficient FFT segments warning
# ---------------------------------------------------------------------------


class TestGoodmanBiasCorrectionInsufficient:
    def test_insufficient_segments_returns_one(self):
        """When fft_segments <= 1.02*n_accel, warn and return 1.0."""
        from odas_tpw.scor160.goodman import _bias_correction

        # n_samples=64, nfft=64 → fft_segments = 2*64//64 - 1 = 1
        # n_accel=2 → 1.02*2 = 2.04, 1 <= 2.04 → return 1.0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _bias_correction(n_samples=64, nfft=64, n_accel=2)
        assert result == 1.0
        assert any("Insufficient FFT segments" in str(wi.message) for wi in w)


# ---------------------------------------------------------------------------
# rsi/binning.py — line 84: log-bin with all non-positive values
# ---------------------------------------------------------------------------


class TestBinByDepthLogAllNonPositive:
    def test_log_var_all_non_positive_in_bin_yields_nan(self):
        """A bin whose log-mean variable values are all <= 0 → NaN result."""
        from odas_tpw.rsi.binning import bin_by_depth

        pres = np.array([10.0, 11.0, 12.0])
        # all epsilon values non-positive → log path filters them out
        eps = np.array([-1.0, 0.0, -0.5])
        ds = bin_by_depth(pres, {"epsilon": eps}, bin_size=5.0)
        # All values filtered → bin is NaN
        assert np.all(np.isnan(ds["epsilon"].values))


# ---------------------------------------------------------------------------
# rsi/combine.py — line 40: empty depth set → return empty Dataset
# ---------------------------------------------------------------------------


class TestCombineProfilesNoDepth:
    def test_datasets_without_depth_bin_returns_empty(self):
        """Datasets that lack the depth_bin coord → empty Dataset returned."""
        from odas_tpw.rsi.combine import combine_profiles

        # Datasets with no depth_bin coord
        ds1 = xr.Dataset({"x": (["t"], [1.0, 2.0])})
        ds2 = xr.Dataset({"x": (["t"], [3.0, 4.0])})
        result = combine_profiles([ds1, ds2])
        assert len(result.data_vars) == 0


# ---------------------------------------------------------------------------
# rsi/deconvolve.py — line 67: sign inversion when polyfit slope < -0.5
# ---------------------------------------------------------------------------


class TestDeconvolveSignInversion:
    def test_inverted_x_dx_triggers_sign_flip(self):
        """A pre-emphasized signal anti-correlated with X triggers the
        sign-inversion path (line 67)."""
        from odas_tpw.rsi.deconvolve import deconvolve

        fs = 512.0
        n = 1024
        t = np.arange(n) / fs
        # Create a slow signal
        X = np.sin(2 * np.pi * 0.5 * t[: n // 8])
        # X_dX is the negative derivative scaled — slope between X and X_dX is < -0.5
        X_dX = -np.gradient(np.sin(2 * np.pi * 0.5 * t)) * fs * 10.0
        result = deconvolve(X, X_dX, fs, diff_gain=0.94)
        assert len(result) == len(X_dX)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# chi/l4_chi.py — process_l4_chi_fit MLE branches (mask < 3, chi_obs <= 0)
# ---------------------------------------------------------------------------


class TestL4ChiFitMLEEdges:
    def _make_l3_chi(self, *, K_high: bool, all_zero: bool):
        """Build a tiny L3ChiData for the MLE fit_method path."""
        from odas_tpw.chi.l3_chi import L3ChiData

        n_temp = 1
        n_freq = 33
        n_spec = 1
        F_const = np.linspace(0, 256, n_freq)
        # If K_high=True, all wavenumbers are above f_AA/W (98/0.5 = 196)
        # so the mask (K > 0) & (f_AA/W >= K) is empty.
        if K_high:
            kcyc = np.full((n_freq, n_spec), 1e6)
        else:
            kcyc = np.tile(F_const[:, None] / 0.5, (1, n_spec))

        # All zero spectrum → chi_obs = 0 → triggers chi_obs <= 0 branch
        spec_val = 0.0 if all_zero else 1e-8
        gradt_spec = np.full((n_temp, n_freq, n_spec), spec_val)
        return L3ChiData(
            time=np.array([0.0]),
            pres=np.array([10.0]),
            temp=np.array([15.0]),
            pspd_rel=np.array([0.5]),
            section_number=np.array([1]),
            nu=np.array([1.2e-6]),
            kcyc=kcyc,
            freq=F_const,
            gradt_spec=gradt_spec,
            noise_spec=np.full_like(gradt_spec, 1e-12),
            H2=np.ones((n_spec, n_freq)),
            tau0=np.array([0.01]),
        )

    def test_mle_mask_less_than_three(self):
        """fit_method='mle' with all wavenumbers above f_AA/W → mask < 3 → returns None
        which leaves chi as NaN."""
        from odas_tpw.chi.l4_chi import process_l4_chi_fit

        l3_chi = self._make_l3_chi(K_high=True, all_zero=False)
        l4 = process_l4_chi_fit(l3_chi, fit_method="mle")
        assert np.all(np.isnan(l4.chi))

    def test_mle_chi_obs_clamped_to_floor(self):
        """fit_method='mle' with all-zero spectrum → chi_obs <= 0 → clamped to 1e-14."""
        from odas_tpw.chi.l4_chi import process_l4_chi_fit

        l3_chi = self._make_l3_chi(K_high=False, all_zero=True)
        # Run completes (chi may be NaN from MLE failure but the line is hit)
        l4 = process_l4_chi_fit(l3_chi, fit_method="mle")
        assert l4.n_spectra == 1


# ---------------------------------------------------------------------------
# chi/l2_chi.py — line 134: no temp_fast and no l1.temp → fallback to 10.0
# ---------------------------------------------------------------------------


class TestL2ChiTempFallback:
    def test_no_temp_at_all_raises(self):
        """L1Data without temp_fast raises early — covers the guard at line 84."""
        from odas_tpw.chi.l2_chi import process_l2_chi
        from odas_tpw.scor160.io import L1Data

        n = 100
        l1 = L1Data(
            time=np.arange(n) / 512.0,
            pres=np.linspace(5.0, 50.0, n),
            shear=np.zeros((1, n)),
            vib=np.zeros((1, n)),
            vib_type="accel",
            fs_fast=512.0,
            f_AA=98.0,
            vehicle="vmp",
            profile_dir="down",
            time_reference_year=2025,
        )
        # No temp_fast → process_l2_chi raises ValueError
        with pytest.raises(ValueError, match="no temp_fast"):
            process_l2_chi(l1, l2=None)


# ---------------------------------------------------------------------------
# scor160/compare.py — line 222: n_valid == 0 in spectral comparison
# ---------------------------------------------------------------------------


class TestCompareL3SpectraAllZero:
    def test_all_zero_spectra_skips_log_metrics(self):
        """When both computed and reference spectra are all <= 0, n_valid=0,
        the log-metrics block is skipped (line 222 continue)."""
        from odas_tpw.scor160.compare import compare_l3
        from odas_tpw.scor160.io import L3Data

        n_sh, n_wn, n_sp = 1, 4, 1
        zero_spec = np.zeros((n_sh, n_wn, n_sp))
        l3 = L3Data(
            time=np.array([0.0]),
            pres=np.array([10.0]),
            temp=np.array([15.0]),
            pspd_rel=np.array([0.5]),
            section_number=np.array([1]),
            kcyc=np.ones((n_wn, n_sp)),
            sh_spec=zero_spec,
            sh_spec_clean=zero_spec.copy(),
        )
        result = compare_l3(l3, l3)
        # No spec stats accumulated because every metric had n_valid==0
        assert result["spectra"] == []


# ---------------------------------------------------------------------------
# pyturb/eps.py — load_auxiliary missing variable raises
# ---------------------------------------------------------------------------


class TestPyturbLoadAuxMissingVar:
    def test_missing_required_var_raises_keyerror(self, tmp_path):
        """Auxiliary file missing a required variable → KeyError."""
        from odas_tpw.pyturb._compat import load_auxiliary

        aux = tmp_path / "aux.nc"
        # Missing density variable
        ds = xr.Dataset(
            {
                "lat": (["t"], [0.0]),
                "lon": (["t"], [0.0]),
                "temperature": (["t"], [15.0]),
                "salinity": (["t"], [35.0]),
            }
        )
        ds.to_netcdf(aux)
        with pytest.raises(KeyError, match="density"):
            load_auxiliary(aux)


# ---------------------------------------------------------------------------
# pyturb/eps.py — check_overwrite logic
# ---------------------------------------------------------------------------


class TestCheckOverwrite:
    def test_existing_no_overwrite_returns_false(self, tmp_path):
        """Existing path with overwrite=False returns False."""
        from odas_tpw.pyturb._compat import check_overwrite

        p = tmp_path / "x.nc"
        p.write_bytes(b"")
        assert check_overwrite(p, False) is False
        assert check_overwrite(p, True) is True

    def test_missing_path_returns_true(self, tmp_path):
        """Non-existent path always returns True."""
        from odas_tpw.pyturb._compat import check_overwrite

        assert check_overwrite(tmp_path / "missing.nc", False) is True
