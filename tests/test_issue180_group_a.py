# Sep-2026, Claude and Pat Welch, pat@mousebrains.com
"""Negative controls for issue #180 group A: fail closed on bad input.

Each test is the reviewer's own counterexample, kept in the shape they ran it
so a regression reproduces the original report rather than a paraphrase of it.
F06, F13, F14, F15, F16, F21.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from scipy import signal

from odas_tpw.rsi.binning import bin_by_depth
from odas_tpw.rsi.channels import convert_shear, convert_therm
from odas_tpw.scor160.goodman import _bias_correction
from odas_tpw.scor160.l4 import _estimate_epsilon
from odas_tpw.scor160.nasmyth import nasmyth_grid
from odas_tpw.scor160.spectral import (
    _get_window,
    csd_matrix,
    csd_matrix_batch,
    n_segments,
)


class TestF06MalformedCalibration:
    """A present-but-malformed calibration coefficient must not become a default."""

    def test_decimal_comma_diff_gain_raises(self):
        """`0,09` silently became 1.0: shear x0.09, shear VARIANCE x0.0081."""
        good = convert_shear(np.array([1000.0]), {"sens": ".1", "diff_gain": ".09"})[0]
        assert np.isfinite(good).all()
        with pytest.raises(ValueError, match=r"diff_gain.*unparseable"):
            convert_shear(np.array([1000.0]), {"sens": ".1", "diff_gain": "0,09"})

    def test_missing_diff_gain_still_only_warns(self):
        """Absent stays a warning — legacy corpora genuinely omit keys."""
        with pytest.warns(UserWarning, match="diff_gain"):
            out, _ = convert_shear(np.array([1000.0]), {"sens": ".1"})
        assert np.isfinite(out).all()

    @pytest.mark.parametrize("bad", ["nan", "inf", "0", "-0.09"])
    def test_nonphysical_diff_gain_raises(self, bad):
        with pytest.raises(ValueError, match="diff_gain"):
            convert_shear(np.array([1000.0]), {"sens": ".1", "diff_gain": bad})

    def test_malformed_adc_scaling_raises(self):
        """adc_fs/adc_bits scale every converted sample."""
        with pytest.raises(ValueError, match="adc_fs"):
            convert_shear(np.array([1000.0]), {"sens": ".1", "diff_gain": ".09", "adc_fs": "4,096"})

    def test_malformed_therm_beta_names_the_coefficient(self):
        """beta_2 is a RECIPROCAL: a 0.0 default raised ZeroDivisionError from
        inside an arithmetic expression instead of naming the bad key."""
        params = {"t_0": "290", "beta_1": "3000", "beta_2": "1e30", "g": "6", "e_b": "0.68"}
        ok, _ = convert_therm(np.array([1000.0]), params)
        assert np.isfinite(ok).all()
        with pytest.raises(ValueError, match="beta_2"):
            convert_therm(np.array([1000.0]), {**params, "beta_2": "1,2e30"})


class TestF13AntiAliasCeiling:
    """K_LIMIT_MIN must not readmit bins above the anti-alias wavenumber."""

    def test_corrupting_only_the_excluded_band_no_longer_moves_epsilon(self):
        K = np.arange(0.0, 50.25, 0.25)
        baseline = nasmyth_grid(1e-9, 1e-6, K)
        corrupt = baseline.copy()
        corrupt[(K > 2) & (K <= 7)] *= 5  # ONLY above the declared K_AA = 2 cpm
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = _estimate_epsilon(K, baseline, 1e-6, 2.0, 3)
            corr = _estimate_epsilon(K, corrupt, 1e-6, 2.0, 3)
        # Before the fix: kmax 7.0 cpm on both, epsilon 1.017e-9 -> 7.441e-9.
        assert not np.isfinite(base[0]), "too few trusted bins below K_AA=2: expect a refusal"
        assert not np.isfinite(corr[0])

    def test_warns_naming_the_anti_alias_limit(self):
        K = np.arange(0.0, 50.25, 0.25)
        spec = nasmyth_grid(1e-9, 1e-6, K)
        with pytest.warns(UserWarning, match="anti-alias limit"):
            _estimate_epsilon(K, spec, 1e-6, 2.0, 3)

    def test_production_anti_alias_limit_is_unaffected(self):
        """f_AA = 98 Hz at 0.7 m/s -> K_AA = 126 cpm: nothing changes."""
        K = np.arange(0.0, 150.5, 0.5)
        spec = nasmyth_grid(1e-8, 1e-6, K)
        eps, kmax, *_ = _estimate_epsilon(K, spec, 1e-6, 0.9 * 98.0 / 0.7, 3)
        assert np.isfinite(eps) and eps > 0
        assert kmax <= 0.9 * 98.0 / 0.7


class TestF16OddFFTLength:
    """Odd nfft has no Nyquist bin and a different half-overlap step."""

    @pytest.mark.parametrize("nfft", [63, 64, 255, 256])
    def test_last_bin_matches_scipy_welch(self, nfft):
        rng = np.random.default_rng(7391)
        x = rng.normal(size=(8 * nfft, 1))
        s = csd_matrix(x, None, nfft, 512, detrend="none").Cxy[:, 0, 0].real
        b = csd_matrix_batch(x[None, ...], None, nfft, 512, detrend="none").Cxy[0, :, 0, 0].real
        _, ref = signal.welch(
            x[:, 0],
            fs=512,
            window=_get_window(nfft),
            nperseg=nfft,
            noverlap=nfft // 2,
            detrend=False,
        )
        assert s[-1] / ref[-1] == pytest.approx(1.0, rel=1e-9)
        assert b[-1] / ref[-1] == pytest.approx(1.0, rel=1e-9)

    @pytest.mark.parametrize("nfft", [63, 64])
    def test_segment_count_matches_the_estimator(self, nfft):
        n = 2 * nfft
        overlap = nfft // 2
        actual = 1 + (n - nfft) // (nfft - overlap)
        assert n_segments(n, nfft) == actual
        expected = 1.0 / (1.0 - 1.02 / actual)
        assert _bias_correction(n, nfft, 1) == pytest.approx(expected)

    def test_even_nfft_segment_count_is_unchanged_everywhere(self):
        """The old `2*N//nfft - 1` is EXACTLY right for even nfft — proving the
        parity fix is a no-op on every production configuration."""
        for nfft in range(2, 2050, 2):
            for n in range(nfft, 8 * nfft, max(1, nfft // 5)):
                assert n_segments(n, nfft) == 2 * n // nfft - 1


class TestF14BinnedVariance:
    """E[x^2] - E[x]^2 loses precision that clipping cannot restore."""

    def test_large_offset_small_spread(self):
        from odas_tpw.perturb.binning import _bin_std

        values = 1000.0 + np.linspace(-1e-5, 1e-5, 1000)
        got = _bin_std(values, np.zeros(1000), np.array([-0.5, 0.5]))[0]
        assert got == pytest.approx(np.std(values), rel=1e-9)

    def test_offset_invariance(self):
        from odas_tpw.perturb.binning import _bin_std

        rng = np.random.default_rng(11)
        base = rng.normal(0.0, 1e-4, 500)
        coords = np.zeros(500)
        edges = np.array([-0.5, 0.5])
        shifted = base + 1.7e9  # epoch-seconds-like offset
        a = _bin_std(base, coords, edges)[0]
        b = _bin_std(shifted, coords, edges)[0]
        # Exact against a centered reference on the SAME (quantized) input --
        # this is the algorithm. The single-pass form returned 0.0 here: the
        # cancellation error (~1e-16 * 2.9e18) swamps a variance of 9.4e-9.
        assert b == pytest.approx(float(np.std(shifted)), rel=1e-6)
        # And loosely against the unshifted case: what is left is the float64
        # representation of the shifted samples, not the estimator.
        assert b == pytest.approx(a, rel=1e-4)

    def test_constant_bin_is_zero_not_negative(self):
        from odas_tpw.perturb.binning import _bin_std

        got = _bin_std(np.full(10, 3.25), np.zeros(10), np.array([-0.5, 0.5]))[0]
        assert got == pytest.approx(0.0, abs=1e-15)


class TestF15DimensionAlignment:
    """Equal dimension length is not sample alignment."""

    def test_unrelated_axis_is_not_depth_binned(self, tmp_path: Path):
        from odas_tpw.perturb.binning import _bin_snapshot, _load_profile_snapshot

        path = tmp_path / "profile.nc"
        xr.Dataset(
            {
                "depth": ("time", [1.0, 2.0, 3.0]),
                "temperature": ("time", [10.0, 11.0, 12.0]),
                "calibration_coeff": ("coefficient", [100.0, 200.0, 300.0]),
            }
        ).to_netcdf(path)
        snap = _load_profile_snapshot(path)
        assert "calibration_coeff" not in snap["vars"]
        assert "temperature" in snap["vars"]
        out = _bin_snapshot(snap, np.array([0.0, 2.0, 4.0]), np.nanmean, False)
        assert "calibration_coeff" not in out["vars"]


class TestF21DegenerateBinRange:
    """Constant pressure on an exact bin boundary produced zero bins."""

    def test_single_sample_on_a_bin_edge(self):
        ds = bin_by_depth(np.array([10.0]), {"epsilon": np.array([1e-7])})
        assert ds.sizes["depth_bin"] == 1
        assert float(ds["epsilon"].values[0]) == pytest.approx(1e-7)

    def test_repeated_identical_pressures(self):
        ds = bin_by_depth(np.full(5, 10.0), {"epsilon": np.full(5, 1e-7)})
        assert ds.sizes["depth_bin"] == 1
        assert float(ds["epsilon"].values[0]) == pytest.approx(1e-7)

    def test_off_grid_singleton_still_works(self):
        ds = bin_by_depth(np.array([10.4]), {"epsilon": np.array([1e-7])})
        assert ds.sizes["depth_bin"] >= 1
        assert np.isfinite(ds["epsilon"].values).any()
