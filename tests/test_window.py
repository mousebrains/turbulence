# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for the per-window epsilon and chi computation (window.py)."""

import numpy as np
import pytest
from microstructure_tpw.rsi.window import (
    ChiWindowResult,
    EpsWindowResult,
    compute_chi_window,
    compute_eps_window,
)

from microstructure_tpw.scor160.ocean import visc35

# ---------------------------------------------------------------------------
# compute_eps_window
# ---------------------------------------------------------------------------


class TestComputeEpsWindow:
    """Tests for compute_eps_window using synthetic white noise shear data."""

    def _make_eps_inputs(self, diss_length=512, fft_length=256, n_shear=2, n_accel=3):
        """Create synthetic inputs for compute_eps_window."""
        rng = np.random.default_rng(42)
        shear = rng.standard_normal((diss_length, n_shear)) * 1e-2
        accel = rng.standard_normal((diss_length, n_accel)) * 0.01
        speed = np.full(diss_length, 0.6)
        P = np.linspace(10, 12, diss_length)
        T_mean = 15.0
        fs_fast = 512.0
        f_AA = 98.0
        return shear, accel, speed, P, T_mean, fs_fast, fft_length, f_AA

    def test_output_shapes(self):
        """Verify output array shapes match expected dimensions."""
        shear, accel, speed, P, T_mean, fs_fast, fft_length, f_AA = self._make_eps_inputs()
        result = compute_eps_window(shear, accel, speed, P, T_mean, fs_fast, fft_length, f_AA)

        assert isinstance(result, EpsWindowResult)
        n_shear = 2
        n_freq = fft_length // 2 + 1  # 129

        # Per-probe arrays
        assert result.epsilon.shape == (n_shear,)
        assert result.K_max.shape == (n_shear,)
        assert result.mad.shape == (n_shear,)
        assert result.fom.shape == (n_shear,)
        assert result.FM.shape == (n_shear,)
        assert result.K_max_ratio.shape == (n_shear,)
        assert result.method.shape == (n_shear,)

        # Spectral arrays
        assert result.K.shape == (n_freq,)
        assert result.F.shape == (n_freq,)

        # Per-probe spectral lists
        assert len(result.shear_specs) == n_shear
        assert len(result.nasmyth_specs) == n_shear
        for spec in result.shear_specs:
            assert spec.shape == (n_freq,)

        # Scalars
        assert result.W > 0
        assert result.nu > 0
        assert np.isfinite(result.mean_P)
        assert np.isfinite(result.mean_T)

    def test_positive_epsilon(self):
        """Epsilon values should be positive and finite for white noise input."""
        shear, accel, speed, P, T_mean, fs_fast, fft_length, f_AA = self._make_eps_inputs()
        result = compute_eps_window(shear, accel, speed, P, T_mean, fs_fast, fft_length, f_AA)

        assert np.all(np.isfinite(result.epsilon))
        assert np.all(result.epsilon > 0)

    def test_speed_warning(self):
        """Very low speed should trigger a UserWarning about clamping."""
        shear, accel, speed, P, T_mean, fs_fast, fft_length, f_AA = self._make_eps_inputs()
        speed_low = np.full(len(speed), 0.001)

        with pytest.warns(UserWarning, match="clamped"):
            result = compute_eps_window(
                shear, accel, speed_low, P, T_mean, fs_fast, fft_length, f_AA
            )

        # Should still produce a result with the clamped speed
        assert result.W >= 0.01  # SPEED_MIN

    def test_no_goodman(self):
        """Setting do_goodman=False with accel=None should still work."""
        shear, _, speed, P, T_mean, fs_fast, fft_length, f_AA = self._make_eps_inputs()
        result = compute_eps_window(
            shear, None, speed, P, T_mean, fs_fast, fft_length, f_AA, do_goodman=False
        )

        assert isinstance(result, EpsWindowResult)
        assert result.epsilon.shape == (2,)
        assert np.all(np.isfinite(result.epsilon))
        assert np.all(result.epsilon > 0)

    def test_short_segment(self):
        """Data shorter than 2*fft_length should return NaN epsilon, not crash."""
        fft_length = 256
        diss_length = fft_length  # shorter than 2*fft_length
        shear, accel, speed, P, T_mean, fs_fast, _, f_AA = self._make_eps_inputs(
            diss_length=diss_length, fft_length=fft_length
        )

        result = compute_eps_window(shear, accel, speed, P, T_mean, fs_fast, fft_length, f_AA)

        assert isinstance(result, EpsWindowResult)
        assert result.epsilon.shape == (2,)
        assert np.all(np.isnan(result.epsilon))
        assert np.all(np.isnan(result.K_max))
        assert np.all(np.isnan(result.fom))


# ---------------------------------------------------------------------------
# compute_chi_window
# ---------------------------------------------------------------------------


class TestComputeChiWindow:
    """Tests for compute_chi_window using synthetic temperature gradient data."""

    def _make_chi_inputs(self, diss_length=512, fft_length=256, n_therm=1):
        """Create synthetic inputs for compute_chi_window."""
        rng = np.random.default_rng(42)
        therm_segs = [rng.standard_normal(diss_length) * 0.001 for _ in range(n_therm)]
        diff_gains = [0.94] * n_therm
        W = 0.6
        T_mean = 15.0
        nu = float(visc35(T_mean))
        fs_fast = 512.0
        f_AA = 98.0
        return therm_segs, diff_gains, W, T_mean, nu, fs_fast, fft_length, f_AA

    def test_output_shapes(self):
        """Verify output array shapes for method=2."""
        therm_segs, diff_gains, W, T_mean, nu, fs_fast, fft_length, f_AA = (
            self._make_chi_inputs()
        )
        result = compute_chi_window(
            therm_segs, diff_gains, W, T_mean, nu, fs_fast, fft_length, f_AA,
            method=2,
        )

        assert isinstance(result, ChiWindowResult)
        n_therm = 1
        n_freq = fft_length // 2 + 1  # 129

        # Per-probe arrays
        assert result.chi.shape == (n_therm,)
        assert result.kB.shape == (n_therm,)
        assert result.K_max_T.shape == (n_therm,)
        assert result.fom.shape == (n_therm,)
        assert result.K_max_ratio.shape == (n_therm,)

        # Spectral arrays
        assert result.K.shape == (n_freq,)
        assert result.F.shape == (n_freq,)
        assert result.H2.shape == (n_freq,)

        # Per-probe spectral lists
        assert len(result.grad_specs) == n_therm
        assert len(result.model_specs) == n_therm
        assert len(result.model_specs_raw) == n_therm

        # Noise spectrum
        assert result.noise_K is not None
        assert result.noise_K.shape == (n_freq,)

    def test_method1_with_epsilon(self):
        """Method 1 with provided epsilon should produce finite chi."""
        therm_segs, diff_gains, W, T_mean, nu, fs_fast, fft_length, f_AA = (
            self._make_chi_inputs()
        )
        epsilon = np.array([1e-7])  # realistic epsilon for method 1

        result = compute_chi_window(
            therm_segs, diff_gains, W, T_mean, nu, fs_fast, fft_length, f_AA,
            epsilon=epsilon, method=1,
        )

        assert isinstance(result, ChiWindowResult)
        assert np.all(np.isfinite(result.chi))
        assert np.all(result.chi > 0)

    def test_method2_iterative(self):
        """Method 2 (iterative fit) should produce finite chi."""
        therm_segs, diff_gains, W, T_mean, nu, fs_fast, fft_length, f_AA = (
            self._make_chi_inputs()
        )

        result = compute_chi_window(
            therm_segs, diff_gains, W, T_mean, nu, fs_fast, fft_length, f_AA,
            method=2,
        )

        assert isinstance(result, ChiWindowResult)
        assert np.all(np.isfinite(result.chi))
        assert np.all(result.chi > 0)

    def test_short_segment(self):
        """Data shorter than 2*fft_length should return NaN chi, not crash."""
        fft_length = 256
        diss_length = fft_length  # shorter than 2*fft_length
        therm_segs, diff_gains, W, T_mean, nu, fs_fast, _, f_AA = self._make_chi_inputs(
            diss_length=diss_length, fft_length=fft_length
        )

        result = compute_chi_window(
            therm_segs, diff_gains, W, T_mean, nu, fs_fast, fft_length, f_AA,
            method=2,
        )

        assert isinstance(result, ChiWindowResult)
        assert result.chi.shape == (1,)
        assert np.all(np.isnan(result.chi))
        assert np.all(np.isnan(result.kB))
        assert np.all(np.isnan(result.fom))
