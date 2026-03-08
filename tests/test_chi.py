# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for the chi (thermal variance dissipation) pipeline modules."""

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# batchelor.py
# ---------------------------------------------------------------------------


class TestBatchelorKB:
    def test_higher_eps_higher_kB(self):
        from rsi_python.batchelor import batchelor_kB

        nu = 1.2e-6
        kB_low = batchelor_kB(1e-9, nu)
        kB_high = batchelor_kB(1e-5, nu)
        assert kB_high > kB_low

    def test_known_value(self):
        """Check kB against hand calculation."""
        from rsi_python.batchelor import KAPPA_T, batchelor_kB

        nu = 1.2e-6
        eps = 1e-7
        expected = (1 / (2 * np.pi)) * (eps / (nu * KAPPA_T**2)) ** 0.25
        np.testing.assert_allclose(batchelor_kB(eps, nu), expected)


class TestBatchelorNondim:
    def test_shape_and_positive(self):
        from rsi_python.batchelor import batchelor_nondim

        alpha = np.linspace(0.01, 10, 200)
        f = batchelor_nondim(alpha)
        assert f.shape == (200,)
        assert np.all(f >= 0)

    def test_peak_near_alpha_1(self):
        from rsi_python.batchelor import batchelor_nondim

        alpha = np.linspace(0.01, 5, 1000)
        f = batchelor_nondim(alpha)
        peak_alpha = alpha[np.argmax(f)]
        # Peak should be near alpha ~1
        assert 0.3 < peak_alpha < 2.0, f"Peak at alpha={peak_alpha}"


class TestBatchelorGrad:
    def test_integral_equals_chi_over_6kT(self):
        """Batchelor gradient spectrum should integrate to chi/(6*kappa_T)."""
        from rsi_python.batchelor import KAPPA_T, batchelor_grad, batchelor_kB

        chi = 1e-7
        nu = 1.2e-6
        eps = 1e-7
        kB = batchelor_kB(eps, nu)

        k = np.linspace(0.01, kB * 5, 100000)
        S = batchelor_grad(k, kB, chi)
        integral = np.trapezoid(S, k)
        expected = chi / (6 * KAPPA_T)
        np.testing.assert_allclose(integral, expected, rtol=0.01)

    def test_higher_eps_broader(self):
        from rsi_python.batchelor import batchelor_grad, batchelor_kB

        chi = 1e-7
        nu = 1.2e-6
        kB_low = batchelor_kB(1e-9, nu)
        kB_high = batchelor_kB(1e-5, nu)
        k = np.logspace(0, 3, 100)
        S_low = batchelor_grad(k, kB_low, chi)
        S_high = batchelor_grad(k, kB_high, chi)
        # Higher kB should shift spectrum to higher wavenumbers
        mean_k_low = np.average(k, weights=S_low + 1e-30)
        mean_k_high = np.average(k, weights=S_high + 1e-30)
        assert mean_k_high > mean_k_low


class TestKraichnanGrad:
    def test_integral_equals_chi_over_6kT(self):
        """Kraichnan gradient spectrum should integrate to chi/(6*kappa_T)."""
        from rsi_python.batchelor import KAPPA_T, batchelor_kB, kraichnan_grad

        chi = 1e-7
        nu = 1.2e-6
        eps = 1e-7
        kB = batchelor_kB(eps, nu)

        k = np.linspace(0.01, kB * 10, 100000)
        S = kraichnan_grad(k, kB, chi)
        integral = np.trapezoid(S, k)
        expected = chi / (6 * KAPPA_T)
        np.testing.assert_allclose(integral, expected, rtol=0.01)

    def test_exponential_vs_gaussian_rolloff(self):
        """Kraichnan should have more power at high wavenumbers than Batchelor."""
        from rsi_python.batchelor import batchelor_grad, batchelor_kB, kraichnan_grad

        chi = 1e-7
        nu = 1.2e-6
        eps = 1e-7
        kB = batchelor_kB(eps, nu)
        # At 2*kB, Kraichnan should be higher (exponential vs Gaussian rolloff)
        k_high = np.array([2.0 * kB])
        S_batch = batchelor_grad(k_high, kB, chi)
        S_kraich = kraichnan_grad(k_high, kB, chi)
        assert S_kraich[0] > S_batch[0], "Kraichnan should exceed Batchelor at k = 2*kB"


# ---------------------------------------------------------------------------
# fp07.py
# ---------------------------------------------------------------------------


class TestFP07Transfer:
    def test_unity_at_dc(self):
        from rsi_python.fp07 import fp07_transfer

        H2 = fp07_transfer(np.array([0.0]), 0.01)
        np.testing.assert_allclose(H2, 1.0)

    def test_rolloff(self):
        from rsi_python.fp07 import fp07_transfer

        f = np.logspace(-1, 3, 100)
        H2 = fp07_transfer(f, 0.01)
        assert H2[0] > H2[-1]
        # Should be monotonically decreasing
        assert np.all(np.diff(H2) <= 0)

    def test_double_pole_faster_rolloff(self):
        from rsi_python.fp07 import fp07_double_pole, fp07_transfer

        f = np.array([50.0])
        H2_single = fp07_transfer(f, 0.01)
        H2_double = fp07_double_pole(f, 0.01)
        assert H2_double[0] < H2_single[0]


class TestFP07Tau:
    def test_lueck(self):
        from rsi_python.fp07 import fp07_tau

        tau = fp07_tau(1.0, model="lueck")
        assert 0.005 < tau < 0.02

    def test_speed_dependence(self):
        from rsi_python.fp07 import fp07_tau

        tau_slow = fp07_tau(0.5)
        tau_fast = fp07_tau(1.5)
        assert tau_slow > tau_fast  # slower speed = larger time constant


class TestNoiseModel:
    def test_noise_positive(self):
        from rsi_python.fp07 import noise_thermchannel

        F = np.logspace(-1, 2, 50)
        noise = noise_thermchannel(F, 10.0)
        assert np.all(noise > 0)

    def test_noise_shape(self):
        from rsi_python.fp07 import noise_thermchannel

        F = np.logspace(-1, 2, 50)
        noise = noise_thermchannel(F, 10.0)
        assert noise.shape == (50,)


# ---------------------------------------------------------------------------
# chi.py — synthetic spectrum recovery
# ---------------------------------------------------------------------------


class TestChiFromEpsilon:
    def test_synthetic_batchelor_recovery(self):
        """Method 1: recover chi from a synthetic Batchelor spectrum with FP07 rolloff."""
        from rsi_python.batchelor import batchelor_grad, batchelor_kB
        from rsi_python.chi import _chi_from_epsilon
        from rsi_python.fp07 import fp07_tau, fp07_transfer

        chi_true = 1e-7
        eps_true = 1e-7
        nu = 1.2e-6
        kB = batchelor_kB(eps_true, nu)
        speed = 0.7

        fs = 512
        fft_length = 512
        n_freq = fft_length // 2 + 1
        F = np.arange(n_freq) * fs / fft_length
        K = F / speed

        # Synthetic observed spectrum with FP07 rolloff applied
        spec_true = batchelor_grad(K, kB, chi_true)
        tau0 = fp07_tau(speed)
        H2 = fp07_transfer(F, tau0)
        spec_obs = spec_true * H2
        spec_obs = np.maximum(spec_obs, 1e-20)

        chi_est, kB_est, K_max, _, fom, K_max_ratio = _chi_from_epsilon(
            spec_obs,
            K,
            eps_true,
            nu,
            10.0,
            speed,
            98.0,
            "single_pole",
            "batchelor",
            fs,
            0.94,
            fft_length,
        )

        assert np.isfinite(chi_est)
        # Should recover within a factor of 3
        ratio = chi_est / chi_true
        assert 0.3 < ratio < 3.0, f"chi ratio = {ratio}"


class TestMLEFit:
    def test_synthetic_recovery(self):
        """Method 2 MLE: fit kB from synthetic Batchelor spectrum."""
        from rsi_python.batchelor import batchelor_grad, batchelor_kB
        from rsi_python.chi import _mle_fit_kB

        chi_true = 1e-7
        eps_true = 1e-7
        nu = 1.2e-6
        kB_true = batchelor_kB(eps_true, nu)
        speed = 0.7

        fs = 512
        fft_length = 512
        n_freq = fft_length // 2 + 1
        F = np.arange(n_freq) * fs / fft_length
        K = F / speed

        spec_obs = batchelor_grad(K, kB_true, chi_true)
        spec_obs = np.maximum(spec_obs, 1e-20)

        kB_fit, chi_fit, eps_fit, K_max, _, fom, K_max_ratio = _mle_fit_kB(
            spec_obs,
            K,
            chi_true,
            nu,
            10.0,
            speed,
            98.0,
            "single_pole",
            "batchelor",
            fs,
            0.94,
            fft_length,
        )

        assert np.isfinite(kB_fit)
        # kB should be within 50% (broader tolerance for grid search)
        ratio = kB_fit / kB_true
        assert 0.5 < ratio < 2.0, f"kB ratio = {ratio}"


# ---------------------------------------------------------------------------
# Integration test on real VMP data
# ---------------------------------------------------------------------------

TEST_DATA_DIR = Path(__file__).parent / "data"
PROFILE_FILE = TEST_DATA_DIR / "SN479_0006.p"


class TestChiIntegration:
    @pytest.fixture
    def skip_no_data(self):
        if not PROFILE_FILE.exists():
            pytest.skip("Test data not available")

    def test_get_chi_method2(self, skip_no_data):
        """Method 2 (no epsilon) should produce chi in reasonable range."""
        from rsi_python.chi import get_chi

        results = get_chi(PROFILE_FILE, fft_length=512)
        assert len(results) >= 1
        ds = results[0]

        chi = ds["chi"].values
        valid = chi[np.isfinite(chi) & (chi > 0)]
        assert len(valid) > 0, "No valid chi estimates"
        # Chi should be in ~1e-11 to 1e-4 range for ocean data
        assert np.min(valid) > 1e-14, f"min chi too small: {np.min(valid)}"
        assert np.max(valid) < 1e0, f"max chi too large: {np.max(valid)}"

    def test_get_chi_method1(self, skip_no_data):
        """Method 1 (with epsilon) should produce chi in reasonable range."""
        from rsi_python.chi import get_chi
        from rsi_python.dissipation import get_diss

        # First compute epsilon
        eps_results = get_diss(PROFILE_FILE, fft_length=256, goodman=True)
        assert len(eps_results) >= 1

        # Then compute chi using epsilon
        results = get_chi(PROFILE_FILE, epsilon_ds=eps_results[0], fft_length=512)
        assert len(results) >= 1
        ds = results[0]

        chi = ds["chi"].values
        valid = chi[np.isfinite(chi) & (chi > 0)]
        assert len(valid) > 0, "No valid chi estimates"
        assert np.min(valid) > 1e-14
        assert np.max(valid) < 1e0

    def test_output_dataset_structure(self, skip_no_data):
        """Output dataset should have expected variables and dimensions."""
        from rsi_python.chi import get_chi

        results = get_chi(PROFILE_FILE, fft_length=512)
        ds = results[0]

        # Check required variables exist (including new QC variables)
        for var in [
            "chi",
            "epsilon_T",
            "kB",
            "K_max_T",
            "fom",
            "K_max_ratio",
            "speed",
            "nu",
            "P_mean",
            "T_mean",
            "spec_gradT",
            "spec_batch",
            "spec_noise",
            "K",
            "F",
        ]:
            assert var in ds, f"Missing variable: {var}"

        # Check dimensions
        assert "probe" in ds.dims
        assert "time" in ds.dims
        assert "freq" in ds.dims

    def test_qc_variables_in_output(self, skip_no_data):
        """Output should contain fom and K_max_ratio QC variables."""
        from rsi_python.chi import get_chi

        results = get_chi(PROFILE_FILE, fft_length=512)
        ds = results[0]
        assert "fom" in ds, "Missing fom variable"
        assert "K_max_ratio" in ds, "Missing K_max_ratio variable"
        # Should have same shape as chi
        assert ds["fom"].shape == ds["chi"].shape
        assert ds["K_max_ratio"].shape == ds["chi"].shape

    def test_chi_file_output(self, skip_no_data, tmp_path):
        """compute_chi_file should write valid NetCDF."""
        from rsi_python.chi import compute_chi_file

        out_paths = compute_chi_file(PROFILE_FILE, tmp_path, fft_length=512)
        assert len(out_paths) >= 1

        import xarray as xr

        ds = xr.open_dataset(out_paths[0])
        assert "chi" in ds
        assert "kB" in ds
        ds.close()

    def test_python_api(self, skip_no_data):
        """Top-level import should work."""
        from rsi_python import get_chi

        results = get_chi(PROFILE_FILE, fft_length=512)
        assert len(results) >= 1
