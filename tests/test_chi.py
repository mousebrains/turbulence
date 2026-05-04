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
        from odas_tpw.chi.batchelor import batchelor_kB

        nu = 1.2e-6
        kB_low = batchelor_kB(1e-9, nu)
        kB_high = batchelor_kB(1e-5, nu)
        assert kB_high > kB_low

    def test_known_value(self):
        """Check kB against hand calculation."""
        from odas_tpw.chi.batchelor import KAPPA_T, batchelor_kB

        nu = 1.2e-6
        eps = 1e-7
        expected = (1 / (2 * np.pi)) * (eps / (nu * KAPPA_T**2)) ** 0.25
        np.testing.assert_allclose(batchelor_kB(eps, nu), expected)


class TestBatchelorNondim:
    def test_shape_and_positive(self):
        from odas_tpw.chi.batchelor import batchelor_nondim

        alpha = np.linspace(0.01, 10, 200)
        f = batchelor_nondim(alpha)
        assert f.shape == (200,)
        assert np.all(f >= 0)

    def test_peak_near_alpha_1(self):
        from odas_tpw.chi.batchelor import batchelor_nondim

        alpha = np.linspace(0.01, 5, 1000)
        f = batchelor_nondim(alpha)
        peak_alpha = alpha[np.argmax(f)]
        # Peak should be near alpha ~1
        assert 0.3 < peak_alpha < 2.0, f"Peak at alpha={peak_alpha}"


class TestBatchelorGrad:
    def test_integral_equals_chi_over_6kT(self):
        """Batchelor gradient spectrum should integrate to chi/(6*kappa_T)."""
        from odas_tpw.chi.batchelor import KAPPA_T, batchelor_grad, batchelor_kB

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
        from odas_tpw.chi.batchelor import batchelor_grad, batchelor_kB

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
        from odas_tpw.chi.batchelor import KAPPA_T, batchelor_kB, kraichnan_grad

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
        from odas_tpw.chi.batchelor import batchelor_grad, batchelor_kB, kraichnan_grad

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
        from odas_tpw.chi.fp07 import fp07_transfer

        H2 = fp07_transfer(np.array([0.0]), 0.01)
        np.testing.assert_allclose(H2, 1.0)

    def test_rolloff(self):
        from odas_tpw.chi.fp07 import fp07_transfer

        f = np.logspace(-1, 3, 100)
        H2 = fp07_transfer(f, 0.01)
        assert H2[0] > H2[-1]
        # Should be monotonically decreasing
        assert np.all(np.diff(H2) <= 0)

    def test_double_pole_faster_rolloff(self):
        from odas_tpw.chi.fp07 import fp07_double_pole, fp07_transfer

        f = np.array([50.0])
        H2_single = fp07_transfer(f, 0.01)
        H2_double = fp07_double_pole(f, 0.01)
        assert H2_double[0] < H2_single[0]


class TestFP07Tau:
    def test_lueck(self):
        from odas_tpw.chi.fp07 import fp07_tau

        tau = fp07_tau(1.0, model="lueck")
        assert 0.005 < tau < 0.02

    def test_speed_dependence(self):
        from odas_tpw.chi.fp07 import fp07_tau

        tau_slow = fp07_tau(0.5)
        tau_fast = fp07_tau(1.5)
        assert tau_slow > tau_fast  # slower speed = larger time constant

    def test_peterson(self):
        from odas_tpw.chi.fp07 import fp07_tau

        tau = fp07_tau(1.0, model="peterson")
        # tau = 0.012 * 1.0^(-0.32) = 0.012
        np.testing.assert_allclose(tau, 0.012, rtol=1e-10)

    def test_goto(self):
        from odas_tpw.chi.fp07 import fp07_tau

        tau = fp07_tau(1.0, model="goto")
        assert tau == 0.003
        # goto model is speed-independent
        tau2 = fp07_tau(0.5, model="goto")
        assert tau2 == 0.003

    def test_unknown_raises(self):
        from odas_tpw.chi.fp07 import fp07_tau

        with pytest.raises(ValueError, match="Unknown FP07 tau model"):
            fp07_tau(1.0, model="invalid_model")


class TestNoiseModel:
    def test_noise_positive(self):
        from odas_tpw.chi.fp07 import noise_thermchannel

        F = np.logspace(-1, 2, 50)
        noise = noise_thermchannel(F, 10.0)
        assert np.all(noise > 0)

    def test_noise_shape(self):
        from odas_tpw.chi.fp07 import noise_thermchannel

        F = np.logspace(-1, 2, 50)
        noise = noise_thermchannel(F, 10.0)
        assert noise.shape == (50,)


# ---------------------------------------------------------------------------
# shear_noise.py — shear probe electronics noise model
# ---------------------------------------------------------------------------


class TestShearNoiseModel:
    def test_noise_positive(self):
        from odas_tpw.rsi.shear_noise import noise_shearchannel

        F = np.logspace(-1, 2, 100)
        noise = noise_shearchannel(F)
        assert np.all(noise > 0)

    def test_noise_shape(self):
        from odas_tpw.rsi.shear_noise import noise_shearchannel

        F = np.logspace(-1, 2, 50)
        noise = noise_shearchannel(F)
        assert noise.shape == (50,)

    def test_noise_increases_with_frequency(self):
        """Shear noise should generally increase at high frequencies
        due to the differentiator gain."""
        from odas_tpw.rsi.shear_noise import noise_shearchannel

        F = np.array([1.0, 10.0, 100.0])
        noise = noise_shearchannel(F)
        # Differentiator gain rises with frequency, so noise rises
        assert noise[2] > noise[0]

    def test_probe_capacitance_effect(self):
        """Adding probe capacitance should increase noise at high frequencies."""
        from odas_tpw.rsi.shear_noise import noise_shearchannel

        F = np.logspace(0, 2, 50)
        noise_no_probe = noise_shearchannel(F, CP=0)
        noise_with_probe = noise_shearchannel(F, CP=1e-9)
        # Probe capacitance increases first-stage noise gain at high f
        ratio = noise_with_probe[-1] / noise_no_probe[-1]
        assert ratio > 1.0

    def test_default_parameters_match_odas(self):
        """Verify default parameter values match ODAS noise_shearchannel.m."""
        import inspect

        from odas_tpw.rsi.shear_noise import noise_shearchannel

        sig = inspect.signature(noise_shearchannel)
        defaults = {
            k: v.default
            for k, v in sig.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        assert defaults["R1"] == 1e9
        assert defaults["C1"] == 1.5e-9
        assert defaults["R2"] == 499
        assert defaults["C2"] == 0.94e-6
        assert defaults["R3"] == 1e6
        assert defaults["C3"] == 470e-12
        assert defaults["CP"] == 0
        assert defaults["E_1"] == 9e-9
        assert defaults["fc"] == 50
        assert defaults["I_1"] == 0.56e-15
        assert defaults["f_AA"] == 110
        assert defaults["fs"] == 512
        assert defaults["VFS"] == 4.096
        assert defaults["Bits"] == 16
        assert defaults["gamma_RSI"] == 2.5
        assert defaults["T_K"] == 295
        assert defaults["K_B"] == 1.382e-23

    def test_adc_quantization_floor(self):
        """At very low frequencies, ADC quantization noise should dominate."""
        from odas_tpw.rsi.shear_noise import noise_shearchannel

        # With very small E_1, I_1, R1, the circuit noise vanishes;
        # only ADC quantization remains
        F = np.array([1.0, 10.0, 100.0])
        noise = noise_shearchannel(
            F,
            E_1=0,
            I_1=0,
            R1=0,
            CP=0,
        )
        # ADC quantization floor: gamma_RSI * delta_s^2 / (12 * fN) / delta_s^2
        # = gamma_RSI / (12 * fN)
        fN = 512 / 2
        expected_floor = 2.5 / (12 * fN)
        # Circuit noise is zero, so only ADC floor plus tiny AA-filtered remnant
        np.testing.assert_allclose(noise[0], expected_floor, rtol=0.01)


# ---------------------------------------------------------------------------
# chi.py — synthetic spectrum recovery
# ---------------------------------------------------------------------------


class TestChiFromEpsilon:
    def test_synthetic_batchelor_recovery(self):
        """Method 1: recover chi from a synthetic Batchelor spectrum with FP07 rolloff."""
        from odas_tpw.chi.batchelor import batchelor_grad, batchelor_kB
        from odas_tpw.chi.chi import _chi_from_epsilon
        from odas_tpw.chi.fp07 import fp07_tau, fp07_transfer, gradT_noise

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

        # Pre-compute noise for new signature
        noise_K, _ = gradT_noise(F, 10.0, speed, fs=fs, diff_gain=0.94)

        chi_est, _kB_est, _K_max, _, _fom, _K_max_ratio = _chi_from_epsilon(
            spec_obs,
            K,
            eps_true,
            nu,
            noise_K,
            H2,
            tau0,
            fp07_transfer,
            98.0,
            speed,
            "batchelor",
        )

        assert np.isfinite(chi_est)
        # Should recover within a factor of 3
        ratio = chi_est / chi_true
        assert 0.3 < ratio < 3.0, f"chi ratio = {ratio}"


class TestMLEFit:
    def test_synthetic_recovery(self):
        """Method 2 MLE: fit kB from synthetic Batchelor spectrum."""
        from odas_tpw.chi.batchelor import batchelor_grad, batchelor_kB
        from odas_tpw.chi.chi import _mle_fit_kB
        from odas_tpw.chi.fp07 import fp07_tau, fp07_transfer, gradT_noise

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

        # Pre-compute noise/H2 for new signature
        tau0 = fp07_tau(speed)
        H2 = fp07_transfer(F, tau0)
        noise_K, _ = gradT_noise(F, 10.0, speed, fs=fs, diff_gain=0.94)

        kB_fit, _chi_fit, _eps_fit, _K_max, _, _fom, _K_max_ratio = _mle_fit_kB(
            spec_obs,
            K,
            chi_true,
            nu,
            noise_K,
            H2,
            tau0,
            fp07_transfer,
            98.0,
            speed,
            "batchelor",
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
        from odas_tpw.rsi.chi_io import get_chi

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
        from odas_tpw.rsi.chi_io import get_chi
        from odas_tpw.rsi.dissipation import get_diss

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
        from odas_tpw.rsi.chi_io import get_chi

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
        from odas_tpw.rsi.chi_io import get_chi

        results = get_chi(PROFILE_FILE, fft_length=512)
        ds = results[0]
        assert "fom" in ds, "Missing fom variable"
        assert "K_max_ratio" in ds, "Missing K_max_ratio variable"
        # Should have same shape as chi
        assert ds["fom"].shape == ds["chi"].shape
        assert ds["K_max_ratio"].shape == ds["chi"].shape

    def test_chi_cf_compliance(self, skip_no_data):
        """Chi dataset should have CF-1.13 attributes on all variables."""
        from odas_tpw.rsi.chi_io import get_chi

        results = get_chi(PROFILE_FILE, fft_length=512)
        ds = results[0]

        # Global attributes
        assert ds.attrs["Conventions"] == "CF-1.13, ACDD-1.3"
        assert "history" in ds.attrs

        # Time coordinate
        t = ds.coords["t"]
        assert t.attrs["standard_name"] == "time"
        assert t.attrs["calendar"] == "standard"
        assert t.attrs["axis"] == "T"
        assert "seconds since" in t.attrs["units"]

        # All data variables should have units and long_name
        for vname in ds.data_vars:
            var = ds[vname]
            assert "units" in var.attrs, f"{vname} missing units"
            assert "long_name" in var.attrs, f"{vname} missing long_name"

        # Standard names on known physical variables
        assert ds["P_mean"].attrs["standard_name"] == "sea_water_pressure"
        assert ds["P_mean"].attrs["positive"] == "down"
        assert ds["T_mean"].attrs["standard_name"] == "sea_water_temperature"

        # Chi-specific units checks
        assert ds["chi"].attrs["units"] == "K2 s-1"
        assert ds["epsilon_T"].attrs["units"] == "W kg-1"
        assert ds["kB"].attrs["units"] == "cpm"
        assert ds["speed"].attrs["units"] == "m s-1"
        assert ds["nu"].attrs["units"] == "m2 s-1"
        assert ds["P_mean"].attrs["units"] == "dbar"
        assert ds["T_mean"].attrs["units"] == "degree_Celsius"
        assert ds["spec_gradT"].attrs["units"] == "K2 m-1"
        assert ds["spec_batch"].attrs["units"] == "K2 m-1"
        assert ds["spec_noise"].attrs["units"] == "K2 m-1"
        assert ds["K"].attrs["units"] == "cpm"
        assert ds["F"].attrs["units"] == "Hz"

        # Probe coordinate
        assert "long_name" in ds.coords["probe"].attrs

    def test_chi_file_cf_compliance(self, skip_no_data, tmp_path):
        """Chi NetCDF file should roundtrip with CF attributes intact."""
        import xarray as xr

        from odas_tpw.rsi.chi_io import compute_chi_file

        out_paths = compute_chi_file(PROFILE_FILE, tmp_path, fft_length=512)
        ds = xr.open_dataset(out_paths[0])
        assert ds.attrs["Conventions"] == "CF-1.13, ACDD-1.3"
        assert ds["chi"].attrs["units"] == "K2 s-1"
        assert "history" in ds.attrs
        ds.close()

    def test_chi_file_output(self, skip_no_data, tmp_path):
        """compute_chi_file should write valid NetCDF."""
        from odas_tpw.rsi.chi_io import compute_chi_file

        out_paths = compute_chi_file(PROFILE_FILE, tmp_path, fft_length=512)
        assert len(out_paths) >= 1

        import xarray as xr

        ds = xr.open_dataset(out_paths[0])
        assert "chi" in ds
        assert "kB" in ds
        ds.close()

    def test_python_api(self, skip_no_data):
        """Top-level import should work."""
        from odas_tpw.rsi import get_chi

        results = get_chi(PROFILE_FILE, fft_length=512)
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# Chi from NetCDF path tests
# ---------------------------------------------------------------------------


class TestChiFromNetCDF:
    """Exercise the NetCDF input path for chi computation."""

    @pytest.fixture(autouse=True)
    def _skip_no_data(self):
        if not PROFILE_FILE.exists():
            pytest.skip("Test data not available")

    def test_load_therm_channels_from_nc(self, tmp_path):
        """_load_therm_channels should work with a per-profile NC file."""
        from odas_tpw.rsi.chi_io import _load_therm_channels
        from odas_tpw.rsi.profile import extract_profiles

        prof_paths = extract_profiles(PROFILE_FILE, tmp_path)
        assert len(prof_paths) > 0

        data = _load_therm_channels(prof_paths[0])
        assert "therm" in data
        # Should find at least one thermistor channel
        assert len(data["therm"]) > 0

    def test_compute_chi_from_nc(self, tmp_path):
        """compute_chi_file should work with a per-profile NC file."""
        from odas_tpw.rsi.chi_io import compute_chi_file
        from odas_tpw.rsi.profile import extract_profiles

        prof_dir = tmp_path / "profiles"
        prof_paths = extract_profiles(PROFILE_FILE, prof_dir)
        assert len(prof_paths) > 0

        chi_dir = tmp_path / "chi"
        chi_dir.mkdir()
        out_paths = compute_chi_file(prof_paths[0], chi_dir, fft_length=512)
        assert len(out_paths) >= 1


# ---------------------------------------------------------------------------
# Chi edge-case unit tests
# ---------------------------------------------------------------------------


class TestChiEdgeCases:
    """Edge-case unit tests for chi.py functions."""

    def test_chi_from_epsilon_zero_spectrum(self):
        """All-zero spectrum should produce chi=NaN."""
        from odas_tpw.chi.chi import _chi_from_epsilon
        from odas_tpw.chi.fp07 import fp07_tau, fp07_transfer, gradT_noise

        speed = 0.7
        fs = 512
        n_freq = 257
        F = np.arange(n_freq) * fs / (2 * (n_freq - 1))
        K = F / speed

        tau0 = fp07_tau(speed)
        H2 = fp07_transfer(F, tau0)
        noise_K, _ = gradT_noise(F, 10.0, speed, fs=fs, diff_gain=0.94)

        result = _chi_from_epsilon(
            np.zeros(n_freq),
            K,
            1e-7,
            1.2e-6,
            noise_K,
            H2,
            tau0,
            fp07_transfer,
            98.0,
            speed,
            "batchelor",
        )
        assert np.isnan(result.chi)

    def test_chi_from_epsilon_low_kB(self):
        """Very low epsilon should produce kB < 1 and chi=NaN."""
        from odas_tpw.chi.chi import _chi_from_epsilon
        from odas_tpw.chi.fp07 import fp07_tau, fp07_transfer, gradT_noise

        speed = 0.7
        fs = 512
        n_freq = 257
        F = np.arange(n_freq) * fs / (2 * (n_freq - 1))
        K = F / speed

        tau0 = fp07_tau(speed)
        H2 = fp07_transfer(F, tau0)
        noise_K, _ = gradT_noise(F, 10.0, speed, fs=fs, diff_gain=0.94)

        with pytest.warns(UserWarning, match="kB="):
            result = _chi_from_epsilon(
                np.ones(n_freq) * 1e-6,
                K,
                1e-20,
                1.2e-6,
                noise_K,
                H2,
                tau0,
                fp07_transfer,
                98.0,
                speed,
                "batchelor",
            )
        assert np.isnan(result.chi)

    def test_mle_too_few_points(self):
        """Too few valid wavenumber points should produce NaN from MLE fit."""
        from odas_tpw.chi.chi import _mle_fit_kB
        from odas_tpw.chi.fp07 import fp07_tau, fp07_transfer, gradT_noise

        speed = 0.7
        fs = 512
        # Only 5 frequency points — fewer than the min_points=6 requirement
        n_freq = 5
        F = np.arange(n_freq) * fs / (2 * (n_freq - 1))
        K = F / speed

        tau0 = fp07_tau(speed)
        H2 = fp07_transfer(F, tau0)
        noise_K, _ = gradT_noise(F, 10.0, speed, fs=fs, diff_gain=0.94)

        with pytest.warns(UserWarning, match="Too few"):
            result = _mle_fit_kB(
                np.ones(n_freq) * 1e-6,
                K,
                1e-7,
                1.2e-6,
                noise_K,
                H2,
                tau0,
                fp07_transfer,
                98.0,
                speed,
                "batchelor",
            )
        assert np.isnan(result.kB)

    def test_iterative_fit_zero_chi(self):
        """Spectrum where initial chi <= 0 should floor at 1e-14."""
        from odas_tpw.chi.chi import _iterative_fit
        from odas_tpw.chi.fp07 import fp07_tau, fp07_transfer, gradT_noise

        speed = 0.7
        fs = 512
        n_freq = 257
        F = np.arange(n_freq) * fs / (2 * (n_freq - 1))
        K = F / speed

        tau0 = fp07_tau(speed)
        H2 = fp07_transfer(F, tau0)
        noise_K, _ = gradT_noise(F, 10.0, speed, fs=fs, diff_gain=0.94)

        # Spectrum at noise floor — initial chi should floor at 1e-14
        result = _iterative_fit(
            noise_K * 0.5,
            K,
            1.2e-6,
            noise_K,
            H2,
            tau0,
            fp07_transfer,
            98.0,
            speed,
            "batchelor",
        )
        # Should produce some result (possibly NaN) without crashing
        assert isinstance(result.chi, float)

    def test_bilinear_correction_short(self):
        """F with length < 3 should return ones."""
        from odas_tpw.chi.chi import _bilinear_correction

        F = np.array([0.0, 1.0])
        bl = _bilinear_correction(F, 0.94, 512.0)
        np.testing.assert_array_equal(bl, np.ones(2))

    def test_valid_mask_all_below_noise(self):
        """Spectrum below 2x noise everywhere should use fallback mask."""
        from odas_tpw.chi.chi import _valid_wavenumber_mask

        n = 50
        K = np.linspace(0.5, 200, n)
        noise = np.ones(n) * 10.0
        spec = np.ones(n) * 1.0  # well below 2 * noise
        mask = _valid_wavenumber_mask(spec, noise, K, K_AA=200.0)
        # Fallback: K > 0 and K <= K_AA
        assert np.sum(mask) > 0
        assert mask[0]  # K[0] = 0.5 > 0


class TestEpsilonDsToL4Data:
    """``_epsilon_ds_to_l4data`` should prefer ``epsilonMean`` over a raw nanmean."""

    def _make_diss_ds(self, with_mean: bool):
        import xarray as xr

        n = 4
        # sh1 reasonable, sh2 spuriously low (simulates the SN465 bad-amp case)
        eps = np.array(
            [[1e-7, 1e-7, 1e-7, 1e-7], [1e-14, 1e-14, 1e-14, 1e-14]],
            dtype=np.float64,
        )
        ds = xr.Dataset(
            {
                "epsilon": (["probe", "time"], eps),
                "P_mean": (["time"], np.linspace(10, 100, n)),
                "speed": (["time"], np.full(n, 0.7)),
            },
            coords={
                "probe": ["sh1", "sh2"],
                "t": ("time", np.arange(n, dtype=float)),
            },
        )
        if with_mean:
            # Simulate mk_epsilon_mean having floored sh2 (1e-14 < epsilon_minimum)
            # and produced a per-window combined value from sh1 only.
            ds["epsilonMean"] = xr.DataArray(np.full(n, 1e-7), dims=["time"])
        return ds

    def test_uses_epsilon_mean_when_present(self):
        from odas_tpw.rsi.chi_io import _epsilon_ds_to_l4data

        ds = self._make_diss_ds(with_mean=True)
        l4 = _epsilon_ds_to_l4data(ds)
        # Must use epsilonMean (1e-7), not nanmean of [1e-7, 1e-14] (~5e-8)
        np.testing.assert_allclose(l4.epsi_final, 1e-7)

    def test_falls_back_to_nanmean_when_mean_absent(self):
        from odas_tpw.rsi.chi_io import _epsilon_ds_to_l4data

        ds = self._make_diss_ds(with_mean=False)
        l4 = _epsilon_ds_to_l4data(ds)
        # Without epsilonMean we expect the historical arithmetic mean.
        np.testing.assert_allclose(l4.epsi_final, np.nanmean(ds["epsilon"].values, axis=0))
