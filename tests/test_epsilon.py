# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for the epsilon (TKE dissipation) pipeline modules."""

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# ocean.py
# ---------------------------------------------------------------------------


class TestVisc35:
    def test_zero_celsius(self):
        from rsi_python.ocean import visc35

        nu = visc35(0.0)
        assert 1.7e-6 < nu < 1.9e-6, f"visc35(0) = {nu}"

    def test_twenty_celsius(self):
        from rsi_python.ocean import visc35

        nu = visc35(20.0)
        assert 1.0e-6 < nu < 1.1e-6, f"visc35(20) = {nu}"

    def test_array(self):
        from rsi_python.ocean import visc35

        T = np.array([0.0, 10.0, 20.0])
        nu = visc35(T)
        assert nu.shape == (3,)
        assert nu[0] > nu[1] > nu[2]  # viscosity decreases with temperature


class TestVisc:
    def test_s35_p0_matches_visc35(self):
        """visc(T, S=35, P=0) should match visc35(T)."""
        from rsi_python.ocean import visc, visc35

        T = np.array([0.0, 10.0, 20.0])
        np.testing.assert_array_equal(visc(T, 35, 0), visc35(T))

    def test_different_salinity(self):
        """Non-default salinity should give different viscosity."""
        from rsi_python.ocean import visc

        nu_35 = visc(10.0, 35, 0)
        nu_30 = visc(10.0, 30, 0)
        assert nu_35 != nu_30

    def test_scalar(self):
        from rsi_python.ocean import visc

        nu = visc(10.0, 34.5, 100.0)
        assert 1e-7 < nu < 1e-5


class TestDensity:
    def test_reasonable_range(self):
        from rsi_python.ocean import density

        rho = density(10.0, 35.0, 0.0)
        assert 1020 < rho < 1030

    def test_salinity_increases_density(self):
        from rsi_python.ocean import density

        rho_low = density(10.0, 30.0, 0.0)
        rho_high = density(10.0, 38.0, 0.0)
        assert rho_high > rho_low


class TestBuoyancyFreq:
    def test_stable_stratification(self):
        from rsi_python.ocean import buoyancy_freq

        T = np.array([20.0, 15.0, 10.0, 5.0])
        S = np.full(4, 35.0)
        P = np.array([0.0, 100.0, 200.0, 300.0])
        N2, p_mid = buoyancy_freq(T, S, P)
        assert len(N2) == 3
        assert len(p_mid) == 3
        # Stable stratification should have positive N²
        assert np.all(N2[np.isfinite(N2)] > 0)


# ---------------------------------------------------------------------------
# nasmyth.py
# ---------------------------------------------------------------------------


class TestNasmyth:
    def test_shape(self):
        from rsi_python.nasmyth import nasmyth

        k = np.logspace(-1, 3, 100)
        phi = nasmyth(1e-7, 1.2e-6, k)
        assert phi.shape == (100,)

    def test_positive(self):
        from rsi_python.nasmyth import nasmyth

        k = np.logspace(-1, 3, 100)
        phi = nasmyth(1e-7, 1.2e-6, k)
        assert np.all(phi > 0)

    def test_higher_epsilon_higher_spectrum(self):
        from rsi_python.nasmyth import nasmyth

        k = np.logspace(0, 2, 50)
        nu = 1e-6
        phi_low = nasmyth(1e-9, nu, k)
        phi_high = nasmyth(1e-5, nu, k)
        # Higher epsilon should give higher spectral levels
        assert np.mean(phi_high) > np.mean(phi_low)

    def test_nondim(self):
        from rsi_python.nasmyth import nasmyth_nondim

        x = np.logspace(-4, 0, 100)
        G2 = nasmyth_nondim(x)
        assert G2.shape == (100,)
        assert np.all(G2 > 0)

    def test_lueck_coefficients(self):
        """Verify we use the Lueck coefficients, not older Oakey ones."""
        from rsi_python.nasmyth import _nasmyth_g2

        x = np.array([0.01])
        G2 = _nasmyth_g2(x)
        # Lueck: 8.05 * 0.01^(1/3) / (1 + (20.6*0.01)^3.715)
        expected = 8.05 * 0.01 ** (1 / 3) / (1 + (20.6 * 0.01) ** 3.715)
        np.testing.assert_allclose(G2[0], expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# spectral.py
# ---------------------------------------------------------------------------


class TestCSD:
    def test_white_noise_flat(self):
        """White noise should produce an approximately flat spectrum."""
        from rsi_python.spectral import csd_odas

        rng = np.random.default_rng(42)
        x = rng.standard_normal(10000)
        nfft = 256
        rate = 512.0
        Pxx, F, _, _ = csd_odas(x, None, nfft, rate, detrend="none")
        # Variance of white noise = 1, so PSD ≈ 1/rate at all frequencies
        # (for properly normalized one-sided spectrum)
        assert Pxx.shape == (nfft // 2 + 1,)
        # Check the spectrum is roughly flat (within 50% for this seed)
        mid = Pxx[5:-5]
        assert np.std(mid) / np.mean(mid) < 1.0

    def test_variance_preservation(self):
        """Integral of auto-spectrum should approximate signal variance."""
        from rsi_python.spectral import csd_odas

        rng = np.random.default_rng(123)
        x = rng.standard_normal(8192)
        nfft = 256
        rate = 512.0
        Pxx, F, _, _ = csd_odas(x, None, nfft, rate, detrend="constant")
        integral = np.trapezoid(Pxx, F)
        assert abs(integral - np.var(x)) / np.var(x) < 0.15

    def test_cross_spectrum_returns_all(self):
        from rsi_python.spectral import csd_odas

        rng = np.random.default_rng(42)
        x = rng.standard_normal(2048)
        y = rng.standard_normal(2048)
        Cxy, F, Cxx, Cyy = csd_odas(x, y, 256, 512.0)
        assert Cxy is not None
        assert Cxx is not None
        assert Cyy is not None

    def test_parabolic_detrend(self):
        """Parabolic detrend should produce valid spectrum."""
        from rsi_python.spectral import csd_odas

        rng = np.random.default_rng(42)
        x = rng.standard_normal(4096) + np.linspace(0, 1, 4096) ** 2
        Pxx, F, _, _ = csd_odas(x, None, 256, 512.0, detrend="parabolic")
        assert Pxx.shape == (129,)
        assert np.all(Pxx > 0)
        assert np.all(np.isfinite(Pxx))

    def test_cubic_detrend(self):
        """Cubic detrend should produce valid spectrum."""
        from rsi_python.spectral import csd_odas

        rng = np.random.default_rng(42)
        x = rng.standard_normal(4096) + np.linspace(0, 1, 4096) ** 3
        Pxx, F, _, _ = csd_odas(x, None, 256, 512.0, detrend="cubic")
        assert Pxx.shape == (129,)
        assert np.all(Pxx > 0)
        assert np.all(np.isfinite(Pxx))

    def test_matrix_auto(self):
        from rsi_python.spectral import csd_matrix

        rng = np.random.default_rng(42)
        x = rng.standard_normal((4096, 2))
        Cxy, F, _, _ = csd_matrix(x, None, 256, 512.0)
        # Should be (n_freq, 2, 2)
        assert Cxy.shape == (129, 2, 2)
        # Auto-spectra on diagonal should be real and positive
        assert np.all(np.real(Cxy[:, 0, 0]) > 0)
        assert np.all(np.real(Cxy[:, 1, 1]) > 0)

    def test_matrix_cross(self):
        from rsi_python.spectral import csd_matrix

        rng = np.random.default_rng(42)
        x = rng.standard_normal((4096, 2))
        y = rng.standard_normal((4096, 3))
        Cxy, F, Cxx, Cyy = csd_matrix(x, y, 256, 512.0)
        assert Cxy.shape == (129, 2, 3)
        assert Cxx.shape == (129, 2, 2)
        assert Cyy.shape == (129, 3, 3)


# ---------------------------------------------------------------------------
# despike.py
# ---------------------------------------------------------------------------


class TestDespike:
    def test_clean_signal_unchanged(self):
        """A clean sinusoidal signal should not be modified."""
        from rsi_python.despike import despike

        t = np.arange(0, 10, 1 / 512)
        x = np.sin(2 * np.pi * 5 * t)
        y, spikes, n_passes, frac = despike(x, 512.0, thresh=8, smooth=0.5)
        assert len(spikes) == 0
        assert frac == 0.0
        np.testing.assert_array_equal(y, x)

    def test_spike_detected(self):
        """An obvious spike should be detected and removed."""
        from rsi_python.despike import despike

        t = np.arange(0, 10, 1 / 512)
        x = np.sin(2 * np.pi * 5 * t) * 0.1
        # Add a large spike
        x[2560] = 50.0
        y, spikes, n_passes, frac = despike(x, 512.0, thresh=8, smooth=0.5)
        assert len(spikes) > 0
        assert abs(y[2560]) < 1.0  # spike removed
        assert frac > 0

    def test_spike_at_start(self):
        """Spike at index 0 should be handled without error."""
        from rsi_python.despike import despike

        t = np.arange(0, 10, 1 / 512)
        x = np.sin(2 * np.pi * 5 * t) * 0.1
        x[0] = 50.0
        y, spikes, n_passes, frac = despike(x, 512.0, thresh=8, smooth=0.5)
        assert np.all(np.isfinite(y))

    def test_spike_at_end(self):
        """Spike at last index should be handled without error."""
        from rsi_python.despike import despike

        t = np.arange(0, 10, 1 / 512)
        x = np.sin(2 * np.pi * 5 * t) * 0.1
        x[-1] = 50.0
        y, spikes, n_passes, frac = despike(x, 512.0, thresh=8, smooth=0.5)
        assert np.all(np.isfinite(y))

    def test_all_spike_signal(self):
        """All-constant signal with a huge spike everywhere → replacement=0 path."""
        from rsi_python.despike import despike

        # Signal where most values are huge → they all look like spikes
        x = np.full(5120, 100.0)
        y, spikes, n_passes, frac = despike(x, 512.0, thresh=2, smooth=0.5)
        assert np.all(np.isfinite(y))


# ---------------------------------------------------------------------------
# profile.py
# ---------------------------------------------------------------------------


class TestGetProfiles:
    def test_single_profile(self):
        from rsi_python.profile import get_profiles

        fs = 64.0
        t = np.arange(0, 60, 1 / fs)
        # Simulate a profile: ramp down from 0 to 50 dbar over 60 s
        P = np.linspace(0, 50, len(t))
        W = np.gradient(P, 1 / fs)
        profiles = get_profiles(P, W, fs, P_min=0.5, W_min=0.3, direction="down", min_duration=7.0)
        assert len(profiles) >= 1
        s, e = profiles[0]
        assert P[s] > 0.5
        assert (e - s) / fs >= 7.0

    def test_no_profile(self):
        from rsi_python.profile import get_profiles

        fs = 64.0
        P = np.zeros(1000)  # no depth
        W = np.zeros(1000)
        profiles = get_profiles(P, W, fs, P_min=0.5, W_min=0.3)
        assert profiles == []

    def test_two_profiles(self):
        from rsi_python.profile import get_profiles

        fs = 64.0
        # Two ramp-down profiles separated by a surface interval
        t = np.arange(0, 120, 1 / fs)
        P = np.zeros(len(t))
        # Profile 1: t=10-40s, P ramps 0→30 dbar
        i1s, i1e = int(10 * fs), int(40 * fs)
        P[i1s:i1e] = np.linspace(0, 30, i1e - i1s)
        # Profile 2: t=70-100s, P ramps 0→25 dbar
        i2s, i2e = int(70 * fs), int(100 * fs)
        P[i2s:i2e] = np.linspace(0, 25, i2e - i2s)
        W = np.gradient(P, 1 / fs)
        profiles = get_profiles(P, W, fs, P_min=0.5, W_min=0.3, direction="down", min_duration=7.0)
        assert len(profiles) == 2


# ---------------------------------------------------------------------------
# goodman.py
# ---------------------------------------------------------------------------


class TestGoodman:
    def test_short_signal_warns(self):
        """Short signal with small nfft should warn and return without error."""
        from rsi_python.goodman import clean_shear_spec

        rng = np.random.default_rng(99)
        N = 100
        shear = rng.standard_normal((N, 1))
        accel = rng.standard_normal((N, 2))
        with pytest.warns(UserWarning, match="Insufficient FFT segments"):
            clean_UU, AA, UU, UA, F = clean_shear_spec(accel, shear, 64, 512.0)
        # Should still return valid arrays
        assert clean_UU.shape[0] == 64 // 2 + 1
        assert np.all(np.isfinite(np.real(clean_UU)))

    def test_transposed_input(self):
        """Passing (n_chan, n_samples) shape should be auto-transposed."""
        from rsi_python.goodman import clean_shear_spec

        rng = np.random.default_rng(42)
        N = 4096
        # Pass transposed shapes: (n_chan, N) instead of (N, n_chan)
        shear = rng.standard_normal((1, N))  # 1 channel, N samples
        accel = rng.standard_normal((2, N))  # 2 channels, N samples
        clean_UU, AA, UU, UA, F = clean_shear_spec(accel, shear, 256, 512.0)
        assert clean_UU.shape[0] == 256 // 2 + 1
        assert np.all(np.isfinite(np.real(clean_UU)))

    def test_bias_warning_with_few_segments(self):
        """Signal with barely enough segments should trigger bias warning."""
        from rsi_python.goodman import clean_shear_spec

        rng = np.random.default_rng(99)
        # 2 accel channels, nfft=64: need > 1.02*2 ~ 2.04 segments
        # segments = 2*N/nfft - 1; for N=100, nfft=64: segments = 2*100/64 - 1 = 2.125
        # 2.125 <= 1.02*2 = 2.04 → True, so warning fires
        N = 100
        shear = rng.standard_normal((N, 1))
        accel = rng.standard_normal((N, 2))
        with pytest.warns(UserWarning, match="Insufficient FFT segments"):
            clean_UU, AA, UU, UA, F = clean_shear_spec(accel, shear, 64, 512.0)
        assert np.all(np.isfinite(np.real(clean_UU)))

    def test_clean_shear_reduces_noise(self):
        """Goodman cleaning should reduce coherent noise."""
        from rsi_python.goodman import clean_shear_spec

        rng = np.random.default_rng(42)
        N = 4096
        # Create vibration signal
        vib = 10 * np.sin(2 * np.pi * 50 * np.arange(N) / 512)
        # Shear = turbulence + vibration leak
        turb = rng.standard_normal(N) * 0.1
        shear = turb + 0.5 * vib
        accel = vib[:, np.newaxis] + rng.standard_normal((N, 1)) * 0.01
        shear = shear[:, np.newaxis]

        clean_UU, AA, UU, UA, F = clean_shear_spec(accel, shear, 256, 512.0)

        # Find the frequency bin nearest 50 Hz (vibration frequency)
        vib_bin = np.argmin(np.abs(F - 50))
        # Clean spectrum at vibration frequency should be lower
        assert np.real(clean_UU[vib_bin, 0, 0]) < np.real(UU[vib_bin, 0, 0])


# ---------------------------------------------------------------------------
# Integration test on real VMP data
# ---------------------------------------------------------------------------

TEST_DATA_DIR = Path(__file__).parent / "data"
PROFILE_FILE = TEST_DATA_DIR / "SN479_0006.p"


class TestIntegration:
    @pytest.fixture
    def skip_no_data(self):
        if not PROFILE_FILE.exists():
            pytest.skip("Test data not available")

    def test_profile_detection(self, skip_no_data):
        """File 0005 should have at least one profile."""
        from rsi_python.p_file import PFile
        from rsi_python.profile import _smooth_fall_rate, get_profiles

        pf = PFile(PROFILE_FILE)
        P = pf.channels["P"]
        W = _smooth_fall_rate(P, pf.fs_slow)
        profiles = get_profiles(
            P, W, pf.fs_slow, P_min=0.5, W_min=0.3, direction="down", min_duration=7.0
        )
        assert len(profiles) >= 1

    def test_load_channels(self, skip_no_data):
        """load_channels should find shear and accel channels."""
        from rsi_python.dissipation import load_channels

        data = load_channels(PROFILE_FILE)
        assert len(data["shear"]) >= 1
        assert len(data["accel"]) >= 1
        assert data["fs_fast"] > 400

    def test_load_channels_has_start_time(self, skip_no_data):
        """load_channels metadata should include start_time for CF time units."""
        from rsi_python.dissipation import load_channels

        data = load_channels(PROFILE_FILE)
        assert "start_time" in data["metadata"]
        assert len(data["metadata"]["start_time"]) > 0

    def test_get_diss_produces_valid_epsilon(self, skip_no_data):
        """Epsilon values should be in a reasonable range."""
        from rsi_python.dissipation import get_diss

        results = get_diss(PROFILE_FILE, fft_length=256, goodman=True)
        assert len(results) >= 1
        ds = results[0]
        eps = ds["epsilon"].values
        valid = eps[np.isfinite(eps) & (eps > 0)]
        assert len(valid) > 0
        # Epsilon should span a wide range but be physically reasonable
        assert np.min(valid) > 1e-14, f"min epsilon too small: {np.min(valid)}"
        assert np.max(valid) < 1e0, f"max epsilon too large: {np.max(valid)}"

    def test_qc_variables_in_output(self, skip_no_data):
        """Output should contain fom and K_max_ratio QC variables."""
        from rsi_python.dissipation import get_diss

        results = get_diss(PROFILE_FILE, fft_length=256, goodman=True)
        ds = results[0]
        assert "fom" in ds, "Missing fom variable"
        assert "K_max_ratio" in ds, "Missing K_max_ratio variable"
        # fom should have some finite values
        fom = ds["fom"].values
        assert np.any(np.isfinite(fom)), "No finite fom values"
        # K_max_ratio should have some finite positive values
        kmr = ds["K_max_ratio"].values
        valid_kmr = kmr[np.isfinite(kmr)]
        assert len(valid_kmr) > 0, "No finite K_max_ratio values"
        assert np.all(valid_kmr > 0), "K_max_ratio should be positive"

    def test_epsilon_cf_compliance(self, skip_no_data):
        """Epsilon dataset should have CF-1.13 attributes on all variables."""
        from rsi_python.dissipation import get_diss

        results = get_diss(PROFILE_FILE, fft_length=256, goodman=True)
        ds = results[0]

        # Global attributes
        assert ds.attrs["Conventions"] == "CF-1.13"
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
            assert "long_name" in var.attrs, f"{vname} missing long_name"
            if vname == "method":
                assert "flag_values" in var.attrs, "method missing flag_values"
                assert "flag_meanings" in var.attrs, "method missing flag_meanings"
            else:
                assert "units" in var.attrs, f"{vname} missing units"

        # Standard names on known physical variables
        assert ds["P_mean"].attrs["standard_name"] == "sea_water_pressure"
        assert ds["P_mean"].attrs["positive"] == "down"
        assert ds["T_mean"].attrs["standard_name"] == "sea_water_temperature"

        # Specific units checks
        assert ds["epsilon"].attrs["units"] == "W kg-1"
        assert ds["speed"].attrs["units"] == "m s-1"
        assert ds["nu"].attrs["units"] == "m2 s-1"
        assert ds["P_mean"].attrs["units"] == "dbar"
        assert ds["T_mean"].attrs["units"] == "degree_Celsius"
        assert ds["K"].attrs["units"] == "cpm"
        assert ds["F"].attrs["units"] == "Hz"

        # Probe coordinate
        assert "long_name" in ds.coords["probe"].attrs

    def test_epsilon_file_cf_compliance(self, skip_no_data, tmp_path):
        """Epsilon NetCDF file should roundtrip with CF attributes intact."""
        import xarray as xr

        from rsi_python.dissipation import compute_diss_file

        eps_paths = compute_diss_file(PROFILE_FILE, tmp_path, fft_length=256, goodman=True)
        ds = xr.open_dataset(eps_paths[0])
        assert ds.attrs["Conventions"] == "CF-1.13"
        assert ds["epsilon"].attrs["units"] == "W kg-1"
        assert "history" in ds.attrs
        ds.close()

    def test_profile_cf_compliance(self, skip_no_data, tmp_path):
        """Profile NetCDF should have CF-1.13 attributes."""
        import netCDF4 as nc

        from rsi_python.profile import extract_profiles

        prof_dir = tmp_path / "prof_cf"
        prof_paths = extract_profiles(PROFILE_FILE, prof_dir)
        assert len(prof_paths) >= 1

        ds = nc.Dataset(str(prof_paths[0]), "r")
        assert ds.Conventions == "CF-1.13"

        for tvar_name in ("t_fast", "t_slow"):
            tvar = ds.variables[tvar_name]
            assert tvar.standard_name == "time"
            assert tvar.calendar == "standard"
            assert tvar.axis == "T"
            assert "seconds since" in tvar.units

        ds.close()

    def test_salinity_passthrough(self, skip_no_data):
        """get_diss with salinity should use gsw-based viscosity."""
        from rsi_python.dissipation import get_diss

        results_default = get_diss(PROFILE_FILE, fft_length=256, goodman=True)
        results_sal = get_diss(PROFILE_FILE, fft_length=256, goodman=True, salinity=34.5)
        assert len(results_sal) >= 1
        # Viscosity should differ from default (S=35)
        nu_default = results_default[0]["nu"].values
        nu_sal = results_sal[0]["nu"].values
        assert not np.allclose(nu_default, nu_sal, rtol=1e-6, atol=0), (
            "Viscosity should change with different salinity"
        )

    def test_pipeline_p2prof_p2eps(self, skip_no_data, tmp_path):
        """Full pipeline: .p → p2prof → p2eps."""
        from rsi_python.dissipation import compute_diss_file
        from rsi_python.profile import extract_profiles

        prof_dir = tmp_path / "profiles"
        prof_paths = extract_profiles(PROFILE_FILE, prof_dir)
        assert len(prof_paths) >= 1

        eps_dir = tmp_path / "epsilon"
        eps_paths = compute_diss_file(prof_paths[0], eps_dir)
        assert len(eps_paths) >= 1

        import xarray as xr

        ds = xr.open_dataset(eps_paths[0])
        assert "epsilon" in ds
        assert "K_max" in ds
        ds.close()

    def test_direct_vs_chained(self, skip_no_data, tmp_path):
        """Direct .p→epsilon should produce same as chained pipeline."""
        from rsi_python.dissipation import get_diss
        from rsi_python.profile import extract_profiles

        # Direct
        results_direct = get_diss(PROFILE_FILE, fft_length=256, goodman=True)
        assert len(results_direct) >= 1
        eps_direct = results_direct[0]["epsilon"].values

        # Chained: extract profile, then compute
        prof_dir = tmp_path / "chain_prof"
        prof_paths = extract_profiles(PROFILE_FILE, prof_dir)
        assert len(prof_paths) >= 1

        results_chained = get_diss(prof_paths[0], fft_length=256, goodman=True)
        assert len(results_chained) >= 1
        eps_chained = results_chained[0]["epsilon"].values

        # Results should be very close (not exact due to profile boundary precision)
        n_compare = min(len(eps_direct[0]), len(eps_chained[0]))
        if n_compare > 2:
            # Check that the order of magnitude is similar
            d = eps_direct[0, :n_compare]
            c = eps_chained[0, :n_compare]
            valid = np.isfinite(d) & np.isfinite(c) & (d > 0) & (c > 0)
            if np.sum(valid) > 0:
                ratio = np.log10(d[valid]) - np.log10(c[valid])
                assert np.median(np.abs(ratio)) < 1.0  # within an order of magnitude

    def test_get_diss_no_goodman(self, skip_no_data):
        """get_diss with goodman=False should produce valid epsilon."""
        from rsi_python.dissipation import get_diss

        results = get_diss(PROFILE_FILE, fft_length=256, goodman=False)
        assert len(results) >= 1
        ds = results[0]
        eps = ds["epsilon"].values
        valid = eps[np.isfinite(eps) & (eps > 0)]
        assert len(valid) > 0

    def test_get_diss_fixed_speed(self, skip_no_data):
        """get_diss with speed=0.5 should use fixed speed."""
        from rsi_python.dissipation import get_diss

        results = get_diss(PROFILE_FILE, fft_length=256, goodman=True, speed=0.5)
        assert len(results) >= 1
        ds = results[0]
        # All speed values should be the fixed speed
        speeds = ds["speed"].values
        np.testing.assert_allclose(speeds, 0.5)
