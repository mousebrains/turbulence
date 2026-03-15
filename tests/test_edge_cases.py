"""Edge-case tests for various rsi modules."""

from pathlib import Path

import numpy as np
import pytest

from odas_tpw.chi.batchelor import batchelor_grad
from odas_tpw.chi.fp07 import FP07NoiseConfig, noise_thermchannel
from odas_tpw.scor160.despike import despike
from odas_tpw.scor160.goodman import clean_shear_spec
from odas_tpw.scor160.nasmyth import nasmyth
from odas_tpw.scor160.ocean import visc35
from odas_tpw.scor160.spectral import csd_odas

VMP_DIR = Path(__file__).resolve().parent.parent / "VMP"
P_FILES = sorted(VMP_DIR.glob("*.p"))


class TestNasmythEdgeCases:
    def test_negative_epsilon_warns(self):
        with pytest.warns(UserWarning):
            result = nasmyth(-1e-8, 1e-6, np.array([1, 10, 100]))
        assert np.all(np.isnan(result))

    def test_zero_epsilon_warns(self):
        with pytest.warns(UserWarning):
            result = nasmyth(0.0, 1e-6, np.array([1, 10, 100]))
        assert np.all(np.isnan(result))

    def test_zero_nu_raises(self):
        with pytest.raises(ValueError):
            nasmyth(1e-8, 0.0, np.array([1, 10]))

    def test_negative_nu_raises(self):
        with pytest.raises(ValueError):
            nasmyth(1e-8, -1e-6, np.array([1, 10]))


class TestBatchelorEdgeCases:
    def test_zero_chi(self):
        K = np.linspace(1, 500, 100)
        result = batchelor_grad(K, kB=100, chi=0)
        np.testing.assert_array_equal(result, np.zeros_like(K))

    def test_negative_kB(self):
        K = np.linspace(1, 500, 100)
        result = batchelor_grad(K, kB=-1, chi=1e-8)
        assert np.all(np.isfinite(result) | (result == 0))


class TestDespikeEdgeCases:
    def test_empty_array(self):
        with pytest.raises(ValueError):
            despike(np.array([]), 512.0)

    def test_short_array(self):
        x = np.array([1.0, 2.0, 3.0])
        y, _spike_indices, _n_passes, _fraction = despike(x, 512.0)
        assert len(y) == 3

    def test_all_nan(self):
        x = np.full(1024, np.nan)
        y, _spike_indices, _n_passes, _fraction = despike(x, 512.0)
        assert len(y) == 1024

    def test_constant_signal(self):
        x = np.ones(1024)
        y, _spike_indices, _n_passes, _fraction = despike(x, 512.0)
        np.testing.assert_array_equal(y, np.ones(1024))


class TestVisc35EdgeCases:
    def test_extreme_cold(self):
        nu = visc35(-2.0)
        assert np.isfinite(nu)
        assert nu > 0

    def test_extreme_hot(self):
        nu = visc35(40.0)
        assert np.isfinite(nu)
        assert nu > 0


class TestCsdOdasEdgeCases:
    def test_short_signal(self):
        with pytest.raises(ValueError):
            csd_odas(np.array([1.0, 2.0]), None, 256, 512.0)

    def test_all_zeros(self):
        Cxy, _F, _Cxx, _Cyy = csd_odas(np.zeros(1024), None, 256, 512.0)
        np.testing.assert_array_equal(Cxy, np.zeros_like(Cxy))


# ---------------------------------------------------------------------------
# profile.py — extract_profiles file-writing path
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not P_FILES, reason="No VMP .p files available")
class TestExtractProfiles:
    def test_extract_profiles_writes_nc(self, tmp_path):
        from odas_tpw.rsi.profile import extract_profiles

        paths = extract_profiles(P_FILES[0], tmp_path)
        assert len(paths) > 0
        for p in paths:
            assert isinstance(p, Path)
            assert p.suffix == ".nc"
            assert p.exists()

        # Verify contents of first profile
        import netCDF4 as nc

        ds = nc.Dataset(str(paths[0]), "r")
        assert ds.profile_number >= 1
        assert ds.fs_fast > 0
        assert ds.fs_slow > 0
        assert "t_fast" in ds.variables
        assert "t_slow" in ds.variables
        assert "P" in ds.variables
        ds.close()

    def test_extract_profiles_roundtrip(self, tmp_path):
        import netCDF4 as nc

        from odas_tpw.rsi.profile import extract_profiles

        paths = extract_profiles(P_FILES[0], tmp_path)
        assert len(paths) > 0

        ds = nc.Dataset(str(paths[0]), "r")
        assert isinstance(float(ds.profile_P_start), float)
        assert isinstance(float(ds.profile_duration_s), float)
        assert ds.profile_P_start > 0
        assert ds.profile_duration_s > 0
        ds.close()

    def test_extract_profiles_from_nc(self, tmp_path):
        """Exercise _load_from_nc by extracting profiles from a .nc file."""
        from odas_tpw.rsi.convert import p_to_L1
        from odas_tpw.rsi.profile import extract_profiles

        # Convert .p to .nc first
        nc_dir = tmp_path / "nc"
        nc_dir.mkdir()
        _pf, nc_path = p_to_L1(P_FILES[0], nc_dir / "full.nc")

        # Extract profiles from the .nc file
        prof_dir = tmp_path / "profiles"
        paths_nc = extract_profiles(nc_path, prof_dir)

        # Compare with direct .p extraction
        prof_dir_p = tmp_path / "profiles_p"
        paths_p = extract_profiles(P_FILES[0], prof_dir_p)

        assert len(paths_nc) == len(paths_p)
        assert len(paths_nc) > 0

    def test_extract_one_worker(self, tmp_path):
        """Exercise _extract_one parallel worker wrapper."""
        from odas_tpw.rsi.profile import _extract_one

        source_str, count = _extract_one((P_FILES[0], tmp_path, {}))
        assert source_str == str(P_FILES[0])
        assert count > 0
        nc_files = list(tmp_path.glob("*.nc"))
        assert len(nc_files) == count


# ---------------------------------------------------------------------------
# profile.py — NC-path tests using sample_nc_file fixture
# ---------------------------------------------------------------------------


class TestExtractProfilesFromNC:
    """Tests for extract_profiles() and _load_from_nc() with NC input."""

    def test_extract_profiles_from_nc_sample(self, sample_nc_file, tmp_path):
        """extract_profiles should work with NC input and produce profile NCs."""
        from odas_tpw.rsi.profile import extract_profiles

        prof_dir = tmp_path / "profiles"
        paths = extract_profiles(sample_nc_file, prof_dir)
        assert len(paths) > 0
        for p in paths:
            assert p.suffix == ".nc"
            assert p.exists()

    def test_load_from_nc_fields(self, sample_nc_file):
        """_load_source should return dict with expected keys for NC input."""
        from odas_tpw.rsi.profile import _load_source

        data = _load_source(sample_nc_file)
        assert "P" in data
        assert "fs_fast" in data
        assert "fs_slow" in data
        assert "channels" in data
        assert "stem" in data
        assert data["fs_fast"] > 0

    def test_extract_profiles_flat_pressure(self, tmp_path):
        """Flat pressure (no profiling) should produce no profiles."""
        import netCDF4

        from odas_tpw.rsi.profile import extract_profiles

        # Create a synthetic NC with flat pressure
        nc_path = tmp_path / "flat.nc"
        ds = netCDF4.Dataset(str(nc_path), "w", format="NETCDF4")
        n_fast = 4096
        n_slow = 512
        ds.fs_fast = 512.0
        ds.fs_slow = 64.0
        ds.createDimension("n_fast", n_fast)
        ds.createDimension("n_slow", n_slow)
        t_f = ds.createVariable("t_fast", "f8", ("n_fast",))
        t_f[:] = np.arange(n_fast) / 512.0
        t_s = ds.createVariable("t_slow", "f8", ("n_slow",))
        t_s[:] = np.arange(n_slow) / 64.0
        P = ds.createVariable("P", "f8", ("n_slow",))
        P[:] = np.full(n_slow, 5.0)  # flat pressure
        ds.close()

        prof_dir = tmp_path / "profiles"
        paths = extract_profiles(nc_path, prof_dir)
        assert len(paths) == 0


# ---------------------------------------------------------------------------
# fp07.py — FP07NoiseConfig override path and beta_2 correction
# ---------------------------------------------------------------------------


class TestFP07NoiseConfig:
    F = np.arange(1, 257) * 1.0

    def test_config_overrides_kwargs(self):
        cfg = FP07NoiseConfig(R_0=5000)
        result_cfg = noise_thermchannel(self.F, T_mean=15, config=cfg)
        result_default = noise_thermchannel(self.F, T_mean=15)
        assert np.all(np.isfinite(result_cfg))
        assert np.all(np.isfinite(result_default))
        assert not np.allclose(result_cfg, result_default)

    def test_config_matches_kwargs(self):
        cfg = FP07NoiseConfig()
        result_cfg = noise_thermchannel(self.F, T_mean=15, config=cfg)
        result_kw = noise_thermchannel(self.F, T_mean=15)
        np.testing.assert_array_equal(result_cfg, result_kw)

    def test_beta_2_correction(self):
        result_with = noise_thermchannel(self.F, T_mean=15, beta_2=5000.0)
        result_without = noise_thermchannel(self.F, T_mean=15)
        assert np.all(np.isfinite(result_with))
        assert np.all(np.isfinite(result_without))
        assert not np.allclose(result_with, result_without)


# ---------------------------------------------------------------------------
# goodman.py — short-signal fallback and singular-matrix fallback
# ---------------------------------------------------------------------------


class TestGoodmanEdgeCases:
    def test_short_signal_fallback(self):
        """Signal shorter than 2*nfft triggers the fallback path."""
        nfft = 256
        rate = 512.0
        # Length between nfft and 2*nfft: cross-spectrum fails, auto-spectrum works
        n = nfft + nfft // 2  # 384
        rng = np.random.default_rng(42)
        shear = rng.standard_normal((n, 1))
        accel = rng.standard_normal((n, 3))

        with pytest.warns(UserWarning, match="Insufficient FFT segments"):
            clean_UU, _AA, _UU, _UA, F = clean_shear_spec(accel, shear, nfft, rate)

        n_freq = nfft // 2 + 1
        assert clean_UU.shape == (n_freq, 1, 1)
        assert F.shape == (n_freq,)

    def test_very_short_signal_zeros(self):
        """Signal shorter than nfft: even auto-spectrum fails, returns zeros."""
        nfft = 256
        rate = 512.0
        n = 100  # shorter than nfft
        rng = np.random.default_rng(42)
        shear = rng.standard_normal((n, 1))
        accel = rng.standard_normal((n, 3))

        with pytest.warns(UserWarning, match="Insufficient FFT segments"):
            clean_UU, _AA, UU, _UA, F = clean_shear_spec(accel, shear, nfft, rate)

        n_freq = nfft // 2 + 1
        assert clean_UU.shape == (n_freq, 1, 1)
        assert F.shape == (n_freq,)
        # UU should be zeros since auto-spectrum also failed
        np.testing.assert_array_equal(np.real(UU), np.zeros_like(np.real(UU)))

    def test_singular_matrix(self):
        """Singular AA matrix (duplicate accel columns) doesn't crash."""
        nfft = 256
        rate = 512.0
        n = 2048
        rng = np.random.default_rng(42)
        shear = rng.standard_normal((n, 1))
        a1 = rng.standard_normal((n, 1))
        # Make accel columns identical to create singular AA
        accel = np.hstack([a1, a1, a1])

        clean_UU, _AA, _UU, _UA, _F = clean_shear_spec(accel, shear, nfft, rate)

        assert clean_UU.shape[0] == nfft // 2 + 1
        assert np.all(np.isfinite(clean_UU))
