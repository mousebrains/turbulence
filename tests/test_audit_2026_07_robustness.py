"""Regression tests for the 2026-07-01 deep-audit robustness fixes (Batch C).

Each test pins a bad-data / edge-case behavior that the audit flagged: a
below-detection chi window must be NaN (not a finite sentinel), a flatlined
FP07 segment must not poison the alignment lag, an out-of-coverage GPS query
must warn, a UTC ``+00:00`` timestamp must parse, and a never-settled top_trim
cast must never return a trim deeper than every sample.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest


class TestChiBelowNoiseIsNaN:
    """chi/chi.py: a window whose whole band is at/below the noise floor -> NaN."""

    def test_iterative_below_noise_returns_nan(self):
        from odas_tpw.rsi.window import compute_chi_window
        from odas_tpw.scor160.ocean import visc35

        rng = np.random.default_rng(1)
        # Near-zero thermistor signal -> gradient spectrum far below the modeled
        # FP07 electronics noise floor. Pre-fix this returned the 1e-14 sentinel
        # (or a grid-edge kB) as finite chi; it must be NaN (below detection).
        therm = [rng.standard_normal(512) * 1e-9]
        r = compute_chi_window(
            therm, [0.94], 0.6, 15.0, float(visc35(15.0)), 512.0, 256, 98.0, method=2
        )
        assert np.all(np.isnan(r.chi))

    def test_real_signal_still_finite(self):
        # Guard against over-masking: a normal Method-1 window stays finite.
        from odas_tpw.rsi.window import compute_chi_window
        from odas_tpw.scor160.ocean import visc35

        rng = np.random.default_rng(42)
        therm = [rng.standard_normal(512) * 0.001]
        r = compute_chi_window(
            therm,
            [0.94],
            0.6,
            15.0,
            float(visc35(15.0)),
            512.0,
            256,
            98.0,
            epsilon=np.array([1e-7]),
            method=1,
        )
        assert np.all(np.isfinite(r.chi)) and np.all(r.chi > 0)


class TestFP07LagGuard:
    """perturb/fp07_cal.py: flatlined / non-finite segments must not poison lag."""

    def test_flatline_returns_nan(self):
        from odas_tpw.perturb import fp07_cal as F

        lag, corr = F._calc_lag(
            np.full(2000, 22.0), np.full(2000, 1000.0), 64.0, must_be_negative=True
        )
        assert np.isnan(lag) and np.isnan(corr)

    def test_nan_sample_returns_nan(self):
        from odas_tpw.perturb import fp07_cal as F

        ref = np.full(2000, 22.0)
        ref[7] = np.nan
        lag, corr = F._calc_lag(ref, np.full(2000, 1000.0), 64.0)
        assert np.isnan(lag) and np.isnan(corr)

    def test_clean_signal_recovers_lag(self):
        # A real (shifted) signal still returns a finite negative lag.
        from odas_tpw.perturb import fp07_cal as F

        fs = 64.0
        t = np.arange(4000) / fs
        ref = 22.0 + 0.5 * np.sin(2 * np.pi * 0.3 * t)
        shift = 5  # samples
        fp = np.empty_like(ref)
        fp[shift:] = -1000.0 * (ref[:-shift] - 22.0) + 5000.0
        fp[:shift] = fp[shift]
        lag, corr = F._calc_lag(ref, fp, fs, must_be_negative=True)
        assert np.isfinite(lag)


class TestGPSCoverageWindow:
    """perturb/gps.py: coverage = finite-node span, so extrapolations warn."""

    def _make_nc(self, tmp_path, drop_after=800):
        import netCDF4 as nc

        f = tmp_path / "gps.nc"
        ds = nc.Dataset(str(f), "w")
        ds.createDimension("obs", 1001)
        tv = ds.createVariable("time", "f8", ("obs",))
        av = ds.createVariable("lat", "f8", ("obs",))
        ov = ds.createVariable("lon", "f8", ("obs",))
        tv.units = "seconds since 1970-01-01"
        t = np.arange(1001.0)
        lat = 15 + t * 1e-4
        lon = 145 + t * 1e-4
        lat[t > drop_after] = np.nan
        lon[t > drop_after] = np.nan
        tv[:] = t
        av[:] = lat
        ov[:] = lon
        ds.close()
        return f

    def test_tmax_is_last_finite_fix(self, tmp_path):
        from odas_tpw.perturb.gps import GPSFromNetCDF

        g = GPSFromNetCDF(str(self._make_nc(tmp_path)), max_time_diff=60.0)
        assert g._t_max == pytest.approx(800.0)

    def test_query_past_last_fix_warns(self, tmp_path):
        from odas_tpw.perturb.gps import GPSFromNetCDF

        g = GPSFromNetCDF(str(self._make_nc(tmp_path)), max_time_diff=60.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            g.lat(np.array([950.0]))
        assert len(w) >= 1


class TestParseTimeZeroOffset:
    """perturb/plot/sections.py: a zero UTC offset is accepted; non-zero rejected."""

    def test_trailing_z(self):
        from odas_tpw.perturb.plot.sections import parse_time

        assert parse_time("2025-01-15T00:00:00Z") == np.datetime64("2025-01-15T00:00:00")

    def test_space_zero_offset(self):
        # YAML resolves an unquoted '...Z' timestamp to '... +00:00' with a space.
        from odas_tpw.perturb.plot.sections import parse_time

        assert parse_time("2025-01-15 00:00:00+00:00") == np.datetime64(
            "2025-01-15T00:00:00"
        )

    def test_t_zero_offset(self):
        from odas_tpw.perturb.plot.sections import parse_time

        assert parse_time("2025-01-15T00:00:00+00:00") == np.datetime64(
            "2025-01-15T00:00:00"
        )

    def test_nonzero_offset_rejected(self):
        from odas_tpw.perturb.plot.sections import parse_time

        with pytest.raises(ValueError, match="must be UTC"):
            parse_time("2025-01-15T00:00:00+09:00")
        with pytest.raises(ValueError, match="must be UTC"):
            parse_time("2025-01-15T00:00:00-05:00")


class TestConvertAccelODASParity:
    """channels.py: convert_accel with only coef0/coef1 must match ODAS (adc_bits=0)."""

    def test_counts_based_cal_matches_odas(self):
        from odas_tpw.rsi.channels import convert_accel

        d = np.array([1000, -2000, 15000], dtype=np.int16)
        out, unit = convert_accel(d, {"coef0": "-100.0", "coef1": "1500.0"})
        expected = 9.81 * (d.astype(float) - (-100.0)) / 1500.0
        assert np.allclose(out, expected)
        assert unit == "m_s-2"

    def test_explicit_adc_bits_still_honored(self):
        # SN479 configs carry adc_bits=16 explicitly; that path is unchanged.
        from odas_tpw.rsi.channels import convert_accel

        d = np.array([1000, 2000], dtype=np.int16)
        out, _ = convert_accel(
            d, {"coef0": "0", "coef1": "1", "adc_fs": "1", "adc_bits": "16"}
        )
        exp = 9.81 * (d.astype(float) / 2**16)
        assert np.allclose(out, exp)


class TestChiFomTwoSided:
    """Chi obs/model variance-ratio QC must reject fom << 1, not only fom > limit."""

    def test_qc_chi_final_rejects_low_fom_probe(self):
        from odas_tpw.rsi.pipeline import _qc_chi_final

        chi = np.array([[1e-8], [1e-6]])  # probe0 good, probe1 much larger
        fom = np.array([[1.0], [0.2]])  # probe1 model overestimates 5x -> bad
        kmr = np.array([[0.9], [0.9]])
        out = _qc_chi_final(chi, fom, kmr)
        # Only probe0 passes the two-sided gate, so the combined chi is probe0's.
        assert out[0] == pytest.approx(1e-8)

    def test_apply_fom_cut_two_sided_nans_low_fom(self):
        xr = pytest.importorskip("xarray")
        from odas_tpw.perturb.pipeline import _apply_fom_cut

        ds = xr.Dataset(
            {
                "fom": (("probe", "time"), np.array([[1.0, 0.1], [1.0, 1.0]])),
                "chi": (("probe", "time"), np.array([[1e-8, 2e-8], [3e-8, 4e-8]])),
                "chi_1": (("time",), np.array([1e-8, 2e-8])),
                "chi_2": (("time",), np.array([3e-8, 4e-8])),
            }
        )
        n = _apply_fom_cut(ds, 1.15, "test", two_sided=True)
        assert n == 1
        assert np.isnan(ds["chi"].values[0, 1])  # fom=0.1 <= 1/1.15
        assert np.isnan(ds["chi_1"].values[1])
        assert np.isfinite(ds["chi"].values[0, 0])  # fom=1.0 kept


class TestAtomicToNetcdf:
    """perturb/pipeline.py: _atomic_to_netcdf leaves no partial NC on failure."""

    def test_success_writes_and_leaves_no_tmp(self, tmp_path):
        xr = pytest.importorskip("xarray")
        from odas_tpw.perturb.pipeline import _atomic_to_netcdf

        ds = xr.Dataset({"x": (("t",), np.arange(5.0))})
        out = tmp_path / "a.nc"
        _atomic_to_netcdf(ds, out)
        assert out.exists()
        assert not list(tmp_path.glob(".a.nc.*.tmp"))

    def test_failure_leaves_no_partial(self, tmp_path, monkeypatch):
        xr = pytest.importorskip("xarray")
        from odas_tpw.perturb import pipeline as P

        ds = xr.Dataset({"x": (("t",), np.arange(5.0))})
        out = tmp_path / "b.nc"

        def boom(self, *a, **k):
            # Simulate a mid-write abort after the tmp file is touched.
            raise OSError("disk full")

        monkeypatch.setattr(xr.Dataset, "to_netcdf", boom)
        with pytest.raises(OSError):
            P._atomic_to_netcdf(ds, out)
        assert not out.exists()  # no partial at the live path
        assert not list(tmp_path.glob(".b.nc.*.tmp"))


class TestChiSchemaHeadlineVars:
    """CHI_SCHEMA must carry the published chiMean/chiLnSigma so combos get attrs."""

    def test_chimean_and_chilnsigma_have_units(self):
        from odas_tpw.perturb.netcdf_schema import CHI_SCHEMA

        for name in ("chiMean", "chiLnSigma"):
            assert name in CHI_SCHEMA
            assert "units" in CHI_SCHEMA[name]
            assert "long_name" in CHI_SCHEMA[name]


class TestChiKappaTemperatureDependence:
    """chi core: chi scales with the per-window kappa_T, epsilon_T with kappa_T^2.

    The old fixed KAPPA_T=1.4e-7 biased chi low in warm water and slightly high
    in cold water; the threaded ocean.kappa_T(T,S,P) removes that.
    """

    def _synthetic_obs(self, kap):
        from odas_tpw.chi.batchelor import batchelor_kB, kraichnan_grad
        from odas_tpw.chi.fp07 import fp07_tau, fp07_transfer, gradT_noise

        W, nu, fs, nfft = 0.7, 1.0e-6, 512.0, 1024
        F = np.arange(nfft // 2 + 1) * fs / nfft
        K = F / W
        K[0] = 1e-9
        tau0 = float(fp07_tau(W))
        H2 = fp07_transfer(F, tau0)
        noise_K, _ = gradT_noise(F, 15.0, W, fs=fs, diff_gain=0.94)
        kB_true = float(batchelor_kB(1e-8, nu, kap))
        spec_clean = kraichnan_grad(K, kB_true, 2e-6, kappa_T=kap)
        return spec_clean * H2 + noise_K, K, nu, noise_K, H2, tau0, fp07_transfer, W

    def test_kappa_T_is_temperature_dependent(self):
        from odas_tpw.scor160.ocean import kappa_T

        warm, cold = float(kappa_T(28.0)), float(kappa_T(-1.0))
        assert warm > cold  # thermal conductivity rises with T
        assert 1.45e-7 < warm < 1.55e-7  # ~1.498e-7 (Sharqawy/Jamieson-Tudhope)
        assert 1.35e-7 < cold < 1.40e-7  # ~1.385e-7

    def test_method2_chi_and_epsilonT_scale_with_kappa(self):
        from odas_tpw.chi.batchelor import KAPPA_T
        from odas_tpw.chi.chi import _iterative_fit
        from odas_tpw.scor160.ocean import kappa_T

        kap = float(kappa_T(28.0))  # warm: kap > KAPPA_T
        spec_obs, K, nu, noise_K, H2, tau0, h2f, W = self._synthetic_obs(kap)
        fixed = _iterative_fit(spec_obs, K, nu, noise_K, H2, tau0, h2f, 88.2, W, "kraichnan", KAPPA_T)
        true = _iterative_fit(spec_obs, K, nu, noise_K, H2, tau0, h2f, 88.2, W, "kraichnan", kap)
        r = kap / KAPPA_T
        # chi is linear in kappa_T (kB fit is kappa_T-independent in Method 2).
        assert true.chi / fixed.chi == pytest.approx(r, rel=0.02)
        # epsilon_T = (2*pi*kB)^4 * nu * kappa_T^2 -> exact square law.
        assert true.epsilon / fixed.epsilon == pytest.approx(r**2, rel=1e-6)
        # Using the true kappa_T recovers the injected chi=2e-6 more accurately.
        assert true.chi == pytest.approx(2e-6, rel=0.05)


class TestMixingPerVariableCeilings:
    """mixing.py: K_T and Gamma get their own implausibility ceilings, not K_rho's."""

    def test_kt_and_gamma_ceilings_fire_independently(self):
        from odas_tpw.processing.mixing import mixing_coefficients

        # Window 0: tiny epsilon -> huge Gamma; weak dT/dz + big chi -> huge K_T;
        # but K_rho = 0.2*eps/N2 stays tiny (must NOT be masked).
        eps = np.array([1e-12, 1e-7])
        chi = np.array([1e-4, 1e-8])
        N2 = np.array([1e-3, 1e-4])
        dTdz = np.array([1e-3, 1e-2])
        with np.errstate(all="ignore"):
            r = mixing_coefficients(eps, chi, N2, dTdz)
        assert np.isnan(r.K_T[0]) and np.isfinite(r.K_T[1])
        assert np.isnan(r.Gamma[0]) and np.isfinite(r.Gamma[1])
        # K_rho at window 0 is tiny and physical -> kept despite K_T/Gamma masks.
        assert np.isfinite(r.K_rho[0])

    def test_high_kt_in_low_n2_water_survives(self):
        # A legitimately large K_T where N2 is at the floor must NOT be masked
        # just because K_rho would be (per-variable gating).
        from odas_tpw.processing.mixing import mixing_coefficients

        eps = np.array([1e-9])
        chi = np.array([1e-7])
        N2 = np.array([1e-10])  # below N2_min -> K_rho/Gamma NaN
        dTdz = np.array([1e-2])  # strong gradient -> modest, finite K_T
        with np.errstate(all="ignore"):
            r = mixing_coefficients(eps, chi, N2, dTdz)
        assert np.isfinite(r.K_T[0])  # K_T survives in unstratified water
        assert np.isnan(r.K_rho[0])


class TestBinCoordinateMeters:
    """binning.py: a pressure-only profile is binned on depth in METRES."""

    def _make_profile(self, path, lat=None):
        import netCDF4 as nc

        ds = nc.Dataset(str(path), "w")
        ds.createDimension("time", 4)
        p = ds.createVariable("P_mean", "f8", ("time",))
        p[:] = [100.0, 500.0, 1000.0, 1500.0]
        e = ds.createVariable("epsilonMean", "f8", ("time",))
        e[:] = [1e-8, 2e-8, 3e-8, 4e-8]
        if lat is not None:
            la = ds.createVariable("lat", "f8", ())
            la[()] = lat
        ds.close()

    def test_pressure_converted_with_lat(self, tmp_path):
        import gsw

        from odas_tpw.perturb.binning import _load_profile_snapshot

        f = tmp_path / "prof.nc"
        self._make_profile(f, lat=15.0)
        snap = _load_profile_snapshot(f)
        expected = -gsw.z_from_p(np.array([100.0, 500.0, 1000.0, 1500.0]), 15.0)
        np.testing.assert_allclose(snap["depth"], expected)
        # depth (m) is strictly shallower than pressure (dbar) in magnitude here
        assert np.all(snap["depth"] < np.array([100.0, 500.0, 1000.0, 1500.0]))

    def test_pressure_converted_default_lat_when_missing(self, tmp_path):
        import gsw

        from odas_tpw.perturb.binning import _DEFAULT_BIN_LATITUDE, _load_profile_snapshot

        f = tmp_path / "prof_nolat.nc"
        self._make_profile(f, lat=None)
        snap = _load_profile_snapshot(f)
        expected = -gsw.z_from_p(
            np.array([100.0, 500.0, 1000.0, 1500.0]), _DEFAULT_BIN_LATITUDE
        )
        np.testing.assert_allclose(snap["depth"], expected)


class TestTopTrimNeverDeeperThanSamples:
    """processing/top_trim.py: a returned trim is always applicable to the cast."""

    def test_trim_within_sample_range(self):
        # A normal prop-wash cast: elevated near the surface, settling with depth.
        # Whatever trim is returned, at least one in-range sample must be at or
        # below it, or the caller would apply no trim at all.
        from odas_tpw.processing.top_trim import compute_trim_depth

        rng = np.random.default_rng(7)
        depth = np.linspace(1.0, 30.0, 30000)
        def chan():
            s = rng.standard_normal(depth.size) * 0.01
            surf = depth < 3.0
            s[surf] += rng.standard_normal(int(surf.sum())) * 3.0
            return s

        trim = compute_trim_depth(depth, {"Ax": chan(), "Ay": chan()})
        if trim is not None:
            in_range = depth[(depth >= 1.0) & (depth <= 50.0)]
            assert (in_range >= trim).any()
