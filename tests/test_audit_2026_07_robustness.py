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
