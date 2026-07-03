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
