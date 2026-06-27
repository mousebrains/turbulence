# Jun-2026, Claude and Pat Welch, pat@mousebrains.com
"""Regression tests for the adversarial-review fixes on the audit-fix branch.

These cover the blocker/major issues the per-file review surfaced in the
agent-applied fixes:
  ocean.visc      — Sharqawy branch must floor viscosity (no negative nu)
  pair_nearest    — a NaN source must not shadow a finite neighbor within max_dt
  bin_by_time     — a non-grid time span must not drop its final partial bin
"""

from __future__ import annotations

import numpy as np


class TestOceanViscFloor:
    def test_negative_temperature_viscosity_floored(self):
        from odas_tpw.scor160.ocean import visc

        # A spurious-cold window mean drove the Sharqawy polynomial negative.
        assert np.all(np.asarray(visc(-42.0, 34, 100)) >= 1.0e-7)
        # A normal value is unaffected (sanity).
        assert visc(10.0, 35, 0) > 1.0e-6


class TestPairNearestFinite:
    def test_nan_source_does_not_shadow_finite_neighbor(self):
        from odas_tpw.processing.mixing import pair_nearest

        src_t = np.array([0.0, 8.0, 16.0])
        src_v = np.array([1e-7, np.nan, 3e-7])  # nearest to dst=8 is the NaN
        out = pair_nearest(src_t, src_v, np.array([8.0]), max_dt=10.0)
        assert np.isfinite(out[0]), "NaN source shadowed a finite neighbor"


class TestBinByTimeNoDataLoss:
    def test_non_grid_span_keeps_final_partial_bin(self, tmp_path):
        import xarray as xr

        from odas_tpw.perturb.binning import bin_by_time

        # Span 3.4 s is not an integer multiple of bin_width=1.0; the final
        # samples (t=3.0..3.4) must land in a [3,4) bin, not be dropped.
        n = 35
        t = np.linspace(0.0, 3.4, n)
        ds = xr.Dataset(
            {"T": (["time"], np.full(n, 5.0))},
            coords={"time": t},
        )
        path = tmp_path / "prof00.nc"
        ds.to_netcdf(path)
        result = bin_by_time([path], bin_width=1.0)
        # 4 bins ([0,1),[1,2),[2,3),[3,4)); the last retains the constant value.
        assert result.sizes["time"] == 4
        assert float(result["T"].values[-1]) == 5.0
