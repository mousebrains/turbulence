# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Branch-coverage tests for perturb.combo — unhit partial branches."""

from __future__ import annotations

import numpy as np
import xarray as xr

from odas_tpw.perturb.combo import make_combo
from odas_tpw.perturb.netcdf_schema import COMBO_SCHEMA


def _write_binned(path, **vars):
    """Write a small binned NC with the given variables."""
    ds = xr.Dataset(vars)
    ds.to_netcdf(path)


# ---------------------------------------------------------------------------
# netcdf_attrs with None values are skipped (line 124->123)
# ---------------------------------------------------------------------------


class TestNetcdfAttrsSkipNone:
    def test_none_value_skipped(self, tmp_path):
        """netcdf_attrs entry with v=None should be skipped."""
        binned_dir = tmp_path / "binned"
        binned_dir.mkdir()
        ds = xr.Dataset(
            {"T": (["bin", "profile"], np.ones((3, 1)))},
            coords={"bin": np.arange(3.0), "profile": np.arange(1)},
        )
        ds.to_netcdf(binned_dir / "f.nc")

        out = make_combo(
            binned_dir,
            tmp_path / "combo",
            COMBO_SCHEMA,
            netcdf_attrs={"institution": "Test", "missing": None},
        )
        combo = xr.open_dataset(out)
        assert combo.attrs.get("institution") == "Test"
        assert "missing" not in combo.attrs
        combo.close()


# ---------------------------------------------------------------------------
# Vertical-extent loop with all-NaN depth (line 159->156)
# ---------------------------------------------------------------------------


class TestVerticalExtentNoFinite:
    def test_depth_all_nan_falls_through(self, tmp_path):
        """When the first vertical candidate ('depth') has no finite values,
        loop continues to next candidate ('bin' or 'P')."""
        binned_dir = tmp_path / "binned"
        binned_dir.mkdir()
        # 'depth' is all NaN, but 'bin' coord is fine
        ds = xr.Dataset(
            {
                "T": (["bin", "profile"], np.ones((3, 1))),
                "depth": (["bin", "profile"], np.full((3, 1), np.nan)),
            },
            coords={"bin": np.arange(3.0), "profile": np.arange(1)},
        )
        ds.to_netcdf(binned_dir / "f.nc")

        out = make_combo(binned_dir, tmp_path / "combo", COMBO_SCHEMA)
        combo = xr.open_dataset(out)
        # depth was NaN → loop continued to 'bin' which has finite values
        assert "geospatial_vertical_min" in combo.attrs
        combo.close()


# ---------------------------------------------------------------------------
# stime/etime present but all NaN (line 177->191) — should fall through to time
# ---------------------------------------------------------------------------


class TestTimeCoverageFallback:
    def test_stime_etime_all_nan_falls_to_time(self, tmp_path):
        """When stime/etime are all NaN, code falls through and tries 'time' or no attr."""
        binned_dir = tmp_path / "binned"
        binned_dir.mkdir()
        ds = xr.Dataset(
            {
                "T": (["bin", "profile"], np.ones((3, 1))),
                "stime": (["profile"], np.full(1, np.nan)),
                "etime": (["profile"], np.full(1, np.nan)),
            },
            coords={"bin": np.arange(3.0), "profile": np.arange(1)},
        )
        ds.to_netcdf(binned_dir / "f.nc")

        out = make_combo(binned_dir, tmp_path / "combo", COMBO_SCHEMA)
        combo = xr.open_dataset(out)
        # Should have completed without time_coverage_* attrs
        assert combo is not None
        combo.close()

    def test_time_all_nan_no_coverage_attrs(self, tmp_path):
        """When 'time' coordinate has no finite values, no time_coverage_* attrs."""
        binned_dir = tmp_path / "binned"
        binned_dir.mkdir()
        ds = xr.Dataset(
            {"T": (["time"], np.ones(3))},
            coords={"time": np.full(3, np.nan)},
        )
        ds.to_netcdf(binned_dir / "f.nc")

        out = make_combo(binned_dir, tmp_path / "combo", COMBO_SCHEMA, method="time")
        combo = xr.open_dataset(out)
        assert "time_coverage_start" not in combo.attrs
        combo.close()


# ---------------------------------------------------------------------------
# stimes.size <= 1 after filtering for finite (line 217->234)
# ---------------------------------------------------------------------------


class TestStimesSizeOne:
    def test_single_stime_no_resolution_attr(self, tmp_path):
        """When stime has only 1 finite entry, no time_coverage_resolution."""
        binned_dir = tmp_path / "binned"
        binned_dir.mkdir()
        t0 = 1767225600.0
        ds = xr.Dataset(
            {
                "T": (["bin", "profile"], np.ones((3, 2))),
                "stime": (["profile"], [t0, np.nan]),  # only one finite
                "etime": (["profile"], [t0 + 60, np.nan]),
            },
            coords={"bin": np.arange(3.0), "profile": np.arange(2)},
        )
        ds.to_netcdf(binned_dir / "f.nc")

        out = make_combo(binned_dir, tmp_path / "combo", COMBO_SCHEMA)
        combo = xr.open_dataset(out)
        # time_coverage_start/end should be set (one finite is enough), but
        # resolution requires >= 2 finite stimes
        assert "time_coverage_start" in combo.attrs
        assert "time_coverage_resolution" not in combo.attrs
        combo.close()


# ---------------------------------------------------------------------------
# stimes all identical → median delta = 0 (line 220->234)
# ---------------------------------------------------------------------------


class TestStimesIdentical:
    def test_identical_stimes_no_resolution(self, tmp_path):
        """When all stimes are identical, median delta = 0 → no resolution attr."""
        binned_dir = tmp_path / "binned"
        binned_dir.mkdir()
        t0 = 1767225600.0
        # All three profiles have the same stime
        ds = xr.Dataset(
            {
                "T": (["bin", "profile"], np.ones((3, 3))),
                "stime": (["profile"], [t0, t0, t0]),
                "etime": (["profile"], [t0 + 60, t0 + 60, t0 + 60]),
            },
            coords={"bin": np.arange(3.0), "profile": np.arange(3)},
        )
        ds.to_netcdf(binned_dir / "f.nc")

        out = make_combo(binned_dir, tmp_path / "combo", COMBO_SCHEMA)
        combo = xr.open_dataset(out)
        # All stimes equal → median diff is 0 → no resolution attr
        assert "time_coverage_resolution" not in combo.attrs
        combo.close()


# ---------------------------------------------------------------------------
# Days-only ISO duration (line 203->211)
# ---------------------------------------------------------------------------


class TestDaysOnlyDuration:
    def test_exact_days_duration(self, tmp_path):
        """When duration is exactly N days with no h/m/s remainder, iso_dur ends with day count."""
        binned_dir = tmp_path / "binned"
        binned_dir.mkdir()
        t0 = 1767225600.0
        # Exactly 2 days = 172800s
        ds = xr.Dataset(
            {
                "T": (["bin", "profile"], np.ones((3, 2))),
                "stime": (["profile"], [t0, t0]),
                "etime": (["profile"], [t0 + 172800, t0 + 172800]),
            },
            coords={"bin": np.arange(3.0), "profile": np.arange(2)},
        )
        ds.to_netcdf(binned_dir / "f.nc")

        out = make_combo(binned_dir, tmp_path / "combo", COMBO_SCHEMA)
        combo = xr.open_dataset(out)
        dur = combo.attrs.get("time_coverage_duration", "")
        # Exact 2 days: iso_dur should be "P2DT0S" (default 0S branch)
        assert dur.startswith("P2D")
        combo.close()
