# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb.combo — combined NetCDF assembly."""

import numpy as np
import xarray as xr

from odas_tpw.perturb.combo import (
    _glue_lengthwise,
    _glue_widthwise,
    make_combo,
    make_ctd_combo,
)
from odas_tpw.perturb.netcdf_schema import COMBO_SCHEMA, CTD_SCHEMA


class TestGlueWidthwise:
    def test_empty(self):
        """Empty list returns empty Dataset."""
        result = _glue_widthwise([])
        assert isinstance(result, xr.Dataset)
        assert len(result.data_vars) == 0


class TestGlueLengthwise:
    def test_empty(self):
        """Empty list returns empty Dataset."""
        result = _glue_lengthwise([])
        assert isinstance(result, xr.Dataset)
        assert len(result.data_vars) == 0


class TestMakeCombo:
    def test_depth_combo(self, tmp_path):
        """Two binned files → combo with widthwise glue."""
        binned_dir = tmp_path / "binned"
        binned_dir.mkdir()
        output_dir = tmp_path / "combo"

        for i in range(2):
            n_bins = 10
            ds = xr.Dataset(
                {
                    "T": (["bin", "profile"], np.random.randn(n_bins, 3)),
                    "depth": (
                        ["bin", "profile"],
                        np.tile(np.arange(n_bins, dtype=float)[:, None], (1, 3)),
                    ),
                },
                coords={
                    "bin": np.arange(n_bins, dtype=float),
                    "profile": np.arange(3),
                },
            )
            ds.to_netcdf(binned_dir / f"file{i:02d}.nc")

        out = make_combo(binned_dir, output_dir, COMBO_SCHEMA, method="depth")
        assert out is not None
        assert out.exists()

        combo = xr.open_dataset(out)
        # 2 files x 3 profiles = 6 total profiles
        assert combo.sizes["profile"] == 6
        assert "Conventions" in combo.attrs
        combo.close()

    def test_time_combo(self, tmp_path):
        """Time-based combo → lengthwise glue."""
        binned_dir = tmp_path / "binned"
        binned_dir.mkdir()
        output_dir = tmp_path / "combo"

        for i in range(2):
            n = 10
            ds = xr.Dataset(
                {"T": (["time"], np.random.randn(n))},
                coords={"time": np.arange(i * 10, i * 10 + n, dtype=float)},
            )
            ds.to_netcdf(binned_dir / f"file{i:02d}.nc")

        out = make_combo(binned_dir, output_dir, COMBO_SCHEMA, method="time")
        assert out is not None
        combo = xr.open_dataset(out)
        assert combo.sizes["time"] == 20
        combo.close()

    def test_no_files(self, tmp_path):
        binned_dir = tmp_path / "empty"
        binned_dir.mkdir()
        out = make_combo(binned_dir, tmp_path / "combo", COMBO_SCHEMA)
        assert out is None

    def test_netcdf_attrs_applied(self, tmp_path):
        binned_dir = tmp_path / "binned"
        binned_dir.mkdir()
        ds = xr.Dataset(
            {"T": (["bin", "profile"], np.ones((5, 2)))},
            coords={"bin": np.arange(5.0), "profile": np.arange(2)},
        )
        ds.to_netcdf(binned_dir / "file.nc")

        out = make_combo(
            binned_dir,
            tmp_path / "combo",
            COMBO_SCHEMA,
            netcdf_attrs={"institution": "Test University"},
        )
        assert out is not None
        combo = xr.open_dataset(out)
        assert combo.attrs["institution"] == "Test University"
        combo.close()

    def test_geospatial_attrs(self, tmp_path):
        """Binned files with lat/lon set geospatial extent attrs."""
        binned_dir = tmp_path / "binned"
        binned_dir.mkdir()
        output_dir = tmp_path / "combo"

        n_bins = 5
        ds = xr.Dataset(
            {
                "T": (["bin", "profile"], np.random.randn(n_bins, 3)),
                "lat": (["profile"], [10.0, 20.0, 30.0]),
                "lon": (["profile"], [-150.0, -140.0, -130.0]),
            },
            coords={
                "bin": np.arange(n_bins, dtype=float),
                "profile": np.arange(3),
            },
        )
        ds.to_netcdf(binned_dir / "file00.nc")

        out = make_combo(binned_dir, output_dir, COMBO_SCHEMA, method="depth")
        assert out is not None

        combo = xr.open_dataset(out)
        assert combo.attrs["geospatial_lat_min"] == 10.0
        assert combo.attrs["geospatial_lat_max"] == 30.0
        assert combo.attrs["geospatial_lon_min"] == -150.0
        assert combo.attrs["geospatial_lon_max"] == -130.0
        # bbox + CRS are derivable when both lat and lon are present
        assert combo.attrs["geospatial_bounds_crs"] == "EPSG:4326"
        assert "POLYGON" in combo.attrs["geospatial_bounds"]
        combo.close()

    def test_time_coverage_from_stime_etime(self, tmp_path):
        """stime/etime per-profile vars drive ACDD time_coverage_*."""
        binned_dir = tmp_path / "binned"
        binned_dir.mkdir()
        output_dir = tmp_path / "combo"

        # 2026-01-01 00:00:00 UTC and +2 hours, in epoch seconds
        t0, t1 = 1767225600.0, 1767225600.0 + 7200
        ds = xr.Dataset(
            {
                "T": (["bin", "profile"], np.random.randn(5, 2)),
                "stime": (["profile"], [t0, t0 + 60.0]),
                "etime": (["profile"], [t1, t1 + 60.0]),
            },
            coords={"bin": np.arange(5, dtype=float), "profile": np.arange(2)},
        )
        ds.to_netcdf(binned_dir / "file00.nc")

        out = make_combo(binned_dir, output_dir, COMBO_SCHEMA, method="depth")
        combo = xr.open_dataset(out)
        assert combo.attrs["time_coverage_start"].startswith("2026-01-01T00:00:00")
        assert combo.attrs["time_coverage_end"].startswith("2026-01-01T02:01:00")
        # ISO 8601 duration: 2 hours 1 minute (trailing zero seconds omitted)
        assert combo.attrs["time_coverage_duration"] == "PT2H1M"
        # Resolution is the median delta between successive stimes (60 s).
        assert combo.attrs["time_coverage_resolution"] == "PT1M"
        combo.close()

    def test_seconds_only_resolution(self, tmp_path):
        """Cover the seconds-only ISO duration branch (PT<n>S)."""
        binned_dir = tmp_path / "binned"
        binned_dir.mkdir()
        t0 = 1767225600.0
        # 30 seconds between profile starts → resolution PT30S
        stimes = np.array([t0 + 30.0 * i for i in range(3)])
        ds = xr.Dataset(
            {
                "T": (["bin", "profile"], np.zeros((3, 3))),
                "stime": (["profile"], stimes),
                "etime": (["profile"], stimes + 1.0),
            },
            coords={"bin": np.arange(3, dtype=float), "profile": np.arange(3)},
        )
        ds.to_netcdf(binned_dir / "file00.nc")
        out = make_combo(binned_dir, tmp_path / "combo", COMBO_SCHEMA, method="depth")
        combo = xr.open_dataset(out)
        assert combo.attrs["time_coverage_resolution"] == "PT30S"
        combo.close()

    def test_multi_day_duration_and_hourly_resolution(self, tmp_path):
        """Cover the >1 day branch in iso_dur and the hourly branch in
        time_coverage_resolution (lines combo.py:204, 228)."""
        binned_dir = tmp_path / "binned"
        binned_dir.mkdir()
        output_dir = tmp_path / "combo"
        # Three profiles 1 hour apart, spanning 3 days
        t0 = 1767225600.0  # 2026-01-01T00:00:00
        n_prof = 4
        stimes = np.array([t0 + 3600.0 * 24 * i for i in range(n_prof)])
        etimes = stimes + 60.0
        ds = xr.Dataset(
            {
                "T": (["bin", "profile"], np.zeros((3, n_prof))),
                "stime": (["profile"], stimes),
                "etime": (["profile"], etimes),
            },
            coords={"bin": np.arange(3, dtype=float), "profile": np.arange(n_prof)},
        )
        ds.to_netcdf(binned_dir / "file00.nc")
        out = make_combo(binned_dir, output_dir, COMBO_SCHEMA, method="depth")
        combo = xr.open_dataset(out)
        # Duration carries a D component plus a T component that contains S
        # only because total seconds is a whole multiple of one day.
        assert combo.attrs["time_coverage_duration"].startswith("P3D")
        # Resolution = median(diff(stimes)) = 86400 s = 24 h.
        assert combo.attrs["time_coverage_resolution"] == "PT24H"
        combo.close()

    def test_time_combo_trajectory_id_and_sort(self, tmp_path):
        """Time-mode combos: featureType=trajectory, trajectory_id scalar
        is stamped, and concatenated time vars are sorted strictly
        monotonic."""
        binned_dir = tmp_path / "binned"
        binned_dir.mkdir()
        # Files written out of chronological order: file01 has earlier
        # times than file00, so concatenation is non-monotonic until we
        # sort.
        for i, t_offset in enumerate([3600.0, 0.0]):
            n = 5
            t = 1767225600.0 + t_offset + np.arange(n, dtype=float)
            ds = xr.Dataset(
                {"T": (["time"], np.zeros(n))},
                coords={"time": t},
            )
            ds["time"].attrs["units"] = "seconds since 1970-01-01"
            ds["time"].attrs["calendar"] = "standard"
            ds.to_netcdf(binned_dir / f"file{i:02d}.nc")

        out = make_combo(binned_dir, tmp_path / "combo", COMBO_SCHEMA, method="time")
        combo = xr.open_dataset(out, decode_times=False)
        assert combo.attrs["featureType"] == "trajectory"
        assert "trajectory_id" in combo
        assert combo["trajectory_id"].attrs["cf_role"] == "trajectory_id"
        # Time is now strictly increasing across the concatenated files.
        t = combo["time"].values
        assert np.all(np.diff(t) > 0)
        combo.close()


class TestMakeCtdCombo:
    def test_delegates_to_make_combo(self, tmp_path):
        """make_ctd_combo delegates to make_combo with method='time'."""
        ctd_dir = tmp_path / "ctd"
        ctd_dir.mkdir()
        output_dir = tmp_path / "combo"

        for i in range(2):
            n = 10
            ds = xr.Dataset(
                {"T": (["time"], np.random.randn(n))},
                coords={"time": np.arange(i * 10, i * 10 + n, dtype=float)},
            )
            ds.to_netcdf(ctd_dir / f"ctd{i:02d}.nc")

        out = make_ctd_combo(ctd_dir, output_dir, CTD_SCHEMA)
        assert out is not None
        assert out.exists()

        combo = xr.open_dataset(out)
        assert combo.sizes["time"] == 20
        combo.close()
