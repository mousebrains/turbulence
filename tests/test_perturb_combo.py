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
