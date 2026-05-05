# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb.binning — depth/time binning."""

import numpy as np
import xarray as xr

from odas_tpw.perturb.binning import (
    _bin_array,
    bin_by_depth,
    bin_by_time,
    bin_chi,
    bin_diss,
)


class TestBinArray:
    def test_basic_mean(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        coords = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        bin_edges = np.array([0.0, 2.0, 4.0, 6.0])
        binned, counts = _bin_array(values, coords, bin_edges, np.nanmean)
        assert counts[0] == 2
        assert counts[1] == 2
        assert counts[2] == 1
        np.testing.assert_allclose(binned[0], 1.5)  # mean(1,2)
        np.testing.assert_allclose(binned[1], 3.5)  # mean(3,4)
        np.testing.assert_allclose(binned[2], 5.0)

    def test_median(self):
        values = np.array([1.0, 2.0, 100.0, 4.0])
        coords = np.array([0.5, 1.5, 2.0, 3.5])
        bin_edges = np.array([0.0, 5.0])
        binned, counts = _bin_array(values, coords, bin_edges, np.nanmedian)
        assert counts[0] == 4
        np.testing.assert_allclose(binned[0], np.median([1.0, 2.0, 100.0, 4.0]))

    def test_empty_bin(self):
        values = np.array([1.0, 2.0])
        coords = np.array([0.5, 1.5])
        bin_edges = np.array([0.0, 1.0, 5.0, 10.0])
        binned, counts = _bin_array(values, coords, bin_edges, np.nanmean)
        assert counts[0] == 1
        assert counts[1] == 1
        assert counts[2] == 0
        assert np.isnan(binned[2])

    def test_nan_values(self):
        values = np.array([1.0, np.nan, 3.0])
        coords = np.array([0.5, 1.5, 2.5])
        bin_edges = np.array([0.0, 3.0])
        binned, counts = _bin_array(values, coords, bin_edges, np.nanmean)
        assert counts[0] == 3
        np.testing.assert_allclose(binned[0], 2.0)  # nanmean(1, nan, 3) = 2


class TestBinByDepth:
    def test_two_profiles(self, tmp_path):
        # Create two synthetic profile NetCDFs
        for i in range(2):
            n = 100
            depth = np.linspace(0, 50, n)
            T = 20.0 - 0.1 * depth + np.random.randn(n) * 0.01
            ds = xr.Dataset(
                {"T": (["time_slow"], T), "depth": (["time_slow"], depth)},
                coords={"time_slow": np.arange(n, dtype=float)},
            )
            ds.to_netcdf(tmp_path / f"prof{i:02d}.nc")

        files = sorted(tmp_path.glob("*.nc"))
        result = bin_by_depth(files, bin_width=5.0)
        assert "T" in result
        assert result["T"].dims == ("bin", "profile")
        assert result.sizes["profile"] == 2
        assert result.sizes["bin"] > 0

    def test_per_profile_scalars_propagate(self, tmp_path):
        """Scalar lat/lon/stime/etime on each per-profile NetCDF should
        appear as 1-D ``(profile,)`` vars on the binned output."""
        # 2026-01-01 00:00:00 UTC in epoch seconds
        t0 = 1767225600.0
        lats = [10.0, 20.0]
        lons = [-150.0, -140.0]
        stimes = [t0, t0 + 100.0]
        etimes = [t0 + 60.0, t0 + 200.0]
        for i in range(2):
            n = 50
            depth = np.linspace(0, 50, n)
            T = 20.0 - 0.1 * depth
            ds = xr.Dataset(
                {
                    "T": (["time_slow"], T),
                    "depth": (["time_slow"], depth),
                    "lat": ((), lats[i]),
                    "lon": ((), lons[i]),
                    "stime": ((), stimes[i]),
                    "etime": ((), etimes[i]),
                },
                coords={"time_slow": np.arange(n, dtype=float)},
            )
            ds.to_netcdf(tmp_path / f"prof{i:02d}.nc")

        files = sorted(tmp_path.glob("*.nc"))
        result = bin_by_depth(files, bin_width=5.0)
        for v in ("lat", "lon", "stime", "etime"):
            assert v in result.data_vars, f"{v} missing from binned output"
            assert result[v].dims == ("profile",)
        np.testing.assert_allclose(result["lat"].values, lats)
        np.testing.assert_allclose(result["lon"].values, lons)
        np.testing.assert_allclose(result["stime"].values, stimes)
        np.testing.assert_allclose(result["etime"].values, etimes)

    def test_datetime64_stime_etime_decoded_to_epoch_seconds(self, tmp_path):
        """When stime/etime are written with CF units, xarray decodes them
        to datetime64 on read.  bin_by_depth must reduce back to epoch
        seconds — combo's auto-fill expects numeric timestamps."""
        import netCDF4 as nc

        for i in range(2):
            n = 30
            path = tmp_path / f"prof{i:02d}.nc"
            ds = nc.Dataset(str(path), "w")
            ds.createDimension("z", n)
            d = ds.createVariable("depth", "f8", ("z",))
            t = ds.createVariable("T", "f8", ("z",))
            d[:] = np.linspace(0, 50, n)
            t[:] = np.linspace(15, 5, n)
            for sname, val in (("stime", 1767225600.0 + i * 100.0),
                               ("etime", 1767225700.0 + i * 100.0)):
                v = ds.createVariable(sname, "f8", ())
                v[...] = val
                v.units = "seconds since 1970-01-01"
                v.calendar = "standard"
                v.standard_name = "time"
            ds.close()

        files = sorted(tmp_path.glob("*.nc"))
        result = bin_by_depth(files, bin_width=5.0)
        # After decode-then-reduce, values are float epoch seconds again.
        assert result["stime"].dtype.kind == "f"
        np.testing.assert_allclose(
            result["stime"].values, [1767225600.0, 1767225700.0], rtol=0, atol=1e-3
        )
        # ``units`` and ``calendar`` were stripped to avoid CF-encoder
        # conflicts when writing the combo.
        assert "units" not in result["stime"].attrs
        assert "calendar" not in result["stime"].attrs

    def test_mean_vs_median(self, tmp_path):
        n = 200
        depth = np.linspace(0, 20, n)
        T = np.ones(n) * 10.0
        T[50] = 100.0  # outlier
        ds = xr.Dataset(
            {"T": (["time_slow"], T), "depth": (["time_slow"], depth)},
            coords={"time_slow": np.arange(n, dtype=float)},
        )
        ds.to_netcdf(tmp_path / "prof.nc")

        mean_result = bin_by_depth([tmp_path / "prof.nc"], bin_width=20.0, aggregation="mean")
        median_result = bin_by_depth([tmp_path / "prof.nc"], bin_width=20.0, aggregation="median")
        # Median should be more robust
        assert median_result["T"].values[0, 0] < mean_result["T"].values[0, 0]

    def test_uses_P_when_no_depth(self, tmp_path):
        """When 'depth' is absent, bin_by_depth falls back to 'P'."""
        n = 50
        P = np.linspace(0, 20, n)
        T = 15.0 - 0.2 * P
        ds = xr.Dataset(
            {"T": (["time_slow"], T), "P": (["time_slow"], P)},
            coords={"time_slow": np.arange(n, dtype=float)},
        )
        ds.to_netcdf(tmp_path / "prof.nc")

        result = bin_by_depth([tmp_path / "prof.nc"], bin_width=5.0)
        assert "T" in result
        assert result["T"].dims == ("bin", "profile")
        assert result.sizes["bin"] > 0
        # Verify actual values are reasonable (first bin centered near 2.5 m)
        np.testing.assert_allclose(result["T"].values[0, 0], 15.0 - 0.2 * P[:13].mean(), atol=0.5)

    def test_no_depth_returns_empty(self, tmp_path):
        """Profile with neither 'depth' nor 'P' returns an empty Dataset."""
        n = 30
        ds = xr.Dataset(
            {"T": (["time_slow"], np.ones(n))},
            coords={"time_slow": np.arange(n, dtype=float)},
        )
        ds.to_netcdf(tmp_path / "prof.nc")

        result = bin_by_depth([tmp_path / "prof.nc"], bin_width=1.0)
        assert len(result.data_vars) == 0

    def test_diagnostics(self, tmp_path):
        """diagnostics=True produces *_std and n_samples variables."""
        n = 100
        depth = np.linspace(0, 10, n)
        T = 20.0 + np.random.randn(n) * 0.5
        ds = xr.Dataset(
            {"T": (["time_slow"], T), "depth": (["time_slow"], depth)},
            coords={"time_slow": np.arange(n, dtype=float)},
        )
        ds.to_netcdf(tmp_path / "prof.nc")

        result = bin_by_depth([tmp_path / "prof.nc"], bin_width=5.0, diagnostics=True)
        assert "T" in result
        assert "T_std" in result
        assert "n_samples" in result
        # T_std values should be finite where data exists
        assert np.any(np.isfinite(result["T_std"].values))

    def test_skips_multidim(self, tmp_path):
        """A 2D variable is skipped; 1D variables are still binned."""
        n = 50
        depth = np.linspace(0, 10, n)
        T = 10.0 * np.ones(n)
        matrix = np.ones((n, 3))  # 2D — should be skipped
        ds = xr.Dataset(
            {
                "T": (["time_slow"], T),
                "depth": (["time_slow"], depth),
                "matrix": (["time_slow", "col"], matrix),
            },
            coords={
                "time_slow": np.arange(n, dtype=float),
                "col": np.arange(3),
            },
        )
        ds.to_netcdf(tmp_path / "prof.nc")

        result = bin_by_depth([tmp_path / "prof.nc"], bin_width=5.0)
        assert "T" in result
        assert "matrix" not in result


class TestBinByTime:
    def test_basic(self, tmp_path):
        """Create profile NC with t_slow + T variable and verify time-binned output."""
        n = 200
        t_slow = np.linspace(0, 10, n)
        T = 20.0 + 0.1 * t_slow
        ds = xr.Dataset(
            {"T": (["time_slow"], T), "t_slow": (["time_slow"], t_slow)},
            coords={"time_slow": np.arange(n, dtype=float)},
        )
        ds.to_netcdf(tmp_path / "prof.nc")

        result = bin_by_time([tmp_path / "prof.nc"], bin_width=2.0)
        assert "T" in result
        assert "time" in result.dims
        assert result.sizes["time"] > 0

    def test_no_time_coord(self, tmp_path):
        """Profile NC without any recognised time coord returns empty Dataset."""
        n = 30
        ds = xr.Dataset(
            {"T": (["samples"], np.ones(n))},
            coords={"samples": np.arange(n, dtype=float)},
        )
        ds.to_netcdf(tmp_path / "prof.nc")

        result = bin_by_time([tmp_path / "prof.nc"], bin_width=1.0)
        assert len(result.data_vars) == 0


class TestBinDiss:
    def test_depth_method(self, tmp_path):
        """bin_diss with method='depth' delegates to bin_by_depth."""
        n = 80
        depth = np.linspace(0, 40, n)
        epsilon = np.random.lognormal(-20, 1, n)
        ds = xr.Dataset(
            {"epsilon": (["time_slow"], epsilon), "depth": (["time_slow"], depth)},
            coords={"time_slow": np.arange(n, dtype=float)},
        )
        ds.to_netcdf(tmp_path / "diss.nc")

        result = bin_diss([tmp_path / "diss.nc"], bin_width=10.0, method="depth")
        assert "epsilon" in result
        assert result["epsilon"].dims == ("bin", "profile")

    def test_time_method(self, tmp_path):
        """bin_diss with method='time' delegates to bin_by_time."""
        n = 80
        t_slow = np.linspace(0, 20, n)
        epsilon = np.random.lognormal(-20, 1, n)
        ds = xr.Dataset(
            {"epsilon": (["time_slow"], epsilon), "t_slow": (["time_slow"], t_slow)},
            coords={"time_slow": np.arange(n, dtype=float)},
        )
        ds.to_netcdf(tmp_path / "diss.nc")

        result = bin_diss([tmp_path / "diss.nc"], bin_width=5.0, method="time")
        assert "epsilon" in result
        assert "time" in result.dims


class TestBinChi:
    def test_time_method(self, tmp_path):
        """bin_chi with method='time' delegates to bin_by_time."""
        n = 60
        t_slow = np.linspace(0, 15, n)
        chi = np.random.lognormal(-14, 1, n)
        ds = xr.Dataset(
            {"chi": (["time_slow"], chi), "t_slow": (["time_slow"], t_slow)},
            coords={"time_slow": np.arange(n, dtype=float)},
        )
        ds.to_netcdf(tmp_path / "chi.nc")

        result = bin_chi([tmp_path / "chi.nc"], bin_width=5.0, method="time")
        assert "chi" in result
        assert "time" in result.dims


# ---------------------------------------------------------------------------
# Edge cases for bin_by_depth and bin_by_time
# ---------------------------------------------------------------------------


class TestBinByDepthEdges:
    def test_skip_profile_with_no_depth_var(self, tmp_path):
        """A profile NC with no depth/P/P_mean is silently skipped (line 148)."""
        # Profile with depth/P/P_mean
        good = xr.Dataset(
            {
                "P": (["t"], np.linspace(5, 50, 30)),
                "epsilon": (["t"], np.random.lognormal(-20, 1, 30)),
            }
        )
        good.to_netcdf(tmp_path / "good.nc")
        # Profile without any pressure/depth variable → skipped
        bad = xr.Dataset({"epsilon": (["t"], np.full(10, 1e-8))})
        bad.to_netcdf(tmp_path / "bad.nc")

        result = bin_by_depth(
            [tmp_path / "good.nc", tmp_path / "bad.nc"], bin_width=5.0
        )
        # The skipped profile leaves NaNs in its column; the good one populates
        assert "epsilon" in result
        assert result.sizes["profile"] == 2

    def test_skip_datetime_variable(self, tmp_path):
        """Datetime64 data variables are skipped in bin_by_depth (line 180)."""
        from datetime import datetime, timedelta

        n = 30
        dts = np.array(
            [datetime(2025, 1, 1) + timedelta(seconds=i) for i in range(n)],
            dtype="datetime64[ns]",
        )
        ds = xr.Dataset(
            {
                "P": (["t"], np.linspace(5, 50, n)),
                "epsilon": (["t"], np.random.lognormal(-20, 1, n)),
                "ts": (["t"], dts),  # datetime64 → skipped
            }
        )
        ds.to_netcdf(tmp_path / "p.nc")
        result = bin_by_depth([tmp_path / "p.nc"], bin_width=5.0)
        # The datetime variable should not appear as a binned data variable
        assert "ts" not in result.data_vars
        assert "epsilon" in result


class TestBinByTimeEdges:
    def test_short_time_range_pads_bin_edges(self, tmp_path):
        """When (t_max - t_min) < bin_width, bin_edges has only 1 entry → pad."""
        n = 5
        ds = xr.Dataset(
            {
                "epsilon": (["t"], np.full(n, 1e-8)),
                "t_slow": (["t"], np.linspace(0.0, 0.1, n)),
            },
            coords={"t": np.arange(n, dtype=float)},
        )
        ds.to_netcdf(tmp_path / "tiny.nc")
        # bin_width=10.0 >> t-range (0..0.1), triggering the padding fallback
        result = bin_by_time([tmp_path / "tiny.nc"], bin_width=10.0)
        # Should not crash; should produce a single-time-bin dataset
        assert "epsilon" in result
        assert result.sizes["time"] >= 1

    def test_skip_mismatched_length_var(self, tmp_path):
        """A variable whose length doesn't match the time coord is skipped."""
        n = 30
        ds = xr.Dataset(
            {
                "epsilon": (["t"], np.full(n, 1e-8)),
                "t_slow": (["t"], np.linspace(0, 1, n)),
                # mismatched_var on a different dim — skipped at line 264
                "mismatched_var": (["other"], np.zeros(5)),
            }
        )
        ds.to_netcdf(tmp_path / "p.nc")
        result = bin_by_time([tmp_path / "p.nc"], bin_width=0.1)
        assert "epsilon" in result
        assert "mismatched_var" not in result.data_vars

    def test_skip_datetime_variable_in_bin_by_time(self, tmp_path):
        """Datetime64 vars are skipped in bin_by_time (line 266)."""
        from datetime import datetime, timedelta

        n = 30
        dts = np.array(
            [datetime(2025, 1, 1) + timedelta(seconds=i) for i in range(n)],
            dtype="datetime64[ns]",
        )
        ds = xr.Dataset(
            {
                "epsilon": (["t"], np.full(n, 1e-8)),
                "t_slow": (["t"], np.linspace(0, 1, n)),
                "stime": (["t"], dts),
            }
        )
        ds.to_netcdf(tmp_path / "p.nc")
        result = bin_by_time([tmp_path / "p.nc"], bin_width=0.1)
        assert "epsilon" in result
        assert "stime" not in result.data_vars
