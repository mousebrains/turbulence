# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb.binning — depth/time binning."""

import numpy as np
import pytest
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
    def test_lnsigma_quadrature_holds_under_median(self, tmp_path):
        """*LnSigma vars combine as RMS = sqrt(mean(sigma^2)) regardless of the
        dataset aggregation. Under median binning the old code computed
        sqrt(median(sigma^2)) and silently mis-stated the uncertainty (#81)."""
        n = 3
        # All three samples land in one wide depth bin.
        depth = np.array([1.0, 2.0, 3.0])
        sig = np.array([0.2, 0.4, 3.0])
        ds = xr.Dataset(
            {
                "epsilonLnSigma": (["time_slow"], sig),
                "depth": (["time_slow"], depth),
            },
            coords={"time_slow": np.arange(n, dtype=float)},
        )
        ds.to_netcdf(tmp_path / "prof00.nc")
        files = sorted(tmp_path.glob("*.nc"))
        result = bin_by_depth(files, bin_width=100.0, aggregation="median")
        val = float(np.asarray(result["epsilonLnSigma"].values).ravel()[0])
        rms = float(np.sqrt(np.mean(sig**2)))      # 1.751
        assert abs(val - rms) < 1e-9
        assert abs(val - float(np.median(sig))) > 1.0  # not the median (0.4)

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

    def test_non_binary_exact_bin_width_no_spurious_trailing_bin(self, tmp_path):
        """np.arange edge construction emitted a spurious extra trailing bin for
        non-binary-exact bin_width (e.g. 0.7), stealing the deepest boundary
        sample into a near-empty bin past d_max.

        Depths 0..7 m at bin_width=0.7 must yield 10 bins (edges 0,0.7,...,7.0),
        not 11 (the old np.arange added a spurious [7.0,7.7) bin). The deepest
        sample at depth==7.0 must land in the last real bin [6.3,7.0).
        On the OLD code result.sizes['bin']==11 and that sample lands in the
        spurious bin, so both assertions below fail.
        """
        depth = np.arange(0.0, 7.0 + 0.001, 0.1)  # 0.0 .. 7.0 m, includes 7.0
        # Tag each sample by depth so we can identify where 7.0 landed.
        T = depth.copy()
        ds = xr.Dataset(
            {"T": (["time_slow"], T), "depth": (["time_slow"], depth)},
            coords={"time_slow": np.arange(len(depth), dtype=float)},
        )
        ds.to_netcdf(tmp_path / "prof00.nc")
        files = sorted(tmp_path.glob("*.nc"))

        result = bin_by_depth(files, bin_width=0.7)
        assert result.sizes["bin"] == 10
        # Last bin center is 6.65 (bin [6.3, 7.0)); it must be populated by the
        # deepest samples including depth==7.0, not left near-empty.
        centers = result.coords["bin"].values
        np.testing.assert_allclose(centers[-1], 6.65, atol=1e-9)
        last_bin_mean = float(np.asarray(result["T"].values)[-1, 0])
        # Samples in [6.3, 7.0] (inclusive upper edge): mean of 6.3..7.0 by 0.1.
        in_last = depth[(depth >= 6.3 - 1e-9) & (depth <= 7.0 + 1e-9)]
        np.testing.assert_allclose(last_bin_mean, float(np.mean(in_last)), atol=1e-9)

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
        assert int(result["n_samples"].sum()) == n
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

    def test_cf_time_units_use_common_epoch_seconds(self, tmp_path):
        """Decoded CF time variables with different origins should bin without crashing."""
        starts = [
            "2026-01-01 00:00:00",
            "2026-01-01 00:00:10",
        ]
        for i, units_start in enumerate(starts):
            ds = xr.Dataset(
                {
                    "T": (["sample"], np.array([10.0 + i, 11.0 + i, 12.0 + i])),
                    "t_slow": (["sample"], np.array([0.0, 1.0, 2.0])),
                },
                coords={"sample": np.arange(3)},
            )
            ds["t_slow"].attrs["units"] = f"seconds since {units_start}"
            ds["t_slow"].attrs["calendar"] = "standard"
            ds.to_netcdf(tmp_path / f"prof{i}.nc")

        result = bin_by_time(sorted(tmp_path.glob("prof*.nc")), bin_width=1.0)
        assert "T" in result
        assert result["time"].dtype.kind == "f"
        assert float(result["time"].min()) > 1_700_000_000.0

    def test_no_time_coord(self, tmp_path):
        """Profile NC without any recognized time coord returns empty Dataset."""
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

    def test_fill_value_and_packing_are_cf_decoded(self, tmp_path):
        """CF `_FillValue` sentinels and `scale_factor` packing must be decoded
        before binning — not averaged as raw values.

        Regression for the 2026-07-03 review (GPT-5.5 F2): the raw netCDF4 read
        had mask+scale disabled, so a `-999` fill was averaged as data and a
        packed int16 was averaged in raw counts.
        """
        import netCDF4 as nc

        # (a) a -999 fill on a data var -> the filled cell drops out (NaN), so the
        #     bin is the mean of the two real values (1, 3) = 2, not -331.7.
        p1 = tmp_path / "fill.nc"
        ds = nc.Dataset(str(p1), "w")
        ds.createDimension("z", 3)
        ds.createVariable("depth", "f8", ("z",))[:] = [0.0, 1.0, 2.0]
        t = ds.createVariable("T", "f8", ("z",), fill_value=-999.0)
        t[0] = 1.0
        t[1] = t._FillValue
        t[2] = 3.0
        ds.close()
        assert bin_by_depth([p1], bin_width=10.0)["T"].values.ravel()[0] == pytest.approx(2.0)

        # (b) a packed int16 (scale_factor=0.1): physical values [10,20] are
        #     stored as counts [100,200] and must read back as [10,20] -> binned
        #     15.0, not the raw-count mean 150.
        p2 = tmp_path / "packed.nc"
        ds = nc.Dataset(str(p2), "w")
        ds.createDimension("z", 2)
        ds.createVariable("depth", "f8", ("z",))[:] = [0.0, 1.0]
        tp = ds.createVariable("T", "i2", ("z",))
        tp.scale_factor = 0.1
        tp[:] = [10.0, 20.0]  # physical; stored as packed counts 100, 200
        ds.close()
        assert bin_by_depth([p2], bin_width=5.0)["T"].values.ravel()[0] == pytest.approx(15.0)

    def test_nan_fill_happy_path_and_int_flag_dtype_preserved(self, tmp_path):
        """The package's own files (NaN fill, unpacked int flags) are unchanged:
        NaN cells drop out and integer qc flags keep their integer dtype."""
        import netCDF4 as nc

        from odas_tpw.perturb.binning import _load_profile_snapshot

        p = tmp_path / "happy.nc"
        ds = nc.Dataset(str(p), "w")
        ds.createDimension("z", 3)
        ds.createVariable("depth", "f8", ("z",))[:] = [0.0, 1.0, 2.0]
        e = ds.createVariable("epsilonMean", "f8", ("z",), fill_value=np.nan)
        e[:] = [1e-8, np.nan, 3e-8]
        ds.createVariable("qc_drop_epsilon", "i4", ("z",))[:] = [0, 1, 0]
        ds.close()

        snap = _load_profile_snapshot(p)
        assert snap["vars"]["qc_drop_epsilon"].dtype.kind in ("i", "u")  # not widened
        binned = bin_by_depth([p], bin_width=10.0)["epsilonMean"].values.ravel()[0]
        assert binned == pytest.approx(2e-8)

    def test_uint8_qc_flag_255_is_or_pooled_not_averaged(self, tmp_path):
        """A uint8 qc bitfield whose value equals the type's default fill (255,
        all 8 flags set) must be OR-pooled, not masked-and-averaged.

        Regression for the adversarial review of F2: enabling CF decode made
        netCDF4 auto-mask cells == the uint8 default fill (255) even with NO
        explicit _FillValue, widening the bitfield to float and averaging it
        (e.g. [1,255,2] -> ~86) instead of OR-pooling to 255.
        """
        import netCDF4 as nc

        from odas_tpw.perturb.binning import _load_profile_snapshot

        p = tmp_path / "qc255.nc"
        ds = nc.Dataset(str(p), "w")
        ds.createDimension("z", 3)
        ds.createVariable("depth", "f8", ("z",))[:] = [0.0, 1.0, 2.0]
        ds.createVariable("epsilonMean", "f8", ("z",))[:] = [1e-8, 2e-8, 3e-8]
        # uint8 flag, NO explicit _FillValue; 255 is a legal all-flags-set value.
        ds.createVariable("qc_drop_epsilon", "u1", ("z",))[:] = [1, 255, 2]
        ds.close()

        snap = _load_profile_snapshot(p)
        # Read raw, dtype preserved, the legal 255 not masked away.
        assert snap["vars"]["qc_drop_epsilon"].dtype.kind == "u"
        assert 255 in snap["vars"]["qc_drop_epsilon"]
        # All three depths fall in one bin -> OR-pool 1|255|2 == 255 (not ~86).
        binned = bin_by_depth([p], bin_width=10.0)["qc_drop_epsilon"].values.ravel()[0]
        assert binned == 255


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

    def test_diagnostics_unsupported_warns_once(self, tmp_path, caplog):
        """diagnostics=True on time binning warns that *_std / *_n / n_samples
        are not produced (the depth path emits them; the time path does not),
        and the data is still binned to means (#40/#53)."""
        n = 30
        ds = xr.Dataset(
            {
                "epsilon": (["t"], np.full(n, 1e-8)),
                "t_slow": (["t"], np.linspace(0.0, 5.0, n)),
            }
        )
        ds.to_netcdf(tmp_path / "p.nc")
        with caplog.at_level("WARNING", logger="odas_tpw.perturb.binning"):
            result = bin_by_time([tmp_path / "p.nc"], bin_width=1.0, diagnostics=True)
        assert "epsilon" in result  # means still produced
        assert "epsilon_std" not in result.data_vars
        assert "epsilon_n" not in result.data_vars
        assert "n_samples" not in result.data_vars
        assert "not produced for time binning" in caplog.text

    def test_diagnostics_warns_on_dropped_file_no_time_coord(self, tmp_path, caplog):
        """diagnostics=True surfaces the silent drop of a file lacking any time
        coordinate; without diagnostics the drop stays quiet (#18/#24)."""
        n = 30
        ds = xr.Dataset(
            {"T": (["samples"], np.ones(n))},
            coords={"samples": np.arange(n, dtype=float)},
        )
        ds.to_netcdf(tmp_path / "prof.nc")

        with caplog.at_level("WARNING", logger="odas_tpw.perturb.binning"):
            result = bin_by_time(
                [tmp_path / "prof.nc"], bin_width=1.0, diagnostics=True
            )
        assert len(result.data_vars) == 0
        assert "no time coordinate" in caplog.text
        assert "prof.nc" in caplog.text

    def test_no_diagnostics_is_quiet_on_dropped_file(self, tmp_path, caplog):
        """The same dropped file is NOT logged when diagnostics is off."""
        n = 30
        ds = xr.Dataset(
            {"T": (["samples"], np.ones(n))},
            coords={"samples": np.arange(n, dtype=float)},
        )
        ds.to_netcdf(tmp_path / "prof.nc")

        with caplog.at_level("WARNING", logger="odas_tpw.perturb.binning"):
            bin_by_time([tmp_path / "prof.nc"], bin_width=1.0, diagnostics=False)
        assert "no time coordinate" not in caplog.text

    def test_diagnostics_warns_on_no_finite_times(self, tmp_path, caplog):
        """A file whose times are all NaN is dropped; diagnostics warns (#18/#24)."""
        n = 10
        ds = xr.Dataset(
            {
                "epsilon": (["t"], np.full(n, 1e-8)),
                "t_slow": (["t"], np.full(n, np.nan)),
            }
        )
        ds.to_netcdf(tmp_path / "allnan.nc")

        with caplog.at_level("WARNING", logger="odas_tpw.perturb.binning"):
            bin_by_time([tmp_path / "allnan.nc"], bin_width=1.0, diagnostics=True)
        assert "no finite times" in caplog.text
        assert "allnan.nc" in caplog.text
