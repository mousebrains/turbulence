# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for rsi.binning.bin_by_depth and rsi.combine.combine_profiles."""

import numpy as np
import pytest
import xarray as xr

from odas_tpw.rsi.binning import bin_by_depth
from odas_tpw.rsi.combine import combine_profiles

# ---------------------------------------------------------------------------
# bin_by_depth
# ---------------------------------------------------------------------------


class TestBinByDepthLogMean:
    """Log-mean (geometric mean) averaging for epsilon/chi variables."""

    def test_epsilon_uses_log_mean(self):
        pres = np.array([0.5, 1.5, 0.7, 1.3])
        eps_vals = np.array([1e-8, 1e-6, 1e-9, 1e-7])
        ds = bin_by_depth(pres, {"epsilon": eps_vals}, bin_size=2.0)
        # All four points land in a single 2-dbar bin.
        # Geometric mean = exp(mean(log(vals)))
        expected = np.exp(np.mean(np.log(eps_vals)))
        np.testing.assert_allclose(ds["epsilon"].values[0], expected, rtol=1e-10)

    def test_chi_uses_log_mean(self):
        pres = np.array([0.5, 1.5])
        chi_vals = np.array([1e-10, 1e-8])
        ds = bin_by_depth(pres, {"chi": chi_vals}, bin_size=2.0)
        expected = np.exp(np.mean(np.log(chi_vals)))
        np.testing.assert_allclose(ds["chi"].values[0], expected, rtol=1e-10)

    def test_custom_log_mean_vars(self):
        pres = np.array([0.5, 1.5])
        vals = np.array([1e-4, 1e-2])
        ds = bin_by_depth(pres, {"my_var": vals}, bin_size=2.0, log_mean_vars={"my_var"})
        expected = np.exp(np.mean(np.log(vals)))
        np.testing.assert_allclose(ds["my_var"].values[0], expected, rtol=1e-10)


class TestBinByDepthArithmeticMean:
    """Arithmetic mean for non-log variables (temperature, etc.)."""

    def test_temperature_uses_arithmetic_mean(self):
        pres = np.array([0.5, 1.5, 0.8])
        temp = np.array([10.0, 12.0, 11.0])
        ds = bin_by_depth(pres, {"temperature": temp}, bin_size=2.0)
        np.testing.assert_allclose(ds["temperature"].values[0], np.mean(temp))

    def test_multiple_bins(self):
        pres = np.array([0.5, 1.5, 2.5, 3.5])
        temp = np.array([10.0, 12.0, 14.0, 16.0])
        ds = bin_by_depth(pres, {"temperature": temp}, bin_size=2.0)
        # Bin 0-2: mean(10, 12) = 11; Bin 2-4: mean(14, 16) = 15
        np.testing.assert_allclose(ds["temperature"].values[0], 11.0)
        np.testing.assert_allclose(ds["temperature"].values[1], 15.0)


class TestBinByDepthBinSize:
    """Bin-size parameter controls bin width."""

    def test_small_bin_size(self):
        pres = np.arange(0.25, 5.0, 0.5)  # 0.25, 0.75, ..., 4.75
        temp = np.ones_like(pres) * 20.0
        ds = bin_by_depth(pres, {"T": temp}, bin_size=1.0)
        assert ds.sizes["depth_bin"] == 5

    def test_large_bin_size(self):
        pres = np.arange(0.5, 10.0, 0.5)
        temp = np.ones_like(pres)
        ds = bin_by_depth(pres, {"T": temp}, bin_size=5.0)
        assert ds.sizes["depth_bin"] == 2

    def test_pres_range_overrides_default(self):
        pres = np.array([3.5, 4.5, 5.5])
        temp = np.array([10.0, 11.0, 12.0])
        ds = bin_by_depth(pres, {"T": temp}, bin_size=2.0, pres_range=(0.0, 10.0))
        assert ds.coords["depth_bin"].values[0] == pytest.approx(1.0)
        assert ds.sizes["depth_bin"] == 5


class TestBinByDepthNaN:
    """NaN handling — NaN values should be excluded from means."""

    def test_nan_excluded_from_arithmetic_mean(self):
        pres = np.array([0.5, 1.0, 1.5])
        temp = np.array([10.0, np.nan, 14.0])
        ds = bin_by_depth(pres, {"T": temp}, bin_size=2.0)
        np.testing.assert_allclose(ds["T"].values[0], 12.0)  # mean(10, 14)

    def test_nan_excluded_from_log_mean(self):
        pres = np.array([0.5, 1.0, 1.5])
        eps = np.array([1e-8, np.nan, 1e-6])
        ds = bin_by_depth(pres, {"epsilon": eps}, bin_size=2.0)
        expected = np.exp(np.mean(np.log([1e-8, 1e-6])))
        np.testing.assert_allclose(ds["epsilon"].values[0], expected, rtol=1e-10)

    def test_all_nan_yields_nan_bin(self):
        pres = np.array([0.5, 1.5])
        temp = np.array([np.nan, np.nan])
        ds = bin_by_depth(pres, {"T": temp}, bin_size=2.0)
        assert np.isnan(ds["T"].values[0])

    def test_nan_pressure_returns_empty(self):
        pres = np.array([np.nan, np.nan, np.nan])
        ds = bin_by_depth(pres, {"T": np.array([1.0, 2.0, 3.0])})
        assert len(ds.data_vars) == 0


class TestBinByDepthEdgeCases:
    """Edge cases: empty data, single point, 2-D arrays."""

    def test_single_point(self):
        # Use a mid-bin value; exact bin-edge values produce zero-width range.
        ds = bin_by_depth(np.array([5.3]), {"T": np.array([20.0])}, bin_size=1.0)
        assert ds.sizes["depth_bin"] >= 1
        np.testing.assert_allclose(ds["T"].values[0], 20.0)

    def test_2d_array_probe_reduction_arithmetic(self):
        pres = np.array([0.5, 1.5])
        # 2 probes x 2 times
        temp_2d = np.array([[10.0, 12.0], [14.0, 16.0]])
        ds = bin_by_depth(pres, {"T": temp_2d}, bin_size=2.0)
        # Probe mean first: [12, 14], then bin mean: 13
        np.testing.assert_allclose(ds["T"].values[0], 13.0)

    def test_2d_array_probe_reduction_log(self):
        pres = np.array([0.5, 1.5])
        eps_2d = np.array([[1e-8, 1e-6], [1e-10, 1e-4]])
        ds = bin_by_depth(pres, {"epsilon": eps_2d}, bin_size=2.0)
        # Geometric mean across probes first, then across bin
        probe_geomean = np.exp(np.nanmean(np.log(eps_2d), axis=0))
        expected = np.exp(np.mean(np.log(probe_geomean)))
        np.testing.assert_allclose(ds["epsilon"].values[0], expected, rtol=1e-8)


class TestBinByDepthOutputStructure:
    """Output Dataset structure and attributes."""

    def test_returns_xr_dataset(self):
        pres = np.array([1.0, 2.0, 3.0])
        ds = bin_by_depth(pres, {"T": np.array([10.0, 11.0, 12.0])})
        assert isinstance(ds, xr.Dataset)

    def test_depth_bin_coord(self):
        pres = np.array([0.5, 1.5, 2.5])
        ds = bin_by_depth(pres, {"T": np.ones(3)}, bin_size=1.0)
        assert "depth_bin" in ds.coords
        assert ds.coords["depth_bin"].attrs["units"] == "dbar"

    def test_bin_size_attr(self):
        pres = np.array([0.5, 1.5])
        ds = bin_by_depth(pres, {"T": np.ones(2)}, bin_size=2.5)
        assert ds.attrs["bin_size"] == 2.5

    def test_multiple_variables(self):
        pres = np.array([0.5, 1.5])
        ds = bin_by_depth(
            pres,
            {"epsilon": np.array([1e-8, 1e-7]), "T": np.array([10.0, 12.0])},
            bin_size=2.0,
        )
        assert "epsilon" in ds.data_vars
        assert "T" in ds.data_vars


# ---------------------------------------------------------------------------
# combine_profiles
# ---------------------------------------------------------------------------


class TestCombineProfiles:
    """Tests for combine_profiles."""

    def _make_binned(self, depth_vals, var_dict):
        """Helper: build a bin_by_depth-style Dataset."""
        data_vars = {k: (["depth_bin"], v) for k, v in var_dict.items()}
        return xr.Dataset(data_vars, coords={"depth_bin": depth_vals})

    def test_combine_two_profiles(self):
        ds1 = self._make_binned([0.5, 1.5], {"T": [10.0, 11.0]})
        ds2 = self._make_binned([0.5, 1.5], {"T": [12.0, 13.0]})
        result = combine_profiles([ds1, ds2])
        assert result.sizes["profile"] == 2
        assert result.sizes["depth_bin"] == 2
        np.testing.assert_allclose(result["T"].values[0], [10.0, 11.0])
        np.testing.assert_allclose(result["T"].values[1], [12.0, 13.0])

    def test_combine_three_profiles(self):
        ds1 = self._make_binned([1.0, 2.0], {"T": [10.0, 11.0]})
        ds2 = self._make_binned([1.0, 2.0], {"T": [12.0, 13.0]})
        ds3 = self._make_binned([1.0, 2.0], {"T": [14.0, 15.0]})
        result = combine_profiles([ds1, ds2, ds3])
        assert result.sizes["profile"] == 3
        assert result["T"].dims == ("profile", "depth_bin")

    def test_different_depth_ranges_aligned(self):
        ds1 = self._make_binned([0.5, 1.5, 2.5], {"T": [10.0, 11.0, 12.0]})
        ds2 = self._make_binned([1.5, 2.5, 3.5], {"T": [20.0, 21.0, 22.0]})
        result = combine_profiles([ds1, ds2])
        # Union grid: [0.5, 1.5, 2.5, 3.5]
        assert result.sizes["depth_bin"] == 4
        depths = result.coords["depth_bin"].values
        np.testing.assert_allclose(depths, [0.5, 1.5, 2.5, 3.5])

    def test_nan_fill_for_unmatched_depths(self):
        ds1 = self._make_binned([0.5, 1.5], {"T": [10.0, 11.0]})
        ds2 = self._make_binned([1.5, 2.5], {"T": [20.0, 21.0]})
        result = combine_profiles([ds1, ds2])
        # ds1 has no data at depth=2.5 → NaN
        assert np.isnan(result["T"].values[0, 2])
        # ds2 has no data at depth=0.5 → NaN
        assert np.isnan(result["T"].values[1, 0])
        # Overlap at depth=1.5 → finite
        assert np.isfinite(result["T"].values[0, 1])
        assert np.isfinite(result["T"].values[1, 1])

    def test_metadata_propagation(self):
        ds1 = self._make_binned([0.5], {"T": [10.0]})
        ds2 = self._make_binned([0.5], {"T": [12.0]})
        meta = [
            {"start_time": "2025-01-10T00:00", "file": "cast001.p"},
            {"start_time": "2025-01-10T01:00", "file": "cast002.p"},
        ]
        result = combine_profiles([ds1, ds2], profile_metadata=meta)
        assert result.attrs["profile_0_file"] == "cast001.p"
        assert result.attrs["profile_1_start_time"] == "2025-01-10T01:00"

    def test_output_dimensions(self):
        ds1 = self._make_binned([1.0, 2.0, 3.0], {"eps": [1e-8, 1e-7, 1e-6]})
        ds2 = self._make_binned([1.0, 2.0, 3.0], {"eps": [2e-8, 2e-7, 2e-6]})
        result = combine_profiles([ds1, ds2])
        assert result["eps"].dims == ("profile", "depth_bin")
        assert result["eps"].shape == (2, 3)

    def test_depth_bin_attributes(self):
        ds1 = self._make_binned([0.5], {"T": [10.0]})
        result = combine_profiles([ds1])
        assert result.coords["depth_bin"].attrs["units"] == "dbar"
        assert result.coords["profile"].attrs["long_name"] == "profile number"

    def test_empty_list_returns_empty(self):
        result = combine_profiles([])
        assert isinstance(result, xr.Dataset)
        assert len(result.data_vars) == 0

    def test_mixed_variables_across_profiles(self):
        ds1 = self._make_binned([0.5], {"T": [10.0], "S": [35.0]})
        ds2 = self._make_binned([0.5], {"T": [12.0]})
        result = combine_profiles([ds1, ds2])
        assert "T" in result.data_vars
        assert "S" in result.data_vars
        # ds2 has no S → NaN
        assert np.isnan(result["S"].values[1, 0])
        np.testing.assert_allclose(result["S"].values[0, 0], 35.0)
