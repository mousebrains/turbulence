# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for processing.epsilon_combine — mk_epsilon_mean."""

import numpy as np
import xarray as xr

from odas_tpw.processing.epsilon_combine import mk_epsilon_mean


def _make_diss_ds(e1, e2, speed=None, nu=None, diss_length=512, fs_fast=512.0):
    """Build a minimal dissipation dataset for testing."""
    n = len(e1)
    ds = xr.Dataset(
        {
            "e_1": (["time"], e1),
            "e_2": (["time"], e2),
            "speed": (["time"], speed if speed is not None else np.full(n, 0.5)),
            "nu": (["time"], nu if nu is not None else np.full(n, 1e-6)),
        },
        coords={"time": np.arange(n, dtype=float)},
        attrs={"diss_length": diss_length, "fs_fast": fs_fast},
    )
    return ds


class TestMkEpsilonMean:
    def test_identical_probes_geometric_mean(self):
        """Two identical probes → geometric mean = same value."""
        n = 20
        e = np.full(n, 1e-8)
        ds = mk_epsilon_mean(_make_diss_ds(e, e))
        assert "epsilonMean" in ds
        np.testing.assert_allclose(ds["epsilonMean"].values, 1e-8, rtol=1e-6)

    def test_consistent_probes_both_survive(self):
        """Two consistent probes → both contribute to mean."""
        n = 20
        e1 = np.full(n, 1e-8)
        e2 = np.full(n, 2e-8)  # factor of 2 — well within CI
        ds = mk_epsilon_mean(_make_diss_ds(e1, e2))
        mean_val = ds["epsilonMean"].values
        # Geometric mean of 1e-8 and 2e-8 = sqrt(2e-16) ≈ 1.41e-8
        expected = np.sqrt(1e-8 * 2e-8)
        np.testing.assert_allclose(mean_val, expected, rtol=0.1)

    def test_epsilon_minimum_floor(self):
        """Values below epsilon_minimum should be NaN'd."""
        n = 10
        e1 = np.full(n, 1e-8)
        e2 = np.full(n, 1e-15)  # below default minimum
        ds = mk_epsilon_mean(_make_diss_ds(e1, e2), epsilon_minimum=1e-13)
        # e2 should be NaN'd, so mean = e1
        mean_val = ds["epsilonMean"].values
        np.testing.assert_allclose(mean_val, 1e-8, rtol=0.1)

    def test_nan_handling(self):
        """NaN values in input should not crash."""
        n = 10
        e1 = np.full(n, 1e-7)
        e2 = np.full(n, 1e-7)
        e1[3] = np.nan
        e2[5] = np.nan
        ds = mk_epsilon_mean(_make_diss_ds(e1, e2))
        assert "epsilonMean" in ds
        # Non-NaN positions should have values
        assert np.isfinite(ds["epsilonMean"].values[0])

    def test_epsilonLnSigma_added(self):
        n = 10
        e = np.full(n, 1e-8)
        ds = mk_epsilon_mean(_make_diss_ds(e, e))
        assert "epsilonLnSigma" in ds
        assert ds["epsilonLnSigma"].shape == (n,)

    def test_no_probe_vars(self):
        """Dataset without e_* variables should warn and return unchanged."""
        ds = xr.Dataset(
            {"speed": (["time"], [0.5, 0.5])},
            coords={"time": [0.0, 1.0]},
        )
        result = mk_epsilon_mean(ds)
        assert "epsilonMean" not in result

    def test_does_not_modify_original(self):
        """mk_epsilon_mean should not modify the input dataset."""
        n = 10
        e = np.full(n, 1e-8)
        ds = _make_diss_ds(e.copy(), e.copy())
        original_e1 = ds["e_1"].values.copy()
        mk_epsilon_mean(ds)
        np.testing.assert_array_equal(ds["e_1"].values, original_e1)

    def test_empty_time_dimension(self):
        """n_time==0 → return ds unchanged (no epsilonMean added)."""
        ds = xr.Dataset(
            {
                "e_1": (["time"], np.array([], dtype=float)),
                "e_2": (["time"], np.array([], dtype=float)),
            },
            coords={"time": np.array([], dtype=float)},
        )
        result = mk_epsilon_mean(ds)
        assert "epsilonMean" not in result

    def test_defaults_no_speed_nu(self):
        """Dataset with e_1, e_2 but no speed or nu uses defaults."""
        n = 10
        e = np.full(n, 1e-7)
        ds = xr.Dataset(
            {
                "e_1": (["time"], e.copy()),
                "e_2": (["time"], e.copy()),
            },
            coords={"time": np.arange(n, dtype=float)},
            attrs={"diss_length": 512, "fs_fast": 512.0},
        )
        result = mk_epsilon_mean(ds)
        assert "epsilonMean" in result
        np.testing.assert_allclose(result["epsilonMean"].values, 1e-7, rtol=1e-6)

    def test_defaults_no_diss_length_fs(self):
        """Dataset without diss_length or fs_fast attrs uses 512.0 defaults."""
        n = 10
        e = np.full(n, 1e-7)
        ds = xr.Dataset(
            {
                "e_1": (["time"], e.copy()),
                "e_2": (["time"], e.copy()),
                "speed": (["time"], np.full(n, 0.5)),
                "nu": (["time"], np.full(n, 1e-6)),
            },
            coords={"time": np.arange(n, dtype=float)},
        )
        result = mk_epsilon_mean(ds)
        assert "epsilonMean" in result
        np.testing.assert_allclose(result["epsilonMean"].values, 1e-7, rtol=1e-6)

    def test_diss_length_as_variable(self):
        """diss_length as a data variable (not attr) should be read correctly."""
        n = 10
        e = np.full(n, 1e-7)
        ds = xr.Dataset(
            {
                "e_1": (["time"], e.copy()),
                "e_2": (["time"], e.copy()),
                "speed": (["time"], np.full(n, 0.5)),
                "nu": (["time"], np.full(n, 1e-6)),
                "diss_length": 256.0,
                "fs_fast": 512.0,
            },
            coords={"time": np.arange(n, dtype=float)},
        )
        # No attrs — diss_length and fs_fast come from data variables
        assert "diss_length" not in ds.attrs
        assert "diss_length" in ds.data_vars
        result = mk_epsilon_mean(ds)
        assert "epsilonMean" in result
        np.testing.assert_allclose(result["epsilonMean"].values, 1e-7, rtol=1e-6)

    def test_single_probe(self):
        """Only e_1, no e_2 — geometric mean should equal e_1."""
        n = 10
        e = np.full(n, 5e-8)
        ds = xr.Dataset(
            {
                "e_1": (["time"], e.copy()),
                "speed": (["time"], np.full(n, 0.5)),
                "nu": (["time"], np.full(n, 1e-6)),
            },
            coords={"time": np.arange(n, dtype=float)},
            attrs={"diss_length": 512, "fs_fast": 512.0},
        )
        result = mk_epsilon_mean(ds)
        assert "epsilonMean" in result
        np.testing.assert_allclose(result["epsilonMean"].values, 5e-8, rtol=1e-6)

    def test_outlier_probe_removed(self):
        """Probes many orders apart — CI should remove outlier, mean ~ e_1."""
        n = 20
        e1 = np.full(n, 1e-8)
        e2 = np.full(n, 1e-3)  # 5 orders of magnitude apart
        ds = xr.Dataset(
            {
                "e_1": (["time"], e1),
                "e_2": (["time"], e2),
                "speed": (["time"], np.full(n, 0.5)),
                "nu": (["time"], np.full(n, 1e-6)),
            },
            coords={"time": np.arange(n, dtype=float)},
            attrs={"diss_length": 512, "fs_fast": 512.0},
        )
        result = mk_epsilon_mean(ds)
        assert "epsilonMean" in result
        # After removing e_2 as outlier, mean should be close to e_1
        np.testing.assert_allclose(result["epsilonMean"].values, 1e-8, rtol=0.1)

    def test_2d_epsilon_probe_time(self):
        """2D epsilon(probe, time) is split into per-probe variables."""
        n = 20
        e = np.full(n, 1e-8)
        ds = xr.Dataset(
            {
                "epsilon": (["probe", "time"], np.stack([e, e * 2])),
                "speed": (["time"], np.full(n, 0.5)),
                "nu": (["time"], np.full(n, 1e-6)),
            },
            coords={
                "probe": ["sh1", "sh2"],
                "time": np.arange(n, dtype=float),
            },
            attrs={"diss_length": 512, "fs_fast": 512.0},
        )
        result = mk_epsilon_mean(ds)
        assert "epsilonMean" in result
        expected = np.sqrt(1e-8 * 2e-8)
        np.testing.assert_allclose(result["epsilonMean"].values, expected, rtol=0.1)
