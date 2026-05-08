# May-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for processing.chi_combine — mk_chi_mean.

Mirrors tests/test_processing_epsilon_combine.py against the chi
analog so the two combiners stay in lockstep.
"""

import numpy as np
import xarray as xr

from odas_tpw.processing.chi_combine import mk_chi_mean


def _make_chi_ds(c1, c2, speed=None, nu=None, diss_length=512, fs_fast=512.0):
    """Build a minimal chi dataset for testing (1-D per-probe form)."""
    n = len(c1)
    return xr.Dataset(
        {
            "chi_1": (["time"], c1),
            "chi_2": (["time"], c2),
            "speed": (["time"], speed if speed is not None else np.full(n, 0.5)),
            "nu": (["time"], nu if nu is not None else np.full(n, 1e-6)),
        },
        coords={"time": np.arange(n, dtype=float)},
        attrs={"diss_length": diss_length, "fs_fast": fs_fast},
    )


class TestMkChiMean:
    def test_identical_probes_geometric_mean(self):
        n = 20
        c = np.full(n, 1e-8)
        ds = mk_chi_mean(_make_chi_ds(c, c))
        assert "chiMean" in ds
        np.testing.assert_allclose(ds["chiMean"].values, 1e-8, rtol=1e-6)

    def test_consistent_probes_both_survive(self):
        n = 20
        c1 = np.full(n, 1e-8)
        c2 = np.full(n, 2e-8)
        ds = mk_chi_mean(_make_chi_ds(c1, c2))
        expected = np.sqrt(1e-8 * 2e-8)
        np.testing.assert_allclose(ds["chiMean"].values, expected, rtol=0.1)

    def test_chi_minimum_floor(self):
        n = 10
        c1 = np.full(n, 1e-8)
        c2 = np.full(n, 1e-15)
        ds = mk_chi_mean(_make_chi_ds(c1, c2), chi_minimum=1e-13)
        np.testing.assert_allclose(ds["chiMean"].values, 1e-8, rtol=0.1)

    def test_nan_handling(self):
        n = 10
        c1 = np.full(n, 1e-7)
        c2 = np.full(n, 1e-7)
        c1[3] = np.nan
        c2[5] = np.nan
        ds = mk_chi_mean(_make_chi_ds(c1, c2))
        assert "chiMean" in ds
        assert np.isfinite(ds["chiMean"].values[0])

    def test_all_nan_row_does_not_raise(self):
        """Row where every probe is NaN (e.g. fom_max NaN'd both) should
        produce NaN in chiMean rather than tripping nanargmax's
        all-NaN-slice ValueError.
        """
        n = 10
        c1 = np.full(n, 1e-7)
        c2 = np.full(n, 5e-3)  # huge spread so iterative removal kicks in
        c1[4] = np.nan
        c2[4] = np.nan          # row 4 is all-NaN
        ds = mk_chi_mean(_make_chi_ds(c1, c2))
        chi_mean = ds["chiMean"].values
        assert np.isnan(chi_mean[4])
        # Other rows still computable.
        assert np.all(np.isfinite(chi_mean[[0, 1, 2, 3, 5, 6, 7, 8, 9]]))

    def test_chiLnSigma_added(self):
        n = 10
        c = np.full(n, 1e-8)
        ds = mk_chi_mean(_make_chi_ds(c, c))
        assert "chiLnSigma" in ds
        assert ds["chiLnSigma"].shape == (n,)

    def test_no_probe_vars(self):
        ds = xr.Dataset(
            {"speed": (["time"], [0.5, 0.5])},
            coords={"time": [0.0, 1.0]},
        )
        result = mk_chi_mean(ds)
        assert "chiMean" not in result

    def test_does_not_modify_original(self):
        n = 10
        c = np.full(n, 1e-8)
        ds = _make_chi_ds(c.copy(), c.copy())
        original_c1 = ds["chi_1"].values.copy()
        mk_chi_mean(ds)
        np.testing.assert_array_equal(ds["chi_1"].values, original_c1)

    def test_empty_time_dimension(self):
        ds = xr.Dataset(
            {
                "chi_1": (["time"], np.array([], dtype=float)),
                "chi_2": (["time"], np.array([], dtype=float)),
            },
            coords={"time": np.array([], dtype=float)},
        )
        result = mk_chi_mean(ds)
        assert "chiMean" not in result

    def test_defaults_no_speed_nu(self):
        n = 10
        c = np.full(n, 1e-7)
        ds = xr.Dataset(
            {
                "chi_1": (["time"], c.copy()),
                "chi_2": (["time"], c.copy()),
            },
            coords={"time": np.arange(n, dtype=float)},
            attrs={"diss_length": 512, "fs_fast": 512.0},
        )
        result = mk_chi_mean(ds)
        assert "chiMean" in result
        np.testing.assert_allclose(result["chiMean"].values, 1e-7, rtol=1e-6)

    def test_defaults_no_diss_length_fs(self):
        n = 10
        c = np.full(n, 1e-7)
        ds = xr.Dataset(
            {
                "chi_1": (["time"], c.copy()),
                "chi_2": (["time"], c.copy()),
                "speed": (["time"], np.full(n, 0.5)),
                "nu": (["time"], np.full(n, 1e-6)),
            },
            coords={"time": np.arange(n, dtype=float)},
        )
        result = mk_chi_mean(ds)
        assert "chiMean" in result
        np.testing.assert_allclose(result["chiMean"].values, 1e-7, rtol=1e-6)

    def test_diss_length_as_variable(self):
        n = 10
        c = np.full(n, 1e-7)
        ds = xr.Dataset(
            {
                "chi_1": (["time"], c.copy()),
                "chi_2": (["time"], c.copy()),
                "speed": (["time"], np.full(n, 0.5)),
                "nu": (["time"], np.full(n, 1e-6)),
                "diss_length": 256.0,
                "fs_fast": 512.0,
            },
            coords={"time": np.arange(n, dtype=float)},
        )
        result = mk_chi_mean(ds)
        assert "chiMean" in result
        np.testing.assert_allclose(result["chiMean"].values, 1e-7, rtol=1e-6)

    def test_single_probe(self):
        n = 10
        c = np.full(n, 5e-8)
        ds = xr.Dataset(
            {
                "chi_1": (["time"], c.copy()),
                "speed": (["time"], np.full(n, 0.5)),
                "nu": (["time"], np.full(n, 1e-6)),
            },
            coords={"time": np.arange(n, dtype=float)},
            attrs={"diss_length": 512, "fs_fast": 512.0},
        )
        result = mk_chi_mean(ds)
        assert "chiMean" in result
        np.testing.assert_allclose(result["chiMean"].values, 5e-8, rtol=1e-6)

    def test_outlier_probe_removed(self):
        n = 20
        c1 = np.full(n, 1e-8)
        c2 = np.full(n, 1e-3)
        ds = xr.Dataset(
            {
                "chi_1": (["time"], c1),
                "chi_2": (["time"], c2),
                "speed": (["time"], np.full(n, 0.5)),
                "nu": (["time"], np.full(n, 1e-6)),
            },
            coords={"time": np.arange(n, dtype=float)},
            attrs={"diss_length": 512, "fs_fast": 512.0},
        )
        result = mk_chi_mean(ds)
        assert "chiMean" in result
        np.testing.assert_allclose(result["chiMean"].values, 1e-8, rtol=0.1)

    def test_2d_chi_probe_time_split(self):
        """2-D chi(probe, time) is split into per-probe chi_1, chi_2 vars."""
        n = 20
        c = np.full(n, 1e-8)
        ds = xr.Dataset(
            {
                "chi": (["probe", "time"], np.stack([c, c * 2])),
                "speed": (["time"], np.full(n, 0.5)),
                "nu": (["time"], np.full(n, 1e-6)),
            },
            coords={
                "probe": ["t1", "t2"],
                "time": np.arange(n, dtype=float),
            },
            attrs={"diss_length": 512, "fs_fast": 512.0},
        )
        result = mk_chi_mean(ds)
        assert "chi_1" in result.data_vars
        assert "chi_2" in result.data_vars
        assert result["chi_1"].dims == ("time",)
        assert "chiMean" in result
        expected = np.sqrt(1e-8 * 2e-8)
        np.testing.assert_allclose(result["chiMean"].values, expected, rtol=0.1)

    def test_uses_epsilon_T_when_present(self):
        """epsilon_T(probe, time) drives the Kolmogorov length used for the
        CI, so swapping epsilon_T magnitude must change chiLnSigma."""
        n = 20
        c = np.full(n, 1e-8)
        eps_low = np.stack([np.full(n, 1e-12), np.full(n, 1e-12)])
        eps_hi = np.stack([np.full(n, 1e-6), np.full(n, 1e-6)])
        base = xr.Dataset(
            {
                "chi": (["probe", "time"], np.stack([c, c])),
                "speed": (["time"], np.full(n, 0.5)),
                "nu": (["time"], np.full(n, 1e-6)),
            },
            coords={"probe": ["t1", "t2"], "time": np.arange(n, dtype=float)},
            attrs={"diss_length": 512, "fs_fast": 512.0},
        )
        ds_low = base.copy()
        ds_low["epsilon_T"] = (["probe", "time"], eps_low)
        ds_hi = base.copy()
        ds_hi["epsilon_T"] = (["probe", "time"], eps_hi)
        sigma_low = mk_chi_mean(ds_low)["chiLnSigma"].values
        sigma_hi = mk_chi_mean(ds_hi)["chiLnSigma"].values
        # Different epsilon_T → different L_K → different sigma_ln_chi.
        assert not np.allclose(sigma_low, sigma_hi)
