# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for processing.epsilon_combine — mk_epsilon_mean."""

import numpy as np
import pytest
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

    def test_all_nan_row_does_not_raise(self):
        """Row where every probe is NaN (e.g. fom_max NaN'd both) must
        not trip nanargmax's all-NaN-slice ValueError -- the iterative
        removal loop has nothing to do on an empty row.
        """
        n = 10
        e1 = np.full(n, 1e-7)
        e2 = np.full(n, 1e-3)  # huge spread so iterative removal triggers
        e1[4] = np.nan
        e2[4] = np.nan          # row 4 is all-NaN
        ds = mk_epsilon_mean(_make_diss_ds(e1, e2))
        eps_mean = ds["epsilonMean"].values
        assert np.isnan(eps_mean[4])
        assert np.all(np.isfinite(eps_mean[[0, 1, 2, 3, 5, 6, 7, 8, 9]]))

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

    def test_probe_sort_key_orders_by_trailing_int(self):
        """e_2 must sort before e_10; plain lexicographic order puts e_10
        first (#24)."""
        from odas_tpw.processing.epsilon_combine import _probe_sort_key

        names = ["e_10", "e_2", "e_1"]
        assert sorted(names, key=_probe_sort_key) == ["e_1", "e_2", "e_10"]

    def test_combine_never_prunes_below_two_probes(self):
        """A 3-probe row whose estimates all mutually disagree must collapse to
        the geometric mean of the two closest, never to a single survivor (the
        documented 'keep both' rule applied per-row) (#23/#49). A huge speed
        forces CF95~0 so the loop would otherwise drop to one probe."""
        # Probes 1e-12, 1e-7, 1e-6: the unambiguous furthest-from-ln-mean is
        # 1e-12, dropped first, leaving (1e-7, 1e-6) -> geometric mean
        # sqrt(1e-13) ~ 3.16e-7. With the >=2 floor the loop stops there; without
        # it the loop continues to a single survivor (1e-7 or 1e-6).
        ds = xr.Dataset(
            {
                "e_1": (["time"], [1e-12]),
                "e_2": (["time"], [1e-7]),
                "e_3": (["time"], [1e-6]),
                "speed": (["time"], [1.0e6]),  # drives L_hat huge -> CF95 ~ 0
                "nu": (["time"], [1.0e-6]),
            },
            coords={"time": [0.0]},
            attrs={"diss_length": 512, "fs_fast": 512.0},
        )
        out = mk_epsilon_mean(ds)
        assert out["epsilonMean"].values[0] == pytest.approx(np.sqrt(1e-13), rel=1e-6)

    def test_all_nan_row_with_three_probes_does_not_raise(self):
        """An all-NaN row in a 3-probe dataset must not raise in the removal
        loop (nanargmax over an empty slice): the loop runs only with >=3
        probes, and the all-NaN row must be skipped, not indexed (#23/#49)."""
        ds = xr.Dataset(
            {
                "e_1": (["time"], [np.nan, 1e-9]),
                "e_2": (["time"], [np.nan, 1e-7]),
                "e_3": (["time"], [np.nan, 1e-5]),
                "speed": (["time"], [1.0e6, 1.0e6]),  # CF95 ~ 0 -> loop active
                "nu": (["time"], [1.0e-6, 1.0e-6]),
            },
            coords={"time": [0.0, 1.0]},
            attrs={"diss_length": 512, "fs_fast": 512.0},
        )
        out = mk_epsilon_mean(ds)  # must not raise
        assert np.isnan(out["epsilonMean"].values[0])  # all-NaN row stays NaN
        assert np.isfinite(out["epsilonMean"].values[1])  # other row combined

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

    def test_two_probe_disagreement_keeps_both(self):
        """With only 2 probes there is no identifiable outlier, so both are
        kept (geometric mean). The old always-drop-max rule would have kept the
        lower probe (systematic low bias); keep-both is unbiased (audit #52)."""
        n = 20
        e1 = np.full(n, 1e-8)
        e2 = np.full(n, 1e-3)  # 5 orders apart — but which is wrong is unknowable
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
        # Geometric mean of both probes, not either probe alone.
        np.testing.assert_allclose(
            result["epsilonMean"].values, np.sqrt(1e-8 * 1e-3), rtol=0.1
        )

    def _make_three_probe_ds(self, e1, e2, e3, n=20):
        return xr.Dataset(
            {
                "e_1": (["time"], np.full(n, e1)),
                "e_2": (["time"], np.full(n, e2)),
                "e_3": (["time"], np.full(n, e3)),
                "speed": (["time"], np.full(n, 0.5)),
                "nu": (["time"], np.full(n, 1e-6)),
            },
            coords={"time": np.arange(n, dtype=float)},
            attrs={"diss_length": 512, "fs_fast": 512.0},
        )

    def test_three_probe_high_outlier_removed(self):
        """Three probes, one high outlier → furthest-from-ln-mean (the high one)
        is removed, mean tracks the consistent pair."""
        result = mk_epsilon_mean(self._make_three_probe_ds(1e-8, 1e-8, 1e-3))
        np.testing.assert_allclose(result["epsilonMean"].values, 1e-8, rtol=0.1)

    def test_three_probe_low_outlier_removed(self):
        """The audit's failing case: a LOW junk probe must now be the one
        removed (the old drop-max kept it and biased the mean low)."""
        result = mk_epsilon_mean(self._make_three_probe_ds(1e-7, 1e-7, 1e-11))
        np.testing.assert_allclose(result["epsilonMean"].values, 1e-7, rtol=0.1)

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


def _make_diss_ds_vf(eps, var_resolved=None, n=8):
    """2-D epsilon(probe, time) diss ds, optionally with var_resolved(probe, time)
    (issue #104 U4-F1 eq (18) truncation-correction tests)."""
    data = {
        "epsilon": (["probe", "time"], np.full((2, n), eps)),
        "speed": (["time"], np.full(n, 0.7)),
        "nu": (["time"], np.full(n, 1e-6)),
    }
    if var_resolved is not None:
        data["var_resolved"] = (["probe", "time"], np.full((2, n), var_resolved))
    return xr.Dataset(
        data,
        coords={"probe": [0, 1], "time": np.arange(n, dtype=float)},
        attrs={"diss_length": 1024.0, "fs_fast": 512.0},
    )


class TestEpsilonLnSigmaVfTruncation:
    """Lueck (2022) eq (18) L_hat_f = L_hat * V_f**0.75 correction (#104 U4-F1)."""

    def test_absent_var_resolved_backward_compatible(self):
        base = mk_epsilon_mean(_make_diss_ds_vf(1e-8))["epsilonLnSigma"].values
        assert np.all(np.isfinite(base))
        # No var_resolved -> unchanged behavior (the fixture without it).

    def test_vf_equal_one_is_no_op(self):
        base = mk_epsilon_mean(_make_diss_ds_vf(1e-8))["epsilonLnSigma"].values
        vf1 = mk_epsilon_mean(_make_diss_ds_vf(1e-8, var_resolved=1.0))["epsilonLnSigma"].values
        np.testing.assert_allclose(vf1, base, rtol=1e-9)

    def test_vf_below_one_increases_sigma(self):
        # Truncation reduces dof -> sigma_ln(epsilon) must INCREASE (the omission
        # understated it). Monotone: smaller V_f -> larger sigma.
        base = mk_epsilon_mean(_make_diss_ds_vf(1e-8))["epsilonLnSigma"].values
        vf95 = mk_epsilon_mean(_make_diss_ds_vf(1e-8, var_resolved=0.95))["epsilonLnSigma"].values
        vf60 = mk_epsilon_mean(_make_diss_ds_vf(1e-8, var_resolved=0.60))["epsilonLnSigma"].values
        assert np.all(vf95 > base)
        assert np.all(vf60 > vf95)

    def test_hand_value(self):
        vf = 0.95
        out = mk_epsilon_mean(_make_diss_ds_vf(1e-8, var_resolved=vf))["epsilonLnSigma"].values
        L = 0.7 * 1024.0 / 512.0
        L_K = (1e-6**3 / 1e-8) ** 0.25
        L_hat = (L / L_K) * vf**0.75
        expected = np.sqrt(5.5 / (1.0 + (L_hat / 4.0) ** (7.0 / 9.0)))
        np.testing.assert_allclose(out, expected, rtol=1e-9)

    def test_zero_and_nan_vf_do_not_crash(self):
        # V_f=0 is clipped (no L_hat collapse); a NaN slot propagates to NaN sigma.
        out0 = mk_epsilon_mean(_make_diss_ds_vf(1e-8, var_resolved=0.0))["epsilonLnSigma"].values
        assert np.all(np.isfinite(out0))
        ds = _make_diss_ds_vf(1e-8, var_resolved=0.9)
        ds["var_resolved"].values[0, 0] = np.nan
        out_nan = mk_epsilon_mean(ds)["epsilonLnSigma"].values
        assert np.all(np.isfinite(out_nan[1:]))  # other windows unaffected
