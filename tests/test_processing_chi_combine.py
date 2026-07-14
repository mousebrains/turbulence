# May-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for processing.chi_combine — mk_chi_mean.

Mirrors tests/test_processing_epsilon_combine.py against the chi
analog so the two combiners stay in lockstep.
"""

import numpy as np
import pytest
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

    def test_probe_sort_key_orders_by_trailing_int(self):
        """chi_2 must sort before chi_10; plain lexicographic order puts chi_10
        first (#24)."""
        from odas_tpw.processing.chi_combine import _probe_sort_key

        names = ["chi_10", "chi_2", "chi_1"]
        assert sorted(names, key=_probe_sort_key) == ["chi_1", "chi_2", "chi_10"]

    def test_combine_never_prunes_below_two_probes(self):
        """Mirror of the epsilon-side floor: a 3-probe row must collapse to the
        geometric mean of the two closest, never a single survivor (#23/#49).
        epsilon_T(probe,time) enables the CI-removal loop; huge speed -> CF95~0."""
        ds = xr.Dataset(
            {
                "chi_1": (["time"], [1e-12]),
                "chi_2": (["time"], [1e-7]),
                "chi_3": (["time"], [1e-6]),
                "epsilon_T": (["probe", "time"], [[1e-7], [1e-7], [1e-7]]),
                "speed": (["time"], [1.0e6]),
                "nu": (["time"], [1.0e-6]),
            },
            coords={"time": [0.0], "probe": [0, 1, 2]},
            attrs={"diss_length": 512, "fs_fast": 512.0},
        )
        out = mk_chi_mean(ds)
        assert out["chiMean"].values[0] == pytest.approx(np.sqrt(1e-13), rel=1e-6)

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

    def test_two_probe_disagreement_keeps_both(self):
        """With only 2 probes no outlier is identifiable, so both are kept
        (geometric mean) rather than dropping one. The old drop-max rule biased
        chiMean low by always keeping the lower probe (audit #52)."""
        n = 20
        c1 = np.full(n, 1e-8)
        c2 = np.full(n, 1e-3)
        eps = np.full((2, n), 1e-7)
        ds = xr.Dataset(
            {
                "chi_1": (["time"], c1),
                "chi_2": (["time"], c2),
                "epsilon_T": (["probe", "time"], eps),
                "speed": (["time"], np.full(n, 0.5)),
                "nu": (["time"], np.full(n, 1e-6)),
            },
            coords={"probe": ["t1", "t2"], "time": np.arange(n, dtype=float)},
            attrs={"diss_length": 512, "fs_fast": 512.0},
        )
        result = mk_chi_mean(ds)
        np.testing.assert_allclose(
            result["chiMean"].values, np.sqrt(1e-8 * 1e-3), rtol=0.1
        )

    def test_three_probe_low_outlier_removed(self):
        """Three probes with epsilon_T: the LOW junk probe (furthest from the
        ln-mean) is removed, not unconditionally the max, so chiMean tracks the
        consistent pair instead of being dragged down."""
        n = 20
        ds = xr.Dataset(
            {
                "chi_1": (["time"], np.full(n, 1e-7)),
                "chi_2": (["time"], np.full(n, 1e-7)),
                "chi_3": (["time"], np.full(n, 1e-11)),  # low junk
                "epsilon_T": (["probe", "time"], np.full((3, n), 1e-7)),
                "speed": (["time"], np.full(n, 0.5)),
                "nu": (["time"], np.full(n, 1e-6)),
            },
            coords={"probe": ["t1", "t2", "t3"], "time": np.arange(n, dtype=float)},
            attrs={"diss_length": 512, "fs_fast": 512.0},
        )
        result = mk_chi_mean(ds)
        np.testing.assert_allclose(result["chiMean"].values, 1e-7, rtol=0.1)

    def test_no_epsilon_keeps_all_probes(self):
        """Without epsilon_T the CI is nominal: no probe removal occurs."""
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
        with pytest.warns(UserWarning, match="nominal"):
            result = mk_chi_mean(ds)
        # Geometric mean of both probes — neither was discarded
        np.testing.assert_allclose(
            result["chiMean"].values, np.sqrt(1e-8 * 1e-3), rtol=0.01
        )
        assert "comment" in result["chiLnSigma"].attrs

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


def _make_chi_qc_ds(chi_2d, fom_2d, kmr_2d):
    """Build a (probe, time) chi dataset carrying per-probe fom / K_max_ratio,
    matching the perturb pipeline's chi product (issue #104 U3-C2 QC tests)."""
    chi_2d = np.asarray(chi_2d, dtype=float)
    n_probe, n_time = chi_2d.shape
    return xr.Dataset(
        {
            "chi": (["probe", "time"], chi_2d),
            "fom": (["probe", "time"], np.asarray(fom_2d, dtype=float)),
            "K_max_ratio": (["probe", "time"], np.asarray(kmr_2d, dtype=float)),
            "epsilon_T": (["probe", "time"], np.full((n_probe, n_time), 1e-8)),
            "speed": (["time"], np.full(n_time, 0.7)),
            "nu": (["time"], np.full(n_time, 1e-6)),
        },
        coords={"probe": np.arange(n_probe), "time": np.arange(n_time, dtype=float)},
        attrs={"diss_length": 1024.0, "fs_fast": 512.0},
    )


class TestMkChiMeanSpectralQC:
    """Soft fom + K_max_ratio QC on chiMean (issue #104 U3-C2)."""

    from odas_tpw.chi.l4_chi import _CHI_FOM_LIMIT as FOM
    from odas_tpw.chi.l4_chi import _CHI_K_MAX_RATIO_MIN as KMR

    def _qc(self, chi, fom, kmr):
        """mk_chi_mean with the shared spectral-QC thresholds enabled."""
        ds = _make_chi_qc_ds(chi, fom, kmr)
        return mk_chi_mean(ds, fom_limit=self.FOM, k_max_ratio_min=self.KMR)

    def test_off_by_default_is_backward_compatible(self):
        # No fom_limit/k_max_ratio_min -> QC disabled even when fom/Kmr present.
        chi = np.array([[1e-8, 1e-8], [2e-8, 2e-8]])  # (probe, time)
        fom = np.array([[1.0, 1.0], [3.0, 3.0]])  # probe 2 out of band
        kmr = np.array([[0.9, 0.9], [0.9, 0.9]])
        out = mk_chi_mean(_make_chi_qc_ds(chi, fom, kmr))
        np.testing.assert_allclose(out["chiMean"].values, np.sqrt(1e-8 * 2e-8), rtol=1e-6)
        assert "comment" not in out["chiMean"].attrs

    def test_drops_failing_probe_when_another_passes(self):
        chi = np.array([[1e-8, 1e-8], [2e-8, 2e-8]])
        fom = np.array([[1.0, 1.0], [3.0, 3.0]])  # probe 2 fails both windows
        kmr = np.array([[0.9, 0.9], [0.9, 0.9]])
        out = self._qc(chi, fom, kmr)
        np.testing.assert_allclose(out["chiMean"].values, 1e-8, rtol=1e-6)  # only probe 1
        assert "spectral QC applied" in out["chiMean"].attrs.get("comment", "")

    def test_kmax_ratio_floor_drops_underresolved_probe(self):
        chi = np.array([[1e-8, 1e-8], [2e-8, 2e-8]])
        fom = np.array([[1.0, 1.0], [1.0, 1.0]])  # both in fom band
        kmr = np.array([[0.9, 0.9], [0.2, 0.2]])  # probe 2 under-resolved
        out = self._qc(chi, fom, kmr)
        np.testing.assert_allclose(out["chiMean"].values, 1e-8, rtol=1e-6)

    def test_all_fail_falls_back_no_window_dropped(self):
        chi = np.array([[1e-8, 1e-8], [2e-8, 2e-8]])
        fom = np.array([[3.0, 3.0], [3.0, 3.0]])  # every probe fails
        kmr = np.array([[0.2, 0.2], [0.2, 0.2]])
        out = self._qc(chi, fom, kmr)
        # Fallback keeps both -> geometric mean, never NaN.
        assert np.all(np.isfinite(out["chiMean"].values))
        np.testing.assert_allclose(out["chiMean"].values, np.sqrt(1e-8 * 2e-8), rtol=1e-6)

    def test_both_pass_unchanged(self):
        chi = np.array([[1e-8, 1e-8], [2e-8, 2e-8]])
        fom = np.array([[1.0, 1.0], [1.0, 1.0]])
        kmr = np.array([[0.9, 0.9], [0.9, 0.9]])
        qc = self._qc(chi, fom, kmr)
        noqc = mk_chi_mean(_make_chi_qc_ds(chi, fom, kmr))
        np.testing.assert_allclose(qc["chiMean"].values, noqc["chiMean"].values, rtol=1e-6)

    def test_no_fom_present_skips_qc_and_writes_no_comment(self):
        # Limits passed but the dataset lacks fom/K_max_ratio (old-format NC):
        # the QC is silently skipped and no misleading QC-applied comment is set.
        n = 6
        c = np.full(n, 1e-8)
        ds = xr.Dataset(
            {
                "chi_1": (["time"], c),
                "chi_2": (["time"], c * 2),
                "speed": (["time"], np.full(n, 0.7)),
                "nu": (["time"], np.full(n, 1e-6)),
            },
            coords={"time": np.arange(n, dtype=float)},
            attrs={"diss_length": 1024.0, "fs_fast": 512.0},
        )
        out = mk_chi_mean(ds, fom_limit=self.FOM, k_max_ratio_min=self.KMR)
        np.testing.assert_allclose(out["chiMean"].values, np.sqrt(1e-8 * 2e-8), rtol=1e-6)
        assert "comment" not in out["chiMean"].attrs

    def test_matches_compute_chi_final_probe_selection(self):
        """The probe set entering chiMean must match _compute_chi_final's `passes`
        mask on the same inputs (single shared-threshold definition)."""
        from odas_tpw.chi.l4_chi import _compute_chi_final

        rng = np.random.default_rng(0)
        chi = 10.0 ** rng.uniform(-9, -7, size=(2, 6))
        fom = rng.uniform(0.6, 1.6, size=(2, 6))
        kmr = rng.uniform(0.2, 1.0, size=(2, 6))
        out = self._qc(chi, fom, kmr)
        # _compute_chi_final applies the identical soft QC (with fallback).
        expected = _compute_chi_final(chi, fom, kmr)
        np.testing.assert_allclose(out["chiMean"].values, expected, rtol=1e-6)


class TestChiLnSigmaVfTruncation:
    """chi_combine's Lueck eq (18) var_resolved hook (issue #104 U4-F1).

    The chi diss product now stores a Batchelor variance-resolved fraction
    (``var_resolved``; produced in ``chi.py`` / ``chi_io.py``), so this hook is
    live: ``mk_chi_mean`` applies ``L_hat_f = L_hat * V_f**0.75`` to widen
    chiLnSigma where the spectrum is truncated. These tests lock the
    consumer behavior — no-op when var_resolved is absent (old products) or
    == 1, widening when V_f < 1 — in parity with the epsilon combiner."""

    def _ds(self, var_resolved=None, n=6):
        data = {
            "chi": (["probe", "time"], np.full((2, n), 1e-8)),
            "epsilon_T": (["probe", "time"], np.full((2, n), 1e-8)),
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

    def test_absent_var_resolved_unchanged(self):
        # No var_resolved (the real chi path today) -> plain L_hat, finite sigma.
        out = mk_chi_mean(self._ds())["chiLnSigma"].values
        assert np.all(np.isfinite(out))

    def test_vf_present_applies_correction_parity_with_epsilon(self):
        base = mk_chi_mean(self._ds())["chiLnSigma"].values
        vf = mk_chi_mean(self._ds(var_resolved=0.6))["chiLnSigma"].values
        # Same direction as the epsilon combiner: V_f<1 raises sigma.
        assert np.all(vf > base)

    def test_vf_one_is_no_op(self):
        base = mk_chi_mean(self._ds())["chiLnSigma"].values
        vf1 = mk_chi_mean(self._ds(var_resolved=1.0))["chiLnSigma"].values
        np.testing.assert_allclose(vf1, base, rtol=1e-9)

    def test_nan_var_resolved_does_not_poison_sigma(self):
        # A U3-C3-dropped window carries NaN var_resolved (chi is NaN too). The
        # NaN in one probe must not blank the combined chiLnSigma where the other
        # probe is valid.
        ds = self._ds(var_resolved=0.7, n=6)
        ds["var_resolved"].values[0, 2] = np.nan  # one probe/time dropped
        out = mk_chi_mean(ds)["chiLnSigma"].values
        assert np.all(np.isfinite(out))
