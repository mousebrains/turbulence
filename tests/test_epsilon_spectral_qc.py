"""``epsilon.spectral_qc`` — ATOMIX-style FM / var_resolved cut, plus the
two-probe pair policy.

Scope is deliberately narrow and the tests pin the boundaries:

  * bit 1  (FM > FM_max) and bit 16 (var_resolved < min, **method == 0 only**)
    are implemented.
  * bits 2 and 8 (despike fraction / passes) are NOT — the perturb diss
    product carries no despike diagnostics.
  * bit 4 is the ``pair_policy``, exposed rather than decided, because
    ``mk_epsilon_mean`` (keep both) and ATOMIX (keep the minimum) genuinely
    disagree for a two-probe instrument. See
    ``test_interprobe_consistency_forms.py`` for that disagreement in full.
"""

import numpy as np
import pytest
import xarray as xr

from odas_tpw.perturb.pipeline import PAIR_POLICIES, _apply_spectral_qc
from odas_tpw.scor160.l4 import DEFAULT_DISS_RATIO_LIMIT

FS = 512.0
DISS = 2048.0


def _ds(eps, FM=None, var_resolved=None, method=None):
    eps = np.asarray(eps, dtype=float)
    n_probe, n_time = eps.shape
    fm = np.full_like(eps, 0.5) if FM is None else np.asarray(FM, float)
    vr = np.full_like(eps, 0.95) if var_resolved is None else np.asarray(var_resolved, float)
    me = np.zeros_like(eps) if method is None else np.asarray(method, float)
    d = xr.Dataset(
        {
            "epsilon": (("probe", "time"), eps.copy()),
            "FM": (("probe", "time"), fm),
            "var_resolved": (("probe", "time"), vr),
            "method": (("probe", "time"), me),
            "speed": ("time", np.ones(n_time)),
            "nu": ("time", np.full(n_time, 1e-6)),
        },
        coords={"probe": np.arange(n_probe), "time": np.arange(n_time, dtype=float)},
    )
    for i in range(n_probe):
        d[f"e_{i + 1}"] = ("time", eps[i].copy())
    d.attrs["diss_length"] = DISS
    d.attrs["fs_fast"] = FS
    return d


def _run(ds, **kw):
    kw.setdefault("FM_max", 1.15)
    kw.setdefault("var_resolved_min", 0.5)
    kw.setdefault("pair_policy", "keep_both")
    kw.setdefault("pair_limit", DEFAULT_DISS_RATIO_LIMIT)
    return _apply_spectral_qc(ds, "test.p", **kw)


class TestBit1FM:
    def test_cuts_only_windows_over_the_limit(self):
        eps = np.full((2, 4), 1e-8)
        FM = np.array([[0.5, 2.0, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]])
        ds = _ds(eps, FM=FM)
        info = _run(ds)
        assert info["spectral_qc_n_cut_FM"] == 1
        assert np.isnan(ds["epsilon"].values[0, 1])
        assert np.isfinite(ds["epsilon"].values[0, 0])
        assert np.isfinite(ds["epsilon"].values[1, 1])

    def test_masks_the_e_N_companion_too(self):
        """Masking one but not the other lets a cut probe back into the mean."""
        eps = np.full((2, 3), 1e-8)
        FM = np.array([[3.0, 0.5, 0.5], [0.5, 0.5, 0.5]])
        ds = _ds(eps, FM=FM)
        _run(ds)
        assert np.isnan(ds["e_1"].values[0])
        assert np.isfinite(ds["e_2"].values[0])

    def test_thresholds_FM_not_fom(self):
        """A high `fom` with a clean FM must NOT be cut -- different statistic."""
        eps = np.full((2, 3), 1e-8)
        ds = _ds(eps)
        ds["fom"] = (("probe", "time"), np.full((2, 3), 9.9))
        info = _run(ds)
        assert info["spectral_qc_n_cut_FM"] == 0
        assert np.all(np.isfinite(ds["epsilon"].values))


class TestBit16VarResolvedMethodGate:
    def test_cuts_under_resolved_variance_method_estimates(self):
        eps = np.full((2, 3), 1e-8)
        vr = np.array([[0.1, 0.9, 0.9], [0.9, 0.9, 0.9]])
        ds = _ds(eps, var_resolved=vr, method=np.zeros((2, 3)))
        info = _run(ds)
        assert info["spectral_qc_n_cut_var_resolved"] == 1
        assert np.isnan(ds["epsilon"].values[0, 0])

    def test_does_NOT_cut_an_ISR_estimate_with_the_same_var_resolved(self):
        """The method gate. An ISR fit never integrates the dissipation range,
        so a low resolved fraction is expected, not diagnostic -- ungating this
        rejected 3.51% of ARCTERX-2022 instead of 0.18%."""
        eps = np.full((2, 3), 1e-8)
        vr = np.array([[0.1, 0.9, 0.9], [0.9, 0.9, 0.9]])
        ds = _ds(eps, var_resolved=vr, method=np.ones((2, 3)))  # method != 0
        info = _run(ds)
        assert info["spectral_qc_n_cut_var_resolved"] == 0
        assert np.all(np.isfinite(ds["epsilon"].values))

    def test_criterion_is_SKIPPED_when_method_is_absent(self):
        """Refuse to apply an ISR-inappropriate cut blind."""
        eps = np.full((2, 3), 1e-8)
        vr = np.full((2, 3), 0.1)
        ds = _ds(eps, var_resolved=vr)
        ds = ds.drop_vars("method")
        info = _run(ds)
        assert info["spectral_qc_n_cut_var_resolved"] == 0
        assert np.all(np.isfinite(ds["epsilon"].values))


class TestPairPolicy:
    """Two probes disagreeing far beyond 1.96*sqrt(2)*mean(sigma_ln)."""

    @staticmethod
    def _disagreeing():
        return _ds(np.array([[1e-7] * 4, [1e-9] * 4]))  # 100x apart

    def test_keep_both_is_the_default_and_masks_nothing(self):
        ds = self._disagreeing()
        info = _run(ds)
        assert info["spectral_qc_pair_policy"] == "keep_both"
        assert info["spectral_qc_n_pair_windows_flagged"] == 0
        assert np.all(np.isfinite(ds["epsilon"].values))

    def test_drop_high_removes_the_larger_probe_only(self):
        ds = self._disagreeing()
        info = _run(ds, pair_policy="drop_high")
        assert info["spectral_qc_n_pair_windows_flagged"] == 4
        assert np.all(np.isnan(ds["epsilon"].values[0]))
        assert np.all(np.isfinite(ds["epsilon"].values[1]))

    def test_flag_only_counts_without_masking(self):
        ds = self._disagreeing()
        info = _run(ds, pair_policy="flag_only")
        assert info["spectral_qc_n_pair_windows_flagged"] == 4
        assert np.all(np.isfinite(ds["epsilon"].values))

    def test_an_agreeing_pair_is_untouched_by_every_policy(self):
        for policy in PAIR_POLICIES:
            ds = _ds(np.array([[1.02e-8] * 4, [1e-8] * 4]))
            info = _run(ds, pair_policy=policy)
            assert info["spectral_qc_n_pair_windows_flagged"] == 0, policy
            assert np.all(np.isfinite(ds["epsilon"].values)), policy

    def test_three_FINITE_probes_are_left_to_mk_epsilon_mean(self):
        """>=3 finite probes stay with the symmetric rule; two rules must not
        contest the same window."""
        eps = np.array([[1e-7] * 4, [1e-9] * 4, [1e-8] * 4])
        ds = _ds(eps)
        info = _run(ds, pair_policy="drop_high")
        assert info["spectral_qc_n_pair_windows_flagged"] == 0
        assert np.all(np.isfinite(ds["epsilon"].values))

    def test_a_three_probe_window_with_only_TWO_finite_is_covered(self):
        """The gap that gating on n_probe would leave: mk_epsilon_mean declines
        `finite_count > 2`, so a 3-probe instrument with one probe NaN in a
        window falls into the same hole as a 2-probe instrument. Gating on the
        FINITE count makes the two rules exhaustive."""
        eps = np.array([[1e-7] * 4, [1e-9] * 4, [np.nan] * 4])
        ds = _ds(eps)
        info = _run(ds, pair_policy="drop_high")
        assert info["spectral_qc_n_pair_windows_flagged"] == 4
        assert np.all(np.isnan(ds["epsilon"].values[0]))  # the larger goes
        assert np.all(np.isfinite(ds["epsilon"].values[1]))  # the smaller stays

    def test_pair_limit_is_the_full_coefficient(self):
        """It must equal scor160's DEFAULT_DISS_RATIO_LIMIT, not 1.96, so a
        value moved between the two settings means the same thing."""
        eps = np.array([[1e-7] * 4, [1e-9] * 4])
        # Just under the threshold at the full coefficient -> no cut.
        info = _run(_ds(eps), pair_policy="drop_high", pair_limit=1e6)
        assert info["spectral_qc_n_pair_windows_flagged"] == 0

    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError, match="pair_policy"):
            _run(_ds(np.full((2, 2), 1e-8)), pair_policy="whatever")


class TestProvenance:
    def test_thresholds_and_counts_are_recorded(self):
        eps = np.full((2, 4), 1e-8)
        FM = np.array([[2.0, 2.0, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]])
        info = _run(_ds(eps, FM=FM), FM_max=1.15)
        assert info["spectral_qc_applied"] == "true"
        assert info["spectral_qc_FM_max"] == 1.15
        assert info["spectral_qc_var_resolved_min"] == 0.5
        assert info["spectral_qc_pair_policy"] == "keep_both"
        assert info["spectral_qc_n_cut_FM"] == 2
        assert info["spectral_qc_rejected_fraction"] == pytest.approx(2 / 8)

    def test_rejected_fraction_is_over_FINITE_cells(self):
        eps = np.full((2, 4), 1e-8)
        eps[1, :] = np.nan  # one probe entirely absent
        FM = np.array([[2.0, 0.5, 0.5, 0.5], [0.5] * 4])
        info = _run(_ds(eps, FM=FM))
        assert info["spectral_qc_rejected_fraction"] == pytest.approx(1 / 4)


class TestScopeBoundaries:
    def test_no_epsilon_variable_is_a_no_op(self):
        ds = xr.Dataset({"x": ("time", np.zeros(3))})
        info = _run(ds)
        assert info["spectral_qc_rejected_fraction"] == 0.0

    def test_missing_FM_skips_bit_1_rather_than_guessing(self):
        eps = np.full((2, 3), 1e-8)
        ds = _ds(eps)
        ds = ds.drop_vars("FM")
        info = _run(ds)
        assert info["spectral_qc_n_cut_FM"] == 0
        assert np.all(np.isfinite(ds["epsilon"].values))

    def test_all_probes_failing_leaves_the_window_empty_not_backfilled(self):
        """rsi semantics: no clean probe -> the window drops. NOT chi's
        never-drop fallback. A finite-but-wrong epsilon rescales Method-1 chi
        while the chi fom stays ~1, so it cannot be rejected downstream."""
        eps = np.full((2, 2), 1e-8)
        ds = _ds(eps, FM=np.full((2, 2), 5.0))
        _run(ds)
        assert np.all(np.isnan(ds["epsilon"].values))
        assert np.all(np.isnan(ds["e_1"].values))
        assert np.all(np.isnan(ds["e_2"].values))


class TestConfigKeysReachTheRightConsumer:
    """The bug this file structurally could not catch on its own.

    Every ``epsilon.*`` key is splatted into ``_compute_epsilon``; the
    post-processing ones must be stripped first. When they were not, every
    perturb run with epsilon enabled raised ``TypeError`` inside the
    per-profile ``except Exception``, so the symptom was "every profile
    errored, ``diss/`` empty, chi silently skipped" -- no traceback, and no
    unit test touched it because the pipeline tests patch ``_compute_epsilon``
    and the tests above call ``_apply_spectral_qc`` directly.
    """

    def test_no_epsilon_config_key_is_rejected_by_compute_epsilon(self):
        import inspect

        from odas_tpw.perturb.config import merge_config
        from odas_tpw.rsi.dissipation import _compute_epsilon

        cfg = merge_config("epsilon", None)
        # Mirrors the strip tuple at the _compute_epsilon call site.
        post_processing = {
            "epsilon_minimum",
            "T_source",
            "fom_max",
            "diagnostics",
            "salinity",
            "spectral_qc",
            "FM_max",
            "var_resolved_min",
            "pair_policy",
            "pair_limit",
        }
        accepted = set(inspect.signature(_compute_epsilon).parameters)
        # Duration keys are resolved away by resolve_window_config upstream.
        resolved_away = {"fft_sec", "diss_sec", "overlap_sec"}
        leftover = set(cfg) - post_processing - accepted - resolved_away
        assert not leftover, (
            f"epsilon config keys {sorted(leftover)} are neither stripped nor "
            f"accepted by _compute_epsilon -- they would raise TypeError"
        )

    def test_every_spectral_qc_key_is_in_the_strip_list(self):
        """A new knob added to DEFAULTS but not to the strip tuple is the bug."""
        import inspect

        from odas_tpw.perturb import pipeline
        from odas_tpw.perturb.config import DEFAULTS

        src = inspect.getsource(pipeline)
        for key in ("spectral_qc", "FM_max", "var_resolved_min", "pair_policy", "pair_limit"):
            assert key in DEFAULTS["epsilon"], key
            assert f'"{key}",' in src, f"{key} missing from the _compute_epsilon strip list"
