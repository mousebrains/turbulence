"""The two inter-probe consistency rules in this package, side by side.

There are two, they are NOT equivalent, and both are deliberate:

  A. ``mk_epsilon_mean`` (perturb / processing.epsilon_combine)
     Drops the probe FURTHEST FROM THE CROSS-PROBE ln-MEAN -- symmetric, so a
     low junk probe is as removable as a high one -- but only when
     ``n_probes >= 3``. With two probes it KEEPS BOTH by documented choice:
     the pair is equidistant from its own mean, so there is no identifiable
     outlier, and always dropping the maximum would systematically retain the
     lower probe and bias epsilonMean low.

  B. ``scor160.l4._compute_flags`` bit 4 (the rsi / ATOMIX path)
     Flags every probe with ``ln(e_i/e_min) > limit * mean(sigma_ln)`` --
     one-sided, so the window MINIMUM is never flagged -- with no probe-count
     gate. With two probes it therefore fires, and it keeps the lower one.

Both use the SAME threshold, ``1.96 * sqrt(2) * mean(sigma_ln)``. What differs
is the action and the probe-count gate. On a 2-probe VMP -- the common
configuration, and what ARCTERX-2022 used -- rule A is inactive and rule B is
active, so the two paths give materially different answers on identical input.

These tests pin both, and pin the disagreement, so neither can drift silently.
"""

import numpy as np
import pytest
import xarray as xr

from odas_tpw.processing.epsilon_combine import mk_epsilon_mean
from odas_tpw.scor160.l4 import DEFAULT_DISS_RATIO_LIMIT, _compute_flags

FS = 512.0
DISS_LENGTH = 2048.0
NU = 1.0e-6
SPEED = 1.0


def _sigma_ln(eps_col: np.ndarray) -> np.ndarray:
    """sigma_ln per (window, probe), by the same Lueck-2022a model both rules use."""
    L_K = (NU**3 / eps_col) ** 0.25
    L = SPEED * DISS_LENGTH / FS
    L_hat = L / L_K
    return np.sqrt(5.5 / (1.0 + (L_hat / 4.0) ** (7.0 / 9.0)))


def _threshold(eps_col: np.ndarray) -> np.ndarray:
    """The shared threshold: 1.96*sqrt(2) * mean(sigma_ln) across probes."""
    return DEFAULT_DISS_RATIO_LIMIT * np.nanmean(_sigma_ln(eps_col), axis=1)


def _ds(eps_col: np.ndarray) -> xr.Dataset:
    """(n_time, n_probe) -> the Dataset mk_epsilon_mean consumes."""
    n_time, n_probe = eps_col.shape
    ds = xr.Dataset(
        {f"e_{i + 1}": ("time", eps_col[:, i].copy()) for i in range(n_probe)},
        coords={"time": np.arange(n_time, dtype=float)},
    )
    ds["speed"] = ("time", np.full(n_time, SPEED))
    ds["nu"] = ("time", np.full(n_time, NU))
    ds.attrs["diss_length"] = DISS_LENGTH
    ds.attrs["fs_fast"] = FS
    return ds


def _rule_a(eps_col: np.ndarray) -> np.ndarray:
    """epsilonMean from mk_epsilon_mean."""
    return np.asarray(mk_epsilon_mean(_ds(eps_col))["epsilonMean"].values)


def _rule_b_flags(eps_col: np.ndarray) -> np.ndarray:
    """ATOMIX bit-4 mask, shape (n_probe, n_time). Other bits neutralised."""
    epsi = eps_col.T  # (n_probe, n_time)
    flags = _compute_flags(
        epsi,
        fom=np.zeros_like(epsi),  # bit 1 off
        var_resolved=np.ones_like(epsi),  # bit 16 off
        fom_limit=1.15,
        var_resolved_limit=0.5,
        sigma_ln=_sigma_ln(eps_col).T,
        diss_ratio_limit=DEFAULT_DISS_RATIO_LIMIT,
        method=np.zeros_like(epsi),
    )
    return (flags.astype(int) & 4) != 0


def _rule_b_final(eps_col: np.ndarray) -> np.ndarray:
    """Geometric mean over probes bit 4 did not flag."""
    bad = _rule_b_flags(eps_col)
    epsi = np.where(bad, np.nan, eps_col.T)
    with np.errstate(invalid="ignore"):
        return np.exp(np.nanmean(np.log(epsi), axis=0))


def _pair(ratio: float, base: float = 1e-8, n: int = 4) -> np.ndarray:
    """n windows of two probes differing by *ratio* (probe 0 the larger)."""
    return np.column_stack([np.full(n, base * ratio), np.full(n, base)])


class TestSharedThreshold:
    def test_both_rules_use_the_same_limit(self):
        assert pytest.approx(1.96 * np.sqrt(2.0)) == DEFAULT_DISS_RATIO_LIMIT

    def test_the_disagreement_is_action_not_threshold(self):
        """A pair inside the threshold is untouched by BOTH rules."""
        eps = _pair(1.05)
        assert np.all(np.log(eps[:, 0] / eps[:, 1]) < _threshold(eps))
        np.testing.assert_allclose(
            _rule_a(eps), np.sqrt(eps[:, 0] * eps[:, 1]), rtol=1e-12
        )
        assert not _rule_b_flags(eps).any()


class TestTwoProbes:
    """The configuration that matters: a 2-probe VMP."""

    @pytest.mark.parametrize("ratio", [5.0, 20.0, 100.0])
    def test_rule_a_keeps_both_however_bad_the_disagreement(self, ratio):
        eps = _pair(ratio)
        assert np.all(np.log(eps[:, 0] / eps[:, 1]) > _threshold(eps)), "not past threshold"
        # Documented choice: no identifiable outlier in a pair -> keep both.
        np.testing.assert_allclose(
            _rule_a(eps), np.sqrt(eps[:, 0] * eps[:, 1]), rtol=1e-12
        )

    @pytest.mark.parametrize("ratio", [5.0, 20.0, 100.0])
    def test_rule_b_fires_and_flags_only_the_larger(self, ratio):
        eps = _pair(ratio)
        bad = _rule_b_flags(eps)
        assert bad[0].all(), "the larger probe should be flagged"
        assert not bad[1].any(), "the window minimum must never be flagged"

    def test_the_two_rules_therefore_disagree_by_sqrt_of_the_ratio(self):
        """Rule A returns the geometric mean; rule B returns the minimum."""
        ratio = 16.0
        eps = _pair(ratio)
        a, b = _rule_a(eps), _rule_b_final(eps)
        np.testing.assert_allclose(b, eps[:, 1], rtol=1e-12)  # B keeps the min
        np.testing.assert_allclose(a / b, np.sqrt(ratio), rtol=1e-9)

    def test_the_disagreement_is_signed_rule_a_is_always_the_higher(self):
        # Ratios comfortably past the threshold (ln-ratio ~0.67 at these
        # epsilons). 2.0 would sit at 0.69 vs 0.67 -- too marginal to assert on.
        for ratio in (5.0, 10.0, 50.0):
            eps = _pair(ratio)
            assert np.all(np.log(eps[:, 0] / eps[:, 1]) > 1.5 * _threshold(eps))
            assert np.all(_rule_a(eps) > _rule_b_final(eps))


class TestThreeProbesWhereBothAct:
    def test_rule_a_removes_a_LOW_outlier(self):
        """Symmetric: a low junk probe is as removable as a high one."""
        good = 1e-8
        eps = np.column_stack(
            [np.full(4, good), np.full(4, good), np.full(4, good / 50.0)]
        )
        out = _rule_a(eps)
        # The low probe is dropped, so the mean is the two good probes.
        np.testing.assert_allclose(out, good, rtol=1e-9)

    def test_rule_b_KEEPS_that_low_outlier_and_flags_the_good_probes(self):
        """One-sided: the minimum is never flagged, so junk-low survives."""
        good = 1e-8
        eps = np.column_stack(
            [np.full(4, good), np.full(4, good), np.full(4, good / 50.0)]
        )
        bad = _rule_b_flags(eps)
        assert not bad[2].any(), "the minimum is never flagged"
        assert bad[0].all() and bad[1].all(), "the two good probes get flagged instead"
        # And the surviving estimate is the junk probe.
        np.testing.assert_allclose(_rule_b_final(eps), good / 50.0, rtol=1e-9)

    def test_this_is_the_documented_bias_direction(self):
        """Rule A guards against a low junk probe; rule B is exposed to it.

        This is precisely the failure mode epsilon_combine's comment cites for
        rejecting always-drop-max, and it is the strongest argument for NOT
        replacing rule A with rule B wholesale.
        """
        good = 1e-8
        eps = np.column_stack(
            [np.full(4, good), np.full(4, good), np.full(4, good / 50.0)]
        )
        assert np.all(_rule_a(eps) > _rule_b_final(eps) * 10)

    def test_rule_a_removes_a_HIGH_outlier_too(self):
        good = 1e-8
        eps = np.column_stack(
            [np.full(4, good), np.full(4, good), np.full(4, good * 50.0)]
        )
        np.testing.assert_allclose(_rule_a(eps), good, rtol=1e-9)

    def test_on_a_HIGH_outlier_the_two_rules_agree(self):
        """They only diverge on low outliers and on 2-probe windows."""
        good = 1e-8
        eps = np.column_stack(
            [np.full(4, good), np.full(4, good), np.full(4, good * 50.0)]
        )
        np.testing.assert_allclose(_rule_a(eps), _rule_b_final(eps), rtol=1e-9)


class TestWhatThisMeansForEpsilonSpectralQC:
    """Why `epsilon.spectral_qc` should implement bits 1 and 16 only."""

    def test_bit_4_is_not_redundant_on_a_two_probe_instrument(self):
        """mk_epsilon_mean cannot cover bit 4 when n_probes == 2."""
        eps = _pair(20.0)
        assert not np.any(_rule_a(eps) == eps[:, 1]), "rule A did not defer to the min"
        assert _rule_b_flags(eps).any(), "rule B did fire"

    def test_bit_4_IS_covered_for_three_or_more_probes_on_high_outliers(self):
        good = 1e-8
        eps = np.column_stack(
            [np.full(4, good), np.full(4, good), np.full(4, good * 50.0)]
        )
        np.testing.assert_allclose(_rule_a(eps), _rule_b_final(eps), rtol=1e-9)
