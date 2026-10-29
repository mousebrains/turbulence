# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""``fp07-cal`` QC follow-ups to #238, #241 and #243.

#246  Both robust loops kept a pair when ``|resid| <= 4 sigma`` with sigma
      measured about the median: a minority population offset by many sigma
      dragged the least-squares zero off the majority, nothing was rejected and
      the channel was refused at the 1 K gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from odas_tpw.fp07cal.cli import MAX_REJECTED_FRACTION, run_calibration
from odas_tpw.fp07cal.fit import _subset, fit_calibration
from odas_tpw.fp07cal.geometry import joint_fit, local_dTdz
from odas_tpw.fp07cal.pairs import PairConfig, build_pairs_multi
from odas_tpw.fp07cal.synth import SynthConfig, make_deployment


def _dep(**kw):
    """A small one-channel deployment: CT on every yo, no clock offset or delay."""
    cfg = dict(n_yos=12, yo_seconds=1200.0, fs=4.0, ct_every_n=1, files_per_deployment=4)
    cfg.update(kw)
    return make_deployment(SynthConfig(**cfg), channels=("T1",))


def _cfg(*, order=1, geometry=False):
    return {
        "pairs": {"max_gap": 30.0, "min_corr": 0.7},
        "lag": {"max_lag": 6.0, "step": 0.5},
        "fit": {"order": order, "geometry": geometry, "robust": True},
        "stability": {"n_blocks": 4},
        "channels": ["T1"],
    }


# --------------------------------------------- #246: the robust loops' center
def _bimodal_pairs(share=0.3, dT=2.0):
    """Pairs at the true lag, the last *share* of them with the reference off by *dT*."""
    probes, ref, truth = _dep()
    pairs = build_pairs_multi(probes, ref, "T1", lag=0.0, cfg=PairConfig(max_gap=30.0))
    minority = pairs.time > np.quantile(pairs.time, 1.0 - share)
    pairs.T_ref = np.where(minority, pairs.T_ref + dT, pairs.T_ref)
    return pairs, minority, truth


def test_the_calibration_loop_rejects_a_minority_about_the_median():
    """30% of the reference 2 K off: least squares puts zero 0.6 K from the majority."""
    pairs, minority, truth = _bimodal_pairs()
    fit = fit_calibration(pairs, order=1)
    assert fit.outlier[minority].all()
    assert fit.outlier[~minority].sum() <= 2  # the noise's own 4-sigma tail
    majority = fit_calibration(_subset(pairs, ~minority), order=1)
    assert fit.config_equivalent["t_0"] == pytest.approx(majority.config_equivalent["t_0"],
                                                         abs=1e-6)
    assert fit.config_equivalent["t_0"] == pytest.approx(truth["t_0"], abs=1e-3)


def test_the_joint_loop_rejects_a_minority_about_the_median():
    """The geometry fit's loop, on pairs that all have a reference gradient."""
    pairs, minority, truth = _bimodal_pairs()
    g = local_dTdz(pairs)
    has = np.isfinite(g)
    pairs, minority, g = _subset(pairs, has), minority[has], g[has]
    fit, _geo = joint_fit(pairs, order=1, dTdz=g)
    assert fit.outlier[minority].all()
    assert fit.outlier[~minority].sum() <= 2
    assert fit.config_equivalent["t_0"] == pytest.approx(truth["t_0"], abs=1e-3)


def test_a_bimodal_reference_flags_the_minority_and_exports(tmp_path):
    """Issue #246's acceptance case: main refused it (1.80 K on the pairs it kept)."""
    probes, ref, truth = _dep()
    off = ref.time > np.quantile(ref.time, 0.85)
    ref.value[off] += 2.0
    c = run_calibration(probes, ref, _cfg(), tmp_path, make_figure=False)["channels"]["T1"]
    assert "config_equivalent" in c, c
    assert c["config_equivalent"]["t_0"] == pytest.approx(truth["t_0"], abs=1e-3)
    assert c["config_equivalent"]["beta_1"] == pytest.approx(truth["beta_1"], rel=1e-3)
    beyond = c["outliers_beyond_gate"]
    assert beyond["n"] > 0.14 * c["n_pairs"]  # the 15% minority, flagged
    assert beyond["max_error_K"] == pytest.approx(2.0, abs=0.05)


def test_a_clean_reference_loses_almost_nothing_to_the_loop(tmp_path):
    pairs, _minority, _truth = _bimodal_pairs(dT=0.0)
    assert fit_calibration(pairs, order=1).outlier.mean() < 1e-3
    probes, ref, _ = _dep()
    c = run_calibration(probes, ref, _cfg(), tmp_path, make_figure=False)["channels"]["T1"]
    assert "robust_rejection" not in c
    assert "All diagnostic gates passed." in (tmp_path / "T1_report.md").read_text(
        encoding="utf-8")


@pytest.mark.parametrize("q", [0.6, 0.95])
def test_a_population_below_the_gate_is_warned_of(q, tmp_path):
    """A reference 50 mK off for the last 40% (5%) of the deployment.

    Main averaged 40% in (rms 24 mK, the step visible in the block table); the
    loop now leaves it out of the fit and of the blocks, below the 1 K gate's
    notice, so the record and the report have to say so.
    """
    probes, ref, truth = _dep()
    ref.value[ref.time > np.quantile(ref.time, q)] += 0.05
    c = run_calibration(probes, ref, _cfg(), tmp_path, make_figure=False)["channels"]["T1"]
    assert c["config_equivalent"]["t_0"] == pytest.approx(truth["t_0"], abs=1e-3)
    assert "outliers_beyond_gate" not in c
    rr = c["robust_rejection"]
    assert rr["fraction"] == pytest.approx(1.0 - q, abs=0.01)
    assert rr["share_in_those_profiles"] > 0.8
    assert rr["first"] >= np.quantile(ref.time, q) - 1200.0
    report = (tmp_path / "T1_report.md").read_text(encoding="utf-8")
    assert "a second population, not a tail" in report


def test_the_population_rule_needs_a_fraction_or_whole_profiles():
    from odas_tpw.fp07cal.cli import _population_warning

    base = {"n_rejected": 50, "n_judged": 1000, "n_profiles": 20, "first": 1.7e9,
            "last": 1.7e9 + 600.0, "files": {"a.p": 50}}
    tail = {**base, "fraction": 0.069, "n_profiles_mostly_rejected": 7,
            "share_in_those_profiles": 0.127}  # the worst of 23 real channels, each way
    assert _population_warning(tail) is None
    assert _population_warning({**tail, "fraction": MAX_REJECTED_FRACTION + 0.01})
    assert _population_warning({**tail, "share_in_those_profiles": 0.5})
    assert _population_warning({**tail, "n_profiles_mostly_rejected": 0,
                                "share_in_those_profiles": 0.0, "first": None}) is None
