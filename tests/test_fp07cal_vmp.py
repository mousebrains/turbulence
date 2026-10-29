# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""``fp07-cal`` on a VMP against its own JAC CT (issue #238).

The tool was designed on gliders against a separate CTD.  Fitting VMP beads
against the instrument's own JAC CT exposed six defects; each has a test here:

1. the 1 K export gate counted pairs the robust fit had already rejected, and
   a fit that is not an FP07 curve at all could pass it;
2. the lag-trust gate refused every narrow (VMP) search window;
3. the geometry term crashed on profiles shorter than ~30 dbar;
4. one NaN count turned the rest of the file to NaN through the single pole;
5. the reference-free T1 - T2 check used each file's embedded coefficients;
6. the +/-40 s clock search ran even when the reference is the instrument's
   own record.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.signal import lfilter

import odas_tpw.fp07cal.cli as cli
from odas_tpw.fp07cal.cli import run_calibration
from odas_tpw.fp07cal.lag import LagResult, pressure_offset, shares_clock, temperature_lag
from odas_tpw.fp07cal.logr import log_r, temperature
from odas_tpw.fp07cal.pairs import (
    SETTLE_TAUS,
    PairConfig,
    _single_pole,
    build_pairs,
    build_pairs_multi,
)
from odas_tpw.fp07cal.patch import validate_record
from odas_tpw.fp07cal.series import sanitize_reference
from odas_tpw.fp07cal.stability import t1_t2_series
from odas_tpw.fp07cal.synth import SynthConfig, make_deployment


def _dep(**kw):
    """A small two-channel deployment: CT on every yo, 4 files."""
    cfg = dict(n_yos=12, yo_seconds=1200.0, fs=4.0, ct_every_n=1, files_per_deployment=4)
    cfg.update(kw)
    return make_deployment(SynthConfig(**cfg))


def _cfg(*, geometry=False, order=1, channels=("T1",), lag=None, reference=None):
    return {
        "pairs": {"max_gap": 30.0, "min_corr": 0.7},
        "lag": lag or {"max_lag": 6.0, "step": 0.5},
        "fit": {"order": order, "geometry": geometry, "robust": True},
        "stability": {"n_blocks": 4},
        "channels": list(channels),
        **({"reference": reference} if reference is not None else {}),
    }


def _own_record(probes, every=2):
    """A reference made of the instrument's own samples: a VMP's JAC CT."""
    t = np.concatenate([p.time[::every] for p in probes])
    T = np.concatenate([
        temperature(log_r(p.counts["T1"], p.bridge["T1"])[0], p.factory["T1"])[::every]
        for p in probes
    ])
    P = np.concatenate([p.pressure[::every] for p in probes])
    return sanitize_reference(t, T, pressure=P, source="own:JAC_T")


# ---------------------------------------------------------- 1. export gates
def test_the_gate_ignores_pairs_the_robust_fit_rejected(tmp_path):
    """One 4.2 K reference event refused SUNRISE 2022 PointSur (fit rms 6-7 mK)."""
    probes, ref, _ = _dep()
    hit = np.flatnonzero((ref.pressure > 40) & (ref.pressure < 42))[:3]
    ref.value[hit] += 4.2
    res = run_calibration(probes, ref, _cfg(), tmp_path, make_figure=False)
    c = res["channels"]["T1"]
    assert "config_equivalent" in c, c  # exported
    assert c["rms_K"] < 0.01
    out = c["outliers_beyond_gate"]
    assert out["n"] >= 1 and out["max_error_K"] == pytest.approx(4.2, abs=0.1)
    assert out["worst_pressure_dbar"] == pytest.approx(41.0, abs=2.0)
    report = (tmp_path / "T1_report.md").read_text(encoding="utf-8")
    assert "left out of the fit and of the export gate" in report
    # The old gate, over every pair, would have refused this fit.
    _lr, pairs = temperature_lag(probes, ref, "T1", cfg=PairConfig(max_gap=30.0),
                                 max_lag=6.0, step=0.5)
    pred = temperature(pairs.L, np.asarray(c["coefficients"]))
    assert np.nanmax(np.abs(pred - pairs.T_ref)) > cli.MAX_PREDICTION_ERROR_K


def test_rejected_outliers_set_neither_the_range_nor_the_folds(tmp_path):
    """Water 13-21 degC plus three reference samples at 2.0 degC.  The exported fit
    claimed T_range [2.0, 20.98], and the 2 degC set the middle of the warm/cold
    fold split, which left no fold to score the orders on."""
    from odas_tpw.fp07cal.fit import fit_calibration, outlier_screen, temperature_folds

    probes, ref, _ = _dep()
    hit = np.flatnonzero((ref.pressure > 40) & (ref.pressure < 42))[:3]
    ref.value[hit] = 2.0
    c = run_calibration(probes, ref, _cfg(order="auto"), tmp_path / "a",
                        make_figure=False)["channels"]["T1"]
    assert "config_equivalent" in c and c["outliers_beyond_gate"]["n"] == 3
    assert c["T_range"][0] > 12.0  # the water, not the glitch
    clean_probes, clean_ref, _ = _dep()
    clean = run_calibration(clean_probes, clean_ref, _cfg(order="auto"), tmp_path / "b",
                            make_figure=False)["channels"]["T1"]
    assert c["order"] == clean["order"] and set(c["order_selection"]) == {"1", "2", "3"}
    for o, v in clean["order_selection"].items():  # the glitch also moves the lag a little
        assert c["order_selection"][o]["held_out_K"] == pytest.approx(v["held_out_K"], rel=0.2)
    # The pieces: the fit's ranges are its kept pairs', and the screen takes the
    # glitch out of both the split and the folds.
    _lr, pairs = temperature_lag(probes, ref, "T1", cfg=PairConfig(max_gap=30.0),
                                 max_lag=6.0, step=0.5)
    fit = fit_calibration(pairs, order=1)
    kept = ~fit.outlier
    assert fit.T_range == (pairs.T_ref[kept].min(), pairs.T_ref[kept].max())
    assert fit.L_range == (pairs.L[kept].min(), pairs.L[kept].max())
    screen = outlier_screen(pairs)
    glitch = pairs.T_ref < 3.0
    assert glitch.sum() == 3 and screen[glitch].all()
    for train, test in temperature_folds(pairs, screen):
        assert not (train | test)[glitch].any()
    # Without an exclusion the folds are what they always were.
    a, b = temperature_folds(pairs), temperature_folds(pairs, None)
    assert all(np.array_equal(x, y) for fa, fb in zip(a, b) for x, y in zip(fa, fb))


def test_a_clean_record_carries_no_new_keys(tmp_path):
    """None of the #238 additions appear where they do not apply."""
    probes, ref, _ = _dep(clock_offset=3.0, ctd_delay=1.0)
    c = run_calibration(probes, ref, _cfg(), tmp_path, make_figure=False)
    assert "clock_source" not in c
    extra = {"outliers_beyond_gate", "thermistor_gaps", "geometry_skipped", "lag_resolution"}
    assert not extra & set(c["channels"]["T1"])
    report = (tmp_path / "T1_report.md").read_text(encoding="utf-8")
    assert "## Notes" not in report and "All diagnostic gates passed." in report


def test_the_gate_still_refuses_a_fit_that_mispredicts_what_it_kept(tmp_path, monkeypatch):
    """F20: an unidentified fit is wrong on the pairs it kept, not only on outliers."""
    real = cli.fit_calibration

    def off_by_5K(pairs, **kw):
        f = real(pairs, **kw)
        f.coeffs = f.coeffs.copy()
        f.coeffs[0] -= 5.0 / 290.0**2  # ~5 K warmer everywhere
        return f

    monkeypatch.setattr(cli, "fit_calibration", off_by_5K)
    probes, ref, _ = _dep()
    c = run_calibration(probes, ref, _cfg(), tmp_path, make_figure=False)["channels"]["T1"]
    assert c["error"] == "unidentifiable fit" and c["max_prediction_error_K"] > 4.0


def test_a_fit_that_is_not_ntc_is_refused(tmp_path):
    """Isothermal water, no prior: order 1 fits noise with beta_1 = -1.06e5, T RISING with L.

    It predicts its own pairs to a few mK, so the 1 K gate cannot see it.
    """
    probes, ref = _iso()
    res = run_calibration(probes, ref, _cfg(), tmp_path, make_figure=False)
    c = res["channels"]["T1"]
    assert c["error"] == "not an FP07 curve"
    assert "does not fall as L rises" in c["reason"]
    assert 1.0 / c["refused_coefficients"][1] < 0  # beta_1 negative
    assert "config_equivalent" not in c and "coefficients" not in c
    validate_record(res)  # patch skips the channel instead of refusing the record
    assert not (tmp_path / "T1_report.md").exists()


def _iso(seed=7):
    from odas_tpw.fp07cal.synth import DEFAULT_COEFFS

    cfg = SynthConfig(n_yos=16, yo_seconds=1200.0, fs=4.0, ct_every_n=1, T_surface=28.0,
                      T_deep=28.0, files_per_deployment=4, seed=seed)
    return make_deployment(cfg, coeffs=DEFAULT_COEFFS, channels=("T1",))[:2]


# ------------------------------------------------------------- 2. lag trust
FS = 16.0  # a lag grid fine enough to resolve a +/-1 s peak (a VMP searches at 1/64 s)


@pytest.fixture(scope="module")
def fine():
    return make_deployment(SynthConfig(n_yos=12, yo_seconds=1200.0, fs=FS, ct_every_n=1,
                                       ref_rate=FS, files_per_deployment=4))


def _narrow(dep, detrend_s=10.0):
    probes, ref, _ = dep
    return temperature_lag(probes, ref, "T1", cfg=PairConfig(max_gap=30.0, kernel_width=3 / FS),
                           max_lag=1.0, step=1 / FS, detrend_s=detrend_s)


def test_a_narrow_window_peak_is_trusted_on_curvature_and_halves(fine):
    """A +/-1 s search cannot show a 0.10 fall however well the lag is identified."""
    lr, _pairs = _narrow(fine)
    assert lr.dynamic_range < 0.10 and lr.narrow_window() and lr.max_lag == 1.0
    assert lr.trustworthy(), lr.summary()
    assert lr.fall_at_reference() >= 0.10 and lr.peak_width >= 2 / FS
    a, b = lr.half_lags
    assert abs(a - b) <= 2 / FS and lr.half_interior
    assert "narrow window" in lr.summary()


def test_a_narrow_window_still_refuses_a_mean_removed_ramp(fine):
    """The flat plateau the glider gate exists for, now in a +/-1 s window."""
    lr, _pairs = _narrow(fine, detrend_s=1e9)
    assert lr.dynamic_range < 1e-3
    assert not lr.trustworthy(), lr.summary()
    assert lr.fall_at_reference() < 0.10  # refused on curvature, whatever the halves say
    assert "flat" in lr.summary()


def _parabola(R, half, step, peak=0.1):
    lags = np.arange(-half, half + step / 2, step)
    c = R / (half + abs(peak)) ** 2
    return lags, 0.9999 - c * (lags - peak) ** 2


def test_the_narrow_window_rule_needs_curvature_and_halves():
    from odas_tpw.fp07cal.lag import _summarize

    # A VMP-like peak: falls 0.005 across +/-1 s on a 1/64 s grid.
    lags, s = _parabola(0.005, 1.0, 1 / 64)
    good = _summarize(lags, s, "vmp", 0)
    good.half_lags, good.half_interior = (0.098, 0.103), True
    assert good.trustworthy() and good.fall_at_reference() > 1.0
    split = _summarize(lags, s, "vmp", 0)
    split.half_lags, split.half_interior = (0.0, 0.4), True
    assert not split.trustworthy() and "halves do not find the peak" in split.summary()
    edge = _summarize(lags, s, "vmp", 0)
    edge.half_lags, edge.half_interior = (0.098, 0.103), False
    assert not edge.trustworthy()
    # A glider's mean-removed ramp: 5e-4 across +/-20 s, halves agreeing.
    lags, s = _parabola(5e-4, 20.0, 0.25, peak=7.4)
    flat = _summarize(lags, s, "ramp", 0)
    flat.half_lags, flat.half_interior = (7.36, 7.54), True
    assert not flat.trustworthy() and flat.fall_at_reference() < 1e-3
    # No halves at all (a hand-built result): the classic rule alone.
    assert not LagResult(lag=0.1, score=0.99, dynamic_range=0.005, width=0.14).trustworthy()
    assert LagResult(lag=4.0, score=0.97, dynamic_range=0.97, width=0.8).trustworthy()


@pytest.mark.parametrize("step,configured", [
    *[(s, c) for s in (0.25, 0.2, 0.1, 0.05) for c in (True, False)],
    (0.3, True), (0.15, True),  # grids that stop short of 20 s: the configured max_lag rules
])
def test_a_glider_width_search_is_judged_on_its_fall_alone(step, configured):
    """A +/-20 s search MEASURES the fall: a sharp, replicating peak that falls by
    0.05 across it stays refused, as before issue #238 --- whatever the step.
    np.arange puts the last lag of a +/-20 s search at 19.99999999999993 s for a
    0.2 s step, which a strict `< 20` read as a narrow window."""
    from odas_tpw.fp07cal.lag import _summarize

    lags = np.arange(-20.0, 20.0 + step / 2, step)  # exactly as temperature_lag builds it
    s = 0.9 - 0.05 * np.minimum(np.abs(lags - 7.0) / 2.0, 1.0)
    weak = _summarize(lags, s, "glider", 0)
    if configured:
        weak.max_lag = 20.0
    weak.half_lags, weak.half_interior = (7.0, 7.0), True
    assert not weak.narrow_window() and not weak.trustworthy(), weak.summary()


def _grid(step=1 / 64):
    return np.arange(-1.0, 1.0 + step / 2, step)  # the runbook's VMP search


def test_two_lobes_are_an_ambiguity_not_a_peak():
    """Two near-equal lobes at -0.40 and +0.50 s, each half picking one: the
    width of every lag within 1% spans both, so a width-based tolerance let it
    through."""
    from odas_tpw.fp07cal.lag import _summarize

    lags = _grid()
    s = 0.99 - 0.010 * np.minimum(((lags + 0.40) / 0.15) ** 2,
                                  ((lags - 0.50) / 0.15) ** 2).clip(0, 1)
    s[np.argmin(abs(lags + 0.40))] += 5e-5
    lr = _summarize(lags, s, "two lobes", 0)
    lr.half_lags, lr.half_interior, lr.max_lag = (-0.40, 0.50), True, 1.0
    assert lr.width > 0.8 and lr.peak_width < 0.1
    assert not lr.halves_agree() and not lr.trustworthy()
    # Broad lobes the grid resolves, the second 2e-5 lower: within 1% of the
    # peak, so not the only top, even with both halves on the first lobe.
    s = 0.99 - 0.010 * np.minimum(((lags + 0.40) / 0.6) ** 2,
                                  ((lags - 0.50) / 0.6) ** 2 + 2e-3).clip(0, 1)
    lr = _summarize(lags, s, "two lobes", 0)
    lr.half_lags, lr.half_interior, lr.max_lag = (-0.40, -0.40), True, 1.0
    assert lr.peak_width >= 2 / 64 and lr.halves_agree()
    assert "more than one top" in lr.narrow_problem() and not lr.trustworthy()


def test_a_flat_top_is_refused():
    """A 0.8 s flat top falling 0.02 at the edges: curvature alone passes it."""
    from odas_tpw.fp07cal.lag import _summarize

    lags = _grid()
    s = 0.99 - 0.02 * np.clip((np.abs(lags - 0.1) - 0.4) / 0.5, 0, 1) ** 2
    s += 1e-6 * np.cos(7 * lags)
    lr = _summarize(lags, s, "plateau", 0)
    lr.max_lag = 1.0
    lr.half_lags, lr.half_interior = (0.10, 0.11), True  # even with agreeing halves
    assert lr.fall_at_reference() > 0.10
    assert "flat top" in lr.narrow_problem() and not lr.trustworthy()
    lr.half_lags = (-0.25, 0.45)
    assert not lr.halves_agree()


def test_a_one_step_peak_is_not_a_measured_curvature():
    """On the VMP's 1/64 s grid a one-point bump of 3.6e-5 passes the curvature
    bound (fall 0.59) even widened by a step at each edge, and replicates on both
    halves: the grid must resolve the peak (two steps or more)."""
    from odas_tpw.fp07cal.lag import _summarize

    lags = _grid()
    s = 1.0 - 3.6e-5 * (np.abs(lags) > 1e-12)
    lr = _summarize(lags, s, "one step", 0)
    lr.half_lags, lr.half_interior, lr.max_lag = (0.0, 0.0), True, 1.0
    assert lr.peak_width == 0.0 and lr.fall_at_reference() > 0.10 and lr.halves_agree()
    assert "does not resolve" in lr.narrow_problem() and not lr.trustworthy()


def test_the_estimators_populate_the_halves():
    probes, ref, _ = _dep(clock_offset=3.0, ctd_delay=1.0)
    lr, _ = temperature_lag(probes, ref, "T1", cfg=PairConfig(max_gap=30.0),
                            max_lag=8.0, step=0.5)
    po = pressure_offset(probes, ref, max_lag=8.0, step=0.5)
    for r in (lr, po):
        assert r.half_interior and all(np.isfinite(r.half_lags)), r
        assert abs(r.half_lags[0] - r.half_lags[1]) < 1.0


def test_the_record_carries_the_narrow_window_evidence(fine, tmp_path):
    probes, ref, _ = fine
    cfg = _cfg(lag={"max_lag": 1.0, "step": 1 / FS, "detrend_s": 10.0})
    cfg["pairs"]["kernel_width"] = 3 / FS
    c = run_calibration(probes, ref, cfg, tmp_path, make_figure=False)["channels"]["T1"]
    assert c["lag_trustworthy"] is True and c["lag_dynamic_range"] < 0.10
    ev = c["lag_resolution"]
    assert ev["halves_agree"] and ev["fall_at_20s"] >= 0.10 and ev["search_s"] == [-1.0, 1.0]
    assert ev["problem"] == "" and ev["peak_width_s"] >= 2 / FS
    from odas_tpw.fp07cal.patch import build_edits

    rec = {"schema": "fp07-cal/1", "instrument_sn": "SYNTH-01", "channels": {"T1": c}}
    config = {"instrument_info": {"sn": "SYNTH-01"},
              "channels": [{"name": "T1", **{k: str(v) for k, v in c["bridge"].items()},
                            "t_0": "288", "beta_1": "3100"}]}
    plan = build_edits(rec, config)
    assert not plan.errors, plan.errors  # patch would accept it


# -------------------------------------------------------------- 3. geometry
def test_short_profiles_skip_the_geometry_term_with_a_note(tmp_path):
    """SUNRISE 2021 descents span ~22 dbar; local_dTdz needs 30. It used to raise."""
    probes, ref, _ = _dep(P_max=20.0, thermocline_dbar=8.0, T_surface=29.0, T_deep=27.0)
    res = run_calibration(probes, ref, _cfg(geometry=True, order="auto"), tmp_path,
                          make_figure=False)
    c = res["channels"]["T1"]
    assert "config_equivalent" in c and c["geometry"] is None
    assert "30 dbar" in c["geometry_skipped"] and "20.0 dbar" in c["geometry_skipped"]
    assert c["order_selection"]  # the order was still chosen by held-out error
    report = (tmp_path / "T1_report.md").read_text(encoding="utf-8")
    assert "## Notes" in report and "geometry term was requested" in report


# ------------------------------------------------------------------ 4. NaN
def _old_single_pole(x, dt, tau):
    alpha = dt / (tau + dt)
    y, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], x, zi=np.array([(1.0 - alpha) * x[0]]))
    return y


def test_single_pole_is_unchanged_on_finite_input():
    x = np.random.default_rng(1).normal(size=5000).cumsum()
    st: dict = {}
    np.testing.assert_array_equal(_single_pole(x, 1 / 64, 0.17, stats=st),
                                  _old_single_pole(x, 1 / 64, 0.17))
    assert st == {"n_gaps": 0, "n_nonfinite": 0, "n_settling": 0}


def test_single_pole_restarts_after_a_gap():
    dt, tau = 1 / 64, 0.17
    x = np.random.default_rng(2).normal(size=4000).cumsum() * 1e-3
    x[1000:1010] = np.nan
    st: dict = {}
    y = _single_pole(x, dt, tau, stats=st)
    n_settle = int(np.ceil(SETTLE_TAUS * tau / dt))
    np.testing.assert_array_equal(y[:1000], _old_single_pole(x[:1000], dt, tau))
    assert np.all(np.isnan(y[1000 : 1010 + n_settle]))
    np.testing.assert_array_equal(y[1010 + n_settle :],
                                  _old_single_pole(x[1010:], dt, tau)[n_settle:])
    assert st == {"n_gaps": 1, "n_nonfinite": 10, "n_settling": n_settle}
    # The old filter carried the NaN through every later sample.
    assert np.all(np.isnan(_old_single_pole(x, dt, tau)[1000:]))


def test_a_leading_nan_does_not_poison_the_record():
    x = np.linspace(-0.1, 0.1, 500)
    x[:3] = np.nan
    y = _single_pole(x, 0.25, 0.7)
    assert np.all(np.isnan(y[:3])) and np.all(np.isfinite(y[3 + 14 :]))


def test_a_nan_count_costs_only_the_pairs_near_it(tmp_path):
    probes, ref, _ = _dep()
    pc = PairConfig(max_gap=30.0)
    before = build_pairs(probes[1], ref, "T1", lag=1.0, cfg=pc)
    t_gap = probes[1].time[600]
    probes[1].counts["T1"][600:604] = np.nan
    after = build_pairs(probes[1], ref, "T1", lag=1.0, cfg=pc)
    lost = len(before) - len(after)
    # The gap (1 s), the settling (5 x 0.7 s) and the kernel: a few pairs, not the file.
    assert 0 < lost <= 8 and after.rejected["thermistor_gap"] == lost
    assert after.gaps == {probes[1].label: {"n_gaps": 1, "n_nonfinite": 4, "n_settling": 14}}
    assert np.sum(before.time > t_gap) > 100  # what the old filter would have lost
    res = run_calibration(probes, ref, _cfg(), tmp_path, make_figure=False)
    g = res["channels"]["T1"]["thermistor_gaps"]
    assert g["n_files"] == 1 and g["n_nonfinite_samples"] == 4 and g["n_pairs_lost"] > 0
    assert "thermistor sample(s)" in (tmp_path / "T1_report.md").read_text(encoding="utf-8")


def test_no_kept_kernel_touches_a_gap():
    """A kernel the gap or the settling blank only partly empties is dropped too:
    its mean would be over a shorter window than every other pair's."""
    from odas_tpw.fp07cal.pairs import _boxcar_at, prepare_probe

    probes, ref, _ = _dep()
    pc = PairConfig(max_gap=30.0, kernel_width=3.0)  # wide kernels: partial overlaps
    p = probes[1]
    p.counts["T1"][600:640] = np.nan
    after = build_pairs(p, ref, "T1", lag=1.0, cfg=pc)
    pp = prepare_probe(p, "T1", pc)
    holed, _ = _boxcar_at(p.time, np.isnan(pp.L_slow).astype(float), after.time - 1.0, 3.0)
    assert len(after) > 0 and not np.any(holed > 0)
    assert after.rejected["thermistor_gap"] > 0


def test_pairs_without_gaps_are_bit_identical():
    """No gap: the parent's algorithm (2e1b660) exactly, the parent's single pole
    and boxcar reproduce every L bit for bit, and the counts are its golden values."""
    from odas_tpw.fp07cal.pairs import _boxcar_at

    probes, ref, _ = _dep()
    pc = PairConfig(max_gap=30.0)
    a = build_pairs_multi(probes, ref, "T1", lag=1.0, cfg=pc)
    # Golden values from the parent commit, same synthetic deployment.
    assert len(a) == 14388
    assert dict(a.rejected) == {"kernel_all_nan": 0, "kernel_too_few_samples": 0,
                                "clipped_counts": 0, "outside_profile": 0,
                                "below_min_speed": 0, "reference_not_finite": 0}
    assert list(a.rejected) == ["kernel_all_nan", "kernel_too_few_samples", "clipped_counts",
                                "outside_profile", "below_min_speed", "reference_not_finite"]
    assert a.per_file == {f"synth_{i:04d}.p": 3597 for i in range(4)} and a.gaps == {}
    # Every L is the parent's: its single pole, then the boxcar at the pair's centre.
    width = ref.median_interval()
    for p in probes:
        m = a.file_label == p.label
        L_raw, _ = log_r(p.counts["T1"], p.bridge["T1"])
        old = _old_single_pole(L_raw, 1.0 / p.fs, pc.kernel_tau)
        L_k, _ = _boxcar_at(p.time, old, a.time[m] - 1.0, width)
        np.testing.assert_array_equal(a.L[m], L_k)


# ----------------------------------------------------------------- 5. T1-T2
def _patched_per_file(probes, dT=0.5):
    """Each file's embedded t_0 moved by +/-dT, as per-file patches leave them."""
    for i, p in enumerate(probes):
        for ch in ("T1", "T2"):
            c = p.factory[ch].copy()
            T0 = 1.0 / c[0]
            c[0] = 1.0 / (T0 + (dT if (i + (ch == "T2")) % 2 else -dT))
            p.factory[ch] = c


def test_t1_t2_refuses_per_file_coefficients_and_uses_the_fitted_ones():
    probes, _ref, truth = _dep()
    _patched_per_file(probes)
    stale = t1_t2_series(probes)
    assert not stale["available"] and "differ between files" in stale["reason"]
    fitted = t1_t2_series(probes, coeffs={"T1": truth["coeffs"], "T2": truth["coeffs"]})
    assert fitted["available"] and fitted["coefficients"] == "fitted"
    assert np.std(fitted["value"]) < 1e-3  # one set: no step between files
    assert abs(fitted["slope_K_per_day"]) < 1e-3


def test_t1_t2_never_mixes_a_fit_with_embedded_coefficients():
    """One channel fitted, the other not: both embedded when both are one set
    across the files, else unavailable --- never this run's T1 against an
    embedded T2, which compares two calibrations."""
    probes, _ref, truth = _dep()
    only_t1 = {"T1": truth["coeffs"] + np.array([1e-6, 0.0])}  # a fit unlike the embedded
    both_embedded = t1_t2_series(probes, coeffs=only_t1)
    assert both_embedded["available"] and both_embedded["coefficients"] == "embedded"
    np.testing.assert_array_equal(both_embedded["value"], t1_t2_series(probes)["value"])
    # T1 patched per file (its embedded sets vary), T2 one set: the embedded
    # pair is not one set per channel, and the fitted T1 cannot stand in.
    for i, p in enumerate(probes):
        p.factory["T1"] = p.factory["T1"] * np.array([1.0 + 1e-5 * (i % 2), 1.0])
    t1_varies = t1_t2_series(probes, coeffs=only_t1)
    assert not t1_varies["available"] and t1_varies["coefficients"] is None
    assert "embedded coefficients of T1 differ" in t1_varies["reason"]
    _patched_per_file(probes)
    mixed = t1_t2_series(probes, coeffs=only_t1)
    assert not mixed["available"] and mixed["coefficients"] is None
    assert "T2 was not fitted" in mixed["reason"]


def test_the_report_t1_t2_check_uses_this_runs_fit(tmp_path):
    probes, ref, _ = _dep()
    _patched_per_file(probes)
    run_calibration(probes, ref, _cfg(channels=("T1", "T2")), tmp_path, make_figure=False)
    report = (tmp_path / "T1_report.md").read_text(encoding="utf-8")
    assert "**T1-T2 check** (with this run's fitted coefficients):" in report
    assert "unavailable" not in report
    # With T2 left out of the run, the per-file embedded sets make it unavailable.
    run_calibration(probes, ref, _cfg(channels=("T1",)), tmp_path / "t1", make_figure=False)
    report = (tmp_path / "t1" / "T1_report.md").read_text(encoding="utf-8")
    assert "T1-T2 check unavailable** (T2 was not fitted in this run" in report


# ------------------------------------------------------------ 6. same clock
def test_shares_clock_needs_the_instruments_own_samples():
    probes, ref, _ = _dep(clock_offset=3.0, ctd_delay=1.0)
    assert not shares_clock(probes, ref)  # the demo: whole-second stamps on a 4 Hz grid
    own = _own_record(probes)
    assert shares_clock(probes, own)
    moved = sanitize_reference(own.time, own.value, pressure=own.pressure + 0.05)
    assert not shares_clock(probes, moved)  # same stamps, another pressure sensor
    assert not shares_clock(probes, sanitize_reference(own.time, own.value))


def test_the_instruments_own_reference_skips_the_clock_search(tmp_path, monkeypatch):
    probes, _ref, _ = _dep()
    own = _own_record(probes)

    def boom(*a, **k):
        raise AssertionError("the pressure search ran")

    monkeypatch.setattr(cli, "pressure_offset", boom)
    res = run_calibration(probes, own, _cfg(), tmp_path, make_figure=False)
    c = res["channels"]["T1"]
    assert res["clock_offset_s"] == 0.0 and "by construction" in res["clock_source"]
    assert c["clock_trustworthy"] and c["thermal_lag_s"] == c["lag_s"]
    assert "0 by construction" in (tmp_path / "T1_report.md").read_text(encoding="utf-8")
    json.dumps(res)  # still serialisable


def test_same_clock_false_forces_the_search_and_true_skips_it(tmp_path, monkeypatch):
    probes, ref, _ = _dep(clock_offset=3.0, ctd_delay=1.0)
    calls = []
    real = cli.pressure_offset
    monkeypatch.setattr(cli, "pressure_offset", lambda *a, **k: calls.append(1) or real(*a, **k))
    own = _own_record(probes)
    run_calibration(probes, own, _cfg(reference={"same_clock": False}), tmp_path / "a",
                    make_figure=False)
    assert calls == [1]
    res = run_calibration(probes, ref, _cfg(reference={"same_clock": True}), tmp_path / "b",
                          make_figure=False)
    assert calls == [1] and "same_clock: true" in res["clock_source"]
    # An asserted zero measured nothing: the record says so, and says not to
    # copy it into perturb's hotel.time_offset as a measured offset is.
    assert "ASSERTED" in res["clock_source"] and "not measured" in res["clock_source"]
    assert "hotel.time_offset" in res["clock_source"]


def test_profiles_direction_selects_vmp_descents():
    """A glider config reaches the loader unchanged; a VMP asks for its descents."""
    assert "direction" not in cli._profile_kwargs({"profiles": {"W_min": 0.05}})
    assert cli._profile_kwargs({"profiles": {"direction": "Down"}})["direction"] == "down"
    with pytest.raises(ValueError, match=r"profiles\.direction"):
        cli._profile_kwargs({"profiles": {"direction": "sideways"}})


@pytest.mark.parametrize("value", ["maybe", 1, "yes"])
def test_same_clock_is_validated(value, tmp_path, capsys):
    with pytest.raises(ValueError, match="same_clock"):
        cli._same_clock_setting({"reference": {"same_clock": value}})
    cfg = tmp_path / "fp07-cal.yaml"
    cfg.write_text(f"reference:\n  file: nowhere.nc\n  same_clock: {value}\n", encoding="utf-8")
    assert cli.main(["fit", "-c", str(cfg)]) == 1
    assert "same_clock" in capsys.readouterr().err
