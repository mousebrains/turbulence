# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for the FP07 in-situ calibration (pre-pipeline coefficient extraction).

The load-bearing ones are the recovery tests: a least-squares fit always returns
numbers, so "it ran" proves nothing.  Each estimator is checked against a
deployment synthesised from known coefficients, a known clock offset and a known
drift.
"""

import numpy as np
import pytest

from odas_tpw.fp07cal.fit import fit_calibration
from odas_tpw.fp07cal.lag import highpass, pressure_offset, temperature_lag
from odas_tpw.fp07cal.logr import (
    BridgeParams,
    coeffs_to_config,
    config_to_coeffs,
    live_beta_key,
    log_r,
    temperature,
)
from odas_tpw.fp07cal.pairs import PairConfig, build_pairs_multi
from odas_tpw.fp07cal.geometry import geometry_fit, local_dTdz
from odas_tpw.fp07cal.series import (
    ReferenceSeries,
    load_hotel_reference,
    sanitize_reference,
)
from odas_tpw.fp07cal.stability import blocked_offsets, corroborates, drift_fit, t1_t2_series
from odas_tpw.fp07cal.synth import DEFAULT_BRIDGE, SynthConfig, make_deployment


# --------------------------------------------------------------- coefficients
def test_coeff_config_roundtrip():
    c = np.array([1.0 / 288.0, 1.0 / 3100.0, 1.0 / 2.5e5])
    cfg = coeffs_to_config(c)
    assert cfg["t_0"] == pytest.approx(288.0)
    assert cfg["beta_1"] == pytest.approx(3100.0)
    assert cfg["beta_2"] == pytest.approx(2.5e5)
    np.testing.assert_allclose(config_to_coeffs(cfg), c, rtol=1e-12)


def test_legacy_beta_shadows_beta_1():
    """convert_therm checks `beta` FIRST; patching beta_1 on such a file is a no-op."""
    cfg = {"t_0": "288", "beta": "3000", "beta_1": "9999"}
    assert live_beta_key(cfg) == "beta"
    assert config_to_coeffs(cfg)[1] == pytest.approx(1.0 / 3000.0)
    assert live_beta_key({"t_0": "288", "beta_1": "3100"}) == "beta_1"


def test_bridge_requires_every_parameter():
    """No defaults in the calibration path: a guessed e_b silently breaks the fit."""
    with pytest.raises(ValueError, match="missing bridge parameter"):
        BridgeParams.from_channel_config({"a": "0", "b": "1", "g": "6"}, "T1")


def test_log_r_matches_convert_therm():
    """The fit's L and the reader's L must be identical, or the coefficients lie."""
    from odas_tpw.rsi.channels import convert_therm

    counts = np.linspace(-20000, 20000, 501)
    params = {"a": "-12.3", "b": "0.99921", "g": "6.0", "e_b": "0.68280",
              "adc_fs": "4.096", "adc_bits": "16", "t_0": "289.301",
              "beta_1": "3143.55", "beta_2": "2.5e5"}
    bp = BridgeParams.from_channel_config(params, "T1")
    L, _ = log_r(counts, bp)
    mine = temperature(L, config_to_coeffs(params))
    theirs, _ = convert_therm(counts, params)
    np.testing.assert_allclose(mine, theirs, rtol=1e-12, atol=1e-12)


def test_clipped_samples_are_flagged():
    bp = BridgeParams(a=0.0, b=1.0, g=6.0, e_b=0.68, adc_fs=4.096, adc_bits=16)
    counts = np.array([0.0, 1e9, -1e9])
    _, clipped = log_r(counts, bp)
    assert not clipped[0] and clipped[1] and clipped[2]


# ------------------------------------------------------------------ reference
def test_sanitize_drops_slocum_junk_and_dedupes():
    t = np.array([0.0, 1.7e9, 1.7e9, 1.7e9 + 1, 1.7e9 + 2, np.nan])
    v = np.array([15.0, 20.0, 22.0, 21.0, 999.0, 18.0])
    ref = sanitize_reference(t, v)
    assert ref.time.size == 2  # 0.0 stamp, the 999 value and the NaN all dropped
    assert ref.value[0] == pytest.approx(21.0)  # duplicates averaged
    assert np.all(np.diff(ref.time) > 0)


def test_valid_spans_do_not_bridge_gaps():
    """The every-n-th-yo case: no reference between spans, not a wide one."""
    t = np.concatenate([np.arange(0, 60.0), np.arange(3600.0, 3660.0)])
    ref = ReferenceSeries(time=t, value=np.full(t.size, 20.0))
    spans = ref.valid_spans(max_gap=30.0)
    assert len(spans) == 2
    assert spans[0] == (0, 59)


# ----------------------------------------------------------------- V1/V2/V3
def _deployment(**kw):
    cfg = SynthConfig(n_yos=24, yo_seconds=1200, fs=8.0, files_per_deployment=4,
                      clock_offset=kw.pop("clock_offset", 3.0),
                      ctd_delay=kw.pop("ctd_delay", 1.0), **kw)
    return make_deployment(cfg)


def test_v1_dense_reference_recovers_coefficients():
    probes, ref, truth = _deployment(ct_every_n=1)
    pc = PairConfig(max_gap=30.0)
    lr, pairs = temperature_lag(probes, ref, "T1", cfg=pc, max_lag=12.0, step=0.5)
    fit = fit_calibration(pairs, order=1)
    assert fit.config_equivalent["t_0"] == pytest.approx(truth["t_0"], abs=2e-3)
    assert fit.config_equivalent["beta_1"] == pytest.approx(truth["beta_1"], rel=2e-3)


def test_v2_sparse_reference_matches_dense():
    """CT on every 3rd yo must give the same coefficients as continuous CT."""
    dense = _deployment(ct_every_n=1)
    sparse = _deployment(ct_every_n=3)
    pc = PairConfig(max_gap=30.0)
    out = {}
    for name, (probes, ref, _t) in (("dense", dense), ("sparse", sparse)):
        _lr, pairs = temperature_lag(probes, ref, "T1", cfg=pc, max_lag=12.0, step=0.5)
        out[name] = fit_calibration(pairs, order=1).config_equivalent
    assert out["sparse"]["t_0"] == pytest.approx(out["dense"]["t_0"], abs=3e-3)
    assert out["sparse"]["beta_1"] == pytest.approx(out["dense"]["beta_1"], rel=3e-3)
    # And the sparse set really is sparse — roughly a third of the pairs.
    assert len(sparse[1].time) < 0.5 * len(dense[1].time)


def test_v3_no_pairs_are_invented_across_a_gap():
    """The A1 regression: a fabricated mid-gap reference must yield ZERO pairs."""
    probes, ref, _truth = _deployment(ct_every_n=3)
    pc = PairConfig(max_gap=30.0)
    pairs = build_pairs_multi(probes, ref, "T1", lag=0.0, cfg=pc)
    # Every pair must sit within max_gap of an actual reference sample.
    nearest = np.abs(pairs.time[:, None] - ref.time[None, :]).min(axis=1) if len(pairs) < 4000 \
        else np.array([np.abs(ref.time - t).min() for t in pairs.time[:2000]])
    assert nearest.max() <= 1e-6  # pairs live exactly ON reference samples


def test_zero_pairs_when_reference_does_not_overlap():
    probes, _ref, _t = _deployment(ct_every_n=1)
    far = ReferenceSeries(time=np.arange(0.0, 100.0) + 1.0e8,
                          value=np.full(100, 20.0))
    pairs = build_pairs_multi(probes, far, "T1", lag=0.0, cfg=PairConfig())
    assert len(pairs) == 0
    assert pairs.rejected["no_reference_coverage"] > 0


# ------------------------------------------------------------------ V7: lags
def test_v7_clock_offset_and_sensor_response_separate():
    """P-vs-P must recover the clock offset; the remainder is the CTD response."""
    probes, ref, truth = _deployment(ct_every_n=1, clock_offset=5.0, ctd_delay=2.0)
    po = pressure_offset(probes, ref, max_lag=20.0, step=0.25)
    assert po.lag == pytest.approx(truth["clock_offset"], abs=0.6)
    lr, _pairs = temperature_lag(probes, ref, "T1", cfg=PairConfig(max_gap=30.0),
                                 max_lag=20.0, step=0.25)
    assert lr.lag - po.lag == pytest.approx(truth["ctd_delay"], abs=0.8)


def test_lag_gate_rejects_a_flat_peak():
    """A high correlation is not evidence — a flat score curve must be refused."""
    from odas_tpw.fp07cal.lag import LagResult

    flat = LagResult(lag=4.0, score=0.999999, dynamic_range=2e-5, width=30.0)
    sharp = LagResult(lag=4.0, score=0.97, dynamic_range=0.97, width=0.8)
    assert not flat.trustworthy()
    assert sharp.trustworthy()


def test_highpass_removes_a_ramp():
    t = np.arange(0.0, 600.0)
    assert np.max(np.abs(highpass(t, 3.0 + 0.5 * t, 30.0)[40:-40])) < 1e-9


# -------------------------------------------------------------- V14: drift
def _stability(drift, t2_drift=0.0, n_blocks=6):
    cfg = SynthConfig(n_yos=90, yo_seconds=2400, fs=4.0, ct_every_n=3,
                      files_per_deployment=10, clock_offset=2.0, ctd_delay=1.0,
                      drift_K_per_day=drift)
    probes, ref, truth = make_deployment(cfg, t2_drift_K_per_day=t2_drift)
    pc = PairConfig(max_gap=30.0)
    _lr, pairs = temperature_lag(probes, ref, "T1", cfg=pc, max_lag=10.0, step=0.5)
    fit = fit_calibration(pairs, order=1)
    stab = drift_fit(blocked_offsets(pairs, fit, n_blocks=n_blocks), n_permutations=800)
    stab.channel = "T1"
    return stab, t1_t2_series(probes), truth


def test_v14_drift_recovered_and_control_is_quiet():
    stab, _t1t2, truth = _stability(drift=0.02)
    assert stab.significant
    assert stab.probe_drift_K_per_day == pytest.approx(truth["drift_K_per_day"], rel=0.15)

    control, _t, _tr = _stability(drift=0.0)
    assert not control.significant


def test_t1_t2_discriminates_common_from_probe_specific_drift():
    """Both probes moving together is not a bead story; the report must say so."""
    both, t1t2_both, _ = _stability(drift=0.02, t2_drift=0.0)
    verdict = corroborates(both, t1t2_both)
    assert "moved together" in verdict

    only_t1, t1t2_one, _ = _stability(drift=0.02, t2_drift=-0.02)
    assert "probe-specific" in corroborates(only_t1, t1t2_one)


# ------------------------------------------------------------------ the fit
def test_centering_keeps_the_design_matrix_conditioned():
    probes, ref, _t = _deployment(ct_every_n=1)
    _lr, pairs = temperature_lag(probes, ref, "T1", cfg=PairConfig(max_gap=30.0),
                                 max_lag=10.0, step=1.0)
    fit = fit_calibration(pairs, order=2)
    assert fit.condition < 100.0  # raw Vandermonde in L would be orders worse


def test_fit_refuses_too_few_pairs():
    from odas_tpw.fp07cal.pairs import PairSet

    tiny = PairSet(time=np.arange(2.0), T_ref=np.array([20.0, 21.0]),
                   L=np.array([-0.1, -0.11]), channel="T1")
    with pytest.raises(ValueError, match="too few"):
        fit_calibration(tiny, order=2)


def test_boundary_peak_is_never_trustworthy():
    """A score still climbing at the search edge has no maximum inside it."""
    from odas_tpw.fp07cal.lag import LagResult

    edge = LagResult(lag=25.0, score=0.9, dynamic_range=0.8, width=0.5,
                     at_boundary=True)
    assert not edge.trustworthy()
    assert "boundary" in edge.summary()


def test_joint_fit_recovers_an_injected_depth_offset():
    """dz must be recovered AND kept out of t_0 — which needs a joint fit."""
    from odas_tpw.fp07cal.geometry import joint_fit

    probes, ref, truth = _deployment(ct_every_n=1)
    _lr, pairs = temperature_lag(probes, ref, "T1", cfg=PairConfig(max_gap=30.0),
                                 max_lag=10.0, step=1.0)
    g = local_dTdz(pairs)
    ok = np.isfinite(g)
    assert ok.sum() > 100

    dz = 0.25
    pairs.T_ref = pairs.T_ref + np.where(ok, dz * g, 0.0)
    fit, geo = joint_fit(pairs, order=1, dTdz=g)
    assert geo.dz_m == pytest.approx(dz, abs=0.05)
    # ...and the coefficients must come back clean rather than having eaten it.
    assert fit.config_equivalent["beta_1"] == pytest.approx(truth["beta_1"], rel=5e-3)


def test_two_step_geometry_is_absorbed_by_the_calibration():
    """Why joint_fit exists: post-fit residuals badly underestimate dz."""
    from odas_tpw.fp07cal.geometry import joint_fit

    probes, ref, _t = _deployment(ct_every_n=1)
    _lr, pairs = temperature_lag(probes, ref, "T1", cfg=PairConfig(max_gap=30.0),
                                 max_lag=10.0, step=1.0)
    g = local_dTdz(pairs)
    dz = 0.25
    pairs.T_ref = pairs.T_ref + np.where(np.isfinite(g), dz * g, 0.0)

    two_step = geometry_fit(pairs, fit_calibration(pairs, order=1), dTdz=g)
    _f, joint = joint_fit(pairs, order=1, dTdz=g)
    assert abs(two_step.dz_m - dz) > abs(joint.dz_m - dz)


def test_local_dTdz_sees_the_thermocline():
    probes, ref, _t = _deployment(ct_every_n=1)
    _lr, pairs = temperature_lag(probes, ref, "T1", cfg=PairConfig(max_gap=30.0),
                                 max_lag=6.0, step=2.0)
    g = local_dTdz(pairs)
    fin = g[np.isfinite(g)]
    assert fin.size > 100
    # The synthetic column warms toward the surface, and the thermocline gives
    # dT/dz a dynamic range -- which is the lever arm geometry_fit relies on.
    assert np.nanmedian(fin) < 0
    assert np.nanmax(np.abs(fin)) / max(np.nanmedian(np.abs(fin)), 1e-9) > 2.0


def test_hotel_pressure_scale_applied():
    """Slocum reports bar; a raw ebd.nc read must be able to convert to dbar."""
    import tempfile
    from pathlib import Path

    import xarray as xr

    ds = xr.Dataset({
        "sci_ctd41cp_timestamp": ("i", np.arange(1.7e9, 1.7e9 + 10)),
        "sci_water_temp": ("i", np.full(10, 20.0)),
        "sci_water_pressure": ("i", np.full(10, 10.0)),
    })
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "h.nc"
        ds.to_netcdf(p)
        raw = load_hotel_reference(p)
        scaled = load_hotel_reference(p, pressure_scale=10.0)
    assert raw.pressure[0] == pytest.approx(10.0)
    assert scaled.pressure[0] == pytest.approx(100.0)


def test_beta_2_zero_is_not_deletion_but_neutral_is():
    """Setting beta_2=0 crashes the reader; only beta_2 -> infinity removes the term."""
    from odas_tpw.fp07cal.patch import NEUTRAL
    from odas_tpw.rsi.channels import convert_therm

    base = {"a": "-12.3", "b": "0.99921", "g": "6.0", "e_b": "0.68280",
            "adc_fs": "4.096", "adc_bits": "16", "t_0": "286.65", "beta_1": "3051.45"}
    counts = np.array([-5000.0, 0.0, 5000.0])
    absent, _ = convert_therm(counts, dict(base))

    with pytest.raises(ZeroDivisionError):
        convert_therm(counts, {**base, "beta_2": "0"})

    neutral, _ = convert_therm(counts, {**base, "beta_2": NEUTRAL})
    np.testing.assert_array_equal(neutral, absent)


def test_patch_edits_neutralise_a_stale_higher_order_term():
    from odas_tpw.fp07cal.patch import NEUTRAL, build_edits

    record = {
        "instrument_sn": "435",
        "channels": {"T1": {
            "config_equivalent": {"t_0": 286.65, "beta_1": 3051.45},
            "bridge": {"a": -12.3, "b": 0.99921, "g": 6.0, "e_b": 0.6828,
                       "adc_fs": 4.096, "adc_bits": 16.0},
            "beta_key": "beta_1", "lag_trustworthy": True,
        }},
    }
    config = {"instrument_info": {"sn": "435"}, "channels": [
        {"name": "T1", "a": "-12.3", "b": "0.99921", "g": "6.0", "e_b": "0.68280",
         "adc_fs": "4.096", "adc_bits": "16", "t_0": "289.301",
         "beta_1": "3143.55", "beta_2": "2.5e5"},
    ]}
    plan = build_edits(record, config)
    assert plan.ok
    assert plan.edits["T1"]["beta_2"] == NEUTRAL
    assert any("neutralising" in w for w in plan.warnings)


def test_patch_refuses_a_bridge_mismatch():
    from odas_tpw.fp07cal.patch import build_edits

    record = {
        "instrument_sn": "435",
        "channels": {"T1": {
            "config_equivalent": {"t_0": 286.65, "beta_1": 3051.45},
            "bridge": {"a": -12.3, "b": 0.99921, "g": 6.0, "e_b": 0.6828,
                       "adc_fs": 4.096, "adc_bits": 16.0},
            "beta_key": "beta_1", "lag_trustworthy": True,
        }},
    }
    config = {"instrument_info": {"sn": "435"}, "channels": [
        {"name": "T1", "a": "-12.3", "b": "0.99921", "g": "3.0", "e_b": "0.68280",
         "adc_fs": "4.096", "adc_bits": "16", "t_0": "289.301", "beta_1": "3143.55"},
    ]}
    plan = build_edits(record, config)
    assert not plan.ok
    assert any("bridge parameter g differs" in e for e in plan.errors)


def test_patch_refuses_an_untrustworthy_lag():
    from odas_tpw.fp07cal.patch import build_edits

    record = {"instrument_sn": "435", "channels": {"T1": {
        "config_equivalent": {"t_0": 286.65, "beta_1": 3051.45},
        "bridge": {}, "beta_key": "beta_1", "lag_trustworthy": False}}}
    config = {"instrument_info": {"sn": "435"},
              "channels": [{"name": "T1", "t_0": "289.301", "beta_1": "3143.55"}]}
    plan = build_edits(record, config)
    assert not plan.ok
    assert any("sharpness gate" in e for e in plan.errors)


def test_patch_writes_to_the_live_beta_key():
    """convert_therm checks `beta` first, so patching beta_1 there is a no-op."""
    from odas_tpw.fp07cal.patch import build_edits

    record = {"instrument_sn": "435", "channels": {"T1": {
        "config_equivalent": {"t_0": 286.65, "beta_1": 3051.45},
        "bridge": {}, "beta_key": "beta_1", "lag_trustworthy": True}}}
    config = {"instrument_info": {"sn": "435"}, "channels": [
        {"name": "T1", "t_0": "289.301", "beta": "3143.55", "beta_1": "9999"}]}
    plan = build_edits(record, config)
    assert "beta" in plan.edits["T1"]
    assert "beta_1" not in plan.edits["T1"]


def test_select_order_prefers_the_order_that_extrapolates():
    """In-sample fit always improves with order; held-out is the real test."""
    from odas_tpw.fp07cal.fit import select_order

    probes, ref, _t = _deployment(ct_every_n=1)
    _lr, pairs = temperature_lag(probes, ref, "T1", cfg=PairConfig(max_gap=30.0),
                                 max_lag=8.0, step=1.0)
    order, scores = select_order(pairs)
    assert order in (1, 2, 3)
    assert scores, "select_order must report its held-out errors"
    # Whatever it picks must actually be the held-out minimum.
    assert order == min(scores, key=lambda o: scores[o]["held_out_K"])
    # In-sample error is monotone non-increasing in order — which is exactly
    # why it cannot be the selection criterion.
    ins = [scores[o]["in_sample_K"] for o in sorted(scores)]
    assert all(b <= a * 1.0001 for a, b in zip(ins, ins[1:]))


def test_bridge_inverse_is_exact():
    from odas_tpw.fp07cal.synth import _L_to_counts

    L = np.linspace(-0.30, 0.10, 200)
    back, _ = log_r(_L_to_counts(L, DEFAULT_BRIDGE), DEFAULT_BRIDGE)
    np.testing.assert_allclose(back, L, rtol=1e-10, atol=1e-12)
