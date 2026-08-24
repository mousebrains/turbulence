# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for the FP07 in-situ calibration (pre-pipeline coefficient extraction).

The load-bearing ones are the recovery tests: a least-squares fit always returns
numbers, so "it ran" proves nothing.  Each estimator is checked against a
deployment synthesised from known coefficients, a known clock offset and a known
drift.
"""

import itertools
from pathlib import Path

import numpy as np
import pytest

from odas_tpw.fp07cal.fit import fit_calibration
from odas_tpw.fp07cal.geometry import geometry_fit, local_dTdz
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
# ------------------------------------------------- shipped example configs
def test_shipped_fp07_cal_example_matches_the_code_defaults():
    """The example must not silently pin a value the code has since corrected.

    #153 changed ``kernel_tau`` from the measured 2.7 s delay to the 0.7 s
    thermistor pole, but ``examples/slocum_glider_hotel/fp07-cal.yaml`` kept
    2.7 -- and because an explicit key OVERRIDES the default, anyone starting
    from the example got the corrected-away value straight back. Nothing
    caught it, because no test loaded that file.
    """
    import yaml

    from odas_tpw.fp07cal.cli import _pair_config

    example = (
        Path(__file__).resolve().parents[1]
        / "examples/slocum_glider_hotel/fp07-cal.yaml"
    )
    if not example.exists():
        pytest.skip("example config not present")
    cfg = yaml.safe_load(example.read_text())

    from_example = _pair_config(cfg)
    defaults = _pair_config({})
    # max_gap is deliberately restated in the example (it is the one knob a
    # user must think about); everything else it sets has to agree with the
    # code, or the example is pinning a stale value.
    for name in ("kernel_tau", "min_speed", "min_corr", "require_profile"):
        assert getattr(from_example, name) == getattr(defaults, name), (
            f"example fp07-cal.yaml pins {name}={getattr(from_example, name)!r} "
            f"but the code default is now {getattr(defaults, name)!r}"
        )


def test_sanitize_drops_slocum_junk_and_dedupes():
    t = np.array([0.0, 1.7e9, 1.7e9, 1.7e9 + 1, 1.7e9 + 2, np.nan])
    v = np.array([15.0, 20.0, 22.0, 21.0, 999.0, 18.0])
    ref = sanitize_reference(t, v)
    assert ref.time.size == 2  # 0.0 stamp, the 999 value and the NaN all dropped
    assert ref.value[0] == pytest.approx(21.0)  # duplicates averaged
    assert np.all(np.diff(ref.time) > 0)


def test_sanitize_reports_why_samples_were_dropped():
    """A reference that mostly evaporates should say so, not just come back short."""
    t = np.array([0.0, 1.7e9, 1.7e9, 1.7e9 + 1, 1.7e9 + 2, np.nan])
    v = np.array([15.0, 20.0, 22.0, 21.0, 999.0, 18.0])
    stats: dict = {}
    ref = sanitize_reference(t, v, stats=stats)
    assert stats == {
        "n_total": 6,
        "n_bad_time": 2,  # the 0.0 sentinel and the NaN
        "n_bad_value": 1,  # 999 degC, on an otherwise fine stamp
        "n_duplicate": 1,
        "n_kept": 2,
    }
    assert stats["n_kept"] == ref.time.size


@pytest.mark.parametrize(
    ("method", "expected"), [("mean", 21.0), ("first", 20.0), ("last", 22.0)]
)
def test_sanitize_dedupe_method_is_selectable(method, expected):
    """Same three rules as dinkum-hotel, because it is the same implementation."""
    t = np.array([1.7e9, 1.7e9, 1.7e9 + 1])
    v = np.array([20.0, 22.0, 30.0])
    ref = sanitize_reference(t, v, dedupe=method)
    assert ref.value[0] == pytest.approx(expected)


def test_sanitize_accepts_iso8601_time_bounds():
    """Inherited from resolve_time_bounds: a date is easier to reason about."""
    t = np.array([1.4e9, 1.7e9, 1.9e9])  # 2014, 2023, 2030
    v = np.full(3, 20.0)
    ref = sanitize_reference(t, v, time_min="2020-01-01", time_max="2026-01-01")
    assert ref.time.size == 1
    assert ref.time[0] == pytest.approx(1.7e9)


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
    _lr, pairs = temperature_lag(probes, ref, "T1", cfg=pc, max_lag=12.0, step=0.5)
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


def test_datetime64_time_base_is_accepted():
    """`dinkum-hotel` writes its time basis as datetime64; ebd.nc uses a float epoch.

    Casting datetime64 straight to float yields ~1.7e18 nanoseconds, which the
    sanitiser then correctly discards as out of range — leaving "0 reference
    samples" with no obvious cause. Both dialects must load identically.
    """
    import tempfile
    from pathlib import Path

    import xarray as xr

    epoch = np.arange(1.7e9, 1.7e9 + 10)
    temp = np.linspace(18.0, 20.0, 10)
    with tempfile.TemporaryDirectory() as d:
        f_num = Path(d) / "epoch.nc"
        xr.Dataset({
            "sci_ctd41cp_timestamp": ("i", epoch),
            "sci_water_temp": ("i", temp),
        }).to_netcdf(f_num)
        f_dt = Path(d) / "dt64.nc"
        xr.Dataset({
            "sci_ctd41cp_timestamp": (
                "i", (epoch * 1e9).astype("datetime64[ns]")),
            "sci_water_temp": ("i", temp),
        }).to_netcdf(f_dt)
        a = load_hotel_reference(f_num, pressure_var=None)
        b = load_hotel_reference(f_dt, pressure_var=None)
    assert a.time.size == 10 and b.time.size == 10
    np.testing.assert_allclose(a.time, b.time, atol=1e-6)
    np.testing.assert_allclose(a.value, b.value)


def test_nat_does_not_become_a_bogus_epoch():
    from odas_tpw.fp07cal.series import _to_epoch_seconds

    v = np.array(["2024-01-01T00:00:00", "NaT"], dtype="datetime64[ns]")
    out = _to_epoch_seconds(v)
    assert np.isfinite(out[0]) and out[0] > 1.7e9
    assert not np.isfinite(out[1])


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


def test_patch_warns_when_extrapolating_outside_the_fitted_range():
    """Applying a polynomial outside its fitted range must be visible."""
    from odas_tpw.fp07cal.patch import build_edits

    record = {"instrument_sn": "435", "channels": {"T1": {
        "config_equivalent": {"t_0": 286.65, "beta_1": 3051.45},
        "bridge": {}, "beta_key": "beta_1", "lag_trustworthy": True,
        "T_range": [10.0, 20.0],
        "validity": {
            "T_fitted": [10.0, 20.0], "T_seen": [4.0, 26.0],
            "extrapolated_below_K": 6.0, "extrapolated_above_K": 6.0,
            "n_profiles_outside": 40, "n_profiles_total": 100,
        },
    }}}
    config = {"instrument_info": {"sn": "435"},
              "channels": [{"name": "T1", "t_0": "289.301", "beta_1": "3143.55"}]}
    plan = build_edits(record, config)
    # A warning, not an error: leaving files on the nominal coefficients would
    # re-create the very discontinuity the tool removes.
    assert plan.ok
    assert any("extrapolating" in w for w in plan.warnings)


def test_patch_warns_on_an_untrusted_probe_serial():
    from odas_tpw.fp07cal.patch import build_edits

    record = {"instrument_sn": "435", "channels": {"T1": {
        "config_equivalent": {"t_0": 286.65, "beta_1": 3051.45},
        "bridge": {}, "beta_key": "beta_1", "lag_trustworthy": True,
        "probe_sn": "T", "probe_sn_trusted": False,
    }}}
    config = {"instrument_info": {"sn": "435"},
              "channels": [{"name": "T1", "t_0": "289.301", "beta_1": "3143.55"}]}
    plan = build_edits(record, config)
    assert plan.ok
    assert any("placeholder" in w for w in plan.warnings)


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
    assert all(b <= a * 1.0001 for a, b in itertools.pairwise(ins))


def test_bridge_inverse_is_exact():
    from odas_tpw.fp07cal.synth import _L_to_counts

    L = np.linspace(-0.30, 0.10, 200)
    back, _ = log_r(_L_to_counts(L, DEFAULT_BRIDGE), DEFAULT_BRIDGE)
    np.testing.assert_allclose(back, L, rtol=1e-10, atol=1e-12)


# --- review findings on PR #149 -------------------------------------------
def _schema_record(sn="435", t_0=286.65, beta_1=3051.45, bridge=None):
    """A complete fp07-cal/1 record, as `fp07-cal fit` writes it."""
    return {
        "schema": "fp07-cal/1",
        "instrument_sn": sn,
        "n_fit_files": 1,
        "channels": {"T1": {
            "config_equivalent": {"t_0": t_0, "beta_1": beta_1},
            "coefficients": [1.0 / t_0, 1.0 / beta_1],
            "bridge": bridge or {"a": -12.3, "b": 0.99921, "g": 6.0, "e_b": 0.6828,
                                 "adc_fs": 4.096, "adc_bits": 16.0},
            "beta_key": "beta_1", "lag_trustworthy": True,
        }},
    }


def _monkeypatched_plan(record, configs, tmp_path):
    """Run patch_deployment (dry) against fake per-file configs."""
    import json

    import odas_tpw.fp07cal.patch as P

    orig = (P.read_config_text, P.parse_config, P.already_patched)
    P.read_config_text = lambda p: Path(p).name
    P.parse_config = lambda name: configs[name]
    P.already_patched = lambda p: False
    try:
        recf = Path(tmp_path) / "rec.json"
        recf.write_text(json.dumps(record))
        srcs = [Path(tmp_path) / name for name in configs]
        plan, _results = P.patch_deployment(recf, srcs, Path(tmp_path) / "out",
                                            dry_run=True)
    finally:
        P.read_config_text, P.parse_config, P.already_patched = orig
    return plan


def _fake_config(sn="435", a="-12.3", extra=None):
    ch = {"name": "T1", "a": a, "t_0": "289.301", "beta_1": "3143.55"}
    ch.update(extra or {})
    return {"instrument_info": {"sn": sn}, "channels": [ch]}


def test_patch_flags_an_sn_mismatch_in_a_later_source(tmp_path):
    """Checking one file while patching a list is not a check (SN gate)."""
    record = _schema_record(bridge={"a": -12.3})
    plan = _monkeypatched_plan(record, {
        "first.p": _fake_config(sn="435"),
        "second.p": _fake_config(sn="479"),  # same bridge, different instrument
    }, tmp_path)
    assert not plan.ok
    joined = " ".join(plan.errors)
    assert "second.p" in joined and "instrument SN mismatch" in joined


def test_patch_flags_a_bridge_mismatch_in_a_later_source(tmp_path):
    """The bridge gate must fire independently of the SN gate."""
    record = _schema_record(bridge={"a": -12.3})
    plan = _monkeypatched_plan(record, {
        "first.p": _fake_config(a="-12.3"),
        "second.p": _fake_config(a="-16.3"),  # same SN, different bridge
    }, tmp_path)
    assert not plan.ok
    joined = " ".join(plan.errors)
    assert "second.p" in joined and "bridge parameter" in joined
    assert "instrument SN mismatch" not in joined


def test_patch_flags_divergent_edits_across_sources(tmp_path):
    """Two files that resolve to different edits do not share one calibration."""
    record = _schema_record(bridge={"a": -12.3})
    plan = _monkeypatched_plan(record, {
        "first.p": _fake_config(),
        # Same SN and bridge, but carries a live beta_2 the first lacks: its
        # edit set gains a neutralisation and so differs from first.p's.
        "second.p": _fake_config(extra={"beta_2": "2.5e5"}),
    }, tmp_path)
    assert not plan.ok
    assert any("different edits" in e for e in plan.errors)


# ---- P1: the record itself is validated ----------------------------------
def test_patch_refuses_a_foreign_schema(tmp_path):
    """A JSON that is not an fp07-cal record must be refused outright."""
    import json

    from odas_tpw.fp07cal.patch import patch_deployment

    rec = tmp_path / "rec.json"
    rec.write_text(json.dumps({
        "schema": "something-else/9",
        "channels": {"T1": {"config_equivalent": {"t_0": 291.0}}},
    }))
    with pytest.raises(ValueError, match="schema"):
        patch_deployment(rec, [Path("tests/data/MR_SL435.p")], tmp_path / "out",
                         dry_run=True)


def test_patch_refuses_a_partial_record(tmp_path):
    """Missing bridge/coefficients would make every safety gate vacuous."""
    import json

    from odas_tpw.fp07cal.patch import patch_deployment, validate_record

    rec = _schema_record()
    del rec["channels"]["T1"]["bridge"]
    with pytest.raises(ValueError, match="missing 'bridge'"):
        validate_record(rec)

    # config_equivalent without t_0 must be a ValueError, not a KeyError later.
    rec2 = _schema_record()
    rec2["channels"]["T1"]["config_equivalent"] = {"T0": 291.0}
    f = tmp_path / "rec.json"
    f.write_text(json.dumps(rec2))
    with pytest.raises(ValueError, match="t_0"):
        patch_deployment(f, [Path("tests/data/MR_SL435.p")], tmp_path / "out",
                         dry_run=True)

    with pytest.raises(ValueError, match="instrument_sn"):
        validate_record({"schema": "fp07-cal/1",
                         "channels": {"T1": {"config_equivalent": {"t_0": 1.0},
                                             "coefficients": [1.0],
                                             "bridge": {"a": 1.0}}}})


def test_patch_real_sn_mismatch_on_disk(tmp_path):
    """Mixed-instrument directory with REAL files: nothing may be written."""
    import json

    from odas_tpw.fp07cal.patch import patch_deployment

    rec = tmp_path / "rec.json"
    rec.write_text(json.dumps(_schema_record()))
    srcs = [Path("tests/data/MR_SL435.p"), Path("tests/data/SN479_0006.p")]
    plan, results = patch_deployment(rec, srcs, tmp_path / "out", dry_run=False)
    assert not plan.ok
    assert results == []
    assert not (tmp_path / "out").exists()
    joined = " ".join(plan.errors)
    assert "instrument SN mismatch" in joined


# ---- P2/(d): fp07-cal banner detection, not any config_patch banner -------
def _patch_435(tmp_path, out_name="patched"):
    import json
    import shutil

    from odas_tpw.fp07cal.patch import patch_deployment

    src = tmp_path / "MR_SL435.p"
    if not src.exists():
        shutil.copy("tests/data/MR_SL435.p", src)
    rec = tmp_path / "rec.json"
    rec.write_text(json.dumps(_schema_record()))
    return patch_deployment(rec, [src], tmp_path / out_name)


def test_generic_config_patch_does_not_block_calibration(tmp_path):
    """A bridge-parameter fix via rsi-tpw patch-config must leave the file
    eligible — fixing the config FIRST is the documented workflow."""
    import json
    import shutil

    from odas_tpw.fp07cal.patch import already_patched, patch_deployment
    from odas_tpw.rsi.config_patch import EditSpec, patch_files

    orig = tmp_path / "MR_SL435.p"
    shutil.copy("tests/data/MR_SL435.p", orig)
    spec = EditSpec(note="fix a bridge value", author="operator",
                    channels={"T1": {"b": "0.99922"}})
    results = patch_files([orig], tmp_path / "fixed", spec, batch_cal=True)
    fixed = results[0][1]
    assert fixed is not None
    assert not already_patched(fixed)  # generic banner, not fp07-cal's

    rec = tmp_path / "rec.json"
    record = _schema_record()
    record["channels"]["T1"]["bridge"]["b"] = 0.99922  # match the fixed file
    rec.write_text(json.dumps(record))
    plan, res = patch_deployment(rec, [fixed], tmp_path / "cal")
    assert plan.ok
    assert res[0][1] is not None


def test_patch_deployment_twice_refuses(tmp_path):
    """A REALLY patched file (not a hand-built one) must be refused."""

    from odas_tpw.fp07cal.patch import already_patched, patch_deployment

    plan, results = _patch_435(tmp_path)
    assert plan.ok
    patched = results[0][1]
    assert already_patched(patched)
    assert not already_patched(tmp_path / "MR_SL435.p")

    rec = tmp_path / "rec.json"  # written by _patch_435
    with pytest.raises(ValueError, match="already carry an fp07-cal"):
        patch_deployment(rec, [patched], tmp_path / "again")
    assert not (tmp_path / "again").exists()


# ---- (e): real .p round-trip through the reader ---------------------------
def test_patched_file_reproduces_the_fit_exactly(tmp_path):
    """PFile on the patched copy must evaluate the fitted polynomial to 0.0 K."""
    from odas_tpw.rsi.p_file import PFile

    plan, results = _patch_435(tmp_path)
    assert plan.ok
    pf = PFile(str(results[0][1]))
    cfg_t1 = next(c for c in pf.config["channels"]
                  if str(c.get("name", "")).strip() == "T1")
    bp = BridgeParams.from_channel_config(cfg_t1, "T1")
    L, _clipped = log_r(np.asarray(pf.channels_raw["T1"], dtype=np.float64), bp)
    rec = _schema_record()["channels"]["T1"]
    mine = temperature(L, np.asarray(rec["coefficients"]))
    np.testing.assert_allclose(mine, pf.channels["T1"], rtol=0, atol=0)


# ---- P3/P4: gathering and the reference -----------------------------------
def test_gather_excludes_the_output_dir(tmp_path, capsys):
    """patch writes output_dir/patched/*.p inside the default layout; the
    recursive glob must not sweep the tool's own output back in as input."""
    from odas_tpw.fp07cal.cli import _gather_paths

    (tmp_path / "a.p").write_bytes(b"x")
    out = tmp_path / "fp07cal" / "patched"
    out.mkdir(parents=True)
    (out / "a.p").write_bytes(b"x")
    cfg = {"files": {"p_file_root": str(tmp_path), "p_file_pattern": "**/*.p",
                     "output_dir": str(tmp_path / "fp07cal")}}
    paths = _gather_paths(cfg)
    assert paths == [tmp_path / "a.p"]
    assert "output_dir" in capsys.readouterr().err


def test_missing_reference_block_is_a_clear_error():
    from odas_tpw.fp07cal.cli import _load_reference

    with pytest.raises(ValueError, match="reference"):
        _load_reference({})
    with pytest.raises(ValueError, match="does not exist"):
        _load_reference({"reference": {"file": "/no/such/hotel.nc"}})


def test_cli_patch_does_not_need_the_hotel_file(tmp_path):
    """`fp07-cal patch` edits configs; a missing hotel.nc must not stop it."""
    import json
    import shutil

    from odas_tpw.fp07cal.cli import main

    root = tmp_path / "deploy"
    root.mkdir()
    shutil.copy("tests/data/MR_SL435.p", root / "MR_SL435.p")
    out_dir = root / "fp07cal"
    out_dir.mkdir()
    (out_dir / "coefficients.json").write_text(json.dumps(_schema_record()))
    cfg = root / "fp07-cal.yaml"
    # POSIX-style paths, and single quotes so YAML does no escape processing:
    # a Windows tmp_path in a DOUBLE-quoted scalar makes "C:\\Users" a \\U
    # unicode escape and ruamel raises before the test can run.
    cfg.write_text(
        "files:\n"
        f"  p_file_root: '{root.as_posix()}'\n"
        "  p_file_pattern: '**/*.p'\n"
        f"  output_dir: '{out_dir.as_posix()}'\n"
        "reference:\n"
        f"  file: '{root.as_posix()}/no_such_hotel.nc'\n"
        "channels: ['T1']\n"
    )
    assert main(["patch", "-c", str(cfg), "--dry-run"]) == 0


# ---- P5: nothing written when a destination exists ------------------------
def test_patch_prechecks_destinations_before_writing(tmp_path):
    import shutil

    from odas_tpw.fp07cal.patch import patch_deployment

    _plan, results = _patch_435(tmp_path)  # occupies tmp_path/patched
    assert results[0][1] is not None
    # Second run into the same out dir: refuse up front, write nothing new.
    rec = tmp_path / "rec.json"
    src2 = tmp_path / "src2"
    src2.mkdir()
    shutil.copy("tests/data/MR_SL435.p", src2 / "MR_SL435.p")
    before = sorted((tmp_path / "patched").iterdir())
    with pytest.raises(ValueError, match="already exist"):
        patch_deployment(rec, [src2 / "MR_SL435.p"], tmp_path / "patched")
    assert sorted((tmp_path / "patched").iterdir()) == before


# ---- P6: streaming skips are counted --------------------------------------
def test_stream_reports_failures(tmp_path):
    from odas_tpw.fp07cal.cli import _stream

    bad = tmp_path / "bad.p"
    bad.write_bytes(b"1234567")
    failures: list = []
    assert list(_stream([bad], None, failures)) == []
    assert len(failures) == 1
    assert failures[0][0] == "bad.p"
    assert "Error" in failures[0][1] or "error" in failures[0][1]


# ---- P7: profile knobs are plumbed ----------------------------------------
def test_profile_kwargs_come_from_the_config():
    from odas_tpw.fp07cal.cli import _profile_kwargs

    kw = _profile_kwargs({"profiles": {"W_min": 0.2, "min_duration": 10,
                                       "speed_var": None}})
    assert kw == {"speed_var": None, "W_min": 0.2, "P_min": 0.5,
                  "min_duration": 10.0}
    assert _profile_kwargs({})["W_min"] == pytest.approx(0.05)


# ---- P8 -------------------------------------------------------------------
def test_sanitize_pressure_dedupe_is_nan_aware():
    """p=[5, nan] on a repeated stamp must give 5.0, not NaN."""
    t = np.array([1.7e9, 1.7e9, 1.7e9 + 1])
    v = np.array([20.0, 20.5, 21.0])
    p = np.array([5.0, np.nan, 6.0])
    ref = sanitize_reference(t, v, pressure=p)
    assert ref.pressure[0] == pytest.approx(5.0)
    assert ref.pressure[1] == pytest.approx(6.0)


# ---- N1: a killing test for order selection -------------------------------
def test_select_order_finds_a_known_quadratic_truth():
    """Thermocline-like density (dense cold sliver, sparse warm tail) with an
    order-2 truth: order 2 must be chosen, and order 3 must score WORSE
    held-out — the overfit direction, so this is not `min(scores)` restated."""
    from odas_tpw.fp07cal.fit import select_order
    from odas_tpw.fp07cal.pairs import PairSet

    rng = np.random.default_rng(7)
    coeffs = np.array([1.0 / 289.0, 1.0 / 3100.0, 1.0 / 2.5e5])
    # The cold cluster spans ~0.27 K, the warm tail ~10 K: under a median
    # split the "cold half" is the sliver alone and extrapolating it across
    # the tail punishes every order >= 2 (a median-split mutant picks 1 here).
    L = np.concatenate([rng.uniform(0.110, 0.120, 4000),  # dense cold sliver
                        rng.uniform(-0.30, 0.110, 800)])  # sparse warm tail
    invT = coeffs[0] + coeffs[1] * L + coeffs[2] * L**2
    T = 1.0 / invT - 273.15 + rng.normal(0.0, 1e-3, L.size)
    n = L.size
    ps = PairSet(time=np.arange(n, dtype=float), T_ref=T, L=L,
                 pressure=np.full(n, 50.0), w=np.full(n, 0.2),
                 direction=np.ones(n),
                 profile_uid=np.array(["p0"] * n, dtype=object),
                 file_label=np.array(["f0"] * n, dtype=object), channel="T1")
    order, scores = select_order(ps)
    assert order == 2
    assert scores[3]["held_out_K"] > scores[2]["held_out_K"]


# ---- (a): the estimators must POPULATE the sharpness gate -----------------
def test_estimators_populate_a_trustworthy_gate():
    """Not the predicate alone: running the estimators on a resolvable
    deployment must yield trustworthy() results."""
    probes, ref, _t = _deployment(ct_every_n=1)
    lr, _pairs = temperature_lag(probes, ref, "T1", cfg=PairConfig(max_gap=30.0),
                                 max_lag=12.0, step=0.5)
    assert lr.trustworthy(), lr.summary()
    po = pressure_offset(probes, ref, max_lag=12.0, step=0.5)
    assert po.trustworthy(), po.summary()


def test_without_highpass_the_peak_is_flat():
    """Mean removal instead of high-passing must FAIL the gate: a shifted
    monotone ramp is the same ramp plus a constant, so the score plateaus."""
    probes, ref, _t = _deployment(ct_every_n=1)
    # detrend_s longer than any file makes highpass degrade to mean removal —
    # exactly the mutation that previously survived the test suite.
    lr, _pairs = temperature_lag(probes, ref, "T1", cfg=PairConfig(max_gap=30.0),
                                 max_lag=12.0, step=0.5, detrend_s=1e9)
    assert not lr.trustworthy(), lr.summary()
    po = pressure_offset(probes, ref, max_lag=12.0, step=0.5, detrend_s=1e9)
    assert not po.trustworthy(), po.summary()


# ---- (b): a peak outside the search range must flag at_boundary -----------
def test_offset_beyond_max_lag_hits_the_boundary():
    probes, ref, _t = _deployment(ct_every_n=1, clock_offset=8.0)
    po = pressure_offset(probes, ref, max_lag=4.0, step=0.5)
    assert po.at_boundary
    assert not po.trustworthy()


# ---- (c): gradient_lag tracks an injected shift ---------------------------
def test_gradient_lag_recovers_an_injected_shift():
    from dataclasses import replace

    from odas_tpw.fp07cal.gradient_lag import gradient_lag

    probes, ref, _t = _deployment(ct_every_n=1)
    pc = PairConfig(max_gap=30.0)
    base = gradient_lag(probes, ref, "T1", cfg=pc, max_lag=8.0, step=0.5)
    assert np.isfinite(base.lag)
    shifted_ref = replace(ref, time=ref.time + 3.0)
    shifted = gradient_lag(probes, shifted_ref, "T1", cfg=pc, max_lag=8.0, step=0.5)
    assert shifted.lag - base.lag == pytest.approx(3.0, abs=0.6)


def test_dinkum_refuses_two_sensors_sharing_an_output_name():
    """build_hotel stores data_vars[out_name]; a collision silently overwrites."""
    from odas_tpw.dinkum.config import normalize_sensors

    with pytest.raises(ValueError, match="same output name"):
        normalize_sensors(
            {"sci_water_temp": {"name": "T"}, "sci_water_cond": {"name": "T"}},
            "sci_m_present_time",
        )


def test_dinkum_allows_distinct_output_names():
    from odas_tpw.dinkum.config import normalize_sensors

    out = normalize_sensors({"a": {"name": "x"}, "b": {}}, "t")
    assert sorted(o["name"] for o in out.values()) == ["b", "x"]


# --- the pole is not the delay --------------------------------------------
def test_pole_mismatch_is_not_absorbed_by_the_lag_search():
    """A wrong kernel_tau cannot be compensated by shifting time.

    A pole attenuates as well as delays, so a mismatched pole leaves residual
    the lag search cannot remove. This is why the measured temperature-vs-
    pressure delay (transit + response) must not be fed back in as the pole.
    """
    cfg = SynthConfig(n_yos=16, yo_seconds=1200, fs=8.0, files_per_deployment=3,
                      ct_every_n=1, clock_offset=2.0, ctd_delay=1.0, ctd_tau=0.7)
    probes, ref, _truth = make_deployment(cfg)

    def rms_for(kernel_tau):
        pc = PairConfig(max_gap=30.0, kernel_tau=kernel_tau)
        _lr, pairs = temperature_lag(probes, ref, "T1", cfg=pc, max_lag=12.0, step=0.5)
        return fit_calibration(pairs, order=1).rms_K

    matched = rms_for(0.7)          # model == truth
    over = rms_for(4.0)             # grossly over-filtered
    assert over > matched, (
        f"a 4 s pole against a 0.7 s CTD gave rms {over:.5f} K, no worse than "
        f"the matched {matched:.5f} K -- the mismatch is being absorbed"
    )


def test_synth_truth_and_estimator_model_are_separable():
    """The recovery tests must be able to specify a mismatch, not just match."""
    matched = SynthConfig()
    assert matched.ctd_tau == PairConfig().kernel_tau
    # ...but nothing forces them equal.
    mismatched = SynthConfig(ctd_tau=2.0)
    assert mismatched.ctd_tau != PairConfig().kernel_tau


# ---------------------------------------------- discovery and wrong configs
def test_find_p_files_follows_a_symlinked_directory(tmp_path):
    """`Path.glob` does not traverse a symlink with `**`, and says nothing.

    A deployment kept on an external volume is normally symlinked in
    (`MR -> /Volumes/.../osu685/MR`). Under the default `**/*.p` that matched
    zero files while `ls MR/*.p` showed 1228, and the only output was
    "no .p files matched".
    """
    from odas_tpw.perturb.discover import find_p_files

    real = tmp_path / "elsewhere"
    real.mkdir()
    (real / "a.p").write_bytes(b"x")
    (real / "b.p").write_bytes(b"x")
    root = tmp_path / "deployment"
    root.mkdir()
    (root / "MR").symlink_to(real, target_is_directory=True)

    found = find_p_files(root, "**/*.p")
    assert {p.name for p in found} == {"a.p", "b.p"}
    # A plain (non-recursive) pattern through the link worked before and must
    # keep working.
    assert len(find_p_files(root, "MR/*.p")) == 2


def test_find_p_files_still_filters_what_it_always_did(tmp_path):
    from odas_tpw.perturb.discover import find_p_files

    (tmp_path / "keep.p").write_bytes(b"x")
    (tmp_path / ".hidden.p").write_bytes(b"x")
    (tmp_path / "thing_original.p").write_bytes(b"x")
    (tmp_path / "notap.q").write_bytes(b"x")
    assert {p.name for p in find_p_files(tmp_path)} == {"keep.p"}


def test_a_perturb_config_handed_to_fp07_cal_is_named_as_such():
    """The two configs share `files.p_file_root`, so the mix-up gets far
    enough to look plausible before failing on a missing section."""
    from odas_tpw.fp07cal.cli import _load_reference

    with pytest.raises(ValueError, match="looks like a perturb config"):
        _load_reference({"files": {}, "epsilon": {}, "chi": {}, "hotel": {}})


def test_a_config_merely_missing_reference_gets_the_plain_message():
    from odas_tpw.fp07cal.cli import _load_reference

    with pytest.raises(ValueError, match="no `reference:` block"):
        _load_reference({"files": {}})


def test_non_overlapping_reference_and_p_files_are_flagged():
    """Zero pairs reported as 'sparse coverage' hides a wrong reference file.

    Caught in the wild: an osu685 MicroRider (2025) against a hotel file built
    from ru33 (2021). Coverage reported a 42.5% duty cycle, which was true of
    the reference alone and irrelevant.
    """
    from odas_tpw.fp07cal.cli import _overlap_warning
    from odas_tpw.fp07cal.series import ReferenceSeries

    class _P:
        def __init__(self, lo, hi):
            self.time = np.array([lo, hi])

    ref = ReferenceSeries(
        time=np.array([1.633e9, 1.634e9]), value=np.array([20.0, 21.0])
    )
    assert "do not overlap" in _overlap_warning([_P(1.738e9, 1.744e9)], ref)
    assert _overlap_warning([_P(1.6335e9, 1.6338e9)], ref) == ""
    assert _overlap_warning([], ref) == ""
