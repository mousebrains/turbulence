# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""``fp07-cal`` QC follow-ups to #238, #241 and #243.

#246  Both robust loops kept a pair when ``|resid| <= 4 sigma`` with sigma
      measured about the median: a minority population offset by many sigma
      dragged the least-squares zero off the majority, nothing was rejected and
      the channel was refused at the 1 K gate.
#247  Without a prior, a fit to isothermal noise could be NTC and still have a
      slope no FP07 has, and export.
#248  Thermistor counts PFile reports as bad buffers (or rebuilt by a framing
      repair) reached the calibration as if they were measurements.
#249  A record carried one time span, the run's, so `registry add` gave a bead
      that stopped early the run's span and file count.
"""

from __future__ import annotations

import copy
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from odas_tpw.fp07cal import registry
from odas_tpw.fp07cal.cli import MAX_REJECTED_FRACTION, run_calibration
from odas_tpw.fp07cal.fit import _subset, fit_calibration
from odas_tpw.fp07cal.geometry import joint_fit, local_dTdz
from odas_tpw.fp07cal.pairs import PairConfig, build_pairs_multi
from odas_tpw.fp07cal.patch import validate_record
from odas_tpw.fp07cal.prior import FP07_BETA_K, fp07_slope
from odas_tpw.fp07cal.series import load_probe_series
from odas_tpw.fp07cal.synth import SynthConfig, make_deployment
from odas_tpw.rsi.p_file import PFile


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


def test_the_joint_loop_judges_the_pairs_it_does_not_fit():
    """Pairs with no reference gradient stay out of the fit set but are judged, by
    the calibration's residual: unjudged, they refused the channel at the 1 K gate."""
    pairs, minority, _truth = _bimodal_pairs()
    g = local_dTdz(pairs)
    has = np.isfinite(g)
    assert (~has & minority).sum() > 100  # the profile ends of the offset population
    fit, _geo = joint_fit(pairs, order=1, dTdz=g)
    assert not fit.kept[~has].any()  # still not fitted
    assert fit.outlier[minority].all()  # gradient or not
    assert fit.outlier[~minority].sum() <= 2
    # ...and the fit set, so every coefficient, is what it was.
    alone, _ = joint_fit(_subset(pairs, has), order=1, dTdz=g[has])
    np.testing.assert_array_equal(fit.coeffs, alone.coeffs)
    np.testing.assert_array_equal(fit.outlier[has], alone.outlier)


@pytest.mark.parametrize("geometry", [False, True])
@pytest.mark.parametrize("share", [0.15, 0.30])
def test_a_bimodal_reference_flags_the_minority_and_exports(share, geometry, tmp_path):
    """Issue #246's acceptance case: main refused it (1.80 K on the pairs it kept),
    and with the geometry term (the default) so did the first fix (2.01 K)."""
    probes, ref, truth = _dep()
    off = ref.time > np.quantile(ref.time, 1.0 - share)
    ref.value[off] += 2.0
    c = run_calibration(probes, ref, _cfg(geometry=geometry), tmp_path,
                        make_figure=False)["channels"]["T1"]
    assert "config_equivalent" in c, c
    assert c["config_equivalent"]["t_0"] == pytest.approx(truth["t_0"], abs=1e-3)
    assert c["config_equivalent"]["beta_1"] == pytest.approx(truth["beta_1"], rel=1e-3)
    beyond = c["outliers_beyond_gate"]
    assert beyond["n"] > (share - 0.01) * c["n_pairs"]  # the minority, flagged
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
    tail = {**base, "fraction": 0.069, "n_profiles_mostly_rejected": 7,  # the worst real
            "share_in_those_profiles": 0.127, "whole_profile_fraction": 0.0043}
    assert _population_warning(tail) is None
    assert _population_warning({**tail, "fraction": MAX_REJECTED_FRACTION + 0.01})
    assert _population_warning({**tail, "share_in_those_profiles": 0.5,
                                "whole_profile_fraction": 0.005})
    assert _population_warning({**tail, "n_profiles_mostly_rejected": 0,
                                "share_in_those_profiles": 0.0, "first": None}) is None
    # One rejected pair in a one-pair profile: all of the rejections, but no population.
    one = {**base, "n_rejected": 1, "fraction": 1e-5, "n_profiles_mostly_rejected": 1,
           "share_in_those_profiles": 1.0, "whole_profile_fraction": 1e-5}
    assert _population_warning(one) is None


# ------------------------------- #247: a slope no FP07 has, without a prior
def _iso(seed):
    """Isothermal water (28 degC throughout, 2 mK noise): nothing to fit a slope to."""
    from odas_tpw.fp07cal.synth import DEFAULT_COEFFS

    cfg = SynthConfig(n_yos=16, yo_seconds=1200.0, fs=4.0, ct_every_n=1, T_surface=28.0,
                      T_deep=28.0, files_per_deployment=4, seed=seed)
    return make_deployment(cfg, coeffs=DEFAULT_COEFFS, channels=("T1",))[:2]


def test_an_isothermal_fit_with_a_wrong_slope_is_refused(tmp_path):
    """Seed 1: order 1 is NTC (T falls as L rises) with beta_1 = +4.47e4, 14x an FP07's.

    It passed the NTC check and predicts its own pairs to a few mK, so before
    #247 it was exported."""
    probes, ref = _iso(1)
    res = run_calibration(probes, ref, _cfg(), tmp_path, make_figure=False)
    c = res["channels"]["T1"]
    assert c["error"] == "not an FP07 curve"
    assert "a slope no FP07 has" in c["reason"]
    lo, hi = c["beta_range_K"]
    assert 4.0e4 < lo <= hi < 5.0e4
    assert 1.0 / c["refused_coefficients"][1] > 0  # the right sign: NTC passed
    assert "config_equivalent" not in c and "coefficients" not in c
    validate_record(res)
    assert not (tmp_path / "T1_report.md").exists()


def test_the_slope_bound_holds_every_calibration_in_the_sheet():
    """FP07_BETA_K is the sheet's measured envelope: inside it, and not looser than 1%."""
    reg = registry.load()
    lo, hi = np.inf, -np.inf
    for c in reg.calibrations:
        ends = [registry._L_at(c.poly, T) for T in c.valid_T]
        assert None not in ends, c.key
        L = np.linspace(min(ends), max(ends), 201)
        d = sum(i * a * L ** (i - 1) for i, a in enumerate(c.poly) if i)
        beta = 1.0 / d
        lo, hi = min(lo, float(beta.min())), max(hi, float(beta.max()))
    assert FP07_BETA_K[0] <= lo and hi <= FP07_BETA_K[1], (lo, hi)
    assert lo < FP07_BETA_K[0] * 1.01 and hi > FP07_BETA_K[1] / 1.01, (lo, hi)


def test_fp07_slope_judges_the_local_beta_over_the_fits_range():
    pairs, _minority, _truth = _bimodal_pairs(dT=0.0)
    good = fit_calibration(pairs, order=2)
    ok, why, span = fp07_slope(good)
    assert ok and why == "" and 3000.0 < span[0] <= span[1] < 3200.0
    for factor, expect in ((1.05, True), (1.25, False), (1 / 1.25, False)):
        f = copy.copy(good)
        f.coeffs = good.coeffs * np.array([1.0, 1 / factor, 1 / factor])  # beta x factor
        assert fp07_slope(f)[0] is expect, factor


# ---------------------------------- #248: bad buffers are not thermistor samples
VMP_P = Path(__file__).parent / "data" / "SN479_0006.p"  # VMP-250IR v6.1, T1/T1_dT1, T2/T2_dT2


def _short_record(raw: bytes, k: int, keep: int = 33) -> bytes:
    """Data record *k* (0-based) keeps only its first *keep* matrices: a mis-framed
    record, which PFile rebuilds by interpolation and reports as a bad buffer."""
    endian = ">" if struct.unpack(">64H", raw[:128])[17] == 128 else "<"
    w = struct.unpack(f"{endian}64H", raw[:128])
    header, first, record, matrix = w[17], w[17] + w[11], w[18], w[30] * (w[28] + w[29]) * 2
    a = first + k * record + header + keep * matrix
    return raw[:a] + raw[first + (k + 1) * record:]


@pytest.fixture(scope="module")
def sn479(tmp_path_factory):
    """SN479_0006 as recorded, and with data record 100 mis-framed."""
    damaged = tmp_path_factory.mktemp("p248") / "damaged.p"
    damaged.write_bytes(_short_record(VMP_P.read_bytes(), 100))
    clean = load_probe_series(VMP_P, profiles=[], speed_var=None)
    with pytest.warns(UserWarning, match="mis-framed"):
        hit = load_probe_series(damaged, profiles=[], speed_var=None)
    pf = PFile(VMP_P)
    per = int(pf.samples_per_record) // round(float(pf.fs_fast) / float(pf.fs_slow))
    return clean, hit, per, float(pf.fs_slow)


def test_a_clean_file_blanks_nothing(sn479):
    clean, _hit, _per, _fs = sn479
    assert clean.bad_buffer == {}
    assert all(np.isfinite(c).all() for c in clean.counts.values())


def test_the_rebuilt_record_is_blanked_with_the_deconvolution_smear(sn479):
    """The slot PFile rebuilt is NaN, and so is the 3 diff_gain the deconvolution
    smears it forward (chi's mask), and nothing else."""
    clean, hit, per, fs = sn479
    for ch in ("T1", "T2"):
        c = hit.counts[ch]
        bad = np.flatnonzero(np.isnan(c))
        assert hit.bad_buffer[ch] == bad.size
        assert bad[0] == 100 * per - 1  # the slow sample whose stencil reaches the slot
        assert np.array_equal(bad, np.arange(bad[0], bad[-1] + 1))  # one run
        smear_s = (bad[-1] + 1 - 101 * per) / fs
        assert 2.5 < smear_s < 3.5, smear_s  # 3 x diff_gain (~1 s)
        # Elsewhere the counts are the clean file's, but for the deconvolution's
        # global regression, which the rebuilt slot moves by a fraction of a count.
        rest = np.ones(c.size, dtype=bool)
        rest[bad[0] - int(5 * fs): bad[-1] + 1 + int(5 * fs)] = False
        np.testing.assert_allclose(c[rest], clean.counts[ch][rest], rtol=0, atol=1.0)


def test_the_blanked_samples_are_counted_in_the_record_and_report(tmp_path, monkeypatch):
    """As load_probe_series leaves them: NaN counts, and their number per channel."""
    import odas_tpw.fp07cal.cli as cli

    probes, ref, _truth = _dep()
    p = probes[1]
    i = np.arange(p.time.size)
    p.counts["T1"] = np.where((i >= i.size // 2) & (i < i.size // 2 + 64), np.nan, p.counts["T1"])
    p.bad_buffer = {"T1": 64}
    cfg = _cfg()
    res = run_calibration(probes, ref, cfg, tmp_path, make_figure=False)
    e = res["channels"]["T1"]
    assert e["bad_buffer_samples"] == {"n_samples": 64, "n_files": 1, "files": [p.label]}
    assert e["thermistor_gaps"]["n_nonfinite_samples"] == 64  # the #238 gap path took them
    assert "RDL bad buffers" in (tmp_path / "T1_report.md").read_text(encoding="utf-8")
    # ...and the streaming pass over every file counts them too.
    monkeypatch.setattr(cli, "_stream", lambda paths, kw, failures=None: iter(probes))
    out = cli._stream_stats([None] * len(probes), ref, cfg, res, tmp_path)
    assert out["channels"]["T1"]["bad_buffer_samples"] == 64


def _sentinels(raw: bytes, name: str, k: int, n: int = 64) -> bytes:
    """*n* consecutive samples of channel *name* in data record *k* (0-based) replaced
    by the RDL bad-buffer sentinel, as a v6.1+ logger writes a missing buffer."""
    from odas_tpw.rsi.p_file import BAD_BUFFER_SENTINEL

    pf = PFile(VMP_P)
    endian = ">" if struct.unpack(">64H", raw[:128])[17] == 128 else "<"
    w = struct.unpack(f"{endian}64H", raw[:128])
    header, first, record = w[17], w[17] + w[11], w[18]
    cid = next(int(str(c["id"]).replace(",", " ").split()[0])
               for c in pf.config["channels"] if c.get("name") == name)
    n_rows, n_cols = pf.matrix.shape
    n_mat = (record - header) // 2 // (n_rows * n_cols)
    rows, cols = np.where(pf.matrix == cid)
    if np.all(pf.matrix[:, cols[0]] == cid):  # a fast channel: its whole column
        slots = [(m * n_rows + r) * n_cols + int(cols[0])
                 for m in range(n_mat) for r in range(n_rows)][n_mat:n_mat + n]
    else:  # a slow one: its first cell, once per matrix
        slots = [(m * n_rows + int(rows[0])) * n_cols + int(cols[0]) for m in range(n)]
    out = bytearray(raw)
    for j in slots:
        struct.pack_into(f"{endian}h", out, first + k * record + header + 2 * j,
                         BAD_BUFFER_SENTINEL)
    return bytes(out)


@pytest.mark.parametrize("name", ["T1_dT1", "T1"])
def test_a_marker_run_on_a_deconvolved_thermistor_drops_it_from_the_file(
        name, sn479, tmp_path):
    """PFile deconvolves T1 from T1 and T1_dT1 with two regressions over the WHOLE
    file, which one sentinel run moves (#259): injected here, T1 more than 20 s away
    read -8..+57 mK (in T1_dT1) and -234..-64 mK (in T1), 5-95%. The 3 diff_gain
    mask removes only the local part, so the channel is left out of the file."""
    clean = sn479[0]
    hit_path = tmp_path / "markers.p"
    hit_path.write_bytes(_sentinels(VMP_P.read_bytes(), name, 200))
    with pytest.warns(UserWarning, match="bad-buffer markers"):
        hit = load_probe_series(hit_path, profiles=[], speed_var=None)
    assert name in PFile(hit_path).bad_buffer_report["confirmed"]
    assert hit.bad_buffer_dropped == ["T1"]
    assert np.isnan(hit.counts["T1"]).all() and hit.bad_buffer["T1"] == hit.time.size
    np.testing.assert_array_equal(hit.counts["T2"], clean.counts["T2"])  # its own pair


def test_a_rebuilt_record_stays_local(sn479):
    """Framing repair interpolates BEFORE the deconvolution: nothing is dropped whole."""
    assert sn479[1].bad_buffer_dropped == []


def test_a_dropped_file_is_named_in_the_record_and_report(tmp_path):
    probes, ref, _truth = _dep()
    p = probes[1]
    p.counts["T1"] = np.full(p.time.size, np.nan)
    p.bad_buffer, p.bad_buffer_dropped = {"T1": p.time.size}, ["T1"]
    c = run_calibration(probes, ref, _cfg(), tmp_path, make_figure=False)["channels"]["T1"]
    b = c["bad_buffer_samples"]
    assert b["n_files_dropped"] == 1 and b["files_dropped"] == [p.label]
    assert c["span"]["n_files"] == len(probes) - 1
    report = (tmp_path / "T1_report.md").read_text(encoding="utf-8")
    assert "left out of each such file whole" in report and "header word 16" in report


# ---------------------------------- #249: each channel's own span in the record
def _dead_t2(tmp_path):
    """Two beads, four files; T2 on the ADC rail through the last file (as SUNRISE 2021
    Pelican's T1593 through SR1P_516)."""
    probes, ref, _truth = make_deployment(SynthConfig(
        n_yos=12, yo_seconds=1200.0, fs=4.0, ct_every_n=1, files_per_deployment=4))
    for p in probes:
        p.instrument_sn, p.probe_sn = "7", {"T1": "T100", "T2": "T101"}
    probes[-1].counts["T2"] = np.full(probes[-1].counts["T2"].size, 32767.0)
    cfg = {**_cfg(order="auto"), "channels": ["T1", "T2"]}
    return probes, run_calibration(probes, ref, cfg, tmp_path, make_figure=False)


def test_a_live_bead_without_reference_is_still_in_service(tmp_path):
    """The span is when the bead was ALIVE (the PI's decision, 2026-10-29), not the
    files that gave it pairs: here the reference stops before the last file."""
    probes, ref, _truth = _dep()
    keep = ref.time < probes[-1].time[0]
    ref.time, ref.value, ref.pressure = ref.time[keep], ref.value[keep], ref.pressure[keep]
    c = run_calibration(probes, ref, _cfg(), tmp_path, make_figure=False)["channels"]["T1"]
    s = c["span"]
    assert s["n_files"] == 4 and s["time_end"] == float(probes[-1].time[-1])
    assert s["last_kept_pair"] < float(probes[-1].time[0])  # the calibration's support


def test_the_alive_rule_is_fp07_pairs_dead_bead():
    """A window is dead when its counts do not move or touch the ADC rail; a file is
    dead when at least half its windows are (fp07_bead_tau.DEAD_FILE_FRAC), or when
    it spans less than ~10 mK: a test probe reading its T_0 with electronics noise."""
    from odas_tpw.fp07cal.logr import log_r
    from odas_tpw.fp07cal.series import ALIVE_MIN_RANGE_L, ALIVE_WINDOW_S, thermistor_alive

    probes, _ref, _truth = _dep()
    p = probes[0]
    c, br, fs = p.counts["T1"], p.bridge["T1"], p.fs
    assert thermistor_alive(c, br, fs)
    assert not thermistor_alive(np.full(c.size, c[0]), br, fs)  # one value
    assert not thermistor_alive(np.full(c.size, 32767.0), br, fs)  # on the rail
    assert not thermistor_alive(np.full(c.size, np.nan), br, fs)  # nothing recorded
    # A test probe: SN465's read 16.145 degC with 0.15-0.22 mK of noise (0.6-1.0 mK
    # 1-99%), about a tenth of a count; the bound is 4e-4 in L, ~10 mK.
    rng = np.random.default_rng(1)
    probe = c[0] + rng.normal(0.0, 0.1, c.size)
    L = log_r(probe, br)[0]
    assert np.ptp(np.percentile(L, [1, 99])) < ALIVE_MIN_RANGE_L
    assert not thermistor_alive(probe, br, fs)
    w = round(ALIVE_WINDOW_S * fs)
    n_win = -(-c.size // w)
    half = np.array(c, copy=True)
    half[: (n_win // 2) * w] = c[0]  # dead in the first half of its windows (rounded down)
    assert thermistor_alive(half, br, fs) == (n_win // 2 < 0.5 * n_win)
    half[: (n_win // 2 + 1) * w] = c[0]
    assert not thermistor_alive(half, br, fs)


def _utc(t: float) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_each_channel_records_when_its_bead_was_alive(tmp_path):
    probes, res = _dead_t2(tmp_path)
    assert res["schema"] == "fp07-cal/2"
    t1, t2 = res["channels"]["T1"], res["channels"]["T2"]
    assert t1["span"]["n_files"] == 4 and t2["span"]["n_files"] == 3
    assert t1["span"]["n_files_dead"] == 0 and t2["span"]["n_files_dead"] == 1
    assert t1["span"]["time_start"] == t2["span"]["time_start"] == res["time_start"]
    assert t1["span"]["time_end"] == res["time_end"] == float(probes[3].time[-1])
    assert t2["span"]["time_end"] == float(probes[2].time[-1]) < res["time_end"]
    for s in (t1["span"], t2["span"]):
        assert s["time_start"] <= s["first_kept_pair"] < s["last_kept_pair"] <= s["time_end"]
    validate_record(res)


def test_registry_add_gives_a_bead_its_own_span(tmp_path):
    probes, res = _dead_t2(tmp_path)
    def add(record, ch):
        n_blocks = record["channels"][ch]["stability"]["n_blocks"]
        return registry.entry_from_record(
            record, ch, deployment="Synthetic", instrument_model="vmp-250",
            method="vmp-own-jac", reference_kind="JAC CT", reference_serial="23",
            reference_cal_date="2016-02-15", reference_pole_s=0.7,
            block_offsets_K=[0.0] * n_blocks)

    e2 = add(res, "T2")
    assert e2["first"] == _utc(res["time_start"])
    assert e2["last"] == _utc(probes[2].time[-1]) and e2["fit"]["n_files"] == 3
    e1 = add(res, "T1")
    assert e1["last"] == _utc(probes[3].time[-1]) and e1["fit"]["n_files"] == 4
    # An fp07-cal/1 record has only the run's span, and still adds with it...
    old = copy.deepcopy(res)
    old["schema"] = "fp07-cal/1"
    for c in old["channels"].values():
        del c["span"]
    validate_record(old)
    e_old = add(old, "T2")
    assert e_old["last"] == _utc(res["time_end"]) and e_old["fit"]["n_files"] == len(probes)
    # ...but an fp07-cal/2 record without one has been edited.
    old["schema"] = "fp07-cal/2"
    with pytest.raises(registry.RegistryError, match="no per-channel span"):
        add(old, "T2")
    bad = copy.deepcopy(res)
    bad["channels"]["T2"]["span"]["n_files"] = 0
    with pytest.raises(registry.RegistryError, match="n_files"):
        add(bad, "T2")


@pytest.mark.parametrize("stream", [True, False])
def test_fit_takes_the_span_from_every_file_not_the_subsample(stream, tmp_path, monkeypatch):
    """The fit holds a subsample; its stride can end a dying bead's span a file early.
    Here it holds files 0 and 1 of 4, and T2 dies in file 3: the streaming pass over
    every file finds T2's pairs through file 2."""
    import odas_tpw.fp07cal.cli as cli

    probes, ref, _truth = make_deployment(SynthConfig(
        n_yos=12, yo_seconds=1200.0, fs=4.0, ct_every_n=1, files_per_deployment=4))
    probes[-1].counts["T2"] = np.full(probes[-1].counts["T2"].size, 32767.0)
    paths = [tmp_path / p.label for p in probes]
    monkeypatch.setattr(cli, "_gather_paths", lambda cfg: paths)
    monkeypatch.setattr(cli, "_load_reference", lambda cfg: ref)
    monkeypatch.setattr(cli, "_load_some", lambda paths, limit, kw=None: (probes[:2], 0))
    monkeypatch.setattr(cli, "_stream", lambda paths, kw, failures=None: iter(probes))
    cfg = tmp_path / "fp07-cal.yaml"
    cfg.write_text(
        "files: {output_dir: '<CONFIG_DIR>/out', max_fit_files: 2}\n"
        "channels: [T1, T2]\n"
        "pairs: {max_gap: 30.0}\nlag: {max_lag: 6.0, step: 0.5}\n"
        "fit: {order: 1, geometry: false}\nstability: {n_blocks: 2}\n", encoding="utf-8")
    argv = ["fit", "-c", str(cfg), "--no-figure"] + ([] if stream else ["--no-stream"])
    assert cli.main(argv) == 0
    rec = json.loads((tmp_path / "out" / "coefficients.json").read_text(encoding="utf-8"))
    s1, s2 = rec["channels"]["T1"]["span"], rec["channels"]["T2"]["span"]
    if stream:
        assert s1["from"] == s2["from"] == "every file"
        assert s1["n_files"] == 4 and s1["time_end"] == float(probes[3].time[-1])
        assert s2["n_files"] == 3 and s2["time_end"] == float(probes[2].time[-1])
        assert "spans" not in rec["per_profile"]
    else:
        assert s2["from"] == "2 fit files of 4"
        assert s2["n_files"] == 2 and s2["time_end"] == float(probes[1].time[-1])


def test_the_reference_recipe_masks_pressure_with_its_partner(tmp_path):
    """Runbook section 8 blanks P where P_dP dropped out too, over the deconvolution's
    3 diff_gain, as chi masks it; a clean channel gets an all-False mask."""
    from odas_tpw.fp07cal.series import bad_buffer_mask
    from odas_tpw.rsi.channels import config_number

    hit = tmp_path / "pdp.p"
    hit.write_bytes(_sentinels(VMP_P.read_bytes(), "P_dP", 200))
    with pytest.warns(UserWarning, match="bad-buffer markers"):
        pf = PFile(hit)
    gain = config_number(next(c for c in pf.config["channels"] if c.get("name") == "P_dP")
                         ["diff_gain"], key="diff_gain", channel="P_dP")
    bad = bad_buffer_mask(pf, "P")
    start, n = pf.bad_buffer_report["confirmed"]["P_dP"]["spans"][0]
    run = np.flatnonzero(bad)
    assert run[0] <= start and run[-1] + 1 >= start + n + round(3 * gain * pf.fs_slow) - 1
    assert not bad_buffer_mask(pf, "JAC_T").any()
