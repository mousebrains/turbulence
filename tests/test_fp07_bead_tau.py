# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""``fp07-pair tau`` (odas_tpw.chi.fp07_bead_tau) and the sheet's ``response:`` block.

The estimator is checked on synthetic chi products with known single-pole
beads: it recovers them, it refuses matched beads and a failing bead, and it
keeps to its span and instrument. ``fp07-pair fit`` is pinned to the output it
gave before this module existed (a golden report and correction factors
computed with main's ``fp07_pair.py``). The sheet side checks the strict
loader, the rendering and the text insertion of ``registry response``.
"""

import copy
import csv
import io
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from odas_tpw.chi import fp07_bead_tau as B
from odas_tpw.chi import fp07_pair as P
from odas_tpw.fp07cal import registry as reg

REPO_ROOT = Path(__file__).resolve().parents[1]
SHEET = REPO_ROOT / "microstructure_sensors" / "fp07_calibrations.yaml"
T0 = datetime(2022, 3, 21, tzinfo=UTC).timestamp()
F = np.arange(0, 129) * 1.0


# ------------------------------------------------------------ synthetics
def _source(path, sn1="T1592", sn2="T2005", inst=194):
    import netCDF4

    cfg = (f"[instrument_info]\nsn = {inst}\n"
           f"[channel]\nname = T1\ntype = therm\nid = 4\nsn = {sn1}\n"
           f"[channel]\nname = T2\ntype = therm\nid = 5\nsn = {sn2}\n")
    with netCDF4.Dataset(path, "w") as ds:
        ds.configuration_string = cfg
    return str(path)


def _chi(path, source, t0, taus, rng, *, n_win=40, W_med=0.95, gain2=1.1, snr=300.0,
         dead2=False):
    """One synthetic chi product: single-pole beads *taus* [s] at W_med, tau ~ W^-1/2.

    *dead2*: probe 2's spectrum identically zero (a dead or railed channel).
    """
    import netCDF4

    W = W_med * np.exp(rng.normal(0, 0.05, n_win))
    kB = np.exp(rng.normal(np.log(70.0), 0.5, n_win))
    lvl = np.exp(rng.normal(0, 1.0, n_win))
    x = np.maximum(F[:, None] / W[None, :] / kB[None, :], 1e-6)
    g = lvl * x * np.exp(-np.sqrt(6 * 5.26) * x) * (1 + 3 * (0.03 / np.maximum(x, 1e-4)) ** 1.5)
    nz = np.max(g, axis=0) / snr * np.exp(rng.normal(0, 1.0, n_win))
    s = (W / W_med) ** -0.5
    S = np.empty((2, F.size, n_win))
    for k, (tau, gain) in enumerate(((taus[0], 1.0), (taus[1], gain2))):
        h = 1 / (1 + (2 * np.pi * F[:, None] * tau * s[None, :]) ** 2)
        S[k] = (gain * g * h + nz) * rng.chisquare(9.5, (F.size, n_win)) / 9.5
    if dead2:
        S[1] = 0.0
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("probe", 2)
        ds.createDimension("freq", F.size)
        ds.createDimension("time", n_win)
        ds.createVariable("spec_gradT", "f8", ("probe", "freq", "time"))[:] = S
        ds.createVariable("spec_noise", "f8", ("probe", "freq", "time"))[:] = np.broadcast_to(
            nz, (2, F.size, n_win))
        ds.createVariable("F", "f8", ("freq", "time"))[:] = np.tile(F[:, None], (1, n_win))
        ds.createVariable("speed", "f8", ("time",))[:] = W
        pv = ds.createVariable("probe", str, ("probe",))
        pv[0], pv[1] = "T1_dT1", "T2_dT2"
        ds.source = source
        ds.start_time = datetime.fromtimestamp(t0, UTC).isoformat()
        ds.fp07_model = "single_pole"
    return str(path)


def _deploy(d, taus, *, n=24, seed=0, per_start=3, step=3600.0, t0=T0, taus2=None, **kw):
    """*n* products, *per_start* sharing each start_time; *taus2* for the second half."""
    rng = np.random.default_rng(seed)
    src = _source(Path(d) / f"src{seed}.nc")
    return [_chi(Path(d) / f"c{seed}_{i:03d}.nc", src, t0 + step * (i // per_start),
                 taus2 if (taus2 is not None and i >= n // 2) else taus, rng, **kw)
            for i in range(n)]


def _measure(paths, **kw):
    kw.setdefault("n_boot", 40)
    return B.measure(paths, instrument="194", first=T0 - 10, last=T0 + 1e8, **kw)


# ------------------------------------------------------------- the fit
def test_fit_single_pole_recovers_noise_free_taus():
    x = P._grid(25.0)
    x = x[x >= 1.0]
    for t1, t2 in ((0.013, 0.0195), (0.025, 0.010), (0.006, 0.004)):
        y = 0.2 + np.log(B.h2(x, t2)) - np.log(B.h2(x, t1))
        lg, a, b, rms = B.fit_single_pole(x, y)
        assert (a, b) == pytest.approx((t1, t2), rel=1e-3)
        assert lg == pytest.approx(0.2, abs=1e-3) and rms < 1e-4


def test_split_halves_never_splits_a_shared_start_time():
    assert B.split_halves([0, 0, 0, 1, 1, 1]) == 3
    assert B.split_halves([0, 0, 0, 0, 1, 2]) == 4  # nearest the middle between distinct times
    assert B.split_halves([5, 5, 5]) is None
    assert B.split_halves([5.2, 5.7, 9.0]) == 2  # one printed second is one time


def test_noise_check_reads_the_quietest_windows():
    rng = np.random.default_rng(1)
    quiet = rng.uniform(10, 100, (2, 500))  # S/N over 1-10 Hz per window
    high = np.where(quiet < 25, 1.0, 5.0)  # signal left at 60-100 Hz except in quiet windows
    high[1] *= np.where(quiet[1] < 25, 1.6, 1.0)  # probe 2's model 1.6x too low
    # dead windows (zero spectrum) are not "quiet": they must not drag the check to 0
    quiet = np.concatenate([quiet, np.zeros((2, 100))], axis=1)
    high = np.concatenate([high, np.zeros((2, 100))], axis=1)
    obs = B._noise_check(quiet, high)
    assert obs[0] == pytest.approx(1.0) and obs[1] == pytest.approx(1.6)
    assert B._noise_check(quiet[:, :5], high[:, :5]) == [None, None]  # too few windows


def test_drift_is_a_two_sigma_test_in_ln_tau():
    def fit(t1, ci):
        return B.TauFit(first=0, last=1, n_files=20, n_recordings=10, n_windows=100,
                        identified=True, reasons=(),
                        tau=(t1, 0.02), tau_fit=(t1, 0.02), tau_ci=(ci, (0.019, 0.021)),
                        ln_G_fit=0, ln_G=0, rms=0, fit_band=(1, 25), n_bins=25,
                        ln_ratio={}, ln_ratio_ci={}, n_boot=10, n_boot_ok=10)
    a = fit(0.010, (0.0095, 0.0105))  # half-width ~0.05 in ln
    assert B.drifted(a, fit(0.0116, (0.0110, 0.0122)), 0)  # +15%: beyond 2 x 0.07
    assert not B.drifted(a, fit(0.0108, (0.0103, 0.0113)), 0)  # +8%: within
    assert not B.drifted(a, fit(0.0116, (0.0110, 0.0122)), 1)  # the other bead held still


def test_a_ratio_shift_between_halves_is_model_free():
    # osu685 2025: the pair's 20 Hz ratio went 1.23 -> 0.81 while each tau's
    # interval was too wide for `drift` to say which bead moved
    def half(r20, hw):
        return B.TauFit(first=0, last=1, n_files=20, n_recordings=10, n_windows=100,
                        identified=False, reasons=(), tau=None, tau_fit=(0.01, 0.01),
                        tau_ci=((0.008, 0.012), (0.008, 0.012)), ln_G_fit=0, ln_G=0, rms=0,
                        fit_band=(1, 25), n_bins=25, ln_ratio={20.0: r20},
                        ln_ratio_ci={20.0: (r20 - hw, r20 + hw)}, n_boot=10, n_boot_ok=10)
    assert B.ratio_shifted(half(np.log(1.23), 0.02), half(np.log(0.81), 0.04))
    assert not B.ratio_shifted(half(0.20, 0.05), half(0.25, 0.05))
    assert not B.ratio_shifted(half(0.20, 0.05), half(np.nan, 0.05))
    few = half(np.log(0.81), 0.04)
    few.n_recordings = 3  # too few to test against
    assert not B.ratio_shifted(half(np.log(1.23), 0.02), few)


# ------------------------------------------------------- the estimator
def test_recovers_a_mismatched_pair(tmp_path):
    res = _measure(_deploy(tmp_path, (0.013, 0.0195), n=36, n_win=60))
    (pt,) = res.pairs
    f = pt.all
    assert f.identified, f.reasons
    assert pt.serials == ("T1592", "T2005") and pt.tau_model == "lueck"
    assert f.tau == pytest.approx((0.013, 0.0195), rel=0.15)
    for k, truth in enumerate((0.013, 0.0195)):
        lo, hi = f.tau_ci[k]
        assert lo < hi and hi / lo < B.CI_MAX_FACTOR
    # the model-free ratio: truth relative to the reference bins
    lnr20 = np.log(B.h2(20.0, 0.0195) / B.h2(20.0, 0.013))
    assert f.ln_ratio[20.0] == pytest.approx(lnr20 - np.log(B.h2(2.5, 0.0195) / B.h2(2.5, 0.013)),
                                             abs=0.06)
    assert pt.halves is not None and sum(h.n_files for h in pt.halves) == f.n_files
    assert not pt.flags


def test_matched_beads_give_no_tau(tmp_path):
    res = _measure(_deploy(tmp_path, (0.010, 0.010), n=36, n_win=60))
    (pt,) = res.pairs
    assert not pt.all.identified and pt.all.tau is None
    assert any("not distinguishable" in r for r in pt.all.reasons)
    assert np.isfinite(pt.all.ln_ratio[20.0])  # the ratio is still measured
    assert pt.to_dict()["all"]["tau_s"] is None


def test_a_failing_bead_is_not_a_time_constant(tmp_path):
    res = _measure(_deploy(tmp_path, (0.013, 0.0195), n=24, gain2=0.3))
    (pt,) = res.pairs
    assert "amplitude" in pt.flags and not pt.all.identified
    assert any("amplitude-loss" in r for r in pt.all.reasons)


def test_a_dead_bead_is_left_out_flagged_and_refused(tmp_path):
    # SUNRISE 2021 Pelican: T2's spectrum was identically zero through the last
    # file before that bead was replaced. Those files carry no ratio, and the
    # pair is no tau measurement.
    good = _deploy(tmp_path, (0.013, 0.0195), n=30, n_win=50)
    src = _source(tmp_path / "src_dead.nc")
    rng = np.random.default_rng(4)
    dead = [_chi(tmp_path / f"d{i}.nc", src, T0 + 30 * 86400, (0.013, 0.0195), rng, dead2=True)
            for i in range(3)]
    (pt,) = B.measure(good + dead, instrument="194", first=T0 - 10, last=T0 + 1e8,
                      n_boot=20).pairs
    assert "dead-bead" in pt.flags and pt.dead_files == ([], ["d0.nc", "d1.nc", "d2.nc"])
    assert pt.all.n_files == 30 and not pt.all.identified and pt.all.tau is None
    assert any("T2005 (T2_dT2) dead in 3 chi file(s)" in r for r in pt.all.reasons)
    # the dead files came last: only the second half is refused for them
    h1, h2 = pt.halves
    assert any("dead in 3" in r for r in h2.reasons) and not h2.identified
    assert not any("dead in" in r for r in h1.reasons)
    assert pt.to_dict()["dead_files"] == [[], ["d0.nc", "d1.nc", "d2.nc"]]
    assert np.isfinite(pt.all.ln_ratio[20.0])  # the ratio of the files kept is still recorded
    # the same products without the dead files identify the pair
    (ok,) = B.measure(good, instrument="194", first=T0 - 10, last=T0 + 1e8, n_boot=20).pairs
    assert ok.all.identified and "dead-bead" not in ok.flags


def test_too_few_files_is_refused(tmp_path):
    res = _measure(_deploy(tmp_path, (0.013, 0.0195), n=6, per_start=1, n_win=80))
    (pt,) = res.pairs
    assert not pt.all.identified
    assert any("chi files <" in r for r in pt.all.reasons)


def test_too_few_recordings_is_refused(tmp_path):
    # 24 profiles cut from 3 .p files are 3 recordings: the bootstrap resamples
    # recordings, and 3 say little however many profiles they hold
    (pt,) = _measure(_deploy(tmp_path, (0.013, 0.0195), n=24, per_start=8)).pairs
    assert pt.all.n_recordings == 3 and not pt.all.identified
    assert any("3 recordings" in r for r in pt.all.reasons)


def test_instrument_serials_are_compared_as_numbers(tmp_path):
    src = _source(tmp_path / "s0194.nc", inst="0194")
    rng = np.random.default_rng(2)
    paths = [_chi(tmp_path / f"z{i}.nc", src, T0 + 3600 * i, (0.013, 0.0195), rng)
             for i in range(3)]
    for inst in ("194", "SN194", "sn 0194"):
        (pt,) = B.measure(paths, instrument=inst, first=T0 - 1, last=T0 + 1e6, n_boot=2).pairs
        assert pt.instrument == "SN194"
    assert B.canonical_instrument("MR046") == "SNMR046"  # not a number: compared as text
    assert B.measure(paths, instrument="MR046", first=T0 - 1, last=T0 + 1e6).pairs == []


def test_fit_points_never_bridge_a_hole_in_the_band():
    grid = P._grid(100.0)
    curve = np.where((grid >= 1) & (grid <= 12), 0.1, np.nan)
    curve[np.argmin(abs(grid - 24.0))] = 0.3  # one lone bin far beyond a 12 Hz hole
    fx = B.fit_points((1.0, 25.0))
    y = B._curve_at(curve, grid, fx)
    assert np.isfinite(y[(fx >= 2) & (fx <= 11)]).all()
    assert np.isnan(y[(fx >= 13) & (fx <= 23)]).all()  # inside the hole: not measured
    # 2 Hz FFT bins (SUNRISE) are bridged
    sparse = np.full(grid.size, np.nan)
    for f0 in np.arange(2.0, 26.0, 2.0):
        sparse[np.argmin(abs(grid - f0))] = 0.0
    assert np.isfinite(B._curve_at(sparse, grid, np.arange(3.0, 24.0))).all()


def test_span_and_instrument_select_the_files(tmp_path):
    a = _deploy(tmp_path, (0.013, 0.0195), n=12, seed=1)
    late = _deploy(tmp_path, (0.013, 0.0195), n=6, seed=2, t0=T0 + 30 * 86400)
    other = [_chi(tmp_path / "o.nc", _source(tmp_path / "osrc.nc", inst=428), T0 + 60,
                  (0.01, 0.01), np.random.default_rng(3))]
    res = B.measure(a + late + other, instrument="SN194", first=T0 - 10, last=T0 + 86400,
                    n_boot=10)
    (pt,) = res.pairs
    assert pt.all.n_files == 12 and pt.instrument == "SN194"
    assert res.report[0].startswith("19 chi files: 6 outside")
    assert "1 on other instruments" in res.report[0]
    assert B.measure(a, instrument="194", first=T0 - 10, last=T0 + 1e8, beads=["T9999"],
                     n_boot=5).pairs == []


def test_the_source_is_read_once_per_recording(tmp_path, monkeypatch):
    paths = _deploy(tmp_path, (0.013, 0.0195), n=12, per_start=4)  # 3 recordings
    calls = []
    real = P.identity_from_source
    monkeypatch.setattr(P, "identity_from_source",
                        lambda *a, **k: calls.append(a[0]) or real(*a, **k))
    (pt,) = _measure(paths, n_boot=5).pairs
    assert pt.all.n_files == 12 and len(calls) == 3


def test_drift_between_halves_is_flagged(tmp_path):
    res = _measure(_deploy(tmp_path, (0.010, 0.016), taus2=(0.018, 0.025), n=40, n_win=60))
    (pt,) = res.pairs
    h1, h2 = pt.halves
    assert h1.identified and h2.identified
    assert h1.tau[0] < h2.tau[0] and h1.tau[1] < h2.tau[1]
    assert "drift" in pt.flags and pt.drift[0]  # T1: 10 -> 18 ms, far beyond 2 sigma


def test_record_round_trip_and_cli(tmp_path):
    from odas_tpw.chi.fp07_pair_cli import main

    _deploy(tmp_path, (0.013, 0.0195), n=24)
    out = tmp_path / "out"
    rc = main(["tau", str(tmp_path / "c0_*.nc"), "--instrument", "194",
               "--first", "2022-03-20T00:00:00Z", "--last", "2022-03-25T00:00:00Z",
               "-o", str(out), "--n-boot", "20"])
    assert rc == 0
    (js,) = out.glob("fp07tau_SN194_T1592-T2005_*.json")
    rec = json.loads(js.read_text(encoding="utf-8"))
    assert rec["format"] == reg.RESPONSE_FORMAT and rec["method"] == "pair-ratio-sp"
    assert rec["speed_scaling"] == "lueck" and rec["band_Hz"] == [1.0, 25.0]
    assert js.with_suffix(".txt").read_text(encoding="utf-8").startswith("SN194 T1_dT1=T1592")
    assert main(["tau", str(tmp_path / "c0_*.nc"), "--instrument", "428", "--first",
                 "2022-03-20T00:00:00Z", "--last", "2022-03-25T00:00:00Z", "-o",
                 str(out)]) == 1


# --------------------------------------------- fp07-pair fit is unchanged
# Report lines and correction factors from `estimate` with main's fp07_pair.py
# (commit 2e1b660) on _deploy(..., seed=11) below, before this module existed.
# The report's numbers are rounded to 3 decimals, so they are platform-proof.
_GOLDEN_REPORT = [
    "30 chi files: 30 read, 0 skipped, 1 bead pair(s)",
    "SN194 T1_dT1=T1592 T2_dT2=T2005: 30 files, W_ref 0.947 m/s, 1 segment(s)",
    "  SN194 T1_dT1=T1592 T2_dT2=T2005 [2022-03-21T00:00:00+00:00..2022-03-21T14:00:00+00:00]: "
    "30 files, 1200 windows; band 1.0-47.2 Hz; ln G +0.061 (low-band slope +0.004); "
    "ln R' at 20 Hz +0.572; per-file mid-band spread 0.099; flags -",
]
_GOLDEN_FACTORS = [[
    1.043635463681836, 0.9352813924197867, 0.8537886599466638, 0.7517682553139281,
    0.7053154268965084, 0.9581889795811513, 1.0691969369910925, 1.1712500375238881,
    1.3301971623986886, 1.417805370286805,
]]


def test_fp07_pair_fit_output_is_unchanged(tmp_path):
    paths = _deploy(tmp_path, (0.015, 0.010), n=30, seed=11, per_start=2)
    res = P.estimate(paths)
    assert res.report == _GOLDEN_REPORT
    got = [[float(c.factor(k, np.array([f0]), 1.0)[0]) for k in (0, 1) for f0 in (2, 5, 10, 20, 40)]
           for c in res.corrections]
    np.testing.assert_allclose(got, _GOLDEN_FACTORS, rtol=1e-9)


# ------------------------------------------------- the sheet's response block
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_fp07_registry import _entry, _sheet  # noqa: E402


def _resp(**over):
    r = {
        "method": "pair-ratio-sp", "partner": "T200",
        "first": "2024-01-02T00:00:00Z", "last": "2024-01-09T00:00:00Z",
        "n_chi_files": 40, "n_recordings": 8, "n_windows": 2000, "W_ref_m_s": 0.95,
        "speed_scaling": "lueck",
        "band_Hz": [1.0, 25.0], "identified": True, "tau_ms": 13.0, "tau_ms_16_84": [12.0, 14.0],
        "halves": {
            "first": {"first": "2024-01-02T00:00:00Z", "last": "2024-01-05T00:00:00Z",
                      "n_chi_files": 20, "tau_ms": 11.0, "tau_ms_16_84": [10.0, 12.0],
                      "power_ratio_20Hz": 1.25},
            "second": {"first": "2024-01-05T06:00:00Z", "last": "2024-01-09T00:00:00Z",
                       "n_chi_files": 20, "tau_ms": 15.0, "tau_ms_16_84": [14.0, 16.0],
                       "power_ratio_20Hz": 1.36},
        },
        "power_ratio": {10: 1.12, 20: 1.31},
        "power_ratio_16_84": {10: [1.1, 1.14], 20: [1.28, 1.35]},
        "software_version": "test",
    }
    r.update(over)
    return r


def _pair_sheet(resp=None, partner_resp=None):
    a = _entry(bead="T100", channel="T1")
    b = _entry(bead="T200", channel="T2")
    if resp is not None:
        a["response"] = resp
    if partner_resp is not None:
        b["response"] = partner_resp
    return _sheet(a, b)


def _reciprocal(r):
    q = copy.deepcopy(r)
    q["partner"] = "T100"
    q["power_ratio"] = {k: round(1 / v, 6) for k, v in r["power_ratio"].items()}
    q["power_ratio_16_84"] = {k: [round(1 / hi, 6), round(1 / lo, 6)]
                              for k, (lo, hi) in r["power_ratio_16_84"].items()}
    q["tau_ms"], q["tau_ms_16_84"] = 19.5, [18.0, 21.0]
    for h in q["halves"].values():
        h["power_ratio_20Hz"] = round(1 / h["power_ratio_20Hz"], 6)
    return q


def test_a_response_parses_and_derives_tau_at_one_meter_per_second():
    c = reg.parse(_pair_sheet(_resp())).history("T100")[0]
    r = c.response
    assert r is not None and r.identified and r.tau_ms == 13.0 and r.partner == "T200"
    assert r.tau_1mps_ms == pytest.approx(13.0 * np.sqrt(0.95))  # lueck: tau ~ W^-1/2
    assert r.halves[1].tau_ms == 15.0
    assert c.to_dict()["response"]["tau_1mps_ms_derived"] == pytest.approx(r.tau_1mps_ms)


def test_a_response_with_its_partner_reads_the_same_from_both_sides():
    r = _resp()
    reg.parse(_pair_sheet(r, _reciprocal(r)))
    bad = _reciprocal(r)
    bad["power_ratio"][20] = 1.31  # not the reciprocal
    with pytest.raises(reg.RegistryError, match="reciprocal"):
        reg.parse(_pair_sheet(r, bad))
    bad = _reciprocal(r)
    bad["n_windows"] = 1999
    with pytest.raises(reg.RegistryError, match="differs"):
        reg.parse(_pair_sheet(r, bad))


@pytest.mark.parametrize("over, err, match", [
    ({"tau": 1}, reg.UnknownKeyError, "unknown key"),
    ({"tau_ms": None}, reg.RegistryError, "together"),
    ({"tau_ms": 0.013, "tau_ms_16_84": [0.012, 0.014]}, reg.PlausibilityError, "MILLISECONDS"),
    ({"tau_ms": 200.0, "tau_ms_16_84": [150.0, 200.0]}, reg.PlausibilityError, "bound"),
    ({"identified": False}, reg.RegistryError, "records no tau"),
    ({"identified": False, "tau_ms": None, "tau_ms_16_84": None}, reg.MissingKeyError, "reason"),
    ({"reason": "x"}, reg.RegistryError, "unidentified"),
    ({"partner": "T100"}, reg.RegistryError, "own partner"),
    ({"partner": "T"}, reg.SerialPatternError, "bead serial"),
    ({"first": "2023-12-31T00:00:00Z"}, reg.DateOrderError, "inside the entry"),
    ({"method": "eyeball"}, reg.RegistryError, "pair-ratio-sp"),
    ({"speed_scaling": "nope"}, reg.RegistryError, "speed_scaling"),
    ({"W_ref_m_s": 95.0}, reg.PlausibilityError, "m/s"),
    ({"power_ratio": {10: 1.1, 30: 1.3}}, reg.UnknownKeyError, "frequencies"),
    ({"power_ratio": {10: 1.1, 20: None}}, reg.RegistryError, "without a power_ratio"),
    ({"flags": ["drifting"]}, reg.RegistryError, "unknown flag"),
    ({"flags": ["amplitude"]}, reg.RegistryError, "failing bead"),
    ({"flags": ["dead-bead"]}, reg.RegistryError, "failing bead"),
    ({"n_windows": 10}, reg.RegistryError, ">= 40"),
    ({"n_recordings": 41}, reg.RegistryError, "recordings in 40"),
])
def test_the_response_loader_is_strict(over, err, match):
    with pytest.raises(err, match=match):
        reg.parse(_pair_sheet(_resp(**over)))


def test_missing_response_key_is_refused():
    r = _resp()
    del r["halves"]
    with pytest.raises(reg.MissingKeyError, match="halves"):
        reg.parse(_pair_sheet(r))


def test_unidentified_response_keeps_its_ratios():
    r = _resp(identified=False, tau_ms=None, tau_ms_16_84=None, reason="beads matched")
    c = reg.parse(_pair_sheet(r)).history("T100")[0]
    assert c.response.tau_1mps_ms is None and c.response.power_ratio[20] == 1.31
    # one recording: a ratio, but no bootstrap interval
    r["power_ratio_16_84"] = {10: None, 20: None}
    c = reg.parse(_pair_sheet(r)).history("T100")[0]
    assert c.response.power_ratio[20] == 1.31 and c.response.power_ratio_16_84[20] is None


def test_halves_must_split_the_span():
    r = _resp()
    r["halves"]["first"]["first"] = "2024-01-03T00:00:00Z"
    with pytest.raises(reg.DateOrderError, match="split"):
        reg.parse(_pair_sheet(r))
    r = _resp()
    r["halves"]["second"]["n_chi_files"] = 21
    with pytest.raises(reg.RegistryError, match="n_chi_files"):
        reg.parse(_pair_sheet(r))
    r = _resp(flags=["drift"])
    r["halves"]["second"].update(tau_ms=None, tau_ms_16_84=None)
    with pytest.raises(reg.RegistryError, match="drift"):
        reg.parse(_pair_sheet(r))


def test_the_partner_must_be_on_this_instrument():
    a = _entry(bead="T100", channel="T1")
    a["response"] = _resp()
    b = _entry(bead="T200", channel="T2", serial="9")
    with pytest.raises(reg.DateOrderError, match="not on SN 7"):
        reg.parse(_sheet(a, b))


def test_a_response_renders_in_both_views():
    sheet = reg.parse(_pair_sheet(_resp()))
    md = reg.render_markdown(sheet)
    assert "### FP07 time constant (`response`)" in md
    assert "**The pole form is not determined**" in md
    assert "| 1 | T200 | 0.950 | 1-25 | **13.0** (12.0-14.0) | 12.7 (lueck) |" in md
    rows = list(csv.DictReader(io.StringIO(reg.render_csv(sheet))))
    (row,) = [r for r in rows if r["bead"] == "T100"]
    assert float(row["response_tau_ms"]) == 13.0
    assert float(row["response_tau_1mps_ms"]) == round(13.0 * np.sqrt(0.95), 2)
    assert row["response_partner"] == "T200" and row["response_identified"] == "true"
    (other,) = [r for r in rows if r["bead"] == "T200"]
    assert other["response_tau_ms"] == ""


# --------------------------------------------- recording a response as text
def _strip_responses(text):
    """The sheet without its `response:` blocks (as `registry response` writes them)."""
    out, skip = [], False
    for ln in text.splitlines(keepends=True):
        if ln.rstrip("\r\n") == "    response:":
            skip = True
            continue
        if skip and ln.startswith("      "):
            continue
        skip = False
        out.append(ln)
    return "".join(out)


def _sheet_copy(tmp_path):
    """The repository sheet as it was before any response was recorded."""
    dst = tmp_path / reg.YAML_NAME
    dst.write_text(_strip_responses(SHEET.read_text(encoding="utf-8")), encoding="utf-8",
                   newline="\n")
    assert all(c.response is None for c in reg.load(dst).calibrations)
    reg.write_views(dst)
    return dst


def _real_record(tmp_path, n=24, seed=5):
    """A record from synthetic products inside ARCTERX-2022 Interior's SN194 span."""
    d = tmp_path / f"prod{seed}"
    d.mkdir()
    paths = _deploy(d, (0.013, 0.0195), n=n, seed=seed, t0=T0 + 6 * 3600)
    (pt,) = B.measure(paths, instrument="194", first=T0, last=T0 + 30 * 86400, n_boot=20).pairs
    js, _ = B.write_record(pt, tmp_path / f"out{seed}")
    return json.loads(js.read_text(encoding="utf-8")), js


def _cal(sheet, bead, deployment="ARCTERX-2022 Interior"):
    (c,) = [c for c in reg.load(sheet).history(bead) if c.deployment == deployment]
    return c


def test_registry_response_records_both_beads_of_a_measurement(tmp_path):
    sheet = _sheet_copy(tmp_path)
    old = sheet.read_text(encoding="utf-8")
    rec, _ = _real_record(tmp_path)
    inserted = []
    for bead in ("T1592", "T2005"):
        block = reg.response_from_record(rec, _cal(sheet, bead), notes="synthetic")
        inserted.append(reg.add_response(sheet, bead, "ARCTERX-2022 Interior", block))
    r1, r2 = _cal(sheet, "T1592").response, _cal(sheet, "T2005").response
    assert r1.partner == "T2005" and r2.partner == "T1592"
    assert r1.tau_ms == pytest.approx(rec["all"]["tau_s"][0] * 1e3, abs=0.006)
    assert r2.tau_ms == pytest.approx(rec["all"]["tau_s"][1] * 1e3, abs=0.006)
    # T1592 is on T1: its ratio is the record's ln(S2/S1) turned round
    assert r1.power_ratio[20] == pytest.approx(np.exp(-rec["all"]["ln_ratio"]["20"]), abs=6e-5)
    assert r1.power_ratio[20] * r2.power_ratio[20] == pytest.approx(1.0, abs=2e-3)
    assert r1.notes == "synthetic"
    assert not reg.stale_views(sheet)
    # every byte outside the two inserted blocks is untouched
    new = sheet.read_text(encoding="utf-8")
    assert all(new.count(b) == 1 for b in inserted)
    assert new.replace(inserted[0], "").replace(inserted[1], "") == old
    with pytest.raises(reg.DuplicateEntryError, match="already has a response"):
        reg.add_response(sheet, "T1592", "ARCTERX-2022 Interior",
                         reg.response_from_record(rec, _cal(sheet, "T1592")))


def test_registry_response_refuses_the_wrong_entry(tmp_path):
    sheet = _sheet_copy(tmp_path)
    before = sheet.read_bytes()
    rec, _ = _real_record(tmp_path)
    with pytest.raises(reg.AddError, match="SN194"):
        reg.response_from_record(rec, _cal(sheet, "T1592", "ARCTERX-2025 Interior"))
    with pytest.raises(reg.AddError, match="not T1593"):
        reg.response_from_record(rec, _cal(sheet, "T1593", "SUNRISE 2019 Pelican"))
    # the right bead, the wrong deployment: outside its span, so the loader refuses it
    block = reg.response_from_record(rec, _cal(sheet, "T1592", "SUNRISE 2019 Pelican"))
    with pytest.raises(reg.DateOrderError, match="inside the entry"):
        reg.add_response(sheet, "T1592", "SUNRISE 2019 Pelican", block)
    swapped = copy.deepcopy(rec)
    swapped["channels"] = swapped["channels"][::-1]
    with pytest.raises(reg.AddError, match="on T2 in the record"):
        reg.response_from_record(swapped, _cal(sheet, "T1592"))
    placeholder = copy.deepcopy(rec)
    placeholder["serials"][1] = "T"
    with pytest.raises(reg.AddError, match="placeholder"):
        reg.response_from_record(placeholder, _cal(sheet, "T1592"))
    assert sheet.read_bytes() == before


def test_registry_response_keeps_crlf_and_mode(tmp_path):
    sheet = _sheet_copy(tmp_path)
    sheet.write_bytes(sheet.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    if os.name == "posix":
        sheet.chmod(0o640)
    rec, _ = _real_record(tmp_path)
    reg.add_response(sheet, "T2005", "ARCTERX-2022 Interior",
                     reg.response_from_record(rec, _cal(sheet, "T2005")))
    raw = sheet.read_bytes()
    assert b"\r\n" in raw and raw.count(b"\n") == raw.count(b"\r\n")
    if os.name == "posix":
        assert (sheet.stat().st_mode & 0o777) == 0o640


def test_registry_response_keeps_comments_inside_the_entry(tmp_path):
    sheet = _sheet_copy(tmp_path)
    text = sheet.read_text(encoding="utf-8")
    anchor = "  - bead: T2005\n    deployment: ARCTERX-2022 Interior\n"
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + "# a column-0 comment inside the entry\n"
                        "    # and an indented one\n")
    sheet.write_text(text, encoding="utf-8", newline="\n")
    rec, _ = _real_record(tmp_path)
    block = reg.add_response(sheet, "T2005", "ARCTERX-2022 Interior",
                             reg.response_from_record(rec, _cal(sheet, "T2005")))
    new = sheet.read_text(encoding="utf-8")
    assert new.replace(block, "") == text
    assert _cal(sheet, "T2005").response.partner == "T1592"


def test_registry_response_cli_dry_run_and_show(tmp_path, capsys):
    from odas_tpw.fp07cal.cli import main as fp07cal_main

    sheet = _sheet_copy(tmp_path)
    _rec, js = _real_record(tmp_path)
    before = sheet.read_bytes()
    assert fp07cal_main(["registry", "response", str(js), "--bead", "T1592", "--deployment",
                         "ARCTERX-2022 Interior", "-r", str(sheet), "--dry-run"]) == 0
    assert sheet.read_bytes() == before
    assert "response:" in capsys.readouterr().out
    assert fp07cal_main(["registry", "response", str(js), "--bead", "T1592", "--deployment",
                         "arcterx-2022  interior", "-r", str(sheet)]) == 0
    capsys.readouterr()
    assert fp07cal_main(["registry", "show", "T1592", "-r", str(sheet)]) == 0
    assert "FP07 response vs T2005" in capsys.readouterr().out
    assert fp07cal_main(["registry", "response", str(js), "--bead", "T1592", "--deployment",
                         "No Such Cruise", "-r", str(sheet)]) == 1
    bad = tmp_path / "list.json"
    bad.write_text("[1, 2]", encoding="utf-8")
    assert fp07cal_main(["registry", "response", str(bad), "--bead", "T2005", "--deployment",
                         "ARCTERX-2022 Interior", "-r", str(sheet)]) == 1
    assert "not an `fp07-pair tau` record" in capsys.readouterr().err
