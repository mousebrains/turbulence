#!/usr/bin/env python3
"""Analyse a still-water bench zero for a MicroRider's JAC EM flowmeter.

WHY. On six of seven MicroRider-on-Slocum deployments, taking U_EM at face
value requires a NEGATIVE angle of attack, which a wing cannot have. Four
explanations have been tested and rejected (local flow acceleration, a nominal
EM slope, roll, and the on-deck reading as the zero). The field data cannot
separate the angle of attack from a sensor zero because the two are degenerate:
the lever is the range of speeds flown at fixed pitch, and that is only 1.3x to
5x. A fresh/salt tank series gives a lever of ~130x in 1/sigma, which is why the
bench can settle what the deployments could not.

HOW BIG. Pinning alpha at 4.5 deg (Tanaka + Welch) turns the degeneracy into a
number. Across the three EM units that were actually tracking -- 051, 066 and
046 -- every one reads HIGH, by +44 to +61 mm/s (380-530 counts, 17-23% of the
a coefficient). Decomposed:

    common bias  52 mm/s, SAME SIGN on all three
    unit scatter  8 mm/s sd -- which is JFE's own a reproducibility (9.4 mm/s)

So the scatter is ordinary manufacturing spread and the bias is not. Nothing
here needs alpha = 4.5 to be exactly right: requiring only alpha >= 0, which a
climbing wing must satisfy, still demands +11 to +41 mm/s.

WHY THE UNITS CAN STILL BE IN SPEC. The JFE AEM1-G datasheet (indexed in
../AEM1-G/docs/datasheets/README.md) gives range 0-500 cm/s and accuracy
+/-0.5 cm/s or +/-2%RD. A CONSTANT 47 mm/s zero error is 0.94% of reading at 5 m/s -- inside
+/-2%RD -- and only breaches the spec below ~2.4 m/s. We fly at 20-50 cm/s, the
bottom 4-10% of the calibrated span, where the same error is 12-17% of reading.
A unit can therefore pass its acceptance test and still be unusable for us,
which is the whole argument for measuring the zero ourselves.

THE MODEL. The Faraday EMF is conductivity-independent -- that is the virtue of
an EM meter -- but the ZERO is not:

    c(sigma) = A + B / sigma

    A     amplifier input offset, plus electrode asymmetry that does not scale
    B/s   bias current x source impedance;  Z_source ~ 1/sigma

Two unknowns. Each can gives one (sigma, c) pair, so two cans determine it and
three or more TEST the functional form rather than assuming it.

INPUT, per can:
  * one or more MicroRider .p files recorded with the head in still water
  * the conductivity and temperature of that can, either as --sigma on the
    command line or from an RBR NetCDF/CSV logged in the can

    python em_bench_zero.py \\
        --can seawater=bench/sw/*.p    --sigma 5.31 --temp 19.4 \\
        --can half=bench/half/*.p      --sigma 2.86 --temp 19.5 \\
        --can tap=bench/tap/*.p        --sigma 0.041 --temp 19.6 \\
        -o bench_report

WHAT IT DOES
  1. Reads U_EM and Incl_T per can. Incl_T is a CASE thermometer, not a head
     thermometer -- measured against the glider CTD it lags the water by ~6.5
     min (median 390 s over 21 profiles) and spans only 67% of the water's
     range, i.e. a first-order tau of 6-10 min. That makes it the SLOWEST thing
     in the can, so a flat Incl_T is a conservative "everything has settled"
     signal. Use the ASYMPTOTE, never the whole-record mean.

     Preferred protocol is an OVERNIGHT soak (~16 h) per can: ~100 time
     constants, so every thermal and electrochemical transient is long dead and
     the whole record is usable.
  2. Reports the settled zero per can with a block-bootstrap error bar (blocks,
     not samples: the residual drift is correlated over minutes).
  3. Fits c(sigma) = A + B/sigma and reports both terms.
  4. Predicts c at each deployment's in-situ sigma and re-runs the `excess`
     table, so the success criterion -- every deployment giving alpha > 0 -- is
     checked automatically.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

# Per-LEG medians from steady climbs (second half of each leg, so alpha has
# settled), inside a +/-1.5 deg pitch band. These SUPERSEDE an earlier sample-
# level table whose footprint was not matched leg to leg.
# (name, EM SN, theta_deg, |W| m/s, U_EM m/s, n_legs)
#
# osu684-2025 is split at the wing failure (2025-02-06 05:57:17Z): roll went
# 12.4 -> 21.0 deg and climb pitch 43.6 -> 37.3, so pre and post are different
# aircraft and must not be pooled.
DEPLOYMENTS = [
    ("RU33 2021", "042", 25.7, 0.158, 0.301, 0),      # not yet re-reduced per leg
    ("osu685 2023 d0-17", "046", 38.2, 0.319, 0.528, 107),
    ("osu684 2025 pre", "051", 42.5, 0.174, 0.274, 42),
    ("osu684 2025 post", "051", 37.3, 0.228, 0.380, 48),
    ("osu685 2025", "066", 42.5, 0.234, 0.366, 726),
    ("sl684 2026", "079", 44.9, 0.180, 0.244, 0),     # not yet re-reduced per leg
    ("sl685 2026", "066", 44.7, 0.238, 0.360, 0),     # not yet re-reduced per leg
]
# EM 044 (osu684 2023) is DELIBERATELY ABSENT: it sat at ~2727 counts whatever
# the vehicle did, and 126 of its 128 legs report U_EM < |W|, which is
# impossible. That unit was not tracking, and it is a failure, not a zero error.
#
# EM 046 fell apart during 2023 as well -- the leg-median count leaves the
# physical range at day 18 and reaches 45 758 (5.08 m/s) by day 23 -- so only
# days 0-17 are used. The cut is on TIME, taken from the count trend, so it
# does not select on U_EM itself.


def read_can(paths: list[Path]) -> dict:
    """Concatenate U_EM and Incl_T from one can's .p files onto one clock."""
    from odas_tpw.rsi.p_file import PFile

    t, u, it = [], [], []
    for p in sorted(paths):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                pf = PFile(p)
            except Exception as exc:
                print(f"    skip {p.name}: {exc}", file=sys.stderr)
                continue
        ch = pf.channels
        if "U_EM" not in ch:
            print(f"    skip {p.name}: no U_EM channel", file=sys.stderr)
            continue
        ts = np.asarray(pf.t_slow, dtype=np.float64)
        U = np.asarray(ch["U_EM"], dtype=np.float64)
        T = np.asarray(ch.get("Incl_T", np.full(U.size, np.nan)), dtype=np.float64)
        n = min(ts.size, U.size, T.size)
        t0 = pf.start_time.timestamp() if pf.start_time else 0.0
        t.append(t0 + ts[:n])
        u.append(U[:n])
        it.append(T[:n])
    if not t:
        return {}
    t = np.concatenate(t)
    o = np.argsort(t)
    return {"t": t[o], "U": np.concatenate(u)[o], "IT": np.concatenate(it)[o]}


def settled(d: dict, skip_s: float, rng: np.random.Generator) -> tuple[float, float, dict]:
    """Zero from the settled tail, with a BLOCK bootstrap.

    Blocks of 60 s, not individual samples: the residual after the thermal
    transient drifts on a timescale of minutes, so a sample bootstrap would
    report an error bar perhaps 50x too tight.
    """
    t, U = d["t"], d["U"]
    t = t - t[0]
    keep = np.isfinite(U) & (t >= skip_s)
    if keep.sum() < 100:
        return np.nan, np.nan, {"n": int(keep.sum())}
    tt, uu = t[keep], U[keep]
    blk = (tt // 60.0).astype(int)
    ub = np.unique(blk)
    means = np.array([np.median(uu[blk == b]) for b in ub])
    boot = [np.median(rng.choice(means, means.size, replace=True)) for _ in range(4000)]
    lo, mid, hi = np.percentile(boot, [16, 50, 84])
    # residual drift across the retained window -- a stability check
    half = len(means) // 2
    drift = float(np.median(means[half:]) - np.median(means[:half])) if half > 2 else np.nan
    return float(mid), float((hi - lo) / 2), {
        "n": int(keep.sum()), "blocks": int(ub.size),
        "minutes": float(tt.max() / 60), "drift": drift,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--can", action="append", required=True, metavar="NAME=GLOB",
                    help="one per can; repeatable")
    ap.add_argument("--sigma", action="append", type=float, required=True,
                    help="conductivity [S/m] of each can, in the same order")
    ap.add_argument("--temp", action="append", type=float, default=None,
                    help="temperature [degC] of each can, for the record")
    ap.add_argument("--skip", type=float, default=1200.0,
                    help="[s] discarded after immersion as thermal transient "
                         "(default 1200 = 20 min)")
    ap.add_argument("--hourly", action="store_true",
                    help="break a long (overnight) record into hourly blocks, to show "
                         "drift and to expose the within-can sigma lever from the room's "
                         "diurnal temperature cycle")
    ap.add_argument("-o", "--output", type=Path, default=Path("em_bench"))
    args = ap.parse_args()

    if len(args.can) != len(args.sigma):
        raise SystemExit("--can and --sigma must be given the same number of times")
    temps = args.temp or [float("nan")] * len(args.can)
    rng = np.random.default_rng(11)
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"discarding the first {args.skip/60:.0f} min of each can as thermal transient\n")
    print(f"{'can':>14s} {'sigma':>7s} {'T':>6s} {'min':>6s} {'blocks':>7s} "
          f"{'ZERO [m/s]':>13s} {'drift':>9s} {'Incl_T span':>14s}")
    sig, cz, ce, names = [], [], [], []
    for spec, s, T in zip(args.can, args.sigma, temps):
        name, _, glob = spec.partition("=")
        paths = sorted(Path().glob(glob)) or [Path(glob)]
        d = read_can([p for p in paths if p.exists()])
        if not d:
            print(f"{name:>14s}  no usable .p files")
            continue
        z, e, st = settled(d, args.skip, rng)
        itv = d["IT"][np.isfinite(d["IT"])]
        itspan = f"{itv.min():.2f}-{itv.max():.2f}" if itv.size else "n/a"
        print(f"{name:>14s} {s:7.3f} {T:6.1f} {st['minutes']:6.1f} {st['blocks']:7d} "
              f"{z:+8.4f}±{e:.4f} {st['drift']:+9.4f} {itspan:>14s}")
        if args.hourly:
            t0 = d["t"] - d["t"][0]
            keep = np.isfinite(d["U"]) & (t0 >= args.skip)
            hh = ((t0[keep] - args.skip) // 3600).astype(int)
            print(f"{'':>14s}   hourly:", end="")
            for h in np.unique(hh):
                m = hh == h
                if m.sum() < 600:
                    continue
                it_h = d["IT"][keep][m]
                it_h = np.median(it_h[np.isfinite(it_h)]) if np.isfinite(it_h).any() else np.nan
                print(f"  h{h}: {np.median(d['U'][keep][m]):+.4f}"
                      f"@{it_h:.2f}C", end="")
            print()
        if np.isfinite(z):
            sig.append(s)
            cz.append(z)
            ce.append(max(e, 1e-4))
            names.append(name)

    if len(sig) < 2:
        raise SystemExit("\nneed at least two cans with a settled zero")
    sig = np.array(sig)
    cz = np.array(cz)
    ce = np.array(ce)
    inv = 1.0 / sig
    w = 1.0 / ce
    B, A = np.polyfit(inv, cz, 1, w=w)
    boot = [np.polyfit(inv, cz + rng.normal(0, ce), 1, w=w) for _ in range(4000)]
    Bs = np.array([b[0] for b in boot])
    As = np.array([b[1] for b in boot])

    print(f"\n{'='*78}\nc(sigma) = A + B/sigma      lever in 1/sigma: {inv.max()/inv.min():.0f}x")
    print(f"  A = {A:+.5f} +/- {np.std(As):.5f} m/s      (sigma-independent)")
    print(f"  B = {B:+.5f} +/- {np.std(Bs):.5f} m/s.S/m  (bias-current term)")
    if len(sig) >= 3:
        resid = cz - (A + B * inv)
        print("  fit residuals [mm/s]: " + " ".join(f"{1000*r:+.2f}" for r in resid))
        print("  (3+ cans TEST the 1/sigma form; large structured residuals mean it is wrong)")
    else:
        print("  only 2 cans: the 1/sigma form is ASSUMED, not tested. Add a third.")

    print(f"\n{'='*78}\nApplied to the deployments — success criterion is alpha > 0 everywhere\n")
    print(f"{'deployment':>14s} {'EM':>4s} {'sigma':>6s} {'c pred':>9s} "
          f"{'alpha before':>13s} {'alpha after':>12s}")
    for name, em, th, W, U, nleg in DEPLOYMENTS:
        s_situ = 4.4          # representative in-situ conductivity [S/m]
        c = A + B / s_situ
        Wm = W               # already m/s in this table, NOT dbar/s
        a0 = np.degrees(np.arcsin(np.clip(Wm / U, -1, 1))) - th
        a1 = np.degrees(np.arcsin(np.clip(Wm / (U - c), -1, 1))) - th
        flag = "" if a1 > 0 else "   <-- STILL NEGATIVE"
        print(f"{name:>14s} {em:>4s} {s_situ:6.2f} {c:+9.4f} {a0:+13.2f} {a1:+12.2f}{flag}")
    print("\n  NOTE: one A and B are fitted here, so this assumes ONE flowmeter was")
    print("  benched. Bench every unit you can and key the correction by EM serial;")
    print("  the deployments differ by unit, not by vehicle.")


if __name__ == "__main__":
    main()
