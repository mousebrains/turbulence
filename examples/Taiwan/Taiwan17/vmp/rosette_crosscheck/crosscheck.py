#!/usr/bin/env python3
"""VMP SN142 thermistors vs the ship SBE 911plus, rosette cast RR1704_10 / DAT_166.

The SBE file is 1-dbar bin-averaged with no time column, so everything is
compared in PRESSURE, on the SBE's own bins (bin k = mean over (k-0.5, k+0.5]).
Each difference is fit as

    dX(p) = a + b * dRef/dp(p)

so `a` is the sensor offset (what remains where the water is uniform) and `b` is
the effective vertical offset in dbar (mounting height on the rosette, pressure
sensor offsets, and sensor lag x descent rate -- inseparable on one downcast).

Chains tested:
  A  JAC_T, JAC_C, SP(JAC)          vs SBE        -- the reference the FP07s are fit to
  B  pipeline T1/T2 (perturb run)   vs SBE        -- its real output, its 102.7-248.7 dbar fit range
  C  fp07_calibrate over the whole downcast vs SBE -- what a free-fall cast would have given
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import gsw
import numpy as np
import xarray as xr

from odas_tpw.perturb.fp07_cal import fp07_calibrate
from odas_tpw.processing.ct_align import ct_align
from odas_tpw.rsi import PFile

HERE = Path(__file__).resolve().parent
T17 = HERE.parent.parent                       # this lives in Taiwan17/vmp/rosette_crosscheck/
CNV = T17 / "uctd/calibration_casts/RR1704_10.cnv"
PFILE = T17 / "vmp/raw/ctd_rosette/SN142/DAT_166.P"
PROF_B = HERE / "Processed/profiles_00/raw__ctd_rosette__SN142__DAT_166_prof001.nc"
LAT, LON = 21.7702, 119.4809           # from the CNV header
P_MIN = 5.0                            # SBE pumps / surface soak above this


def mad(x):
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if x.size else np.nan


def read_cnv(path):
    names, rows = [], []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("# name "):
            names.append(line.split("=", 1)[1].split(":", 1)[0].strip())
        elif not line.startswith(("*", "#")) and line.strip():
            rows.append([float(v) for v in line.split()])
    a = np.array(rows)
    return {n: a[:, i] for i, n in enumerate(names)}


def bin_to(pbins, P, X, keep):
    out = np.full(pbins.size, np.nan)
    n = np.zeros(pbins.size, int)
    idx = np.rint(P[keep]).astype(int)          # (k-0.5, k+0.5] -> k
    x = X[keep]
    for j, k in enumerate(pbins.astype(int)):
        s = (idx == k) & np.isfinite(x)
        n[j] = s.sum()
        if n[j]:
            out[j] = x[s].mean()
    return out, n


def fit(label, dX, grad, p, sel, unit, scale=1.0):
    s = sel & np.isfinite(dX) & np.isfinite(grad)
    A = np.column_stack([np.ones(s.sum()), grad[s]])
    coef, *_ = np.linalg.lstsq(A, dX[s], rcond=None)
    res = dX[s] - A @ coef
    weak = s & (np.abs(grad) < 0.005 if unit == "mK" else np.abs(grad) < 0.02)
    print(f"  {label:34s} n={s.sum():3d}  raw median {np.median(dX[s])*scale:+8.2f} {unit} (MAD {mad(dX[s])*scale:6.2f})"
          f" | weak-gradient n={weak.sum():3d} median {np.median(dX[weak])*scale:+8.2f} (MAD {mad(dX[weak])*scale:5.2f})"
          f" | fit a={coef[0]*scale:+8.2f} {unit}, b={coef[1]:+6.2f} dbar, resid MAD {mad(res)*scale:5.2f}")
    return coef


def main():
    sbe = read_cnv(CNV)
    pb = sbe["prDM"]
    T0, T1s, C0, C1s = sbe["t090C"], sbe["t190C"], sbe["c0mS/cm"], sbe["c1mS/cm"]
    SP0 = gsw.SP_from_C(C0, T0, pb)
    gT = np.gradient(T0, pb)
    gC = np.gradient(C0, pb)
    gS = np.gradient(SP0, pb)
    deep = pb >= P_MIN
    print(f"SBE cast: {pb.size} bins, {pb.min():.0f}-{pb.max():.0f} dbar, T {T0.min():.3f}-{T0.max():.3f} C\n")

    pf = PFile(str(PFILE))
    c = pf.channels
    t, P = pf.t_slow, c["P"]
    imax = int(np.nanargmax(P))
    i0 = int(np.argmax(P > 2.0))
    down = np.zeros(P.size, bool)
    down[i0:imax + 1] = True
    Ps = np.convolve(P, np.ones(64) / 64, "same")
    W = np.gradient(Ps, t)
    prof = [(i0, imax)]
    JAC_T = c["JAC_T"]
    JAC_C, clag = ct_align(JAC_T, c["JAC_C"], pf.fs_slow, prof)
    print(f"DAT_166 downcast: slow idx {i0}-{imax}, {P[i0]:.1f}-{P[imax]:.1f} dbar, {t[imax]-t[i0]:.0f} s; "
          f"CT-align lag {clag}\n")

    print("REFERENCE SELF-CONSISTENCY (SBE secondary - primary)")
    fit("t190C - t090C", T1s - T0, gT, pb, deep, "mK", 1e3)
    fit("c1 - c0", C1s - C0, gC, pb, deep, "uS/cm", 1e3)

    for tag, keep in (("all downcast samples", down), ("downcast, W > 0.2 dbar/s", down & (W > 0.2))):
        print(f"\nA  JAC vs SBE primary   [{tag}]")
        jt, n = bin_to(pb, P, JAC_T, keep)
        jc, _ = bin_to(pb, P, JAC_C, keep)
        jsp = gsw.SP_from_C(jc, jt, pb)
        print(f"  VMP samples per bin: median {np.median(n[deep]):.0f}, min {n[deep].min()}")
        fit("JAC_T - t090C", jt - T0, gT, pb, deep, "mK", 1e3)
        fit("JAC_T - t190C", jt - T1s, np.gradient(T1s, pb), pb, deep, "mK", 1e3)
        fit("JAC_C - c0", jc - C0, gC, pb, deep, "uS/cm", 1e3)
        fit("SP(JAC) - SP(SBE)", jsp - SP0, gS, pb, deep, "mPSU", 1e3)
        for lo, hi in ((5, 100), (100, 252)):
            fit(f"JAC_T - t090C, {lo}-{hi} dbar", jt - T0, gT, pb, deep & (pb >= lo) & (pb < hi), "mK", 1e3)

    print("\nB  pipeline T1/T2 (perturb run, fit over its one detected profile) vs SBE primary")
    d = xr.open_dataset(PROF_B, decode_times=False)
    pP = d["P"].values
    lo_b, hi_b = float(np.nanmin(pP)), float(np.nanmax(pP))
    inb = deep & (pb >= np.ceil(lo_b)) & (pb <= np.floor(hi_b))
    print(f"  profile range {lo_b:.1f}-{hi_b:.1f} dbar; SBE T there {np.nanmin(T0[inb]):.3f}-{np.nanmax(T0[inb]):.3f} C")
    for ch in ("T1", "T2", "JAC_T"):
        x, _ = bin_to(pb, pP, d[ch].values, np.ones(pP.size, bool))
        fit(f"{ch} (pipeline) - t090C", x - T0, gT, pb, inb, "mK", 1e3)

    print("\nC  fp07_calibrate refit over the WHOLE downcast (same function, Taiwan17 settings) vs SBE primary")
    for order in (2, 1):
        pfc = copy.deepcopy(pf)
        cal = fp07_calibrate(pfc, prof, reference="JAC_T", order=order, max_lag_seconds=10.0, must_be_negative=True)
        lags = {k: v for k, v in cal.items() if "lag" in k.lower()}
        print(f"  order {order}: lag info {lags if lags else '(not reported)'}")
        for ch in ("T1", "T2"):
            x, _ = bin_to(pb, P, cal["channels"][ch], down)
            fit(f"{ch} order {order} - t090C", x - T0, gT, pb, deep, "mK", 1e3)
            fit(f"{ch} order {order} - JAC_T (own fit)", x - bin_to(pb, P, JAC_T, down)[0], gT, pb, deep, "mK", 1e3)

    print("\nFactory coefficients, for scale:")
    for ch in ("T1", "T2"):
        x, _ = bin_to(pb, P, c[ch], down)
        print(f"  {ch} factory - t090C: median {np.nanmedian((x - T0)[deep]):+.3f} C")


if __name__ == "__main__":
    sys.exit(main())
