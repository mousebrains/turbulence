#!/usr/bin/env python3
"""Pin the ABSOLUTE FP07 tau by requiring Method 1 and Method 2 to agree.

The idea
--------
Method 1 fixes kB from the shear epsilon and fits chi alone.
Method 2 fits kB from the temperature spectrum itself, along with chi.

tau and kB both control the high-wavenumber rolloff, but they enter
differently, so a WRONG tau makes the free fit compensate:

  * tau too small  -> the model is not attenuated enough, the observed
                      spectrum falls faster, and the free fit picks kB too LOW
  * tau too large  -> the model is over-attenuated and the free fit picks
                      kB too HIGH

So kB_fitted / kB_epsilon crosses 1 at the correct tau. That is an absolute
constraint on tau which uses NO inter-probe assumption -- it is independent of
the ratio work in issue #189, and independent of the shape-misfit metric in
fit_tau_absolute.py (which shares the "compare a model to the data" weakness).

kB is only weakly sensitive to epsilon errors (kB ~ epsilon^(1/4)), so even a
2x epsilon error is 1.19x in kB.

Run:  python3 fit_tau_method12.py [chi_root] [stride] [name_filter]
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import netCDF4

from odas_tpw.chi.batchelor import KAPPA_T, batchelor_grad, kraichnan_grad

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/SeaChest/CASPER/CasperWest/VMP/Processed"
STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 8
FILT = sys.argv[3] if len(sys.argv) > 3 else None
MODEL = kraichnan_grad          # CASPER-West and ARCTERX both use spectrum_model: kraichnan

S_GRID = np.round(np.arange(0.30, 3.01, 0.10), 3)
KB_GRID = np.round(np.exp(np.linspace(np.log(0.25), np.log(4.0), 33)), 4)   # x kB_eps


def load(files):
    out = {k: [] for k in ("K", "GT", "NZ", "SB", "CHI", "KB", "KMT", "KMR", "SP", "T")}
    for fn in files:
        try:
            d = netCDF4.Dataset(fn)
            g = {k: np.asarray(d[k][:]) for k in
                 ("chi", "spec_gradT", "spec_batch", "spec_noise", "K", "kB",
                  "K_max_T", "K_max_ratio", "speed")}
            st = float(np.asarray(d["stime"][:]).ravel()[0]); d.close()
        except Exception:
            continue
        chi = g["chi"]
        if chi.ndim != 2 or chi.shape[0] < 2:
            continue
        i = np.arange(chi.shape[1])
        ok = (np.all(np.isfinite(chi[:, i]), 0) & np.all(chi[:, i] > 0, 0)
              & np.isfinite(g["K_max_ratio"][0, i]) & (g["speed"][i] > 0)
              & np.all(np.isfinite(g["kB"][:, i]), 0) & np.all(g["kB"][:, i] > 0, 0))
        i = i[ok]
        if i.size == 0:
            continue
        out["K"].append(g["K"][:, i].T); out["SP"].append(g["speed"][i])
        out["CHI"].append(chi[:, i].T); out["KB"].append(g["kB"][:, i].T)
        out["KMT"].append(g["K_max_T"][:, i].T); out["KMR"].append(g["K_max_ratio"][:, i].T)
        out["T"].append(np.full(i.size, st))
        for key, src in (("GT", "spec_gradT"), ("NZ", "spec_noise"), ("SB", "spec_batch")):
            out[key].append(np.stack([g[src][p][:, i].T for p in (0, 1)], 1))
    return {k: np.concatenate(v) for k, v in out.items()}


files = sorted(glob.glob(f"{ROOT}/chi_*[0-9]/*_prof*.nc"))
if FILT:
    files = [f for f in files if FILT in f]
files = files[::STRIDE]
print(f"{len(files)} chi profiles" + (f", filter {FILT!r}" if FILT else ""), flush=True)
D = load(files)
# A free-kB fit can only constrain kB if the fit band actually reaches past it.
# Without this the "fit" slides along a flat objective -- the same trap as the
# inter-probe degeneracy, one level down.
KMR_MIN = float(sys.argv[4]) if len(sys.argv) > 4 else 1.2
sel = D["KMR"][:, 0] > KMR_MIN
D = {k: (v[sel] if getattr(v, "shape", (0,))[0] == len(sel) else v) for k, v in D.items()}
print(f"K_max_ratio > {KMR_MIN}: {int(sel.sum()):,} of {len(sel):,} windows kept")
U = float(np.median(D["SP"]))
print(f"{len(D['CHI']):,} windows, median U {U:.2f} m/s, tau_lueck "
      f"{1000 * 0.01 / np.sqrt(U):.1f} ms")

# ---- CONTROL: does our model reproduce the pipeline's saved spec_batch? -----
p = 0
mine = MODEL(D["K"], D["KB"][:, p][:, None], 1.0, KAPPA_T) * D["CHI"][:, p][:, None]
sb = D["SB"][:, p, :]
m = np.isfinite(mine) & np.isfinite(sb) & (mine > 0) & (sb > 0)
rel = np.abs(np.log10(np.where(m, sb, 1.0) / np.where(m, mine, 1.0)))
print(f"CONTROL  our {MODEL.__name__} vs saved spec_batch: "
      f"median |log10 ratio| {np.nanmedian(np.where(m, rel, np.nan)):.4f} "
      f"(0 = identical model)\n")

tauL = 0.01 * (1.0 / D["SP"]) ** 0.5
F = D["K"] * D["SP"][:, None]


def fit_at(p, s, free_kB):
    """Return (chi_hat, kB_hat). free_kB=False pins kB at the epsilon value."""
    H2 = 1.0 / (1.0 + (2 * np.pi * F * (s * tauL)[:, None]) ** 2)
    obs = D["GT"][:, p, :] - D["NZ"][:, p, :]
    base = (np.isfinite(obs) & (D["K"] > 0) & (D["K"] <= D["KMT"][:, p][:, None])
            & (obs > 0) & ((D["GT"][:, p, :]) > 2.0 * D["NZ"][:, p, :]))
    kbs = [1.0] if not free_kB else KB_GRID
    best_r = np.full(len(obs), np.inf); best_c = np.full(len(obs), np.nan)
    best_k = np.full(len(obs), np.nan)
    for f in kbs:
        kb = D["KB"][:, p] * f
        shape = MODEL(D["K"], kb[:, None], 1.0, KAPPA_T) * H2
        band = base & np.isfinite(shape) & (shape > 0)
        o = np.where(band, obs, 0.0); sh = np.where(band, shape, 0.0)
        kk = np.where(band, D["K"], 0.0)
        num = np.trapezoid(o * sh, kk, axis=1); den = np.trapezoid(sh * sh, kk, axis=1)
        c = np.where(den > 0, num / den, np.nan)
        model = c[:, None] * shape
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(band & (model > 0), np.abs(np.log10(obs / np.where(model > 0, model, 1.0))), np.nan)
            r = np.nanmedian(r, axis=1)
        upd = np.isfinite(r) & (r < best_r)
        best_r = np.where(upd, r, best_r); best_c = np.where(upd, c, best_c)
        best_k = np.where(upd, kb, best_k)
    return best_c, best_k


for p, nm in ((0, "T1"), (1, "T2")):
    print(f"--- {nm}: does the FREE-kB fit recover the epsilon-derived kB?")
    print(f"  {'s':>5}{'tau[ms]':>9}{'kB_fit/kB_eps':>15}{'chi_M2/chi_M1':>15}{'n':>8}")
    rows = []
    for s in S_GRID:
        c1, _ = fit_at(p, s, False)
        c2, k2 = fit_at(p, s, True)
        g = np.isfinite(c1) & np.isfinite(c2) & (c1 > 0) & (c2 > 0) & np.isfinite(k2)
        if g.sum() < 200:
            continue
        rk = float(np.median(k2[g] / D["KB"][g, p]))
        rc = float(np.median(c2[g] / c1[g]))
        rows.append((s, rk, rc, int(g.sum())))
        if round(s, 2) in (0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.1, 2.5, 3.0):
            print(f"  {s:5.2f}{1000 * s * 0.01 / np.sqrt(U):9.1f}{rk:15.3f}{rc:15.3f}{g.sum():8d}")
    # where does kB_fit/kB_eps cross 1?
    ss = np.array([r[0] for r in rows]); rk = np.array([r[1] for r in rows])
    cross = None
    for i in range(len(ss) - 1):
        if (rk[i] - 1) * (rk[i + 1] - 1) <= 0 and rk[i] != rk[i + 1]:
            cross = ss[i] + (1 - rk[i]) * (ss[i + 1] - ss[i]) / (rk[i + 1] - rk[i])
            break
    if cross is not None:
        print(f"  => kB_fit / kB_eps crosses 1 at s = {cross:.2f}  "
              f"({1000 * cross * 0.01 / np.sqrt(U):.1f} ms)\n")
    else:
        print(f"  => NO CROSSING (range {rk.min():.2f}-{rk.max():.2f}); tau not "
              f"constrained by this test on these data\n")
