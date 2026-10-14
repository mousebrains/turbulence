"""Synthetic check of abs_anchor.py on a real deployment's geometry."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import abs_anchor as A
from analyze_cache import load

cachedir, label, inst, pair = sys.argv[1:5]
D = [d for d in load(cachedir, label) if np.any((d["inst"][d["file"]] == inst) & (d["pair"][d["file"]] == pair))][0]
kb = D["kB"]
sel = (D["inst"][D["file"]] == inst) & (D["pair"][D["file"]] == pair) & np.isfinite(kb)
F = D["F"]; rng = np.random.default_rng(4)
a0, fc10, fc20 = 0.94, 21.0, 5.1
def synth(scale, distort=True, excess=True):
    S = D["S"].astype(float).copy()
    idx = np.flatnonzero(sel)
    for i in idx:
        x = np.maximum(F / (D["W"][i] * kb[i]), 1e-6)
        g = x * np.exp(-np.sqrt(6 * 5.26) * x)
        if excess: g = g * (1 + 3 * (0.03 / x) ** 1.5)
        if distort: g = g * (1 - 0.45 * np.exp(-0.5 * (np.log(x / 0.4) / 0.5) ** 2))
        lvl = np.exp(rng.normal(0, 1)) * np.nanmax(D["N"][i, 0]) * 3e3 / np.max(g)
        for p, fc in ((0, fc10), (1, fc20)):
            t = lvl * (1.0 if p == 0 else 0.8) * g * A.h2a(F, scale * fc, a0) + D["N"][i, p]
            S[i, p] = t * rng.chisquare(9.5, F.size) / 9.5
    return S
clean = "clean" in sys.argv
for scale in (1.0, 2.0):
    Ds = dict(D); Ds["S"] = synth(scale, distort=not clean, excess=not clean)
    y, lx, f, w, p, n = A.build(Ds, kb, sel, x_min=0.1)
    cs = np.geomspace(0.2, 5.0, 41)
    ss = A.scan(y, lx, f, w, p, n, a0, fc10, fc20, cs)
    i = int(np.argmin(ss))
    print(f"truth scale {scale}: recovered c = {cs[i]:.3f}  (windows {n}, points {y.size})", flush=True)
