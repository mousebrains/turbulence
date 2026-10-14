# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""Absolute FP07 response level for a bead pair, given its RATIO-derived shape.

The T1/T2 ratio (analyze_cache.py) fixes the response family (free shape
|H|^2 = 1/(1 + (f/fc)^a)) and fc1/fc2, but not the common scale. Here both
probes' spectra, at x = k/kB_eps >= x_min (above the low-k excess, where the
spectral shape is well behaved; #191), are modelled as

    ln(S_i - N_i) = a_w + g(log x) + b_i + ln|H(f; c*fc_i0, a)|^2

a_w per window, g free in log-x bins (no Kraichnan form imposed), b_i per
probe, and ONE scale c scanned. c is identified because kB*W varies between
windows, so one frequency falls at different x. kB is from the shear epsilon
(cache_kb.py), never a fitted kB.

usage: abs_anchor.py CACHEDIR LABEL INSTRUMENT PAIR ALPHA FC1 FC2 [x_min]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decomp  # noqa: E402  (fit_additive, bins)
import medlog  # noqa: E402
from analyze_cache import load  # noqa: E402


def h2a(f, fc, a):
    return 1.0 / (1.0 + (f / fc) ** a)


def build(D, kb, sel, x_min=0.1, x_max=1.5, f_max=70.0, snr=10.0):
    """Long-format observations for both probes of the selected windows."""
    F = D["F"]
    ys, lx, ff, ww, pp = [], [], [], [], []
    idx = np.flatnonzero(sel & np.isfinite(kb) & (kb > 0))
    for p in (0, 1):
        S = D["S"][idx, p, :].astype(float)
        N = D["N"][idx, p, :].astype(float)
        sm = medlog.smoothed_snr(S, N)
        x = F[None, :] / (D["W"][idx, None] * kb[idx, None])
        ok = (sm >= snr) & (S > N) & (x >= x_min) & (x <= x_max) & (F[None, :] >= 1) & (F[None, :] <= f_max)
        w, j = np.nonzero(ok)
        ys.append(np.log(S[w, j] - N[w, j]))
        lx.append(np.log10(x[w, j]))
        ff.append(F[j])
        ww.append(w)
        pp.append(np.full(w.size, p))
    return (np.concatenate(ys), np.concatenate(lx), np.concatenate(ff), np.concatenate(ww),
            np.concatenate(pp), idx.size)


def scan(y, lx, f, w, p, n_win, a, fc1, fc2, cs, x_width=0.03):
    bx, _ = decomp.bins(lx, x_width)
    ss = []
    for c in cs:
        fc = np.where(p == 0, c * fc1, c * fc2)
        yy = y - np.log(h2a(f, fc, a))
        _, r = decomp.fit_additive(yy, [(w, n_win, None), (bx, bx.max() + 1, None), (p, 2, None)], n_iter=200)
        ss.append(r @ r)
    return np.array(ss)


def main():
    cachedir, label, inst, pair = sys.argv[1:5]
    a, fc1, fc2 = map(float, sys.argv[5:8])
    x_min = float(sys.argv[8]) if len(sys.argv) > 8 else 0.1
    Ds = [D for D in load(cachedir, label)
          if np.any((D["inst"][D["file"]] == inst) & (D["pair"][D["file"]] == pair))]
    if not Ds:
        raise SystemExit("no such pair")
    D = Ds[0]
    kb = D["kB"]
    sel = (D["inst"][D["file"]] == inst) & (D["pair"][D["file"]] == pair)
    y, lx, f, w, p, n_win = build(D, kb, sel, x_min=x_min)
    cs = np.geomspace(0.2, 5.0, 41)
    ss = scan(y, lx, f, w, p, n_win, a, fc1, fc2, cs)
    i = int(np.argmin(ss))
    s2 = ss[i] / max(y.size - n_win - 50, 1)
    print(f"{label} {inst} {pair}: windows {n_win} points {y.size}; family a={a:.2f}, ratio fc {fc1:.1f}/{fc2:.1f}")
    print(f"  best scale c = {cs[i]:.3f} -> fc {cs[i]*fc1:.1f} / {cs[i]*fc2:.1f} Hz  (edge: {i in (0, len(cs)-1)})")
    print("  SS(c) - min: " + " ".join(f"{c:.2f}:{(v - ss[i]) / s2:.0f}" for c, v in zip(cs[::5], ss[::5])))


if __name__ == "__main__":
    main()
