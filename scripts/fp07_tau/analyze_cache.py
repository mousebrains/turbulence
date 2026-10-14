# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""Bead-pair response comparison from cached spectra (cache_spectra.py).

usage: analyze_cache.py CACHEDIR OUT.csv [LABEL ...]

For every (deployment, instrument, T1 serial / T2 serial) and time half:
  * the MODEL-FREE normalized ratio R(f)/R(f_ref): how much more one bead
    attenuates than the other at 5..40 Hz, relative to 1-2 Hz;
  * single- and double-pole fits on a common 1-25 Hz band (comparable across
    deployments) and on the full usable band;
  * a free-shape fit, |H_i|^2 = 1/(1 + (f/fc_i)^a), common a;
with 16-84% intervals from a bootstrap over files. The ratio is the median
over windows of ln(S2/S1) (medlog.py).
"""
import csv
import glob
import os
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import medlog  # noqa: E402
import ratio as R  # noqa: E402

FQ = (5, 10, 20, 30, 40, 60)
F_TOP = 70.0  # above this, the two channels' anti-alias filters (corner ~110 Hz)
              # can differ by component tolerance and bend the ratio


def load(cachedir, label):
    """All cached chunks of a deployment, as ONE dataset per frequency grid.

    Instruments in one deployment can differ in fs or fft length (ARCTERX-22:
    SN194 has 100 bins below 100 Hz, SN428 101), so chunks are grouped by grid
    rather than dropped. Returns a list of dicts; kB (cache_kb.py) is attached
    where a matching kb_NNN.npz exists.
    """
    by_grid = {}
    for p in sorted(glob.glob(os.path.join(cachedir, label, "chunk_*.npz"))):
        z = np.load(p, allow_pickle=False)
        if "empty" in z.files:
            continue
        sig = (z["F"].size, round(float(z["F"][1] - z["F"][0]), 4))
        kbp = p.replace("chunk_", "kb_")
        kb = np.load(kbp)["kB"] if os.path.exists(kbp) else np.full(z["W"].size, np.nan)
        by_grid.setdefault(sig, []).append(({k: z[k] for k in z.files}, kb))
    out = []
    for sig, parts in sorted(by_grid.items()):
        F = parts[0][0]["F"]
        D = {"F": F, "S": [], "N": [], "W": [], "T": [], "chi": [], "file": [], "kB": [],
             "inst": [], "pair": [], "t0": []}
        off = 0
        for z, kb in parts:
            for k, src in (("S", "S"), ("N", "N"), ("W", "W"), ("T", "T"), ("chi", "chi")):
                D[k].append(z[src])
            D["kB"].append(kb if kb.size == z["W"].size else np.full(z["W"].size, np.nan))
            D["file"].append(z["fileidx"] + off)
            D["inst"].extend(z["inst"]); D["pair"].extend(z["pair"]); D["t0"].extend(z["t0"])
            off += len(z["names"])
        for k in ("S", "N", "W", "T", "chi", "file", "kB"):
            D[k] = np.concatenate(D[k])
        for k in ("inst", "pair", "t0"):
            D[k] = np.array(D[k])
        out.append(D)
    return out

def free_shape(f, lnR):
    def model(p):
        lg, l1, l2, a = p
        return lg + np.log(1 / (1 + (f / np.exp(l2)) ** a)) - np.log(1 / (1 + (f / np.exp(l1)) ** a))

    best = None
    for a0 in (1.0, 2.0, 3.0):
        for c1, c2 in ((10, 15), (15, 10), (8, 25), (25, 8)):
            r = least_squares(lambda p: lnR - model(p), [0, np.log(c1), np.log(c2), a0],
                              bounds=([-3, np.log(0.5), np.log(0.5), 0.3], [3, np.log(400), np.log(400), 6.0]))
            if best is None or r.cost < best.cost:
                best = r
    lg, l1, l2, a = best.x
    return np.exp(l1), np.exp(l2), a, float(np.sqrt(2 * best.cost / f.size))


def h2_half(f, tau):
    """|H|^2 for H = 1/(1 + sqrt(i w tau)) -- heat diffusing through a layer
    into a core; falls as f^-1 at high frequency (single pole: f^-2)."""
    q = np.sqrt(np.pi * f * tau)  # sqrt(w tau / 2)
    return 1.0 / ((1 + q) ** 2 + q ** 2)


def fit_half(f, lnR):
    def res(p):
        lg, l1, l2 = p
        return lnR - (lg + np.log(h2_half(f, np.exp(l2))) - np.log(h2_half(f, np.exp(l1))))

    best = min((least_squares(res, [0.0, np.log(a), np.log(b)],
                              bounds=([-3, np.log(1e-4), np.log(1e-4)], [3, np.log(2.0), np.log(2.0)]))
                for a in (0.003, 0.02, 0.1) for b in (0.003, 0.02, 0.1)), key=lambda r: r.cost)
    lg, l1, l2 = best.x
    return np.exp(l1), np.exp(l2), float(np.sqrt(2 * best.cost / f.size))


def f3db_half(tau):
    """Half-power frequency of h2_half: solve (1+q)^2 + q^2 = 2."""
    q = (-1 + np.sqrt(3)) / 2  # 2q^2 + 2q - 1 = 0
    return q**2 / (np.pi * tau)


def summarize(F, lr):
    """Normalized ratio at FQ, fits on 1-25 Hz and full band."""
    ok = np.isfinite(lr) & (F >= 1) & (F <= F_TOP)
    if ok.sum() < 12:
        return None
    # Reference level: the lowest usable 1-3 Hz (some grids start at 2 Hz).
    rb = (F >= 1) & (F <= 3) & np.isfinite(lr)
    ref = np.nanmean(lr[rb]) if rb.any() else np.nan
    out = {"f_hi": float(F[ok].max())}
    for q in FQ:
        i = int(np.argmin(abs(F - q)))
        out[f"nr{q}"] = float(np.exp(lr[i] - ref)) if np.isfinite(lr[i]) else np.nan
    for tag, fhi in (("b25", 25.0), ("full", F_TOP)):
        m = ok & (fhi >= F)
        if m.sum() < 12:
            continue
        for p in (1, 2):
            G, t1, t2, rms = R.fit_ratio(F[m], np.exp(lr[m]), p)
            out[f"{tag}_p{p}_tau1"], out[f"{tag}_p{p}_tau2"], out[f"{tag}_p{p}_rms"] = t1 * 1e3, t2 * 1e3, rms
    if ok.sum() >= 20:
        c1, c2, a, rms = free_shape(F[ok], lr[ok])
        out.update({"fs_fc1": c1, "fs_fc2": c2, "fs_alpha": a, "fs_rms": rms})
        t1, t2, rms = fit_half(F[ok], lr[ok])
        out.update({"ho_tau1": t1 * 1e3, "ho_tau2": t2 * 1e3, "ho_rms": rms,
                    "ho_f3db1": f3db_half(t1), "ho_f3db2": f3db_half(t2)})
    return out


def analyse_group(D, sel, n_boot=100, rng=None):
    rng = rng or np.random.default_rng(0)
    F = D["F"]
    S1, S2 = D["S"][sel, 0, :].astype(float), D["S"][sel, 1, :].astype(float)
    N1, N2 = D["N"][sel, 0, :].astype(float), D["N"][sel, 1, :].astype(float)
    lr, n = medlog.ratio_curve(S1, S2, N1, N2)
    base = summarize(F, lr)
    if base is None:
        return None
    files = D["file"][sel]
    uf = np.unique(files)
    idx_by_file = {f: np.flatnonzero(files == f) for f in uf}
    boots = []
    for _ in range(n_boot):
        pick = np.concatenate([idx_by_file[f] for f in rng.choice(uf, size=uf.size, replace=True)])
        lrb, _ = medlog.ratio_curve(S1[pick], S2[pick], N1[pick], N2[pick])
        sb = summarize(F, lrb)
        if sb:
            boots.append(sb)
    for k in list(base):
        vals = np.array([b.get(k, np.nan) for b in boots], float)
        vals = vals[np.isfinite(vals)]
        if vals.size >= n_boot // 2:
            base[k + "_lo"], base[k + "_hi"] = np.percentile(vals, [16, 84])
    base.update({"windows": int(sel.sum()), "files": int(uf.size), "W": float(np.median(D["W"][sel])),
                 "T": float(np.nanmedian(D["T"][sel]))})
    return base


def main():
    cachedir, out = sys.argv[1], sys.argv[2]
    labels = sys.argv[3:] or sorted(os.path.basename(p) for p in glob.glob(os.path.join(cachedir, "*")))
    rows = []
    for label in labels:
      for D in load(cachedir, label):
        inst_w = D["inst"][D["file"]]
        pair_w = D["pair"][D["file"]]
        t0_w = D["t0"][D["file"]]
        for key in sorted(set(zip(inst_w, pair_w))):
            sel_pair = (inst_w == key[0]) & (pair_w == key[1])
            fl = np.unique(D["file"][sel_pair])
            order = fl[np.argsort([np.nan_to_num(D["t0"][f]) for f in fl], kind="stable")]
            h1 = set(order[: len(order) // 2])
            halves = {"all": sel_pair,
                      "H1": sel_pair & np.isin(D["file"], list(h1)),
                      "H2": sel_pair & ~np.isin(D["file"], list(h1))}
            for hname, sel in halves.items():
                if sel.sum() < 500:
                    continue
                r = analyse_group(D, sel)
                if r is None:
                    continue
                t = t0_w[sel]
                t = t[np.isfinite(t)]
                r.update({"deployment": label, "instrument": key[0], "pair": key[1], "half": hname,
                          "t_start": float(t.min()) if t.size else np.nan, "t_end": float(t.max()) if t.size else np.nan})
                rows.append(r)
                print(f"{label:14s} {key[0]:7s} {key[1]:13s} {hname:3s} win {r['windows']:6d} W {r['W']:.2f} T {r['T']:4.1f}  "
                      f"nr10 {r.get('nr10', np.nan):.2f} nr20 {r.get('nr20', np.nan):.2f} nr40 {r.get('nr40', np.nan):.2f}  "
                      f"SP25 {r.get('b25_p1_tau1', np.nan):5.1f}/{r.get('b25_p1_tau2', np.nan):5.1f}  "
                      f"SPfull {r.get('full_p1_tau1', np.nan):5.1f}/{r.get('full_p1_tau2', np.nan):5.1f}  "
                      f"free a {r.get('fs_alpha', np.nan):.2f} fc {r.get('fs_fc1', np.nan):.1f}/{r.get('fs_fc2', np.nan):.1f} (rms {r.get('fs_rms', np.nan):.4f})  "
                      f"half-order f3dB {r.get('ho_f3db1', np.nan):.1f}/{r.get('ho_f3db2', np.nan):.1f} Hz (rms {r.get('ho_rms', np.nan):.4f}; SP {r.get('full_p1_rms', np.nan):.4f})  f_hi {r['f_hi']:.0f}",
                      flush=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("deployment", "instrument", "pair", "half"), k))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
