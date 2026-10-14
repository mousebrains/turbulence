# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""Add shear-epsilon kB to an existing spectra cache (cache_spectra.py).

usage: cache_kb.py CACHEDIR LABEL=GLOB [LABEL=GLOB ...]   (same GLOBs as the cache)

For each cached chunk, writes kb_NNN.npz with kB_eps (window, probe-agnostic)
aligned with the chunk's windows (same file order, every window of each file).
Method-1 products (chi_method 'epsilon') carry an epsilon-derived kB; for
fitted products ('fit', mostly gliders) the fitted kB is CIRCULAR (fitted with
the old tau), so kB is rebuilt from the sibling diss directory's epsilon
(geometric mean of the probes) with nu from the chi file and kappa_T(T, 35, P).
Windows with no usable epsilon get NaN.
"""
import glob
import os
import sys

import netCDF4
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from odas_tpw.scor160.ocean import kappa_T  # noqa: E402


def diss_path(chi_path):
    d, b = os.path.split(chi_path)
    parent, name = os.path.split(d)  # .../chi_NN
    suffix = name.split("_", 1)[1] if "_" in name else "00"
    for cand in (f"diss_{suffix}", "diss_00"):
        p = os.path.join(parent, cand, b)
        if os.path.exists(p):
            return p
    return None


def kb_for(chi_path):
    with netCDF4.Dataset(chi_path) as c:
        method = getattr(c, "chi_method", "?")
        n = c["speed"].shape[0]
        if method == "epsilon" and "kB" in c.variables:
            kb = np.asarray(c["kB"][:], float)
            return np.nanmean(kb, axis=0) if kb.ndim == 2 else kb, method
        nu = np.asarray(c["nu"][:], float)
        T = np.asarray(c["T_mean"][:], float)
        P = np.asarray(c["P_mean"][:], float) if "P_mean" in c.variables else np.zeros(n)
        tc = np.asarray(c["t"][:], float)
    dp = diss_path(chi_path)
    if dp is None:
        return np.full(n, np.nan), method + "/no-diss"
    with netCDF4.Dataset(dp) as d:
        td = np.asarray(d["t"][:], float)
        eps = np.asarray(d["epsilon"][:], float)
    if td.shape != tc.shape or np.nanmax(np.abs(td - tc)) > 1e-2:
        return np.full(n, np.nan), method + "/misaligned"
    with np.errstate(invalid="ignore", divide="ignore"):
        e = np.exp(np.nanmean(np.log(np.where(eps > 0, eps, np.nan)), axis=0))
    kap = kappa_T(T, 35.0, P)
    return (1 / (2 * np.pi)) * (e / (nu * kap**2)) ** 0.25, method + "/diss"


def main():
    cachedir = sys.argv[1]
    for spec in sys.argv[2:]:
        label, pattern = spec.split("=", 1)
        srcdir = os.path.dirname(pattern)
        for ch in sorted(glob.glob(os.path.join(cachedir, label, "chunk_*.npz"))):
            out = ch.replace("chunk_", "kb_")
            if os.path.exists(out):
                continue
            z = np.load(ch)
            if "empty" in z.files:
                continue
            kbs, methods = [], []
            for name in z["names"]:
                kb, m = kb_for(os.path.join(srcdir, str(name)))
                kbs.append(kb)
                methods.append(m)
            kb = np.concatenate(kbs)
            if kb.size != z["W"].size:
                print(f"{ch}: window count mismatch ({kb.size} vs {z['W'].size}); skipped", flush=True)
                continue
            np.savez_compressed(out, kB=kb, method=np.array(methods))
            print(f"{label} {os.path.basename(out)}: {np.isfinite(kb).mean():.0%} windows with kB; "
                  f"{sorted(set(methods))}", flush=True)


if __name__ == "__main__":
    main()
