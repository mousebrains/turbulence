# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""Cache every window's two-probe gradient spectra (f <= F_MAX) from chi products.

usage: cache_spectra.py OUTDIR LABEL=GLOB [LABEL=GLOB ...]

Writes OUTDIR/<LABEL>/chunk_NNN_gK.npz, one per CHUNK files and frequency grid K, each holding float32
S (window, probe, freq), N (noise model, same shape), the frequency grid F,
and per-window W, T_mean, P_mean, chi (window, probe), fileidx (file index), and the
bead serials and instrument of each file. A chunk that exists is skipped, so
an interrupted run resumes (chunk_NNN.done marks a finished chunk). Only files with two FP07 probes are kept.
Reading SeaChest is the slow part; every analysis afterwards is local.
"""
import glob
import os
import sys

import netCDF4
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_fleet2 import file_meta  # noqa: E402

F_MAX = 100.0
CHUNK = 250


def main():
    outdir = sys.argv[1]
    for spec in sys.argv[2:]:
        label, pattern = spec.split("=", 1)
        files = sorted(glob.glob(pattern))
        d = os.path.join(outdir, label)
        os.makedirs(d, exist_ok=True)
        for c0 in range(0, len(files), CHUNK):
            done_marker = os.path.join(d, f"chunk_{c0 // CHUNK:03d}.done")
            if os.path.exists(done_marker):
                continue
            # One bucket per frequency grid: instruments in one deployment can
            # differ in fs or fft length, and a file must never be dropped for it.
            buckets: dict = {}
            for f in files[c0:c0 + CHUNK]:
                try:
                    m = file_meta(f)
                    if m is None:
                        continue
                    with netCDF4.Dataset(f) as ds:
                        F = np.asarray(ds["F"][:, 0], float)
                        keep = F <= F_MAX
                        sig = (int(keep.sum()), round(float(F[1] - F[0]), 4))
                        S = np.asarray(ds["spec_gradT"][:2, keep, :], np.float32)
                        N = np.asarray(ds["spec_noise"][:2, keep, :], np.float32)
                        W = np.asarray(ds["speed"][:], float)
                        T = np.asarray(ds["T_mean"][:], float) if "T_mean" in ds.variables else np.full(W.size, np.nan)
                        P = np.asarray(ds["P_mean"][:], float) if "P_mean" in ds.variables else np.full(W.size, np.nan)
                        chi = np.asarray(ds["chi"][:2, :], float)
                except OSError:
                    continue
                b = buckets.setdefault(sig, {"F": F[keep], "S": [], "N": [], "W": [], "T": [], "P": [], "chi": [],
                                             "fi": [], "names": [], "inst": [], "pair": [], "t0": []})
                fidx = len(b["names"])
                b["names"].append(os.path.basename(f)); b["inst"].append(m["inst"])
                b["pair"].append("/".join(m["pair"])); b["t0"].append(m["t0"] if m["t0"] else np.nan)
                b["S"].append(np.moveaxis(S, 2, 0)); b["N"].append(np.moveaxis(N, 2, 0))
                b["W"].append(W); b["T"].append(T); b["P"].append(P); b["chi"].append(chi.T)
                b["fi"].append(np.full(W.size, fidx))
            for gi, (sig, b) in enumerate(sorted(buckets.items())):
                out = os.path.join(d, f"chunk_{c0 // CHUNK:03d}_g{gi}.npz")
                np.savez_compressed(
                    out, F=b["F"], S=np.concatenate(b["S"]), N=np.concatenate(b["N"]),
                    W=np.concatenate(b["W"]), T=np.concatenate(b["T"]), P=np.concatenate(b["P"]),
                    chi=np.concatenate(b["chi"]), fileidx=np.concatenate(b["fi"]),
                    names=np.array(b["names"]), inst=np.array(b["inst"]), pair=np.array(b["pair"]),
                    t0=np.array(b["t0"]))
                print(f"{label} chunk {c0 // CHUNK} grid {sig}: {len(b['names'])} files", flush=True)
            open(done_marker, "w").close()
        print(f"DONE {label}", flush=True)


if __name__ == "__main__":
    main()
