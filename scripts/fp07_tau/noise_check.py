# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""Observed spectrum / noise model in the quietest windows, per instrument and probe.

Where the signal is negligible the ratio should be ~1 if the noise model is
right. usage: noise_check.py GLOB [quantile]
"""
import glob
import os
import re
import sys

import netCDF4
import numpy as np


def inst(path, ds):
    sn = getattr(ds, "instrument_sn", None) or getattr(ds, "sn", None)
    if sn:
        return f"SN{sn}"
    m = re.search(r"SN(\d+)", os.path.basename(path))
    return f"SN{m.group(1)}" if m else "SN?"


pattern = sys.argv[1]
q = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
rows = {}
for f in sorted(glob.glob(pattern)):
    try:
        ds = netCDF4.Dataset(f)
    except OSError:
        continue
    with ds:
        if "spec_gradT" not in ds.variables:
            continue
        S = np.asarray(ds["spec_gradT"][:], float)
        N = np.asarray(ds["spec_noise"][:], float)
        F = np.asarray(ds["F"][:], float)
        chi = np.asarray(ds["chi"][:], float)
        key = inst(f, ds)
    for p in range(S.shape[0]):
        for j in range(S.shape[2]):
            if not np.isfinite(chi[p, j]):
                continue
            rows.setdefault((key, p), []).append((chi[p, j], S[p, :, j] / N[p, :, j], F[:, j]))
fq = (20, 30, 40, 50, 60, 70, 80, 90, 100)
print(f"windows in the lowest {q:.0%} of chi; median obs/model at f [Hz]")
for (key, p), lst in sorted(rows.items()):
    chis = np.array([c for c, _, _ in lst])
    cut = np.quantile(chis, q)
    sel = [r for c, r, _ in lst if c <= cut]
    Fs = lst[0][2]
    med = np.nanmedian(np.array([r for r in sel if r.shape == Fs.shape]), axis=0)
    cells = " ".join(f"{fv}:{med[np.argmin(abs(Fs - fv))]:6.2f}" for fv in fq)
    print(f"  {key} probe {p + 1}: n={len(sel):6d}  {cells}")
