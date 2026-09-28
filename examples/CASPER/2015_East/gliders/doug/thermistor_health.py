#!/usr/bin/env python3
"""Per-file FP07 health for the CASPER MicroRiders, resumable.

A failing FP07 rails: the trace pins near a fixed value and its standard
deviation collapses while the sibling thermistor keeps tracking. Records
min/max/sd and the fraction of samples within 0.05 C of the file max (the
rail), for T1 and T2, one row per file, flushed as it goes.
"""
import csv, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
from odas_tpw.rsi.p_file import PFile

src, out = sys.argv[1], sys.argv[2]
done = set()
if os.path.exists(out):
    with open(out) as fh:
        done = {r["file"] for r in csv.DictReader(fh)}

files = sorted(f for f in os.listdir(src) if f.endswith(".P"))
new = not os.path.exists(out) or os.path.getsize(out) == 0
fh = open(out, "a", newline=""); w = csv.writer(fh)
if new:
    w.writerow(["file", "start_utc", "n_slow",
                "T1_min", "T1_max", "T1_sd", "T1_frac_at_rail",
                "T2_min", "T2_max", "T2_sd", "T2_frac_at_rail",
                "P_max"]); fh.flush()

for i, f in enumerate(files, 1):
    stem = f[:-2]
    if stem in done:
        continue
    try:
        pf = PFile(os.path.join(src, f))
        row = [stem, pf.start_time.isoformat(), int(np.size(pf.t_slow))]
        for ch in ("T1", "T2"):
            x = np.asarray(pf.channels[ch], float)
            x = x[np.isfinite(x)]
            if x.size == 0:
                row += ["", "", "", ""]; continue
            mx = float(x.max())
            row += ["%.4f" % x.min(), "%.4f" % mx, "%.5f" % x.std(),
                    "%.5f" % float(np.mean(x > mx - 0.05))]
        P = np.asarray(pf.channels["P"], float)
        row.append("%.2f" % np.nanmax(P))
        w.writerow(row); fh.flush()
        print("[%d/%d] %s" % (i, len(files), stem), flush=True)
    except Exception as exc:
        w.writerow([stem, "ERROR", str(exc)] + [""] * 9); fh.flush()
        print("[%d/%d] %s FAILED: %s" % (i, len(files), stem, exc), flush=True)
