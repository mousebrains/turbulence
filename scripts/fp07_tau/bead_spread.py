"""Model-free bead-to-bead spread from paired-bead normalized ratios.

For each INDEPENDENT bead pair (one row per physical pair; first deployment
kept when a pair recurs), ln nr(f) = ln of the T2/T1 power ratio at f relative
to 1-2 Hz. If beads scatter independently with log-response SD sigma(f) at f,
Var(ln nr) = 2 sigma^2, so sigma = SD(ln nr)/sqrt(2). Pairs whose serials are
unknown ("?" or "T") are kept but counted once per (deployment, instrument).
"""
import csv
import math
import sys

import numpy as np

rows = [r for f in sys.argv[1:] for r in csv.DictReader(open(f)) if r["half"] == "all"]
seen, use = set(), []
for r in rows:
    key = r["pair"] if all(s not in ("?", "T", "") for s in r["pair"].split("/")) else (r["deployment"], r["instrument"], r["pair"])
    k2 = tuple(sorted(r["pair"].split("/"))) if isinstance(key, str) else key
    if k2 in seen: continue
    seen.add(k2); use.append(r)
print(f"independent pairs: {len(use)}")
rng = np.random.default_rng(0)
for q in (5, 10, 20, 30, 40, 60):
    v = np.array([float(r.get(f"nr{q}") or "nan") for r in use]); v = np.log(v[np.isfinite(v) & (v > 0)])
    if v.size < 4: continue
    sig = v.std(ddof=1) / math.sqrt(2)
    b = [rng.choice(v, v.size).std(ddof=1) / math.sqrt(2) for _ in range(2000)]
    lo, hi = np.percentile(b, [16, 84])
    print(f"  {q:2d} Hz: n={v.size:2d}  |ln nr| median {np.median(abs(v)):.2f}  sigma_bead (ln power) {sig:.2f} [{lo:.2f}-{hi:.2f}]  -> x{math.exp(sig):.2f} (1 sd)")
