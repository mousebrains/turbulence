"""Print the analyze_cache.py CSV as a compact table (all-halves rows by default)."""
import csv
import math
import sys

rows = list(csv.DictReader(open(sys.argv[1])))
halves = sys.argv[2].split(",") if len(sys.argv) > 2 else ["all"]
g = lambda r, k: float(r.get(k) or "nan")
def ci(r, k, fmt="{:.2f}"):
    v, lo, hi = g(r, k), g(r, k + "_lo"), g(r, k + "_hi")
    return (fmt.format(v) + " [" + fmt.format(lo) + "-" + fmt.format(hi) + "]") if not math.isnan(v) else "-"
print(f"{'deployment':12s} {'inst':6s} {'pair':13s} {'h':3s} {'win':>6s} {'W':>4s} {'T':>4s} | {'alpha [16-84]':>18s} | {'fc1 / fc2 [Hz]':>34s} | {'nr20 (T2/T1 @20Hz vs 1-2Hz)':>26s} | {'SP 1-25 ms':>11s} | rms free/SP")
for r in rows:
    if r["half"] not in halves: continue
    print(f"{r['deployment']:12s} {r['instrument']:6s} {r['pair']:13s} {r['half']:3s} {int(float(r['windows'])):6d} {g(r,'W'):4.2f} {g(r,'T'):4.1f} | "
          f"{ci(r,'fs_alpha'):>18s} | {ci(r,'fs_fc1','{:.0f}'):>16s} / {ci(r,'fs_fc2','{:.0f}'):<16s}| {ci(r,'nr20'):>26s} | "
          f"{g(r,'b25_p1_tau1'):4.1f}/{g(r,'b25_p1_tau2'):4.1f}  | {g(r,'fs_rms'):.3f}/{g(r,'full_p1_rms'):.3f}")
