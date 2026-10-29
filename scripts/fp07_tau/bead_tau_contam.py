# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""Unmodelled content on one probe, or a noise model off on both: tau bias vs the floor test.

usage: bead_tau_contam.py [OUT.csv] [--seeds N]

The evidence behind fp07_bead_tau's rule 7 (NOISE_FLOOR_MAX). Synthetic products
(bead_tau_synth.write_files: truth 13/19.5 ms, 48 files in 16 recordings) get,
on probe 2, content the noise model does not know about, of three shapes:
white, rising as f^2 above 10 Hz ("blue"), and a peak at 30 Hz inside the fit
band; or both probes' noise MODEL is scaled up. Per case: each tau's bias in
units of its own 16-84% half-width (z) and probe 2's floor ratio (observed /
model over 60-100 Hz in its quietest usable windows) as the module measures it.
"""

from __future__ import annotations

import csv
import math
import sys
import tempfile
from pathlib import Path

import netCDF4
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bead_tau_synth as S

from odas_tpw.chi import fp07_bead_tau as B

F = np.arange(0, 129) * 1.0
SHAPES = {
    "white": lambda a: a * np.ones_like(F),
    "blue": lambda a: a * np.clip((F - 10) / 50.0, 0, None) ** 2,
    "peak30": lambda a: a * np.exp(-0.5 * (np.log10(np.maximum(F, 1e-3) / 30.0) / 0.15) ** 2),
}
CASES = ([("none", 0.0)] + [("white", a) for a in (0.25, 0.5, 1.0)]
         + [("blue", a) for a in (0.3, 1.0, 3.0)] + [("peak30", a) for a in (1.0, 3.0, 10.0)]
         + [("model_high_both", m) for m in (2.0, 5.0)])


def run(shape, amp, seed):
    rng = np.random.default_rng(100 + seed)
    with tempfile.TemporaryDirectory() as d:
        paths = S.write_files(d, taus=(0.013, 0.0195), n_files=48, n_win=60, per_start=3,
                              rng=rng)
        for p in paths:
            with netCDF4.Dataset(p, "a") as ds:
                Sx, Nx = ds["spec_gradT"][:], ds["spec_noise"][:]
                if shape in SHAPES:
                    Sx[1] = Sx[1] + SHAPES[shape](amp)[:, None] * Nx[1]
                    ds["spec_gradT"][:] = Sx
                elif shape == "model_high_both":
                    ds["spec_noise"][:] = Nx * amp
        (pt,) = B.measure(paths, instrument="194", first=S.T0 - 1, last=S.T0 + 1e7,
                          n_boot=100, rng_seed=seed).pairs
    f = pt.all
    row = {"case": shape, "amp": amp, "seed": seed, "identified": f.identified,
           "floor1": pt.noise[0], "floor2": pt.noise[1]}
    for k, truth in enumerate((0.013, 0.0195), start=1):
        t, (lo, hi) = f.tau_fit[k - 1], f.tau_ci[k - 1]
        hw = 0.5 * math.log(hi / lo) if lo > 0 and hi > 0 else float("nan")
        row[f"tau{k}_ms"] = t * 1e3
        row[f"z{k}"] = math.log(t / truth) / hw if hw > 0 else float("nan")
    return row


def main():
    args = sys.argv[1:]
    seeds = 3
    if "--seeds" in args:
        i = args.index("--seeds")
        seeds = int(args[i + 1])
        del args[i:i + 2]
    rows = []
    for shape, amp in CASES:
        for seed in range(seeds):
            r = run(shape, amp, seed)
            rows.append(r)
            print(f"{shape:16s} {amp:5g} seed {seed}: floor2 {r['floor2']:.2f} "
                  f"tau {r['tau1_ms']:.1f}/{r['tau2_ms']:.1f} z {r['z1']:+.1f}/{r['z2']:+.1f} "
                  f"{'identified' if r['identified'] else 'refused'}", flush=True)
    if args:
        with open(args[0], "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
