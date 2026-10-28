# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""Synthetic validation of `fp07-pair tau` and its identifiability rule.

usage: bead_tau_synth.py [OUT.csv] [--seeds N]

Writes synthetic chi products (spec_gradT, spec_noise, F, speed, start_time,
source with bead serials) for known single-pole beads, runs
odas_tpw.chi.fp07_bead_tau.measure on them exactly as on real products, and
prints, per case, the recovered taus with their 16-84% intervals and the rule's
verdict. The rule (fp07_bead_tau.RULE) was set from these cases and a
multi-seed coverage study (docs/fp07pair/runbook.md section 6, which records
which parts came after the first real measurement).

Each window draws its own speed (W_med x lognormal 5%), Batchelor wavenumber,
level and noise floor; the spectral SHAPE is deliberately non-universal (a
low-k excess and a mid-band dip that vary window to window), which the ratio
must not care about. Spectra carry chi-squared estimation noise at 9.5 DOF.
Profiles come in groups of 10 sharing one start_time, as perturb cuts them
from one .p file.
"""

from __future__ import annotations

import csv
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import netCDF4
import numpy as np

from odas_tpw.chi import fp07_bead_tau as B

DOF = 9.5
F = np.arange(0, 129) * 1.0  # Hz
T0 = datetime(2022, 3, 21, tzinfo=UTC).timestamp()


def shape(x, rng):
    """Kraichnan gradient shape with a per-window low-k excess and a mid-band dip."""
    g = x * np.exp(-np.sqrt(6 * 5.26) * x)
    amp = np.exp(rng.normal(np.log(3.0), 0.7))
    g = g * (1 + amp * (0.03 / np.maximum(x, 1e-4)) ** 1.5)
    return g * (1 - 0.4 * np.exp(-0.5 * (np.log(x / 0.3) / 0.4) ** 2))


def h2(f, tau, poles=1):
    return (1.0 + (2 * np.pi * f * tau) ** 2) ** (-poles)


def write_source(path, sn1, sn2, inst=194):
    cfg = (f"[instrument_info]\nsn = {inst}\n"
           f"[channel]\nname = T1\ntype = therm\nid = 4\nsn = {sn1}\n"
           f"[channel]\nname = T2\ntype = therm\nid = 5\nsn = {sn2}\n")
    with netCDF4.Dataset(path, "w") as ds:
        ds.configuration_string = cfg
    return str(path)


def write_files(d, *, taus, n_files=120, n_win=60, W_med=0.95, kB_med=70.0, snr=300.0,
                noise_true=(1.0, 1.0), gain2=1.1, poles=1, rng, taus2=None, per_start=10,
                df=1.0):
    """Synthetic products; *taus2* (if given) applies to the second half of the files.

    *df* is the FFT resolution [Hz]: 1 as on ARCTERX-2022, 2 as on SUNRISE.
    """
    F = np.arange(0, int(128 / df) + 1) * df  # shadows the module grid on purpose
    src = write_source(Path(d) / "src.nc", "T1592", "T2005")
    paths = []
    for i in range(n_files):
        t1, t2 = taus2 if (taus2 is not None and i >= n_files // 2) else taus
        S = np.zeros((2, F.size, n_win))
        N = np.zeros_like(S)
        W = W_med * np.exp(rng.normal(0, 0.05, n_win))
        for j in range(n_win):
            kB = np.exp(rng.normal(np.log(kB_med), 0.6))
            x = np.maximum(F / W[j] / kB, 1e-6)
            g = shape(x, rng)
            lvl = np.exp(rng.normal(0, 1.2))
            nz = np.full(F.size, np.max(g) / snr) * np.exp(rng.normal(0, 1.0))
            s = (W[j] / W_med) ** -0.5
            a = lvl * g * h2(F, t1 * s, poles) + noise_true[0] * nz
            b = gain2 * lvl * g * h2(F, t2 * s, poles) + noise_true[1] * nz
            S[0, :, j] = a * rng.chisquare(DOF, F.size) / DOF
            S[1, :, j] = b * rng.chisquare(DOF, F.size) / DOF
            N[:, :, j] = nz  # the MODEL, which the true noise may not equal
        p = Path(d) / f"c{i:04d}.nc"
        with netCDF4.Dataset(p, "w") as ds:
            ds.createDimension("probe", 2)
            ds.createDimension("freq", F.size)
            ds.createDimension("time", n_win)
            ds.createVariable("spec_gradT", "f8", ("probe", "freq", "time"))[:] = S
            ds.createVariable("spec_noise", "f8", ("probe", "freq", "time"))[:] = N
            ds.createVariable("F", "f8", ("freq", "time"))[:] = np.tile(F[:, None], (1, n_win))
            ds.createVariable("speed", "f8", ("time",))[:] = W
            pv = ds.createVariable("probe", str, ("probe",))
            pv[0], pv[1] = "T1_dT1", "T2_dT2"
            ds.source = src
            ds.start_time = datetime.fromtimestamp(T0 + 3600.0 * (i // per_start),
                                                   UTC).isoformat()
            ds.fp07_model = "single_pole"
        paths.append(str(p))
    return paths


CASES = [
    # name, kwargs
    ("matched 10/10", dict(taus=(0.010, 0.010))),
    ("near-matched 11/10", dict(taus=(0.011, 0.010))),
    ("12/10", dict(taus=(0.012, 0.010))),
    ("13/19.5 (A22 SN194-like)", dict(taus=(0.013, 0.0195))),
    ("15/10", dict(taus=(0.015, 0.010))),
    ("25/10", dict(taus=(0.025, 0.010))),
    ("glider 20/15, W 0.3", dict(taus=(0.020, 0.015), W_med=0.3, kB_med=150.0)),
    ("glider 30/12, W 0.3", dict(taus=(0.030, 0.012), W_med=0.3, kB_med=150.0)),
    ("13/19.5, noise 1.5x model on T2, S/N 300", dict(taus=(0.013, 0.0195),
                                                       noise_true=(1.0, 1.5))),
    ("13/19.5, noise 1.5x model on T2, S/N 60", dict(taus=(0.013, 0.0195), snr=60.0,
                                                      noise_true=(1.0, 1.5))),
    ("13/19.5, noise 0.67x model on T1, S/N 60", dict(taus=(0.013, 0.0195), snr=60.0,
                                                       noise_true=(0.67, 1.0))),
    ("13/19.5, low S/N 20", dict(taus=(0.013, 0.0195), snr=20.0)),
    ("13/19.5, 2 Hz FFT bins (SUNRISE-like)", dict(taus=(0.013, 0.0195), df=2.0, n_win=15)),
    ("13/19.5, 4 files", dict(taus=(0.013, 0.0195), n_files=4, per_start=1)),
    ("13/19.5, 8 files", dict(taus=(0.013, 0.0195), n_files=8, per_start=1)),
    ("13/19.5, 16 files", dict(taus=(0.013, 0.0195), n_files=16, per_start=2)),
    ("matched 10/10, 16 files", dict(taus=(0.010, 0.010), n_files=16, per_start=2)),
    ("failing bead (gain 0.3)", dict(taus=(0.013, 0.0195), gain2=0.3)),
    ("drift 10.8/17.7 -> 16.1/22.0", dict(taus=(0.0108, 0.0177), taus2=(0.0161, 0.0220))),
    ("double-pole truth 8/5 (SP fitted)", dict(taus=(0.008, 0.005), poles=2)),
]


def run(name, kw, seed, n_boot):
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory() as d:
        paths = write_files(d, rng=rng, **kw)
        res = B.measure(paths, instrument="194", first=T0 - 1, last=T0 + 1e7, n_boot=n_boot,
                        rng_seed=seed)
    (pt,) = res.pairs
    f = pt.all
    truth = kw["taus"]
    row = {"case": name, "seed": seed, "truth1_ms": truth[0] * 1e3, "truth2_ms": truth[1] * 1e3,
           "identified": f.identified, "reasons": "; ".join(f.reasons),
           "tau1_ms": f.tau_fit[0] * 1e3, "tau1_lo": f.tau_ci[0][0] * 1e3,
           "tau1_hi": f.tau_ci[0][1] * 1e3, "tau2_ms": f.tau_fit[1] * 1e3,
           "tau2_lo": f.tau_ci[1][0] * 1e3, "tau2_hi": f.tau_ci[1][1] * 1e3,
           "lnR20": f.ln_ratio[20.0], "lnR20_lo": f.ln_ratio_ci[20.0][0],
           "lnR20_hi": f.ln_ratio_ci[20.0][1], "flags": ",".join(pt.flags),
           "noise": ",".join("-" if v is None else f"{v:.2f}" for v in pt.noise),
           "n_files": f.n_files, "W_ref": pt.W_ref}
    if pt.halves is not None:
        for k, h in enumerate(pt.halves, start=1):
            row[f"h{k}_ident"] = h.identified
            row[f"h{k}_tau1"] = h.tau_fit[0] * 1e3
            row[f"h{k}_tau2"] = h.tau_fit[1] * 1e3
    return row


def main():
    args = sys.argv[1:]
    seeds = 1
    if "--seeds" in args:
        i = args.index("--seeds")
        seeds = int(args[i + 1])
        del args[i:i + 2]
    out = args[0] if args else None
    rows = []
    for name, kw in CASES:
        for seed in range(seeds):
            r = run(name, kw, seed, n_boot=200 if seeds == 1 else 100)
            rows.append(r)
            ok = "IDENTIFIED" if r["identified"] else "not identified"
            print(f"{name:42s} seed {seed}: truth {r['truth1_ms']:.1f}/{r['truth2_ms']:.1f} -> "
                  f"{r['tau1_ms']:.1f} [{r['tau1_lo']:.1f}-{r['tau1_hi']:.1f}] / "
                  f"{r['tau2_ms']:.1f} [{r['tau2_lo']:.1f}-{r['tau2_hi']:.1f}] ms; lnR'20 "
                  f"{r['lnR20']:+.3f} [{r['lnR20_lo']:+.3f},{r['lnR20_hi']:+.3f}]; {ok}"
                  + (f" ({r['reasons']})" if r["reasons"] else "")
                  + (f"; flags {r['flags']}" if r["flags"] else "")
                  + (f"; halves {r.get('h1_tau1', np.nan):.1f}/{r.get('h1_tau2', np.nan):.1f} -> "
                     f"{r.get('h2_tau1', np.nan):.1f}/{r.get('h2_tau2', np.nan):.1f}"
                     if "drift" in name else ""), flush=True)
    if out:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "case", k))
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
