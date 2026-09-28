#!/usr/bin/env python3
"""Campaign summary across all nine SUNRISE deployments.

Reads each deployment's newest COMBO generation -- one file per stage, not the
thousands of per-profile files the first version walked (which was minutes of
SMB round-trips for an identical answer).

Reports MEDIAN and log10-MAD, never mean/std: epsilon and chi are lognormal-ish
and heavy-tailed, and a mean is dominated by the tail.

THE FLAG THAT CHANGES THE ANSWER. ``chiQCFallback == 1`` means NO probe passed
QC in that window -- the chi value is a fallback, not a measurement. On SUNRISE
that is 18-59% of bins, because eps here is ~2e-8 and kB ~165 cpm so K_max/kB
is only ~0.49 and the Batchelor rolloff is never resolved. Including those bins
biases the campaign chi HIGH by 1.3-2.0x. Both numbers are printed, because the
QC-passing subset is not a random sample of the water column -- it is biased
toward the resolvable (weaker-chi, better-fom) windows, so neither column alone
is the truth.

Probe-pair ratios are the internal consistency check and constrain only the
RELATIVE scale. They are computed pairwise on windows where BOTH probes are
good, never from the two medians -- a dead probe in a subset of profiles would
otherwise shift the two medians independently and fake a sensitivity error.
"""
from __future__ import annotations

import glob
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")

BASE = Path("/Volumes/SeaChest/SUNRISE/Data")
# Which product tree to summarise: VMP_results (top-down) or VMP_results_bbl
# (bottom-up). Both are built from the same raw files and the same probes;
# only the window layout and the bottom datum differ, so the two columns are
# directly comparable.
TREE = sys.argv[1] if len(sys.argv) > 1 else "VMP_results"
DEPS = [
    (2019, "Pelican", 142), (2019, "Pelican", 194),
    (2021, "Pelican", 194), (2021, "Pelican", 412),
    (2021, "WaltonSmith", 142), (2021, "WaltonSmith", 194),
    (2022, "Pelican", 142), (2022, "Pelican", 412),
    (2022, "PointSur", 194),
]


def logmad(a):
    """MAD in log10 space -- the right spread for a lognormal quantity."""
    a = np.asarray(a, float)
    a = a[np.isfinite(a) & (a > 0)]
    if a.size < 3:
        return np.nan
    l = np.log10(a)
    return float(np.median(np.abs(l - np.median(l))))


def newest_combo(res: Path, stage: str) -> Path | None:
    """Newest ``<stage>_NN`` -- an exact match, so ``diss_combo_*`` never
    matches ``diss_combo_binned_*`` and sort order never picks the wrong tree."""
    pat = re.compile(r"%s_\d+$" % re.escape(stage))
    dirs = sorted(d for d in glob.glob(str(res / (stage + "_*"))) if pat.search(d))
    if not dirs:
        return None
    f = Path(dirs[-1]) / "combo.nc"
    return f if f.exists() else None


def col(ds, name):
    """(profile, bin) orientation, flattened."""
    v = np.asarray(ds[name].values, float)
    if v.ndim == 2 and v.shape[0] != ds.sizes["profile"]:
        v = v.T
    return v.ravel()


def med(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a) & (a > 0)]
    return float(np.median(a)) if a.size else np.nan


def ratio(a, b):
    """Median pairwise ratio on windows where BOTH are good."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    return (float(np.median(a[m] / b[m])), int(m.sum())) if m.sum() > 10 else (np.nan, 0)


print("tree: %s\n" % TREE)
hdr = ("%-22s %7s %7s %10s %6s %10s %10s %6s %6s %8s" %
       ("deployment", "prof", "eps n", "eps med", "lMAD", "chi all", "chi QCok",
        "lMAD", "fb %", "Gamma"))
print(hdr)
print("-" * len(hdr))

rows = {}
for (y, s, sn) in DEPS:
    d = BASE / str(y) / s / "VMP" / "analysis" / ("SN%d" % sn)
    tag = "%d %s SN%d" % (y, s, sn)
    fe = newest_combo(d / TREE, "diss_combo")
    fc = newest_combo(d / TREE, "chi_combo")
    if fe is None:
        print("%-22s  (not processed)" % tag)
        continue
    de = xr.open_dataset(fe)
    eps = col(de, "epsilonMean")
    e1, e2 = col(de, "e_1"), col(de, "e_2")
    nprof = de.sizes["profile"]
    de.close()

    chi_all = chi_ok = gam = np.nan
    c1 = c2 = np.array([])
    fbpct = np.nan
    lmc = np.nan
    if fc is not None:
        dc = xr.open_dataset(fc)
        cm = col(dc, "chiMean")
        fb = col(dc, "chiQCFallback")
        c1, c2 = col(dc, "chi_1"), col(dc, "chi_2")
        gam = med(col(dc, "Gamma"))
        good = np.isfinite(cm) & (cm > 0)
        chi_all, chi_ok = med(cm[good]), med(cm[good & (fb == 0)])
        lmc = logmad(cm[good & (fb == 0)])
        fbpct = 100.0 * float(np.mean(fb[good] != 0)) if good.any() else np.nan
        dc.close()

    rows[tag] = (e1, e2, c1, c2)
    print("%-22s %7d %7d %10.3e %6.2f %10.3e %10.3e %6.2f %5.0f%% %8.3f" %
          (tag, nprof, int((np.isfinite(eps) & (eps > 0)).sum()),
           med(eps), logmad(eps), chi_all, chi_ok, lmc, fbpct, gam))

print("\n--- probe-pair consistency (RELATIVE scale only; both-good windows) ---")
print("%-22s %16s %8s %16s %8s" % ("deployment", "eps sh1/sh2", "n", "chi T1/T2", "n"))
for tag, (e1, e2, c1, c2) in rows.items():
    er, en = ratio(e1, e2)
    cr, cn = ratio(c1, c2)
    print("%-22s %16s %8d %16s %8d" % (
        tag, ("%.3f" % er) if np.isfinite(er) else "-", en,
        ("%.3f" % cr) if np.isfinite(cr) else "-", cn))
