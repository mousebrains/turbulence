#!/usr/bin/env python3
"""Generate sections.yaml for ASTRAL 2023 -- one section per deployment.

Unlike ARCTERX, no geometric splitting is needed: each `.p` file IS a section.
Measured on the products, every deployment is a short, straight westward drift
(net displacement == path length for all 13, bearings 251-280 deg, 1-3 h,
1.5-6.5 km, 10-20 casts), separated from the next by 9-22 h. There are no
heading reversals to cut on and no long lines to split.

A deployment is also exactly one SHEAR PROBE CONFIGURATION, so each section
carries its probe pair -- which matters here because the probes changed six
times and one of them (M1196, D10) is unusable.
"""
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

V = Path(__file__).resolve().parent
R = 6371.0
BAD_PROBES = {"M1196"}  # implied true sens ~0.023, below the whole fleet range

probes = {}
for f in sorted(V.glob("*.p")):
    from odas_tpw.rsi.p_file import parse_config, read_config_string

    ch = {c.get("name"): c for c in parse_config(read_config_string(f))["channels"]}
    dep = f"D{int(re.search(r'_D(\d+)_', f.name).group(1)):02d}"
    probes[dep] = (ch["sh1"]["sn"], ch["sh2"]["sn"])

prof_dirs = sorted(glob.glob(str(V / "Processed" / "profiles_[0-9]*")))
if not prof_dirs:
    raise SystemExit("no Processed/profiles_NN -- run the pipeline first")

casts = {}
for f in sorted(glob.glob(prof_dirs[-1] + "/*.nc")):
    m = re.search(r"_D(\d+)_", f)
    if not m:
        continue
    dep = f"D{int(m.group(1)):02d}"
    with xr.open_dataset(f) as ds:
        casts.setdefault(dep, []).append(
            (
                pd.Timestamp(np.atleast_1d(ds["stime"].values).ravel()[0]),
                pd.Timestamp(np.atleast_1d(ds["etime"].values).ravel()[-1]),
                float(np.nanmedian(ds["lat"].values)),
                float(np.nanmedian(ds["lon"].values)),
                float(np.nanmax(ds["P"].values)),
            )
        )

lines = ['''# Plotting sections — ASTRAL 2023 (VMP250IR-DL3 SN 92, Arabian Sea)
#
# Consumed by perturb-plot via --sections, e.g.
#   perturb-plot scalar --root <VMP>/Processed --sections <VMP>/sections.yaml \\
#       --out-dir figs/ --var JAC_T --var SP --var sigma0
#
# ONE SECTION = ONE DEPLOYMENT (.p file). No geometric splitting is needed:
# every deployment is a short, STRAIGHT westward drift — net displacement
# equals path length for all 13, bearings 251–280°, 1–3 h, 1.5–6.5 km — and
# consecutive deployments are 9–22 h apart. There are no heading reversals to
# cut on, which is what ARCTERX needed.
#
# WHY xaxis = signed_distance, measured FROM THE MIDPOINT. Every deployment is
# a single-direction drift, so the along-track coordinate is unambiguous:
# `signed_distance_axis` fits the principal axis by total least squares THROUGH
# THE CENTROID and reports distance from it, earliest sample negative. Sections
# run 1.5–6.5 km, so values span roughly ±0.8 to ±3.3 km.
#
# Two consequences of every section running the SAME way (bearings 251–280°):
# the sign convention is consistent between sections — negative is always the
# eastern, earlier end — and the axes are directly comparable in extent and
# orientation. They are NOT comparable in absolute position: each section's
# origin is its OWN midpoint, so x = 0 is a different place in each.
#
# `time` remains defensible: these are drift stations at 0.9–2.9 km/h, so the
# displacement is partly incidental advection. Each comment carries the hours,
# bearing and net distance to support switching back.
#
# EACH SECTION IS ALSO ONE PROBE CONFIGURATION, named below. That matters: the
# shear probes were changed six times, and D10's sh1 (M1196) is UNUSABLE — its
# implied true sensitivity is below the entire fleet range. See
# calibration/README.md.
#
# Regenerate with make_sections.py rather than hand-editing.

sections:

  # ---- whole programme ----------------------------------------------------
  - name: full_timeseries
    xaxis: {method: time}
''']

for dep in sorted(casts):
    v = sorted(casts[dep])
    t0, t1 = v[0][0], v[-1][1]
    lat = np.array([r[2] for r in v])
    lon = np.array([r[3] for r in v])
    x = np.radians(lon) * R * np.cos(np.radians(lat.mean()))
    y = np.radians(lat) * R
    net = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    brg = float(np.degrees(np.arctan2(x[-1] - x[0], y[-1] - y[0])) % 360)
    hrs = (t1 - t0).total_seconds() / 3600
    s1, s2 = probes[dep]
    flag = "  ** sh1 UNUSABLE (see calibration/README.md) **" if s1 in BAD_PROBES else ""
    lines.append(
        f"""
  # {len(v)} casts, {hrs:.1f} h, {net:.1f} km on bearing {brg:.0f} deg, to {max(r[4] for r in v):.0f} dbar
  # probes: sh1 {s1}, sh2 {s2}{flag}
  - name: {dep}_{s1}_{s2}
    start: "{(t0 - pd.Timedelta('2min')).strftime('%Y-%m-%dT%H:%M:%SZ')}"
    stop:  "{(t1 + pd.Timedelta('2min')).strftime('%Y-%m-%dT%H:%M:%SZ')}"
    xaxis: {{method: signed_distance, units: km}}"""
    )

out = V / "sections.yaml"
out.write_text("\n".join(lines) + "\n")
n = sum(len(v) for v in casts.values())
print(f"{len(casts)} deployment sections -> {out}")
print(f"casts covered: {n}/{n} (100.0%) — a section is a whole file, so none can fall outside")
