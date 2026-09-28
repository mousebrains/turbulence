#!/usr/bin/env python3
"""Drift-corrected track for glider doug (CASPER-East 2015) -> gps.nc, on the MR clock.

    python make_gps.py

Same algorithm as Taiwan13's
`gliders/20130526_husker/hotel/make_corrected_track.py` (Pat's): the glider
dead-reckons underwater and cannot see the depth-averaged current, so DR error
grows through a dive; at the next surfacing GPS is truth and the closing error
is redistributed LINEARLY IN TIME back to the previous fix. Read that file for
why "strictly before" and "last valid DR sample" matter.

Differences here:
  * Source is `../post-recovery/flight/logs/*.dbd`.
  * The output is written **on the MicroRider's clock** (glider time minus
    L(t) from mr_clock_model.json). The 2015 MR clock was nearly right
    (-64 s, then +8 s, then -8 s) but it JUMPED TWICE, so no single offset
    works: +73.5 s across the probe-change gap and -12.75 s on 10-28.

perturb reads it as:

    gps: {source: "netcdf", file: "<...>/flight/gps.nc", time_col: "time",
          lat_col: "lat", lon_col: "lon", max_time_diff: 600}
"""
from __future__ import annotations

import glob
import json
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")
from odas_tpw.dinkum.reader import load_dinkum  # noqa: E402

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent / "post-recovery" / "flight" / "logs"
CACHE = HERE / "cache"
OUT = HERE / "gps.nc"
NO_FIX = 69696969.0                      # Slocum's "no GPS fix" sentinel
T_LO, T_HI = 1444262400.0, 1446940800.0  # 2015-10-08 .. 2015-11-08
# Working-area box, the shelf off Duck, North Carolina. Real fixes span
# 35.99-36.84 N, -76.27 to -74.98 E. The margin catches transit and any stored
# pre-deployment position without admitting a stale one (OSU Corvallis, which
# polluted the Taiwan run, is 44.6 N -123.3 E and far outside).
LAT_LO, LAT_HI, LON_LO, LON_HI = 35.0, 37.5, -77.5, -74.0


def ddmm_to_deg(x):
    """Slocum ddmm.mmmm -> decimal degrees (sign preserved)."""
    s, a = np.sign(x), np.abs(x)
    d = np.floor(a / 100.0)
    return s * (d + (a - 100.0 * d) / 60.0)


def lag_model(t):
    m = json.loads((HERE / "mr_clock_model.json").read_text())["blocks"]
    L = np.full(np.shape(t), np.nan)
    for b in m.values():
        # Each block carries its own half-open selection window. The old
        # single-"split_epoch" form could only express two blocks and silently
        # mis-assigned a third; refuse it rather than guess.
        if "sel_lo_epoch" not in b:
            raise SystemExit("mr_clock_model.json predates per-block sel_lo/sel_hi; re-run mr_clock.py solve")
        lo = -np.inf if b["sel_lo_epoch"] is None else b["sel_lo_epoch"]
        hi = np.inf if b["sel_hi_epoch"] is None else b["sel_hi_epoch"]
        sel = (t >= lo) & (t < hi)
        L[sel] = b["lag_at_ref_s"] + b["drift_s_per_day"] * (t[sel] - b["t_ref_epoch"]) / 86400.0
    return L


def main() -> None:
    want = ["m_present_time", "m_lat", "m_lon", "m_gps_lat", "m_gps_lon"]
    cols: dict[str, list] = {w: [] for w in want}
    for p in sorted(glob.glob(str(LOGS / "*.dbd"))):
        try:
            ds = load_dinkum([p], cache=str(CACHE), sensors=want)
        except Exception:
            continue
        if not ds.sizes:
            continue
        n = ds.sizes["record"]
        for w in want:
            cols[w].append(np.asarray(ds[w].values, float) if w in ds else np.full(n, np.nan))
    f = {w: np.concatenate(v) for w, v in cols.items() if v}
    order = np.argsort(f["m_present_time"])
    f = {w: v[order] for w, v in f.items()}

    t = f["m_present_time"]
    keep = np.isfinite(t) & (t > T_LO) & (t < T_HI)
    t = t[keep]
    lat_dr, lon_dr = ddmm_to_deg(f["m_lat"][keep]), ddmm_to_deg(f["m_lon"][keep])
    g_lat_raw, g_lon_raw = f["m_gps_lat"][keep], f["m_gps_lon"][keep]
    good_fix = (np.isfinite(g_lat_raw) & np.isfinite(g_lon_raw)
                & (g_lat_raw != NO_FIX) & (g_lon_raw != NO_FIX) & (np.abs(g_lat_raw) > 1.0))
    g_lat, g_lon = ddmm_to_deg(g_lat_raw), ddmm_to_deg(g_lon_raw)
    in_area = ((lat_dr > LAT_LO) & (lat_dr < LAT_HI) & (lon_dr > LON_LO) & (lon_dr < LON_HI))
    n_out = int((np.isfinite(lat_dr) & np.isfinite(lon_dr) & ~in_area).sum())
    if n_out:
        print(f"dropped {n_out} DR position(s) outside the working area (stale pre-deployment fix)")
    have_dr = np.isfinite(lat_dr) & np.isfinite(lon_dr) & in_area
    good_fix &= (g_lat > LAT_LO) & (g_lat < LAT_HI) & (g_lon > LON_LO) & (g_lon < LON_HI)

    at_surf = good_fix & have_dr
    edges = np.flatnonzero(np.diff(at_surf.astype(int)) != 0) + 1
    runs = np.split(np.arange(t.size), edges)
    surf_runs = [r for r in runs if r.size and at_surf[r[0]]]
    print(f"flight samples: {t.size:,}   DR positions: {have_dr.sum():,}   surface intervals: {len(surf_runs):,}")

    lat_c, lon_c = lat_dr.copy(), lon_dr.copy()
    corrected = np.zeros(t.size, bool)
    drifts, hours = [], []
    for s_k, s_next in zip(surf_runs[:-1], surf_runs[1:]):
        a_i, b_i = s_k[-1], s_next[0]
        dive = np.arange(a_i, b_i)
        dive = dive[have_dr[dive] & ~at_surf[dive]]
        if dive.size < 3:
            continue
        e_i = dive[-1]
        t0, t1 = t[a_i], t[e_i]
        if not (t1 > t0) or (t1 - t0) > 12 * 3600:
            continue
        w = (t[a_i:e_i + 1] - t0) / (t1 - t0)
        seg = np.arange(a_i, e_i + 1)
        m = have_dr[seg]
        lat_c[seg[m]] = lat_dr[seg[m]] + (g_lat[b_i] - lat_dr[e_i]) * w[m]
        lon_c[seg[m]] = lon_dr[seg[m]] + (g_lon[b_i] - lon_dr[e_i]) * w[m]
        corrected[seg[m]] = True
        km = np.hypot((g_lat[b_i] - lat_dr[e_i]) * 111.0,
                      (g_lon[b_i] - lon_dr[e_i]) * 111.0 * np.cos(np.deg2rad(g_lat[b_i])))
        drifts.append(km * 1000.0)
        hours.append((t1 - t0) / 3600.0)

    d, h = np.array(drifts), np.array(hours)
    print(f"dive segments corrected: {d.size}")
    if d.size:
        print(f"  closing drift m : median {np.median(d):8.1f}  p90 {np.percentile(d, 90):8.1f}  max {d.max():8.1f}")
        print(f"  implied current m/s: median {np.median(d / (h * 3600)):.3f}  p90 {np.percentile(d / (h * 3600), 90):.3f}")
    print(f"  samples corrected: {corrected.sum():,} of {have_dr.sum():,}")

    ok = have_dr & np.isfinite(t)
    t_mr = t[ok] - lag_model(t[ok])          # glider_time = MR_time + L
    o = np.argsort(t_mr)
    ds = xr.Dataset(
        {"lat": ("time", lat_c[ok][o], {"units": "degrees_north", "standard_name": "latitude",
                                        "comment": "dead-reckoned m_lat, drift redistributed linearly in time between GPS fixes"}),
         "lon": ("time", lon_c[ok][o], {"units": "degrees_east", "standard_name": "longitude",
                                        "comment": "dead-reckoned m_lon, same correction"}),
         "lat_corrected": ("time", corrected[ok][o].astype("int8"),
                           {"long_name": "sample lies inside a GPS-bracketed segment",
                            "comment": "0 where the track is raw DR (before the first fix or after the last)"}),
         "glider_time": ("time", t[ok][o], {"units": "seconds since 1970-01-01T00:00:00+00:00",
                                            "long_name": "doug flight-computer time (m_present_time)"})},
        coords={"time": ("time", t_mr[o], {"units": "seconds since 1970-01-01T00:00:00+00:00",
                                           "calendar": "standard", "standard_name": "time", "axis": "T",
                                           "long_name": "time on the MicroRider SN 134 CLOCK (glider time minus mr_clock_lag)"})},
        attrs={"title": "Glider doug drift-corrected track on the MicroRider clock, CASPER-East 2015",
               "source": "m_lat/m_lon/m_gps_lat/m_gps_lon from post-recovery flight dbd",
               "comment": "Algorithm: Taiwan13 gliders/20130526_husker/hotel/make_corrected_track.py (Pat's). "
                          "Times shifted onto the MR clock with mr_clock_model.json.",
               "Conventions": "CF-1.13"})
    ds.to_netcdf(OUT)
    print(f"wrote {OUT.name}: {ds.sizes['time']:,} positions, "
          f"lat {float(ds.lat.min()):.4f}..{float(ds.lat.max()):.4f}, lon {float(ds.lon.min()):.4f}..{float(ds.lon.max()):.4f}")


if __name__ == "__main__":
    main()
