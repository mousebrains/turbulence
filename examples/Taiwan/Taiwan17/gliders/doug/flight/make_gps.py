#!/usr/bin/env python3
"""Drift-corrected track for glider doug (Taiwan 2017) -> gps.nc, on the MR clock.

    python make_gps.py

Same algorithm as Taiwan13's
`gliders/20130526_husker/hotel/make_corrected_track.py` (Pat's): the glider
dead-reckons underwater and cannot see the depth-averaged current, so DR error
grows through a dive; at the next surfacing GPS is truth and the closing error
is redistributed LINEARLY IN TIME back to the previous fix. Read that file for
why "strictly before" and "last valid DR sample" matter.

Differences here:
  * Source is `../../20170216_doug_Taiwan/post-recovery/flight/logs/*.dbd`.
  * The output is written **on the MicroRider's clock** (glider time minus
    L(t) from mr_clock_model.json), because doug's MR ran 9 min 21 s fast.
    perturb matches a profile's time against this file, so a track left on the
    glider clock would put every position ~170 m away (0.3 m/s x 561 s).

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
LOGS = HERE.parent.parent / "20170216_doug_Taiwan" / "post-recovery" / "flight" / "logs"
CACHE = HERE / "cache"
OUT = HERE / "gps.nc"
NO_FIX = 69696969.0                      # Slocum's "no GPS fix" sentinel
T_LO, T_HI = 1487116800.0, 1488412800.0  # 2017-02-15 .. 2017-03-02
# Working-area box. The FIRST flight sample (2017-02-16 06:38:50, the deck
# STATUS.MI) still carries the glider's stored position from OSU in Corvallis,
# Oregon (44.5583 N, -123.2840 E) -- one sample, 10,000 km away. Anything
# outside this box is dropped rather than fed to the drift correction.
LAT_LO, LAT_HI, LON_LO, LON_HI = 19.0, 25.0, 117.0, 122.0


def ddmm_to_deg(x):
    """Slocum ddmm.mmmm -> decimal degrees (sign preserved)."""
    s, a = np.sign(x), np.abs(x)
    d = np.floor(a / 100.0)
    return s * (d + (a - 100.0 * d) / 60.0)


def lag_model(t):
    m = json.loads((HERE / "mr_clock_model.json").read_text())["blocks"]
    L = np.full(np.shape(t), np.nan)
    for b in m.values():
        sel = (t < b["split_epoch"]) if b["t_max_epoch"] < b["split_epoch"] else (t >= b["split_epoch"])
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
        attrs={"title": "Glider doug drift-corrected track on the MicroRider clock, Taiwan 2017",
               "source": "m_lat/m_lon/m_gps_lat/m_gps_lon from post-recovery flight dbd",
               "comment": "Algorithm: Taiwan13 gliders/20130526_husker/hotel/make_corrected_track.py (Pat's). "
                          "Times shifted onto the MR clock with mr_clock_model.json.",
               "Conventions": "CF-1.13"})
    ds.to_netcdf(OUT)
    print(f"wrote {OUT.name}: {ds.sizes['time']:,} positions, "
          f"lat {float(ds.lat.min()):.4f}..{float(ds.lat.max()):.4f}, lon {float(ds.lon.min()):.4f}..{float(ds.lon.max()):.4f}")


if __name__ == "__main__":
    main()
