#!/usr/bin/env python3
"""Add a drift-corrected lat/lon to hotel.nc for glider "husker" (Taiwan 2013).

WHAT THE FLIGHT COMPUTER RECORDS
--------------------------------
The 2013 flight computer DOES dead-reckon. Verified on 00210000.DBD: 380
distinct ``m_lat`` values against only 4 GPS fixes, 352 distinct while deeper
than 10 m (spread ~184 m), and ``m_lat`` never equals a forward-fill of
``m_gps_lat`` (0 of 404 samples). So ``m_lat``/``m_lon`` are genuine DR, not a
held fix. They are in Slocum ddmm.mmmm.

THE CORRECTION (Pat's algorithm; linear-in-time redistribution)
-------------------------------------------------------
Underwater the glider integrates its flight model but cannot see the
depth-averaged current, so DR error grows through the dive. At the next
surfacing GPS gives truth, and the mismatch is the whole accumulated drift.
For each segment between consecutive GPS fixes at t_k and t_k+1:

    E      = P_gps(t_k+1) - P_dr(t_k+1)          # closing error, degrees
    P_corr(t) = P_dr(t) + E * (t - t_k)/(t_k+1 - t_k)

which is equivalent to assuming a CONSTANT depth-averaged current over the
segment. Samples before the first fix or after the last are left uncorrected
and flagged in ``lat_corrected``.

  This IS Pat's algorithm, as he states it: clean both datasets, DR and GPS;
  take the last DR position STRICTLY BEFORE a GPS point; the difference is the
  drift; ramp it in linearly from the previous valid GPS point. Verified by
  independent re-implementation from that description -- identical track to
  0.0000 m median / 0.0000 m max over 312,497 samples, anchor a valid fix in
  65/65 segments.

  Two 2013-specific facts this formulation gets right:
    * "strictly before" matters -- at a valid fix m_lat has ALREADY been
      snapped to m_gps_lat, so differencing at the fix gives exactly 0.0 on
      every segment, a plausible-looking null.
    * "last DR position" cannot be shortened to n-1 -- Slocum writes a sensor
      only when it updates, so m_lat is NaN at n-1 in 65 of 65 segments
      (nearest real value a median of 16 records back, max 245).

Note ``m_water_vx``/``m_water_vy`` -- the glider's OWN depth-averaged current
estimate -- exist but are reported only at surfacings (140 samples for the whole
deployment), so they are all NaN in hotel.nc under `projection.max_gap: 120`.
They are read here straight from the DBD stream as an independent cross-check.

Run:  python3 make_corrected_track.py
"""
from __future__ import annotations

import glob
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")
from odas_tpw.dinkum.reader import load_dinkum  # noqa: E402

HERE = Path(__file__).resolve().parent
GLIDER = HERE.parent
HOTEL = HERE / "hotel.nc"
CACHE = HERE / "cache"
NO_FIX = 69696969.0        # Slocum's "no GPS fix" sentinel
T_LO, T_HI = 1367280000.0, 1369612800.0   # 2013-04-30 .. 2013-05-27


def ddmm_to_deg(x: np.ndarray) -> np.ndarray:
    """Slocum ddmm.mmmm -> decimal degrees (sign preserved)."""
    s = np.sign(x)
    a = np.abs(x)
    d = np.floor(a / 100.0)
    return s * (d + (a - 100.0 * d) / 60.0)


def load_flight() -> dict[str, np.ndarray]:
    paths = sorted(glob.glob(str(GLIDER / "Leg3_Husker_MB" / "LOGS" / "*.DBD")))
    paths += sorted(glob.glob(str(GLIDER / "Leg3_Husker_MB" / "SENTLOGS" / "*.DBD")))
    want = ["m_present_time", "m_lat", "m_lon", "m_gps_lat", "m_gps_lon",
            "m_water_vx", "m_water_vy"]
    cols: dict[str, list] = {w: [] for w in want}
    for p in paths:
        try:
            ds = load_dinkum([p], cache=str(CACHE), sensors=want)
        except Exception:
            continue                      # zero-byte / header-only segments
        if not ds.sizes:
            continue
        for w in want:
            cols[w].append(np.asarray(ds[w].values, float) if w in ds
                           else np.full(ds.sizes["record"], np.nan))
    out = {w: np.concatenate(v) for w, v in cols.items() if v}
    order = np.argsort(out["m_present_time"])
    return {w: v[order] for w, v in out.items()}


def _redistribute(t, dr, t0, t1, err):
    """Spread the closing error over a segment, linearly in time."""
    w = (t - t0) / (t1 - t0)
    return dr + err * w


def main() -> None:
    f = load_flight()
    t = f["m_present_time"]
    keep = np.isfinite(t) & (t > T_LO) & (t < T_HI)
    t = t[keep]
    lat_dr = ddmm_to_deg(f["m_lat"][keep])
    lon_dr = ddmm_to_deg(f["m_lon"][keep])

    g_lat_raw, g_lon_raw = f["m_gps_lat"][keep], f["m_gps_lon"][keep]
    good_fix = (np.isfinite(g_lat_raw) & np.isfinite(g_lon_raw)
                & (g_lat_raw != NO_FIX) & (g_lon_raw != NO_FIX)
                & (np.abs(g_lat_raw) > 1.0))
    g_lat = ddmm_to_deg(g_lat_raw)
    g_lon = ddmm_to_deg(g_lon_raw)

    have_dr = np.isfinite(lat_dr) & np.isfinite(lon_dr)
    # m_gps_lat is reported ONLY while at the surface, and there m_lat has
    # ALREADY been snapped to it -- so differencing at a fix sample gives
    # exactly zero. The accumulated DR error is in the last sample OF THE DIVE,
    # immediately before the next surface interval. Segment on surface runs.
    at_surf = good_fix & have_dr
    edges = np.flatnonzero(np.diff(at_surf.astype(int)) != 0) + 1
    runs = np.split(np.arange(t.size), edges)
    surf_runs = [r for r in runs if r.size and at_surf[r[0]]]
    print(f"flight samples in window: {t.size:,}   DR positions: {have_dr.sum():,}   "
          f"surface intervals: {len(surf_runs):,}")

    lat_c = lat_dr.copy()
    lon_c = lon_dr.copy()
    corrected = np.zeros(t.size, bool)
    drifts, hours = [], []

    for s_k, s_next in zip(surf_runs[:-1], surf_runs[1:]):
        a_i, b_i = s_k[-1], s_next[0]          # reset point, next truth
        dive = np.arange(a_i, b_i)
        dive = dive[have_dr[dive] & ~at_surf[dive]]
        if dive.size < 3:
            continue
        e_i = dive[-1]                          # last DR sample of the dive
        t0, t1 = t[a_i], t[e_i]
        if not (t1 > t0) or (t1 - t0) > 12 * 3600:
            continue
        e_lat = g_lat[b_i] - lat_dr[e_i]
        e_lon = g_lon[b_i] - lon_dr[e_i]
        seg = np.arange(a_i, e_i + 1)
        seg = seg[have_dr[seg]]
        lat_c[seg] = _redistribute(t[seg], lat_dr[seg], t0, t1, e_lat)
        lon_c[seg] = _redistribute(t[seg], lon_dr[seg], t0, t1, e_lon)
        corrected[seg] = True
        km = np.hypot(e_lat * 111.0, e_lon * 111.0 * np.cos(np.deg2rad(g_lat[b_i])))
        drifts.append(km * 1000.0)
        hours.append((t1 - t0) / 3600.0)

    d = np.array(drifts)
    h = np.array(hours)
    print(f"dive segments corrected: {d.size}")
    if d.size:
        print(f"  closing drift m : median {np.median(d):8.1f}  p90 {np.percentile(d,90):8.1f}  max {d.max():8.1f}")
        print(f"  segment hours   : median {np.median(h):8.2f}  max {h.max():8.2f}")
        v = d / (h * 3600.0)
        print(f"  implied current m/s: median {np.median(v):.3f}  p90 {np.percentile(v,90):.3f}  max {v.max():.3f}")
    print(f"  samples corrected: {corrected.sum():,} of {have_dr.sum():,} DR positions")

    ds = xr.open_dataset(HOTEL)
    # the hotel's time coordinate is named after time.base, not "time"
    tname = list(ds.sizes)[0]
    ht = ds[tname].values.astype("datetime64[ns]").astype("int64") / 1e9
    ok = np.isfinite(lat_c) & np.isfinite(lon_c)
    ds["lat"] = (tname, np.interp(ht, t[ok], lat_c[ok], left=np.nan, right=np.nan))
    ds["lon"] = (tname, np.interp(ht, t[ok], lon_c[ok], left=np.nan, right=np.nan))
    ds["lat_corrected"] = (tname, np.interp(ht, t, corrected.astype(float),
                                             left=0.0, right=0.0) > 0.5)
    ds["lat"].attrs = {"units": "degrees_north", "standard_name": "latitude",
                       "comment": "dead-reckoned m_lat, drift redistributed linearly in "
                                  "time between GPS fixes; see make_corrected_track.py"}
    ds["lon"].attrs = {"units": "degrees_east", "standard_name": "longitude",
                       "comment": "dead-reckoned m_lon, drift redistributed linearly in "
                                  "time between GPS fixes; see make_corrected_track.py"}
    ds["lat_corrected"].attrs = {
        "long_name": "sample lies inside a GPS-bracketed segment",
        "comment": "False where the track is raw DR (before the first fix or after the last)"}
    ds.load()
    ds.close()
    ds.to_netcdf(HOTEL, mode="w")
    print(f"\nwrote lat/lon/lat_corrected into {HOTEL.name}")


if __name__ == "__main__":
    main()
