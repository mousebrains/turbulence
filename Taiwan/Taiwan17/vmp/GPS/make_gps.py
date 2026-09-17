#!/usr/bin/env python3
"""Build GPS/gps.nc for the Taiwan-2017 VMP SN142 casts from the ship met stream.

Source: ``../../ship_data/met/met.mat`` (``../../DATA/met.mat`` before the
2026-09-15 reorganization) -- the shipboard met/nav record, 1,230,258
samples at ~1 Hz, 2017-02-15T01:18Z .. 2017-03-01T12:00Z. Position is ``LA`` /
``LO``; ``met_nav_export.mat`` here is a v7 export of (ZD, LA, LO) made with
MATLAB because ``met.mat`` is MATLAB v7.3 (HDF5) and scipy cannot read it.

WHICH CLOCK
-----------
``met.mat`` carries TWO time bases and they disagree: ``Time`` (MATLAB datenum)
and ``ZD`` (unix epoch seconds). Their difference has a median of 0.00 s but a
minimum of **exactly -86400 s** -- a one-day rollover in ``Time``. ``ZD`` is
used, and it checks out independently:

  * strictly monotonic, **zero** backward steps
  * cadence median 1.00 s, max gap 39 s (inside the 60 s tolerance)
  * span brackets every VMP cast (2017-02-22T14:20Z .. 2017-02-26T18:11Z)

Do not use ``Time`` without first repairing the rollover.

WHY THIS MATTERS
----------------
The config previously pinned ``gps: {source: fixed, lat: 13, lon: 130}`` --
copied unedited from an ARCTERX-2025 config. That is the Philippine Sea, about
1,200 km from where this cruise worked (21.1-22.2 N, 119.0-120.4 E). Position
feeds the TEOS-10 Absolute Salinity conversion, so density, N^2 and Gamma all
inherited it.

Output: one-minute medians as ``t`` (minutes since the first retained minute,
CF-encoded), ``lat``, ``lon`` -- same layout as the CASPER-West, ARCTERX-2022
and Taiwan-2013 GPS files.

Run:  python3 make_gps.py
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import scipy.io as sio
import xarray as xr

HERE = Path(__file__).resolve().parent
NAV = HERE / "met_nav_export.mat"
OUT = HERE / "gps.nc"


def main() -> None:
    m = sio.loadmat(NAV, squeeze_me=True)["out"]
    t, lat, lon = m[:, 0].astype(float), m[:, 1].astype(float), m[:, 2].astype(float)
    ok = np.isfinite(t) & np.isfinite(lat) & np.isfinite(lon)
    ok &= (np.abs(lat) <= 90) & (np.abs(lon) <= 360)
    t, lat, lon = t[ok], lat[ok], lon[ok]
    if np.any(np.diff(t) < 0):
        raise SystemExit("ZD is not monotonic; investigate before trusting it")
    print(f"source: {t.size:,} fixes  "
          f"{dt.datetime.utcfromtimestamp(t[0]):%Y-%m-%dT%H:%MZ} .. "
          f"{dt.datetime.utcfromtimestamp(t[-1]):%Y-%m-%dT%H:%MZ}")

    minute = np.floor(t / 60.0).astype(np.int64)
    edges = np.flatnonzero(np.diff(minute)) + 1
    groups = np.split(np.arange(t.size), edges)
    m0 = minute[0]
    t_min = np.array([minute[g[0]] - m0 for g in groups], dtype=np.int32)
    la = np.array([np.median(lat[g]) for g in groups])
    lo = np.array([np.median(lon[g]) for g in groups])

    t0 = dt.datetime(1970, 1, 1) + dt.timedelta(minutes=int(m0))
    ds = xr.Dataset({"lat": ("t", la), "lon": ("t", lo)}, coords={"t": ("t", t_min)})
    ds["t"].attrs = {"units": f"minutes since {t0:%Y-%m-%d %H:%M:%S}", "calendar": "standard"}
    ds["lat"].attrs = {"units": "degrees_north", "standard_name": "latitude"}
    ds["lon"].attrs = {"units": "degrees_east", "standard_name": "longitude"}
    ds.attrs = {
        "title": "Taiwan 2017 ship track, one-minute medians",
        "source": "DATA/met.mat fields ZD/LA/LO (unix-epoch clock, not the datenum)",
        "history": f"created {dt.datetime.utcnow():%Y-%m-%d} by make_gps.py",
        "comment": "Replaces a fixed 13N/130E position copied from an ARCTERX config.",
    }
    ds.to_netcdf(OUT)
    print(f"wrote {OUT}  n={t_min.size:,} minutes  "
          f"lat {la.min():.4f}..{la.max():.4f}  lon {lo.min():.4f}..{lo.max():.4f}")


if __name__ == "__main__":
    main()
