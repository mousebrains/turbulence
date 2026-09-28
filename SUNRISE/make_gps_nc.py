"""SUNRISE ship GPS: gps.mat -> gps.nc, one per vessel-year.

Source is <year>/<ship>/GPS/gps.mat (lat, lon, MATLAB datetime). Read via a
MATLAB-side export to -v7 with time already converted to epoch seconds, so
both the v5 and v7.3 originals are handled by one path and scipy suffices.

Fixes applied, each counted in the output and recorded as a global attribute:
  * 2022 PointSur carries 4305 samples (one contiguous 72-minute block on
    2022-06-26) with a FLIPPED LONGITUDE SIGN: +92.43..+92.45 where the rest
    of the track runs -93.01..-90.57. |lon| lands inside the good range and
    the latitudes are normal, so this is a sign error, not a position error.
    Applied as lon = -|lon| for lon > 0.
  * 2019 Pelican has 13 NaN lat/lon samples -> dropped.
  * 2021 WaltonSmith has 4 non-increasing timestamps. They are OUT OF ORDER,
    not duplicated: a stable sort by time repairs all four and nothing is
    dropped (nonmonotonic_dropped=0 in the output). The dedupe below still
    runs, and would drop an exact time tie if one ever appeared.
Times are UTC: the MATLAB datetimes carry no TimeZone, and posixtime() on an
unzoned datetime treats it as UTC.
"""
import pathlib
import numpy as np, scipy.io as sio, xarray as xr

SRC = pathlib.Path("/private/tmp/claude-501/-Users-pat-tpw-turbulence/9f39321a-b2dd-4f6f-b90d-504505782a96/scratchpad/sunrise")
DST = pathlib.Path("/Volumes/SeaChest/SUNRISE/Data")
TAGS = ["2019_Pelican", "2021_Pelican", "2021_WaltonSmith", "2022_Pelican", "2022_PointSur"]

for tag in TAGS:
    m = sio.loadmat(str(SRC / ("gps_%s.mat" % tag)), squeeze_me=True)
    t = np.asarray(m["t"], float).ravel()
    lat = np.asarray(m["lat"], float).ravel()
    lon = np.asarray(m["lon"], float).ravel()
    n0 = t.size
    n_signflip = int((lon > 0).sum())
    lon = np.where(lon > 0, -np.abs(lon), lon)
    good = np.isfinite(t) & np.isfinite(lat) & np.isfinite(lon)
    n_nan = int((~good).sum())
    t, lat, lon = t[good], lat[good], lon[good]
    o = np.argsort(t, kind="stable")
    t, lat, lon = t[o], lat[o], lon[o]
    keep = np.ones(t.size, bool)
    keep[1:] = np.diff(t) > 0
    n_dup = int((~keep).sum())
    t, lat, lon = t[keep], lat[keep], lon[keep]
    year, ship = tag.split("_", 1)
    ds = xr.Dataset(
        {"lat": ("time", lat, {"units": "degrees_north", "standard_name": "latitude"}),
         "lon": ("time", lon, {"units": "degrees_east", "standard_name": "longitude"})},
        coords={"time": ("time", t, {"units": "seconds since 1970-01-01T00:00:00+00:00",
                                     "calendar": "standard", "standard_name": "time", "axis": "T"})},
        attrs={"title": "Ship GPS track, SUNRISE %s %s" % (year, ship),
               "source": "%s/%s/GPS/gps.mat" % (year, ship),
               "comment": ("Converted from the MATLAB original. "
                           "sign_flipped_lon_fixed=%d; nan_dropped=%d; "
                           "nonmonotonic_dropped=%d; n_in=%d; n_out=%d"
                           % (n_signflip, n_nan, n_dup, n0, t.size)),
               "Conventions": "CF-1.13"})
    out = DST / year / ship / "GPS" / "gps.nc"
    ds.to_netcdf(out)
    print("%-20s n %7d -> %7d  (signflip %4d, nan %3d, nonmono %3d)  "
          "lat %.4f..%.4f  lon %.4f..%.4f"
          % (tag, n0, t.size, n_signflip, n_nan, n_dup,
             lat.min(), lat.max(), lon.min(), lon.max()))
