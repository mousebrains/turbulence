#!/usr/bin/env python3
"""Build gps.nc for ASTRAL 2023 from the ship's gps.mat.

`gps.mat` is a MATLAB **v7.3** file, i.e. HDF5, holding a MATLAB TABLE with
columns t / lat / lon. Two wrinkles:

  * scipy.io.loadmat cannot read v7.3 at all, and `h5py` is not installed here,
    so we read it with **netCDF4**, which opens plain HDF5 happily.
  * the top-level `gps` variable is an HDF5 OBJECT REFERENCE, not data. The
    columns live in the `#refs#` group as separate datasets. We locate them by
    SHAPE AND RANGE rather than by their one-letter `#refs#` names, which are
    assigned by MATLAB and must not be relied on.

`t` is epoch MILLISECONDS (median dt = 1000), not seconds and not a MATLAB
datenum -- read as seconds it lands in the year 55406.

Output mirrors the ARCTERX gps.nc layout (t/lat/lon on an `index` dimension,
t as int minutes since a reference) so the same `gps:` config block shape works.

The neighbouring `mat2nc.py` is an earlier, unfinished attempt (it prints and
writes nothing); it is left untouched.
"""
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr

SRC = Path(__file__).resolve().parent / "gps.mat"
OUT = Path(__file__).resolve().parent / "gps.nc"

with nc.Dataset(SRC, "r") as ds:
    refs = ds.groups["#refs#"]
    # The table's data columns are the equal-length float64 datasets. Take them
    # in dataset-name order, which is MATLAB's table-variable order (t, lat,
    # lon -- the names are stored separately in #refs# as uint16 char arrays).
    numeric = []
    for name in sorted(refs.variables):
        a = np.asarray(refs.variables[name][:]).ravel()
        if a.dtype == np.float64 and a.size >= 1000:
            numeric.append((name, a))
    if len(numeric) != 3:
        raise SystemExit(f"expected 3 data columns in {SRC}, found {len(numeric)}")

    # Time is unambiguous: epoch milliseconds are ~1e12, far outside any
    # coordinate range.
    t_cols = [(n, a) for n, a in numeric if np.nanmin(a) > 1e12]
    if len(t_cols) != 1:
        raise SystemExit(f"could not identify the time column in {SRC}")
    t_name = t_cols[0][0]
    coords = [(n, a) for n, a in numeric if n != t_name]

    # lat/lon CANNOT be told apart by range here -- this cruise sits at
    # 66-72 E, which is inside the +/-90 latitude bound too. So take them in
    # column order and VALIDATE against the working area, swapping if needed
    # rather than trusting the order blindly.
    (lat_name, lat), (lon_name, lon) = coords
    def in_box(la, lo):
        return (5 <= np.nanmin(la) and np.nanmax(la) <= 25
                and 60 <= np.nanmin(lo) and np.nanmax(lo) <= 80)
    if not in_box(lat, lon):
        lat, lon = lon, lat
        lat_name, lon_name = lon_name, lat_name
        if not in_box(lat, lon):
            raise SystemExit(
                f"neither assignment of {lat_name}/{lon_name} lands in the "
                f"Arabian Sea working area; check {SRC} by hand"
            )
    cols = {"t": t_cols[0][1], "lat": lat, "lon": lon}
    print(f"columns: t={t_name}  lat={lat_name}  lon={lon_name}")

n_raw = cols["t"].size
df = pd.DataFrame(
    {
        "t": pd.to_datetime(cols["t"] / 1e3, unit="s", utc=True),
        "lat": cols["lat"],
        "lon": cols["lon"],
    }
).dropna()

# Plausibility: the Arabian Sea working area. A wild fix would otherwise drag a
# cast's position.
box = df.lat.between(5, 25) & df.lon.between(60, 80)
n_out = int((~box).sum())
df = df[box].set_index("t").sort_index()

per_min = df.resample("1min").median().dropna()
ref = per_min.index[0].floor("min")
t_min = ((per_min.index - ref).total_seconds() / 60.0).round().astype("int32")

out = xr.Dataset(
    {"t": ("index", t_min), "lat": ("index", per_min.lat.values),
     "lon": ("index", per_min.lon.values)},
    coords={"index": ("index", np.arange(per_min.shape[0], dtype="int32"))},
)
out.t.attrs = {"units": f"minutes since {ref.strftime('%Y-%m-%d %H:%M:%S')}",
               "calendar": "proleptic_gregorian"}
out.lat.attrs = {"units": "degrees_north", "standard_name": "latitude"}
out.lon.attrs = {"units": "degrees_east", "standard_name": "longitude"}
out.attrs = {"title": "ASTRAL 2023 ship GPS, 1-minute medians",
             "source": "GPS/gps.mat (MATLAB v7.3 table: t [epoch ms], lat, lon)",
             "history": "built by GPS/make_gps.py"}
out.to_netcdf(OUT, encoding={"t": {"dtype": "int32"}})

print(f"{n_raw} raw fixes, {n_out} outside the plausibility box")
print(f"wrote {OUT}  n={per_min.shape[0]} minutes")
print(f"  {per_min.index[0]} .. {per_min.index[-1]}")
print(f"  lat {per_min.lat.min():.4f}..{per_min.lat.max():.4f}  "
      f"lon {per_min.lon.min():.4f}..{per_min.lon.max():.4f}")
