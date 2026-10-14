#!/usr/bin/env python3
"""SeaExplorer SEA025 navigation (mission 24, Kuroshio, June 2017) -> nav.nc.

The SeaExplorer counterpart of `dinkum-hotel build` for these files. Reads the
66 `sea025.24.gli.sub.N.gz` navigation files (semicolon CSV, ~20 s cadence,
the glider's own clock) from ../post-recovery/glider and writes nav.nc on that
clock. Nothing is resampled or gap-filled.

Columns, verified against rows in these files (the vendor manual,
../../../reference/SEA06-50-MUM-UsermanualSeaExplorer-revG.pdf, has the rest):
  Timestamp  dd/mm/yyyy HH:MM:SS, UTC
  Pitch, Roll, Heading   degrees (Pitch POSITIVE nose UP: +19 deg at the surface on a climb)
  Depth      m (the only water depth/pressure in these files)
  Pa         INTERNAL HULL pressure in pascal (75-82 kPa throughout a 700 m
             dive) -- NOT water pressure, despite the name
  Lat, Lon   NMEA ddmm.mmm / dddmm.mmm (3404.422 = 34 deg 04.422')
  BallastCmd, BallastPos   oil volume, ml (-500 fully in .. +500 fully out)
  LinCmd, LinPos           battery fore-aft position, mm
  AngCmd, AngPos           battery roll (angular actuator), deg
  NavState   as OBSERVED in these files (not from the vendor): 100 = dive
             (pitch -21, oil -499), 117 = CLIMB (pitch +29, oil +259; 5-701 m),
             118 = the bottom turn (~704 m), 110/115/116 = at or near the
             surface (0.5-9 m). 117 is NOT the surface state.

    python make_nav.py      # -> nav.nc
"""
import gzip
import io
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "post-recovery" / "glider"
T_MIN = pd.Timestamp("2017-06-15", tz="UTC")
T_MAX = pd.Timestamp("2017-07-01", tz="UTC")


def nmea(x):
    x = np.asarray(x, float)
    d = np.trunc(x / 100.0)
    return d + (x - 100.0 * d) / 60.0


def main():
    files = sorted(SRC.glob("sea025.24.gli.sub.*.gz"), key=lambda p: int(p.name.split(".")[-2]))
    frames = []
    for f in files:
        with gzip.open(f, "rt", errors="replace") as fh:
            df = pd.read_csv(io.StringIO(fh.read()), sep=";")
        df["segment"] = int(f.name.split(".")[-2])
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df["time"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", utc=True)
    # A few rows carry an unset clock (one reads 2009-06-19); keep the deployment only.
    inwin = (df["time"] >= T_MIN) & (df["time"] < T_MAX)
    print(f"dropped {int((~inwin).sum())} rows outside {T_MIN.date()}..{T_MAX.date()}: "
          f"{sorted(df.loc[~inwin, 'Timestamp'].astype(str).unique())[:5]}")
    df = df[inwin].sort_values("time").drop_duplicates("time", keep="last")
    t = (df["time"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy()
    lat, lon = nmea(df["Lat"]), nmea(df["Lon"])
    bad = ~((lat > 0) & (lat < 60) & (lon > 100) & (lon < 180))
    lat[bad] = np.nan
    lon[bad] = np.nan

    def v(col, units, long_name, scale=1.0):
        return ("time", df[col].to_numpy(float) * scale, dict(units=units, long_name=long_name))

    ds = xr.Dataset(
        {
            "pitch": v("Pitch", "degree", "pitch, positive nose UP"),
            "roll": v("Roll", "degree", "roll"),
            "heading": v("Heading", "degree", "heading"),
            "depth": v("Depth", "m", "depth"),
            "hull_pressure": v("Pa", "Pa", "internal hull pressure (file column Pa); NOT water pressure"),
            "temperature": v("Temperature", "degree_Celsius", "navigation-board temperature (not a CTD)"),
            "ballast_cmd": v("BallastCmd", "cm3", "commanded oil volume"),
            "ballast_pos": v("BallastPos", "cm3", "measured oil volume (buoyancy change)"),
            "lin_pos": v("LinPos", "mm", "battery fore-aft position"),
            "ang_cmd": v("AngCmd", "degree", "commanded battery roll"),
            "ang_pos": v("AngPos", "degree", "battery roll (angular actuator)"),
            "nav_state": v("NavState", "1", "navigation state; observed: 100 dive, 117 climb, 118 bottom turn, 110/115/116 surface"),
            "lat": ("time", lat, dict(units="degrees_north", long_name="navigation latitude: the last GPS fix, HELD while underwater; NaN only if out of range")),
            "lon": ("time", lon, dict(units="degrees_east", long_name="navigation longitude: the last GPS fix, HELD while underwater; NaN only if out of range")),
            "segment": ("time", df["segment"].to_numpy(), dict(long_name="gli.sub file number")),
        },
        coords={"time": ("time", t, dict(units="seconds since 1970-01-01T00:00:00+00:00", standard_name="time",
                                         long_name="SeaExplorer navigation clock"))},
        attrs=dict(title="SeaExplorer SEA025 mission 24 navigation, Kuroshio 2017",
                   source="sea025.24.gli.sub.*.gz, Tanaka/gliders/20170620_sea025/post-recovery/glider"),
    )
    enc = {k: {"zlib": True, "complevel": 4} for k in ds.data_vars}
    ds.to_netcdf(HERE / "nav.nc", encoding=enc)
    dt = np.diff(t)
    print(f"{len(files)} files, {t.size} records, {pd.Timestamp(t[0], unit='s')} .. {pd.Timestamp(t[-1], unit='s')}, "
          f"median step {np.median(dt):.1f} s, max gap {dt.max()/60:.1f} min, {np.isfinite(lat).sum()} position fixes")
    for k in ("pitch", "depth", "hull_pressure", "ballast_pos", "lin_pos", "ang_pos"):
        x = ds[k].values
        print(f"  {k:12s} p1/50/99 {np.nanpercentile(x, [1, 50, 99]).round(2)}")


if __name__ == "__main__":
    main()
