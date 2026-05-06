# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Combined NetCDF assembly — merge binned data across all files.

Reference: Code/save2combo.m, Code/glue_widthwise.m, Code/glue_lengthwise.m,
           Code/save2NetCDF.m
"""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xarray as xr

from odas_tpw.perturb.netcdf_schema import GLOBAL_ATTRS, apply_schema


def _glue_widthwise(datasets: list[xr.Dataset]) -> xr.Dataset:
    """Glue depth-binned datasets widthwise (bins x all profiles).

    Each input dataset has dims (bin, profile).  Profiles are concatenated
    along the profile dimension.
    """
    if not datasets:
        return xr.Dataset()
    return xr.concat(datasets, dim="profile")


def _glue_lengthwise(datasets: list[xr.Dataset]) -> xr.Dataset:
    """Glue time-binned datasets lengthwise (concatenate time bins).

    Each input dataset has a time dimension.
    """
    if not datasets:
        return xr.Dataset()
    return xr.concat(datasets, dim="time")


def make_combo(
    binned_dir: str | Path,
    output_dir: str | Path,
    schema: dict,
    netcdf_attrs: dict | None = None,
    method: str = "depth",
) -> Path | None:
    """Merge all binned NetCDFs into a single combo.nc.

    Parameters
    ----------
    binned_dir : Path
        Directory containing per-file binned NetCDFs.
    output_dir : Path
        Directory for combo output.
    schema : dict
        CF/ACDD schema for variable metadata.
    netcdf_attrs : dict, optional
        Additional global attributes from config.
    method : str
        "depth" (widthwise glue) or "time" (lengthwise glue).

    Returns
    -------
    Path or None
        Path to combo.nc, or None if no input files.
    """
    binned_dir = Path(binned_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nc_files = sorted(binned_dir.glob("*.nc"))
    if not nc_files:
        return None

    # xarray decodes datetime variables and stores their original ``units``
    # in attrs.  The CF encoder later refuses to overwrite that attr when
    # we re-write — so for binned files (which we always re-encode) it is
    # cleaner to keep numeric times throughout.
    datasets = [xr.open_dataset(f, decode_times=False) for f in nc_files]

    combo = _glue_widthwise(datasets) if method == "depth" else _glue_lengthwise(datasets)

    for ds in datasets:
        ds.close()

    # Chronologically sort the depth-combo profiles so time_coverage_start /
    # end agree with stime[0] / stime[-1].  ACDD's checker compares these
    # explicitly and raises a "Date time mismatch" warning otherwise.
    if method == "depth" and "stime" in combo and "profile" in combo.dims:
        sk = combo["stime"].values
        order = np.argsort(np.where(np.isfinite(sk), sk, np.inf))
        combo = combo.isel(profile=order)
        combo = combo.assign_coords(profile=np.arange(combo.sizes["profile"], dtype=np.int32))
    elif method == "time" and "time" in combo.coords and combo["time"].size:
        # CF requires the time coordinate to be strictly monotonic; per-file
        # outputs concatenated in filename order are not (e.g. ARCTERX has
        # ``data_*`` April + ``OH3465_*`` May files alphabetically reversed).
        tk = combo["time"].values
        order = np.argsort(tk)
        combo = combo.isel(time=order)

    # Apply schema
    combo = apply_schema(combo, schema)

    # Apply global attributes — featureType is the one GLOBAL_ATTRS entry
    # that legitimately differs by combo style (depth ⇒ ``profile``,
    # time ⇒ ``timeSeries``), so resolve it after copying the defaults.
    combo.attrs.update(GLOBAL_ATTRS)
    # CF §9 featureType: depth combos are profile sets; CTD time combos are
    # along a moving ship track, which CF calls ``trajectory``.
    combo.attrs["featureType"] = "profile" if method == "depth" else "trajectory"
    # CF §9.4: a trajectory-featured dataset must carry an instance variable
    # with ``cf_role="trajectory_id"``.  For our single-platform CTD combos
    # a scalar is sufficient.
    if method == "time" and "trajectory_id" not in combo:
        traj_id = combo.attrs.get("id") or "platform_track"
        combo["trajectory_id"] = xr.DataArray(
            np.asarray(str(traj_id), dtype="S64"),
            attrs={"cf_role": "trajectory_id", "long_name": "trajectory identifier"},
        )
    combo.attrs["date_created"] = datetime.now(UTC).isoformat()
    combo.attrs["history"] = f"Created by perturb on {datetime.now(UTC).isoformat()}"

    if netcdf_attrs:
        for k, v in netcdf_attrs.items():
            if v is not None:
                combo.attrs[k] = v

    # Auto-fill geospatial/temporal attributes
    has_lat = "lat" in combo and np.any(np.isfinite(combo["lat"].values))
    has_lon = "lon" in combo and np.any(np.isfinite(combo["lon"].values))
    if has_lat:
        lat_vals = combo["lat"].values
        combo.attrs.setdefault("geospatial_lat_min", float(np.nanmin(lat_vals)))
        combo.attrs.setdefault("geospatial_lat_max", float(np.nanmax(lat_vals)))
        combo.attrs.setdefault("geospatial_lat_units", "degrees_north")
        combo.attrs.setdefault("geospatial_lat_resolution", "point")
    if has_lon:
        lon_vals = combo["lon"].values
        combo.attrs.setdefault("geospatial_lon_min", float(np.nanmin(lon_vals)))
        combo.attrs.setdefault("geospatial_lon_max", float(np.nanmax(lon_vals)))
        combo.attrs.setdefault("geospatial_lon_units", "degrees_east")
        combo.attrs.setdefault("geospatial_lon_resolution", "point")
    if has_lat and has_lon:
        # WGS84 is the standard for surface oceanographic GPS feeds.
        combo.attrs.setdefault("geospatial_bounds_crs", "EPSG:4326")
        lat_min = combo.attrs["geospatial_lat_min"]
        lat_max = combo.attrs["geospatial_lat_max"]
        lon_min = combo.attrs["geospatial_lon_min"]
        lon_max = combo.attrs["geospatial_lon_max"]
        combo.attrs.setdefault(
            "geospatial_bounds",
            f"POLYGON(({lon_min} {lat_min}, {lon_max} {lat_min}, "
            f"{lon_max} {lat_max}, {lon_min} {lat_max}, {lon_min} {lat_min}))",
        )

    # Vertical extent: prefer ``depth`` (m, positive down) over ``bin`` or ``P``
    for vname in ("depth", "bin", "P"):
        if vname in combo.coords or vname in combo:
            vals = combo[vname].values
            if np.any(np.isfinite(vals)):
                combo.attrs.setdefault("geospatial_vertical_min", float(np.nanmin(vals)))
                combo.attrs.setdefault("geospatial_vertical_max", float(np.nanmax(vals)))
                combo.attrs.setdefault("geospatial_vertical_units", "m")
                # EPSG:5831 = "Mean Sea Level depth" (positive down, in
                # metres) — the right vertical CRS for our depth coord.
                combo.attrs.setdefault("geospatial_bounds_vertical_crs", "EPSG:5831")
                combo.attrs.setdefault("geospatial_vertical_resolution", "1.0 m")
                break

    # Time coverage: prefer per-profile stime/etime (depth combos), fall back
    # to ``time`` for time combos (CTD).  ISO 8601 with timezone designator,
    # plus a derived duration (ACDD recommended).
    t_start = None
    t_end = None
    if "stime" in combo and "etime" in combo:
        stime = combo["stime"].values
        etime = combo["etime"].values
        if np.any(np.isfinite(stime)) and np.any(np.isfinite(etime)):
            t_start = np.datetime64("1970-01-01") + np.timedelta64(
                int(np.nanmin(stime)), "s"
            )
            t_end = np.datetime64("1970-01-01") + np.timedelta64(int(np.nanmax(etime)), "s")
    elif "time" in combo:
        # ``decode_times=False`` keeps numeric time on read, so values are
        # epoch seconds — convert via timedelta64 for ISO formatting.
        t = combo["time"].values
        if t.size and np.any(np.isfinite(t.astype("float64"))):
            t_start = np.datetime64("1970-01-01") + np.timedelta64(
                int(np.nanmin(t)), "s"
            )
            t_end = np.datetime64("1970-01-01") + np.timedelta64(int(np.nanmax(t)), "s")
    if t_start is not None and t_end is not None:
        combo.attrs.setdefault("time_coverage_start", f"{t_start}Z")
        combo.attrs.setdefault("time_coverage_end", f"{t_end}Z")
        # ISO 8601 duration — avoid pulling in a heavy dep just for this.
        secs = int((t_end - t_start) / np.timedelta64(1, "s"))
        if secs >= 0:
            d, rem = divmod(secs, 86400)
            h, rem = divmod(rem, 3600)
            m, s = divmod(rem, 60)
            iso_dur = "P"
            if d:
                iso_dur += f"{d}D"
            if h or m or s or iso_dur == "P":
                iso_dur += "T"
                if h:
                    iso_dur += f"{h}H"
                if m:
                    iso_dur += f"{m}M"
                if s or iso_dur == "PT":
                    iso_dur += f"{s}S"
            combo.attrs.setdefault("time_coverage_duration", iso_dur)
        # ACDD recommends a typical sampling interval — for profile combos
        # the median between successive profile starts is the right choice.
        if "stime" in combo and combo["stime"].size > 1:
            stimes = combo["stime"].values
            stimes = stimes[np.isfinite(stimes)]
            if stimes.size > 1:
                stimes_sorted = np.sort(stimes)
                med = float(np.median(np.diff(stimes_sorted)))
                if med > 0:
                    secs = round(med)
                    h, rem = divmod(secs, 3600)
                    m, s = divmod(rem, 60)
                    res = "PT"
                    if h:
                        res += f"{h}H"
                    if m:
                        res += f"{m}M"
                    if s or res == "PT":
                        res += f"{s}S"
                    combo.attrs.setdefault("time_coverage_resolution", res)

    # Auto-derive ``id`` from output path if the user didn't set it
    combo.attrs.setdefault("id", output_dir.name)

    # CF profile featureType requires the profile coordinate to be int (not
    # int64) and to carry cf_role. xarray's default ``np.arange`` produces
    # int64 on 64-bit platforms, which compliance-checker rejects.
    if "profile" in combo.dims and "profile" in combo.coords:
        if combo["profile"].dtype.kind == "i" and combo["profile"].dtype.itemsize > 4:
            combo = combo.assign_coords(profile=combo["profile"].astype(np.int32))
        combo["profile"].attrs.setdefault("cf_role", "profile_id")
        combo["profile"].attrs.setdefault("long_name", "profile index")

    # Strip the auto-applied _FillValue from coordinate variables — CF-1.13
    # §2.5.1 forbids _FillValue on coordinates. xarray writes one by default
    # for floats; clear it via encoding instead of mutating attrs.
    encoding: dict = {}
    for cname in combo.coords:
        encoding[cname] = {"_FillValue": None}

    out_path = output_dir / "combo.nc"
    combo.to_netcdf(out_path, encoding=encoding)
    return out_path


def make_ctd_combo(
    ctd_dir: str | Path,
    output_dir: str | Path,
    schema: dict,
    netcdf_attrs: dict | None = None,
) -> Path | None:
    """Concatenate all CTD time-binned files into a single combo.

    Parameters
    ----------
    ctd_dir : Path
        Directory containing per-file CTD NetCDFs.
    output_dir : Path
        Directory for combo output.
    schema : dict
        CF/ACDD schema.
    netcdf_attrs : dict, optional
        Additional global attributes.

    Returns
    -------
    Path or None
        Path to combo.nc, or None if no input files.
    """
    return make_combo(ctd_dir, output_dir, schema, netcdf_attrs, method="time")
