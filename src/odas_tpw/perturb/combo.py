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

    datasets = [xr.open_dataset(f) for f in nc_files]

    # Sort temporally if possible
    for ds in datasets:
        if "stime" in ds:
            pass  # already sorted by filename order

    combo = _glue_widthwise(datasets) if method == "depth" else _glue_lengthwise(datasets)

    for ds in datasets:
        ds.close()

    # Apply schema
    combo = apply_schema(combo, schema)

    # Apply global attributes
    combo.attrs.update(GLOBAL_ATTRS)
    combo.attrs["date_created"] = datetime.now(UTC).isoformat()
    combo.attrs["history"] = f"Created by perturb on {datetime.now(UTC).isoformat()}"

    if netcdf_attrs:
        for k, v in netcdf_attrs.items():
            if v is not None:
                combo.attrs[k] = v

    # Auto-fill geospatial/temporal attributes
    if "lat" in combo:
        lat_vals = combo["lat"].values
        if np.any(np.isfinite(lat_vals)):
            combo.attrs.setdefault("geospatial_lat_min", float(np.nanmin(lat_vals)))
            combo.attrs.setdefault("geospatial_lat_max", float(np.nanmax(lat_vals)))
    if "lon" in combo:
        lon_vals = combo["lon"].values
        if np.any(np.isfinite(lon_vals)):
            combo.attrs.setdefault("geospatial_lon_min", float(np.nanmin(lon_vals)))
            combo.attrs.setdefault("geospatial_lon_max", float(np.nanmax(lon_vals)))

    out_path = output_dir / "combo.nc"
    combo.to_netcdf(out_path)
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
