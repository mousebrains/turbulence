# May-2026, Claude and Pat Welch, pat@mousebrains.com
"""Hand-rolled NetCDF4 writer for ``xarray.Dataset`` outputs.

Replaces ``xr.Dataset.to_netcdf()`` in the per-profile output paths
(diss, chi, ctd, binned, combo).  Bypasses xarray's
``ScipyDataStore``/``H5NetCDFStore`` indirection — measured at ~10 ms per
write under the SN465 cprofile — and goes straight to ``netCDF4.Dataset``.
Output is functionally equivalent: same variable names, dimensions,
attributes, and CF-1.13 / ACDD-1.3 metadata.

Behavioural rules mirrored from xarray's default ``to_netcdf`` path:
- Format: NETCDF4 (HDF5).
- No compression or chunking (the existing pipeline does not enable any).
- ``_FillValue=NaN`` is added for every floating-point variable that
  doesn't already carry one in its attrs/encoding.  This matches
  xarray's default CF encoder behaviour, which applies the rule to
  data, coord, and 1D index variables alike.
- String coordinate variables (e.g. the ``probe`` axis) are written as
  variable-length strings — ``netCDF4`` supports this natively in
  NETCDF4 format.
- All variable attributes survive verbatim, including ``flag_values``
  arrays for the ``method`` enum.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import xarray as xr


def _needs_fill_value(dtype: np.dtype) -> bool:
    """Mirror xarray's "add _FillValue=NaN to float vars" rule."""
    return bool(np.issubdtype(dtype, np.floating))


_UNSET = object()


_CF_TIME_UNIT_FACTORS_NS: dict[str, float] = {
    "seconds": 1e9,
    "second": 1e9,
    "s": 1e9,
    "sec": 1e9,
    "secs": 1e9,
    "milliseconds": 1e6,
    "millisecond": 1e6,
    "ms": 1e6,
    "microseconds": 1e3,
    "microsecond": 1e3,
    "us": 1e3,
    "nanoseconds": 1.0,
    "nanosecond": 1.0,
    "ns": 1.0,
    "minutes": 60e9,
    "minute": 60e9,
    "min": 60e9,
    "hours": 3600e9,
    "hour": 3600e9,
    "h": 3600e9,
    "hr": 3600e9,
    "days": 86400e9,
    "day": 86400e9,
    "d": 86400e9,
}


def _encode_cf_datetime(values: np.ndarray, units: str) -> np.ndarray:
    """Convert datetime64 values to numeric epoch offsets per CF ``units`` string.

    Mirrors what xarray's CF encoder does for ``"<unit> since <iso-datetime>"``.
    """
    from datetime import UTC, datetime

    units_str = str(units).strip().lower()
    unit_part, _, ref_part = units_str.partition(" since ")
    factor_ns = _CF_TIME_UNIT_FACTORS_NS.get(unit_part.strip())
    if factor_ns is None:
        raise ValueError(f"Unsupported CF time unit: {units!r}")
    ref_str = ref_part.strip().rstrip("z").rstrip("Z")
    try:
        ref_dt = datetime.fromisoformat(ref_str)
    except ValueError:
        ref_dt = datetime.fromisoformat(ref_str.replace(" ", "T"))
    if ref_dt.tzinfo is None:
        ref_dt = ref_dt.replace(tzinfo=UTC)
    epoch_ns = int(ref_dt.timestamp() * 1e9)
    ns_int = values.astype("datetime64[ns]").astype(np.int64)
    return (ns_int - epoch_ns) / factor_ns


def write_dataset(
    ds: xr.Dataset,
    path: str | Path,
    encoding: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Write an :class:`xarray.Dataset` to ``path`` in NETCDF4 format.

    Functionally equivalent to ``ds.to_netcdf(path, encoding=encoding)``
    for our pipeline's output datasets, but ~2-3x faster per call because
    it skips xarray's encoder/store/dump indirection.

    Honours the subset of xarray ``encoding`` we use:

    - ``_FillValue`` (passed in ``encoding`` or already on the variable)
      becomes the netCDF fill value; ``encoding[var]['_FillValue'] = None``
      suppresses it (used by combo to strip fill from CF coordinate
      variables, per CF-1.13 §2.5.1).
    - ``datetime64`` variables are encoded back to numeric epoch offsets
      using ``var.encoding['units']`` and ``var.encoding['dtype']``,
      matching xarray's CF encoder.
    """
    import netCDF4 as nc

    encoding = encoding or {}

    coord_names = set(ds.coords)
    data_var_names = set(ds.data_vars)
    # CF §5: an auxiliary coordinate is a coord whose name is *not* the
    # name of its sole dimension.  These need to be advertised on each
    # data variable that shares a dim with them via the ``coordinates``
    # attribute.  Dimension coordinates (var name == dim name) are
    # implicit and never go in ``coordinates``.
    aux_coord_dims: dict[str, frozenset[str]] = {}
    for cname in coord_names:
        coord = ds.variables[cname]
        cdims = tuple(coord.dims)
        if len(cdims) == 1 and cdims[0] == cname:
            continue
        aux_coord_dims[cname] = frozenset(cdims)

    with nc.Dataset(str(path), "w", format="NETCDF4") as f:
        for dim_name, dim_size in ds.sizes.items():
            f.createDimension(dim_name, int(dim_size))

        for var_name, var in ds.variables.items():
            data = var.values
            dtype = data.dtype if hasattr(data, "dtype") else np.asarray(data).dtype
            dims = tuple(var.dims)
            attrs = dict(var.attrs)
            var_encoding = dict(var.encoding) if hasattr(var, "encoding") else {}

            if (
                var_name in data_var_names
                and "coordinates" not in attrs
                and aux_coord_dims
            ):
                vdims = set(dims)
                shared = [
                    cname
                    for cname, cdims in aux_coord_dims.items()
                    if cdims & vdims
                ]
                if shared:
                    attrs["coordinates"] = " ".join(sorted(shared))

            if np.issubdtype(dtype, np.datetime64):
                units = var_encoding.get("units") or attrs.get("units")
                if not units:
                    raise ValueError(
                        f"datetime64 variable {var_name!r} has no CF 'units' "
                        "in attrs or encoding — cannot encode"
                    )
                target_dtype = np.dtype(var_encoding.get("dtype", "float64"))
                numeric = _encode_cf_datetime(np.asarray(data), units)
                data = numeric.astype(target_dtype)
                dtype = data.dtype
                attrs.setdefault("units", units)
                cal = var_encoding.get("calendar") or attrs.get("calendar")
                if cal:
                    attrs.setdefault("calendar", cal)

            existing_fill = attrs.pop("_FillValue", None)
            override = encoding.get(str(var_name), {}).get("_FillValue", _UNSET)
            fill_value: Any = None
            if override is not _UNSET:
                fill_value = override
            elif existing_fill is not None:
                fill_value = existing_fill
            elif _needs_fill_value(dtype):
                fill_value = np.nan

            if dtype.kind in ("U", "S", "O"):
                v = f.createVariable(var_name, str, dims)
                if data.ndim == 0:
                    v[...] = str(data.item())
                else:
                    flat = np.asarray(data).ravel()
                    out = np.empty(flat.shape, dtype=object)
                    for i, elem in enumerate(flat):
                        out[i] = str(elem)
                    v[:] = out.reshape(data.shape)
            else:
                v = f.createVariable(
                    var_name,
                    dtype,
                    dims,
                    fill_value=fill_value,
                )
                if data.ndim == 0:
                    v[...] = data.item()
                else:
                    v[:] = data

            for k, val in attrs.items():
                v.setncattr(k, val)

        for k, val in ds.attrs.items():
            f.setncattr(k, val)
