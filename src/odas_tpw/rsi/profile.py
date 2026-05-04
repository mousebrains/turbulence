# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Profile detection and extraction for vertical profilers.

Profile detection and smoothed fall-rate computation live in
:mod:`odas_tpw.scor160.profile` (instrument-independent).  This module
re-exports them for backward compatibility and adds RSI-specific I/O
(NetCDF extraction from PFile).
"""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from odas_tpw.rsi.p_file import PFile

# Re-export from scor160 — canonical implementations live there
from odas_tpw.scor160.profile import get_profiles, smooth_fall_rate

logger = logging.getLogger(__name__)

# Backward-compatible alias (previously underscore-prefixed)
_smooth_fall_rate = smooth_fall_rate

# Backward-compatible alias — computed from VEHICLE_ATTRIBUTES
def _build_vehicle_tau() -> dict[str, float]:
    from odas_tpw.rsi.vehicle import VEHICLE_ATTRIBUTES
    return {k: v[1] for k, v in VEHICLE_ATTRIBUTES.items()}

_VEHICLE_TAU = _build_vehicle_tau()


def extract_profiles(
    source: "PFile | str | Path",
    output_dir: str | Path,
    profiles: list[tuple[int, int]] | None = None,
    gps: Any = None,
    **profile_kwargs: Any,
) -> list[Path]:
    """Extract profiles from a PFile or full-record NetCDF.

    Parameters
    ----------
    source : PFile, str, or Path
        A PFile object, path to .p file, or path to full-record .nc file.
    output_dir : Path
        Directory for per-profile NetCDF files.
    profiles : list of (start, end) tuples, optional
        Pre-computed profile bounds (slow-rate indices). If supplied, the
        internal call to :func:`get_profiles` is skipped — use this when
        the caller has already adjusted the bounds (e.g. via top-trim or
        bottom-crash detection).
    gps : GPSProvider, optional
        If supplied, lat/lon/stime/etime are written as scalar coordinate
        variables on each per-profile NetCDF for CF §9 profile featureType
        compliance.  ``gps`` must expose ``lat(t)`` and ``lon(t)`` taking a
        time array in *seconds since pf.start_time* (the same domain as
        ``pf.t_slow``).
    **profile_kwargs
        Keyword arguments passed to get_profiles (P_min, W_min, etc.).

    Returns
    -------
    list of Path
        Paths to per-profile NetCDF files written.
    """
    import netCDF4 as nc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = _load_source(source)
    P_slow = data["P"]
    fs_slow = data["fs_slow"]
    fs_fast = data["fs_fast"]
    ratio = round(fs_fast / fs_slow)
    start_epoch_s = data.get("start_epoch_s")

    # Compute smoothed fall rate from slow pressure
    W = _smooth_fall_rate(P_slow, fs_slow)

    if profiles is None:
        profiles = get_profiles(P_slow, W, fs_slow, **profile_kwargs)
    if not profiles:
        logger.warning(f"No profiles found in {data['stem']}")
        return []

    output_paths = []
    for pi, (s_slow, e_slow) in enumerate(profiles, 1):
        s_fast = s_slow * ratio
        e_fast = (e_slow + 1) * ratio
        s_slow_end = e_slow + 1

        prof_path = output_dir / f"{data['stem']}_prof{pi:03d}.nc"

        ds = nc.Dataset(str(prof_path), "w", format="NETCDF4")

        # Copy global attributes
        for attr in data["global_attrs"]:
            setattr(ds, attr, data["global_attrs"][attr])

        ds.Conventions = "CF-1.13, ACDD-1.3"
        ds.title = f"{data['stem']} profile {pi}"
        ds.featureType = "profile"

        # Profile metadata
        ds.profile_number = pi
        ds.profile_start_index_slow = int(s_slow)
        ds.profile_end_index_slow = int(e_slow)
        ds.profile_P_start = float(P_slow[s_slow])
        ds.profile_P_end = float(P_slow[e_slow])
        ds.profile_duration_s = float((e_slow - s_slow) / fs_slow)
        ds.profile_mean_speed = float(np.mean(np.abs(W[s_slow:s_slow_end])))
        ds.history = f"Profile {pi} extracted on {datetime.now(UTC).isoformat()}"

        # Per-profile scalar coordinate variables (CF §9 profile featureType).
        # Keep these on the slow-rate timeline; gps must be in seconds since
        # the file's start_time, matching ``data["t_slow"]`` semantics.
        t_prof_s = float(data["t_slow"][s_slow])
        t_end_s = float(data["t_slow"][e_slow])
        if start_epoch_s is not None:
            stime_var = ds.createVariable("stime", "f8", ())
            stime_var[...] = start_epoch_s + t_prof_s
            stime_var.units = "seconds since 1970-01-01"
            stime_var.standard_name = "time"
            stime_var.long_name = "profile start time"
            stime_var.calendar = "standard"
            stime_var.units_metadata = "leap_seconds: utc"
            stime_var.axis = "T"

            etime_var = ds.createVariable("etime", "f8", ())
            etime_var[...] = start_epoch_s + t_end_s
            etime_var.units = "seconds since 1970-01-01"
            etime_var.standard_name = "time"
            etime_var.long_name = "profile end time"
            etime_var.calendar = "standard"
            etime_var.units_metadata = "leap_seconds: utc"
        if gps is not None:
            # GPSProvider expects epoch seconds — t_prof_s is file-relative.
            t_query = np.array([t_prof_s + (start_epoch_s or 0.0)])
            lat_val = float(np.asarray(gps.lat(t_query))[0])
            lon_val = float(np.asarray(gps.lon(t_query))[0])
            lat_var = ds.createVariable("lat", "f8", ())
            lat_var[...] = lat_val
            lat_var.units = "degrees_north"
            lat_var.standard_name = "latitude"
            lat_var.long_name = "profile latitude"
            lat_var.axis = "Y"
            lon_var = ds.createVariable("lon", "f8", ())
            lon_var[...] = lon_val
            lon_var.units = "degrees_east"
            lon_var.standard_name = "longitude"
            lon_var.long_name = "profile longitude"
            lon_var.axis = "X"

        n_fast = e_fast - s_fast
        n_slow = s_slow_end - s_slow
        ds.createDimension("time_fast", n_fast)
        ds.createDimension("time_slow", n_slow)

        # Time variables
        t_fast_var = ds.createVariable("t_fast", "f8", ("time_fast",), zlib=True)
        t_fast_var[:] = data["t_fast"][s_fast:e_fast]
        t_fast_var.units = data.get("t_fast_units", "seconds")
        t_fast_var.long_name = "time (fast channels)"
        t_fast_var.standard_name = "time"
        t_fast_var.calendar = "standard"
        t_fast_var.units_metadata = "leap_seconds: utc"
        t_fast_var.axis = "T"

        t_slow_var = ds.createVariable("t_slow", "f8", ("time_slow",), zlib=True)
        t_slow_var[:] = data["t_slow"][s_slow:s_slow_end]
        t_slow_var.units = data.get("t_slow_units", "seconds")
        t_slow_var.long_name = "time (slow channels)"
        t_slow_var.standard_name = "time"
        t_slow_var.calendar = "standard"
        t_slow_var.units_metadata = "leap_seconds: utc"
        t_slow_var.axis = "T"

        # Channel data — RSI's INI emits unit strings like ``umol_L-1`` and
        # ``deg`` that UDUNITS won't parse; canonicalise via the schema.
        from odas_tpw.perturb.netcdf_schema import COMBO_SCHEMA, canonicalize_units

        for ch_name, ch_data, dim, attrs in data["channels"]:
            trimmed = ch_data[s_fast:e_fast] if dim == "time_fast" else ch_data[s_slow:s_slow_end]
            var_name = ch_name.replace(" ", "_")
            var = ds.createVariable(var_name, "f4", (dim,), zlib=True)
            var[:] = trimmed.astype(np.float32)
            schema_entry = COMBO_SCHEMA.get(var_name, {})
            for k, v in attrs.items():
                if k == "units":
                    v = canonicalize_units(str(v))
                setattr(var, k, v)
            # Layer the canonical schema attrs (long_name, standard_name,
            # units_metadata, …) on top — schema wins where it has an opinion.
            for k, v in schema_entry.items():
                if k != "nc_name":
                    setattr(var, k, v)

        ds.fs_fast = float(fs_fast)
        ds.fs_slow = float(fs_slow)
        ds.close()
        output_paths.append(prof_path)
        logger.info(
            f"Profile {pi}: P={P_slow[s_slow]:.1f}–{P_slow[e_slow]:.1f} dbar, "  # noqa: RUF001
            f"{(e_slow - s_slow) / fs_slow:.1f} s -> {prof_path.name}"
        )

    return output_paths


def _load_source(source: "PFile | str | Path") -> dict[str, Any]:
    """Load data from PFile, .p path, or .nc path into a common dict."""
    from odas_tpw.rsi.p_file import PFile

    if isinstance(source, PFile):
        return _load_from_pfile(source)

    source = Path(source)
    if source.suffix.lower() == ".p":
        return _load_from_pfile(PFile(source))
    elif source.suffix.lower() == ".nc":
        return _load_from_nc(source)
    else:
        raise ValueError(f"Unsupported file type: {source.suffix}")


def _load_from_pfile(pf: "PFile") -> dict[str, Any]:
    """Extract data dict from a PFile."""
    channels = []
    for ch_name, ch_data in pf.channels.items():
        dim = "time_fast" if pf.is_fast(ch_name) else "time_slow"
        attrs = {
            "units": pf.channel_info[ch_name]["units"],
            "sensor_type": pf.channel_info[ch_name]["type"],
            "long_name": ch_name,
        }
        channels.append((ch_name, ch_data, dim, attrs))

    global_attrs = {
        "Conventions": "CF-1.13, ACDD-1.3",
        "instrument_model": pf.config["instrument_info"].get("model", ""),
        "instrument_sn": pf.config["instrument_info"].get("sn", ""),
        "operator": pf.config["cruise_info"].get("operator", ""),
        "project": pf.config["cruise_info"].get("project", ""),
        "start_time": pf.start_time.isoformat(),
        "source_file": pf.filepath.name,
    }
    if hasattr(pf, "config_str"):
        global_attrs["configuration_string"] = pf.config_str

    return {
        "P": pf.channels["P"],
        "fs_fast": pf.fs_fast,
        "fs_slow": pf.fs_slow,
        "t_fast": pf.t_fast,
        "t_slow": pf.t_slow,
        "t_fast_units": f"seconds since {pf.start_time.isoformat()}",
        "t_slow_units": f"seconds since {pf.start_time.isoformat()}",
        "start_epoch_s": pf.start_time.timestamp(),
        "channels": channels,
        "global_attrs": global_attrs,
        "stem": pf.filepath.stem,
    }


def _load_from_nc(nc_path: Path) -> dict[str, Any]:
    """Extract data dict from a full-record NetCDF file."""
    import netCDF4 as nc

    ds = nc.Dataset(str(nc_path), "r")

    # Read global attributes
    global_attrs = {}
    for attr in ds.ncattrs():
        if attr != "configuration_string":
            global_attrs[attr] = getattr(ds, attr)
    if "configuration_string" in ds.ncattrs():
        global_attrs["configuration_string"] = ds.configuration_string

    # Attributes may be on root or in L1_converted group
    def _get_nc_attr(name: str, default=None):
        for src in [ds]:
            if name in {a for a in src.ncattrs()}:
                return src.getncattr(name)
        if "L1_converted" in ds.groups:
            g = ds.groups["L1_converted"]
            if name in {a for a in g.ncattrs()}:
                return g.getncattr(name)
        if default is not None:
            return default
        raise AttributeError(f"Attribute {name!r} not found in NetCDF file")

    fs_fast = float(_get_nc_attr("fs_fast"))
    fs_slow = float(_get_nc_attr("fs_slow"))

    # Time vectors may be at root or inside L1_converted group
    if "t_fast" in ds.variables:
        t_fast = ds.variables["t_fast"][:].data
        t_slow = ds.variables["t_slow"][:].data
    elif "L1_converted" in ds.groups:
        g = ds.groups["L1_converted"]
        # L1_converted uses TIME/TIME_SLOW dimension names
        t_fast = g.variables["TIME"][:].data
        t_slow = g.variables["TIME_SLOW"][:].data
    else:
        raise ValueError("No time variables found in NetCDF file")
    # Determine time units
    if "t_fast" in ds.variables:
        t_fast_units = (
            ds.variables["t_fast"].units if hasattr(ds.variables["t_fast"], "units") else "seconds"
        )
        t_slow_units = (
            ds.variables["t_slow"].units if hasattr(ds.variables["t_slow"], "units") else "seconds"
        )
    elif "L1_converted" in ds.groups:
        g = ds.groups["L1_converted"]
        t_fast_units = (
            g.variables["TIME"].units if hasattr(g.variables["TIME"], "units") else "days"
        )
        t_slow_units = (
            g.variables["TIME_SLOW"].units if hasattr(g.variables["TIME_SLOW"], "units") else "days"
        )
    else:
        t_fast_units = "seconds"
        t_slow_units = "seconds"

    # Pressure — need slow-rate pressure for profile detection
    if "P" in ds.variables:
        P = ds.variables["P"][:].data
    elif "L1_converted" in ds.groups:
        g = ds.groups["L1_converted"]
        if "PRES_SLOW" in g.variables:
            P = g.variables["PRES_SLOW"][:].data
        elif "PRES" in g.variables:
            P = g.variables["PRES"][:].data
        else:
            raise ValueError("No pressure variable found in NetCDF file")
    else:
        raise ValueError("No pressure variable found in NetCDF file")

    # Channels — scan both root and L1_converted group
    channels = []
    _seen = set()

    def _scan_vars(source, time_dim_fast, time_dim_slow):
        for vname in source.variables:
            if vname in _seen or vname in ("t_fast", "t_slow", "TIME", "TIME_SLOW"):
                continue
            var = source.variables[vname]
            dims = var.dimensions
            if len(dims) != 1:
                continue
            dim = dims[0]
            if dim == time_dim_fast:
                mapped_dim = "time_fast"
            elif dim == time_dim_slow:
                mapped_dim = "time_slow"
            else:
                continue
            attrs = {}
            for a in var.ncattrs():
                attrs[a] = getattr(var, a)
            channels.append((vname, var[:].data.astype(np.float64), mapped_dim, attrs))
            _seen.add(vname)

    # Scan root-level variables (old format)
    _scan_vars(ds, "time_fast", "time_slow")
    # Scan L1_converted group (new ATOMIX format)
    if "L1_converted" in ds.groups:
        _scan_vars(ds.groups["L1_converted"], "TIME", "TIME_SLOW")

    # If time is in days, convert to seconds for consistency
    if "day" in t_fast_units.lower():
        t_fast = (t_fast - t_fast[0]) * 86400.0
        t_slow = (t_slow - t_slow[0]) * 86400.0
        t_fast_units = "seconds"
        t_slow_units = "seconds"

    ds.close()

    return {
        "P": P.astype(np.float64),
        "fs_fast": fs_fast,
        "fs_slow": fs_slow,
        "t_fast": t_fast,
        "t_slow": t_slow,
        "t_fast_units": t_fast_units,
        "t_slow_units": t_slow_units,
        "channels": channels,
        "global_attrs": global_attrs,
        "stem": Path(nc_path).stem,
    }


def _extract_one(args: tuple) -> tuple[str, int]:
    """Worker function for parallel profile extraction."""
    source_path, output_dir, kwargs = args
    paths = extract_profiles(source_path, output_dir, **kwargs)
    return str(source_path), len(paths)
