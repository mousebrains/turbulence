# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Hotel file support — external telemetry from gliders/AUVs/Remus.

Hotel files provide vehicle-mounted sensor data (speed, pitch, roll, heading,
CTD) that gets interpolated onto instrument time axes as new channels.

The ``hotel.channels`` YAML block supports three forms per source name:

    channels:
      # Take with all defaults from the section
      lat:
      # Legacy flat name map: rename source -> output
      m_speed: "speed"
      # Per-variable options
      pitch:
        name: "theta"        # rename target (default: same as source)
        interp: "nearest"    # override hotel.interpolation for this var
        scale: 0.0174533     # multiplicative factor (default 1.0)
        offset: 0.0          # additive offset (default 0.0)
        units: "rad"         # CF units string (default: source file's units)
        fast: false          # interpolate to fast rate? (default: name in hotel.fast_channels)

If ``channels`` is empty or omitted, every source variable is loaded with
default options. Otherwise only the source names listed are kept.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Interpolation kinds we accept. ``pchip`` uses scipy's PchipInterpolator; the
# others are passed straight through to ``scipy.interpolate.interp1d`` as the
# ``kind`` argument.
_INTERP_KINDS = frozenset({
    "pchip", "linear", "nearest", "previous", "next",
    "zero", "slinear", "quadratic", "cubic",
})

_CHANNEL_OPTION_KEYS = frozenset({
    "name", "interp", "scale", "offset", "units", "fast",
})


@dataclass
class HotelData:
    """Container for loaded hotel file data.

    Attributes
    ----------
    time : np.ndarray
        Time vector in epoch seconds (or relative seconds).
    channels : dict[str, np.ndarray]
        Source channel name → data array. Renaming and per-variable
        transforms are applied later by :func:`merge_hotel_into_pfile`.
    units : dict[str, str]
        Source channel name → CF-compatible units string from the file.
        Empty if not known. The merge step may override these.
    time_is_relative : bool
        True if time values are relative seconds (not epoch).
    """

    time: np.ndarray
    channels: dict[str, np.ndarray] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    time_is_relative: bool = False


def _normalize_channels_cfg(
    channels_cfg: dict | None,
) -> tuple[bool, dict[str, dict]]:
    """Parse the YAML ``hotel.channels`` block.

    Returns ``(filter_active, options)`` where:

    - ``filter_active`` is ``False`` for an empty / missing block (meaning
      "take every source channel with default options"), and ``True`` when
      the user explicitly listed source names.
    - ``options`` maps source_name → ``{"name", "interp", "scale", "offset",
      "units", "fast"}`` (any subset; missing keys mean "use the default").

    Per-source values may be:

    - ``None`` or ``{}`` — include this source with all defaults.
    - a string — legacy "rename to this output name".
    - a dict — full per-variable options.
    """
    if not channels_cfg:
        return False, {}
    out: dict[str, dict] = {}
    for src, val in channels_cfg.items():
        if val is None or val == {}:
            out[src] = {}
        elif isinstance(val, str):
            out[src] = {"name": val}
        elif isinstance(val, dict):
            unknown = set(val) - _CHANNEL_OPTION_KEYS
            if unknown:
                raise ValueError(
                    f"hotel.channels[{src!r}]: unknown options {sorted(unknown)}. "
                    f"Valid: {sorted(_CHANNEL_OPTION_KEYS)}"
                )
            interp_kind = val.get("interp")
            if interp_kind is not None and interp_kind not in _INTERP_KINDS:
                raise ValueError(
                    f"hotel.channels[{src!r}].interp={interp_kind!r}: not in "
                    f"{sorted(_INTERP_KINDS)}"
                )
            out[src] = dict(val)
        else:
            raise ValueError(
                f"hotel.channels[{src!r}]: must be a string, dict, or null; "
                f"got {type(val).__name__}"
            )
    return True, out


def _parse_time(raw_time: np.ndarray, time_format: str) -> tuple[np.ndarray, bool]:
    """Convert raw time values to epoch seconds.

    Returns (time_array, is_relative).
    """
    if time_format == "seconds":
        return raw_time.astype(np.float64), True
    elif time_format == "epoch":
        return raw_time.astype(np.float64), False
    elif time_format == "iso":
        import pandas as pd

        t = pd.to_datetime(raw_time).values.astype("datetime64[ns]").astype(np.int64) / 1e9
        return t, False
    elif time_format == "auto":
        vals = raw_time.astype(np.float64)
        median = np.median(vals)
        if median < 1e6:
            return vals, True
        elif median > 1e9:
            return vals, False
        else:
            # Try ISO parse
            import pandas as pd

            try:
                t = pd.to_datetime(raw_time).values.astype("datetime64[ns]").astype(np.int64) / 1e9
                return t, False
            except Exception:
                return vals, True
    else:
        raise ValueError(f"Unknown time_format: {time_format!r}")


def load_hotel(
    path: str | Path,
    time_column: str = "time",
    time_format: str = "auto",
    channels: dict | None = None,
) -> HotelData:
    """Load a hotel file (CSV, NetCDF, or .mat).

    Parameters
    ----------
    path : str or Path
        Path to hotel file.
    time_column : str
        Name of the time column/variable.
    time_format : str
        Time format: ``"auto"``, ``"seconds"``, ``"epoch"``, ``"iso"``.
    channels : dict, optional
        ``hotel.channels`` block (see module docstring). Only the source
        names listed are loaded; any rename / scale / offset / units /
        interp / fast options are applied later by
        :func:`merge_hotel_into_pfile`. ``None`` or ``{}`` loads every
        source channel under its native name.

    Returns
    -------
    HotelData
        Channels and units keyed by *source* name (no rename applied).
    """
    path = Path(path)
    ext = path.suffix.lower()

    filter_active, _ = _normalize_channels_cfg(channels)
    allowed = set(channels.keys()) if filter_active and channels else None

    if ext == ".csv":
        return _load_csv(path, time_column, time_format, allowed)
    elif ext in (".nc", ".nc4"):
        return _load_netcdf(path, time_column, time_format, allowed)
    elif ext == ".mat":
        return _load_mat(path, time_column, time_format, allowed)
    else:
        raise ValueError(f"Unsupported hotel file format: {ext!r}. Supported: .csv, .nc, .mat")


def _load_csv(
    path: Path,
    time_column: str,
    time_format: str,
    allowed: set[str] | None,
) -> HotelData:
    import pandas as pd

    df = pd.read_csv(path)
    raw_time = np.asarray(df[time_column].values)
    time, is_relative = _parse_time(raw_time, time_format)

    data_cols = [c for c in df.columns if c != time_column]
    if allowed is not None:
        data_cols = [c for c in data_cols if c in allowed]
    ch = {c: df[c].values.astype(np.float64) for c in data_cols}
    units = dict.fromkeys(ch, "")
    return HotelData(time=time, channels=ch, units=units, time_is_relative=is_relative)


def _load_netcdf(
    path: Path,
    time_column: str,
    time_format: str,
    allowed: set[str] | None,
) -> HotelData:
    import netCDF4 as nc

    ds = nc.Dataset(str(path), "r")
    raw_time = ds.variables[time_column][:].data
    time, is_relative = _parse_time(raw_time, time_format)

    data_vars = [v for v in ds.variables if v != time_column]
    if allowed is not None:
        data_vars = [v for v in data_vars if v in allowed]
    ch: dict[str, np.ndarray] = {}
    units: dict[str, str] = {}
    for v in data_vars:
        var = ds.variables[v]
        ch[v] = var[:].data.astype(np.float64)
        units[v] = getattr(var, "units", "") or ""
    ds.close()

    return HotelData(time=time, channels=ch, units=units, time_is_relative=is_relative)


def _load_mat(
    path: Path,
    time_column: str,
    time_format: str,
    allowed: set[str] | None,
) -> HotelData:
    from scipy.io import loadmat

    mat = loadmat(str(path), squeeze_me=True)

    # ODAS convention: struct fields with .data/.time subfields
    # Also handle flat arrays
    ch: dict[str, np.ndarray] = {}
    raw_time: np.ndarray | None = None

    for key, val in mat.items():
        if key.startswith("_"):
            continue
        if (
            hasattr(val, "dtype")
            and val.dtype.names
            and "time" in val.dtype.names
            and "data" in val.dtype.names
        ):
            struct = val.flat[0] if val.ndim > 0 else val
            if key == time_column:
                raw_time = np.asarray(struct["time"]).flatten().astype(np.float64)
            else:
                if allowed is None or key in allowed:
                    ch[key] = np.asarray(struct["data"]).flatten().astype(np.float64)
                    if raw_time is None:
                        raw_time = np.asarray(struct["time"]).flatten().astype(np.float64)
                continue
        # Flat array
        arr = np.asarray(val).flatten()
        if key == time_column:
            raw_time = arr.astype(np.float64)
        else:
            if allowed is None or key in allowed:
                ch[key] = arr.astype(np.float64)

    if raw_time is None:
        raise ValueError(f"Time column {time_column!r} not found in .mat file")

    time, is_relative = _parse_time(raw_time, time_format)
    units = dict.fromkeys(ch, "")
    return HotelData(time=time, channels=ch, units=units, time_is_relative=is_relative)


def _interp_one(
    hotel_t: np.ndarray,
    data: np.ndarray,
    target_t: np.ndarray,
    kind: str,
) -> np.ndarray:
    """Interpolate one channel onto ``target_t`` with the requested kind."""
    from scipy.interpolate import PchipInterpolator, interp1d

    if kind == "pchip":
        interp = PchipInterpolator(hotel_t, data, extrapolate=False)
        out = interp(target_t)
        # Fill NaN edges with boundary values, matching the historical behaviour.
        mask = np.isnan(out)
        if np.any(mask):
            below = target_t < hotel_t[0]
            above = target_t > hotel_t[-1]
            out[mask & below] = data[0]
            out[mask & above] = data[-1]
        return np.asarray(out, dtype=np.float64)
    interp = interp1d(
        hotel_t, data, kind=kind, bounds_error=False,
        fill_value=(data[0], data[-1]),
    )
    return np.asarray(interp(target_t), dtype=np.float64)


def interpolate_hotel(hotel_data: HotelData, pf, hotel_cfg: dict) -> dict[str, np.ndarray]:
    """Interpolate hotel channels onto PFile time axes.

    Honors per-variable ``interp`` overrides from the ``hotel.channels``
    block; falls back to the global ``hotel.interpolation`` default for any
    channel without an explicit override.

    Returns a dict keyed by *source* channel name. The merge helper applies
    rename / scale / offset / units / fast overrides on top of this.
    """
    fast_channels = set(hotel_cfg.get("fast_channels", ["speed", "P"]))
    default_kind = hotel_cfg.get("interpolation", "pchip")
    if default_kind not in _INTERP_KINDS:
        raise ValueError(
            f"hotel.interpolation={default_kind!r}: not in {sorted(_INTERP_KINDS)}"
        )
    _, channels_opts = _normalize_channels_cfg(hotel_cfg.get("channels"))

    if hotel_data.time_is_relative:
        hotel_t = hotel_data.time
    else:
        hotel_t = hotel_data.time - pf.start_time.timestamp()

    result: dict[str, np.ndarray] = {}
    for src, data in hotel_data.channels.items():
        opts = channels_opts.get(src, {})
        out_name = opts.get("name", src)
        kind = opts.get("interp") or default_kind
        if "fast" in opts:
            target_t = pf.t_fast if opts["fast"] else pf.t_slow
        else:
            target_t = pf.t_fast if out_name in fast_channels else pf.t_slow
        result[src] = _interp_one(hotel_t, data, target_t, kind)

    return result


def merge_hotel_into_pfile(hotel_data: HotelData, pf, hotel_cfg: dict) -> None:
    """Interpolate hotel channels and register them on ``pf`` in-place.

    Applies the full ``hotel.channels`` schema: per-variable rename, interp
    method, ``scale`` / ``offset`` linear transform, ``units`` override, and
    ``fast`` rate override.

    Adds each resulting channel to ``pf.channels`` under the (possibly
    renamed) output name, registers a ``pf.channel_info`` entry so
    :func:`extract_profiles` can read units, and updates
    ``pf._fast_channels`` so :meth:`PFile.is_fast` returns the correct dim.
    Names that already exist on ``pf`` are overwritten and their fast/slow
    membership is rewritten to match the hotel choice.
    """
    fast_channels = set(hotel_cfg.get("fast_channels", ["speed", "P"]))
    _, channels_opts = _normalize_channels_cfg(hotel_cfg.get("channels"))
    interpolated = interpolate_hotel(hotel_data, pf, hotel_cfg)

    for src, data in interpolated.items():
        opts = channels_opts.get(src, {})
        out_name = opts.get("name", src)
        scale = float(opts.get("scale", 1.0))
        offset = float(opts.get("offset", 0.0))
        if scale != 1.0 or offset != 0.0:
            data = data * scale + offset
        pf.channels[out_name] = data

        units = opts.get("units")
        if units is None:
            units = hotel_data.units.get(src, "")
        pf.channel_info[out_name] = {
            "units": units,
            "type": "hotel",
            "name": out_name,
        }

        is_fast = bool(opts["fast"]) if "fast" in opts else out_name in fast_channels
        if is_fast:
            pf._fast_channels.add(out_name)
        else:
            pf._fast_channels.discard(out_name)
