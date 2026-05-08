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
        time_column: "time_flight"   # NetCDF time variable for this channel
                                     # (default: hotel.time_column). Lets a single
                                     # hotel file carry channels on multiple native
                                     # time grids, e.g. CTD on time_sci, flight
                                     # vars on time_flight, sparse modem events
                                     # on time_modem_event.

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
    "name", "interp", "scale", "offset", "units", "fast", "time_column",
})


@dataclass
class HotelData:
    """Container for loaded hotel file data.

    Attributes
    ----------
    time : np.ndarray
        Default time vector for channels that don't override
        ``time_column``. Epoch seconds (or relative seconds — see
        ``time_is_relative``).
    channels : dict[str, np.ndarray]
        Source channel name → data array. Renaming and per-variable
        transforms are applied later by :func:`merge_hotel_into_pfile`.
    channel_times : dict[str, np.ndarray]
        Per-channel time array override. Keys are source channel names;
        values are time arrays in the same units as ``time``. Channels
        not in this dict use the default ``time``. Lets a single hotel
        file carry channels on multiple native time grids.
    units : dict[str, str]
        Source channel name → CF-compatible units string from the file.
        Empty if not known. The merge step may override these.
    time_is_relative : bool
        True if time values are relative seconds (not epoch). Applies to
        every time array (default and per-channel) — mixing relative and
        absolute time arrays in one hotel file is not supported.
    """

    time: np.ndarray
    channels: dict[str, np.ndarray] = field(default_factory=dict)
    channel_times: dict[str, np.ndarray] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    time_is_relative: bool = False

    def time_for(self, channel: str) -> np.ndarray:
        """Time array a channel lives on (override → default fallback)."""
        return self.channel_times.get(channel, self.time)


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


def _per_channel_time_columns(
    options: dict[str, dict], default_time_column: str,
) -> dict[str, str]:
    """Map source-name → time variable name for any channel overriding it.

    Returns ``{channel: time_column}`` only for channels whose option dict
    has a ``time_column`` different from *default_time_column*. Channels
    that don't override are left out (they implicitly use the default).
    """
    out: dict[str, str] = {}
    for src, opts in options.items():
        tc = opts.get("time_column")
        if tc and tc != default_time_column:
            out[src] = tc
    return out


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
        Name of the *default* time column / variable. Channels in the
        ``channels`` mapping may override this with their own
        ``time_column`` to ride a different time grid in the same file.
    time_format : str
        Time format: ``"auto"``, ``"seconds"``, ``"epoch"``, ``"iso"``.
    channels : dict, optional
        ``hotel.channels`` block (see module docstring). Only the source
        names listed are loaded; rename / scale / offset / units /
        interp / fast / time_column options are applied later by
        :func:`merge_hotel_into_pfile`. ``None`` or ``{}`` loads every
        source channel under its native name on the default time grid.

    Returns
    -------
    HotelData
        Channels and units keyed by *source* name (no rename applied).
        ``HotelData.channel_times`` is populated for any channel using
        a non-default ``time_column``; the rest use ``HotelData.time``.
    """
    path = Path(path)
    ext = path.suffix.lower()

    filter_active, options = _normalize_channels_cfg(channels)
    allowed = set(channels.keys()) if filter_active and channels else None
    per_chan_time = _per_channel_time_columns(options, time_column)

    if ext == ".csv":
        return _load_csv(path, time_column, time_format, allowed, per_chan_time)
    elif ext in (".nc", ".nc4"):
        return _load_netcdf(path, time_column, time_format, allowed, per_chan_time)
    elif ext == ".mat":
        return _load_mat(path, time_column, time_format, allowed, per_chan_time)
    else:
        raise ValueError(f"Unsupported hotel file format: {ext!r}. Supported: .csv, .nc, .mat")


def _load_csv(
    path: Path,
    time_column: str,
    time_format: str,
    allowed: set[str] | None,
    per_chan_time: dict[str, str],
) -> HotelData:
    import pandas as pd

    df = pd.read_csv(path)
    raw_time = np.asarray(df[time_column].values)
    time, is_relative = _parse_time(raw_time, time_format)

    extra_time_cols = set(per_chan_time.values())
    data_cols = [c for c in df.columns if c != time_column and c not in extra_time_cols]
    if allowed is not None:
        data_cols = [c for c in data_cols if c in allowed]
    ch = {c: df[c].values.astype(np.float64) for c in data_cols}
    units = dict.fromkeys(ch, "")

    channel_times: dict[str, np.ndarray] = {}
    for src, tc in per_chan_time.items():
        if allowed is not None and src not in allowed:
            continue
        if tc not in df.columns:
            raise ValueError(f"hotel: time_column {tc!r} not found in {path}")
        t_arr, _ = _parse_time(np.asarray(df[tc].values), time_format)
        channel_times[src] = t_arr

    return HotelData(time=time, channels=ch, channel_times=channel_times,
                     units=units, time_is_relative=is_relative)


def _load_netcdf(
    path: Path,
    time_column: str,
    time_format: str,
    allowed: set[str] | None,
    per_chan_time: dict[str, str],
) -> HotelData:
    import netCDF4 as nc

    ds = nc.Dataset(str(path), "r")
    raw_time = ds.variables[time_column][:].data
    time, is_relative = _parse_time(raw_time, time_format)

    extra_time_cols = set(per_chan_time.values())
    skip = {time_column} | extra_time_cols
    data_vars = [v for v in ds.variables if v not in skip]
    if allowed is not None:
        data_vars = [v for v in data_vars if v in allowed]
    ch: dict[str, np.ndarray] = {}
    units: dict[str, str] = {}
    for v in data_vars:
        var = ds.variables[v]
        ch[v] = var[:].data.astype(np.float64)
        units[v] = getattr(var, "units", "") or ""

    channel_times: dict[str, np.ndarray] = {}
    for src, tc in per_chan_time.items():
        if allowed is not None and src not in allowed:
            continue
        if tc not in ds.variables:
            ds.close()
            raise ValueError(f"hotel: time_column {tc!r} not found in {path}")
        raw = ds.variables[tc][:].data
        t_arr, _ = _parse_time(raw, time_format)
        channel_times[src] = t_arr
    ds.close()

    return HotelData(time=time, channels=ch, channel_times=channel_times,
                     units=units, time_is_relative=is_relative)


def _load_mat(
    path: Path,
    time_column: str,
    time_format: str,
    allowed: set[str] | None,
    per_chan_time: dict[str, str],
) -> HotelData:
    from scipy.io import loadmat

    mat = loadmat(str(path), squeeze_me=True)

    # ODAS convention: struct fields with .data/.time subfields
    # Also handle flat arrays
    ch: dict[str, np.ndarray] = {}
    raw_time: np.ndarray | None = None
    extra_time_cols = set(per_chan_time.values())
    extra_time_arrays: dict[str, np.ndarray] = {}

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
            elif key in extra_time_cols:
                extra_time_arrays[key] = (
                    np.asarray(struct["time"]).flatten().astype(np.float64)
                )
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
        elif key in extra_time_cols:
            extra_time_arrays[key] = arr.astype(np.float64)
        else:
            if allowed is None or key in allowed:
                ch[key] = arr.astype(np.float64)

    if raw_time is None:
        raise ValueError(f"Time column {time_column!r} not found in .mat file")

    time, is_relative = _parse_time(raw_time, time_format)
    units = dict.fromkeys(ch, "")

    channel_times: dict[str, np.ndarray] = {}
    for src, tc in per_chan_time.items():
        if allowed is not None and src not in allowed:
            continue
        if tc not in extra_time_arrays:
            raise ValueError(f"hotel: time_column {tc!r} not found in {path}")
        t_arr, _ = _parse_time(extra_time_arrays[tc], time_format)
        channel_times[src] = t_arr

    return HotelData(time=time, channels=ch, channel_times=channel_times,
                     units=units, time_is_relative=is_relative)


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

    pf_start_offset = 0.0 if hotel_data.time_is_relative else pf.start_time.timestamp()

    result: dict[str, np.ndarray] = {}
    for src, data in hotel_data.channels.items():
        opts = channels_opts.get(src, {})
        out_name = opts.get("name", src)
        kind = opts.get("interp") or default_kind
        if "fast" in opts:
            target_t = pf.t_fast if opts["fast"] else pf.t_slow
        else:
            target_t = pf.t_fast if out_name in fast_channels else pf.t_slow
        hotel_t = hotel_data.time_for(src) - pf_start_offset
        # Sparse time grids (e.g. one-row-per-event modem listings) can have
        # a single sample, which interpolators won't accept. Skip — caller
        # gets nothing for that channel, same as a missing channel.
        if hotel_t.size < 2:
            continue
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
