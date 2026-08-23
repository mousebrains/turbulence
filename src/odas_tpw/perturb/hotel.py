# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Hotel file support — external telemetry from gliders/AUVs/Remus.

Hotel files provide vehicle-mounted sensor data (speed, pitch, roll, heading,
CTD) that gets interpolated onto instrument time axes as new channels.

The ``hotel.channels`` YAML block supports three forms per source name:

    channels:
      # Take with all defaults from the section
      lat:
      # Legacy flat name map: rename source -> output. Mapping a source
      # variable onto the output name "speed" pairs with the perturb
      # ``speed.method: "hotel"`` (see ``speed.hotel_var``), which uses the
      # merged channel as the through-water speed for epsilon/chi. The
      # default ``hotel.fast_channels`` puts "speed" on the FAST grid.
      m_speed: "speed"
      # Per-variable options
      pitch:
        name: "theta"        # rename target (default: same as source)
        interp: "nearest"    # override hotel.interpolation for this var
        max_gap: 30.0        # [s] NaN the output where the two bracketing
                             # SOURCE samples are farther apart than this,
                             # instead of ruling a straight line across the
                             # hole (default: hotel.max_gap, which is
                             # REQUIRED -- a number or "unlimited"; a
                             # per-channel null inherits it)
        extrapolate: false   # NaN outside the source's own time range instead
                             # of holding the end values (default:
                             # hotel.extrapolate, itself false; a per-channel
                             # null inherits it)
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

Gaps
----
``hotel.max_gap`` and ``hotel.extrapolate`` (with per-channel overrides above)
control what happens where the source has **no data**.

``max_gap`` is **required**. There is no safe default: the right limit is the
sensor's own rate --- tens of seconds for a 1 Hz CTD, minutes for a flight-state
variable --- and guessing wrong is silent in both directions. Omitting it raises.
``extrapolate`` defaults to ``False``.

The reason for the strictness: on an instrument whose CTD ran on only some
profiles, an ungated merge produces a smooth *fabricated* ramp between real
samples hours apart, and every consumer (``ct``, ``ctd``, ``stratification``,
``salinity: "measured"``, ``epsilon.T_source``) reads it as data. It also
silently undid gap control applied upstream: a builder that NaN-marks its own
dropouts (``dinkum-hotel``'s ``projection.max_gap``) had that erased, because
the loader drops non-finite samples and then interpolates across the hole.

``max_gap: "unlimited"`` restores the old interpolate-across-anything behaviour
for a channel or for the whole block --- deliberately, in writing --- and warns
whenever it actually fabricates. That warning is scaled to the channel's own
median sample interval, so it fires on real dropouts and not on sampling jitter.
"""

import warnings
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
    "name", "interp", "scale", "offset", "units", "fast", "time_column", "replace",
    "max_gap", "extrapolate",
})

# A gap this many times the channel's own median sample interval is not normal
# sampling jitter --- it is a dropout, and interpolating across it manufactures
# data.  Used only to decide when to WARN, so it needs no configuration and
# scales itself to each channel's rate.
_GAP_WARN_FACTOR = 10.0

# The one value that opts out of the required gap limit.  A string rather than
# ``null`` on purpose: ``merge_config`` drops nulls, so an explicit null is
# indistinguishable from "not set" -- and the opt-out has to be something the
# operator typed deliberately, not something they left blank.
UNLIMITED = "unlimited"


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
      "units", "fast", "replace"}`` (any subset; missing keys mean "use the
      default"). ``replace: true`` permits a hotel channel to overwrite a
      native instrument channel of the same output name (refused otherwise;
      U5-1) — pair it with ``fast: false`` when overwriting a slow native
      channel such as ``P``, or the fast-gridded replacement is dropped.

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


def _nc_array(var) -> np.ndarray:
    """Read a netCDF4 variable, mapping masked/_FillValue cells to NaN.

    netCDF4 auto-masks by default; reaching for ``.data`` would expose the raw
    _FillValue (e.g. -999, 1e20) as a real measurement. Hotel telemetry stores
    gaps as fill values, so a -999 speed/pitch/CTD would be interpolated onto
    the instrument axis and corrupt the dissipation/chi estimates silently.
    """
    arr = var[:]
    if np.ma.isMaskedArray(arr):
        # Cast to float before filling: filling an integer-dtype masked array
        # with NaN raises (NaN cannot be cast to int).
        return np.asarray(np.ma.filled(arr.astype(np.float64), np.nan))
    return np.asarray(arr)


def _dt64_to_epoch_s(values: np.ndarray) -> np.ndarray:
    """datetime64 -> epoch seconds (float64), mapping NaT to NaN.

    NaT casts to the int64 minimum; converting straight to seconds would yield a
    bogus ~-9.2e9 s epoch that then poisons interpolation. Detect the sentinel
    and emit NaN instead.
    """
    ns = np.asarray(values).astype("datetime64[ns]").astype(np.int64)
    secs = ns.astype(np.float64) / 1e9
    secs[ns == np.iinfo(np.int64).min] = np.nan
    return secs


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

        t = _dt64_to_epoch_s(pd.to_datetime(raw_time).values)
        return t, False
    elif time_format == "auto":
        if not np.issubdtype(raw_time.dtype, np.number):
            # Non-numeric (ISO strings / datetime64 / object) -> parse as dates.
            # Feeding these to ``.astype(float64)`` below would raise, so the
            # old code never actually reached its ISO fallback for string input.
            import pandas as pd

            t = _dt64_to_epoch_s(pd.to_datetime(raw_time).values)
            return t, False
        vals = raw_time.astype(np.float64)
        # nanmedian: a single NaN timestamp must not blank the whole test and
        # push every file through the ambiguous branch.
        median = np.nanmedian(vals)
        if median < 1e6:
            return vals, True
        elif median > 1e9:
            return vals, False
        else:
            # Ambiguous numeric magnitude: epoch seconds here mean 1970..2001,
            # while relative seconds mean an 11.6-day..31.7-year span -- both
            # plausible. pd.to_datetime would silently read bare numbers as
            # nanoseconds-since-epoch (garbage), so refuse to guess.
            raise ValueError(
                f"hotel time_format='auto' cannot disambiguate numeric times "
                f"with median {median:g} s (in [1e6, 1e9)). Set time_format "
                f"explicitly to 'seconds' (relative) or 'epoch' (POSIX)."
            )
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

    # Context manager closes the Dataset even if _parse_time or a variable read
    # raises partway through (e.g. an ambiguous-time ValueError).
    with nc.Dataset(str(path), "r") as ds:
        raw_time = _nc_array(ds.variables[time_column])
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
            ch[v] = _nc_array(var).astype(np.float64)
            units[v] = getattr(var, "units", "") or ""

        channel_times: dict[str, np.ndarray] = {}
        for src, tc in per_chan_time.items():
            if allowed is not None and src not in allowed:
                continue
            if tc not in ds.variables:
                raise ValueError(f"hotel: time_column {tc!r} not found in {path}")
            raw = _nc_array(ds.variables[tc])
            t_arr, _ = _parse_time(raw, time_format)
            channel_times[src] = t_arr

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


# Relative tolerance for "the target lands ON a source sample" (see
# _bridged_gap), scaled by max(1 s, median source interval).
_MEASURED_TOL = 1e-6


def _bridged_gap(hotel_t: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    """Width of the source gap each target sample is interpolated ACROSS.

    Zero for targets outside the source range (extrapolation, handled
    separately) and zero for a target that lands exactly ON a source sample:
    that value is **measured**, not interpolated, so no gap was bridged to
    obtain it.

    The exact-match case is not academic.  ``searchsorted(side="left")`` returns
    the index of the matching sample, so without the check the first real
    sample after a dropout inherits the dropout's width and is thrown away by
    ``max_gap`` --- discarding an observation to protect against interpolation
    that never happened.  It also matters whenever the source and target share
    a clock, where every target is an exact match and the gate would fire on
    all of them.
    """
    n = hotel_t.size
    idx = np.searchsorted(hotel_t, target_t, side="left")
    inside = (target_t >= hotel_t[0]) & (target_t <= hotel_t[-1])
    lo = np.clip(idx - 1, 0, n - 1)
    hi = np.clip(idx, 0, n - 1)
    # Float tolerance: target times are exact k/fs but hotel epochs shifted by
    # the file start have ~1e-7 s resolution, so an exact match is luck. A
    # target within a sliver of a source sample is that sample, not the gap.
    dt = np.diff(hotel_t)
    median_dt = float(np.median(dt)) if dt.size else 1.0
    tol = _MEASURED_TOL * max(1.0, median_dt)
    # Check both bracketing neighbours: a target a sliver AFTER a sample has
    # ``hi`` pointing at the next one, so hi alone would miss the match.
    measured = (np.abs(hotel_t[hi] - target_t) <= tol) | (
        np.abs(hotel_t[lo] - target_t) <= tol
    )
    return np.where(inside & ~measured, hotel_t[hi] - hotel_t[lo], 0.0)


def _interp_one(
    hotel_t: np.ndarray,
    data: np.ndarray,
    target_t: np.ndarray,
    kind: str,
    *,
    max_gap: float | None = None,
    extrapolate: bool = False,
    stats: dict | None = None,
) -> np.ndarray:
    """Interpolate one channel onto ``target_t`` with the requested kind.

    ``max_gap`` [s] NaNs the output wherever the two bracketing source samples
    are farther apart than that, instead of ruling a straight line across the
    hole.  ``extrapolate=False`` NaNs the output outside the source's own time
    range instead of holding the end values.

    ``extrapolate`` defaults to False: an edge-held constant outside the
    source's coverage correlates with nothing and is not a measurement.
    ``max_gap=None`` leaves this function ungated --- the *requirement* is
    enforced one level up in :func:`interpolate_hotel`, where the user's config
    lives, so library callers keep a usable primitive.

    Either way this function unconditionally *measures* how much of the output
    was manufactured and reports it through ``stats``.

    Why this matters: on an instrument whose CTD ran on only some profiles, the
    merged channel is a smooth fabricated ramp between real samples hours
    apart, and every consumer --- ``ct``, ``ctd``, ``stratification``,
    ``salinity: "measured"``, ``epsilon.T_source`` --- reads it as if it were
    data.
    """
    from scipy.interpolate import PchipInterpolator, interp1d

    target_t = np.asarray(target_t, dtype=np.float64)
    # Drop NaN samples (fill-valued gaps / NaT times): PchipInterpolator raises
    # on a NaN in the data, and interp1d would propagate it. With < 2 valid
    # points there is nothing to interpolate.
    finite = np.isfinite(hotel_t) & np.isfinite(data)
    if int(finite.sum()) < 2:
        if stats is not None:
            stats.update(n_target=int(target_t.size), n_gap=int(target_t.size),
                         n_outside=0, widest_gap=float("inf"), median_dt=float("nan"))
        return np.full(np.shape(target_t), np.nan, dtype=np.float64)
    hotel_t = np.asarray(hotel_t, dtype=np.float64)[finite]
    data = np.asarray(data, dtype=np.float64)[finite]
    # Sort: interp1d sorts internally, but the range / gap bookkeeping below
    # reads hotel_t[0] and hotel_t[-1] directly, and PchipInterpolator requires
    # increasing x. No loader sorts, and a CSV/NetCDF need not be in order.
    if np.any(np.diff(hotel_t) < 0):
        order = np.argsort(hotel_t, kind="stable")
        hotel_t = hotel_t[order]
        data = data[order]

    if kind == "pchip":
        interp = PchipInterpolator(hotel_t, data, extrapolate=False)
        out = np.asarray(interp(target_t), dtype=np.float64)
        mask = np.isnan(out)
        if np.any(mask):
            out[mask & (target_t < hotel_t[0])] = data[0]
            out[mask & (target_t > hotel_t[-1])] = data[-1]
    else:
        interp = interp1d(
            hotel_t, data, kind=kind, bounds_error=False,
            fill_value=(data[0], data[-1]),
        )
        out = np.asarray(interp(target_t), dtype=np.float64)

    gap = _bridged_gap(hotel_t, target_t)
    outside = (target_t < hotel_t[0]) | (target_t > hotel_t[-1])
    dt = np.diff(hotel_t)
    median_dt = float(np.median(dt)) if dt.size else float("nan")
    warn_gap = (
        _GAP_WARN_FACTOR * median_dt
        if np.isfinite(median_dt) and median_dt > 0
        else np.inf
    )
    notable = gap > warn_gap
    gated = max_gap is not None and np.isfinite(max_gap) and max_gap > 0
    rejected = (gap > max_gap) if gated else np.zeros_like(gap, dtype=bool)

    if stats is not None:
        # Rejected and merely-notable are tracked apart, because they are
        # different claims: one says data was thrown away, the other says data
        # was manufactured. A warning that conflates them can state the
        # opposite of what happened -- with max_gap above the warning
        # threshold, notable gaps are interpolated across and kept.
        stats.update(
            n_target=int(target_t.size),
            n_notable=int(np.count_nonzero(notable & ~rejected)),
            n_rejected=int(np.count_nonzero(rejected)),
            n_outside=int(np.count_nonzero(outside)),
            widest_gap=float(gap.max()) if gap.size else 0.0,
            widest_kept=float(gap[~rejected].max()) if np.any(~rejected) else 0.0,
            median_dt=median_dt,
            gated=bool(gated),
        )

    if gated:
        out[rejected] = np.nan
    if not extrapolate:
        out[outside] = np.nan
    return out


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
    # max_gap is required (a number or "unlimited") and extrapolate defaults to
    # False; resolve_gap_settings validates both, globally and per channel.
    gap_settings = resolve_gap_settings(hotel_cfg)
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
        max_gap, extrapolate = gap_settings.get(src, gap_settings[None])
        stats: dict = {}
        result[src] = _interp_one(
            hotel_t, data, target_t, kind,
            max_gap=None if max_gap is None else float(max_gap),
            extrapolate=extrapolate,
            stats=stats,
        )
        _warn_if_fabricated(src, stats, max_gap, extrapolate)

    return result


def resolve_gap_settings(hotel_cfg: dict) -> dict[str | None, tuple[float | None, bool]]:
    """Validate ``max_gap`` / ``extrapolate`` once, globally and per channel.

    Returns ``{source_name: (max_gap, extrapolate)}`` with the global defaults
    under the key ``None``; ``max_gap`` is a float in seconds or ``None`` for
    ``"unlimited"``.  A per-channel ``null`` (or an absent key) inherits the
    global value for either setting.

    Raises ``ValueError`` on a missing or malformed value.  Called at config
    time by the pipeline so a bad config fails once, up front, rather than
    once per file inside the worker pool.
    """
    default_max_gap = _resolve_max_gap(hotel_cfg.get("max_gap"), "hotel.max_gap")
    default_extrapolate = bool(hotel_cfg.get("extrapolate") or False)
    _, channels_opts = _normalize_channels_cfg(hotel_cfg.get("channels"))
    out: dict[str | None, tuple[float | None, bool]] = {
        None: (default_max_gap, default_extrapolate)
    }
    for src, opts in channels_opts.items():
        max_gap = (
            _resolve_max_gap(opts["max_gap"], f"hotel.channels[{src!r}].max_gap")
            if opts.get("max_gap") is not None
            else default_max_gap
        )
        extrapolate = (
            bool(opts["extrapolate"])
            if opts.get("extrapolate") is not None
            else default_extrapolate
        )
        out[src] = (max_gap, extrapolate)
    return out


def _resolve_max_gap(value, where: str) -> float | None:
    """Validate a gap limit.  Unset is an error; ``"unlimited"`` opts out.

    There is no defensible default. The right limit is the sensor's own rate --
    tens of seconds for a 1 Hz CTD, minutes for a flight-state variable -- and
    guessing it wrong in either direction is silent: too tight throws away good
    data, too loose manufactures it. So the operator has to say.
    """
    if isinstance(value, str):
        if value.strip().lower() == UNLIMITED:
            return None
        raise ValueError(
            f"{where}={value!r}: expected a number of seconds or the string "
            f"{UNLIMITED!r}"
        )
    if value is None:
        raise ValueError(
            f"{where} is required when hotel.enable is true. Where two "
            f"bracketing source samples are farther apart than this, the merge "
            f"NaNs the output instead of ruling a straight line across the "
            f"hole. There is no safe default -- the right limit is the sensor's "
            f"own sample rate (~30 for a 1 Hz CTD, minutes for a flight-state "
            f"variable). Interpolating across an unbounded gap manufactures "
            f"data that ct, ctd, stratification, salinity:\"measured\" and "
            f"epsilon.T_source all read as measurement. To deliberately keep "
            f"the old behaviour, set {where}: {UNLIMITED!r}."
        )
    gap = float(value)
    if not np.isfinite(gap) or gap <= 0:
        raise ValueError(f"{where}={value!r}: must be a positive number of seconds")
    return gap


def _warn_if_fabricated(
    src: str, stats: dict, max_gap: float | None, extrapolate: bool
) -> None:
    """Say how much of a merged channel was manufactured, or thrown away.

    Unconditional, and scaled to the channel's own median sample interval, so
    it fires on a real dropout without needing to be configured and without
    nagging about ordinary sampling jitter.  Silence here is the failure mode
    that motivated it: a CTD that ran on only some profiles produced a smooth
    fabricated ramp that every downstream consumer read as data.

    The two outcomes are reported separately and never conflated.  Saying
    "NaN-ed" about samples that were in fact interpolated across and kept is
    the same class of error the gate exists to prevent, one level up.
    """
    n = int(stats.get("n_target", 0))
    if not n:
        return
    n_notable = int(stats.get("n_notable", 0))
    n_rejected = int(stats.get("n_rejected", 0))
    n_out = int(stats.get("n_outside", 0))
    if not (n_notable or n_rejected or n_out):
        return

    med = stats.get("median_dt", float("nan"))
    widest = stats.get("widest_gap", float("nan"))
    widest_kept = stats.get("widest_kept", float("nan"))
    parts = []
    if n_rejected:
        parts.append(
            f"{100.0 * n_rejected / n:.1f}% of samples NaN-ed for falling in a "
            f"gap wider than max_gap={max_gap:g} s (widest {widest:.4g} s)"
        )
    if n_notable:
        parts.append(
            f"{100.0 * n_notable / n:.1f}% of samples interpolated across gaps "
            f"wider than {_GAP_WARN_FACTOR:g}x the median interval "
            f"({med:.3g} s), widest kept {widest_kept:.4g} s"
        )
    if n_out:
        verb = "NaN-ed" if not extrapolate else "edge-held"
        parts.append(f"{100.0 * n_out / n:.1f}% of samples {verb} outside coverage")

    hint = ""
    if n_notable and not stats.get("gated", False):
        hint = " — set hotel.max_gap to reject these instead"
    elif n_notable:
        hint = f" — max_gap={max_gap:g} s is above that threshold, so they are kept"
    warnings.warn(f"hotel channel {src!r}: " + "; ".join(parts) + hint, stacklevel=3)


def _native_interval(hotel_t: np.ndarray) -> float:
    """Median sample interval [s] of a hotel channel's *own* time vector.

    Recorded on the PFile so a consumer (``fp07_calibrate``) can match the
    reference's real bandwidth instead of guessing it from the interpolated
    array --- which is impossible for a pchip/nearest merge.
    """
    t = np.asarray(hotel_t, dtype=np.float64)
    t = t[np.isfinite(t)]
    if t.size < 2:
        return float("nan")
    dt = np.diff(np.sort(t))
    dt = dt[dt > 0]
    return float(np.median(dt)) if dt.size else float("nan")


def merge_hotel_into_pfile(hotel_data: HotelData, pf, hotel_cfg: dict) -> None:
    """Interpolate hotel channels and register them on ``pf`` in-place.

    Applies the full ``hotel.channels`` schema: per-variable rename, interp
    method, ``scale`` / ``offset`` linear transform, ``units`` override, and
    ``fast`` rate override.

    Adds each resulting channel to ``pf.channels`` under the (possibly
    renamed) output name, registers a ``pf.channel_info`` entry so
    :func:`extract_profiles` can read units, and updates
    ``pf._fast_channels`` so :meth:`PFile.is_fast` returns the correct dim.
    A hotel channel whose output name collides with one the instrument
    already carries raises ``ValueError`` unless it sets ``replace: true`` —
    this protects instrument channels that downstream processing depends on
    (notably ``P``, whose fast-rate hotel version silently breaks profile
    detection; issue #104 U5-1).
    """
    fast_channels = set(hotel_cfg.get("fast_channels", ["speed", "P"]))
    _, channels_opts = _normalize_channels_cfg(hotel_cfg.get("channels"))
    interpolated = interpolate_hotel(hotel_data, pf, hotel_cfg)
    gap_settings = resolve_gap_settings(hotel_cfg)  # validated above already

    # Snapshot the instrument's own channels so a hotel variable cannot silently
    # clobber one profile detection depends on (a hotel "P" would overwrite the
    # native slow pressure with a fast-rate array, understating dP/dt ~8x and
    # silently dropping the file). Require explicit opt-in to replace. (U5-1.)
    native = set(pf.channels)

    for src, data in interpolated.items():
        opts = channels_opts.get(src, {})
        out_name = opts.get("name", src)
        if out_name in native and not bool(opts.get("replace", False)):
            # If replacing would still land the hotel channel on a different grid
            # than the native one it overwrites (e.g. native slow P but hotel P
            # defaults to the fast set), replace: true alone is a trap — the
            # regridded channel is later dropped downstream. Name the second
            # remedy (fast:) so the escape hatch actually works. (U5-1.)
            native_is_fast = out_name in getattr(pf, "_fast_channels", set())
            would_be_fast = bool(opts["fast"]) if "fast" in opts else out_name in fast_channels
            msg = (
                f"hotel channel {src!r} would overwrite the instrument's own "
                f"{out_name!r} channel; rename it (hotel.channels.{src}.name: ...) "
                f"or set hotel.channels.{src}.replace: true to override."
            )
            if would_be_fast != native_is_fast:
                grid = "fast" if native_is_fast else "slow"
                msg += (
                    f" Note: the native {out_name!r} is on the {grid} grid, so also "
                    f"set hotel.channels.{src}.fast: {str(native_is_fast).lower()} "
                    f"so the replacement matches it (otherwise it is dropped)."
                )
            raise ValueError(msg)
        scale = float(opts.get("scale", 1.0))
        offset = float(opts.get("offset", 0.0))
        if scale != 1.0 or offset != 0.0:
            data = data * scale + offset
        pf.channels[out_name] = data
        if not hasattr(pf, "hotel_native_dt"):
            pf.hotel_native_dt = {}
        pf.hotel_native_dt[out_name] = _native_interval(hotel_data.time_for(src))

        units = opts.get("units")
        if units is None:
            units = hotel_data.units.get(src, "")
        info = {
            "units": units,
            "type": "hotel",
            "name": out_name,
        }
        # Carry the gate downstream: the per-profile NetCDF writer copies this
        # to a ``hotel_max_gap`` attr, so consumers that interp-fill NaN
        # samples (stratification / viscosity salinity) can refuse to rule
        # across the very hole the merge just NaN-ed.  None = unlimited.
        max_gap, _ = gap_settings.get(src, gap_settings[None])
        if max_gap is not None:
            info["hotel_max_gap"] = float(max_gap)
        pf.channel_info[out_name] = info

        is_fast = bool(opts["fast"]) if "fast" in opts else out_name in fast_channels
        if is_fast:
            pf._fast_channels.add(out_name)
        else:
            pf._fast_channels.discard(out_name)
