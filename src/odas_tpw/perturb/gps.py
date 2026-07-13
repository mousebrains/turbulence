# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""GPS providers for position interpolation.

All providers expect query times in **epoch seconds** (POSIX time, the
``datetime.timestamp()`` value).  Callers holding file-relative seconds
(e.g. ``pf.t_slow``) must add ``pf.start_time.timestamp()`` before
querying.  The interpolator stored in :class:`GPSFromNetCDF` /
:class:`GPSFromCSV` normalizes datetime64 columns to epoch seconds at
construction time.

Reference: Code/GPS_base_class.m, GPS_NaN.m, GPS_fixed.m
"""

import warnings
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


def _to_epoch_seconds(values: np.ndarray) -> np.ndarray:
    """Normalize a time column to epoch seconds (float64).

    datetime64[*] columns are converted via integer ns since epoch.  Numeric
    columns are passed through unchanged — callers are expected to hand in
    epoch seconds when no datetime metadata is available.

    NaT decodes to NaN (not the int64-min sentinel, which would otherwise become
    a bogus ~-9.2e9 s epoch and poison the interpolation node).
    """
    if np.issubdtype(values.dtype, np.datetime64):
        ns = values.astype("datetime64[ns]").astype(np.int64)
        secs = ns.astype(np.float64) / 1e9
        secs[ns == np.iinfo(np.int64).min] = np.nan
        return secs
    return values.astype(np.float64)


def _nc_values(var) -> np.ndarray:
    """Read a netCDF4 variable, mapping any masked/_FillValue cells to NaN.

    netCDF4 returns masked arrays by default (auto-masking is on in this
    module).  Reaching for ``.data`` would expose the raw _FillValue (e.g.
    -999, 1e20) as if it were a real measurement — a missing GPS fix would
    become a -999 latitude and drive ``interp1d`` to wildly wrong positions.
    ``.filled(np.nan)`` turns those gaps into NaN instead.
    """
    arr = var[:]
    if np.ma.isMaskedArray(arr):
        # Cast to float before filling: filling an integer-dtype masked array
        # with NaN raises (NaN cannot be cast to int).
        return np.asarray(np.ma.filled(arr.astype(np.float64), np.nan))
    return np.asarray(arr)


def _finite_interp1d(t: np.ndarray, vals: np.ndarray):
    """Build a linear interpolator over the FINITE (t, vals) pairs.

    ``interp1d`` propagates NaN to every interval touching a NaN node, so a
    single fill-valued lat/lon (or time) would blank whole position spans.
    Dropping the non-finite nodes interpolates across the gap instead.
    """
    from scipy.interpolate import interp1d

    t = np.asarray(t, dtype=np.float64)
    vals = np.asarray(vals, dtype=np.float64)
    good = np.isfinite(t) & np.isfinite(vals)
    if int(good.sum()) >= 2:
        return interp1d(
            t[good], vals[good], bounds_error=False, fill_value="extrapolate"
        )
    # < 2 finite nodes: constant (or all-NaN) -> a constant function.
    fill = float(vals[good][0]) if good.any() else np.nan
    return interp1d([0.0, 1.0], [fill, fill], bounds_error=False, fill_value=fill)


class _DatelineSafeLon:
    """Callable that evaluates an unwrapped-phase longitude interpolator and
    wraps only out-of-[-180, 180) results back into range.

    A module-level class rather than a closure so a GPS provider holding one
    stays picklable: the perturb pipeline ships providers to a
    ``ProcessPoolExecutor``, and a local-function closure would raise
    ``Can't pickle local object`` there. ``base`` is a scipy ``interp1d``,
    which pickles cleanly. (#104 U5-4 follow-up.)
    """

    def __init__(self, base) -> None:
        self._base = base

    def __call__(self, query: npt.ArrayLike) -> np.ndarray:
        out = np.asarray(self._base(query), dtype=np.float64)
        wrap = np.isfinite(out) & ((out < -180.0) | (out >= 180.0))
        return np.where(wrap, (out + 180.0) % 360.0 - 180.0, out)


def _lon_interp_dateline_safe(t: np.ndarray, lon: np.ndarray):
    """Linear longitude interpolator that is safe across the +/-180 seam.

    A plain linear interp of raw longitude sweeps ~360 deg the *wrong* way when
    a track crosses the antimeridian (e.g. 179.9 -> -179.9 would pass through 0,
    landing on the opposite side of the globe). Unwrap the finite nodes to a
    continuous phase (``period=360``), interpolate that, then wrap only the
    results that land outside [-180, 180) back into range. A track that never
    crosses the seam unwraps to itself, and any interpolated value already in
    [-180, 180) passes through untouched — so the result is bit-identical to the
    old raw interpolation everywhere the position is in range (all of ARCTERX,
    whose positions and time-bounded extrapolations stay near 145 E). Two kinds
    of value differ from the old raw interp, both corrections rather than
    regressions: (1) points on a seam-crossing segment, where the old code swept
    the wrong way through 0 to the far side of the globe (still in range, but
    wrong) while the new code takes the short path across +/-180; and (2) an
    extreme extrapolation the old code pushed past +/-180, which the new code
    wraps to the equivalent in-range meridian instead of a meaningless >180 deg
    longitude.

    Convention: output is normalized to [-180, 180). Inputs in the usual
    [-180, 180) convention are unchanged (subject to the wrap above); a
    [0, 360)-convention track is re-expressed in [-180, 180) — the same physical
    positions, a different label. (#104 U5-4.)
    """
    from scipy.interpolate import interp1d

    t = np.asarray(t, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    good = np.isfinite(t) & np.isfinite(lon)
    if int(good.sum()) >= 2:
        # unwrap needs the nodes in time order; interp1d otherwise sorts itself.
        order = np.argsort(t[good])
        t_g = t[good][order]
        lon_u = np.unwrap(lon[good][order], period=360.0)
        base = interp1d(t_g, lon_u, bounds_error=False, fill_value="extrapolate")
    else:
        fill = float(lon[good][0]) if good.any() else np.nan
        base = interp1d([0.0, 1.0], [fill, fill], bounds_error=False, fill_value=fill)

    return _DatelineSafeLon(base)


# CF time unit -> seconds factor.  Only common units; falls back to
# ``num2date`` for the long tail (months, years, weird calendars, etc.)
_CF_TIME_UNIT_FACTORS: dict[str, float] = {
    "s": 1.0,
    "sec": 1.0,
    "secs": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "ms": 1e-3,
    "msec": 1e-3,
    "msecs": 1e-3,
    "millisecond": 1e-3,
    "milliseconds": 1e-3,
    "us": 1e-6,
    "microsecond": 1e-6,
    "microseconds": 1e-6,
    "ns": 1e-9,
    "nanosecond": 1e-9,
    "nanoseconds": 1e-9,
    "min": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hrs": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
    "d": 86400.0,
    "day": 86400.0,
    "days": 86400.0,
}


def _decode_cf_times_fast(
    raw: np.ndarray, units: str | None, calendar: str | None
) -> np.ndarray | None:
    """Decode CF "<unit> since <iso-datetime>" times to epoch seconds.

    Returns the decoded float64 array, or ``None`` if the units string
    isn't a form we can decode without falling back to ``num2date``.

    Used as a fast path inside :class:`GPSFromNetCDF` for plain numeric
    time variables; on a million-row GPS file this replaces a
    per-element ``datetime.timestamp()`` loop with a single vector
    multiply-add.
    """
    if units is None:
        return None
    units_str = str(units).strip().lower()
    if " since " not in units_str:
        return None
    unit_part, _, ref_part = units_str.partition(" since ")
    factor = _CF_TIME_UNIT_FACTORS.get(unit_part.strip())
    if factor is None:
        return None
    if calendar and str(calendar).lower() not in (
        "standard",
        "gregorian",
        "proleptic_gregorian",
    ):
        return None
    if not np.issubdtype(raw.dtype, np.number):
        return None

    from datetime import UTC, datetime

    ref_str = ref_part.strip().rstrip("z").rstrip("Z")
    # Tolerate a few common CF reference-time spellings: pure ISO, with
    # space separator, with or without timezone.
    try:
        ref_dt = datetime.fromisoformat(ref_str)
    except ValueError:
        try:
            ref_dt = datetime.fromisoformat(ref_str.replace(" ", "T"))
        except ValueError:
            return None
    if ref_dt.tzinfo is None:
        ref_dt = ref_dt.replace(tzinfo=UTC)
    epoch_s = ref_dt.timestamp()

    return raw.astype(np.float64) * factor + epoch_s


@runtime_checkable
class GPSProvider(Protocol):
    """Protocol for GPS position providers."""

    def lat(self, t: npt.ArrayLike) -> np.ndarray: ...
    def lon(self, t: npt.ArrayLike) -> np.ndarray: ...


class GPSNaN:
    """Returns NaN for all GPS queries."""

    def lat(self, t: npt.ArrayLike) -> np.ndarray:
        """Return NaN latitude for all times."""
        return np.full_like(np.asarray(t, dtype=np.float64), np.nan)

    def lon(self, t: npt.ArrayLike) -> np.ndarray:
        """Return NaN longitude for all times."""
        return np.full_like(np.asarray(t, dtype=np.float64), np.nan)


class GPSFixed:
    """Returns constant lat/lon."""

    def __init__(self, lat: float, lon: float) -> None:
        self._lat = float(lat)
        self._lon = float(lon)

    def lat(self, t: npt.ArrayLike) -> np.ndarray:
        """Return fixed latitude for all times."""
        return np.full_like(np.asarray(t, dtype=np.float64), self._lat)

    def lon(self, t: npt.ArrayLike) -> np.ndarray:
        """Return fixed longitude for all times."""
        return np.full_like(np.asarray(t, dtype=np.float64), self._lon)


class GPSFromCSV:
    """Interpolate GPS from a CSV file (time column expected in epoch seconds).

    Positions requested more than ``max_time_diff`` seconds outside the
    GPS record's time coverage are linearly extrapolated AND trigger a
    warning — extrapolated positions feed the TEOS-10 absolute-salinity
    conversion and the file geospatial attributes.
    """

    def __init__(
        self,
        file: str | Path,
        time_col: str = "t",
        lat_col: str = "lat",
        lon_col: str = "lon",
        max_time_diff: float = 60.0,
    ) -> None:
        import pandas as pd

        df = pd.read_csv(file)
        t = _to_epoch_seconds(np.asarray(df[time_col].values))
        lat = np.asarray(df[lat_col].values, dtype=np.float64)
        lon = np.asarray(df[lon_col].values, dtype=np.float64)
        # Coverage window = the finite (time, lat, lon) span the interpolators
        # honor without extrapolating (see GPSFromNetCDF for the rationale).
        _cov = np.isfinite(t) & np.isfinite(lat) & np.isfinite(lon)
        if _cov.any():
            self._t_min = float(np.min(t[_cov]))
            self._t_max = float(np.max(t[_cov]))
        else:
            self._t_min = float(np.nanmin(t))
            self._t_max = float(np.nanmax(t))
        self._max_time_diff = float(max_time_diff)
        self._lat_interp = _finite_interp1d(t, lat)
        self._lon_interp = _lon_interp_dateline_safe(t, lon)

    def lat(self, t: npt.ArrayLike) -> np.ndarray:
        """Interpolate latitude from CSV at the given times."""
        t_arr = np.asarray(t, dtype=np.float64)
        _warn_outside_coverage(t_arr, self._t_min, self._t_max, self._max_time_diff)
        return np.asarray(self._lat_interp(t_arr))

    def lon(self, t: npt.ArrayLike) -> np.ndarray:
        """Interpolate longitude from CSV at the given times."""
        t_arr = np.asarray(t, dtype=np.float64)
        _warn_outside_coverage(t_arr, self._t_min, self._t_max, self._max_time_diff)
        return np.asarray(self._lon_interp(t_arr))


def _warn_outside_coverage(
    t: np.ndarray, t_min: float, t_max: float, max_time_diff: float
) -> None:
    """Warn when requested times fall outside the GPS record coverage.

    Beyond the coverage, positions are linear extrapolations without
    bound; ``max_time_diff`` (config key ``gps.max_time_diff``) sets the
    tolerance before warning.
    """
    if t.size == 0 or not np.any(np.isfinite(t)):
        return
    lo = np.nanmin(t) - (t_min - max_time_diff)
    hi = np.nanmax(t) - (t_max + max_time_diff)
    if lo < 0 or hi > 0:
        worst = max(-lo if lo < 0 else 0.0, hi if hi > 0 else 0.0)
        warnings.warn(
            f"GPS position requested {worst:.0f} s outside the GPS record "
            f"coverage (tolerance {max_time_diff:.0f} s); positions are "
            "linearly extrapolated and may be far from the instrument",
            stacklevel=3,
        )


class GPSFromNetCDF:
    """Interpolate GPS from a NetCDF file.

    Positions requested more than ``max_time_diff`` seconds outside the
    GPS record's time coverage are linearly extrapolated AND trigger a
    warning (see :class:`GPSFromCSV`).
    """

    def __init__(
        self,
        file: str | Path,
        time_var: str = "time",
        lat_var: str = "lat",
        lon_var: str = "lon",
        max_time_diff: float = 60.0,
    ) -> None:
        import netCDF4 as nc

        # Read raw arrays inside the context manager so the Dataset is closed
        # even if a read raises; decode times afterwards (raw/lat/lon are
        # in-memory copies that outlive the file handle).
        with nc.Dataset(str(file), "r") as ds:
            t_var = ds.variables[time_var]
            units = getattr(t_var, "units", None)
            calendar = getattr(t_var, "calendar", "standard")
            raw = _nc_values(t_var)
            lat = _nc_values(ds.variables[lat_var]).astype(np.float64)
            lon = _nc_values(ds.variables[lon_var]).astype(np.float64)

        # Fast path: when the variable is plain numeric and its CF units
        # are "<unit> since <iso-datetime>", we can compute epoch seconds
        # by a single scalar+vector arithmetic step instead of going
        # through ``num2date`` + a per-element ``datetime.timestamp()``
        # list comprehension (which dominated startup on multi-million-
        # row GPS files).
        t = _decode_cf_times_fast(raw, units, calendar)
        if t is None:
            if units and "since" in str(units):
                from datetime import UTC

                decoded = nc.num2date(
                    raw,
                    units=str(units),
                    calendar=str(calendar),
                    only_use_cftime_datetimes=False,
                )
                dts_list = (
                    [decoded] if not hasattr(decoded, "__iter__") else list(decoded)  # type: ignore[redundant-expr]
                )
                t = np.array(
                    [
                        (d.replace(tzinfo=UTC) if d.tzinfo is None else d).timestamp()
                        for d in dts_list
                    ],
                    dtype=np.float64,
                )
            else:
                t = _to_epoch_seconds(raw)

        # Coverage window = the span the interpolators can actually honor
        # without extrapolating: the finite (time, lat, lon) nodes. Using the
        # full decoded time range would hide fabricated extrapolations when
        # lat/lon drop out at the record's temporal extremes (a GPS fix dropout
        # at the start/end of the file), since _finite_interp1d silently
        # extrapolates past the last finite fix.
        _cov = np.isfinite(t) & np.isfinite(lat) & np.isfinite(lon)
        if _cov.any():
            self._t_min = float(np.min(t[_cov]))
            self._t_max = float(np.max(t[_cov]))
        else:
            self._t_min = float(np.nanmin(t))
            self._t_max = float(np.nanmax(t))
        self._max_time_diff = float(max_time_diff)
        self._lat_interp = _finite_interp1d(t, lat)
        self._lon_interp = _lon_interp_dateline_safe(t, lon)

    def lat(self, t: npt.ArrayLike) -> np.ndarray:
        """Interpolate latitude from NetCDF at the given times."""
        t_arr = np.asarray(t, dtype=np.float64)
        _warn_outside_coverage(t_arr, self._t_min, self._t_max, self._max_time_diff)
        return np.asarray(self._lat_interp(t_arr))

    def lon(self, t: npt.ArrayLike) -> np.ndarray:
        """Interpolate longitude from NetCDF at the given times."""
        t_arr = np.asarray(t, dtype=np.float64)
        _warn_outside_coverage(t_arr, self._t_min, self._t_max, self._max_time_diff)
        return np.asarray(self._lon_interp(t_arr))


def create_gps(config: dict) -> GPSProvider:
    """Factory: create a GPSProvider from the ``gps`` config section.

    Parameters
    ----------
    config : dict
        Merged GPS config (source, lat, lon, file, time_col, etc.).

    Returns
    -------
    GPSProvider
    """
    source = config.get("source", "nan").lower()

    if source == "nan":
        return GPSNaN()
    elif source == "fixed":
        return GPSFixed(config["lat"], config["lon"])
    elif source == "csv":
        # time_col default is source-dependent ("t" for CSV, "time" for NetCDF);
        # the shared gps.time_col default is null so each branch can pick its own.
        return GPSFromCSV(
            config["file"],
            time_col=config.get("time_col") or "t",
            lat_col=config.get("lat_col", "lat"),
            lon_col=config.get("lon_col", "lon"),
            max_time_diff=float(config.get("max_time_diff", 60)),
        )
    elif source == "netcdf":
        return GPSFromNetCDF(
            config["file"],
            time_var=config.get("time_col") or "time",
            lat_var=config.get("lat_col", "lat"),
            lon_var=config.get("lon_col", "lon"),
            max_time_diff=float(config.get("max_time_diff", 60)),
        )
    else:
        raise ValueError(f"Unknown GPS source: {source!r}")
