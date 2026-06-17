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
    """
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[ns]").astype(np.int64) / 1e9
    return values.astype(np.float64)


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
        from scipy.interpolate import interp1d

        df = pd.read_csv(file)
        t = _to_epoch_seconds(np.asarray(df[time_col].values))
        self._t_min = float(np.nanmin(t))
        self._t_max = float(np.nanmax(t))
        self._max_time_diff = float(max_time_diff)
        self._lat_interp = interp1d(
            t,
            df[lat_col].values,
            bounds_error=False,
            fill_value="extrapolate",
        )
        self._lon_interp = interp1d(
            t,
            df[lon_col].values,
            bounds_error=False,
            fill_value="extrapolate",
        )

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
        from scipy.interpolate import interp1d

        ds = nc.Dataset(str(file), "r")
        t_var = ds.variables[time_var]
        units = getattr(t_var, "units", None)
        calendar = getattr(t_var, "calendar", "standard")
        raw = t_var[:].data
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
        lat = ds.variables[lat_var][:].data.astype(np.float64)
        lon = ds.variables[lon_var][:].data.astype(np.float64)
        ds.close()

        self._t_min = float(np.nanmin(t))
        self._t_max = float(np.nanmax(t))
        self._max_time_diff = float(max_time_diff)
        self._lat_interp = interp1d(t, lat, bounds_error=False, fill_value="extrapolate")
        self._lon_interp = interp1d(t, lon, bounds_error=False, fill_value="extrapolate")

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
