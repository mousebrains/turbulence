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
    """Interpolate GPS from a CSV file (time column expected in epoch seconds)."""

    def __init__(
        self,
        file: str | Path,
        time_col: str = "t",
        lat_col: str = "lat",
        lon_col: str = "lon",
    ) -> None:
        import pandas as pd
        from scipy.interpolate import interp1d

        df = pd.read_csv(file)
        t = _to_epoch_seconds(np.asarray(df[time_col].values))
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
        return np.asarray(self._lat_interp(np.asarray(t, dtype=np.float64)))

    def lon(self, t: npt.ArrayLike) -> np.ndarray:
        """Interpolate longitude from CSV at the given times."""
        return np.asarray(self._lon_interp(np.asarray(t, dtype=np.float64)))


class GPSFromNetCDF:
    """Interpolate GPS from a NetCDF file."""

    def __init__(
        self,
        file: str | Path,
        time_var: str = "time",
        lat_var: str = "lat",
        lon_var: str = "lon",
    ) -> None:
        import netCDF4 as nc
        from scipy.interpolate import interp1d

        ds = nc.Dataset(str(file), "r")
        t_var = ds.variables[time_var]
        # Decode CF time units to epoch seconds when the variable carries
        # ``units = "<unit> since <reference>"``.  netCDF4's ``num2date`` is
        # the canonical decoder; otherwise we treat the column as raw values
        # (epoch seconds, or datetime64 written as int64 ns).
        units = getattr(t_var, "units", None)
        calendar = getattr(t_var, "calendar", "standard")
        raw = t_var[:].data
        if units and "since" in str(units):
            from datetime import UTC

            decoded = nc.num2date(
                raw,
                units=str(units),
                calendar=str(calendar),
                only_use_cftime_datetimes=False,
            )
            # ``num2date`` returns naive datetimes (or a single one for a
            # 0-D input); treat the reference epoch as UTC so
            # ``timestamp()`` doesn't pick up the local-time offset.
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

        self._lat_interp = interp1d(t, lat, bounds_error=False, fill_value="extrapolate")
        self._lon_interp = interp1d(t, lon, bounds_error=False, fill_value="extrapolate")

    def lat(self, t: npt.ArrayLike) -> np.ndarray:
        """Interpolate latitude from NetCDF at the given times."""
        return np.asarray(self._lat_interp(np.asarray(t, dtype=np.float64)))

    def lon(self, t: npt.ArrayLike) -> np.ndarray:
        """Interpolate longitude from NetCDF at the given times."""
        return np.asarray(self._lon_interp(np.asarray(t, dtype=np.float64)))


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
        return GPSFromCSV(
            config["file"],
            time_col=config.get("time_col", "t"),
            lat_col=config.get("lat_col", "lat"),
            lon_col=config.get("lon_col", "lon"),
        )
    elif source == "netcdf":
        return GPSFromNetCDF(
            config["file"],
            time_var=config.get("time_col", "time"),
            lat_var=config.get("lat_col", "lat"),
            lon_var=config.get("lon_col", "lon"),
        )
    else:
        raise ValueError(f"Unknown GPS source: {source!r}")
