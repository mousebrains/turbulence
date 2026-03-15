# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""GPS providers for position interpolation.

Reference: Code/GPS_base_class.m, GPS_NaN.m, GPS_fixed.m
"""

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


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
    """Interpolate GPS from a CSV file."""

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
        t = df[time_col].values.astype(np.float64)
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
        t = ds.variables[time_var][:].data.astype(np.float64)
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
