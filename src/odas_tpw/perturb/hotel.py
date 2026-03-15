# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Hotel file support — external telemetry from gliders/AUVs/Remus.

Hotel files provide vehicle-mounted sensor data (speed, pitch, roll, heading,
CTD) that gets interpolated onto instrument time axes as new channels.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class HotelData:
    """Container for loaded hotel file data.

    Attributes
    ----------
    time : np.ndarray
        Time vector in epoch seconds (or relative seconds).
    channels : dict[str, np.ndarray]
        Channel name → data array.
    time_is_relative : bool
        True if time values are relative seconds (not epoch).
    """

    time: np.ndarray
    channels: dict[str, np.ndarray] = field(default_factory=dict)
    time_is_relative: bool = False


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
        Mapping of hotel column names to output names.  If empty or None,
        all non-time columns are loaded with their original names.

    Returns
    -------
    HotelData
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".csv":
        return _load_csv(path, time_column, time_format, channels)
    elif ext in (".nc", ".nc4"):
        return _load_netcdf(path, time_column, time_format, channels)
    elif ext == ".mat":
        return _load_mat(path, time_column, time_format, channels)
    else:
        raise ValueError(
            f"Unsupported hotel file format: {ext!r}. "
            f"Supported: .csv, .nc, .mat"
        )


def _load_csv(
    path: Path,
    time_column: str,
    time_format: str,
    channels: dict | None,
) -> HotelData:
    import pandas as pd

    df = pd.read_csv(path)
    raw_time = np.asarray(df[time_column].values)
    time, is_relative = _parse_time(raw_time, time_format)

    data_cols = [c for c in df.columns if c != time_column]
    if channels:
        ch = {
            channels.get(c, c): df[c].values.astype(np.float64)
            for c in data_cols if c in channels
        }
    else:
        ch = {c: df[c].values.astype(np.float64) for c in data_cols}

    return HotelData(time=time, channels=ch, time_is_relative=is_relative)


def _load_netcdf(
    path: Path,
    time_column: str,
    time_format: str,
    channels: dict | None,
) -> HotelData:
    import netCDF4 as nc

    ds = nc.Dataset(str(path), "r")
    raw_time = ds.variables[time_column][:].data
    time, is_relative = _parse_time(raw_time, time_format)

    data_vars = [v for v in ds.variables if v != time_column]
    if channels:
        ch = {
            channels.get(v, v): ds.variables[v][:].data.astype(np.float64)
            for v in data_vars if v in channels
        }
    else:
        ch = {v: ds.variables[v][:].data.astype(np.float64) for v in data_vars}
    ds.close()

    return HotelData(time=time, channels=ch, time_is_relative=is_relative)


def _load_mat(
    path: Path,
    time_column: str,
    time_format: str,
    channels: dict | None,
) -> HotelData:
    from scipy.io import loadmat

    mat = loadmat(str(path), squeeze_me=True)

    # ODAS convention: struct fields with .data/.time subfields
    # Also handle flat arrays
    ch = {}
    raw_time = None

    for key, val in mat.items():
        if key.startswith("_"):
            continue
        # Check for structured array with .time and .data subfields
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
                name = channels.get(key, key) if channels else key
                if not channels or key in channels:
                    ch[name] = np.asarray(struct["data"]).flatten().astype(np.float64)
                    if raw_time is None:
                        raw_time = np.asarray(struct["time"]).flatten().astype(np.float64)
                continue
        # Flat array
        arr = np.asarray(val).flatten()
        if key == time_column:
            raw_time = arr.astype(np.float64)
        else:
            name = channels.get(key, key) if channels else key
            if not channels or key in channels:
                ch[name] = arr.astype(np.float64)

    if raw_time is None:
        raise ValueError(f"Time column {time_column!r} not found in .mat file")

    time, is_relative = _parse_time(raw_time, time_format)
    return HotelData(time=time, channels=ch, time_is_relative=is_relative)


def interpolate_hotel(hotel_data: HotelData, pf, hotel_cfg: dict) -> dict[str, np.ndarray]:
    """Interpolate hotel channels onto PFile time axes.

    Parameters
    ----------
    hotel_data : HotelData
        Loaded hotel data.
    pf : PFile
        Loaded PFile with ``t_fast``, ``t_slow``, ``start_time``.
    hotel_cfg : dict
        Hotel config section (``fast_channels``, ``interpolation``).

    Returns
    -------
    dict mapping channel names to interpolated arrays.
    """
    from scipy.interpolate import PchipInterpolator, interp1d

    fast_channels = set(hotel_cfg.get("fast_channels", ["speed", "P"]))
    method = hotel_cfg.get("interpolation", "pchip")

    # Convert hotel time to relative seconds matching pf.t_fast / pf.t_slow
    if hotel_data.time_is_relative:
        hotel_t = hotel_data.time
    else:
        # Hotel time is epoch — convert to relative seconds from pf.start_time
        hotel_t = hotel_data.time - pf.start_time.timestamp()

    result = {}
    for name, data in hotel_data.channels.items():
        target_t = pf.t_fast if name in fast_channels else pf.t_slow

        if method == "pchip":
            interp = PchipInterpolator(hotel_t, data, extrapolate=False)
            interpolated = interp(target_t)
            # Fill NaN edges with boundary values
            mask = np.isnan(interpolated)
            if np.any(mask):
                first_valid = data[0]
                last_valid = data[-1]
                below = target_t < hotel_t[0]
                above = target_t > hotel_t[-1]
                interpolated[mask & below] = first_valid
                interpolated[mask & above] = last_valid
        else:
            interp = interp1d(
                hotel_t, data, kind="linear",
                bounds_error=False, fill_value=(data[0], data[-1]),
            )
            interpolated = interp(target_t)

        result[name] = interpolated

    return result
