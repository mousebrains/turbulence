# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Profile detection and extraction for vertical profilers.

Port of get_profile.m from the ODAS MATLAB library, plus NetCDF I/O
for per-profile files.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, filtfilt

if TYPE_CHECKING:
    from rsi_python.p_file import PFile

# Default vehicle tau values matching ODAS default_vehicle_attributes.ini
_VEHICLE_TAU = {
    "vmp": 1.5,
    "rvmp": 1.5,
    "xmp": 1.5,
    "micro_squid": 1.5,
    "stand": 1.5,
    "sea_glider": 5.0,
    "slocum_glider": 3.0,
    "sea_explorer": 3.0,
    "auv": 10.0,
    "auv_emc": 10.0,
    "nemo": 60.0,
    "argo_float": 60.0,
}


def _smooth_fall_rate(P: np.ndarray, fs: float, tau: float = 1.5) -> np.ndarray:
    """Compute smoothed fall rate from pressure.

    Matches ODAS odas_p2mat.m lines 699-701: central-difference gradient
    followed by a zero-phase first-order Butterworth low-pass filter at
    cutoff frequency ``0.68 / tau``.

    Parameters
    ----------
    P : ndarray
        Pressure [dbar].
    fs : float
        Sampling rate [Hz].
    tau : float
        Smoothing time constant [s]. Default: 1.5 (VMP).

    Returns
    -------
    W : ndarray
        Smoothed fall rate [dbar/s].
    """
    W = np.gradient(P.astype(np.float64), 1.0 / fs)
    f_c = 0.68 / tau
    b, a = butter(1, f_c / (fs / 2.0))
    return filtfilt(b, a, W)


def get_profiles(
    P: npt.ArrayLike,
    W: npt.ArrayLike,
    fs: float,
    P_min: float = 0.5,
    W_min: float = 0.3,
    direction: str = "down",
    min_duration: float = 7.0,
) -> list[tuple[int, int]]:
    """Find profiling segments in pressure data.

    Parameters
    ----------
    P : array_like
        Pressure vector [dbar].
    W : array_like
        Rate of change of pressure [dbar/s] (positive = downward).
    fs : float
        Sampling rate of P and W [Hz].
    P_min : float
        Minimum pressure for a valid profile [dbar].
    W_min : float
        Minimum fall/rise rate magnitude [dbar/s].
    direction : str
        'down' or 'up'.
    min_duration : float
        Minimum profile duration [s].

    Returns
    -------
    list of (int, int)
        Start and end indices (inclusive) of each detected profile.
    """
    P = np.asarray(P, dtype=np.float64).ravel()
    W = np.asarray(W, dtype=np.float64).ravel()
    min_samples = int(min_duration * fs)

    if direction.lower() == "up":
        W = -W

    # Find valid samples
    mask = (P_min < P) & (W_min <= W)
    n = np.where(mask)[0]

    if len(n) < min_samples:
        return []

    # Find breaks between contiguous segments
    dn = np.diff(n)
    breaks = np.where(dn > 1)[0]

    if len(breaks) == 0:
        profiles = [(int(n[0]), int(n[-1]))]
    else:
        profiles = []
        profiles.append((int(n[0]), int(n[breaks[0]])))
        for i in range(1, len(breaks)):
            profiles.append((int(n[breaks[i - 1] + 1]), int(n[breaks[i]])))
        profiles.append((int(n[breaks[-1] + 1]), int(n[-1])))

    # Filter by minimum duration
    profiles = [(s, e) for s, e in profiles if (e - s) >= min_samples]
    return profiles


def extract_profiles(
    source: "PFile | str | Path",
    output_dir: str | Path,
    **profile_kwargs: Any,
) -> list[Path]:
    """Extract profiles from a PFile or full-record NetCDF.

    Parameters
    ----------
    source : PFile, str, or Path
        A PFile object, path to .p file, or path to full-record .nc file.
    output_dir : Path
        Directory for per-profile NetCDF files.
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

    # Compute smoothed fall rate from slow pressure
    W = _smooth_fall_rate(P_slow, fs_slow)

    profiles = get_profiles(P_slow, W, fs_slow, **profile_kwargs)
    if not profiles:
        print(f"  No profiles found in {data['stem']}")
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

        ds.Conventions = "CF-1.13"

        # Profile metadata
        ds.profile_number = pi
        ds.profile_start_index_slow = int(s_slow)
        ds.profile_end_index_slow = int(e_slow)
        ds.profile_P_start = float(P_slow[s_slow])
        ds.profile_P_end = float(P_slow[e_slow])
        ds.profile_duration_s = float((e_slow - s_slow) / fs_slow)
        ds.profile_mean_speed = float(np.mean(np.abs(W[s_slow:s_slow_end])))
        ds.history = f"Profile {pi} extracted on {datetime.now(UTC).isoformat()}"

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
        t_fast_var.axis = "T"

        t_slow_var = ds.createVariable("t_slow", "f8", ("time_slow",), zlib=True)
        t_slow_var[:] = data["t_slow"][s_slow:s_slow_end]
        t_slow_var.units = data.get("t_slow_units", "seconds")
        t_slow_var.long_name = "time (slow channels)"
        t_slow_var.standard_name = "time"
        t_slow_var.calendar = "standard"
        t_slow_var.axis = "T"

        # Channel data
        for ch_name, ch_data, dim, attrs in data["channels"]:
            trimmed = ch_data[s_fast:e_fast] if dim == "time_fast" else ch_data[s_slow:s_slow_end]
            var_name = ch_name.replace(" ", "_")
            var = ds.createVariable(var_name, "f4", (dim,), zlib=True)
            var[:] = trimmed.astype(np.float32)
            for k, v in attrs.items():
                setattr(var, k, v)

        ds.fs_fast = float(fs_fast)
        ds.fs_slow = float(fs_slow)
        ds.close()
        output_paths.append(prof_path)
        print(
            f"  Profile {pi}: P={P_slow[s_slow]:.1f}–{P_slow[e_slow]:.1f} dbar, "  # noqa: RUF001
            f"{(e_slow - s_slow) / fs_slow:.1f} s -> {prof_path.name}"
        )

    return output_paths


def _load_source(source: "PFile | str | Path") -> dict[str, Any]:
    """Load data from PFile, .p path, or .nc path into a common dict."""
    from rsi_python.p_file import PFile

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
        "Conventions": "CF-1.13",
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

    fs_fast = float(ds.fs_fast)
    fs_slow = float(ds.fs_slow)

    t_fast = ds.variables["t_fast"][:].data
    t_slow = ds.variables["t_slow"][:].data
    t_fast_units = (
        ds.variables["t_fast"].units if hasattr(ds.variables["t_fast"], "units") else "seconds"
    )
    t_slow_units = (
        ds.variables["t_slow"].units if hasattr(ds.variables["t_slow"], "units") else "seconds"
    )

    P = ds.variables["P"][:].data

    channels = []
    for vname in ds.variables:
        if vname in ("t_fast", "t_slow"):
            continue
        var = ds.variables[vname]
        dims = var.dimensions
        if len(dims) != 1:
            continue
        dim = dims[0]
        if dim not in ("time_fast", "time_slow"):
            continue
        attrs = {}
        for a in var.ncattrs():
            attrs[a] = getattr(var, a)
        channels.append((vname, var[:].data.astype(np.float64), dim, attrs))

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
