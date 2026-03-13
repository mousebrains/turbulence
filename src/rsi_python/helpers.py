# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Shared helpers for loading instrument data and preparing profiles.

These functions bridge PFile/NetCDF reading with spectral processing.
Originally part of dissipation.py, extracted so that both epsilon and
chi packages can use them without circular dependencies.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from rsi_python.p_file import PFile

from rsi_python.ocean import visc, visc35

# ---------------------------------------------------------------------------
# Channel loading
# ---------------------------------------------------------------------------


def load_channels(
    source: PFile | str | Path,
    shear_pattern: str = r"^sh\d+$",
    accel_pattern: str = r"^A[xyz]$",
    pressure_name: str = "P",
    temperature_name: str = "T1",
) -> dict[str, Any]:
    """Load channel data from any supported source.

    Parameters
    ----------
    source : PFile, str, or Path
        A PFile object, .p file path, full-record .nc path,
        or per-profile .nc path.
    shear_pattern : str
        Regex pattern matching shear channel names.
    accel_pattern : str
        Regex pattern matching accelerometer channel names.
    pressure_name : str
        Name of the pressure channel.
    temperature_name : str
        Name of the temperature channel.

    Returns
    -------
    dict with keys:
        shear : list of (name, ndarray) — shear probe signals
        accel : list of (name, ndarray) — accelerometer signals
        P : ndarray — pressure (slow)
        T : ndarray — temperature (slow)
        t_fast : ndarray — fast time vector
        t_slow : ndarray — slow time vector
        fs_fast : float
        fs_slow : float
        is_profile : bool — whether source is a per-profile file
        metadata : dict
    """
    from rsi_python.p_file import PFile

    if isinstance(source, PFile):
        return _channels_from_pfile(
            source, shear_pattern, accel_pattern, pressure_name, temperature_name
        )

    source = Path(source)
    if source.suffix.lower() == ".p":
        pf = PFile(source)
        return _channels_from_pfile(
            pf, shear_pattern, accel_pattern, pressure_name, temperature_name
        )
    elif source.suffix.lower() == ".nc":
        return _channels_from_nc(
            source, shear_pattern, accel_pattern, pressure_name, temperature_name
        )
    else:
        raise ValueError(f"Unsupported file type: {source.suffix}")


def _channels_from_pfile(
    pf: PFile, sh_pat: str, ac_pat: str, p_name: str, t_name: str
) -> dict[str, Any]:
    sh_re = re.compile(sh_pat)
    ac_re = re.compile(ac_pat)
    shear = sorted(
        [(n, pf.channels[n]) for n in pf._fast_channels if sh_re.match(n)],
        key=lambda x: x[0],
    )
    accel = sorted(
        [(n, pf.channels[n]) for n in pf._fast_channels if ac_re.match(n)],
        key=lambda x: x[0],
    )
    return {
        "shear": shear,
        "accel": accel,
        "P": pf.channels[p_name],
        "T": pf.channels[t_name],
        "t_fast": pf.t_fast,
        "t_slow": pf.t_slow,
        "fs_fast": pf.fs_fast,
        "fs_slow": pf.fs_slow,
        "is_profile": False,
        "vehicle": pf.config["instrument_info"].get("vehicle", "").lower(),
        "metadata": {
            "source": str(pf.filepath),
            "instrument": pf.config["instrument_info"].get("model", ""),
            "sn": pf.config["instrument_info"].get("sn", ""),
            "start_time": pf.start_time.isoformat(),
        },
    }


def _channels_from_nc(
    nc_path: Path, sh_pat: str, ac_pat: str, p_name: str, t_name: str
) -> dict[str, Any]:
    import netCDF4 as nc

    ds = nc.Dataset(str(nc_path), "r")
    sh_re = re.compile(sh_pat)
    ac_re = re.compile(ac_pat)

    fs_fast = float(ds.fs_fast)
    fs_slow = float(ds.fs_slow)
    is_profile = hasattr(ds, "profile_number")

    t_fast = ds.variables["t_fast"][:].data
    t_slow = ds.variables["t_slow"][:].data
    P = ds.variables[p_name][:].data.astype(np.float64)
    T = ds.variables[t_name][:].data.astype(np.float64)

    shear = []
    accel = []
    for vname in sorted(ds.variables.keys()):
        var = ds.variables[vname]
        if var.dimensions == ("time_fast",):
            data = var[:].data.astype(np.float64)
            if sh_re.match(vname):
                shear.append((vname, data))
            elif ac_re.match(vname):
                accel.append((vname, data))

    metadata = {"source": str(nc_path)}
    for attr in ("instrument_model", "instrument_sn", "source_file", "start_time"):
        if hasattr(ds, attr):
            metadata[attr] = getattr(ds, attr)

    vehicle = ""
    if hasattr(ds, "instrument_model"):
        model = ds.instrument_model.lower()
        if "vmp" in model:
            vehicle = "vmp"
        elif "xmp" in model:
            vehicle = "xmp"

    ds.close()

    return {
        "shear": shear,
        "accel": accel,
        "P": P,
        "T": T,
        "t_fast": t_fast,
        "t_slow": t_slow,
        "fs_fast": fs_fast,
        "fs_slow": fs_slow,
        "is_profile": is_profile,
        "vehicle": vehicle,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Profile preparation
# ---------------------------------------------------------------------------


def prepare_profiles(
    data: dict[str, Any],
    speed: float | None,
    direction: str,
    salinity: npt.ArrayLike | None,
    tau: float = 1.5,
    speed_cutout: float = 0.05,
) -> tuple | None:
    """Profile detection, speed computation, and salinity interpolation.

    Matches the ODAS odas_p2mat.m speed pipeline:
      1. W = gradient(P) filtered with Butterworth at 0.68/tau
      2. speed = abs(W)
      3. speed filtered again with Butterworth at 0.68/tau
      4. speed clamped to speed_cutout minimum

    Returns (profiles_slow, speed_fast, P_fast, T_fast, sal_fast, fs_fast,
    fs_slow, ratio, t_fast).
    """
    from scipy.signal import butter, filtfilt

    fs_fast = data["fs_fast"]
    fs_slow = data["fs_slow"]
    ratio = round(fs_fast / fs_slow)

    P_slow = data["P"]
    T_slow = data["T"]
    t_fast = data["t_fast"]
    t_slow = data["t_slow"]

    if data["is_profile"]:
        profiles_slow = [(0, len(P_slow) - 1)]
    else:
        from rsi_python.profile import _smooth_fall_rate, get_profiles

        W_slow = _smooth_fall_rate(P_slow, fs_slow, tau=tau)
        profiles_slow = get_profiles(P_slow, W_slow, fs_slow, direction=direction)
        if not profiles_slow:
            return None

    if speed is not None:
        speed_fast = np.full(len(t_fast), abs(speed))
    else:
        from rsi_python.profile import _smooth_fall_rate

        W_slow = _smooth_fall_rate(P_slow, fs_slow, tau=tau)

        speed_slow = np.abs(W_slow)
        speed_fast = np.interp(t_fast, t_slow, speed_slow)

        f_c = 0.68 / tau
        b_slow, a_slow = butter(1, f_c / (fs_slow / 2.0))
        speed_slow = filtfilt(b_slow, a_slow, speed_slow)
        b_fast, a_fast = butter(1, f_c / (fs_fast / 2.0))
        speed_fast = filtfilt(b_fast, a_fast, speed_fast)

        speed_slow[speed_slow < speed_cutout] = speed_cutout
        speed_fast[speed_fast < speed_cutout] = speed_cutout

    P_fast = np.interp(t_fast, t_slow, P_slow)
    T_fast = np.interp(t_fast, t_slow, T_slow)

    if salinity is not None:
        salinity = np.asarray(salinity, dtype=float)
        if salinity.ndim > 0:
            if len(salinity) == len(t_slow):
                sal_fast = np.interp(t_fast, t_slow, salinity)
            elif len(salinity) == len(t_fast):
                sal_fast = salinity
            else:
                raise ValueError(
                    f"salinity array length {len(salinity)} doesn't match "
                    f"slow ({len(t_slow)}) or fast ({len(t_fast)}) time series"
                )
        else:
            sal_fast = float(salinity)
    else:
        sal_fast = None

    return (profiles_slow, speed_fast, P_fast, T_fast, sal_fast, fs_fast, fs_slow, ratio, t_fast)


def compute_nu(
    mean_T: float, mean_P: float, salinity: float | np.ndarray | None, sel: slice
) -> float:
    """Compute kinematic viscosity, dispatching to visc35 or visc."""
    if salinity is not None:
        mean_S = float(np.mean(salinity[sel]) if np.ndim(salinity) > 0 else salinity)
        return float(visc(mean_T, mean_S, mean_P))
    return float(visc35(mean_T))
