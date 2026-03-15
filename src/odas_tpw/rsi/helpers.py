# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Shared helpers for loading instrument data and preparing profiles.

These functions bridge PFile/NetCDF reading with spectral processing.
Originally part of dissipation.py, extracted so that both epsilon and
chi packages can use them without circular dependencies.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from odas_tpw.rsi.p_file import PFile


class ChannelsDict(TypedDict, total=False):
    """Type for the dict returned by :func:`load_channels`."""

    shear: list[tuple[str, np.ndarray]]
    accel: list[tuple[str, np.ndarray]]
    P: np.ndarray
    T: np.ndarray
    t_fast: np.ndarray
    t_slow: np.ndarray
    fs_fast: float
    fs_slow: float
    is_profile: bool
    metadata: dict[str, str]
    vehicle: str

# ---------------------------------------------------------------------------
# Channel name patterns for RSI instruments
# ---------------------------------------------------------------------------

SH_PATTERN = re.compile(r"^sh\d+$")
AC_PATTERN = re.compile(r"^A[xyz]$")
T_PATTERN = re.compile(r"^T\d+$")
DT_PATTERN = re.compile(r"^T\d+_dT\d+$")

# ---------------------------------------------------------------------------
# Channel loading
# ---------------------------------------------------------------------------


def load_channels(
    source: PFile | str | Path,
    shear_pattern: str = r"^sh\d+$",
    accel_pattern: str = r"^A[xyz]$",
    pressure_name: str = "P",
    temperature_name: str = "T1",
) -> ChannelsDict:
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
    from odas_tpw.rsi.p_file import PFile

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
) -> ChannelsDict:
    sh_re = re.compile(sh_pat) if sh_pat != SH_PATTERN.pattern else SH_PATTERN
    ac_re = re.compile(ac_pat) if ac_pat != AC_PATTERN.pattern else AC_PATTERN
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
) -> ChannelsDict:
    import netCDF4 as nc

    ds = nc.Dataset(str(nc_path), "r")
    sh_re = re.compile(sh_pat) if sh_pat != SH_PATTERN.pattern else SH_PATTERN
    ac_re = re.compile(ac_pat) if ac_pat != AC_PATTERN.pattern else AC_PATTERN

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
        from odas_tpw.rsi.profile import _smooth_fall_rate, get_profiles

        W_slow = _smooth_fall_rate(P_slow, fs_slow, tau=tau)
        profiles_slow = get_profiles(P_slow, W_slow, fs_slow, direction=direction)
        if not profiles_slow:
            return None

    if speed is not None:
        speed_fast = np.full(len(t_fast), abs(speed))
    else:
        from odas_tpw.scor160.profile import compute_speed_fast

        speed_fast, _W_slow = compute_speed_fast(
            P_slow, t_fast, t_slow, fs_fast, fs_slow,
            tau=tau, speed_min=speed_cutout,
        )

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


# ---------------------------------------------------------------------------
# L1Data construction from load_channels output
# ---------------------------------------------------------------------------


def _build_l1data_from_channels(
    data: dict[str, Any],
    s_fast: int,
    e_fast: int,
    speed_fast: np.ndarray,
    P_fast: np.ndarray,
    T_fast: np.ndarray,
    direction: str = "down",
    *,
    therm_list: list[tuple[str, np.ndarray]] | None = None,
    diff_gains: list[float] | None = None,
) -> Any:
    """Build L1Data from load_channels output and interpolated profile arrays.

    Parameters
    ----------
    data : dict
        Output of load_channels().
    s_fast, e_fast : int
        Fast-rate slice indices for this profile.
    speed_fast : ndarray
        Full-length speed array (fast rate).
    P_fast, T_fast : ndarray
        Full-length pressure and temperature (interpolated to fast rate).
    direction : str
        Profile direction.
    therm_list : list of (name, array), optional
        Thermistor data for chi computation.
    diff_gains : list of float, optional
        Per-thermistor differentiator gains.
    """
    from odas_tpw.scor160.io import L1Data

    shear_arrays = [s[1] for s in data["shear"]]
    accel_arrays = [a[1] for a in data["accel"]]
    fs_fast = data["fs_fast"]
    n = e_fast - s_fast

    speed_prof = speed_fast[s_fast:e_fast]

    # Shear: normalize by speed^2
    if shear_arrays:
        shear = np.stack(
            [arr[s_fast:e_fast] / speed_prof**2 for arr in shear_arrays],
            axis=0,
        )
    else:
        shear = np.zeros((0, n), dtype=np.float64)

    # Vibration/accelerometer
    if accel_arrays:
        vib = np.stack([arr[s_fast:e_fast] for arr in accel_arrays], axis=0)
        vib_type = "ACC"
    else:
        vib = np.zeros((0, n), dtype=np.float64)
        vib_type = "NONE"

    # Optional fast temperature
    if therm_list:
        tf = np.stack([arr[s_fast:e_fast] for _, arr in therm_list], axis=0)
    else:
        tf = np.zeros((0, 0), dtype=np.float64)

    return L1Data(
        time=data["t_fast"][s_fast:e_fast],
        pres=P_fast[s_fast:e_fast],
        shear=shear,
        vib=vib,
        vib_type=vib_type,
        fs_fast=fs_fast,
        f_AA=98.0,
        vehicle=data.get("vehicle", ""),
        profile_dir=direction,
        time_reference_year=2000,
        pspd_rel=speed_prof,
        temp=T_fast[s_fast:e_fast],
        temp_fast=tf,
        diff_gains=diff_gains or [],
    )


# ---------------------------------------------------------------------------
# Shared file-level output
# ---------------------------------------------------------------------------


def write_profile_results(
    results: list,
    source_path: Path,
    output_dir: Path,
    suffix: str,
) -> list[Path]:
    """Write per-profile xarray Datasets to NetCDF files.

    Parameters
    ----------
    results : list of xr.Dataset
        Per-profile results.
    source_path : Path
        Original source file (stem used for naming).
    output_dir : Path
        Output directory (created if needed).
    suffix : str
        File suffix, e.g. 'eps' or 'chi'.

    Returns
    -------
    list of Path
        Paths to output files written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for i, ds in enumerate(results):
        if len(results) == 1:
            out_name = f"{source_path.stem}_{suffix}.nc"
        else:
            out_name = f"{source_path.stem}_prof{i + 1:03d}_{suffix}.nc"
        out_path = output_dir / out_name
        ds.to_netcdf(out_path)
        output_paths.append(out_path)
        print(
            f"  {out_path.name}: {ds.sizes['time']} estimates, "
            f"P={float(ds.P_mean.min()):.0f}-{float(ds.P_mean.max()):.0f} dbar"
        )
    return output_paths


# ---------------------------------------------------------------------------
# Shared dataset builder
# ---------------------------------------------------------------------------


def _build_result_dataset(
    variables: list[tuple[str, list[str], np.ndarray, dict]],
    probe_names: list[str],
    t_out: np.ndarray,
    probe_long_name: str,
    global_attrs: dict,
) -> "xr.Dataset":
    """Build an xarray Dataset from a list of variable specs.

    Parameters
    ----------
    variables : list of (name, dims, data, attrs)
        Variable definitions.
    probe_names : list of str
        Probe/sensor coordinate labels.
    t_out : ndarray
        Time coordinate values.
    probe_long_name : str
        Long name for the probe coordinate.
    global_attrs : dict
        Dataset-level attributes.
    """
    import xarray as xr

    data_vars = {
        name: (dims, data, attrs)
        for name, dims, data, attrs in variables
    }
    ds = xr.Dataset(
        data_vars,
        coords={
            "probe": probe_names,
            "t": (["time"], t_out),
        },
        attrs=global_attrs,
    )
    ds.coords["probe"].attrs["long_name"] = probe_long_name
    return ds
