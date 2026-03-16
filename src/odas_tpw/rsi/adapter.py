"""Adapters from RSI PFile / NetCDF to scor160 L1Data.

Bridges instrument-specific I/O with the generic scor160 processing pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from odas_tpw.rsi.helpers import AC_PATTERN, DT_PATTERN, SH_PATTERN, T_PATTERN
from odas_tpw.scor160.io import L1Data

if TYPE_CHECKING:
    from odas_tpw.rsi.p_file import PFile


def pfile_to_l1data(
    pf: PFile,
    profile_slice: tuple[int, int] | None = None,
    speed: float | None = None,
    direction: str = "auto",
    speed_tau: float = 1.5,
) -> L1Data:
    """Convert a PFile (or profile slice) to scor160 L1Data.

    Parameters
    ----------
    pf : PFile
        Parsed .P file.
    profile_slice : tuple of (start, end) slow-rate indices, optional
        If provided, extract only this slice (from profile detection).
    speed : float, optional
        Fixed profiling speed [m/s]. If None, computed from dP/dt.
    direction : str
        'up' or 'down' for speed sign convention.
    speed_tau : float
        Speed smoothing time constant [s].

    Returns
    -------
    L1Data
    """
    from odas_tpw.rsi.vehicle import resolve_direction

    vehicle = pf.config["instrument_info"].get("vehicle", "").lower()
    direction = resolve_direction(direction, vehicle)

    ratio = round(pf.fs_fast / pf.fs_slow)

    if profile_slice is not None:
        s_slow, e_slow = profile_slice
        s_fast = s_slow * ratio
        e_fast = min((e_slow + 1) * ratio, len(pf.t_fast))
    else:
        s_slow, e_slow = 0, len(pf.t_slow) - 1
        s_fast, e_fast = 0, len(pf.t_fast)

    t_fast = pf.t_fast[s_fast:e_fast]
    t_slow = pf.t_slow[s_slow : e_slow + 1]

    # Pressure
    P_slow = pf.channels.get("P_dP", pf.channels.get("P"))
    if P_slow is None:
        raise ValueError("No pressure channel (P or P_dP) found")
    P_slow_slice = P_slow[s_slow : e_slow + 1]
    pres = np.interp(t_fast, t_slow, P_slow_slice)

    # Temperature (slow, for L1.temp)
    T_slow = pf.channels.get("T1", np.full(len(pf.t_slow), np.nan))
    T_slow_slice = T_slow[s_slow : e_slow + 1]
    temp = np.interp(t_fast, t_slow, T_slow_slice)

    # Profiling speed
    if speed is not None:
        pspd_rel = np.full(len(t_fast), abs(speed))
    else:
        from odas_tpw.scor160.profile import smooth_fall_rate, smooth_speed_interp

        # Must filter full P_slow before slicing to avoid Butterworth edge effects
        W_slow_full = smooth_fall_rate(P_slow, pf.fs_slow, tau=speed_tau)
        speed_slow = np.abs(W_slow_full[s_slow : e_slow + 1])
        pspd_rel = smooth_speed_interp(
            speed_slow,
            t_fast,
            t_slow,
            pf.fs_fast,
            speed_tau,
        )

    # Shear channels — normalize by speed^2 to get du/dz [s^-1]
    shear_names = sorted(n for n in pf._fast_channels if SH_PATTERN.match(n))
    if shear_names:
        shear = np.stack(
            [pf.channels[n][s_fast:e_fast] / pspd_rel**2 for n in shear_names],
            axis=0,
        )
    else:
        shear = np.zeros((0, len(t_fast)), dtype=np.float64)

    # Vibration / accelerometer channels
    acc_names = sorted(n for n in pf._fast_channels if AC_PATTERN.match(n))
    vib_names = sorted(n for n in pf.channels if pf.channel_info[n]["type"] == "piezo")
    if acc_names:
        vib = np.stack(
            [pf.channels[n][s_fast:e_fast] for n in acc_names],
            axis=0,
        )
        vib_type = "ACC"
    elif vib_names:
        vib = np.stack(
            [pf.channels[n][s_fast:e_fast] for n in vib_names],
            axis=0,
        )
        vib_type = "VIB"
    else:
        vib = np.zeros((0, len(t_fast)), dtype=np.float64)
        vib_type = "NONE"

    # Fast thermistor temperature (for chi)
    temp_fast_list: list[np.ndarray] = []
    diff_gains: list[float] = []

    # Prefer deconvolved pre-emphasized channels (T1_dT1, T2_dT2)
    for name in sorted(pf._fast_channels):
        if DT_PATTERN.match(name):
            temp_fast_list.append(pf.channels[name][s_fast:e_fast])
            ch_cfg: dict = next(
                (ch for ch in pf.config["channels"] if ch.get("name") == name),
                {},
            )
            diff_gains.append(float(ch_cfg.get("diff_gain", "0.94")))

    # Fallback to T channels
    if not temp_fast_list:
        for name in sorted(pf._fast_channels):
            if T_PATTERN.match(name):
                temp_fast_list.append(pf.channels[name][s_fast:e_fast])
                diff_gains.append(0.94)

    if temp_fast_list:
        temp_fast = np.stack(temp_fast_list, axis=0)
    else:
        temp_fast = np.zeros((0, 0), dtype=np.float64)

    # Time reference
    ref_year = pf.start_time.year

    return L1Data(
        time=t_fast,
        pres=pres,
        shear=shear,
        vib=vib,
        vib_type=vib_type,
        fs_fast=pf.fs_fast,
        f_AA=98.0,
        vehicle=pf.config["instrument_info"].get("vehicle", "").lower(),
        profile_dir=direction,
        time_reference_year=ref_year,
        pspd_rel=pspd_rel,
        time_slow=t_slow,
        pres_slow=P_slow_slice,
        temp=temp,
        fs_slow=pf.fs_slow,
        temp_fast=temp_fast,
        diff_gains=diff_gains,
    )


def nc_to_l1data(nc_path: str | Path) -> L1Data:
    """Read a per-profile L1_converted NetCDF into L1Data.

    Parameters
    ----------
    nc_path : str or Path
        Path to an L1_converted NetCDF file (output of ``p_to_L1`` +
        ``extract_profiles``).

    Returns
    -------
    L1Data
    """
    import netCDF4

    ds = netCDF4.Dataset(str(nc_path), "r")
    try:
        # Navigate to L1_converted group if present, else use root
        g = ds.groups.get("L1_converted", ds)

        time = np.asarray(g.variables["TIME"][:], dtype=np.float64)

        # Pressure
        pres = np.asarray(g.variables["PRES"][:], dtype=np.float64)
        if pres.shape[0] != len(time):
            pres = np.interp(
                np.linspace(0, 1, len(time)),
                np.linspace(0, 1, len(pres)),
                pres,
            )

        # Shear
        if "SHEAR" in g.variables:
            shear = np.asarray(g.variables["SHEAR"][:], dtype=np.float64)
        else:
            shear = np.zeros((0, len(time)), dtype=np.float64)

        # Vibration / Accelerometer
        if "ACC" in g.variables:
            vib = np.asarray(g.variables["ACC"][:], dtype=np.float64)
            vib_type = "ACC"
        elif "VIB" in g.variables:
            vib = np.asarray(g.variables["VIB"][:], dtype=np.float64)
            vib_type = "VIB"
        else:
            vib = np.zeros((0, len(time)), dtype=np.float64)
            vib_type = "NONE"

        # Speed
        pspd_rel = np.array([])
        if "PSPD_REL" in g.variables:
            pspd_rel = np.asarray(g.variables["PSPD_REL"][:], dtype=np.float64)
            if pspd_rel.shape[0] != len(time):
                pspd_rel = np.interp(
                    np.linspace(0, 1, len(time)),
                    np.linspace(0, 1, len(pspd_rel)),
                    pspd_rel,
                )

        # Temperature (fast-rate FP07)
        temp_fast = np.zeros((0, 0), dtype=np.float64)
        if "TEMP" in g.variables:
            temp_arr = np.asarray(g.variables["TEMP"][:], dtype=np.float64)
            if temp_arr.ndim == 2 and temp_arr.shape[1] == len(time):
                temp_fast = temp_arr

        # Temperature gradient (for slow-rate temp)
        temp = np.zeros_like(time)
        if temp_fast.size > 0:
            temp = np.mean(temp_fast, axis=0) if temp_fast.ndim == 2 else temp_fast

        # Slow-rate arrays
        time_slow = np.array([])
        pres_slow = np.array([])
        if "TIME_SLOW" in g.variables:
            time_slow = np.asarray(g.variables["TIME_SLOW"][:], dtype=np.float64)
        if "PRES_SLOW" in g.variables:
            pres_slow = np.asarray(g.variables["PRES_SLOW"][:], dtype=np.float64)

        # Attributes
        def _get_attr(name, default):
            for src in [g, ds]:
                if name in {a for a in src.ncattrs()}:
                    return src.getncattr(name)
            return default

        fs_fast = float(_get_attr("fs_fast", 512.0))
        fs_slow = float(_get_attr("fs_slow", 0.0))

        return L1Data(
            time=time,
            pres=pres,
            shear=shear,
            vib=vib,
            vib_type=vib_type,
            fs_fast=fs_fast,
            f_AA=float(_get_attr("f_AA", 98.0)),
            vehicle=str(_get_attr("vehicle", "unknown")),
            profile_dir=str(_get_attr("profile_dir", "down")),
            time_reference_year=int(_get_attr("time_reference_year", 2000)),
            pspd_rel=pspd_rel,
            time_slow=time_slow,
            pres_slow=pres_slow,
            temp=temp,
            fs_slow=fs_slow,
            temp_fast=temp_fast,
            diff_gains=[],
        )
    finally:
        ds.close()
