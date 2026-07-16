"""Adapters from RSI PFile / NetCDF to scor160 L1Data.

Bridges instrument-specific I/O with the generic scor160 processing pipeline.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from odas_tpw.rsi.helpers import (
    AC_PATTERN,
    DT_PATTERN,
    SH_PATTERN,
    T_PATTERN,
    reference_temperature_qc,
    resolve_conductivity_channel,
    resolve_temperature_channel,
    temperature_candidates,
)
from odas_tpw.scor160.io import L1Data

if TYPE_CHECKING:
    from odas_tpw.rsi.p_file import PFile


def pfile_to_l1data(
    pf: PFile,
    profile_slice: tuple[int, int] | None = None,
    speed: float | None = None,
    direction: str = "auto",
    speed_tau: float = 1.5,
    speed_method: str = "pressure",
    aoa_deg: float = 3.0,
    temperature: str | float = "auto",
    conductivity: str = "auto",
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
    temperature : str or float
        Reference-temperature source for ``L1.temp``: ``"auto"`` (first
        plausible of T1..Tn, T, JAC_T, QC'd on the full channel and sliced
        after), an explicit channel name (QC failure warns but proceeds), or
        a number = constant reference temperature [degC] (ODAS
        ``constant_temp`` parity).
    conductivity : str
        Conductivity channel for the measured practical salinity
        (``"auto"`` = JAC_C when present, else no salinity).

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

    # Reference temperature (slow, for L1.temp): resolved on the FULL channel
    # (QC over the whole file), sliced after. An instrument with no slow
    # temperature channel at all still converts (historical tolerance —
    # downstream L4 substitutes 10 degC for NaN loudly); channels that exist
    # but are all implausible raise instead of yielding silent products.
    n_slow_full = len(pf.t_slow)
    ctx = str(getattr(pf, "filepath", "") or "")
    if isinstance(temperature, bool):
        raise ValueError(
            f"temperature={temperature!r} is not valid; use a channel name or a number"
        )
    if isinstance(temperature, (int, float)):
        T_slow_full = np.full(n_slow_full, float(temperature))
    elif temperature == "auto" and not temperature_candidates(pf.channels, n_slow_full):
        warnings.warn(
            "no slow temperature channel found; L1.temp will be NaN "
            "(downstream substitutes 10 degC for seawater properties)",
            stacklevel=2,
        )
        T_slow_full = np.full(n_slow_full, np.nan)
    else:
        t_chan, _t_reason = resolve_temperature_channel(
            pf.channels, n_slow_full, temperature, pressure=P_slow, context=ctx
        )
        T_slow_full = np.asarray(pf.channels[t_chan], dtype=np.float64)
    T_slow_slice = T_slow_full[s_slow : e_slow + 1]
    temp = np.interp(t_fast, t_slow, T_slow_slice)

    # Practical salinity (fast) from CTD conductivity when the .p file carries
    # a conductivity channel (VMP JAC_C, or an explicit selection). MicroRiders
    # have no conductivity at the .p level, so this stays empty and
    # stratification falls back to a supplied/assumed S. The pair temperature
    # is the co-located JAC_T when conductivity is JAC_C and JAC_T passes
    # plausibility QC, else the resolved reference temperature (a railed JAC_T
    # must not silently poison the measured salinity).
    salinity_fast = np.array([])
    c_chan = resolve_conductivity_channel(pf.channels, n_slow_full, conductivity, context=ctx)
    if c_chan is not None:
        import gsw

        pair_T = None
        if c_chan == "JAC_C":
            JAC_T = pf.channels.get("JAC_T")
            if JAC_T is not None and len(JAC_T) == n_slow_full:
                reason = reference_temperature_qc(JAC_T, pressure=P_slow)
                if reason is None:
                    pair_T = JAC_T
                else:
                    warnings.warn(
                        f"JAC_T fails plausibility QC ({reason}); pairing JAC_C "
                        "with the reference temperature for measured salinity",
                        stacklevel=2,
                    )
        if pair_T is None:
            pair_T = T_slow_full
        pair_slice = pair_T[s_slow : e_slow + 1]
        if np.isfinite(pair_slice).any():
            sp_slow = gsw.SP_from_C(
                pf.channels[c_chan][s_slow : e_slow + 1], pair_slice, P_slow_slice
            )
            salinity_fast = np.interp(t_fast, t_slow, sp_slow)

    # Profiling speed
    if speed is not None:
        # Floor at the shared speed cutout (0.05 m/s, = speed.py's default
        # speed_cutout and its 'constant' method). Without it, shear /=
        # pspd_rel**2 below yields inf/nan for --speed 0 and huge
        # over-amplification for a tiny speed, baked into L1.shear.
        pspd_rel = np.full(len(t_fast), max(abs(speed), 0.05))
    elif speed_method in ("em", "flight"):
        # EM flowmeter (U_EM) or inviscid flight model U=|W|/sin(|pitch|-aoa)
        # from the inclinometers, for MicroRiders/gliders where |dP/dt| is the
        # vertical speed, not the through-water flow. Shared with perturb.
        from odas_tpw.rsi.speed import compute_speed_for_pfile

        speed_fast_full, _ = compute_speed_for_pfile(
            pf, {"method": speed_method, "aoa_deg": aoa_deg, "tau": speed_tau}, vehicle
        )
        pspd_rel = speed_fast_full[s_fast:e_fast]
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
        f_AA=98.0,  # RSI/ODAS default AA cutoff, hardcoded (not parsed from config); #104 U1-4
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
        salinity=salinity_fast,
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

    Notes
    -----
    ``temp`` here is the nanmean of the FP07 stack (a different path from
    ``pfile_to_l1data``); wiring it through the reference-temperature
    resolver is deferred to #131 W5.
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

        # Temperature gradient (for slow-rate temp). nanmean over probes so an
        # all-NaN thermistor doesn't poison temp at every sample (mirrors the
        # nanmean tolerance elsewhere in the chi/epsilon paths).
        temp = np.zeros_like(time)
        if temp_fast.size > 0:
            temp = np.nanmean(temp_fast, axis=0) if temp_fast.ndim == 2 else temp_fast

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
