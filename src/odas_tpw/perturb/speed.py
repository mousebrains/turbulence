# May-2026, Pat Welch, pat@mousebrains.com
"""Configurable through-water speed computation for perturb.

Run *after* the ``.p`` file is loaded and any hotel data has been merged
into ``pf.channels``, so ``compute_speed_for_pfile`` sees the union of
both sources.

Methods
-------
``"pressure"``
    ODAS-style smoothed ``|dP/dt|`` at slow rate, interpolated and
    Butterworth-smoothed onto fast rate. This is the historical default
    (correct for VMP, returns vertical speed for a glider).

``"em"``
    Use the ``U_EM`` channel from the ``.p`` file directly — the
    MicroRider's electromagnetic flowmeter measurement of along-axis
    flow. Errors out if ``U_EM`` is missing.

``"flight"``
    Construct an inviscid flight-model along-axis speed from the MR's
    own pressure and inclinometers:

        U_along = |W| / (sin(|pitch| - aoa) * cos(|roll|))

    where ``aoa`` is the angle of attack (``aoa_deg``, default 3 deg
    matching ODAS Slocum trim). The pitch axis is auto-picked from the
    inclinometer axis with the larger swing -- Slocum mountings some-
    times have ``Incl_X`` carrying pitch, sometimes ``Incl_Y``.

``"constant"``
    Use the scalar in ``speed.value`` for every fast-rate sample.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_speed_for_pfile(
    pf: Any,
    speed_cfg: dict | None,
    vehicle: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (speed_fast, W_slow) from configured method, in m/s and dbar/s.

    Parameters
    ----------
    pf : PFile
        Has ``channels`` (already merged with hotel), ``t_fast``,
        ``t_slow``, ``fs_fast``, ``fs_slow``.
    speed_cfg : dict
        Merged ``speed:`` section from the perturb config.
    vehicle : str | None
        Vehicle string from instrument_info (``"slocum_glider"``,
        ``"vmp"`` etc.). Used only to auto-resolve ``tau``.

    Returns
    -------
    speed_fast : (n_fast,) float64, m/s, >= ``speed_cutout``.
    W_slow     : (n_slow,) float64, dbar/s. Always the smoothed |dP/dt|
                 -- independent of method, useful for QC/binning.
    """
    from odas_tpw.rsi.vehicle import resolve_tau
    from odas_tpw.scor160.profile import compute_speed_fast as _ode_speed
    from odas_tpw.scor160.profile import smooth_fall_rate

    cfg = speed_cfg or {}
    method = cfg.get("method", "pressure")
    speed_cutout = float(cfg.get("speed_cutout", 0.05))
    tau = float(cfg.get("tau") or resolve_tau(vehicle or ""))

    P_slow = np.asarray(pf.channels["P"], dtype=np.float64)
    fs_slow = float(pf.fs_slow)
    fs_fast = float(pf.fs_fast)
    t_fast = np.asarray(pf.t_fast, dtype=np.float64)
    t_slow = np.asarray(pf.t_slow, dtype=np.float64)

    # W_slow always: smoothed |dP/dt|, regardless of method.
    W_slow = smooth_fall_rate(P_slow, fs_slow, tau=tau)

    if method == "pressure":
        speed_fast, _ = _ode_speed(
            P_slow, t_fast, t_slow, fs_fast, fs_slow,
            tau=tau, speed_min=speed_cutout,
        )
        return speed_fast, W_slow

    if method == "constant":
        v = cfg.get("value")
        if v is None:
            raise ValueError("speed.method='constant' but speed.value is null")
        speed_fast = np.full(len(t_fast), max(abs(float(v)), speed_cutout))
        return speed_fast, W_slow

    if method == "em":
        if "U_EM" not in pf.channels:
            raise ValueError(
                "speed.method='em' but channel U_EM is missing from the "
                ".p file. Use 'flight' or 'pressure' instead."
            )
        U_em_slow = np.abs(np.asarray(pf.channels["U_EM"], dtype=np.float64))
        return _slow_to_fast(U_em_slow, t_fast, t_slow, fs_fast, fs_slow,
                             tau=tau, speed_min=speed_cutout), W_slow

    if method == "flight":
        aoa_deg = float(cfg.get("aoa_deg", 3.0))
        min_pitch_deg = float(cfg.get("min_pitch_deg", 5.0))
        speed_slow = _flight_model_slow(
            W_slow, pf, aoa_deg=aoa_deg, min_pitch_deg=min_pitch_deg,
        )
        return _slow_to_fast(speed_slow, t_fast, t_slow, fs_fast, fs_slow,
                             tau=tau, speed_min=speed_cutout), W_slow

    raise ValueError(
        f"Unknown speed.method={method!r}. "
        "Expected: pressure | em | flight | constant."
    )


def _slow_to_fast(
    arr_slow: np.ndarray,
    t_fast: np.ndarray,
    t_slow: np.ndarray,
    fs_fast: float,
    fs_slow: float,
    tau: float,
    speed_min: float,
) -> np.ndarray:
    """Match scor160 speed pipeline: linear interp + Butterworth smoothing."""
    from scipy.signal import butter, filtfilt

    # NaN-safe interp by filling on the slow grid first.
    finite = np.isfinite(arr_slow)
    if not finite.any():
        return np.full(len(t_fast), speed_min)
    arr_slow = np.where(
        finite, arr_slow,
        np.interp(t_slow, t_slow[finite], arr_slow[finite]),
    )
    arr_fast = np.interp(t_fast, t_slow, arr_slow)
    f_c = 0.68 / tau
    b, a = butter(1, f_c / (fs_fast / 2.0))
    arr_fast = np.asarray(filtfilt(b, a, arr_fast))
    return np.asarray(np.maximum(arr_fast, speed_min), dtype=np.float64)


def _flight_model_slow(
    W_slow: np.ndarray,
    pf: Any,
    aoa_deg: float,
    min_pitch_deg: float,
) -> np.ndarray:
    """U_along = |W| / (sin(|pitch|-aoa) * cos|roll|), at slow rate.

    Pitch axis auto-picked from whichever of ``Incl_X``/``Incl_Y`` has
    the larger swing -- Slocum mountings vary.
    """
    iX = np.asarray(pf.channels.get("Incl_X"), dtype=np.float64) \
        if "Incl_X" in pf.channels else None
    iY = np.asarray(pf.channels.get("Incl_Y"), dtype=np.float64) \
        if "Incl_Y" in pf.channels else None
    if iX is None or iY is None:
        raise ValueError(
            "speed.method='flight' needs Incl_X and Incl_Y channels in the "
            ".p file. Use 'pressure' or 'em' instead."
        )
    iX_rng = float(np.nanmax(iX) - np.nanmin(iX))
    iY_rng = float(np.nanmax(iY) - np.nanmin(iY))
    pitch_deg = iX if iX_rng >= iY_rng else iY
    roll_deg = iY if iX_rng >= iY_rng else iX

    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)
    aoa = np.deg2rad(aoa_deg)
    eff = np.maximum(np.abs(pitch) - aoa, 0.0)
    sin_path = np.sin(eff) * np.cos(roll)
    sin_floor = np.sin(np.deg2rad(min_pitch_deg))
    sin_path = np.where(sin_path > sin_floor, sin_path, np.nan)
    return np.asarray(np.abs(W_slow) / sin_path, dtype=np.float64)
