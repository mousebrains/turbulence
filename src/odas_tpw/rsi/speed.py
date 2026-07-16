# May-2026, Pat Welch, pat@mousebrains.com
"""Configurable through-water speed computation from a loaded ``.p`` file.

Shared by both the ``rsi`` and ``perturb`` pipelines (``perturb.speed``
re-exports ``compute_speed_for_pfile``). Run *after* the ``.p`` file is loaded
and any hotel data has been merged into ``pf.channels``, so
``compute_speed_for_pfile`` sees the union of both sources.

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

        U_along = |W| / sin(|pitch| + aoa)

    where ``aoa`` is the angle of attack (``aoa_deg``, default 3 deg
    matching ODAS Slocum trim). The glide path is steeper than the
    pitch attitude by the angle of attack (ODAS odas_p2mat.m uses
    ``abs(Incl_Y) + aoa``). The pitch axis is auto-picked from the
    inclinometer axis with the larger swing -- Slocum mountings some-
    times have ``Incl_X`` carrying pitch, sometimes ``Incl_Y``.

``"constant"``
    Use the scalar in ``speed.value`` for every fast-rate sample.

``"hotel"``
    Use a hotel-merged channel (named by ``speed.hotel_var``, default
    ``"speed"``) as the through-water speed — external vehicle telemetry
    interpolated onto the instrument grids by the perturb hotel merge.
    The channel's grid is taken from ``pf.is_fast`` (the default
    ``hotel.fast_channels`` puts ``"speed"`` on the FAST grid); a
    slow-grid channel is interpolated and Butterworth-smoothed to fast
    rate exactly like the em/flight methods. An unusable channel
    (missing, on neither grid, or mostly non-finite) is an ERROR — an
    explicitly requested hotel speed is never silently replaced by the
    ``speed_cutout`` floor (issue #131 M10).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np


def compute_speed_for_pfile(
    pf: Any,
    speed_cfg: dict | None,
    vehicle: str | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (speed_fast, W_slow, source) from the configured method.

    Parameters
    ----------
    pf : PFile
        Has ``channels`` (already merged with hotel), ``t_fast``,
        ``t_slow``, ``fs_fast``, ``fs_slow``.
    speed_cfg : dict
        Merged ``speed:`` section from the perturb config.
    vehicle : str | None
        Vehicle string from instrument_info (``"slocum_glider"``,
        ``"vmp"`` etc.). Auto-resolves ``tau`` and, for the ``pressure``
        method, triggers a bias warning on glide/horizontal platforms
        (where ``|dP/dt|`` is not the through-water speed).

    Returns
    -------
    speed_fast : (n_fast,) float64, m/s, >= ``speed_cutout``.
                 For the pressure method this is numerically |dP/dt| in
                 dbar/s treated as m/s (the ODAS convention). 1 dbar is
                 ~0.99 m (equivalently 1 m ~ 1.006-1.009 dbar depending on
                 latitude/density), so |dP/dt| in dbar/s slightly OVER-states
                 the true speed in m/s, by <1%. Through epsilon's strong
                 inverse speed leverage (~U^-4 in the shear conversion and
                 wavenumber transform) that biases the reported epsilon
                 slightly LOW.
    W_slow     : (n_slow,) float64, dbar/s. Always the smoothed |dP/dt|
                 -- independent of method, useful for QC/binning.
    source     : str, the provenance vocabulary for the speed actually
                 computed: ``"pressure"`` | ``"em"`` | ``"flight"`` |
                 ``"constant:<v>"`` | ``"hotel:<var>"``. Callers stamp
                 product provenance from this return value rather than
                 re-deriving it from the cfg.
    """
    from odas_tpw.rsi.vehicle import resolve_direction, resolve_tau
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
        # For a glider / horizontal platform, |dP/dt| is the VERTICAL speed and
        # underestimates the along-path flow past the sensors; epsilon has ~U^4
        # leverage, so this strongly biases epsilon high. The rsi path warns
        # about this in helpers.prepare_profiles, but perturb precomputes speed
        # here and so never reaches that warning — emit the equivalent here so
        # the perturb path is not silent (audit r2-3). VMP (down/up) is unaffected.
        direction = resolve_direction("auto", vehicle or "")
        if direction in ("glide", "horizontal"):
            warnings.warn(
                f"Vehicle direction '{direction}' but speed.method='pressure' "
                "computes speed from |dP/dt| (vertical); set speed.method="
                "'flight' (glider) or 'em' (horizontal), or speed.value — "
                "epsilon scales as ~U^4 and will be strongly biased",
                stacklevel=2,
            )
        speed_fast, _ = _ode_speed(
            P_slow, t_fast, t_slow, fs_fast, fs_slow,
            tau=tau, speed_min=speed_cutout,
        )
        return speed_fast, W_slow, "pressure"

    # Finite-coverage guards for EXPLICITLY requested non-pressure methods,
    # applied BEFORE _slow_to_fast's cutout floor: _slow_to_fast fills an
    # all-NaN input with speed_min (that fill is load-bearing for the
    # pressure path and stays), which would otherwise publish a constant
    # 0.05 m/s with provenance "em"/"flight" — missing telemetry would be
    # indistinguishable from a real 0.05 m/s speed, and the perturb
    # explicit-method abort could never fire (the array is already finite).
    # Threshold asymmetry vs the hotel method (50% rule): the flight model
    # legitimately NaNs every sample below min_pitch_deg (dive/climb
    # inflections), so a fraction threshold could false-error real casts —
    # em/flight only reject ZERO finite samples.

    if method == "constant":
        v = cfg.get("value")
        if v is None:
            raise ValueError("speed.method='constant' but speed.value is null")
        v = float(v)
        if not np.isfinite(v):
            raise ValueError(
                f"speed.method='constant' but speed.value={v!r} is not finite"
            )
        speed_fast = np.full(len(t_fast), max(abs(v), speed_cutout))
        return speed_fast, W_slow, f"constant:{v:g}"

    if method == "em":
        if "U_EM" not in pf.channels:
            raise ValueError(
                "speed.method='em' but channel U_EM is missing from the "
                "source. Use 'flight' or 'pressure' instead."
            )
        U_em_slow = np.abs(np.asarray(pf.channels["U_EM"], dtype=np.float64))
        if not np.isfinite(U_em_slow).any():
            raise ValueError(
                "speed.method='em': channel U_EM has no finite samples "
                "(dead or disconnected flowmeter) — refusing to publish "
                f"the {speed_cutout:g} m/s speed_cutout floor as "
                "through-water speed."
            )
        return _slow_to_fast(U_em_slow, t_fast, t_slow, fs_fast, fs_slow,
                             tau=tau, speed_min=speed_cutout), W_slow, "em"

    if method == "flight":
        aoa_deg = float(cfg.get("aoa_deg", 3.0))
        min_pitch_deg = float(cfg.get("min_pitch_deg", 5.0))
        # Reject implausible flight parameters up front: a negative aoa can
        # drive the effective glide angle negative, and the resulting negative
        # speed would be silently replaced by the 0.05 m/s cutout — a
        # plausible-looking wrong answer for an accepted input. The physical
        # attack angle is a small positive number (ODAS default 3 deg).
        if not np.isfinite(aoa_deg) or aoa_deg < 0:
            raise ValueError(
                f"speed.aoa_deg={aoa_deg!r} is not valid: the angle of attack "
                "must be a finite value >= 0 deg (ODAS Slocum default: 3)"
            )
        if not np.isfinite(min_pitch_deg) or min_pitch_deg < 0:
            raise ValueError(
                f"speed.min_pitch_deg={min_pitch_deg!r} is not valid: the "
                "minimum pitch gate must be a finite value >= 0 deg"
            )
        aq = cfg.get("amplitude_quantile") or (1.0, 99.0)
        speed_slow = _flight_model_slow(
            W_slow, pf, aoa_deg=aoa_deg, min_pitch_deg=min_pitch_deg,
            amplitude_quantile=(float(aq[0]), float(aq[1])),
        )
        if not np.isfinite(speed_slow).any():
            raise ValueError(
                "speed.method='flight': the flight model produced no finite "
                "samples — the effective pitch never cleared min_pitch_deg "
                f"({min_pitch_deg:g} deg; level flight / all-inflection "
                "record) or the inclinometer/pressure inputs are all-NaN — "
                f"refusing to publish the {speed_cutout:g} m/s speed_cutout "
                "floor as through-water speed."
            )
        return _slow_to_fast(speed_slow, t_fast, t_slow, fs_fast, fs_slow,
                             tau=tau, speed_min=speed_cutout), W_slow, "flight"

    if method == "hotel":
        hotel_var = str(cfg.get("hotel_var") or "speed")
        speed_fast = _hotel_speed_fast(
            pf, hotel_var, t_fast, t_slow, fs_fast, fs_slow,
            tau=tau, speed_min=speed_cutout,
        )
        return speed_fast, W_slow, f"hotel:{hotel_var}"

    raise ValueError(
        f"Unknown speed.method={method!r}. "
        "Expected: pressure | em | flight | constant | hotel."
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


# A hotel speed channel that is mostly non-finite (external feed dropped out,
# wrong hotel.channels mapping, derived-variable NaN cascade) must be an error,
# not data: _slow_to_fast would silently fill an all-NaN channel with the
# speed_cutout floor, and epsilon has ~U^4 leverage on speed (#131 M10 / F6).
_HOTEL_MIN_FINITE_FRACTION = 0.5


def _hotel_speed_fast(
    pf: Any,
    hotel_var: str,
    t_fast: np.ndarray,
    t_slow: np.ndarray,
    fs_fast: float,
    fs_slow: float,
    tau: float,
    speed_min: float,
) -> np.ndarray:
    """Through-water speed from a hotel-merged channel, on the fast grid.

    Raises ``ValueError`` (never falls back) when the channel is missing,
    matches neither grid, or is less than ``_HOTEL_MIN_FINITE_FRACTION``
    finite — an explicitly requested hotel speed must not silently publish
    the ``speed_min`` floor as data.
    """
    if hotel_var not in pf.channels:
        raise ValueError(
            f"speed.method='hotel' but channel {hotel_var!r} is not present "
            "after the hotel merge. Map a hotel variable onto it via "
            f"hotel.channels (e.g. hotel.channels.m_speed: {hotel_var!r}) or "
            "point speed.hotel_var at the merged channel name."
        )
    arr = np.asarray(pf.channels[hotel_var], dtype=np.float64)
    n_fast, n_slow = len(t_fast), len(t_slow)

    # Grid resolution: trust the merge's fast/slow registration when the
    # source exposes it (the default hotel.fast_channels puts "speed" on the
    # FAST grid); fall back to length matching for duck-typed sources.
    is_fast: bool | None
    if hasattr(pf, "is_fast"):
        is_fast = bool(pf.is_fast(hotel_var))
    elif len(arr) == n_fast:
        is_fast = True
    elif len(arr) == n_slow:
        is_fast = False
    else:
        is_fast = None
    if is_fast is None:
        raise ValueError(
            f"speed.method='hotel': channel {hotel_var!r} has length "
            f"{len(arr)}, matching neither the fast grid ({n_fast}) nor the "
            f"slow grid ({n_slow}); check the hotel.channels mapping (its "
            "'fast:' option controls the target grid)."
        )
    n_expected = n_fast if is_fast else n_slow
    if len(arr) != n_expected:
        # Phrased as "resolves to", not "registered on": PFile.is_fast
        # returns False for names never registered at all, and those land
        # here too when their length is not the slow-grid length.
        grid = "fast" if is_fast else "slow"
        raise ValueError(
            f"speed.method='hotel': channel {hotel_var!r} has length "
            f"{len(arr)}, which does not match the {grid} grid it resolves "
            f"to ({n_expected} samples); check the hotel.channels mapping "
            "(its 'fast:' option controls the target grid)."
        )

    finite = np.isfinite(arr)
    finite_fraction = float(finite.mean()) if arr.size else 0.0
    if finite_fraction < _HOTEL_MIN_FINITE_FRACTION:
        raise ValueError(
            f"speed.method='hotel': channel {hotel_var!r} is only "
            f"{100.0 * finite_fraction:.1f}% finite "
            f"(< {100.0 * _HOTEL_MIN_FINITE_FRACTION:.0f}%) — refusing to "
            f"publish the {speed_min:g} m/s speed_cutout floor as "
            "through-water speed. Check the hotel file's coverage and the "
            f"hotel.channels mapping for {hotel_var!r}."
        )

    arr = np.abs(arr)
    if is_fast:
        return _clean_fast_speed(arr, t_fast, fs_fast, tau=tau, speed_min=speed_min)
    return _slow_to_fast(arr, t_fast, t_slow, fs_fast, fs_slow,
                         tau=tau, speed_min=speed_min)


def _clean_fast_speed(
    arr_fast: np.ndarray,
    t_fast: np.ndarray,
    fs_fast: float,
    tau: float,
    speed_min: float,
) -> np.ndarray:
    """Mirror ``_slow_to_fast``'s NaN-interp/Butterworth/floor treatment for
    an array already on the fast grid (no regridding step).

    The caller guarantees at least one finite sample (finite-fraction gate).
    """
    from scipy.signal import butter, filtfilt

    finite = np.isfinite(arr_fast)
    if not finite.all():
        arr_fast = np.where(
            finite, arr_fast,
            np.interp(t_fast, t_fast[finite], arr_fast[finite]),
        )
    f_c = 0.68 / tau
    b, a = butter(1, f_c / (fs_fast / 2.0))
    arr_fast = np.asarray(filtfilt(b, a, arr_fast))
    return np.asarray(np.maximum(arr_fast, speed_min), dtype=np.float64)


def _flight_model_slow(
    W_slow: np.ndarray,
    pf: Any,
    aoa_deg: float,
    min_pitch_deg: float,
    amplitude_quantile: tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    """U_along = |W| / sin(glide angle), at slow rate.

    Steady-flight kinematics (Merckelbach et al. 2010,
    doi:10.1175/2009JTECHO710.1): W = U * sin(gamma) with glide-path
    angle |gamma| = |pitch| + angle of attack, so U = |W| / sin(|gamma|).
    The glide path is STEEPER than the pitch attitude on both dive and
    climb: the flow must meet the wing at a nonzero attack angle for its
    lift to balance the net (negative or positive) buoyancy, which slips
    the velocity vector away from the nose toward the buoyancy force —
    the attack angle's MAGNITUDE adds to |pitch| in both directions
    (in Merckelbach's signed convention alpha flips sign on climb).
    ODAS agrees:
    odas_p2mat.m computes glide_angle = abs(Incl_Y) + aoa. An earlier
    version here SUBTRACTED aoa, inflating U by sin(|pitch|+aoa)/
    sin(|pitch|-aoa) (1.24x at 26 deg pitch) and, through epsilon's
    ~U^-4 leverage, biasing epsilon ~2.4x low (issue #131 M7).
    Roll does not enter the along-path relation (it rotates the body
    about the flight axis without changing the vertical velocity
    component); an earlier version multiplied by cos(roll), which
    inflated U by 1/cos(roll).

    Pitch axis auto-picked from whichever of ``Incl_X``/``Incl_Y`` has
    the larger percentile-spread amplitude (default 99-1) -- Slocum
    mountings vary, and percentiles keep brief outlier spikes from
    masquerading as the high-amplitude flight axis.
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
    lo_q, hi_q = float(amplitude_quantile[0]), float(amplitude_quantile[1])
    iX_rng = float(np.nanpercentile(iX, hi_q) - np.nanpercentile(iX, lo_q))
    iY_rng = float(np.nanpercentile(iY, hi_q) - np.nanpercentile(iY, lo_q))
    pitch_deg = iX if iX_rng >= iY_rng else iY

    pitch = np.deg2rad(pitch_deg)
    aoa = np.deg2rad(aoa_deg)
    # Clamp at 90 deg: inclinometers rail at +/-90 and sin() is non-monotonic
    # past vertical; a vertical glide path means U = |W| exactly.
    eff = np.minimum(np.abs(pitch) + aoa, np.pi / 2)
    sin_path = np.sin(eff)
    # Gate on ATTITUDE, not the aoa-shifted path angle: the steady-glide
    # model is invalid near dive/climb inflections, and that invalidity is
    # a property of |pitch| itself. (min_pitch_deg=5 -> samples with
    # |pitch| < 5 deg are NaN'd and bridged by _slow_to_fast's interp.)
    sin_path = np.where(np.abs(pitch_deg) >= min_pitch_deg, sin_path, np.nan)
    speed_slow = np.asarray(np.abs(W_slow) / sin_path, dtype=np.float64)

    # Cross-check against the EM flowmeter when the platform carries one:
    # flight and U_EM measure the same along-axis speed, so a large
    # systematic disagreement means a bad aoa, a mis-picked pitch axis, or
    # a mis-calibrated U_EM — and epsilon (~U^-4) inherits it either way.
    # A fast-rate U_EM (hotel channel with fast: true) fails the length
    # match and intentionally skips the check.
    u_em = pf.channels.get("U_EM")
    if u_em is not None and len(u_em) == len(speed_slow):
        u_em = np.abs(np.asarray(u_em, dtype=np.float64))
        both = np.isfinite(speed_slow) & np.isfinite(u_em) & (u_em > 0.01)
        if both.sum() >= 10:
            ratio = float(np.median(speed_slow[both] / u_em[both]))
            if not 0.8 <= ratio <= 1.25:
                warnings.warn(
                    f"flight-model speed disagrees with U_EM by a median "
                    f"factor {ratio:.2f} (outside [0.8, 1.25]); check "
                    f"aoa_deg, the pitch-axis pick, and the U_EM "
                    f"calibration — epsilon scales as ~U^-4",
                    stacklevel=3,
                )
    return speed_slow
