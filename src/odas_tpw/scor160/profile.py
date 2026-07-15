# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Profile detection for vertical profilers.

General-purpose algorithms for detecting profiling segments in pressure
time series.  Not instrument-specific — works with any pressure/fall-rate
data.

Port of get_profile.m from the ODAS MATLAB library.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, filtfilt


def _safe_filtfilt(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """filtfilt that no-ops on inputs too short for its default padlen.

    A 1st-order Butterworth gives padlen = 3*max(len(a), len(b)) = 6, so
    filtfilt raises ``ValueError`` for len(x) <= 6. For such short series there
    is nothing meaningful to zero-phase smooth, so return the input unchanged
    rather than crashing the whole profile/file.
    """
    x = np.asarray(x)
    padlen = 3 * max(len(a), len(b))
    if x.shape[0] <= padlen:
        return x
    return np.asarray(filtfilt(b, a, x))


def smooth_fall_rate(P: np.ndarray, fs: float, tau: float = 1.5) -> np.ndarray:
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
    P = P.astype(np.float64)
    if P.shape[0] < 2:
        # np.gradient needs >= 2 samples; a single pressure sample has no rate.
        return np.zeros_like(P)
    W = np.gradient(P, 1.0 / fs)
    f_c = 0.68 / tau
    b, a = butter(1, f_c / (fs / 2.0))
    return _safe_filtfilt(b, a, W)


def compute_speed_fast(
    P_slow: np.ndarray,
    t_fast: np.ndarray,
    t_slow: np.ndarray,
    fs_fast: float,
    fs_slow: float,
    tau: float = 1.5,
    speed_min: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute profiling speed from pressure, interpolated to fast rate.

    Steps:
      1. W = gradient(P_slow) zero-phase filtered with Butterworth at
         0.68/tau (= ODAS odas_p2mat.m fall-rate smoothing)
      2. speed = abs(W), interpolated to the fast time grid
      3. Second zero-phase smoothing pass at 0.68/tau on the fast grid
      4. Clamped to speed_min

    Note: ODAS computes the fast-rate speed directly from gradient(P_fast)
    in a single smoothing pass (odas_p2mat.m:699-707); the slow-grid
    interpolation plus second pass used here gives a slightly smoother
    speed but is not identical.

    Parameters
    ----------
    P_slow : ndarray
        Pressure [dbar] at slow rate.
    t_fast, t_slow : ndarray
        Time vectors for fast and slow rates.
    fs_fast, fs_slow : float
        Sampling rates [Hz].
    tau : float
        Smoothing time constant [s].
    speed_min : float
        Minimum profiling speed [m/s].

    Returns
    -------
    speed_fast : ndarray
        Profiling speed at fast rate [dbar/s, numerically treated as m/s].
    W_slow : ndarray
        Smoothed fall rate at slow rate [dbar/s].
    """
    W_slow = smooth_fall_rate(P_slow, fs_slow, tau=tau)
    speed_fast = np.interp(t_fast, t_slow, np.abs(W_slow))

    f_c = 0.68 / tau
    b_f, a_f = butter(1, f_c / (fs_fast / 2.0))
    speed_fast = _safe_filtfilt(b_f, a_f, speed_fast)

    speed_fast = np.maximum(speed_fast, speed_min)

    return speed_fast, W_slow


def smooth_speed_interp(
    speed_slow: np.ndarray,
    t_fast: np.ndarray,
    t_slow: np.ndarray,
    fs_fast: float,
    tau: float,
    speed_min: float = 0.05,
) -> np.ndarray:
    """Interpolate and smooth profiling speed from slow to fast rate.

    Applies a first-order Butterworth low-pass filter at cutoff
    ``0.68 / tau`` to the interpolated speed, then clamps to speed_min.

    Parameters
    ----------
    speed_slow : ndarray
        Absolute profiling speed at slow rate [m/s].
    t_fast, t_slow : ndarray
        Time vectors for fast and slow rates.
    fs_fast : float
        Fast sampling rate [Hz].
    tau : float
        Smoothing time constant [s].
    speed_min : float
        Minimum profiling speed [m/s].

    Returns
    -------
    speed_fast : ndarray
        Smoothed profiling speed at fast rate [m/s].
    """
    pspd_rel = np.interp(t_fast, t_slow, speed_slow)
    f_c = 0.68 / tau
    b_f, a_f = butter(1, f_c / (fs_fast / 2.0))
    pspd_rel = _safe_filtfilt(b_f, a_f, pspd_rel)
    return np.asarray(np.maximum(pspd_rel, speed_min))


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
        ``'down'``, ``'up'``, ``'glide'`` (both up and down), or
        ``'horizontal'`` (either sign, for towed/AUV instruments).
    min_duration : float
        Minimum profile duration [s].

    Returns
    -------
    list of (int, int)
        Start and end indices (inclusive) of each detected profile.
    """
    d = direction.lower()

    if d == "glide":
        # Detect both up and down segments, merge and sort by start index
        down = get_profiles(P, W, fs, P_min, W_min, "down", min_duration)
        up = get_profiles(P, W, fs, P_min, W_min, "up", min_duration)
        merged = sorted(down + up, key=lambda t: t[0])
        return merged

    P = np.asarray(P, dtype=np.float64).ravel()
    W = np.asarray(W, dtype=np.float64).ravel()
    min_samples = int(min_duration * fs)

    if d == "up":
        W = -W
    elif d == "horizontal":
        W = np.abs(W)

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


def explain_no_profiles(
    P: npt.ArrayLike,
    W: npt.ArrayLike,
    P_min: float = 0.5,
    W_min: float = 0.3,
    direction: str = "down",
) -> str:
    """Build an actionable message explaining why :func:`get_profiles` found none.

    Reports the observed pressure span and peak fall/rise rate against the
    ``P_min`` / ``W_min`` thresholds and, when a threshold is the binding
    constraint, suggests the flag to relax. A common cause is a slow or
    glider-style cast whose fall rate never reaches the VMP-tuned ``W_min``
    default (0.3 dbar/s).
    """
    P = np.asarray(P, dtype=np.float64).ravel()
    W = np.asarray(W, dtype=np.float64).ravel()
    d = direction.lower()
    if d == "up":
        rate, sense = -W, "upward"
    elif d in ("glide", "horizontal"):
        rate, sense = np.abs(W), "vertical"
    else:
        rate, sense = W, "downward"

    # Reduce over finite values only: an all-NaN pressure (e.g. a bad pressure
    # coefficient) is itself the likely cause and must neither crash the reduce
    # nor be silently ignored — nanmax on an all-NaN slice returns NaN and warns.
    finite_P = P[np.isfinite(P)]
    finite_rate = rate[np.isfinite(rate)]
    peak = float(finite_rate.max()) if finite_rate.size else float("nan")
    p_lo = float(finite_P.min()) if finite_P.size else float("nan")
    p_hi = float(finite_P.max()) if finite_P.size else float("nan")

    parts = [
        f"No profiles detected (direction={d}, W_min={W_min:g} dbar/s, P_min={P_min:g} dbar). "
        f"Observed pressure {p_lo:.3g}-{p_hi:.3g} dbar, peak {sense} rate {peak:.3g} dbar/s."
    ]
    if not np.isfinite(p_hi) or not np.isfinite(peak):
        parts.append(
            "Pressure or fall rate is non-finite (NaN/inf) — check the pressure "
            "calibration; a bad coefficient can blank out or explode the depth."
        )
    elif p_hi <= P_min:
        parts.append(
            "Pressure never exceeds P_min — check the pressure calibration "
            "(a bad coefficient can flatten or explode the depth)."
        )
    elif peak < W_min:
        suggest = max(round(peak * 0.5, 3), 0.01)
        hint = (
            f"The {sense} rate never reaches W_min. For a slow or glider-style cast, "
            f"lower it (e.g. --W-min {suggest:g})"
        )
        # Only nudge toward glide when a single-sense search might be missing the
        # other half of an up/down cast.
        hint += (
            " and/or use --direction glide to catch both down and up."
            if d in ("down", "up")
            else "."
        )
        parts.append(hint)
    else:
        # Each threshold is individually cleared but no contiguous run satisfies
        # both at once for long enough (e.g. fast while shallow, then slow at
        # depth). Don't assert the thresholds "were cleared" — state the joint
        # condition honestly.
        parts.append(
            f"No run of samples satisfied P > {P_min:g} dbar and {sense} rate "
            f">= {W_min:g} dbar/s together for long enough; try --direction glide, "
            "a lower --W-min, or a shorter min-duration."
        )
    return " ".join(parts)
