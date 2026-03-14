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
