"""Peak-finding profile detection (Jesse's algorithm).

Uses scipy.signal.find_peaks on smoothed pressure to identify
up-cast and down-cast segments.  Does not modify existing
rsi/profile.py or scor160/profile.py.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def find_profiles_peaks(
    pressure: np.ndarray,
    fs: float,
    *,
    direction: str = "down",
    min_pressure: float = 0.0,
    peaks_height: float = 25.0,
    peaks_distance: int = 200,
    peaks_prominence: float = 25.0,
    min_speed: float = 0.2,
    smoothing_tau: float = 2.0,
) -> list[tuple[int, int]]:
    """Find profiles using scipy peak-finding on smoothed pressure.

    Parameters
    ----------
    pressure : ndarray
        Pressure time series [dbar] at slow rate.
    fs : float
        Sampling rate of *pressure* [Hz].
    direction : str
        'down', 'up', or 'both'.
    min_pressure : float
        Minimum pressure for valid profile segments [dbar].
    peaks_height : float
        Minimum peak height for ``find_peaks`` [dbar].
    peaks_distance : int
        Minimum distance between peaks [samples].
    peaks_prominence : float
        Minimum peak prominence [dbar].
    min_speed : float
        Minimum vertical speed to include a sample [m/s].
    smoothing_tau : float
        Time constant for Butterworth low-pass smoothing [s].

    Returns
    -------
    list of (start_idx, end_idx)
        Start and end indices (inclusive) in slow-rate samples.
    """
    pressure = np.asarray(pressure, dtype=np.float64)
    if len(pressure) < 10:
        return []

    # Low-pass smooth pressure
    f_c = 0.68 / max(smoothing_tau, 0.1)
    nyquist = fs / 2.0
    if f_c >= nyquist:
        f_c = nyquist * 0.9
    b, a = butter(2, f_c / nyquist)
    p_smooth = filtfilt(b, a, pressure)

    # Find pressure maxima (bottom of downcasts)
    maxima, _ = find_peaks(
        p_smooth,
        height=peaks_height,
        distance=peaks_distance,
        prominence=peaks_prominence,
    )

    # Find pressure minima (top of casts) — peaks of inverted pressure
    minima, _ = find_peaks(
        -p_smooth,
        distance=peaks_distance,
    )

    if len(maxima) == 0:
        return []

    # Build sorted array of all extrema with type labels
    events: list[tuple[int, str]] = []
    for idx in maxima:
        events.append((int(idx), "max"))
    for idx in minima:
        events.append((int(idx), "min"))
    events.sort(key=lambda x: x[0])

    profiles: list[tuple[int, int]] = []

    if direction in ("down", "both"):
        # Downcasts: from min to max
        for i, (idx, typ) in enumerate(events):
            if typ != "max":
                continue
            # Find preceding minimum
            prev_min = 0
            for j in range(i - 1, -1, -1):
                if events[j][1] == "min":
                    prev_min = events[j][0]
                    break
            if p_smooth[idx] - p_smooth[prev_min] >= min_pressure:
                profiles.append((prev_min, idx))

    if direction in ("up", "both"):
        # Upcasts: from max to min
        for i, (idx, typ) in enumerate(events):
            if typ != "max":
                continue
            # Find following minimum
            next_min = len(pressure) - 1
            for j in range(i + 1, len(events)):
                if events[j][1] == "min":
                    next_min = events[j][0]
                    break
            if p_smooth[idx] - p_smooth[next_min] >= min_pressure:
                profiles.append((idx, next_min))

    # Sort by start index
    profiles.sort(key=lambda x: x[0])

    return profiles
