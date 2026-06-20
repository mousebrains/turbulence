# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""CT sensor alignment via cross-correlation.

Aligns conductivity to temperature by finding the lag of maximum
cross-correlation between diff(T) and diff(C), using a weighted median
across profiles.

Reference: Code/CT_align.m (70 lines)
"""

import numpy as np
from scipy.signal import butter, correlate, lfilter


def shift_edge_hold(x: np.ndarray, shift: int) -> np.ndarray:
    """Shift *x* by *shift* samples, holding the edge value.

    Unlike ``np.roll``, samples shifted past the array ends are filled
    with the first/last value instead of wrapping around — wrapping
    would splice one end of the record into the other.
    """
    if shift == 0:
        return x.copy()
    out = np.empty_like(x)
    if shift > 0:
        out[:shift] = x[0]
        out[shift:] = x[:-shift]
    else:
        out[shift:] = x[-1]
        out[:shift] = x[-shift:]
    return out


def ct_align(
    T: np.ndarray,
    C: np.ndarray,
    fs: float,
    profiles: list[tuple[int, int]],
    max_lag_seconds: float = 5.0,
) -> tuple[np.ndarray, float]:
    """Align conductivity to temperature by cross-correlation.

    Parameters
    ----------
    T : ndarray
        Temperature array (slow rate).
    C : ndarray
        Conductivity array (slow rate).
    fs : float
        Sampling rate [Hz].
    profiles : list of (start, end) tuples
        Profile indices into slow-rate arrays.
    max_lag_seconds : float
        Maximum lag to search [s].

    Returns
    -------
    C_aligned : ndarray
        Shifted conductivity array.
    lag_seconds : float
        Applied lag in seconds.
    """
    if not profiles:
        return C.copy(), 0.0

    max_lag_samples = round(max_lag_seconds * fs)
    fc = min(4.0, fs / 2.0 * 0.99)
    bb, aa = butter(2, fc / (fs / 2.0))

    per_profile = []
    for s, e in profiles:
        seg_T = T[s : e + 1]
        seg_C = C[s : e + 1]
        if len(seg_T) < 10:
            continue

        dx = lfilter(bb, aa, np.diff(seg_T) - np.mean(np.diff(seg_T)))
        dy = lfilter(bb, aa, np.diff(seg_C) - np.mean(np.diff(seg_C)))

        corr = correlate(dx, dy, mode="full")
        norm = np.sqrt(np.sum(dx**2) * np.sum(dy**2))
        if norm <= 0:
            # A flatlined/constant T or C segment (norm == 0) carries no
            # alignment information. Skip it: leaving it in would make
            # argmax(|corr|) of an all-zero correlation pick index 0, i.e. the
            # most-negative lag (-max_lag_seconds), a maximally-wrong estimate.
            continue
        corr = corr / norm

        n = len(dx)
        lags = np.arange(-(n - 1), n)

        mask = np.abs(lags) <= max_lag_samples
        corr = corr[mask]
        lags = lags[mask]

        idx = np.argmax(np.abs(corr))
        lag = lags[idx]
        max_corr = np.abs(corr[idx])
        n_samples = e - s + 1

        per_profile.append(
            {
                "lag": lag / fs,
                "max_corr": max_corr,
                "n_samples": n_samples,
            }
        )

    if not per_profile:
        return C.copy(), 0.0

    # Weighted median: smallest lag whose cumulative weight reaches half the
    # total. searchsorted gives the true (lower) weighted median; the old
    # argmin(|cum_w - total/2|) picked the entry *closest* to the midpoint,
    # which is biased toward the pre-crossing entry for skewed weights.
    per_profile.sort(key=lambda x: x["lag"])
    weights = np.array([p["max_corr"] * p["n_samples"] for p in per_profile])
    cum_w = np.cumsum(weights)
    total = float(cum_w[-1])
    if total <= 0:
        # No usable correlation across any profile -> apply no shift.
        return C.copy(), 0.0
    mid_idx = int(np.searchsorted(cum_w, total / 2.0))
    mid_idx = min(mid_idx, len(per_profile) - 1)
    median_lag = per_profile[mid_idx]["lag"]

    i_shift = round(median_lag * fs)
    # Edge-hold instead of np.roll: wrapping would splice the start of
    # the record into the end (ODAS trims instead).
    C_aligned = shift_edge_hold(C, i_shift)

    return C_aligned, median_lag
