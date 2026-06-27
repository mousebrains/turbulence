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


def _repair_nonfinite(
    seg: np.ndarray,
    min_finite: int = 10,
    max_nonfinite_frac: float = 0.25,
) -> np.ndarray | None:
    """Linearly interpolate isolated non-finite samples in a profile segment.

    Returns a repaired copy, or ``None`` when too little finite data remains
    (fewer than *min_finite* finite samples, or the non-finite fraction exceeds
    *max_nonfinite_frac*) to carry useful alignment information. A fully-finite
    segment is returned unchanged.
    """
    finite = np.isfinite(seg)
    if finite.all():
        return seg
    n_finite = int(finite.sum())
    if n_finite < min_finite or (len(seg) - n_finite) > max_nonfinite_frac * len(seg):
        return None
    idx = np.arange(len(seg))
    repaired = seg.copy()
    # np.interp clamps to the edge finite values for leading/trailing gaps.
    repaired[~finite] = np.interp(idx[~finite], idx[finite], seg[finite])
    return repaired


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

        # Audit over-discard fix: convert_jac_c emits NaN for *isolated* corrupt
        # conductivity samples (v_part==0 bit errors), not whole-segment
        # corruption. The earlier guard skipped the ENTIRE profile on any single
        # non-finite sample, silently forfeiting that cast's alignment whenever
        # every profile carried one transient bad sample. Instead, linearly
        # interpolate the few non-finite points (so the IIR lfilter cannot smear
        # a NaN across the segment and drive a maximally-wrong lag), and only
        # skip when too little finite data remains to carry alignment info.
        rep_T = _repair_nonfinite(seg_T)
        rep_C = _repair_nonfinite(seg_C)
        if rep_T is None or rep_C is None:
            continue
        seg_T, seg_C = rep_T, rep_C

        dx = lfilter(bb, aa, np.diff(seg_T) - np.mean(np.diff(seg_T)))
        dy = lfilter(bb, aa, np.diff(seg_C) - np.mean(np.diff(seg_C)))

        corr = correlate(dx, dy, mode="full")
        norm = np.sqrt(np.sum(dx**2) * np.sum(dy**2))
        if not np.isfinite(norm) or norm <= 0:
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
    if not np.isfinite(total) or total <= 0:
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
