# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Bottom-crash detection — find seafloor impact from a vibration signal.

Caller supplies any number of fast-rate vibration channels as a dict
(e.g. ``{"Ax": …, "Ay": …}`` for two-axis accel, or just
``{"vibration": …}`` for a pre-computed RMS). The algorithm forms the
elementwise root-sum-square magnitude across all supplied channels, bins
its std by depth, and flags the deepest bin whose std exceeds
``vibration_factor`` times the median.

Reference: Code/bottom_crash_profiles.m (currently a stub in Matlab)
"""

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt


def detect_bottom_crash(
    depth_fast: npt.ArrayLike,
    vibration_channels: Mapping[str, npt.ArrayLike],
    fs: float,
    *,
    depth_window: float = 4.0,
    depth_minimum: float = 10.0,
    speed_factor: float = 0.3,
    median_factor: float = 1.0,
    vibration_frequency: int = 16,
    vibration_factor: float = 4.0,
) -> float | None:
    """Detect bottom crash from a set of vibration channels.

    Parameters
    ----------
    depth_fast : array_like
        Depth array (positive downward, fast rate) [m].
    vibration_channels : mapping of name -> 1-D array
        Fast-rate vibration / motion channels. Two-axis accel
        ``{"Ax": …, "Ay": …}`` reproduces the historical 2-axis
        magnitude. A single pre-aggregated channel
        (e.g. ``{"vibration_rms": rms}``) is also fine. All arrays
        must match ``depth_fast`` in length.
    fs : float
        Fast sampling rate [Hz].  Currently UNUSED — accepted for
        backward compatibility; the algorithm operates on depth bins,
        not time.
    depth_window : float
        Depth window for variance computation [m].
    depth_minimum : float
        Minimum depth to start searching [m].
    speed_factor : float
        Currently UNUSED — accepted for backward compatibility (the
        speed-drop confirmation from the original MATLAB design is not
        implemented).  Tuning it has no effect.
    median_factor : float
        Currently UNUSED — accepted for backward compatibility.
        Tuning it has no effect.
    vibration_frequency : int
        Currently UNUSED — accepted for backward compatibility.
        Tuning it has no effect.
    vibration_factor : float
        Vibration std dev acceptance factor: the deepest bin whose
        magnitude std exceeds ``vibration_factor`` times the median
        marks the crash.

    Returns
    -------
    float or None
        Bottom depth [m], or None if no crash detected.

    Notes
    -----
    The reported depth is the *mean* depth of the samples in the flagged
    ``depth_window``-wide bin. When the crash onset falls early in that bin the
    mean lies below the onset, so up to roughly half a bin width (~``depth_window
    / 2``) of contaminated data can remain *above* the reported depth and be left
    un-trimmed. This is a fundamental limit of the bin resolution; size
    ``depth_window`` accordingly if tighter bottom trimming is required.
    """
    depth = np.asarray(depth_fast, dtype=np.float64)

    if not vibration_channels:
        return None

    # Elementwise root-sum-square across all supplied channels. For two-axis
    # ``{Ax, Ay}`` this matches the previous ``sqrt(Ax**2 + Ay**2)`` exactly.
    mag_sq: np.ndarray | None = None
    for arr in vibration_channels.values():
        a = np.asarray(arr, dtype=np.float64)
        if len(a) != len(depth):
            return None
        sq = a * a
        mag_sq = sq if mag_sq is None else mag_sq + sq
    assert mag_sq is not None  # unreachable: guarded by `if not vibration_channels`
    accel_mag = np.sqrt(mag_sq)

    # An all-NaN depth segment makes nanmax NaN; np.arange(..., NaN, ...) below
    # then raises. Bail out cleanly instead of crashing the profile.
    #
    # Audit fix: short-circuit BEFORE nanmax on an all-NaN depth. np.errstate
    # only governs IEEE FP flags (invalid/divide/...), NOT numpy's Python-level
    # "All-NaN slice encountered" RuntimeWarning — so the old errstate guard let
    # that warning escape (spurious log noise; promotable to an error by callers
    # that filter warnings). Guarding on np.isfinite up front avoids the warning
    # and the NaN max_depth entirely.
    if not np.any(np.isfinite(depth)):
        return None
    max_depth = np.nanmax(depth)
    if not np.isfinite(max_depth) or max_depth < depth_minimum:
        return None

    # Bin acceleration variance by depth
    bin_size = depth_window
    bins = np.arange(depth_minimum, max_depth + bin_size, bin_size)
    if len(bins) < 2:
        return None

    n_bins = len(bins) - 1
    idx = np.searchsorted(bins, depth, side="right") - 1
    in_range = (idx >= 0) & (idx < n_bins) & np.isfinite(accel_mag)
    idx_v = idx[in_range]
    vals_v = accel_mag[in_range]
    counts = np.bincount(idx_v, minlength=n_bins)
    sums = np.bincount(idx_v, weights=vals_v, minlength=n_bins)
    sums_sq = np.bincount(idx_v, weights=vals_v * vals_v, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
        var = sums_sq / np.maximum(counts, 1) - means * means
        var = np.maximum(var, 0.0)
        bin_std = np.where(counts > 1, np.sqrt(var), np.nan)

    valid = np.isfinite(bin_std)
    if np.sum(valid) < 3:
        return None

    med_std = np.nanmedian(bin_std[valid])
    threshold = med_std * vibration_factor

    # Search from deepest bin upward for spike
    for i in range(len(bin_std) - 1, -1, -1):
        if np.isfinite(bin_std[i]) and bin_std[i] > threshold:
            # Report the MEAN depth of the samples that actually fell in the
            # flagged bin, not the bin's geometric center. The deepest bin's
            # right edge overhangs nanmax(depth) by up to one bin width, so its
            # center can lie BELOW the deepest real sample; the caller's
            # `P >= bottom_depth` then matches nothing and the crash is silently
            # left un-trimmed (~half of all detections). The sample mean is
            # guaranteed to lie within the observed depths of the bin (and so is
            # robust to the fast/slow rate mismatch between this depth series and
            # the slow pressure the caller trims against).
            sel_depth = depth[in_range & (idx == i)]
            return float(np.nanmean(sel_depth))

    return None
