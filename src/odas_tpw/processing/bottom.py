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
        Fast sampling rate [Hz].
    depth_window : float
        Depth window for variance computation [m].
    depth_minimum : float
        Minimum depth to start searching [m].
    speed_factor : float
        Speed reduction factor for crash identification.
    median_factor : float
        Acceleration std dev filter factor.
    vibration_frequency : int
        Frequency for vibration binning [Hz].
    vibration_factor : float
        Vibration std dev acceptance factor.

    Returns
    -------
    float or None
        Bottom depth [m], or None if no crash detected.
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

    max_depth = np.nanmax(depth)
    if max_depth < depth_minimum:
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
            bottom_depth = bins[i]
            return float(bottom_depth)

    return None
