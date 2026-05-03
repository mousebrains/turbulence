# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Top trimming — drop initial instabilities from a profile.

Caller supplies any number of fast-rate channels (shear, acceleration,
heading-rate, …) as a dict. The algorithm bins each by depth, computes
per-bin std, and reports the deepest depth at which all channels have
settled below their own quantile threshold. The instrument-specific
question of which channels to feed in lives in the caller.

Reference: Code/trim_top_profiles.m (85 lines)
"""

import numpy as np
import numpy.typing as npt


def _bin_std(values: np.ndarray, depth: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Compute standard deviation per depth bin."""
    n_bins = len(bin_edges) - 1
    result = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (depth >= bin_edges[i]) & (depth < bin_edges[i + 1])
        v = values[mask]
        if len(v) > 1:
            result[i] = np.nanstd(v)
    return result


def compute_trim_depth(
    depth_fast: npt.ArrayLike,
    channels: dict[str, np.ndarray],
    *,
    dz: float = 0.5,
    min_depth: float = 1.0,
    max_depth: float = 50.0,
    quantile: float = 0.6,
) -> float | None:
    """Compute the trim depth for a single profile.

    Parameters
    ----------
    depth_fast : array_like
        Depth (positive downward, fast rate) [m].
    channels : dict
        Fast-rate channel data, name -> 1-D array. Each channel must
        match ``depth_fast`` in length. Any vibration / motion proxy
        works (shear, acceleration, gyro rate, …).
    dz : float
        Depth bin size [m].
    min_depth : float
        Minimum search depth [m].
    max_depth : float
        Maximum search depth [m].
    quantile : float
        Quantile threshold for detecting stable profiling (0-1).

    Returns
    -------
    float or None
        Trim depth [m], or None if no trim point found.
    """
    depth = np.asarray(depth_fast, dtype=np.float64)
    bin_edges = np.arange(min_depth - dz / 2, max_depth + dz, dz)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    if len(bin_edges) < 2:
        return None

    # Compute std per bin for each channel
    all_stds = []
    for name, data in channels.items():
        data = np.asarray(data, dtype=np.float64)
        if len(data) != len(depth):
            continue
        std_per_bin = _bin_std(data, depth, bin_edges)
        all_stds.append(std_per_bin)

    if not all_stds:
        return None

    # Find first depth where ALL channels drop below their quantile threshold
    trim_depths = []
    for std_arr in all_stds:
        valid = np.isfinite(std_arr)
        if np.sum(valid) < 3:
            continue
        threshold = np.nanquantile(std_arr[valid], quantile)
        # Find first bin (from top) where std drops below threshold
        for i in range(len(std_arr)):
            if np.isfinite(std_arr[i]) and std_arr[i] < threshold:
                trim_depths.append(bin_centers[i])
                break

    if not trim_depths:
        return None

    # Use the deepest trim depth across all channels (most conservative)
    return float(np.max(trim_depths))


def compute_trim_depths(
    profiles_data: list[dict],
    **params,
) -> list[float | None]:
    """Compute trim depths for multiple profiles.

    Parameters
    ----------
    profiles_data : list of dict
        Each dict has keys 'depth_fast' and 'channels' (dict of arrays).
    **params
        Keyword arguments passed to :func:`compute_trim_depth`.

    Returns
    -------
    list of (float or None)
        Per-profile trim depths.
    """
    return [compute_trim_depth(pd["depth_fast"], pd["channels"], **params) for pd in profiles_data]
