# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Top trimming — remove initial instabilities from profiles.

Bins shear, acceleration, and speed by depth, computes std per bin,
and finds the depth where variance drops below a quantile threshold.
This identifies where the VMP exits propeller wash and enters stable
profiling.

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
        Fast-rate channel data. Should include some of: sh1, sh2, Ax, Ay, W_fast.
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
    return [
        compute_trim_depth(pd["depth_fast"], pd["channels"], **params)
        for pd in profiles_data
    ]
