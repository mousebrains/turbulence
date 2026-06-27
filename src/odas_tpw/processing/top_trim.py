# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Top trimming — drop initial instabilities from a profile.

Caller supplies any number of fast-rate channels (shear, acceleration,
heading-rate, …) as a dict. The algorithm bins each by depth, computes
per-bin std, and reports the depth below which every channel's variance
has settled to its background level. The settling depth is located as
the first bin beneath the *deepest* still-elevated (prop-wash) bin, so a
momentarily quiet near-surface bin cannot end the search prematurely and
leave noisy data in the profile. The instrument-specific question of
which channels to feed in lives in the caller.

Reference: Code/trim_top_profiles.m (85 lines)
"""

import numpy as np
import numpy.typing as npt


def _bin_std(values: np.ndarray, depth: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Compute standard deviation per depth bin (NaN-aware, ddof=0)."""
    n_bins = len(bin_edges) - 1
    # Bin assignment via searchsorted gives the same [edges[i], edges[i+1])
    # half-open bins as the original mask comparison.
    idx = np.searchsorted(bin_edges, depth, side="right") - 1
    in_range = (idx >= 0) & (idx < n_bins) & np.isfinite(values)
    idx_v = idx[in_range]
    vals_v = values[in_range]
    counts = np.bincount(idx_v, minlength=n_bins)
    sums = np.bincount(idx_v, weights=vals_v, minlength=n_bins)
    sums_sq = np.bincount(idx_v, weights=vals_v * vals_v, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
        var = sums_sq / np.maximum(counts, 1) - means * means
        var = np.maximum(var, 0.0)
        result = np.where(counts > 1, np.sqrt(var), np.nan)
    return result


def compute_trim_depth(
    depth_fast: npt.ArrayLike,
    channels: dict[str, np.ndarray],
    *,
    dz: float = 0.5,
    min_depth: float = 1.0,
    max_depth: float = 50.0,
    quantile: float = 0.25,
    noise_factor: float = 2.0,
) -> float | None:
    """Compute the trim depth for a single profile.

    For each channel the per-bin standard deviation is compared against a
    *settled background* level — a low ``quantile`` of that channel's
    per-bin std, which estimates the quiet floor. Taking a low quantile
    (rather than a central one) keeps the estimate inside the quiet
    population even when the prop wash inflates a large fraction of the
    bins: the estimate holds while the prop wash spans less than
    ``1 - quantile`` of the binned range, so a *lower* ``quantile``
    tolerates *deeper* prop wash. A bin is treated as still inside the prop
    wash when its std exceeds ``noise_factor`` times that background. The
    channel's prop-wash exit is the first bin *below the deepest
    still-elevated bin*: scanning for the deepest elevated bin (rather than
    the first quiet one) prevents a momentarily quiet near-surface bin from
    ending the search early. The profile trim depth is the deepest such
    exit across all channels — the depth below which every channel has
    settled.

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
        Low quantile of per-bin std taken as the settled quiet-floor
        level; must be in ``(0, 1)``. Trimming is robust while the prop
        wash spans less than ``1 - quantile`` of the search range.
    noise_factor : float
        A bin counts as still in the prop wash when its std exceeds
        ``noise_factor`` times the settled background. Must be > 1 so the
        background's own bin-to-bin scatter is not mistaken for prop wash.

    Returns
    -------
    float or None
        Trim depth [m], or None if no trim point found.

    Raises
    ------
    ValueError
        If ``quantile`` is not in ``(0, 1)`` or ``noise_factor`` is not
        greater than 1.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    if not noise_factor > 1.0:
        raise ValueError(f"noise_factor must be > 1, got {noise_factor}")
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

    # Per channel, locate the prop-wash exit: the bin just below the deepest
    # bin whose std is still elevated above the settled background. Using the
    # deepest elevated bin (not the first quiet one) prevents a momentarily
    # quiet near-surface bin from ending the search early (audit #66).
    trim_depths = []
    for std_arr in all_stds:
        valid = np.isfinite(std_arr)
        if np.sum(valid) < 3:
            continue
        background = np.nanquantile(std_arr[valid], quantile)
        elevated = valid & (std_arr > noise_factor * background)
        elevated_idx = np.flatnonzero(elevated)
        if elevated_idx.size == 0:
            # No bin is elevated: this channel never left its background level
            # within the search range, so it implies a minimal trim (the
            # shallowest resolved bin).
            trim_depths.append(bin_centers[np.flatnonzero(valid)[0]])
            continue
        deepest = int(elevated_idx[-1])
        # First bin below the deepest elevated bin. If the deepest search bin is
        # itself still elevated the profile never settled within range; fall
        # back to that bin (the most conservative, deepest trim).
        exit_idx = min(deepest + 1, len(bin_centers) - 1)
        trim_depths.append(bin_centers[exit_idx])

    if not trim_depths:
        return None

    # Combine channels: the profile is settled only below the deepest
    # per-channel prop-wash exit (most conservative).
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
