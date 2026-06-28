# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Top trimming — drop initial instabilities from a profile.

Caller supplies one or more fast-rate motion proxies as a dict. The
algorithm bins each by depth, computes per-bin std, and reports the depth
below which the instrument's motion has settled. The settling depth is
located as the first bin beneath the *deepest* still-elevated (prop-wash)
bin, so a momentarily quiet near-surface bin cannot end the search
prematurely; channels are combined with the median, robust to one bad
channel.

The instrument-specific question of which channels to feed lives in the
caller. They must reflect the *instrument's* state, not the ocean: on a
VMP the accelerometers settle at the true prop-wash exit, whereas shear
probes, inclinometers, and fall rate stay elevated through deep ocean
turbulence and would over-trim.

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
    quantile: float = 0.6,
    noise_factor: float = 2.0,
) -> float | None:
    """Compute the trim depth for a single profile.

    For each channel the per-bin standard deviation is compared against a
    *settled background* level — the ``quantile`` of that channel's per-bin
    std (a robust estimate of the quiet level, valid while the prop wash
    spans less than ``1 - quantile`` of the binned range). A bin is treated
    as still inside the prop wash when its std exceeds ``noise_factor``
    times that background. The channel's prop-wash exit is the first bin
    *below the deepest still-elevated bin*: scanning for the deepest
    elevated bin (rather than the first quiet one) prevents a momentarily
    quiet near-surface bin from ending the search early. The profile trim
    depth is the **median** exit across channels — robust to a single
    misbehaving or dead channel (a flat / zero-variance channel is dropped).

    The caller chooses which channels best mark the instrument's settling.
    On VMP data the accelerometers are the right choice: they capture the
    mechanical entry transient. Shear probes, inclinometers, and fall rate
    respond to the *ocean* turbulence the instrument falls through, so their
    per-bin std stays elevated at depth and would over-trim.

    Parameters
    ----------
    depth_fast : array_like
        Depth (positive downward, fast rate) [m].
    channels : dict
        Fast-rate channel data, name -> 1-D array. Each channel must
        match ``depth_fast`` in length. Use instrument-motion proxies that
        settle once the descent stabilizes (accelerometers); avoid channels
        driven by ocean turbulence (shear).
    dz : float
        Depth bin size [m].
    min_depth : float
        Minimum search depth [m].
    max_depth : float
        Maximum search depth [m].
    quantile : float
        Quantile of per-bin std taken as the settled background level;
        must be in ``(0, 1)``. Trimming is robust while the prop wash
        spans less than ``1 - quantile`` of the search range.
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

    # Compute std per bin for each channel that carries a real signal.
    all_stds = []
    for name, data in channels.items():
        data = np.asarray(data, dtype=np.float64)
        if len(data) != len(depth):
            continue
        finite = np.isfinite(data)
        if int(finite.sum()) < 2 or np.ptp(data[finite]) == 0:
            # All-NaN or constant: a dead / non-functional sensor carries no
            # settling information, so it must not vote (it would otherwise drag
            # the median toward a minimal trim).
            continue
        all_stds.append(_bin_std(data, depth, bin_edges))

    if not all_stds:
        return None

    # Per channel, locate the prop-wash exit: the bin just below the deepest
    # bin whose std is still elevated above the settled background. Using the
    # deepest elevated bin (not the first quiet one) prevents a momentarily
    # quiet near-surface bin from ending the search early (audit #66).
    exits = []
    any_live = False
    for std_arr in all_stds:
        valid = np.isfinite(std_arr)
        if np.sum(valid) < 3:
            continue
        any_live = True
        background = np.nanquantile(std_arr[valid], quantile)
        elevated = valid & (std_arr > noise_factor * background)
        elevated_idx = np.flatnonzero(elevated)
        if elevated_idx.size == 0:
            # This channel sees no prop wash; it abstains rather than voting a
            # shallow trim that would pull the median up against channels that
            # do detect one.
            continue
        deepest = int(elevated_idx[-1])
        # First bin below the deepest elevated bin. If the deepest search bin is
        # itself still elevated the profile never settled within range; fall
        # back to that bin (the most conservative, deepest trim).
        exit_idx = min(deepest + 1, len(bin_centers) - 1)
        exits.append(bin_centers[exit_idx])

    if exits:
        # Median exit across the channels that detected prop wash: robust to a
        # single misbehaving channel (a `max` lets one bad channel drag the
        # trim arbitrarily deep).
        return float(np.median(exits))
    if any_live:
        # Live channels but none saw prop wash: trim minimally (top of range).
        return float(bin_centers[0])
    return None


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
