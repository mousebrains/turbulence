# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Top trimming — drop initial instabilities from a profile.

Caller supplies one or more fast-rate motion proxies as a dict. The
algorithm bins each by depth, computes per-bin std, and reports the depth
below which the instrument's motion has settled. The settling depth is
located as the first bin beneath the *surface-attached* elevated
(prop-wash) run — the run of elevated bins anchored to the top of the
search range, bridging quiet lulls up to ``max_gap`` bins. Bridging lulls
keeps a momentarily quiet near-surface bin from ending the search early;
anchoring to the surface keeps an isolated *deep* transient from
over-trimming the quiet band above it. Channels are combined with the
median (fully robust to one bad channel only for three or more voters; the
VMP caller feeds two accelerometers, where median == mean).

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


def _surface_run_end(elevated: np.ndarray, max_gap: int) -> int | None:
    """Index of the deepest bin of the surface-attached elevated run.

    Prop wash is a *surface-attached* transient: a run of elevated bins that
    begins at (or within ``max_gap`` bins of) the top of the search range and
    extends downward. Quiet lulls of up to ``max_gap`` bins inside the run are
    bridged so a momentarily quiet near-surface bin does not end the search
    early (audit #66). A wider quiet band separates the surface wash from any
    deeper *isolated* elevated bin (a cable snap-load or mid-column turbulence
    patch), which is contamination — not prop wash — and must not drive the
    trim (audit r1-2).

    ``elevated`` is a boolean array over a channel's valid bins, shallow→deep.
    Returns the index (into ``elevated``) of the deepest bin in the
    surface-attached run, or None when there is no surface-attached prop wash
    (no elevated bins, or the only elevated bins are detached from the
    surface).
    """
    idx = np.flatnonzero(elevated)
    if idx.size == 0:
        return None
    first = int(idx[0])
    if first > max_gap:
        # Elevated bins exist, but none within max_gap of the surface: the
        # descent was already settled at the top, so these are detached
        # mid-column transients, not prop wash.
        return None
    last = first
    gap = 0
    for pos in range(first + 1, len(elevated)):
        if elevated[pos]:
            last = pos
            gap = 0
        else:
            gap += 1
            if gap > max_gap:
                break
    return last


def compute_trim_depth(
    depth_fast: npt.ArrayLike,
    channels: dict[str, np.ndarray],
    *,
    dz: float = 0.5,
    min_depth: float = 1.0,
    max_depth: float = 50.0,
    quantile: float = 0.6,
    noise_factor: float = 2.0,
    max_gap: int = 3,
) -> float | None:
    """Compute the trim depth for a single profile.

    For each channel the per-bin standard deviation is compared against a
    *settled background* level — the ``quantile`` of that channel's per-bin
    std (a robust estimate of the quiet level, valid while the prop wash
    spans less than ``1 - quantile`` of the binned range). A bin is treated
    as still inside the prop wash when its std exceeds ``noise_factor``
    times that background. The channel's prop-wash exit is the first bin
    *below the surface-attached elevated run* — the run of elevated bins
    that starts within ``max_gap`` bins of the top of the search range and
    extends down, bridging quiet lulls of up to ``max_gap`` bins. Bridging
    lulls prevents a momentarily quiet near-surface bin from ending the
    search early (audit #66); anchoring the run to the surface prevents an
    isolated *deep* transient (a cable snap-load or mid-column turbulence
    patch) from over-trimming the entire quiet band above it (audit r1-2).
    A channel whose only elevated bins are detached from the surface sees
    no prop wash and abstains. The profile trim depth is the **median**
    exit across the channels that detected prop wash. Note the median is
    fully robust to one bad channel only for *three or more* voters; the
    production VMP caller feeds exactly two accelerometers (``Ax``/``Ay``),
    for which the median equals the mean, so a disagreeing channel pulls
    the trim halfway and a lone surface-detecting channel sets it outright.
    The surface-attachment rule above (not the median) is what rejects a
    single channel's spurious *deep* transient. A flat / zero-variance
    channel carries no settling information and is dropped before voting.

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
    max_gap : int
        Maximum run of quiet bins bridged within the surface-attached
        prop-wash run, and the maximum offset of the first elevated bin
        from the surface for the run to count as surface-attached [bins].
        Must be >= 0; ``>= 1`` preserves the audit-#66 momentary-lull
        tolerance. At the default ``dz=0.5`` m, ``max_gap=3`` bridges lulls
        up to ~1.5 m.

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
    if max_gap < 0:
        raise ValueError(f"max_gap must be >= 0, got {max_gap}")
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
    # bin of the surface-attached elevated run. Anchoring the run to the
    # surface (rather than taking the globally deepest elevated bin) bridges a
    # momentary near-surface lull (audit #66) while rejecting isolated deep
    # transients that would otherwise over-trim the whole quiet band above them
    # (audit r1-2).
    exits = []
    any_live = False
    for std_arr in all_stds:
        valid = np.isfinite(std_arr)
        if np.sum(valid) < 3:
            continue
        any_live = True
        valid_pos = np.flatnonzero(valid)
        background = np.nanquantile(std_arr[valid], quantile)
        elevated = std_arr[valid_pos] > noise_factor * background
        run_end = _surface_run_end(elevated, max_gap)
        if run_end is None:
            # No surface-attached prop wash (none, or only detached deep
            # contamination): abstain rather than voting a trim — a shallow
            # vote would pull the median up, a deep vote to a transient would
            # discard valid near-surface data.
            continue
        # Bin just below the deepest surface-run bin. If that is the deepest
        # search bin the profile never settled within range; fall back to it
        # (the most conservative, deepest trim).
        deepest = int(valid_pos[run_end])
        exit_idx = min(deepest + 1, len(bin_centers) - 1)
        exits.append(bin_centers[exit_idx])

    if exits:
        # Median exit across the channels that detected prop wash: robust to a
        # single misbehaving channel (a `max` lets one bad channel drag the
        # trim arbitrarily deep).
        trim = float(np.median(exits))
        # A never-settled cast (the surface run reaches the deepest populated
        # bin) yields an exit bin center in the empty tail of the search range,
        # deeper than every real sample; the caller's ``P >= trim_depth`` would
        # then be empty and apply NO trim, silently keeping the whole
        # wash-contaminated column. Clamp to the deepest observed in-range
        # sample so the trim is always applicable (the cast is then trimmed to
        # its bottom, i.e. flagged as all-wash, rather than passed untouched).
        in_range = depth[(depth >= min_depth) & (depth <= max_depth) & np.isfinite(depth)]
        if in_range.size:
            trim = min(trim, float(np.max(in_range)))
        return trim
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
