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
caller, and it is not a free choice -- a channel that is *physically blind*
to the contamination cannot detect it, however well it behaves.

On a Rockland VMP, ``Ax``/``Ay`` are piezo VIBRATION sensors (for Goodman
coherent-noise removal), not accelerometers, and there is no ``Az``. They
measure the body ringing, which damps out within ~5 m, so they are blind to
prop wash -- an advected FLOW perturbation that persists far deeper. Measured
on ASTRAL 2023 (4 deployments, ratio to the 40-60 m background): Ax/Ay std
peaks at 1.44x and is flat below 5 m, while the high-passed FALL-RATE
RESIDUAL peaks at 44x and decays over ~30 m. Feeding only Ax/Ay trims to a
median of 3 dbar and leaves two decades of prop-wash epsilon in the product.

The fall-rate residual is the right proxy for prop wash. Note this is the
*residual* about a smooth descent, not the raw fall rate: raw fall rate does
stay elevated at depth through ocean turbulence (which is why it was
previously excluded), but the high-passed residual settles to a flat
background.

THE SMOOTHING WINDOW IS NOT A FREE PARAMETER. The residual is
``w - movmean(w, window)``, so the window sets the high-pass corner and is
only meaningful *relative to the cast length*. Once it is a large fraction of
the cast, the "smooth descent" baseline absorbs the wash and the residual
loses the signal. Measured on SUNRISE (18-24 m casts, ~20 s) the peak residual
ratio falls from 6.1x at a 2 m window to 2.3x at 6 m, and two of five
vessel-years read as having *no prop wash at all* at the 6 m default -- while
at 2 m the wash is plainly there in every one. Roughly 10% of the cast span
worked on both ASTRAL (long casts) and SUNRISE (short). A peak ratio near 1.0
is evidence about the window until the short-window case has been checked; the
caller warns when the configured window exceeds 30% of the cast span.

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


def inclinometer_is_usable(
    incl_segment: npt.ArrayLike,
    *,
    min_range_deg: float = 0.5,
    valid_range: tuple[float, float] = (-95.0, 95.0),
    min_finite_fraction: float = 0.5,
) -> bool:
    """Is this inclinometer reporting *during this cast*? Check PER PROFILE.

    A dead channel is not a harmless abstention for
    :func:`attitude_trim_depth`. **A channel frozen at or above the vertical
    threshold mimics "already vertical"**, so the voter returns None, the
    caller reads that as "no attitude transient to trim", and a tow-yo cast
    whose first metres are physically meaningless passes through untrimmed.
    That failure must not be silent.

    **Why this takes ONE CAST and not the whole file.** VMP SN 412 developed
    exactly this fault in SUNRISE 2021 and 2022 (both Incl_X and Incl_Y). The
    value is **frozen within each descent** — median within-cast range 0.0
    deg, against 64-78 deg on the healthy SN 142 and SN 194 — yet it *changes
    between* casts, so over the whole file it spans 6.3 to 90.0 deg across
    3033 distinct values. A file-level range test therefore passes SN 412 as
    healthy and then hands the voter a frozen angle on every cast. The
    discriminator only exists within a cast.

    A live sensor always dithers, so a genuinely vertical cast on working
    hardware still moves more than a few tenths of a degree; a frozen one
    reports bit-identical samples.

    Deliberately NOT tested here: whether the values are *physically
    sensible* beyond the range check. A miscalibrated but moving sensor
    passes, because this answers "is the hardware reporting?", not "do I
    believe the numbers".

    Parameters
    ----------
    incl_segment : array_like
        One profile's worth of the inclinometer channel [degrees].
    min_range_deg : float
        Minimum peak-to-peak excursion within the cast for the channel to
        count as reporting. Frozen channels give ~0; healthy ones on a tow-yo
        give tens of degrees, and even a steady vertical cast dithers.
    valid_range : tuple of float
        Physically admissible bounds [degrees]. The sensor is capped at
        +/-90; the default allows margin for a calibration offset.
    min_finite_fraction : float
        Minimum fraction of samples that must be finite.

    Returns
    -------
    bool
        True when the channel looks alive for this cast and should be
        trusted by :func:`attitude_trim_depth`.
    """
    a = np.asarray(incl_segment, dtype=np.float64).ravel()
    if a.size == 0:
        return False
    finite = np.isfinite(a)
    if finite.sum() < max(2, int(min_finite_fraction * a.size)):
        return False
    a = a[finite]
    lo, hi = valid_range
    if not (lo <= float(np.median(a)) <= hi):
        return False
    return bool(np.ptp(a) >= min_range_deg)


def attitude_trim_depth(
    depth: npt.ArrayLike,
    incl_y: npt.ArrayLike,
    *,
    threshold_deg: float = 85.0,
    relax_deg: float = 5.0,
    hold_m: float = 1.0,
    margin_m: float = 2.0,
    min_range_deg: float = 0.5,
) -> float | None:
    """Depth below which a tow-yo VMP has come vertical, or None.

    This is a *different kind of voter* from :func:`compute_trim_depth`. That
    one asks whether a channel's VARIANCE is still elevated. This one reads
    the inclinometer's ABSOLUTE VALUE — the instrument's attitude — and is
    therefore immune to the cast-length / window-scaling problem that afflicts
    the fall-rate residual (see the module docstring). There is nothing to
    tune against record length.

    Why it exists. In a tow-yo the VMP is launched nearly parallel to the
    surface, the reel brake is released, and it pitches down as it falls
    (Pat, 2026-09-20). Until it is vertical the shear probes see a MEAN
    CROSS-FLOW rather than turbulence, and since epsilon ~ shear^2 / U^4 the
    reported dissipation is meaningless, not merely contaminated. Measured on
    SUNRISE 2021 Walton Smith SN 194 (773 descents): Incl_Y starts at a median
    of 20.2 deg (i.e. ~70 deg off vertical) with 56 deg of roll, and reaches
    85 deg at a median of 3.9 m but anywhere in 2.7-5.7 m. Epsilon there runs
    4.5e5 x the interior at 1 m and 6.2e4 x at 2 m -- consistent with the
    geometry, since at 38 deg off vertical the mean cross-flow is U*sin(38)
    ~ 0.46 m/s against turbulent fluctuations of order 1e-3 m/s.

    Because the vertical depth varies by ~3 m between casts of one
    deployment, a flat floor necessarily over-trims the quick drops and
    under-trims the slow ones. This voter gives the per-cast answer.

    **VMP-specific.** On a Rockland VMP ``Incl_Y`` is capped at 90 deg with
    +90 = pointing straight down, so a vertically falling VMP reads ~+90. On
    a MicroRider ``Incl_Y`` is approximately pitch and this reading does not
    apply; the caller must not feed one in. The value is used literally and
    is deliberately not passed through ``abs()``, so a genuinely inverted
    instrument fails the test rather than being hidden.

    If the instrument is ALREADY vertical at the first valid sample there is
    no attitude transient and the function returns None, so a conventional
    carefully-lowered cast is untouched and pays no ``margin_m``.

    Parameters
    ----------
    depth : array_like
        Depth (positive downward) [m], same rate as ``incl_y``.
    incl_y : array_like
        Inclinometer Y [degrees]; +90 = vertical, nose down.
    threshold_deg : float
        Attitude at or above which the instrument counts as vertical.
    relax_deg : float
        The attitude may dip this far below ``threshold_deg`` during the
        hold without restarting the search, absorbing inclinometer noise.
    hold_m : float
        The attitude must stay above ``threshold_deg - relax_deg`` over this
        much further descent, so a momentary swing through vertical while
        still tumbling does not end the search early.
    margin_m : float
        Added below the vertical depth. Reaching vertical is not the same as
        the flow having settled: measured on SUNRISE, epsilon is still 6.7x
        the interior at the vertical depth and 2.2x one metre below it, but
        1.2x two metres below. Hence the default of 2 m.
    min_range_deg : float
        Passed to :func:`inclinometer_is_usable`. A channel that does not
        move this much within the cast is frozen and the function abstains,
        rather than mistaking a stuck-high reading for "already vertical".

    Returns
    -------
    float or None
        Depth [m] below which the cast is usable on attitude grounds, or
        None when the instrument started vertical (nothing to trim), never
        reached ``threshold_deg``, or the channel was frozen for this cast.
        The caller is expected to distinguish those — the last two are
        failures and must be reported, not silently kept. Use
        :func:`inclinometer_is_usable` to separate them.

    Raises
    ------
    ValueError
        If ``hold_m`` or ``margin_m`` is negative, or ``relax_deg`` < 0.
    """
    if hold_m < 0:
        raise ValueError(f"hold_m must be >= 0, got {hold_m}")
    if margin_m < 0:
        raise ValueError(f"margin_m must be >= 0, got {margin_m}")
    if relax_deg < 0:
        raise ValueError(f"relax_deg must be >= 0, got {relax_deg}")

    z = np.asarray(depth, dtype=np.float64)
    y = np.asarray(incl_y, dtype=np.float64)
    if z.shape != y.shape or z.size == 0:
        return None
    ok = np.isfinite(z) & np.isfinite(y)
    if int(ok.sum()) < 2:
        return None
    z, y = z[ok], y[ok]

    # A frozen channel must never reach the "already vertical" shortcut
    # below: SN 412's Incl_Y holds one value for a whole cast, and if that
    # value happens to sit above threshold_deg the shortcut would report a
    # clean vertical cast on a sensor that is not reporting at all. Checked
    # here as well as in the caller so the function cannot be misused.
    if not inclinometer_is_usable(y, min_range_deg=min_range_deg):
        return None

    # Already vertical at the start: no attitude transient to trim.
    if y[0] >= threshold_deg:
        return None

    floor_deg = threshold_deg - relax_deg
    cand = np.flatnonzero(y >= threshold_deg)
    for i in cand:
        # The hold runs over DEPTH, not samples, so it does not depend on the
        # sampling rate or on how fast this particular cast is falling.
        window = (z >= z[i]) & (z <= z[i] + hold_m)
        if not np.any(window):
            continue
        if np.all(y[window] >= floor_deg):
            return float(z[i] + margin_m)
    return None


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
    combine: str = "median",
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
    mechanical entry transient. Shear probes, inclinometers, and raw fall
    rate respond to the *ocean* turbulence the instrument falls through, so
    their per-bin std stays elevated at depth and would over-trim. Two
    channels are exceptions and are handled elsewhere rather than here: the
    high-passed fall-rate *residual*, which does settle (see the module
    docstring, and mind its window scaling), and the inclinometer's
    *absolute value*, which is not a variance question at all --
    :func:`attitude_trim_depth` reads it directly to find where a tow-yo VMP
    has come vertical. "Do not feed the inclinometer here" and "use the
    inclinometer there" are both correct; they use different properties of
    the same channel.

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
    combine : {"median", "max"}
        How to combine the per-channel exits.

        ``"median"`` (default) suits REDUNDANT channels measuring the same
        mechanism, where a disagreeing channel is a fault to be outvoted.

        ``"max"`` -- deepest exit wins -- is required when the channels
        measure DIFFERENT mechanisms with different reach. A channel that is
        blind to the contamination (VMP vibration sensors against prop wash)
        reports a shallow exit that is *correct for what it measures* and
        would, under a median, outvote the channel that can actually see the
        contamination. Two blind channels plus one sensitive one median to
        the blind answer. Use ``"max"`` whenever the channel set is
        heterogeneous, and accept that it trims conservatively.
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
    if combine not in ("median", "max"):
        raise ValueError(f"combine must be 'median' or 'max', got {combine!r}")
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
        # Combine the per-channel exits. median: robust to one misbehaving
        # channel among redundant ones. max: required for a heterogeneous set,
        # where a channel blind to the contamination would otherwise outvote
        # the one that sees it (see `combine` in the docstring).
        trim = float(np.median(exits)) if combine == "median" else float(np.max(exits))
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
