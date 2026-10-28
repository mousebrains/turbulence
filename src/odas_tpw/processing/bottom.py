# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Bottom-crash detection — find seafloor impact from a vibration signal.

Caller supplies any number of fast-rate vibration channels as a dict
(e.g. ``{"Ax": …, "Ay": …}`` for two-axis accel, or just
``{"vibration": …}`` for a pre-computed RMS). The algorithm forms the
elementwise root-sum-square magnitude across all supplied channels, bins
its std by depth, and flags the deepest bin whose std exceeds
``vibration_factor`` times the median — but only within a few bins of the
deepest sampled depth, since a real seafloor strike is at the cast bottom.
A supra-threshold spike far above the bottom is mid-column vibration, not a
crash, and is reported via a warning rather than truncating the cast.

Reference: Code/bottom_crash_profiles.m (currently a stub in Matlab)
"""

import warnings
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
    proximity_bins: int = 2,
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
    proximity_bins : int
        A crash is accepted only in a bin within ``proximity_bins`` of the
        deepest sampled bin — a real seafloor strike is at the cast bottom,
        so a supra-threshold spike higher up is mid-column vibration, not a
        crash (audit r1-3). Must be >= 0; at the default ``depth_window=4``
        m, ``proximity_bins=2`` accepts a strike within ~8 m of the deepest
        sample. This is the geometric stand-in for the unimplemented
        ``speed_factor`` fall-rate-collapse confirmation.

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
    # speed_factor / median_factor / vibration_frequency are accepted for
    # backward compatibility but are not yet wired into the algorithm (see the
    # parameter docs above). They are kept as valid config keys — strict
    # validate_config would reject a YAML that still lists them — but warn if a
    # caller tunes one off its default so it is no longer a SILENT no-op.
    # (#104 U4-F2.)
    _unwired = {
        "speed_factor": (speed_factor, 0.3),
        "median_factor": (median_factor, 1.0),
        "vibration_frequency": (vibration_frequency, 16),
    }
    _tuned = [name for name, (val, default) in _unwired.items() if val != default]
    if _tuned:
        warnings.warn(
            "detect_bottom_crash: parameter(s) "
            f"{', '.join(_tuned)} are accepted but not yet implemented; the "
            "supplied value(s) have no effect on bottom detection",
            stacklevel=2,
        )

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

    # A real bottom crash is at the seafloor — the deepest point of the cast.
    # Restrict crash candidates to the near-bottom zone (within `proximity_bins`
    # of the deepest sampled bin). A supra-threshold spike far above the bottom
    # is mid-column vibration (a cable transient or a turbulence patch), not a
    # crash; truncating the cast there would silently delete valid deep data
    # (audit r1-3). This proximity gate is the geometric stand-in for the
    # unimplemented MATLAB speed-drop (`speed_factor`) confirmation.
    valid_idx = np.flatnonzero(valid)
    deepest_valid = int(valid_idx[-1])
    zone_top = deepest_valid - max(proximity_bins, 0)

    def _bin_mean_depth(i: int) -> float:
        # MEAN depth of the samples in bin i, not its geometric center: the
        # deepest bin's right edge overhangs nanmax(depth) by up to one bin
        # width, so its center can lie BELOW the deepest real sample and the
        # caller's `P >= bottom_depth` would match nothing. The sample mean is
        # guaranteed to lie within the bin's observed depths (robust to the
        # fast/slow rate mismatch against the slow pressure the caller trims).
        return float(np.nanmean(depth[in_range & (idx == i)]))

    # Search the near-bottom zone from the deepest bin upward for the spike.
    for i in range(deepest_valid, zone_top - 1, -1):
        if valid[i] and bin_std[i] > threshold:
            return _bin_mean_depth(i)

    # No crash at the bottom. Surface any supra-threshold spike higher up —
    # what the pre-fix deepest-upward scan would have mis-reported as a crash —
    # so genuine mid-column contamination is visible rather than silently
    # ignored, without truncating the (valid) deep cast.
    for i in range(zone_top - 1, -1, -1):
        if valid[i] and bin_std[i] > threshold:
            spike_depth = _bin_mean_depth(i)
            deepest_depth = float(np.nanmax(depth[in_range]))
            warnings.warn(
                f"detect_bottom_crash: vibration spike at ~{spike_depth:.1f} m "
                f"is {deepest_depth - spike_depth:.1f} m above the deepest "
                "sample; treated as mid-column contamination, not a bottom "
                "crash (cast not truncated)",
                stacklevel=2,
            )
            break

    return None


def detect_bottom_fallrate(
    depth_slow: npt.ArrayLike,
    fs_slow: float,
    *,
    fraction: float = 0.9,
    min_terminal: float = 0.2,
    smooth_s: float = 0.05,
    min_span: float = 8.0,
) -> float | None:
    """Bottom from the FALL-RATE COLLAPSE, not from vibration.

    Returns the deepest depth at which the instrument was still falling at
    ``fraction`` of its terminal rate — the end of free fall. Above it the
    profiler is in undisturbed descent; below it, it is decelerating into the
    seabed and the shear is meaningless (``epsilon ~ shear**2 / U**4``, so a
    collapsing ``U`` inflates epsilon violently — the bottom mirror of the
    launch-attitude transient at the surface).

    WHY THIS EXISTS. :func:`detect_bottom_crash` bins vibration std on a
    ``depth_window``-wide grid and returns the flagged bin's MEAN depth, so its
    resolution is about half a bin. Measured against this function on SUNRISE
    2022 Point Sur (134 descents, one file), it reported the bottom a median
    **1.27 m above** the depth at which the instrument actually stopped
    (p10-90 0.34-1.65 m). Pressure at ``fs_slow`` resolves ~1.5 cm per sample,
    which is the resolution a bottom-boundary-layer analysis needs. This is the
    fall-rate confirmation that ``detect_bottom_crash``'s ``speed_factor``
    documents but has never implemented.

    Parameters
    ----------
    depth_slow : array_like
        Depth (positive down) at the SLOW rate — the pressure record, not an
        interpolation of it onto the fast grid. [m or dbar]
    fs_slow : float
        Slow sampling rate [Hz].
    fraction : float
        Fraction of terminal fall rate defining the end of free fall.
        Default 0.9. **Do not raise this toward 1.0**: measured on two ships,
        the 0.90 and 0.95 thresholds are tight (p10-90 spans of 0.03 m and
        0.04 m), but at 0.98-0.99 the p90 blows out to 2-4 m because ordinary
        fall-rate variability from stratification and turbulence crosses the
        threshold far above the seabed.
    min_terminal : float
        A descent must exceed this rate to be considered falling [units/s].
    smooth_s : float
        Boxcar applied to depth before differencing [s]. **This is not a free
        parameter.** The deceleration length it measures converges to 0.082 m
        for smoothing below ~0.06 s and is then inflated by the smoother
        itself: 0.094 m at 0.125 s, 0.130 m at 0.25 s, 0.222 m at 0.50 s.
        Keep it well under the answer you are trying to measure.
    min_span : float
        A descent shorter than this is ignored [same units as depth].

    Returns
    -------
    float or None
        Depth of the end of free fall, or None if no usable descent.

    Notes
    -----
    The caller still owes a clearance above this depth before the data are
    clean — see the ``height`` idea in :func:`bottom_anchor_depth`.
    """
    depth = np.asarray(depth_slow, dtype=np.float64)
    if depth.size < 8 or not np.any(np.isfinite(depth)):
        return None
    if not np.isfinite(fs_slow) or fs_slow <= 0:
        return None
    if not (0.0 < fraction < 1.0):
        raise ValueError(f"fraction must be in (0, 1), got {fraction!r}")

    n = max(round(smooth_s * fs_slow), 1)
    smoothed = depth if n < 2 else np.convolve(depth, np.ones(n) / n, mode="same")
    t = np.arange(depth.size, dtype=np.float64) / fs_slow
    with np.errstate(invalid="ignore"):
        w = np.gradient(smoothed, t)

    falling = np.isfinite(w) & (w > min_terminal)
    if not falling.any():
        return None

    # Longest contiguous falling run is the descent.
    d = np.diff(falling.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1) + 1
    if falling[0]:
        starts = np.concatenate(([0], starts))
    if falling[-1]:
        ends = np.concatenate((ends, [falling.size]))
    if starts.size == 0:
        return None
    best = int(np.argmax(ends - starts))
    s, e = int(starts[best]), int(ends[best])

    d_seg, w_seg = depth[s:e], w[s:e]
    finite = np.isfinite(d_seg)
    if finite.sum() < 8:
        return None
    if float(np.nanmax(d_seg) - np.nanmin(d_seg)) < min_span:
        return None

    # Terminal rate from the middle half — the ends are the launch transient
    # and the crash, neither of which is terminal.
    mid = slice(d_seg.size // 4, 3 * d_seg.size // 4)
    w_term = float(np.nanmedian(w_seg[mid]))
    if not np.isfinite(w_term) or w_term <= 0:
        return None

    ok = np.flatnonzero(np.isfinite(w_seg) & (w_seg >= fraction * w_term))
    if ok.size == 0:
        return None
    return float(d_seg[ok[-1]])


def bottom_anchor_depth(bottom_depth: float | None, height: float) -> float | None:
    """The depth the deepest analysis window is anchored at: ``bottom - height``.

    ``height`` is clearance ABOVE the detected bottom, in the same units as
    ``bottom_depth``, and it buys two different things at once:

    * **Deceleration clearance.** Measured on SUNRISE, free fall ends a median
      0.086 m (Point Sur) to 0.109 m (Walton Smith) above the stop, p90 0.13 m.
      A height of 0.15 m clears it on well over 90% of casts.
    * **Pre-impact data only.** At 0.15 m and ~1 m/s the last retained sample
      predates the crash by ~0.15 s, so the resuspension cloud a silty seabed
      throws up does not exist yet in the retained record.

    It does NOT need to include the probe-to-crasher-tip offset. A bottom
    crasher's ring leads the shear probes by 6-8 inches, so at the deepest
    pre-impact sample the probes are ALREADY ~0.15-0.20 m above the seabed.
    With ``height = 0.15`` the probes sit roughly 0.30-0.35 m above the
    sediment interface.

    The detected bottom is not the sediment-water interface: over silt the
    ring must penetrate before anything is detectable, so the datum is biased
    deep by an unmeasured penetration depth. The deceleration is abrupt
    (terminal to half speed in ~0.03 m), which bounds that penetration as
    small, but it is not zero.
    """
    if bottom_depth is None or not np.isfinite(bottom_depth):
        return None
    if not np.isfinite(height):
        raise ValueError(f"height must be finite, got {height!r}")
    if height < 0:
        raise ValueError(f"height must be >= 0, got {height!r}")
    return float(bottom_depth) - float(height)
