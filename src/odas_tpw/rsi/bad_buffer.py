# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""Turn RDL bad-buffer dropouts into epsilon/chi processing masks.

From v6.1 the RDL substitutes ``BAD_BUFFER_SENTINEL`` for individual missing
samples (TN-051 rev. 2026-01-12, section 3.2).  :class:`~odas_tpw.rsi.p_file.PFile`
detects them and reports ``(start, length)`` spans into each channel's own
extracted samples, deliberately without modifying the data.  This module is the
consumer side of that contract.

**Short gaps are interpolated, long gaps are dropped.**  Interpolating across a
gap removes roughly its own fraction of the variance, so for a gap that is a
small part of an FFT segment the bias is far inside a dissipation estimate's
own uncertainty -- while a long contiguous gap has no information to
interpolate across at any scale.  The boundary is :data:`MAX_INTERP_S`
(0.25 s), which is where the RDL's fixed 64-sample buffer loss actually
separates: 0.125 s on a fast channel (interpolate) against 1.0 s on a slow one
(drop).  A window is dropped anyway once more than
:data:`MAX_INTERP_FRACTION` of it has been interpolated, bounding the
accumulated variance loss.

Masks are graded, not boolean: :data:`CLEAN`, :data:`INTERPOLATED`,
:data:`DROPPED`.

The scoping is the other half.  A dropout only invalidates an estimate if the
contaminated channel feeds it:

==================  ====================================================
channel             what it invalidates
==================  ====================================================
``sh{i}``           epsilon for probe *i* only
``T{i}_dT{i}``      chi for thermistor *i* only (and its ``T{i}`` base)
accelerometer /     epsilon for EVERY probe, but only when Goodman
piezo               coherent-noise removal is on -- it mixes the
                    vibration reference into every shear spectrum
``P`` / ``P_dP``    both, always (depth), plus speed on the pressure
                    and flight paths
``U_EM``            ONLY when the speed actually used came from the EM
                    flowmeter.  Under a flight model U_EM is not an
                    input -- it is at most a cross-check -- so a U_EM
                    dropout neither masks nor repairs anything.
``Incl_X/Y``        ONLY when the speed came from the flight model
reference T, C      both, via viscosity / kappa_T
==================  ====================================================

That table is the reason this module exists rather than a blanket "any
dropout kills the file": the speed provenance decides whether a telemetry
channel is load-bearing or ignorable, and only the code that *chose* the
speed knows which.  :func:`~odas_tpw.rsi.helpers.prepare_profiles` therefore
stamps ``metadata["speed_channels"]`` with what it consumed, and this module
reads that rather than re-deriving it from a method name.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

import numpy as np

from odas_tpw.scor160.io import (
    BAD_CLEAN,
    BAD_DROPPED,
    BAD_INTERPOLATED,
    BAD_MAX_INTERP_FRACTION,
)

# Mask grades. A graded int8 array rather than two boolean arrays: every
# consumer (probe combination, window fractions, NetCDF spans) then carries one
# field, and combining channels is an elementwise maximum -- dropped beats
# interpolated beats clean. Defined in scor160.io beside the structures that
# carry them, so the generic layers need not import this package.
CLEAN = BAD_CLEAN
INTERPOLATED = BAD_INTERPOLATED
DROPPED = BAD_DROPPED

# Gap duration at or below which a run is repaired by linear interpolation
# rather than rejected [s]. Set from where the observed dropouts separate: the
# RDL substitutes one fixed 64-sample buffer, which is 0.125 s on a 512 Hz fast
# channel but 1.0 s on a 64 Hz slow one. 0.25 s is 12.5% of the default 2 s FFT
# segment and 3.1% of the default 8 s dissipation window.
MAX_INTERP_S = 0.25

# Per-window ceiling on the interpolated fraction (defined in scor160.io, which
# the L4 stages can import without depending on this package).
MAX_INTERP_FRACTION = BAD_MAX_INTERP_FRACTION

# Which channels each speed method actually reads.  Sourced from
# speed.compute_speed_for_pfile, not from the method's name:
#   pressure  -> _ode_speed(P_slow, ...)
#   em        -> |U_EM|                       (W_slow is computed but unused)
#   flight    -> _flight_model_slow(W_slow, Incl_X, Incl_Y)  -- W_slow is from P,
#                so the flight path depends on pressure too, but NOT on U_EM
#                (speed.py only compares against U_EM to raise a warning)
#   constant  -> nothing
#   hotel     -> a merged hotel channel, which is not in the .p bad-buffer report
SPEED_INPUT_CHANNELS: dict[str, tuple[str, ...]] = {
    "pressure": ("P",),
    "em": ("U_EM",),
    "flight": ("P", "Incl_X", "Incl_Y"),
    "constant": (),
    "hotel": (),
}

# A pre-emphasized channel and its base are one measurement: deconvolution
# (Mudge & Lueck 1994) reconstructs X from X_dX, so a dropout in either
# contaminates both.
_PRE_EMPHASIS = re.compile(r"^(?P<base>\w+)_d(?P=base)$")

# Deconvolution runs X_dX through a first-order Butterworth at
# f_c = 1/(2*pi*diff_gain) (deconvolve.py), whose impulse response decays with
# an e-folding time of diff_gain seconds.  A dropout therefore smears FORWARD;
# three time constants leaves <5% of the disturbance.  (The polyfit that sets
# the regression coefficients is global, but a few dozen bad samples in ~1e6
# move it negligibly.)
DECONV_DILATE_TAU = 3.0

# Speed is smoothed with a Butterworth at 0.68/tau before use, so a bad sample
# in a speed input spreads either side by roughly the smoothing time constant.
SPEED_DILATE_TAU = 1.0


def sample_masks(
    report: Mapping[str, Any],
    n_fast: int,
    n_slow: int,
    *,
    fs_fast: float,
    fs_slow: float,
    max_interp_s: float = MAX_INTERP_S,
) -> dict[str, np.ndarray]:
    """Per-channel graded masks of RDL-substituted samples.

    Values are :data:`CLEAN` / :data:`INTERPOLATED` / :data:`DROPPED`, graded
    per RUN by its duration: ``length / rate`` at or below *max_interp_s* is
    repairable, longer is not.  Duration, not sample count -- the RDL loses a
    fixed 64-sample buffer, which is 0.125 s of a fast channel but 1.0 s of a
    slow one, and only one of those is short enough to interpolate through.

    *report* is :attr:`PFile.bad_buffer_report`.  Each entry declares the
    ``rate`` its spans index (``"fast"`` or ``"slow"``), which is why the
    lengths are passed in rather than read off ``PFile.channels``: detection
    runs before deconvolution, which then demotes a base channel sampled as a
    full fast column to a slow-length view (and reclassifies it, so
    ``is_fast`` follows).  Neither the stored length nor ``is_fast`` still
    reports the axis the scan ran on.

    Only confirmed runs are graded -- the isolated single hits are consistent
    with ordinary data riding the negative rail, and masking them would
    reject windows on most real files (see ``p_file.BAD_BUFFER_MIN_RUN``).
    """
    out: dict[str, np.ndarray] = {}
    for name, found in (report.get("confirmed") or {}).items():
        is_fast = str(found.get("rate", "slow")) == "fast"
        n = n_fast if is_fast else n_slow
        rate = float(fs_fast if is_fast else fs_slow)
        if n <= 0 or not np.isfinite(rate) or rate <= 0:
            continue
        mask = np.zeros(n, dtype=np.int8)
        for start, length in found.get("spans") or []:
            if start >= n:
                continue
            stop = min(start + length, n)
            grade = INTERPOLATED if (stop - start) / rate <= max_interp_s else DROPPED
            mask[start:stop] = grade
        if mask.any():
            out[name] = mask
    return out


def repair(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Linear-interpolate the :data:`INTERPOLATED` samples of *values*.

    Returns a repaired copy; *values* is never modified, since the arrays come
    straight off ``PFile.channels`` and are shared with every other consumer.
    Samples marked :data:`DROPPED` are left alone -- those windows are rejected
    outright, and inventing data there would only hide the rejection.

    Endpoint runs have no bracketing sample on one side, so ``np.interp``
    holds the nearest good value flat rather than extrapolating.
    """
    if mask.size != values.size:
        return values
    fix = mask == INTERPOLATED
    if not fix.any():
        return values
    good = ~fix & (mask != DROPPED)
    if not good.any():
        return values
    out = np.array(values, dtype=np.float64, copy=True)
    idx = np.arange(values.size)
    out[fix] = np.interp(idx[fix], idx[good], out[good])
    return out


def dilate(mask: np.ndarray, before: int = 0, after: int = 0) -> np.ndarray:
    """Widen each run of :data:`DROPPED` by *before* left and *after* right.

    Only the dropped grade widens: an interpolated sample has been replaced by
    a plausible value before anything downstream sees it, so it no longer
    contaminates the filters that would otherwise smear it.

    Run-based rather than convolution-based: the windows here are ~10^3
    samples wide over ~10^6-sample channels, and a moving-window maximum would
    be O(N*w) for a handful of runs.
    """
    hit = mask == DROPPED
    if (before <= 0 and after <= 0) or not hit.any():
        return mask
    out = mask.copy()
    d = np.diff(hit.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1) + 1
    if hit[0]:
        starts = np.concatenate(([0], starts))
    if hit[-1]:
        ends = np.concatenate((ends, [hit.size]))
    for st, en in zip(starts, ends):
        out[max(0, st - before) : min(hit.size, en + after)] = DROPPED
    return out


def expand_to_fast(
    mask_slow: np.ndarray,
    t_slow: np.ndarray,
    t_fast: np.ndarray,
) -> np.ndarray:
    """Project a slow-rate graded mask onto the fast time base.

    Slow channels reach the fast grid through ``np.interp``, so a bad slow
    sample contaminates every fast sample whose two-point stencil touches it
    -- the interval from the previous slow sample to the next.
    """
    if mask_slow.size == 0 or t_slow.size == 0 or mask_slow.size != t_slow.size:
        # Length disagreement means the mask does not belong to this time
        # base; refusing to guess beats masking the wrong samples.
        return np.zeros(len(t_fast), dtype=np.int8)
    idx = np.searchsorted(t_slow, t_fast, side="right") - 1
    lo = np.clip(idx, 0, mask_slow.size - 1)
    hi = np.clip(idx + 1, 0, mask_slow.size - 1)
    return np.asarray(np.maximum(mask_slow[lo], mask_slow[hi]), dtype=np.int8)


def _to_fast(
    name: str,
    mask: np.ndarray,
    *,
    t_fast: np.ndarray,
    t_slow: np.ndarray,
    fs_fast: float,
    diff_gains: Mapping[str, float] | None = None,
) -> np.ndarray:
    """One channel's mask on the fast grid, with deconvolution smear."""
    n_fast = len(t_fast)
    if mask.size == n_fast:
        fast = mask
    elif mask.size == len(t_slow):
        fast = expand_to_fast(mask, t_slow, t_fast)
    else:
        return np.zeros(n_fast, dtype=np.int8)
    m = _PRE_EMPHASIS.match(name)
    if m is not None:
        gain = float((diff_gains or {}).get(name, 0.94))
        after = round(DECONV_DILATE_TAU * gain * fs_fast)
        fast = dilate(fast, after=after)
    return fast


def speed_channels(metadata: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Channels the speed actually used, from ``prepare_profiles`` provenance.

    Falls back to the method name when the explicit stamp is absent (an older
    product, or a caller that bypassed ``prepare_profiles``).  An unknown or
    missing method yields ``()`` rather than a guess: over-masking on a
    channel the speed never read is exactly the failure this module exists to
    avoid.
    """
    meta = metadata or {}
    declared = meta.get("speed_channels")
    if declared is not None:
        # Comma-joined string (NetCDF-safe) or an already-split sequence.
        if isinstance(declared, str):
            return tuple(c for c in (part.strip() for part in declared.split(",")) if c)
        return tuple(str(c) for c in declared)
    method = str(meta.get("speed_method") or "").strip().lower()
    return SPEED_INPUT_CHANNELS.get(method, ())


def probe_masks(
    *,
    masks: Mapping[str, np.ndarray],
    probe_names: Sequence[str],
    shared_names: Sequence[str],
    t_fast: np.ndarray,
    t_slow: np.ndarray,
    fs_fast: float,
    speed_names: Sequence[str] = (),
    speed_tau: float = 1.5,
    diff_gains: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, str]]:
    """Assemble the per-probe fast-rate masks for one product.

    Parameters
    ----------
    masks
        Per-channel graded masks from :func:`sample_masks`.
    probe_names
        The per-probe channels, in output order: shear channels for epsilon,
        thermistor gradient channels for chi.  A dropout here affects only
        that probe.
    shared_names
        Channels every probe depends on (pressure, reference temperature,
        conductivity, and the vibration stack when Goodman is on).
    speed_names
        Channels the speed actually consumed -- see :func:`speed_channels`.
        Masked with the extra smoothing dilation, and left out entirely when
        the speed came from somewhere else.  This is what keeps a U_EM
        dropout from masking a flight-model run.
    speed_tau
        Speed smoothing time constant [s], for that dilation.

    Returns
    -------
    (mask, provenance)
        ``mask`` is ``(len(probe_names), len(t_fast))`` int8 grades (CLEAN /
        INTERPOLATED / DROPPED, combined across channels by maximum);
        ``provenance`` maps each contributing channel to the role that pulled
        it in, for the product attributes.
    """
    n_fast = len(t_fast)
    out = np.zeros((len(probe_names), n_fast), dtype=np.int8)
    provenance: dict[str, str] = {}
    if not masks:
        return out, provenance

    common = np.zeros(n_fast, dtype=np.int8)
    for name in shared_names:
        m = masks.get(name)
        if m is None:
            continue
        fast = _to_fast(
            name, m, t_fast=t_fast, t_slow=t_slow, fs_fast=fs_fast, diff_gains=diff_gains
        )
        if fast.any():
            common = np.maximum(common, fast)
            provenance[name] = "shared"

    pad = round(SPEED_DILATE_TAU * speed_tau * fs_fast)
    for name in speed_names:
        m = masks.get(name)
        if m is None:
            continue
        fast = _to_fast(
            name, m, t_fast=t_fast, t_slow=t_slow, fs_fast=fs_fast, diff_gains=diff_gains
        )
        fast = dilate(fast, before=pad, after=pad)
        if fast.any():
            common = np.maximum(common, fast)
            provenance[name] = "speed"

    for i, name in enumerate(probe_names):
        probe = common
        m = masks.get(name)
        if m is not None:
            fast = _to_fast(
                name, m, t_fast=t_fast, t_slow=t_slow, fs_fast=fs_fast, diff_gains=diff_gains
            )
            if fast.any():
                probe = np.maximum(common, fast)
                provenance[name] = "probe"
        # A pre-emphasized probe channel carries its base channel's dropouts
        # too (deconvolution couples them).
        m2 = _PRE_EMPHASIS.match(name)
        base = masks.get(m2.group("base")) if m2 is not None else None
        if base is not None:
            fast = _to_fast(
                name, base, t_fast=t_fast, t_slow=t_slow, fs_fast=fs_fast, diff_gains=diff_gains
            )
            if fast.any():
                probe = np.maximum(probe, fast)
                provenance[m2.group("base")] = "probe"  # type: ignore[union-attr]
        out[i] = probe
    return out, provenance


def encode_spans(mask: np.ndarray) -> str:
    """Serialize a graded mask as ``"start:length:grade,..."`` for a NetCDF attr.

    Per-profile NetCDFs are an intermediate in the ``prof -> eps/chi`` and
    perturb routes; without this the dropouts are known only to whoever read
    the ``.p`` file, and the handling would silently stop at the file boundary.
    Spans rather than a companion variable: there are a handful per file, and
    an attribute needs no new dimension.

    The grade travels with the span because it cannot be recovered downstream:
    it depends on the run's duration on its ORIGINAL axis, and a profile slice
    can also cut a run short.
    """
    if mask.size == 0 or not mask.any():
        return ""
    graded = mask.astype(np.int8)
    # Split wherever the grade changes, so an interpolated run abutting a
    # dropped one is not merged into one span with a single grade.
    edges = np.flatnonzero(np.diff(graded)) + 1
    bounds = np.concatenate(([0], edges, [graded.size]))
    parts = []
    for st, en in pairwise(bounds):
        grade = int(graded[st])
        if grade != CLEAN:
            parts.append(f"{int(st)}:{int(en - st)}:{grade}")
    return ",".join(parts)


def decode_spans(text: str, n: int) -> np.ndarray:
    """Inverse of :func:`encode_spans`; malformed entries are skipped."""
    mask = np.zeros(n, dtype=np.int8)
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        fields = part.split(":")
        try:
            start, length = int(fields[0]), int(fields[1])
            grade = int(fields[2]) if len(fields) > 2 else DROPPED
        except (ValueError, IndexError):
            continue
        if 0 <= start < n and length > 0 and grade in (INTERPOLATED, DROPPED):
            mask[start : min(start + length, n)] = grade
    return mask


def window_fractions(
    mask: np.ndarray,
    starts: np.ndarray,
    diss_length: int,
    grade: int = DROPPED,
) -> np.ndarray:
    """Fraction of each dissipation window carrying *grade*, per probe.

    *mask* is ``(n_probe, n_time)`` of grades; *grade* selects which one is
    counted.  Returns ``(n_probe, len(starts))``.
    """
    starts = np.asarray(starts, dtype=np.int64)
    if mask.size == 0 or starts.size == 0:
        return np.zeros((mask.shape[0], starts.size))
    hit = (mask == grade).astype(np.int64)
    csum = np.concatenate(
        [np.zeros((mask.shape[0], 1), dtype=np.int64), np.cumsum(hit, axis=1)], axis=1
    )
    n = mask.shape[1]
    lo = np.clip(starts, 0, n)
    hi = np.clip(starts + diss_length, 0, n)
    return np.asarray((csum[:, hi] - csum[:, lo]) / float(diss_length), dtype=np.float64)
