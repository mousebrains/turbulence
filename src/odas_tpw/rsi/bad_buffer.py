# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""Turn RDL bad-buffer dropouts into epsilon/chi processing masks.

From v6.1 the RDL substitutes ``BAD_BUFFER_SENTINEL`` for individual missing
samples (TN-051 rev. 2026-01-12, section 3.2).  :class:`~odas_tpw.rsi.p_file.PFile`
detects them and reports ``(start, length)`` spans into each channel's own
extracted samples, deliberately without modifying the data.  This module is the
consumer side of that contract: it turns those spans into per-probe boolean
masks on the fast time base, so a dissipation estimate whose window overlaps a
dropout on a channel it actually depends on can be rejected.

The whole point is *actually depends on*.  A dropout only invalidates an
estimate if the contaminated channel feeds it:

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
                    dropout masks nothing.
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
from typing import Any

import numpy as np

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
) -> dict[str, np.ndarray]:
    """Per-channel boolean masks of RDL-substituted samples.

    *report* is :attr:`PFile.bad_buffer_report`.  Each entry declares the
    ``rate`` its spans index (``"fast"`` or ``"slow"``), which is why the
    lengths are passed in rather than read off ``PFile.channels``: detection
    runs before deconvolution, which then demotes a base channel sampled as a
    full fast column to a slow-length view (and reclassifies it, so
    ``is_fast`` follows).  Neither the stored length nor ``is_fast`` still
    reports the axis the scan ran on.

    Only confirmed runs are masked -- the isolated single hits are consistent
    with ordinary data riding the negative rail, and masking them would
    reject windows on most real files (see ``p_file.BAD_BUFFER_MIN_RUN``).
    """
    out: dict[str, np.ndarray] = {}
    for name, found in (report.get("confirmed") or {}).items():
        n = n_fast if str(found.get("rate", "slow")) == "fast" else n_slow
        if n <= 0:
            continue
        mask = np.zeros(n, dtype=bool)
        for start, length in found.get("spans") or []:
            if start >= n:
                continue
            mask[start : min(start + length, n)] = True
        if mask.any():
            out[name] = mask
    return out


def dilate(mask: np.ndarray, before: int = 0, after: int = 0) -> np.ndarray:
    """Widen each run of True by *before* samples left and *after* right.

    Run-based rather than convolution-based: the windows here are ~10^3
    samples wide over ~10^6-sample channels, and a moving-window OR would be
    O(N*w) for a handful of runs.
    """
    if (before <= 0 and after <= 0) or not mask.any():
        return mask
    out = mask.copy()
    d = np.diff(mask.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1) + 1
    if mask[0]:
        starts = np.concatenate(([0], starts))
    if mask[-1]:
        ends = np.concatenate((ends, [mask.size]))
    for s, e in zip(starts, ends):
        out[max(0, s - before) : min(mask.size, e + after)] = True
    return out


def expand_to_fast(
    mask_slow: np.ndarray,
    t_slow: np.ndarray,
    t_fast: np.ndarray,
) -> np.ndarray:
    """Project a slow-rate mask onto the fast time base.

    Slow channels reach the fast grid through ``np.interp``, so a bad slow
    sample contaminates every fast sample whose two-point stencil touches it
    -- the interval from the previous slow sample to the next.
    """
    if mask_slow.size == 0 or t_slow.size == 0:
        return np.zeros(len(t_fast), dtype=bool)
    if mask_slow.size != t_slow.size:
        # Length disagreement means the mask does not belong to this time
        # base; refusing to guess beats masking the wrong samples.
        return np.zeros(len(t_fast), dtype=bool)
    idx = np.searchsorted(t_slow, t_fast, side="right") - 1
    lo = np.clip(idx, 0, mask_slow.size - 1)
    hi = np.clip(idx + 1, 0, mask_slow.size - 1)
    return np.asarray(mask_slow[lo] | mask_slow[hi], dtype=bool)


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
        return np.zeros(n_fast, dtype=bool)
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
        Per-channel masks from :func:`sample_masks`.
    probe_names
        The per-probe channels, in output order: shear channels for epsilon,
        thermistor gradient channels for chi.  A dropout here masks only that
        probe.
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
        ``mask`` is ``(len(probe_names), len(t_fast))`` boolean; ``provenance``
        maps each contributing channel to the role that pulled it in, for the
        product attributes.
    """
    n_fast = len(t_fast)
    out = np.zeros((len(probe_names), n_fast), dtype=bool)
    provenance: dict[str, str] = {}
    if not masks:
        return out, provenance

    common = np.zeros(n_fast, dtype=bool)
    for name in shared_names:
        m = masks.get(name)
        if m is None:
            continue
        fast = _to_fast(
            name, m, t_fast=t_fast, t_slow=t_slow, fs_fast=fs_fast, diff_gains=diff_gains
        )
        if fast.any():
            common |= fast
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
            common |= fast
            provenance[name] = "speed"

    for i, name in enumerate(probe_names):
        probe = common
        m = masks.get(name)
        if m is not None:
            fast = _to_fast(
                name, m, t_fast=t_fast, t_slow=t_slow, fs_fast=fs_fast, diff_gains=diff_gains
            )
            if fast.any():
                probe = common | fast
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
                probe = probe | fast
                provenance[m2.group("base")] = "probe"  # type: ignore[union-attr]
        out[i] = probe
    return out, provenance


def encode_spans(mask: np.ndarray) -> str:
    """Serialize a mask as ``"start:length,start:length"`` for a NetCDF attr.

    Per-profile NetCDFs are an intermediate in the ``prof -> eps/chi`` and
    perturb routes; without this the dropouts are known only to whoever read
    the ``.p`` file, and the masking would silently stop at the file boundary.
    Spans rather than a companion variable: there are a handful per file, and
    an attribute needs no new dimension.
    """
    if mask.size == 0 or not mask.any():
        return ""
    d = np.diff(mask.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1) + 1
    if mask[0]:
        starts = np.concatenate(([0], starts))
    if mask[-1]:
        ends = np.concatenate((ends, [mask.size]))
    return ",".join(f"{int(s)}:{int(e - s)}" for s, e in zip(starts, ends))


def decode_spans(text: str, n: int) -> np.ndarray:
    """Inverse of :func:`encode_spans`; malformed entries are skipped."""
    mask = np.zeros(n, dtype=bool)
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            start_s, length_s = part.split(":")
            start, length = int(start_s), int(length_s)
        except ValueError:
            continue
        if 0 <= start < n and length > 0:
            mask[start : min(start + length, n)] = True
    return mask


def window_fractions(
    mask: np.ndarray,
    starts: np.ndarray,
    diss_length: int,
) -> np.ndarray:
    """Fraction of each dissipation window that is masked, per probe.

    *mask* is ``(n_probe, n_time)``; returns ``(n_probe, len(starts))``.
    """
    starts = np.asarray(starts, dtype=np.int64)
    if mask.size == 0 or starts.size == 0:
        return np.zeros((mask.shape[0], starts.size))
    csum = np.concatenate(
        [np.zeros((mask.shape[0], 1), dtype=np.int64), np.cumsum(mask, axis=1)], axis=1
    )
    n = mask.shape[1]
    lo = np.clip(starts, 0, n)
    hi = np.clip(starts + diss_length, 0, n)
    return np.asarray((csum[:, hi] - csum[:, lo]) / float(diss_length), dtype=np.float64)
