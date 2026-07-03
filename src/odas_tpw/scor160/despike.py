# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Spike removal for shear probe and micro-conductivity signals.

Port of despike.m from the ODAS MATLAB library.
"""

import warnings
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, filtfilt


def _matlab_padlen(b: np.ndarray, a: np.ndarray) -> int:
    """Return the ``filtfilt`` *padlen* matching MATLAB's convention.

    MATLAB pads with ``3*(nfilt-1)`` reflected samples where
    ``nfilt = max(len(a), len(b))``; scipy defaults to ``3*nfilt``.
    For a 1st-order Butterworth (nfilt=2) this is padlen=3 vs scipy's 6.
    Replicated locally (rather than imported from ``l2``) to avoid a
    circular import, mirroring ``l2._matlab_padlen``.
    """
    return 3 * (max(len(a), len(b)) - 1)


class DespikeResult(NamedTuple):
    """Result of iterative spike removal."""

    y: np.ndarray
    spike_indices: np.ndarray
    n_passes: int
    fraction: float


def despike(
    signal_in: npt.ArrayLike,
    fs: float,
    thresh: float = 8,
    smooth: float = 0.5,
    N: int | None = None,
    max_passes: int = 10,
) -> DespikeResult:
    """Iterative spike removal.

    Identifies spikes by comparing the instantaneous rectified HP-filtered
    signal against its smoothed envelope. Replaces spike neighborhoods with
    local mean.

    Parameters
    ----------
    signal_in : array_like
        Signal to despike.
    fs : float
        Sampling rate [Hz].
    thresh : float
        Ratio threshold for spike detection. Default: 8.
    smooth : float
        Low-pass cutoff [Hz] for envelope smoothing. Default: 0.5.
    N : int or None
        Spike removal half-width [samples]. A total of ~1.5*N points are
        removed around each spike. Default: round(0.04 * fs) (~40 ms).
    max_passes : int
        Maximum despiking iterations. Default: 10.

    Returns
    -------
    y : ndarray
        Despiked signal.
    spike_indices : ndarray
        Indices of detected spikes in original signal.
    n_passes : int
        Number of iterations performed.
    fraction : float
        Fraction of data replaced.
    """
    if N is None:
        N = round(0.04 * fs)

    y = np.array(signal_in, dtype=np.float64).ravel()
    # Non-finite samples propagate through filtfilt and turn the entire
    # detection ratio into NaN, silently disabling ALL spike removal so
    # genuine spikes survive. ODAS removes/interpolates NaNs upstream; this
    # port does not, so at minimum warn to make the silent no-op observable.
    if y.size and not np.all(np.isfinite(y)):
        warnings.warn(
            "despike: input contains non-finite samples; filtfilt will "
            "propagate them and spike detection is disabled for this array. "
            "Remove or interpolate NaN/Inf before despiking.",
            RuntimeWarning,
            stacklevel=2,
        )
    original = y.copy()
    all_spikes: set[int] = set()
    n_passes = 0

    for _ in range(max_passes):
        y_new, spikes = _single_despike(y, thresh, smooth, fs, N)
        if len(spikes) == 0:
            break
        y = y_new
        all_spikes.update(spikes)
        n_passes += 1

    spike_indices = np.sort(np.array(list(all_spikes), dtype=np.intp))
    # NaN-safe change count: ``y != original`` reports True at unchanged NaN
    # positions (NaN != NaN), so a no-op pass on NaN-containing input used to
    # report a spurious nonzero fraction. Exclude the NaN==NaN positions.
    changed = (y != original) & ~(np.isnan(y) & np.isnan(original))
    fraction = float(np.sum(changed)) / len(y) if len(y) else 0.0
    return DespikeResult(y, spike_indices, n_passes, fraction)


def _single_despike(
    dv: np.ndarray, thresh: float, smooth: float, fs: float, N: int
) -> tuple[np.ndarray, np.ndarray]:
    """Single pass of spike detection and removal."""
    # Port note: MATLAB despike.m uses a float half-width (N/2) and rounds the
    # resulting window boundaries, so for odd N its bad-window can differ from
    # this floored N//2 by ~1 sample on each side. The physical effect is
    # sub-sample and the default N=round(0.04*fs)=20 (fs=512) is even, so the
    # default path matches MATLAB exactly; documented rather than changed.
    N_half = N // 2
    dv = dv.copy()
    length = len(dv)

    # Zero-pad with reflected ends to handle edge effects
    pad_len = min(length, 2 * int(fs / smooth))
    dv_padded = np.concatenate(
        [
            dv[pad_len - 1 :: -1],
            dv,
            dv[-1 : -pad_len - 1 : -1],
        ]
    )
    rng = slice(pad_len, pad_len + length)

    # High-pass filter at 0.5 Hz, rectify.
    # Use MATLAB's padlen (3) not scipy's default (6): restores port
    # faithfulness and avoids a ValueError crash on short (e.g. length-2,
    # padded-to-6) sections where scipy requires len(x) > padlen. Guard the
    # still-too-short remainder (e.g. length-1, padded-to-3) so despike
    # degrades to a no-op instead of crashing.
    b_hp, a_hp = butter(1, 0.5 / (fs / 2), btype="high")
    scipy_min_len = _matlab_padlen(b_hp, a_hp) + 1
    if len(dv_padded) < scipy_min_len:
        return dv, np.array([], dtype=np.intp)
    dv_hp = np.abs(filtfilt(b_hp, a_hp, dv_padded, padlen=_matlab_padlen(b_hp, a_hp)))

    # Smooth envelope
    b_lp, a_lp = butter(1, smooth / (fs / 2))
    dv_lp = filtfilt(b_lp, a_lp, dv_hp, padlen=_matlab_padlen(b_lp, a_lp))

    # Detect spikes (avoid divide-by-zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = dv_hp / dv_lp
    spike_idx = np.where(ratio > thresh)[0]

    # Ignore spikes in padding
    spike_idx = spike_idx[(spike_idx >= rng.start) & (spike_idx < rng.stop)]
    if len(spike_idx) == 0:
        return dv, np.array([], dtype=np.intp)

    # Mark bad points: N_half before spike, 2*N_half after spike
    good = np.ones(len(dv_padded), dtype=bool)
    R = round(fs / (4 * smooth))  # averaging region size

    for s in spike_idx:
        lo = max(0, s - N_half)
        hi = min(len(good), s + 2 * N_half + 1)
        good[lo:hi] = False

    # Find contiguous bad regions and replace with local mean
    transitions = np.diff(good.astype(np.int8))
    starts = np.where(transitions == -1)[0] + 1
    ends = np.where(transitions == 1)[0] + 1

    # Handle edge cases
    if not good[0]:
        starts = np.concatenate([[0], starts])
    if not good[-1]:
        ends = np.concatenate([ends, [len(good)]])

    boundary_nan = False
    for s_bad, e_bad in zip(starts, ends):
        # Average from region R before start
        before_idx = np.arange(max(0, s_bad - R), s_bad)
        before_vals = (
            dv_padded[before_idx[good[before_idx]]] if len(before_idx) > 0 else np.array([])
        )

        # Average from region R after end
        after_idx = np.arange(e_bad, min(len(good), e_bad + R))
        after_vals = dv_padded[after_idx[good[after_idx]]] if len(after_idx) > 0 else np.array([])

        # Match MATLAB despike.m: start_value/stop_value are sum/length, so an
        # empty region gives 0/0 = NaN and the average (NaN + x)/2 = NaN. The
        # prior port fabricated 0.0 (and one-sided fallbacks), silently
        # injecting plausible-looking data instead of a detectable NaN.
        start_value = np.mean(before_vals) if len(before_vals) > 0 else np.nan
        stop_value = np.mean(after_vals) if len(after_vals) > 0 else np.nan
        replacement = (start_value + stop_value) / 2

        # A spike region touching a boundary can have no clean samples on one
        # side, so the MATLAB-parity replacement above is NaN. Keep that NaN
        # (never fabricate), but make it observable rather than silent —
        # mirroring the non-finite-input warning. Only reachable for very short
        # sections; a full-length record never trips it.
        if not np.isfinite(replacement):
            boundary_nan = True

        dv_padded[s_bad:e_bad] = replacement

    if boundary_nan:
        warnings.warn(
            "despike: a spike region adjacent to the array boundary had no "
            "clean neighboring samples on one side; those samples are set to "
            "NaN (matching ODAS despike.m) rather than fabricated. This "
            "typically affects only very short sections.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Remove padding, return spike indices in original coordinates
    result = dv_padded[rng]
    spikes_orig = spike_idx - pad_len
    return result, spikes_orig
