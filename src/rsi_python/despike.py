"""Spike removal for shear probe and micro-conductivity signals.

Port of despike.m from the ODAS MATLAB library.
"""

import numpy as np
from scipy.signal import butter, filtfilt


def despike(signal_in, fs, thresh=8, smooth=0.5, N=None, max_passes=10):
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

    y = np.asarray(signal_in, dtype=np.float64).copy().ravel()
    original = y.copy()
    all_spikes = set()
    n_passes = 0

    for _ in range(max_passes):
        y_new, spikes = _single_despike(y, thresh, smooth, fs, N)
        if len(spikes) == 0:
            break
        y = y_new
        all_spikes.update(spikes)
        n_passes += 1

    spike_indices = np.sort(np.array(list(all_spikes), dtype=np.intp))
    fraction = np.sum(y != original) / len(y)
    return y, spike_indices, n_passes, fraction


def _single_despike(dv, thresh, smooth, fs, N):
    """Single pass of spike detection and removal."""
    N_half = N // 2
    dv = dv.copy()
    length = len(dv)

    # Zero-pad with reflected ends to handle edge effects
    pad_len = min(length, 2 * int(fs / smooth))
    dv_padded = np.concatenate([
        dv[pad_len - 1::-1],
        dv,
        dv[-1:-pad_len - 1:-1],
    ])
    rng = slice(pad_len, pad_len + length)

    # High-pass filter at 0.5 Hz, rectify
    b_hp, a_hp = butter(1, 0.5 / (fs / 2), btype="high")
    dv_hp = np.abs(filtfilt(b_hp, a_hp, dv_padded))

    # Smooth envelope
    b_lp, a_lp = butter(1, smooth / (fs / 2))
    dv_lp = filtfilt(b_lp, a_lp, dv_hp)

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

    for s_bad, e_bad in zip(starts, ends):
        # Average from region R before start
        before_idx = np.arange(max(0, s_bad - R), s_bad)
        before_vals = dv_padded[before_idx[good[before_idx]]] if len(before_idx) > 0 else np.array([])

        # Average from region R after end
        after_idx = np.arange(e_bad, min(len(good), e_bad + R))
        after_vals = dv_padded[after_idx[good[after_idx]]] if len(after_idx) > 0 else np.array([])

        if len(before_vals) > 0 and len(after_vals) > 0:
            replacement = (np.mean(before_vals) + np.mean(after_vals)) / 2
        elif len(before_vals) > 0:
            replacement = np.mean(before_vals)
        elif len(after_vals) > 0:
            replacement = np.mean(after_vals)
        else:
            replacement = 0.0

        dv_padded[s_bad:e_bad] = replacement

    # Remove padding, return spike indices in original coordinates
    result = dv_padded[rng]
    spikes_orig = spike_idx - pad_len
    return result, spikes_orig
