# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Deconvolve pre-emphasized signals to produce high-resolution data.

Port of deconvolve.m from the ODAS MATLAB Library (v4.5.1).

Pre-emphasized (differentiated) channels encode high-frequency content
that the non-pre-emphasized channel cannot capture at the slow sample
rate.  Deconvolution recovers a high-resolution version of the original
signal by low-pass filtering the pre-emphasized data, then linearly
regressing against the non-pre-emphasized channel to remove the DC
offset introduced by the derivative circuit.

Reference
---------
Mudge, T.D. and R.G. Lueck, 1994: Digital signal processing to enhance
    oceanographic observations.  J. Atmos. Oceanogr. Techn., 11, 825-836.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, lfilter


def deconvolve(
    X: np.ndarray | None,
    X_dX: np.ndarray,
    fs: float,
    diff_gain: float,
) -> np.ndarray:
    """Deconvolve a pre-emphasized signal to yield high-resolution data.

    Parameters
    ----------
    X : ndarray or None
        Low-resolution (non-pre-emphasized) signal.  May be shorter than
        *X_dX* (slow-rate) or ``None`` if unavailable.  Providing it
        improves initial-condition estimation and adds a linear
        correction that preserves the factory calibration.
    X_dX : ndarray
        Pre-emphasized (high-rate, differentiated) signal.
    fs : float
        Sampling rate of *X_dX* in Hz.
    diff_gain : float
        Differentiator gain (time constant in seconds) of the
        pre-emphasis circuit.

    Returns
    -------
    ndarray
        High-resolution deconvolved signal, same length as *X_dX*.
    """
    if len(X_dX) == 0:
        return X_dX.copy()

    # Interpolate X to match X_dX length if needed
    X_interp = _interp_if_required(X, X_dX, fs)

    # First-order Butterworth low-pass at the pre-emphasis cutoff
    f_c = 1.0 / (2.0 * np.pi * diff_gain)
    b, a = butter(1, f_c / (fs / 2.0))

    # --- First pass: compute initial conditions and filter ---
    if X_interp is not None and len(X_interp) > 1:
        # Use polyfit to check for sign inversion in derivative circuit
        p = np.polyfit(X_interp, X_dX, 1)
        if p[0] < -0.5:
            X_dX = -X_dX

        # filtic(b, a, y0, x0):  zi = b[1]*x0 - a[1]*y0
        zi = np.array([b[1] * X_dX[0] - a[1] * X_interp[0]])
        X_hires = lfilter(b, a, X_dX, zi=zi)[0]

        # --- Second pass: linear regression to match non-pre-emphasized ---
        # Fit deconvolved to original, then re-filter with corrected IC
        p = np.polyfit(X_hires, X_interp, 1)
        p2_0 = 2.0 - p[0]
        p2_1 = -p[1]

        initial_output = p2_0 * X_interp[0] + p2_1
        zi = np.array([b[1] * X_dX[0] - a[1] * initial_output])
        X_hires = lfilter(b, a, X_dX, zi=zi)[0]

        # Apply the linear correction
        X_hires = p[0] * X_hires + p[1]
    else:
        # No non-pre-emphasized channel: estimate IC from first samples
        n_init = min(len(X_dX), int(2.0 * diff_gain * fs) + 1)
        n_init = max(n_init, 2)
        t_init = np.arange(n_init) / fs
        p = np.polyfit(t_init, X_dX[:n_init], 1)
        previous_output = p[1] - diff_gain * p[0]

        zi = np.array([b[1] * X_dX[0] - a[1] * previous_output])
        X_hires = lfilter(b, a, X_dX, zi=zi)[0]

    return X_hires


def _interp_if_required(
    X: np.ndarray | None,
    X_dX: np.ndarray,
    fs: float,
) -> np.ndarray | None:
    """Interpolate *X* to match the length of *X_dX* if they differ.

    Uses PCHIP (shape-preserving cubic Hermite) interpolation, matching
    MATLAB's interp1(..., 'pchip', 'extrap').
    """
    if X is None or len(X) <= 1:
        return X

    if len(X) == len(X_dX):
        return X

    fs_slow = fs * len(X) / len(X_dX)
    t_slow = np.arange(len(X)) / fs_slow
    t_fast = np.arange(len(X_dX)) / fs
    return PchipInterpolator(t_slow, X, extrapolate=True)(t_fast)
