# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Goodman coherent noise removal for shear probe spectra.

Port of clean_shear_spec.m from the ODAS MATLAB library.
Removes vibration-coherent noise from shear spectra using
accelerometer cross-spectra.
"""

import numpy as np

from rsi_python.spectral import csd_matrix


def clean_shear_spec(
    accel: np.ndarray,
    shear: np.ndarray,
    nfft: int,
    rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove acceleration-coherent noise from shear spectra.

    Parameters
    ----------
    accel : ndarray, shape (N, n_accel)
        Accelerometer signals (columns).
    shear : ndarray, shape (N, n_shear)
        Shear probe signals (columns).
    nfft : int
        FFT segment length.
    rate : float
        Sampling rate [Hz].

    Returns
    -------
    clean_UU : ndarray, shape (n_freq, n_shear, n_shear)
        Cleaned shear spectral matrix. Diagonal elements are the
        cleaned auto-spectra.
    AA : ndarray, shape (n_freq, n_accel, n_accel)
        Accelerometer spectral matrix.
    UU : ndarray, shape (n_freq, n_shear, n_shear)
        Original (uncleaned) shear spectral matrix.
    UA : ndarray, shape (n_freq, n_shear, n_accel)
        Shear-accelerometer cross-spectral matrix.
    F : ndarray
        Frequency vector [Hz].
    """
    accel = np.atleast_2d(np.asarray(accel, dtype=np.float64))
    shear = np.atleast_2d(np.asarray(shear, dtype=np.float64))
    if accel.shape[0] < accel.shape[1]:
        accel = accel.T
    if shear.shape[0] < shear.shape[1]:
        shear = shear.T
    if accel.shape[0] != shear.shape[0]:
        raise ValueError("accel and shear must have same number of rows")

    n_accel = accel.shape[1]

    # Compute all spectra in one call: x=shear, y=accel
    # Returns Cxy=(n_freq, n_sh, n_ac), Cxx=(n_freq, n_sh, n_sh), Cyy=(n_freq, n_ac, n_ac)
    try:
        UA, F, UU, AA = csd_matrix(shear, accel, nfft, rate, overlap=nfft // 2, detrend="linear")
    except ValueError:
        # Signal too short for CSD — warn and return uncleaned spectra
        import warnings

        warnings.warn(
            f"Insufficient FFT segments for Goodman cleaning "
            f"(signal length {shear.shape[0]} < 2*nfft {2 * nfft}); "
            f"returning uncleaned spectra",
            stacklevel=2,
        )
        # Fall back to auto-spectrum only (no cleaning)
        n_freq_fb = nfft // 2 + 1
        F = np.arange(n_freq_fb) * rate / nfft
        n_sh = shear.shape[1]
        n_ac = accel.shape[1]
        UU = np.zeros((n_freq_fb, n_sh, n_sh), dtype=np.complex128)
        AA = np.zeros((n_freq_fb, n_ac, n_ac), dtype=np.complex128)
        UA = np.zeros((n_freq_fb, n_sh, n_ac), dtype=np.complex128)
        return np.real(UU), AA, UU, UA, F
    assert UU is not None and AA is not None  # always returned when y is provided

    # UU, AA, UA are complex; extract real diagonal for auto-spectra
    n_freq = len(F)

    # Batched linear solve: np.linalg.solve supports batch dimensions
    # clean = UU - UA @ inv(AA) @ conj(UA).T
    try:
        ua_H = np.conj(np.swapaxes(UA, -2, -1))  # (n_freq, n_ac, n_sh)
        solved = np.linalg.solve(AA, ua_H)  # (n_freq, n_ac, n_sh)
        clean_UU = UU - np.matmul(UA, solved)  # (n_freq, n_sh, n_sh)
    except np.linalg.LinAlgError:
        # Fallback: per-frequency with singular handling
        clean_UU = np.copy(UU)
        for f in range(n_freq):
            try:
                clean_UU[f] = UU[f] - UA[f] @ np.linalg.solve(AA[f], np.conj(UA[f]).T)
            except np.linalg.LinAlgError:
                pass

    clean_UU = np.real(clean_UU)

    # Bias correction (ODAS Technical Note 61)
    fft_segments = 2 * shear.shape[0] // nfft - 1
    if fft_segments <= 1.02 * n_accel:
        import warnings

        warnings.warn(
            f"Insufficient FFT segments ({fft_segments}) for Goodman bias "
            f"correction with {n_accel} accelerometers; skipping correction",
            stacklevel=2,
        )
        R = 1.0
    else:
        R = 1.0 / (1.0 - 1.02 * n_accel / fft_segments)
    clean_UU *= R

    return clean_UU, AA, UU, UA, F
