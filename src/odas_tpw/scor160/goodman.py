# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Goodman coherent noise removal for shear probe spectra.

Port of clean_shear_spec.m from the ODAS MATLAB library.
Removes vibration-coherent noise from shear spectra using
accelerometer cross-spectra.
"""

from typing import NamedTuple

import numpy as np

from odas_tpw.scor160.spectral import csd_matrix, csd_matrix_batch


def _sanitize_autospectra(clean_UU: np.ndarray) -> np.ndarray:
    """Sanitize cleaned auto-spectrum (diagonal) bins.

    Two corrections, both on the diagonal auto-spectra:

    * Non-finite -> NaN. A singular accelerometer matrix AA cannot be cleaned.
      np.linalg.solve raises LinAlgError on some LAPACK builds (handled by the
      per-frequency fallback, which NaNs the bin) but on others returns inf/NaN
      WITHOUT raising -- so the cleaned bin is inf/NaN here instead. Map any
      non-finite bin to NaN so l4's bad-bin handling excludes it *portably*,
      rather than the result depending on whether solve happened to raise.
    * Finite negative -> 0. An ill-conditioned (not exactly singular) AA can
      over-subtract and drive a cleaned auto-spectrum slightly negative, which
      is physically impossible (a PSD cannot be < 0). Floor to 0 (matching the
      variance-method polyfit's zero-bin handling) so the epsilon integral / FOM
      see consistent non-negative power.

    Operates on the last two (n_sh, n_sh) axes, so it works for both the single
    (n_freq, ...) and batched (n_win, n_freq, ...) shapes.
    """
    n_sh = clean_UU.shape[-1]
    for i in range(n_sh):
        diag_i = clean_UU[..., i, i]
        clean_UU[..., i, i] = np.where(
            ~np.isfinite(diag_i), np.nan, np.where(diag_i < 0, 0.0, diag_i)
        )
    return clean_UU


def _nan_singular_bins(clean_UU: np.ndarray, AA: np.ndarray) -> np.ndarray:
    """NaN whole frequency bins whose accelerometer matrix AA is singular.

    ``np.linalg.solve``'s output for a singular AA is platform-dependent -- it
    raises ``LinAlgError`` on some LAPACK builds (handled by the per-frequency
    fallback) but returns inf/NaN or even finite garbage on others. Detect the
    singular bins portably via the condition number (cond == inf or astronomical
    for an exactly rank-deficient AA, e.g. duplicate/collinear accelerometer
    channels) and NaN those bins so the result does not depend on the solver's
    behavior. Merely ill-conditioned bins (cond <= 1e13) are left to
    :func:`_sanitize_autospectra`'s floor. clean_UU: (..., n_sh, n_sh); AA:
    (..., n_ac, n_ac) with matching leading dims.
    """
    # A NaN/inf in AA (a vibration dropout flows into the accel auto-spectrum)
    # makes np.linalg.cond's SVD raise LinAlgError ("SVD did not converge"),
    # which the errstate does NOT catch -- it would abort the whole .p file.
    # Exclude non-finite AA matrices from the cond/SVD and treat them as bad
    # bins directly, preserving l4's graceful NaN-bin handling.
    finite_AA = np.all(np.isfinite(AA), axis=(-2, -1))
    cond = np.full(AA.shape[:-2], np.inf)
    if np.any(finite_AA):
        with np.errstate(invalid="ignore", divide="ignore"):
            cond[finite_AA] = np.linalg.cond(AA[finite_AA])
    bad = ~np.isfinite(cond) | (cond > 1.0e13)
    if np.any(bad):
        clean_UU[bad] = np.nan
    return clean_UU


class CleanShearResult(NamedTuple):
    """Result of Goodman coherent noise removal."""

    clean_UU: np.ndarray
    AA: np.ndarray
    UU: np.ndarray
    UA: np.ndarray
    F: np.ndarray


def _bias_correction(n_samples: int, nfft: int, n_accel: int) -> float:
    """Goodman bias correction factor (ODAS TN-061, Eq. 3).

    Factor 1.02 accounts for effective degrees-of-freedom reduction
    from overlapping FFT segments (Goodman 2006, ODAS TN-061).
    """
    fft_segments = 2 * n_samples // nfft - 1
    if fft_segments <= 1.02 * n_accel:
        import warnings

        warnings.warn(
            f"Insufficient FFT segments ({fft_segments}) for Goodman bias "
            f"correction with {n_accel} accelerometers; skipping correction",
            stacklevel=3,
        )
        return 1.0
    return 1.0 / (1.0 - 1.02 * n_accel / fft_segments)


def clean_shear_spec(
    accel: np.ndarray,
    shear: np.ndarray,
    nfft: int,
    rate: float,
) -> CleanShearResult:
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
        # Signal too short for cross-spectrum — fall back to shear auto-spectrum
        import warnings

        warnings.warn(
            f"Insufficient FFT segments for Goodman cleaning "
            f"(signal length {shear.shape[0]} < 2*nfft {2 * nfft}); "
            f"returning uncleaned spectra",
            stacklevel=2,
        )
        n_sh = shear.shape[1]
        n_ac = accel.shape[1]
        try:
            # Compute the uncleaned shear auto-spectrum so callers still
            # get a usable spectrum.  Subject to the same 2*nfft length
            # check as the cross-spectrum, so this succeeds only when the
            # failure above was not a length error.
            UU, F, _, _ = csd_matrix(shear, None, nfft, rate, overlap=nfft // 2, detrend="linear")
        except ValueError:
            # Even auto-spectrum fails — return zeros
            n_freq_fb = nfft // 2 + 1
            F = np.arange(n_freq_fb) * rate / nfft
            UU = np.zeros((n_freq_fb, n_sh, n_sh), dtype=np.complex128)
        n_freq_fb = len(F)
        AA = np.zeros((n_freq_fb, n_ac, n_ac), dtype=np.complex128)
        UA = np.zeros((n_freq_fb, n_sh, n_ac), dtype=np.complex128)
        return CleanShearResult(np.real(UU), AA, UU, UA, F)
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
        # Fallback: per-frequency with singular handling. A singular AA[f]
        # cannot be cleaned; NaN that bin so downstream treats it as bad rather
        # than passing the UNCLEANED raw UU through (and then bias-correcting it
        # as if it had been cleaned).
        clean_UU = np.copy(UU)
        for f in range(n_freq):
            try:
                clean_UU[f] = UU[f] - UA[f] @ np.linalg.solve(AA[f], np.conj(UA[f]).T)
            except np.linalg.LinAlgError:
                clean_UU[f] = np.nan

    clean_UU = _sanitize_autospectra(np.real(clean_UU))
    clean_UU = _nan_singular_bins(clean_UU, AA)

    # Bias correction (ODAS Technical Note 61, Eq. 3)
    clean_UU *= _bias_correction(shear.shape[0], nfft, n_accel)

    return CleanShearResult(clean_UU, AA, UU, UA, F)


def clean_shear_spec_batch(
    accel_windows: np.ndarray,
    shear_windows: np.ndarray,
    nfft: int,
    rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched Goodman noise removal for all dissipation windows at once.

    Parameters
    ----------
    accel_windows : ndarray, shape (n_windows, diss_length, n_accel)
    shear_windows : ndarray, shape (n_windows, diss_length, n_shear)
    nfft : int
    rate : float

    Returns
    -------
    clean_UU : ndarray, shape (n_windows, n_freq, n_shear, n_shear)
        Cleaned shear spectral matrices (real).
    F : ndarray, shape (n_freq,)
        Frequency vector [Hz].
    """
    n_accel = accel_windows.shape[2]

    # Batched CSD: x=shear, y=accel
    # UA: (n_win, n_freq, n_sh, n_ac)
    # UU: (n_win, n_freq, n_sh, n_sh)
    # AA: (n_win, n_freq, n_ac, n_ac)
    UA, F, UU, AA = csd_matrix_batch(
        shear_windows,
        accel_windows,
        nfft,
        rate,
        overlap=nfft // 2,
        detrend="linear",
    )
    assert UU is not None and AA is not None  # guaranteed when y is not None

    # Goodman cleaning: clean = UU - UA @ inv(AA) @ conj(UA).T
    # np.linalg.solve broadcasts over leading dims (n_win, n_freq)
    ua_H = np.conj(np.swapaxes(UA, -2, -1))  # (n_win, n_freq, n_ac, n_sh)
    try:
        solved = np.linalg.solve(AA, ua_H)  # (n_win, n_freq, n_ac, n_sh)
        clean_UU = UU - np.matmul(UA, solved)
    except np.linalg.LinAlgError:
        clean_UU = UU.copy()
        n_win, n_freq = UU.shape[:2]
        for w in range(n_win):
            for fi in range(n_freq):
                try:
                    clean_UU[w, fi] = UU[w, fi] - UA[w, fi] @ np.linalg.solve(
                        AA[w, fi], np.conj(UA[w, fi]).T
                    )
                except np.linalg.LinAlgError:
                    # Singular AA: NaN the bin rather than keep the raw UU.
                    clean_UU[w, fi] = np.nan

    clean_UU = _sanitize_autospectra(np.real(clean_UU))
    clean_UU = _nan_singular_bins(clean_UU, AA)

    # Bias correction
    clean_UU *= _bias_correction(shear_windows.shape[1], nfft, n_accel)

    return clean_UU, F
