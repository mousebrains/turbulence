# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Cross-spectral density estimation.

Port of csd_odas.m and csd_matrix_odas.m from the ODAS MATLAB library.
"""

from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from scipy import signal
from scipy.fft import rfft as _rfft


class CSDResult(NamedTuple):
    """Cross-spectral density result."""

    Cxy: np.ndarray
    F: np.ndarray
    Cxx: np.ndarray | None
    Cyy: np.ndarray | None


def _cosine_window(n: int) -> np.ndarray:
    """ODAS cosine window, normalized to RMS = 1.

    Window = 1 + cos(pi * (-1 + 2*i/n)) for i = 0..n-1.
    This is the Hann window.
    """
    w = 1.0 + np.cos(np.pi * (-1.0 + 2.0 * np.arange(n) / n))
    w /= np.sqrt(np.mean(w**2))
    return w


_window_cache: dict[int, np.ndarray] = {}


def _get_window(nfft: int) -> np.ndarray:
    """Cached cosine window — avoids recomputing for repeated nfft values."""
    if nfft not in _window_cache:
        _window_cache[nfft] = _cosine_window(nfft)
    return _window_cache[nfft]


def _detrend_segment(seg: np.ndarray, method: str, ramp: np.ndarray) -> np.ndarray:
    """Detrend a single FFT segment."""
    if method == "none":
        return seg
    if method == "constant":
        return seg - np.mean(seg)
    if method == "linear":
        return signal.detrend(seg, type="linear")
    # parabolic or cubic
    order = {"parabolic": 2, "cubic": 3}[method]
    p = np.polyfit(ramp, seg, order)
    return seg - np.polyval(p, ramp)


def csd_odas(
    x: npt.ArrayLike,
    y: npt.ArrayLike | None,
    nfft: int,
    rate: float,
    window: npt.ArrayLike | None = None,
    overlap: int | None = None,
    detrend: str = "linear",
) -> CSDResult:
    """Cross-spectral density using Welch's method with cosine window.

    Thin wrapper around :func:`csd_matrix` for 1-D signals.

    Parameters
    ----------
    x : array_like
        First signal.
    y : array_like or None
        Second signal. If None, compute auto-spectrum only.
    nfft : int
        FFT segment length.
    rate : float
        Sampling rate [Hz].
    window : array_like or None
        Window of length nfft. Default: cosine window normalized to RMS=1.
    overlap : int or None
        Overlap in samples. Default: nfft // 2.
    detrend : str
        Detrending: 'none', 'constant', 'linear', 'parabolic', 'cubic'.

    Returns
    -------
    Cxy : ndarray
        Cross-spectrum (complex) or auto-spectrum (real).
    F : ndarray
        Frequency vector [Hz], 0 to Nyquist.
    Cxx : ndarray or None
        Auto-spectrum of x (only if cross-spectrum computed).
    Cyy : ndarray or None
        Auto-spectrum of y (only if cross-spectrum computed).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if y is not None:
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        if len(x) != len(y_arr):
            raise ValueError("x and y must have same length")
        y_arr = None if np.array_equal(x, y_arr) else y_arr[:, np.newaxis]
    else:
        y_arr = None

    x_2d = x[:, np.newaxis]  # (N, 1)
    Cxy_m, F, Cxx_m, Cyy_m = csd_matrix(
        x_2d,
        y_arr,
        nfft,
        rate,
        window=window,
        overlap=overlap,
        detrend=detrend,
    )

    if Cxx_m is None:
        # Auto-spectrum: Cxy_m is (n_freq, 1, 1) complex → squeeze to real 1-D
        return CSDResult(np.real(Cxy_m[:, 0, 0]), F, None, None)

    # Cross-spectrum: squeeze matrix dims
    return CSDResult(Cxy_m[:, 0, 0], F, np.real(Cxx_m[:, 0, 0]), np.real(Cyy_m[:, 0, 0]))


def csd_matrix(
    x: np.ndarray,
    y: np.ndarray | None,
    nfft: int,
    rate: float,
    window: npt.ArrayLike | None = None,
    overlap: int | None = None,
    detrend: str = "linear",
) -> CSDResult:
    """Multi-channel cross-spectral density matrix.

    Parameters
    ----------
    x : ndarray, shape (N, n_x)
        Matrix of signals, one per column.
    y : ndarray or None, shape (N, n_y)
        Second matrix. If None, compute auto-spectral matrix of x only.
    nfft : int
        FFT segment length.
    rate : float
        Sampling rate [Hz].
    window, overlap, detrend : as in csd_odas.

    Returns
    -------
    Cxy : ndarray, shape (n_freq, n_x, n_y) or (n_freq, n_x, n_x)
        Cross-spectral matrix. If y is None, auto-spectral matrix of x.
    F : ndarray
        Frequency vector [Hz].
    Cxx : ndarray or None
        Auto-spectral matrix of x (only if y is not None).
    Cyy : ndarray or None
        Auto-spectral matrix of y (only if y is not None).
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if x.ndim == 2 and x.shape[0] < x.shape[1]:
        x = x.T
    if x.shape[0] < 2 * nfft:
        raise ValueError(f"Input length ({x.shape[0]}) must be at least 2*nfft ({2 * nfft})")
    n_x = x.shape[1]

    auto = y is None
    if not auto:
        y = np.atleast_2d(np.asarray(y, dtype=np.float64))
        if y.ndim == 2 and y.shape[0] < y.shape[1]:
            y = y.T
        n_y = y.shape[1]
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have same number of rows")

    if overlap is None:
        overlap = nfft // 2
    window = _get_window(nfft) if window is None else np.asarray(window, dtype=np.float64)

    n_freq = nfft // 2 + 1
    step = nfft - overlap
    n_seg = (x.shape[0] - overlap) // step
    ramp = np.arange(nfft, dtype=np.float64)

    if auto:
        Cxy = np.zeros((n_freq, n_x, n_x), dtype=np.complex128)
        for i in range(n_seg):
            s = i * step
            seg = np.empty((nfft, n_x))
            for c in range(n_x):
                seg[:, c] = _detrend_segment(x[s : s + nfft, c], detrend, ramp) * window
            fft_x = _rfft(seg, n=nfft, axis=0)  # (n_freq, n_x)
            # Cxy[f, i, j] = fft_x[f, j] * conj(fft_x[f, i])
            Cxy += np.conj(fft_x[:, :, np.newaxis]) * fft_x[:, np.newaxis, :]
        Cxy /= n_seg
        Cxy /= nfft * rate / 2
        Cxy[0] /= 2
        Cxy[-1] /= 2
        F = np.arange(n_freq) * rate / nfft
        return CSDResult(Cxy, F, None, None)

    # Cross case
    Cxx = np.zeros((n_freq, n_x, n_x), dtype=np.complex128)
    Cyy = np.zeros((n_freq, n_y, n_y), dtype=np.complex128)
    Cxy = np.zeros((n_freq, n_x, n_y), dtype=np.complex128)

    for i in range(n_seg):
        s = i * step
        seg_x = np.empty((nfft, n_x))
        seg_y = np.empty((nfft, n_y))
        for c in range(n_x):
            seg_x[:, c] = _detrend_segment(x[s : s + nfft, c], detrend, ramp) * window
        for c in range(n_y):
            seg_y[:, c] = _detrend_segment(y[s : s + nfft, c], detrend, ramp) * window
        fft_x = _rfft(seg_x, n=nfft, axis=0)
        fft_y = _rfft(seg_y, n=nfft, axis=0)
        # Cxx[f,i,j] = fft_x[f,j] * conj(fft_x[f,i])
        Cxx += np.conj(fft_x[:, :, np.newaxis]) * fft_x[:, np.newaxis, :]
        Cyy += np.conj(fft_y[:, :, np.newaxis]) * fft_y[:, np.newaxis, :]
        # Cxy[f,i,j] = conj(fft_x[f,i]) * fft_y[f,j]
        Cxy += np.conj(fft_x[:, :, np.newaxis]) * fft_y[:, np.newaxis, :]

    norm = n_seg * nfft * rate / 2
    for arr in (Cxx, Cyy, Cxy):
        arr /= norm
        arr[0] /= 2
        arr[-1] /= 2
    F = np.arange(n_freq) * rate / nfft
    return CSDResult(Cxy, F, Cxx, Cyy)


def _detrend_batch(segments: np.ndarray, method: str, axis: int) -> np.ndarray:
    """Detrend an array of FFT segments along *axis* (the sample axis).

    NaN/inf values are replaced with zero before detrending (so that
    scipy.signal.detrend does not raise) and restored afterward, so that
    downstream FFT produces NaN for corrupted windows.

    Parameters
    ----------
    segments : ndarray
        Array whose *axis* dimension has length nfft.
    method : str
        One of 'none', 'constant', 'linear', 'parabolic', 'cubic'.
    axis : int
        Axis corresponding to the nfft (sample) dimension.

    Returns
    -------
    ndarray — detrended copy (or the original, for 'none').
    """
    if method == "none":
        return segments

    # Guard against NaN/inf — scipy.signal.detrend raises on non-finite
    bad = ~np.isfinite(segments)
    has_bad = np.any(bad)
    if has_bad:
        segments = segments.copy()
        segments[bad] = 0.0

    if method == "constant":
        result = segments - np.mean(segments, axis=axis, keepdims=True)
    elif method == "linear":
        result = signal.detrend(segments, axis=axis, type="linear")
    else:
        # parabolic / cubic — vectorised polynomial fit along *axis*
        nfft = segments.shape[axis]
        order = {"parabolic": 2, "cubic": 3}[method]
        ramp = np.arange(nfft, dtype=np.float64)
        moved = np.moveaxis(segments, axis, -1)  # (..., nfft)
        flat = moved.reshape(-1, nfft)  # (M, nfft)
        coeffs = np.polynomial.polynomial.polyfit(ramp, flat.T, order)
        trend = np.polynomial.polynomial.polyval(ramp, coeffs)
        detrended = flat - trend
        result = np.moveaxis(detrended.reshape(moved.shape), -1, axis)

    if has_bad:
        result[bad] = np.nan

    return result


def csd_matrix_batch(
    x_windows: np.ndarray,
    y_windows: np.ndarray | None,
    nfft: int,
    rate: float,
    overlap: int | None = None,
    detrend: str = "linear",
) -> CSDResult:
    """Batched cross-spectral density for multiple dissipation windows.

    Computes the same result as calling :func:`csd_matrix` independently on
    each window, but extracts all FFT segments, detrends, windows, and
    transforms them in one vectorised pass — eliminating per-window Python
    loops.

    Parameters
    ----------
    x_windows : ndarray, shape (n_windows, diss_length, n_x)
        Stacked dissipation windows for the *x* channels.
    y_windows : ndarray or None, shape (n_windows, diss_length, n_y)
        Stacked dissipation windows for the *y* channels.
        If None, only auto-spectral matrices of *x* are computed.
    nfft : int
        FFT segment length (e.g. 256).
    rate : float
        Sampling rate [Hz].
    overlap : int or None
        Overlap in samples.  Default: ``nfft // 2``.
    detrend : str
        Detrending method: 'none', 'constant', 'linear', 'parabolic',
        'cubic'.

    Returns
    -------
    Cxy : ndarray, shape (n_windows, n_freq, n_x, n_y)
        Cross-spectral matrix between *x* and *y* channels.
        If *y_windows* is None, this is the auto-spectral matrix of *x*
        with shape (n_windows, n_freq, n_x, n_x).
    F : ndarray, shape (n_freq,)
        Frequency vector [Hz], from 0 to Nyquist.
    Cxx : ndarray or None
        Auto-spectral matrix of *x*, shape (n_windows, n_freq, n_x, n_x).
        Returned only when *y_windows* is not None.
    Cyy : ndarray or None
        Auto-spectral matrix of *y*, shape (n_windows, n_freq, n_y, n_y).
        Returned only when *y_windows* is not None.
    """
    x_windows = np.asarray(x_windows, dtype=np.float64)
    if x_windows.ndim != 3:
        raise ValueError(
            f"x_windows must be 3-D (n_windows, diss_length, n_x), got shape {x_windows.shape}"
        )
    n_windows, diss_length, _n_x = x_windows.shape

    if overlap is None:
        overlap = nfft // 2
    step = nfft - overlap
    n_freq = nfft // 2 + 1

    # --- Segment extraction ------------------------------------------------
    # Segment start indices within each window
    n_seg = (diss_length - overlap) // step
    if n_seg < 1:
        raise ValueError(
            f"diss_length ({diss_length}) too short for nfft={nfft}, overlap={overlap}"
        )
    seg_starts = np.arange(n_seg) * step  # (n_seg,)

    # Build a (nfft,) index offset and broadcast to extract all segments.
    # indices shape: (n_seg, nfft)
    indices = seg_starts[:, np.newaxis] + np.arange(nfft)[np.newaxis, :]

    # x_windows[:, indices, :] → (n_windows, n_seg, nfft, n_x)
    x_segs = x_windows[:, indices, :]

    # --- Detrend along the nfft axis (axis=2) ------------------------------
    x_segs = _detrend_batch(x_segs, detrend, axis=2)

    # --- Apply cosine window (broadcast over windows, segments, channels) --
    win = _get_window(nfft)  # (nfft,)
    x_segs *= win[np.newaxis, np.newaxis, :, np.newaxis]

    # --- FFT along the nfft axis -------------------------------------------
    # fft_x shape: (n_windows, n_seg, n_freq, n_x)
    fft_x = _rfft(x_segs, n=nfft, axis=2)

    # Free memory — segments no longer needed
    del x_segs

    # --- Normalization constant --------------------------------------------
    norm = nfft * rate / 2.0

    auto = y_windows is None

    if auto:
        # Auto-spectral matrix: Cxy[w, f, i, j] = <conj(X_i) * X_j>
        # Average over segments, then normalise.
        # einsum: for each (w, f), outer product over channel dimension
        # fft_x: (n_windows, n_seg, n_freq, n_x)
        Cxy = np.einsum("wsfi,wsfj->wfij", np.conj(fft_x), fft_x)  # (n_windows, n_freq, n_x, n_x)
        Cxy /= n_seg
        Cxy /= norm
        Cxy[:, 0, :, :] /= 2
        Cxy[:, -1, :, :] /= 2
        F = np.arange(n_freq) * rate / nfft
        return CSDResult(Cxy, F, None, None)

    # --- Cross-spectral case -----------------------------------------------
    y_windows = np.asarray(y_windows, dtype=np.float64)
    if y_windows.ndim != 3:
        raise ValueError(
            f"y_windows must be 3-D (n_windows, diss_length, n_y), got shape {y_windows.shape}"
        )
    if y_windows.shape[0] != n_windows or y_windows.shape[1] != diss_length:
        raise ValueError(
            f"y_windows shape {y_windows.shape} incompatible with x_windows shape {x_windows.shape}"
        )
    y_segs = y_windows[:, indices, :]  # (n_windows, n_seg, nfft, n_y)
    y_segs = _detrend_batch(y_segs, detrend, axis=2)
    y_segs *= win[np.newaxis, np.newaxis, :, np.newaxis]
    fft_y = _rfft(y_segs, n=nfft, axis=2)
    del y_segs

    # Cxx[w,f,i,j] = <conj(Xi) * Xj>
    Cxx = np.einsum("wsfi,wsfj->wfij", np.conj(fft_x), fft_x)
    # Cyy[w,f,i,j] = <conj(Yi) * Yj>
    Cyy = np.einsum("wsfi,wsfj->wfij", np.conj(fft_y), fft_y)
    # Cxy[w,f,i,j] = <conj(Xi) * Yj>
    Cxy = np.einsum("wsfi,wsfj->wfij", np.conj(fft_x), fft_y)

    for arr in (Cxx, Cyy, Cxy):
        arr /= n_seg * norm
        arr[:, 0, :, :] /= 2
        arr[:, -1, :, :] /= 2

    F = np.arange(n_freq) * rate / nfft
    return CSDResult(Cxy, F, Cxx, Cyy)
