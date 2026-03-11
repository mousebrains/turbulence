# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Cross-spectral density estimation.

Port of csd_odas.m and csd_matrix_odas.m from the ODAS MATLAB library.
"""

import numpy as np
import numpy.typing as npt
from scipy import signal


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Cross-spectral density using Welch's method with cosine window.

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
    if len(x) < 2 * nfft:
        raise ValueError(f"Input length ({len(x)}) must be at least 2*nfft ({2 * nfft})")
    auto = y is None
    if not auto:
        y = np.asarray(y, dtype=np.float64).ravel()
        if len(x) != len(y):
            raise ValueError("x and y must have same length")
        if np.array_equal(x, y):
            auto = True

    if overlap is None:
        overlap = nfft // 2
    if window is None:
        window = _get_window(nfft)
    else:
        window = np.asarray(window, dtype=np.float64)

    n_freq = nfft // 2 + 1
    step = nfft - overlap
    n_seg = (len(x) - overlap) // step
    ramp = np.arange(nfft, dtype=np.float64)

    if auto:
        Cxy = np.zeros(n_freq)
        for i in range(n_seg):
            s = i * step
            seg = _detrend_segment(x[s : s + nfft], detrend, ramp) * window
            X = np.fft.rfft(seg, n=nfft)
            Cxy += np.abs(X) ** 2
        Cxy /= n_seg
        Cxy /= nfft * rate / 2
        Cxy[0] /= 2
        Cxy[-1] /= 2
        F = np.arange(n_freq) * rate / nfft
        return Cxy, F, None, None

    # Cross-spectrum
    Cxy = np.zeros(n_freq, dtype=np.complex128)
    Cxx = np.zeros(n_freq)
    Cyy = np.zeros(n_freq)
    for i in range(n_seg):
        s = i * step
        sx = _detrend_segment(x[s : s + nfft], detrend, ramp) * window
        sy = _detrend_segment(y[s : s + nfft], detrend, ramp) * window
        X = np.fft.rfft(sx, n=nfft)
        Y = np.fft.rfft(sy, n=nfft)
        Cxy += Y * np.conj(X)
        Cxx += np.abs(X) ** 2
        Cyy += np.abs(Y) ** 2
    Cxy /= n_seg
    Cxx /= n_seg
    Cyy /= n_seg
    norm = nfft * rate / 2
    Cxy /= norm
    Cxx /= norm
    Cyy /= norm
    for arr in (Cxy, Cxx, Cyy):
        arr[0] /= 2
        arr[-1] /= 2
    F = np.arange(n_freq) * rate / nfft
    return Cxy, F, Cxx, Cyy


def csd_matrix(
    x: np.ndarray,
    y: np.ndarray | None,
    nfft: int,
    rate: float,
    window: npt.ArrayLike | None = None,
    overlap: int | None = None,
    detrend: str = "linear",
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
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
    if window is None:
        window = _get_window(nfft)
    else:
        window = np.asarray(window, dtype=np.float64)

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
            fft_x = np.fft.rfft(seg, n=nfft, axis=0)  # (n_freq, n_x)
            # Cxy[f, i, j] = fft_x[f, j] * conj(fft_x[f, i])
            Cxy += np.conj(fft_x[:, :, np.newaxis]) * fft_x[:, np.newaxis, :]
        Cxy /= n_seg
        Cxy /= nfft * rate / 2
        Cxy[0] /= 2
        Cxy[-1] /= 2
        F = np.arange(n_freq) * rate / nfft
        return Cxy, F, None, None

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
        fft_x = np.fft.rfft(seg_x, n=nfft, axis=0)
        fft_y = np.fft.rfft(seg_y, n=nfft, axis=0)
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
    return Cxy, F, Cxx, Cyy
