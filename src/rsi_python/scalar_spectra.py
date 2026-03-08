"""Temperature gradient spectrum computation.

Ports get_scalar_spectra_odas.m and make_gradT_odas.m from the ODAS library.
"""

import numpy as np

from rsi_python.spectral import csd_odas


def get_scalar_spectra(scalar_vectors, speed, P, t_fast, fs,
                       fft_length=512, spec_length=None, overlap=None,
                       f_AA=98.0, gradient_method="first_difference",
                       diff_gain=None):
    """Compute wavenumber spectra of scalar (temperature gradient) signals.

    Ports get_scalar_spectra_odas.m.

    Parameters
    ----------
    scalar_vectors : ndarray, shape (N,) or (N, n_signals)
        Scalar gradient signals (e.g., temperature gradient [K/m]).
    speed : ndarray or float
        Profiling speed [m/s].  If array, same length as scalar_vectors.
    P : ndarray, shape (N,)
        Pressure [dbar].
    t_fast : ndarray, shape (N,)
        Time vector [s].
    fs : float
        Sampling rate [Hz].
    fft_length : int
        FFT segment length [samples].
    spec_length : int or None
        Dissipation window length [samples]. Default: 3 * fft_length.
    overlap : int or None
        Window overlap [samples]. Default: spec_length // 2.
    f_AA : float
        Anti-aliasing filter cutoff [Hz].
    gradient_method : str
        'first_difference' — apply first-difference and deconvolution correction.
        'high_pass' — no spectral correction.
    diff_gain : float or array or None
        Pre-emphasis differentiator gain(s) [s].  Required if
        gradient_method='first_difference'.

    Returns
    -------
    dict with keys:
        scalar_spec : ndarray (n_freq, n_signals, n_estimates) — wavenumber spectra
        K : ndarray (n_freq, n_estimates) — wavenumber vectors [cpm]
        F : ndarray (n_freq, n_estimates) — frequency vectors [Hz]
        speed : ndarray (n_estimates,) — mean speed per window
        P : ndarray (n_estimates,) — mean pressure per window
        t : ndarray (n_estimates,) — mean time per window
    """
    scalar_vectors = np.atleast_2d(np.asarray(scalar_vectors, dtype=np.float64))
    if scalar_vectors.shape[0] < scalar_vectors.shape[1]:
        scalar_vectors = scalar_vectors.T
    N, n_signals = scalar_vectors.shape

    if np.isscalar(speed):
        speed_vec = np.full(N, float(speed))
    else:
        speed_vec = np.asarray(speed, dtype=np.float64)

    if spec_length is None:
        spec_length = 3 * fft_length
    if overlap is None:
        overlap = spec_length // 2
    if overlap >= spec_length:
        overlap = spec_length // 2

    step = spec_length - overlap
    n_estimates = max(1, 1 + (N - spec_length) // step)
    n_freq = fft_length // 2 + 1

    # Pre-allocate
    out_spec = np.full((n_freq, n_signals, n_estimates), np.nan)
    out_K = np.full((n_freq, n_estimates), np.nan)
    out_F = np.full((n_freq, n_estimates), np.nan)
    out_speed = np.full(n_estimates, np.nan)
    out_P = np.full(n_estimates, np.nan)
    out_t = np.full(n_estimates, np.nan)

    # Pre-compute first-difference correction (frequency-independent)
    correction = None
    bl_corrections = None
    if gradient_method == "first_difference":
        if diff_gain is None:
            raise ValueError("diff_gain required for gradient_method='first_difference'")
        diff_gain = np.atleast_1d(np.asarray(diff_gain, dtype=np.float64))
        if len(diff_gain) < n_signals:
            diff_gain = np.full(n_signals, diff_gain[0])

        F_template = np.arange(n_freq) * fs / fft_length
        # First-difference correction
        correction = np.ones(n_freq)
        with np.errstate(divide='ignore', invalid='ignore'):
            correction[1:] = (np.pi * F_template[1:] / (fs * np.sin(np.pi * F_template[1:] / fs)))**2
        correction = np.where(np.isfinite(correction), correction, correction[-2] if n_freq > 2 else 1.0)

        # Bilinear transform correction for deconvolution filter
        from scipy.signal import butter, freqz
        bl_corrections = np.ones((n_freq, n_signals))
        n_valid = np.arange(1, n_freq)
        for ki in range(n_signals):
            fc = 1 / (2 * np.pi * diff_gain[ki])
            b, a = butter(1, fc / (fs / 2))
            _, h = freqz(b, a, F_template[n_valid], fs=fs)
            H_actual = np.abs(h)**2
            H_ideal = 1.0 / (1 + (2 * np.pi * F_template[n_valid] * diff_gain[ki])**2)
            bl_corrections[n_valid, ki] = H_ideal / H_actual
            bl_corrections[-1, ki] = bl_corrections[-2, ki]

    for idx in range(n_estimates):
        s = idx * step
        e = s + spec_length
        if e > N:
            break

        sel = slice(s, e)
        W = np.mean(np.abs(speed_vec[sel]))
        if W < 0.01:
            W = 0.01

        out_speed[idx] = W
        out_P[idx] = np.mean(P[sel])
        out_t[idx] = np.mean(t_fast[sel])

        for ki in range(n_signals):
            Pxx, F = csd_odas(
                scalar_vectors[sel, ki], None, fft_length, fs,
                overlap=fft_length // 2, detrend="linear",
            )[:2]
            # Convert to wavenumber spectrum
            Pxx = Pxx * W  # [units^2/Hz] * [m/s] = [units^2/cpm]

            if gradient_method == "first_difference" and correction is not None:
                Pxx = Pxx * correction * bl_corrections[:, ki]

            out_spec[:, ki, idx] = Pxx

        out_F[:, idx] = F
        out_K[:, idx] = F / W

    return {
        "scalar_spec": out_spec,
        "K": out_K,
        "F": out_F,
        "speed": out_speed,
        "P": out_P,
        "t": out_t,
    }


def make_gradT_first_diff(T, fs, speed):
    """Compute temperature gradient via first difference.

    dT/dz ≈ fs * diff(T) / speed

    Parameters
    ----------
    T : ndarray
        Temperature signal [deg C].
    fs : float
        Sampling rate [Hz].
    speed : float or ndarray
        Profiling speed [m/s].

    Returns
    -------
    gradT : ndarray
        Temperature gradient [K/m], same length as T.
    """
    T = np.asarray(T, dtype=np.float64)
    dTdt = np.empty_like(T)
    dTdt[:-1] = np.diff(T) * fs
    dTdt[-1] = dTdt[-2]
    return dTdt / speed
