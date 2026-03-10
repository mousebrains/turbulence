# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""FP07 thermistor transfer function and electronics noise model.

References
----------
Lueck, R.G., O. Hertzman, and T.R. Osborn, 1977: The spectral response of
    thermistors. Deep-Sea Res., 24, 951-970.
Gregg, M.C. and T.B. Meagher, 1980: The dynamic response of glass rod
    thermistors. J. Geophys. Res., 85, 2779-2786.
Peterson, A.K. and I. Fer, 2014: Dissipation measurements using temperature
    microstructure from an underwater glider. Methods in Oceanography, 10, 44-69.
RSI Technical Note 040: Noise in Temperature Gradient Measurements.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# FP07 transfer functions
# ---------------------------------------------------------------------------


def fp07_transfer(f: npt.ArrayLike, tau0: float) -> np.ndarray:
    """Single-pole FP07 transfer function |H(f)|^2.

    H^2 = 1 / (1 + (2*pi*f*tau0)^2)

    Lueck et al. 1977.

    Parameters
    ----------
    f : array_like
        Frequency [Hz].
    tau0 : float
        Time constant [s].

    Returns
    -------
    H2 : ndarray
        Squared magnitude of transfer function.
    """
    f = np.asarray(f, dtype=np.float64)
    return 1.0 / (1.0 + (2 * np.pi * f * tau0) ** 2)


def fp07_double_pole(f: npt.ArrayLike, tau0: float) -> np.ndarray:
    """Double-pole FP07 transfer function |H(f)|^2.

    H^2 = 1 / (1 + (2*pi*f*tau0)^2)^2

    Gregg & Meagher 1980.  Accounts for glass bead thermal mass
    plus thermal boundary layer.

    Parameters
    ----------
    f : array_like
        Frequency [Hz].
    tau0 : float
        Time constant [s].

    Returns
    -------
    H2 : ndarray
    """
    f = np.asarray(f, dtype=np.float64)
    return 1.0 / (1.0 + (2 * np.pi * f * tau0) ** 2) ** 2


def fp07_tau(speed: float, model: str = "lueck") -> float:
    """Speed-dependent FP07 time constant [s].

    Parameters
    ----------
    speed : float
        Profiling speed [m/s].
    model : str
        'lueck'    : tau = 0.01 * (1.0/speed)^0.5 (Lueck et al. 1977)
        'peterson' : tau = 0.012 * speed^(-0.32) (Peterson & Fer 2014)
        'goto'     : tau = 0.003 (Goto et al. 2016, for double-pole)

    Returns
    -------
    tau0 : float
        Time constant [s].
    """
    if model == "lueck":
        return 0.01 * (1.0 / speed) ** 0.5
    elif model == "peterson":
        return 0.012 * speed ** (-0.32)
    elif model == "goto":
        return 0.003
    else:
        raise ValueError(f"Unknown FP07 tau model: {model!r}")


# ---------------------------------------------------------------------------
# Electronics noise model — ports noise_thermchannel.m + gradT_noise_odas.m
# ---------------------------------------------------------------------------


def noise_thermchannel(
    F: npt.ArrayLike,
    T_mean: float,
    fs: float = 512,
    diff_gain: float = 0.94,
    R_0: float = 3000,
    gain: float = 6,
    f_AA: float = 110,
    adc_fs: float = 4.096,
    adc_bits: int = 16,
    E_n: float = 4e-9,
    fc: float = 18.7,
    E_n2: float = 8e-9,
    fc_2: float = 42,
    gamma_RSI: float = 3,
    T_K: float = 295,
    K_B: float = 1.382e-23,
    e_b: float = 0.68,
    b: float = 1.0,
    beta_1: float = 3000.0,
    T_0: float = 289.3,
    beta_2: float | None = None,
) -> np.ndarray:
    """Electronics noise spectrum for FP07 thermistor [(K/m)^2 / Hz].

    Computes the temperature gradient noise spectrum including:
    - Johnson noise from thermistor resistance
    - Amplifier voltage noise (two stages with 1/f knee)
    - Pre-emphasis gain
    - Anti-aliasing filter (two-stage 4th-order Butterworth)
    - ADC quantization noise
    - Conversion to physical gradient units via scale factor

    Ports noise_thermchannel.m and gradT_noise_odas.m from ODAS.

    Parameters
    ----------
    F : array_like
        Frequency vector [Hz].
    T_mean : float
        Mean temperature [deg C] for computing scale factor.
    fs : float
        Sampling rate [Hz].
    diff_gain : float
        Pre-emphasis differentiator gain [s].
    R_0 : float
        Nominal thermistor resistance [Ohm] at operating temperature.
    gain : float
        First-stage amplifier gain.
    f_AA : float
        Anti-aliasing filter cutoff [Hz] (each stage).
    adc_fs : float
        ADC full-scale voltage [V].
    adc_bits : int
        ADC resolution [bits].
    E_n : float
        First-stage amplifier voltage noise [V/sqrt(Hz)].
    fc : float
        First-stage flicker noise knee [Hz].
    E_n2 : float
        Second-stage amplifier voltage noise [V/sqrt(Hz)].
    fc_2 : float
        Second-stage flicker noise knee [Hz].
    gamma_RSI : float
        ADC noise factor (RSI sampler excess).
    T_K : float
        Operating temperature [K].
    K_B : float
        Boltzmann constant [J/K].
    e_b : float
        Bridge excitation voltage [V].
    b : float
        Steinhart-Hart 'b' coefficient.
    beta_1 : float
        Steinhart-Hart beta_1 coefficient [K].
    T_0 : float
        Steinhart-Hart reference temperature [K]. Default 289.3 (~16 deg C).
    beta_2 : float or None
        Steinhart-Hart beta_2 coefficient [K]. When finite, applies
        scale factor correction (MATLAB gradT_noise_odas.m l.335-338).

    Returns
    -------
    gradT_noise : ndarray
        Temperature gradient noise spectrum [(K/m)^2 / Hz].
    """
    F = np.asarray(F, dtype=np.float64)

    delta_s = adc_fs / 2**adc_bits  # ADC step size
    fN = fs / 2  # Nyquist frequency

    # Compute operating resistance for Johnson noise (MATLAB gradT_noise_odas.m)
    T_kelvin = T_mean + 273.15
    R_ratio = np.exp(beta_1 * (1.0 / T_kelvin - 1.0 / T_0))
    R_ratio = 1.0 if R_ratio < 0.1 else R_ratio  # guard: broken thermistor
    R_actual = R_ratio * R_0

    # --- Stage 1: First amplifier + Johnson noise ---
    with np.errstate(divide="ignore", invalid="ignore"):
        V1 = 2 * E_n**2 * np.sqrt(1 + (F / fc) ** 2) / (F / fc)
    V1 = np.where(np.isfinite(V1), V1, V1[np.isfinite(V1)].max() if np.any(np.isfinite(V1)) else 0)
    phi_R = 4 * K_B * R_actual * T_K  # Johnson noise (actual operating resistance)
    Noise_1 = gain**2 * (V1 + phi_R)

    # --- Stage 2: Pre-emphasis + second amplifier ---
    G_2 = 1 + (2 * np.pi * diff_gain * F) ** 2  # Pre-emphasis gain
    with np.errstate(divide="ignore", invalid="ignore"):
        V2 = 2 * E_n2**2 * np.sqrt(1 + (F / fc_2) ** 2) / (F / fc_2)
    V2 = np.where(np.isfinite(V2), V2, V2[np.isfinite(V2)].max() if np.any(np.isfinite(V2)) else 0)
    Noise_2 = G_2 * (Noise_1 + V2)

    # --- Anti-aliasing: two-stage 4th-order Butterworth ---
    G_AA = 1.0 / (1 + (F / f_AA) ** 8) ** 2
    Noise_3 = Noise_2 * G_AA

    # --- Sampling: ADC quantization noise ---
    Noise_4 = Noise_3 + gamma_RSI * delta_s**2 / (12 * fN)

    # --- Convert to counts^2/Hz ---
    Noise_counts = Noise_4 / delta_s**2

    # --- High-pass transfer function from pre-emphasis deconvolution ---
    w = 2 * np.pi * diff_gain * F
    G_HP = (1 / diff_gain) ** 2 * w**2 / (1 + w**2)
    Noise_counts = Noise_counts * G_HP

    # --- Convert to physical units ---
    # Scale factor: convert counts to physical gradient units (gradT_noise_odas.m)
    # R_ratio and T_kelvin already computed above
    eta = (b / 2) * 2**adc_bits * gain * e_b / adc_fs
    scale_factor = T_kelvin**2 * (1 + R_ratio) ** 2 / (2 * eta * beta_1 * R_ratio)
    # beta_2 correction (MATLAB gradT_noise_odas.m l.335-338)
    if beta_2 is not None and np.isfinite(beta_2):
        scale_factor *= 1 + 2 * (beta_1 / beta_2) * np.log(R_ratio)
    # Note: speed is NOT included here — caller divides by speed when converting
    # frequency spectrum to wavenumber spectrum.

    gradT_noise = Noise_counts * scale_factor**2

    return gradT_noise


def gradT_noise(
    F: npt.ArrayLike,
    T_mean: float,
    speed: float,
    fs: float = 512,
    diff_gain: float = 0.94,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Temperature gradient noise spectrum [(K/m)^2 / cpm].

    Convenience wrapper that converts noise_thermchannel output to
    wavenumber units using profiling speed.

    Parameters
    ----------
    F : array_like
        Frequency vector [Hz].
    T_mean : float
        Mean temperature [deg C].
    speed : float
        Profiling speed [m/s].
    fs : float
        Sampling rate [Hz].
    diff_gain : float
        Pre-emphasis differentiator gain [s].
    **kwargs
        Additional parameters passed to noise_thermchannel.

    Returns
    -------
    noise_K : ndarray
        Noise spectrum [(K/m)^2 / cpm].
    K : ndarray
        Wavenumber vector [cpm].
    """
    noise_f = noise_thermchannel(F, T_mean, fs=fs, diff_gain=diff_gain, **kwargs)
    F_arr = np.asarray(F, dtype=np.float64)
    K = F_arr / speed
    noise_K = noise_f * speed  # convert from per-Hz to per-cpm
    return noise_K, K
