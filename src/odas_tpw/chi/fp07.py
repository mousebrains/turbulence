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

import warnings
from dataclasses import dataclass
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


def default_tau_model(fp07_model: str) -> str:
    """Auto-select FP07 tau model based on transfer function model.

    double_pole pairs with Goto (tau=0.003); single_pole with Lueck.
    """
    return "goto" if fp07_model == "double_pole" else "lueck"


def fp07_tau(speed: npt.ArrayLike, model: str = "lueck") -> np.ndarray | float:
    """Speed-dependent FP07 time constant [s].

    Handles both scalar and array inputs via broadcasting.

    Parameters
    ----------
    speed : float or array_like
        Profiling speed [m/s].
    model : str
        'lueck'    : tau = 0.01 * (1.0/speed)^0.5 (Lueck et al. 1977)
        'peterson' : tau = 0.012 * speed^(-0.32) (Peterson & Fer 2014)
        'goto'     : tau = 0.003 (Goto et al. 2016, for double-pole)

    Returns
    -------
    tau0 : float or ndarray
        Time constant [s].
    """
    is_scalar = np.ndim(speed) == 0
    speed = np.asarray(speed, dtype=np.float64)
    if model == "lueck":
        result = 0.01 * (1.0 / speed) ** 0.5
    elif model == "peterson":
        result = 0.012 * speed ** (-0.32)
    elif model == "goto":
        result = np.full_like(speed, 0.003) if speed.ndim > 0 else 0.003
    else:
        raise ValueError(f"Unknown FP07 tau model: {model!r}")
    return float(result) if is_scalar else result


def fp07_tau_batch(speeds: npt.ArrayLike, model: str = "lueck") -> np.ndarray:
    """Vectorized FP07 time constants for multiple speeds.

    Thin wrapper around :func:`fp07_tau` ensuring array output.

    Parameters
    ----------
    speeds : array_like
        Profiling speeds [m/s], shape ``(n,)``.
    model : str
        Same as :func:`fp07_tau`.

    Returns
    -------
    tau0 : ndarray, shape ``(n,)``
    """
    return np.atleast_1d(fp07_tau(speeds, model))


def fp07_transfer_batch(
    F: npt.ArrayLike,
    tau0: npt.ArrayLike,
    model: str = "single_pole",
) -> np.ndarray:
    """Vectorized FP07 |H(f)|^2 for multiple time constants.

    Parameters
    ----------
    F : array_like, shape ``(n_freq,)``
        Frequency vector [Hz].
    tau0 : array_like, shape ``(n_est,)``
        Per-window time constants [s].
    model : str
        'single_pole' or 'double_pole'.

    Returns
    -------
    H2 : ndarray, shape ``(n_est, n_freq)``
    """
    F = np.asarray(F, dtype=np.float64)
    tau0 = np.asarray(tau0, dtype=np.float64)
    omega_tau_sq = (2 * np.pi * F[:, np.newaxis] * tau0[np.newaxis, :]) ** 2
    if model == "single_pole":
        return (1.0 / (1.0 + omega_tau_sq)).T
    else:
        return (1.0 / (1.0 + omega_tau_sq) ** 2).T


# ---------------------------------------------------------------------------
# Electronics noise model — ports noise_thermchannel.m + gradT_noise_odas.m
# ---------------------------------------------------------------------------


@dataclass
class FP07NoiseConfig:
    """Hardware configuration for FP07 electronics noise model.

    Consolidates the 17 hardware parameters used by ``noise_thermchannel``
    with RSI defaults.  Pass an instance as the ``config`` argument to
    override any subset of defaults.
    """

    R_0: float = 3000  # nominal thermistor resistance [Ohm]
    gain: float = 6  # first-stage amplifier gain
    f_AA: float = 110  # anti-aliasing filter cutoff [Hz]
    adc_fs: float = 4.096  # ADC full-scale voltage [V]
    adc_bits: int = 16  # ADC resolution [bits]
    E_n: float = 4e-9  # first-stage voltage noise [V/sqrt(Hz)]
    fc: float = 18.7  # first-stage flicker noise knee [Hz]
    E_n2: float = 8e-9  # second-stage voltage noise [V/sqrt(Hz)]
    fc_2: float = 42  # second-stage flicker noise knee [Hz]
    gamma_RSI: float = 3  # ADC noise factor
    T_K: float = 295  # operating temperature [K]
    K_B: float = 1.382e-23  # Boltzmann constant [J/K]
    e_b: float = 0.68  # bridge excitation voltage [V]
    b: float = 1.0  # Steinhart-Hart 'b' coefficient
    beta_1: float = 3000.0  # Steinhart-Hart beta_1 [K]
    T_0: float = 289.3  # Steinhart-Hart reference temperature [K]
    beta_2: float | None = None  # Steinhart-Hart beta_2 [K]


def _unpack_noise_config(
    config: FP07NoiseConfig | None,
    *,
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
) -> dict:
    """Resolve hardware parameters from config dataclass or kwargs."""
    if config is not None:
        return {
            "R_0": config.R_0, "gain": config.gain, "f_AA": config.f_AA,
            "adc_fs": config.adc_fs, "adc_bits": config.adc_bits,
            "E_n": config.E_n, "fc": config.fc, "E_n2": config.E_n2,
            "fc_2": config.fc_2, "gamma_RSI": config.gamma_RSI,
            "T_K": config.T_K, "K_B": config.K_B, "e_b": config.e_b,
            "b": config.b, "beta_1": config.beta_1, "T_0": config.T_0,
            "beta_2": config.beta_2,
        }
    return {
        "R_0": R_0, "gain": gain, "f_AA": f_AA,
        "adc_fs": adc_fs, "adc_bits": adc_bits,
        "E_n": E_n, "fc": fc, "E_n2": E_n2,
        "fc_2": fc_2, "gamma_RSI": gamma_RSI,
        "T_K": T_K, "K_B": K_B, "e_b": e_b,
        "b": b, "beta_1": beta_1, "T_0": T_0,
        "beta_2": beta_2,
    }


def _noise_f_intermediates(
    F: np.ndarray,
    fs: float,
    diff_gain: float,
    p: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute F-dependent noise intermediates shared between scalar and batch.

    Returns (base_f, johnson_gain_f, counts_factor, delta_s).
    """
    delta_s = p["adc_fs"] / 2 ** p["adc_bits"]
    fN = fs / 2

    with np.errstate(divide="ignore", invalid="ignore"):
        V1 = 2 * p["E_n"] ** 2 * np.sqrt(1 + (F / p["fc"]) ** 2) / (F / p["fc"])
    V1 = np.where(np.isfinite(V1), V1, V1[np.isfinite(V1)].max() if np.any(np.isfinite(V1)) else 0)

    G_2 = 1 + (2 * np.pi * diff_gain * F) ** 2

    with np.errstate(divide="ignore", invalid="ignore"):
        V2 = 2 * p["E_n2"] ** 2 * np.sqrt(1 + (F / p["fc_2"]) ** 2) / (F / p["fc_2"])
    V2 = np.where(np.isfinite(V2), V2, V2[np.isfinite(V2)].max() if np.any(np.isfinite(V2)) else 0)

    G_AA = 1.0 / (1 + (F / p["f_AA"]) ** 8) ** 2
    adc_floor = p["gamma_RSI"] * delta_s ** 2 / (12 * fN)

    w = 2 * np.pi * diff_gain * F
    G_HP = (1 / diff_gain) ** 2 * w ** 2 / (1 + w ** 2)

    base_f = G_AA * G_2 * (p["gain"] ** 2 * V1 + V2) + adc_floor
    johnson_gain_f = G_AA * G_2 * p["gain"] ** 2
    counts_factor = G_HP / delta_s ** 2

    return base_f, johnson_gain_f, counts_factor, delta_s


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
    config: FP07NoiseConfig | None = None,
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

    config : FP07NoiseConfig or None
        Hardware configuration dataclass.  When provided, its fields
        override the corresponding keyword arguments.

    Returns
    -------
    gradT_noise : ndarray
        Temperature gradient noise spectrum [(K/m)^2 / Hz].
    """
    p = _unpack_noise_config(
        config, R_0=R_0, gain=gain, f_AA=f_AA, adc_fs=adc_fs,
        adc_bits=adc_bits, E_n=E_n, fc=fc, E_n2=E_n2, fc_2=fc_2,
        gamma_RSI=gamma_RSI, T_K=T_K, K_B=K_B, e_b=e_b, b=b,
        beta_1=beta_1, T_0=T_0, beta_2=beta_2,
    )

    F = np.asarray(F, dtype=np.float64)
    base_f, johnson_gain_f, counts_factor, delta_s = _noise_f_intermediates(F, fs, diff_gain, p)

    # T-dependent terms
    T_kelvin = T_mean + 273.15
    R_ratio = np.exp(p["beta_1"] * (1.0 / T_kelvin - 1.0 / p["T_0"]))
    if R_ratio < 0.1:
        warnings.warn(
            f"R_ratio={R_ratio:.4g} < 0.1 (possible broken thermistor at T={T_mean:.1f}°C); "
            "clamped to 1.0",
            stacklevel=2,
        )
        R_ratio = 1.0
    R_actual = R_ratio * p["R_0"]
    phi_R = 4 * p["K_B"] * R_actual * p["T_K"]

    # Scale factor
    eta = (p["b"] / 2) * 2 ** p["adc_bits"] * p["gain"] * p["e_b"] / p["adc_fs"]
    scale_factor = T_kelvin ** 2 * (1 + R_ratio) ** 2 / (2 * eta * p["beta_1"] * R_ratio)
    if p["beta_2"] is not None and np.isfinite(p["beta_2"]):
        scale_factor *= 1 + 2 * (p["beta_1"] / p["beta_2"]) * np.log(R_ratio)

    Noise_counts = (base_f + johnson_gain_f * phi_R) * counts_factor
    return Noise_counts * scale_factor ** 2


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


def noise_thermchannel_batch(
    F: npt.ArrayLike,
    T_means: npt.ArrayLike,
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
    config: FP07NoiseConfig | None = None,
) -> np.ndarray:
    """Vectorized electronics noise for multiple T_mean values.

    Same physics as :func:`noise_thermchannel` but computes F-dependent
    intermediates once and broadcasts T_means through T-dependent terms
    (Johnson noise and scale factor).

    Parameters
    ----------
    F : array_like
        Frequency vector [Hz], shape ``(n_freq,)``.
    T_means : array_like
        Mean temperatures [deg C], shape ``(n_est,)``.
    Other parameters same as :func:`noise_thermchannel`.

    Returns
    -------
    gradT_noise : ndarray
        Shape ``(n_est, n_freq)``.
    """
    p = _unpack_noise_config(
        config, R_0=R_0, gain=gain, f_AA=f_AA, adc_fs=adc_fs,
        adc_bits=adc_bits, E_n=E_n, fc=fc, E_n2=E_n2, fc_2=fc_2,
        gamma_RSI=gamma_RSI, T_K=T_K, K_B=K_B, e_b=e_b, b=b,
        beta_1=beta_1, T_0=T_0, beta_2=beta_2,
    )

    F = np.asarray(F, dtype=np.float64)
    T_means = np.asarray(T_means, dtype=np.float64)

    base_f, johnson_gain_f, counts_factor, _delta_s = _noise_f_intermediates(F, fs, diff_gain, p)

    # --- T-dependent terms ---
    T_kelvin = T_means + 273.15  # (n_est,)
    R_ratio = np.exp(p["beta_1"] * (1.0 / T_kelvin - 1.0 / p["T_0"]))  # (n_est,)
    R_ratio = np.where(R_ratio < 0.1, 1.0, R_ratio)
    R_actual = R_ratio * p["R_0"]
    phi_R = 4 * p["K_B"] * R_actual * p["T_K"]  # (n_est,)

    # Scale factor
    eta = (p["b"] / 2) * 2 ** p["adc_bits"] * p["gain"] * p["e_b"] / p["adc_fs"]
    scale_factor = T_kelvin ** 2 * (1 + R_ratio) ** 2 / (2 * eta * p["beta_1"] * R_ratio)
    if p["beta_2"] is not None and np.isfinite(p["beta_2"]):
        scale_factor = scale_factor * (1 + 2 * (p["beta_1"] / p["beta_2"]) * np.log(R_ratio))

    Noise_counts = (
        base_f[np.newaxis, :] + johnson_gain_f[np.newaxis, :] * phi_R[:, np.newaxis]
    ) * counts_factor[np.newaxis, :]

    return Noise_counts * (scale_factor ** 2)[:, np.newaxis]


def gradT_noise_batch(
    F: npt.ArrayLike,
    T_means: npt.ArrayLike,
    speeds: npt.ArrayLike,
    fs: float = 512,
    diff_gain: float = 0.94,
    **kwargs: Any,
) -> np.ndarray:
    """Batch temperature gradient noise in wavenumber units.

    Parameters
    ----------
    F : shape ``(n_freq,)``
    T_means : shape ``(n_est,)``
    speeds : shape ``(n_est,)``

    Returns
    -------
    noise_K : ndarray, shape ``(n_est, n_freq)``
        Noise spectrum [(K/m)^2 / cpm].
    """
    noise_f = noise_thermchannel_batch(F, T_means, fs=fs, diff_gain=diff_gain, **kwargs)
    speeds = np.asarray(speeds, dtype=np.float64)
    return noise_f * speeds[:, np.newaxis]
