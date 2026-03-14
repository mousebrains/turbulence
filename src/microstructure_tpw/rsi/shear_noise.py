# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Electronics noise model for shear probe channels.

Port of noise_shearchannel.m from the ODAS MATLAB library (v4.5.1).

The shear probe circuit consists of:
  1. Charge-transfer amplifier (LTC6240, R1=1 GΩ, C1=1.5 nF)
  2. Differentiator (R2, C2, R3, C3)
  3. Anti-aliasing filter (two cascaded 4th-order Butterworth)
  4. ADC sampler

References
----------
RSI Technical Note 042: Noise in Shear Measurements.
"""

import numpy as np
import numpy.typing as npt


def noise_shearchannel(
    F: npt.ArrayLike,
    *,
    R1: float = 1e9,
    C1: float = 1.5e-9,
    R2: float = 499,
    C2: float = 0.94e-6,
    R3: float = 1e6,
    C3: float = 470e-12,
    CP: float = 0,
    E_1: float = 9e-9,
    fc: float = 50,
    I_1: float = 0.56e-15,
    f_AA: float = 110,
    fs: float = 512,
    VFS: float = 4.096,
    Bits: int = 16,
    gamma_RSI: float = 2.5,
    T_K: float = 295,
    K_B: float = 1.382e-23,
) -> np.ndarray:
    """Compute the output noise spectrum of the shear probe channel.

    Parameters
    ----------
    F : array_like
        Frequency vector [Hz].
    R1 : float
        First-stage feedback resistance [Ohm]. Default: 1 GΩ.
    C1 : float
        First-stage charge-transfer capacitance [F]. Default: 1.5 nF.
    R2 : float
        Differentiator input resistance [Ohm]. Default: 499 Ω.
    C2 : float
        Differentiator capacitance [F]. Default: 0.94 µF.
    R3 : float
        Differentiator feedback resistance [Ohm]. Default: 1 MΩ.
    C3 : float
        Differentiator feedback capacitance [F]. Default: 470 pF.
    CP : float
        Probe capacitance [F]. Default: 0 (no probe). Set to ~1 nF with probe.
    E_1 : float
        First-stage amplifier voltage noise [V/√Hz]. Default: 9e-9.
    fc : float
        First-stage flicker noise knee frequency [Hz]. Default: 50.
    I_1 : float
        First-stage amplifier current noise [A/√Hz]. Default: 0.56e-15.
    f_AA : float
        Anti-aliasing filter cutoff [Hz]. Default: 110.
    fs : float
        Sampling rate [Hz]. Default: 512.
    VFS : float
        ADC full-scale voltage [V]. Default: 4.096.
    Bits : int
        ADC resolution [bits]. Default: 16.
    gamma_RSI : float
        ADC noise factor (RSI sampler excess). Default: 2.5.
    T_K : float
        Operating temperature [K]. Default: 295.
    K_B : float
        Boltzmann constant [J/K]. Default: 1.382e-23.

    Returns
    -------
    noise : ndarray
        Noise spectrum [counts² / Hz].
    """
    F = np.asarray(F, dtype=np.float64)
    omega = 2 * np.pi * F

    delta_s = VFS / 2**Bits  # ADC step size
    fN = fs / 2  # Nyquist frequency

    # --- Stage 1: Charge-transfer amplifier (LTC6240) ---
    # Voltage noise (1/f model from spec sheet)
    with np.errstate(divide="ignore", invalid="ignore"):
        V_V1 = E_1**2 * (fc / F) * np.sqrt(1 + (F / fc) ** 2)
    V_V1 = np.where(np.isfinite(V_V1), V_V1, 0.0)

    # Current noise (frequency-independent up to 300 Hz)
    V_I1 = I_1**2 * R1**2 / (1 + (omega * R1 * C1) ** 2)

    # Johnson (thermal) noise from R1
    V_R1 = 4 * K_B * T_K * R1 / (1 + (omega * R1 * C1) ** 2)

    # Noise gain of first stage (includes probe capacitance)
    G_1 = (1 + (omega * R1 * (CP + C1)) ** 2) / (1 + (omega * R1 * C1) ** 2)

    Noise_1 = G_1 * (V_V1 + V_I1) + V_R1

    # --- Stage 2: Differentiator ---
    G_2 = (omega * R3 * C2) ** 2 / (1 + (omega * R2 * C2) ** 2) / (1 + (omega * R3 * C3) ** 2)

    # Note: MATLAB uses (Noise_1 + V_V1) not (Noise_1 + V_V2) —
    # same first-stage voltage noise applied as second-stage input noise
    Noise_2 = (Noise_1 + V_V1) * G_2

    # --- Anti-aliasing: two cascaded 4th-order Butterworth ---
    G_AA = 1.0 / (1 + (F / f_AA) ** 8) ** 2
    Noise_3 = Noise_2 * G_AA

    # --- Sampling: ADC quantization noise ---
    Noise_4 = Noise_3 + gamma_RSI * delta_s**2 / (12 * fN)

    # Convert to counts²/Hz
    noise_counts = Noise_4 / delta_s**2

    return noise_counts
