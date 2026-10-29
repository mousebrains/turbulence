# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""FP07 thermistor transfer function and electronics noise model.

References
----------
Lueck, R.G., O. Hertzman, and T.R. Osborn, 1977: The spectral response of
    thermistors. Deep-Sea Res., 24, 951-970.
Gregg, M.C. and T.B. Meagher, 1980: The dynamic response of glass rod
    thermistors. J. Geophys. Res., 85, 2779-2786.
Vachon, P. and R.G. Lueck, 1984: A small combined temperature-conductivity
    probe. Proc. 1984 STD Conf. and Workshop, Marine Technology Society.
Hill, K.D., 1987: Observations on the velocity scaling of thermistor dynamic
    response functions. Rev. Sci. Instrum., 58, 1235-1238.
Goto, Y., I. Yasuda, and M. Nagasawa, 2016: Turbulence estimation using
    fast-response thermistors attached to a free-fall vertical microstructure
    profiler. J. Atmos. Oceanic Technol., 33, 2065-2078.
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

    ``double_pole`` pairs with 'goto' (tau = 3 ms * W^-0.32, Goto et al.
    2016); ``single_pole`` pairs with 'lueck' (10 ms * W^-0.5).  Note
    that Peterson & Fer (2014) used a SINGLE-pole response,
    H^2(f) = (1 + (f/fc)^2)^-1, with a speed-dependent cut-off, so
    reproducing them means the ``single_pole`` transfer with the
    'peterson' tau (not the default 'lueck') — see the :func:`fp07_tau`
    notes for how to do that explicitly.
    """
    # Defect: an unrecognized fp07_model previously fell through to 'lueck'
    # silently, pairing a single-pole tau with a double-pole transfer on a typo.
    if fp07_model == "double_pole":
        return "goto"
    elif fp07_model == "single_pole":
        return "lueck"
    raise ValueError(f"Unknown FP07 model: {fp07_model!r}")


def fp07_tau(speed: npt.ArrayLike, model: str = "lueck") -> np.ndarray | float:
    """Speed-dependent FP07 time constant [s].

    Handles both scalar and array inputs via broadcasting.

    Parameters
    ----------
    speed : float or array_like
        Profiling speed [m/s].
    model : str
        'lueck'    : tau = 0.01 * speed^(-0.5), for the single pole.  The
            10 ms is Sommer et al. (2013, JTECH 30, 1907; Lueck a co-author),
            but there it is a speed-independent DOUBLE pole (f_half-power
            10.2 Hz at 0.19-0.89 m/s; they rejected velocity scaling), so
            this single pole with W^-1/2 is not their model.
            The -1/2 exponent is Vachon & Lueck's (1984), as Rockland
            cites it, and Hill's (1987), as Goto et al. (2016) cite it.
            But ODAS ``quick_look.m`` builds that law as a DOUBLE pole with
            f_3dB = 25 * sqrt(speed) Hz; this single pole puts f_3dB at
            15.9 * sqrt(speed) Hz, a response 1.6x slower.  (Compare the
            3-dB points, not the taus: a double pole with a given tau
            rolls off sooner than a single pole with the same tau.)
            Lueck et al. (1977) found 0.15-mm beads "not very dependent
            on water speed" at 0.5 and 1.25 m/s; U^-1/2 appears there only
            as one candidate boundary-layer scaling.  Nothing fits the
            prefactor (perturb's ``fp07_tau_scale`` is an optional
            multiplier, default 1), so the 10 ms sets the FP07 correction
            in nearly every chi product, and a change to it would move
            them all.
        'peterson' : tau = 0.012 * speed^(-0.32), single pole: Peterson &
            Fer (2014, doi:10.1016/j.mio.2014.05.002), whose -0.32 is Gregg
            & Meagher's (1980) U^1/3 scaling of the 3-dB frequency (cited
            in their text as "1995"; their reference list has only 1980).
        'goto'     : tau = 0.003 * speed^(-0.32), double pole (Goto et al.
            2016, doi:10.1175/JTECH-D-15-0220.1, section 2d: tau = tau0 W^g
            with g = -0.32; section 3c: tau0 = 3 ms best of 2, 3, 4 ms).  Their fit used
            a VMP at 0.6-0.7 m/s, so the exponent itself was adopted, not
            tested.  A version of this code omitted the W^-0.32 and used a
            fixed 3 ms, which is 13% short at 0.64 m/s and 2.1x short at
            0.1 m/s.

    Notes
    -----
    The pipeline pairs tau models with transfer-function models via
    :func:`default_tau_model`: ``fp07_model='single_pole'`` uses 'lueck'
    and ``'double_pole'`` uses 'goto'.  Be aware that Peterson & Fer
    (2014) used a SINGLE-pole response, H^2(f) = (1 + (f/fc)^2)^-1, with
    a speed-dependent cut-off fc = (2*pi*tau*U^-0.32)^-1, tau = 12 ms
    (their double-pole tau = 8 ms is only a Fig. 15 sensitivity test).
    The default 'lueck' tau therefore does NOT reproduce them.  To
    reproduce Peterson & Fer exactly, call the single-pole transfer with
    ``fp07_transfer`` and ``fp07_tau(speed, model='peterson')``.

    Returns
    -------
    tau0 : float or ndarray
        Time constant [s].
    """
    is_scalar = np.ndim(speed) == 0
    speed = np.asarray(speed, dtype=np.float64)
    # Defect: MATLAB gradT_noise_odas.m validates speed > 0; without it a
    # non-positive speed yields NaN (lueck/peterson) or inf downstream.
    if np.any(speed <= 0):
        raise ValueError(f"FP07 speed must be > 0, got min {np.min(speed)!r}")
    if model == "lueck":
        result = 0.01 * (1.0 / speed) ** 0.5
    elif model == "peterson":
        result = 0.012 * speed ** (-0.32)
    elif model == "goto":
        result = 0.003 * speed ** (-0.32)
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
    # Defect: an unrecognized model string silently fell through to
    # double_pole, mixing physics on a config typo. Validate like fp07_tau.
    if model == "single_pole":
        return (1.0 / (1.0 + omega_tau_sq)).T
    elif model == "double_pole":
        return (1.0 / (1.0 + omega_tau_sq) ** 2).T
    else:
        raise ValueError(f"Unknown FP07 model: {model!r}")


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
    beta_3: float | None = None  # Steinhart-Hart beta_3 [K] (exact inversion only)
    exact_inversion: bool = False  # solve R(T) exactly (an in-situ fit's curve)


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
    beta_3: float | None = None,
    exact_inversion: bool = False,
) -> dict:
    """Resolve hardware parameters from config dataclass or kwargs."""
    if config is not None:
        return {
            "R_0": config.R_0,
            "gain": config.gain,
            "f_AA": config.f_AA,
            "adc_fs": config.adc_fs,
            "adc_bits": config.adc_bits,
            "E_n": config.E_n,
            "fc": config.fc,
            "E_n2": config.E_n2,
            "fc_2": config.fc_2,
            "gamma_RSI": config.gamma_RSI,
            "T_K": config.T_K,
            "K_B": config.K_B,
            "e_b": config.e_b,
            "b": config.b,
            "beta_1": config.beta_1,
            "T_0": config.T_0,
            "beta_2": config.beta_2,
            "beta_3": config.beta_3,
            "exact_inversion": config.exact_inversion,
        }
    return {
        "R_0": R_0,
        "gain": gain,
        "f_AA": f_AA,
        "adc_fs": adc_fs,
        "adc_bits": adc_bits,
        "E_n": E_n,
        "fc": fc,
        "E_n2": E_n2,
        "fc_2": fc_2,
        "gamma_RSI": gamma_RSI,
        "T_K": T_K,
        "K_B": K_B,
        "e_b": e_b,
        "b": b,
        "beta_1": beta_1,
        "T_0": T_0,
        "beta_2": beta_2,
        "beta_3": beta_3,
        "exact_inversion": exact_inversion,
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
    adc_floor = p["gamma_RSI"] * delta_s**2 / (12 * fN)

    w = 2 * np.pi * diff_gain * F
    G_HP = (1 / diff_gain) ** 2 * w**2 / (1 + w**2)

    base_f = G_AA * G_2 * (p["gain"] ** 2 * V1 + V2) + adc_floor
    johnson_gain_f = G_AA * G_2 * p["gain"] ** 2
    counts_factor = G_HP / delta_s**2

    return base_f, johnson_gain_f, counts_factor, delta_s


def bead_inversion(
    T_mean: npt.ArrayLike,
    T_0: float,
    beta_1: float,
    beta_2: float | None = None,
    beta_3: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact ``L = ln(R/R_0)`` and ``d(1/T)/dL`` of a bead's polynomial at *T_mean*.

    Solves ``1/T = 1/T_0 + L/beta_1 + L^2/beta_2 [+ L^3/beta_3]`` (the
    ``convert_therm`` form) for its real roots at each temperature [deg C].
    The bead's branch is the NTC one, ``d(1/T)/dL > 0`` (R falls as T rises);
    among its roots the one nearest the order-1 estimate ``beta_1 (1/T - 1/T_0)``
    is taken. The two roots of a quadratic share ``|d(1/T)/dL|`` and differ
    only in R, but a curved fit puts the order-1 estimate on the wrong one: on
    CASPER 2015 doug CAS_024 the nearest root was 0.04 in L from the file's
    data, the NTC root inside them (and on every profile sample of the six
    beta_1 < 0 CASPER channels checked).

    Returns ``(L, slope, solved)``; ``solved`` is False (L and slope NaN) where
    the curve has no NTC root at that temperature -- no real root at all (past
    a quadratic's turning point), or only roots where T rises with R, which no
    working FP07 reads -- and at a non-finite temperature.
    """
    T_K = np.atleast_1d(np.asarray(T_mean, dtype=np.float64)) + 273.15
    a = [1.0 / T_0, 1.0 / beta_1]
    for beta in (beta_2, beta_3):
        a.append(1.0 / beta if beta is not None and np.isfinite(beta) else 0.0)
    while len(a) > 2 and a[-1] == 0.0:
        a.pop()
    poly = np.polynomial.Polynomial(a)
    dpoly = poly.deriv()
    L = np.full(T_K.shape, np.nan)
    for i, tk in enumerate(T_K):
        if not np.isfinite(tk):
            continue
        roots = (poly - 1.0 / tk).roots()
        real = roots[np.abs(roots.imag) <= 1e-9 * np.maximum(1.0, np.abs(roots.real))].real
        if real.size == 0:
            continue
        ntc = real[dpoly(real) > 0]
        if ntc.size == 0:
            continue
        x = float(ntc[np.argmin(np.abs(ntc - beta_1 * (1.0 / tk - 1.0 / T_0)))])
        for _ in range(2):  # polish the companion-matrix root
            d = float(dpoly(x))
            if d <= 0.0:
                break
            x -= (float(poly(x)) - 1.0 / tk) / d
        L[i] = x
    solved = np.isfinite(L)
    return L, dpoly(L), solved


def _exact_R_and_slope(T_kelvin: np.ndarray, p: dict) -> tuple[np.ndarray, np.ndarray]:
    """R/R_0 and d(1/T)/dL from :func:`bead_inversion`, for the scale factor.

    Where the curve has no NTC root at the temperature both are NaN, so is the
    noise model, and so the window gets no chi (every chi estimator returns NaN
    on a NaN noise spectrum, as for a window without a temperature). Falling
    back to the ODAS form there would put back the very value the exact
    inversion exists to replace, far off on such curves, and it would jump at
    a quadratic's turning point where the exact noise goes to zero. The
    count is warned here and recorded per chi file. R < 0.1 is clamped to 1, as
    in the ODAS form.
    """
    L, slope, solved = bead_inversion(
        T_kelvin - 273.15, p["T_0"], p["beta_1"], p["beta_2"], p["beta_3"]
    )
    unsolved = ~solved & np.isfinite(T_kelvin)
    if np.any(unsolved):
        warnings.warn(
            f"FP07 noise model: the bead's fitted curve has no NTC root at "
            f"{int(unsolved.sum())} of {unsolved.size} temperatures; no noise model "
            "(NaN) there",
            stacklevel=3,
        )
    R_ratio = np.exp(L)
    low = R_ratio < 0.1
    if np.any(low):
        warnings.warn(
            f"R_ratio < 0.1 for {int(low.sum())} of {R_ratio.size} T_means "
            "(possible broken thermistor); clamped to 1.0",
            stacklevel=3,
        )
        R_ratio = np.where(low, 1.0, R_ratio)
        slope = np.where(low, 1.0 / p["beta_1"], slope)
    return R_ratio, slope


def _inverts_exactly(p: dict) -> bool:
    """Exact inversion asked for AND a curved polynomial to invert.

    An order-1 curve is inverted exactly by the ODAS form already, so it keeps
    that code path bit for bit.
    """
    return bool(p["exact_inversion"]) and any(
        p[k] is not None and np.isfinite(p[k]) for k in ("beta_2", "beta_3")
    )


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
    beta_3: float | None = None,
    exact_inversion: bool = False,
    config: FP07NoiseConfig | None = None,
) -> np.ndarray:
    """Electronics noise spectrum for FP07 thermistor [(K/s)^2 / Hz].

    Returns the noise of the temperature *time derivative* dT/dt.  Use
    :func:`gradT_noise` to convert to a spatial-gradient spectrum
    [(K/m)^2 / cpm] using the profiling speed.

    Computes the noise spectrum including:
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
    beta_3 : float or None
        Steinhart-Hart beta_3 coefficient [K]; used only with
        ``exact_inversion``.
    exact_inversion : bool
        Solve the bead's polynomial for R and its slope exactly
        (:func:`bead_inversion`) instead of the ODAS order-1 inversion with a
        first-order beta_2 correction. For an in-situ fit's curve (#245): a
        strongly curved per-file fit (beta_2 comparable to beta_1) put the
        ODAS form more than 2x off in power on 126 of CASPER 2015 doug's 272
        fitted channels. Where the curve has no NTC root at *T_mean* the noise
        is NaN, with a warning. An order-1 curve is unchanged.

    config : FP07NoiseConfig or None
        Hardware configuration dataclass.  When provided, its fields
        override the corresponding keyword arguments.

    Returns
    -------
    noise_f : ndarray
        Temperature time-derivative noise spectrum [(K/s)^2 / Hz].
    """
    p = _unpack_noise_config(
        config,
        R_0=R_0,
        gain=gain,
        f_AA=f_AA,
        adc_fs=adc_fs,
        adc_bits=adc_bits,
        E_n=E_n,
        fc=fc,
        E_n2=E_n2,
        fc_2=fc_2,
        gamma_RSI=gamma_RSI,
        T_K=T_K,
        K_B=K_B,
        e_b=e_b,
        b=b,
        beta_1=beta_1,
        T_0=T_0,
        beta_2=beta_2,
        beta_3=beta_3,
        exact_inversion=exact_inversion,
    )

    F = np.asarray(F, dtype=np.float64)
    base_f, johnson_gain_f, counts_factor, _delta_s = _noise_f_intermediates(F, fs, diff_gain, p)

    # T-dependent terms
    T_kelvin = T_mean + 273.15
    eta = (p["b"] / 2) * 2 ** p["adc_bits"] * p["gain"] * p["e_b"] / p["adc_fs"]
    if _inverts_exactly(p):
        R_exact, slope = _exact_R_and_slope(np.array([T_kelvin], dtype=np.float64), p)
        R_ratio = float(R_exact[0])
        scale_factor = T_kelvin**2 * (1 + R_ratio) ** 2 / (2 * eta * R_ratio) * float(slope[0])
    else:
        R_ratio = np.exp(p["beta_1"] * (1.0 / T_kelvin - 1.0 / p["T_0"]))
        if R_ratio < 0.1:
            warnings.warn(
                f"R_ratio={R_ratio:.4g} < 0.1 (possible broken thermistor at "
                f"T={T_mean:.1f}°C); clamped to 1.0",
                stacklevel=2,
            )
            R_ratio = 1.0
        # Scale factor
        scale_factor = T_kelvin**2 * (1 + R_ratio) ** 2 / (2 * eta * p["beta_1"] * R_ratio)
        if p["beta_2"] is not None and np.isfinite(p["beta_2"]):
            scale_factor *= 1 + 2 * (p["beta_1"] / p["beta_2"]) * np.log(R_ratio)
    R_actual = R_ratio * p["R_0"]
    phi_R = 4 * p["K_B"] * R_actual * p["T_K"]

    Noise_counts = (base_f + johnson_gain_f * phi_R) * counts_factor
    return Noise_counts * scale_factor**2


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
    # Defect: a non-positive speed gives a negative noise PSD (impossible) or
    # inf; MATLAB gradT_noise_odas.m validates speed > 0 (val_speed).
    if not (np.isfinite(speed) and speed > 0):
        raise ValueError(f"FP07 speed must be > 0, got {speed!r}")
    noise_f = noise_thermchannel(F, T_mean, fs=fs, diff_gain=diff_gain, **kwargs)
    F_arr = np.asarray(F, dtype=np.float64)
    K = F_arr / speed
    # noise_f is the time-derivative noise [(K/s)^2/Hz].  Converting to a
    # gradient spectrum divides by speed^2 (dT/dz = (dT/dt)/W), and the
    # per-Hz -> per-cpm change of variable multiplies by speed: net /speed.
    # Matches ODAS gradT_noise_odas.m ("scale_factor ./ speed", squared).
    noise_K = noise_f / speed
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
    beta_3: float | None = None,
    exact_inversion: bool = False,
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
    noise_f : ndarray
        Temperature time-derivative noise [(K/s)^2 / Hz],
        shape ``(n_est, n_freq)``.
    """
    p = _unpack_noise_config(
        config,
        R_0=R_0,
        gain=gain,
        f_AA=f_AA,
        adc_fs=adc_fs,
        adc_bits=adc_bits,
        E_n=E_n,
        fc=fc,
        E_n2=E_n2,
        fc_2=fc_2,
        gamma_RSI=gamma_RSI,
        T_K=T_K,
        K_B=K_B,
        e_b=e_b,
        b=b,
        beta_1=beta_1,
        T_0=T_0,
        beta_2=beta_2,
        beta_3=beta_3,
        exact_inversion=exact_inversion,
    )

    F = np.asarray(F, dtype=np.float64)
    T_means = np.asarray(T_means, dtype=np.float64)

    base_f, johnson_gain_f, counts_factor, _delta_s = _noise_f_intermediates(F, fs, diff_gain, p)

    # --- T-dependent terms ---
    T_kelvin = T_means + 273.15  # (n_est,)
    eta = (p["b"] / 2) * 2 ** p["adc_bits"] * p["gain"] * p["e_b"] / p["adc_fs"]
    if _inverts_exactly(p):
        R_ratio, slope = _exact_R_and_slope(np.atleast_1d(T_kelvin), p)
        scale_factor = T_kelvin**2 * (1 + R_ratio) ** 2 / (2 * eta * R_ratio) * slope
    else:
        R_ratio = np.exp(p["beta_1"] * (1.0 / T_kelvin - 1.0 / p["T_0"]))  # (n_est,)
        # Defect: the batch path clamped R_ratio < 0.1 silently while the scalar
        # noise_thermchannel warns; surface the broken-thermistor condition here too.
        n_clamped = int(np.count_nonzero(R_ratio < 0.1))
        if n_clamped:
            warnings.warn(
                f"R_ratio < 0.1 for {n_clamped} of {R_ratio.size} T_means "
                "(possible broken thermistor); clamped to 1.0",
                stacklevel=2,
            )
        R_ratio = np.where(R_ratio < 0.1, 1.0, R_ratio)
        # Scale factor
        scale_factor = T_kelvin**2 * (1 + R_ratio) ** 2 / (2 * eta * p["beta_1"] * R_ratio)
        if p["beta_2"] is not None and np.isfinite(p["beta_2"]):
            scale_factor = scale_factor * (1 + 2 * (p["beta_1"] / p["beta_2"]) * np.log(R_ratio))
    R_actual = R_ratio * p["R_0"]
    phi_R = 4 * p["K_B"] * R_actual * p["T_K"]  # (n_est,)

    Noise_counts = (
        base_f[np.newaxis, :] + johnson_gain_f[np.newaxis, :] * phi_R[:, np.newaxis]
    ) * counts_factor[np.newaxis, :]

    return Noise_counts * (scale_factor**2)[:, np.newaxis]


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
    speeds = np.asarray(speeds, dtype=np.float64)
    # Defect: non-positive speeds give a negative/inf noise PSD; MATLAB
    # gradT_noise_odas.m validates speed > 0 (val_speed).
    if not np.all(np.isfinite(speeds) & (speeds > 0)):
        raise ValueError(f"FP07 speeds must all be > 0, got min {np.min(speeds)!r}")
    noise_f = noise_thermchannel_batch(F, T_means, fs=fs, diff_gain=diff_gain, **kwargs)
    # Time-derivative noise -> gradient noise per cpm: /speed^2 * speed
    # (see gradT_noise).
    return noise_f / speeds[:, np.newaxis]
