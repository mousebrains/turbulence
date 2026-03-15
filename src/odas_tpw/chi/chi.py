# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Chi (thermal variance dissipation rate) calculation.

Two methods:
1. Chi from known epsilon (shear probes) — Dillon & Caldwell 1980
2. Chi without epsilon (Batchelor spectrum fitting) — Ruddick et al. 2000,
   Peterson & Fer 2014

References
----------
Dillon, T.M. and D.R. Caldwell, 1980: The Batchelor spectrum and dissipation
    in the upper ocean. J. Geophys. Res., 85, 1910-1916.
Ruddick, B., A. Anis, and K. Thompson, 2000: Maximum likelihood spectral
    fitting: The Batchelor spectrum. J. Atmos. Oceanic Technol., 17, 1541-1555.
Peterson, A.K. and I. Fer, 2014: Dissipation measurements using temperature
    microstructure from an underwater glider. Methods in Oceanography, 10, 44-69.
"""

import warnings
from typing import NamedTuple

import numpy as np
from scipy.signal import butter, freqz


class ChiEpsilonResult(NamedTuple):
    """Result from Method 1: chi from known epsilon."""

    chi: float
    kB: float
    K_max: float
    spec_batch: np.ndarray
    fom: float
    K_max_ratio: float


class ChiFitResult(NamedTuple):
    """Result from Methods 2 & 3: Batchelor spectrum fitting."""

    kB: float
    chi: float
    epsilon: float
    K_max: float
    spec_batch: np.ndarray
    fom: float
    K_max_ratio: float

from odas_tpw.chi.batchelor import (
    KAPPA_T,
    Q_BATCHELOR,
    Q_KRAICHNAN,
    batchelor_grad,
    batchelor_kB,
    kraichnan_grad,
)

# ---------------------------------------------------------------------------
# Spectrum model dispatcher
# ---------------------------------------------------------------------------

_N_FINE = 2000  # Points for fine-grid variance correction (was 10000)
_KB_COARSE = np.logspace(0, 4.5, 100)  # Coarse MLE grid: 1 to ~31623 cpm
_KB_COARSE_2D = _KB_COARSE[:, np.newaxis]  # (100, 1) for broadcasting


def _spectrum_func(model: str) -> tuple:
    """Return (grad_func, q_constant) for the named spectrum model."""
    if model == "batchelor":
        return batchelor_grad, Q_BATCHELOR
    elif model == "kraichnan":
        return kraichnan_grad, Q_KRAICHNAN
    else:
        raise ValueError(f"Unknown spectrum model: {model!r}")


def _variance_correction(
    kB: float,
    K_max: float,
    speed: float,
    tau0: float,
    _h2,
    grad_func,
    n_fine: int = _N_FINE,
) -> float:
    """Compute V_total / V_resolved for the unresolved-variance correction.

    The correction factor is independent of chi (linear scaling cancels
    in the ratio).  Uses a 2000-point grid (sufficient for <0.1% accuracy).

    Returns correction factor, or NaN if integration fails.
    """
    K_upper = max(K_max * 5, kB * 5)
    K_fine = np.linspace(K_upper / n_fine * 0.01, K_upper, n_fine)
    spec_fine = grad_func(K_fine, kB, 1.0)  # chi=1.0 (cancels in ratio)

    V_total = np.trapezoid(spec_fine, K_fine)
    if V_total <= 0:
        return np.nan

    F_fine = K_fine * speed
    H2_fine = _h2(F_fine, tau0)
    mask = K_fine <= K_max
    V_resolved = np.trapezoid(spec_fine[mask] * H2_fine[mask], K_fine[mask])

    if V_resolved <= 0:
        return np.nan
    return float(V_total / V_resolved)


# ---------------------------------------------------------------------------
# Shared wavenumber masking helper
# ---------------------------------------------------------------------------


def _valid_wavenumber_mask(
    spec_obs: np.ndarray,
    noise: np.ndarray,
    K: np.ndarray,
    K_AA: float,
    min_points: int = 3,
) -> np.ndarray:
    """Build a boolean mask for valid wavenumber points.

    Selects points where the observed spectrum exceeds 2x noise, K > 0,
    and K <= K_AA.  Falls back to K > 0 & K <= K_AA if too few points
    pass the noise criterion.

    Parameters
    ----------
    spec_obs : ndarray
        Observed spectrum.
    noise : ndarray
        Noise floor spectrum (same units as spec_obs).
    K : ndarray
        Wavenumber vector [cpm].
    K_AA : float
        Anti-aliasing wavenumber limit [cpm].
    min_points : int
        Minimum valid points required.

    Returns
    -------
    mask : ndarray of bool
    """
    above_noise = spec_obs > 2 * noise
    mask = above_noise & (K > 0) & (K <= K_AA)
    if np.sum(mask) < min_points:
        mask = (K > 0) & (K <= K_AA)
    return mask


# ---------------------------------------------------------------------------
# Method 1: Chi from known epsilon
# ---------------------------------------------------------------------------


def _chi_from_epsilon(
    spec_obs: np.ndarray,
    K: np.ndarray,
    epsilon: float,
    nu: float,
    noise_K: np.ndarray,
    H2: np.ndarray,
    tau0: float,
    _h2,
    f_AA: float,
    speed: float,
    spectrum_model: str,
) -> ChiEpsilonResult:
    """Compute chi for one probe in one window, given epsilon (Method 1).

    Parameters
    ----------
    spec_obs : ndarray
        Observed temperature gradient spectrum [(K/m)²/cpm].
    K : ndarray
        Wavenumber vector [cpm].
    epsilon : float
        TKE dissipation rate [W/kg] from shear probes.
    nu : float
        Kinematic viscosity [m²/s].
    noise_K : ndarray
        Pre-computed noise spectrum [(K/m)²/cpm].
    H2 : ndarray
        Pre-computed FP07 transfer function |H(f)|².
    tau0 : float
        Pre-computed FP07 time constant [s].
    _h2 : callable
        FP07 transfer function (fp07_transfer or fp07_double_pole).
    f_AA : float
        Anti-aliasing cutoff [Hz].
    speed : float
        Profiling speed [m/s].
    spectrum_model : str
        Theoretical spectrum model ('batchelor' or 'kraichnan').

    Returns
    -------
    ChiEpsilonResult
        Named tuple: (chi, kB, K_max, spec_batch, fom, K_max_ratio).
    """
    grad_func, _q = _spectrum_func(spectrum_model)

    kB = float(batchelor_kB(epsilon, nu))
    if kB < 1:
        warnings.warn(f"kB={kB:.1f} < 1 cpm; epsilon too low for chi estimation", stacklevel=2)
        return ChiEpsilonResult(np.nan, kB, np.nan, np.zeros_like(K), np.nan, np.nan)

    # Find integration limit: where observed meets noise
    K_AA = f_AA / speed
    valid = _valid_wavenumber_mask(spec_obs, noise_K, K, K_AA, min_points=3)
    if np.sum(valid) < 3:
        warnings.warn("Too few valid wavenumber points for chi integration", stacklevel=2)
        return ChiEpsilonResult(np.nan, kB, np.nan, np.zeros_like(K), np.nan, np.nan)

    valid_idx = np.where(valid)[0]
    K_max = K[valid_idx[-1]]

    # Integrate observed spectrum
    obs_var = np.trapezoid(spec_obs[valid], K[valid])

    # Correction factor for FP07 rolloff and unresolved variance
    # (chi-independent: Batchelor spectrum scales linearly with chi)
    if obs_var <= 0:
        warnings.warn("Trial chi <= 0; observed variance too low", stacklevel=2)
        return ChiEpsilonResult(np.nan, kB, K_max, np.zeros_like(K), np.nan, np.nan)

    correction = _variance_correction(kB, K_max, speed, tau0, _h2, grad_func)
    if not np.isfinite(correction):
        warnings.warn("Batchelor variance non-positive; cannot compute correction", stacklevel=2)
        return ChiEpsilonResult(np.nan, kB, K_max, np.zeros_like(K), np.nan, np.nan)

    chi = 6 * KAPPA_T * obs_var * correction

    # Compute fitted Batchelor spectrum for output
    spec_batch = grad_func(K, kB, chi)

    # Figure of merit: observed vs attenuated model (Batchelor * H2 + noise)
    if np.sum(valid) >= 3 and np.isfinite(chi):
        mod_v = np.trapezoid(spec_batch[valid] * H2[valid] + noise_K[valid], K[valid])
        fom = obs_var / mod_v if mod_v > 0 else np.nan
    else:
        fom = np.nan

    K_max_ratio_val = K_max / kB if kB > 0 else np.nan

    return ChiEpsilonResult(chi, kB, K_max, spec_batch, fom, K_max_ratio_val)


# ---------------------------------------------------------------------------
# Method 2: MLE spectral fitting (Ruddick et al. 2000)
# ---------------------------------------------------------------------------


def _mle_find_kB(
    spec_obs: np.ndarray,
    K: np.ndarray,
    chi_obs: float,
    noise_K: np.ndarray,
    H2: np.ndarray,
    f_AA: float,
    speed: float,
    grad_func,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Core MLE grid search for kB (no chi correction or output spectra).

    Returns
    -------
    kB_best : float
        Best-fit Batchelor wavenumber [cpm], or NaN if fit fails.
    fit_mask : ndarray of bool
        Valid wavenumber mask used for fitting.
    K_fit : ndarray
        Wavenumber vector at fit points.
    """
    K_AA = f_AA / speed
    fit_mask = _valid_wavenumber_mask(spec_obs, noise_K, K, K_AA, min_points=6)
    if np.sum(fit_mask) < 6:
        return np.nan, fit_mask, np.empty(0)

    fit_idx = np.where(fit_mask)[0]
    K_fit = K[fit_idx]
    spec_fit = spec_obs[fit_idx]
    H2_fit = H2[fit_idx]
    noise_fit = noise_K[fit_idx]

    # --- Coarse grid search (vectorized) ---
    spec_models = grad_func(K_fit, _KB_COARSE_2D, chi_obs) * H2_fit + noise_fit
    spec_models = np.maximum(spec_models, 1e-30)
    nll_coarse = np.sum(np.log(spec_models) + spec_fit / spec_models, axis=1)
    nll_coarse = np.where(np.isfinite(nll_coarse), nll_coarse, np.inf)

    if np.all(np.isinf(nll_coarse)):
        return np.nan, fit_mask, K_fit

    best_coarse = _KB_COARSE[np.argmin(nll_coarse)]

    # --- Fine grid search (vectorized) ---
    kB_lo = max(best_coarse * 0.5, 1.0)
    kB_hi = best_coarse * 2.0
    kB_fine = np.linspace(kB_lo, kB_hi, 100)
    kB_fine_2d = kB_fine[:, np.newaxis]
    spec_models = grad_func(K_fit, kB_fine_2d, chi_obs) * H2_fit + noise_fit
    spec_models = np.maximum(spec_models, 1e-30)
    nll_fine = np.sum(np.log(spec_models) + spec_fit / spec_models, axis=1)
    nll_fine = np.where(np.isfinite(nll_fine), nll_fine, np.inf)

    kB_best = best_coarse if np.all(np.isinf(nll_fine)) else kB_fine[np.argmin(nll_fine)]
    return kB_best, fit_mask, K_fit


def _mle_fit_kB(
    spec_obs: np.ndarray,
    K: np.ndarray,
    chi_obs: float,
    nu: float,
    noise_K: np.ndarray,
    H2: np.ndarray,
    tau0: float,
    _h2,
    f_AA: float,
    speed: float,
    spectrum_model: str,
) -> ChiFitResult:
    """Maximum-likelihood fit for Batchelor wavenumber kB (Ruddick et al. 2000).

    Performs a coarse+fine grid search over kB to minimise the MLE
    negative log-likelihood for chi-squared spectral estimates.

    Parameters
    ----------
    spec_obs : ndarray
        Observed temperature gradient spectrum [(K/m)²/cpm].
    K : ndarray
        Wavenumber vector [cpm].
    chi_obs : float
        Initial chi estimate [K²/s] from integration.
    nu : float
        Kinematic viscosity [m²/s].
    noise_K, H2, tau0, _h2, f_AA, speed, spectrum_model
        Same as :func:`_chi_from_epsilon`.

    Returns
    -------
    ChiFitResult
        Named tuple: (kB, chi, epsilon, K_max, spec_batch, fom, K_max_ratio).
    """
    grad_func, _q = _spectrum_func(spectrum_model)

    kB_best, fit_mask, K_fit = _mle_find_kB(
        spec_obs, K, chi_obs, noise_K, H2, f_AA, speed, grad_func,
    )

    if not np.isfinite(kB_best):
        if len(K_fit) == 0:
            warnings.warn("Too few valid points for MLE fit", stacklevel=2)
        else:
            warnings.warn("All NLL values infinite; MLE fit failed", stacklevel=2)
        return ChiFitResult(np.nan, np.nan, np.nan, np.nan, np.zeros_like(K), np.nan, np.nan)

    K_max_fit = K_fit[-1]

    # Recover epsilon from kB
    epsilon = (2 * np.pi * kB_best) ** 4 * nu * KAPPA_T**2

    # Re-estimate chi with unresolved-variance correction (like Method 1)
    correction = _variance_correction(
        kB_best, K_max_fit, speed, tau0, _h2, grad_func,
    )
    if np.isfinite(correction):
        obs_var = np.trapezoid(spec_obs[fit_mask], K[fit_mask])
        chi = 6 * KAPPA_T * obs_var * correction
    else:
        chi = chi_obs

    # Fitted spectrum for output
    spec_batch = grad_func(K, kB_best, chi)

    # Figure of merit: observed vs attenuated model (Batchelor * H2 + noise)
    fit_idx = np.where(fit_mask)[0]
    if np.isfinite(chi) and len(fit_idx) >= 3:
        mod_v = np.trapezoid(spec_batch[fit_idx] * H2[fit_idx] + noise_K[fit_idx], K_fit)
        obs_v = np.trapezoid(spec_obs[fit_idx], K_fit)
        fom = obs_v / mod_v if mod_v > 0 else np.nan
    else:
        fom = np.nan

    K_max_ratio_val = K_max_fit / kB_best if kB_best > 0 else np.nan

    return ChiFitResult(kB_best, chi, epsilon, K_max_fit, spec_batch, fom, K_max_ratio_val)


def _iterative_fit(
    spec_obs: np.ndarray,
    K: np.ndarray,
    nu: float,
    noise_K: np.ndarray,
    H2: np.ndarray,
    tau0: float,
    _h2,
    f_AA: float,
    speed: float,
    spectrum_model: str,
) -> ChiFitResult:
    """Iterative MLE fitting (Peterson & Fer 2014, Method 2).

    Three iterations refining integration limits and correcting for
    unresolved variance below ``k_l`` and above ``k_u``.

    Parameters
    ----------
    spec_obs : ndarray
        Observed temperature gradient spectrum [(K/m)²/cpm].
    K : ndarray
        Wavenumber vector [cpm].
    nu : float
        Kinematic viscosity [m²/s].
    noise_K, H2, tau0, _h2, f_AA, speed, spectrum_model
        Same as :func:`_chi_from_epsilon`.

    Returns
    -------
    ChiFitResult
        Named tuple: (kB, chi, epsilon, K_max, spec_batch, fom, K_max_ratio).
    """
    grad_func, _q = _spectrum_func(spectrum_model)

    # Initial integration limits
    K_AA = f_AA / speed
    valid = _valid_wavenumber_mask(spec_obs, noise_K, K, K_AA, min_points=6)

    valid_idx = np.where(valid)[0]
    if len(valid_idx) < 6:
        warnings.warn("Too few valid points for iterative fit", stacklevel=2)
        return ChiFitResult(np.nan, np.nan, np.nan, np.nan, np.zeros_like(K), np.nan, np.nan)

    k_u = K[valid_idx[-1]]

    # Initial chi estimate
    mask_init = valid & (spec_obs > noise_K)
    if np.sum(mask_init) < 3:
        mask_init = valid
    chi_obs = (
        6
        * KAPPA_T
        * np.trapezoid(np.maximum(spec_obs[mask_init] - noise_K[mask_init], 0), K[mask_init])
    )

    if chi_obs <= 0:
        chi_obs = 1e-14

    # Iterative refinement (up to 3 iterations, with convergence check)
    kB_best = np.nan
    kB_prev = np.nan
    epsilon = np.nan

    for iteration in range(3):
        # MLE fit for kB (core search only — no variance correction)
        kB_best, _, _ = _mle_find_kB(
            spec_obs, K, chi_obs, noise_K, H2, f_AA, speed, grad_func,
        )

        if not np.isfinite(kB_best) or kB_best < 1:
            break

        # Check convergence
        if iteration > 0 and abs(kB_best - kB_prev) / kB_prev < 0.01:
            break
        kB_prev = kB_best

        # Refine integration limits
        k_star = 0.04 * kB_best * np.sqrt(KAPPA_T / nu)
        k_l_new = max(K[1], 3 * k_star)
        k_u_new = k_u

        # Recompute chi_obs with refined limits
        mask_refined = (k_l_new <= K) & (k_u_new >= K)
        if np.sum(mask_refined) < 3:
            break

        chi_obs_new = (
            6
            * KAPPA_T
            * np.trapezoid(
                np.maximum(spec_obs[mask_refined] - noise_K[mask_refined], 0), K[mask_refined]
            )
        )

        # Unresolved variance from Batchelor model
        K_fine = np.linspace(K[1] * 0.01, kB_best * 5, _N_FINE)
        chi_use = chi_obs_new if chi_obs_new > 0 else chi_obs
        spec_fine = grad_func(K_fine, kB_best, chi_use)

        chi_low = 6 * KAPPA_T * np.trapezoid(spec_fine[K_fine < k_l_new], K_fine[K_fine < k_l_new])
        chi_high = 6 * KAPPA_T * np.trapezoid(spec_fine[K_fine > k_u_new], K_fine[K_fine > k_u_new])

        chi_obs = max(chi_obs_new, 0) + chi_low + chi_high

        if chi_obs <= 0:
            chi_obs = 1e-14

    # Final values
    if np.isfinite(kB_best):
        epsilon = (2 * np.pi * kB_best) ** 4 * nu * KAPPA_T**2
    chi = chi_obs
    spec_batch = grad_func(K, kB_best, chi) if np.isfinite(kB_best) else np.zeros_like(K)

    # Figure of merit: observed vs attenuated model (Batchelor * H2 + noise)
    if np.isfinite(kB_best) and np.isfinite(chi):
        valid_fom = (K > 0) & (k_u >= K)
        if np.sum(valid_fom) >= 3:
            obs_v = np.trapezoid(spec_obs[valid_fom], K[valid_fom])
            mod_v = np.trapezoid(
                spec_batch[valid_fom] * H2[valid_fom] + noise_K[valid_fom],
                K[valid_fom],
            )
            fom = obs_v / mod_v if mod_v > 0 else np.nan
        else:
            fom = np.nan
    else:
        fom = np.nan

    K_max_ratio_val = k_u / kB_best if np.isfinite(kB_best) and kB_best > 0 else np.nan

    return ChiFitResult(kB_best, chi, epsilon, k_u, spec_batch, fom, K_max_ratio_val)


def _bilinear_correction(F: np.ndarray, diff_gain: float, fs: float) -> np.ndarray:
    """Bilinear transform correction for deconvolution filter.

    Matches ODAS get_scalar_spectra_odas.m lines 264-270:
    corrects for the difference between the ideal analog LP filter
    H = 1/(1+(2*pi*f*diff_gain)^2) and the actual digital Butterworth
    used in deconvolution.
    """
    # Digital Butterworth used in deconvolution
    b, a = butter(1, 1 / (2 * np.pi * diff_gain * fs / 2))
    # Evaluate at the frequency points
    n = len(F)
    bl = np.ones(n)
    if n < 3:
        return bl
    # Compute for indices 1..N-2 (exclude DC and Nyquist), matching ODAS
    idx = np.arange(1, n - 1)
    F_eval = F[idx]
    # freqz expects normalized frequency (0 to pi)
    w_norm = 2 * np.pi * F_eval / fs
    _, h = freqz(b, a, worN=w_norm)
    H_digital = np.abs(h) ** 2
    # Ideal analog response
    H_analog = 1.0 / (1 + (2 * np.pi * F_eval * diff_gain) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        bl[idx] = H_analog / H_digital
    bl = np.where(np.isfinite(bl), bl, 1.0)
    # Copy second-to-last value to Nyquist bin (ODAS convention)
    bl[-1] = bl[-2]
    return bl


