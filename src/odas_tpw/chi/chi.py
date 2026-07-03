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

import functools
import warnings
from typing import NamedTuple

import numpy as np
from scipy.signal import butter, freqz

from odas_tpw.chi.batchelor import (
    KAPPA_T,
    Q_BATCHELOR,
    Q_KRAICHNAN,
    batchelor_grad,
    batchelor_kB,
    kraichnan_grad,
)


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


# ---------------------------------------------------------------------------
# Spectrum model dispatcher
# ---------------------------------------------------------------------------

# The variance-correction grid resolves the model spectrum by its OWN scale kB,
# not by K_max: a fixed point count spread over ~5*K_max leaves the steep
# Batchelor/Kraichnan peak near kB unresolved when kB << K_max (low-epsilon
# windows where K_max is pushed out near K_AA), biasing the correction by up to
# ~15%. Span 40*kB (the spectrum is negligible beyond that, so V_total is fully
# captured and the V_resolved mask can be clamped there) at ~2000 points per kB.
_K_SPAN_KB = 40  # grid extends to _K_SPAN_KB * kB
_PTS_PER_KB = 2000  # points per kB of span -> dK = kB / _PTS_PER_KB
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
    K_min: float = 0.0,
) -> float:
    """Compute V_total / V_resolved for the unresolved-variance correction.

    V_total is the model variance over all wavenumbers; V_resolved is the
    model variance within [K_min, K_max] *after* FP07 attenuation (|H|²),
    so the ratio simultaneously corrects for the unresolved band edges and
    for in-band sensor response.  The correction factor is independent of
    chi (linear scaling cancels in the ratio).  The grid resolution is set by
    kB (dK ~ kB/_PTS_PER_KB over [0, 40*kB]), giving <0.1% accuracy independent
    of how far K_max is pushed out — verified against a high-n reference for kB
    in [2, 400] including the K_min-masked iterative path.

    Returns correction factor, or NaN if integration fails.
    """
    # Tightened guard: kB=+inf passes ``kB > 0`` but makes K_fine all-inf, and
    # the inf-inf subtract in np.trapezoid leaks a RuntimeWarning (audit nit).
    if not (0 < kB < np.inf):
        return np.nan
    # Fixed dK ~ kB/_PTS_PER_KB so the peak near kB is always resolved; span
    # 40*kB so the model variance is fully captured.
    K_upper = _K_SPAN_KB * kB
    n = _K_SPAN_KB * _PTS_PER_KB
    K_fine = np.arange(1, n + 1, dtype=np.float64) * (K_upper / n)
    spec_fine = grad_func(K_fine, kB, 1.0)  # chi=1.0 (cancels in ratio)

    V_total = np.trapezoid(spec_fine, K_fine)
    if V_total <= 0:
        return np.nan

    F_fine = K_fine * speed
    H2_fine = _h2(F_fine, tau0)
    # Clamp the mask to the grid: past 40*kB the model spectrum is ~0, so a
    # larger K_max contributes nothing to V_resolved.
    mask = (K_fine >= K_min) & (K_fine <= min(K_max, K_upper))
    if not np.any(mask):
        return np.nan
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
    kappa_T: float = KAPPA_T,
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
    kappa_T : float
        Molecular thermal diffusivity [m²/s]. Per-window T/S/P value from
        :func:`odas_tpw.scor160.ocean.kappa_T`; defaults to the fixed
        ``batchelor.KAPPA_T`` for backward compatibility.

    Returns
    -------
    ChiEpsilonResult
        Named tuple: (chi, kB, K_max, spec_batch, fom, K_max_ratio).
    """
    grad_func, _q = _spectrum_func(spectrum_model)
    # Bind kappa_T into the spectrum shape so every grad_func(...) call here and
    # in _variance_correction uses the per-window value (its amplitude carries
    # chi/(kB*kappa_T)).
    grad_func = functools.partial(grad_func, kappa_T=kappa_T)

    kB = float(batchelor_kB(epsilon, nu, kappa_T))
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

    # Log-space least-squares fit for chi with kB fixed from epsilon.
    # Model: chi * Batchelor_unit(K) * H2(K) + noise(K).
    # Minimises Σ [log(model) - log(obs)]² over valid K, which penalises
    # over- and under-estimation symmetrically on the log-log plot.
    # This is more robust than the variance-correction approach when the
    # epsilon-derived kB doesn't perfectly match the temperature spectrum.
    B_unit = grad_func(K[valid], kB, 1.0)  # unit-chi Batchelor at valid K
    s = spec_obs[valid]
    h2v = H2[valid]
    nv = noise_K[valid]

    # Initial chi estimate from variance correction (used as grid centre)
    obs_var = np.trapezoid(s, K[valid])
    if obs_var <= 0:
        warnings.warn("Trial chi <= 0; observed variance too low", stacklevel=2)
        return ChiEpsilonResult(np.nan, kB, K_max, np.zeros_like(K), np.nan, np.nan)

    correction = _variance_correction(kB, K_max, speed, tau0, _h2, grad_func)
    if np.isfinite(correction):
        chi_vc = 6 * kappa_T * obs_var * correction
    else:
        chi_vc = 6 * kappa_T * obs_var

    # Vectorized grid search: 200 chi values spanning 4 decades around chi_vc
    chi_lo = max(chi_vc * 0.01, 1e-15)
    chi_hi = chi_vc * 100
    chi_grid = np.logspace(np.log10(chi_lo), np.log10(chi_hi), 200)
    # model shape: (200, n_valid)
    models = (
        chi_grid[:, np.newaxis] * B_unit[np.newaxis, :] * h2v[np.newaxis, :] + nv[np.newaxis, :]
    )
    models = np.maximum(models, 1e-30)
    log_s = np.log(np.maximum(s, 1e-30))
    cost = np.sum((np.log(models) - log_s[np.newaxis, :]) ** 2, axis=1)
    cost = np.where(np.isfinite(cost), cost, np.inf)

    if not np.all(np.isinf(cost)):
        i_min = int(np.argmin(cost))
        chi = float(chi_grid[i_min])
        # Parabolic refinement of the minimum in log10(chi): the grid is
        # ~4.7% coarse (200 points over 4 decades), a visible quantization
        # on the primary output.  The vertex of the parabola through the
        # minimum and its neighbours (uniform log spacing h) is
        # x1 + 0.5*h*(y0 - y2)/(y0 - 2*y1 + y2); curvature > 0 and
        # |offset| <= h are guaranteed when y1 is the strict minimum.
        if 0 < i_min < len(chi_grid) - 1:
            y0, y1, y2 = cost[i_min - 1 : i_min + 2]
            if np.all(np.isfinite([y0, y1, y2])):
                denom = y0 - 2.0 * y1 + y2
                if denom > 0:
                    h = np.log10(chi_grid[1] / chi_grid[0])
                    dx = 0.5 * h * (y0 - y2) / denom
                    chi = float(10.0 ** (np.log10(chi_grid[i_min]) + dx))
    else:
        chi = chi_vc

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
    kappa_T: float = KAPPA_T,
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
    grad_func = functools.partial(grad_func, kappa_T=kappa_T)

    kB_best, fit_mask, K_fit = _mle_find_kB(
        spec_obs,
        K,
        chi_obs,
        noise_K,
        H2,
        f_AA,
        speed,
        grad_func,
    )

    if not np.isfinite(kB_best):
        if len(K_fit) == 0:
            warnings.warn("Too few valid points for MLE fit", stacklevel=2)
        else:
            warnings.warn("All NLL values infinite; MLE fit failed", stacklevel=2)
        return ChiFitResult(np.nan, np.nan, np.nan, np.nan, np.zeros_like(K), np.nan, np.nan)

    K_max_fit = K_fit[-1]

    # Recover epsilon from kB
    epsilon = (2 * np.pi * kB_best) ** 4 * nu * kappa_T**2

    # Re-estimate chi with unresolved-variance correction (like Method 1).
    # Pass the same lower band edge that obs_var uses (K[fit_mask][0]) so
    # V_resolved spans the identical [K_fit_low, K_max] band: with the prior
    # default K_min=0 the correction included model variance in [0, K_fit_low]
    # that obs_var excludes, inflating V_resolved and biasing chi low for
    # low-epsilon windows (audit finding) — matching the _iterative_fit path.
    K_fit_low = float(K[fit_mask][0])
    correction = _variance_correction(
        kB_best,
        K_max_fit,
        speed,
        tau0,
        _h2,
        grad_func,
        K_min=K_fit_low,
    )
    if np.isfinite(correction):
        obs_var = np.trapezoid(np.maximum(spec_obs[fit_mask] - noise_K[fit_mask], 0), K[fit_mask])
        chi = 6 * kappa_T * obs_var * correction
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
    kappa_T: float = KAPPA_T,
) -> ChiFitResult:
    """Iterative MLE fitting (Peterson & Fer 2014, Method 2).

    Three iterations refining the LOWER integration limit
    ``k_l = max(K[1], 3*k_star)`` (with ``k_star = 0.04*kB*sqrt(kappa/nu)``,
    a viscous-convective onset scale; the 0.04 coefficient follows the
    original implementation and has not been verified against Peterson &
    Fer's text) and applying a model-based correction for the variance
    outside [``k_l``, ``k_u``] and for in-band FP07 attenuation (see
    :func:`_variance_correction`).  The UPPER limit ``k_u`` is fixed at
    the last valid wavenumber (noise/anti-aliasing criterion) and is not
    refined between iterations.

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
    grad_func = functools.partial(grad_func, kappa_T=kappa_T)

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
        * kappa_T
        * np.trapezoid(np.maximum(spec_obs[mask_init] - noise_K[mask_init], 0), K[mask_init])
    )

    if chi_obs <= 0:
        chi_obs = 1e-14

    # Iterative refinement (up to 3 iterations, with convergence check)
    kB_best = np.nan
    kB_prev = np.nan
    epsilon = np.nan
    # True once the refined integration band carries above-noise variance.
    # If it never does (every in-band bin has Phi_obs <= Phi_noise), the
    # window is below the detection limit and chi must be NaN, NOT the
    # 1e-14 sentinel that chi_obs falls back to for model shaping.
    band_has_signal = False

    for iteration in range(3):
        # MLE fit for kB (core search only — no variance correction)
        kB_best, _, _ = _mle_find_kB(
            spec_obs,
            K,
            chi_obs,
            noise_K,
            H2,
            f_AA,
            speed,
            grad_func,
        )

        if not np.isfinite(kB_best) or kB_best < 1:
            break

        # Check convergence
        if iteration > 0 and abs(kB_best - kB_prev) / kB_prev < 0.01:
            break
        kB_prev = kB_best

        # Refine integration limits
        k_star = 0.04 * kB_best * np.sqrt(kappa_T / nu)
        k_l_new = max(K[1], 3 * k_star)
        k_u_new = k_u

        # Recompute chi_obs with refined limits
        mask_refined = (k_l_new <= K) & (k_u_new >= K)
        if np.sum(mask_refined) < 3:
            break

        chi_band = (
            6
            * kappa_T
            * np.trapezoid(
                np.maximum(spec_obs[mask_refined] - noise_K[mask_refined], 0), K[mask_refined]
            )
        )

        # Model-based correction for variance outside [k_l, k_u] AND for
        # in-band FP07 attenuation (the observed spectrum is never divided
        # by |H|², so V_resolved in the ratio includes |H|² instead).
        # Peterson & Fer (2014) equivalently boost the measured spectrum
        # by the response function before integrating.
        correction = _variance_correction(
            kB_best, k_u_new, speed, tau0, _h2, grad_func, K_min=k_l_new
        )
        if chi_band > 0 and np.isfinite(correction):
            chi_obs = chi_band * correction
            band_has_signal = True
        elif chi_band > 0:
            chi_obs = chi_band
            band_has_signal = True
        else:
            chi_obs = 1e-14

    # Recompute chi_obs once from the *final* kB_best so the returned kB,
    # epsilon, chi and spec_batch are mutually consistent. On the convergence
    # break the loop exits before re-deriving chi_obs for the newly-converged
    # kB_best, leaving chi tied to the prior iteration's kB (audit finding:
    # k_l_new and the variance correction both depend on kB).
    if np.isfinite(kB_best) and kB_best >= 1:
        k_star = 0.04 * kB_best * np.sqrt(kappa_T / nu)
        k_l_final = max(K[1], 3 * k_star)
        mask_final = (k_l_final <= K) & (k_u >= K)
        if np.sum(mask_final) >= 3:
            chi_band = (
                6
                * kappa_T
                * np.trapezoid(
                    np.maximum(spec_obs[mask_final] - noise_K[mask_final], 0),
                    K[mask_final],
                )
            )
            correction = _variance_correction(
                kB_best, k_u, speed, tau0, _h2, grad_func, K_min=k_l_final
            )
            if chi_band > 0 and np.isfinite(correction):
                chi_obs = chi_band * correction
                band_has_signal = True
            elif chi_band > 0:
                chi_obs = chi_band
                band_has_signal = True

    # Final values
    if np.isfinite(kB_best):
        epsilon = (2 * np.pi * kB_best) ** 4 * nu * kappa_T**2
    # A finite chi requires BOTH a converged Batchelor/kB fit AND above-noise
    # variance in the integration band. A failed fit (kB NaN) leaks the initial
    # noise-dominated band integral; a below-detection window (band_has_signal
    # False) leaks the 1e-14 model-shaping sentinel. Either way fom=NaN and no
    # downstream QC cut can reach it, so return NaN — excluding it from
    # chi_final/chiMean (and thus K_T/Gamma), mirroring the epsilon side where a
    # failed shear fit yields NaN epsilon.
    chi = chi_obs if (np.isfinite(kB_best) and band_has_signal) else np.nan
    spec_batch = grad_func(K, kB_best, chi) if np.isfinite(kB_best) else np.zeros_like(K)

    # Figure of merit: observed vs attenuated model (Batchelor * H2 + noise).
    # Restrict to above-noise bins (spec_obs > 2*noise) bounded by the
    # integration limit k_u, so the FOM is computed over the same band the chi
    # was integrated over; otherwise noise-dominated bins dilute the ratio
    # toward 1.0. This band is intentionally NOT identical to Method 1 / the MLE
    # path (which use _valid_wavenumber_mask with a K_AA bound and a min-points
    # fallback): those affect only this diagnostic FOM, never chi/kB (#48).
    if np.isfinite(kB_best) and np.isfinite(chi):
        valid_fom = (spec_obs > 2 * noise_K) & (K > 0) & (k_u >= K)
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
    # Evaluate at the frequency points
    n = len(F)
    bl = np.ones(n)
    if n < 3:
        return bl
    # Normalized cutoff for the digital Butterworth. butter() requires
    # 0 < Wn < 1; a tiny diff_gain (e.g. a corrupt/patched .p config value)
    # pushes Wn >= 1 and scipy raises, aborting chi for the whole cast (audit
    # finding). Degrade to no bilinear correction (all-ones) instead.
    wn = 1 / (2 * np.pi * diff_gain * fs / 2)
    if not (0 < wn < 1):
        warnings.warn(
            f"bilinear cutoff Wn={wn:.3g} out of (0,1) for diff_gain={diff_gain:.3g}; "
            "skipping bilinear correction",
            stacklevel=2,
        )
        return bl
    # Digital Butterworth used in deconvolution
    b, a = butter(1, wn)
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
