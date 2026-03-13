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
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import xarray as xr

if TYPE_CHECKING:
    from rsi_python.p_file import PFile

from scipy.signal import butter, freqz

from chi_tpw.batchelor import (
    KAPPA_T,
    Q_BATCHELOR,
    Q_KRAICHNAN,
    batchelor_grad,
    batchelor_kB,
    kraichnan_grad,
)
from chi_tpw.fp07 import (
    fp07_double_pole,
    fp07_transfer,
    gradT_noise_batch,
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


def _default_tau_model(fp07_model: str) -> str:
    """Auto-select FP07 tau model based on transfer function model.

    double_pole pairs with Goto (tau=0.003); single_pole with Lueck.
    """
    return "goto" if fp07_model == "double_pole" else "lueck"


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
) -> tuple:
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
    chi : float
        Thermal variance dissipation rate [K²/s].
    kB : float
        Batchelor wavenumber [cpm].
    K_max : float
        Upper integration limit [cpm].
    spec_batch : ndarray
        Fitted Batchelor spectrum.
    fom : float
        Figure of merit (observed/model variance ratio).
    K_max_ratio : float
        K_max / kB spectral resolution ratio.
    """
    grad_func, _q = _spectrum_func(spectrum_model)

    kB = float(batchelor_kB(epsilon, nu))
    if kB < 1:
        warnings.warn(f"kB={kB:.1f} < 1 cpm; epsilon too low for chi estimation", stacklevel=2)
        return np.nan, kB, np.nan, np.zeros_like(K), np.nan, np.nan

    # Find integration limit: where observed meets noise
    K_AA = f_AA / speed
    valid = _valid_wavenumber_mask(spec_obs, noise_K, K, K_AA, min_points=3)
    if np.sum(valid) < 3:
        warnings.warn("Too few valid wavenumber points for chi integration", stacklevel=2)
        return np.nan, kB, np.nan, np.zeros_like(K), np.nan, np.nan

    valid_idx = np.where(valid)[0]
    K_max = K[valid_idx[-1]]

    # Integrate observed spectrum
    obs_var = np.trapezoid(spec_obs[valid], K[valid])

    # Correction factor for FP07 rolloff and unresolved variance
    # (chi-independent: Batchelor spectrum scales linearly with chi)
    if obs_var <= 0:
        warnings.warn("Trial chi <= 0; observed variance too low", stacklevel=2)
        return np.nan, kB, K_max, np.zeros_like(K), np.nan, np.nan

    correction = _variance_correction(kB, K_max, speed, tau0, _h2, grad_func)
    if not np.isfinite(correction):
        warnings.warn("Batchelor variance non-positive; cannot compute correction", stacklevel=2)
        return np.nan, kB, K_max, np.zeros_like(K), np.nan, np.nan

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

    return chi, kB, K_max, spec_batch, fom, K_max_ratio_val


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
) -> tuple[float, float, float, float, np.ndarray, float, float]:
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
    kB_best : float
        Best-fit Batchelor wavenumber [cpm].
    chi : float
        Chi corrected for unresolved variance [K²/s].
    epsilon : float
        Epsilon recovered from kB [W/kg].
    K_max : float
        Upper fit limit [cpm].
    spec_batch : ndarray
        Fitted Batchelor spectrum.
    fom : float
        Figure of merit.
    K_max_ratio : float
        K_max / kB ratio.
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
        return np.nan, np.nan, np.nan, np.nan, np.zeros_like(K), np.nan, np.nan

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

    return kB_best, chi, epsilon, K_max_fit, spec_batch, fom, K_max_ratio_val


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
) -> tuple[float, float, float, float, np.ndarray, float, float]:
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
    kB, chi, epsilon, K_max, spec_batch, fom, K_max_ratio
        Same as :func:`_mle_fit_kB`.
    """
    grad_func, _q = _spectrum_func(spectrum_model)

    # Initial integration limits
    K_AA = f_AA / speed
    valid = _valid_wavenumber_mask(spec_obs, noise_K, K, K_AA, min_points=6)

    valid_idx = np.where(valid)[0]
    if len(valid_idx) < 6:
        warnings.warn("Too few valid points for iterative fit", stacklevel=2)
        return np.nan, np.nan, np.nan, np.nan, np.zeros_like(K), np.nan, np.nan

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

    return kB_best, chi, epsilon, k_u, spec_batch, fom, K_max_ratio_val


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def get_chi(
    source: "PFile | str | Path",
    epsilon_ds: xr.Dataset | None = None,
    fft_length: int = 512,
    diss_length: int | None = None,
    overlap: int | None = None,
    speed: float | None = None,
    direction: str = "down",
    fp07_model: str = "single_pole",
    goodman: bool = True,
    f_AA: float = 98.0,
    fit_method: str = "iterative",
    spectrum_model: str = "kraichnan",
    salinity: npt.ArrayLike | None = None,
) -> list[xr.Dataset]:
    """Compute chi from temperature gradient spectra.

    Parameters
    ----------
    source : PFile, str, or Path
        Input data (same multi-source support as get_diss).
    epsilon_ds : xarray.Dataset or None
        If provided: Method 1 — use known epsilon from shear probes.
        If None: Method 2 — fit Batchelor spectrum for kB.
    fft_length : int
        FFT segment length [samples], default 512.
    diss_length : int or None
        Dissipation window length [samples], default 3 * fft_length.
    overlap : int or None
        Window overlap [samples], default diss_length // 2.
    speed : float or None
        Fixed profiling speed [m/s]. If None, from dP/dt.
    direction : str
        'up' or 'down' for speed sign convention.
    fp07_model : str
        'single_pole' or 'double_pole' for FP07 transfer function.
    goodman : bool
        Apply Goodman coherent noise removal using accelerometers (default True).
    f_AA : float
        Anti-aliasing filter cutoff [Hz].
    fit_method : str
        Method 2 only: 'mle' (Ruddick et al. 2000) or 'iterative'
        (Peterson & Fer 2014).
    spectrum_model : str
        'batchelor' or 'kraichnan'.
    salinity : float or array_like or None
        Practical salinity [PSU]. If provided, uses gsw-based viscosity
        instead of visc35. Scalar or array matching slow time series.

    Returns
    -------
    list of xarray.Dataset, one per profile, with variables:
        chi         (probe, time) — thermal dissipation rate [K^2/s]
        epsilon_T   (probe, time) — epsilon from T (Method 2) or input epsilon
        kB          (probe, time) — Batchelor wavenumber [cpm]
        K_max_T     (probe, time) — integration limit [cpm]
        fom         (probe, time) — figure of merit (obs/model variance)
        K_max_ratio (probe, time) — K_max / kB (spectral resolution)
        speed       (time) — profiling speed [m/s]
        nu          (time) — kinematic viscosity [m^2/s]
        T_mean      (time) — mean temperature [deg C]
        P_mean      (time) — mean pressure [dbar]
        spec_gradT  (probe, freq, time) — temperature gradient spectra
        spec_batch  (probe, freq, time) — fitted Batchelor spectra
        spec_noise  (probe, freq, time) — noise spectra (per-probe)
        K           (freq, time) — wavenumber vectors
    """
    if diss_length is None:
        diss_length = 3 * fft_length
    if overlap is None:
        overlap = diss_length // 2

    # Load channels including thermistor data
    data = _load_therm_channels(source)

    therm_names = [t[0] for t in data["therm"]]
    therm_arrays = [t[1] for t in data["therm"]]
    n_therm = len(therm_arrays)
    diff_gains = data.get("diff_gains", [0.94] * n_therm)
    therm_cal = data.get("therm_cal", [{}] * n_therm)

    # Accelerometer arrays for scalar Goodman cleaning
    accel_arrays = [a[1] for a in data.get("accel", [])]
    if not goodman:
        accel_arrays = []

    if n_therm == 0:
        raise ValueError("No thermistor gradient channels found")

    # Profile detection, speed, P/T interpolation, salinity
    from rsi_python.helpers import prepare_profiles
    from rsi_python.profile import _VEHICLE_TAU

    vehicle = data.get("vehicle", "")
    tau = _VEHICLE_TAU.get(vehicle, 1.5)
    prepared = prepare_profiles(data, speed, direction, salinity, tau=tau)
    if prepared is None:
        return []
    (profiles_slow, speed_fast, P_fast, T_fast, sal_fast, fs_fast, _fs_slow, ratio, t_fast) = (
        prepared
    )

    f_AA_eff = 0.9 * f_AA

    results = []
    for s_slow, e_slow in profiles_slow:
        s_fast = s_slow * ratio
        e_fast = min((e_slow + 1) * ratio, len(t_fast))

        therm_prof = [arr[s_fast:e_fast] for arr in therm_arrays]
        accel_prof = [arr[s_fast:e_fast] for arr in accel_arrays] if accel_arrays else None
        p_prof = P_fast[s_fast:e_fast]
        t_prof = T_fast[s_fast:e_fast]
        spd_prof = speed_fast[s_fast:e_fast]
        time_prof = t_fast[s_fast:e_fast]

        sal_prof = sal_fast[s_fast:e_fast] if isinstance(sal_fast, np.ndarray) else sal_fast

        ds = _compute_profile_chi(
            therm_prof,
            therm_names,
            diff_gains,
            p_prof,
            t_prof,
            spd_prof,
            time_prof,
            fs_fast,
            fft_length,
            diss_length,
            overlap,
            f_AA_eff,
            fp07_model,
            spectrum_model,
            fit_method,
            epsilon_ds,
            salinity=sal_prof,
            therm_cal=therm_cal,
            accel_arrays=accel_prof,
        )
        ds.attrs.update(data["metadata"])
        ds.attrs["history"] = (
            f"Computed with rsi-python on {datetime.now(UTC).isoformat()}"
        )
        # CF time coordinate attributes
        start_time = data["metadata"].get("start_time", "")
        t_units = f"seconds since {start_time}" if start_time else "seconds"
        ds.coords["t"].attrs.update(
            {
                "standard_name": "time",
                "long_name": "time of chi estimate",
                "units": t_units,
                "calendar": "standard",
                "axis": "T",
            }
        )
        results.append(ds)

    return results


def _extract_therm_cal(ch_cfg: dict[str, Any]) -> dict[str, float]:
    """Extract thermistor calibration parameters from PFile channel config."""
    cal = {}
    for key in ("e_b", "b", "g", "beta_1", "beta_2", "adc_fs", "adc_bits", "T_0"):
        val = ch_cfg.get(key)
        if val is not None:
            cal[key] = float(val)
    # Map 'g' to 'gain' for noise_thermchannel
    if "g" in cal:
        cal["gain"] = cal.pop("g")
    return cal


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


def _load_therm_channels(source: "PFile | str | Path") -> dict[str, Any]:
    """Load thermistor gradient channels from any source.

    Looks for channels matching T*_dT* pattern (pre-emphasized temperature),
    or falls back to T1/T2 channels for first-difference gradient.
    """
    import re

    from rsi_python.helpers import load_channels

    # First load standard channels
    data = load_channels(source)

    # Now find thermistor channels
    from rsi_python.p_file import PFile

    therm = []
    diff_gains = []

    if isinstance(source, PFile):
        pf = source
    elif Path(source).suffix.lower() == ".p":
        pf = PFile(source)
    else:
        pf = None

    therm_cal = []  # per-probe calibration dicts for noise model

    if pf is not None:
        # Look for pre-emphasized channels (T1_dT1, T2_dT2)
        dT_re = re.compile(r"^T\d+_dT\d+$")
        T_re = re.compile(r"^T\d+$")

        for name in sorted(pf._fast_channels):
            if dT_re.match(name):
                therm.append((name, pf.channels[name]))
                # Get diff_gain from config
                ch_cfg: dict = next(
                    (ch for ch in pf.config["channels"] if ch.get("name") == name),
                    {},
                )
                dg = ch_cfg.get("diff_gain", "0.94")
                diff_gains.append(float(dg))
                # Extract calibration from base channel (T1 for T1_dT1)
                m = re.match(r"^(\w+)_d\1$", name)
                base_name = m.group(1) if m else name
                base_cfg: dict = next(
                    (ch for ch in pf.config["channels"] if ch.get("name") == base_name),
                    {},
                )
                therm_cal.append(_extract_therm_cal(base_cfg))

        # If no pre-emphasized channels, use T channels with first-difference
        if not therm:
            for name in sorted(pf._fast_channels):
                if T_re.match(name):
                    # Compute gradient via first-difference
                    T_data = pf.channels[name]
                    therm.append((name, T_data))
                    diff_gains.append(0.94)
                    ch_cfg = next(
                        (ch for ch in pf.config["channels"] if ch.get("name") == name),
                        {},
                    )
                    therm_cal.append(_extract_therm_cal(ch_cfg))
    else:
        # NetCDF source — look for gradient channels or T channels
        import netCDF4 as nc

        ds = nc.Dataset(str(source), "r")
        dT_re = re.compile(r"^T\d+_dT\d+$")
        T_re = re.compile(r"^T\d+$")

        for vname in sorted(ds.variables.keys()):
            var = ds.variables[vname]
            if var.dimensions == ("time_fast",):
                arr = var[:].data.astype(np.float64)
                if dT_re.match(vname):
                    therm.append((vname, arr))
                    diff_gains.append(0.94)

        if not therm:
            for vname in sorted(ds.variables.keys()):
                var = ds.variables[vname]
                if var.dimensions == ("time_fast",) and T_re.match(vname):
                    therm.append((vname, var[:].data.astype(np.float64)))
                    diff_gains.append(0.94)

        ds.close()

    data["therm"] = therm
    data["diff_gains"] = diff_gains
    data["therm_cal"] = therm_cal if therm_cal else [{}] * len(therm)
    return data


def _compute_profile_chi(
    therm_arrays,
    therm_names,
    diff_gains,
    P,
    T,
    speed,
    t_fast,
    fs_fast,
    fft_length,
    diss_length,
    overlap,
    f_AA,
    fp07_model,
    spectrum_model,
    fit_method,
    epsilon_ds,
    salinity=None,
    therm_cal=None,
    accel_arrays=None,
):
    """Compute chi for all windows in a single profile.

    Iterates over dissipation windows, computes temperature gradient spectra
    (with optional Goodman cleaning), and fits Batchelor/Kraichnan models
    using Method 1 (from epsilon) or Method 2 (MLE/iterative).

    Returns an xarray.Dataset with chi, kB, spectra, and QC variables.
    """
    N = len(P)
    n_therm = len(therm_arrays)

    # Convert temperature to spatial gradient (first-difference method).
    # Matches MATLAB make_gradT_odas: gradT = fs * diff(T) / speed.
    # therm_arrays contains temperature from convert_therm on pre-emphasized
    # channels; the first-difference approximates dT/dt, dividing by speed
    # gives dT/dz in K/m.
    spd_safe = np.maximum(np.abs(speed), 0.01)
    for ci in range(n_therm):
        dTdt = np.diff(therm_arrays[ci]) * fs_fast  # dT/dt [K/s]
        dTdt = np.append(dTdt, dTdt[-1])  # pad to original length
        therm_arrays[ci] = dTdt / spd_safe  # dT/dz [K/m]
    n_freq = fft_length // 2 + 1

    if overlap >= diss_length:
        overlap = diss_length // 2
    step = diss_length - overlap
    n_est = max(1, 1 + (N - diss_length) // step)

    # Pre-allocate
    chi_out = np.full((n_therm, n_est), np.nan)
    eps_out = np.full((n_therm, n_est), np.nan)
    kB_out = np.full((n_therm, n_est), np.nan)
    K_max_out = np.full((n_therm, n_est), np.nan)
    fom_out = np.full((n_therm, n_est), np.nan)
    K_max_ratio_out = np.full((n_therm, n_est), np.nan)
    nu_out = np.full(n_est, np.nan)
    speed_out = np.full(n_est, np.nan)
    P_out = np.full(n_est, np.nan)
    T_out = np.full(n_est, np.nan)
    t_out = np.full(n_est, np.nan)

    spec_gradT = np.full((n_therm, n_freq, n_est), np.nan)
    spec_batch = np.full((n_therm, n_freq, n_est), np.nan)
    spec_noise_out = np.full((n_therm, n_freq, n_est), np.nan)
    K_out = np.full((n_freq, n_est), np.nan)
    F_out = np.full((n_freq, n_est), np.nan)

    # Degrees of freedom
    num_ffts = 2 * (diss_length // fft_length) - 1
    dof_spec = 1.9 * num_ffts

    # Hoisted imports
    from rsi_python.goodman import clean_shear_spec_batch
    from rsi_python.ocean import visc, visc35
    from rsi_python.spectral import csd_matrix_batch

    # Pre-compute constant frequency vector
    F_const = np.arange(n_freq) * fs_fast / fft_length

    # Pre-compute first-difference correction (constant across windows/probes)
    fd_correction = np.ones(n_freq)
    with np.errstate(divide="ignore", invalid="ignore"):
        fd_correction[1:] = (
            np.pi * F_const[1:] / (fs_fast * np.sin(np.pi * F_const[1:] / fs_fast))
        ) ** 2
    fd_correction = np.where(np.isfinite(fd_correction), fd_correction, 1.0)

    # Pre-compute bilinear corrections per probe (constant across windows)
    bl_corrections = [_bilinear_correction(F_const, dg, fs_fast) for dg in diff_gains]

    # ---- Vectorized window extraction ----
    win_starts = np.arange(n_est) * step
    n_valid = int(np.sum(win_starts + diss_length <= N))

    if n_valid > 0:
        indices = win_starts[:n_valid, np.newaxis] + np.arange(diss_length)[np.newaxis, :]

        # ---- Batch Goodman cleaning ----
        therm_windows = np.stack([arr[indices] for arr in therm_arrays], axis=-1)
        do_goodman = accel_arrays is not None and len(accel_arrays) > 0

        if do_goodman:
            accel_windows = np.stack([a[indices] for a in accel_arrays], axis=-1)
            clean_TT_all, _ = clean_shear_spec_batch(
                accel_windows, therm_windows, fft_length, fs_fast,
            )
            del accel_windows
            clean_spectra_all = np.real(np.diagonal(clean_TT_all, axis1=2, axis2=3))
            del clean_TT_all
        else:
            TT_all, _, _, _ = csd_matrix_batch(
                therm_windows, None, fft_length, fs_fast,
                overlap=fft_length // 2, detrend="linear",
            )
            clean_spectra_all = np.real(np.diagonal(TT_all, axis1=2, axis2=3))
            del TT_all
        del therm_windows

        # ---- Vectorized window means ----
        speed_means = np.maximum(np.mean(np.abs(speed[indices]), axis=1), 0.01)
        T_means = np.mean(T[indices], axis=1)
        P_means = np.mean(P[indices], axis=1)
        t_means = np.mean(t_fast[indices], axis=1)

        # Vectorized viscosity (ocean functions support array inputs)
        if salinity is not None:
            if np.ndim(salinity) > 0:
                sal_means = np.mean(salinity[indices], axis=1)
            else:
                sal_means = np.full(n_valid, float(salinity))
            nu_all = np.asarray(visc(T_means, sal_means, P_means), dtype=np.float64)
        else:
            nu_all = np.asarray(visc35(T_means), dtype=np.float64)

        # Store into output arrays
        speed_out[:n_valid] = speed_means
        T_out[:n_valid] = T_means
        P_out[:n_valid] = P_means
        t_out[:n_valid] = t_means
        nu_out[:n_valid] = nu_all

        # ---- Vectorized wavenumber ----
        K_all = F_const[:, np.newaxis] / speed_means[np.newaxis, :]  # (n_freq, n_valid)
        K_out[:, :n_valid] = K_all
        F_out[:, :n_valid] = F_const[:, np.newaxis]

        # ---- Vectorized spectrum correction ----
        spec_obs_all = np.empty((n_therm, n_valid, n_freq))
        for ci in range(n_therm):
            spec_obs_all[ci] = (
                clean_spectra_all[:, :, ci]
                * speed_means[:, np.newaxis]
                * fd_correction[np.newaxis, :]
                * bl_corrections[ci][np.newaxis, :]
            )
        del clean_spectra_all

        # Store observed spectra
        spec_gradT[:, :, :n_valid] = spec_obs_all.transpose(0, 2, 1)

        # ---- Batch noise computation ----
        noise_K_all = np.empty((n_therm, n_valid, n_freq))
        for ci in range(n_therm):
            cal_ci = (therm_cal[ci] if therm_cal else {}) if therm_cal else {}
            noise_kwargs = {
                k: v
                for k, v in cal_ci.items()
                if k in ("e_b", "gain", "beta_1", "adc_fs", "adc_bits")
            }
            noise_K_all[ci] = gradT_noise_batch(
                F_const, T_means, speed_means,
                fs=fs_fast, diff_gain=diff_gains[ci], **noise_kwargs,
            )
        spec_noise_out[:, :, :n_valid] = noise_K_all.transpose(0, 2, 1)

        # ---- Vectorized H2 (FP07 transfer) ----
        tau_model = _default_tau_model(fp07_model)
        _h2 = fp07_transfer if fp07_model == "single_pole" else fp07_double_pole
        # Vectorized tau0 computation (simple formula)
        if tau_model == "lueck":
            tau0_all = 0.01 * (1.0 / speed_means) ** 0.5
        elif tau_model == "peterson":
            tau0_all = 0.012 * speed_means ** (-0.32)
        else:  # goto
            tau0_all = np.full(n_valid, 0.003)
        # Vectorized H2: broadcast F_const (n_freq,) with tau0 (n_valid,)
        omega_tau = (2 * np.pi * F_const[:, np.newaxis] * tau0_all[np.newaxis, :]) ** 2
        if fp07_model == "single_pole":
            H2_all = (1.0 / (1.0 + omega_tau)).T  # (n_valid, n_freq)
        else:
            H2_all = (1.0 / (1.0 + omega_tau) ** 2).T

        # ---- Epsilon interpolation setup (vectorized) ----
        eps_vals_all = None  # (n_valid,) or None
        if epsilon_ds is not None:
            eps_times = epsilon_ds.coords["t"].values if "t" in epsilon_ds.coords else None
            if eps_times is not None:
                if np.issubdtype(eps_times.dtype, np.datetime64):
                    eps_times = (eps_times - eps_times[0]).astype("timedelta64[ns]").astype(
                        np.float64
                    ) / 1e9 + t_fast[0]
                eps_vals = np.nanmean(epsilon_ds["epsilon"].values, axis=0)
                # Batch nearest-neighbor lookup for all windows
                idx_eps_all = np.argmin(
                    np.abs(eps_times[:, np.newaxis] - t_means[np.newaxis, :]), axis=0,
                )
                eps_vals_all = eps_vals[idx_eps_all]

        # ---- Per-window fitting loop ----
        for idx in range(n_valid):
            K = K_all[:, idx]
            W = speed_means[idx]
            nu = nu_all[idx]
            tau0 = tau0_all[idx]
            H2 = H2_all[idx]

            # Epsilon for Method 1
            epsilon_val = None
            if eps_vals_all is not None:
                epsilon_val = eps_vals_all[idx]
                if not np.isfinite(epsilon_val) or epsilon_val <= 0:
                    epsilon_val = None

            for ci in range(n_therm):
                spec_obs = spec_obs_all[ci, idx]
                noise_K = noise_K_all[ci, idx]

                if epsilon_val is not None:
                    # Method 1: chi from known epsilon
                    chi_val, kB_val, K_max_val, batch_spec, fom_val, K_max_ratio_val = (
                        _chi_from_epsilon(
                            spec_obs, K, epsilon_val, nu,
                            noise_K, H2, tau0, _h2, f_AA, W, spectrum_model,
                        )
                    )
                    chi_out[ci, idx] = chi_val
                    eps_out[ci, idx] = epsilon_val
                    kB_out[ci, idx] = kB_val
                    K_max_out[ci, idx] = K_max_val
                    fom_out[ci, idx] = fom_val
                    K_max_ratio_out[ci, idx] = K_max_ratio_val
                    spec_batch[ci, :, idx] = batch_spec
                else:
                    # Method 2: fit kB
                    obs_above_noise = np.maximum(spec_obs - noise_K, 0)
                    mask = (K > 0) & (f_AA / W >= K)
                    if np.sum(mask) < 3:
                        continue
                    chi_obs = 6 * KAPPA_T * np.trapezoid(obs_above_noise[mask], K[mask])
                    if chi_obs <= 0:
                        chi_obs = 1e-14

                    if fit_method == "iterative":
                        (kB_val, chi_val, eps_val, K_max_val,
                         batch_spec, fom_val, K_max_ratio_val) = (
                            _iterative_fit(
                                spec_obs, K, nu,
                                noise_K, H2, tau0, _h2,
                                f_AA, W, spectrum_model,
                            )
                        )
                    else:
                        (kB_val, chi_val, eps_val, K_max_val,
                         batch_spec, fom_val, K_max_ratio_val) = (
                            _mle_fit_kB(
                                spec_obs, K, chi_obs, nu,
                                noise_K, H2, tau0, _h2,
                                f_AA, W, spectrum_model,
                            )
                        )

                    chi_out[ci, idx] = chi_val
                    eps_out[ci, idx] = eps_val
                    kB_out[ci, idx] = kB_val
                    K_max_out[ci, idx] = K_max_val
                    fom_out[ci, idx] = fom_val
                    K_max_ratio_out[ci, idx] = K_max_ratio_val
                    spec_batch[ci, :, idx] = batch_spec

    return _build_chi_dataset(
        chi_out=chi_out,
        eps_out=eps_out,
        kB_out=kB_out,
        K_max_out=K_max_out,
        fom_out=fom_out,
        K_max_ratio_out=K_max_ratio_out,
        speed_out=speed_out,
        nu_out=nu_out,
        P_out=P_out,
        T_out=T_out,
        t_out=t_out,
        spec_gradT=spec_gradT,
        spec_batch=spec_batch,
        spec_noise_out=spec_noise_out,
        K_out=K_out,
        F_out=F_out,
        therm_names=therm_names,
        fft_length=fft_length,
        diss_length=diss_length,
        overlap=overlap,
        fs_fast=fs_fast,
        dof_spec=dof_spec,
        fp07_model=fp07_model,
        spectrum_model=spectrum_model,
        fit_method=fit_method,
        f_AA=f_AA,
    )


def _build_chi_dataset(
    *,
    chi_out: np.ndarray,
    eps_out: np.ndarray,
    kB_out: np.ndarray,
    K_max_out: np.ndarray,
    fom_out: np.ndarray,
    K_max_ratio_out: np.ndarray,
    speed_out: np.ndarray,
    nu_out: np.ndarray,
    P_out: np.ndarray,
    T_out: np.ndarray,
    t_out: np.ndarray,
    spec_gradT: np.ndarray,
    spec_batch: np.ndarray,
    spec_noise_out: np.ndarray,
    K_out: np.ndarray,
    F_out: np.ndarray,
    therm_names: list[str],
    fft_length: int,
    diss_length: int,
    overlap: int,
    fs_fast: float,
    dof_spec: float,
    fp07_model: str,
    spectrum_model: str,
    fit_method: str,
    f_AA: float,
) -> xr.Dataset:
    """Build an xarray Dataset from chi estimation output arrays."""
    ds = xr.Dataset(
        {
            "chi": (
                ["probe", "time"],
                chi_out,
                {
                    "units": "K2 s-1",
                    "long_name": "thermal variance dissipation rate",
                },
            ),
            "epsilon_T": (
                ["probe", "time"],
                eps_out,
                {
                    "units": "W kg-1",
                    "long_name": "TKE dissipation rate from temperature",
                },
            ),
            "kB": (
                ["probe", "time"],
                kB_out,
                {
                    "units": "cpm",
                    "long_name": "Batchelor wavenumber",
                },
            ),
            "K_max_T": (
                ["probe", "time"],
                K_max_out,
                {
                    "units": "cpm",
                    "long_name": "upper wavenumber integration limit for chi",
                },
            ),
            "fom": (
                ["probe", "time"],
                fom_out,
                {
                    "units": "1",
                    "long_name": "figure of merit (observed/model variance ratio)",
                },
            ),
            "K_max_ratio": (
                ["probe", "time"],
                K_max_ratio_out,
                {
                    "units": "1",
                    "long_name": "K_max / kB spectral resolution ratio",
                },
            ),
            "speed": (
                ["time"],
                speed_out,
                {
                    "units": "m s-1",
                    "long_name": "profiling speed",
                },
            ),
            "nu": (
                ["time"],
                nu_out,
                {
                    "units": "m2 s-1",
                    "long_name": "kinematic viscosity of sea water",
                },
            ),
            "P_mean": (
                ["time"],
                P_out,
                {
                    "units": "dbar",
                    "long_name": "mean sea water pressure",
                    "standard_name": "sea_water_pressure",
                    "positive": "down",
                },
            ),
            "T_mean": (
                ["time"],
                T_out,
                {
                    "units": "degree_Celsius",
                    "long_name": "mean sea water temperature",
                    "standard_name": "sea_water_temperature",
                },
            ),
            "spec_gradT": (
                ["probe", "freq", "time"],
                spec_gradT,
                {
                    "units": "K2 m-1",
                    "long_name": "temperature gradient wavenumber spectrum (observed)",
                },
            ),
            "spec_batch": (
                ["probe", "freq", "time"],
                spec_batch,
                {
                    "units": "K2 m-1",
                    "long_name": "fitted Batchelor temperature gradient spectrum",
                },
            ),
            "spec_noise": (
                ["probe", "freq", "time"],
                spec_noise_out,
                {
                    "units": "K2 m-1",
                    "long_name": "FP07 electronics noise spectrum",
                },
            ),
            "K": (
                ["freq", "time"],
                K_out,
                {
                    "units": "cpm",
                    "long_name": "wavenumber (cycles per metre)",
                },
            ),
            "F": (
                ["freq", "time"],
                F_out,
                {
                    "units": "Hz",
                    "long_name": "frequency",
                },
            ),
        },
        coords={
            "probe": therm_names,
            "t": (["time"], t_out),
        },
        attrs={
            "Conventions": "CF-1.13",
            "fft_length": fft_length,
            "diss_length": diss_length,
            "overlap": overlap,
            "fs_fast": fs_fast,
            "dof_spec": dof_spec,
            "fp07_model": fp07_model,
            "spectrum_model": spectrum_model,
            "fit_method": fit_method,
            "f_AA": f_AA,
        },
    )
    ds.coords["probe"].attrs["long_name"] = "thermistor probe name"
    return ds


# ---------------------------------------------------------------------------
# File-level processing
# ---------------------------------------------------------------------------


def compute_chi_file(
    source_path: str | Path,
    output_dir: str | Path,
    **chi_kwargs: Any,
) -> list[Path]:
    """Compute chi for one file and write NetCDF output(s).

    Parameters
    ----------
    source_path : str or Path
        Input file.
    output_dir : str or Path
        Output directory.
    **chi_kwargs
        Keyword arguments passed to get_chi.

    Returns
    -------
    list of Path
        Paths to output files written.
    """
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = get_chi(source_path, **chi_kwargs)

    output_paths = []
    for i, ds in enumerate(results):
        if len(results) == 1:
            out_name = f"{source_path.stem}_chi.nc"
        else:
            out_name = f"{source_path.stem}_prof{i + 1:03d}_chi.nc"
        out_path = output_dir / out_name
        ds.to_netcdf(out_path)
        output_paths.append(out_path)
        print(
            f"  {out_path.name}: {ds.sizes['time']} estimates, "
            f"P={float(ds.P_mean.min()):.0f}-{float(ds.P_mean.max()):.0f} dbar"
        )

    return output_paths


def _compute_chi_one(args: tuple) -> tuple[str, int]:
    """Worker for parallel chi computation."""
    source_path, output_dir, kwargs = args
    paths = compute_chi_file(source_path, output_dir, **kwargs)
    return str(source_path), len(paths)
