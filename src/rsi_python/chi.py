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
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import xarray as xr

if TYPE_CHECKING:
    from rsi_python.p_file import PFile

from scipy.signal import butter, freqz

from rsi_python.batchelor import (
    KAPPA_T,
    Q_BATCHELOR,
    Q_KRAICHNAN,
    batchelor_grad,
    batchelor_kB,
    kraichnan_grad,
)
from rsi_python.fp07 import fp07_double_pole, fp07_tau, fp07_transfer, gradT_noise
from rsi_python.spectral import csd_odas

# ---------------------------------------------------------------------------
# Spectrum model dispatcher
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Method 1: Chi from known epsilon
# ---------------------------------------------------------------------------


def _chi_from_epsilon(
    spec_obs: np.ndarray,
    K: np.ndarray,
    epsilon: float,
    nu: float,
    T_mean: float,
    speed: float,
    f_AA: float,
    fp07_model: str,
    spectrum_model: str,
    fs: float,
    diff_gain: float,
    fft_length: int,
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
    T_mean : float
        Mean temperature [°C] for noise model.
    speed : float
        Profiling speed [m/s].
    f_AA : float
        Anti-aliasing cutoff [Hz].
    fp07_model : str
        FP07 transfer function model ('single_pole' or 'double_pole').
    spectrum_model : str
        Theoretical spectrum model ('batchelor' or 'kraichnan').
    fs : float
        Sampling rate [Hz].
    diff_gain : float
        Differentiator gain [s].
    fft_length : int
        FFT segment length [samples].

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
    grad_func, q = _spectrum_func(spectrum_model)

    kB = batchelor_kB(epsilon, nu)
    if kB < 1:
        warnings.warn(f"kB={kB:.1f} < 1 cpm; epsilon too low for chi estimation", stacklevel=2)
        return np.nan, kB, np.nan, np.zeros_like(K), np.nan, np.nan

    # FP07 transfer function
    tau0 = fp07_tau(speed, model=_default_tau_model(fp07_model))
    F = K * speed
    # FP07 transfer function selector
    _h2 = fp07_transfer if fp07_model == "single_pole" else fp07_double_pole
    H2 = _h2(F, tau0)

    # Noise spectrum
    noise_f = gradT_noise(F, T_mean, speed, fs=fs, diff_gain=diff_gain)[0]
    # Noise is already in cpm units from gradT_noise wrapper

    # Find integration limit: where observed meets noise
    K_AA = f_AA / speed
    above_noise = spec_obs > 2 * noise_f
    valid = above_noise & (K > 0) & (K <= K_AA)
    if np.sum(valid) < 3:
        # Fall back to AA limit
        valid = (K > 0) & (K <= K_AA)
    if np.sum(valid) < 3:
        warnings.warn("Too few valid wavenumber points for chi integration", stacklevel=2)
        return np.nan, kB, np.nan, np.zeros_like(K), np.nan, np.nan

    valid_idx = np.where(valid)[0]
    K_max = K[valid_idx[-1]]

    # Integrate observed spectrum
    obs_var = np.trapezoid(spec_obs[valid], K[valid])

    # Correction factor for FP07 rolloff and unresolved variance
    # Use trial chi = 6*kappa_T*obs_var as initial
    chi_trial = 6 * KAPPA_T * obs_var
    if chi_trial <= 0:
        warnings.warn("Trial chi <= 0; observed variance too low", stacklevel=2)
        return np.nan, kB, K_max, np.zeros_like(K), np.nan, np.nan

    # Theoretical Batchelor spectrum
    K_fine = np.linspace(0, max(K_max * 5, kB * 5), 10000)
    K_fine[0] = K_fine[1] * 0.01  # avoid zero
    spec_batch_fine = grad_func(K_fine, kB, chi_trial)

    V_total = np.trapezoid(spec_batch_fine, K_fine)

    # Resolved Batchelor variance (with FP07 rolloff)
    F_fine = K_fine * speed
    H2_fine = _h2(F_fine, tau0)
    mask_resolved = K_fine <= K_max
    V_resolved = np.trapezoid(
        spec_batch_fine[mask_resolved] * H2_fine[mask_resolved], K_fine[mask_resolved]
    )

    if V_resolved <= 0 or V_total <= 0:
        warnings.warn("Batchelor variance non-positive; cannot compute correction", stacklevel=2)
        return np.nan, kB, K_max, np.zeros_like(K), np.nan, np.nan

    correction = V_total / V_resolved
    chi = 6 * KAPPA_T * obs_var * correction

    # Compute fitted Batchelor spectrum for output
    spec_batch = grad_func(K, kB, chi)

    # Figure of merit: observed vs attenuated model (Batchelor * H2 + noise)
    if np.sum(valid) >= 3 and np.isfinite(chi):
        mod_v = np.trapezoid(spec_batch[valid] * H2[valid] + noise_f[valid], K[valid])
        fom = obs_var / mod_v if mod_v > 0 else np.nan
    else:
        fom = np.nan

    K_max_ratio_val = K_max / kB if kB > 0 else np.nan

    return chi, kB, K_max, spec_batch, fom, K_max_ratio_val


# ---------------------------------------------------------------------------
# Method 2: MLE spectral fitting (Ruddick et al. 2000)
# ---------------------------------------------------------------------------


def _mle_fit_kB(
    spec_obs: np.ndarray,
    K: np.ndarray,
    chi_obs: float,
    nu: float,
    T_mean: float,
    speed: float,
    f_AA: float,
    fp07_model: str,
    spectrum_model: str,
    fs: float,
    diff_gain: float,
    fft_length: int,
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
    T_mean, speed, f_AA, fp07_model, spectrum_model, fs, diff_gain, fft_length
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
    grad_func, q = _spectrum_func(spectrum_model)

    # FP07 transfer function — respect fp07_model parameter
    _h2 = fp07_transfer if fp07_model == "single_pole" else fp07_double_pole
    tau0 = fp07_tau(speed, model=_default_tau_model(fp07_model))
    F = K * speed
    H2 = _h2(F, tau0)

    # Noise spectrum
    noise_K, _ = gradT_noise(F, T_mean, speed, fs=fs, diff_gain=diff_gain)

    # Fitting range
    K_AA = f_AA / speed
    above_noise = spec_obs > 2 * noise_K
    fit_mask = above_noise & (K > 0) & (K <= K_AA)
    if np.sum(fit_mask) < 6:
        fit_mask = (K > 0) & (K <= K_AA)
    if np.sum(fit_mask) < 6:
        warnings.warn("Too few valid points for MLE fit", stacklevel=2)
        return np.nan, np.nan, np.nan, np.nan, np.zeros_like(K), np.nan, np.nan

    fit_idx = np.where(fit_mask)[0]
    K_fit = K[fit_idx]
    spec_fit = spec_obs[fit_idx]
    H2_fit = H2[fit_idx]
    noise_fit = noise_K[fit_idx]

    K_max_fit = K_fit[-1]

    # --- Coarse grid search (vectorized) ---
    kB_coarse = np.logspace(0, 4.5, 100)  # 1 to ~31623 cpm
    kB_2d = kB_coarse[:, np.newaxis]  # (100, 1) for broadcasting
    spec_models = grad_func(K_fit, kB_2d, chi_obs) * H2_fit + noise_fit  # (100, n_fit)
    spec_models = np.maximum(spec_models, 1e-30)
    nll_coarse = np.sum(np.log(spec_models) + spec_fit / spec_models, axis=1)
    nll_coarse = np.where(np.isfinite(nll_coarse), nll_coarse, np.inf)

    if np.all(np.isinf(nll_coarse)):
        warnings.warn("All NLL values infinite; MLE fit failed", stacklevel=2)
        return np.nan, np.nan, np.nan, np.nan, np.zeros_like(K), np.nan, np.nan

    best_coarse = kB_coarse[np.argmin(nll_coarse)]

    # --- Fine grid search (vectorized) ---
    kB_lo = max(best_coarse * 0.5, 1.0)
    kB_hi = best_coarse * 2.0
    kB_fine = np.linspace(kB_lo, kB_hi, 100)
    kB_fine_2d = kB_fine[:, np.newaxis]  # (100, 1) for broadcasting
    spec_models = grad_func(K_fit, kB_fine_2d, chi_obs) * H2_fit + noise_fit
    spec_models = np.maximum(spec_models, 1e-30)
    nll_fine = np.sum(np.log(spec_models) + spec_fit / spec_models, axis=1)
    nll_fine = np.where(np.isfinite(nll_fine), nll_fine, np.inf)

    if np.all(np.isinf(nll_fine)):
        kB_best = best_coarse
    else:
        kB_best = kB_fine[np.argmin(nll_fine)]

    # Recover epsilon from kB
    epsilon = (2 * np.pi * kB_best) ** 4 * nu * KAPPA_T**2

    # Re-estimate chi with unresolved-variance correction (like Method 1)
    K_fine = np.linspace(K[1] * 0.01, max(K_max_fit * 5, kB_best * 5), 10000)
    spec_fine = grad_func(K_fine, kB_best, chi_obs)
    V_total = np.trapezoid(spec_fine, K_fine)
    F_fine = K_fine * speed
    H2_fine = _h2(F_fine, tau0)
    mask_resolved = K_fine <= K_max_fit
    V_resolved = np.trapezoid(
        spec_fine[mask_resolved] * H2_fine[mask_resolved], K_fine[mask_resolved]
    )
    if V_resolved > 0 and V_total > 0:
        obs_var = np.trapezoid(spec_obs[fit_mask], K[fit_mask])
        chi = 6 * KAPPA_T * obs_var * V_total / V_resolved
    else:
        chi = 6 * KAPPA_T * np.trapezoid(spec_fine, K_fine)

    # Fitted spectrum for output
    spec_batch = grad_func(K, kB_best, chi)

    # Figure of merit: observed vs attenuated model (Batchelor * H2 + noise)
    if np.isfinite(chi) and np.sum(fit_mask) >= 3:
        mod_v = np.trapezoid(spec_batch[fit_idx] * H2[fit_idx] + noise_K[fit_idx], K_fit)
        obs_v = np.trapezoid(spec_fit, K_fit)
        fom = obs_v / mod_v if mod_v > 0 else np.nan
    else:
        fom = np.nan

    K_max_ratio_val = K_max_fit / kB_best if kB_best > 0 else np.nan

    return kB_best, chi, epsilon, K_max_fit, spec_batch, fom, K_max_ratio_val


def _iterative_fit(
    spec_obs: np.ndarray,
    K: np.ndarray,
    nu: float,
    T_mean: float,
    speed: float,
    f_AA: float,
    fp07_model: str,
    spectrum_model: str,
    fs: float,
    diff_gain: float,
    fft_length: int,
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
    T_mean, speed, f_AA, fp07_model, spectrum_model, fs, diff_gain, fft_length
        Same as :func:`_chi_from_epsilon`.

    Returns
    -------
    kB, chi, epsilon, K_max, spec_batch, fom, K_max_ratio
        Same as :func:`_mle_fit_kB`.
    """
    grad_func, q = _spectrum_func(spectrum_model)

    # FP07 transfer function
    tau0 = fp07_tau(speed, model=_default_tau_model(fp07_model))
    F = K * speed
    _h2 = fp07_transfer if fp07_model == "single_pole" else fp07_double_pole
    H2 = _h2(F, tau0)

    # Noise spectrum
    noise_K, _ = gradT_noise(F, T_mean, speed, fs=fs, diff_gain=diff_gain)

    # Initial integration limits
    K_AA = f_AA / speed
    above_noise = spec_obs > 2 * noise_K
    valid = above_noise & (K > 0) & (K <= K_AA)
    if np.sum(valid) < 6:
        valid = (K > 0) & (K <= K_AA)

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
        # MLE fit for kB
        kB_best, chi_fit, epsilon, K_max, _, _, _ = _mle_fit_kB(
            spec_obs,
            K,
            chi_obs,
            nu,
            T_mean,
            speed,
            f_AA,
            fp07_model,
            spectrum_model,
            fs,
            diff_gain,
            fft_length,
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
        mask_refined = (K >= k_l_new) & (K <= k_u_new)
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
        K_fine = np.linspace(K[1] * 0.01, kB_best * 5, 10000)
        spec_fine = grad_func(K_fine, kB_best, chi_obs_new if chi_obs_new > 0 else chi_obs)

        chi_low = 6 * KAPPA_T * np.trapezoid(spec_fine[K_fine < k_l_new], K_fine[K_fine < k_l_new])
        chi_high = 6 * KAPPA_T * np.trapezoid(spec_fine[K_fine > k_u_new], K_fine[K_fine > k_u_new])

        chi_obs = max(chi_obs_new, 0) + chi_low + chi_high

        if chi_obs <= 0:
            chi_obs = 1e-14

    # Final values
    chi = chi_obs
    spec_batch = grad_func(K, kB_best, chi) if np.isfinite(kB_best) else np.zeros_like(K)

    # Figure of merit: observed vs attenuated model (Batchelor * H2 + noise)
    if np.isfinite(kB_best) and np.isfinite(chi):
        valid_fom = (K > 0) & (K <= k_u)
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
    from rsi_python.dissipation import _prepare_profiles
    from rsi_python.profile import _VEHICLE_TAU

    vehicle = data.get("vehicle", "")
    tau = _VEHICLE_TAU.get(vehicle, 1.5)
    prepared = _prepare_profiles(data, speed, direction, salinity, tau=tau)
    if prepared is None:
        return []
    (profiles_slow, speed_fast, P_fast, T_fast, sal_fast, fs_fast, fs_slow, ratio, t_fast) = (
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

        if isinstance(sal_fast, np.ndarray):
            sal_prof = sal_fast[s_fast:e_fast]
        else:
            sal_prof = sal_fast  # None or scalar

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
            f"Computed with rsi-python on {datetime.now(timezone.utc).isoformat()}"
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

    from rsi_python.dissipation import load_channels

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

    # Hoisted imports (avoid per-window import overhead)
    from rsi_python.dissipation import _compute_nu
    from rsi_python.goodman import clean_shear_spec

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

    # Get epsilon values if provided (Method 1)
    eps_interp = None
    if epsilon_ds is not None:
        # Interpolate epsilon to our time grid
        eps_times = epsilon_ds.coords["t"].values if "t" in epsilon_ds.coords else None
        if eps_times is not None:
            # xarray decodes CF time attrs to datetime64; convert back to float
            # seconds so we can subtract float mean_t values later.
            if np.issubdtype(eps_times.dtype, np.datetime64):
                eps_times = (eps_times - eps_times[0]).astype("timedelta64[ns]").astype(
                    np.float64
                ) / 1e9 + t_fast[0]
            # Average across probes for Method 1
            eps_vals = np.nanmean(epsilon_ds["epsilon"].values, axis=0)
            eps_interp = (eps_times, eps_vals)

    for idx in range(n_est):
        s = idx * step
        e = s + diss_length
        if e > N:
            break

        sel = slice(s, e)
        W = np.mean(np.abs(speed[sel]))
        if W < 0.01:
            warnings.warn(
                f"Speed {W:.4f} m/s below minimum; clamped to 0.01 m/s",
                stacklevel=2,
            )
            W = 0.01  # minimum speed to avoid wavenumber singularity
        mean_T = np.mean(T[sel])
        mean_P = np.mean(P[sel])
        mean_t = np.mean(t_fast[sel])
        nu = _compute_nu(mean_T, mean_P, salinity, sel)

        K = F_const / W
        F = F_const

        K_out[:, idx] = K
        F_out[:, idx] = F
        nu_out[idx] = nu
        speed_out[idx] = W
        P_out[idx] = mean_P
        T_out[idx] = mean_T
        t_out[idx] = mean_t

        # Scalar Goodman cleaning: remove accelerometer-coherent noise
        do_goodman = accel_arrays is not None and len(accel_arrays) > 0
        clean_spectra = {}
        if do_goodman:
            accel_seg = np.column_stack([a[sel] for a in accel_arrays])
            therm_seg = np.column_stack([therm_arrays[ci][sel] for ci in range(n_therm)])
            clean_TT, _, _, _, _ = clean_shear_spec(accel_seg, therm_seg, fft_length, fs_fast)
            for ci in range(n_therm):
                clean_spectra[ci] = np.real(clean_TT[:, ci, ci])

        # Get epsilon for Method 1
        epsilon_val = None
        if eps_interp is not None:
            eps_times, eps_vals = eps_interp
            # Find nearest epsilon estimate
            idx_eps = np.argmin(np.abs(eps_times - mean_t))
            epsilon_val = eps_vals[idx_eps]
            if not np.isfinite(epsilon_val) or epsilon_val <= 0:
                epsilon_val = None

        # NOTE: gradient method is 'first_difference' (Python default).
        # ODAS uses 'high_pass' for VMP data, which applies a scale factor
        # involving beta_2/beta_3 from Steinhart-Hart. This is a known
        # difference; see TODO for high_pass gradient implementation.

        for ci in range(n_therm):
            # Per-probe noise spectrum
            cal_ci = (therm_cal[ci] if therm_cal else {}) if therm_cal else {}
            noise_kwargs = {
                k: v
                for k, v in cal_ci.items()
                if k
                in (
                    "e_b",
                    "gain",
                    "beta_1",
                    "adc_fs",
                    "adc_bits",
                )
            }
            noise_K, _ = gradT_noise(
                F, mean_T, W, fs=fs_fast, diff_gain=diff_gains[ci], **noise_kwargs
            )
            spec_noise_out[ci, :, idx] = noise_K

            if do_goodman and ci in clean_spectra:
                # Use Goodman-cleaned spectrum
                Pxx = clean_spectra[ci]
            else:
                # Compute auto-spectrum of gradient signal
                seg = therm_arrays[ci][sel]
                Pxx = csd_odas(
                    seg,
                    None,
                    fft_length,
                    fs_fast,
                    overlap=fft_length // 2,
                    detrend="linear",
                )[0]

            # Convert to wavenumber spectrum with pre-computed corrections
            spec_obs = Pxx * W * fd_correction * bl_corrections[ci]

            spec_gradT[ci, :, idx] = spec_obs

            if epsilon_val is not None:
                # Method 1: chi from known epsilon
                chi_val, kB_val, K_max_val, batch_spec, fom_val, K_max_ratio_val = (
                    _chi_from_epsilon(
                        spec_obs,
                        K,
                        epsilon_val,
                        nu,
                        mean_T,
                        W,
                        f_AA,
                        fp07_model,
                        spectrum_model,
                        fs_fast,
                        diff_gains[ci],
                        fft_length,
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
                # Initial chi estimate from observed spectrum (per-probe noise)
                obs_above_noise = np.maximum(spec_obs - noise_K, 0)
                mask = (K > 0) & (K <= f_AA / W)
                if np.sum(mask) < 3:
                    continue
                chi_obs = 6 * KAPPA_T * np.trapezoid(obs_above_noise[mask], K[mask])
                if chi_obs <= 0:
                    chi_obs = 1e-14

                if fit_method == "iterative":
                    kB_val, chi_val, eps_val, K_max_val, batch_spec, fom_val, K_max_ratio_val = (
                        _iterative_fit(
                            spec_obs,
                            K,
                            nu,
                            mean_T,
                            W,
                            f_AA,
                            fp07_model,
                            spectrum_model,
                            fs_fast,
                            diff_gains[ci],
                            fft_length,
                        )
                    )
                else:
                    kB_val, chi_val, eps_val, K_max_val, batch_spec, fom_val, K_max_ratio_val = (
                        _mle_fit_kB(
                            spec_obs,
                            K,
                            chi_obs,
                            nu,
                            mean_T,
                            W,
                            f_AA,
                            fp07_model,
                            spectrum_model,
                            fs_fast,
                            diff_gains[ci],
                            fft_length,
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
