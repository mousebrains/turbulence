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

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import xarray as xr

if TYPE_CHECKING:
    from rsi_python.p_file import PFile

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


def _spectrum_func(model):
    if model == "batchelor":
        return batchelor_grad, Q_BATCHELOR
    elif model == "kraichnan":
        return kraichnan_grad, Q_KRAICHNAN
    else:
        raise ValueError(f"Unknown spectrum model: {model!r}")


# ---------------------------------------------------------------------------
# Method 1: Chi from known epsilon
# ---------------------------------------------------------------------------


def _chi_from_epsilon(
    spec_obs,
    K,
    epsilon,
    nu,
    T_mean,
    speed,
    f_AA,
    fp07_model,
    spectrum_model,
    fs,
    diff_gain,
    fft_length,
):
    """Compute chi for one probe in one window, given epsilon.

    Returns (chi, kB, K_max, spec_batch).
    """
    grad_func, q = _spectrum_func(spectrum_model)

    kB = batchelor_kB(epsilon, nu)
    if kB < 1:
        return np.nan, kB, np.nan, np.zeros_like(K), np.nan, np.nan

    # FP07 transfer function
    tau0 = fp07_tau(speed)
    F = K * speed
    # FP07 transfer function selector
    _h2 = fp07_transfer if fp07_model == "single_pole" else fp07_double_pole

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
        return np.nan, kB, np.nan, np.zeros_like(K), np.nan, np.nan

    valid_idx = np.where(valid)[0]
    K_max = K[valid_idx[-1]]

    # Integrate observed spectrum
    obs_var = np.trapezoid(spec_obs[valid], K[valid])

    # Correction factor for FP07 rolloff and unresolved variance
    # Use trial chi = 6*kappa_T*obs_var as initial
    chi_trial = 6 * KAPPA_T * obs_var
    if chi_trial <= 0:
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
        return np.nan, kB, K_max, np.zeros_like(K), np.nan, np.nan

    correction = V_total / V_resolved
    chi = 6 * KAPPA_T * obs_var * correction

    # Compute fitted Batchelor spectrum for output
    spec_batch = grad_func(K, kB, chi)

    # Figure of merit: observed/model variance in integration range
    if np.sum(valid) >= 3 and np.isfinite(chi):
        mod_v = np.trapezoid(spec_batch[valid], K[valid])
        fom = obs_var / mod_v if mod_v > 0 else np.nan
    else:
        fom = np.nan

    K_max_ratio_val = K_max / kB if kB > 0 else np.nan

    return chi, kB, K_max, spec_batch, fom, K_max_ratio_val


# ---------------------------------------------------------------------------
# Method 2: MLE spectral fitting (Ruddick et al. 2000)
# ---------------------------------------------------------------------------


def _mle_fit_kB(
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
):
    """Maximum likelihood fit for kB, one probe, one window.

    Returns (kB_best, chi, epsilon, K_max, spec_batch).
    """
    grad_func, q = _spectrum_func(spectrum_model)

    # FP07 transfer function — respect fp07_model parameter
    _h2 = fp07_transfer if fp07_model == "single_pole" else fp07_double_pole
    tau0 = fp07_tau(speed)
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
        return np.nan, np.nan, np.nan, np.nan, np.zeros_like(K), np.nan, np.nan

    fit_idx = np.where(fit_mask)[0]
    K_fit = K[fit_idx]
    spec_fit = spec_obs[fit_idx]
    H2_fit = H2[fit_idx]
    noise_fit = noise_K[fit_idx]

    K_max_fit = K_fit[-1]

    # --- Coarse grid search ---
    kB_coarse = np.logspace(0, 4.5, 100)  # 1 to ~31623 cpm
    nll_coarse = np.full(len(kB_coarse), np.inf)

    for i, kB_try in enumerate(kB_coarse):
        spec_model = grad_func(K_fit, kB_try, chi_obs) * H2_fit + noise_fit
        spec_model = np.maximum(spec_model, 1e-30)
        # Negative log-likelihood for chi-squared distributed spectral estimates
        nll = np.sum(np.log(spec_model) + spec_fit / spec_model)
        if np.isfinite(nll):
            nll_coarse[i] = nll

    if np.all(np.isinf(nll_coarse)):
        return np.nan, np.nan, np.nan, np.nan, np.zeros_like(K), np.nan, np.nan

    best_coarse = kB_coarse[np.argmin(nll_coarse)]

    # --- Fine grid search ---
    kB_lo = max(best_coarse * 0.5, 1.0)
    kB_hi = best_coarse * 2.0
    kB_fine = np.linspace(kB_lo, kB_hi, 100)
    nll_fine = np.full(len(kB_fine), np.inf)

    for i, kB_try in enumerate(kB_fine):
        spec_model = grad_func(K_fit, kB_try, chi_obs) * H2_fit + noise_fit
        spec_model = np.maximum(spec_model, 1e-30)
        nll = np.sum(np.log(spec_model) + spec_fit / spec_model)
        if np.isfinite(nll):
            nll_fine[i] = nll

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

    # Figure of merit
    if np.isfinite(chi) and np.sum(fit_mask) >= 3:
        mod_v = np.trapezoid(spec_batch[fit_idx], K_fit)
        obs_v = np.trapezoid(spec_fit, K_fit)
        fom = obs_v / mod_v if mod_v > 0 else np.nan
    else:
        fom = np.nan

    K_max_ratio_val = K_max_fit / kB_best if kB_best > 0 else np.nan

    return kB_best, chi, epsilon, K_max_fit, spec_batch, fom, K_max_ratio_val


def _iterative_fit(
    spec_obs, K, nu, T_mean, speed, f_AA, fp07_model, spectrum_model, fs, diff_gain, fft_length
):
    """Iterative MLE fitting (Peterson & Fer 2014).

    Three iterations refining the integration limits and unresolved variance.

    Returns (kB, chi, epsilon, K_max, spec_batch).
    """
    grad_func, q = _spectrum_func(spectrum_model)

    # FP07 transfer function
    tau0 = fp07_tau(speed)
    F = K * speed
    fp07_transfer(F, tau0)  # evaluated but not used directly here

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

    # Iterative refinement (3 iterations)
    kB_best = np.nan
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

    # Figure of merit
    if np.isfinite(kB_best) and np.isfinite(chi):
        valid_fom = (K > 0) & (K <= k_u)
        if np.sum(valid_fom) >= 3:
            obs_v = np.trapezoid(spec_obs[valid_fom], K[valid_fom])
            mod_v = np.trapezoid(spec_batch[valid_fom], K[valid_fom])
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
    goodman: bool = False,
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
        Apply Goodman coherent noise removal using accelerometers.
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
        spec_noise  (freq, time) — noise spectra
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


def _extract_therm_cal(ch_cfg: dict) -> dict:
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


def _bilinear_correction(F, diff_gain, fs):
    """Bilinear transform correction for deconvolution filter.

    Matches ODAS get_scalar_spectra_odas.m lines 264-270:
    corrects for the difference between the ideal analog LP filter
    H = 1/(1+(2*pi*f*diff_gain)^2) and the actual digital Butterworth
    used in deconvolution.
    """
    from scipy.signal import butter, freqz

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


def _load_therm_channels(source):
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
                ch_cfg = next(
                    (ch for ch in pf.config["channels"] if ch.get("name") == name),
                    {},
                )
                dg = ch_cfg.get("diff_gain", "0.94")
                diff_gains.append(float(dg))
                # Extract calibration from base channel (T1 for T1_dT1)
                base_name = re.match(r"^(\w+)_d\1$", name).group(1)
                base_cfg = next(
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
    """Compute chi for a single profile."""
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
    spec_noise_out = np.full((n_freq, n_est), np.nan)
    K_out = np.full((n_freq, n_est), np.nan)
    F_out = np.full((n_freq, n_est), np.nan)

    # Degrees of freedom
    num_ffts = 2 * (diss_length // fft_length) - 1
    dof_spec = 1.9 * num_ffts

    # Get epsilon values if provided (Method 1)
    eps_interp = None
    if epsilon_ds is not None:
        # Interpolate epsilon to our time grid
        eps_times = epsilon_ds.coords["t"].values if "t" in epsilon_ds.coords else None
        if eps_times is not None:
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
            W = 0.01
        mean_T = np.mean(T[sel])
        mean_P = np.mean(P[sel])
        mean_t = np.mean(t_fast[sel])
        from rsi_python.dissipation import _compute_nu

        nu = _compute_nu(mean_T, mean_P, salinity, sel)

        K = np.arange(n_freq) * fs_fast / fft_length / W
        F = np.arange(n_freq) * fs_fast / fft_length

        K_out[:, idx] = K
        F_out[:, idx] = F
        nu_out[idx] = nu
        speed_out[idx] = W
        P_out[idx] = mean_P
        T_out[idx] = mean_T
        t_out[idx] = mean_t

        # Noise spectrum for this window (use first probe's calibration)
        cal_0 = (therm_cal[0] if therm_cal else {}) if therm_cal else {}
        noise_kwargs = {k: v for k, v in cal_0.items() if k in (
            "e_b", "gain", "beta_1", "adc_fs", "adc_bits",
        )}
        noise_K, _ = gradT_noise(
            F, mean_T, W, fs=fs_fast, diff_gain=diff_gains[0], **noise_kwargs
        )
        spec_noise_out[:, idx] = noise_K

        # Scalar Goodman cleaning: remove accelerometer-coherent noise
        do_goodman = accel_arrays is not None and len(accel_arrays) > 0
        clean_spectra = {}
        if do_goodman:
            from rsi_python.goodman import clean_shear_spec

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

        for ci in range(n_therm):
            if do_goodman and ci in clean_spectra:
                # Use Goodman-cleaned spectrum
                Pxx = clean_spectra[ci]
                F_spec = np.arange(n_freq) * fs_fast / fft_length
            else:
                # Compute auto-spectrum of gradient signal
                seg = therm_arrays[ci][sel]
                Pxx, F_spec = csd_odas(
                    seg,
                    None,
                    fft_length,
                    fs_fast,
                    overlap=fft_length // 2,
                    detrend="linear",
                )[:2]

            # Convert to wavenumber spectrum
            spec_obs = Pxx * W

            # First-difference correction
            correction = np.ones(n_freq)
            with np.errstate(divide="ignore", invalid="ignore"):
                correction[1:] = (
                    np.pi * F_spec[1:] / (fs_fast * np.sin(np.pi * F_spec[1:] / fs_fast))
                ) ** 2
            correction = np.where(np.isfinite(correction), correction, 1.0)

            # Bilinear correction (ODAS get_scalar_spectra_odas.m l.264-270)
            bl_corr = _bilinear_correction(F_spec, diff_gains[ci], fs_fast)
            spec_obs = spec_obs * correction * bl_corr

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
                # Initial chi estimate from observed spectrum
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

    # Build xarray Dataset
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
                ["freq", "time"],
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


def _compute_chi_one(args):
    """Worker for parallel chi computation."""
    source_path, output_dir, kwargs = args
    paths = compute_chi_file(source_path, output_dir, **kwargs)
    return str(source_path), len(paths)
