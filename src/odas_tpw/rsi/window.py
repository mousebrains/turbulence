# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Shared per-window epsilon and chi computation.

These functions encapsulate the single-window computation used by both
the pipeline (dissipation.py, chi.py) and the interactive viewers
(diss_look.py, quick_look.py).  Having one implementation prevents
divergence bugs (missing speed² division, wrong f_AA, etc.).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from odas_tpw.chi.fp07 import fp07_double_pole, fp07_tau, fp07_transfer, gradT_noise
from odas_tpw.rsi.dissipation import (
    MACOUN_LUECK_DENOM,
    MACOUN_LUECK_K,
    SPEED_MIN,
)
from odas_tpw.scor160.goodman import clean_shear_spec
from odas_tpw.scor160.l4 import _estimate_epsilon
from odas_tpw.scor160.ocean import visc35
from odas_tpw.scor160.spectral import csd_odas

# ---------------------------------------------------------------------------
# Epsilon for one dissipation window
# ---------------------------------------------------------------------------


@dataclass
class EpsWindowResult:
    """Results from epsilon computation for one dissipation window."""

    epsilon: np.ndarray  # (n_shear,)
    K_max: np.ndarray  # (n_shear,)
    mad: np.ndarray  # (n_shear,)
    fom: np.ndarray  # (n_shear,)
    FM: np.ndarray  # (n_shear,)
    K_max_ratio: np.ndarray  # (n_shear,)
    method: np.ndarray  # (n_shear,)
    epsilon_isr: np.ndarray = field(default_factory=lambda: np.array([]))  # (n_shear,)
    epsilon_var: np.ndarray = field(default_factory=lambda: np.array([]))  # (n_shear,)
    shear_specs: list[np.ndarray] = field(default_factory=list)  # per-probe
    nasmyth_specs: list[np.ndarray] = field(default_factory=list)
    K: np.ndarray = field(default_factory=lambda: np.array([]))
    F: np.ndarray = field(default_factory=lambda: np.array([]))
    W: float = 0.0
    nu: float = 0.0
    mean_P: float = float("nan")
    mean_T: float = float("nan")


def compute_eps_window(
    shear: np.ndarray,
    accel: np.ndarray | None,
    speed: np.ndarray,
    P: np.ndarray,
    T_mean: float,
    fs_fast: float,
    fft_length: int,
    f_AA: float,
    do_goodman: bool = True,
    fit_order: int = 3,
    e_isr_threshold: float = 1.5e-5,
) -> EpsWindowResult:
    """Compute epsilon for one dissipation window.

    Parameters
    ----------
    shear : ndarray, shape (diss_length, n_shear)
        Shear probe data **already divided by speed²** (du/dz in s⁻¹).
    accel : ndarray or None, shape (diss_length, n_accel)
        Accelerometer data.  None disables Goodman cleaning.
    speed : ndarray, shape (diss_length,)
        Profiling speed [m/s].
    P : ndarray, shape (diss_length,)
        Pressure [dbar].
    T_mean : float
        Mean temperature [°C] for viscosity calculation.
    fs_fast : float
        Fast sampling rate [Hz].
    fft_length : int
        FFT segment length [samples].
    f_AA : float
        Anti-aliasing frequency [Hz].  No 0.9 factor applied here
        (matching MATLAB get_diss_odas).
    do_goodman : bool
        Whether to apply Goodman coherent noise removal.
    """
    shear = np.atleast_2d(shear)
    if shear.shape[0] < shear.shape[1]:
        shear = shear.T
    n_shear = shear.shape[1]
    diss_length = shear.shape[0]

    W = float(np.mean(np.abs(speed)))
    if W < SPEED_MIN:
        warnings.warn(f"Speed {W:.4f} m/s below minimum; clamped to {SPEED_MIN} m/s", stacklevel=2)
        W = SPEED_MIN
    nu = float(visc35(T_mean))
    mean_P = float(np.mean(P))

    n_freq = fft_length // 2 + 1
    F = np.arange(n_freq) * fs_fast / fft_length
    K_AA = f_AA / W
    num_ffts = 2 * (diss_length // fft_length) - 1
    n_v = (accel.shape[1] if accel is not None else 0) if do_goodman else 0

    eps = np.full(n_shear, np.nan)
    k_maxes = np.full(n_shear, np.nan)
    mads = np.full(n_shear, np.nan)
    foms = np.full(n_shear, np.nan)
    FMs = np.full(n_shear, np.nan)
    Kmrs = np.full(n_shear, np.nan)
    methods = np.full(n_shear, np.nan)
    eps_isr = np.full(n_shear, np.nan)
    eps_var = np.full(n_shear, np.nan)
    shear_specs: list[np.ndarray] = []
    nasmyth_specs: list[np.ndarray] = []
    K_out = F / W

    if diss_length < 2 * fft_length:
        return EpsWindowResult(
            epsilon=eps,
            K_max=k_maxes,
            mad=mads,
            fom=foms,
            FM=FMs,
            K_max_ratio=Kmrs,
            method=methods,
            K=K_out,
            F=F,
            W=W,
            nu=nu,
            mean_P=mean_P,
            mean_T=T_mean,
        )

    if do_goodman and accel is not None and shear.shape[0] > 2 * fft_length:
        clean_UU, _, _, _, F_g = clean_shear_spec(accel, shear, fft_length, fs_fast)
        K_g = F_g / W

        # Macoun-Lueck wavenumber correction
        correction = np.ones_like(K_g)
        mask = K_g <= MACOUN_LUECK_K
        correction[mask] = 1 + (K_g[mask] / MACOUN_LUECK_DENOM) ** 2

        for ci in range(n_shear):
            spec_k = np.real(clean_UU[:, ci, ci]) * W * correction
            e4, k_max, mad, meth, fom, _vr, nas, Kmr, FM, e_i, e_v = _estimate_epsilon(
                K_g,
                spec_k,
                nu,
                K_AA,
                fit_order,
                e_isr_threshold,
                num_ffts=num_ffts,
                n_v=n_v,
            )
            eps[ci] = e4
            k_maxes[ci] = k_max
            mads[ci] = mad
            foms[ci] = fom
            FMs[ci] = FM
            Kmrs[ci] = Kmr
            methods[ci] = meth
            eps_isr[ci] = e_i
            eps_var[ci] = e_v
            shear_specs.append(spec_k)
            nasmyth_specs.append(nas)
        K_out = K_g
    else:
        for ci in range(n_shear):
            Pxx, F_s, _, _ = csd_odas(shear[:, ci], None, fft_length, fs_fast)
            K_s = F_s / W
            correction = np.ones_like(K_s)
            mask_c = K_s <= MACOUN_LUECK_K
            correction[mask_c] = 1 + (K_s[mask_c] / MACOUN_LUECK_DENOM) ** 2
            spec_k = Pxx * W * correction

            e4, k_max, mad, meth, fom, _vr, nas, Kmr, FM, e_i, e_v = _estimate_epsilon(
                K_s,
                spec_k,
                nu,
                f_AA / W,
                fit_order,
                e_isr_threshold,
                num_ffts=num_ffts,
                n_v=0,
            )
            eps[ci] = e4
            k_maxes[ci] = k_max
            mads[ci] = mad
            foms[ci] = fom
            FMs[ci] = FM
            Kmrs[ci] = Kmr
            methods[ci] = meth
            eps_isr[ci] = e_i
            eps_var[ci] = e_v
            shear_specs.append(spec_k)
            nasmyth_specs.append(nas)
        if n_shear > 0:
            K_out = K_s

    return EpsWindowResult(
        epsilon=eps,
        K_max=k_maxes,
        mad=mads,
        fom=foms,
        FM=FMs,
        K_max_ratio=Kmrs,
        method=methods,
        epsilon_isr=eps_isr,
        epsilon_var=eps_var,
        shear_specs=shear_specs,
        nasmyth_specs=nasmyth_specs,
        K=K_out,
        F=F,
        W=W,
        nu=nu,
        mean_P=mean_P,
        mean_T=T_mean,
    )


# ---------------------------------------------------------------------------
# Chi for one dissipation window
# ---------------------------------------------------------------------------


@dataclass
class ChiWindowResult:
    """Results from chi computation for one dissipation window."""

    chi: np.ndarray  # (n_therm,)
    kB: np.ndarray  # (n_therm,)
    K_max_T: np.ndarray  # (n_therm,)
    fom: np.ndarray  # (n_therm,)
    K_max_ratio: np.ndarray  # (n_therm,)
    grad_specs: list[np.ndarray] = field(default_factory=list)  # observed
    model_specs: list[np.ndarray] = field(default_factory=list)  # fitted (with H2)
    model_specs_raw: list[np.ndarray] = field(default_factory=list)  # unattenuated
    noise_K: np.ndarray | None = None
    K: np.ndarray = field(default_factory=lambda: np.array([]))
    F: np.ndarray = field(default_factory=lambda: np.array([]))
    H2: np.ndarray = field(default_factory=lambda: np.array([]))


def compute_chi_window(
    therm_segs: list[np.ndarray],
    diff_gains: list[float],
    W: float,
    T_mean: float,
    nu: float,
    fs_fast: float,
    fft_length: int,
    f_AA: float,
    spectrum_model: str = "kraichnan",
    fp07_model: str = "single_pole",
    epsilon: np.ndarray | None = None,
    method: int = 1,
) -> ChiWindowResult:
    """Compute chi for one dissipation window.

    Parameters
    ----------
    therm_segs : list of ndarray
        Per-probe temperature segments (°C, deconvolved fast-rate).
    diff_gains : list of float
        Per-probe differentiator gain [s].
    W : float
        Mean profiling speed [m/s].
    T_mean : float
        Mean temperature [°C].
    nu : float
        Kinematic viscosity [m²/s].
    fs_fast : float
        Fast sampling rate [Hz].
    fft_length : int
        FFT segment length [samples].
    f_AA : float
        Anti-aliasing frequency [Hz].  Caller should apply the 0.9
        safety margin (matching MATLAB get_chi).
    spectrum_model : str
        'batchelor' or 'kraichnan'.
    fp07_model : str
        'single_pole' or 'double_pole'.
    epsilon : ndarray or None, shape (n_therm,)
        Per-probe epsilon for Method 1.  None → Method 2 (iterative fit).
    method : int
        1 = chi from epsilon (requires epsilon), 2 = iterative fit.
    """
    from odas_tpw.chi.chi import _chi_from_epsilon, _iterative_fit
    from odas_tpw.chi.fp07 import default_tau_model

    n_therm = len(therm_segs)
    n_freq = fft_length // 2 + 1
    F = np.arange(n_freq) * fs_fast / fft_length
    K = F / W

    # Temperature-to-gradient conversion: (2πK)² converts Φ_T(K) to Φ_{dT/dz}(K)
    grad_factor = np.zeros(n_freq)
    grad_factor[1:] = (2 * np.pi * K[1:]) ** 2

    # FP07 transfer function
    _h2_func = fp07_transfer if fp07_model == "single_pole" else fp07_double_pole
    tau0 = float(fp07_tau(W, model=default_tau_model(fp07_model)))
    H2 = _h2_func(F, tau0)

    # Noise spectrum (use first probe's diff_gain)
    dg0 = diff_gains[0] if diff_gains else 0.94
    noise_K, _ = gradT_noise(F, T_mean, W, fs=fs_fast, diff_gain=dg0)

    chi_arr = np.full(n_therm, np.nan)
    kB_arr = np.full(n_therm, np.nan)
    Kmax_arr = np.full(n_therm, np.nan)
    fom_arr = np.full(n_therm, np.nan)
    Kmr_arr = np.full(n_therm, np.nan)
    grad_specs: list[np.ndarray] = []
    model_specs: list[np.ndarray] = []
    model_specs_raw: list[np.ndarray] = []

    for ci in range(n_therm):
        seg = therm_segs[ci]
        if len(seg) < 2 * fft_length:
            grad_specs.append(np.full(n_freq, np.nan))
            model_specs.append(np.full(n_freq, np.nan))
            model_specs_raw.append(np.full(n_freq, np.nan))
            continue

        Pxx_t, _ = csd_odas(seg, None, fft_length, fs_fast, overlap=fft_length // 2)[:2]
        spec_obs = Pxx_t * W * grad_factor
        grad_specs.append(spec_obs)

        dg = diff_gains[ci] if ci < len(diff_gains) else 0.94
        # Per-probe noise (may differ from shared noise_K if diff_gain differs)
        noise_K_ci, _ = gradT_noise(F, T_mean, W, fs=fs_fast, diff_gain=dg)

        if method == 1 and epsilon is not None:
            eps_ci = epsilon[ci] if ci < len(epsilon) else np.nan
            if np.isfinite(eps_ci) and eps_ci > 0:
                chi_val, kB, K_max, spec_raw, fom_val, Kmr = _chi_from_epsilon(
                    spec_obs,
                    K,
                    eps_ci,
                    nu,
                    noise_K_ci,
                    H2,
                    tau0,
                    _h2_func,
                    f_AA,
                    W,
                    spectrum_model,
                )
                chi_arr[ci] = chi_val
                kB_arr[ci] = kB
                Kmax_arr[ci] = K_max
                fom_arr[ci] = fom_val
                Kmr_arr[ci] = Kmr
                model_specs.append(spec_raw * H2)
                model_specs_raw.append(spec_raw)
            else:
                model_specs.append(np.zeros(n_freq))
                model_specs_raw.append(np.zeros(n_freq))
        else:
            # Method 2: iterative fit
            _, chi_val, kB, K_max, spec_raw, fom_val, Kmr = _iterative_fit(
                spec_obs,
                K,
                nu,
                noise_K_ci,
                H2,
                tau0,
                _h2_func,
                f_AA,
                W,
                spectrum_model,
            )
            chi_arr[ci] = chi_val
            kB_arr[ci] = kB
            Kmax_arr[ci] = K_max
            fom_arr[ci] = fom_val
            Kmr_arr[ci] = Kmr
            if np.isfinite(chi_val):
                model_specs.append(spec_raw * H2)
                model_specs_raw.append(spec_raw)
            else:
                model_specs.append(np.zeros(n_freq))
                model_specs_raw.append(np.zeros(n_freq))

    return ChiWindowResult(
        chi=chi_arr,
        kB=kB_arr,
        K_max_T=Kmax_arr,
        fom=fom_arr,
        K_max_ratio=Kmr_arr,
        grad_specs=grad_specs,
        model_specs=model_specs,
        model_specs_raw=model_specs_raw,
        noise_K=noise_K,
        K=K,
        F=F,
        H2=H2,
    )
