"""L4 chi: chi estimation from L3 temperature gradient spectra.

Two methods:
1. Chi from known epsilon (L4Data from shear probes) — Method 1
2. Chi from spectral fitting (MLE or iterative) — Method 2

Both call existing, tested chi fitting functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from odas_tpw.chi.batchelor import KAPPA_T
from odas_tpw.chi.chi import (
    _chi_from_epsilon,
    _iterative_fit,
    _mle_fit_kB,
)
from odas_tpw.chi.fp07 import fp07_double_pole, fp07_transfer
from odas_tpw.chi.l3_chi import L3ChiData
from odas_tpw.scor160.io import L4Data


def _to_float64_seconds(arr: np.ndarray) -> np.ndarray:
    """Convert time array to float64, handling datetime64 from xarray."""
    if np.issubdtype(arr.dtype, np.datetime64):
        # Convert datetime64 → float64 seconds since epoch
        return arr.astype("datetime64[ns]").astype("int64") / 1e9
    return np.asarray(arr, dtype=np.float64)


@dataclass
class L4ChiData:
    """Level-4 chi (thermal variance dissipation) estimates."""

    time: np.ndarray  # (N_SPECTRA,)
    pres: np.ndarray  # (N_SPECTRA,)
    pspd_rel: np.ndarray  # (N_SPECTRA,)
    section_number: np.ndarray  # (N_SPECTRA,)
    chi: np.ndarray  # (N_GRADT, N_SPECTRA)
    chi_final: np.ndarray  # (N_SPECTRA,)
    epsilon_T: np.ndarray  # (N_GRADT, N_SPECTRA)
    kB: np.ndarray  # (N_GRADT, N_SPECTRA)
    K_max: np.ndarray  # (N_GRADT, N_SPECTRA)
    fom: np.ndarray  # (N_GRADT, N_SPECTRA)
    K_max_ratio: np.ndarray  # (N_GRADT, N_SPECTRA)
    method: str  # "epsilon" or "fit"

    @property
    def n_spectra(self) -> int:
        return self.time.shape[0]

    @property
    def n_gradt(self) -> int:
        return self.chi.shape[0]


def _process_l4_chi(
    l3_chi: L3ChiData,
    chi_func,
    method_name: str,
    f_AA: float,
) -> L4ChiData:
    """Shared L4 chi processing loop.

    Parameters
    ----------
    l3_chi : L3ChiData
        Level-3 temperature gradient spectra.
    chi_func : callable
        ``chi_func(j, ci, spec_obs, noise_K, K, W, nu, tau0, H2, _h2, f_AA_eff)``
        returns ``(chi_val, eps_val, kB_val, K_max_val, fom_val, K_max_ratio_val)``
        or None to skip.
    method_name : str
        "epsilon" or "fit".
    f_AA : float
        Anti-aliasing filter cutoff [Hz].
    """
    n_spec = l3_chi.n_spectra
    n_gradt = l3_chi.n_gradt
    f_AA_eff = 0.9 * f_AA

    _h2 = fp07_transfer if l3_chi.fp07_model == "single_pole" else fp07_double_pole

    chi_out = np.full((n_gradt, n_spec), np.nan)
    eps_out = np.full((n_gradt, n_spec), np.nan)
    kB_out = np.full((n_gradt, n_spec), np.nan)
    K_max_out = np.full((n_gradt, n_spec), np.nan)
    fom_out = np.full((n_gradt, n_spec), np.nan)
    K_max_ratio_out = np.full((n_gradt, n_spec), np.nan)

    for j in range(n_spec):
        K = l3_chi.kcyc[:, j]
        W = l3_chi.pspd_rel[j]
        nu = l3_chi.nu[j]
        tau0 = l3_chi.tau0[j]
        H2 = l3_chi.H2[j]

        for ci in range(n_gradt):
            spec_obs = l3_chi.gradt_spec[ci, :, j]
            noise_K = l3_chi.noise_spec[ci, :, j]

            result = chi_func(j, ci, spec_obs, noise_K, K, W, nu, tau0, H2, _h2, f_AA_eff)
            if result is None:
                continue

            chi_val, eps_val, kB_val, K_max_val, fom_val, K_max_ratio_val = result
            chi_out[ci, j] = chi_val
            eps_out[ci, j] = eps_val
            kB_out[ci, j] = kB_val
            K_max_out[ci, j] = K_max_val
            fom_out[ci, j] = fom_val
            K_max_ratio_out[ci, j] = K_max_ratio_val

    chi_final = _compute_chi_final(chi_out)

    return L4ChiData(
        time=l3_chi.time.copy(),
        pres=l3_chi.pres.copy(),
        pspd_rel=l3_chi.pspd_rel.copy(),
        section_number=l3_chi.section_number.copy(),
        chi=chi_out,
        chi_final=chi_final,
        epsilon_T=eps_out,
        kB=kB_out,
        K_max=K_max_out,
        fom=fom_out,
        K_max_ratio=K_max_ratio_out,
        method=method_name,
    )


def process_l4_chi_epsilon(
    l3_chi: L3ChiData,
    l4_diss: L4Data,
    *,
    spectrum_model: str = "kraichnan",
    f_AA: float = 98.0,
) -> L4ChiData:
    """Compute chi from known epsilon (Method 1).

    Parameters
    ----------
    l3_chi : L3ChiData
        Level-3 temperature gradient spectra.
    l4_diss : L4Data
        Level-4 dissipation estimates (epsilon from shear probes).
    spectrum_model : str
        'batchelor' or 'kraichnan'.
    f_AA : float
        Anti-aliasing filter cutoff [Hz].

    Returns
    -------
    L4ChiData
    """
    # Pre-compute time matching
    epsi_final = l4_diss.epsi_final
    epsi_times = _to_float64_seconds(l4_diss.time)
    chi_times = _to_float64_seconds(l3_chi.time)

    def _chi_eps_func(j, _ci, spec_obs, noise_K, K, W, nu, tau0, H2, _h2, f_AA_eff):
        if len(epsi_times) > 0:
            idx_eps = int(np.argmin(np.abs(epsi_times - chi_times[j])))
            epsilon_val = float(epsi_final[idx_eps])
        else:
            epsilon_val = np.nan
        if not np.isfinite(epsilon_val) or epsilon_val <= 0:
            return None
        chi_val, kB_val, K_max_val, _, fom_val, K_max_ratio_val = _chi_from_epsilon(
            spec_obs, K, epsilon_val, nu, noise_K, H2, tau0, _h2,
            f_AA_eff, W, spectrum_model,
        )
        return chi_val, epsilon_val, kB_val, K_max_val, fom_val, K_max_ratio_val

    return _process_l4_chi(l3_chi, _chi_eps_func, "epsilon", f_AA)


def process_l4_chi_fit(
    l3_chi: L3ChiData,
    *,
    spectrum_model: str = "kraichnan",
    fit_method: str = "iterative",
    f_AA: float = 98.0,
) -> L4ChiData:
    """Compute chi by spectral fitting (Method 2).

    Parameters
    ----------
    l3_chi : L3ChiData
        Level-3 temperature gradient spectra.
    spectrum_model : str
        'batchelor' or 'kraichnan'.
    fit_method : str
        'iterative' (Peterson & Fer 2014) or 'mle' (Ruddick et al. 2000).
    f_AA : float
        Anti-aliasing filter cutoff [Hz].

    Returns
    -------
    L4ChiData
    """
    def _chi_fit_func(_j, _ci, spec_obs, noise_K, K, W, nu, tau0, H2, _h2, f_AA_eff):
        if fit_method == "iterative":
            kB_val, chi_val, eps_val, K_max_val, _, fom_val, K_max_ratio_val = _iterative_fit(
                spec_obs, K, nu, noise_K, H2, tau0, _h2, f_AA_eff, W, spectrum_model,
            )
        else:
            mask = (K > 0) & (f_AA_eff / W >= K)
            if np.sum(mask) < 3:
                return None
            chi_obs = (
                6 * KAPPA_T
                * np.trapezoid(
                    np.maximum(spec_obs[mask] - noise_K[mask], 0), K[mask],
                )
            )
            if chi_obs <= 0:
                chi_obs = 1e-14
            kB_val, chi_val, eps_val, K_max_val, _, fom_val, K_max_ratio_val = _mle_fit_kB(
                spec_obs, K, chi_obs, nu, noise_K, H2, tau0, _h2,
                f_AA_eff, W, spectrum_model,
            )
        return chi_val, eps_val, kB_val, K_max_val, fom_val, K_max_ratio_val

    return _process_l4_chi(l3_chi, _chi_fit_func, "fit", f_AA)


def _compute_chi_final(chi: np.ndarray) -> np.ndarray:
    """Compute chi_final: geometric mean of probes with valid chi."""
    _n_gradt, n_spec = chi.shape
    chi_final = np.full(n_spec, np.nan)
    for j in range(n_spec):
        good = np.isfinite(chi[:, j]) & (chi[:, j] > 0)
        if good.any():
            chi_final[j] = np.exp(np.mean(np.log(chi[good, j])))
    return chi_final
