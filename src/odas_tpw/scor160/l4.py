"""L3 -> L4: dissipation rate estimation from wavenumber spectra.

Implements Sec. 3.4 of Lueck et al. (2024).

Steps:
  1. For each dissipation window, estimate epsilon from shear wavenumber spectrum.
  2. Use variance method (low epsilon) or ISR method (high epsilon).
  3. Iterative variance correction for unresolved spectral tail.
  4. Compute QC metrics: FOM, MAD, VAR_RESOLVED.
  5. Apply QC flags and compute EPSI_FINAL (geometric mean of passing probes).
"""

from __future__ import annotations

import warnings

import numpy as np

from odas_tpw.scor160.io import L3Data, L4Data
from odas_tpw.scor160.nasmyth import LUECK_A, X_95, nasmyth_grid
from odas_tpw.scor160.ocean import visc35

# ---------------------------------------------------------------------------
# Constants (matching rsi.dissipation)
# ---------------------------------------------------------------------------

EPSILON_FLOOR = 1e-15
E_ISR_THRESHOLD = 1.5e-5
K_LIMIT_MIN = 7
K_LIMIT_MAX = 150
K_INITIAL_CUTOFF = 10
ISOTROPY_FACTOR = 7.5
# Upper limit of the inertial-subrange fit, as a fraction of the
# Kolmogorov wavenumber k_s = (epsilon/nu^3)^(1/4).  ODAS used 0.01 for
# years; v4.5.1 experimentally doubles it ("test pushing this upward",
# get_diss_odas.m:434-436).  The ATOMIX benchmark reference values match
# 0.01 distinctly better (verified by rerunning the L2->L4 comparison
# with 0.02: per-probe mean log10 errors on MR1000 Minas Passage grow
# from ~+0.02 to +0.05..+0.08), so we keep 0.01.
X_ISR = 0.01

# Margin applied to the variance->ISR method-switch threshold:
# use_isr = e_1 >= e_isr_threshold * isr_margin.  ODAS switches at
# e_isr_threshold (1.5e-5 W/kg) with no margin.  Empirically, this
# pipeline's preliminary estimate e_1 runs high enough near the
# threshold that a 1.0 margin overshoots the benchmark ISR fraction
# (e.g. VMP250 tidal: 37.5% ISR vs 21.9% in the reference, dropping
# method agreement from 100% to 84%); 1.6 maximises method agreement
# with the ATOMIX benchmark references.  Both epsilon estimates are
# always computed, so the margin only affects which one is reported.
DEFAULT_ISR_MARGIN = 1.6

VARIANCE_TANH_COEFF = 48
VARIANCE_EXP_COEFF = 2.9
VARIANCE_EXP_DECAY = 22.3
VARIANCE_CONVERGENCE = 1.02
LOW_K_CORRECTION_THRESHOLD = 1.1
LOW_K_CORRECTION_FACTOR = 0.25

ISR_FLYER_THRESHOLD = 0.5
ISR_FLYER_FRAC = 0.2

FM_SIGMA_COEFF = 1.25
FM_SIGMA_EXPONENT = -7 / 9
FM_TM_OFFSET = 0.8
FM_TM_COEFF = 1.56

# Default QC thresholds (from ATOMIX benchmark file attributes;
# VMP250_TidalChannel_024.nc L4_dissipation group)
DEFAULT_FOM_LIMIT = 1.15
DEFAULT_VAR_RESOLVED_LIMIT = 0.5
DEFAULT_DESPIKE_FRACTION_LIMIT = 0.05  # despike_shear_fraction_limit
DEFAULT_DESPIKE_PASSES_LIMIT = 8  # despike_shear_iterations_limit
# z-factor on the expected sigma_ln for the inter-probe consistency
# test: ln(e_i/e_min) > diss_ratio_limit * mean(sigma_ln) flags probe i.
# The benchmark stores exactly 1.96*sqrt(2) (95% CI for the difference
# of two equal-variance lognormal estimates).
DEFAULT_DISS_RATIO_LIMIT = 1.96 * np.sqrt(2.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def process_l4(
    l3: L3Data,
    *,
    temp: np.ndarray | None = None,
    f_AA: float = 98.0,
    fit_order: int = 3,
    fom_limit: float = DEFAULT_FOM_LIMIT,
    var_resolved_limit: float = DEFAULT_VAR_RESOLVED_LIMIT,
    num_ffts: int = 0,
    n_v: int = 0,
    diss_length_s: float = 0.0,
    despike_fraction_limit: float = DEFAULT_DESPIKE_FRACTION_LIMIT,
    despike_passes_limit: int = DEFAULT_DESPIKE_PASSES_LIMIT,
    diss_ratio_limit: float = DEFAULT_DISS_RATIO_LIMIT,
) -> L4Data:
    """Compute L4 dissipation estimates from L3 wavenumber spectra.

    Parameters
    ----------
    l3 : L3Data
        Level-3 wavenumber spectra (clean spectra used for epsilon).
    temp : ndarray, optional
        Per-window temperature for viscosity. If None, uses 10 C default.
    f_AA : float
        Anti-alias filter cutoff [Hz].
    fit_order : int
        Polynomial order for spectral minimum detection.
    fom_limit : float
        QC threshold for the figure of merit.  Applied to the Lueck
        (2022a,b) MAD-based FM statistic when ``num_ffts > 0`` (the
        ATOMIX convention; recommended limit ~1.15), otherwise to the
        observed/Nasmyth variance ratio ``fom``.
    var_resolved_limit : float
        QC threshold for fraction of variance resolved.
    num_ffts : int
        Number of FFT segments per dissipation window,
        ``2 * (diss_length // fft_length) - 1`` for 50 % overlap.
        Required for the Lueck FM statistic; 0 skips FM.
    n_v : int
        Number of coherent-noise (vibration) signals removed by Goodman
        cleaning (reduces the effective degrees of freedom in FM).
        Pass 0 when Goodman cleaning was not applied.
    diss_length_s : float
        Dissipation window duration [s] (``diss_length / fs_fast``).
        Required for the inter-probe consistency flag (bit 4, via the
        Lueck 2022 variance model); 0 skips that flag.
    despike_fraction_limit : float
        Flag bit 2 when the fraction of shear samples replaced by
        despiking in a window exceeds this (ATOMIX
        ``despike_shear_fraction_limit``).
    despike_passes_limit : int
        Flag bit 8 when despiking needed more than this many passes
        (ATOMIX ``despike_shear_iterations_limit``).
    diss_ratio_limit : float
        z-factor for the inter-probe consistency test (bit 4):
        ``ln(e_i/e_min) > diss_ratio_limit * mean(sigma_ln)`` flags
        probe i.  The ATOMIX benchmark value is 1.96*sqrt(2).
    """
    n_spec = l3.n_spectra
    n_shear = l3.n_shear

    if n_spec == 0:
        return _empty_l4(n_shear)

    # Temperature for viscosity
    if temp is None:
        temp = l3.temp if l3.temp.size == n_spec else np.full(n_spec, 10.0)
    temp = np.asarray(temp, dtype=np.float64)
    if not np.all(np.isfinite(temp)):
        warnings.warn(
            "Non-finite window temperature(s); using 10 degC for viscosity "
            "in those windows",
            stacklevel=2,
        )
        temp = np.where(np.isfinite(temp), temp, 10.0)

    # Output arrays
    epsi = np.full((n_shear, n_spec), np.nan)
    fom = np.full((n_shear, n_spec), np.nan)
    mad = np.full((n_shear, n_spec), np.nan)
    kmax = np.full((n_shear, n_spec), np.nan)
    method_arr = np.full((n_shear, n_spec), np.nan)
    var_resolved = np.full((n_shear, n_spec), np.nan)
    FM = np.full((n_shear, n_spec), np.nan)
    nu_arr = np.full(n_spec, np.nan)

    for j in range(n_spec):
        # Per-window wavenumber grid: kcyc is (N_WAVENUMBER, N_SPECTRA)
        K = l3.kcyc[:, j]  # cpm
        W = l3.pspd_rel[j]
        nu = float(visc35(temp[j]))
        nu_arr[j] = nu

        # Anti-alias wavenumber limit
        K_AA = 0.9 * f_AA / max(W, 0.05)

        for i in range(n_shear):
            # Use cleaned spectra
            spec = l3.sh_spec_clean[i, :, j]

            if not np.any(np.isfinite(spec) & (spec > 0)):
                continue

            e, km, md, meth, fom_val, var_res, _nas, _kmr, fm, _eisr, _evar = _estimate_epsilon(
                K,
                spec,
                nu,
                K_AA,
                fit_order,
                num_ffts=num_ffts,
                n_v=n_v,
            )

            epsi[i, j] = e
            fom[i, j] = fom_val
            mad[i, j] = md
            kmax[i, j] = km
            method_arr[i, j] = meth
            var_resolved[i, j] = var_res
            FM[i, j] = fm

    # Expected sigma_ln per probe/window (Lueck 2022a variance model)
    # for the inter-probe consistency flag.
    sigma_ln = None
    if diss_length_s > 0:
        sigma_ln = _sigma_ln_epsilon(epsi, nu_arr, diss_length_s * l3.pspd_rel)

    # QC flags and EPSI_FINAL.  Flag on the MAD-based FM statistic when
    # available (the ATOMIX convention the fom_limit default of 1.15 was
    # written for); fall back to the variance-ratio fom otherwise.
    fom_for_flags = FM if num_ffts > 0 else fom
    epsi_flags = _compute_flags(
        epsi,
        fom_for_flags,
        var_resolved,
        fom_limit,
        var_resolved_limit,
        despike_fraction=l3.despike_fraction if l3.despike_fraction.size > 0 else None,
        despike_passes=l3.despike_passes if l3.despike_passes.size > 0 else None,
        despike_fraction_limit=despike_fraction_limit,
        despike_passes_limit=despike_passes_limit,
        sigma_ln=sigma_ln,
        diss_ratio_limit=diss_ratio_limit,
        method=method_arr,
    )
    epsi_final = _compute_epsi_final(epsi, epsi_flags)

    return L4Data(
        time=l3.time.copy(),
        pres=l3.pres.copy(),
        pspd_rel=l3.pspd_rel.copy(),
        section_number=l3.section_number.copy(),
        epsi=epsi,
        epsi_final=epsi_final,
        epsi_flags=epsi_flags,
        fom=fom,
        mad=mad,
        kmax=kmax,
        method=method_arr,
        var_resolved=var_resolved,
        FM=FM,
        despike_fraction=(
            l3.despike_fraction.copy() if l3.despike_fraction.size > 0 else None
        ),
    )


def _empty_l4(n_shear: int) -> L4Data:
    return L4Data(
        time=np.array([]),
        pres=np.array([]),
        pspd_rel=np.array([]),
        section_number=np.array([]),
        epsi=np.zeros((n_shear, 0)),
        epsi_final=np.array([]),
        epsi_flags=np.zeros((n_shear, 0)),
        fom=np.zeros((n_shear, 0)),
        mad=np.zeros((n_shear, 0)),
        kmax=np.zeros((n_shear, 0)),
        method=np.zeros((n_shear, 0)),
        var_resolved=np.zeros((n_shear, 0)),
        FM=np.zeros((n_shear, 0)),
    )


# ---------------------------------------------------------------------------
# Core epsilon estimation (per-spectrum)
# ---------------------------------------------------------------------------


def _estimate_epsilon(
    K: np.ndarray,
    shear_spectrum: np.ndarray,
    nu: float,
    K_AA: float,
    fit_order: int,
    e_isr_threshold: float = E_ISR_THRESHOLD,
    num_ffts: int = 0,
    n_v: int = 0,
    isr_margin: float = DEFAULT_ISR_MARGIN,
) -> tuple[float, float, float, int, float, float, np.ndarray, float, float, float, float]:
    """Estimate epsilon from a single shear wavenumber spectrum.

    This is the canonical implementation used by both the scor160 pipeline
    and the RSI ``get_diss()`` function.

    Parameters
    ----------
    K : ndarray
        Wavenumber vector [cpm].
    shear_spectrum : ndarray
        Observed shear spectrum [s^-2 cpm^-1].
    nu : float
        Kinematic viscosity [m^2/s].
    K_AA : float
        Anti-aliasing wavenumber limit [cpm].
    fit_order : int
        Polynomial order for spectral minimum detection.
    e_isr_threshold : float
        Epsilon threshold for switching to ISR method [W/kg].
    num_ffts : int
        Number of FFT segments (for Lueck FM statistic). 0 to skip FM.
    n_v : int
        Number of vibration signals removed by Goodman cleaning (for the
        Lueck FM statistic). Pass 0 when Goodman was not applied.
    isr_margin : float
        Multiplier on ``e_isr_threshold`` for the method switch (see
        ``DEFAULT_ISR_MARGIN``).  1.0 reproduces the ODAS convention.

    Returns
    -------
    epsilon : float
        Dissipation rate [W/kg].
    K_max : float
        Upper integration limit [cpm].
    mad : float
        Mean absolute deviation of spectral fit (log10).
    method : int
        0 = variance, 1 = inertial subrange.
    fom : float
        Figure of merit (observed/Nasmyth variance ratio).
    var_resolved : float
        Fraction of Nasmyth variance resolved.
    nasmyth_spectrum : ndarray
        Nasmyth spectrum at final epsilon.
    K_max_ratio : float
        K_max / K_95 spectral resolution ratio.
    FM : float
        Lueck (2022a,b) figure of merit (NaN if num_ffts == 0).
    epsilon_isr : float
        Inertial subrange epsilon estimate [W/kg].
    epsilon_var : float
        Variance method epsilon estimate [W/kg].
    """
    n_freq = len(K)

    # Corrupted spectral bins (ODAS get_diss_odas.m:577-587 convention):
    # a single interior NaN/Inf bin is interpolated from its neighbours;
    # multiple bad bins invalidate the estimate.  Zeroing bad bins (the
    # previous behaviour) silently removes variance, biasing epsilon low
    # and corrupting the log-space polynomial minimum search.
    bad = np.where(~np.isfinite(shear_spectrum))[0]
    if len(bad) == 0:
        spec_safe = shear_spectrum
    elif len(bad) == 1 and 0 < bad[0] < n_freq - 1:
        spec_safe = shear_spectrum.copy()
        spec_safe[bad[0]] = 0.5 * (spec_safe[bad[0] - 1] + spec_safe[bad[0] + 1])
    else:
        nan_spec = np.full(n_freq, np.nan)
        return (np.nan, np.nan, np.nan, 0, np.nan, np.nan, nan_spec, np.nan, np.nan, np.nan, np.nan)

    # Initial estimate: integrate to K_INITIAL_CUTOFF cpm.  K is monotone
    # nondecreasing (it's an FFT wavenumber grid divided by a positive
    # speed), so ``np.where(K <= cutoff)[0]`` is equivalent to
    # ``np.arange(searchsorted(K, cutoff, 'right'))`` — and slicing with
    # the count avoids building both the boolean and the index array.
    n_init = int(np.searchsorted(K, K_INITIAL_CUTOFF, side="right"))
    if n_init < 3:
        n_init = min(3, n_freq)

    e_10 = ISOTROPY_FACTOR * nu * np.trapezoid(spec_safe[:n_init], K[:n_init])
    if e_10 <= 0:
        e_10 = EPSILON_FLOOR
    e_1 = e_10 * np.sqrt(1 + LUECK_A * e_10)

    use_isr = e_1 >= e_isr_threshold * isr_margin

    # Always compute both estimates so viewers can show the alternative
    K_limit = min(K_AA, K_LIMIT_MAX)
    isr_e, isr_km, isr_range = _inertial_subrange(K, spec_safe, e_1, nu, K_limit)
    var_e, var_km, n_var = _variance_method(
        K,
        spec_safe,
        e_1,
        nu,
        K_AA,
        fit_order,
        n_freq,
    )

    # qc_idx is the wavenumber-bin index set the MAD/FOM/FM statistics
    # are computed over, excluding the DC bin (K[0]=0 by construction;
    # F starts at 0 in csd_matrix_batch and K=F/W with W>0).
    if use_isr:
        method = 1
        # Use the flyer-pruned ISR fit range, as ODAS computes MAD over
        # the flyer-reduced Range (get_diss_odas.m:660-671); a contiguous
        # range would re-include the excluded flyer bins.
        qc_idx = isr_range[K[isr_range] > 0]
        e_4, k_max = isr_e, isr_km
    else:
        method = 0
        e_4, k_max = var_e, var_km
        qc_idx = np.arange(1, max(n_var, 1))

    # Nasmyth spectrum at final epsilon
    nas_spec = nasmyth_grid(max(e_4, EPSILON_FLOOR), nu, K + 1e-30)

    if len(qc_idx) > 0:
        K_pos = K[qc_idx]
        spec_safe_pos = spec_safe[qc_idx]
        nas_spec_pos = nas_spec[qc_idx]
        spec_ratio = spec_safe_pos / (nas_spec_pos + 1e-30)
        spec_ratio = spec_ratio[spec_ratio > 0]
        mad_val = float(np.mean(np.abs(np.log10(spec_ratio)))) if len(spec_ratio) > 0 else np.nan
        obs_var = np.trapezoid(spec_safe_pos, K_pos)
        nas_var = np.trapezoid(nas_spec_pos, K_pos)
        fom_val = obs_var / nas_var if nas_var > 0 else np.nan
    else:
        mad_val = np.nan
        fom_val = np.nan
        spec_ratio = np.array([])

    # Variance resolved fraction
    var_res = _variance_resolved_fraction(k_max, e_4, nu)

    # K_max / K_95 spectral resolution ratio
    K_95 = X_95 * (max(e_4, EPSILON_FLOOR) / nu**3) ** 0.25
    K_max_ratio_val = k_max / K_95 if K_95 > 0 else np.nan

    # Lueck (2022a,b) figure of merit
    FM_val = np.nan
    if num_ffts > 0 and len(spec_ratio) > 0:
        N_s = len(spec_ratio)
        mad_ln = np.mean(np.abs(np.log(spec_ratio)))  # natural log
        N_eff = max(num_ffts - n_v, 1)
        sigma_ln = np.sqrt(FM_SIGMA_COEFF * N_eff**FM_SIGMA_EXPONENT)
        T_M = FM_TM_OFFSET + np.sqrt(FM_TM_COEFF / N_s)
        FM_val = mad_ln / (T_M * sigma_ln)

    return (
        e_4,
        k_max,
        mad_val,
        method,
        fom_val,
        var_res,
        nas_spec,
        K_max_ratio_val,
        FM_val,
        isr_e,
        var_e,
    )


def _variance_method(
    K: np.ndarray,
    spec_safe: np.ndarray,
    e_1: float,
    nu: float,
    K_AA: float,
    fit_order: int,
    n_freq: int,
) -> tuple[float, float, int]:
    """Variance integration method for epsilon estimation.

    Returns (epsilon, K_max, n_range), where ``n_range`` is the count
    of leading wavenumber bins used for the integration; the caller
    treats it as an implicit ``np.arange(n_range)``.
    """
    isr_limit = X_ISR * (e_1 / nu**3) ** 0.25
    # K is monotone nondecreasing — use searchsorted to count points
    # below isr_limit instead of building a bool+index array.
    if int(np.searchsorted(K, isr_limit, side="right")) >= 20:
        e_2, _, _ = _inertial_subrange(K, spec_safe, e_1, nu, min(K_LIMIT_MAX, K_AA))
    else:
        e_2 = e_1

    K_95 = X_95 * (e_2 / nu**3) ** 0.25
    valid_K_limit = min(K_AA, K_95)
    # Plain min/max on scalars produces the same float64 result as
    # ``np.clip`` here without the numpy dispatch overhead — and
    # ``np.clip`` was being called per spectrum in the cProfile hot-list.
    valid_K_limit = min(max(valid_K_limit, K_LIMIT_MIN), K_LIMIT_MAX)
    index_limit = int(np.searchsorted(K, valid_K_limit, side="right"))

    if index_limit <= 1:
        index_limit = min(3, n_freq)

    # Polynomial fit to find spectral minimum.  Zero bins (possible after
    # Goodman cleaning) are excluded: log10(0 + 1e-30) = -30 would enter
    # the fit as an extreme outlier and drag the fitted minimum.
    y = np.log10(spec_safe[1:index_limit] + 1e-30)
    x = np.log10(K[1:index_limit] + 1e-30)

    fit_order_eff = min(max(fit_order, 3), 8)
    K_limit_log = np.log10(K_95)

    if index_limit > fit_order_eff + 2:
        valid = np.isfinite(y) & np.isfinite(x) & (spec_safe[1:index_limit] > 0)
        if np.sum(valid) > fit_order_eff + 2:
            p = np.polyfit(x[valid], y[valid], fit_order_eff)
            pd1 = np.polyder(p)
            roots = np.roots(pd1)
            roots = roots[np.isreal(roots)].real
            pd2 = np.polyder(pd1)
            roots = roots[np.polyval(pd2, roots) > 0]
            roots = roots[roots >= np.log10(K_INITIAL_CUTOFF)]
            # np.roots returns eigenvalues in no particular order; take
            # the LOWEST qualifying minimum, as ODAS does via
            # ``pr1 = sort(roots(pd1))`` (get_diss_odas.m:590,597).
            K_limit_log = np.min(roots) if len(roots) > 0 else np.log10(K_95)

    # Final integration limit
    K_limit_log = min(K_limit_log, np.log10(K_95), np.log10(K_AA))
    K_limit_log = min(max(K_limit_log, np.log10(K_LIMIT_MIN)), np.log10(K_LIMIT_MAX))

    # K is monotone nondecreasing — searchsorted gives the count.  The
    # implicit Range is ``np.arange(n_var)``; we use ``[:n_var]`` slicing
    # below to avoid building the index array (slicing returns a view).
    n_var = int(np.searchsorted(K, 10**K_limit_log, side="right"))
    if n_var > 0 and K[n_var - 1] < K_LIMIT_MIN:
        n_var = min(n_var + 1, n_freq)
    if n_var < 3:
        n_var = min(3, n_freq)

    e_3 = ISOTROPY_FACTOR * nu * np.trapezoid(spec_safe[:n_var], K[:n_var])
    e_3 = max(e_3, EPSILON_FLOOR)

    K_last = K[n_var - 1]

    # Iterative variance correction
    e_4 = _variance_correction(e_3, K_last, nu)

    # Low-wavenumber correction
    if len(K) > 2:
        e_4_vc = e_4
        phi_low = nasmyth_grid(e_4, nu, K[1:3])
        e_4 = e_4 + LOW_K_CORRECTION_FACTOR * ISOTROPY_FACTOR * nu * K[1] * phi_low[0]
        if e_4 / e_4_vc > LOW_K_CORRECTION_THRESHOLD:
            e_4 = _variance_correction(e_4, K_last, nu)

    return e_4, K_last, n_var


def _compute_mad(
    K: np.ndarray,
    spec_safe: np.ndarray,
    epsilon: float,
    nu: float,
    Range: np.ndarray,
) -> float:
    """Compute MAD (log10) for a given epsilon estimate over a spectral range."""
    nas = nasmyth_grid(max(epsilon, EPSILON_FLOOR), nu, K + 1e-30)
    Range_noDC = Range[K[Range] > 0]
    if len(Range_noDC) == 0:
        return np.inf
    ratio = spec_safe[Range_noDC] / (nas[Range_noDC] + 1e-30)
    ratio = ratio[ratio > 0]
    if len(ratio) == 0:
        return np.inf
    return float(np.mean(np.abs(np.log10(ratio))))


def _variance_correction(e_3: float, K_upper: float, nu: float, max_iter: int = 50) -> float:
    """Iterative variance correction using Lueck's resolved-variance model."""
    e_new = e_3
    for _ in range(max_iter):
        x_limit = K_upper * (nu**3 / e_new) ** 0.25
        x_limit = x_limit ** (4 / 3)
        variance_resolved = np.tanh(
            VARIANCE_TANH_COEFF * x_limit
        ) - VARIANCE_EXP_COEFF * x_limit * np.exp(-VARIANCE_EXP_DECAY * x_limit)
        if variance_resolved <= 0:
            break
        e_old = e_new
        e_new = e_3 / variance_resolved
        if e_new / e_old < VARIANCE_CONVERGENCE:
            break
    return e_new


def _variance_resolved_fraction(K_upper: float, epsilon: float, nu: float) -> float:
    """Fraction of Nasmyth variance resolved up to K_upper."""
    epsilon = max(epsilon, EPSILON_FLOOR)
    x_limit = K_upper * (nu**3 / epsilon) ** 0.25
    x_limit = x_limit ** (4 / 3)
    vr = np.tanh(VARIANCE_TANH_COEFF * x_limit) - VARIANCE_EXP_COEFF * x_limit * np.exp(
        -VARIANCE_EXP_DECAY * x_limit
    )
    return float(np.clip(vr, 0, 1))


def _inertial_subrange(
    K: np.ndarray, shear_spectrum: np.ndarray, e: float, nu: float, K_limit: float
) -> tuple[float, float, np.ndarray]:
    """Fit to the inertial subrange to estimate epsilon.

    Returns (epsilon, K_max, fit_range) where ``fit_range`` is the index
    array of wavenumber bins retained after flyer removal (used by the
    caller for the MAD/FOM/FM statistics).
    """
    isr_limit = min(X_ISR * (e / nu**3) ** 0.25, K_limit)
    # K is monotone nondecreasing — searchsorted is equivalent to
    # ``np.where(isr_limit >= K)[0]`` while skipping the bool array.
    n_fit = int(np.searchsorted(K, isr_limit, side="right"))
    if n_fit < 3:
        n_fit = min(3, len(K))
    fit_range = np.arange(n_fit)

    k_max = K[n_fit - 1]

    # Cache the post-DC slices that the iterative loops actually need.
    # The previous code computed nas at K_safe[0] = K[0]+1e-30 ≈ 0 and
    # threw it away via ``nas[1:]``; passing K_inner = K[1:n_fit]+1e-30
    # to nasmyth_grid trims that wasted element from each interp call
    # (8138 spectra * 6 calls each = ~49k calls).  fit_range[0] = 0
    # always (the flyer loop never marks keep[0] = False), so
    # fit_range[1:] is the contiguous K[1:n_fit] view here.
    K_inner = K[1:n_fit] + 1e-30
    shear_tail = shear_spectrum[1:n_fit]

    # Iterative fitting (3 passes)
    for _ in range(3):
        nas = nasmyth_grid(max(e, EPSILON_FLOOR), nu, K_inner)
        ratio = shear_tail / (nas + 1e-30)
        ratio = ratio[ratio > 0]
        if len(ratio) == 0:
            break
        fit_error = float(np.mean(np.log10(ratio)))
        e = e * 10 ** (3 * fit_error / 2)

    # Remove flyers.  The error vector keeps full length (non-positive
    # ratio bins become NaN, which compares False below) so that index i
    # in ``fit_error_vec`` always corresponds to ``fit_range[i+1]`` —
    # compressing the array first (the previous behaviour) misaligned
    # the flyer indices whenever a zero bin was present.  Note: ODAS
    # itself has an off-by-one here (get_diss_odas.m:790-797 deletes
    # ``fit_range(index)`` where ``index`` refers to ``fit_range(2:end)``);
    # the ``keep[b + 1]`` mapping below is the corrected version.
    nas = nasmyth_grid(max(e, EPSILON_FLOOR), nu, K_inner)
    if len(fit_range) > 2:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_full = shear_tail / (nas + 1e-30)
            fit_error_vec = np.where(ratio_full > 0, np.log10(ratio_full), np.nan)
        with np.errstate(invalid="ignore"):
            bad = np.where(np.abs(fit_error_vec) > ISR_FLYER_THRESHOLD)[0]
        if len(bad) > 0:
            bad_limit = max(1, int(np.ceil(ISR_FLYER_FRAC * len(fit_range))))
            if len(bad) > bad_limit:
                order = np.argsort(fit_error_vec[bad])[::-1]
                bad = bad[order[:bad_limit]]
            keep = np.ones(len(fit_range), dtype=bool)
            for b in bad:
                if b + 1 < len(keep):
                    keep[b + 1] = False
            fit_range = fit_range[keep]
            k_max = K[fit_range[-1]]
            # fit_range may now be non-contiguous — fancy index.
            K_inner = K[fit_range[1:]] + 1e-30
            shear_tail = shear_spectrum[fit_range[1:]]

    # Re-fit (2 more passes)
    for _ in range(2):
        nas = nasmyth_grid(max(e, EPSILON_FLOOR), nu, K_inner)
        ratio = shear_tail / (nas + 1e-30)
        ratio = ratio[ratio > 0]
        if len(ratio) == 0:
            break
        fit_error = float(np.mean(np.log10(ratio)))
        e = e * 10 ** (3 * fit_error / 2)

    return e, k_max, fit_range


# ---------------------------------------------------------------------------
# QC flags and EPSI_FINAL
# ---------------------------------------------------------------------------


def _sigma_ln_epsilon(
    epsi: np.ndarray, nu: np.ndarray, L: np.ndarray
) -> np.ndarray:
    """Expected sigma of ln(epsilon) per probe/window (Lueck 2022a).

    var(ln eps) = 5.5 / (1 + (L_hat/4)^(7/9)) with L_hat = L/L_K and
    L_K = (nu^3/eps)^(1/4) — the same model used by mk_epsilon_mean.

    Parameters
    ----------
    epsi : (n_shear, n_spec)
    nu : (n_spec,)
        Kinematic viscosity per window [m^2/s].
    L : (n_spec,)
        Physical dissipation-window length [m] (duration * speed).
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        L_K = (nu[np.newaxis, :] ** 3 / epsi) ** 0.25
        L_hat = L[np.newaxis, :] / L_K
        var_ln = 5.5 / (1.0 + (L_hat / 4.0) ** (7.0 / 9.0))
    return np.asarray(np.sqrt(var_ln))


def _compute_flags(
    epsi: np.ndarray,
    fom: np.ndarray,
    var_resolved: np.ndarray,
    fom_limit: float,
    var_resolved_limit: float,
    despike_fraction: np.ndarray | None = None,
    despike_passes: np.ndarray | None = None,
    despike_fraction_limit: float = DEFAULT_DESPIKE_FRACTION_LIMIT,
    despike_passes_limit: int = DEFAULT_DESPIKE_PASSES_LIMIT,
    sigma_ln: np.ndarray | None = None,
    diss_ratio_limit: float = DEFAULT_DISS_RATIO_LIMIT,
    method: np.ndarray | None = None,
) -> np.ndarray:
    """Compute ATOMIX QC flags (conventions: shear probes group).

    Bit flags (matching the benchmark EPSI_FLAGS flag_meanings):
      0 = good
      1 = FOM > limit
      2 = despike fraction > limit
      4 = inter-probe consistency: ln(e_i/e_min) > limit * mean(sigma_ln)
          — only the larger (suspect) estimates are flagged, never the
          window minimum, matching the benchmark convention
      8 = too many despike passes
      16 = variance resolved < limit — applied only to variance-method
          estimates (method 0): the ISR fit never integrates the
          dissipation range, so the criterion is not applicable there
          (matching the benchmark, which leaves ISR estimates with
          var_resolved ~ 0.1 unflagged)
      255 = invalid estimate
    """
    flags = np.zeros_like(epsi, dtype=np.float64)
    flags[fom > fom_limit] += 1

    if despike_fraction is not None and despike_fraction.shape == epsi.shape:
        with np.errstate(invalid="ignore"):
            flags[despike_fraction > despike_fraction_limit] += 2

    if sigma_ln is not None and epsi.shape[0] > 1:
        with np.errstate(invalid="ignore", divide="ignore"):
            epsi_pos = np.where(np.isfinite(epsi) & (epsi > 0), epsi, np.nan)
            e_min = np.nanmin(epsi_pos, axis=0)  # (n_spec,)
            ln_ratio = np.log(epsi_pos / e_min[np.newaxis, :])
            mu_sigma = np.nanmean(
                np.where(np.isfinite(epsi_pos), sigma_ln, np.nan), axis=0
            )
            inconsistent = ln_ratio > diss_ratio_limit * mu_sigma[np.newaxis, :]
        flags[inconsistent] += 4

    if despike_passes is not None and len(despike_passes) == epsi.shape[0]:
        flags[despike_passes > despike_passes_limit, :] += 8

    under_resolved = var_resolved < var_resolved_limit
    if method is not None:
        under_resolved &= method == 0
    flags[under_resolved] += 16
    flags[~np.isfinite(epsi)] = 255
    return flags


def _compute_epsi_final(epsi: np.ndarray, flags: np.ndarray) -> np.ndarray:
    """Compute EPSI_FINAL: geometric mean of probes with flags == 0."""
    _n_shear, n_spec = epsi.shape
    epsi_final = np.full(n_spec, np.nan)
    for j in range(n_spec):
        good = (flags[:, j] == 0) & np.isfinite(epsi[:, j]) & (epsi[:, j] > 0)
        if good.any():
            epsi_final[j] = np.exp(np.mean(np.log(epsi[good, j])))
        else:
            # Fall back to geometric mean of all finite probes
            finite = np.isfinite(epsi[:, j]) & (epsi[:, j] > 0)
            if finite.any():
                epsi_final[j] = np.exp(np.mean(np.log(epsi[finite, j])))
    return epsi_final
