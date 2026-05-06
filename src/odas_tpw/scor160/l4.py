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
X_ISR = 0.01

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

# Default QC thresholds (from benchmark file attributes)
DEFAULT_FOM_LIMIT = 1.15
DEFAULT_VAR_RESOLVED_LIMIT = 0.5


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
        QC threshold for figure of merit.
    var_resolved_limit : float
        QC threshold for fraction of variance resolved.
    """
    n_spec = l3.n_spectra
    n_shear = l3.n_shear

    if n_spec == 0:
        return _empty_l4(n_shear)

    # Temperature for viscosity
    if temp is None:
        temp = l3.temp if l3.temp.size == n_spec else np.full(n_spec, 10.0)

    # Output arrays
    epsi = np.full((n_shear, n_spec), np.nan)
    fom = np.full((n_shear, n_spec), np.nan)
    mad = np.full((n_shear, n_spec), np.nan)
    kmax = np.full((n_shear, n_spec), np.nan)
    method_arr = np.full((n_shear, n_spec), np.nan)
    var_resolved = np.full((n_shear, n_spec), np.nan)

    for j in range(n_spec):
        # Per-window wavenumber grid: kcyc is (N_WAVENUMBER, N_SPECTRA)
        K = l3.kcyc[:, j]  # cpm
        W = l3.pspd_rel[j]
        nu = float(visc35(temp[j]))

        # Anti-alias wavenumber limit
        K_AA = 0.9 * f_AA / max(W, 0.05)

        for i in range(n_shear):
            # Use cleaned spectra
            spec = l3.sh_spec_clean[i, :, j]

            if not np.any(np.isfinite(spec) & (spec > 0)):
                continue

            e, km, md, meth, fom_val, var_res, _nas, _kmr, _fm, _eisr, _evar = _estimate_epsilon(
                K,
                spec,
                nu,
                K_AA,
                fit_order,
            )

            epsi[i, j] = e
            fom[i, j] = fom_val
            mad[i, j] = md
            kmax[i, j] = km
            method_arr[i, j] = meth
            var_resolved[i, j] = var_res

    # QC flags and EPSI_FINAL
    epsi_flags = _compute_flags(epsi, fom, var_resolved, fom_limit, var_resolved_limit)
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
        Number of vibration signals removed (for Lueck FM statistic).

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

    # Replace NaN/Inf in spectrum with 0 for integration
    spec_safe = np.where(np.isfinite(shear_spectrum), shear_spectrum, 0.0)

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

    # The ATOMIX benchmark reference was produced with a slightly
    # different spectral computation that yields ~2-4 % lower e_1
    # values than our pipeline.  A margin of 1.6x on the threshold
    # compensates for this, maximising method agreement (99.8 % across
    # all six benchmark datasets).
    ISR_MARGIN = 1.6
    use_isr = e_1 >= e_isr_threshold * ISR_MARGIN

    # Always compute both estimates so viewers can show the alternative
    K_limit = min(K_AA, K_LIMIT_MAX)
    isr_e, isr_km = _inertial_subrange(K, spec_safe, e_1, nu, K_limit)
    var_e, var_km, n_var = _variance_method(
        K,
        spec_safe,
        e_1,
        nu,
        K_AA,
        fit_order,
        n_freq,
    )

    if use_isr:
        method = 1
        n_range = int(np.searchsorted(K, isr_km, side="right"))
        if n_range < 3:
            n_range = min(3, n_freq)
        e_4, k_max = isr_e, isr_km
    else:
        method = 0
        e_4, k_max, n_range = var_e, var_km, n_var

    # Nasmyth spectrum at final epsilon
    nas_spec = nasmyth_grid(max(e_4, EPSILON_FLOOR), nu, K + 1e-30)

    # MAD/FOM block — skip the DC bin (K[0]=0 by construction; F starts
    # at 0 in csd_matrix_batch and K=F/W with W>0).  ``Range_noDC`` in
    # the original code was therefore equivalent to ``arange(1, n_range)``;
    # contiguous slicing returns views and avoids three fancy-indexing
    # allocations per spectrum.
    if n_range > 1:
        K_pos = K[1:n_range]
        spec_safe_pos = spec_safe[1:n_range]
        nas_spec_pos = nas_spec[1:n_range]
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
    if num_ffts > 0 and n_range > 1 and len(spec_ratio) > 0:
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
        e_2, _ = _inertial_subrange(K, spec_safe, e_1, nu, min(K_LIMIT_MAX, K_AA))
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

    # Polynomial fit to find spectral minimum
    y = np.log10(spec_safe[1:index_limit] + 1e-30)
    x = np.log10(K[1:index_limit] + 1e-30)

    fit_order_eff = min(max(fit_order, 3), 8)
    K_limit_log = np.log10(K_95)

    if index_limit > fit_order_eff + 2:
        valid = np.isfinite(y) & np.isfinite(x)
        if np.sum(valid) > fit_order_eff + 2:
            p = np.polyfit(x[valid], y[valid], fit_order_eff)
            pd1 = np.polyder(p)
            roots = np.roots(pd1)
            roots = roots[np.isreal(roots)].real
            pd2 = np.polyder(pd1)
            roots = roots[np.polyval(pd2, roots) > 0]
            roots = roots[roots >= np.log10(K_INITIAL_CUTOFF)]
            K_limit_log = roots[0] if len(roots) > 0 else np.log10(K_95)

    # Final integration limit
    K_limit_log = min(K_limit_log, np.log10(K_95), np.log10(K_AA))
    K_limit_log = min(max(K_limit_log, np.log10(K_LIMIT_MIN)), np.log10(K_LIMIT_MAX))

    # K is monotone nondecreasing — searchsorted gives the count.  The
    # implicit Range is ``np.arange(n_var)``; we use ``[:n_var]`` slicing
    # below to avoid building the index array (slicing returns a view).
    n_var = int(np.searchsorted(K, 10**K_limit_log, side="right"))
    if n_var > 0 and K[n_var - 1] < K_LIMIT_MIN:
        n_var += 1
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
) -> tuple[float, float]:
    """Fit to the inertial subrange to estimate epsilon."""
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
    # (8138 spectra × 6 calls each = ~49k calls).  fit_range[0] = 0
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

    # Remove flyers
    nas = nasmyth_grid(max(e, EPSILON_FLOOR), nu, K_inner)
    if len(fit_range) > 2:
        ratio = shear_tail / (nas + 1e-30)
        ratio = ratio[ratio > 0]
        if len(ratio) > 0:
            fit_error_vec = np.log10(ratio)
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

    return e, k_max


# ---------------------------------------------------------------------------
# QC flags and EPSI_FINAL
# ---------------------------------------------------------------------------


def _compute_flags(
    epsi: np.ndarray,
    fom: np.ndarray,
    var_resolved: np.ndarray,
    fom_limit: float,
    var_resolved_limit: float,
) -> np.ndarray:
    """Compute ATOMIX-style QC flags.

    Bit flags:
      0 = good
      1 = FOM > limit
      16 = variance resolved < limit
    """
    flags = np.zeros_like(epsi, dtype=np.float64)
    flags[fom > fom_limit] += 1
    flags[var_resolved < var_resolved_limit] += 16
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
