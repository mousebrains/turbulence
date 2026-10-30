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
    # Fraction of the model temperature-gradient variance resolved in
    # [K_min, K_max] (Batchelor V_f; see _batchelor_resolved_fraction). Feeds
    # the Lueck (2022) eq (18) L_hat*V_f^0.75 truncation in chiLnSigma. Trailing
    # default so failure-path constructions need not supply it. (#104 U4-F1.)
    var_resolved: float = np.nan
    # Spread of log10(observed/model) across the fom band [dex]: the SHAPE
    # diagnostic fom cannot be (see _shape_mad, issue #191). NaN wherever fom is.
    mad_shape: float = np.nan


class ChiFitResult(NamedTuple):
    """Result from Methods 2 & 3: Batchelor spectrum fitting."""

    kB: float
    chi: float
    epsilon: float
    K_max: float
    spec_batch: np.ndarray
    fom: float
    K_max_ratio: float
    var_resolved: float = np.nan  # Batchelor V_f; see ChiEpsilonResult.
    mad_shape: float = np.nan  # shape statistic [dex]; see ChiEpsilonResult.


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


def _batchelor_resolved_fraction(
    kB: float,
    K_max: float,
    grad_func,
    K_min: float = 0.0,
) -> float:
    """Fraction of the model temperature-gradient variance resolved in [K_min, K_max].

    The chi-side analog of the shear/Nasmyth ``_variance_resolved_fraction``
    (``scor160.l4``, Lueck 2022 Part 1 eq (17)):
    ``V_f = integral_{K_min}^{K_max} S(k) dk / integral_0^inf S(k) dk`` for the
    model (Batchelor/Kraichnan) temperature-gradient spectrum ``S``.

    Unlike :func:`_variance_correction` this does **not** apply the FP07 ``|H|²``
    attenuation: it is the pure spectral band-truncation fraction, analogous to
    the shear V_f (which is likewise a band fraction of the universal Nasmyth
    spectrum, with no sensor response). It feeds the Lueck 2022 Part 1 eq (18)
    degrees-of-freedom reduction ``L_hat_f = L_hat * V_f**0.75`` used to widen
    ``chiLnSigma`` when the Batchelor spectrum is truncated at ``K_max``.
    Including ``|H|²`` here would double-count the sensor response that the chi
    amplitude correction (``_variance_correction``'s ``V_total/V_resolved``)
    already undoes. (#104 U4-F1 chi-side follow-up.)

    One difference from the shear V_f: this uses the *same* ``[K_min, K_max]``
    band as the chi amplitude correction (band-consistent, so the two are not
    double-counted), whereas the shear ``_variance_resolved_fraction`` integrates
    ``[0, K_upper]`` with no low-k bound. Because the chi fit band starts at
    ``K_min ~ 0.044*kB``, a few percent of the gradient variance sits below
    ``K_min``, so this V_f is ~0.97 (not 1.0) even for a window fully resolved at
    the top — chiLnSigma therefore gets a small non-zero widening the epsilon
    path never applies.

    Shares the kB-scaled integration grid of :func:`_variance_correction`
    (span ``_K_SPAN_KB*kB`` at ``_PTS_PER_KB`` points per kB) so ``V_total`` is
    fully captured regardless of how far ``K_max`` is pushed out. Returns the
    fraction clipped to ``[0, 1]``, or ``NaN`` if the model variance cannot be
    integrated (non-finite ``kB`` etc.).
    """
    if not (0 < kB < np.inf):
        return np.nan
    K_upper = _K_SPAN_KB * kB
    n = _K_SPAN_KB * _PTS_PER_KB
    K_fine = np.arange(1, n + 1, dtype=np.float64) * (K_upper / n)
    spec_fine = grad_func(K_fine, kB, 1.0)  # chi=1.0 cancels in the ratio
    V_total = np.trapezoid(spec_fine, K_fine)
    if not (V_total > 0):
        return np.nan
    # Same band the correction uses: [K_min, min(K_max, K_upper)], but without |H|².
    mask = (K_fine >= K_min) & (K_fine <= min(K_max, K_upper))
    if not np.any(mask):
        return np.nan
    V_resolved = np.trapezoid(spec_fine[mask], K_fine[mask])
    return float(np.clip(V_resolved / V_total, 0.0, 1.0))


def _shape_mad(spec_obs: np.ndarray, spec_model: np.ndarray, min_points: int = 3) -> float:
    """Spread of ``log10(observed / model)`` across the fit band [dex].

    ``mad_shape = mean_k |r_k - median(r)|`` with ``r_k = log10(obs_k / model_k)``,
    where ``model`` is the same attenuated model the figure of merit uses
    (Batchelor/Kraichnan x FP07 ``|H|^2`` + noise) and the bins are the same
    bins ``fom`` integrates over.

    ``fom`` is an INTEGRATED variance ratio, so a mid-band deficit cancels
    against excesses at the band ends: on CASPER-West a model wrong by ~2x
    bin-by-bin scored fom = 1.001 (issue #191). This is the per-bin view that
    cannot cancel.

    It is the epsilon side's ``mad`` (``scor160.l4``: ``mean |log10(obs/Nasmyth)|``)
    with one change: the deviation is taken about the window's own MEDIAN log
    ratio instead of about 0. The median minimizes a mean absolute deviation,
    so this is the epsilon-style ``mad`` after the best constant rescaling of
    the model -- the part of the misfit that no choice of level can remove. It
    is therefore a diagnostic of SHAPE, not of level (level is ``fom``'s job).
    Centering also absorbs the bias of a log spectral estimate: a chi-squared
    estimate with ``d`` degrees of freedom has its median, and the mean of its
    log, BELOW its mean (``E[ln X] = psi(d/2) - ln(d/2)``), which about 0 would
    read as misfit.

    A perfect model does not give 0. Spectral scatter alone floors it at the
    mean absolute deviation of ``log10(chi2_d / d)`` about its median
    (:func:`mad_shape_floor`): 0.214, 0.166, 0.138, 0.115 dex for d = 6, 9.5,
    13.3, 19. ``d`` is the window's OWN spectral DOF -- a BBL window averages
    fewer FFT segments than the file's ``dof_spec`` attribute assumes, so its
    floor is higher. Adjacent Welch bins are correlated, which leaves that
    expectation alone but widens the scatter of the statistic between windows.
    With few bins it reads low (one term of ``|r - median|`` is always 0 for
    odd n).

    Bins with a non-finite or non-positive ratio are dropped (the epsilon
    ``mad`` likewise drops ``ratio <= 0``); fewer than *min_points* usable bins,
    the same 3-bin floor ``fom`` requires, returns NaN. A diagnostic only:
    nothing gates on it.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.asarray(spec_obs, dtype=np.float64) / np.asarray(spec_model, dtype=np.float64)
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    if ratio.size < min_points:
        return np.nan
    r = np.log10(ratio)
    return float(np.mean(np.abs(r - np.median(r))))


def mad_shape_floor(dof: float) -> float:
    """``mad_shape`` of a PERFECT model with ``dof`` spectral degrees of freedom [dex].

    ``E|log10(X) - median(log10 X)|`` for ``X ~ chi2_dof / dof``: what chi-squared
    scatter alone gives in a band of many bins. Evaluate it at the window's own
    DOF, ``1.9 * (n_segments - n_v)`` (``n_v`` = vibration signals Goodman
    removed). A chi product's ``dof_spec`` attribute is that value for its
    ``diss_length``; near-bed windows laid out with ``bbl_diss_length`` average
    fewer segments, so their floor is higher than ``mad_shape_floor(dof_spec)``.
    """
    from scipy import integrate, stats

    if not (dof > 0 and np.isfinite(dof)):
        return np.nan
    dist = stats.chi2(dof, scale=1.0 / dof)
    med = float(dist.median())
    m = np.log10(med)

    def integrand(x: float) -> float:
        return abs(np.log10(x) - m) * float(dist.pdf(x))

    lo = integrate.quad(integrand, 0.0, med, limit=200)[0]
    hi = integrate.quad(integrand, med, np.inf, limit=200)[0]
    return float(lo + hi)


# ---------------------------------------------------------------------------
# Shared wavenumber masking helper
# ---------------------------------------------------------------------------

# The lowest FFT bin any chi estimate uses, counting DC as bin 0 (issue #191
# step 2). Every FFT segment is LINEARLY detrended, and a straight line is not
# orthogonal to bin 1's one-cycle sinusoid, so the detrend removes part of bin
# 1's own variance while leakage from its neighbors adds some back: bin 1 reads
# the local spectral SLOPE as much as the level. Through csd_matrix_batch (fft
# 512, long records) it reads ln -0.33 on a flat spectrum, +0.22 on one rising
# as k and -0.59 on one falling as 1/k. At real window geometry with the
# low-wavenumber variance real spectra carry, it is low by a median ln 0.36 /
# 0.38 / 0.31 (fft_length 256 / 512 / 1024) with bins 2-8 within 3%; with
# Kraichnan + noise alone, 2-5% low. So bin 1 does not estimate the spectrum at
# its own wavenumber. ``chi.first_bin: 1`` restores the earlier band (bin 1 and
# up) to reproduce a product made before the change.
FIRST_BIN = 2


def validate_first_bin(value: object, where: str = "chi.first_bin") -> int:
    """*value* as the lowest FFT bin chi may use (an int >= 1), or ValueError.

    Bin 0 is DC and never usable. A bool is refused although Python counts it
    as an int (``True`` would silently mean bin 1), and so is a float, even a
    whole one: a bin is an index.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{where}: must be an integer FFT bin >= 1, got {value!r}")
    if value < 1:
        raise ValueError(
            f"{where}: must be >= 1 (bin 0 is DC, which chi never uses), got {value}"
        )
    return int(value)


# The fewest bins any chi path can work with (Method 1's band, fom, the detection
# gate); Method 2 needs 6.
_MIN_BAND_BINS = 3


def check_first_bin_fits(first_bin: int, n_freq: int, where: str = "chi.first_bin") -> None:
    """ValueError when *first_bin* leaves fewer than 3 bins up to Nyquist.

    *n_freq* is the one-sided grid's length, ``fft_length // 2 + 1`` (DC and
    Nyquist included). Such a value would make every window NaN; refusing it
    names the cause instead of a non-detection warning per window. Whether
    enough bins remain below the anti-alias limit depends on the speed, so that
    is reported per window (:func:`_first_bin_starves`).
    """
    left = int(n_freq) - int(first_bin)
    if left < _MIN_BAND_BINS:
        raise ValueError(
            f"{where} = {first_bin} leaves {max(left, 0)} FFT bin(s) up to Nyquist at "
            f"fft_length {2 * (int(n_freq) - 1)}; chi needs at least {_MIN_BAND_BINS}"
        )


def _first_bin_starves(K: np.ndarray, K_AA: float, first_bin: int, need: int) -> str | None:
    """Why a window has too few bins, when *first_bin* is the reason; else None.

    Fewer than *need* bins lie from bin *first_bin* up to ``K_AA`` while bin 1
    and up would give enough. The window is NaN either way; this only makes the
    warning name the cause. ``None`` when bin 1 and up would be too few as well
    (the anti-alias limit's doing) and always at ``first_bin = 1``, so those
    paths warn exactly as before.
    """
    if first_bin <= 1:
        return None
    in_aa = K <= K_AA
    n = int(np.sum(_usable_bins(K, first_bin) & in_aa))
    if n >= need:
        return None
    n1 = int(np.sum((K > 0) & in_aa))
    if n1 < need:
        return None
    return (
        f"only {n} FFT bin(s) from first_bin = {first_bin} up to K_AA = {K_AA:.3g} cpm "
        f"({n1} from bin 1); chi needs {need}, so it is NaN"
    )


def _usable_bins(K: np.ndarray, first_bin: int = FIRST_BIN) -> np.ndarray:
    """Bins a chi estimate may use: positive wavenumber, at or above bin *first_bin*.

    DC (``K <= 0``) is never usable, and of the positive-wavenumber bins the
    lowest ``first_bin - 1`` are dropped. On the production grid (``K[0]`` = DC,
    ``K[i]`` = bin ``i``) that is exactly ``index >= first_bin``; a caller whose
    grid starts at bin 1 loses the same bins. *K* is monotone increasing, as
    every FFT grid is. ``first_bin = 1`` is ``K > 0``, the band before
    :data:`FIRST_BIN` existed.
    """
    usable = np.asarray(K) > 0
    if first_bin > 1:
        usable[np.flatnonzero(usable)[: first_bin - 1]] = False
    return usable


def _band_slice(valid: np.ndarray) -> slice:
    """The CONTIGUOUS index band spanned by a selection mask.

    ``_valid_wavenumber_mask`` returns a possibly non-contiguous mask, and
    ``np.trapezoid(y[mask], K[mask])`` integrates the SUBSET GRID: every excluded
    interval is bridged with a straight line between its neighbors, so
    unobserved area is counted as measured. Three isolated excess bins at
    k = 1, 80, 160 integrate to 53x their true area on the original grid; on the
    repo's own VMP corpus (29 files, 2944 probe-windows) 28.8% of selected bands
    have an interior hole, 5.50% of windows were inflated by more than 5% and
    1.19% by more than 50%, worst case 3.79x (issue #180 F03).

    Integrating over ``[first, last]`` on the ORIGINAL grid keeps the true bin
    weights, and it is also what the rest of the estimator already assumes:
    ``_variance_correction`` and ``_batchelor_resolved_fraction`` both treat
    ``[K_min, K_max]`` as a CONTINUOUS interval, so a subset-grid integral
    contradicts its own correction factor. Sub-threshold bins inside the band
    contribute their own (small, noise-clipped) excess rather than a bridge.
    """
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return slice(0, 0)
    return slice(int(idx[0]), int(idx[-1]) + 1)


def _valid_wavenumber_mask(
    spec_obs: np.ndarray,
    noise: np.ndarray,
    K: np.ndarray,
    K_AA: float,
    min_points: int = 3,
    first_bin: int = FIRST_BIN,
) -> np.ndarray:
    """Build a boolean mask for valid wavenumber points.

    Selects points where the observed spectrum exceeds 2x noise, at or above
    FFT bin *first_bin* (:func:`_usable_bins`), and K <= K_AA.  Falls back to
    every usable bin with K <= K_AA if too few points pass the noise criterion.

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
    first_bin : int
        Lowest FFT bin used, DC = bin 0 (:data:`FIRST_BIN`).

    Returns
    -------
    mask : ndarray of bool
    """
    above_noise = spec_obs > 2 * noise
    usable = _usable_bins(K, first_bin) & (K <= K_AA)
    mask = above_noise & usable
    if np.sum(mask) < min_points:
        mask = usable
    return mask


# Window-level detection floor. A FIXED "3 bins above 2x noise" ignores how many
# bins were searched and how tight each one is: with 7 overlapping FFTs the
# per-bin chi^2 scatter alone puts several hundred bins over 2x the mean, so
# 554/1000 pure-noise windows produced a finite chi -- 393 of them inside the
# two-sided FOM band, which cannot catch it because the model amplitude is
# fitted to the same observations (issue #180 F03). These scale the floor with
# the searched bandwidth and the window's own degrees of freedom.
#
# HONEST LIMIT: Welch bins from 50%-overlapped segments are CORRELATED, so the
# binomial tail below understates the true spread of the null count. The
# inflation factor is a deliberate allowance for that, not a derivation, and the
# achieved false-positive rate is MEASURED by a Monte-Carlo regression rather
# than asserted. This is a calibrated floor, not the full window-level null test
# the review asks for.
_DETECT_Z = 3.0  # one-sided normal margin on the null count
_DETECT_CORR_INFLATION = 2.0  # variance allowance for overlapped-Welch correlation


def detection_floor(n_bins: int, dof: float, min_points: int = 3) -> int:
    """Bins that must clear 2x noise before a window counts as a detection.

    ``dof`` is the spectral degrees of freedom per bin (this package's
    convention: ``1.9 * (num_ffts - n_vib_removed)``).  ``dof <= 0`` means
    "unknown" and returns ``min_points`` — exactly the pre-#180 behaviour, so an
    older product or a direct call is never made stricter by accident.
    """
    if not (dof and dof > 0) or n_bins <= 0:
        return min_points
    from scipy.stats import chi2

    p = float(chi2.sf(2.0 * dof, dof))  # P(bin > 2x its mean) under noise only
    mu = n_bins * p
    sd = np.sqrt(max(mu * (1.0 - p), 0.0) * _DETECT_CORR_INFLATION)
    return max(min_points, int(np.ceil(mu + _DETECT_Z * sd)))


def _below_detection(
    spec_obs: np.ndarray,
    noise: np.ndarray,
    K: np.ndarray,
    K_AA: float,
    min_points: int = 3,
    dof: float = 0.0,
    first_bin: int = FIRST_BIN,
) -> bool:
    """True when a window is a non-detection.

    Fewer than :func:`detection_floor` bins clear 2x the noise floor within the
    resolved band, bin *first_bin* up to ``K_AA`` — the strict ``above_noise``
    criterion of :func:`_valid_wavenumber_mask` *without* its whole-band
    fallback.  Used to reject noise-only windows before any chi path (the
    known-epsilon integral or an MLE Batchelor fit) can shape a curve to the
    noise floor and emit a spurious finite chi whose figure of merit (model
    fitted to obs) sits near 1. Defined once so all three chi paths share the
    identical detection limit, over the same bins they may use.
    """
    in_band = _usable_bins(K, first_bin) & (K <= K_AA)
    above_noise = (spec_obs > 2.0 * noise) & in_band
    need = detection_floor(int(np.sum(in_band)), dof, min_points)
    return int(np.sum(above_noise)) < need


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
    dof: float = 0.0,
    first_bin: int = FIRST_BIN,
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
    dof : float
        Spectral degrees of freedom of the window, for the detection floor
        (:func:`detection_floor`); 0 = unknown.
    first_bin : int
        Lowest FFT bin used, DC = bin 0 (:data:`FIRST_BIN`). *K* is the full
        one-sided FFT grid, so ``K[i]`` is bin ``i``. The detection gate, the
        band, the variance correction's ``K_min``, ``fom``, ``mad_shape`` and
        ``var_resolved`` all start there.

    Returns
    -------
    ChiEpsilonResult
        Named tuple: (chi, kB, K_max, spec_batch, fom, K_max_ratio, var_resolved,
        mad_shape).
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

    # Signal-presence gate.  A window whose gradient spectrum never rises above
    # the FP07 electronics noise floor carries no resolvable thermal signal.
    # Without this gate the variance integral below still returns a small positive
    # chi from residual above-noise scatter, and the figure of merit cannot catch
    # it (fom = obs / model with the model amplitude derived from obs, so
    # fom ~= 1 for the noise floor itself).  Reject such windows as non-detections
    # here, at the source, so they become NaN and drop out of chiMean rather than
    # biasing its low tail and the derived mixing products.  (2026-07-03 review.)
    # Too few bins from first_bin up is said as such, not as a non-detection
    # (the same NaN result the gate would return).
    starved = _first_bin_starves(K, K_AA, first_bin, _MIN_BAND_BINS)
    if starved:
        warnings.warn(starved, stacklevel=2)
        return ChiEpsilonResult(np.nan, kB, np.nan, np.zeros_like(K), np.nan, np.nan)
    if _below_detection(spec_obs, noise_K, K, K_AA, dof=dof, first_bin=first_bin):
        warnings.warn(
            "No thermal signal above the FP07 noise floor; chi is a non-detection",
            stacklevel=2,
        )
        return ChiEpsilonResult(np.nan, kB, np.nan, np.zeros_like(K), np.nan, np.nan)

    valid = _valid_wavenumber_mask(spec_obs, noise_K, K, K_AA, min_points=3, first_bin=first_bin)
    if np.sum(valid) < 3:
        warnings.warn("Too few valid wavenumber points for chi integration", stacklevel=2)
        return ChiEpsilonResult(np.nan, kB, np.nan, np.zeros_like(K), np.nan, np.nan)

    # Integrate the CONTIGUOUS band the selection spans, on the original grid --
    # never the subset grid, which bridges every excluded interval (see
    # _band_slice, issue #180 F03).
    band = _band_slice(valid)
    K_band = K[band]
    K_max = float(K_band[-1])

    # Chi amplitude from the noise-subtracted resolved temperature-gradient
    # variance, with kB fixed from epsilon.  This is the linear-space
    # (method-of-moments) estimator: E[obs] = signal + noise, so subtracting the
    # noise floor and integrating recovers the in-band gradient variance, and the
    # analytic V_total / V_resolved correction (which also undoes the in-band FP07
    # |H|^2 attenuation) scales it to chi = 6 * kappa_T * <(dT/dz)^2>.
    #
    # It replaces the former log-space least-squares grid fit, which is biased low
    # by exactly exp(psi(d/2) - ln(d/2)) (~ -6% at the production dof ~13) because
    # for chi^2-distributed Welch spectra E[ln(obs)] = ln(signal) + (psi(d/2) -
    # ln(d/2)) < ln E[obs] (Jensen).  The variance integral is unbiased in linear
    # space, and it is the SAME estimator Method 2 already uses (_mle_fit_kB /
    # _iterative_fit), so the two methods now share one amplitude form.  A
    # Monte-Carlo bias test (test_chi) pins the ~1% high/moderate-SNR bias; the
    # residual near-detection-floor overshoot is a pre-existing property of the
    # above-2x-noise band selection, not of this estimator.  (issue #104 U3-C1.)
    s = spec_obs[band]
    nv = noise_K[band]

    # Raw in-band variance (NO noise subtraction) — retained only for the figure
    # of merit below (matching the historical fom numerator).  The chi estimate
    # uses the noise-subtracted variance instead; do not conflate the two.
    obs_var = np.trapezoid(s, K_band)
    if obs_var <= 0:
        warnings.warn("Observed variance <= 0; chi is a non-detection", stacklevel=2)
        return ChiEpsilonResult(np.nan, kB, K_max, np.zeros_like(K), np.nan, np.nan)

    # V_resolved spans the SAME measured band [K_min, K_max] that the observed
    # integral covers, mirroring the Method-2 path so the unresolved-variance
    # correction is not double-counted. K_min is bin first_bin or above, so the
    # correction models all of [0, K_min], bin 1 included.
    K_min = float(K_band[0])
    correction = _variance_correction(kB, K_max, speed, tau0, _h2, grad_func, K_min=K_min)
    chi_var = np.trapezoid(np.maximum(s - nv, 0.0), K_band)
    chi = 6.0 * kappa_T * chi_var * (correction if np.isfinite(correction) else 1.0)

    # Compute fitted Batchelor spectrum for output
    spec_batch = grad_func(K, kB, chi)

    # Figure of merit: observed vs attenuated model (Batchelor * H2 + noise) over
    # the same contiguous band as obs_var, so numerator and denominator agree.
    # mad_shape compares the same two spectra bin by bin (see _shape_mad).
    mad_shape = np.nan
    if K_band.size >= 3 and np.isfinite(chi):
        model_band = spec_batch[band] * H2[band] + noise_K[band]
        mod_v = np.trapezoid(model_band, K_band)
        fom = obs_var / mod_v if mod_v > 0 else np.nan
        if np.isfinite(fom):
            mad_shape = _shape_mad(s, model_band)
    else:
        fom = np.nan

    K_max_ratio_val = K_max / kB if kB > 0 else np.nan

    # Batchelor variance-resolved fraction over the SAME [K_min, K_max] band the
    # correction/integral use — feeds the eq (18) chiLnSigma truncation (U4-F1).
    var_resolved_val = _batchelor_resolved_fraction(kB, K_max, grad_func, K_min=K_min)

    return ChiEpsilonResult(
        chi, kB, K_max, spec_batch, fom, K_max_ratio_val, var_resolved_val, mad_shape
    )


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
    first_bin: int = FIRST_BIN,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Core MLE grid search for kB (no chi correction or output spectra).

    The likelihood sums over bins at or above FFT bin *first_bin*
    (:data:`FIRST_BIN`).

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
    fit_mask = _valid_wavenumber_mask(spec_obs, noise_K, K, K_AA, min_points=6, first_bin=first_bin)
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
    dof: float = 0.0,
    first_bin: int = FIRST_BIN,
) -> ChiFitResult:
    """Maximum-likelihood fit for Batchelor wavenumber kB (Ruddick et al. 2000).

    Performs a coarse+fine grid search over kB to minimize the MLE
    negative log-likelihood for chi-squared spectral estimates, over bins at
    or above FFT bin *first_bin* (:data:`FIRST_BIN`), which also starts the
    band of the chi integral, its variance correction, ``fom``, ``mad_shape``
    and ``var_resolved``.

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
    noise_K, H2, tau0, _h2, f_AA, speed, spectrum_model, kappa_T, dof, first_bin
        Same as :func:`_chi_from_epsilon`.

    Returns
    -------
    ChiFitResult
        Named tuple: (kB, chi, epsilon, K_max, spec_batch, fom, K_max_ratio,
        var_resolved, mad_shape).
    """
    grad_func, _q = _spectrum_func(spectrum_model)
    grad_func = functools.partial(grad_func, kappa_T=kappa_T)

    # Signal-presence gate (mirrors Method 1): reject a noise-only window before
    # the MLE fits a Batchelor curve to noise and returns a spurious finite
    # chi/kB.  (2026-07-03 adversarial review of the Method-1 gate.)
    starved = _first_bin_starves(K, f_AA / speed, first_bin, 6)
    if starved:
        warnings.warn(starved, stacklevel=2)
        return ChiFitResult(np.nan, np.nan, np.nan, np.nan, np.zeros_like(K), np.nan, np.nan)
    if _below_detection(spec_obs, noise_K, K, f_AA / speed, dof=dof, first_bin=first_bin):
        warnings.warn(
            "No thermal signal above the FP07 noise floor; chi is a non-detection",
            stacklevel=2,
        )
        return ChiFitResult(np.nan, np.nan, np.nan, np.nan, np.zeros_like(K), np.nan, np.nan)

    kB_best, fit_mask, K_fit = _mle_find_kB(
        spec_obs,
        K,
        chi_obs,
        noise_K,
        H2,
        f_AA,
        speed,
        grad_func,
        first_bin=first_bin,
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
    # K_fit_low is bin first_bin or above.
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
    # The kB LIKELIHOOD above is a sum over selected bins and is unaffected by
    # mask gaps, but every INTEGRAL below runs on the contiguous band on the
    # original grid: trapezoid over the subset grid bridges each excluded
    # interval (see _band_slice, issue #180 F03). The band's endpoints are the
    # same K_fit_low / K_max_fit the correction already assumes.
    band = _band_slice(fit_mask)
    K_band = K[band]
    if np.isfinite(correction):
        obs_var = np.trapezoid(np.maximum(spec_obs[band] - noise_K[band], 0), K_band)
        chi = 6 * kappa_T * obs_var * correction
    else:
        chi = chi_obs

    # Fitted spectrum for output
    spec_batch = grad_func(K, kB_best, chi)

    # Figure of merit: observed vs attenuated model (Batchelor * H2 + noise);
    # mad_shape compares the same two spectra bin by bin (see _shape_mad).
    mad_shape = np.nan
    if np.isfinite(chi) and K_band.size >= 3:
        model_band = spec_batch[band] * H2[band] + noise_K[band]
        mod_v = np.trapezoid(model_band, K_band)
        obs_v = np.trapezoid(spec_obs[band], K_band)
        fom = obs_v / mod_v if mod_v > 0 else np.nan
        if np.isfinite(fom):
            mad_shape = _shape_mad(spec_obs[band], model_band)
    else:
        fom = np.nan

    K_max_ratio_val = K_max_fit / kB_best if kB_best > 0 else np.nan
    var_resolved_val = _batchelor_resolved_fraction(
        kB_best, K_max_fit, grad_func, K_min=K_fit_low
    )

    return ChiFitResult(
        kB_best,
        chi,
        epsilon,
        K_max_fit,
        spec_batch,
        fom,
        K_max_ratio_val,
        var_resolved_val,
        mad_shape,
    )


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
    dof: float = 0.0,
    first_bin: int = FIRST_BIN,
) -> ChiFitResult:
    """Iterative MLE fitting (Peterson & Fer 2014, Method 2).

    Three iterations refining the LOWER integration limit
    ``k_l = max(K[first_bin], 3*k_star)`` (with ``k_star = 0.04*kB*sqrt(kappa/nu)``,
    the inertial-convective to viscous-convective transition; source below)
    and applying a model-based correction for the variance
    outside [``k_l``, ``k_u``] and for in-band FP07 attenuation (see
    :func:`_variance_correction`).  The UPPER limit ``k_u`` is fixed at
    the last valid wavenumber (noise/anti-aliasing criterion) and is not
    refined between iterations.

    Note: ``k_l`` is only the lower edge of the *integration* band, not a hard
    clip — :func:`_variance_correction` adds back the modeled variance below
    ``k_l``, so the recovered chi is insensitive to the exact 0.04 value when
    the model matches the observed spectrum.  A 2026-07-03 review confirmed
    numerically that ``chi`` is invariant (to <0.001%) across coefficients
    0.02-0.16 for a model-consistent spectrum; only a model-vs-observed shape
    mismatch in the low-k sliver below ``k_l`` makes it matter, a second-order
    effect.  So this is a documented-provenance note, not a correctness risk.

    Source of the 0.04: ``kB*sqrt(kappa/nu) = kB*Pr**-0.5`` is the Kolmogorov
    wavenumber ``k_K``, so ``k_star = 0.04*k_K``.  Gibson et al. (1970) found
    k*/k_K = 0.03-0.04 (0.01-0.02 of kB) in the laboratory, and Steinbuck et
    al. (2009) proposed k* = 0.04 k_K (~0.02 kB) as the lowest wavenumber of a
    Batchelor fit, both as quoted by Sanchez et al. (2011, section 2b, p. 2159;
    doi:10.1175/JPO-D-11-025.1), who also start their chi integral there (their
    eq. 12).  The factor 3 in ``k_l`` is not in that account and is unsourced
    here; Steinbuck et al. was not read directly.

    No bin below FFT bin *first_bin* (:data:`FIRST_BIN`) enters the detection
    gate, the MLE, the initial or refined chi integral, ``fom``, ``mad_shape``
    or ``var_resolved``.

    Parameters
    ----------
    spec_obs : ndarray
        Observed temperature gradient spectrum [(K/m)²/cpm].
    K : ndarray
        Wavenumber vector [cpm].
    nu : float
        Kinematic viscosity [m²/s].
    noise_K, H2, tau0, _h2, f_AA, speed, spectrum_model, kappa_T, dof, first_bin
        Same as :func:`_chi_from_epsilon`.

    Returns
    -------
    ChiFitResult
        Named tuple: (kB, chi, epsilon, K_max, spec_batch, fom, K_max_ratio,
        var_resolved, mad_shape).
    """
    grad_func, _q = _spectrum_func(spectrum_model)
    grad_func = functools.partial(grad_func, kappa_T=kappa_T)

    # Initial integration limits
    K_AA = f_AA / speed

    # Signal-presence gate (mirrors Method 1's _chi_from_epsilon): a window with
    # fewer than 3 bins above 2x the noise floor is a non-detection. Without it
    # the MLE below fits a Batchelor curve to noise and, together with the weak
    # band_has_signal check (spec_obs > 1x noise), emits a spurious finite
    # chi/kB. Reject at the source -> NaN, dropping it from chiMean rather than
    # biasing the low tail.  (2026-07-03 adversarial review of the Method-1 gate.)
    starved = _first_bin_starves(K, K_AA, first_bin, 6)
    if starved:
        warnings.warn(starved, stacklevel=2)
        return ChiFitResult(np.nan, np.nan, np.nan, np.nan, np.zeros_like(K), np.nan, np.nan)
    if _below_detection(spec_obs, noise_K, K, K_AA, dof=dof, first_bin=first_bin):
        warnings.warn(
            "No thermal signal above the FP07 noise floor; chi is a non-detection",
            stacklevel=2,
        )
        return ChiFitResult(np.nan, np.nan, np.nan, np.nan, np.zeros_like(K), np.nan, np.nan)

    usable = _usable_bins(K, first_bin)
    valid = _valid_wavenumber_mask(spec_obs, noise_K, K, K_AA, min_points=6, first_bin=first_bin)

    valid_idx = np.where(valid)[0]
    if len(valid_idx) < 6:
        warnings.warn("Too few valid points for iterative fit", stacklevel=2)
        return ChiFitResult(np.nan, np.nan, np.nan, np.nan, np.zeros_like(K), np.nan, np.nan)

    k_u = K[valid_idx[-1]]
    # The refined band's floor: bin first_bin's wavenumber (K[1] before
    # FIRST_BIN). `valid` is a subset of `usable`, so it is non-empty here.
    k_first = K[usable][0]

    # Initial chi estimate, integrated over the CONTIGUOUS band the selection
    # spans (see _band_slice, #180 F03): the subset grid bridges every excluded
    # interval. mask_refined / mask_final below are range conditions on a
    # monotone K, hence already contiguous, and are left alone.
    mask_init = valid & (spec_obs > noise_K)
    if np.sum(mask_init) < 3:
        mask_init = valid
    band_init = _band_slice(mask_init)
    chi_obs = (
        6
        * kappa_T
        * np.trapezoid(
            np.maximum(spec_obs[band_init] - noise_K[band_init], 0), K[band_init]
        )
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
            first_bin=first_bin,
        )

        if not np.isfinite(kB_best) or kB_best < 1:
            break

        # Check convergence
        if iteration > 0 and abs(kB_best - kB_prev) / kB_prev < 0.01:
            break
        kB_prev = kB_best

        # Refine integration limits
        k_star = 0.04 * kB_best * np.sqrt(kappa_T / nu)
        k_l_new = max(k_first, 3 * k_star)
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
    var_resolved_val = np.nan
    if np.isfinite(kB_best) and kB_best >= 1:
        k_star = 0.04 * kB_best * np.sqrt(kappa_T / nu)
        k_l_final = max(k_first, 3 * k_star)
        # Batchelor V_f over the final [k_l_final, k_u] band (eq (18); U4-F1).
        var_resolved_val = _batchelor_resolved_fraction(
            kB_best, k_u, grad_func, K_min=k_l_final
        )
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
    # mad_shape compares the same two spectra bin by bin (see _shape_mad).
    mad_shape = np.nan
    if np.isfinite(kB_best) and np.isfinite(chi):
        valid_fom = (spec_obs > 2 * noise_K) & usable & (k_u >= K)
        band_fom = _band_slice(valid_fom)
        K_fom = K[band_fom]
        if K_fom.size >= 3:
            obs_v = np.trapezoid(spec_obs[band_fom], K_fom)
            model_fom = spec_batch[band_fom] * H2[band_fom] + noise_K[band_fom]
            mod_v = np.trapezoid(model_fom, K_fom)
            fom = obs_v / mod_v if mod_v > 0 else np.nan
            if np.isfinite(fom):
                mad_shape = _shape_mad(spec_obs[band_fom], model_fom)
        else:
            fom = np.nan
    else:
        fom = np.nan

    K_max_ratio_val = k_u / kB_best if np.isfinite(kB_best) and kB_best > 0 else np.nan

    return ChiFitResult(
        kB_best, chi, epsilon, k_u, spec_batch, fom, K_max_ratio_val, var_resolved_val, mad_shape
    )


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
