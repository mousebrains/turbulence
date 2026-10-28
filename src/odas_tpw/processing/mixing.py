# Jun-2026, Claude and Pat Welch, pat@mousebrains.com
"""Derived mixing quantities from co-located epsilon and chi estimates.

Microstructure profilers measure two dissipation rates — epsilon (TKE,
from shear probes) and chi (thermal variance, from FP07 thermistors).
The quantities oceanographers ultimately want are usually the derived
ones:

- ``K_T``    — Osborn-Cox eddy diffusivity of heat,
  ``K_T = chi / (2 * (dT/dz)^2)``  [m^2/s]
- ``Gamma``  — dimensionless mixing coefficient (flux coefficient),
  ``Gamma = N^2 * chi / (2 * epsilon * (dT/dz)^2)``  (Oakey 1982)
- ``K_rho``  — Osborn diapycnal diffusivity with a constant mixing
  coefficient, ``K_rho = Gamma_0 * epsilon / N^2`` (Osborn 1980,
  canonical ``Gamma_0 = 0.2``)

Note the algebraic identity ``Gamma * epsilon / N^2 == K_T``: when the
mixing coefficient is *measured* via Oakey's method, the Osborn
diffusivity computed with it reproduces the Osborn-Cox diffusivity by
construction.  ``K_rho`` here therefore uses the canonical constant
``Gamma_0`` so that comparing ``K_rho`` with ``K_T`` (equivalently,
``Gamma`` with ``Gamma_0``) is a physically meaningful consistency
check, not a tautology.

Both ``Gamma`` and the diffusivities divide by background gradients that
can vanish in weakly stratified water, so estimates are masked where
``N^2`` or ``|dT/dz|`` fall below configurable floors.

References
----------
Osborn, T.R. and C.S. Cox, 1972: Oceanic fine structure.
    Geophys. Fluid Dyn., 3, 321-345.
    https://doi.org/10.1080/03091927208236085
Osborn, T.R., 1980: Estimates of the local rate of vertical diffusion
    from dissipation measurements. J. Phys. Oceanogr., 10, 83-89.
    https://doi.org/10.1175/1520-0485(1980)010<0083:EOTLRO>2.0.CO;2
Oakey, N.S., 1982: Determination of the rate of dissipation of
    turbulent energy from simultaneous temperature and velocity shear
    microstructure measurements. J. Phys. Oceanogr., 12, 256-271.
    https://doi.org/10.1175/1520-0485(1982)012<0256:DOTROD>2.0.CO;2
Gregg, M.C., E.A. D'Asaro, J.J. Riley, and E. Kunze, 2018: Mixing
    efficiency in the ocean. Annu. Rev. Mar. Sci., 10, 443-473.
    https://doi.org/10.1146/annurev-marine-121916-063643
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import gsw
import numpy as np
import numpy.typing as npt

# Stratification floor [s^-2]. Below ~1e-9 s^-2 (N ~ 0.02 cph) the Osborn
# estimate diverges (division-by-zero guard). A 1e-9 floor prevents the divide
# but not unphysical inflation: a window whose N^2 sits just above it still
# yields an absurd K_rho = Gamma_0 * epsilon / N^2. Rather than raise the floor
# (which would NaN genuinely weakly-stratified bins without benchmark support),
# K_rho is bounded from above by DEFAULT_K_RHO_MAX below (audit #30).
DEFAULT_N2_MIN = 1e-9

# Upper sanity bound on K_rho [m^2/s] (audit #30). The N2_min floor stops the
# divide-by-zero but not magnitude inflation: K_rho = Gamma_0 * epsilon / N2 grows
# without bound as N^2 shrinks toward the floor, so the artifact produces values
# of tens to thousands of m^2/s. Windows above this bound are masked (set to NaN).
# 10 m^2/s sits above even the most energetic real diapycnal mixing — overflows
# and hydraulic jumps reach ~0.1-1 m^2/s, and energetic near-surface mixing on
# ARCTERX VMP casts was observed up to a few m^2/s — so genuine signal is kept
# and only physically implausible magnitudes are removed (the unbounded
# near-floor-N^2 artifact, or extreme contaminated near-surface windows where
# epsilon is itself spurious). (A 1 m^2/s bound was tried first but clipped real
# energetic near-surface mixing.) Configurable via K_rho_max.
DEFAULT_K_RHO_MAX = 10.0

# Upper sanity bound on K_T [m^2/s]. K_T = chi/(2*(dT/dz)^2) is a turbulent
# thermal diffusivity on the same physical scale as K_rho (by the module
# identity K_T = 5*Gamma*K_rho, so K_T ~ K_rho when Gamma ~ 0.2). It has no N^2
# in it, so it does NOT blow up on the near-floor-N^2 artifact — but a weak
# dT/dz combined with a contaminated chi can still push it past any real ocean
# value, so it gets its OWN ceiling (not K_rho's mask). 10 m^2/s sits above the
# most energetic real thermal mixing. Configurable via K_T_max.
DEFAULT_K_T_MAX = 10.0

# Upper sanity bound on the measured mixing coefficient Gamma [1]. Gamma is
# dimensionless with a canonical value ~0.2 and rarely exceeds ~1-2 even in
# energetic mixing (flux Richardson number Rf < ~0.5 => Gamma = Rf/(1-Rf) < ~1);
# values of tens to hundreds arise only when a near-zero shear epsilon lands in
# the denominator (Gamma = N2*chi/(2*epsilon*(dT/dz)^2)). 5.0 keeps all
# physically plausible mixing efficiency while removing those gross artifacts.
# Configurable via Gamma_max.
DEFAULT_GAMMA_MAX = 5.0

# Background temperature-gradient floor [K/m].  chi/(dT/dz)^2 diverges
# as the mean gradient vanishes (well-mixed layers); 1e-4 K/m over a
# ~5 m window corresponds to ~0.5 mK of signal, near FP07 resolution.
DEFAULT_DTDZ_MIN = 1e-4

# Canonical Osborn (1980) mixing coefficient.
GAMMA_OSBORN = 0.2

# Minimum pressure span [dbar] within a window for gradient estimates;
# below this (stalled instrument) vertical gradients are unconstrained.
DEFAULT_MIN_DP = 0.2

# Default background vertical window [dbar] for profile/CTD stratification.
DEFAULT_STRAT_WINDOW = 2.0

# Hard ceiling [s] on the AUTOMATIC epsilon<->chi pairing tolerance. Epsilon and
# chi windows sit seconds apart by construction (a 4096-sample dissipation
# window at 512 Hz is 8 s), so a pairing wider than this is not "the nearest
# estimate" but a different piece of water. Only the max_dt=None default is
# capped -- an explicit max_dt is the caller's declared choice (#180 F05).
MAX_PAIR_DT = 120.0

# How a window's parcels are reordered before its gradients are taken (see
# _stable_window). "density" is the adiabatic leveling the module did until
# 2026-10; "temperature" orders by conservative temperature alone (the
# sorted-scalar background of Winters & D'Asaro 1996, which is what chi works
# against); "none" keeps the measured order.
SORT_MODES = ("density", "temperature", "none")

# The default is "temperature" (since 2026-10; "density" before). A temperature
# sensor is a steadier observation than a salinity derived from a conductivity
# cell, and the salinity's noise scrambles a density order. On stored VMP-250
# products with unpumped JAC CTDs, the density-sorted chi-window dT/dz was
# 0.70-0.97 of the temperature-sorted value across five campaigns (K_T 1.06-2x
# high; 2-s windows worst) and N2 1.12-1.48x the unsorted value; on an unpumped
# SBE4 the ratio was 0.28-0.46.
DEFAULT_SORT = "temperature"

# Sort-agreement diagnostic (sort_kappa). Whatever the product's sort, kappa is
# the median over windows of dTdz(density order) / dTdz(temperature order), both
# from the per-sample conservative temperature. A
# temperature-monotonic order gives the steepest least-squares slope of any
# ordering, so kappa < 1 means salinity is reordering the parcels: salinity noise
# (unpumped or mismatched T/C sensors) or density-compensated structure. Below
# KAPPA_WARN a density-sorted background gradient -- and K_T, which goes as its
# inverse square -- is off by more than 1/0.8^2 = 1.56x; under the other sorts
# dT/dz is spared, but N2 carries the same salinity. Windows whose temperature-sorted
# |dT/dz| is under KAPPA_MIN_DTDZ [K/m] are too weakly stratified to vote, and
# fewer than KAPPA_MIN_WINDOWS voting windows leave kappa undefined (NaN).
# On stored products from the (unpumped) JAC CTDs of VMP-250s (2-dbar windows, 2026-10) the
# per-campaign median kappa ran 0.67-0.88; on an unpumped SBE4 0.28-0.46.
KAPPA_WARN = 0.8
KAPPA_MIN_DTDZ = 1e-3
KAPPA_MIN_WINDOWS = 5
# A window votes only where its MEASURED-order gradient is resolved: the
# temperature-sorted slope cannot be the gate, because sorting noise inflates it
# (a 4e-4 K/m mixed-layer window sorts to ~5e-3), and mixed layers would vote.
# A salinity failure can be confined to part of a cast, so kappa is also taken
# per KAPPA_BAND_DBAR band (with at least KAPPA_BAND_MIN_WINDOWS voting windows).
# The worst of several noisy band medians is biased low, so a band warns only
# below KAPPA_BAND_WARN. Calibrated on stored products (2026-10, 30 casts each,
# chi windows): warnings on 0-3 of 30 casts where stored K_T is 1.06-1.18x high
# (ASTRAL, CASPER-West, ARCTERX-2023 Wake; 4-s windows) and 13-20 of 30 where
# it is ~2x high (ARCTERX-2025 Interior, Taiwan17; 2-s windows); the band rule
# alone at 0.8 warned on 11-29 of 30 everywhere.
KAPPA_BAND_DBAR = 50.0
KAPPA_BAND_MIN_WINDOWS = 10
KAPPA_BAND_WARN = 0.6

# Temperature-sort direction (see _trend_direction): a quadratic fit over a span
# TSORT_TREND_SPAN times the window, centered on it. Where its gradient at the
# window center is below TSORT_MIN_TREND [K/m] -- the K_T well-mixed floor --
# or its extremum falls inside the window, the measured order is kept.
TSORT_TREND_SPAN = 5.0
TSORT_MIN_TREND = DEFAULT_DTDZ_MIN
# ...and where the fitted temperature change across the window is under
# TSORT_SNR times the scatter about the fit (MAD-based sigma over the span):
# there a full sort would order noise and inflate the gradient (on ASTRAL's
# weak-gradient windows, median |dT/dz| 4.4e-4 K/m unsorted, ~5e-3 sorted).
TSORT_SNR = 2.0


class StratificationResult(NamedTuple):
    """Per-window background stratification."""

    N2: np.ndarray  # buoyancy frequency squared [s^-2]
    dTdz: np.ndarray  # background conservative-temperature (θ) gradient vs depth [K/m]
    # The per-sample CT gradient with the parcels in temperature order and in
    # density order, for the sort_kappa diagnostic. Computed under every sort
    # when the salinity is per-sample; NaN otherwise, and where the window is
    # too weakly stratified to vote.
    dTdz_T: np.ndarray
    dTdz_rho: np.ndarray


class ProfileStratification(NamedTuple):
    """Per-target-pressure background stratification for one cast."""

    target_P: np.ndarray  # window-center pressures [dbar]
    N2: np.ndarray  # [s^-2]
    dTdz: np.ndarray  # [K/m vs depth, positive down]
    dTdz_T: np.ndarray  # temperature-order dT/dz for sort_kappa (NaN when not voting)
    dTdz_rho: np.ndarray  # density-order dT/dz for sort_kappa (NaN when not voting)


class MixingResult(NamedTuple):
    """Per-window derived mixing quantities."""

    K_T: np.ndarray  # Osborn-Cox heat diffusivity [m^2/s]
    Gamma: np.ndarray  # measured mixing coefficient [1]
    K_rho: np.ndarray  # Osborn diffusivity with Gamma_0 [m^2/s]


def window_stratification(
    win_times: npt.ArrayLike,
    win_half_width: float,
    t: npt.ArrayLike,
    P: npt.ArrayLike,
    T: npt.ArrayLike,
    S: float | npt.ArrayLike | None = None,
    lat: float = 0.0,
    lon: float = 0.0,
    min_samples: int = 4,
    min_dp: float = DEFAULT_MIN_DP,
) -> StratificationResult:
    """Background N² and dT/dz for each dissipation window.

    For each window (center time ± half width):

    - ``dTdz`` is the slope of a least-squares fit of *conservative*
      temperature (θ, TEOS-10) against depth (positive down), i.e. the
      *background* gradient at the window scale — the same quantity whose
      square normalizes chi in the Osborn-Cox model. Conservative (not
      in-situ) temperature is used so the adiabatic lapse rate, which
      produces no thermal variance, does not enter the gradient.
    - ``N2`` is gsw.Nsquared (TEOS-10) evaluated between the mean
      (SA, CT, p) of the shallow and deep halves of the window, which
      averages out microstructure fluctuations.

    Parameters
    ----------
    win_times : array_like, shape (n_win,)
        Window center times [s] (same time base as ``t``).
    win_half_width : float
        Half the window duration [s].
    t, P, T : array_like, shape (n,)
        Sample times [s], pressure [dbar], in-situ temperature [degC].
    S : float, array_like, or None
        Practical salinity: scalar, per-sample array, or None.
        None assumes 35 PSU — N² then reflects temperature
        stratification only, which can be badly wrong where salinity
        stratification matters (the integrating callers record which case
        applied in a free-text ``sal_note`` on the output variables).
    lat, lon : float
        Position for TEOS-10 conversions (defaults 0/0 introduce only
        small absolute-salinity-anomaly and gravity errors).
    min_samples : int
        Minimum samples in a window; fewer gives NaN.
    min_dp : float
        Minimum pressure span [dbar] in a window; less gives NaN
        (vertical gradients unconstrained, e.g. stalled instrument).

    Returns
    -------
    StratificationResult
        ``N2`` [s^-2] and ``dTdz`` [K/m vs depth, positive down] per
        window; NaN where the window has too few samples or span.
        ``N2 > 0`` is stable stratification.
    """
    win_times = np.asarray(win_times, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    n_win = len(win_times)

    if S is None or np.ndim(S) == 0:
        S_const = 35.0 if S is None else float(S)  # type: ignore[arg-type]
        S_arr = None
    else:
        S_const = np.nan
        S_arr = np.asarray(S, dtype=np.float64)

    N2 = np.full(n_win, np.nan)
    dTdz = np.full(n_win, np.nan)

    # Depth (positive down) from pressure; lat=0 underestimates gravity
    # by <0.3%, negligible against the gradient-estimation noise.
    depth = -gsw.z_from_p(P, lat)

    for j, tau in enumerate(win_times):
        sel = np.abs(t - tau) <= win_half_width
        if not np.any(sel):
            continue
        Pw = P[sel]
        Tw = T[sel]
        zw = depth[sel]
        Sw = S_arr[sel] if S_arr is not None else None

        good = np.isfinite(Pw) & np.isfinite(Tw) & np.isfinite(zw)
        if Sw is not None:
            good &= np.isfinite(Sw)
        if np.sum(good) < min_samples:
            continue
        Pw, Tw, zw = Pw[good], Tw[good], zw[good]
        if Sw is not None:
            Sw = Sw[good]

        if np.ptp(Pw) < min_dp:
            continue

        # Background conservative-temperature gradient vs depth (least
        # squares). Osborn-Cox variance production is against the
        # adiabatically-referenced (conservative) temperature gradient; fitting
        # in-situ T folds in the adiabatic lapse rate (~2.4e-4 K/m in warm
        # water), which biases K_T/Gamma and defeats the well-mixed mask
        # (audit r1-1). chi itself is a microstructure quantity and is
        # unaffected — in-situ and conservative temperature *fluctuations*
        # coincide at those scales.
        Sw_vals = Sw if Sw is not None else np.full(len(Pw), S_const)
        SA_w = gsw.SA_from_SP(Sw_vals, Pw, lon, lat)
        CT_w = gsw.CT_from_t(SA_w, Tw, Pw)
        dTdz[j] = np.polyfit(zw, CT_w, 1)[0]

        # N² between the shallow- and deep-half means (TEOS-10)
        order = np.argsort(Pw)
        half = len(order) // 2
        sh, dp = order[:half], order[half:]
        p_pair = np.array([np.mean(Pw[sh]), np.mean(Pw[dp])])
        t_pair = np.array([np.mean(Tw[sh]), np.mean(Tw[dp])])
        if Sw is not None:
            s_pair = np.array([np.mean(Sw[sh]), np.mean(Sw[dp])])
        else:
            s_pair = np.array([S_const, S_const])
        if p_pair[1] - p_pair[0] < min_dp / 2:
            continue
        SA = gsw.SA_from_SP(s_pair, p_pair, lon, lat)
        CT = gsw.CT_from_t(SA, t_pair, p_pair)
        n2_val, _ = gsw.Nsquared(SA, CT, p_pair, lat)
        N2[j] = float(n2_val[0])

    return StratificationResult(
        N2=N2, dTdz=dTdz, dTdz_T=np.full(n_win, np.nan), dTdz_rho=np.full(n_win, np.nan)
    )


def sorted_stratification(
    win_times: npt.ArrayLike,
    win_half_width: float,
    t: npt.ArrayLike,
    P: npt.ArrayLike,
    T: npt.ArrayLike,
    S: float | npt.ArrayLike | None = None,
    lat: float = 0.0,
    lon: float = 0.0,
    min_samples: int = 4,
    min_dp: float = DEFAULT_MIN_DP,
    sort: str = DEFAULT_SORT,
) -> StratificationResult:
    """Background N² and dT/dz per window from a sorted profile.

    Like :func:`window_stratification`, but within each window the parcels are
    reordered before differencing, so the gradients reflect the *background*
    stratification rather than the instantaneous profile with its overturns:

    - The window's (SA, CT, p) are formed via TEOS-10 and put in the order
      *sort* picks (:data:`SORT_MODES`; see :func:`_stable_window`): by
      conservative temperature alone (``"temperature"``, the default
      :data:`DEFAULT_SORT`), by potential density (``"density"``, ``gsw.sigma0``,
      the i-th densest parcel at the i-th deepest pressure: Thorpe-sorted to
      static stability), or as measured (``"none"``).
    - ``N2`` is ``gsw.Nsquared`` between the shallow- and deep-half means of
      the reordered profile. Under the density sort it is ≥ 0 for ordinary
      stratification, but can be slightly negative in strongly
      density-compensated water: ``gsw.Nsquared`` averages SA and CT separately
      and re-evaluates the nonlinear equation of state, which (via cabbeling) is
      not guaranteed monotone under the sigma0 sort. Under the other sorts
      nothing orders the density, so N² is negative wherever the reordered
      half-means are unstable (salinity noise, density-compensated structure,
      an overturn left in measured order). Such windows are treated as
      effectively unstratified by the downstream Osborn scaling.
    - ``dTdz`` is the least-squares slope of the reordered *conservative*
      temperature (θ) against depth (positive down); see
      :func:`window_stratification` for why conservative, not in-situ.

    A salinity whose noise is large against its gradient scrambles the density
    order; ``"temperature"`` is immune to that in dT/dz, but its N² still
    carries the salinity, and it also removes real density-compensated
    inversions.

    Parameters match :func:`window_stratification`, and NaN is returned for
    windows with fewer than *min_samples* valid samples or a pressure span below
    *min_dp*. With a per-sample salinity, the result's ``dTdz_rho`` and
    ``dTdz_T`` carry the density- and temperature-order gradients for
    :func:`sort_kappa`, whatever *sort* is.
    """
    _check_sort(sort)
    win_times = np.asarray(win_times, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    n_win = len(win_times)

    if S is None or np.ndim(S) == 0:
        S_const = 35.0 if S is None else float(S)  # type: ignore[arg-type]
        S_arr = None
    else:
        S_const = np.nan
        S_arr = np.asarray(S, dtype=np.float64)

    N2 = np.full(n_win, np.nan)
    dTdz = np.full(n_win, np.nan)
    dTdz_T = np.full(n_win, np.nan)
    dTdz_rho = np.full(n_win, np.nan)

    depth = -gsw.z_from_p(P, lat)
    # Conservative temperature over the whole series, for the temperature-sort
    # direction only (_trend_direction looks beyond the window). Each window's
    # own SA/CT are still formed from its samples below, so the density path's
    # arithmetic is unchanged.
    # kappa needs a salinity that varies sample to sample: with a constant one
    # the density and temperature orders coincide.
    want_kappa = S_arr is not None
    CT_all = None
    if sort == "temperature" or want_kappa:
        S_all = S_arr if S_arr is not None else np.full(len(P), S_const)
        CT_all = gsw.CT_from_t(gsw.SA_from_SP(S_all, P, lon, lat), T, P)
    ext_half = TSORT_TREND_SPAN * win_half_width

    for j, tau in enumerate(win_times):
        sel = np.abs(t - tau) <= win_half_width
        if not np.any(sel):
            continue
        Pw = P[sel]
        Tw = T[sel]
        zw = depth[sel]
        Sw = S_arr[sel] if S_arr is not None else None

        good = np.isfinite(Pw) & np.isfinite(Tw) & np.isfinite(zw)
        if Sw is not None:
            good &= np.isfinite(Sw)
        if np.sum(good) < min_samples:
            continue
        Pw, Tw, zw = Pw[good], Tw[good], zw[good]
        Sw = Sw[good] if Sw is not None else None
        if np.ptp(Pw) < min_dp:
            continue

        direction = 0
        if CT_all is not None:
            ext = np.abs(t - tau) <= ext_half
            direction = _trend_direction(depth[ext], CT_all[ext], float(zw.min()), float(zw.max()))

        Sw_vals = Sw if Sw is not None else np.full(len(Pw), S_const)
        SA = gsw.SA_from_SP(Sw_vals, Pw, lon, lat)
        CT = gsw.CT_from_t(SA, Tw, Pw)
        sigma = gsw.sigma0(SA, CT)
        N2[j], dTdz[j], dTdz_rho[j], dTdz_T[j] = _stable_window(
            Pw, zw, Tw, SA, CT, sigma, lat, min_dp, sort=sort, direction=direction,
            want_kappa=want_kappa,
        )

    return StratificationResult(N2=N2, dTdz=dTdz, dTdz_T=dTdz_T, dTdz_rho=dTdz_rho)


def _check_sort(sort: str, allowed: tuple[str, ...] = SORT_MODES) -> None:
    """Raise ValueError unless *sort* is one of *allowed*."""
    if sort not in allowed:
        raise ValueError(f"sort={sort!r}: expected one of {list(allowed)}")


def _slope(z: np.ndarray, x: np.ndarray) -> float:
    """Least-squares slope of *x* against *z*."""
    return float(np.polyfit(z, x, 1)[0])


def _trend_direction(z_ext: np.ndarray, ct_ext: np.ndarray, z_lo: float, z_hi: float) -> int:
    """Which way a temperature sort runs across the window ``[z_lo, z_hi]``.

    +1 where conservative temperature rises with depth, -1 where it falls, and 0
    -- keep the measured order -- where that is not resolved. Decided from a
    quadratic fit over a span :data:`TSORT_TREND_SPAN` times the window
    (*z_ext*, *ct_ext*), not from the window itself, for two reasons:

    - an overturn larger than about half the window reverses the window's own
      trend, and sorting along it would keep the overturn unstable;
    - at a temperature extremum there is no monotonic background, and any sort
      would invent a gradient of the window's temperature range. The vertex of
      the quadratic marks the extremum: inside the window, the result is 0.

    Also 0 where the fitted gradient at the window center is below
    :data:`TSORT_MIN_TREND`, where the fitted change across the window is below
    :data:`TSORT_SNR` times the scatter about the fit (a sort would order noise),
    or where there are fewer than six finite samples. 0 is always safe: the
    measured order is immune to salinity and invents nothing.
    """
    ok = np.isfinite(z_ext) & np.isfinite(ct_ext)
    if int(np.count_nonzero(ok)) < 6:
        return 0
    zc = 0.5 * (z_lo + z_hi)
    zz = z_ext[ok] - zc
    if np.ptp(zz) <= 0:
        return 0
    coef = np.polyfit(zz, ct_ext[ok], 2)
    c2, c1 = coef[0], coef[1]
    if c2 != 0.0:
        vertex = -c1 / (2.0 * c2)
        if z_lo - zc <= vertex <= z_hi - zc:
            return 0
    if not abs(c1) >= TSORT_MIN_TREND:
        return 0
    resid = ct_ext[ok] - np.polyval(coef, zz)
    sigma = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
    if abs(c1) * (z_hi - z_lo) < TSORT_SNR * sigma:
        return 0
    return 1 if c1 > 0 else -1


def _temperature_order(ct_key: np.ndarray, direction: int, pos_order: np.ndarray) -> np.ndarray:
    """Parcel order for a temperature sort of one window.

    The k-th shallowest position gets the k-th coldest parcel for *direction*
    +1, the k-th warmest for -1, and the measured order (*pos_order*) for 0
    (see :func:`_trend_direction`).
    """
    if direction == 0:
        return pos_order
    ascending = np.argsort(ct_key, kind="stable")
    return ascending if direction > 0 else ascending[::-1]


def _stable_window(
    Pw, zw, Tw, SAw, CTw, sigmaw, lat, min_dp, sort=DEFAULT_SORT, direction=0, want_kappa=True
):
    """N², dT/dz, and the two sort_kappa gradients for one window's samples.

    Reconstructs a reference profile in which the k-th shallowest position holds
    the k-th parcel of the chosen order:

    - ``"density"``: least dense first (Thorpe-sorted to static stability);
    - ``"temperature"``: by conservative temperature, the way *direction* says
      (:func:`_temperature_order`; 0 keeps the measured order);
    - ``"none"``: the measured order.

    Returns ``(N2, dTdz, dTdz_rho, dTdz_T)``: N² from ``gsw.Nsquared`` between
    the shallow/deep-half means of the reference profile's (SA, CT) (NaN if the
    half-mean pressure separation is below ``min_dp/2``), dT/dz as the
    least-squares slope of its *conservative* temperature (θ) vs depth, and the
    two sort_kappa gradients.

    The temperature order is keyed on CT formed with the window's median SA, so
    salinity noise cannot reorder it (dCT/dSA is about -0.01 to -0.03 K per
    g/kg at fixed in-situ temperature, so a 0.5 g/kg spike moves a parcel's CT
    by 5-15 mK). Under ``"temperature"`` that same CT gives dT/dz. With
    *want_kappa*, the last two values are the slopes of the per-sample CT in
    density order and in temperature order, whatever *sort* is, so
    :func:`sort_kappa` compares the two orders on the same values and gives the
    same kappa under every sort. They are NaN otherwise, and also where the
    measured-order slope is under :data:`KAPPA_MIN_DTDZ` (such a window does not
    vote). Inputs must be finite with length ≥ 2.
    """
    _check_sort(sort)
    pos_order = np.argsort(Pw, kind="stable")
    p_stable = Pw[pos_order]
    z_stable = zw[pos_order]
    den_order = np.argsort(sigmaw, kind="stable")
    ct_key = gsw.CT_from_t(np.full(len(Pw), np.median(SAw)), Tw, Pw)
    t_order = _temperature_order(ct_key, direction, pos_order)
    dTdz_rho = dTdz_T = np.nan
    if want_kappa and abs(_slope(z_stable, CTw[pos_order])) >= KAPPA_MIN_DTDZ:
        dTdz_rho = _slope(z_stable, CTw[den_order])
        dTdz_T = _slope(z_stable, CTw[t_order])
    # Fit the conservative-temperature gradient (not in-situ T): the adiabatic
    # lapse rate is a reversible gradient that produces no thermal variance, so
    # the Osborn-Cox K_T/Gamma denominator must use dCT/dz (audit r1-1).
    if sort == "density":
        SA_stable = SAw[den_order]
        CT_stable = CTw[den_order]
        dTdz = _slope(z_stable, CT_stable)
    elif sort == "temperature":
        SA_stable = SAw[t_order]
        CT_stable = CTw[t_order]
        dTdz = _slope(z_stable, ct_key[t_order])
    else:  # "none"
        SA_stable = SAw[pos_order]
        CT_stable = CTw[pos_order]
        dTdz = _slope(z_stable, CT_stable)

    half = len(p_stable) // 2
    p_pair = np.array([np.mean(p_stable[:half]), np.mean(p_stable[half:])])
    if p_pair[1] - p_pair[0] < min_dp / 2:
        return np.nan, dTdz, dTdz_rho, dTdz_T
    sa_pair = np.array([np.mean(SA_stable[:half]), np.mean(SA_stable[half:])])
    ct_pair = np.array([np.mean(CT_stable[:half]), np.mean(CT_stable[half:])])
    n2_val, _ = gsw.Nsquared(sa_pair, ct_pair, p_pair, lat)
    return float(n2_val[0]), dTdz, dTdz_rho, dTdz_T


def profile_stratification(
    P: npt.ArrayLike,
    T: npt.ArrayLike,
    S: float | npt.ArrayLike | None = None,
    lat: float = 0.0,
    lon: float = 0.0,
    window: float = DEFAULT_STRAT_WINDOW,
    min_samples: int = 4,
    min_dp: float = DEFAULT_MIN_DP,
    sort: str = DEFAULT_SORT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(target_P, N2, dTdz)`` of :func:`profile_stratification_full`."""
    res = profile_stratification_full(
        P, T, S, lat=lat, lon=lon, window=window, min_samples=min_samples, min_dp=min_dp,
        sort=sort,
    )
    return res.target_P, res.N2, res.dTdz


def profile_stratification_full(
    P: npt.ArrayLike,
    T: npt.ArrayLike,
    S: float | npt.ArrayLike | None = None,
    lat: float = 0.0,
    lon: float = 0.0,
    window: float = DEFAULT_STRAT_WINDOW,
    min_samples: int = 4,
    min_dp: float = DEFAULT_MIN_DP,
    sort: str = DEFAULT_SORT,
) -> ProfileStratification:
    """Sorted N²/dT/dz vs pressure for one cast, on a coarse depth grid.

    The background (profile- and CTD-product) analog of
    :func:`sorted_stratification`: instead of dissipation-window *times*, it
    evaluates at target pressures spaced by ``window/2`` over the cast's
    pressure range. Within each ``±window/2`` pressure window the samples are
    sorted and differenced by the same :func:`_stable_window` method, so the two
    paths report consistent stratification.

    Evaluating on the coarse target grid (rather than every slow sample) keeps
    the cost to ~``range/window`` windows per cast; callers interpolate onto
    their own grid (e.g. the slow grid for the profile product, depth bins for
    CTD).

    Parameters
    ----------
    P, T : array_like, shape (n,)
        Pressure [dbar] and in-situ temperature [degC] for one cast.
    S : float, array_like, or None
        Practical salinity (scalar, per-sample, or None → 35 PSU).
    lat, lon : float
        Position for TEOS-10 conversions.
    window : float
        Background vertical window [dbar].
    min_samples, min_dp : int, float
        A window with fewer samples or a smaller pressure span yields NaN.
    sort : str
        Parcel order, as in :func:`sorted_stratification`.

    Returns
    -------
    ProfileStratification
        Target pressures [dbar], the per-target N² [s^-2] and dT/dz [K/m vs
        depth], and the temperature- and density-order dT/dz for
        :func:`sort_kappa` (NaN without a per-sample salinity). All empty when
        the cast has too few valid samples.
    """
    _check_sort(sort)
    # Explicit ndarray locals (not reassignments of the ArrayLike params) so
    # mypy keeps the concrete type through .min()/.max(), indexing, and the
    # scalar subtraction below.
    P_arr: np.ndarray = np.asarray(P, dtype=np.float64)
    T_arr: np.ndarray = np.asarray(T, dtype=np.float64)

    S_arr: np.ndarray | None
    if S is None or np.ndim(S) == 0:
        S_const = 35.0 if S is None else float(S)  # type: ignore[arg-type]
        S_arr = None
    else:
        S_arr = np.asarray(S, dtype=np.float64)

    good = np.isfinite(P_arr) & np.isfinite(T_arr)
    if S_arr is not None:
        good &= np.isfinite(S_arr)
    P_arr, T_arr = P_arr[good], T_arr[good]
    S_arr = S_arr[good] if S_arr is not None else None
    empty = np.empty(0, dtype=np.float64)
    if len(P_arr) < min_samples or np.ptp(P_arr) < min_dp:
        return ProfileStratification(empty, empty, empty, empty, empty)

    S_vals = S_arr if S_arr is not None else np.full(len(P_arr), S_const)
    SA = gsw.SA_from_SP(S_vals, P_arr, lon, lat)
    CT = gsw.CT_from_t(SA, T_arr, P_arr)
    sigma = gsw.sigma0(SA, CT)
    depth = -gsw.z_from_p(P_arr, lat)

    target_P = _profile_targets(P_arr, window, min_dp)
    half_w = window / 2.0
    ext_half = TSORT_TREND_SPAN * half_w
    want_kappa = S_arr is not None  # see sorted_stratification
    N2 = np.full(len(target_P), np.nan)
    dTdz = np.full(len(target_P), np.nan)
    dTdz_T = np.full(len(target_P), np.nan)
    dTdz_rho = np.full(len(target_P), np.nan)
    for k, pt in enumerate(target_P):
        sel = np.abs(P_arr - pt) <= half_w
        if np.sum(sel) < min_samples:
            continue
        Pw = P_arr[sel]
        if np.ptp(Pw) < min_dp:
            continue
        zw = depth[sel]
        direction = 0
        if sort == "temperature" or want_kappa:
            ext = np.abs(P_arr - pt) <= ext_half
            direction = _trend_direction(depth[ext], CT[ext], float(zw.min()), float(zw.max()))
        N2[k], dTdz[k], dTdz_rho[k], dTdz_T[k] = _stable_window(
            Pw, zw, T_arr[sel], SA[sel], CT[sel], sigma[sel], lat, min_dp,
            sort=sort, direction=direction, want_kappa=want_kappa,
        )

    return ProfileStratification(target_P, N2, dTdz, dTdz_T, dTdz_rho)


def _profile_targets(P_arr: np.ndarray, window: float, min_dp: float) -> np.ndarray:
    """Window-center pressures for :func:`profile_stratification_full` and twins.

    Spaced by ``max(window/2, min_dp)`` from the shallowest to past the deepest
    finite sample. The caller has checked ``ptp(P_arr) >= min_dp``, so with a
    positive step this always yields at least one target.
    """
    step = max(window / 2.0, min_dp)
    p_lo, p_hi = float(P_arr.min()), float(P_arr.max())
    return np.arange(p_lo, p_hi + step, step)


def _window_salinity(S_win: np.ndarray | None, S_fallback: float) -> float:
    """One practical salinity for a window: the median of its finite samples.

    Falls back to *S_fallback* when the window holds none. Used to convert a
    temperature that has no co-registered salinity (an FP07) to conservative
    temperature, so salinity noise cannot reach the gradient (see
    :func:`_stable_window` for the size of that effect).
    """
    if S_win is not None:
        finite = S_win[np.isfinite(S_win)]
        if finite.size:
            return float(np.median(finite))
    return S_fallback


def _gradient_window(Pw, zw, Tw, S_scalar, lat, lon, sort, direction):
    """Least-squares dCT/dz of one window, temperature-sorted or as measured."""
    SA = gsw.SA_from_SP(np.full(len(Pw), S_scalar), Pw, lon, lat)
    CT = gsw.CT_from_t(SA, Tw, Pw)
    pos_order = np.argsort(Pw, kind="stable")
    order = _temperature_order(CT, direction, pos_order) if sort == "temperature" else pos_order
    return _slope(zw[pos_order], CT[order])


def _salinity_inputs(S, n: int) -> tuple[np.ndarray | None, float]:
    """``(per-sample array or None, fallback scalar)`` from a salinity argument."""
    if S is None:
        return None, 35.0
    if np.ndim(S) == 0:
        return None, float(S)
    S_arr = np.asarray(S, dtype=np.float64)
    if S_arr.shape != (n,):
        raise ValueError(f"S has shape {S_arr.shape}; expected a scalar or shape ({n},)")
    finite = S_arr[np.isfinite(S_arr)]
    return S_arr, (float(np.median(finite)) if finite.size else 35.0)


def _fallback_ct(P_arr, T_arr, S_fallback, lat, lon):
    """CT of a whole temperature series at one salinity (trend direction only)."""
    SA = gsw.SA_from_SP(np.full(len(P_arr), S_fallback), P_arr, lon, lat)
    return gsw.CT_from_t(SA, T_arr, P_arr)


def window_temperature_gradient(
    win_times: npt.ArrayLike,
    win_half_width: float,
    t: npt.ArrayLike,
    P: npt.ArrayLike,
    T: npt.ArrayLike,
    S: float | npt.ArrayLike | None = None,
    lat: float = 0.0,
    lon: float = 0.0,
    sort: str = "temperature",
    min_samples: int = 4,
    min_dp: float = DEFAULT_MIN_DP,
) -> np.ndarray:
    """Background dCT/dz per window from a temperature alone (e.g. an FP07).

    The temperature-only twin of :func:`sorted_stratification`, for a sensor with
    no co-registered salinity. Conservative temperature is formed with one
    salinity per window (:func:`_window_salinity`; *S* may be a scalar, a
    per-sample array on *t*, or None for 35 PSU), so salinity noise cannot enter
    the gradient. *sort* is ``"temperature"`` (direction from
    :func:`_trend_direction`) or ``"none"``; a density order is not defined
    without a co-registered salinity.

    Returns dT/dz [K/m vs depth, positive down] per window; NaN where the window
    has fewer than *min_samples* finite samples or a pressure span below
    *min_dp*.
    """
    _check_sort(sort, ("temperature", "none"))
    win_t = np.asarray(win_times, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)
    P_arr = np.asarray(P, dtype=np.float64)
    T_arr = np.asarray(T, dtype=np.float64)
    S_arr, S_fallback = _salinity_inputs(S, len(P_arr))
    depth = -gsw.z_from_p(P_arr, lat)
    finite = np.isfinite(P_arr) & np.isfinite(T_arr)
    CT_all = _fallback_ct(P_arr, T_arr, S_fallback, lat, lon) if sort == "temperature" else None
    ext_half = TSORT_TREND_SPAN * win_half_width
    dTdz = np.full(len(win_t), np.nan)
    for j, tau in enumerate(win_t):
        sel = (np.abs(t_arr - tau) <= win_half_width) & finite
        if np.sum(sel) < min_samples or np.ptp(P_arr[sel]) < min_dp:
            continue
        zw = depth[sel]
        direction = 0
        if CT_all is not None:
            ext = (np.abs(t_arr - tau) <= ext_half) & finite
            direction = _trend_direction(depth[ext], CT_all[ext], float(zw.min()), float(zw.max()))
        S_w = _window_salinity(S_arr[sel] if S_arr is not None else None, S_fallback)
        dTdz[j] = _gradient_window(P_arr[sel], zw, T_arr[sel], S_w, lat, lon, sort, direction)
    return dTdz


def profile_temperature_gradient(
    P: npt.ArrayLike,
    T: npt.ArrayLike,
    S: float | npt.ArrayLike | None = None,
    lat: float = 0.0,
    lon: float = 0.0,
    window: float = DEFAULT_STRAT_WINDOW,
    sort: str = "temperature",
    min_samples: int = 4,
    min_dp: float = DEFAULT_MIN_DP,
) -> tuple[np.ndarray, np.ndarray]:
    """``(target_P, dTdz)`` for one cast from a temperature alone (e.g. an FP07).

    The temperature-only twin of :func:`profile_stratification_full`, with the
    salinity and *sort* handling of :func:`window_temperature_gradient`. The
    target pressures follow the same rule as the N² path but from this
    temperature's own finite samples, so the two grids can differ; callers
    interpolate each onto their own grid.
    """
    _check_sort(sort, ("temperature", "none"))
    P_arr = np.asarray(P, dtype=np.float64)
    T_arr = np.asarray(T, dtype=np.float64)
    S_arr, S_fallback = _salinity_inputs(S, len(P_arr))
    good = np.isfinite(P_arr) & np.isfinite(T_arr)
    P_arr, T_arr = P_arr[good], T_arr[good]
    S_arr = S_arr[good] if S_arr is not None else None
    empty = np.empty(0, dtype=np.float64)
    if len(P_arr) < min_samples or np.ptp(P_arr) < min_dp:
        return empty, empty
    depth = -gsw.z_from_p(P_arr, lat)
    CT_all = _fallback_ct(P_arr, T_arr, S_fallback, lat, lon) if sort == "temperature" else None
    target_P = _profile_targets(P_arr, window, min_dp)
    half_w = window / 2.0
    ext_half = TSORT_TREND_SPAN * half_w
    dTdz = np.full(len(target_P), np.nan)
    for k, pt in enumerate(target_P):
        sel = np.abs(P_arr - pt) <= half_w
        if np.sum(sel) < min_samples or np.ptp(P_arr[sel]) < min_dp:
            continue
        zw = depth[sel]
        direction = 0
        if CT_all is not None:
            ext = np.abs(P_arr - pt) <= ext_half
            direction = _trend_direction(depth[ext], CT_all[ext], float(zw.min()), float(zw.max()))
        S_w = _window_salinity(S_arr[sel] if S_arr is not None else None, S_fallback)
        dTdz[k] = _gradient_window(P_arr[sel], zw, T_arr[sel], S_w, lat, lon, sort, direction)
    return target_P, dTdz


class KappaSummary(NamedTuple):
    """:func:`sort_kappa` over a whole cast and in its worst depth band."""

    overall: float  # NaN when undefined
    worst: float  # lowest band kappa (= overall when no band qualifies)
    worst_P: float  # center pressure of that band [dbar]; NaN when none


def sort_kappa(
    dTdz_rho: npt.ArrayLike,
    dTdz_T: npt.ArrayLike,
    min_abs: float = KAPPA_MIN_DTDZ,
    min_windows: int = KAPPA_MIN_WINDOWS,
) -> float:
    """Median of density-order over temperature-order dT/dz (see KAPPA_WARN).

    Takes a stratification result's ``dTdz_rho`` and ``dTdz_T``, over windows
    where both are finite and ``|dTdz_T| >= min_abs``; the stratification
    functions already leave them NaN where the measured-order gradient is under
    :data:`KAPPA_MIN_DTDZ`, which is the gate that keeps mixed layers from
    voting. NaN when fewer than *min_windows* windows qualify. ~1 means salinity
    leaves the temperature order alone; well below 1 means it scrambles it.
    """
    d = np.asarray(dTdz_rho, dtype=np.float64)
    d_T = np.asarray(dTdz_T, dtype=np.float64)
    ok = np.isfinite(d) & np.isfinite(d_T) & (np.abs(d_T) >= min_abs)
    if int(np.count_nonzero(ok)) < min_windows:
        return float("nan")
    return float(np.median(d[ok] / d_T[ok]))


def kappa_summary(
    dTdz_rho: npt.ArrayLike,
    dTdz_T: npt.ArrayLike,
    P_win: npt.ArrayLike,
    band: float = KAPPA_BAND_DBAR,
    band_min_windows: int = KAPPA_BAND_MIN_WINDOWS,
) -> KappaSummary:
    """:func:`sort_kappa` for the whole set of windows and per depth band.

    A salinity failure can be confined to part of a cast (on the motivating
    VMP-5500 casts it was worst at 150-250 dbar of 1000), where a cast-wide
    median would dilute it. Bands are *band* dbar wide from 0; a band votes
    with at least *band_min_windows* qualifying windows.
    """
    d = np.asarray(dTdz_rho, dtype=np.float64)
    d_T = np.asarray(dTdz_T, dtype=np.float64)
    pw = np.asarray(P_win, dtype=np.float64)
    overall = sort_kappa(d, d_T)
    worst_band: tuple[float, float] | None = None
    finite_p = np.flatnonzero(np.isfinite(pw))
    band_idx = np.maximum(np.floor(pw[finite_p] / band).astype(int), 0)  # P < 0 -> top band
    for b in np.unique(band_idx):
        members = finite_p[band_idx == b]
        k = sort_kappa(d[members], d_T[members], min_windows=band_min_windows)
        if np.isfinite(k) and (worst_band is None or k < worst_band[0]):
            worst_band = (k, (b + 0.5) * band)
    worst, worst_P = worst_band if worst_band is not None else (overall, float("nan"))
    return KappaSummary(overall, worst, worst_P)


def kappa_is_low(summary: KappaSummary) -> bool:
    """True when the cast's kappa is below KAPPA_WARN or a band's below KAPPA_BAND_WARN."""
    over = np.isfinite(summary.overall) and summary.overall < KAPPA_WARN
    band = (
        np.isfinite(summary.worst_P)
        and np.isfinite(summary.worst)
        and summary.worst < KAPPA_BAND_WARN
    )
    return bool(over or band)


def kappa_warning(summary: KappaSummary, label: str, sort: str = DEFAULT_SORT) -> str | None:
    """Warning text when :func:`kappa_is_low`, else None.

    *sort* is the order the product's dT/dz used (an FP07 gradient is
    temperature-sorted even under ``stratification.sort: density``): kappa is the
    same under every sort, but only a density-sorted dT/dz is the one it says is
    biased.
    """
    _check_sort(sort)
    if not kappa_is_low(summary):
        return None
    band_is_low = (
        np.isfinite(summary.worst_P)
        and summary.worst < KAPPA_BAND_WARN
        and not (np.isfinite(summary.overall) and summary.overall <= summary.worst)
    )
    if band_is_low:
        low = summary.worst
        lo_p = summary.worst_P - KAPPA_BAND_DBAR / 2
        where = (
            f" in the {lo_p:g}-{lo_p + KAPPA_BAND_DBAR:g} dbar band "
            f"({summary.overall:.2f} over the cast); a band warns below "
            f"{KAPPA_BAND_WARN:g}"
        )
    else:
        low = summary.overall
        where = f" over the cast; warning below {KAPPA_WARN:g}"
    cause = (
        "Salinity is reordering the parcels: salinity noise (an unpumped or "
        "T/C-mismatched conductivity cell) or density-compensated structure."
    )
    if sort == "density":
        return (
            f"{label}: the density-sorted dT/dz is {low:.2f} of the "
            f"temperature-sorted value{where}. {cause} If it is noise, dT/dz "
            "is biased low and N2 high, so K_T and Gamma are biased high. "
            "stratification.sort: temperature (or dTdz_source: fp07) fixes dT/dz; N2 "
            "still carries the salinity, so for N2 use a fixed salinity or one from a "
            "T-S relation."
        )
    order = "temperature-sorted" if sort == "temperature" else "in measured order"
    return (
        f"{label}: a density-sorted dT/dz would be {low:.2f} of the "
        f"temperature-sorted value{where}. {cause} dT/dz here is {order}, so the "
        "salinity does not reorder it, but N2 carries the salinity, and Gamma and "
        "K_rho with it. If it is noise, use a fixed salinity or one from a T-S "
        "relation for N2."
    )


def mixing_coefficients(
    epsilon: npt.ArrayLike,
    chi: npt.ArrayLike,
    N2: npt.ArrayLike,
    dTdz: npt.ArrayLike,
    *,
    N2_min: float = DEFAULT_N2_MIN,
    dTdz_min: float = DEFAULT_DTDZ_MIN,
    gamma_osborn: float = GAMMA_OSBORN,
    K_rho_max: float = DEFAULT_K_RHO_MAX,
    K_T_max: float = DEFAULT_K_T_MAX,
    Gamma_max: float = DEFAULT_GAMMA_MAX,
) -> MixingResult:
    """Derived mixing quantities from epsilon, chi, and stratification.

    .. math::

        K_T = \\frac{\\chi}{2 (\\partial T/\\partial z)^2}, \\qquad
        \\Gamma = \\frac{N^2 \\chi}{2 \\varepsilon
                  (\\partial T/\\partial z)^2}, \\qquad
        K_\\rho = \\Gamma_0 \\frac{\\varepsilon}{N^2}

    Masking (NaN):

    - ``K_T`` and ``Gamma`` where ``|dT/dz| < dTdz_min`` (well-mixed:
      the temperature-variance budget no longer constrains a
      diffusivity).
    - ``Gamma`` and ``K_rho`` where ``N2 < N2_min`` (unstratified or
      statically unstable at the window scale: the Osborn scaling does
      not apply).
    - Each coefficient where it exceeds its OWN upper sanity bound —
      ``K_rho`` > ``K_rho_max`` (near-floor-``N2`` artifact), ``K_T`` >
      ``K_T_max`` (weak ``dT/dz`` with contaminated ``chi``), ``Gamma`` >
      ``Gamma_max`` (near-zero ``epsilon`` in the denominator). Gating each on
      its own ceiling (rather than masking all three wherever ``K_rho`` is
      masked) keeps a legitimately large ``K_T`` in low-``N2`` water. The masked
      count per variable is reported via :mod:`warnings`.
    - Any output where the corresponding ``chi`` or ``epsilon`` is
      non-finite or non-positive (a positive dissipation rate is required
      for every quantity).

    Parameters
    ----------
    epsilon : array_like
        TKE dissipation rate [W/kg].
    chi : array_like
        Thermal variance dissipation rate [K^2/s].
    N2 : array_like
        Buoyancy frequency squared [s^-2].
    dTdz : array_like
        Background temperature gradient [K/m] (sign irrelevant; only
        its square is used).
    N2_min, dTdz_min : float
        Validity floors (see module constants).
    gamma_osborn : float
        Constant mixing coefficient for ``K_rho`` (default 0.2).
    K_rho_max : float
        Upper sanity bound on ``K_rho`` [m^2/s]; windows exceeding it are
        masked as physically implausible (see ``DEFAULT_K_RHO_MAX``).
    K_T_max : float
        Upper sanity bound on ``K_T`` [m^2/s] (see ``DEFAULT_K_T_MAX``,
        default 10).
    Gamma_max : float
        Upper sanity bound on the measured ``Gamma`` [1] (see
        ``DEFAULT_GAMMA_MAX``, default 5).

    Returns
    -------
    MixingResult
        ``K_T`` [m^2/s], ``Gamma`` [1], ``K_rho`` [m^2/s].
    """
    epsilon = np.asarray(epsilon, dtype=np.float64)
    chi = np.asarray(chi, dtype=np.float64)
    N2 = np.asarray(N2, dtype=np.float64)
    dTdz = np.asarray(dTdz, dtype=np.float64)

    grad_ok = np.isfinite(dTdz) & (np.abs(dTdz) >= dTdz_min)
    strat_ok = np.isfinite(N2) & (N2_min <= N2)
    chi_ok = np.isfinite(chi) & (chi > 0)
    eps_ok = np.isfinite(epsilon) & (epsilon > 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        K_T = np.where(grad_ok & chi_ok, chi / (2.0 * dTdz**2), np.nan)
        Gamma = np.where(
            grad_ok & strat_ok & chi_ok & eps_ok,
            N2 * chi / (2.0 * epsilon * dTdz**2),
            np.nan,
        )
        K_rho = np.where(strat_ok & eps_ok, gamma_osborn * epsilon / N2, np.nan)

    # Mask physically implausible magnitudes per variable (audit #30, r-2026-07):
    # each coefficient is gated on its OWN ceiling, not on K_rho's, because they
    # blow up in different regimes — K_rho on near-floor N^2, K_T on a weak dT/dz
    # with contaminated chi, Gamma on a near-zero shear epsilon. A NaN compares
    # False, so only finite over-ceiling windows are masked/counted.
    for name, arr, ceiling, unit in (
        ("K_rho", K_rho, K_rho_max, "m^2/s"),
        ("K_T", K_T, K_T_max, "m^2/s"),
        ("Gamma", Gamma, Gamma_max, ""),
    ):
        over = arr > ceiling
        n_over = int(np.count_nonzero(over))
        if n_over:
            arr[over] = np.nan
            unit_str = f" {unit}" if unit else ""
            warnings.warn(
                f"mixing_coefficients: masked {n_over} {name} value(s) exceeding "
                f"{ceiling}{unit_str} (physically implausible)",
                stacklevel=2,
            )

    return MixingResult(K_T=K_T, Gamma=Gamma, K_rho=K_rho)


def pair_nearest(
    src_times: npt.ArrayLike,
    src_values: npt.ArrayLike,
    dst_times: npt.ArrayLike,
    max_dt: float | None = None,
) -> np.ndarray:
    """Pair each destination time with the nearest source value.

    Used to align epsilon estimates onto the chi window grid when the
    two were computed with different window lengths.  ``max_dt`` (None
    = one median source spacing) rejects pairings with no temporally
    co-located source estimate.
    """
    # Fresh ndarray-typed names (params are npt.ArrayLike; reusing them keeps
    # mypy's declared type as ArrayLike and flags every later index/len).
    src_t = np.asarray(src_times, dtype=np.float64)
    src_v = np.asarray(src_values, dtype=np.float64)
    dst_t = np.asarray(dst_times, dtype=np.float64)

    out = np.full(len(dst_t), np.nan)
    # The tolerance describes the SOURCE SAMPLING GEOMETRY, so it comes from the
    # full time base BEFORE the finite filter below. Taking the median spacing of
    # the SURVIVING estimates let QC gaps redefine the cadence: with sources
    # every 1 s from 0..1000 but only t=0 and t=1000 finite, the median spacing
    # became 1000 s and a query at t=500 paired to an estimate 500 s away
    # (issue #180 F05). It is also capped, so a source series that is sparse by
    # construction cannot pair across an arbitrary span.
    if max_dt is None:
        # Defect (audit): a zero/negative median spacing (duplicate or
        # clamped source times) would collapse the tolerance to <= 0 and
        # silently drop every pairing that is not an exact time match;
        # fall back to a usable positive floor instead.
        med = float(np.median(np.diff(np.sort(src_t)))) if len(src_t) > 1 else 0.0
        max_dt = min(med if med > 0 else 30.0, MAX_PAIR_DT)

    # Only finite source estimates are candidates: a NaN (QC-rejected) epsilon
    # window must not shadow a valid epsilon at an adjacent window within max_dt
    # (else Gamma/K_rho are silently dropped while K_T survives).
    finite = np.isfinite(src_v)
    if not finite.all():
        src_t = src_t[finite]
        src_v = src_v[finite]
    if len(src_t) == 0:
        return out
    order = np.argsort(src_t)
    st = src_t[order]
    sv = src_v[order]
    if len(st) == 1:
        ok = np.abs(st[0] - dst_t) <= max_dt
        out[ok] = sv[0]
        return out
    idx = np.clip(np.searchsorted(st, dst_t), 1, len(st) - 1)
    left = idx - 1
    pick = np.where(np.abs(st[idx] - dst_t) < np.abs(st[left] - dst_t), idx, left)
    dt = np.abs(st[pick] - dst_t)
    ok = dt <= max_dt
    out[ok] = sv[pick[ok]]
    return out
