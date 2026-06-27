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

from typing import NamedTuple

import gsw
import numpy as np
import numpy.typing as npt

# Stratification floor [s^-2].  Defect (audit): a 1e-9 s^-2 floor only
# averts division-by-zero; just above it (N ~ 0.02 cph) the water is
# effectively unstratified, where Osborn K ~ epsilon/N^2 still diverges
# to physically meaningless diffusivities (e.g. eps=1e-7, N2=2e-9 ->
# K_rho ~ 10 m^2/s).  Floor at 1e-7 s^-2 (N ~ 0.26 cph), below which the
# Osborn estimate is no longer meaningful, so near-floor windows are
# masked rather than leaked into the product.
DEFAULT_N2_MIN = 1e-7

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


class StratificationResult(NamedTuple):
    """Per-window background stratification."""

    N2: np.ndarray  # buoyancy frequency squared [s^-2]
    dTdz: np.ndarray  # background temperature gradient vs depth [K/m]


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

    - ``dTdz`` is the slope of a least-squares fit of in-situ
      temperature against depth (positive down), i.e. the *background*
      gradient at the window scale — the same quantity whose square
      normalizes chi in the Osborn-Cox model.
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

        # Background temperature gradient vs depth (least squares)
        dTdz[j] = np.polyfit(zw, Tw, 1)[0]

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

    return StratificationResult(N2=N2, dTdz=dTdz)


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
) -> StratificationResult:
    """Background N² and dT/dz per window from the adiabatically sorted profile.

    Like :func:`window_stratification`, but within each window the parcels are
    Thorpe-sorted to a statically stable ordering (densest deepest) before
    differencing. Sorting removes density inversions / overturns so the
    gradients reflect the *background* stratification rather than the
    instantaneous (possibly unstable) profile — the adiabatic-leveling idea
    applied window-by-window:

    - The window's (SA, CT, p) are formed via TEOS-10 and ranked by potential
      density (``gsw.sigma0``); the i-th densest parcel is assigned to the
      i-th deepest pressure, giving the stable reference profile.
    - ``N2`` is ``gsw.Nsquared`` between the shallow- and deep-half means of
      that stable profile. After sorting it is ≥ 0 for ordinary stratification,
      but can be slightly negative in strongly density-compensated water:
      ``gsw.Nsquared`` averages SA and CT separately and re-evaluates the
      nonlinear equation of state, which (via cabbeling) is not guaranteed
      monotone under the potential-density (sigma0) sort. Such windows are
      treated as effectively unstratified by the downstream Osborn scaling.
    - ``dTdz`` is the least-squares slope of the stably-sorted in-situ
      temperature against depth (positive down).

    Parameters and return value match :func:`window_stratification`; NaN is
    returned for windows with fewer than *min_samples* valid samples or a
    pressure span below *min_dp*.
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
        Sw = Sw[good] if Sw is not None else None
        if np.ptp(Pw) < min_dp:
            continue

        Sw_vals = Sw if Sw is not None else np.full(len(Pw), S_const)
        SA = gsw.SA_from_SP(Sw_vals, Pw, lon, lat)
        CT = gsw.CT_from_t(SA, Tw, Pw)
        sigma = gsw.sigma0(SA, CT)
        N2[j], dTdz[j] = _stable_window(Pw, zw, Tw, SA, CT, sigma, lat, min_dp)

    return StratificationResult(N2=N2, dTdz=dTdz)


def _stable_window(Pw, zw, Tw, SAw, CTw, sigmaw, lat, min_dp):
    """N² and dT/dz for one window's samples, Thorpe-sorted to stability.

    Reconstructs the statically stable reference profile (the k-th shallowest
    position holds the k-th least-dense parcel), then returns ``(N2, dTdz)``:
    N² from ``gsw.Nsquared`` between the shallow/deep-half means of the stable
    profile (NaN if the half-mean pressure separation is below ``min_dp/2``),
    and dT/dz as the least-squares slope of the stably-sorted temperature vs
    depth. Inputs must be finite with length ≥ 2.
    """
    pos_order = np.argsort(Pw, kind="stable")
    den_order = np.argsort(sigmaw, kind="stable")
    p_stable = Pw[pos_order]
    z_stable = zw[pos_order]
    SA_stable = SAw[den_order]
    CT_stable = CTw[den_order]
    T_stable = Tw[den_order]

    dTdz = float(np.polyfit(z_stable, T_stable, 1)[0])

    half = len(p_stable) // 2
    p_pair = np.array([np.mean(p_stable[:half]), np.mean(p_stable[half:])])
    if p_pair[1] - p_pair[0] < min_dp / 2:
        return np.nan, dTdz
    sa_pair = np.array([np.mean(SA_stable[:half]), np.mean(SA_stable[half:])])
    ct_pair = np.array([np.mean(CT_stable[:half]), np.mean(CT_stable[half:])])
    n2_val, _ = gsw.Nsquared(sa_pair, ct_pair, p_pair, lat)
    return float(n2_val[0]), dTdz


def profile_stratification(
    P: npt.ArrayLike,
    T: npt.ArrayLike,
    S: float | npt.ArrayLike | None = None,
    lat: float = 0.0,
    lon: float = 0.0,
    window: float = DEFAULT_STRAT_WINDOW,
    min_samples: int = 4,
    min_dp: float = DEFAULT_MIN_DP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sorted N²/dT/dz vs pressure for one cast, on a coarse depth grid.

    The background (profile- and CTD-product) analogue of
    :func:`sorted_stratification`: instead of dissipation-window *times*, it
    evaluates at target pressures spaced by ``window/2`` over the cast's
    pressure range. Within each ``±window/2`` pressure window the samples are
    Thorpe-sorted and differenced by the same :func:`_stable_window` method, so
    the two paths report consistent stratification.

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

    Returns
    -------
    (target_P, N2, dTdz) : tuple of ndarray
        Target pressures [dbar] and the per-target N² [s^-2] and dT/dz
        [K/m vs depth]. ``target_P`` is empty when the cast has too few
        valid samples.
    """
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
        return empty, empty, empty

    S_vals = S_arr if S_arr is not None else np.full(len(P_arr), S_const)
    SA = gsw.SA_from_SP(S_vals, P_arr, lon, lat)
    CT = gsw.CT_from_t(SA, T_arr, P_arr)
    sigma = gsw.sigma0(SA, CT)
    depth = -gsw.z_from_p(P_arr, lat)

    half_w = window / 2.0
    step = max(half_w, min_dp)
    p_lo, p_hi = float(P_arr.min()), float(P_arr.max())
    # p_lo < p_hi (ptp >= min_dp guard above) and step > 0, so arange always
    # yields >= 1 target.
    target_P = np.arange(p_lo, p_hi + step, step)

    N2 = np.full(len(target_P), np.nan)
    dTdz = np.full(len(target_P), np.nan)
    for k, pt in enumerate(target_P):
        sel = np.abs(P_arr - pt) <= half_w
        if np.sum(sel) < min_samples:
            continue
        Pw = P_arr[sel]
        if np.ptp(Pw) < min_dp:
            continue
        N2[k], dTdz[k] = _stable_window(
            Pw, depth[sel], T_arr[sel], SA[sel], CT[sel], sigma[sel], lat, min_dp
        )

    return target_P, N2, dTdz


def mixing_coefficients(
    epsilon: npt.ArrayLike,
    chi: npt.ArrayLike,
    N2: npt.ArrayLike,
    dTdz: npt.ArrayLike,
    *,
    N2_min: float = DEFAULT_N2_MIN,
    dTdz_min: float = DEFAULT_DTDZ_MIN,
    gamma_osborn: float = GAMMA_OSBORN,
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
    src_times = np.asarray(src_times, dtype=np.float64)
    src_values = np.asarray(src_values, dtype=np.float64)
    dst_times = np.asarray(dst_times, dtype=np.float64)

    out = np.full(len(dst_times), np.nan)
    # Only finite source estimates are candidates: a NaN (QC-rejected) epsilon
    # window must not shadow a valid epsilon at an adjacent window within max_dt
    # (else Gamma/K_rho are silently dropped while K_T survives).
    finite = np.isfinite(src_values)
    if not finite.all():
        src_times = src_times[finite]
        src_values = src_values[finite]
    if len(src_times) == 0:
        return out
    if max_dt is None:
        # Defect (audit): a zero/negative median spacing (duplicate or
        # clamped source times) would collapse the tolerance to <= 0 and
        # silently drop every pairing that is not an exact time match;
        # fall back to a usable positive floor instead.
        med = float(np.median(np.diff(np.sort(src_times)))) if len(src_times) > 1 else 0.0
        max_dt = med if med > 0 else 30.0
    order = np.argsort(src_times)
    st = src_times[order]
    sv = src_values[order]
    if len(st) == 1:
        ok = np.abs(st[0] - dst_times) <= max_dt
        out[ok] = sv[0]
        return out
    idx = np.clip(np.searchsorted(st, dst_times), 1, len(st) - 1)
    left = idx - 1
    pick = np.where(np.abs(st[idx] - dst_times) < np.abs(st[left] - dst_times), idx, left)
    dt = np.abs(st[pick] - dst_times)
    ok = dt <= max_dt
    out[ok] = sv[pick[ok]]
    return out
