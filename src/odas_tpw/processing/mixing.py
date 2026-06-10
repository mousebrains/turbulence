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

# Stratification floor [s^-2].  Below ~1e-9 s^-2 (N ~ 0.02 cph) the
# water column is effectively unstratified at the window scale and the
# Osborn scaling K ~ epsilon/N^2 diverges.
DEFAULT_N2_MIN = 1e-9

# Background temperature-gradient floor [K/m].  chi/(dT/dz)^2 diverges
# as the mean gradient vanishes (well-mixed layers); 1e-4 K/m over a
# ~5 m window corresponds to ~0.5 mK of signal, near FP07 resolution.
DEFAULT_DTDZ_MIN = 1e-4

# Canonical Osborn (1980) mixing coefficient.
GAMMA_OSBORN = 0.2

# Minimum pressure span [dbar] within a window for gradient estimates;
# below this (stalled instrument) vertical gradients are unconstrained.
DEFAULT_MIN_DP = 0.2


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
        stratification matters (note the ``salinity_assumed`` flag in
        the returned dataset attributes when integrating).
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
    if len(src_times) == 0:
        return out
    if max_dt is None:
        max_dt = (
            float(np.median(np.diff(np.sort(src_times)))) if len(src_times) > 1 else 30.0
        )
    order = np.argsort(src_times)
    st = src_times[order]
    sv = src_values[order]
    if len(st) == 1:
        ok = np.abs(st[0] - dst_times) <= max_dt
        out[ok] = sv[0]
        return out
    idx = np.clip(np.searchsorted(st, dst_times), 1, len(st) - 1)
    left = idx - 1
    pick = np.where(
        np.abs(st[idx] - dst_times) < np.abs(st[left] - dst_times), idx, left
    )
    dt = np.abs(st[pick] - dst_times)
    ok = dt <= max_dt
    out[ok] = sv[pick[ok]]
    return out
