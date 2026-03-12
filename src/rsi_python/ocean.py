# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Seawater properties for turbulence calculations."""

import numpy as np
import numpy.typing as npt


def visc35(T: npt.ArrayLike) -> np.ndarray:
    """Kinematic viscosity of seawater at S=35, P=0 [m²/s].

    Parameters
    ----------
    T : float or array_like
        Temperature in °C. Valid for 0-20 °C.

    Returns
    -------
    nu : float or ndarray
        Kinematic viscosity in m²/s.

    Reference: ODAS visc35.m — 3rd-order polynomial fit,
    error < 1% for 0 ≤ T ≤ 20 °C, 30 ≤ S ≤ 40 PSU, atmospheric pressure.
    """
    pol = np.array(
        [
            -1.131311019739306e-011,
            1.199552027472192e-009,
            -5.864346822839289e-008,
            1.828297985908266e-006,
        ]
    )
    return np.polyval(pol, T)


def visc(T: npt.ArrayLike, S: npt.ArrayLike = 35, P: npt.ArrayLike = 0) -> np.ndarray:
    """Kinematic viscosity of seawater [m²/s].

    For S=35, P=0 falls back to the visc35 polynomial (matches ODAS).
    Otherwise uses Sharqawy et al. (2010) for dynamic viscosity
    and gsw (TEOS-10) for in-situ density.

    Parameters
    ----------
    T : float or array_like
        In-situ temperature [°C].
    S : float or array_like
        Practical salinity [PSU]. Default: 35.
    P : float or array_like
        Pressure [dbar]. Default: 0.

    Returns
    -------
    nu : float or ndarray
        Kinematic viscosity [m²/s].
    """
    T = np.asarray(T, dtype=float)
    S = np.asarray(S, dtype=float)
    P = np.asarray(P, dtype=float)

    if np.all(S == 35) and np.all(P == 0):
        return visc35(T)

    import gsw

    # Dynamic viscosity of pure water [Pa·s] — Sharqawy et al. (2010) Eq. 22
    mu_w = 4.2844e-5 + (0.157 * (T + 64.993) ** 2 - 91.296) ** (-1)

    # Salinity correction — Sharqawy et al. (2010) Eq. 23
    A = 1.541 + 1.998e-2 * T - 9.52e-5 * T**2
    B = 7.974 - 7.561e-2 * T + 4.724e-4 * T**2
    S_frac = S * 1e-3  # PSU to mass fraction
    mu = mu_w * (1 + A * S_frac + B * S_frac**2)

    # In-situ density from gsw (TEOS-10)
    SA = gsw.SA_from_SP(S, P, 0, 0)
    CT = gsw.CT_from_t(SA, T, P)
    rho = gsw.rho(SA, CT, P)

    return mu / rho


def density(T: npt.ArrayLike, S: npt.ArrayLike, P: npt.ArrayLike) -> np.ndarray:
    """In-situ density of seawater [kg/m³] via TEOS-10.

    Parameters
    ----------
    T : float or array_like
        In-situ temperature [°C].
    S : float or array_like
        Practical salinity [PSU].
    P : float or array_like
        Pressure [dbar].

    Returns
    -------
    rho : float or ndarray
        In-situ density [kg/m³].
    """
    import gsw

    SA = gsw.SA_from_SP(S, P, 0, 0)
    CT = gsw.CT_from_t(SA, T, P)
    return gsw.rho(SA, CT, P)


def buoyancy_freq(
    T: npt.ArrayLike,
    S: npt.ArrayLike,
    P: npt.ArrayLike,
    lat: float = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Buoyancy frequency squared N² [s⁻²] from profiles.

    Parameters
    ----------
    T : array_like
        In-situ temperature profile [°C].
    S : array_like
        Practical salinity profile [PSU].
    P : array_like
        Pressure profile [dbar].
    lat : float
        Latitude [°N]. Default: 0.

    Returns
    -------
    N2 : ndarray
        Buoyancy frequency squared [s⁻²], length len(P)-1.
    p_mid : ndarray
        Mid-point pressures [dbar], length len(P)-1.
    """
    import gsw

    SA = gsw.SA_from_SP(S, P, 0, 0)
    CT = gsw.CT_from_t(SA, T, P)
    N2, p_mid = gsw.Nsquared(SA, CT, P, lat)
    return N2, p_mid
