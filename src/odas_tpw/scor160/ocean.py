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
    # The cubic fit (valid 0-20 C) crosses zero near 64.5 C and goes negative
    # beyond it. A single spurious-hot temperature sample would then yield a
    # negative viscosity and make nasmyth raise on sqrt(negative), crashing the
    # whole file. Floor at a small positive value (well below any realistic
    # seawater viscosity, so 0-~60 C is unaffected).
    return np.maximum(np.polyval(pol, T), 1.0e-7)


def visc(T: npt.ArrayLike, S: npt.ArrayLike = 35, P: npt.ArrayLike = 0) -> np.ndarray:
    """Kinematic viscosity of seawater [m²/s].

    For S=35, P=0 falls back to the visc35 polynomial (matches ODAS).
    Otherwise uses Sharqawy et al. (2010) for dynamic viscosity
    and gsw (TEOS-10) for in-situ density.

    Note: the two branches differ by up to ~1.4% at the S=35/P=0 seam (max at
    0 °C; ~0.7% at 5-15 °C), so nu is mildly discontinuous there. This is
    intentional — visc35 is the ODAS reference used by the ATOMIX shear
    benchmark at exactly S=35/P=0, so the seam is not blended. The seam is
    unreachable in normal processing (in-situ P>0 keeps a cast on the Sharqawy
    branch), and a ~1.4% nu change propagates to <0.5% in epsilon (#20).

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

    # Floor to a small positive value, mirroring visc35: a spurious-cold window
    # mean (T well below 0) drives the Sharqawy polynomial negative, and a
    # negative viscosity crashes the Nasmyth fit downstream.
    return np.maximum(mu / rho, 1.0e-7)


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
    min_dp: float = 1e-4,
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
        Latitude [°N]. Default: 0. N² is scaled by the latitude-dependent
        gravitational acceleration g(lat) inside ``gsw.Nsquared``; away from
        the equator this matters (≈0.9% bias at 70°N), so supply the cast
        latitude rather than relying on the equatorial default.
    min_dp : float
        Minimum adjacent pressure spacing [dbar]; pairs with smaller spacing
        yield NaN instead of a spurious ``inf`` from the dp=0 division.
        Default: 1e-4.

    Returns
    -------
    N2 : ndarray
        Buoyancy frequency squared [s⁻²], length len(P)-1.
    p_mid : ndarray
        Mid-point pressures [dbar], length len(P)-1.
    """
    import gsw

    # Defect (audit 103): lat=0 silently uses equatorial gravity, biasing N²
    # for non-equatorial casts. Warn once so the caller knows lat matters.
    if lat == 0:
        import warnings

        warnings.warn(
            "buoyancy_freq called with lat=0 (equatorial gravity); N² is "
            "gravity-scaled, so supply the cast latitude for non-equatorial "
            "data to avoid a systematic N² bias.",
            stacklevel=2,
        )

    SA = gsw.SA_from_SP(S, P, 0, 0)
    CT = gsw.CT_from_t(SA, T, P)
    N2, p_mid = gsw.Nsquared(SA, CT, P, lat)
    # Defect (audit 104): gsw.Nsquared divides by adjacent dp; duplicate or
    # non-monotonic pressures (common in binned/turn-around data) give a silent
    # inf. Replace degenerate-spacing entries with NaN, matching mixing.py.
    dp = np.diff(np.asarray(P, dtype=float))
    N2 = np.where(np.abs(dp) < min_dp, np.nan, N2)
    return N2, p_mid
