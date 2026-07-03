# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Batchelor and Kraichnan temperature gradient spectra.

References
----------
Dillon, T.M. and D.R. Caldwell, 1980: The Batchelor spectrum and dissipation
    in the upper ocean. J. Geophys. Res., 85, 1910-1916.
Bogucki, D., J.A. Domaradzki, and P.K. Yeung, 1997: Direct numerical
    simulations of passive scalars with Pr > 1 advected by turbulent flow.
    J. Fluid Mech., 343, 111-130.
Oakey, N.S., 1982: Determination of the rate of dissipation of turbulent
    energy. J. Phys. Oceanogr., 12, 256-271.
"""

import numpy as np
import numpy.typing as npt
from scipy.special import erfc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Thermal diffusivity of seawater [m^2/s], held FIXED at a ~15 degC value.
# NOTE (audit 2026-07-01): kappa_T is temperature-dependent (kappa_T = k/(rho*cp),
# ~1.40e-7 at 15 degC rising to ~1.50e-7 at 28 degC, +7%). Because chi = 6*kappa_T*I
# and epsilon_T = (2*pi*kB)^4*nu*kappa_T^2, the fixed value biases chi LOW by ~6.5%
# and Method-2 epsilon_T LOW by ~12.6% in ARCTERX warm (~28 degC) surface water,
# while nu in the same pipeline IS temperature-resolved. Threading a per-window
# kappa_T(T_mean, S, P) through the ~13 KAPPA_T sites in chi.py/l4_chi.py is the
# correct fix; it changes every published chi and is deferred pending review.
# See docs/chi_mathematics.md Section 11 for the bias envelope.
KAPPA_T = 1.4e-7
Q_BATCHELOR = 3.7  # Batchelor universal constant (Oakey 1982)
Q_KRAICHNAN = 5.26  # Kraichnan universal constant (Bogucki et al. 1997)
_SQ6Q_KRAICHNAN = np.sqrt(6 * Q_KRAICHNAN)  # Precomputed for kraichnan_grad


# ---------------------------------------------------------------------------
# Batchelor wavenumber
# ---------------------------------------------------------------------------


def batchelor_kB(
    epsilon: float | np.ndarray, nu: float, kappa_T: float = KAPPA_T
) -> float | np.ndarray:
    """Batchelor wavenumber [cpm].

    kB = (1/(2*pi)) * (epsilon / (nu * kappa_T^2))^(1/4)

    Parameters
    ----------
    epsilon : float or array
        TKE dissipation rate [W/kg].
    nu : float
        Kinematic viscosity [m^2/s].
    kappa_T : float
        Thermal diffusivity [m^2/s].

    Returns
    -------
    kB : float or array
        Batchelor wavenumber [cpm].
    """
    return (1 / (2 * np.pi)) * (epsilon / (nu * kappa_T**2)) ** 0.25


# ---------------------------------------------------------------------------
# Batchelor spectrum (Dillon & Caldwell 1980)
# ---------------------------------------------------------------------------


def batchelor_nondim(alpha: npt.ArrayLike) -> np.ndarray:
    """Non-dimensional Batchelor gradient spectrum f_B(alpha).

    f(alpha) = alpha * [exp(-alpha^2/2) - alpha * sqrt(pi/2) * erfc(alpha/sqrt(2))]

    Dillon & Caldwell 1980, eq. 3.  Integral over [0, inf) = 1/3.

    Parameters
    ----------
    alpha : array_like
        Non-dimensional wavenumber: alpha = sqrt(2*q) * k / kB.

    Returns
    -------
    f : ndarray
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    return alpha * (np.exp(-(alpha**2) / 2) - alpha * np.sqrt(np.pi / 2) * erfc(alpha / np.sqrt(2)))


def batchelor_grad(
    k: npt.ArrayLike,
    kB: float | np.ndarray,
    chi: float,
    kappa_T: float = KAPPA_T,
    q: float = Q_BATCHELOR,
) -> np.ndarray:
    """Batchelor temperature gradient spectrum [(K/m)^2 / cpm].

    S(k) = sqrt(q/2) * chi / (kB * kappa_T) * f(alpha)
    where alpha = sqrt(2*q) * k / kB

    Dillon & Caldwell 1980, eq. 1.  k and kB in cpm.
    Integrates to chi / (6 * kappa_T).

    Parameters
    ----------
    k : array_like
        Wavenumber [cpm].
    kB : float
        Batchelor wavenumber [cpm].
    chi : float
        Thermal variance dissipation rate [K^2/s].
    kappa_T : float
        Thermal diffusivity [m^2/s].
    q : float
        Universal constant.

    Returns
    -------
    S : ndarray
        Gradient spectrum [(K/m)^2 / cpm].
    """
    k = np.asarray(k, dtype=np.float64)
    # Mirror kraichnan_grad: a degenerate kB (0 / NaN) otherwise yields
    # NaN-with-warnings; sanitize to 0 so the spectrum stays finite (#16).
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.sqrt(2 * q) * k / kB
        f = batchelor_nondim(alpha)
        S = np.sqrt(q / 2) * chi / (kB * kappa_T) * f
    return np.where(np.isfinite(S), S, 0.0)


# ---------------------------------------------------------------------------
# Kraichnan spectrum (Bogucki et al. 1997)
# ---------------------------------------------------------------------------


def kraichnan_grad(
    k: npt.ArrayLike,
    kB: float | np.ndarray,
    chi: float,
    kappa_T: float = KAPPA_T,
    q: float = Q_KRAICHNAN,
) -> np.ndarray:
    """Kraichnan temperature gradient spectrum [(K/m)^2 / cpm].

    S(k) = chi*q / (kappa_T*kB^2) * k * exp(-sqrt(6q)*y)
    where y = k / kB.

    This is the one-dimensional (along-path) gradient spectrum obtained by
    applying the isotropic transform G1(k1) = k1^2 * int_{k1}^inf E(k)/k dk
    to the three-dimensional Kraichnan scalar spectrum of Bogucki et al.
    1997, eq. 11.  The transform evaluates in closed form to the simple
    exponential above (the same form as Peterson & Fer 2014, eq. 8).
    Integrates to chi / (6 * kappa_T); peaks at k = kB / sqrt(6q).

    Key difference from Batchelor: exponential rolloff instead of Gaussian,
    which fits DNS and observed high-Prandtl-number spectra better
    (Bogucki et al. 1997).

    Parameters
    ----------
    k : array_like
        Wavenumber [cpm].
    kB : float
        Batchelor wavenumber [cpm].
    chi : float
        Thermal variance dissipation rate [K^2/s].
    kappa_T : float
        Thermal diffusivity [m^2/s].
    q : float
        Universal constant (default: 5.26).

    Returns
    -------
    S : ndarray
        Gradient spectrum [(K/m)^2 / cpm].
    """
    k = np.asarray(k, dtype=np.float64)
    y = k / kB
    sq6q = _SQ6Q_KRAICHNAN if q == Q_KRAICHNAN else np.sqrt(6 * q)
    sq6q_y = sq6q * y
    with np.errstate(divide="ignore", invalid="ignore"):
        S = chi * q / (kappa_T * kB**2) * k * np.exp(-sq6q_y)
    S = np.where(np.isfinite(S), S, 0.0)
    return S
