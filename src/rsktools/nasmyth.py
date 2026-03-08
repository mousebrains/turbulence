"""Nasmyth universal shear spectrum.

Uses Lueck's improved fit from ODAS v4.5.1, documented in
McMillan et al. (2016), J. Atmos. Oceanic Techno., 33, 817–837.
"""

import numpy as np

# Constants from ODAS get_diss_odas.m
LUECK_A = 1.0774e9   # e/e_10 model constant
X_95 = 0.1205        # non-dimensional wavenumber for 95% variance


def nasmyth(epsilon, nu, k):
    """Nasmyth shear spectrum at given wavenumbers.

    Parameters
    ----------
    epsilon : float
        Dissipation rate [W/kg].
    nu : float
        Kinematic viscosity [m²/s].
    k : array_like
        Wavenumber(s) [cpm].

    Returns
    -------
    phi : ndarray
        Shear spectrum [s⁻² cpm⁻¹].
    """
    k = np.asarray(k, dtype=np.float64)
    ks = (epsilon / nu**3) ** 0.25  # Kolmogorov wavenumber
    x = k / ks
    G2 = _nasmyth_g2(x)
    return epsilon ** (3 / 4) * nu ** (-1 / 4) * G2


def nasmyth_nondim(x):
    """Non-dimensional Nasmyth spectrum G2(x) where x = k/ks.

    Parameters
    ----------
    x : array_like
        Non-dimensional wavenumber k/ks.

    Returns
    -------
    G2 : ndarray
        Non-dimensional spectrum.
    """
    return _nasmyth_g2(np.asarray(x, dtype=np.float64))


def _nasmyth_g2(x):
    """Lueck's improved fit to the Nasmyth spectrum.

    G2 = 8.05 * x^(1/3) / (1 + (20.6*x)^3.715)
    """
    return 8.05 * x ** (1 / 3) / (1 + (20.6 * x) ** 3.715)
