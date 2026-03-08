"""Seawater properties for turbulence calculations."""

import numpy as np


def visc35(T):
    """Kinematic viscosity of seawater at S=35, P=0 [m²/s].

    Parameters
    ----------
    T : float or array_like
        Temperature in °C. Valid for 0–20 °C.

    Returns
    -------
    nu : float or ndarray
        Kinematic viscosity in m²/s.

    Reference: ODAS visc35.m — 3rd-order polynomial fit,
    error < 1% for 0 ≤ T ≤ 20 °C, 30 ≤ S ≤ 40 PSU, atmospheric pressure.
    """
    pol = np.array([
        -1.131311019739306e-011,
         1.199552027472192e-009,
        -5.864346822839289e-008,
         1.828297985908266e-006,
    ])
    return np.polyval(pol, T)
