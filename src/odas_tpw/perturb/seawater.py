# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Seawater properties via TEOS-10 (gsw).

Reference: Code/add_seawater_properties.m
"""

import numpy as np
import numpy.typing as npt


def add_seawater_properties(
    T: npt.ArrayLike,
    C: npt.ArrayLike,
    P: npt.ArrayLike,
    lat: npt.ArrayLike,
    lon: npt.ArrayLike,
) -> dict[str, np.ndarray]:
    """Compute seawater properties from T, C, P, lat, lon.

    Parameters
    ----------
    T : array_like
        In-situ temperature [degC] (from slow CT sensor, e.g. JAC_T).
    C : array_like
        Conductivity [mS/cm].
    P : array_like
        Pressure [dbar].
    lat, lon : array_like
        Latitude and longitude [degrees].

    Returns
    -------
    dict with keys: SP, SA, CT, sigma0, rho, depth
        SP = practical salinity [PSU]
        SA = absolute salinity [g/kg]
        CT = conservative temperature [degC]
        sigma0 = potential density anomaly [kg/m^3]
        rho = in-situ density - 1000 [kg/m^3]
        depth = depth (positive downward) [m]
    """
    import gsw

    T = np.asarray(T, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    # Replace NaN lat/lon with 0 for gsw (matches Matlab default)
    lat_safe = np.where(np.isnan(lat), 0.0, lat)
    lon_safe = np.where(np.isnan(lon), 0.0, lon)

    SP = gsw.SP_from_C(C, T, P)
    SA = gsw.SA_from_SP(SP, P, lon_safe, lat_safe)
    CT = gsw.CT_from_t(SA, T, P)
    sigma0 = gsw.sigma0(SA, CT)
    rho = gsw.rho(SA, CT, P) - 1000.0
    z = gsw.z_from_p(P, lat_safe)
    depth = np.abs(z)

    return {
        "SP": np.asarray(SP),
        "SA": np.asarray(SA),
        "CT": np.asarray(CT),
        "sigma0": np.asarray(sigma0),
        "rho": np.asarray(rho),
        "depth": np.asarray(depth),
    }
