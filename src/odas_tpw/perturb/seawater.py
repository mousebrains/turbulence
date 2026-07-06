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

    # Fill masked cells with NaN (do NOT np.asarray a masked array — that drops
    # the mask and exposes raw _FillValue, e.g. -999, as if it were real T/C/P).
    def _filled(x):
        return np.ma.filled(np.ma.asarray(x).astype(np.float64), np.nan)

    T = _filled(T)
    C = _filled(C)
    P = _filled(P)
    lat = _filled(lat)
    lon = _filled(lon)

    # Replace NaN lat/lon with 0 for gsw (matches Matlab default)
    lat_safe = np.where(np.isnan(lat), 0.0, lat)
    lon_safe = np.where(np.isnan(lon), 0.0, lon)

    SP = gsw.SP_from_C(C, T, P)
    SA = gsw.SA_from_SP(SP, P, lon_safe, lat_safe)
    CT = gsw.CT_from_t(SA, T, P)
    sigma0 = gsw.sigma0(SA, CT)
    rho = gsw.rho(SA, CT, P) - 1000.0
    # gsw.z_from_p returns height Z (m, <=0 below the surface); positive-down
    # depth is -Z. Using -Z (not |Z|) keeps above-surface samples (P<0, e.g.
    # deck calibrations) as negative depths instead of folding them back down.
    z = gsw.z_from_p(P, lat_safe)
    depth = -z

    return {
        "SP": np.asarray(SP),
        "SA": np.asarray(SA),
        "CT": np.asarray(CT),
        "sigma0": np.asarray(sigma0),
        "rho": np.asarray(rho),
        "depth": np.asarray(depth),
    }
