# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Nasmyth universal shear spectrum.

Uses Lueck's improved fit from ODAS v4.5.1, documented in
McMillan et al. (2016), J. Atmos. Oceanic Techno., 33, 817-837.
"""

import warnings

import numpy as np
import numpy.typing as npt

# Lueck (2022), J. Atmos. Oceanic Technol., 39, 1803-1816,
# https://doi.org/10.1175/JTECH-D-21-0051.1
LUECK_A = 1.0774e9  # e/e_10 model constant
X_95 = 0.1205  # non-dimensional wavenumber for 95% variance


def nasmyth(epsilon: float, nu: float, k: npt.ArrayLike) -> np.ndarray:
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
    if nu <= 0:
        raise ValueError(f"Kinematic viscosity must be positive, got nu={nu}")
    k = np.asarray(k, dtype=np.float64)
    if epsilon <= 0:
        warnings.warn(
            f"epsilon={epsilon:.2e} <= 0; Nasmyth spectrum will contain NaN",
            stacklevel=2,
        )
        return np.full_like(k, np.nan)
    ks = (epsilon / nu**3) ** 0.25  # Kolmogorov wavenumber
    x = k / ks
    G2 = _nasmyth_g2(x)
    return epsilon ** (3 / 4) * nu ** (-1 / 4) * G2


def nasmyth_nondim(x: npt.ArrayLike) -> np.ndarray:
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


def _nasmyth_g2(x: np.ndarray) -> np.ndarray:
    """Lueck's improved fit to the Nasmyth spectrum.

    G2 = 8.05 * x^(1/3) / (1 + (20.6*x)^3.715)
    """
    return 8.05 * x ** (1 / 3) / (1 + (20.6 * x) ** 3.715)


# ---------------------------------------------------------------------------
# Grid-based fast lookup for G2(x)
# ---------------------------------------------------------------------------


class NasmythGrid:
    """Pre-tabulated Nasmyth G2(x) on a log-spaced grid for fast interpolation.

    The non-dimensional Nasmyth spectrum G2(x) depends only on x = k/ks.
    By tabulating log10(G2) vs log10(x) once, subsequent evaluations reduce
    to ``np.interp`` in log-space -- avoiding repeated power operations.

    Parameters
    ----------
    n : int, optional
        Number of grid points (default 2000).
    x_min : float, optional
        Minimum x value (default 1e-6).
    x_max : float, optional
        Maximum x value (default 1.0).
    """

    def __init__(self, n: int = 2000, x_min: float = 1e-6, x_max: float = 1.0) -> None:
        self._log10_x = np.linspace(np.log10(x_min), np.log10(x_max), n)
        x_grid = 10.0**self._log10_x
        g2_grid = _nasmyth_g2(x_grid)
        self._log10_g2 = np.log10(g2_grid)
        # Cache boundary values for extrapolation fallback
        self._x_min = x_min
        self._x_max = x_max

    def interp_g2(self, x: np.ndarray) -> np.ndarray:
        """Interpolate G2 for arbitrary x values.

        For x inside [x_min, x_max], log-space interpolation is used.
        For x outside that range, G2 is computed directly via ``_nasmyth_g2``.

        Parameters
        ----------
        x : ndarray
            Non-dimensional wavenumber(s) k/ks.

        Returns
        -------
        G2 : ndarray
            Non-dimensional spectrum values.
        """
        x = np.asarray(x, dtype=np.float64)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)

        result = np.empty_like(x)

        in_range = (x >= self._x_min) & (x <= self._x_max)
        out_range = ~in_range

        if np.any(in_range):
            log10_x_query = np.log10(x[in_range])
            log10_g2 = np.interp(log10_x_query, self._log10_x, self._log10_g2)
            result[in_range] = 10.0**log10_g2

        if np.any(out_range):
            result[out_range] = _nasmyth_g2(x[out_range])

        return result.item() if scalar else result


# Module-level singleton, lazily initialized on first call to nasmyth_grid().
_grid: NasmythGrid | None = None


def _get_grid() -> NasmythGrid:
    """Return the module-level NasmythGrid, creating it on first access."""
    global _grid
    if _grid is None:
        _grid = NasmythGrid()
    return _grid


def nasmyth_grid(epsilon: float, nu: float, k: npt.ArrayLike) -> np.ndarray:
    """Nasmyth shear spectrum using pre-tabulated G2 interpolation.

    Drop-in replacement for :func:`nasmyth` that avoids repeated power
    operations by interpolating on a pre-computed log-spaced grid of the
    non-dimensional spectrum G2(x).

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
    if nu <= 0:
        raise ValueError(f"Kinematic viscosity must be positive, got nu={nu}")
    k = np.asarray(k, dtype=np.float64)
    if epsilon <= 0:
        warnings.warn(
            f"epsilon={epsilon:.2e} <= 0; Nasmyth spectrum will contain NaN",
            stacklevel=2,
        )
        return np.full_like(k, np.nan)
    ks = (epsilon / nu**3) ** 0.25  # Kolmogorov wavenumber
    x = k / ks
    G2 = _get_grid().interp_g2(x)
    return epsilon ** (3 / 4) * nu ** (-1 / 4) * G2
