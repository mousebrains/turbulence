# Tests for odas_tpw.scor160.nasmyth
"""Unit tests for Nasmyth universal shear spectrum."""

import warnings

import numpy as np
import pytest

from odas_tpw.scor160.nasmyth import (
    LUECK_A,
    X_95,
    NasmythGrid,
    nasmyth,
    nasmyth_grid,
    nasmyth_nondim,
)


class TestNasmythNondim:
    """Tests for the non-dimensional G2(x) function."""

    def test_positive_for_positive_x(self):
        x = np.logspace(-4, -0.5, 100)
        G2 = nasmyth_nondim(x)
        assert np.all(G2 > 0)

    def test_formula_values(self):
        """Spot-check the Lueck formula: G2 = 8.05*x^(1/3) / (1+(20.6*x)^3.715)."""
        x = np.array([0.001, 0.01, 0.1])
        expected = 8.05 * x ** (1 / 3) / (1 + (20.6 * x) ** 3.715)
        np.testing.assert_allclose(nasmyth_nondim(x), expected)

    def test_rises_then_falls(self):
        """G2 should peak in the inertial subrange then decay."""
        x = np.logspace(-5, 0, 1000)
        G2 = nasmyth_nondim(x)
        peak_idx = np.argmax(G2)
        # Peak should be somewhere in the middle
        assert 50 < peak_idx < 950

    def test_scalar_input(self):
        G2 = nasmyth_nondim(0.01)
        assert G2 > 0


class TestNasmyth:
    """Tests for nasmyth(epsilon, nu, k)."""

    def test_dimensional_output(self):
        """Spectrum should be positive for physical parameters."""
        k = np.arange(1, 100, 1.0)
        phi = nasmyth(1e-8, 1.3e-6, k)
        assert np.all(phi > 0)

    def test_negative_epsilon_warns_nan(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            phi = nasmyth(-1e-8, 1.3e-6, np.array([1.0, 10.0]))
            assert len(w) == 1
            assert np.all(np.isnan(phi))

    def test_zero_epsilon_warns_nan(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            phi = nasmyth(0, 1.3e-6, np.array([1.0]))
            assert len(w) == 1
            assert np.all(np.isnan(phi))

    def test_negative_nu_raises(self):
        with pytest.raises(ValueError, match="positive"):
            nasmyth(1e-8, -1e-6, np.array([1.0]))

    def test_zero_nu_raises(self):
        with pytest.raises(ValueError, match="positive"):
            nasmyth(1e-8, 0, np.array([1.0]))

    def test_scaling_with_epsilon(self):
        """Higher epsilon → higher spectrum amplitude."""
        k = np.arange(1, 50)
        phi_lo = nasmyth(1e-10, 1.3e-6, k)
        phi_hi = nasmyth(1e-6, 1.3e-6, k)
        assert np.all(phi_hi > phi_lo)

    def test_integral_variance(self):
        """Integral of Nasmyth over all k gives isotropic variance ~ eps/(7.5*nu)."""
        eps = 1e-7
        nu = 1.3e-6
        ks = (eps / nu**3) ** 0.25
        k = np.linspace(0.1, ks, 5000)
        phi = nasmyth(eps, nu, k)
        variance = np.trapezoid(phi, k)
        expected = eps / (7.5 * nu)
        # Allow some tolerance since we can't integrate to infinity
        assert variance == pytest.approx(expected, rel=0.15)


class TestNasmythGrid:
    """Tests for the pre-tabulated NasmythGrid interpolation."""

    def test_matches_nasmyth_nondim(self):
        """Grid-interpolated G2 should closely match the analytic formula."""
        grid = NasmythGrid(n=5000)
        x = np.logspace(-5, -0.1, 200)
        G2_direct = nasmyth_nondim(x)
        G2_interp = grid.interp_g2(x)
        np.testing.assert_allclose(G2_interp, G2_direct, rtol=1e-3)

    def test_out_of_range_fallback(self):
        """Values outside grid range should fall back to direct computation."""
        grid = NasmythGrid(x_min=1e-4, x_max=0.5)
        x_out = np.array([1e-6, 0.9])
        G2 = grid.interp_g2(x_out)
        expected = nasmyth_nondim(x_out)
        np.testing.assert_allclose(G2, expected)

    def test_scalar_input(self):
        grid = NasmythGrid()
        result = grid.interp_g2(np.float64(0.01))
        assert isinstance(result, (float, np.floating))


class TestNasmythGridFunction:
    """Tests for nasmyth_grid() — drop-in replacement for nasmyth()."""

    def test_matches_nasmyth(self):
        """nasmyth_grid should closely match nasmyth()."""
        k = np.arange(1, 100, 1.0)
        eps, nu = 1e-8, 1.3e-6
        phi_direct = nasmyth(eps, nu, k)
        phi_grid = nasmyth_grid(eps, nu, k)
        np.testing.assert_allclose(phi_grid, phi_direct, rtol=1e-2)

    def test_negative_epsilon_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            phi = nasmyth_grid(-1e-8, 1.3e-6, np.array([1.0]))
            assert len(w) == 1
            assert np.all(np.isnan(phi))

    def test_negative_nu_raises(self):
        with pytest.raises(ValueError, match="positive"):
            nasmyth_grid(1e-8, -1e-6, np.array([1.0]))


class TestConstants:
    """Verify critical constants."""

    def test_lueck_a(self):
        assert pytest.approx(1.0774e9) == LUECK_A

    def test_x_95(self):
        assert pytest.approx(0.1205) == X_95
