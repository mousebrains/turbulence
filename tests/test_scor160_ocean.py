# Tests for odas_tpw.scor160.ocean
"""Unit tests for seawater property functions."""

import numpy as np
import pytest

from odas_tpw.scor160.ocean import buoyancy_freq, density, visc, visc35


class TestVisc35:
    """Tests for visc35(T) — kinematic viscosity at S=35, P=0."""

    def test_known_value_0C(self):
        """At 0 °C, viscosity ~1.83e-6 m²/s (seawater at S=35)."""
        nu = visc35(0.0)
        assert 1.7e-6 < nu < 1.9e-6

    def test_known_value_10C(self):
        """At 10 °C, viscosity ~1.35e-6 m²/s."""
        nu = visc35(10.0)
        assert 1.2e-6 < nu < 1.5e-6

    def test_known_value_20C(self):
        """At 20 °C, viscosity ~1.05e-6 m²/s."""
        nu = visc35(20.0)
        assert 0.9e-6 < nu < 1.2e-6

    def test_monotonic_decrease(self):
        """Viscosity should decrease with increasing temperature."""
        T = np.arange(0, 21, 1.0)
        nu = visc35(T)
        assert np.all(np.diff(nu) < 0)

    def test_scalar_input(self):
        nu = visc35(10.0)
        assert isinstance(nu, (float, np.floating))

    def test_array_input(self):
        T = np.array([5.0, 10.0, 15.0])
        nu = visc35(T)
        assert nu.shape == (3,)


class TestVisc:
    """Tests for visc(T, S, P) — general kinematic viscosity."""

    def test_default_s35_p0_matches_visc35(self):
        """With defaults (S=35, P=0), visc() should match visc35()."""
        T = np.array([5.0, 10.0, 15.0])
        np.testing.assert_allclose(visc(T), visc35(T))

    def test_scalar_s35_p0(self):
        np.testing.assert_allclose(visc(10.0, 35, 0), visc35(10.0))

    def test_nondefault_salinity(self):
        """visc(T, S=20, P=0) should differ from visc35."""
        nu_20 = visc(10.0, S=20, P=0)
        nu_35 = visc35(10.0)
        assert nu_20 != pytest.approx(nu_35, rel=0.01)

    def test_pressure_effect(self):
        """Higher pressure should slightly change viscosity."""
        nu_0 = visc(10.0, S=35, P=0)
        nu_1000 = visc(10.0, S=35, P=1000)
        # Pressure effect is small but nonzero
        assert nu_1000 != pytest.approx(nu_0, rel=0.001)

    def test_positive_values(self):
        nu = visc(10.0, 35, 500)
        assert nu > 0


class TestDensity:
    """Tests for density(T, S, P) — in-situ density via TEOS-10."""

    def test_approximate_seawater(self):
        """Density of seawater at T=10, S=35, P=0 ~ 1027 kg/m³."""
        rho = density(10.0, 35.0, 0.0)
        assert 1020 < rho < 1030

    def test_fresh_water(self):
        """Fresh water (S=0, P=0) at 4°C ~ 1000 kg/m³."""
        rho = density(4.0, 0.0, 0.0)
        assert 998 < rho < 1002

    def test_density_increases_with_pressure(self):
        rho_0 = density(10.0, 35.0, 0.0)
        rho_4000 = density(10.0, 35.0, 4000.0)
        assert rho_4000 > rho_0

    def test_density_increases_with_salinity(self):
        rho_30 = density(10.0, 30.0, 0.0)
        rho_38 = density(10.0, 38.0, 0.0)
        assert rho_38 > rho_30

    def test_array_input(self):
        T = np.array([5.0, 10.0, 15.0])
        S = np.array([34.0, 35.0, 36.0])
        P = np.array([0.0, 100.0, 200.0])
        rho = density(T, S, P)
        assert rho.shape == (3,)
        assert np.all(rho > 1000)


class TestBuoyancyFreq:
    """Tests for buoyancy_freq(T, S, P)."""

    def test_stable_stratification(self):
        """Warm-over-cold stable profile should give positive N²."""
        T = np.array([20.0, 15.0, 10.0, 5.0])
        S = np.full(4, 35.0)
        P = np.array([0.0, 50.0, 100.0, 150.0])
        N2, p_mid = buoyancy_freq(T, S, P)
        assert len(N2) == 3
        assert len(p_mid) == 3
        assert np.all(N2 > 0)

    def test_output_lengths(self):
        """Output length should be len(P) - 1."""
        n = 10
        T = np.linspace(20, 5, n)
        S = np.full(n, 35.0)
        P = np.linspace(0, 200, n)
        N2, p_mid = buoyancy_freq(T, S, P)
        assert N2.shape == (n - 1,)
        assert p_mid.shape == (n - 1,)

    def test_mid_pressures(self):
        """Mid-point pressures should be between adjacent input pressures."""
        P = np.array([0.0, 100.0, 200.0])
        T = np.array([20.0, 15.0, 10.0])
        S = np.full(3, 35.0)
        _, p_mid = buoyancy_freq(T, S, P)
        assert np.all(p_mid > P[:-1])
        assert np.all(p_mid < P[1:])
