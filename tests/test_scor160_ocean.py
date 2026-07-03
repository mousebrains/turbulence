# Tests for odas_tpw.scor160.ocean
"""Unit tests for seawater property functions."""

import numpy as np
import pytest

from odas_tpw.scor160.ocean import buoyancy_freq, density, kappa_T, visc, visc35


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

    def test_high_temperature_stays_positive(self):
        """The cubic fit goes negative above ~64.5 C; a spurious-hot sample
        must not yield a negative viscosity (which crashes nasmyth) (#39)."""
        assert float(visc35(70.0)) > 0.0
        assert float(visc35(100.0)) > 0.0
        # Valid-range values are untouched by the floor.
        assert float(visc35(10.0)) == pytest.approx(1.35e-6, rel=0.05)


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

    def test_lon_lat_thread_into_absolute_salinity(self):
        """lon/lat feed the TEOS-10 composition anomaly (2026-07-03 review Gem-A).

        Default (0,0) is unchanged; a real position shifts density slightly
        (a few 1e-5, so this is correctness hygiene, not a material change).
        """
        base = density(15.0, 34.5, 100.0)
        assert density(15.0, 34.5, 100.0, lon=0, lat=0) == base  # default unchanged
        deep_npac = density(15.0, 34.5, 100.0, lon=145.7, lat=15.2)  # Saipan
        assert deep_npac != base                                    # anomaly applied
        assert abs(deep_npac / base - 1.0) < 1e-4                    # but tiny

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


class TestKappaT:
    """Tests for kappa_T(T, S, P) — molecular thermal diffusivity for chi."""

    def test_reference_values(self):
        # Sharqawy/Jamieson-Tudhope + gsw: ~1.45e-7 at 15 C, ~1.498e-7 at 28 C.
        assert float(kappa_T(15.0, 35.0, 0.0)) == pytest.approx(1.45e-7, rel=0.01)
        assert float(kappa_T(28.0, 35.0, 0.0)) == pytest.approx(1.498e-7, rel=0.01)

    def test_increases_with_temperature(self):
        # Thermal conductivity (hence diffusivity) rises with T over the ocean
        # range; span the full -1..32 C these casts cover.
        vals = np.asarray(kappa_T(np.array([-1.0, 0.0, 15.0, 28.0, 32.0])))
        assert np.all(np.diff(vals) > 0)

    def test_physical_magnitude(self):
        # All values sit in the accepted seawater band ~1.3-1.6e-7 m^2/s.
        v = np.asarray(kappa_T(np.linspace(-1.0, 32.0, 12)))
        assert np.all((v > 1.3e-7) & (v < 1.6e-7))

    def test_spurious_temperature_clipped_positive(self):
        # A garbage window-mean T must never produce a negative diffusivity
        # (gsw cp/rho extrapolate negative far out of range).
        for T in (200.0, -50.0, 1e4):
            assert float(kappa_T(T)) > 0

    def test_array_shape_preserved(self):
        out = kappa_T(np.full((2, 3), 20.0))
        assert np.asarray(out).shape == (2, 3)


class TestBuoyancyFreq:
    """Tests for buoyancy_freq(T, S, P)."""

    def test_stable_stratification(self):
        """Warm-over-cold stable profile should give positive N²."""
        T = np.array([20.0, 15.0, 10.0, 5.0])
        S = np.full(4, 35.0)
        P = np.array([0.0, 50.0, 100.0, 150.0])
        N2, p_mid = buoyancy_freq(T, S, P, lat=15.0)
        assert len(N2) == 3
        assert len(p_mid) == 3
        assert np.all(N2 > 0)

    def test_output_lengths(self):
        """Output length should be len(P) - 1."""
        n = 10
        T = np.linspace(20, 5, n)
        S = np.full(n, 35.0)
        P = np.linspace(0, 200, n)
        N2, p_mid = buoyancy_freq(T, S, P, lat=15.0)
        assert N2.shape == (n - 1,)
        assert p_mid.shape == (n - 1,)

    def test_mid_pressures(self):
        """Mid-point pressures should be between adjacent input pressures."""
        P = np.array([0.0, 100.0, 200.0])
        T = np.array([20.0, 15.0, 10.0])
        S = np.full(3, 35.0)
        _, p_mid = buoyancy_freq(T, S, P, lat=15.0)
        assert np.all(p_mid > P[:-1])
        assert np.all(p_mid < P[1:])

    def test_lat_zero_warns(self):
        """Defect (audit 103): lat=0 uses equatorial gravity and biases N²;
        the convenience API must warn so the caller supplies the cast lat."""
        T = np.array([20.0, 15.0, 10.0])
        S = np.full(3, 35.0)
        P = np.array([0.0, 50.0, 100.0])
        with pytest.warns(UserWarning, match="equatorial gravity"):
            buoyancy_freq(T, S, P)  # lat defaults to 0

    def test_lat_supplied_no_warn(self):
        """Supplying a non-zero latitude must not warn."""
        T = np.array([20.0, 15.0, 10.0])
        S = np.full(3, 35.0)
        P = np.array([0.0, 50.0, 100.0])
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes an error
            buoyancy_freq(T, S, P, lat=15.0)

    def test_lat_scales_n2(self):
        """N² is gravity-scaled by g(lat), so high-latitude N² exceeds the
        equatorial value for the same profile (audit 103 evidence)."""
        T = np.array([20.0, 15.0, 10.0])
        S = np.full(3, 35.0)
        P = np.array([0.0, 50.0, 100.0])
        n2_eq, _ = buoyancy_freq(T, S, P, lat=0.0)
        n2_hi, _ = buoyancy_freq(T, S, P, lat=70.0)
        assert np.all(n2_hi > n2_eq)

    def test_duplicate_pressure_no_inf(self):
        """Defect (audit 104): duplicate pressure (dp=0) made gsw.Nsquared
        emit a silent inf; the dp guard must yield NaN instead, with no inf."""
        T = np.array([20.0, 15.0, 12.0, 10.0])
        S = np.full(4, 35.0)
        P = np.array([0.0, 50.0, 50.0, 100.0])  # duplicate pressure at index 2
        # gsw divides by dp=0 internally (RuntimeWarning) before our NaN guard.
        with np.errstate(divide="ignore", invalid="ignore"):
            N2, _ = buoyancy_freq(T, S, P, lat=15.0)
        assert not np.any(np.isinf(N2))
        assert np.isnan(N2[1])  # the degenerate-spacing pair
        # The well-spaced pairs are unaffected.
        assert np.isfinite(N2[0])
        assert np.isfinite(N2[2])
