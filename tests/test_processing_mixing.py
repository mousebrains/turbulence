"""Tests for derived mixing quantities (processing/mixing.py)."""

import gsw
import numpy as np
import pytest

from odas_tpw.processing.mixing import (
    DEFAULT_DTDZ_MIN,
    DEFAULT_N2_MIN,
    GAMMA_OSBORN,
    mixing_coefficients,
    pair_nearest,
    sorted_stratification,
    window_stratification,
)

# ---------------------------------------------------------------------------
# Synthetic profile: instrument falling at constant speed through linear
# stratification.
# ---------------------------------------------------------------------------

FS = 64.0  # slow-ish sampling rate [Hz]
W = 0.7  # fall speed [m/s] -> dbar/s (approximately)
T0 = 20.0  # surface temperature [degC]
DTDZ_TRUE = -0.05  # temperature decreases downward [K/m]
S_CONST = 35.0


def _make_profile(duration=120.0, p0=10.0):
    t = np.arange(0, duration, 1.0 / FS)
    P = p0 + W * t  # dbar
    depth = -gsw.z_from_p(P, 0.0)
    T = T0 + DTDZ_TRUE * depth
    return t, P, T


def _expected_n2(P, T):
    """Reference N² from gsw over the full profile span."""
    SA = gsw.SA_from_SP(np.full_like(P, S_CONST), P, 0.0, 0.0)
    CT = gsw.CT_from_t(SA, T, P)
    p_pair = np.array([P[0], P[-1]])
    sa_pair = np.array([SA[0], SA[-1]])
    ct_pair = np.array([CT[0], CT[-1]])
    n2, _ = gsw.Nsquared(sa_pair, ct_pair, p_pair, 0.0)
    return float(n2[0])


class TestSortedStratification:
    def test_matches_window_when_monotonic(self):
        # A monotonic, statically stable profile has no overturns, so the
        # Thorpe sort is the identity and dT/dz is unchanged; N² agrees to
        # within the mean-then-convert vs convert-then-mean difference.
        t, P, T = _make_profile()
        win_times = np.array([30.0, 60.0, 90.0])
        res_s = sorted_stratification(win_times, 4.0, t, P, T, S=S_CONST)
        res_w = window_stratification(win_times, 4.0, t, P, T, S=S_CONST)
        np.testing.assert_allclose(res_s.dTdz, res_w.dTdz, rtol=1e-7)
        np.testing.assert_allclose(res_s.N2, res_w.N2, rtol=0.02)

    def test_recovers_linear_gradient(self):
        t, P, T = _make_profile()
        res = sorted_stratification(np.array([60.0]), 8.0, t, P, T, S=S_CONST)
        np.testing.assert_allclose(res.dTdz, DTDZ_TRUE, rtol=1e-3)
        assert res.N2[0] > 0

    def test_sorting_restores_stability(self):
        # Build a statically unstable column: warm (light) water at depth.
        t = np.arange(0, 120.0, 1.0 / FS)
        P = 10.0 + W * t
        depth = -gsw.z_from_p(P, 0.0)
        T = T0 + 0.05 * depth  # temperature increases downward -> unstable
        res_w = window_stratification(np.array([60.0]), 8.0, t, P, T, S=S_CONST)
        res_s = sorted_stratification(np.array([60.0]), 8.0, t, P, T, S=S_CONST)
        # The raw profile is unstable; sorting restores a stable column.
        assert res_w.N2[0] < 0
        assert res_s.N2[0] > 0
        # ...and the stable temperature gradient is cooling-downward.
        assert res_w.dTdz[0] > 0
        assert res_s.dTdz[0] < 0


class TestWindowStratification:
    def test_recovers_linear_gradient(self):
        t, P, T = _make_profile()
        win_times = np.array([30.0, 60.0, 90.0])
        res = window_stratification(win_times, 4.0, t, P, T, S=S_CONST)
        np.testing.assert_allclose(res.dTdz, DTDZ_TRUE, rtol=1e-3)

    def test_n2_matches_gsw(self):
        t, P, T = _make_profile()
        win_times = np.array([60.0])
        res = window_stratification(win_times, 8.0, t, P, T, S=S_CONST)
        n2_ref = _expected_n2(P, T)
        # Window estimate vs whole-profile reference: linear profile, so
        # they should agree closely (compressibility variation is tiny).
        np.testing.assert_allclose(res.N2, n2_ref, rtol=0.05)
        assert res.N2[0] > 0  # cooling downward = stable

    def test_salinity_array_accepted(self):
        t, P, T = _make_profile()
        S = np.full_like(P, S_CONST)
        res = window_stratification(np.array([60.0]), 8.0, t, P, T, S=S)
        res_const = window_stratification(np.array([60.0]), 8.0, t, P, T, S=S_CONST)
        np.testing.assert_allclose(res.N2, res_const.N2, rtol=1e-10)

    def test_none_salinity_assumes_35(self):
        t, P, T = _make_profile()
        res = window_stratification(np.array([60.0]), 8.0, t, P, T, S=None)
        res35 = window_stratification(np.array([60.0]), 8.0, t, P, T, S=35.0)
        np.testing.assert_allclose(res.N2, res35.N2, rtol=1e-12)

    def test_unstable_stratification_negative_n2(self):
        t, P, _ = _make_profile()
        depth = -gsw.z_from_p(P, 0.0)
        T_unstable = 10.0 + 0.05 * depth  # warming downward = unstable
        res = window_stratification(np.array([60.0]), 8.0, t, P, T_unstable, S=S_CONST)
        assert res.N2[0] < 0

    def test_too_few_samples_nan(self):
        t, P, T = _make_profile(duration=1.0)
        # Window centred far outside the data
        res = window_stratification(np.array([500.0]), 2.0, t, P, T)
        assert np.isnan(res.N2[0]) and np.isnan(res.dTdz[0])

    def test_stalled_instrument_nan(self):
        # Constant pressure: no vertical span -> gradients unconstrained
        t = np.arange(0, 60, 1.0 / FS)
        P = np.full_like(t, 50.0)
        T = np.full_like(t, 10.0)
        res = window_stratification(np.array([30.0]), 8.0, t, P, T)
        assert np.isnan(res.N2[0]) and np.isnan(res.dTdz[0])

    def test_nan_samples_excluded(self):
        t, P, T = _make_profile()
        T = T.copy()
        T[::7] = np.nan
        res = window_stratification(np.array([60.0]), 8.0, t, P, T, S=S_CONST)
        np.testing.assert_allclose(res.dTdz, DTDZ_TRUE, rtol=1e-3)


class TestMixingCoefficients:
    def test_closed_form(self):
        eps = np.array([1e-8])
        chi = np.array([1e-8])
        N2 = np.array([1e-5])
        dTdz = np.array([0.05])
        res = mixing_coefficients(eps, chi, N2, dTdz)
        np.testing.assert_allclose(res.K_T, chi / (2 * dTdz**2))
        np.testing.assert_allclose(res.Gamma, N2 * chi / (2 * eps * dTdz**2))
        np.testing.assert_allclose(res.K_rho, GAMMA_OSBORN * eps / N2)

    def test_gamma_identity(self):
        """Gamma * epsilon / N2 == K_T by construction (Oakey)."""
        rng = np.random.default_rng(0)
        eps = 10.0 ** rng.uniform(-10, -6, 50)
        chi = 10.0 ** rng.uniform(-10, -7, 50)
        N2 = 10.0 ** rng.uniform(-6, -4, 50)
        dTdz = rng.uniform(0.01, 0.1, 50)
        res = mixing_coefficients(eps, chi, N2, dTdz)
        np.testing.assert_allclose(res.Gamma * eps / N2, res.K_T, rtol=1e-12)

    def test_weak_gradient_masked(self):
        res = mixing_coefficients(
            np.array([1e-8, 1e-8]),
            np.array([1e-8, 1e-8]),
            np.array([1e-5, 1e-5]),
            np.array([0.05, DEFAULT_DTDZ_MIN / 2]),
        )
        assert np.isfinite(res.K_T[0]) and np.isnan(res.K_T[1])
        assert np.isfinite(res.Gamma[0]) and np.isnan(res.Gamma[1])
        # K_rho does not depend on dT/dz
        assert np.isfinite(res.K_rho[1])

    def test_weak_stratification_masked(self):
        res = mixing_coefficients(
            np.array([1e-8, 1e-8, 1e-8]),
            np.array([1e-8, 1e-8, 1e-8]),
            np.array([1e-5, DEFAULT_N2_MIN / 2, -1e-6]),
            np.array([0.05, 0.05, 0.05]),
        )
        # weak and unstable N2 mask Gamma and K_rho
        assert np.isnan(res.Gamma[1]) and np.isnan(res.Gamma[2])
        assert np.isnan(res.K_rho[1]) and np.isnan(res.K_rho[2])
        # K_T does not depend on N2
        assert np.all(np.isfinite(res.K_T))

    def test_negative_gradient_same_as_positive(self):
        a = mixing_coefficients(
            np.array([1e-8]), np.array([1e-8]), np.array([1e-5]), np.array([0.05])
        )
        b = mixing_coefficients(
            np.array([1e-8]), np.array([1e-8]), np.array([1e-5]), np.array([-0.05])
        )
        np.testing.assert_allclose(a.K_T, b.K_T)
        np.testing.assert_allclose(a.Gamma, b.Gamma)

    def test_nan_inputs_propagate(self):
        res = mixing_coefficients(
            np.array([np.nan]), np.array([1e-8]), np.array([1e-5]), np.array([0.05])
        )
        assert np.isnan(res.Gamma[0]) and np.isnan(res.K_rho[0])
        assert np.isfinite(res.K_T[0])  # K_T needs only chi and dT/dz


class TestPairNearest:
    def test_identical_grids(self):
        t = np.arange(10.0)
        v = t * 2
        out = pair_nearest(t, v, t)
        np.testing.assert_allclose(out, v)

    def test_offset_grid_within_tolerance(self):
        t = np.arange(0, 10.0)
        v = t * 2
        out = pair_nearest(t, v, t + 0.4)
        np.testing.assert_allclose(out, v)

    def test_far_destination_rejected(self):
        t = np.arange(0, 10.0)
        v = t * 2
        out = pair_nearest(t, v, np.array([100.0]))
        assert np.isnan(out[0])

    def test_single_source(self):
        out = pair_nearest(np.array([5.0]), np.array([7.0]), np.array([5.1, 50.0]))
        assert out[0] == 7.0
        assert np.isnan(out[1])

    def test_empty_source(self):
        out = pair_nearest(np.array([]), np.array([]), np.array([1.0, 2.0]))
        assert np.all(np.isnan(out))

    def test_explicit_max_dt(self):
        t = np.array([0.0, 10.0])
        v = np.array([1.0, 2.0])
        out = pair_nearest(t, v, np.array([4.0]), max_dt=3.0)
        assert np.isnan(out[0])
        out = pair_nearest(t, v, np.array([4.0]), max_dt=5.0)
        assert out[0] == 1.0


class TestEndToEnd:
    def test_synthetic_profile_gamma(self):
        """Full chain: stratification + closed-form chi gives Gamma back."""
        t, P, T = _make_profile()
        win_times = np.arange(20.0, 100.0, 8.0)
        strat = window_stratification(win_times, 4.0, t, P, T, S=S_CONST)

        # Choose chi to give exactly Gamma = 0.2 for the known eps
        eps = np.full(len(win_times), 1e-8)
        gamma_target = 0.2
        chi = gamma_target * 2 * eps * strat.dTdz**2 / strat.N2

        res = mixing_coefficients(eps, chi, strat.N2, strat.dTdz)
        np.testing.assert_allclose(res.Gamma, gamma_target, rtol=1e-10)
        np.testing.assert_allclose(res.K_rho, res.K_T, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
