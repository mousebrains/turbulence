# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for Thorpe-scale overturn analysis (processing/thorpe.py)."""

from __future__ import annotations

import gsw
import numpy as np
import pytest

from odas_tpw.processing.thorpe import (
    EDGE_TRUNCATION_FRACTION,
    cox_number,
    ozmidov,
    patch_n2,
    r_ot,
    reynolds_buoyancy,
    thorpe_displacements,
    thorpe_stats,
    window_thorpe,
)

RNG = np.random.default_rng(20260708)


# --------------------------------------------------------------------------- #
# thorpe_displacements
# --------------------------------------------------------------------------- #


def test_monotonic_profile_has_zero_displacements():
    z = np.linspace(10.0, 12.0, 21)
    sigma = 25.0 + 0.01 * z  # densest deepest: already stable
    disp = thorpe_displacements(z, sigma, increasing_down=True)
    np.testing.assert_array_equal(disp.delta, 0.0)
    np.testing.assert_array_equal(disp.x_sorted, disp.x)


def test_monotonic_temperature_profile_stable_when_decreasing_down():
    z = np.linspace(10.0, 12.0, 21)
    T = 20.0 - 0.05 * z  # warmest shallowest: stable for temperature
    disp = thorpe_displacements(z, T, increasing_down=False)
    np.testing.assert_array_equal(disp.delta, 0.0)


def test_fully_flipped_segment_exact_displacements():
    # A linear profile observed fully inverted: sample at depth z_j holds the
    # value that belongs at depth z_{n-1-j}, so delta_j = z_j - z_{n-1-j}.
    z = np.linspace(0.0, 1.0, 11)
    G = 0.02  # dsigma/dz > 0 (stable orientation)
    sigma_stable = 25.0 + G * z
    sigma_obs = sigma_stable[::-1]  # overturned
    disp = thorpe_displacements(z, sigma_obs, increasing_down=True)
    np.testing.assert_allclose(disp.delta, z - z[::-1])
    # sorted profile recovers the stable one
    np.testing.assert_allclose(disp.x_sorted, sigma_stable)


def test_single_displaced_parcel():
    # Swap two parcels 0.5 m apart in an otherwise stable profile.
    z = np.arange(0.0, 1.1, 0.1)
    sigma = 25.0 + 0.1 * z
    sigma_obs = sigma.copy()
    sigma_obs[2], sigma_obs[7] = sigma[7], sigma[2]
    disp = thorpe_displacements(z, sigma_obs, increasing_down=True)
    expect = np.zeros_like(z)
    expect[2] = z[2] - z[7]  # heavy parcel observed at 0.2 m belongs at 0.7 m
    expect[7] = z[7] - z[2]
    np.testing.assert_allclose(disp.delta, expect, atol=1e-12)


def test_mean_of_sorted_minus_observed_is_exactly_zero():
    # Regression guard for the patch-N2 blocker: sorting permutes the same
    # values, so a plain mean of (x_sorted - x) is identically zero and MUST
    # NOT be used as the patch stratification numerator (rms is required).
    z = np.linspace(0.0, 5.0, 200)
    x = RNG.normal(25.0, 0.1, z.size)
    disp = thorpe_displacements(z, x, increasing_down=True)
    assert abs(np.mean(disp.x_sorted - disp.x)) < 1e-14
    assert np.sqrt(np.mean((disp.x - disp.x_sorted) ** 2)) > 0  # rms is not


def test_unsorted_depth_input_is_depth_ordered_first():
    z = np.array([1.0, 0.0, 2.0])
    x = np.array([25.1, 25.0, 25.2])  # stable once depth-ordered
    disp = thorpe_displacements(z, x, increasing_down=True)
    np.testing.assert_array_equal(disp.z, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(disp.delta, 0.0)


def test_ties_are_stable_and_undisplaced():
    z = np.linspace(0.0, 1.0, 5)
    x = np.full(5, 25.0)  # perfectly mixed: all ties
    disp = thorpe_displacements(z, x, increasing_down=True)
    np.testing.assert_array_equal(disp.delta, 0.0)
    # same for the descending (temperature) orientation
    disp_t = thorpe_displacements(z, x, increasing_down=False)
    np.testing.assert_array_equal(disp_t.delta, 0.0)


def test_displacement_input_validation():
    with pytest.raises(ValueError, match="equal length"):
        thorpe_displacements([0.0, 1.0], [25.0], increasing_down=True)
    with pytest.raises(ValueError, match="at least 2"):
        thorpe_displacements([0.0], [25.0], increasing_down=True)
    with pytest.raises(ValueError, match="finite"):
        thorpe_displacements([0.0, np.nan], [25.0, 25.1], increasing_down=True)


# --------------------------------------------------------------------------- #
# thorpe_stats
# --------------------------------------------------------------------------- #


def test_stats_of_flipped_segment():
    z = np.linspace(0.0, 1.0, 11)
    sigma_obs = (25.0 + 0.02 * z)[::-1]
    stats = thorpe_stats(thorpe_displacements(z, sigma_obs, increasing_down=True))
    delta = z - z[::-1]
    np.testing.assert_allclose(stats.L_T, np.sqrt(np.mean(delta**2)))
    assert stats.frac_displaced == pytest.approx(10 / 11)  # center parcel stays
    assert stats.max_run == 5  # one contiguous same-sign block each side
    assert stats.edge_truncated  # displacement nonzero at both edges
    assert stats.span == pytest.approx(1.0)


def test_stats_no_overturn():
    z = np.linspace(0.0, 1.0, 11)
    stats = thorpe_stats(thorpe_displacements(z, 25.0 + 0.02 * z, increasing_down=True))
    assert stats.L_T == 0.0
    assert stats.rms_fluct == 0.0
    assert stats.frac_displaced == 0.0
    assert stats.max_run == 0
    assert not stats.edge_truncated


def test_stats_interior_overturn_not_edge_truncated():
    # Small interior swap, well below the span fraction: no edge flag.
    z = np.arange(0.0, 2.1, 0.1)
    sigma = 25.0 + 0.1 * z
    sigma[10], sigma[11] = sigma[11], sigma[10]
    stats = thorpe_stats(thorpe_displacements(z, sigma, increasing_down=True))
    assert not stats.edge_truncated
    assert stats.L_T > 0


def test_stats_large_interior_overturn_flags_truncation():
    # An interior displacement exceeding the span fraction raises the flag
    # even without touching the window edges.
    z = np.arange(0.0, 2.01, 0.05)
    sigma = 25.0 + 0.1 * z
    i, k = 10, 10 + int(np.ceil(EDGE_TRUNCATION_FRACTION * 2.0 / 0.05)) + 2
    sigma[i], sigma[k] = sigma[k], sigma[i]
    stats = thorpe_stats(thorpe_displacements(z, sigma, increasing_down=True))
    assert stats.edge_truncated


def test_boundary_jiggle_is_not_edge_truncation():
    # A one-sample noise displacement at the window boundary must NOT raise
    # the truncation flag (with a literal nonzero-at-edge test, essentially
    # every real noisy window would be flagged).
    z = np.arange(0.0, 2.01, 0.05)
    sigma = 25.0 + 0.1 * z
    sigma[0], sigma[1] = sigma[1], sigma[0]  # single-sample boundary swap
    stats = thorpe_stats(thorpe_displacements(z, sigma, increasing_down=True))
    assert stats.L_T > 0
    assert not stats.edge_truncated  # 0.05 m << 0.1 * 2.0 m span
    # ... but a SUBSTANTIAL displacement at the boundary still flags.
    sigma2 = 25.0 + 0.1 * z
    sigma2[0], sigma2[6] = sigma2[6], sigma2[0]  # 0.3 m > 0.1 * 2.0 m
    stats2 = thorpe_stats(thorpe_displacements(z, sigma2, increasing_down=True))
    assert stats2.edge_truncated


def test_noise_yields_short_runs():
    # Uncorrelated noise on zero stratification: Galbraith-Kelley runs stay
    # far below a coherent overturn's run length.
    z = np.linspace(0.0, 2.0, 128)
    x = RNG.normal(25.0, 0.001, z.size)
    stats = thorpe_stats(thorpe_displacements(z, x, increasing_down=True))
    flipped = thorpe_stats(thorpe_displacements(z, (25.0 + 0.01 * z)[::-1], increasing_down=True))
    assert stats.max_run < flipped.max_run


# --------------------------------------------------------------------------- #
# patch_n2 — the Smyth/Kaminski overturn-weighted stratification
# --------------------------------------------------------------------------- #


def test_patch_n2_recovers_background_for_full_overturn_density_route():
    # For a fully overturned linear sigma0 profile, rms(x - x*) = G * L_T,
    # so N2_patch = (g/rho0) * G exactly — the true background N2.
    z = np.linspace(0.0, 2.0, 101)
    G = 0.05  # dsigma0/dz [kg/m^3 per m]
    sigma_obs = (25.0 + G * z)[::-1]
    stats = thorpe_stats(thorpe_displacements(z, sigma_obs, increasing_down=True))
    g = 9.81
    rho0 = 1000.0 + 25.0 + G * 1.0  # 1000 + mean(sigma0)
    n2 = patch_n2(stats.rms_fluct, stats.L_T, g / rho0)
    np.testing.assert_allclose(n2, g / rho0 * G, rtol=1e-12)


def test_patch_n2_recovers_background_for_full_overturn_temperature_route():
    z = np.linspace(0.0, 2.0, 101)
    dTdz = -0.1  # warm above cold: stable, observed flipped
    T_obs = (20.0 + dTdz * z)[::-1]
    stats = thorpe_stats(thorpe_displacements(z, T_obs, increasing_down=False))
    alpha, g = 2.5e-4, 9.81
    n2 = patch_n2(stats.rms_fluct, stats.L_T, alpha * g)
    np.testing.assert_allclose(n2, alpha * g * abs(dTdz), rtol=1e-12)


def test_patch_n2_nan_when_no_overturn():
    n2 = patch_n2(0.0, 0.0, 9.81 / 1025.0)
    assert np.isnan(n2)
    # array form
    out = patch_n2([0.01, 0.0], [0.1, 0.0], 9.81 / 1025.0)
    assert np.isfinite(out[0]) and np.isnan(out[1])


# --------------------------------------------------------------------------- #
# derived scales
# --------------------------------------------------------------------------- #


def test_ozmidov_and_r_ot_values_and_guards():
    lo = ozmidov(1e-8, 1e-4)
    np.testing.assert_allclose(lo, np.sqrt(1e-8 / 1e-6))  # (eps/N^3)^1/2, N^3=1e-6
    assert np.isnan(ozmidov(0.0, 1e-4))
    assert np.isnan(ozmidov(1e-8, 0.0))
    assert np.isnan(ozmidov(1e-8, -1e-5))
    np.testing.assert_allclose(r_ot(0.2, 0.1), 2.0)
    assert np.isnan(r_ot(0.2, 0.0))
    assert np.isnan(r_ot(np.nan, 0.1))


def test_reynolds_buoyancy_values_and_guards():
    np.testing.assert_allclose(reynolds_buoyancy(1e-8, 1e-6, 1e-4), 100.0)
    assert np.isnan(reynolds_buoyancy(-1e-8, 1e-6, 1e-4))
    assert np.isnan(reynolds_buoyancy(1e-8, 1e-6, 0.0))
    out = reynolds_buoyancy([1e-8, np.nan], [1e-6, 1e-6], [1e-4, 1e-4])
    assert np.isfinite(out[0]) and np.isnan(out[1])


def test_cox_number_values_and_guards():
    # chi = 2 * kappa * dTdz^2 * Cx  =>  Cx = 50 for these inputs
    kappa = 1.4e-7
    dTdz = 0.01
    chi = 50 * 2 * kappa * dTdz**2
    np.testing.assert_allclose(cox_number(chi, kappa, dTdz), 50.0)
    assert np.isnan(cox_number(chi, kappa, 0.0))
    assert np.isnan(cox_number(-chi, kappa, dTdz))
    # sign of the gradient is irrelevant (squared)
    np.testing.assert_allclose(cox_number(chi, kappa, -dTdz), 50.0)


# --------------------------------------------------------------------------- #
# window_thorpe — the cast-level window loop
# --------------------------------------------------------------------------- #


def _cast(n=2048, fs=64.0, w=0.7, p0=10.0):
    """Synthetic downcast: times, pressures, and a stable sigma0 profile."""
    t = np.arange(n) / fs
    P = p0 + w * t
    sigma = 25.0 + 0.02 * P
    return t, P, sigma


def test_window_thorpe_flags_only_the_overturned_window():
    t, P, sigma = _cast()
    # Insert one overturn: flip ~1 m of profile centered at t=16 s.
    mid = np.abs(t - 16.0) <= 0.7  # ~0.98 m at w=0.7
    sigma_obs = sigma.copy()
    sigma_obs[mid] = sigma[mid][::-1]
    win_times = np.array([8.0, 16.0, 24.0])
    res = window_thorpe(win_times, 2.0, t, P, sigma_obs, increasing_down=True)
    assert res.L_T[0] == 0.0 and res.L_T[2] == 0.0
    assert res.L_T[1] > 0.1  # a ~1-m overturn has a decimeter-scale L_T
    assert res.max_run[1] > 10
    assert res.n[0] > 0 and res.n[1] > 0
    # sanity: the sorted rms fluctuation and L_T reproduce the background
    # gradient through the patch-N2 identity (rms_fluct ~= G * L_T)
    np.testing.assert_allclose(res.rms_fluct[1] / res.L_T[1], 0.02, rtol=0.15)


def test_window_thorpe_handles_nan_and_sparse_windows():
    t, P, sigma = _cast(n=512)
    sigma = sigma.copy()
    sigma[100:110] = np.nan
    # window centered past the end of the cast -> no samples -> NaN row
    res = window_thorpe(np.array([2.0, 500.0]), 2.0, t, P, sigma, increasing_down=True)
    assert np.isfinite(res.L_T[0])
    assert np.isnan(res.L_T[1])
    assert res.n[1] == 0
    assert res.max_run[1] == 0
    assert not res.edge_truncated[1]


def test_window_thorpe_min_samples_and_min_dp():
    t, P, sigma = _cast(n=512)
    # min_samples: only 3 samples in a tiny window
    res = window_thorpe(np.array([4.0]), 0.02, t, P, sigma, increasing_down=True, min_samples=8)
    assert np.isnan(res.L_T[0])
    # min_dp: stalled instrument (constant pressure)
    P_stall = np.full_like(P, 10.0)
    res2 = window_thorpe(np.array([4.0]), 2.0, t, P_stall, sigma, increasing_down=True)
    assert np.isnan(res2.L_T[0])


def test_window_thorpe_float32_sigma0_promoted():
    # Products store sigma0 as float32; the window loop must promote to
    # float64 so small-fluctuation arithmetic keeps precision.
    t, P, sigma = _cast(n=512)
    res = window_thorpe(
        np.array([4.0]),
        2.0,
        t,
        P,
        sigma.astype(np.float32),
        increasing_down=True,
    )
    assert res.L_T.dtype == np.float64
    assert np.isfinite(res.L_T[0])


def test_window_thorpe_depth_conversion_uses_latitude():
    # The depth grid comes from -gsw.z_from_p; make sure the window span is
    # in meters (close to, but not identical to, the dbar span).
    t, P, sigma = _cast(n=512)
    res = window_thorpe(np.array([4.0]), 4.0, t, P, sigma, increasing_down=True, lat=13.0)
    p_span = np.ptp(P[np.abs(t - 4.0) <= 4.0])
    z_span = float(res.span[0])
    depth_expected = float(-gsw.z_from_p(P.max(), 13.0) + gsw.z_from_p(P.min(), 13.0))
    # rel=1e-6 is deliberate: it catches both a dbar-for-meters mix-up
    # (rel ~5e-3) and a hard-coded lat=0 (rel ~3e-4); the correct code
    # matches to machine precision (the window spans the whole cast).
    assert z_span == pytest.approx(p_span * depth_expected / np.ptp(P), rel=1e-6)
    assert z_span != pytest.approx(p_span, rel=1e-6)  # meters, not dbar


def test_window_thorpe_temperature_route_matches_density_route_geometry():
    # With T the mirror proxy of sigma0 (linearly related), both routes must
    # find identical displacement geometry for the same overturn.
    t, P, sigma = _cast()
    T = 20.0 - (sigma - 25.0) / 0.2  # linear in sigma, decreasing with depth
    mid = np.abs(t - 16.0) <= 0.7
    sigma_obs, T_obs = sigma.copy(), T.copy()
    sigma_obs[mid] = sigma[mid][::-1]
    T_obs[mid] = T[mid][::-1]
    win = np.array([16.0])
    res_s = window_thorpe(win, 2.0, t, P, sigma_obs, increasing_down=True)
    res_t = window_thorpe(win, 2.0, t, P, T_obs, increasing_down=False)
    np.testing.assert_allclose(res_s.L_T, res_t.L_T)
    np.testing.assert_allclose(res_s.max_run, res_t.max_run)
