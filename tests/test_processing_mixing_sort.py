"""Sort order, temperature-only gradients and the kappa diagnostic (processing/mixing.py).

The motivating failure: a salinity whose noise is large against its gradient
scrambles the density sort, which then flattens dT/dz and inflates N2. These
tests build that case synthetically and check that the temperature sort (the
default since 2026-10) and the temperature-only gradient recover the truth, and
that sort_kappa sees it under every sort.
"""

import gsw
import numpy as np
import pytest

from odas_tpw.processing.mixing import (
    DEFAULT_SORT,
    KAPPA_MIN_WINDOWS,
    KAPPA_WARN,
    SORT_MODES,
    profile_stratification,
    profile_stratification_full,
    profile_temperature_gradient,
    sort_kappa,
    sorted_stratification,
    window_stratification,
    window_temperature_gradient,
)

FS = 64.0
W = 0.8  # [dbar/s], a VMP-5500's fall rate
DTDZ = -0.07  # [K/m]
DSDZ = -0.0037  # [psu/m]; destabilizing, R_rho ~ 2 with DTDZ (finger regime)


def _cast(seconds=150.0, p0=150.0):
    t = np.arange(0.0, seconds, 1.0 / FS)
    P = p0 + W * t
    z = -gsw.z_from_p(P, 0.0)
    T = 18.0 + DTDZ * (z - z[0])
    S = 36.0 + DSDZ * (z - z[0])
    return t, P, z, T, S


def _structured_noise(n, sigma, corr_samples, seed=0):
    """Red noise with ~corr_samples correlation, scaled to rms sigma."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n + 4 * corr_samples)
    k = np.exp(-np.arange(-2 * corr_samples, 2 * corr_samples + 1) ** 2 / (2.0 * corr_samples**2))
    y = np.convolve(x, k / k.sum(), mode="valid")[:n]
    return sigma * y / y.std()


def _true_dCTdz(P, z, T, S):
    SA = gsw.SA_from_SP(S, P, 0.0, 0.0)
    return float(np.polyfit(z, gsw.CT_from_t(SA, T, P), 1)[0])


def _linear_N2(P, T, S):
    SA = gsw.SA_from_SP(S[[0, -1]], P[[0, -1]], 0.0, 0.0)
    CT = gsw.CT_from_t(SA, T[[0, -1]], P[[0, -1]])
    return float(gsw.Nsquared(SA, CT, P[[0, -1]], 0.0)[0][0])


class TestSortModes:
    def test_modes_are_the_documented_three(self):
        assert SORT_MODES == ("density", "temperature", "none")

    @pytest.mark.parametrize("bad", ["Density", "salinity", "", None])
    def test_unknown_sort_raises(self, bad):
        t, P, _, T, S = _cast(30.0)
        with pytest.raises(ValueError, match="sort="):
            sorted_stratification(np.array([15.0]), 4.0, t, P, T, S=S, sort=bad)
        with pytest.raises(ValueError, match="sort="):
            profile_stratification(P, T, S, sort=bad)

    def test_clean_profile_all_sorts_agree(self):
        # Monotonic, stable, noise-free: every order is the measured order, so
        # N2 agrees exactly. dT/dz under "temperature" is taken from CT at the
        # window's median salinity, which drops the haline term dCT/dSA*dSA/dz
        # (here ~0.03 K/(g/kg) * 0.0037 /m against 0.07 K/m, i.e. ~0.16%).
        t, P, _, T, S = _cast()
        wt = np.array([30.0, 75.0, 120.0])
        res = {m: sorted_stratification(wt, 4.0, t, P, T, S=S, sort=m) for m in SORT_MODES}
        np.testing.assert_allclose(res["none"].dTdz, res["density"].dTdz, rtol=1e-10)
        np.testing.assert_allclose(res["temperature"].dTdz, res["density"].dTdz, rtol=5e-3)
        for m in ("temperature", "none"):
            np.testing.assert_allclose(res[m].N2, res["density"].N2, rtol=1e-10)

    def test_default_is_temperature(self):
        from odas_tpw.perturb.config import DEFAULTS

        assert DEFAULT_SORT == "temperature"
        assert DEFAULTS["stratification"]["sort"] == DEFAULT_SORT
        t, P, _, T, S = _cast()
        S_noisy = S + _structured_noise(len(S), 0.2, 64)
        wt = np.arange(10.0, 140.0, 2.5)
        a = sorted_stratification(wt, 1.25, t, P, T, S=S_noisy)
        b = sorted_stratification(wt, 1.25, t, P, T, S=S_noisy, sort="temperature")
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(
            profile_stratification(P, T, S_noisy)[2],
            profile_stratification(P, T, S_noisy, sort="temperature")[2],
        )

    def test_kappa_gradients_under_every_sort(self):
        # Both orders' gradients are computed from the same per-sample CT
        # whatever the product's sort, so kappa does not depend on the sort...
        t, P, _, T, S = _cast()
        S_noisy = S + _structured_noise(len(S), 0.2, 64)
        wt = np.arange(10.0, 140.0, 2.5)
        res = {m: sorted_stratification(wt, 1.25, t, P, T, S=S_noisy, sort=m) for m in SORT_MODES}
        prof = {m: profile_stratification_full(P, T, S_noisy, window=2.0, sort=m)
                for m in SORT_MODES}
        for m in SORT_MODES:
            assert np.isfinite(res[m].dTdz_T).all() and np.isfinite(res[m].dTdz_rho).all()
            for got, ref in ((res[m], res["density"]), (prof[m], prof["density"])):
                np.testing.assert_array_equal(got.dTdz_rho, ref.dTdz_rho)
                np.testing.assert_array_equal(got.dTdz_T, ref.dTdz_T)
        # ...and under density the density-order gradient is the product's own.
        np.testing.assert_array_equal(res["density"].dTdz_rho, res["density"].dTdz)
        # A constant salinity leaves kappa undefined: the two orders coincide.
        for m in SORT_MODES:
            fixed = sorted_stratification(wt, 1.25, t, P, T, S=35.0, sort=m)
            assert np.isnan(fixed.dTdz_T).all() and np.isnan(fixed.dTdz_rho).all()

    def test_none_matches_window_stratification(self):
        # The unsorted fit is window_stratification's fit; N2 differs only by
        # convert-then-mean vs mean-then-convert.
        t, P, _, T, S = _cast()
        S_noisy = S + _structured_noise(len(S), 0.2, 40)
        wt = np.array([30.0, 75.0, 120.0])
        a = sorted_stratification(wt, 4.0, t, P, T, S=S_noisy, sort="none")
        b = window_stratification(wt, 4.0, t, P, T, S=S_noisy)
        np.testing.assert_allclose(a.dTdz, b.dTdz, rtol=1e-9)
        np.testing.assert_allclose(a.N2, b.N2, rtol=0.05)

    def test_profile_wrapper_matches_full(self):
        _, P, _, T, S = _cast()
        full = profile_stratification_full(P, T, S, window=2.0)
        short = profile_stratification(P, T, S, window=2.0)
        for x, y in zip(short, full[:3]):
            np.testing.assert_array_equal(x, y)


class TestNoisySalinity:
    """Structured salinity noise, rms 0.2 psu over ~1 m, as on an unpumped SBE4."""

    def _setup(self):
        t, P, z, T, S = _cast()
        S_noisy = S + _structured_noise(len(S), 0.2, 64)
        wt = np.arange(10.0, 140.0, 2.5)  # 2-dbar-ish windows (half 1.25 s)
        return t, P, z, T, S, S_noisy, wt

    def test_density_sort_flattens_dTdz(self):
        t, P, z, T, S, S_noisy, wt = self._setup()
        truth = _true_dCTdz(P, z, T, S)
        dens = sorted_stratification(wt, 1.25, t, P, T, S=S_noisy, sort="density")
        assert np.nanmedian(dens.dTdz) / truth < 0.8

    def test_temperature_sort_recovers_dTdz(self):
        t, P, z, T, S, S_noisy, wt = self._setup()
        truth = _true_dCTdz(P, z, T, S)
        tsort = sorted_stratification(wt, 1.25, t, P, T, S=S_noisy, sort="temperature")
        assert abs(np.nanmedian(tsort.dTdz) / truth - 1.0) < 0.05

    def test_density_sort_inflates_N2_more_than_temperature_sort(self):
        t, P, _, T, S, S_noisy, wt = self._setup()
        n2_lin = _linear_N2(P, T, S)
        dens = sorted_stratification(wt, 1.25, t, P, T, S=S_noisy, sort="density")
        tsort = sorted_stratification(wt, 1.25, t, P, T, S=S_noisy, sort="temperature")
        assert np.nanmedian(dens.N2) > 2.0 * n2_lin
        assert np.nanmedian(tsort.N2) < np.nanmedian(dens.N2)

    def test_kappa_flags_it(self):
        t, P, _, T, S, S_noisy, wt = self._setup()
        dens = sorted_stratification(wt, 1.25, t, P, T, S=S_noisy, sort="density")
        assert sort_kappa(dens.dTdz_rho, dens.dTdz_T) < KAPPA_WARN
        # The default (temperature) sort reports the same kappa.
        tsort = sorted_stratification(wt, 1.25, t, P, T, S=S_noisy)
        assert sort_kappa(tsort.dTdz_rho, tsort.dTdz_T) == sort_kappa(dens.dTdz_rho, dens.dTdz_T)
        clean = sorted_stratification(wt, 1.25, t, P, T, S=S)
        assert sort_kappa(clean.dTdz_rho, clean.dTdz_T) == pytest.approx(1.0, abs=1e-9)

    def test_profile_path_agrees(self):
        _, P, z, T, S, S_noisy, _ = self._setup()
        truth = _true_dCTdz(P, z, T, S)
        dens = profile_stratification_full(P, T, S_noisy, window=2.0, sort="density")
        tsort = profile_stratification_full(P, T, S_noisy, window=2.0, sort="temperature")
        assert np.nanmedian(dens.dTdz) / truth < 0.8
        assert abs(np.nanmedian(tsort.dTdz) / truth - 1.0) < 0.05
        assert sort_kappa(dens.dTdz_rho, dens.dTdz_T) < KAPPA_WARN
        assert sort_kappa(tsort.dTdz_rho, tsort.dTdz_T) < KAPPA_WARN


class TestDiffusiveWindow:
    def test_temperature_sort_follows_the_trend(self):
        # Cold fresh over warm salty, density-stable: T rises with depth. A
        # warm-on-top sort would flip the gradient; the trend-following one keeps
        # its sign, and small temperature inversions are removed, not inverted.
        t = np.arange(0.0, 60.0, 1.0 / FS)
        P = 50.0 + W * t
        z = -gsw.z_from_p(P, 0.0)
        T = 2.0 + 0.02 * (z - z[0]) + 0.01 * np.sin(2 * np.pi * z / 0.5)
        S = 33.0 + 0.05 * (z - z[0])
        res = sorted_stratification(np.array([30.0]), 8.0, t, P, T, S=S, sort="temperature")
        none = sorted_stratification(np.array([30.0]), 8.0, t, P, T, S=S, sort="none")
        assert res.dTdz[0] > 0
        assert res.dTdz[0] >= none.dTdz[0]
        assert res.N2[0] > 0  # salinity carries the stability


class TestTemperatureSortDirection:
    """The sort direction comes from a span 5x the window (adversarial review)."""

    def _overturned(self):
        # Stable linear background with a 3-m overturn (the column flipped end
        # for end) around 200 dbar; salinity is clean, so density is right.
        _, P, z, T, S = _cast(seconds=60.0, p0=180.0)
        z0 = float(np.interp(200.0, P, z))
        inside = np.abs(z - z0) <= 1.5
        zf = np.where(inside, 2 * z0 - z, z)  # mirror depth inside the patch
        T = 18.0 + DTDZ * (zf - z[0])
        S = 36.0 + DSDZ * (zf - z[0])
        return P, z, T, S

    def test_overturn_larger_than_half_the_window_is_stabilized(self):
        P, _, T, S = self._overturned()
        dens = profile_stratification_full(P, T, S, window=2.0, sort="density")
        tsort = profile_stratification_full(P, T, S, window=2.0, sort="temperature")
        k = int(np.argmin(np.abs(dens.target_P - 200.0)))
        assert dens.N2[k] > 0 and dens.dTdz[k] < 0
        # The window's own trend is reversed; the 10-dbar trend is not.
        assert tsort.N2[k] > 0
        assert tsort.dTdz[k] < 0
        assert tsort.dTdz[k] == pytest.approx(dens.dTdz[k], rel=0.02)

    def test_temperature_extremum_does_not_invent_a_gradient(self):
        # A temperature minimum held up by salinity (density-stable): no
        # monotonic temperature background exists at the minimum, so the sort
        # must keep the measured order rather than make one up.
        t = np.arange(0.0, 40.0, 1.0 / FS)
        P = 40.0 + W * t
        z = -gsw.z_from_p(P, 0.0)
        z_min = float(np.interp(56.0, P, z))
        T = 2.0 + 0.004 * (z - z_min) ** 2
        S = 33.0 + 0.02 * (z - z[0])
        tsort = profile_stratification_full(P, T, S, window=2.0, sort="temperature")
        none = profile_stratification_full(P, T, S, window=2.0, sort="none")
        k = int(np.argmin(np.abs(tsort.target_P - 56.0)))
        assert abs(tsort.dTdz[k]) < 2e-3
        # Measured order kept: N2 (per-sample SA/CT in that order) is the
        # unsorted value. (dT/dz differs from "none" by the haline term, since
        # the temperature path uses median-salinity CT.)
        assert tsort.N2[k] == pytest.approx(none.N2[k], rel=1e-9)
        # ...whereas sorting along the window's own (near-zero) trend would have
        # invented |dT/dz| of order the window's temperature range per meter.


class TestKappaSummary:
    def test_band_limited_failure_is_found(self):
        from odas_tpw.processing.mixing import kappa_summary, kappa_warning

        P_win = np.arange(0.0, 1000.0, 1.0)
        d_T = np.full(P_win.size, -0.05)
        d = d_T.copy()
        bad = (P_win >= 150) & (P_win < 250)
        d[bad] = 0.4 * d_T[bad]
        summ = kappa_summary(d, d_T, P_win)
        assert summ.overall == pytest.approx(1.0)  # the cast-wide median misses it
        assert summ.worst == pytest.approx(0.4)
        assert 150.0 <= summ.worst_P <= 250.0
        for sort in ("density", "temperature", "none"):
            msg = kappa_warning(summ, "x", sort)
            assert msg is not None and "0.40" in msg
            assert "fixed salinity or one from a T-S relation" in msg  # not circular
        # Only under density is the product's own dT/dz the biased one.
        assert "dT/dz is biased low" in kappa_warning(summ, "x", "density")
        msg = kappa_warning(summ, "x")  # the default sort
        assert "would be 0.40" in msg and "biased low" not in msg
        assert "temperature-sorted, so the salinity does not reorder it" in msg
        assert "in measured order" in kappa_warning(summ, "x", "none")
        with pytest.raises(ValueError, match="sort="):
            kappa_warning(summ, "x", "Density")

    def test_no_warning_when_clean_or_undefined(self):
        from odas_tpw.processing.mixing import KappaSummary, kappa_warning

        assert kappa_warning(KappaSummary(1.0, 0.95, 75.0), "x") is None
        assert kappa_warning(KappaSummary(np.nan, np.nan, np.nan), "x") is None


class TestSortKappa:
    def test_too_few_windows_is_nan(self):
        d = np.full(KAPPA_MIN_WINDOWS - 1, -0.05)
        assert np.isnan(sort_kappa(d, d))

    def test_weak_windows_do_not_vote(self):
        strong = np.full(6, -0.05)
        weak = np.full(20, -1e-4)
        d = np.concatenate([0.5 * strong, -weak])  # weak windows disagree in sign
        d_T = np.concatenate([strong, weak])
        assert sort_kappa(d, d_T) == pytest.approx(0.5)

    def test_nan_windows_ignored(self):
        d_T = np.full(8, -0.05)
        d = 0.9 * d_T
        d[:2] = np.nan
        assert sort_kappa(d, d_T) == pytest.approx(0.9)


class TestTemperatureOnlyGradient:
    def test_matches_temperature_sort_with_constant_salinity(self):
        t, P, _, T, _ = _cast()
        wt = np.array([30.0, 75.0, 120.0])
        # Same order and the same salinity; the only difference is that the N2
        # path keys on the window's median *absolute* salinity, which carries
        # SA_from_SP's small pressure dependence (~2e-6 relative here).
        ref = sorted_stratification(wt, 4.0, t, P, T, S=35.0, sort="temperature").dTdz
        np.testing.assert_allclose(window_temperature_gradient(wt, 4.0, t, P, T, S=35.0), ref,
                                   rtol=1e-5)
        tp, dT = profile_temperature_gradient(P, T, S=35.0, window=2.0)
        full = profile_stratification_full(P, T, 35.0, window=2.0, sort="temperature")
        np.testing.assert_array_equal(tp, full.target_P)
        np.testing.assert_allclose(dT, full.dTdz, rtol=1e-5)

    def test_offset_does_not_matter_gain_does(self):
        # Two beads on nominal coefficients read ~3 K warm; that must not reach
        # the gradient. A gain error does, linearly -- which is why K_T built from
        # the same bead's chi and dT/dz cancels it (chi ~ g^2, dT/dz^2 ~ g^2).
        t, P, _, T, _ = _cast()
        wt = np.array([30.0, 75.0, 120.0])
        base = window_temperature_gradient(wt, 4.0, t, P, T)
        warm = window_temperature_gradient(wt, 4.0, t, P, T + 3.1)
        low_gain = window_temperature_gradient(wt, 4.0, t, P, 18.0 + 0.97 * (T - 18.0))
        np.testing.assert_allclose(warm, base, rtol=2e-3)
        np.testing.assert_allclose(low_gain / base, 0.97, rtol=2e-3)

    def test_salinity_spikes_do_not_reach_it(self):
        t, P, _, T, S = _cast()
        S_spiky = S.copy()
        S_spiky[::50] += 2.0  # 2% of samples
        wt = np.array([30.0, 75.0, 120.0])
        a = window_temperature_gradient(wt, 4.0, t, P, T, S=S)
        b = window_temperature_gradient(wt, 4.0, t, P, T, S=S_spiky)
        np.testing.assert_allclose(a, b, rtol=1e-6)

    def test_no_density_order_without_salinity(self):
        t, P, _, T, _ = _cast(30.0)
        with pytest.raises(ValueError, match="sort="):
            window_temperature_gradient(np.array([15.0]), 4.0, t, P, T, sort="density")
        with pytest.raises(ValueError, match="sort="):
            profile_temperature_gradient(P, T, sort="density")

    def test_salinity_shape_checked(self):
        t, P, _, T, _ = _cast(30.0)
        with pytest.raises(ValueError, match="shape"):
            window_temperature_gradient(np.array([15.0]), 4.0, t, P, T, S=np.ones(7))

    def test_nan_samples_and_empty_windows(self):
        t, P, _, T, _ = _cast(30.0)
        T = T.copy()
        T[: len(T) // 2] = np.nan
        out = window_temperature_gradient(np.array([5.0, 25.0, 500.0]), 2.0, t, P, T)
        assert np.isnan(out[0]) and np.isfinite(out[1]) and np.isnan(out[2])
        tp, dT = profile_temperature_gradient(np.full(3, 10.0), np.full(3, 20.0))
        assert tp.size == 0 and dT.size == 0


class TestNullCases:
    """Clean salinity must not warn; noise must not be sorted into a gradient."""

    def _cast_with_mixed_layer(self, seed):
        # 30 m mixed layer (2e-4 K/m) over a thermocline; S = f(T), so salinity
        # carries no error; 1 mK temperature and 0.001 psu white noise.
        rng = np.random.default_rng(seed)
        t = np.arange(0.0, 300.0, 1.0 / FS)
        P = 2.0 + W * t
        z = -gsw.z_from_p(P, 0.0)
        T = np.where(z < 30.0, 25.0 - 2e-4 * z, 25.0 - 2e-4 * 30.0 - 0.05 * (z - 30.0))
        S = 35.0 + 0.02 * (25.0 - T)
        T = T + rng.normal(0.0, 1e-3, T.size)
        S = S + rng.normal(0.0, 1e-3, S.size)
        return t, P, T, S

    def test_kappa_quiet_on_clean_salinity(self):
        from odas_tpw.processing.mixing import kappa_summary, kappa_warning

        for seed in range(10):
            _, P, T, S = self._cast_with_mixed_layer(seed)
            res = profile_stratification_full(P, T, S, window=2.0)
            summ = kappa_summary(res.dTdz_rho, res.dTdz_T, res.target_P)
            assert kappa_warning(summ, "x") is None, (seed, summ)
            # mixed-layer windows (measured gradient under 1e-3) do not vote; the
            # half-empty window at the cast top is left out (its noise slope is
            # twice as large)
            ml = (res.target_P > 3.0) & (res.target_P < 25.0)
            assert np.isnan(res.dTdz_T[ml]).all()

    def test_temperature_sort_does_not_inflate_noise(self):
        _, P, T, S = self._cast_with_mixed_layer(0)
        none = profile_stratification_full(P, T, S, window=2.0, sort="none")
        tsort = profile_stratification_full(P, T, S, window=2.0, sort="temperature")
        ml = (none.target_P > 3.0) & (none.target_P < 25.0)
        # A full sort of 1 mK noise over a 2-dbar window would give ~5e-3 K/m.
        assert np.median(np.abs(tsort.dTdz[ml])) < 2.0 * np.median(np.abs(none.dTdz[ml])) + 1e-4
        # ...while in the thermocline the sort is applied and changes nothing.
        tc = none.target_P > 60.0
        np.testing.assert_allclose(np.median(tsort.dTdz[tc]), -0.05, rtol=0.01)
