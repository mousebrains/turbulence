# Sep-2026, Claude and Pat Welch, pat@mousebrains.com
"""Negative controls for issue #180 groups D and E.

F08, F09, F10, F17, F18, F19, F20 — inference defects: nuisance effects
reappearing as signal, correlated observations counted as replication, an
outlier supplying the scale that absolves it, and asynchronous support turning
temporal variability into vertical shear.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from odas_tpw.clocksync.fit import FileFit, window_pairs
from odas_tpw.clocksync.qc import check_continuity
from odas_tpw.clocksync.series import PressureSeries
from odas_tpw.fp07cal.geometry import MIN_CLUSTERS_FOR_SANDWICH, joint_fit
from odas_tpw.fp07cal.pairs import PairSet
from odas_tpw.fp07cal.stability import (
    StabilityResult,
    blocked_offsets,
    corroborates,
    drift_fit,
)
from odas_tpw.perturb.adcp import AdcpData, window_shear


def _stable_bead_pairs(n_prof: int = 60, n_samp: int = 100, seed: int = 20260907):
    """A bead that does NOT drift, at a fixed dz, in a CHANGING gradient."""
    rng = np.random.default_rng(seed)
    pid = np.repeat(np.arange(n_prof), n_samp)
    phase = np.tile(np.linspace(0, 2 * np.pi, n_samp), n_prof)
    L = np.tile(np.linspace(-0.15, 0.15, n_samp), n_prof)
    truth = 1 / (1 / 290 + L / 3500) - 273.15
    g = 0.03 + 0.12 * pid / (n_prof - 1) + 0.02 * np.sin(phase)
    w = 0.2 * np.where(pid % 2, 1.0, -1.0) * (1 + 0.1 * np.cos(phase))
    pairs = PairSet(
        time=pid * 86400.0 + np.tile(np.arange(n_samp), n_prof),
        L=L,
        T_ref=truth + 0.17 * g + rng.normal(0, 1e-5, n_prof * n_samp),
        pressure=np.tile(np.arange(n_samp), n_prof),
        w=w,
        direction=np.sign(w),
        profile_uid=pid.astype(object),
        file_label=np.full(n_prof * n_samp, "file", object),
        channel="T1",
    )
    return pairs, g


class TestF08GeometryIsNotDrift:
    """A nuisance effect fitted out must not come back as bead drift."""

    def test_zero_drift_changing_gradient_negative_control(self):
        pairs, g = _stable_bead_pairs()
        fit, geo = joint_fit(pairs, dTdz=g, robust=False)
        assert geo.dz_m == pytest.approx(0.17, abs=5e-3)
        assert fit.rms_K < 1e-4

        # Pre-fix: blocked_offsets saw the ORIGINAL reference and a polynomial
        # with the geometry already removed, so the changing gradient became a
        # changing intercept: +3.4578e-4 K/day, permutation p = 0.005,
        # significant = True, against a TRUE drift of exactly zero.
        blind = drift_fit(blocked_offsets(pairs, fit), n_permutations=2000)
        with_geo = drift_fit(
            blocked_offsets(pairs, fit, geo=geo, dTdz=g), n_permutations=2000
        )
        assert abs(with_geo.drift_K_per_day) < abs(blind.drift_K_per_day) / 5
        assert abs(with_geo.drift_K_per_day) < 1e-4

    def test_geometry_free_fit_is_unaffected(self):
        """geo=None must reproduce the old path exactly."""
        pairs, g = _stable_bead_pairs(n_prof=12)
        fit, _ = joint_fit(pairs, dTdz=g, robust=False)
        a = blocked_offsets(pairs, fit)
        b = blocked_offsets(pairs, fit, geo=None)
        assert [x.a0 for x in a] == [x.a0 for x in b]


def _geometry_pairs(seed: int = 20260907, n: int = 600):
    rng = np.random.default_rng(seed)
    L = np.linspace(-0.1, 0.1, n)
    T = 1 / (1 / 290 + L / 3500) - 273.15
    g = 0.07 + 0.03 * np.sin(np.arange(n) / 23)
    w = 0.2 * np.sin(np.arange(n) / 37)
    pairs = PairSet(
        time=np.arange(float(n)),
        L=L,
        T_ref=T + 0.17 * g + 0.7 * w * g + rng.normal(0, 0.003, n),
        pressure=np.arange(float(n)),
        w=w,
        direction=np.sign(w),
        profile_uid=np.repeat(np.arange(6), 100).astype(object),
        file_label=np.full(n, "file", object),
        channel="T1",
    )
    return pairs, g


class TestF09ClusteredUncertainty:
    """Duplicating observations adds no information, so it must not shrink the SE."""

    def test_duplication_leaves_the_clustered_se_alone(self):
        pairs, g = _geometry_pairs()
        _, geo = joint_fit(pairs, dTdz=g, robust=False)

        fields = ("time", "L", "T_ref", "pressure", "w", "direction",
                  "profile_uid", "file_label")
        dup = replace(
            pairs, **{k: np.repeat(getattr(pairs, k), 100) for k in fields}
        )
        _, geo2 = joint_fit(dup, dTdz=np.repeat(g, 100), robust=False)

        assert pairs.n_profiles() == dup.n_profiles() == 6
        assert geo2.dz_m == pytest.approx(geo.dz_m, rel=1e-6)
        # Pre-fix, the ONLY error bar shrank 0.00579 -> 0.000577 m on data
        # carrying no new information.
        assert geo2.dz_se_m < geo.dz_se_m / 5  # the IID error still shrinks...
        # ...but the clustered one, the one to quote, does not.
        assert geo2.dz_se_cluster_m == pytest.approx(geo.dz_se_cluster_m, rel=0.05)
        assert geo2.tau_se_cluster_s == pytest.approx(geo.tau_se_cluster_s, rel=0.05)

    def test_few_clusters_are_flagged_not_silently_quoted(self):
        pairs, g = _geometry_pairs()
        _, geo = joint_fit(pairs, dTdz=g, robust=False)
        assert geo.n_clusters == 6 < MIN_CLUSTERS_FOR_SANDWICH
        assert "6 profile" in geo.cluster_warning
        assert geo.cluster_warning in geo.summary()


class TestF20Identifiability:
    """Rank failure must be a gate, not a diagnostic nobody reads."""

    def test_intercept_aliased_geometry_is_rank_deficient(self):
        pairs, _ = _geometry_pairs()
        L = np.asarray(pairs.L)
        T = 1 / (1 / 290 + L / 3500) - 273.15
        # Makes the geometry regressor -g/T_K^2 constant, hence aliased with the
        # calibration intercept: condition 3.84e30, geometry correlation
        # 4.74e-17 (which the OLD "separately resolved" predicate passed),
        # calibration error 6.5e33 K, dz = 3448 +/- 0.68 m.
        fit, geo = joint_fit(
            replace(pairs, T_ref=T), dTdz=-1e-6 * (T + 273.15) ** 2, robust=False
        )
        # The fit itself is nonsense -- that is the point: it must not be
        # exported. The CLI's hard gate keys on this prediction error.
        assert np.max(np.abs(fit.apply(L) - T)) > 1.0
        assert geo.rank_deficient
        assert not geo.separately_resolved()
        # ...even though the dz/tau correlation alone still looks perfect.
        assert abs(geo.collinearity) < 1e-6

    def test_a_healthy_fit_is_still_resolved(self):
        pairs, g = _geometry_pairs()
        _, geo = joint_fit(pairs, dTdz=g, robust=False)
        assert not geo.rank_deficient
        assert geo.separately_resolved()
        assert np.isfinite(geo.smallest_singular_value) and geo.smallest_singular_value > 0


class TestF19ChannelAwareCorroboration:
    """T1 - T2 tracks T1 positively and T2 negatively."""

    def test_t2_positive_drift_is_corroborated_not_contradicted(self):
        # T2 drifts +0.001 K/day (correction -0.001) against a fixed T1, so the
        # T1-T2 slope is -0.001 -- the CORROBORATING sign. Pre-fix this returned
        # "opposes in sign ... unexplained -- do not apply a drift model".
        stab = StabilityResult(channel="T2", drift_K_per_day=-0.001, significant=True)
        msg = corroborates(stab, {"available": True, "slope_K_per_day": -0.001})
        assert msg is not None and "agrees in sign" in msg

    def test_t1_positive_drift_still_reads_the_same_way(self):
        stab = StabilityResult(channel="T1", drift_K_per_day=-0.001, significant=True)
        msg = corroborates(stab, {"available": True, "slope_K_per_day": +0.001})
        assert msg is not None and "agrees in sign" in msg

    def test_genuine_opposition_is_still_reported(self):
        stab = StabilityResult(channel="T2", drift_K_per_day=-0.001, significant=True)
        msg = corroborates(stab, {"available": True, "slope_K_per_day": +0.001})
        assert msg is not None and "opposes in sign" in msg

    def test_a_third_channel_is_not_silently_read_as_t1(self):
        stab = StabilityResult(channel="T3", drift_K_per_day=-0.001, significant=True)
        msg = corroborates(stab, {"available": True, "slope_K_per_day": -0.001})
        assert msg is not None and "not part of" in msg


class TestF17ClockWindowGap:
    """min_fill is an aggregate; it does not bound the longest hole."""

    def test_long_interior_gap_is_refused(self):
        t = np.arange(0.0, 901.0, 0.5)
        p = 20.0 + 0.1 * np.sin(2 * np.pi * 0.15 * t)
        keep = ~((t >= 400.0) & (t < 500.0))  # a 100 s hole: 11% of a 900 s window
        wp = list(
            window_pairs(
                PressureSeries(t, p, "ref", fs=2.0),
                PressureSeries(t[keep], p[keep], "tgt", fs=2.0),
                2.0,
                900.0,
                max_gap=1.0,
            )
        )
        # Pre-fix: one window, entirely finite, its 100 s fabricated segment
        # unidentified and passed on for lag and uncertainty estimation.
        assert wp == []

    def test_short_holes_are_still_bridged(self):
        t = np.arange(0.0, 901.0, 0.5)
        p = 20.0 + 0.1 * np.sin(2 * np.pi * 0.15 * t)
        keep = np.ones(t.size, dtype=bool)
        keep[100:101] = False  # a single 0.5 s dropout
        wp = list(
            window_pairs(
                PressureSeries(t, p, "ref", fs=2.0),
                PressureSeries(t[keep], p[keep], "tgt", fs=2.0),
                2.0,
                900.0,
                max_gap=1.0,
            )
        )
        assert len(wp) == 1
        assert np.isfinite(wp[0][2]).all()


class TestF18ContinuityScale:
    """An extreme outlier must not supply the scale that absolves it."""

    @pytest.mark.parametrize("jump", [10.0, 10000.0])
    def test_single_slip_is_flagged_at_any_amplitude(self, jump):
        fits = [
            FileFit(name=str(i), t0=float(i), offset=(jump if i == 5 else 0.0),
                    offset_sigma=0.001, ok=True)
            for i in range(10)
        ]
        check_continuity(fits, wave_period=10.0)
        # Pre-fix: MAD is exactly zero, the fallback std INCLUDES the outlier,
        # and |r|/std = N/sqrt(N-1) = 3.33 at N=10 regardless of the amplitude
        # -- below a 5-sigma threshold for a 10 s AND a 10,000 s slip.
        assert sum(bool(f.flags) for f in fits) == 1
        assert fits[5].flags

    def test_a_smoothly_drifting_sequence_is_not_flagged(self):
        fits = [
            FileFit(name=str(i), t0=float(i), offset=0.01 * i,
                    offset_sigma=0.001, ok=True)
            for i in range(12)
        ]
        check_continuity(fits, wave_period=10.0)
        assert not any(f.flags for f in fits)


class TestF10AsynchronousAdcpMasks:
    """A first difference must never span two different ensembles."""

    def test_barotropic_variability_is_not_vertical_shear(self):
        adcp = AdcpData(
            time=np.array([0.0, 1.0]),
            depth=np.array([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]]),
            # Complete fields: [0,0,0] then [1,1,1] -> zero vertical shear at
            # BOTH times. The masks differ by depth.
            u=np.array([[0.0, 0.0, np.nan], [np.nan, 1.0, 1.0]]),
            v=np.zeros((2, 3)),
            name="barotropic",
            path=Path("synthetic"),
        )
        out = window_shear(adcp, [0.5], [20.0], time_tolerance=2.0)
        # Pre-fix: per-cell means [0, 0.5, 1] and S2 = 0.0025 s^-2 from nothing.
        # On common support each adjacent pair is averaged only over the
        # ensembles that carried BOTH of its cells, and recovers the truth.
        assert out.S2[0] == pytest.approx(0.0, abs=1e-15)

    def test_real_shear_on_common_support_survives(self):
        u = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
        adcp = AdcpData(
            time=np.array([0.0, 1.0]),
            depth=np.array([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]]),
            u=u,
            v=np.zeros((2, 3)),
            name="sheared",
            path=Path("synthetic"),
        )
        out = window_shear(adcp, [0.5], [20.0], time_tolerance=2.0)
        assert out.S2[0] == pytest.approx((1.0 / 10.0) ** 2)
        assert out.n_support[0] == 2

    def test_support_count_is_exported(self):
        adcp = AdcpData(
            time=np.array([0.0, 1.0, 2.0]),
            depth=np.tile(np.array([10.0, 20.0, 30.0]), (3, 1)),
            u=np.array([[0.0, 1.0, 2.0], [0.0, 1.0, np.nan], [0.0, 1.0, 2.0]]),
            v=np.zeros((3, 3)),
            name="patchy",
            path=Path("synthetic"),
        )
        out = window_shear(adcp, [1.0], [20.0], time_tolerance=5.0)
        assert out.n_ens[0] == 3
        assert out.n_support[0] == 2  # the deep pair lost one ensemble
