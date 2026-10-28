# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for processing.top_trim — top trimming."""

import numpy as np
import pytest

from odas_tpw.processing.top_trim import (
    attitude_trim_depth,
    compute_trim_depth,
    compute_trim_depths,
    inclinometer_is_usable,
)


class TestComputeTrimDepth:
    def test_high_variance_at_top(self):
        """Synthetic profile with high variance at top, low below."""
        n = 5000
        depth = np.linspace(0, 60, n)

        # High variance in top 10m, low below
        sh1 = np.random.randn(n) * 0.01
        top_mask = depth < 10.0
        sh1[top_mask] = np.random.randn(int(np.count_nonzero(top_mask))) * 10.0

        trim = compute_trim_depth(
            depth,
            {"sh1": sh1},
            dz=1.0,
            min_depth=1.0,
            max_depth=100.0,
            quantile=0.6,
        )
        assert trim is not None
        # Trim depth should be somewhere around 10m (where variance drops)
        assert 1.0 <= trim <= 20.0

    def test_quiet_top_noisy_middle_not_under_trimmed(self):
        """Audit #66: a quiet near-surface bin must not end the search early.

        Quiet 0-2 m, noisy (prop wash) 2-15 m, quiet below. The old
        first-bin-below-threshold logic stopped at the quiet 0-2 m cap and
        trimmed to ~1 m, leaving the 2-15 m noise in the profile. The fix
        must trim past the deepest noisy bin (~15 m).
        """
        rng = np.random.default_rng(0)
        n = 5000
        depth = np.linspace(0, 60, n)
        sh1 = rng.standard_normal(n) * 0.01
        noisy = (depth >= 2) & (depth < 15)
        sh1[noisy] = rng.standard_normal(int(noisy.sum())) * 10.0
        quiet_cap = depth < 2
        sh1[quiet_cap] = rng.standard_normal(int(quiet_cap.sum())) * 0.01

        trim = compute_trim_depth(
            depth, {"sh1": sh1}, dz=1.0, min_depth=1.0, max_depth=100.0, quantile=0.6
        )
        assert trim is not None
        # Must clear the deepest noisy bin (~15 m), not stop at the quiet cap.
        assert trim >= 14.0, f"under-trimmed to {trim} m; should clear the 2-15 m noise"
        assert trim <= 20.0

    def test_momentary_dip_in_noisy_top_not_under_trimmed(self):
        """Audit #66: a single quiet dip inside the prop wash must not stop it.

        Noisy 0-12 m with a one-bin quiet dip at 5-6 m. The trim must reach
        past 12 m, not stop at the 5-6 m dip.
        """
        rng = np.random.default_rng(1)
        n = 5000
        depth = np.linspace(0, 60, n)
        sh1 = rng.standard_normal(n) * 0.01
        sh1[depth < 12] = rng.standard_normal(int((depth < 12).sum())) * 10.0
        dip = (depth >= 5) & (depth < 6)
        sh1[dip] = rng.standard_normal(int(dip.sum())) * 0.01

        trim = compute_trim_depth(
            depth, {"sh1": sh1}, dz=1.0, min_depth=1.0, max_depth=100.0, quantile=0.6
        )
        assert trim is not None
        assert trim >= 12.0, f"dip at 5-6 m ended the search early (trim={trim} m)"
        assert trim <= 16.0

    def test_isolated_deep_transient_does_not_over_trim(self):
        """Audit r1-2: a deep transient detached from the surface prop wash
        must not drag the trim down through the quiet band above it.

        Surface prop wash at 1-2.5 m, a wide quiet band, then an isolated
        high-amplitude transient at 25-33 m on BOTH accelerometer axes (so the
        median combine cannot reject it — the real ARCTERX SN479_0026 prof-8
        case). The pre-fix deepest-elevated-bin rule trimmed to ~33 m,
        discarding valid 3-24 m data; the surface-attached rule trims at the
        surface run (~2.5 m).
        """
        rng = np.random.default_rng(10)
        n = 8000
        depth = np.linspace(0, 60, n)

        def accel():
            a = rng.standard_normal(n) * 0.01  # quiet background
            surf = (depth >= 1.0) & (depth < 2.5)  # surface prop wash
            a[surf] = rng.standard_normal(int(surf.sum())) * 5.0
            deep = (depth >= 25.0) & (depth < 33.0)  # isolated deep transient
            a[deep] = rng.standard_normal(int(deep.sum())) * 8.0
            return a

        trim = compute_trim_depth(
            depth, {"Ax": accel(), "Ay": accel()},
            dz=0.5, min_depth=1.0, max_depth=100.0, quantile=0.6,
        )
        assert trim is not None
        assert trim <= 5.0, (
            f"deep transient over-trimmed to {trim} m; should clear only the "
            "~2.5 m surface wash"
        )

    def test_detached_deep_only_channel_abstains(self):
        """A channel whose only elevated bins are deep (no surface wash) has
        already settled at the surface; it abstains rather than trimming deep."""
        rng = np.random.default_rng(11)
        n = 8000
        depth = np.linspace(0, 60, n)
        # Ax: genuine surface wash 1-3 m. Ay: quiet at surface, lone deep patch.
        ax = rng.standard_normal(n) * 0.01
        ax[(depth >= 1.0) & (depth < 3.0)] = rng.standard_normal(
            int(((depth >= 1.0) & (depth < 3.0)).sum())
        ) * 5.0
        ay = rng.standard_normal(n) * 0.01
        ay[(depth >= 30.0) & (depth < 36.0)] = rng.standard_normal(
            int(((depth >= 30.0) & (depth < 36.0)).sum())
        ) * 8.0
        trim = compute_trim_depth(
            depth, {"Ax": ax, "Ay": ay}, dz=0.5, min_depth=1.0, max_depth=100.0
        )
        # Ay abstains (no surface wash); Ax decides the ~3 m surface exit.
        assert trim is not None and trim <= 6.0, f"trim={trim} m"

    def test_median_combine_robust_to_one_deep_channel(self):
        """One channel elevated to depth must not drag the trim down.

        Two channels settle by ~6 m; a third stays 'elevated' all the way
        down (the failure mode of feeding shear, which tracks deep ocean
        turbulence). The median across channels ignores the outlier — a max
        combine would have followed it to the bottom.
        """
        rng = np.random.default_rng(2)
        n = 6000
        depth = np.linspace(0, 60, n)
        shallow = lambda: np.where(  # noqa: E731 - quiet below 6 m, loud above
            depth < 6.0, rng.standard_normal(n) * 5.0, rng.standard_normal(n) * 0.01
        )
        # Shear-like: loud prop-wash top AND a loud deep patch at 30-40 m, so its
        # own prop-wash exit lands at ~40 m.
        bad = rng.standard_normal(n) * 0.01
        bad[depth < 6.0] = rng.standard_normal(int((depth < 6.0).sum())) * 5.0
        patch = (depth >= 30.0) & (depth < 40.0)
        bad[patch] = rng.standard_normal(int(patch.sum())) * 5.0
        trim = compute_trim_depth(
            depth,
            {"Ax": shallow(), "Ay": shallow(), "bad": bad},
            dz=1.0, min_depth=1.0, max_depth=100.0,
        )
        assert trim is not None
        assert trim <= 9.0, f"deep outlier channel dragged trim to {trim} m"

    def test_dead_channel_dropped(self):
        """A flat / zero-variance channel (dead sensor) is ignored."""
        rng = np.random.default_rng(3)
        n = 6000
        depth = np.linspace(0, 60, n)
        good = np.where(depth < 6.0, rng.standard_normal(n) * 5.0, rng.standard_normal(n) * 0.01)
        dead = np.full(n, 1.234)  # constant -> zero per-bin std everywhere
        trim_with_dead = compute_trim_depth(
            depth, {"Ax": good, "Ay": dead}, dz=1.0, min_depth=1.0, max_depth=100.0
        )
        trim_alone = compute_trim_depth(
            depth, {"Ax": good}, dz=1.0, min_depth=1.0, max_depth=100.0
        )
        # The dead channel contributes nothing; result matches the live channel.
        assert trim_with_dead == trim_alone
        assert trim_with_dead is not None and trim_with_dead <= 9.0

    def test_invalid_quantile_raises(self):
        depth = np.linspace(0, 60, 100)
        sh1 = np.random.randn(100) * 0.01
        for bad in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="quantile"):
                compute_trim_depth(depth, {"sh1": sh1}, quantile=bad)

    def test_invalid_noise_factor_raises(self):
        depth = np.linspace(0, 60, 100)
        sh1 = np.random.randn(100) * 0.01
        for bad in (1.0, 0.5, 0.0):
            with pytest.raises(ValueError, match="noise_factor"):
                compute_trim_depth(depth, {"sh1": sh1}, noise_factor=bad)

    def test_invalid_max_gap_raises(self):
        depth = np.linspace(0, 60, 100)
        sh1 = np.random.randn(100) * 0.01
        with pytest.raises(ValueError, match="max_gap"):
            compute_trim_depth(depth, {"sh1": sh1}, max_gap=-1)

    def test_all_stable(self):
        """Uniform low variance — trim at first bin."""
        n = 5000
        depth = np.linspace(0, 60, n)
        sh1 = np.random.randn(n) * 0.01

        trim = compute_trim_depth(
            depth,
            {"sh1": sh1},
            dz=1.0,
            min_depth=1.0,
            max_depth=100.0,
            quantile=0.6,
        )
        # Should find a trim point near the beginning
        assert trim is not None
        assert trim <= 10.0

    def test_empty_channels(self):
        depth = np.linspace(0, 60, 100)
        trim = compute_trim_depth(depth, {})
        assert trim is None

    def test_bin_edges_too_few_returns_none(self):
        """min_depth==max_depth produces fewer than 2 bin edges."""
        depth = np.linspace(0, 10, 100)
        sh1 = np.random.randn(100) * 0.01
        # min_depth == max_depth → bin_edges has 1 entry → return None (line 67)
        trim = compute_trim_depth(
            depth, {"sh1": sh1}, dz=1.0, min_depth=10.0, max_depth=10.0
        )
        assert trim is None

    def test_mismatched_channel_length_skipped(self):
        """A channel whose length differs from depth is skipped (line 74)."""
        depth = np.linspace(0, 60, 200)
        # short_ch has length 50, mismatches depth (200) → skipped
        # Only short_ch is provided → all_stds stays empty → return None
        trim = compute_trim_depth(
            depth, {"short_ch": np.random.randn(50)},
            dz=1.0, min_depth=1.0, max_depth=100.0,
        )
        assert trim is None

    def test_too_few_valid_stds_skipped(self):
        """Channels with <3 valid bins are skipped (line 86)."""
        # Use a depth that yields only 2 bins, with 1-sample bins (no std)
        # min/max chosen so few bins; data so finite-std count < 3
        depth = np.array([5.0, 5.5, 6.0, 6.5])  # tiny range
        sh1 = np.array([0.0, 1.0, 0.0, 1.0])
        # Most bins will be empty or have <2 samples → finite stds < 3
        # All channels skipped → trim_depths empty → return None (line 95)
        trim = compute_trim_depth(
            depth, {"sh1": sh1}, dz=1.0, min_depth=1.0, max_depth=100.0,
        )
        assert trim is None


class TestComputeTrimDepths:
    def test_multiple_profiles(self):
        profiles_data = []
        for _ in range(3):
            n = 2000
            depth = np.linspace(0, 50, n)
            sh1 = np.random.randn(n) * 0.01
            sh1[depth < 5] = np.random.randn(int(np.count_nonzero(depth < 5))) * 5.0
            profiles_data.append({"depth_fast": depth, "channels": {"sh1": sh1}})

        results = compute_trim_depths(profiles_data, dz=1.0)
        assert len(results) == 3
        for r in results:
            assert r is not None


class TestCombine:
    """`combine` decides how heterogeneous channels vote.

    The case that motivated it: on a Rockland VMP, Ax/Ay are piezo VIBRATION
    sensors blind to prop wash (they settle within ~5 m), while the fall-rate
    residual sees it to ~30 m. Under a median the two blind channels outvote
    the one that can see, and the trim lands at the blind answer.
    """

    # NOTE the search range must be wide enough that the wash is a MINORITY
    # of it: the background is the `quantile` of the per-bin stds, so with
    # quantile=0.6 a wash spanning >40% of the range puts the background
    # INSIDE the wash and nothing reads as elevated. Hence max_depth=100 for
    # a 30 m wash, not 50.
    @staticmethod
    def _channels(n=12000, shallow_exit=5.0, deep_exit=30.0):
        rng = np.random.default_rng(0)
        depth = np.linspace(0, 120, n)
        def chan(exit_depth, amp=10.0):
            x = rng.normal(0, 0.01, n)
            m = depth < exit_depth
            x[m] = rng.normal(0, amp, int(m.sum()))
            return x
        return depth, {
            "Ax": chan(shallow_exit),          # blind: settles early
            "Ay": chan(shallow_exit),          # blind: settles early
            "W_residual": chan(deep_exit),     # sees the wash all the way down
        }

    def test_median_is_outvoted_by_the_blind_majority(self):
        depth, ch = self._channels()
        trim = compute_trim_depth(depth, ch, dz=1.0, min_depth=1.0, max_depth=100.0)
        assert trim is not None
        # two blind voters win: the trim sits near the shallow exit, NOT 30 m
        assert trim < 12.0, f"expected the blind majority to win, got {trim}"

    def test_max_lets_the_sensitive_channel_win(self):
        depth, ch = self._channels()
        trim = compute_trim_depth(
            depth, ch, dz=1.0, min_depth=1.0, max_depth=100.0, combine="max"
        )
        assert trim is not None
        assert 25.0 < trim < 40.0, f"expected the deep exit to win, got {trim}"

    def test_max_and_median_agree_for_homogeneous_channels(self):
        """With redundant channels the two rules must not differ."""
        depth, ch = self._channels(shallow_exit=8.0, deep_exit=8.0)
        a = compute_trim_depth(depth, ch, dz=1.0, min_depth=1.0, max_depth=100.0)
        b = compute_trim_depth(
            depth, ch, dz=1.0, min_depth=1.0, max_depth=100.0, combine="max"
        )
        assert a is not None and b is not None
        assert abs(a - b) <= 2.0

    def test_default_is_median_backward_compatible(self):
        depth, ch = self._channels()
        assert compute_trim_depth(
            depth, ch, dz=1.0, min_depth=1.0, max_depth=100.0
        ) == compute_trim_depth(
            depth, ch, dz=1.0, min_depth=1.0, max_depth=100.0, combine="median"
        )

    def test_rejects_an_unknown_rule(self):
        depth, ch = self._channels()
        with pytest.raises(ValueError, match="combine must be"):
            compute_trim_depth(depth, ch, dz=1.0, combine="mean")


class TestAttitudeTrimDepth:
    """The inclinometer voter for a tow-yo VMP.

    In a tow-yo the VMP is launched nearly parallel to the surface and
    pitches down as it falls, so until it is vertical the shear probes see a
    mean cross-flow and epsilon (~ shear^2/U^4) is meaningless. Measured on
    SUNRISE 2021 Walton Smith SN 194, 773 descents: Incl_Y starts at a median
    20.2 deg and reaches 85 deg at a median 3.9 m, spread 2.7-5.7 m.
    """

    @staticmethod
    def _cast(vertical_at=4.0, start_deg=20.0, z_max=22.0, n=1400):
        """Depth and Incl_Y for a cast that comes vertical at `vertical_at`."""
        z = np.linspace(0.1, z_max, n)
        # Smooth pitch-down from start_deg to ~89 deg, reaching 85 at
        # vertical_at by construction.
        frac = np.clip(z / vertical_at, 0.0, 1.0)
        y = start_deg + (85.0 - start_deg) * frac
        y[z > vertical_at] = 85.0 + 4.0 * (1.0 - np.exp(-(z[z > vertical_at] - vertical_at)))
        return z, y

    def test_finds_the_vertical_depth_plus_margin(self):
        z, y = self._cast(vertical_at=4.0)
        out = attitude_trim_depth(z, y, margin_m=2.0)
        assert out is not None
        assert out == pytest.approx(6.0, abs=0.2)

    def test_margin_zero_returns_the_vertical_depth(self):
        z, y = self._cast(vertical_at=3.5)
        out = attitude_trim_depth(z, y, margin_m=0.0)
        assert out == pytest.approx(3.5, abs=0.2)

    def test_already_vertical_returns_none(self):
        """A conventional lowered cast has no attitude transient and must not
        pay the margin."""
        z = np.linspace(0.1, 200.0, 2000)
        y = np.full_like(z, 89.5)
        assert attitude_trim_depth(z, y) is None

    def test_never_vertical_returns_none(self):
        """A cast that tumbles the whole way has no clean part to keep."""
        z = np.linspace(0.1, 22.0, 1400)
        y = np.linspace(10.0, 40.0, z.size)
        assert attitude_trim_depth(z, y) is None

    def test_momentary_swing_through_vertical_is_not_accepted(self):
        """A tumbling instrument that flicks past 85 deg must not end the
        search: the hold requires it to STAY there."""
        z = np.linspace(0.1, 22.0, 1400)
        y = np.full_like(z, 30.0)
        spike = (z > 2.0) & (z < 2.2)      # brief swing, well under hold_m
        y[spike] = 87.0
        y[z > 10.0] = 88.0                 # genuinely vertical from 10 m
        out = attitude_trim_depth(z, y, hold_m=1.0, margin_m=0.0)
        assert out == pytest.approx(10.0, abs=0.3)

    def test_relax_tolerates_inclinometer_noise_during_the_hold(self):
        z, y = self._cast(vertical_at=4.0)
        rng = np.random.default_rng(0)
        y = y + rng.normal(0.0, 1.0, y.size)     # within the 5 deg relax
        out = attitude_trim_depth(z, y, margin_m=0.0)
        assert out is not None and out < 6.0

    def test_deeper_vertical_depth_gives_a_deeper_trim(self):
        """The whole point: the per-cast answer tracks the cast, where a flat
        floor could not (SUNRISE spread was 2.7-5.7 m)."""
        shallow = attitude_trim_depth(*self._cast(vertical_at=2.7), margin_m=2.0)
        deep = attitude_trim_depth(*self._cast(vertical_at=5.7), margin_m=2.0)
        assert shallow is not None and deep is not None
        assert deep - shallow == pytest.approx(3.0, abs=0.4)

    def test_value_is_literal_not_absolute(self):
        """An inverted instrument must fail the test, not be hidden by abs()."""
        z, y = self._cast(vertical_at=4.0)
        assert attitude_trim_depth(z, -y) is None

    def test_empty_and_mismatched_inputs_return_none(self):
        assert attitude_trim_depth(np.array([]), np.array([])) is None
        assert attitude_trim_depth(np.linspace(0, 10, 50), np.zeros(20)) is None

    def test_all_nan_returns_none(self):
        z = np.linspace(0.1, 22.0, 100)
        assert attitude_trim_depth(z, np.full_like(z, np.nan)) is None

    def test_nan_samples_are_skipped(self):
        z, y = self._cast(vertical_at=4.0)
        y = y.copy()
        y[::7] = np.nan
        out = attitude_trim_depth(z, y, margin_m=0.0)
        assert out == pytest.approx(4.0, abs=0.3)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"hold_m": -1.0}, "hold_m must be >= 0"),
            ({"margin_m": -1.0}, "margin_m must be >= 0"),
            ({"relax_deg": -1.0}, "relax_deg must be >= 0"),
        ],
    )
    def test_rejects_negative_parameters(self, kwargs, match):
        z, y = self._cast()
        with pytest.raises(ValueError, match=match):
            attitude_trim_depth(z, y, **kwargs)


class TestInclinometerIsUsable:
    """Frozen-channel detection.

    VMP SN 412 froze both inclinometer channels in SUNRISE 2021 and 2022:
    median within-cast range 0.0 deg against 64-78 deg on the healthy SN 142
    and SN 194, while still varying BETWEEN casts (whole-file span 6.3-90.0
    deg over 3033 distinct values). Only a per-cast test catches that.
    """

    def test_healthy_towyo_channel_passes(self):
        y = np.linspace(20.0, 88.0, 1400)
        assert inclinometer_is_usable(y) is True

    def test_frozen_channel_fails(self):
        assert inclinometer_is_usable(np.full(1400, 42.0)) is False

    def test_frozen_HIGH_channel_fails(self):
        """The dangerous one: frozen above the vertical threshold, which
        would otherwise be read as 'already vertical' and trim nothing."""
        assert inclinometer_is_usable(np.full(1400, 89.0)) is False

    def test_sn412_pattern_varies_between_casts_but_not_within(self):
        """Whole-file range looks healthy; each cast is frozen. The per-cast
        test must reject every cast, and a whole-file view must not be used."""
        casts = [np.full(1400, v) for v in (6.3, 30.0, 55.0, 90.0)]
        assert all(inclinometer_is_usable(c) is False for c in casts)
        whole_file = np.concatenate(casts)
        assert float(np.ptp(whole_file)) > 80.0        # would pass a file test
        assert inclinometer_is_usable(whole_file) is True   # ... and it does

    def test_dither_on_a_steady_vertical_cast_passes(self):
        rng = np.random.default_rng(1)
        assert inclinometer_is_usable(89.0 + rng.normal(0, 0.3, 1400)) is True

    def test_out_of_physical_range_fails(self):
        assert inclinometer_is_usable(np.linspace(500.0, 600.0, 100)) is False

    def test_mostly_nan_fails(self):
        y = np.full(1000, np.nan)
        y[:100] = np.linspace(20.0, 88.0, 100)
        assert inclinometer_is_usable(y) is False

    def test_empty_fails(self):
        assert inclinometer_is_usable(np.array([])) is False


class TestAttitudeTrimRefusesFrozenChannels:
    def test_frozen_high_does_not_masquerade_as_already_vertical(self):
        """Regression: without the guard this returned None, the caller read
        it as 'nothing to trim', and a tow-yo cast kept its worthless first
        metres."""
        z = np.linspace(0.1, 22.0, 1400)
        assert attitude_trim_depth(z, np.full_like(z, 89.0)) is None
        assert inclinometer_is_usable(np.full_like(z, 89.0)) is False

    def test_frozen_low_is_also_refused(self):
        z = np.linspace(0.1, 22.0, 1400)
        assert attitude_trim_depth(z, np.full_like(z, 20.0)) is None

    def test_a_live_channel_still_trims(self):
        z = np.linspace(0.1, 22.0, 1400)
        y = np.clip(20.0 + (85.0 - 20.0) * (z / 4.0), None, 88.0)
        assert attitude_trim_depth(z, y, margin_m=0.0) == pytest.approx(4.0, abs=0.3)
