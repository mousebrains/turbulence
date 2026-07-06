# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for processing.top_trim — top trimming."""

import numpy as np
import pytest

from odas_tpw.processing.top_trim import compute_trim_depth, compute_trim_depths


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
            max_depth=50.0,
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
            depth, {"sh1": sh1}, dz=1.0, min_depth=1.0, max_depth=50.0, quantile=0.6
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
            depth, {"sh1": sh1}, dz=1.0, min_depth=1.0, max_depth=50.0, quantile=0.6
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
            dz=0.5, min_depth=1.0, max_depth=50.0, quantile=0.6,
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
            depth, {"Ax": ax, "Ay": ay}, dz=0.5, min_depth=1.0, max_depth=50.0
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
            dz=1.0, min_depth=1.0, max_depth=50.0,
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
            depth, {"Ax": good, "Ay": dead}, dz=1.0, min_depth=1.0, max_depth=50.0
        )
        trim_alone = compute_trim_depth(
            depth, {"Ax": good}, dz=1.0, min_depth=1.0, max_depth=50.0
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
            max_depth=50.0,
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
            dz=1.0, min_depth=1.0, max_depth=50.0,
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
            depth, {"sh1": sh1}, dz=1.0, min_depth=1.0, max_depth=50.0,
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
