# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for processing.top_trim — top trimming."""

import numpy as np

from odas_tpw.processing.top_trim import compute_trim_depth, compute_trim_depths


class TestComputeTrimDepth:
    def test_high_variance_at_top(self):
        """Synthetic profile with high variance at top, low below."""
        n = 5000
        depth = np.linspace(0, 60, n)

        # High variance in top 10m, low below
        sh1 = np.random.randn(n) * 0.01
        top_mask = depth < 10.0
        sh1[top_mask] = np.random.randn(np.sum(top_mask)) * 10.0

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
            sh1[depth < 5] = np.random.randn(np.sum(depth < 5)) * 5.0
            profiles_data.append({"depth_fast": depth, "channels": {"sh1": sh1}})

        results = compute_trim_depths(profiles_data, dz=1.0)
        assert len(results) == 3
        for r in results:
            assert r is not None
