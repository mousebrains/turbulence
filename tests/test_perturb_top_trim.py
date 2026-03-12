# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb.top_trim — top trimming."""

import numpy as np

from perturb.top_trim import compute_trim_depth, compute_trim_depths


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
            depth, {"sh1": sh1},
            dz=1.0, min_depth=1.0, max_depth=50.0, quantile=0.6,
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
            depth, {"sh1": sh1},
            dz=1.0, min_depth=1.0, max_depth=50.0, quantile=0.6,
        )
        # Should find a trim point near the beginning
        assert trim is not None
        assert trim <= 10.0

    def test_empty_channels(self):
        depth = np.linspace(0, 60, 100)
        trim = compute_trim_depth(depth, {})
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
