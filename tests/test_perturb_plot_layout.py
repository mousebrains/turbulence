# May-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for odas_tpw.perturb.plot.layout — pure-numpy helpers."""

from __future__ import annotations

import numpy as np

from odas_tpw.perturb.plot import layout


class TestDepthEdges:
    def test_uniform_grid(self):
        d = np.array([1.0, 2.0, 3.0, 4.0])
        e = layout.depth_edges(d)
        np.testing.assert_allclose(e, [0.5, 1.5, 2.5, 3.5, 4.5])
        assert e.size == d.size + 1

    def test_singleton_falls_back(self):
        d = np.array([5.0])
        e = layout.depth_edges(d)
        np.testing.assert_allclose(e, [4.5, 5.5])

    def test_non_uniform_uses_median_spacing(self):
        d = np.array([0.0, 1.0, 2.0, 5.0])  # diffs 1,1,3 → median 1
        e = layout.depth_edges(d)
        np.testing.assert_allclose(e, [-0.5, 0.5, 1.5, 4.5, 5.5])


class TestQuantileLimits:
    def test_inner_quantile(self):
        z = np.linspace(1.0, 100.0, 100)
        lo, hi = layout.quantile_limits(z, q_lo=0.1, q_hi=0.9)
        assert 9.0 <= lo <= 12.0
        assert 88.0 <= hi <= 92.0

    def test_user_lo_hi_override(self):
        z = np.linspace(1.0, 100.0, 100)
        assert layout.quantile_limits(z, lo=0.5)[0] == 0.5
        assert layout.quantile_limits(z, hi=200.0)[1] == 200.0

    def test_no_finite_values(self):
        z = np.array([np.nan, np.nan, -1.0])
        assert layout.quantile_limits(z) == (None, None)

    def test_drops_zero_and_negative(self):
        z = np.array([0.0, -1.0, 1.0, 10.0, 100.0])
        lo, hi = layout.quantile_limits(z, q_lo=0.0, q_hi=1.0)
        assert lo == 1.0
        assert hi == 100.0


class TestFFillDown:
    def test_internal_nan_filled(self):
        z = np.array([
            [1.0, 1.0],
            [np.nan, 2.0],
            [3.0, 2.0],
        ])
        out = layout.ffill_down(z)
        # Internal NaN at (1, 0) gets the value from row 0.
        assert out[1, 0] == 1.0
        assert out[2, 0] == 3.0

    def test_leading_nan_preserved(self):
        z = np.array([
            [np.nan, 1.0],
            [1.0, 2.0],
        ])
        out = layout.ffill_down(z)
        # Leading NaN at top of column 0 stays NaN — colors don't extend up.
        assert np.isnan(out[0, 0])

    def test_trailing_nan_preserved(self):
        z = np.array([
            [1.0, 1.0],
            [np.nan, 2.0],
        ])
        out = layout.ffill_down(z)
        # Bottom-most entry has no value below — stays NaN.
        assert np.isnan(out[1, 0])

    def test_does_not_modify_input(self):
        z = np.array([[1.0], [np.nan], [2.0]])
        before = z.copy()
        layout.ffill_down(z)
        np.testing.assert_array_equal(z, before)


class TestSplitSegments:
    def test_no_gap_one_segment(self):
        t = np.arange(5).astype("datetime64[s]")
        segs = layout.split_segments(t, gap_seconds=10)
        assert segs == [(0, 5)]

    def test_single_gap(self):
        t = np.array([0, 1, 2, 100, 101]).astype("datetime64[s]")
        segs = layout.split_segments(t, gap_seconds=10)
        assert segs == [(0, 3), (3, 5)]

    def test_gap_threshold_inclusive_below(self):
        t = np.array([0, 10]).astype("datetime64[s]")
        # diff=10, threshold=10 → no split (gap must exceed threshold).
        assert layout.split_segments(t, gap_seconds=10) == [(0, 2)]


class TestComputeLayout:
    def test_single_cluster_positions_are_arange(self):
        t = np.arange(4).astype("datetime64[s]")
        cast_x, segs, centers, t_starts, t_ends = layout.compute_layout(
            t, gap_seconds=10
        )
        np.testing.assert_allclose(cast_x, [0, 1, 2, 3])
        assert segs == [(0, 4)]
        np.testing.assert_allclose(centers, [1.5])
        assert len(t_starts) == 1
        assert len(t_ends) == 1

    def test_two_clusters_have_visual_gap(self):
        t = np.array([0, 1, 100, 101]).astype("datetime64[s]")
        cast_x, segs, _c, _s, _e = layout.compute_layout(
            t, gap_seconds=10, cluster_gap=0.5,
        )
        # First cluster {0, 1}, gap, second {2.5, 3.5}.
        np.testing.assert_allclose(cast_x, [0, 1, 2.5, 3.5])
        assert segs == [(0, 2), (2, 4)]


class TestLatestStageDir:
    def test_picks_highest_versioned(self, tmp_path):
        for n in (0, 1, 2, 5):
            (tmp_path / f"diss_{n:02d}").mkdir()
        got = layout.latest_stage_dir(str(tmp_path), "diss")
        assert got is not None and got.endswith("/diss_05")

    def test_legacy_fallback(self, tmp_path):
        (tmp_path / "diss").mkdir()
        got = layout.latest_stage_dir(str(tmp_path), "diss")
        assert got is not None and got.endswith("/diss")

    def test_missing_returns_none(self, tmp_path):
        assert layout.latest_stage_dir(str(tmp_path), "diss") is None
