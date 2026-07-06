# May-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for odas_tpw.perturb.plot.layout — pure-numpy helpers."""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from odas_tpw.perturb.plot import layout  # noqa: E402


class TestPanelGrid:
    """The shared ncols panel layout used by scalar and the profiles family."""

    def _close(self, fig):
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_full_grid_selects_left_and_bottom(self):
        # 4 panels x 2 columns -> full 2x2. Row-major fill: [0,1 / 2,3].
        fig, panels, left, col_bottom = layout.panel_grid(4, 2)
        assert len(panels) == 4
        assert all(ax.get_visible() for ax in panels)
        assert panels[0] in left and panels[2] in left      # left column
        assert panels[1] not in left and panels[3] not in left
        # bottom of each column: col0 -> panel 2, col1 -> panel 3
        assert set(map(id, col_bottom)) == {id(panels[2]), id(panels[3])}
        assert list(fig.get_size_inches()) == [11.0, 7.0]   # 3*nrows + 1, nrows=2
        self._close(fig)

    def test_ragged_last_row_blanks_one_cell(self):
        # 3 panels x 2 columns -> 2x2 with the 4th cell blanked; col1's bottom
        # is panel 1 (row 0), since (row 1, col 1) is empty.
        fig, panels, _left, col_bottom = layout.panel_grid(3, 2)
        assert len(panels) == 3
        assert set(map(id, col_bottom)) == {id(panels[2]), id(panels[1])}
        invisible = [ax for ax in fig.axes if not ax.get_visible()]
        assert len(invisible) == 1
        self._close(fig)

    def test_ncols_clamped_to_range(self):
        # ncols > n clamps to n; ncols < 1 clamps to 1 (a single column).
        fig, _panels, left, col_bottom = layout.panel_grid(2, 9)
        assert len(col_bottom) == 2 and len(left) == 1   # clamped to 2 columns
        self._close(fig)
        fig, _panels, left, col_bottom = layout.panel_grid(3, 0)
        assert len(left) == 3 and len(col_bottom) == 1   # clamped to 1 column
        assert list(fig.get_size_inches()) == [11.0, 10.0]  # 3*3 + 1
        self._close(fig)


class TestColumnHelpers:
    def test_clusters_uniform_is_one(self):
        x = np.arange(6.0)  # uniform spacing 1 -> single cluster
        assert layout.column_clusters(x) == [(0, 6)]

    def test_clusters_split_on_big_gap(self):
        x = np.array([0.0, 1.0, 2.0, 100.0, 101.0])  # gap 98 >> median 1
        assert layout.column_clusters(x) == [(0, 3), (3, 5)]

    def test_clusters_singleton(self):
        assert layout.column_clusters(np.array([5.0])) == [(0, 1)]

    def test_clusters_all_equal_x(self):
        # No positive spacing -> one cluster (don't divide by zero).
        assert layout.column_clusters(np.array([2.0, 2.0, 2.0])) == [(0, 3)]

    def test_column_edges_midpoints(self):
        e = layout.column_edges(np.array([0.0, 2.0, 4.0]))
        np.testing.assert_allclose(e, [-1.0, 1.0, 3.0, 5.0])
        assert e.size == 4

    def test_column_edges_singleton(self):
        np.testing.assert_allclose(layout.column_edges(np.array([3.0])), [2.5, 3.5])

    def test_strictly_increasing_breaks_ties(self):
        x = layout.strictly_increasing(np.array([0.0, 1.0, 1.0, 1.0, 5.0]))
        assert np.all(np.diff(x) > 0)        # tied casts nudged apart
        assert x[0] == 0.0 and x[-1] == 5.0  # endpoints untouched

    def test_strictly_increasing_all_equal(self):
        x = layout.strictly_increasing(np.array([2.0, 2.0, 2.0]))
        assert np.all(np.diff(x) > 0)


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


class TestFitColorbarLabels:
    """``fit_colorbar_labels`` keeps long colorbar labels inside their bars so
    stacked-panel plots don't show colliding colorbar labels."""

    LONG = (
        "FP07 thermistor temperature (probe 1) [degree_Celsius]",
        "FP07 thermistor temperature (probe 2) [degree_Celsius]",
        "buoyancy frequency squared (Thorpe-sorted) [s-2]",
        "background temperature gradient (positive down) [K m-1]",
    )

    @staticmethod
    def _stacked(labels):
        """A figure mirroring the real per-panel layout: N stacked axes, each
        with a vertical colorbar carrying one of *labels*."""
        import matplotlib.pyplot as plt

        n = len(labels)
        fig, axes = plt.subplots(
            n, 1, figsize=(11, 3.0 * n + 1.0),
            constrained_layout=True, squeeze=False,
        )
        for ax, lab in zip(axes[:, 0], labels):
            pcm = ax.pcolormesh([[1.0, 2.0], [3.0, 4.0]])
            fig.colorbar(pcm, ax=ax, label=lab)
        return fig

    @staticmethod
    def _cbar_axes(fig):
        return [ax for ax in fig.axes if getattr(ax, "_colorbar", None) is not None]

    def test_long_labels_overflow_their_bars_without_the_fix(self):
        """Sanity: the failing scenario actually reproduces — at least one label
        is taller than its bar before the helper runs (else the fix is untested)."""
        import matplotlib.pyplot as plt

        fig = self._stacked(self.LONG)
        try:
            fig.draw_without_rendering()
            overflow = [
                ax.yaxis.label.get_window_extent().height
                > ax.get_window_extent().height
                for ax in self._cbar_axes(fig)
            ]
            assert any(overflow)
        finally:
            plt.close(fig)

    def test_fit_removes_overflow_and_collisions(self):
        import matplotlib.pyplot as plt

        fig = self._stacked(self.LONG)
        try:
            layout.fit_colorbar_labels(fig)
            fig.draw_without_rendering()
            cbars = self._cbar_axes(fig)
            # No label taller than its own bar...
            for ax in cbars:
                bar_h = ax.get_window_extent().height
                label_h = ax.yaxis.label.get_window_extent().height
                assert label_h <= bar_h
            # ...and no two colorbar labels overlap.
            boxes = [ax.yaxis.label.get_window_extent() for ax in cbars]

            def _overlap(a, b):
                return not (a.x1 <= b.x0 or b.x1 <= a.x0
                            or a.y1 <= b.y0 or b.y1 <= a.y0)

            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    assert not _overlap(boxes[i], boxes[j])
            # Floor is respected.
            assert all(ax.yaxis.label.get_fontsize() >= 6.0 for ax in cbars)
        finally:
            plt.close(fig)

    def test_short_labels_are_untouched(self):
        """A label that already fits keeps its font size (the helper is a no-op,
        not a blanket shrink)."""
        import matplotlib.pyplot as plt

        fig = self._stacked(("T", "S"))
        try:
            before = [ax.yaxis.label.get_fontsize() for ax in self._cbar_axes(fig)]
            layout.fit_colorbar_labels(fig)
            after = [ax.yaxis.label.get_fontsize() for ax in self._cbar_axes(fig)]
            assert before == after
        finally:
            plt.close(fig)

    def test_floor_clamps_pathological_labels(self):
        """An absurdly tall label can't shrink below the readability floor."""
        import matplotlib.pyplot as plt

        huge = "x" * 400  # taller than any bar even at the floor font size
        fig = self._stacked((huge, huge))
        try:
            layout.fit_colorbar_labels(fig, min_fontsize=6.0)
            fonts = [ax.yaxis.label.get_fontsize() for ax in self._cbar_axes(fig)]
            assert all(abs(f - 6.0) < 1e-9 for f in fonts)
        finally:
            plt.close(fig)


class TestLatestStageDir:
    def test_picks_highest_versioned(self, tmp_path):
        import os

        for n in (0, 1, 2, 5):
            (tmp_path / f"diss_{n:02d}").mkdir()
        got = layout.latest_stage_dir(str(tmp_path), "diss")
        assert got is not None
        assert os.path.basename(got) == "diss_05"

    def test_legacy_fallback(self, tmp_path):
        import os

        (tmp_path / "diss").mkdir()
        got = layout.latest_stage_dir(str(tmp_path), "diss")
        assert got is not None
        assert os.path.basename(got) == "diss"

    def test_missing_returns_none(self, tmp_path):
        assert layout.latest_stage_dir(str(tmp_path), "diss") is None
