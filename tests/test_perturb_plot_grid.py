# Jun-2026, Claude and Pat Welch, pat@mousebrains.com
"""Unit tests for the section scatter-to-grid binning (plot/grid.py)."""

from __future__ import annotations

import numpy as np
import pytest

from odas_tpw.perturb.plot import grid


def test_make_edges_regular():
    e = grid.make_edges(0.0, 10.0, 2.0)
    assert np.allclose(e, [0, 2, 4, 6, 8, 10])


def test_make_edges_covers_non_multiple_span():
    e = grid.make_edges(0.0, 9.0, 2.0)
    assert e[0] == 0.0 and e[-1] >= 9.0  # last edge spans past the data


def test_make_edges_degenerate_span():
    e = grid.make_edges(5.0, 5.0, 1.0)
    assert len(e) == 2 and e[1] > e[0]


def test_make_edges_last_edge_exceeds_max_on_even_divide():
    """An evenly-dividing span nudges the top edge just past the max so a
    sample sitting exactly on it (the bottom turn) is not dropped."""
    e = grid.make_edges(0.0, 10.0, 2.0)
    assert e[-1] > 10.0
    assert e[-1] == pytest.approx(10.0)  # only by an ULP


def test_grid_mean_includes_sample_on_max_edge():
    """A sample at exactly the data max lands in the last bin, not dropped."""
    z_edges = grid.make_edges(0.0, 100.0, 1.0)  # even divide -> nudged
    x_edges = grid.make_edges(0.0, 2.0, 1.0)
    g, c = grid.grid_mean(
        np.array([2.0]), np.array([100.0]), np.array([7.0]), x_edges, z_edges
    )
    assert c.sum() == 1  # the boundary sample was binned
    assert g[-1, -1] == pytest.approx(7.0)  # deepest row, rightmost column


def test_grid_mean_averages_cell_and_marks_empty():
    x = np.array([0.5, 0.5, 1.5])
    z = np.array([0.5, 0.5, 0.5])
    v = np.array([1.0, 3.0, 10.0])
    x_edges = np.array([0.0, 1.0, 2.0])
    z_edges = np.array([0.0, 1.0, 2.0])  # row 1 stays empty
    g, c = grid.grid_mean(x, z, v, x_edges, z_edges)
    assert g.shape == (2, 2)
    assert g[0, 0] == pytest.approx(2.0)  # mean(1, 3)
    assert g[0, 1] == pytest.approx(10.0)
    assert np.isnan(g[1, 0]) and np.isnan(g[1, 1])  # unsampled
    assert c[0, 0] == 2 and c[0, 1] == 1 and c[1, 0] == 0


def test_grid_mean_drops_nonfinite_samples():
    x = np.array([0.5, 0.5, np.nan, 0.5])
    z = np.array([0.5, 0.5, 0.5, 0.5])
    v = np.array([2.0, np.nan, 5.0, 4.0])  # one NaN value, one NaN position
    g, c = grid.grid_mean(x, z, v, np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert g[0, 0] == pytest.approx(3.0)  # mean(2, 4); NaNs excluded
    assert c[0, 0] == 2


def test_grid_mean_out_of_range_dropped():
    x = np.array([5.0])  # outside [0, 2]
    z = np.array([0.5])
    v = np.array([9.0])
    g, c = grid.grid_mean(x, z, v, np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]))
    assert not np.any(np.isfinite(g)) and c.sum() == 0


def test_linear_limits_keeps_negatives():
    z = np.linspace(-5.0, 5.0, 1001)
    lo, hi = grid.linear_limits(z)
    assert lo < 0 < hi
    assert lo == pytest.approx(-4.9, abs=0.05)
    assert hi == pytest.approx(4.9, abs=0.05)


def test_linear_limits_all_nan_returns_overrides():
    z = np.full(10, np.nan)
    assert grid.linear_limits(z, lo=1.0, hi=2.0) == (1.0, 2.0)
    assert grid.linear_limits(z) == (None, None)


def test_auto_step_targets_column_count():
    assert grid.auto_step(0.0, 300.0, target=300) == pytest.approx(1.0)
    assert grid.auto_step(0.0, 0.0) == 1.0  # degenerate fallback
