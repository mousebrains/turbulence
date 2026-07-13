# May-2026, Claude and Pat Welch, pat@mousebrains.com
"""Shared depth-by-cast pcolor helpers used by perturb-plot subcommands.

Pure-numpy / pure-matplotlib utilities — no perturb config, no NetCDF I/O.
Suitable for any 2-D depth-by-cast field a subcommand wants to render.
"""

from __future__ import annotations

import glob
import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def panel_grid(
    n: int,
    ncols: int = 1,
    *,
    figsize: tuple[float, float] | list[float] | None = None,
) -> tuple[Figure, list[Axes], list[Axes], list[Axes]]:
    """Create a figure of *n* panels sharing one x and one y axis.

    Panels are arranged in ``ceil(n/ncols)`` rows by ``ncols`` columns, filled
    row-major (left-to-right, top-to-bottom). ``ncols`` is clamped to
    ``[1, n]``; the default (1) is the classic single vertical stack. Unused
    cells of a ragged last row are blanked. The default figure size scales with
    the grid: ``max(11, 5.5*ncols)`` wide by ``3*nrows + 1`` tall (so ``ncols=1``
    reproduces the historical ``(11, 3n+1)``).

    Returns ``(fig, panels, left, col_bottom)``:

    * ``panels``: the *n* panel Axes, in variable order.
    * ``left``: the left-column Axes. They carry the shared depth label; the
      other columns' y tick labels are hidden by ``sharey``, so a y label there
      would read as unlabeled numbers.
    * ``col_bottom``: the bottom-most panel of each column. They carry the x
      label; their x tick labels are re-enabled here so a ragged short column
      still shows its axis (``sharex`` otherwise hides all but the last row).

    The caller inverts the shared depth axis on ``panels[0]``, fills each panel,
    sets the y label on ``left`` and the x label/formatter on ``col_bottom``.
    """
    import matplotlib.pyplot as plt

    ncols = max(1, min(int(ncols or 1), n))
    nrows = (n + ncols - 1) // ncols
    fig, axes2d = plt.subplots(
        nrows, ncols,
        figsize=tuple(figsize) if figsize else (max(11.0, 5.5 * ncols), 3.0 * nrows + 1.0),
        sharex=True, sharey=True, constrained_layout=True, squeeze=False,
    )
    axes = list(axes2d.ravel())  # row-major
    for ax in axes[n:]:
        ax.set_visible(False)  # blank the unused cells of a ragged last row
    panels = axes[:n]
    left = [ax for i, ax in enumerate(panels) if i % ncols == 0]
    bottom_by_col: dict[int, Axes] = {}
    for i, ax in enumerate(panels):
        bottom_by_col[i % ncols] = ax  # later rows overwrite -> lowest row wins
    col_bottom = list(bottom_by_col.values())
    for ax in col_bottom:
        ax.xaxis.set_tick_params(labelbottom=True)  # re-show ticks hidden by sharex
    return fig, panels, left, col_bottom


def latest_stage_dir(root: str, prefix: str) -> str | None:
    """Path to the highest-numbered ``<prefix>_NN/`` dir under *root*.

    Falls back to the un-versioned ``<prefix>/`` dir if no versioned
    siblings exist. Returns ``None`` if neither is present.
    """
    versioned = sorted(glob.glob(os.path.join(root, f"{prefix}_[0-9][0-9]")))
    if versioned:
        return versioned[-1]
    legacy = os.path.join(root, prefix)
    return legacy if os.path.isdir(legacy) else None


def depth_edges(d: np.ndarray) -> np.ndarray:
    """Center-spaced edges around 1-D depth bin centers.

    Half-bin spacing is the median of ``diff(d)``; for a singleton bin we
    fall back to 0.5 m so the colorbar still renders something visible.
    """
    half = 0.5 if len(d) < 2 else float(np.median(np.diff(d))) / 2
    return np.concatenate((d - half, [d[-1] + half]))


def quantile_limits(
    z: np.ndarray,
    lo: float | None = None,
    hi: float | None = None,
    q_lo: float = 0.01,
    q_hi: float = 0.99,
) -> tuple[float | None, float | None]:
    """Inner 1/99 quantile of finite, positive values; CLI overrides win."""
    finite = z[np.isfinite(z) & (z > 0)]
    if finite.size == 0:
        return lo, hi
    q = np.quantile(finite, [q_lo, q_hi])
    return (
        lo if lo is not None else float(q[0]),
        hi if hi is not None else float(q[1]),
    )


def ffill_down(z: np.ndarray, max_gap: int = 1) -> np.ndarray:
    """Forward-fill internal NaNs along axis 0 (depth) within each column.

    Leading NaNs (above the shallowest valid sample) and trailing NaNs
    (below the deepest valid sample) are left as NaN — colors don't extend
    past either end of a cast. Used so the pcolor reads as a continuous
    profile when there are scattered missing bins inside the cast envelope.

    The fill is bounded by the length of the whole internal NaN run (default
    ``max_gap=1``): a run is bridged only if its entire length is ``<= max_gap``,
    otherwise it is left as a visible gap. So an isolated missing bin is
    cosmetically bridged, but a longer run — e.g. a QC-rejected / pipeline-NaN'd
    region (``drop_action='nan'``) — is left untouched rather than having its
    first bins repainted with the shallower bin's value. Gating on the run
    length (not the distance below the last valid sample) is what keeps the
    *first* bin of a multi-bin dropped block from being silently repainted as if
    it had passed.
    """
    z = z.copy()
    n = z.shape[0]
    rows = np.arange(n)[:, None]
    valid = np.isfinite(z)
    # Nearest valid bin above (>=0) and below (index, else n) each cell.
    last_above = np.maximum.accumulate(np.where(valid, rows, -1), axis=0)
    idx_below = np.where(valid, rows, n)
    first_below = np.minimum.accumulate(idx_below[::-1], axis=0)[::-1]
    has_below = first_below < n
    # Length of the internal NaN run a cell belongs to (all NaNs in one run
    # share last_above/first_below, so they get one run_len and fill together).
    run_len = first_below - last_above - 1
    fill = ~valid & (last_above >= 0) & has_below & (run_len <= max_gap)
    cols = np.broadcast_to(np.arange(z.shape[1]), z.shape)
    z[fill] = z[last_above[fill], cols[fill]]
    return z


def split_segments(t: np.ndarray, gap_seconds: float = 600) -> list[tuple[int, int]]:
    """Indices ``[start, end)`` for runs of profiles separated by < gap_seconds."""
    t_sec = t.astype("datetime64[s]").astype(np.int64)
    breaks = np.where(np.diff(t_sec) > gap_seconds)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks + 1, [len(t)]))
    return list(zip(starts.tolist(), ends.tolist()))


def compute_layout(
    t: np.ndarray,
    gap_seconds: float = 600,
    cluster_gap: float = 0.5,
) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray, list, list]:
    """Cast-index x positions with a small visual gap between time clusters.

    Returns ``(cast_x, segments, centers, t_starts, t_ends)`` where
    ``cast_x`` is the float x-position of every profile, ``segments`` are
    the half-open index ranges of each cluster, and the remaining tuples
    annotate cluster centers / start / end timestamps for axis labels.
    """
    segments = split_segments(t, gap_seconds)
    cast_x = np.zeros(len(t))
    pos = 0.0
    centers: list[float] = []
    t_starts: list = []
    t_ends: list = []
    for k, (s, e) in enumerate(segments):
        if k > 0:
            pos += cluster_gap
        n = e - s
        cast_x[s:e] = pos + np.arange(n)
        centers.append(pos + (n - 1) / 2)
        t_starts.append(t[s])
        t_ends.append(t[e - 1])
        pos += n
    return cast_x, segments, np.asarray(centers), t_starts, t_ends


def plot_panel(
    ax,
    fig,
    cast_x: np.ndarray,
    segments: list[tuple[int, int]],
    depth: np.ndarray,
    z: np.ndarray,
    cmap,
    norm,
    cbar_label: str,
):
    """Pcolor a depth-by-cast field on *ax*, one mesh per cluster.

    Forward-fills internal-NaN holes. Returns the last
    ``QuadMesh`` (or ``None`` when every cluster was empty).
    """
    z = ffill_down(z)
    d_edges = depth_edges(depth)
    pcm = None
    for s, e in segments:
        x_seg = cast_x[s:e]
        if len(x_seg) == 1:
            edges = np.array([x_seg[0] - 0.5, x_seg[0] + 0.5])
        else:
            mids = 0.5 * (x_seg[:-1] + x_seg[1:])
            edges = np.concatenate(([x_seg[0] - 0.5], mids, [x_seg[-1] + 0.5]))
        pcm = ax.pcolormesh(
            edges, d_edges, z[:, s:e],
            cmap=cmap, norm=norm, shading="flat",
        )
    if pcm is not None:
        fig.colorbar(pcm, ax=ax, label=cbar_label)
    return pcm


def column_clusters(x: np.ndarray, gap_factor: float = 4.0) -> list[tuple[int, int]]:
    """Group sorted-ascending column positions *x* into clusters by x-gaps.

    A break is inserted wherever the gap between adjacent columns exceeds
    ``gap_factor`` times the median column spacing.  Returns half-open
    ``[start, end)`` index ranges.  Keeps sparse/irregular casts honest: each
    cluster is drawn as one mesh and the gaps between clusters stay blank
    rather than being stretched across unsampled water/time.
    """
    n = len(x)
    if n <= 1:
        return [(0, n)]
    d = np.diff(x)
    pos = d[d > 0]
    med = float(np.median(pos)) if pos.size else 0.0
    if med <= 0:
        return [(0, n)]
    breaks = np.where(d > gap_factor * med)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks + 1, [n]))
    return list(zip(starts.tolist(), ends.tolist()))


def strictly_increasing(x: np.ndarray) -> np.ndarray:
    """Nudge tied/non-increasing sorted *x* up by a sub-spacing epsilon.

    pcolormesh needs strictly monotonic edges; two casts at the same x (a
    re-occupied station on a lat/longitude/distance axis, or two casts equally
    far from a reference point) would otherwise collapse to zero-width cells.
    The nudge (1e-3 of the median spacing) renders them as adjacent thin
    columns instead of vanishing.
    """
    x = np.array(x, dtype=float)
    if x.size < 2:
        return x
    d = np.diff(x)
    pos = d[d > 0]
    eps = (float(np.median(pos)) if pos.size else 1.0) * 1.0e-3
    for i in range(1, x.size):
        if x[i] <= x[i - 1]:
            x[i] = x[i - 1] + eps
    return x


def column_edges(x: np.ndarray) -> np.ndarray:
    """Pcolor x-edges for columns centered at sorted-ascending positions *x*.

    Interior edges are midpoints; the outer edges extend by half the adjacent
    spacing so the first/last column matches its neighbor's width.  A single
    column gets a half-unit width on each side.
    """
    x = np.asarray(x, dtype=float)
    if len(x) == 1:
        return np.array([x[0] - 0.5, x[0] + 0.5])
    mids = 0.5 * (x[:-1] + x[1:])
    first = x[0] - (mids[0] - x[0])
    last = x[-1] + (x[-1] - mids[-1])
    return np.concatenate(([first], mids, [last]))


def fit_colorbar_labels(
    fig, *, min_fontsize: float = 6.0, fit_frac: float = 0.95
) -> None:
    """Shrink over-long colorbar labels so each fits within its own bar.

    A verbose ``long_name [units]`` label, rotated vertically onto a short
    per-panel colorbar, can be *taller* than the colorbar bar itself.
    matplotlib then centers it on the bar and it overflows above/below into the
    neighboring panel's colorbar label.  ``constrained_layout`` reserves
    horizontal width for the rotated label but does not police this vertical
    collision, so stacked panels (``profiles``/``scalar``/``eps_chi``) show
    colliding colorbar labels.

    We settle the layout (``draw_without_rendering``), measure each colorbar
    label against its bar along the bar's long axis, and scale the font *down*
    to fit within ``fit_frac`` of the bar — never up — with a readability floor.
    The sub-1 ``fit_frac`` leaves headroom for the savefig re-draw: text height
    is not perfectly linear in font size and shrinking fonts frees layout space,
    so the final bar geometry shifts slightly; the margin keeps the fit stable
    rather than letting a 1-px residual creep back over the edge.  A no-op for
    labels that already fit.
    """
    fig.draw_without_rendering()  # settle constrained_layout; cache a renderer
    for ax in fig.axes:
        cbar = getattr(ax, "_colorbar", None)
        if cbar is None:
            continue
        vertical = getattr(cbar, "orientation", "vertical") == "vertical"
        label = ax.yaxis.label if vertical else ax.xaxis.label
        if not label.get_text():
            continue
        bar = ax.get_window_extent()  # no renderer arg: uses fig cached renderer
        avail = (bar.height if vertical else bar.width) * fit_frac
        need_box = label.get_window_extent()
        need = need_box.height if vertical else need_box.width
        if need > avail > 0:
            scaled = label.get_fontsize() * avail / need
            label.set_fontsize(max(scaled, min_fontsize))


def plot_columns(
    ax,
    fig,
    x: np.ndarray,
    depth: np.ndarray,
    z: np.ndarray,
    cmap,
    norm,
    cbar_label: str,
    gap_factor: float = 4.0,
    reverse_cbar: bool = False,
):
    """Pcolor a depth-by-column field at arbitrary x; one mesh per x-cluster.

    *x* must be finite and ascending; *z* is ``(n_depth, n_col)`` with columns
    in x-order.  Each cluster (see :func:`column_clusters`) is drawn with
    midpoint edges; the gaps between clusters are left blank, never stretched.
    Internal-NaN holes are forward-filled within each column and NaN cells are
    masked (so the cmap's ``set_bad`` color shows for unsampled depths).
    Returns the last ``QuadMesh`` (or ``None`` if every cluster was empty).
    """
    x = np.asarray(x, dtype=float)
    z = ffill_down(z)
    d_edges = depth_edges(depth)
    pcm = None
    # Cluster on the ORIGINAL x (tie nudging would deflate the median spacing
    # and spuriously split evenly-spaced casts); break ties only for the edges
    # of each cluster.
    for s, e in column_clusters(x, gap_factor):
        edges = column_edges(strictly_increasing(x[s:e]))
        pcm = ax.pcolormesh(
            edges, d_edges, np.ma.masked_invalid(z[:, s:e]),
            cmap=cmap, norm=norm, shading="flat",
        )
    if pcm is not None:
        cbar = fig.colorbar(pcm, ax=ax, label=cbar_label)
        if reverse_cbar:
            cbar.ax.invert_yaxis()  # smallest value at the top
    return pcm
