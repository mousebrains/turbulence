# May-2026, Claude and Pat Welch, pat@mousebrains.com
"""Shared depth-by-cast pcolor helpers used by perturb-plot subcommands.

Pure-numpy / pure-matplotlib utilities — no perturb config, no NetCDF I/O.
Suitable for any 2-D depth-by-cast field a subcommand wants to render.
"""

from __future__ import annotations

import glob
import os

import numpy as np


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
    """Centre-spaced edges around 1-D depth bin centers.

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

    The fill is bounded to runs of at most ``max_gap`` bins (default 1): an
    isolated missing bin is cosmetically bridged, but a longer run — e.g. a
    QC-rejected / pipeline-NaN'd region (``drop_action='nan'``) — is left as a
    visible gap rather than repainted with the shallower bin's value. Without
    this bound a "QC applied" figure would silently render the dropped bins as
    if they had passed.
    """
    z = z.copy()
    n = z.shape[0]
    rows = np.arange(n)[:, None]
    valid = np.isfinite(z)
    last_above = np.maximum.accumulate(np.where(valid, rows, -1), axis=0)
    valid_or_below = np.cumsum(valid[::-1], axis=0)[::-1] > 0
    has_below = np.zeros_like(valid)
    has_below[:-1] = valid_or_below[1:]
    fill = ~valid & (last_above >= 0) & has_below & ((rows - last_above) <= max_gap)
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
    annotate cluster centres / start / end timestamps for axis labels.
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
    """Pcolor x-edges for columns centred at sorted-ascending positions *x*.

    Interior edges are midpoints; the outer edges extend by half the adjacent
    spacing so the first/last column matches its neighbour's width.  A single
    column gets a half-unit width on each side.
    """
    x = np.asarray(x, dtype=float)
    if len(x) == 1:
        return np.array([x[0] - 0.5, x[0] + 0.5])
    mids = 0.5 * (x[:-1] + x[1:])
    first = x[0] - (mids[0] - x[0])
    last = x[-1] + (x[-1] - mids[-1])
    return np.concatenate(([first], mids, [last]))


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
):
    """Pcolor a depth-by-column field at arbitrary x; one mesh per x-cluster.

    *x* must be finite and ascending; *z* is ``(n_depth, n_col)`` with columns
    in x-order.  Each cluster (see :func:`column_clusters`) is drawn with
    midpoint edges; the gaps between clusters are left blank, never stretched.
    Internal-NaN holes are forward-filled within each column and NaN cells are
    masked (so the cmap's ``set_bad`` colour shows for unsampled depths).
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
        fig.colorbar(pcm, ax=ax, label=cbar_label)
    return pcm
