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


def ffill_down(z: np.ndarray) -> np.ndarray:
    """Forward-fill internal NaNs along axis 0 (depth) within each column.

    Leading NaNs (above the shallowest valid sample) and trailing NaNs
    (below the deepest valid sample) are left as NaN — colors don't extend
    past either end of a cast. Used so the pcolor reads as a continuous
    profile when there are scattered missing bins inside the cast envelope.
    """
    z = z.copy()
    n = z.shape[0]
    rows = np.arange(n)[:, None]
    valid = np.isfinite(z)
    last_above = np.maximum.accumulate(np.where(valid, rows, -1), axis=0)
    valid_or_below = np.cumsum(valid[::-1], axis=0)[::-1] > 0
    has_below = np.zeros_like(valid)
    has_below[:-1] = valid_or_below[1:]
    fill = ~valid & (last_above >= 0) & has_below
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
