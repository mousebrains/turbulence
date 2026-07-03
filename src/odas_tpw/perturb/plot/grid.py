# Jun-2026, Claude and Pat Welch, pat@mousebrains.com
"""Scatter-to-grid binning and signed-safe color limits for section plots.

A distinct coordinate model from :mod:`layout` (which positions already-binned
cast columns): here we bin irregular ``(x, depth, value)`` samples from a CTD
trajectory onto a regular mesh, averaging the samples in each cell and leaving
empty cells NaN.  We never interpolate across unsampled gaps, so the figure
cannot paint values into water the vehicle did not sample.

Pure numpy — no matplotlib, no NetCDF, no colormaps.
"""

from __future__ import annotations

import numpy as np


def auto_step(lo: float, hi: float, target: int = 300) -> float:
    """Bin width giving ~*target* bins across [lo, hi]; 1.0 as a safe fallback."""
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        return 1.0
    return span / float(target)


def make_edges(lo: float, hi: float, step: float) -> np.ndarray:
    """Regular bin edges spanning [lo, hi] with width ~*step* (>= 1 bin)."""
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        width = step if (step and np.isfinite(step) and step > 0) else 1.0
        return np.array([lo, lo + width], dtype=float)
    if not (step and np.isfinite(step) and step > 0):
        step = auto_step(lo, hi)
    n = max(1, int(np.ceil((hi - lo) / step)))
    edges = lo + step * np.arange(n + 1, dtype=float)
    # np.digitize is half-open [edge_i, edge_{i+1}); a sample exactly on the
    # last edge would be dropped.  When the span divides evenly the last edge
    # lands on *hi* (e.g. max depth at the bottom turn, the most interesting
    # row) -- nudge it just past hi so that sample falls inside the last bin.
    if edges[-1] <= hi:
        edges[-1] = np.nextafter(float(hi), np.inf)
    return edges


def grid_mean(
    x: np.ndarray,
    z: np.ndarray,
    v: np.ndarray,
    x_edges: np.ndarray,
    z_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean of *v* per ``(z_bin, x_bin)`` cell.

    Returns ``(grid, count)`` each of shape ``(len(z_edges)-1, len(x_edges)-1)``.
    Samples with non-finite x, z, or v are dropped before binning (so this is
    an effective per-cell nanmean for the supplied variable).  Empty cells are
    NaN in *grid* and 0 in *count*.  Down- and up-cast samples that share a
    cell are averaged together — intended, since re-occupation of a cell only
    happens at the bottom turn of a cast.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    v = np.asarray(v, dtype=float)

    nx = len(x_edges) - 1
    nz = len(z_edges) - 1
    grid = np.full((nz, nx), np.nan)
    count = np.zeros((nz, nx), dtype=float)
    if nx < 1 or nz < 1:
        return grid, count

    good = np.isfinite(x) & np.isfinite(z) & np.isfinite(v)
    x, z, v = x[good], z[good], v[good]
    if x.size == 0:
        return grid, count

    ix = np.digitize(x, x_edges) - 1
    iz = np.digitize(z, z_edges) - 1
    inb = (ix >= 0) & (ix < nx) & (iz >= 0) & (iz < nz)
    ix, iz, v = ix[inb], iz[inb], v[inb]
    if ix.size == 0:
        return grid, count

    flat = iz * nx + ix
    sums = np.bincount(flat, weights=v, minlength=nz * nx)
    cnts = np.bincount(flat, minlength=nz * nx)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    return mean.reshape(nz, nx), cnts.reshape(nz, nx).astype(float)


def linear_limits(
    z: np.ndarray,
    lo: float | None = None,
    hi: float | None = None,
    q_lo: float = 0.01,
    q_hi: float = 0.99,
) -> tuple[float | None, float | None]:
    """Inner 1/99 quantile of finite values, sign-agnostic; overrides win.

    Unlike :func:`layout.quantile_limits` (which filters ``> 0`` for log-scaled
    epsilon/chi), this keeps negative values — essential for signed scalars
    such as ``sigma0`` (a density *anomaly*) or ``dTdz``.  Returns the passed
    overrides unchanged when the data is all-NaN.
    """
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        return lo, hi
    q = np.quantile(finite, [q_lo, q_hi])
    return (
        lo if lo is not None else float(q[0]),
        hi if hi is not None else float(q[1]),
    )
