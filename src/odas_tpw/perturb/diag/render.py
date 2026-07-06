# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""Pure renderers for ``perturb-diag`` — each takes an Axes and stored arrays.

Kept free of interaction state so the same functions serve the interactive
inspector and headless snapshot rendering.  The overview mesh is drawn once;
the spectra and diagnostic-strip renderers clear and redraw their axes on every
cell selection.
"""

from __future__ import annotations

import numpy as np

from odas_tpw.perturb.diag.data import Cell, ProfileFile
from odas_tpw.perturb.plot import layout

# Per-probe colors, matching the rsi ProfileViewer palette.
_PROBE_COLORS = ("C0", "C1", "C4", "C5")


def draw_overview_mesh(ax, cast_x, segments, depth, z, cmap, norm):
    """Pcolor a depth-by-cast field on *ax*, one mesh per cluster (no colorbar).

    Mirrors :func:`layout.plot_panel` but returns the mesh without attaching a
    per-axis colorbar, so the inspector can share one colorbar across the three
    overview panels (they share a color scale).
    """
    z = layout.ffill_down(z)
    d_edges = layout.depth_edges(depth)
    pcm = None
    for s, e in segments:
        x_seg = cast_x[s:e]
        if len(x_seg) == 1:
            edges = np.array([x_seg[0] - 0.5, x_seg[0] + 0.5])
        else:
            mids = 0.5 * (x_seg[:-1] + x_seg[1:])
            edges = np.concatenate(([x_seg[0] - 0.5], mids, [x_seg[-1] + 0.5]))
        pcm = ax.pcolormesh(
            edges, d_edges, z[:, s:e], cmap=cmap, norm=norm, shading="flat"
        )
    return pcm


def band_extent(pf: ProfileFile, window: int) -> tuple[float, float]:
    """Depth interval [lo, hi] of the dissipation cell centered at *window*.

    Half-height is ``0.5 * (diss_length / fs_fast) * speed`` — the along-track
    length of the fft window converted to depth — matching the Matlab tool's
    grey band.  Falls back to +/-0.5 m when the length metadata is absent.
    """
    d0 = float(pf.depth[window])
    diss_len = pf.attrs.get("diss_length")
    fs = pf.attrs.get("fs_fast")
    spd = pf.speed[window] if window < pf.speed.size else np.nan
    if diss_len and fs and np.isfinite(spd):
        half = 0.5 * (float(diss_len) / float(fs)) * float(spd)
    else:
        half = 0.5
    return d0 - half, d0 + half


def draw_eps_spectra(ax, cell: Cell) -> None:
    """Shear wavenumber spectra with Nasmyth fits and K_max markers."""
    from odas_tpw.scor160.nasmyth import nasmyth

    ax.clear()
    pf, w = cell.profile, cell.window
    K = pf.K[:, w]
    kgood = np.isfinite(K) & (K > 0)
    if not kgood.any():
        ax.text(0.5, 0.5, "no spectrum", transform=ax.transAxes,
                ha="center", va="center")
        return

    for p in range(pf.n_probe):
        c = _PROBE_COLORS[p % len(_PROBE_COLORS)]
        shear = pf.spec_shear[p, :, w]
        nas = pf.spec_nasmyth[p, :, w]
        eps = pf.epsilon[p, w]
        kmax = pf.K_max[p, w]
        sgood = kgood & np.isfinite(shear) & (shear > 0)
        if sgood.any():
            ax.loglog(K[sgood], shear[sgood], "-", color=c, linewidth=1.5,
                      label=rf"$\epsilon_{p + 1}$={eps:.2g}")
        ngood = kgood & np.isfinite(nas) & (nas > 0)
        if ngood.any():
            ax.loglog(K[ngood], nas[ngood], "--", color=c, linewidth=0.8,
                      alpha=0.8)
        if np.isfinite(kmax) and sgood.any():
            ki = int(np.argmin(np.abs(K - kmax)))
            ax.loglog(kmax, shear[ki], marker="v", color=c, markersize=8,
                      zorder=5)

    # Faint reference Nasmyth spectra for orientation.
    nu = pf.nu[w]
    if np.isfinite(nu):
        for exp in (-10, -9, -8, -7, -6, -5):
            ax.loglog(K[kgood], nasmyth(10.0**exp, nu, K[kgood]),
                      color="0.85", linewidth=0.3, zorder=0)

    ax.set_xlabel("k (cpm)")
    ax.set_ylabel(r"$\Phi$(k)  (s$^{-2}$ cpm$^{-1}$)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6, loc="lower left")

    em = pf.epsilon_mean[w]
    sig = pf.epsilon_ln_sigma[w]
    lo = f"{np.log10(em):.2f}" if np.isfinite(em) and em > 0 else "?"
    pm = f"{sig / np.log(10):.2f}" if np.isfinite(sig) else "?"
    meth = "/".join(
        "ISR" if pf.method[p, w] == 1 else "var" for p in range(pf.n_probe)
    )
    ax.set_title(
        rf"depth {pf.depth[w]:.1f} m   $\log_{{10}}\langle\epsilon\rangle$="
        rf"{lo}$\pm${pm}   method [{meth}]",
        fontsize=8,
    )


# Diagnostic strip: (label, per-probe 2-D array or None, per-window 1-D array,
# log-x flag).  Rendered left-to-right vs depth.
def draw_diss_strip(axes, cell: Cell) -> tuple[float, float]:
    """Per-profile dissipation diagnostics vs depth, with a cell depth band.

    Returns the (min, max) finite depth so the caller can set a shared,
    inverted y range.  Panels: epsilon (per probe + mean), FM, speed, nu, T.
    """
    pf, w = cell.profile, cell.window
    depth = pf.depth
    lo, hi = band_extent(pf, w)

    # Panel 0: per-probe epsilon (points) + mean (line), log-x.
    ax = axes[0]
    ax.clear()
    for p in range(pf.n_probe):
        c = _PROBE_COLORS[p % len(_PROBE_COLORS)]
        ax.plot(pf.epsilon[p], depth, ".", color=c, markersize=3,
                label=rf"$\epsilon_{p + 1}$")
    ax.plot(pf.epsilon_mean, depth, "-", color="k", linewidth=1.0,
            label=r"$\langle\epsilon\rangle$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\epsilon$ (W kg$^{-1}$)")
    ax.set_ylabel("Depth (m)")
    ax.legend(fontsize=6, loc="best")

    # Remaining scalar panels.
    scalars = [
        ("FM", pf.FM, False),        # (probe, window)
        ("speed (m/s)", pf.speed, False),
        (r"$\nu$ (m$^2$/s)", pf.nu, False),
        ("T (°C)", pf.T_mean, False),
    ]
    for ax, (xlabel, arr, logx) in zip(axes[1:], scalars):
        ax.clear()
        if arr.ndim == 2:
            for p in range(pf.n_probe):
                c = _PROBE_COLORS[p % len(_PROBE_COLORS)]
                ax.plot(arr[p], depth, "-", color=c, linewidth=0.8)
        else:
            ax.plot(arr, depth, "-", color="C2", linewidth=0.8)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)

    # Shared grey depth band + grid on every panel.
    for ax in axes:
        ax.axhspan(lo, hi, color="0.5", alpha=0.25, zorder=0)
        ax.grid(True, alpha=0.3)

    finite = depth[np.isfinite(depth)]
    if finite.size:
        return float(finite.min()), float(finite.max())
    return 0.0, 1.0
