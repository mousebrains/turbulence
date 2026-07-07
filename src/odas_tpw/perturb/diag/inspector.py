# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""The single-figure interactive engine behind ``perturb-diag``.

One matplotlib window holds a time x depth pcolor overview (top) and the
per-cell drill-down (bottom: spectra + a diagnostic strip).  Clicking an
overview cell — or moving with the arrow keys — selects a ``(profile, depth)``
cell; the drill-down panels redraw in place.  This avoids the multiple-figure
juggling that is awkward in matplotlib: there is exactly one figure and one set
of event handlers for its lifetime.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from odas_tpw.perturb.diag import render
from odas_tpw.perturb.diag.data import (
    Cell,
    EpsilonCellSource,
    OverviewData,
    ProfilesCellSource,
    ProfilesFile,
)
from odas_tpw.perturb.plot import layout

# Drill-down renderer signatures. The strip renderers take an explicit depth
# band so the dissipation and scalar strips share one cell band.
SpectraFn = Callable[[object, Cell], None]
StripFn = Callable[[list, Cell, "tuple[float, float]"], "tuple[float, float]"]
ScalarFn = Callable[
    [list, ProfilesFile, "tuple[float, float]"], "tuple[float, float]"
]


class DiagInspector:
    """Interactive overview + click-driven per-cell diagnostics in one figure."""

    def __init__(
        self,
        data: OverviewData,
        source: EpsilonCellSource,
        *,
        field_specs: list[tuple[str, str]],
        spectra_fn: SpectraFn,
        strip_fn: StripFn,
        n_strip: int = 5,
        scalar_source: ProfilesCellSource | None = None,
        scalar_fn: ScalarFn | None = None,
        n_scalar: int = 6,
        title: str = "",
        clim: tuple[float, float] | None = None,
        gap_seconds: float = 600.0,
        figsize: tuple[float, float] | None = None,
    ) -> None:
        self.data = data
        self.source = source
        self.field_specs = field_specs
        self.spectra_fn = spectra_fn
        self.strip_fn = strip_fn
        self.scalar_source = scalar_source
        self.scalar_fn = scalar_fn
        self.title = title
        self.gap_seconds = gap_seconds

        self.iProfile = 0
        self.iDepth = 0

        self._build(n_strip, n_scalar, clim, figsize)

    # ------------------------------------------------------------------ build
    def _build(self, n_strip, n_scalar, clim, figsize) -> None:
        import cmocean
        import matplotlib.pyplot as plt

        self.cast_x, self.segments, *_ = layout.compute_layout(
            self.data.t, gap_seconds=self.gap_seconds
        )

        has_scalar = self.scalar_source is not None and self.scalar_fn is not None
        if figsize is None:
            figsize = (17.0, 13.0) if has_scalar else (16.0, 9.0)

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        self.fig = fig
        n_ov = len(self.field_specs)
        if has_scalar:
            gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.0])
        else:
            gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05])
        gs_top = gs[0].subgridspec(1, n_ov, wspace=0.06)
        gs_bot = gs[1].subgridspec(1, 2 + n_strip, wspace=0.30)

        # Overview panels share the cast (x) and depth (y) axes.
        self.ax_ov: list = []
        for i in range(n_ov):
            share = None if i == 0 else self.ax_ov[0]
            ax = fig.add_subplot(gs_top[0, i], sharex=share, sharey=share)
            self.ax_ov.append(ax)
        self.ax_spec = fig.add_subplot(gs_bot[0, 0:2])
        self.ax_strip = [fig.add_subplot(gs_bot[0, 2 + i]) for i in range(n_strip)]
        for ax in self.ax_strip[1:]:
            ax.sharey(self.ax_strip[0])

        # Optional third row: raw scalar / vehicle diagnostics.
        self.ax_scalar: list = []
        if has_scalar:
            gs_sca = gs[2].subgridspec(1, n_scalar, wspace=0.30)
            self.ax_scalar = [fig.add_subplot(gs_sca[0, i]) for i in range(n_scalar)]
            for ax in self.ax_scalar[1:]:
                ax.sharey(self.ax_scalar[0])
            self.ax_scalar[0].set_ylabel("Depth (m)")

        # Shared log color scale from the finite positive values of all fields.
        stacked = np.concatenate([f.ravel() for f in self.data.fields.values()])
        if clim is not None:
            vmin, vmax = 10.0 ** clim[0], 10.0 ** clim[1]
        else:
            vmin, vmax = layout.quantile_limits(stacked)
        norm = _safe_lognorm(vmin, vmax)
        cmap = cmocean.cm.thermal

        self._xhair: list = []
        self._yhair: list = []
        pcm = None
        for ax, (name, label) in zip(self.ax_ov, self.field_specs):
            z = self.data.fields[name]
            if norm is not None:
                pcm = render.draw_overview_mesh(
                    ax, self.cast_x, self.segments, self.data.bin, z, cmap, norm
                )
            ax.set_title(label)
            ax.set_xlabel("Cast")
            # Crosshair (hidden until a cell is selected).
            self._xhair.append(
                ax.axvline(self.cast_x[0], color="w", lw=0.8, visible=False)
            )
            self._yhair.append(
                ax.axhline(self.data.bin[0], color="w", lw=0.8, visible=False)
            )
        self.ax_ov[0].set_ylabel("Depth (m)")
        if not self.ax_ov[0].yaxis_inverted():
            self.ax_ov[0].invert_yaxis()
        if len(self.cast_x):
            self.ax_ov[0].set_xlim(self.cast_x[0] - 0.5, self.cast_x[-1] + 0.5)
        if pcm is not None:
            fig.colorbar(
                pcm, ax=self.ax_ov, label=r"$\epsilon$  (W kg$^{-1}$)",
                location="right", shrink=0.9,
            )

        self.ax_strip[0].set_ylabel("Depth (m)")
        self._placeholder("click an overview cell (or use the arrow keys)")
        if self.title:
            fig.suptitle(self.title, fontsize=10)

    # -------------------------------------------------------------- selection
    def _placeholder(self, msg: str) -> None:
        for ax in [self.ax_spec, *self.ax_strip, *self.ax_scalar]:
            ax.clear()
        self.ax_spec.text(0.5, 0.5, msg, transform=self.ax_spec.transAxes,
                          ha="center", va="center", fontsize=9)

    def select(self, i_profile: int, i_depth: int) -> None:
        """Select the cell at (profile, depth) and redraw the drill-down."""
        n_prof = len(self.data.stime_epoch)
        n_bin = len(self.data.bin)
        self.iProfile = int(np.clip(i_profile, 0, max(n_prof - 1, 0)))
        self.iDepth = int(np.clip(i_depth, 0, max(n_bin - 1, 0)))

        x = float(self.cast_x[self.iProfile])
        y = float(self.data.bin[self.iDepth])
        for xh, yh in zip(self._xhair, self._yhair):
            xh.set_xdata([x, x])
            xh.set_visible(True)
            yh.set_ydata([y, y])
            yh.set_visible(True)

        stime = float(self.data.stime_epoch[self.iProfile])
        cell = self.source.cell(stime, y)
        # One depth band for both strips: the dissipation cell's extent, or a
        # small fallback around the clicked bin when there is no diss file.
        band = render.band_extent(cell.profile, cell.window) if cell else (y - 1.0, y + 1.0)

        ranges: list[tuple[float, float]] = []
        src_name: str | None = None

        # Spectra + dissipation strip (from the diss product).
        if cell is None:
            self.ax_spec.clear()
            self.ax_spec.text(0.5, 0.5, f"no diss file for cast {self.iProfile + 1}",
                              transform=self.ax_spec.transAxes, ha="center",
                              va="center", fontsize=9)
            for ax in self.ax_strip:
                ax.clear()
        else:
            self.spectra_fn(self.ax_spec, cell)
            ranges.append(self.strip_fn(self.ax_strip, cell, band))
            src_name = cell.profile.path.rsplit("/", 1)[-1]

        # Scalar diagnostics strip (from the profiles product).
        if self.ax_scalar and self.scalar_fn is not None:
            pf = self.scalar_source.profile(stime) if self.scalar_source else None
            if pf is None:
                for ax in self.ax_scalar:
                    ax.clear()
                    ax.text(0.5, 0.5, "no profiles file", transform=ax.transAxes,
                            ha="center", va="center", fontsize=8)
            else:
                ranges.append(self.scalar_fn(self.ax_scalar, pf, band))
                if src_name is None:
                    src_name = pf.path.rsplit("/", 1)[-1]

        # Share one inverted depth range across both strips.
        if ranges:
            dlo = min(r[0] for r in ranges)
            dhi = max(r[1] for r in ranges)
            for ax in [*self.ax_strip, *self.ax_scalar]:
                ax.set_ylim(dhi, dlo)
            for ax in [*self.ax_strip[1:], *self.ax_scalar[1:]]:
                ax.tick_params(labelleft=False)

        tail = f"  —  {src_name}" if src_name else ""
        self.fig.suptitle(
            f"{self.title}\ncast {self.iProfile + 1}/{n_prof}{tail}", fontsize=9
        )
        self.fig.canvas.draw_idle()

    # ----------------------------------------------------------------- events
    def _on_click(self, event) -> None:
        if event.inaxes not in self.ax_ov:
            return
        if event.xdata is None or event.ydata is None:
            return
        i_profile = int(np.argmin(np.abs(self.cast_x - event.xdata)))
        i_depth = int(np.argmin(np.abs(self.data.bin - event.ydata)))
        self.select(i_profile, i_depth)

    def _on_key(self, event) -> None:
        moves = {
            "left": (-1, 0), "right": (1, 0),
            "up": (0, -1), "down": (0, 1),
        }
        if event.key not in moves:
            return
        dp, dd = moves[event.key]
        self.select(self.iProfile + dp, self.iDepth + dd)

    def connect(self) -> None:
        """Wire click + arrow-key handlers, suppressing matplotlib's key nav."""
        canvas = self.fig.canvas
        # Drop matplotlib's built-in key handler so left/right arrows drive the
        # cell selection instead of the toolbar's back/forward navigation.
        mgr = getattr(canvas, "manager", None)
        hid = getattr(mgr, "key_press_handler_id", None)
        if hid is not None:
            canvas.mpl_disconnect(hid)
        canvas.mpl_connect("button_press_event", self._on_click)
        canvas.mpl_connect("key_press_event", self._on_key)

    # -------------------------------------------------------------- launching
    def default_cell(self) -> tuple[int, int]:
        """A representative (profile, depth): the cast with the most data, at a
        mid depth among its finite bins.  Used for a headless snapshot."""
        primary = next(iter(self.data.fields.values()))
        finite = np.isfinite(primary)
        if not finite.any():
            return 0, 0
        j = int(np.argmax(finite.sum(axis=0)))
        rows = np.where(finite[:, j])[0]
        return int(rows[len(rows) // 2]), j

    def show(self) -> None:
        import matplotlib.pyplot as plt

        self.connect()
        plt.show()

    def snapshot(self, out_path: str, *, dpi: int = 110) -> str:
        """Render the default cell and save a static PNG (headless-friendly)."""
        i_depth, i_profile = self.default_cell()
        self.select(i_profile, i_depth)
        self.fig.savefig(out_path, dpi=dpi)
        return out_path


def _safe_lognorm(vmin, vmax):
    """LogNorm for finite positive limits, else None (empty overview)."""
    from matplotlib.colors import LogNorm

    if vmin is None or vmax is None:
        return None
    if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmin <= 0 or vmax <= 0:
        return None
    if vmin >= vmax:
        vmin = vmax / 10.0
    return LogNorm(vmin=vmin, vmax=vmax)
