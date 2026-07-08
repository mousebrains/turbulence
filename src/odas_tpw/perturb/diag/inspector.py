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

import contextlib
from collections.abc import Callable

import numpy as np

from odas_tpw.perturb.diag import render
from odas_tpw.perturb.diag.data import Cell, CellSource, OverviewData
from odas_tpw.perturb.plot import layout

# Signature of a drill-down renderer: (axes-or-ax, cell) -> optional depth range.
SpectraFn = Callable[[object, Cell], None]
StripFn = Callable[[list, Cell], "tuple[float, float]"]
# Raw-diagnostics renderer: (axes, cell) -> None (shares the strip's depth axis).
DiagFn = Callable[[list, Cell], None]


class DiagInspector:
    """Interactive overview + click-driven per-cell diagnostics in one figure."""

    def __init__(
        self,
        data: OverviewData,
        source: CellSource,
        *,
        field_specs: list[tuple[str, str]],
        spectra_fn: SpectraFn,
        strip_fn: StripFn,
        n_strip: int = 5,
        diag_fn: DiagFn | None = None,
        n_diag: int = 0,
        title: str = "",
        cbar_label: str = r"$\epsilon$  (W kg$^{-1}$)",
        clim: tuple[float, float] | None = None,
        per_field_clim: bool = False,
        gap_seconds: float = 600.0,
        figsize: tuple[float, float] = (16.0, 9.0),
    ) -> None:
        self.data = data
        self.source = source
        self.field_specs = field_specs
        self.spectra_fn = spectra_fn
        self.strip_fn = strip_fn
        self.diag_fn = diag_fn
        self.n_diag = int(n_diag)
        self.title = title
        self.cbar_label = cbar_label
        self.per_field_clim = per_field_clim
        self.gap_seconds = gap_seconds

        self.iProfile = 0
        self.iDepth = 0
        self.iField = 0  # which overview panel was last clicked (mixing dispatch)
        self.ax_diag: list = []

        self._build(n_strip, clim, figsize)

    # ------------------------------------------------------------------ build
    def _build(self, n_strip, clim, figsize) -> None:
        import cmocean
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        self.cast_x, self.segments, *_ = layout.compute_layout(
            self.data.t, gap_seconds=self.gap_seconds
        )

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        self.fig = fig
        n_ov = len(self.field_specs)
        has_diag = self.diag_fn is not None and self.n_diag > 0
        # Three rows when the raw-diagnostics strip is present: overview,
        # spectra+diss-strip, raw-diagnostics.  Otherwise the original two.
        if has_diag:
            gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.85])
        else:
            gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05])
        gs_top = gs[0].subgridspec(1, n_ov, wspace=0.06)
        gs_mid = gs[1].subgridspec(1, 2 + n_strip, wspace=0.30)

        # Overview panels share the cast (x) and depth (y) axes.
        self.ax_ov: list = []
        for i in range(n_ov):
            share = None if i == 0 else self.ax_ov[0]
            ax = fig.add_subplot(gs_top[0, i], sharex=share, sharey=share)
            self.ax_ov.append(ax)
        self.ax_spec = fig.add_subplot(gs_mid[0, 0:2])
        self.ax_strip = [fig.add_subplot(gs_mid[0, 2 + i]) for i in range(n_strip)]
        # All depth axes — the overview panels, the diss strip, and the raw
        # diagnostics — share one depth (y) axis, so panning/zooming depth on any
        # of them moves them all together and they stay vertically aligned.  The
        # spectra panel (wavenumber x PSD) is deliberately left independent.
        for ax in self.ax_strip:
            ax.sharey(self.ax_ov[0])
        self.ax_diag = []
        if has_diag:
            gs_diag = gs[2].subgridspec(1, self.n_diag, wspace=0.30)
            self.ax_diag = [
                fig.add_subplot(gs_diag[0, i]) for i in range(self.n_diag)
            ]
            for ax in self.ax_diag:
                ax.sharey(self.ax_ov[0])
            self.ax_diag[0].set_ylabel("Depth (m)")

        cmap = cmocean.cm.thermal
        # One shared log color scale across panels (epsilon/chi: same units)
        # unless per_field_clim (mixing: K_rho / K_T / Gamma differ in units and
        # magnitude), in which case each panel gets its own scale + colorbar.
        shared_norm = None
        if not self.per_field_clim:
            stacked = np.concatenate([f.ravel() for f in self.data.fields.values()])
            if clim is not None:
                vmin, vmax = 10.0 ** clim[0], 10.0 ** clim[1]
            else:
                vmin, vmax = layout.quantile_limits(stacked)
            shared_norm = _safe_lognorm(vmin, vmax)

        self._xhair: list = []
        self._yhair: list = []
        pcm_shared = None
        for ax, (name, label) in zip(self.ax_ov, self.field_specs):
            z = self.data.fields[name]
            if self.per_field_clim:
                vmin, vmax = layout.quantile_limits(z.ravel())
                norm = _safe_lognorm(vmin, vmax)
            else:
                norm = shared_norm
            if norm is not None:
                pcm = render.draw_overview_mesh(
                    ax, self.cast_x, self.segments, self.data.bin, z, cmap, norm
                )
                if self.per_field_clim:
                    fig.colorbar(pcm, ax=ax, location="right", shrink=0.9, pad=0.02)
                else:
                    pcm_shared = pcm
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
        # Cast number is an integer index; force whole-number x ticks (the shared
        # x-axis can otherwise land on fractions, e.g. 2.5, for a narrow section).
        for ax in self.ax_ov:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if len(self.cast_x):
            self.ax_ov[0].set_xlim(self.cast_x[0] - 0.5, self.cast_x[-1] + 0.5)
        if pcm_shared is not None:
            fig.colorbar(
                pcm_shared, ax=self.ax_ov, label=self.cbar_label,
                location="right", shrink=0.9,
            )

        self.ax_strip[0].set_ylabel("Depth (m)")
        self._placeholder("click an overview cell (or use the arrow keys)")
        if self.title:
            fig.suptitle(self.title, fontsize=10)
        # Set the shared depth limits last: _placeholder's clear() of the (now
        # depth-linked) strip/diag axes autoscales the shared group, so the
        # data-span limit has to be applied after it to stick.
        self._set_overview_depth_limits()

    def _set_overview_depth_limits(self) -> None:
        """Set the (inverted) overview depth axis to the span of finite data.

        Takes the union, over the overview fields, of bins holding at least one
        finite cast; the axis then covers only sampled depths (with a small pad)
        rather than the full — often much deeper — combo bin grid.  Falls back to
        a plain inversion of the full range when nothing is finite.
        """
        bins = self.data.bin
        has_data = np.zeros(bins.shape, dtype=bool)
        for z in self.data.fields.values():
            has_data |= np.isfinite(z).any(axis=1)
        if has_data.any():
            sampled = bins[has_data]
            d_lo, d_hi = float(sampled.min()), float(sampled.max())
            pad = 0.03 * (d_hi - d_lo) + 1.0
            self.ax_ov[0].set_ylim(d_hi + pad, max(d_lo - pad, 0.0))
        elif not self.ax_ov[0].yaxis_inverted():
            self.ax_ov[0].invert_yaxis()

    # -------------------------------------------------------------- selection
    def _placeholder(self, msg: str) -> None:
        for ax in [self.ax_spec, *self.ax_strip, *self.ax_diag]:
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

        # Depth is one shared axis across the overview, strip and diag rows;
        # preserve its current limits (the data-span default or the user's zoom)
        # across the in-place redraws, whose clear()+plot would otherwise
        # autoscale the whole linked group to this one profile.
        depth_ylim = self.ax_ov[0].get_ylim()
        stime = float(self.data.stime_epoch[self.iProfile])
        cell = self.source.cell(stime, y)
        if cell is None:
            self._placeholder(f"no per-profile file for cast {self.iProfile + 1}")
        else:
            # Tell the renderer which overview field was selected; the mixing
            # drill-down keys its spectra off it (K_rho -> shear, K_T -> chi).
            cell.field = self.field_specs[self.iField][0]
            self.spectra_fn(self.ax_spec, cell)
            self.strip_fn(self.ax_strip, cell)
            for ax in self.ax_strip[1:]:
                ax.tick_params(labelleft=False)
            if self.diag_fn is not None and self.ax_diag:
                self.diag_fn(self.ax_diag, cell)
                for ax in self.ax_diag[1:]:
                    ax.tick_params(labelleft=False)
            src = cell.profile.path.rsplit("/", 1)[-1]
            self.fig.suptitle(
                f"{self.title}\ncast {self.iProfile + 1}/{n_prof}  —  {src}",
                fontsize=9,
            )
        self.ax_ov[0].set_ylim(depth_ylim)  # restore the shared depth range
        self.fig.canvas.draw_idle()

    # ----------------------------------------------------------------- events
    def _on_click(self, event) -> None:
        if event.inaxes not in self.ax_ov:
            return
        if event.xdata is None or event.ydata is None:
            return
        # Remember which panel was clicked so the drill-down can adapt to the
        # selected field (mixing: K_rho -> shear spectrum, K_T -> chi spectrum).
        self.iField = self.ax_ov.index(event.inaxes)
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
        self._start_prewarm()
        plt.show()

    def _start_prewarm(self) -> None:
        """Scan the per-profile directories on a background thread while the
        user reads the overview, so the first cell click renders promptly.

        The scan is pure file I/O (no matplotlib), and the GUI event loop on the
        main thread releases the GIL, so the thread makes progress during
        ``plt.show()``.  Best-effort: a failure just leaves the lazy click path.
        """
        import threading

        def _work() -> None:
            # Best-effort: a failed scan just leaves the lazy click path intact.
            with contextlib.suppress(Exception):
                self.source.prewarm()

        threading.Thread(
            target=_work, daemon=True, name="perturb-diag-prewarm"
        ).start()

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
