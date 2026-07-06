# Jun-2026, Claude and Pat Welch, pat@mousebrains.com
"""``perturb-plot scalar`` — depth-vs-x scalar sections from the CTD combo.

Reads the CTD trajectory product ``ctd_combo_NN/combo.nc`` (CF
featureType=trajectory, a continuous down/up sawtooth on a ``time`` axis) and
renders, for each *section*, depth (y, inverted) against a chosen x-axis with a
scalar field in color.

A **section** is purely a way of chopping the trajectory and choosing the
x-axis: a name, an optional UTC ``start``/``stop`` window, and an ``xaxis``
method with its parameters.  Sections come from a YAML file (``--sections``) or
from ad-hoc CLI flags.  *Rendering* choices (which variables, depth/x bin
sizes, color limits) are separate CLI options that apply to every section in
the run.

By default the figures are shown on screen.  They are written to PNG files
instead when ``--out-dir`` is given, or when there is no interactive display
available (no controlling tty, or a non-GUI matplotlib backend such as Agg).

The trajectory is gridded by binning samples onto a regular ``(x, depth)``
mesh and averaging each cell (:func:`grid.grid_mean`); empty cells stay NaN —
no interpolation across gaps.

The section/x-axis/CLI machinery lives in :mod:`odas_tpw.perturb.plot.sections`
and is shared with the ``profiles`` subcommand.
"""

from __future__ import annotations

import argparse
import contextlib
import locale
import os
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from odas_tpw.perturb import resolve
from odas_tpw.perturb.plot import grid, layout, xaxis
from odas_tpw.perturb.plot.sections import (
    Section,
    add_section_arguments,
    close_new_figs_on_error,
    closing_figs,
    fig_dpi,
    load_sections,
    resolve_sections,
    save_or_show,
    single_var_limit_guard,
)
from odas_tpw.perturb.plot.sections import (
    can_display as _can_display,
)
from odas_tpw.perturb.plot.sections import (
    grouped as _grouped,
)
from odas_tpw.perturb.plot.sections import (
    override_xaxis as _override_xaxis,
)
from odas_tpw.perturb.plot.sections import (
    parse_clim as _parse_clim,
)
from odas_tpw.perturb.plot.sections import (
    parse_time as _parse_time,
)
from odas_tpw.perturb.plot.sections import (
    parse_waypoints as _parse_waypoints,
)
from odas_tpw.perturb.plot.sections import (
    safe_name as _safe_name,
)
from odas_tpw.perturb.plot.sections import (
    select_sections as _select_sections,
)
from odas_tpw.perturb.plot.sections import (
    var_label as _var_label,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# Helpers extracted to ``sections`` are re-exported here so existing tests that
# reference ``scalar.<helper>`` keep resolving.
__all__ = [
    "Section",
    "_default_variables",
    "_override_xaxis",
    "_parse_clim",
    "_parse_time",
    "_parse_waypoints",
    "_select_sections",
    "add_arguments",
    "load_sections",
    "run",
]

# Default scalar panels when the user names none, plus optional sensor channels
# added automatically when the instrument carries them.
_DEFAULT_VARS: tuple[str, ...] = ("JAC_T", "SP", "sigma0", "rho")
_OPTIONAL_VARS: tuple[str, ...] = ("DO", "Chlorophyll", "Turbidity")
# Default column count when --ncols is not given (a 2-wide grid).
_DEFAULT_NCOLS = 2

# Per-variable cmocean colormap.  Diverging fields center the color scale at 0.
_CMAP: dict[str, str] = {
    "JAC_T": "thermal",
    "CT": "thermal",
    "T": "thermal",
    "SP": "haline",
    "SA": "haline",
    "sigma0": "dense",
    "rho": "dense",
    "DO": "oxy",
    "Chlorophyll": "algae",
    "Turbidity": "turbid",
    "N2": "amp",
    "dTdz": "balance",
}
_DIVERGING: frozenset[str] = frozenset({"dTdz"})
# Variables whose colorbar runs min-at-top to max-at-bottom, mirroring the
# (inverted) depth axis -- salinity and density both increase with depth.
_CBAR_MIN_AT_TOP: frozenset[str] = frozenset({"SP", "sigma0"})

# Explicit colorbar labels that override the CF long_name/units default
# (var_label) for the CTD scalars. "JFE" names the JFE Advantech CTD; the
# density label uses mathtext so it renders as sigma-zero (sigma_0).
_CBAR_LABEL: dict[str, str] = {
    "JAC_T": "in-situ T (°C) (JFE)",
    "SP": "Salinity (PSU) (JFE)",
    "sigma0": r"$\sigma_0$ (kg/m³) (JFE)",
}


def _default_variables(ds: xr.Dataset) -> list[str]:
    """Default panel set: the standard scalars present, plus optional sensors."""
    chosen = [v for v in _DEFAULT_VARS if v in ds.data_vars]
    chosen += [v for v in _OPTIONAL_VARS if v in ds.data_vars]
    return chosen


def _time_subset(ds: xr.Dataset, sec: Section) -> xr.Dataset:
    """Index the trajectory to the section's [start, stop] UTC window."""
    t = ds["time"].values
    # dtype-agnostic: sec.start/stop are datetime64. If the combo's time was
    # decoded to datetime64, compare directly; if it is numeric (epoch seconds,
    # no CF units), compare in epoch seconds so we don't raise a cryptic
    # UFuncTypeError mixing float64 with datetime64.
    numeric = not np.issubdtype(t.dtype, np.datetime64)
    if numeric:
        t = t.astype(np.float64)

    def _bound(b):
        if b is None:
            return None
        return float(xaxis.to_epoch_seconds(np.array([b]))[0]) if numeric else b

    mask = np.ones(t.shape, dtype=bool)
    start, stop = _bound(sec.start), _bound(sec.stop)
    if start is not None:
        mask &= t >= start
    if stop is not None:
        mask &= t <= stop
    return ds.isel(time=np.flatnonzero(mask))


def _build_section_figure(
    ds: xr.Dataset,
    sec: Section,
    variables: list[str],
    args: argparse.Namespace,
    clim: dict[str, tuple[float, float]],
) -> Figure | None:
    """Render one section to a matplotlib Figure (not saved or shown here).

    Returns ``None`` when the section has no plottable data (empty window, no
    requested variable present, no finite x positions). The caller decides
    whether to display or save the returned figure.
    """
    import cmocean
    from matplotlib.colors import Normalize
    from matplotlib.ticker import FuncFormatter

    dss = _time_subset(ds, sec)
    if dss.sizes.get("time", 0) == 0:
        print(f"section {sec.name!r}: no samples in time window; skipped")
        return None

    lat = dss["lat"].values if "lat" in dss else np.full(dss.sizes["time"], np.nan)
    lon = dss["lon"].values if "lon" in dss else np.full(dss.sizes["time"], np.nan)
    depth = dss["depth"].values
    xa = xaxis.compute(sec.method, lat, lon, dss["time"].values, sec.params)

    panel_vars = [v for v in variables if v in dss.data_vars]
    missing = [v for v in variables if v not in dss.data_vars]
    if not panel_vars:
        print(f"section {sec.name!r}: none of {variables} present on dataset; skipped")
        return None
    if missing:
        print(f"section {sec.name!r}: variables not on dataset, skipped: {missing}")

    # x grid: finite samples only; spatial axes can run either direction.
    x = np.asarray(xa.x, dtype=float)
    finite_x = x[np.isfinite(x)]
    if finite_x.size == 0:
        print(f"section {sec.name!r}: no finite x positions; skipped")
        return None
    x_lo, x_hi = float(finite_x.min()), float(finite_x.max())
    x_edges = grid.make_edges(x_lo, x_hi, args.x_bin or grid.auto_step(x_lo, x_hi))

    # depth grid: surface to the deepest sampled bin (or --depth-max).
    finite_z = depth[np.isfinite(depth)]
    z_hi = float(finite_z.max()) if finite_z.size else 1.0
    if args.depth_max is not None:
        z_hi = float(args.depth_max)
    z_edges = grid.make_edges(0.0, z_hi, args.z_bin)

    # Panels are arranged in `ncols` columns (default 1 = a vertical stack),
    # filled left-to-right, top-to-bottom. Sections remain one figure each.
    fig, axes, left_axes, col_bottom = layout.panel_grid(
        len(panel_vars), getattr(args, "ncols", None) or _DEFAULT_NCOLS,
        figsize=getattr(args, "figsize", None),
    )
    left_set = set(left_axes)

    for ax, name in zip(axes, panel_vars):
        # With sharey, only the left column carries the "Depth (m)" label (the
        # other columns' y tick labels are hidden, so a bare label would mislead).
        is_left = ax in left_set
        v = dss[name].values
        g, _count = grid.grid_mean(x, depth, v, x_edges, z_edges)
        if not np.any(np.isfinite(g)):
            ax.text(0.5, 0.5, f"no valid {name}", transform=ax.transAxes,
                    ha="center", va="center")
            if is_left:
                ax.set_ylabel("Depth (m)")
            continue
        # Per-variable --clim wins; else the global --vmin/--vmax (only set for
        # a single-variable plot, enforced in run()); else auto 1/99 percentile.
        lo_ov, hi_ov = clim.get(name, (args.vmin, args.vmax))
        explicit_limits = name in clim or args.vmin is not None or args.vmax is not None
        vmin, vmax = grid.linear_limits(g, lo_ov, hi_ov)
        norm: Normalize
        if (
            name in _DIVERGING
            and not explicit_limits
            and vmin is not None
            and vmax is not None
        ):
            # Symmetric about 0 so the diverging colormap's white midpoint
            # always marks zero gradient -- even on a leg where dT/dz is
            # entirely one sign.  Explicit --vmin/--vmax override this.
            m = max(abs(vmin), abs(vmax))
            norm = Normalize(vmin=-m, vmax=m)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = getattr(cmocean.cm, _CMAP.get(name, "thermal")).copy()
        cmap.set_bad(color="0.85")  # empty cells: light gray, visibly unsampled
        pcm = ax.pcolormesh(x_edges, z_edges, np.ma.masked_invalid(g),
                            cmap=cmap, norm=norm, shading="flat")
        cbar_fmt: str | None = None
        if name == "SP" and vmin is not None and vmax is not None:
            # Salinity sits in a narrow band; show 2 decimals, or 1 when the
            # range is wide enough to read -- avoids false precision on ticks.
            cbar_fmt = "%.1f" if (vmax - vmin) >= 1.0 else "%.2f"
        cbar_label = _CBAR_LABEL.get(name, _var_label(ds, name))
        cbar = fig.colorbar(pcm, ax=ax, label=cbar_label, format=cbar_fmt)
        if name in _CBAR_MIN_AT_TOP:
            cbar.ax.invert_yaxis()  # min at top, max at bottom (mirrors depth)
        if is_left:
            ax.set_ylabel("Depth (m)")

    for ax in axes:
        # Draw the grid over the color mesh (axisbelow False), a thin muted
        # line so it reads on both light (thermal/haline) and dark (dense) maps.
        ax.set_axisbelow(False)
        ax.grid(True, color="0.4", linewidth=0.4, alpha=0.5)

    axes[0].invert_yaxis()  # sharey: inverts every panel's depth axis
    axes[0].set_xlim(x_edges[0], x_edges[-1])

    # The x label/formatter goes on the bottom-most panel of each column
    # (layout.panel_grid already re-enabled their tick labels).
    for ax in col_bottom:
        ax.set_xlabel(xa.label)
        if xa.kind == "time":
            ax.xaxis.set_major_formatter(
                FuncFormatter(
                    lambda val, _pos: np.datetime_as_string(
                        np.datetime64(int(val), "s"), unit="m"
                    ).replace("T", " ")
                )
            )
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(30)
                lbl.set_horizontalalignment("right")

    title_id = ds.attrs.get("id") or os.path.basename(os.path.normpath(args.root))
    npts = int(dss.sizes["time"])
    fig.suptitle(getattr(args, "title", None) or (
        f"{title_id}  —  section: {sec.name}  —  "
        f"x-axis: {sec.method}  —  {_grouped(npts)} samples"
    ))
    layout.fit_colorbar_labels(fig)  # long var labels overflow short per-panel bars
    return fig


# ---------------------------------------------------------------------------
# CLI plumbing (perturb-plot subcommand contract)
# ---------------------------------------------------------------------------


def add_arguments(p: argparse.ArgumentParser) -> None:
    """Register CLI flags for the scalar subcommand on *p*."""
    add_section_arguments(p)
    p.add_argument("--ctd-combo", default=None,
                   help="explicit ctd combo dir or combo.nc (default: latest under --root)")
    p.add_argument("--z-bin", type=float, default=1.0, help="depth bin width [m]")
    p.add_argument("--x-bin", type=float, default=None,
                   help="x bin width in x-axis units (default: ~300 columns)")
    p.add_argument("--depth-max", type=float, default=None,
                   help="clip the depth axis at this value [m]")


def build_figures(args: argparse.Namespace) -> Iterator[tuple[str, Any]]:
    """Yield one ``(stem, Figure)`` per resolved section (no saving/showing).

    A **generator**, lazily, so a streaming caller (``run``'s save path, the
    ``figure`` PDF driver) can save and ``close`` each figure before the next is
    built — bounded to one open figure at a time. Sections with no finite data
    are skipped, so fewer figures may be yielded than there are sections.
    """
    # Honor the user's locale for number formatting in titles. Scoped to
    # LC_NUMERIC so we don't disturb matplotlib's float parsing (LC_CTYPE) or
    # collation; falls back silently when the environment locale is unset.
    with contextlib.suppress(locale.Error):
        locale.setlocale(locale.LC_NUMERIC, "")

    args.root = resolve.require_root(args)  # backfill from --config if needed

    src = args.ctd_combo or resolve.resolve_for_args(args, "ctd_combo")
    if src is None:
        raise SystemExit(f"No ctd_combo dir under {args.root}")
    path = src if src.endswith(".nc") else os.path.join(src, "combo.nc")
    if not os.path.exists(path):
        raise SystemExit(f"CTD combo not found: {path}")

    ds = xr.open_dataset(path)  # default CF decoding -> datetime64 time
    try:
        if "time" not in ds.dims:
            raise SystemExit(f"{path}: expected a CTD trajectory with a 'time' dimension")
        sections = resolve_sections(args)
        variables = list(args.var) if args.var else _default_variables(ds)
        clim = _parse_clim(args.clim)
        single_var_limit_guard(args, variables)
        for sec in sections:
            with close_new_figs_on_error():  # close a half-built figure if it raises
                fig = _build_section_figure(ds, sec, variables, args, clim)
            if fig is not None:
                yield f"scalar_{_safe_name(sec.name)}", fig
    finally:
        ds.close()  # figures hold their own arrays, so the dataset can close now


def run(args: argparse.Namespace) -> str:
    """Render every section; show on screen, or write PNGs and return their dir."""
    args.root = resolve.require_root(args)  # so out_dir/display are known up front
    # Show on screen unless the user asked for files or no display is available.
    display = args.out_dir is None and _can_display()
    # closing_figs releases the generator's dataset handle even if a save raises
    # mid-stream (not left to GC).
    if display:
        with closing_figs(build_figures(args)) as figs:
            shown = save_or_show(figs, None, fig_dpi(args))
        return f"displayed {shown} section(s)"

    out_dir = args.out_dir or args.root
    os.makedirs(out_dir, exist_ok=True)
    with closing_figs(build_figures(args)) as figs:
        save_or_show(figs, out_dir, fig_dpi(args))
    return str(out_dir)
