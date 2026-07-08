# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""``perturb-plot overview`` — dissipation over a stratification/water-mass row.

One figure per *section*, three rows sharing one inverted depth axis and one
section x-axis:

* row 1 (full width): ``⟨ε⟩``  — TKE dissipation, from ``diss_combo``
* row 2 (full width): ``⟨χ⟩``  — thermal-variance dissipation, from ``chi_combo``
* row 3 (side by side): a named *context* set chosen by ``--bottom`` —
  ``ctd`` (default) = ``dT/dz``, ``Θ`` (conservative temperature) and ``S``
  (practical salinity) from ``combo``; ``mixing`` = ``K_rho``, ``K_T`` and
  ``Gamma`` from ``chi_combo``.  ``--var`` overrides the set with an explicit list.

The two dissipation rows and the default context row are all per-cast
``(bin, profile)`` fields, drawn as depth-by-column meshes with the ``profiles``
engine's :func:`layout.plot_columns` (one mesh per x-cluster, blank gaps kept).

Each panel variable is resolved *independently* to the first product that
actually carries it (:data:`_VAR_STAGES`) and drawn from that product's own
grid — the three depth products (``combo`` / ``diss_combo`` / ``chi_combo``)
are binned separately and need not share a ``bin`` grid or profile set, so we
never merge them onto one array (unlike the ``mixing`` product).  A scalar that
is only present as a continuous ``ctd_combo`` trajectory falls back to the
``scalar`` gridding path (:func:`grid.grid_mean`).  A partial run (no chi, no
CTD) still renders what it has and labels the rest "no valid <var>".

Section parsing, ``--select`` / ``--xaxis`` override, ``--clim`` and display
behaviour are shared with ``profiles`` / ``scalar`` via :mod:`sections`; panel
styling (colormap / colorbar label / norm / reversed-bar) is reused from
:mod:`profiles` so a given variable reads identically across subcommands.
"""

from __future__ import annotations

import argparse
import contextlib
import locale
import os
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import xarray as xr

from odas_tpw.perturb import resolve
from odas_tpw.perturb.plot import grid, layout, xaxis
from odas_tpw.perturb.plot import profiles as _prof
from odas_tpw.perturb.plot.scalar import _time_subset
from odas_tpw.perturb.plot.sections import (
    add_section_arguments,
    can_display,
    close_new_figs_on_error,
    closing_figs,
    fig_dpi,
    grouped,
    parse_clim,
    resolve_sections,
    safe_name,
    save_or_show,
    single_var_limit_guard,
    var_label,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# Fixed dissipation rows; the context row (row 3) is chosen by --bottom (a named
# set) or overridden outright by --var.
_DEFAULT_EPS = "epsilonMean"
_DEFAULT_CHI = "chiMean"
_BOTTOM_SETS: dict[str, tuple[str, ...]] = {
    # dT/dz, conservative temperature and salinity — the CTD/water-mass context.
    "ctd": ("dTdz", "CT", "SP"),
    # Osborn/Osborn-Cox diffusivities and the mixing efficiency (from chi_combo).
    "mixing": ("K_rho", "K_T", "Gamma"),
}
_DEFAULT_BOTTOM = "ctd"

# Per variable, the products to look in (first present wins) and how that
# product renders: ``col`` = a (bin, profile) depth-by-column mesh; ``traj`` =
# the ``ctd_combo`` continuous trajectory gridded to depth-vs-x.  CT/SP/sigma0
# are depth-binned into ``combo`` (preferred, same column path as ε/χ) and also
# exist as a (time,) trajectory in ``ctd_combo`` (fallback).  dT/dz and N2 live
# in all three depth products; the background ``combo`` copy pairs most
# naturally with the CTD context row.
_VAR_STAGES: dict[str, tuple[tuple[str, str], ...]] = {
    "epsilonMean": (("diss_combo", "col"),),
    "e_1": (("diss_combo", "col"),),
    "e_2": (("diss_combo", "col"),),
    "chiMean": (("chi_combo", "col"),),
    "chi_1": (("chi_combo", "col"),),
    "chi_2": (("chi_combo", "col"),),
    "dTdz": (("combo", "col"), ("chi_combo", "col"), ("diss_combo", "col")),
    "N2": (("combo", "col"), ("chi_combo", "col"), ("diss_combo", "col")),
    "CT": (("combo", "col"), ("ctd_combo", "traj")),
    "SP": (("combo", "col"), ("ctd_combo", "traj")),
    "SA": (("combo", "col"), ("ctd_combo", "traj")),
    "sigma0": (("combo", "col"), ("ctd_combo", "traj")),
    "rho": (("combo", "col"), ("ctd_combo", "traj")),
    "JAC_T": (("combo", "col"),),
    "T1": (("combo", "col"),),
    "T2": (("combo", "col"),),
    "T_mean": (("diss_combo", "col"), ("chi_combo", "col")),
    "K_T": (("chi_combo", "col"),),
    "K_rho": (("chi_combo", "col"),),
    "Gamma": (("chi_combo", "col"),),
}
# Unknown variables: search the depth products, then the CTD trajectory.
_DEFAULT_STAGES: tuple[tuple[str, str], ...] = (
    ("combo", "col"), ("chi_combo", "col"), ("diss_combo", "col"),
    ("ctd_combo", "traj"),
)

# A dissipation panel is masked by its product's own QC flag (same dataset).
_VAR_QC: dict[str, str] = {
    "epsilonMean": "qc_drop_epsilon", "e_1": "qc_drop_epsilon", "e_2": "qc_drop_epsilon",
    "chiMean": "qc_drop_chi", "chi_1": "qc_drop_chi", "chi_2": "qc_drop_chi",
}


class _Extent(NamedTuple):
    """What a drawn panel contributes to the shared axes / title."""

    z_lo: float
    z_hi: float
    x_lo: float
    x_hi: float
    xlabel: str
    xkind: str
    ncol: int  # cast columns (col panels); 0 for a trajectory panel


class _Stages:
    """Open each product's ``combo.nc`` at most once; close them all at the end.

    ``optional=True, conflict_ok=True`` so a product that simply was not run
    (or is ambiguous under ``--config``) degrades to ``None`` and its panels
    read "no valid <var>", rather than failing the whole figure.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._cache: dict[str, xr.Dataset | None] = {}

    def get(self, prefix: str) -> xr.Dataset | None:
        if prefix not in self._cache:
            src = resolve.resolve_for_args(
                self._args, prefix, optional=True, conflict_ok=True
            )
            ds: xr.Dataset | None = None
            if src is not None:
                path = src if src.endswith(".nc") else os.path.join(src, "combo.nc")
                if os.path.exists(path):
                    # ctd_combo is a CF trajectory (decode time -> datetime64);
                    # the binned products keep stime / bin numeric.
                    ds = xr.open_dataset(path, decode_times=(prefix == "ctd_combo"))
            self._cache[prefix] = ds
        return self._cache[prefix]

    def any_id(self) -> str | None:
        for ds in self._cache.values():
            if ds is not None and ds.attrs.get("id"):
                return str(ds.attrs["id"])
        return None

    def close(self) -> None:
        for ds in self._cache.values():
            if ds is not None:
                ds.close()


def _resolve_var(var: str, stages: _Stages) -> tuple[str, xr.Dataset] | None:
    """First (kind, dataset) whose product exists and carries *var*, else None."""
    for prefix, kind in _VAR_STAGES.get(var, _DEFAULT_STAGES):
        ds = stages.get(prefix)
        if ds is not None and var in ds.data_vars:
            return kind, ds
    return None


def _style(ds: xr.Dataset, var: str):
    """(cmap_name, cbar_label, reverse_cbar) for *var*, reusing the profiles maps."""
    import cmocean

    cmap = getattr(cmocean.cm, _prof._CMAP.get(var, "thermal")).copy()
    cmap.set_bad(color="0.85")  # unsampled cells: light gray
    label = _prof._CBAR_LABEL.get(var, var_label(ds, var))
    return cmap, label, var in _prof._REVERSE_CBAR


def _draw_col_panel(
    ax: Axes, fig: Figure, ds: xr.Dataset, sec, var: str,
    args: argparse.Namespace, clim: dict,
) -> _Extent | None:
    """Draw a (bin, profile) field as a depth-by-column mesh (profiles path)."""
    win = _prof._profile_window(ds["stime"].values, sec)
    dss = ds.isel(profile=np.flatnonzero(win))
    if dss.sizes.get("profile", 0) == 0 or {"bin", "profile"} - set(dss[var].dims):
        return None
    lat, lon = dss["lat"].values, dss["lon"].values
    xa = xaxis.compute(sec.method, lat, lon, dss["stime"].values, sec.params)
    x = np.asarray(xa.x, dtype=float)
    finite = np.flatnonzero(np.isfinite(x))
    if finite.size == 0:
        return None
    col = finite[np.argsort(x[finite])]  # finite casts, ordered by x
    xs = x[col]
    depth = np.asarray(dss["bin"].values, dtype=float)
    z = np.asarray(dss[var].transpose("bin", "profile").values, dtype=float)[:, col]

    qc_var = _VAR_QC.get(var)
    if args.apply_qc and qc_var and qc_var in dss.data_vars:
        flag = np.asarray(dss[qc_var].transpose("bin", "profile").values)[:, col]
        z = np.where(np.isfinite(flag) & (flag > 0), np.nan, z)

    norm = _prof._make_norm(var, z, args, clim)
    if norm is None or not np.any(np.isfinite(z)):
        return None
    cmap, label, reverse = _style(ds, var)
    layout.plot_columns(ax, fig, xs, depth, z, cmap, norm, label,
                        gap_factor=args.gap_factor, reverse_cbar=reverse)

    rows = depth[np.any(np.isfinite(z), axis=1)]
    z_lo, z_hi = (float(rows.min()), float(rows.max())) if rows.size else (np.nan, np.nan)
    return _Extent(z_lo, z_hi, float(xs.min()), float(xs.max()),
                   xa.label, xa.kind, int(xs.size))


def _draw_traj_panel(
    ax: Axes, fig: Figure, ds: xr.Dataset, sec, var: str,
    args: argparse.Namespace, clim: dict,
) -> _Extent | None:
    """Grid a ``ctd_combo`` trajectory scalar onto depth-vs-x (scalar path)."""
    dss = _time_subset(ds, sec)
    if dss.sizes.get("time", 0) == 0 or "depth" not in dss:
        return None  # no window, or a trajectory without a depth axis -> "no valid"
    n = dss.sizes["time"]
    lat = dss["lat"].values if "lat" in dss else np.full(n, np.nan)
    lon = dss["lon"].values if "lon" in dss else np.full(n, np.nan)
    depth = dss["depth"].values
    xa = xaxis.compute(sec.method, lat, lon, dss["time"].values, sec.params)
    x = np.asarray(xa.x, dtype=float)
    finite_x = x[np.isfinite(x)]
    if finite_x.size == 0:
        return None
    x_lo, x_hi = float(finite_x.min()), float(finite_x.max())
    x_edges = grid.make_edges(x_lo, x_hi, args.x_bin or grid.auto_step(x_lo, x_hi))
    finite_z = depth[np.isfinite(depth)]
    z_hi = float(finite_z.max()) if finite_z.size else 1.0
    if args.p_max is not None:
        z_hi = float(args.p_max)
    z_edges = grid.make_edges(0.0, z_hi, args.z_bin)
    g, _count = grid.grid_mean(x, depth, dss[var].values, x_edges, z_edges)
    if not np.any(np.isfinite(g)):
        return None
    norm = _prof._make_norm(var, g, args, clim)
    if norm is None:
        return None
    cmap, label, reverse = _style(ds, var)
    pcm = ax.pcolormesh(x_edges, z_edges, np.ma.masked_invalid(g),
                        cmap=cmap, norm=norm, shading="flat")
    cbar = fig.colorbar(pcm, ax=ax, label=label)
    if reverse:
        cbar.ax.invert_yaxis()

    z_cen = 0.5 * (z_edges[:-1] + z_edges[1:])
    zv = z_cen[np.any(np.isfinite(g), axis=1)]
    z_lo, z_hi2 = (float(zv.min()), float(zv.max())) if zv.size else (np.nan, np.nan)
    return _Extent(z_lo, z_hi2, x_lo, x_hi, xa.label, xa.kind, 0)


def _time_fmt(val: float, _pos) -> str:
    """UTC 'YYYY-MM-DD HH:MM' tick label for an epoch-seconds x-axis."""
    return np.datetime_as_string(np.datetime64(int(val), "s"), unit="m").replace("T", " ")


def _no_data(ax: Axes, var: str) -> None:
    ax.text(0.5, 0.5, f"no valid {var}", transform=ax.transAxes,
            ha="center", va="center")


def _build_overview_figure(
    stages: _Stages, sec, eps_var: str, chi_var: str, bottom: list[str],
    args: argparse.Namespace, clim: dict,
) -> Figure | None:
    """Render one section's 3-row overview to a Figure (not saved/shown here)."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    ncols = max(1, len(bottom))
    figsize = tuple(args.figsize) if args.figsize else (max(11.0, 4.5 * ncols), 11.0)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    # Two independently laid-out regions: the full-width epsilon/chi stack over
    # the multi-panel context row.  Subfigures keep each region's per-panel
    # colorbars aligned *within* the region — a single spanning gridspec detaches
    # the context-row colorbars from their panels (constrained_layout can't
    # reconcile a 3-wide colorbar row under two full-width spanning bars).
    # squeeze=False -> a (2, 1) array of SubFigures. (The default squeeze=True
    # returns a 1-D array here, which unpacks fine at runtime but trips mypy: its
    # overload for the squeezed form is typed as a scalar SubFigure — "not
    # iterable" — so pin squeeze=False and index explicitly.)
    subfigs = fig.subfigures(2, 1, squeeze=False, height_ratios=(2.0, 1.2))
    top_sf, bot_sf = subfigs[0, 0], subfigs[1, 0]
    ax_eps, ax_chi = top_sf.subplots(2, 1, sharex=True, sharey=True)
    bottom_axes = list(
        bot_sf.subplots(1, ncols, sharex=True, sharey=True, squeeze=False)[0]
    )
    axes = [ax_eps, ax_chi, *bottom_axes]
    panel_vars = [eps_var, chi_var, *bottom]

    extents: list[_Extent] = []
    for ax, var in zip(axes, panel_vars):
        res = _resolve_var(var, stages)
        if res is None:
            _no_data(ax, var)
            continue
        kind, ds = res
        draw = _draw_traj_panel if kind == "traj" else _draw_col_panel
        ext = draw(ax, fig, ds, sec, var, args, clim)
        if ext is None:
            _no_data(ax, var)
        else:
            extents.append(ext)

    if not extents:
        plt.close(fig)
        print(f"section {sec.name!r}: no data on any panel; skipped")
        return None

    for ax in axes:
        ax.set_axisbelow(False)  # grid over the mesh so it reads on any colormap
        ax.grid(True, color="0.4", linewidth=0.4, alpha=0.5)
    ax_chi.tick_params(labelbottom=False)  # x labels live only on the context row
    for ax in (ax_eps, ax_chi, bottom_axes[0]):
        ax.set_ylabel("Depth (m)")  # left column of every row

    # Shared inverted depth axis (0 m at top). The two subfigure groups share y
    # only within themselves, so pin the range on one axis of each group;
    # set_ylim(z, 0) inverts without a separate invert_yaxis() call.
    if args.p_max is not None:
        z_top = float(args.p_max)
    else:
        z_his = [e.z_hi for e in extents if np.isfinite(e.z_hi)]
        z_top = max(z_his) * 1.02 if z_his else 1.0
    x_los = [e.x_lo for e in extents if np.isfinite(e.x_lo)]
    x_his = [e.x_hi for e in extents if np.isfinite(e.x_hi)]
    for ax in (ax_eps, bottom_axes[0]):  # one per shared-axis group
        ax.set_ylim(z_top, 0.0)
        if x_los and x_his:
            ax.set_xlim(min(x_los), max(x_his))

    xlabel, xkind = extents[0].xlabel, extents[0].xkind
    if xkind == "time":
        for ax in bottom_axes:
            ax.xaxis.set_major_formatter(FuncFormatter(_time_fmt))
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(30)
                lbl.set_horizontalalignment("right")
    # One shared x label centered under the context row: the three panels share
    # the section x-axis, and its label (often long — e.g. a signed_distance
    # origin + orientation) would overlap if repeated under each panel.
    bot_sf.supxlabel(xlabel)

    ncasts = max((e.ncol for e in extents), default=0)
    title_id = stages.any_id() or os.path.basename(os.path.normpath(args.root))
    cast_note = f"  —  {grouped(ncasts)} casts" if ncasts else ""
    fig.suptitle(getattr(args, "title", None) or (
        f"{title_id}  —  overview: {sec.name}  —  x-axis: {sec.method}{cast_note}"
    ))
    layout.fit_colorbar_labels(fig)  # long var labels overflow short per-panel bars
    return fig


# ---------------------------------------------------------------------------
# CLI plumbing (perturb-plot subcommand contract)
# ---------------------------------------------------------------------------


def add_arguments(p: argparse.ArgumentParser) -> None:
    """Register CLI flags for the overview subcommand on *p*.

    ``--var`` (repeatable) overrides the bottom *context* row only; the two
    dissipation rows come from ``--eps-var`` / ``--chi-var``.
    """
    add_section_arguments(p)
    p.add_argument("--bottom", choices=sorted(_BOTTOM_SETS), default=_DEFAULT_BOTTOM,
                   help="named variable set for the context row: 'ctd' = "
                        "dT/dz, CT, SP (default); 'mixing' = K_rho, K_T, Gamma. "
                        "Overridden by --var.")
    p.add_argument("--eps-var", default=_DEFAULT_EPS,
                   help=f"variable for the top row (default: {_DEFAULT_EPS})")
    p.add_argument("--chi-var", default=_DEFAULT_CHI,
                   help=f"variable for the middle row (default: {_DEFAULT_CHI})")
    p.add_argument("--p-max", type=float, default=None,
                   help="clip the depth axis at this value [m]")
    p.add_argument("--gap-factor", type=float, default=4.0,
                   help="split casts into clusters when the x-gap exceeds this "
                        "multiple of the median cast spacing (default 4)")
    p.add_argument("--z-bin", type=float, default=1.0,
                   help="depth bin width [m] for CTD-trajectory panels (default 1)")
    p.add_argument("--x-bin", type=float, default=None,
                   help="x bin width for CTD-trajectory panels (default: ~300 columns)")
    p.add_argument("--apply-qc", dest="apply_qc", action="store_true", default=True,
                   help="NaN cells flagged by a dissipation product's qc_drop_* "
                        "field (default)")
    p.add_argument("--no-qc", dest="apply_qc", action="store_false",
                   help="ignore qc_drop_* and plot raw values")


def build_figures(args: argparse.Namespace) -> Iterator[tuple[str, Any]]:
    """Yield one ``(stem, Figure)`` per resolved section (no saving/showing).

    A **generator**, lazily, so a streaming caller (``run``'s save path, the
    ``figure`` PDF driver) can save and ``close`` each figure before the next is
    built. Sections with no data on any panel are skipped.
    """
    with contextlib.suppress(locale.Error):
        locale.setlocale(locale.LC_NUMERIC, "")

    args.root = resolve.require_root(args)  # backfill from --config if needed
    if args.gap_factor <= 0:
        raise SystemExit("--gap-factor must be > 0")

    bottom = list(args.var) if args.var else list(_BOTTOM_SETS[args.bottom])
    if len(bottom) > 6:
        raise SystemExit("overview: the context row takes at most 6 --var entries")
    # Overview is inherently multi-panel, so global --vmin/--vmax is ambiguous;
    # this raises the same clear message as the other subcommands (use --clim).
    single_var_limit_guard(args, [args.eps_var, args.chi_var, *bottom])
    clim = parse_clim(args.clim)

    stages = _Stages(args)
    try:
        sections = resolve_sections(args)
        for sec in sections:
            with close_new_figs_on_error():  # close a half-built figure if it raises
                fig = _build_overview_figure(
                    stages, sec, args.eps_var, args.chi_var, bottom, args, clim
                )
            if fig is not None:
                yield f"overview_{safe_name(sec.name)}", fig
    finally:
        stages.close()  # figures hold their own arrays, so datasets can close now


def run(args: argparse.Namespace) -> str:
    """Render every section's overview; show on screen, or write PNGs."""
    args.root = resolve.require_root(args)  # so out_dir/display are known up front
    display = args.out_dir is None and can_display()
    if display:
        with closing_figs(build_figures(args)) as figs:
            shown = save_or_show(figs, None, fig_dpi(args))
        return f"displayed {shown} section(s)"

    out_dir = args.out_dir or args.root
    os.makedirs(out_dir, exist_ok=True)
    with closing_figs(build_figures(args)) as figs:
        save_or_show(figs, out_dir, fig_dpi(args))
    return str(out_dir)
