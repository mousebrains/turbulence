# Jun-2026, Claude and Pat Welch, pat@mousebrains.com
"""``perturb-plot profiles`` — depth-vs-x sections from binned (bin, profile) combos.

One engine for every per-profile product, selected by ``--product``:

* ``profiles`` -> ``combo_NN`` (binned slow channels: T1/T2, N2, dT/dz, ...)
* ``diss``     -> ``diss_combo_NN`` (epsilonMean / e_1 / e_2)
* ``chi``      -> ``chi_combo_NN`` (chiMean / chi_1 / chi_2)
* ``mixing``   -> ``chi_combo_NN`` (K_T / Gamma / K_rho)

These products are already a ``(bin, profile)`` grid with one ``lat``/``lon``/
``stime`` per cast, so — unlike :mod:`scalar` (which grids a continuous
trajectory) — each profile is one *column*.  The x-axis is computed per profile
via the shared :mod:`xaxis` kernel; columns are sorted by x and drawn one mesh
per x-cluster with blank gaps (:func:`layout.plot_columns`), so sparse/irregular
sampling stays honest rather than being stretched across unsampled water.

Vertical axis note: the binned products' ``bin`` coordinate is built from the
profile pressure ``P`` (dbar), so it is **pressure**, labelled here as such (the
product's ``bin:units="m"`` attribute is a known mislabel, separate from this
plot).  Section parsing, ``--select``/``--xaxis`` override, ``--clim``, and
display behaviour are shared with ``scalar`` via :mod:`sections`.
"""

from __future__ import annotations

import argparse
import contextlib
import locale
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from odas_tpw.perturb.plot import diagnostics, grid, layout, xaxis
from odas_tpw.perturb.plot.sections import (
    Section,
    add_section_arguments,
    can_display,
    grouped,
    parse_clim,
    resolve_sections,
    safe_name,
    single_var_limit_guard,
    var_label,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class _Product:
    dir_prefix: str          # latest_stage_dir prefix (combo / diss_combo / ...)
    default_vars: tuple[str, ...]
    qc_var: str | None       # companion qc_drop_* field, NaN'd when --apply-qc


PRODUCTS: dict[str, _Product] = {
    "profiles": _Product("combo", ("T1", "T2", "N2", "dTdz"), None),
    "diss": _Product("diss_combo", ("epsilonMean",), "qc_drop_epsilon"),
    "chi": _Product("chi_combo", ("chiMean",), "qc_drop_chi"),
    "mixing": _Product("chi_combo", ("K_T", "Gamma", "K_rho"), "qc_drop_chi"),
}

# Per-variable cmocean colormap (default thermal).
_CMAP: dict[str, str] = {
    "T1": "thermal", "T2": "thermal", "JAC_T": "thermal", "T_mean": "thermal",
    "N2": "amp", "dTdz": "balance",
    "epsilonMean": "thermal", "e_1": "thermal", "e_2": "thermal",
    "chiMean": "thermal", "chi_1": "thermal", "chi_2": "thermal",
    "K_T": "tempo", "K_rho": "tempo", "Gamma": "matter",
    "Incl_X": "balance", "Incl_Y": "balance",
}
# Log-scaled fields (span orders of magnitude, strictly positive).
_LOG_VARS: frozenset[str] = frozenset({
    "epsilonMean", "e_1", "e_2", "chiMean", "chi_1", "chi_2",
    "K_T", "K_rho", "Gamma",
})
# Log-scaled but sign-bearing (SymLogNorm): N2 is usually positive but can go
# negative in overturning/unstable patches, which LogNorm would hide as no-data.
_SYMLOG_VARS: frozenset[str] = frozenset({"N2"})
# Diverging fields centred on zero.
_DIVERGING: frozenset[str] = frozenset({"dTdz", "Incl_X", "Incl_Y"})


def _profile_window(stime: np.ndarray, sec: Section) -> np.ndarray:
    """Boolean mask of profiles whose start time falls in the section window.

    *stime* is per-profile epoch seconds (these combos have no ``time``
    dimension, so we filter the ``profile`` axis rather than ``isel(time=...)``).
    """
    stime = np.asarray(stime, dtype=float)
    mask = np.ones(stime.shape, dtype=bool)
    if sec.start is not None:
        mask &= stime >= float(xaxis.to_epoch_seconds(np.array([sec.start]))[0])
    if sec.stop is not None:
        mask &= stime <= float(xaxis.to_epoch_seconds(np.array([sec.stop]))[0])
    return mask


def _make_norm(name: str, z: np.ndarray, args: argparse.Namespace, clim: dict):
    """Pick a matplotlib norm for variable *name* on data *z*.

    log for dissipation/diffusivity/N2; symmetric-linear for diverging fields;
    linear otherwise.  Per-variable ``--clim`` wins, then global --vmin/--vmax
    (single-var only).  Returns ``None`` when a log field has no positive data.
    """
    from matplotlib.colors import LogNorm, Normalize, SymLogNorm

    lo, hi = clim.get(name, (args.vmin, args.vmax))
    explicit = name in clim or args.vmin is not None or args.vmax is not None

    # Reversed --clim (MIN >= MAX) would silently invert the colorbar on the
    # linear/diverging paths; validate centrally so every branch errors clearly.
    if lo is not None and hi is not None and lo >= hi:
        raise SystemExit(
            f"colour-scale minimum must be < maximum for {name!r} (got {lo}, {hi})"
        )

    if name in _SYMLOG_VARS:
        # N2 is log-scaled but can go slightly negative (overturning/unstable
        # patches); SymLog keeps the dynamic range while showing negatives
        # distinctly from no-data, where LogNorm would mask them.
        finite = z[np.isfinite(z)]
        if finite.size == 0:
            return None
        amax = float(np.nanmax(np.abs(finite)))
        if amax <= 0:
            return None
        pos = finite[finite > 0]
        linthresh = float(np.quantile(pos, 0.05)) if pos.size else amax * 1.0e-3
        vmin = lo if lo is not None else float(np.nanmin(finite))
        vmax = hi if hi is not None else amax
        return SymLogNorm(linthresh=max(linthresh, amax * 1.0e-9),
                          vmin=vmin, vmax=vmax)

    if name in _DIVERGING:
        vmin, vmax = grid.linear_limits(z, lo, hi)
        if not explicit and vmin is not None and vmax is not None:
            m = max(abs(vmin), abs(vmax))
            return Normalize(vmin=-m, vmax=m)
        return Normalize(vmin=vmin, vmax=vmax)

    if name in _LOG_VARS or diagnostics.is_pseudo_var(name):  # variances are log
        if lo is not None and lo <= 0:
            raise SystemExit(
                f"colour-scale minimum for log variable {name!r} must be > 0 (got {lo})"
            )
        vmin, vmax = layout.quantile_limits(z, lo, hi)  # filters to > 0
        if vmin is None or vmax is None or vmax <= 0:
            return None
        vmin = max(vmin, vmax / 1.0e6)
        if vmin >= vmax:  # all-equal data: show one decade so LogNorm is valid
            vmin = vmax / 10.0
        return LogNorm(vmin=vmin, vmax=vmax)

    vmin, vmax = grid.linear_limits(z, lo, hi)
    return Normalize(vmin=vmin, vmax=vmax)


def _build_profiles_figure(
    ds: xr.Dataset,
    sec: Section,
    variables: list[str],
    args: argparse.Namespace,
    clim: dict[str, tuple[float, float]],
    product: _Product,
) -> Figure | None:
    """Render one section of the (bin, profile) product to a Figure."""
    import cmocean
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    dss = ds.isel(profile=np.flatnonzero(_profile_window(ds["stime"].values, sec)))
    if dss.sizes.get("profile", 0) == 0:
        print(f"section {sec.name!r}: no profiles in time window; skipped")
        return None

    def _available(v: str) -> bool:
        return v in dss.data_vars or diagnostics.is_pseudo_var(v)

    panel_vars = [v for v in variables if _available(v)]
    missing = [v for v in variables if not _available(v)]
    if not panel_vars:
        print(f"section {sec.name!r}: none of {variables} present on product; skipped")
        return None
    if missing:
        print(f"section {sec.name!r}: variables not on product, skipped: {missing}")

    # Per-profile x; drop non-finite-x profiles and order columns by x.
    lat, lon = dss["lat"].values, dss["lon"].values
    xa = xaxis.compute(sec.method, lat, lon, dss["stime"].values, sec.params)
    x = np.asarray(xa.x, dtype=float)
    finite = np.flatnonzero(np.isfinite(x))
    if finite.size == 0:
        print(f"section {sec.name!r}: no finite x positions; skipped")
        return None
    col = finite[np.argsort(x[finite])]   # finite profiles, sorted by x
    xs = x[col]
    depth = np.asarray(dss["bin"].values, dtype=float)  # pressure, dbar

    # Diagnostic pseudo-variables (shear/vibration/T_dT variance) are computed
    # at plot time from the raw per-profile files, matched by stime.
    pseudo_grids: dict[str, np.ndarray] = {}
    pseudo_in_panel = [v for v in panel_vars if diagnostics.is_pseudo_var(v)]
    if pseudo_in_panel:
        pdir = layout.latest_stage_dir(args.root, "profiles")
        if pdir is None:
            print(f"section {sec.name!r}: no profiles_NN dir; skipping diagnostics "
                  f"{pseudo_in_panel}")
            panel_vars = [v for v in panel_vars if v not in pseudo_in_panel]
            if not panel_vars:
                return None
        else:
            for pv in pseudo_in_panel:
                pseudo_grids[pv] = diagnostics.compute_pseudo_grid(
                    pv, dss["stime"].values, depth, pdir,
                    hp_cut=args.hp_cut, despike_thresh=args.despike_thresh,
                    despike_smooth=args.despike_smooth, tol=args.stime_tol,
                )

    qc = None
    if args.apply_qc and product.qc_var and product.qc_var in dss.data_vars:
        qc = np.asarray(dss[product.qc_var].transpose("bin", "profile").values)[:, col]
        n_drop = int((np.isfinite(qc) & (qc > 0)).sum())
        if n_drop:
            print(f"section {sec.name!r}: QC flags NaN {n_drop} cells per panel")

    n = len(panel_vars)
    fig, axes = plt.subplots(
        n, 1, figsize=(11, 3.0 * n + 1.0), sharex=True, sharey=True,
        constrained_layout=True, squeeze=False,
    )
    axes = axes[:, 0]

    for ax, name in zip(axes, panel_vars):
        is_pseudo = diagnostics.is_pseudo_var(name)
        if is_pseudo:
            z = pseudo_grids[name][:, col]            # already (bin, profile)
            cmap_name, label = diagnostics.pseudo_cmap(name), diagnostics.pseudo_label(name)
        else:
            z = np.asarray(dss[name].transpose("bin", "profile").values, dtype=float)[:, col]
            if qc is not None:
                z = np.where(np.isfinite(qc) & (qc > 0), np.nan, z)
            cmap_name, label = _CMAP.get(name, "thermal"), var_label(ds, name)
        norm = _make_norm(name, z, args, clim)
        if norm is None or not np.any(np.isfinite(z)):
            ax.text(0.5, 0.5, f"no valid {name}", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_ylabel("Pressure (dbar)")
            continue
        cmap = getattr(cmocean.cm, cmap_name).copy()
        cmap.set_bad(color="0.85")  # unsampled depths: light grey
        layout.plot_columns(ax, fig, xs, depth, z, cmap, norm, label,
                            gap_factor=args.gap_factor)
        ax.set_ylabel("Pressure (dbar)")

    axes[0].invert_yaxis()
    if args.p_max is not None:
        axes[0].set_ylim(float(args.p_max), 0.0)

    bottom = axes[-1]
    bottom.set_xlabel(xa.label)
    if xa.kind == "time":
        bottom.xaxis.set_major_formatter(
            FuncFormatter(
                lambda val, _pos: np.datetime_as_string(
                    np.datetime64(int(val), "s"), unit="m"
                ).replace("T", " ")
            )
        )
        for lbl in bottom.get_xticklabels():
            lbl.set_rotation(30)
            lbl.set_ha("right")

    title_id = ds.attrs.get("id") or os.path.basename(os.path.normpath(args.root))
    fig.suptitle(
        f"{title_id}  —  {args.product}: {sec.name}  —  "
        f"x-axis: {sec.method}  —  {grouped(int(dss.sizes['profile']))} casts"
    )
    return fig


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def add_arguments(p: argparse.ArgumentParser) -> None:
    """Register CLI flags for the profiles subcommand on *p*."""
    add_section_arguments(p)
    p.add_argument("--product", choices=list(PRODUCTS), default="profiles",
                   help="which (bin, profile) product to plot (default: profiles)")
    p.add_argument("--p-max", type=float, default=None,
                   help="clip the pressure axis at this value [dbar]")
    p.add_argument("--gap-factor", type=float, default=4.0,
                   help="split casts into clusters when the x-gap exceeds this "
                        "multiple of the median cast spacing (default 4)")
    p.add_argument("--apply-qc", dest="apply_qc", action="store_true", default=True,
                   help="NaN cells flagged by the product's qc_drop_* field (default)")
    p.add_argument("--no-qc", dest="apply_qc", action="store_false",
                   help="ignore qc_drop_* and plot raw values")
    # Diagnostic pseudo-variables (--var sh1_var / Ax_var / T1_dT1_var ...):
    # binned variance of a raw fast channel, high-pass filtered + despiked (the
    # epsilon path's two steps, applied over the whole cast), matched to combo
    # casts by stime. Computed at plot time.
    p.add_argument("--hp-cut", type=float, default=1.0,
                   help="high-pass cutoff [Hz] for diagnostic variance channels (default 1)")
    p.add_argument("--despike-thresh", type=float, default=8.0,
                   help="despike threshold for diagnostic variance channels (default 8)")
    p.add_argument("--despike-smooth", type=float, default=0.5,
                   help="despike envelope cutoff [Hz] for diagnostics (default 0.5)")
    p.add_argument("--stime-tol", type=float, default=1.0,
                   help="max stime mismatch [s] when matching casts to raw files (default 1)")


def run(args: argparse.Namespace) -> str:
    """Render every section of the selected product; show or write PNGs."""
    with contextlib.suppress(locale.Error):
        locale.setlocale(locale.LC_NUMERIC, "")

    if args.gap_factor <= 0:
        raise SystemExit("--gap-factor must be > 0")

    product = PRODUCTS[args.product]
    src = layout.latest_stage_dir(args.root, product.dir_prefix)
    if src is None:
        raise SystemExit(f"No {product.dir_prefix} dir under {args.root}")
    path = os.path.join(src, "combo.nc")
    if not os.path.exists(path):
        raise SystemExit(f"{args.product} combo not found: {path}")

    display = args.out_dir is None and can_display()
    out_dir = args.out_dir or args.root
    if not display:
        os.makedirs(out_dir, exist_ok=True)

    import matplotlib.pyplot as plt

    # decode_times=False keeps stime / bin numeric (epoch seconds / dbar).
    ds = xr.open_dataset(path, decode_times=False)
    shown = 0
    try:
        if "profile" not in ds.dims or "bin" not in ds.dims:
            raise SystemExit(f"{path}: expected a (bin, profile) product")
        sections = resolve_sections(args)
        if args.var:
            variables = list(args.var)
        else:
            variables = [v for v in product.default_vars if v in ds.data_vars]
            if not variables:
                raise SystemExit(
                    f"--product {args.product}: none of the default variables "
                    f"{list(product.default_vars)} are present; available: "
                    f"{sorted(str(v) for v in ds.data_vars)}; pass --var"
                )
        clim = parse_clim(args.clim)
        single_var_limit_guard(args, variables)
        for sec in sections:
            fig = _build_profiles_figure(ds, sec, variables, args, clim, product)
            if fig is None:
                continue
            if display:
                shown += 1
            else:
                out = os.path.join(out_dir, f"{args.product}_{safe_name(sec.name)}.png")
                fig.savefig(out, dpi=150)
                plt.close(fig)
                print(f"Wrote {out}")
        if display and shown:
            plt.show()
            plt.close("all")
    finally:
        ds.close()
    return f"displayed {shown} section(s)" if display else str(out_dir)
