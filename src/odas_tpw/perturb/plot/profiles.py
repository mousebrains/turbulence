# Jun-2026, Claude and Pat Welch, pat@mousebrains.com
"""Depth-vs-x sections from binned (bin, profile) combos.

One engine drives four ``perturb-plot`` subcommands, each a :class:`ProductView`
bound to one product (the CLI dispatcher and the ``figure`` driver's presets
share these views):

* ``profiles`` -> ``combo_NN`` (binned slow channels: T1/T2, N2, dT/dz, ...)
* ``epsilon``  -> ``diss_combo_NN`` (epsilonMean / e_1 / e_2)
* ``chi``      -> ``chi_combo_NN`` (chiMean / chi_1 / chi_2)
* ``mixing``   -> ``chi_combo_NN`` (K_T / Gamma / K_rho)

These products are already a ``(bin, profile)`` grid with one ``lat``/``lon``/
``stime`` per cast, so — unlike :mod:`scalar` (which grids a continuous
trajectory) — each profile is one *column*.  The x-axis is computed per profile
via the shared :mod:`xaxis` kernel; columns are sorted by x and drawn one mesh
per x-cluster with blank gaps (:func:`layout.plot_columns`), so sparse/irregular
sampling stays honest rather than being stretched across unsampled water.

Vertical axis note: the binned products' ``bin`` coordinate is **depth in
meters** — profiles binned on pressure ``P`` are converted to depth at write
time via ``gsw.z_from_p`` (:mod:`binning`), so ``bin:units="m"`` is correct and
the y-axis reads ``Depth (m)``.  Section parsing, ``--select``/``--xaxis``
override, ``--clim``, and display behavior are shared with ``scalar`` via
:mod:`sections`.
"""

from __future__ import annotations

import argparse
import contextlib
import locale
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from odas_tpw.perturb import resolve
from odas_tpw.perturb.plot import diagnostics, grid, layout, xaxis
from odas_tpw.perturb.plot.sections import (
    Section,
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
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class _Product:
    dir_prefix: str          # primary latest_stage_dir prefix (combo / chi_combo / ...)
    default_vars: tuple[str, ...]
    qc_vars: tuple[str, ...]  # companion qc_drop_* fields; union-ORed into the mask
    label: str               # user-facing name (subcommand / title / file stem)
    default_ncols: int = 1   # column count when --ncols is not given
    merge_dirs: tuple[str, ...] = ()  # extra combo dirs merged onto the primary grid


PRODUCTS: dict[str, _Product] = {
    # Default profiles preset: a 3x3 overview (temperatures / water-mass /
    # kinematics + stratification). Missing fields (e.g. SP/rho/sigma0 before a
    # pipeline re-run) are silently dropped from the grid.
    "profiles": _Product(
        "combo",
        ("JAC_T", "T1", "T2", "SP", "rho", "sigma0", "W_slow", "dTdz", "N2"),
        (), "profiles", default_ncols=3,
    ),
    # Default epsilon preset: a 3-column overview (kinematics / window means /
    # per-probe + combined dissipation / stratification).
    "diss": _Product(
        "diss_combo",
        ("speed", "nu", "T_mean", "e_1", "e_2", "epsilonMean", "N2", "dTdz"),
        ("qc_drop_epsilon",), "epsilon", default_ncols=3,
    ),
    # Default chi preset: a 3-column overview (kinematics / window means /
    # per-probe + combined chi / stratification / QC flag).
    "chi": _Product(
        "chi_combo",
        ("speed", "nu", "T_mean", "chi_1", "chi_2", "chiMean", "N2", "dTdz",
         "qc_drop_chi"),
        ("qc_drop_chi",), "chi", default_ncols=3,
    ),
    # Mixing draws from BOTH the chi combo (chi / K_T / K_rho / Gamma) and the
    # diss combo (e_*), merged on their shared (bin, profile) grid. A cell is
    # dropped if EITHER product's QC flags it (union of qc_drop_epsilon/chi).
    "mixing": _Product(
        "chi_combo",
        ("e_1", "e_2", "epsilonMean", "chi_1", "chi_2", "chiMean",
         "K_T", "K_rho", "Gamma"),
        ("qc_drop_epsilon", "qc_drop_chi"), "mixing", default_ncols=3,
        merge_dirs=("diss_combo",),
    ),
}

# Per-variable cmocean colormap (default thermal).
_CMAP: dict[str, str] = {
    "T1": "thermal", "T2": "thermal", "JAC_T": "thermal", "T_mean": "thermal",
    "N2": "amp", "dTdz": "balance",
    "epsilonMean": "thermal", "e_1": "thermal", "e_2": "thermal",
    "chiMean": "thermal", "chi_1": "thermal", "chi_2": "thermal",
    "K_T": "tempo", "K_rho": "tempo", "Gamma": "matter",
    "Incl_X": "balance", "Incl_Y": "balance",
    "CT": "thermal", "SP": "haline", "SA": "haline",
    "sigma0": "dense", "rho": "dense",
}
# Log-scaled fields (span orders of magnitude, strictly positive).
_LOG_VARS: frozenset[str] = frozenset({
    "epsilonMean", "e_1", "e_2", "chiMean", "chi_1", "chi_2",
    "K_T", "K_rho", "Gamma",
})
# Log-scaled but sign-bearing (SymLogNorm): N2 is usually positive but can go
# negative in overturning/unstable patches, which LogNorm would hide as no-data.
_SYMLOG_VARS: frozenset[str] = frozenset({"N2"})
# Diverging fields centered on zero.
_DIVERGING: frozenset[str] = frozenset({"dTdz", "Incl_X", "Incl_Y"})

# Explicit colorbar labels overriding the CF long_name/units default
# (var_label) for the profile-product scalars. Mathtext renders the sub/
# superscripts and the dT/dz fraction; the degree sign stays outside mathtext
# as a literal Unicode char. Note dTdz is a *conservative*-temperature (theta)
# gradient (processing.mixing._stable_window), Thorpe-sorted over the background
# window -- the label keeps the familiar dT/dz form.
_CBAR_LABEL: dict[str, str] = {
    "T1": r"$T_1$ (°C)",
    "T2": r"$T_2$ (°C)",
    "N2": r"$N^2$ (s$^{-2}$) (Thorpe-sorted)",
    "dTdz": r"$\frac{dT}{dz}$ (°C m$^{-1}$) (+ downwards)",
    "JAC_T": "JAC_T (°C)",
    "JAC_C": "JAC_C (mS/cm)",
    "CT": r"$\Theta$ (°C)",
    "SP": "Salinity (PSU)",
    "SA": "Salinity (g/kg)",
    "sigma0": r"$\sigma_0$ (kg m$^{-3}$)",
    "rho": r"$\rho$-1000 (kg m$^{-3}$) (in-situ)",
    "P_dP": "P_dP (dbar)",
    "Incl_X": "incl_X (°)",
    "Incl_Y": "incl_Y (°)",
    "Incl_T": "incl_T (°C)",
    "W_slow": r"W_slow (dbar s$^{-1}$)",
    # Epsilon (diss) product: Greek nu for viscosity, angle brackets for the
    # window-mean temperature and combined epsilon (an overline renders faintly
    # at colorbar-label size), and per-probe epsilons as ε_n (the "probe n" is
    # redundant with the subscript).
    "nu": r"$\nu$ (m$^2$ s$^{-1}$)",
    "T_mean": r"$\langle T \rangle$ (°C)",
    "e_1": r"$\epsilon_1$ (W kg$^{-1}$)",
    "e_2": r"$\epsilon_2$ (W kg$^{-1}$)",
    "epsilonMean": r"$\langle \epsilon \rangle$ (W kg$^{-1}$)",
    # Chi (thermal-variance) and mixing product: per-probe χ_n, angle-bracket
    # combined χ, the Osborn-Cox / Osborn diffusivities, and dimensionless Γ.
    "chi_1": r"$\chi_1$ (K$^2$ s$^{-1}$)",
    "chi_2": r"$\chi_2$ (K$^2$ s$^{-1}$)",
    "chiMean": r"$\langle \chi \rangle$ (K$^2$ s$^{-1}$)",
    "K_T": r"$K_T$ (m$^2$ s$^{-1}$)",
    "K_rho": r"$K_\rho$ (m$^2$ s$^{-1}$)",
    "Gamma": r"$\Gamma$",
}

# Variables whose colorbar reads with the smallest value at the top (axis
# inverted) rather than the matplotlib default (smallest at the bottom).
# SP/sigma0 mirror the scalar product (salinity/density rise with depth); nu
# (viscosity) rises with depth as the water cools, so it reads the same way.
_REVERSE_CBAR: frozenset[str] = frozenset({"P_dP", "SP", "sigma0", "nu"})


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
            f"color-scale minimum must be < maximum for {name!r} (got {lo}, {hi})"
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
        # Center on zero only when the robust (1/99%) range actually straddles
        # it.  One-signed data -- an all-stable dTdz (~ -0.17..0 K/m) or an
        # offset Incl_Y (~76..90 deg) -- would otherwise mirror an unused sign
        # and waste half the colorbar, making the range look far too wide.
        if (not explicit and vmin is not None and vmax is not None
                and vmin < 0.0 < vmax):
            m = max(abs(vmin), abs(vmax))
            return Normalize(vmin=-m, vmax=m)
        return Normalize(vmin=vmin, vmax=vmax)

    if name in _LOG_VARS or diagnostics.is_pseudo_var(name):  # variances are log
        if lo is not None and lo <= 0:
            raise SystemExit(
                f"color-scale minimum for log variable {name!r} must be > 0 (got {lo})"
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


def _union_qc_mask(
    dss: xr.Dataset, qc_vars: tuple[str, ...], col: np.ndarray
) -> np.ndarray | None:
    """Boolean (bin, selected-profile) mask: True where ANY qc_drop_* flag is set.

    Products may carry more than one flag field (mixing pulls both
    ``qc_drop_epsilon`` and ``qc_drop_chi``); a cell is dropped if either marks
    it (union). Absent flag fields are skipped; ``None`` means no QC to apply.
    """
    mask: np.ndarray | None = None
    for qv in qc_vars:
        if qv not in dss.data_vars:
            continue
        flag = np.asarray(dss[qv].transpose("bin", "profile").values)[:, col]
        marked = np.isfinite(flag) & (flag > 0)
        mask = marked if mask is None else (mask | marked)
    return mask


def _merge_source_dirs(
    ds: xr.Dataset, product: _Product, args: argparse.Namespace
) -> xr.Dataset:
    """Merge variables from ``product.merge_dirs`` onto the primary dataset.

    Mixing draws from both the diss and chi combos, which share an identical
    (bin, profile) grid from the same pipeline run. Extra variables (and their
    qc_drop_* flags) are copied onto the primary dataset using its own
    coordinates; shared variables keep the primary's copy. A missing extra dir
    is skipped -- its variables simply drop out of the default grid -- but a
    grid that does not line up is a hard error rather than a silent mis-join.
    """
    for prefix in product.merge_dirs:
        # optional=True: a supplemental combo that simply was not produced (e.g.
        # a chi-only run with no diss) degrades to None and is skipped in BOTH
        # --root and --config modes -- matching the diagnostics / eps-chi idiom.
        # (A genuine config *conflict* still raises, so data is never dropped
        # silently over an ambiguous stage.)
        src = resolve.resolve_for_args(args, prefix, optional=True)
        if src is None:
            continue
        path = os.path.join(src, "combo.nc")
        if not os.path.exists(path):
            continue
        with xr.open_dataset(path, decode_times=False) as extra:
            # The assignment below is POSITIONAL (it bypasses xarray coordinate
            # alignment), so the two combos must share the same grid by *value*,
            # not merely the same counts -- otherwise extra data would be placed
            # on the primary's axes and silently mis-joined.
            if (extra.sizes.get("bin") != ds.sizes.get("bin")
                    or extra.sizes.get("profile") != ds.sizes.get("profile")):
                raise SystemExit(
                    f"{prefix} grid {dict(extra.sizes)} != {product.dir_prefix} "
                    f"{dict(ds.sizes)}; cannot merge mixing sources")
            if "bin" in ds.coords and "bin" in extra.coords and not np.allclose(
                    np.asarray(ds["bin"].values, dtype=float),
                    np.asarray(extra["bin"].values, dtype=float),
                    rtol=0.0, atol=1e-6):
                raise SystemExit(
                    f"{prefix} depth bins differ from {product.dir_prefix}; "
                    "cannot merge mixing sources")
            # stime is epoch seconds (~1.7e9), where np.allclose's default
            # rtol=1e-5 spans ~4.8 h -- far too loose to detect different casts.
            # Compare with a 1 s absolute tolerance (and no relative term).
            if "stime" in ds and "stime" in extra and not np.allclose(
                    np.asarray(ds["stime"].values, dtype=float),
                    np.asarray(extra["stime"].values, dtype=float),
                    rtol=0.0, atol=1.0, equal_nan=True):
                raise SystemExit(
                    f"{prefix} profiles (stime) differ from {product.dir_prefix}; "
                    "the combos are not the same casts")
            for name, da in extra.data_vars.items():
                if name in ds.data_vars or not set(da.dims) <= set(ds.dims):
                    continue
                ds[name] = (da.dims, da.load().values, dict(da.attrs))
    return ds


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
    from matplotlib.ticker import FuncFormatter

    dss = ds.isel(profile=np.flatnonzero(_profile_window(ds["stime"].values, sec)))
    if dss.sizes.get("profile", 0) == 0:
        print(f"section {sec.name!r}: no profiles in time window; skipped")
        return None

    def _available(v: str) -> bool:
        if diagnostics.is_pseudo_var(v):
            return True
        if v not in dss.data_vars:
            return False
        # A plottable panel var needs both depth (bin) and profile dims; a 1-D
        # (profile-only) var would crash the transpose("bin","profile") below,
        # so treat it as missing -> clear "not on product" message.
        return {"bin", "profile"} <= set(dss[v].dims)

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
    depth = np.asarray(dss["bin"].values, dtype=float)  # depth, m

    # Diagnostic pseudo-variables (shear/vibration/T_dT variance) are computed
    # at plot time from the raw per-profile files, matched by stime.
    pseudo_grids: dict[str, np.ndarray] = {}
    pseudo_in_panel = [v for v in panel_vars if diagnostics.is_pseudo_var(v)]
    if pseudo_in_panel:
        pdir = resolve.resolve_for_args(args, "profiles", optional=True)
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

    qc = _union_qc_mask(dss, product.qc_vars, col) if args.apply_qc else None
    if qc is not None and qc.any():
        print(f"section {sec.name!r}: QC flags NaN {int(qc.sum())} cells per panel")

    # Panels in `ncols` columns, filled left-to-right, top-to-bottom. When
    # --ncols is not given, fall back to the product's default (profiles = 3;
    # a vertical stack elsewhere). Sections remain one figure each.
    ncols = getattr(args, "ncols", None)
    if ncols is None:
        ncols = product.default_ncols
    fig, axes, left_axes, col_bottom = layout.panel_grid(
        len(panel_vars), ncols,
        figsize=getattr(args, "figsize", None),
    )
    left_set = set(left_axes)
    # Depth rows (bins) that carry finite data in at least one plotted panel;
    # used to fit the shared depth axis to where there is valid data.
    valid_rows = np.zeros(depth.shape, dtype=bool)

    for ax, name in zip(axes, panel_vars):
        is_pseudo = diagnostics.is_pseudo_var(name)
        if is_pseudo:
            z = pseudo_grids[name][:, col]            # already (bin, profile)
            cmap_name, label = diagnostics.pseudo_cmap(name), diagnostics.pseudo_label(name)
        else:
            z = np.asarray(dss[name].transpose("bin", "profile").values, dtype=float)[:, col]
            # Apply the QC mask to the data panels, but never to a flag field
            # itself -- plotting qc_drop_* is how you *see* which cells are
            # flagged, so self-masking it would blank exactly what you asked for.
            if qc is not None and name not in product.qc_vars:
                z = np.where(qc, np.nan, z)
            cmap_name = _CMAP.get(name, "thermal")
            label = _CBAR_LABEL.get(name, var_label(ds, name))
            # Flag in-situ-calibrated channels (FP07 T1/T2) from the variable's
            # own provenance attr -- reflects what was actually applied, not the
            # config intent (calibrate=true can fall back to factory).
            cal = ds[name].attrs.get("calibration", "") if name in ds else ""
            if isinstance(cal, str) and cal.startswith("in-situ"):
                label = f"{label} (in-situ calib)"
        norm = _make_norm(name, z, args, clim)
        if norm is None or not np.any(np.isfinite(z)):
            ax.text(0.5, 0.5, f"no valid {name}", transform=ax.transAxes,
                    ha="center", va="center")
            if ax in left_set:  # only the left column carries the shared y label
                ax.set_ylabel("Depth (m)")
            continue
        valid_rows |= np.any(np.isfinite(z), axis=1)
        cmap = getattr(cmocean.cm, cmap_name).copy()
        cmap.set_bad(color="0.85")  # unsampled depths: light gray
        layout.plot_columns(ax, fig, xs, depth, z, cmap, norm, label,
                            gap_factor=args.gap_factor,
                            reverse_cbar=name in _REVERSE_CBAR)
        if ax in left_set:
            ax.set_ylabel("Depth (m)")

    for ax in axes:
        # Grid over the color mesh (axisbelow False) so it reads on any colormap.
        ax.set_axisbelow(False)
        ax.grid(True, color="0.4", linewidth=0.4, alpha=0.5)

    axes[0].invert_yaxis()  # 0 m at top (shared across panels)
    if args.p_max is not None:
        # Explicit depth clip wins over the data-driven fit.
        axes[0].set_ylim(float(args.p_max), 0.0)
    elif valid_rows.any():
        # Fit the depth axis to where there is valid data rather than spanning
        # the whole combo's bin grid (which pads the section with empty gray
        # below its deepest sample). Pad half a bin so edge cells aren't clipped.
        dv = depth[valid_rows]
        pad = 0.5 * float(np.median(np.diff(depth))) if depth.size > 1 else 0.0
        axes[0].set_ylim(float(dv.max()) + pad, max(float(dv.min()) - pad, 0.0))

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
    fig.suptitle(getattr(args, "title", None) or (
        f"{title_id}  —  {product.label}: {sec.name}  —  "
        f"x-axis: {sec.method}  —  {grouped(int(dss.sizes['profile']))} casts"
    ))
    layout.fit_colorbar_labels(fig)  # long var labels overflow short per-panel bars
    return fig


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def add_arguments(p: argparse.ArgumentParser) -> None:
    """Register the shared (bin, profile) section flags on *p*.

    The product itself is fixed per subcommand (``profiles`` / ``epsilon`` /
    ``chi`` / ``mixing``) by the :class:`ProductView` that owns *p*, so there is
    no ``--product`` flag; the engine reads ``args.product`` set by the view.
    """
    add_section_arguments(p)
    p.add_argument("--p-max", type=float, default=None,
                   help="clip the depth axis at this value [m]")
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


def build_figures(args: argparse.Namespace) -> Iterator[tuple[str, Any]]:
    """Yield one ``(stem, Figure)`` per resolved section (no saving/showing).

    A **generator**, lazily, so a streaming caller (``run``'s save path, the
    ``figure`` PDF driver) can save and ``close`` each figure before the next is
    built — bounded to one open figure at a time. Sections with no finite data
    are skipped, so fewer figures may be yielded than there are sections.
    """
    with contextlib.suppress(locale.Error):
        locale.setlocale(locale.LC_NUMERIC, "")

    args.root = resolve.require_root(args)  # backfill from --config if needed
    if args.gap_factor <= 0:
        raise SystemExit("--gap-factor must be > 0")

    product = PRODUCTS[args.product]
    src = resolve.resolve_for_args(args, product.dir_prefix)
    if src is None:
        raise SystemExit(f"No {product.dir_prefix} dir under {args.root}")
    path = os.path.join(src, "combo.nc")
    if not os.path.exists(path):
        raise SystemExit(f"{product.label} combo not found: {path}")

    # decode_times=False keeps stime / bin numeric (epoch seconds / meters).
    ds = xr.open_dataset(path, decode_times=False)
    try:
        # Validate the primary grid first, so a malformed combo reports the
        # clear "expected a (bin, profile) product" rather than a merge-size
        # mismatch against a None dim.
        if "profile" not in ds.dims or "bin" not in ds.dims:
            raise SystemExit(f"{path}: expected a (bin, profile) product")
        ds = _merge_source_dirs(ds, product, args)  # pull in e.g. diss vars for mixing
        sections = resolve_sections(args)
        if args.var:
            variables = list(args.var)
        else:
            variables = [v for v in product.default_vars if v in ds.data_vars]
            if not variables:
                raise SystemExit(
                    f"{product.label}: none of the default variables "
                    f"{list(product.default_vars)} are present; available: "
                    f"{sorted(str(v) for v in ds.data_vars)}; pass --var"
                )
        clim = parse_clim(args.clim)
        single_var_limit_guard(args, variables)
        for sec in sections:
            with close_new_figs_on_error():  # close a half-built figure if it raises
                fig = _build_profiles_figure(ds, sec, variables, args, clim, product)
            if fig is not None:
                yield f"{product.label}_{safe_name(sec.name)}", fig
    finally:
        ds.close()  # figures hold their own arrays, so the dataset can close now


def run(args: argparse.Namespace) -> str:
    """Render every section of the selected product; show or write PNGs."""
    args.root = resolve.require_root(args)  # so out_dir/display are known up front
    display = args.out_dir is None and can_display()
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


# ---------------------------------------------------------------------------
# Per-product views
# ---------------------------------------------------------------------------
#
# One engine, one subcommand per product. Each view binds ``args.product`` and
# exposes the same ``add_arguments`` / ``build_figures`` / ``run`` surface the
# CLI dispatcher (:mod:`cli`) and the figure driver (:mod:`figure`) consume from
# a plain plot module — so ``perturb-plot epsilon`` and ``preset: epsilon`` reuse
# this file without a ``--product`` flag.


@dataclass(frozen=True)
class ProductView:
    """The profiles engine bound to one product (``diss`` / ``chi`` / ...).

    Duck-types a plot module: ``add_arguments`` registers the shared section
    flags (no ``--product``), and ``build_figures`` / ``run`` set
    ``args.product`` before delegating to the module-level functions.
    """

    product: str

    def _bind(self, args: argparse.Namespace) -> argparse.Namespace:
        args.product = self.product
        return args

    def add_arguments(self, p: argparse.ArgumentParser) -> None:
        add_arguments(p)

    def build_figures(self, args: argparse.Namespace) -> Iterator[tuple[str, Any]]:
        return build_figures(self._bind(args))

    def run(self, args: argparse.Namespace) -> str:
        return run(self._bind(args))


PROFILES = ProductView("profiles")
EPSILON = ProductView("diss")
CHI = ProductView("chi")
MIXING = ProductView("mixing")
