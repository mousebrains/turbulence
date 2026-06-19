# Jun-2026, Claude and Pat Welch, pat@mousebrains.com
"""``perturb-plot scalar`` — depth-vs-x scalar sections from the CTD combo.

Reads the CTD trajectory product ``ctd_combo_NN/combo.nc`` (CF
featureType=trajectory, a continuous down/up sawtooth on a ``time`` axis) and
renders, for each *section*, depth (y, inverted) against a chosen x-axis with a
scalar field in colour.

A **section** is purely a way of chopping the trajectory and choosing the
x-axis: a name, an optional UTC ``start``/``stop`` window, and an ``xaxis``
method with its parameters.  Sections come from a YAML file (``--sections``) or
from ad-hoc CLI flags.  *Rendering* choices (which variables, depth/x bin
sizes, colour limits) are separate CLI options that apply to every section in
the run.

By default the figures are shown on screen.  They are written to PNG files
instead when ``--out-dir`` is given, or when there is no interactive display
available (no controlling tty, or a non-GUI matplotlib backend such as Agg).

The trajectory is gridded by binning samples onto a regular ``(x, depth)``
mesh and averaging each cell (:func:`grid.grid_mean`); empty cells stay NaN —
no interpolation across gaps.
"""

from __future__ import annotations

import argparse
import contextlib
import locale
import os
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from odas_tpw.perturb.plot import grid, layout, xaxis

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# Default scalar panels when the user names none, plus optional sensor channels
# added automatically when the instrument carries them.
_DEFAULT_VARS: tuple[str, ...] = ("JAC_T", "SP", "sigma0")
_OPTIONAL_VARS: tuple[str, ...] = ("DO", "Chlorophyll", "Turbidity")

# Per-variable cmocean colormap.  Diverging fields centre the colour scale at 0.
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

# matplotlib backends that cannot show a window — figures must be saved instead.
_NON_INTERACTIVE_BACKENDS: frozenset[str] = frozenset(
    {"agg", "pdf", "svg", "ps", "cairo", "template", "pgf"}
)


def _can_display() -> bool:
    """True when figures can be shown interactively: a tty and a GUI backend."""
    if not sys.stdout.isatty():
        return False
    import matplotlib

    return matplotlib.get_backend().lower() not in _NON_INTERACTIVE_BACKENDS


# ---------------------------------------------------------------------------
# Section spec + loading
# ---------------------------------------------------------------------------


@dataclass
class Section:
    """One depth-vs-x section: a time window + an x-axis method/params."""

    name: str
    method: str
    start: np.datetime64 | None = None
    stop: np.datetime64 | None = None
    params: dict = field(default_factory=dict)


def _parse_time(value) -> np.datetime64 | None:
    """Parse an ISO-8601 string to a UTC datetime64.

    A trailing ``Z`` is accepted (and stripped); a bare timestamp is treated as
    UTC, matching the CTD product's encoding.  An explicit non-Z timezone
    offset is rejected rather than silently shifted.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1]
    # Reject an explicit numeric offset (e.g. +09:00) on the time-of-day part.
    tail = s.split("T", 1)[-1] if "T" in s else ""
    if "+" in tail or tail.count("-") > 0:
        raise ValueError(
            f"time {value!r} must be UTC: use a trailing 'Z' or no offset, "
            "not an explicit timezone"
        )
    try:
        return np.datetime64(s)
    except ValueError as exc:
        raise ValueError(f"could not parse time {value!r}: {exc}") from None


def _validate_params(method: str, params: dict, name: str) -> dict:
    """Check method-specific params and normalise units; return a clean dict."""
    if method not in xaxis.METHODS:
        raise ValueError(
            f"section {name!r}: unknown xaxis method {method!r}; "
            f"choose from {list(xaxis.METHODS)}"
        )
    out: dict = {}
    units = params.get("units", "km")
    if units not in ("m", "km", "nm"):
        raise ValueError(f"section {name!r}: units must be m/km/nm, got {units!r}")
    out["units"] = units
    if method == "distance_from_point":
        point = params.get("point")
        if point is None or len(point) != 2:
            raise ValueError(f"section {name!r}: distance_from_point needs point: [lat, lon]")
        out["point"] = [float(point[0]), float(point[1])]
    elif method == "along_line":
        wps = params.get("waypoints")
        if wps is None or len(wps) < 2:
            raise ValueError(
                f"section {name!r}: along_line needs waypoints: "
                "[[lat, lon], [lat, lon], ...] (>= 2)"
            )
        out["waypoints"] = [[float(a), float(b)] for a, b in wps]
    return out


def _section_from_dict(raw: dict, idx: int) -> Section:
    name = str(raw.get("name", f"section{idx:02d}"))
    xa = dict(raw.get("xaxis") or {})
    method = str(xa.pop("method", "time"))
    params = _validate_params(method, xa, name)
    return Section(
        name=name,
        method=method,
        start=_parse_time(raw.get("start")),
        stop=_parse_time(raw.get("stop")),
        params=params,
    )


def load_sections(path: str) -> list[Section]:
    """Load and validate a sections YAML (standalone; not the pipeline config).

    Schema::

        sections:
          - name: north_transect
            start: "2025-01-20T00:00:00Z"   # optional, UTC
            stop:  "2025-01-22T00:00:00Z"   # optional
            xaxis:
              method: along_line             # time|latitude|longitude|
                                             #   distance_from_point|along_line
              units: km                      # distance methods
              waypoints: [[18.5, 130.0], [20.0, 132.5]]   # [lat, lon]
    """
    from ruamel.yaml import YAML

    with open(path, encoding="utf-8") as fh:
        raw = YAML().load(fh) or {}
    secs = raw.get("sections")
    if not secs:
        raise ValueError(f"{path}: must contain a non-empty 'sections:' list")
    return [_section_from_dict(dict(s), i) for i, s in enumerate(secs)]


def _select_sections(sections: list[Section], select: list[str]) -> list[Section]:
    """Keep only sections whose ``name`` was requested via ``--select``.

    Each ``--select`` value may be a single name or a comma-separated list;
    values are flattened. An unknown name is a hard error (rather than a silent
    no-op) and lists the available names. Output preserves the file order.
    """
    wanted: list[str] = []
    for item in select:
        wanted.extend(part.strip() for part in item.split(",") if part.strip())
    available = [s.name for s in sections]
    unknown = [w for w in wanted if w not in available]
    if unknown:
        raise SystemExit(
            f"--select: no section named {unknown}; available: {available}"
        )
    wanted_set = set(wanted)
    return [s for s in sections if s.name in wanted_set]


def _xaxis_params_from_args(method: str, args: argparse.Namespace, name: str) -> dict:
    """Build (and validate) xaxis params for *method* from the CLI flags.

    Used both for the ad-hoc single section and for the ``--xaxis`` override of
    config sections, so spatial methods pull --point / --waypoints / --units
    the same way in either path.
    """
    params: dict = {"units": args.units}
    if method == "distance_from_point":
        if not args.point:
            raise SystemExit("--xaxis distance_from_point requires --point LAT LON")
        params["point"] = [float(args.point[0]), float(args.point[1])]
    elif method == "along_line":
        if not args.waypoints:
            raise SystemExit(
                "--xaxis along_line requires --waypoints 'lat,lon;lat,lon;...'"
            )
        params["waypoints"] = _parse_waypoints(args.waypoints)
    return _validate_params(method, params, name)


def _adhoc_section(args: argparse.Namespace) -> Section:
    """Build a single Section from ad-hoc CLI flags."""
    method = args.xaxis or "time"
    return Section(
        name=args.name,
        method=method,
        start=_parse_time(args.start),
        stop=_parse_time(args.stop),
        params=_xaxis_params_from_args(method, args, args.name),
    )


def _override_xaxis(sec: Section, args: argparse.Namespace) -> Section:
    """Return *sec* with its xaxis replaced by the CLI --xaxis (+ its params).

    Keeps the section's name and time window; the method/params come from the
    CLI. Lets ``--xaxis`` re-plot a whole sections file under a different axis.
    """
    return Section(
        name=sec.name,
        method=args.xaxis,
        start=sec.start,
        stop=sec.stop,
        params=_xaxis_params_from_args(args.xaxis, args, sec.name),
    )


def _parse_clim(clim_args) -> dict[str, tuple[float, float]]:
    """Parse repeated ``--clim VAR MIN MAX`` into ``{var: (min, max)}``."""
    out: dict[str, tuple[float, float]] = {}
    for var, lo, hi in clim_args or []:
        try:
            out[str(var)] = (float(lo), float(hi))
        except ValueError:
            raise SystemExit(
                f"--clim {var}: MIN/MAX must be numbers, got {lo!r} {hi!r}"
            ) from None
    return out


def _parse_waypoints(text: str) -> list[list[float]]:
    """Parse ``'lat,lon;lat,lon;...'`` into ``[[lat, lon], ...]``."""
    pts: list[list[float]] = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        a, b = chunk.split(",")
        pts.append([float(a), float(b)])
    if len(pts) < 2:
        raise SystemExit("--waypoints needs >= 2 'lat,lon' pairs separated by ';'")
    return pts


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _grouped(n: int) -> str:
    """Integer with the active locale's digit-grouping separator.

    Uses whatever locale :func:`run` installed from the environment (US
    ``1,234``, German ``1.234``, French ``1 234``, ...).  Falls back to the
    plain integer if the locale provides no grouping (e.g. the ``C`` locale).
    """
    try:
        return locale.format_string("%d", n, grouping=True)
    except (ValueError, locale.Error):
        return str(n)


def _default_variables(ds: xr.Dataset) -> list[str]:
    """Default panel set: the standard scalars present, plus optional sensors."""
    chosen = [v for v in _DEFAULT_VARS if v in ds.data_vars]
    chosen += [v for v in _OPTIONAL_VARS if v in ds.data_vars]
    return chosen


def _var_label(ds: xr.Dataset, name: str) -> str:
    """Colorbar label from the variable's own CF attrs, falling back to name."""
    attrs = ds[name].attrs if name in ds else {}
    long_name = attrs.get("long_name", name)
    units = attrs.get("units")
    return f"{long_name} [{units}]" if units else str(long_name)


def _time_subset(ds: xr.Dataset, sec: Section) -> xr.Dataset:
    """Index the trajectory to the section's [start, stop] UTC window."""
    t = ds["time"].values
    mask = np.ones(t.shape, dtype=bool)
    if sec.start is not None:
        mask &= t >= sec.start
    if sec.stop is not None:
        mask &= t <= sec.stop
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
    import matplotlib.pyplot as plt
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

    n = len(panel_vars)
    fig, axes = plt.subplots(
        n, 1, figsize=(11, 3.0 * n + 1.0), sharex=True, sharey=True,
        constrained_layout=True, squeeze=False,
    )
    axes = axes[:, 0]

    for ax, name in zip(axes, panel_vars):
        v = dss[name].values
        g, _count = grid.grid_mean(x, depth, v, x_edges, z_edges)
        if not np.any(np.isfinite(g)):
            ax.text(0.5, 0.5, f"no valid {name}", transform=ax.transAxes,
                    ha="center", va="center")
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
        cmap.set_bad(color="0.85")  # empty cells: light grey, visibly unsampled
        pcm = ax.pcolormesh(x_edges, z_edges, np.ma.masked_invalid(g),
                            cmap=cmap, norm=norm, shading="flat")
        cbar_fmt: str | None = None
        if name == "SP" and vmin is not None and vmax is not None:
            # Salinity sits in a narrow band; show 2 decimals, or 1 when the
            # range is wide enough to read -- avoids false precision on ticks.
            cbar_fmt = "%.1f" if (vmax - vmin) >= 1.0 else "%.2f"
        cbar = fig.colorbar(pcm, ax=ax, label=_var_label(ds, name), format=cbar_fmt)
        if name in _CBAR_MIN_AT_TOP:
            cbar.ax.invert_yaxis()  # min at top, max at bottom (mirrors depth)
        ax.set_ylabel("Depth (m)")

    axes[0].invert_yaxis()
    axes[0].set_xlim(x_edges[0], x_edges[-1])

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
    npts = int(dss.sizes["time"])
    fig.suptitle(
        f"{title_id}  —  section: {sec.name}  —  "
        f"x-axis: {sec.method}  —  {_grouped(npts)} samples"
    )
    return fig


def _safe_name(name: str) -> str:
    """Filesystem-safe stem for a section name."""
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)


# ---------------------------------------------------------------------------
# CLI plumbing (perturb-plot subcommand contract)
# ---------------------------------------------------------------------------


def add_arguments(p: argparse.ArgumentParser) -> None:
    """Register CLI flags for the scalar subcommand on *p*."""
    p.add_argument("--root", required=True,
                   help="perturb output root (contains ctd_combo_NN/)")
    p.add_argument("--ctd-combo", default=None,
                   help="explicit ctd combo dir or combo.nc (default: latest under --root)")
    p.add_argument("--sections", default=None,
                   help="sections YAML (data-chopping + x-axis). If omitted, "
                        "a single ad-hoc section is built from the flags below.")
    p.add_argument("--select", action="append", default=None, metavar="NAME",
                   help="plot only the named section(s) from --sections, by their "
                        "'name:' in the YAML (repeatable, or comma-separated). "
                        "Default: every section in the file.")
    p.add_argument("--out-dir", default=None,
                   help="write scalar_<name>.png here instead of showing on screen. "
                        "Default: display interactively; with no display available "
                        "(no tty / non-GUI backend) figures are written into --root.")
    # Rendering (apply to every section).
    p.add_argument("--var", dest="var", action="append", default=None,
                   help="scalar variable to panel (repeatable; default: "
                        "JAC_T, SP, sigma0 + DO/Chlorophyll/Turbidity if present)")
    p.add_argument("--z-bin", type=float, default=1.0, help="depth bin width [m]")
    p.add_argument("--x-bin", type=float, default=None,
                   help="x bin width in x-axis units (default: ~300 columns)")
    p.add_argument("--depth-max", type=float, default=None,
                   help="clip the depth axis at this value [m]")
    p.add_argument("--vmin", type=float, default=None,
                   help="colour-scale minimum. Applies only with a single --var; "
                        "for multiple variables use --clim per variable.")
    p.add_argument("--vmax", type=float, default=None,
                   help="colour-scale maximum (single --var only; see --clim)")
    p.add_argument("--clim", action="append", nargs=3, default=None,
                   metavar=("VAR", "MIN", "MAX"),
                   help="per-variable colour limits (repeatable), e.g. "
                        "--clim JAC_T 18 28 --clim SP 34.5 34.9. Wins over "
                        "--vmin/--vmax for that variable.")
    # Ad-hoc single-section flags (ignored when --sections is given, EXCEPT
    # --xaxis, which overrides every section's x-axis when set explicitly).
    p.add_argument("--name", default="section", help="ad-hoc section name")
    p.add_argument("--xaxis", choices=xaxis.METHODS, default=None,
                   help="x-axis method. Without --sections, builds one ad-hoc "
                        "section (default: time). With --sections, overrides "
                        "every section's x-axis when given.")
    p.add_argument("--start", default=None, help="ad-hoc window start (UTC ISO-8601)")
    p.add_argument("--stop", default=None, help="ad-hoc window stop (UTC ISO-8601)")
    p.add_argument("--point", nargs=2, type=float, default=None, metavar=("LAT", "LON"),
                   help="reference point for --xaxis distance_from_point")
    p.add_argument("--waypoints", default=None,
                   help="polyline for --xaxis along_line: 'lat,lon;lat,lon;...'")
    p.add_argument("--units", choices=("m", "km", "nm"), default="km",
                   help="distance units for spatial x-axes (default: km)")


def run(args: argparse.Namespace) -> str:
    """Render every section; show on screen, or write PNGs and return their dir."""
    # Honour the user's locale for number formatting in titles. Scoped to
    # LC_NUMERIC so we don't disturb matplotlib's float parsing (LC_CTYPE) or
    # collation; falls back silently when the environment locale is unset.
    with contextlib.suppress(locale.Error):
        locale.setlocale(locale.LC_NUMERIC, "")

    src = args.ctd_combo or layout.latest_stage_dir(args.root, "ctd_combo")
    if src is None:
        raise SystemExit(f"No ctd_combo dir under {args.root}")
    path = src if src.endswith(".nc") else os.path.join(src, "combo.nc")
    if not os.path.exists(path):
        raise SystemExit(f"CTD combo not found: {path}")

    # Show on screen unless the user asked for files or no display is available.
    display = args.out_dir is None and _can_display()
    out_dir = args.out_dir or args.root
    if not display:
        os.makedirs(out_dir, exist_ok=True)

    import matplotlib.pyplot as plt

    ds = xr.open_dataset(path)  # default CF decoding -> datetime64 time
    shown = 0
    try:
        if "time" not in ds.dims:
            raise SystemExit(f"{path}: expected a CTD trajectory with a 'time' dimension")
        if args.sections:
            sections = load_sections(args.sections)
            if args.select:
                sections = _select_sections(sections, args.select)
            if args.xaxis is not None:  # explicit --xaxis overrides every section
                sections = [_override_xaxis(s, args) for s in sections]
        else:
            if args.select:
                raise SystemExit("--select only applies together with --sections")
            sections = [_adhoc_section(args)]
        variables = list(args.var) if args.var else _default_variables(ds)
        clim = _parse_clim(args.clim)
        if (args.vmin is not None or args.vmax is not None) and len(variables) > 1:
            raise SystemExit(
                "--vmin/--vmax apply only with a single --var; use "
                "--clim VAR MIN MAX for per-variable limits"
            )
        for sec in sections:
            fig = _build_section_figure(ds, sec, variables, args, clim)
            if fig is None:
                continue
            if display:
                shown += 1
            else:
                out = os.path.join(out_dir, f"scalar_{_safe_name(sec.name)}.png")
                fig.savefig(out, dpi=150)
                plt.close(fig)
                print(f"Wrote {out}")
        if display and shown:
            plt.show()  # blocks until the user closes the window(s)
            plt.close("all")
    finally:
        ds.close()
    return f"displayed {shown} section(s)" if display else str(out_dir)
