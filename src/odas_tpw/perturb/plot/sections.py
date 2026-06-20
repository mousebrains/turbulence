# Jun-2026, Claude and Pat Welch, pat@mousebrains.com
"""Shared section/x-axis/CLI helpers for perturb-plot section-style subcommands.

A *section* chops a product and chooses an x-axis: a name, an optional UTC
``start``/``stop`` window, and an ``xaxis`` method with its parameters.  This
module owns the section spec, the ``sections.yaml`` loader, the ``--select`` /
``--xaxis`` override / ``--clim`` logic, and a few pure display/format helpers,
so both ``scalar`` (CTD trajectory) and ``profiles`` (binned (bin, profile)
grids) reuse one implementation.

It deliberately holds only *parsing + x-axis dispatch + UI* — NOT product
reading, gridding, or the per-product default variable sets, which differ by
subcommand.
"""

from __future__ import annotations

import argparse
import locale
import sys
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from odas_tpw.perturb.plot import xaxis

# matplotlib backends that cannot show a window — figures must be saved instead.
_NON_INTERACTIVE_BACKENDS: frozenset[str] = frozenset(
    {"agg", "pdf", "svg", "ps", "cairo", "template", "pgf"}
)


def can_display() -> bool:
    """True when figures can be shown interactively: a tty and a GUI backend."""
    if not sys.stdout.isatty():
        return False
    import matplotlib

    return matplotlib.get_backend().lower() not in _NON_INTERACTIVE_BACKENDS


def grouped(n: int) -> str:
    """Integer with the active locale's digit-grouping separator.

    Uses whatever locale the caller installed from the environment (US
    ``1,234``, German ``1.234``, French ``1 234``, ...).  Falls back to the
    plain integer if the locale provides no grouping (e.g. the ``C`` locale).
    """
    try:
        return locale.format_string("%d", n, grouping=True)
    except (ValueError, locale.Error):
        return str(n)


def safe_name(name: str) -> str:
    """Filesystem-safe stem for a section name."""
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)


def var_label(ds: xr.Dataset, name: str) -> str:
    """Colorbar label from the variable's own CF attrs, falling back to name.

    A dimensionless unit (``"1"`` — the CF/UDUNITS form for ratios such as
    practical salinity, the mixing coefficient, or turbidity) is NOT rendered:
    ``[1]`` is noise on a plot, and the meaning lives in the long_name /
    standard_name. Empty/absent units are likewise omitted (#38).
    """
    attrs = ds[name].attrs if name in ds else {}
    long_name = attrs.get("long_name", name)
    units = attrs.get("units")
    return f"{long_name} [{units}]" if units and units != "1" else str(long_name)


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


def parse_time(value) -> np.datetime64 | None:
    """Parse an ISO-8601 string to a UTC datetime64.

    A trailing ``Z`` is accepted (and stripped); a bare timestamp is treated as
    UTC, matching the products' encoding.  An explicit non-Z timezone offset is
    rejected rather than silently shifted.
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


def validate_params(method: str, params: dict, name: str) -> dict:
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
    params = validate_params(method, xa, name)
    return Section(
        name=name,
        method=method,
        start=parse_time(raw.get("start")),
        stop=parse_time(raw.get("stop")),
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
                                             #   distance_from_point|along_line|
                                             #   signed_distance
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


def select_sections(sections: list[Section], select: list[str]) -> list[Section]:
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


def xaxis_params_from_args(method: str, args: argparse.Namespace, name: str) -> dict:
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
        params["waypoints"] = parse_waypoints(args.waypoints)
    return validate_params(method, params, name)


def adhoc_section(args: argparse.Namespace) -> Section:
    """Build a single Section from ad-hoc CLI flags."""
    method = args.xaxis or "time"
    return Section(
        name=args.name,
        method=method,
        start=parse_time(args.start),
        stop=parse_time(args.stop),
        params=xaxis_params_from_args(method, args, args.name),
    )


def override_xaxis(sec: Section, args: argparse.Namespace) -> Section:
    """Return *sec* with its xaxis replaced by the CLI --xaxis (+ its params).

    Keeps the section's name and time window; the method/params come from the
    CLI. Lets ``--xaxis`` re-plot a whole sections file under a different axis.
    """
    return Section(
        name=sec.name,
        method=args.xaxis,
        start=sec.start,
        stop=sec.stop,
        params=xaxis_params_from_args(args.xaxis, args, sec.name),
    )


def parse_clim(clim_args) -> dict[str, tuple[float, float]]:
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


def parse_waypoints(text: str) -> list[list[float]]:
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


def resolve_sections(args: argparse.Namespace) -> list[Section]:
    """Build the section list from --sections (+ --select / --xaxis override) or ad-hoc.

    Shared by subcommands: with --sections, load the YAML, apply --select, and
    override every section's x-axis when an explicit --xaxis is given; without
    --sections, build a single ad-hoc section (and --select is an error).
    """
    if args.sections:
        sections = load_sections(args.sections)
        if args.select:
            sections = select_sections(sections, args.select)
        if args.xaxis is not None:  # explicit --xaxis overrides every section
            sections = [override_xaxis(s, args) for s in sections]
        return sections
    if args.select:
        raise SystemExit("--select only applies together with --sections")
    return [adhoc_section(args)]


def add_section_arguments(p: argparse.ArgumentParser) -> None:
    """Register the section / x-axis / colour / display flags shared by subcommands.

    Each subcommand adds its own product-reading and rendering flags (e.g.
    --ctd-combo / --z-bin for scalar, --product for profiles).
    """
    p.add_argument("--root", required=True,
                   help="perturb output root (contains the combo_NN/ products)")
    p.add_argument("--sections", default=None,
                   help="sections YAML (data-chopping + x-axis). If omitted, "
                        "a single ad-hoc section is built from the flags below.")
    p.add_argument("--select", action="append", default=None, metavar="NAME",
                   help="plot only the named section(s) from --sections, by their "
                        "'name:' in the YAML (repeatable, or comma-separated). "
                        "Default: every section in the file.")
    p.add_argument("--out-dir", default=None,
                   help="write <name>.png here instead of showing on screen. "
                        "Default: display interactively; with no display available "
                        "(no tty / non-GUI backend) figures are written into --root.")
    p.add_argument("--var", dest="var", action="append", default=None,
                   help="variable to panel (repeatable; default: the product's "
                        "standard set)")
    p.add_argument("--vmin", type=float, default=None,
                   help="colour-scale minimum. Applies only with a single --var; "
                        "for multiple variables use --clim per variable.")
    p.add_argument("--vmax", type=float, default=None,
                   help="colour-scale maximum (single --var only; see --clim)")
    p.add_argument("--clim", action="append", nargs=3, default=None,
                   metavar=("VAR", "MIN", "MAX"),
                   help="per-variable colour limits (repeatable), e.g. "
                        "--clim epsilonMean 1e-10 1e-7. Wins over --vmin/--vmax.")
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


def single_var_limit_guard(args: argparse.Namespace, variables: list[str]) -> None:
    """Reject global --vmin/--vmax when more than one variable is plotted."""
    if (args.vmin is not None or args.vmax is not None) and len(variables) > 1:
        raise SystemExit(
            "--vmin/--vmax apply only with a single --var; use "
            "--clim VAR MIN MAX for per-variable limits"
        )
