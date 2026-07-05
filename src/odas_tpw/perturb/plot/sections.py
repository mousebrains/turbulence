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
import contextlib
import locale
import os
import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

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


# CF temperature unit strings, shown as the compact "°C" symbol.
_CELSIUS_UNITS: frozenset[str] = frozenset(
    {"degree_Celsius", "degrees_Celsius", "degC", "celsius", "Celsius"}
)


def var_label(ds: xr.Dataset, name: str) -> str:
    """Colorbar label from the variable's own CF attrs, falling back to name.

    Units are rendered in curved brackets — ``long_name (unit)`` — consistently
    across every plot, matching the curated ``_CBAR_LABEL`` overrides.  A
    dimensionless unit (``"1"`` — the CF/UDUNITS form for ratios such as
    practical salinity, the mixing coefficient, or turbidity) is NOT rendered:
    ``(1)`` is noise on a plot, and the meaning lives in the long_name /
    standard_name. Empty/absent units are likewise omitted (#38).  Every
    ``degree_Celsius`` field shows the compact ``°C`` symbol.
    """
    attrs = ds[name].attrs if name in ds else {}
    long_name = attrs.get("long_name", name)
    units = attrs.get("units")
    if units in _CELSIUS_UNITS:
        units = "°C"
    return f"{long_name} ({units})" if units and units != "1" else str(long_name)


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
    # Handle an explicit timezone offset on the time-of-day part. The date/time
    # separator may be 'T' OR a space: YAML resolves an unquoted '...Z'
    # timestamp to a tz-AWARE datetime whose str() renders the zone as
    # '+00:00' with a SPACE separator, so we must inspect the post-separator
    # tail. A ZERO offset (Z / +00:00 / -00:00) is UTC and is accepted (and
    # stripped, since numpy datetime64 rejects any offset); a NON-zero offset
    # (e.g. +09:00 / -05:00) is rejected rather than silently shifted.
    sep = "T" if "T" in s else (" " if " " in s else "")
    tail = s.split(sep, 1)[-1] if sep else ""
    off = re.search(r"([+-])(\d{2}):?(\d{2})?$", tail)
    if off:
        if int(off.group(2)) or int(off.group(3) or 0):
            raise ValueError(
                f"time {value!r} must be UTC: use a trailing 'Z' or no offset, "
                "not an explicit timezone"
            )
        s = s[: s.rindex(off.group(0))]
    if sep == " ":  # numpy datetime64 wants the ISO 'T' separator
        s = s.replace(" ", "T", 1)
    try:
        return np.datetime64(s)
    except ValueError as exc:
        raise ValueError(f"could not parse time {value!r}: {exc}") from None


def validate_params(method: str, params: dict, name: str) -> dict:
    """Check method-specific params and normalize units; return a clean dict."""
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
    """Register the section / x-axis / color / display flags shared by subcommands.

    Each subcommand adds its own product-reading and rendering flags (e.g.
    --ctd-combo / --z-bin for scalar, --product for profiles).
    """
    p.add_argument("--root", default=None,
                   help="perturb output root (contains the combo_NN/ products). "
                        "Required unless --config is given.")
    from odas_tpw.perturb import resolve
    resolve.add_resolve_args(p)
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
                   help="color-scale minimum. Applies only with a single --var; "
                        "for multiple variables use --clim per variable.")
    p.add_argument("--vmax", type=float, default=None,
                   help="color-scale maximum (single --var only; see --clim)")
    p.add_argument("--clim", action="append", nargs=3, default=None,
                   metavar=("VAR", "MIN", "MAX"),
                   help="per-variable color limits (repeatable), e.g. "
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
    add_output_arguments(p)


def positive_int(s: str) -> int:
    """argparse ``type`` for a strictly-positive integer (e.g. ``--dpi``).

    Raises ``ValueError`` (not ``ArgumentTypeError``) on a non-positive or
    non-integer value so the same check fires both at the CLI *and* through the
    figure driver's ``_coerce`` (which calls the action's ``type`` and catches
    ``ValueError``) — a 0/negative/float dpi fails up front, not at render time.
    """
    v = int(s)
    if v <= 0:
        raise ValueError(f"must be a positive integer, got {v}")
    return v


def add_output_arguments(p: argparse.ArgumentParser, *, title: bool = True) -> None:
    """Register the shared figure-output flags (figsize / dpi [/ title]).

    Honored by every subcommand's renderer and by the ``figure`` batch driver
    (per-figure ``figsize``/``dpi``/``title``). ``title=False`` for a subcommand
    that already registers its own ``--title`` (eps-chi).
    """
    p.add_argument("--figsize", nargs=2, type=float, default=None, metavar=("W", "H"),
                   help="figure size in inches, e.g. --figsize 11 9 "
                        "(default: preset-specific).")
    p.add_argument("--ncols", type=positive_int, default=None,
                   help="arrange the variable panels in this many columns "
                        "(default: preset-specific — profiles is a 3-column grid, "
                        "others a single vertical stack); e.g. 2 with four "
                        "variables gives a 2x2 grid. Sections stay one image each.")
    p.add_argument("--dpi", type=positive_int, default=None,
                   help="raster resolution for saved PNG/PDF (default: 150).")
    if title:
        p.add_argument("--title", default=None,
                       help="figure title (default: derived from the data).")


def fig_dpi(args: argparse.Namespace) -> int:
    """Raster resolution for a saved figure: ``--dpi`` or the 150 default."""
    return getattr(args, "dpi", None) or 150


@contextlib.contextmanager
def close_new_figs_on_error() -> Iterator[None]:
    """Close any pyplot figure created inside the block if it raises.

    A figure built by ``plt.subplots`` registers in pyplot's global manager
    immediately, so an exception while populating it (a gridding/limits error)
    before it is returned/yielded would leave it open forever — the caller's
    cleanup can't reach a figure it never received. Wrap each figure build in
    this so a build-time error closes the orphan(s) and re-raises; figures that
    existed before the block (already handed to the caller) are left untouched.
    """
    import matplotlib.pyplot as plt

    before = set(plt.get_fignums())
    try:
        yield
    except BaseException:
        for num in set(plt.get_fignums()) - before:
            plt.close(num)
        raise


@contextlib.contextmanager
def closing_figs(
    figs: Iterable[tuple[str, Any]],
) -> Iterator[Iterable[tuple[str, Any]]]:
    """Yield *figs*, then close it on exit if it is a generator.

    A ``build_figures`` generator holds an open dataset until its ``finally``
    runs (on exhaustion or close). Wrapping consumption in this guarantees that
    handle is released deterministically even if a consumer stops early (a save
    error mid-stream), rather than waiting on GC. A plain list/iterator with no
    ``close`` is left untouched.
    """
    try:
        yield figs
    finally:
        close = getattr(figs, "close", None)
        if callable(close):
            close()


def save_or_show(figs: Iterable[tuple[str, Any]], out_dir: str | None,
                 dpi: int) -> int:
    """Consume a ``(stem, Figure)`` iterable and return how many it handled.

    With *out_dir* set, write ``<out_dir>/<stem>.png`` **streaming** — one open
    figure at a time, closing each before the next is pulled (bounded memory,
    and a save error can't leak the not-yet-built figures). With *out_dir* None,
    collect every figure (they must all be open together) and ``show()`` them.
    Always closes the figures it created, even on error.
    """
    import matplotlib.pyplot as plt

    if out_dir is None:  # interactive: all figures open together for show()
        collected = []
        try:
            for _stem, fig in figs:
                collected.append(fig)
        except BaseException:
            for fig in collected:
                plt.close(fig)
            raise
        if collected:
            plt.show()  # blocks until the user closes the window(s)
            plt.close("all")
        return len(collected)

    count = 0
    for stem, fig in figs:  # save path: at most one figure open at a time
        try:
            out = os.path.join(out_dir, f"{stem}.png")
            fig.savefig(out, dpi=dpi)
            print(f"Wrote {out}")
            count += 1
        finally:
            plt.close(fig)
    return count


def single_var_limit_guard(args: argparse.Namespace, variables: list[str]) -> None:
    """Reject global --vmin/--vmax when more than one variable is plotted."""
    if (args.vmin is not None or args.vmax is not None) and len(variables) > 1:
        raise SystemExit(
            "--vmin/--vmax apply only with a single --var; use "
            "--clim VAR MIN MAX for per-variable limits"
        )
