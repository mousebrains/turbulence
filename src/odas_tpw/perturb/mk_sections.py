# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""Auto-generate a plotting ``sections.yaml`` from a perturb run's profiles.

A *section* (see :mod:`odas_tpw.perturb.plot.sections`) groups a contiguous
batch of casts and assigns an x-axis for depth-vs-x plots.  This builds one
automatically by splitting the run's detected profiles wherever the time gap
between consecutive casts exceeds a threshold — the common "casts arrive in
station batches separated by transits" pattern.

The split criterion lives in :func:`split_indices`; a future heading-change
criterion (a new transect leg in lat/lon space) plugs in there without touching
the reader or the writer.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# numpy is imported inside the functions that use it (not at module top) so that
# merely building the CLI parser — cli.build_parser() imports this module to
# register the `sections` subcommand's args on every `perturb` invocation — does
# not pull numpy for subcommands that never touch it. Matches the lazy-import
# discipline in cli.py and the deferred xarray/ruamel imports below.

_DUR_UNITS = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


def parse_duration(text: str) -> float:
    """Parse ``'90m'`` / ``'1.5h'`` / ``'3600s'`` / ``'2d'`` / bare seconds.

    Returns seconds as a float.  A bare number is seconds.
    """
    s = str(text).strip().lower()
    if not s:
        raise ValueError("empty duration")
    unit = _DUR_UNITS.get(s[-1])
    if unit is not None:
        s = s[:-1]
    else:
        unit = 1.0
    try:
        value = float(s)
    except ValueError:
        raise ValueError(
            f"could not parse duration {text!r} (use e.g. 90m, 1.5h, 3600s, 2d)"
        ) from None
    if not (value > 0 and math.isfinite(value)):
        raise ValueError(f"duration must be a positive finite number, got {text!r}")
    return value * unit


@dataclass
class ProfileTimes:
    """Per-profile start times (epoch seconds) and positions, sorted by time."""

    stime: np.ndarray
    lat: np.ndarray
    lon: np.ndarray


def read_profile_times(config: dict, product: str = "combo") -> ProfileTimes:
    """Read per-profile ``stime`` / ``lat`` / ``lon`` from a run's combo product.

    Resolves the ``{product}_NN`` directory for *config* (by signature, via
    :mod:`odas_tpw.perturb.resolve`) and reads ``combo.nc``.  Profiles with a
    non-finite start time are dropped; the result is sorted by time.
    """
    import numpy as np
    import xarray as xr

    from odas_tpw.perturb import resolve

    try:
        stage = resolve.stage_dir(config, product)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"no {product}_NN output found — run the pipeline first, or pass "
            f"--product (e.g. diss_combo). ({exc})"
        ) from exc
    except resolve.StageConflict as exc:
        raise SystemExit(str(exc)) from exc
    except ValueError as exc:  # unknown stage/product name (not in resolve.STAGES)
        raise SystemExit(
            f"{exc} (--product must name a combo product, e.g. combo, diss_combo, "
            "chi_combo)"
        ) from exc
    path = Path(stage) / "combo.nc"
    if not path.exists():
        raise SystemExit(
            f"{path} not found — the {product} stage directory exists but has no "
            "combo.nc (an interrupted run?). Re-run the pipeline's combo stage."
        )
    with xr.open_dataset(path, decode_times=False) as ds:
        if "stime" not in ds.variables:
            raise SystemExit(
                f"{path}: no per-profile 'stime' variable — {product} is not a "
                "(bin, profile) product; try --product combo or diss_combo."
            )
        stime = np.asarray(ds["stime"].values, dtype=float)
        n = stime.size
        lat = (
            np.asarray(ds["lat"].values, dtype=float)
            if "lat" in ds.variables else np.full(n, np.nan)
        )
        lon = (
            np.asarray(ds["lon"].values, dtype=float)
            if "lon" in ds.variables else np.full(n, np.nan)
        )
    finite = np.isfinite(stime)
    stime, lat, lon = stime[finite], lat[finite], lon[finite]
    order = np.argsort(stime, kind="stable")
    return ProfileTimes(stime[order], lat[order], lon[order])


def split_indices(stime: np.ndarray, gap_seconds: float) -> list[np.ndarray]:
    """Split time-sorted profiles into groups at gaps exceeding *gap_seconds*.

    Returns a list of index arrays (into *stime*), one per section, in time
    order.  This is the extension seam: an additional criterion (e.g. a change
    of heading in lat/lon space) would contribute more break points here.
    """
    import numpy as np

    stime = np.asarray(stime, dtype=float)
    if stime.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(stime) > gap_seconds) + 1
    return np.split(np.arange(stime.size), breaks)


def _iso(epoch: float) -> str:
    """UTC ``YYYY-MM-DDTHH:MM:SSZ`` for an integer-second instant.

    *epoch* is expected already floored/ceiled to whole seconds by the caller;
    a fractional value is truncated toward zero.
    """
    import numpy as np

    return np.datetime_as_string(np.datetime64(int(epoch), "s"), unit="s") + "Z"


def build_sections(
    pt: ProfileTimes,
    groups: list[np.ndarray],
    method: str,
    units: str,
    pad: float,
) -> list[dict]:
    """One section dict per group: name, padded UTC window, cast count, xaxis."""
    xaxis: dict = {"method": method}
    if method in ("distance_from_point", "along_line", "signed_distance"):
        xaxis["units"] = units
    sections = []
    for i, idx in enumerate(groups):
        # Floor the start and ceil the stop so the integer-second window always
        # brackets the batch's first/last cast, even at --pad 0 (the plot side
        # filters inclusively: stime >= start & stime <= stop).
        sections.append({
            "name": f"section_{i:02d}",
            "n_casts": int(idx.size),
            "start": _iso(math.floor(float(pt.stime[idx[0]]) - pad)),
            "stop": _iso(math.ceil(float(pt.stime[idx[-1]]) + pad)),
            "xaxis": dict(xaxis),
        })
    return sections


def render_yaml(sections: list[dict], header: str) -> str:
    """Render section dicts to a sections.yaml matching the hand-written style."""
    lines = [header, "", "sections:"]
    for s in sections:
        lines.append(f"  # {s['n_casts']} cast(s)")
        lines.append(f"  - name: {s['name']}")
        lines.append(f'    start: "{s["start"]}"')
        lines.append(f'    stop:  "{s["stop"]}"')
        xa = s["xaxis"]
        if set(xa) == {"method"}:
            lines.append(f"    xaxis: {{method: {xa['method']}}}")
        else:
            lines.append("    xaxis:")
            lines.append(f"      method: {xa['method']}")
            for key, val in xa.items():
                if key != "method":
                    lines.append(f"      {key}: {val}")
    return "\n".join(lines) + "\n"


def _validate(text: str) -> None:
    """Confirm the emitted YAML round-trips through the plot-side loader."""
    from ruamel.yaml import YAML

    from odas_tpw.perturb.plot import sections as plot_sections

    raw = YAML().load(text) or {}
    for i, sec in enumerate(raw.get("sections", [])):
        plot_sections._section_from_dict(dict(sec), i)  # raises on any invalid field


def _report(pt: ProfileTimes, gap: float, sections: list[dict]) -> None:
    """Print a human summary + the largest gaps (so --gap is easy to tune)."""
    import numpy as np

    span_h = (float(pt.stime[-1]) - float(pt.stime[0])) / 3600.0 if pt.stime.size else 0.0
    print(
        f"{pt.stime.size} profiles over {span_h:.1f} h -> {len(sections)} sections "
        f"(gap > {gap / 3600:.2f} h)",
        file=sys.stderr,
    )
    if pt.stime.size > 1:
        top = np.sort(np.diff(pt.stime))[::-1][:8]
        pretty = ", ".join(f"{g / 3600:.2f}h" for g in top)
        print(f"  largest inter-cast gaps: {pretty}", file=sys.stderr)
        print("  (re-run with --gap to change the batching)", file=sys.stderr)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the ``perturb sections`` flags on *parser*."""
    parser.add_argument(
        "-c", "--config", metavar="YAML", required=True,
        help="perturb config (locates the run's combo output)",
    )
    parser.add_argument(
        "-o", "--output", metavar="FILE",
        help="write the sections YAML here (default: stdout)",
    )
    parser.add_argument(
        "--gap", default="1h", metavar="DUR",
        help="start a new section when the gap between consecutive casts exceeds "
             "this (e.g. 90m, 1.5h, 3600s, 2d; default 1h)",
    )
    parser.add_argument(
        "--xaxis", default="time",
        choices=["time", "latitude", "longitude", "signed_distance"],
        help="x-axis method for every section (default time). distance_from_point "
             "and along_line are omitted: they need a point/waypoints this tool "
             "cannot infer — add them by hand afterward.",
    )
    parser.add_argument(
        "--units", default="km", choices=["m", "km", "nm"],
        help="distance units for a spatial --xaxis (default km)",
    )
    parser.add_argument(
        "--pad", type=float, default=30.0, metavar="SEC",
        help="pad each section's time window by this many seconds (default 30)",
    )
    parser.add_argument(
        "--product", default="combo", metavar="P",
        help="combo product to read per-profile times from (default combo)",
    )
    parser.add_argument(
        "-f", "--force", action="store_true",
        help="overwrite --output if it already exists",
    )


def run(args: argparse.Namespace) -> None:
    """Build a sections.yaml from a run's profiles and write it (or print it)."""
    from odas_tpw.perturb.config import load_config

    gap = parse_duration(args.gap)
    if not (math.isfinite(args.pad) and args.pad >= 0):
        raise SystemExit(
            f"--pad must be a non-negative finite number of seconds, got {args.pad}"
        )

    # Validate the output destination before the (expensive) combo read, so a
    # bad -o path fails fast instead of after loading and processing the run.
    out: Path | None = None
    if args.output:
        out = Path(args.output)
        if out.exists() and not args.force:
            raise SystemExit(f"{out} exists; pass --force to overwrite")
        if not out.parent.exists():
            raise SystemExit(
                f"{out.parent} does not exist — create it or choose another "
                "--output path"
            )

    config = load_config(args.config)
    pt = read_profile_times(config, args.product)
    if pt.stime.size == 0:
        raise SystemExit("no profiles with a finite start time were found")

    groups = split_indices(pt.stime, gap)
    sections = build_sections(pt, groups, args.xaxis, args.units, args.pad)
    span_h = (float(pt.stime[-1]) - float(pt.stime[0])) / 3600.0
    header = (
        f"# Auto-generated by `perturb sections` from {args.config}\n"
        f"# {pt.stime.size} profiles over {span_h:.1f} h, split on gaps > "
        f"{args.gap} -> {len(sections)} sections.\n"
        f"# Edit freely: adjust windows, rename, or change each section's xaxis.\n"
        f"# Re-run with --gap (e.g. --gap 2h) to change the batch granularity."
    )
    text = render_yaml(sections, header)
    _validate(text)  # never emit something the plot loader would reject
    _report(pt, gap, sections)

    if out is not None:
        out.write_text(text, encoding="utf-8")
        print(f"wrote {len(sections)} sections -> {out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
