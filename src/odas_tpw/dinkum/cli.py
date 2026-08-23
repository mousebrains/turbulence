# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""``dinkum-hotel`` — build perturb hotel files from Slocum Dinkum files.

Subcommands:

``init``
    Write a commented template YAML.
``sensors``
    List what sensors a set of Dinkum files actually carries, with finite
    counts and ranges. Run this first: it is how you find out whether the
    CTD reported, and which time sensors are populated.
``build``
    Read the files named by a config and write the hotel NetCDF.
``backends``
    Report which readers are available and why.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _add_logging_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--verbose", "-v", action="store_true", help="debug logging")
    parser.add_argument("--quiet", "-q", action="store_true", help="warnings and errors only")


def _install_logging(args: argparse.Namespace) -> None:
    level = logging.INFO
    if getattr(args, "verbose", False):
        level = logging.DEBUG
    elif getattr(args, "quiet", False):
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")


def _cmd_init(args: argparse.Namespace) -> int:
    from odas_tpw.dinkum.config import generate_template

    out = Path(args.output)
    if out.exists() and not args.force:
        print(f"{out} exists; use --force to overwrite", file=sys.stderr)
        return 1
    generate_template(out)
    print(f"Wrote template config to {out}")
    return 0


def _cmd_backends(args: argparse.Namespace) -> int:
    from odas_tpw.dinkum.reader import available_backends

    for name, reason in available_backends().items():
        print(f"{name:14s} {'available' if not reason else 'UNAVAILABLE - ' + reason}")
    return 0


def _cmd_sensors(args: argparse.Namespace) -> int:
    from odas_tpw.dinkum.reader import load_dinkum, sensor_inventory

    paths = [Path(p) for p in args.files]
    ds = load_dinkum(
        paths,
        backend=args.reader,
        cache=args.cache,
        skip_first_record=not args.keep_first_record,
    )
    rows = sensor_inventory(ds)
    if args.match:
        needles = [m.lower() for m in args.match]
        rows = [r for r in rows if any(n in r["name"].lower() for n in needles)]
    if not args.all:
        rows = [r for r in rows if r["n_finite"] > 0]
    if not rows:
        print("No sensors matched.", file=sys.stderr)
        return 1

    width = max(len(r["name"]) for r in rows)
    print(f"{'sensor'.ljust(width)}  {'units':>12}  {'n_finite':>9}  {'frac':>6}  range")
    for r in rows:
        rng = f"{r['min']:.6g} .. {r['max']:.6g}" if r["n_finite"] else "(never reported)"
        print(
            f"{r['name'].ljust(width)}  {r['units'][:12]:>12}  "
            f"{r['n_finite']:>9d}  {r['fraction']:>6.1%}  {rng}"
        )
    print(
        f"\n{len(rows)} sensor(s), {ds.sizes.get('record', 0)} records, "
        f"backend={ds.attrs.get('dinkum_reader', '?')}"
    )
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    from odas_tpw.dinkum.build import build_hotel
    from odas_tpw.dinkum.config import load_config

    cfg_path = Path(args.config).resolve()
    config = load_config(cfg_path)
    out = build_hotel(
        config,
        config_dir=cfg_path.parent,
        output=args.output,
        now=args.now,
    )
    print(f"Wrote hotel file: {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dinkum-hotel",
        description="Build perturb hotel files from Slocum Dinkum Binary Data files",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="write a template config file")
    p_init.add_argument("-o", "--output", default="dinkum-hotel.yaml", help="output path")
    p_init.add_argument("-f", "--force", action="store_true", help="overwrite an existing file")
    _add_logging_args(p_init)
    p_init.set_defaults(func=_cmd_init)

    p_back = sub.add_parser("backends", help="report available DBD readers")
    _add_logging_args(p_back)
    p_back.set_defaults(func=_cmd_backends)

    p_sens = sub.add_parser("sensors", help="list sensors carried by Dinkum files")
    p_sens.add_argument("files", nargs="+", help="Dinkum (or NetCDF) files")
    p_sens.add_argument("-C", "--cache", default=None, help="sensor-list cache directory")
    p_sens.add_argument(
        "-r",
        "--reader",
        default="auto",
        choices=["auto", "xarray-dbd", "dbd2netcdf", "netcdf"],
        help="reader backend",
    )
    p_sens.add_argument(
        "-m",
        "--match",
        nargs="+",
        default=None,
        help="only sensors whose name contains one of these substrings",
    )
    p_sens.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="include sensors that never reported (all-NaN)",
    )
    p_sens.add_argument(
        "--keep-first-record",
        action="store_true",
        help="do not skip each file's first record",
    )
    _add_logging_args(p_sens)
    p_sens.set_defaults(func=_cmd_sensors)

    p_build = sub.add_parser("build", help="build the hotel file from a config")
    p_build.add_argument("-c", "--config", required=True, help="YAML config file")
    p_build.add_argument("-o", "--output", default=None, help="override files.output")
    p_build.add_argument(
        "--now",
        type=float,
        default=None,
        help="epoch seconds used for the 'now + 365 days' fallback time ceiling; "
        "pass it to make a run reproducible",
    )
    _add_logging_args(p_build)
    p_build.set_defaults(func=_cmd_build)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _install_logging(args)
    try:
        rc = args.func(args)
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
        # These carry actionable messages (missing sensor, empty decode, bad
        # bound); a traceback would bury the one line that matters.
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    raise SystemExit(rc)


if __name__ == "__main__":  # pragma: no cover
    main()
