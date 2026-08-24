# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""``erddap-hotel`` — build perturb hotel files from an ERDDAP tabledap dataset.

The ERDDAP twin of ``dinkum-hotel``: same output artifact, different source.
perturb and ``fp07-cal`` never talk to the server -- they open the local file
this writes.

Subcommands:

``init``
    Write a commented template YAML.
``info``
    Probe ``.das``/``.dds``: variables, units, valid ranges, coverage. Run this
    before ``build``, the way ``fp07-cal coverage`` comes before ``fit`` --
    look at what you actually have before computing on it.
``fetch``
    Populate the cache only. ``--dry-run`` prints the URLs and fetches nothing.
``build``
    Fetch (or reuse the cache), sanitise, and write the hotel NetCDF.
``verify``
    Re-fetch just the ``.das`` and report whether the dataset has changed since
    the hotel file was built. No data transfer.
"""

from __future__ import annotations

import argparse
import logging
import re
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


def _load(args: argparse.Namespace) -> tuple[dict, Path]:
    from odas_tpw.erddap.config import load_config

    path = Path(args.config)
    if not path.exists():
        raise SystemExit(f"config not found: {path}\nRun `erddap-hotel init` to write a template.")
    return load_config(path), path.parent


def _cmd_init(args: argparse.Namespace) -> int:
    from odas_tpw.erddap.config import generate_template

    out = Path(args.output)
    if out.exists() and not args.force:
        print(f"{out} exists; use --force to overwrite", file=sys.stderr)
        return 1
    generate_template(out)
    print(f"Wrote template config to {out}")
    return 0


def _das_section(das: str, name: str) -> dict[str, str]:
    m = re.search(rf"\n\s+{re.escape(name)} \{{(.*?)\n\s+\}}", das, re.S)
    if not m:
        return {}
    return dict(re.findall(r"^\s+\S+\s+(\S+)\s+(.*?);\s*$", m.group(1), re.M))


def _cmd_info(args: argparse.Namespace) -> int:
    from odas_tpw.erddap.config import merge_config
    from odas_tpw.erddap.fetch import das_fingerprint, fetch_bytes, probe_das
    from odas_tpw.erddap.query import dds_url

    config, _ = _load(args)
    server = merge_config("server", config.get("server"))
    fetch_cfg = merge_config("fetch", config.get("fetch"))
    timeout = float(server["timeout_s"])

    das = probe_das(server["base_url"], server["dataset_id"], timeout=timeout)
    sha, modified = das_fingerprint(das)
    print(f"dataset       {server['dataset_id']}")
    print(f"server        {server['base_url']}")
    print(f"date_modified {modified or '(not declared)'}")
    print(f"das sha256    {sha}")

    glob = _das_section(das, "NC_GLOBAL")
    for key in ("processing_level", "time_coverage_start", "time_coverage_end", "institution"):
        if key in glob:
            print(f"{key:13s} {glob[key].strip(chr(34))}")

    dds = fetch_bytes(dds_url(server["base_url"], server["dataset_id"]), timeout=timeout)
    present = set(re.findall(r"^\s+\w+\s+(\w+);", dds.decode("utf-8", "replace"), re.M))
    print(f"\nvariables served: {len(present)}")

    print("\nrequested:")
    missing = []
    for name in fetch_cfg.get("variables") or []:
        attrs = _das_section(das, name)
        if name not in present:
            missing.append(name)
            print(f"  {name:26s} *** NOT IN THIS DATASET ***")
            continue
        units = attrs.get("units", "").strip('"') or "-"
        rng = ""
        if "valid_min" in attrs and "valid_max" in attrs:
            rng = f"  valid [{attrs['valid_min']}, {attrs['valid_max']}]"
        fill = f"  _FillValue={attrs['_FillValue']}" if "_FillValue" in attrs else ""
        print(f"  {name:26s} units={units:14s}{rng}{fill}")
        long_name = attrs.get("long_name", "").strip('"')
        if long_name and long_name != name:
            print(f"  {'':26s} long_name={long_name!r}")

    if missing:
        print(
            f"\n{len(missing)} requested variable(s) are absent. ERDDAP 400s the whole "
            "request when any one is, so fix fetch.variables before building.",
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    from odas_tpw.erddap.build import fetch_chunks, plan_requests
    from odas_tpw.erddap.config import validate

    config, config_dir = _load(args)
    validate(config)
    if args.dry_run:
        plan = plan_requests(config)
        print(f"{len(plan)} request(s):")
        for req in plan:
            print(f"\n# {req['start']} .. {req['end']}\n{req['url']}")
        return 0
    paths, meta = fetch_chunks(config, config_dir=config_dir, offline=args.offline)
    print(
        f"{len(paths)} chunk(s) ready "
        f"({meta['chunks_fetched']} fetched, {meta['chunks_cached']} cached, "
        f"{meta['chunks_empty']} empty)"
    )
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    from odas_tpw.erddap.build import build

    config, config_dir = _load(args)
    out = build(config, config_dir=config_dir, output=args.output, offline=args.offline)
    print(f"Wrote {out}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from odas_tpw.erddap.build import verify

    config, config_dir = _load(args)
    hotel = Path(args.hotel_file) if args.hotel_file else None
    if hotel is None:
        from odas_tpw.erddap.config import merge_config
        from odas_tpw.perturb.config import expand_config_dir

        out_cfg = merge_config("output", config.get("output"))
        hotel = Path(expand_config_dir(str(out_cfg["file"]), str(config_dir)))
    result = verify(config, hotel)
    for key in (
        "hotel_file",
        "built_date_modified",
        "live_date_modified",
        "built_das_sha256",
        "live_das_sha256",
    ):
        if key in result:
            print(f"{key:22s} {result[key]}")
    if not result["exists"]:
        print("\nHotel file does not exist yet; nothing to compare against.")
        return 1
    if result["changed"]:
        print(
            "\nCHANGED: the dataset has been revised since this hotel file was built.\n"
            "Rebuild with `fetch.refresh: always` to pick up the revision."
        )
        return 2
    if result["built_das_sha256"] is None:
        print("\nThis hotel file records no .das checksum; it was not built by erddap-hotel.")
        return 1
    print("\nUnchanged: the dataset still matches what this hotel file was built from.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="erddap-hotel",
        description=(
            "Build a perturb hotel file from an ERDDAP tabledap dataset. The "
            "ERDDAP twin of dinkum-hotel: same artifact, different source. "
            "perturb never sees a URL."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="write a template config file")
    p_init.add_argument("-o", "--output", default="erddap-hotel.yaml")
    p_init.add_argument("--force", action="store_true", help="overwrite an existing file")
    p_init.set_defaults(func=_cmd_init)

    p_info = sub.add_parser("info", help="probe the dataset: variables, units, coverage")
    p_info.add_argument("-c", "--config", required=True)
    p_info.set_defaults(func=_cmd_info)

    p_fetch = sub.add_parser("fetch", help="populate the cache only")
    p_fetch.add_argument("-c", "--config", required=True)
    p_fetch.add_argument(
        "--dry-run", action="store_true", help="print the URLs that would be fetched"
    )
    p_fetch.add_argument(
        "--offline", action="store_true", help="use the cache only; never touch the network"
    )
    p_fetch.set_defaults(func=_cmd_fetch)

    p_build = sub.add_parser("build", help="fetch (cached) + QC -> hotel.nc")
    p_build.add_argument("-c", "--config", required=True)
    p_build.add_argument("-o", "--output", default=None, help="override output.file")
    p_build.add_argument(
        "--offline", action="store_true", help="use the cache only; never touch the network"
    )
    p_build.set_defaults(func=_cmd_build)

    p_verify = sub.add_parser("verify", help="has the dataset changed since the build?")
    p_verify.add_argument("-c", "--config", required=True)
    p_verify.add_argument("--hotel-file", default=None)
    p_verify.set_defaults(func=_cmd_verify)

    for p in (p_init, p_info, p_fetch, p_build, p_verify):
        _add_logging_args(p)

    args = parser.parse_args(argv)
    _install_logging(args)
    try:
        return int(args.func(args))
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        from odas_tpw.erddap.fetch import ErddapError

        if isinstance(exc, ErddapError):
            print(f"error: {exc}", file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
