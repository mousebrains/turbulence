# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""
CLI for rsi-python.

Main command:
    rsi-tpw <subcommand> [options]

Subcommands:
    info     — Print summary of .p file(s)
    nc       — Convert .p files to NetCDF
    prof     — Extract profiles from .p or full-record .nc files
    init     — Generate a template configuration file
"""

import argparse
import sys
from pathlib import Path
from typing import Any


def _resolve_p_files(patterns: list[str]) -> list[Path]:
    """Expand glob patterns and return list of .p file Paths."""
    import glob as globmod

    files = []
    for pattern in patterns:
        expanded = sorted(globmod.glob(pattern))
        if not expanded:
            print(f"Warning: no files match '{pattern}'", file=sys.stderr)
        for f in expanded:
            p = Path(f)
            if p.is_file() and p.suffix.lower() == ".p":
                files.append(p)
    if not files:
        print("Error: no .p files found", file=sys.stderr)
        sys.exit(1)
    return files


def _resolve_files(patterns: list[str], extensions: set[str] | None = None) -> list[Path]:
    """Expand glob patterns and return list of matching file Paths.

    Parameters
    ----------
    patterns : list of str
        File paths or glob patterns.
    extensions : set of str or None
        Allowed file extensions (e.g. {'.p', '.nc'}). If None, accept all.
    """
    import glob as globmod

    files = []
    for pattern in patterns:
        expanded = sorted(globmod.glob(pattern))
        if not expanded:
            print(f"Warning: no files match '{pattern}'", file=sys.stderr)
        for f in expanded:
            p = Path(f)
            if p.is_file() and (extensions is None or p.suffix.lower() in extensions):
                files.append(p)
    if not files:
        ext_str = ", ".join(extensions) if extensions else "any"
        print(f"Error: no matching files found ({ext_str})", file=sys.stderr)
        sys.exit(1)
    return files


# ---------------------------------------------------------------------------
# Config integration helpers
# ---------------------------------------------------------------------------


def _load_file_config(args: argparse.Namespace) -> dict[str, Any]:
    """Load config from the -c/--config flag, or return empty dict."""
    if getattr(args, "config", None) is not None:
        from rsi_python.config import load_config

        return load_config(args.config)
    return {}


def _merge_for_section(args: argparse.Namespace, section: str) -> dict[str, Any]:
    """Load config file + CLI overrides and merge for the given section."""
    from rsi_python.config import merge_config

    file_config = _load_file_config(args)
    file_values = file_config.get(section, {})

    # CLI overrides for profiles section
    overrides = {}
    if section == "profiles":
        for attr in ("P_min", "W_min", "direction", "min_duration"):
            val = getattr(args, attr, None)
            if val is not None:
                overrides[attr] = val

    return merge_config(section, file_values, overrides)


def _setup_output_dir(
    args: argparse.Namespace,
    prefix: str,
    section: str,
    params: dict[str, Any],
) -> Path:
    """Resolve the sequential output directory, write signature file and config.yaml."""
    from rsi_python.config import resolve_output_dir, write_resolved_config

    output_dir = resolve_output_dir(args.output, prefix, section, params)
    write_resolved_config(output_dir, section, params)
    return output_dir


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def _cmd_nc(args: argparse.Namespace) -> None:
    """Convert Rockland .p files to NetCDF4."""
    from rsi_python.convert import convert_all, p_to_L1

    p_files = _resolve_p_files(args.files)

    if len(p_files) == 1 and args.output and not Path(args.output).is_dir():
        _, nc_path = p_to_L1(p_files[0], args.output)
        print(f"Wrote {nc_path} ({nc_path.stat().st_size / 1e6:.1f} MB)")
    else:
        output_dir = Path(args.output) if args.output else None
        convert_all(p_files, output_dir, jobs=args.jobs)


def _cmd_info(args: argparse.Namespace) -> None:
    """Print summary information about .p file(s)."""
    from rsi_python.p_file import PFile

    p_files = _resolve_p_files(args.files)
    for i, pf_path in enumerate(p_files):
        if i > 0:
            print("\n" + "=" * 60 + "\n")
        pf = PFile(pf_path)
        pf.summary()


def _cmd_init(args: argparse.Namespace) -> None:
    """Generate a template configuration file."""
    from rsi_python.config import generate_template

    path = Path(args.path)
    if path.exists() and not args.force:
        print(f"Error: {path} already exists (use --force to overwrite)", file=sys.stderr)
        sys.exit(1)
    generate_template(path)
    print(f"Wrote template config to {path}")


def _cmd_prof(args: argparse.Namespace) -> None:
    """Extract profiles from .p or full-record .nc files."""
    from rsi_python.profile import extract_profiles

    files = _resolve_files(args.files, {".p", ".nc"})

    merged = _merge_for_section(args, "profiles")
    output_dir = _setup_output_dir(args, "prof", "profiles", merged)
    print(f"Output directory: {output_dir}")

    for f in files:
        print(f"{f.name}:")
        extract_profiles(f, output_dir, **merged)


# ---------------------------------------------------------------------------
# Subcommand parsers
# ---------------------------------------------------------------------------


def _add_nc_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "nc",
        help="Convert .p files to NetCDF",
        description="Convert Rockland Scientific .p data files to NetCDF4.",
    )
    p.add_argument("files", nargs="+", metavar="FILE", help=".p file(s) or glob pattern(s)")
    p.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        default=None,
        help="Output file (single input) or directory (multiple inputs)",
    )
    p.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="Parallel workers (0 = all cores, default: 1)",
    )
    p.set_defaults(func=_cmd_nc)


def _add_info_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "info",
        help="Print .p file summary",
        description="Print summary of Rockland Scientific .p data files.",
    )
    p.add_argument("files", nargs="+", metavar="FILE", help=".p file(s) or glob pattern(s)")
    p.set_defaults(func=_cmd_info)


def _add_init_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "init",
        help="Generate a template configuration file",
        description="Write a fully-commented template config.yaml with all default values.",
    )
    p.add_argument(
        "path",
        nargs="?",
        default="config.yaml",
        help="Output path (default: config.yaml)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing file",
    )
    p.set_defaults(func=_cmd_init)


def _add_prof_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "prof",
        help="Extract profiles",
        description="Extract profiles from .p or full-record .nc files.",
    )
    p.add_argument("files", nargs="+", metavar="FILE", help=".p or .nc file(s) or glob pattern(s)")
    p.add_argument(
        "-o",
        "--output",
        metavar="DIR",
        required=True,
        help="Output directory for per-profile .nc files",
    )
    p.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="Parallel workers (0 = all cores, default: 1)",
    )
    p.add_argument(
        "--P-min", type=float, default=None, help="Minimum pressure [dbar] (default: 0.5)"
    )
    p.add_argument(
        "--W-min", type=float, default=None, help="Minimum fall rate [dbar/s] (default: 0.3)"
    )
    p.add_argument(
        "--direction",
        default=None,
        choices=["up", "down"],
        help="Profile direction (default: down)",
    )
    p.add_argument(
        "--min-duration", type=float, default=None, help="Minimum profile duration [s] (default: 7)"
    )
    p.set_defaults(func=_cmd_prof)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main CLI entry point for rsi-python."""
    from rsi_python import __version__

    parser = argparse.ArgumentParser(
        prog="rsi-tpw",
        description="rsi-python: Read Rockland Scientific .P files and convert to NetCDF.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-c",
        "--config",
        metavar="YAML",
        default=None,
        help="Configuration file (YAML). Generate a template with 'rsi-tpw init'.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_info_parser(subparsers)
    _add_nc_parser(subparsers)
    _add_init_parser(subparsers)
    _add_prof_parser(subparsers)

    args = parser.parse_args()
    args.func(args)
