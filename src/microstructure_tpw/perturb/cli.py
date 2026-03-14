# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""CLI for perturb.

Main command:
    perturb <subcommand> [options]

Subcommands:
    init     — Generate a template configuration file
    run      — Run full pipeline (discover → trim → merge → process → bin → combo)
    trim     — Trim corrupt final records from .p files
    merge    — Merge split .p files
    profiles — Extract per-profile NetCDFs
    diss     — Compute epsilon (TKE dissipation) per profile
    chi      — Compute chi (thermal variance dissipation) per profile
    ctd      — Time-bin CTD channels per file
    bin      — Depth/time bin profiles, diss, and chi
    combo    — Assemble combo NetCDFs from binned data
"""

import argparse
import sys
from pathlib import Path
from typing import Any


def _load_and_merge(config_path: str | None) -> dict[str, Any]:
    """Load config file or return empty dict."""
    if config_path is not None:
        from microstructure_tpw.perturb.config import load_config

        return load_config(config_path)
    return {}


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def _cmd_init(args: argparse.Namespace) -> None:
    """Generate a template configuration file."""
    from microstructure_tpw.perturb.config import generate_template

    path = Path(args.path)
    if path.exists() and not args.force:
        print(f"Error: {path} already exists (use --force to overwrite)", file=sys.stderr)
        sys.exit(1)
    generate_template(path)
    print(f"Wrote template config to {path}")


def _cmd_run(args: argparse.Namespace) -> None:
    """Run the full pipeline."""
    config = _load_and_merge(args.config)

    # CLI overrides for files section
    if args.p_file_root:
        config.setdefault("files", {})["p_file_root"] = args.p_file_root
    if args.output:
        config.setdefault("files", {})["output_root"] = args.output

    # Parallel override
    if args.jobs is not None:
        config.setdefault("parallel", {})["jobs"] = args.jobs

    # Explicit file list
    p_files = None
    if args.files:
        import glob as globmod

        p_files = []
        for pattern in args.files:
            expanded = sorted(globmod.glob(pattern))
            for f in expanded:
                p = Path(f)
                if p.is_file() and p.suffix.lower() == ".p":
                    p_files.append(p)
        if not p_files:
            print("Error: no .p files found", file=sys.stderr)
            sys.exit(1)

    # Hotel file override
    if hasattr(args, "hotel_file") and args.hotel_file:
        config.setdefault("hotel", {})["enable"] = True
        config.setdefault("hotel", {})["file"] = args.hotel_file

    from microstructure_tpw.perturb.pipeline import run_pipeline

    run_pipeline(config, p_files=p_files)


def _cmd_trim(args: argparse.Namespace) -> None:
    """Trim corrupt final records from .p files."""
    config = _load_and_merge(args.config)

    if args.p_file_root:
        config.setdefault("files", {})["p_file_root"] = args.p_file_root
    if args.output:
        config.setdefault("files", {})["output_root"] = args.output

    from microstructure_tpw.perturb.pipeline import run_trim

    results = run_trim(config)
    print(f"Trimmed {len(results)} files")


def _cmd_merge(args: argparse.Namespace) -> None:
    """Merge split .p files."""
    config = _load_and_merge(args.config)

    if args.p_file_root:
        config.setdefault("files", {})["p_file_root"] = args.p_file_root
    if args.output:
        config.setdefault("files", {})["output_root"] = args.output

    from microstructure_tpw.perturb.pipeline import run_merge

    results = run_merge(config)
    print(f"Merged {len(results)} file groups")


def _cmd_profiles(args: argparse.Namespace) -> None:
    """Extract per-profile NetCDFs."""
    config = _load_and_merge(args.config)

    if args.output:
        config.setdefault("files", {})["output_root"] = args.output
    if args.jobs is not None:
        config.setdefault("parallel", {})["jobs"] = args.jobs

    # Explicit file list
    p_files = None
    if args.files:
        import glob as globmod

        p_files = []
        for pattern in args.files:
            expanded = sorted(globmod.glob(pattern))
            for f in expanded:
                p = Path(f)
                if p.is_file() and p.suffix.lower() == ".p":
                    p_files.append(p)

    from microstructure_tpw.perturb.pipeline import run_pipeline

    # Disable stages we don't need
    config.setdefault("files", {})["trim"] = False
    config.setdefault("files", {})["merge"] = False
    run_pipeline(config, p_files=p_files)


def _cmd_diss(args: argparse.Namespace) -> None:
    """Compute epsilon per profile."""
    config = _load_and_merge(args.config)

    if args.output:
        config.setdefault("files", {})["output_root"] = args.output
    if args.jobs is not None:
        config.setdefault("parallel", {})["jobs"] = args.jobs

    p_files = None
    if args.files:
        import glob as globmod

        p_files = []
        for pattern in args.files:
            expanded = sorted(globmod.glob(pattern))
            for f in expanded:
                p = Path(f)
                if p.is_file() and p.suffix.lower() == ".p":
                    p_files.append(p)

    from microstructure_tpw.perturb.pipeline import run_pipeline

    config.setdefault("files", {})["trim"] = False
    config.setdefault("files", {})["merge"] = False
    run_pipeline(config, p_files=p_files)


def _cmd_chi(args: argparse.Namespace) -> None:
    """Compute chi per profile."""
    config = _load_and_merge(args.config)

    if args.output:
        config.setdefault("files", {})["output_root"] = args.output
    if args.jobs is not None:
        config.setdefault("parallel", {})["jobs"] = args.jobs

    # Enable chi
    config.setdefault("chi", {})["enable"] = True

    p_files = None
    if args.files:
        import glob as globmod

        p_files = []
        for pattern in args.files:
            expanded = sorted(globmod.glob(pattern))
            for f in expanded:
                p = Path(f)
                if p.is_file() and p.suffix.lower() == ".p":
                    p_files.append(p)

    from microstructure_tpw.perturb.pipeline import run_pipeline

    config.setdefault("files", {})["trim"] = False
    config.setdefault("files", {})["merge"] = False
    run_pipeline(config, p_files=p_files)


def _cmd_ctd(args: argparse.Namespace) -> None:
    """Time-bin CTD channels per file."""
    config = _load_and_merge(args.config)

    if args.output:
        config.setdefault("files", {})["output_root"] = args.output
    if args.jobs is not None:
        config.setdefault("parallel", {})["jobs"] = args.jobs

    p_files = None
    if args.files:
        import glob as globmod

        p_files = []
        for pattern in args.files:
            expanded = sorted(globmod.glob(pattern))
            for f in expanded:
                p = Path(f)
                if p.is_file() and p.suffix.lower() == ".p":
                    p_files.append(p)

    from microstructure_tpw.perturb.pipeline import run_pipeline

    config.setdefault("files", {})["trim"] = False
    config.setdefault("files", {})["merge"] = False
    config.setdefault("ctd", {})["enable"] = True
    run_pipeline(config, p_files=p_files)


def _cmd_bin(args: argparse.Namespace) -> None:
    """Depth/time bin profiles, diss, and chi."""
    config = _load_and_merge(args.config)

    if args.output:
        config.setdefault("files", {})["output_root"] = args.output

    from microstructure_tpw.perturb.binning import bin_by_depth, bin_by_time, bin_chi, bin_diss
    from microstructure_tpw.perturb.config import DEFAULTS, merge_config, resolve_output_dir

    files_cfg = config.get("files", {})
    output_root = Path(files_cfg.get("output_root", "results/"))

    binning_cfg = config.get("binning", {})
    bin_method = binning_cfg.get("method", DEFAULTS["binning"]["method"])
    bin_width = binning_cfg.get("width", DEFAULTS["binning"]["width"])
    aggregation = binning_cfg.get("aggregation", DEFAULTS["binning"]["aggregation"])
    diagnostics = binning_cfg.get("diagnostics", False)

    # Find profile output directories
    import glob as globmod

    # Bin profiles
    prof_dirs = sorted(globmod.glob(str(output_root / "profiles_[0-9][0-9]")))
    for prof_dir in prof_dirs:
        prof_ncs = sorted(Path(prof_dir).glob("*.nc"))
        if prof_ncs:
            print(f"Binning profiles from {prof_dir}...")
            binning_params = merge_config("binning", binning_cfg)
            profiles_params = merge_config("profiles", config.get("profiles"))
            prof_binned_dir = resolve_output_dir(
                output_root, "profiles_binned", "binning", binning_params,
                upstream=[("profiles", profiles_params)],
            )
            if bin_method == "depth":
                ds = bin_by_depth(prof_ncs, bin_width, aggregation, diagnostics)
            else:
                ds = bin_by_time(prof_ncs, bin_width, aggregation, diagnostics)
            if ds.data_vars:
                ds.to_netcdf(prof_binned_dir / "binned.nc")

    # Bin diss
    diss_dirs = sorted(globmod.glob(str(output_root / "diss_[0-9][0-9]")))
    for diss_dir in diss_dirs:
        diss_ncs = sorted(Path(diss_dir).glob("*.nc"))
        if diss_ncs:
            print(f"Binning diss from {diss_dir}...")
            diss_width = binning_cfg.get("diss_width") or bin_width
            diss_agg = binning_cfg.get("diss_aggregation") or aggregation
            ds = bin_diss(diss_ncs, diss_width, diss_agg, bin_method, diagnostics)
            if ds.data_vars:
                eps_params = merge_config("epsilon", config.get("epsilon"))
                diss_binned_dir = resolve_output_dir(
                    output_root, "diss_binned", "binning",
                    merge_config("binning", binning_cfg),
                    upstream=[("epsilon", eps_params)],
                )
                ds.to_netcdf(diss_binned_dir / "binned.nc")

    # Bin chi
    chi_dirs = sorted(globmod.glob(str(output_root / "chi_[0-9][0-9]")))
    for chi_dir in chi_dirs:
        chi_ncs = sorted(Path(chi_dir).glob("*.nc"))
        if chi_ncs:
            print(f"Binning chi from {chi_dir}...")
            chi_width = binning_cfg.get("chi_width") or binning_cfg.get("diss_width") or bin_width
            chi_agg = (
                binning_cfg.get("chi_aggregation")
                or binning_cfg.get("diss_aggregation")
                or aggregation
            )
            ds = bin_chi(chi_ncs, chi_width, chi_agg, bin_method, diagnostics)
            if ds.data_vars:
                chi_params = merge_config("chi", config.get("chi"))
                chi_binned_dir = resolve_output_dir(
                    output_root, "chi_binned", "binning",
                    merge_config("binning", binning_cfg),
                    upstream=[("chi", chi_params)],
                )
                ds.to_netcdf(chi_binned_dir / "binned.nc")

    print("Binning complete.")


def _cmd_combo(args: argparse.Namespace) -> None:
    """Assemble combo NetCDFs from binned data."""
    config = _load_and_merge(args.config)

    if args.output:
        config.setdefault("files", {})["output_root"] = args.output

    from microstructure_tpw.perturb.combo import make_combo, make_ctd_combo
    from microstructure_tpw.perturb.netcdf_schema import CHI_SCHEMA, COMBO_SCHEMA, CTD_SCHEMA

    files_cfg = config.get("files", {})
    output_root = Path(files_cfg.get("output_root", "results/"))
    netcdf_attrs = config.get("netcdf", {})

    import glob as globmod

    # Combo from profile binned
    for d in sorted(globmod.glob(str(output_root / "profiles_binned_[0-9][0-9]"))):
        out_dir = output_root / "combo"
        out = make_combo(d, out_dir, COMBO_SCHEMA, netcdf_attrs=netcdf_attrs)
        if out:
            print(f"  Wrote {out}")

    # Combo from diss binned
    for d in sorted(globmod.glob(str(output_root / "diss_binned_[0-9][0-9]"))):
        out_dir = output_root / "diss_combo"
        out = make_combo(d, out_dir, COMBO_SCHEMA, netcdf_attrs=netcdf_attrs)
        if out:
            print(f"  Wrote {out}")

    # Combo from chi binned
    for d in sorted(globmod.glob(str(output_root / "chi_binned_[0-9][0-9]"))):
        out_dir = output_root / "chi_combo"
        out = make_combo(d, out_dir, CHI_SCHEMA, netcdf_attrs=netcdf_attrs)
        if out:
            print(f"  Wrote {out}")

    # CTD combo
    for d in sorted(globmod.glob(str(output_root / "ctd_[0-9][0-9]"))):
        out_dir = output_root / "ctd_combo"
        out = make_ctd_combo(d, out_dir, CTD_SCHEMA, netcdf_attrs=netcdf_attrs)
        if out:
            print(f"  Wrote {out}")

    print("Combo assembly complete.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments shared by most subcommands."""
    parser.add_argument(
        "-c", "--config", metavar="YAML",
        help="configuration file (default: none)",
    )
    parser.add_argument(
        "-o", "--output", metavar="DIR",
        help="output root directory",
    )


def _add_parallel_args(parser: argparse.ArgumentParser) -> None:
    """Add --jobs flag."""
    parser.add_argument(
        "-j", "--jobs", type=int, default=None, metavar="N",
        help="parallel workers (0=auto, 1=serial, default=config or 1)",
    )


def _add_file_args(parser: argparse.ArgumentParser) -> None:
    """Add positional file patterns and --p-file-root."""
    parser.add_argument(
        "files", nargs="*", metavar="FILE",
        help=".p file paths or glob patterns",
    )
    parser.add_argument(
        "--p-file-root", metavar="DIR", default=None,
        help="root directory for .p file discovery",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="perturb",
        description="Batch processing pipeline for Rockland VMP/MicroRider data.",
    )
    sub = parser.add_subparsers(dest="command", help="subcommand")

    # init
    p_init = sub.add_parser("init", help="generate a template config file")
    p_init.add_argument("path", nargs="?", default="config.yaml", help="output path")
    p_init.add_argument("-f", "--force", action="store_true", help="overwrite existing")

    # run
    p_run = sub.add_parser("run", help="run full pipeline")
    _add_common_args(p_run)
    _add_parallel_args(p_run)
    _add_file_args(p_run)
    p_run.add_argument(
        "--hotel-file", metavar="FILE",
        help="hotel file (CSV, NetCDF, or .mat) with external telemetry",
    )

    # trim
    p_trim = sub.add_parser("trim", help="trim corrupt records from .p files")
    _add_common_args(p_trim)
    _add_file_args(p_trim)

    # merge
    p_merge = sub.add_parser("merge", help="merge split .p files")
    _add_common_args(p_merge)
    _add_file_args(p_merge)

    # profiles
    p_prof = sub.add_parser("profiles", help="extract per-profile NetCDFs")
    _add_common_args(p_prof)
    _add_parallel_args(p_prof)
    _add_file_args(p_prof)

    # diss
    p_diss = sub.add_parser("diss", help="compute epsilon per profile")
    _add_common_args(p_diss)
    _add_parallel_args(p_diss)
    _add_file_args(p_diss)

    # chi
    p_chi = sub.add_parser("chi", help="compute chi per profile")
    _add_common_args(p_chi)
    _add_parallel_args(p_chi)
    _add_file_args(p_chi)

    # ctd
    p_ctd = sub.add_parser("ctd", help="time-bin CTD channels per file")
    _add_common_args(p_ctd)
    _add_parallel_args(p_ctd)
    _add_file_args(p_ctd)

    # bin
    p_bin = sub.add_parser("bin", help="depth/time bin profiles, diss, chi")
    _add_common_args(p_bin)

    # combo
    p_combo = sub.add_parser("combo", help="assemble combo NetCDFs")
    _add_common_args(p_combo)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the perturb CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "init": _cmd_init,
        "run": _cmd_run,
        "trim": _cmd_trim,
        "merge": _cmd_merge,
        "profiles": _cmd_profiles,
        "diss": _cmd_diss,
        "chi": _cmd_chi,
        "ctd": _cmd_ctd,
        "bin": _cmd_bin,
        "combo": _cmd_combo,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)
