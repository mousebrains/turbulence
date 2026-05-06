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
import logging
import sys
from pathlib import Path
from typing import Any


def _load_and_merge(config_path: str | None) -> dict[str, Any]:
    """Load config file or return empty dict."""
    if config_path is not None:
        from odas_tpw.perturb.config import load_config

        return load_config(config_path)
    return {}


def _resolve_output_root(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    """Determine the output_root used for ``logs/`` placement.

    Precedence matches every subcommand's later override logic: ``args.output``
    > config ``files.output_root`` > the default ``"results/"``.
    """
    if getattr(args, "output", None):
        return Path(args.output)
    return Path(config.get("files", {}).get("output_root", "results/"))


def _install_logging(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    """Install root logging for this CLI invocation.

    Always writes to ``<output_root>/logs/run_<timestamp>.log``.  ``--stdout``
    additionally streams INFO records to stderr so the user can watch
    progress.  Returns the run log path so callers can mention it.
    """
    from odas_tpw.perturb.logging_setup import current_run_stamp, setup_root_logging

    output_root = _resolve_output_root(args, config)
    log_dir = output_root / "logs"
    log_path = log_dir / f"run_{current_run_stamp()}.log"
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_root_logging(log_path, level=level, stdout=args.stdout)
    return log_path


def _glob_p_files(patterns: list[str] | None) -> list[Path] | None:
    """Expand glob patterns to a list of .p file paths."""
    if not patterns:
        return None
    import glob as globmod

    p_files: list[Path] = []
    for pattern in patterns:
        for f in sorted(globmod.glob(pattern)):
            p = Path(f)
            if p.is_file() and p.suffix.lower() == ".p":
                p_files.append(p)
    return p_files or None


def _run_analysis(
    args: argparse.Namespace,
    extra_config: dict[str, Any] | None = None,
) -> None:
    """Shared handler for profiles/diss/chi/ctd subcommands."""
    config = _load_and_merge(args.config)

    if args.output:
        config.setdefault("files", {})["output_root"] = args.output
    if args.jobs is not None:
        config.setdefault("parallel", {})["jobs"] = args.jobs

    config.setdefault("files", {})["trim"] = False
    config.setdefault("files", {})["merge"] = False

    if extra_config:
        for section, values in extra_config.items():
            config.setdefault(section, {}).update(values)

    log_path = _install_logging(args, config)
    logging.getLogger(__name__).info("CLI %s, log: %s", args.command, log_path)

    from odas_tpw.perturb.pipeline import run_pipeline

    run_pipeline(config, p_files=_glob_p_files(args.files))


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def _cmd_init(args: argparse.Namespace) -> None:
    """Generate a template configuration file.

    No logging setup — ``init`` doesn't run the pipeline and doesn't have
    an ``output_root`` to host a ``logs/`` dir.
    """
    from odas_tpw.perturb.config import generate_template

    path = Path(args.path)
    if path.exists() and not args.force:
        print(f"Error: {path} already exists (use --force to overwrite)", file=sys.stderr)
        sys.exit(1)
    generate_template(path)
    print(f"Wrote template config to {path}")


def _cmd_run(args: argparse.Namespace) -> None:
    """Run the full pipeline."""
    config = _load_and_merge(args.config)

    if args.p_file_root:
        config.setdefault("files", {})["p_file_root"] = args.p_file_root
    if args.output:
        config.setdefault("files", {})["output_root"] = args.output
    if args.jobs is not None:
        config.setdefault("parallel", {})["jobs"] = args.jobs

    p_files = _glob_p_files(args.files)
    if args.files and not p_files:
        print("Error: no .p files found", file=sys.stderr)
        sys.exit(1)

    if hasattr(args, "hotel_file") and args.hotel_file:
        config.setdefault("hotel", {})["enable"] = True
        config.setdefault("hotel", {})["file"] = args.hotel_file

    log_path = _install_logging(args, config)
    logging.getLogger(__name__).info("CLI run, log: %s", log_path)

    from odas_tpw.perturb.pipeline import run_pipeline

    run_pipeline(config, p_files=p_files)


def _cmd_trim(args: argparse.Namespace) -> None:
    """Trim corrupt final records from .p files."""
    config = _load_and_merge(args.config)
    if args.p_file_root:
        config.setdefault("files", {})["p_file_root"] = args.p_file_root
    if args.output:
        config.setdefault("files", {})["output_root"] = args.output

    log_path = _install_logging(args, config)
    logging.getLogger(__name__).info("CLI trim, log: %s", log_path)

    from odas_tpw.perturb.pipeline import run_trim

    results = run_trim(config)
    print(f"Trimmed {len(results)} files")


def _cmd_merge(args: argparse.Namespace) -> None:
    """Merge split .p files."""
    config = _load_and_merge(args.config)
    if args.p_file_root:
        config.setdefault("files", {})["p_file_root"] = args.p_file_root
    if args.output:
        config.setdefault("files", {})["output_root"] = args.output

    log_path = _install_logging(args, config)
    logging.getLogger(__name__).info("CLI merge, log: %s", log_path)

    from odas_tpw.perturb.pipeline import run_merge

    results = run_merge(config)
    print(f"Merged {len(results)} file groups")


def _cmd_profiles(args: argparse.Namespace) -> None:
    _run_analysis(args)


def _cmd_diss(args: argparse.Namespace) -> None:
    _run_analysis(args)


def _cmd_chi(args: argparse.Namespace) -> None:
    _run_analysis(args, extra_config={"chi": {"enable": True}})


def _cmd_ctd(args: argparse.Namespace) -> None:
    _run_analysis(args, extra_config={"ctd": {"enable": True}})


def _cmd_bin(args: argparse.Namespace) -> None:
    """Depth/time bin profiles, diss, and chi."""
    config = _load_and_merge(args.config)

    if args.output:
        config.setdefault("files", {})["output_root"] = args.output

    log_path = _install_logging(args, config)
    logging.getLogger(__name__).info("CLI bin, log: %s", log_path)

    from odas_tpw.perturb.binning import bin_by_depth, bin_by_time, bin_chi, bin_diss
    from odas_tpw.perturb.config import DEFAULTS, merge_config, resolve_output_dir
    from odas_tpw.perturb.pipeline import _upstream_for

    files_cfg = config.get("files", {})
    output_root = Path(files_cfg.get("output_root", "results/"))

    binning_cfg = config.get("binning", {})
    bin_method = binning_cfg.get("method", DEFAULTS["binning"]["method"])
    bin_width = binning_cfg.get("width", DEFAULTS["binning"]["width"])
    aggregation = binning_cfg.get("aggregation", DEFAULTS["binning"]["aggregation"])
    diagnostics = binning_cfg.get("diagnostics", False)

    # Find profile output directories
    import glob as globmod

    # Bin profiles — binned dir resolved up front so per-input-file logs
    # (a.log, b.log, …) can be written into it via stage_log.
    prof_dirs = sorted(globmod.glob(str(output_root / "profiles_[0-9][0-9]")))
    for prof_dir in prof_dirs:
        prof_ncs = sorted(Path(prof_dir).glob("*.nc"))
        if prof_ncs:
            print(f"Binning profiles from {prof_dir}...")
            binning_params = merge_config("binning", binning_cfg)
            prof_binned_dir = resolve_output_dir(
                output_root,
                "profiles_binned",
                "binning",
                binning_params,
                upstream=_upstream_for("profiles_binned", config),
            )
            if bin_method == "depth":
                ds = bin_by_depth(
                    prof_ncs, bin_width, aggregation, diagnostics, log_dir=prof_binned_dir
                )
            else:
                ds = bin_by_time(
                    prof_ncs, bin_width, aggregation, diagnostics, log_dir=prof_binned_dir
                )
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
            diss_binned_dir = resolve_output_dir(
                output_root,
                "diss_binned",
                "binning",
                merge_config("binning", binning_cfg),
                upstream=_upstream_for("diss_binned", config),
            )
            ds = bin_diss(
                diss_ncs, diss_width, diss_agg, bin_method, diagnostics, log_dir=diss_binned_dir
            )
            if ds.data_vars:
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
            chi_binned_dir = resolve_output_dir(
                output_root,
                "chi_binned",
                "binning",
                merge_config("binning", binning_cfg),
                upstream=_upstream_for("chi_binned", config),
            )
            ds = bin_chi(
                chi_ncs, chi_width, chi_agg, bin_method, diagnostics, log_dir=chi_binned_dir
            )
            if ds.data_vars:
                ds.to_netcdf(chi_binned_dir / "binned.nc")

    print("Binning complete.")


def _cmd_combo(args: argparse.Namespace) -> None:
    """Assemble combo NetCDFs from binned data."""
    config = _load_and_merge(args.config)

    if args.output:
        config.setdefault("files", {})["output_root"] = args.output

    log_path = _install_logging(args, config)
    logging.getLogger(__name__).info("CLI combo, log: %s", log_path)

    from odas_tpw.perturb.combo import make_combo, make_ctd_combo
    from odas_tpw.perturb.config import DEFAULTS, merge_config, write_signature
    from odas_tpw.perturb.logging_setup import stage_log
    from odas_tpw.perturb.netcdf_schema import CHI_SCHEMA, COMBO_SCHEMA, CTD_SCHEMA
    from odas_tpw.perturb.pipeline import _upstream_for

    files_cfg = config.get("files", {})
    output_root = Path(files_cfg.get("output_root", "results/"))
    netcdf_attrs = config.get("netcdf", {})

    # Combo glue method follows the binning method: ``depth`` for vertical
    # profilers (widthwise stack of profiles), ``time`` for moored / AUV
    # deployments (lengthwise concat sorted by time).  ctd_combo always
    # uses time regardless — see _run_combo docstring.
    bin_method = config.get("binning", {}).get("method", DEFAULTS["binning"]["method"])

    import glob as globmod

    binning_p = merge_config("binning", config.get("binning"))

    # Combo from profile binned
    for d in sorted(globmod.glob(str(output_root / "profiles_binned_[0-9][0-9]"))):
        out_dir = output_root / "combo"
        out_dir.mkdir(parents=True, exist_ok=True)
        with stage_log(out_dir, "combo"):
            out = make_combo(
                d, out_dir, COMBO_SCHEMA, netcdf_attrs=netcdf_attrs, method=bin_method
            )
            if out:
                write_signature(
                    out_dir, "binning", binning_p, upstream=_upstream_for("combo", config)
                )
                print(f"  Wrote {out}")

    # Combo from diss binned
    for d in sorted(globmod.glob(str(output_root / "diss_binned_[0-9][0-9]"))):
        out_dir = output_root / "diss_combo"
        out_dir.mkdir(parents=True, exist_ok=True)
        with stage_log(out_dir, "combo"):
            out = make_combo(
                d, out_dir, COMBO_SCHEMA, netcdf_attrs=netcdf_attrs, method=bin_method
            )
            if out:
                write_signature(
                    out_dir, "binning", binning_p, upstream=_upstream_for("diss_combo", config)
                )
                print(f"  Wrote {out}")

    # Combo from chi binned
    for d in sorted(globmod.glob(str(output_root / "chi_binned_[0-9][0-9]"))):
        out_dir = output_root / "chi_combo"
        out_dir.mkdir(parents=True, exist_ok=True)
        with stage_log(out_dir, "combo"):
            out = make_combo(d, out_dir, CHI_SCHEMA, netcdf_attrs=netcdf_attrs, method=bin_method)
            if out:
                write_signature(
                    out_dir, "binning", binning_p, upstream=_upstream_for("chi_combo", config)
                )
                print(f"  Wrote {out}")

    # CTD combo
    for d in sorted(globmod.glob(str(output_root / "ctd_[0-9][0-9]"))):
        out_dir = output_root / "ctd_combo"
        out_dir.mkdir(parents=True, exist_ok=True)
        with stage_log(out_dir, "combo"):
            out = make_ctd_combo(d, out_dir, CTD_SCHEMA, netcdf_attrs=netcdf_attrs)
            if out:
                write_signature(
                    out_dir,
                    "ctd",
                    merge_config("ctd", config.get("ctd")),
                    upstream=_upstream_for("ctd_combo", config),
                )
                print(f"  Wrote {out}")

    print("Combo assembly complete.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments shared by most subcommands."""
    parser.add_argument(
        "-c",
        "--config",
        metavar="YAML",
        help="configuration file (default: none)",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="DIR",
        help="output root directory",
    )
    _add_logging_args(parser)


def _add_logging_args(parser: argparse.ArgumentParser) -> None:
    """Add logging-control flags shared by every pipeline-running subcommand."""
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="also stream log records to stderr (default: log to file only)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        metavar="LEVEL",
        help="log level: DEBUG, INFO, WARNING, ERROR (default: INFO)",
    )


def _add_parallel_args(parser: argparse.ArgumentParser) -> None:
    """Add --jobs flag."""
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        metavar="N",
        help="parallel workers (0=auto, 1=serial, default=config or 1)",
    )


def _add_file_args(parser: argparse.ArgumentParser) -> None:
    """Add positional file patterns and --p-file-root."""
    parser.add_argument(
        "files",
        nargs="*",
        metavar="FILE",
        help=".p file paths or glob patterns",
    )
    parser.add_argument(
        "--p-file-root",
        metavar="DIR",
        default=None,
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
        "--hotel-file",
        metavar="FILE",
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
    """Entry point for the perturb CLI.

    Logging is configured per-subcommand inside each handler, after the
    config is merged enough to know ``output_root`` (where the
    ``logs/run_<timestamp>.log`` file lives).
    """
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
