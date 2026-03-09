# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""
CLI for rsi-python.

Main command:
    rsi-tpw <subcommand> [options]

Subcommands:
    info     — Print summary of .p file(s)
    nc       — Convert .p files to NetCDF
    prof     — Extract profiles from .p or full-record .nc files
    eps      — Compute epsilon (TKE dissipation) from any pipeline stage
    chi      — Compute chi (thermal variance dissipation) from any pipeline stage
    pipeline — Run full processing pipeline (.p → profiles → epsilon → chi)
    init     — Generate a template configuration file
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _resolve_p_files(patterns):
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


def _resolve_files(patterns, extensions=None):
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
            if p.is_file():
                if extensions is None or p.suffix.lower() in extensions:
                    files.append(p)
    if not files:
        ext_str = ", ".join(extensions) if extensions else "any"
        print(f"Error: no matching files found ({ext_str})", file=sys.stderr)
        sys.exit(1)
    return files


# ---------------------------------------------------------------------------
# Config integration helpers
# ---------------------------------------------------------------------------


def _load_file_config(args):
    """Load config from the -c/--config flag, or return empty dict."""
    if getattr(args, "config", None) is not None:
        from rsi_python.config import load_config

        return load_config(args.config)
    return {}


def _extract_cli_overrides(args, section):
    """Extract CLI-provided overrides for a config section.

    Only includes values that were explicitly specified on the command line
    (i.e., not None).  Handles arg-name → config-key mapping and the
    --no-goodman inversion.
    """
    # Map from argparse attr name → config key name
    # Most are identity; only renames are listed explicitly.
    if section == "profiles":
        mapping = {
            "P_min": "P_min",
            "W_min": "W_min",
            "direction": "direction",
            "min_duration": "min_duration",
        }
    elif section == "epsilon":
        mapping = {
            "fft_length": "fft_length",
            "diss_length": "diss_length",
            "overlap": "overlap",
            "speed": "speed",
            "direction": "direction",
            "f_AA": "f_AA",
            "salinity": "salinity",
        }
    elif section == "chi":
        mapping = {
            "fft_length": "fft_length",
            "diss_length": "diss_length",
            "overlap": "overlap",
            "speed": "speed",
            "direction": "direction",
            "fp07_model": "fp07_model",
            "f_AA": "f_AA",
            "fit_method": "fit_method",
            "spectrum_model": "spectrum_model",
            "salinity": "salinity",
        }
    elif section == "epsilon_pipeline":
        # pipeline uses different attr names for fft_length
        mapping = {
            "eps_fft_length": "fft_length",
            "direction": "direction",
            "f_AA": "f_AA",
            "speed": "speed",
            "salinity": "salinity",
        }
    elif section == "chi_pipeline":
        mapping = {
            "chi_fft_length": "fft_length",
            "direction": "direction",
            "f_AA": "f_AA",
            "fp07_model": "fp07_model",
            "spectrum_model": "spectrum_model",
            "speed": "speed",
            "salinity": "salinity",
        }
    else:
        return {}

    overrides = {}
    for attr, key in mapping.items():
        val = getattr(args, attr, None)
        if val is not None:
            overrides[key] = val

    # Handle --no-goodman (store_const: None=not specified, True=specified)
    if section in ("epsilon", "epsilon_pipeline"):
        no_goodman = getattr(args, "no_goodman", None)
        if no_goodman is True:
            overrides["goodman"] = False

    return overrides


def _merge_for_section(args, section):
    """Load config file + CLI overrides and merge for the given section.

    Returns the merged kwargs dict (None values stripped).
    """
    from rsi_python.config import merge_config

    file_config = _load_file_config(args)
    # For pipeline pseudo-sections, map to the real config section name
    real_section = section.replace("_pipeline", "")
    file_values = file_config.get(real_section, {})
    cli_overrides = _extract_cli_overrides(args, section)
    return merge_config(real_section, file_values, cli_overrides)


def _setup_output_dir(args, prefix, section, params, upstream=None):
    """Resolve the sequential output directory, write signature file and config.yaml.

    Parameters
    ----------
    upstream : list of (section, params) tuples, optional
        Upstream sections to include in the hash for cumulative tracking.
    """
    from rsi_python.config import resolve_output_dir, write_resolved_config

    real_section = section.replace("_pipeline", "")
    output_dir = resolve_output_dir(args.output, prefix, real_section, params, upstream=upstream)
    write_resolved_config(output_dir, real_section, params, upstream=upstream)
    return output_dir


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def _cmd_nc(args):
    """Convert Rockland .p files to NetCDF4."""
    from rsi_python.convert import convert_all, p_to_netcdf

    p_files = _resolve_p_files(args.files)

    if len(p_files) == 1 and args.output and not Path(args.output).is_dir():
        _, nc_path = p_to_netcdf(p_files[0], args.output)
        print(f"Wrote {nc_path} ({nc_path.stat().st_size / 1e6:.1f} MB)")
    else:
        output_dir = Path(args.output) if args.output else None
        convert_all(p_files, output_dir, jobs=args.jobs)


def _cmd_info(args):
    """Print summary information about .p file(s)."""
    from rsi_python.p_file import PFile

    p_files = _resolve_p_files(args.files)
    for i, pf_path in enumerate(p_files):
        if i > 0:
            print("\n" + "=" * 60 + "\n")
        pf = PFile(pf_path)
        pf.summary()


def _cmd_init(args):
    """Generate a template configuration file."""
    from rsi_python.config import generate_template

    path = Path(args.path)
    if path.exists() and not args.force:
        print(f"Error: {path} already exists (use --force to overwrite)", file=sys.stderr)
        sys.exit(1)
    generate_template(path)
    print(f"Wrote template config to {path}")


def _cmd_prof(args):
    """Extract profiles from .p or full-record .nc files."""
    from rsi_python.profile import extract_profiles

    files = _resolve_files(args.files, {".p", ".nc"})

    merged = _merge_for_section(args, "profiles")
    output_dir = _setup_output_dir(args, "prof", "profiles", merged)
    print(f"Output directory: {output_dir}")

    for f in files:
        print(f"{f.name}:")
        extract_profiles(f, output_dir, **merged)


def _cmd_eps(args):
    """Compute epsilon (TKE dissipation rate) from any pipeline stage."""
    from rsi_python.dissipation import _compute_diss_one, compute_diss_file

    files = _resolve_files(args.files, {".p", ".nc"})

    merged = _merge_for_section(args, "epsilon")
    output_dir = _setup_output_dir(args, "eps", "epsilon", merged)
    print(f"Output directory: {output_dir}")

    jobs = args.jobs
    if jobs == 0:
        jobs = os.cpu_count() or 1

    if jobs == 1:
        for f in files:
            print(f"{f.name}:")
            try:
                compute_diss_file(f, output_dir, **merged)
            except Exception as e:
                print(f"  ERROR: {e}")
    else:
        work = []
        for f in files:
            work.append((f, output_dir, merged))
        print(f"Processing {len(work)} files with {jobs} workers")
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_compute_diss_one, w): w for w in work}
            for future in as_completed(futures):
                src, _, _ = futures[future]
                try:
                    name, n_profiles = future.result()
                    print(f"  {Path(name).name}: {n_profiles} profile(s)")
                except Exception as e:
                    print(f"  {src.name}: ERROR: {e}")


def _cmd_chi(args):
    """Compute chi (thermal variance dissipation rate) from any pipeline stage."""
    from rsi_python.chi import _compute_chi_one, compute_chi_file

    files = _resolve_files(args.files, {".p", ".nc"})

    merged = _merge_for_section(args, "chi")
    output_dir = _setup_output_dir(args, "chi", "chi", merged)
    print(f"Output directory: {output_dir}")

    # Load epsilon datasets if Method 1
    epsilon_dir = getattr(args, "epsilon_dir", None)

    jobs = args.jobs
    if jobs == 0:
        jobs = os.cpu_count() or 1

    if jobs == 1:
        for f in files:
            print(f"{f.name}:")

            kw = dict(merged)
            if epsilon_dir is not None:
                import xarray as xr

                eps_file = Path(epsilon_dir) / f"{f.stem}_eps.nc"
                if eps_file.exists():
                    kw["epsilon_ds"] = xr.open_dataset(eps_file)
                else:
                    print(f"  Warning: no epsilon file {eps_file.name}, using Method 2")

            try:
                compute_chi_file(f, output_dir, **kw)
            except Exception as e:
                print(f"  ERROR: {e}")
    else:
        work = []
        for f in files:
            work.append((f, output_dir, merged))
        print(f"Processing {len(work)} files with {jobs} workers")
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_compute_chi_one, w): w for w in work}
            for future in as_completed(futures):
                src, _, _ = futures[future]
                try:
                    name, n_profiles = future.result()
                    print(f"  {Path(name).name}: {n_profiles} profile(s)")
                except Exception as e:
                    print(f"  {src.name}: ERROR: {e}")


def _cmd_pipeline(args):
    """Run full processing pipeline: .p → profiles → epsilon → chi."""
    from rsi_python.chi import compute_chi_file
    from rsi_python.dissipation import compute_diss_file

    p_files = _resolve_p_files(args.files)

    eps_merged = _merge_for_section(args, "epsilon_pipeline")
    chi_merged = _merge_for_section(args, "chi_pipeline")

    eps_dir = _setup_output_dir(args, "eps", "epsilon_pipeline", eps_merged)
    chi_upstream = [("epsilon", eps_merged)]
    chi_dir = _setup_output_dir(args, "chi", "chi_pipeline", chi_merged, upstream=chi_upstream)
    print(f"Epsilon output: {eps_dir}")
    print(f"Chi output:     {chi_dir}")

    for f in p_files:
        print(f"\n{'=' * 60}")
        print(f"{f.name}")
        print(f"{'=' * 60}")

        # Step 1: Compute epsilon
        print("\n--- Epsilon ---")
        try:
            eps_paths = compute_diss_file(f, eps_dir, **eps_merged)
        except Exception as e:
            print(f"  ERROR computing epsilon: {e}")
            continue

        # Step 2: Compute chi with epsilon (Method 1)
        print("\n--- Chi (Method 1: from epsilon) ---")
        for eps_path in eps_paths:
            import xarray as xr

            eps_ds = xr.open_dataset(eps_path)
            kw = dict(chi_merged)
            kw["epsilon_ds"] = eps_ds
            try:
                compute_chi_file(f, chi_dir, **kw)
            except Exception as e:
                print(f"  ERROR computing chi: {e}")
            finally:
                eps_ds.close()

    print("\nPipeline complete.")
    print(f"  Epsilon: {eps_dir}/")
    print(f"  Chi:     {chi_dir}/")


# ---------------------------------------------------------------------------
# Subcommand parsers
# ---------------------------------------------------------------------------


def _add_nc_parser(subparsers):
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


def _add_info_parser(subparsers):
    p = subparsers.add_parser(
        "info",
        help="Print .p file summary",
        description="Print summary of Rockland Scientific .p data files.",
    )
    p.add_argument("files", nargs="+", metavar="FILE", help=".p file(s) or glob pattern(s)")
    p.set_defaults(func=_cmd_info)


def _add_init_parser(subparsers):
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


def _add_prof_parser(subparsers):
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


def _add_eps_parser(subparsers):
    p = subparsers.add_parser(
        "eps",
        help="Compute epsilon (TKE dissipation)",
        description="Compute TKE dissipation rate (epsilon) from VMP data.",
    )
    p.add_argument(
        "files", nargs="+", metavar="FILE", help=".p, full-record .nc, or per-profile .nc file(s)"
    )
    p.add_argument(
        "-o",
        "--output",
        metavar="DIR",
        required=True,
        help="Output directory for epsilon .nc files",
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
        "--fft-length", type=int, default=None, help="FFT segment length [samples] (default: 256)"
    )
    p.add_argument(
        "--diss-length",
        type=int,
        default=None,
        help="Dissipation window [samples] (default: 2*fft-length)",
    )
    p.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="Window overlap [samples] (default: diss-length//2)",
    )
    p.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Fixed profiling speed [m/s] (default: from dP/dt)",
    )
    p.add_argument(
        "--direction",
        default=None,
        choices=["up", "down"],
        help="Profile direction (default: down)",
    )
    p.add_argument(
        "--no-goodman",
        action="store_const",
        const=True,
        default=None,
        help="Disable Goodman coherent noise removal",
    )
    p.add_argument(
        "--f-AA", type=float, default=None, help="Anti-aliasing filter cutoff [Hz] (default: 98)"
    )
    p.add_argument(
        "--salinity",
        type=float,
        default=None,
        help="Salinity [PSU] for viscosity (default: 35, fixed S)",
    )
    p.set_defaults(func=_cmd_eps)


def _add_chi_parser(subparsers):
    p = subparsers.add_parser(
        "chi",
        help="Compute chi (thermal dissipation)",
        description="Compute thermal variance dissipation rate (chi) from VMP data.",
    )
    p.add_argument(
        "files", nargs="+", metavar="FILE", help=".p, full-record .nc, or per-profile .nc file(s)"
    )
    p.add_argument(
        "-o",
        "--output",
        metavar="DIR",
        required=True,
        help="Output directory for chi .nc files",
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
        "--fft-length", type=int, default=None, help="FFT segment length [samples] (default: 512)"
    )
    p.add_argument(
        "--diss-length",
        type=int,
        default=None,
        help="Dissipation window [samples] (default: 3*fft-length)",
    )
    p.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="Window overlap [samples] (default: diss-length//2)",
    )
    p.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Fixed profiling speed [m/s] (default: from dP/dt)",
    )
    p.add_argument(
        "--direction",
        default=None,
        choices=["up", "down"],
        help="Profile direction (default: down)",
    )
    p.add_argument(
        "--fp07-model",
        default=None,
        choices=["single_pole", "double_pole"],
        help="FP07 transfer function model (default: single_pole)",
    )
    p.add_argument(
        "--epsilon-dir",
        metavar="DIR",
        default=None,
        help="Directory with epsilon .nc files from 'rsi-tpw eps' (Method 1). "
        "If omitted, uses Method 2 (spectral fitting).",
    )
    p.add_argument(
        "--fit-method",
        default=None,
        choices=["mle", "iterative"],
        help="Method 2 fitting: mle or iterative (default: mle)",
    )
    p.add_argument(
        "--spectrum-model",
        default=None,
        choices=["batchelor", "kraichnan"],
        help="Theoretical spectrum model (default: batchelor)",
    )
    p.add_argument(
        "--f-AA", type=float, default=None, help="Anti-aliasing filter cutoff [Hz] (default: 98)"
    )
    p.add_argument(
        "--salinity",
        type=float,
        default=None,
        help="Salinity [PSU] for viscosity (default: 35, fixed S)",
    )
    p.set_defaults(func=_cmd_chi)


def _add_pipeline_parser(subparsers):
    p = subparsers.add_parser(
        "pipeline",
        help="Run full pipeline (.p -> epsilon -> chi)",
        description="Run the full processing pipeline from raw .p files through "
        "epsilon (TKE dissipation) and chi (thermal dissipation). "
        "Profiles are detected automatically. Chi is computed using "
        "Method 1 (from shear-probe epsilon).",
    )
    p.add_argument("files", nargs="+", metavar="FILE", help=".p file(s) or glob pattern(s)")
    p.add_argument(
        "-o",
        "--output",
        metavar="DIR",
        required=True,
        help="Base output directory",
    )
    p.add_argument(
        "--direction",
        default=None,
        choices=["up", "down"],
        help="Profile direction (default: down)",
    )
    p.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Fixed profiling speed [m/s] (default: from dP/dt)",
    )
    p.add_argument(
        "--eps-fft-length",
        type=int,
        default=None,
        help="FFT length for epsilon [samples] (default: 256)",
    )
    p.add_argument(
        "--chi-fft-length",
        type=int,
        default=None,
        help="FFT length for chi [samples] (default: 512)",
    )
    p.add_argument(
        "--no-goodman",
        action="store_const",
        const=True,
        default=None,
        help="Disable Goodman coherent noise removal for epsilon",
    )
    p.add_argument(
        "--fp07-model",
        default=None,
        choices=["single_pole", "double_pole"],
        help="FP07 transfer function model (default: single_pole)",
    )
    p.add_argument(
        "--spectrum-model",
        default=None,
        choices=["batchelor", "kraichnan"],
        help="Theoretical spectrum model for chi (default: batchelor)",
    )
    p.add_argument(
        "--f-AA", type=float, default=None, help="Anti-aliasing filter cutoff [Hz] (default: 98)"
    )
    p.add_argument(
        "--salinity",
        type=float,
        default=None,
        help="Salinity [PSU] for viscosity (default: 35, fixed S)",
    )
    p.set_defaults(func=_cmd_pipeline)


def _cmd_ql(args):
    """Interactive quick-look viewer."""
    from rsi_python.quick_look import quick_look

    p_files = _resolve_p_files(args.files)
    for pf_path in p_files:
        quick_look(
            pf_path,
            fft_length=args.fft_length or 256,
            f_AA=args.f_AA or 98.0,
            goodman=not args.no_goodman,
            direction=args.direction or "down",
        )


def _add_ql_parser(subparsers):
    p = subparsers.add_parser(
        "ql",
        help="Interactive quick-look viewer",
        description="Open an interactive multi-panel viewer with profile navigation.",
    )
    p.add_argument("files", nargs="+", metavar="FILE", help=".p file(s) or glob pattern(s)")
    p.add_argument(
        "--fft-length", type=int, default=None, help="FFT segment length [samples] (default: 256)"
    )
    p.add_argument(
        "--f-AA", type=float, default=None, help="Anti-aliasing filter cutoff [Hz] (default: 98)"
    )
    p.add_argument(
        "--no-goodman",
        action="store_true",
        default=False,
        help="Disable Goodman coherent noise removal",
    )
    p.add_argument(
        "--direction",
        default=None,
        choices=["up", "down"],
        help="Profile direction (default: down)",
    )
    p.set_defaults(func=_cmd_ql)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main CLI entry point for rsi-python."""
    from rsi_python import __version__

    parser = argparse.ArgumentParser(
        prog="rsi-tpw",
        description="rsi-python: Tools for Rockland Scientific microprofiler data.",
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
    _add_eps_parser(subparsers)
    _add_chi_parser(subparsers)
    _add_pipeline_parser(subparsers)
    _add_ql_parser(subparsers)

    args = parser.parse_args()
    args.func(args)
