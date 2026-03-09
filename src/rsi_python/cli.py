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


def _cmd_prof(args):
    """Extract profiles from .p or full-record .nc files."""
    from rsi_python.profile import extract_profiles

    files = _resolve_files(args.files, {".p", ".nc"})

    profile_kwargs = {
        "P_min": args.P_min,
        "W_min": args.W_min,
        "direction": args.direction,
        "min_duration": args.min_duration,
    }

    for f in files:
        output_dir = Path(args.output) if args.output else f.parent
        print(f"{f.name}:")
        extract_profiles(f, output_dir, **profile_kwargs)


def _cmd_eps(args):
    """Compute epsilon (TKE dissipation rate) from any pipeline stage."""
    from rsi_python.dissipation import _compute_diss_one, compute_diss_file

    files = _resolve_files(args.files, {".p", ".nc"})

    diss_kwargs = {
        "fft_length": args.fft_length,
        "goodman": not args.no_goodman,
        "f_AA": args.f_AA,
        "direction": args.direction,
    }
    if args.diss_length is not None:
        diss_kwargs["diss_length"] = args.diss_length
    if args.overlap is not None:
        diss_kwargs["overlap"] = args.overlap
    if args.speed is not None:
        diss_kwargs["speed"] = args.speed
    if args.salinity is not None:
        diss_kwargs["salinity"] = args.salinity

    jobs = args.jobs
    if jobs == 0:
        jobs = os.cpu_count() or 1

    if jobs == 1:
        for f in files:
            output_dir = Path(args.output) if args.output else f.parent
            print(f"{f.name}:")
            try:
                compute_diss_file(f, output_dir, **diss_kwargs)
            except Exception as e:
                print(f"  ERROR: {e}")
    else:
        work = []
        for f in files:
            output_dir = Path(args.output) if args.output else f.parent
            work.append((f, output_dir, diss_kwargs))
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

    chi_kwargs = {
        "fft_length": args.fft_length,
        "f_AA": args.f_AA,
        "direction": args.direction,
        "fp07_model": args.fp07_model,
        "fit_method": args.fit_method,
        "spectrum_model": args.spectrum_model,
    }
    if args.diss_length is not None:
        chi_kwargs["diss_length"] = args.diss_length
    if args.overlap is not None:
        chi_kwargs["overlap"] = args.overlap
    if args.speed is not None:
        chi_kwargs["speed"] = args.speed
    if args.salinity is not None:
        chi_kwargs["salinity"] = args.salinity

    # Load epsilon datasets if Method 1
    if args.epsilon_dir is not None:
        chi_kwargs["_epsilon_dir"] = Path(args.epsilon_dir)

    jobs = args.jobs
    if jobs == 0:
        jobs = os.cpu_count() or 1

    if jobs == 1:
        for f in files:
            output_dir = Path(args.output) if args.output else f.parent
            print(f"{f.name}:")

            kw = dict(chi_kwargs)
            eps_dir = kw.pop("_epsilon_dir", None)
            if eps_dir is not None:
                import xarray as xr

                eps_file = eps_dir / f"{f.stem}_eps.nc"
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
            output_dir = Path(args.output) if args.output else f.parent
            kw = dict(chi_kwargs)
            kw.pop("_epsilon_dir", None)
            work.append((f, output_dir, kw))
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

    output_dir = Path(args.output) if args.output else Path(".")
    eps_dir = output_dir / "epsilon"
    chi_dir = output_dir / "chi"
    eps_dir.mkdir(parents=True, exist_ok=True)
    chi_dir.mkdir(parents=True, exist_ok=True)

    diss_kwargs = {
        "fft_length": args.eps_fft_length,
        "goodman": not args.no_goodman,
        "f_AA": args.f_AA,
        "direction": args.direction,
    }
    if args.speed is not None:
        diss_kwargs["speed"] = args.speed
    if args.salinity is not None:
        diss_kwargs["salinity"] = args.salinity

    chi_kwargs = {
        "fft_length": args.chi_fft_length,
        "f_AA": args.f_AA,
        "direction": args.direction,
        "fp07_model": args.fp07_model,
        "spectrum_model": args.spectrum_model,
    }
    if args.speed is not None:
        chi_kwargs["speed"] = args.speed
    if args.salinity is not None:
        chi_kwargs["salinity"] = args.salinity

    for f in p_files:
        print(f"\n{'=' * 60}")
        print(f"{f.name}")
        print(f"{'=' * 60}")

        # Step 1: Compute epsilon
        print("\n--- Epsilon ---")
        try:
            eps_paths = compute_diss_file(f, eps_dir, **diss_kwargs)
        except Exception as e:
            print(f"  ERROR computing epsilon: {e}")
            continue

        # Step 2: Compute chi with epsilon (Method 1)
        print("\n--- Chi (Method 1: from epsilon) ---")
        for eps_path in eps_paths:
            import xarray as xr

            eps_ds = xr.open_dataset(eps_path)
            kw = dict(chi_kwargs)
            kw["epsilon_ds"] = eps_ds
            try:
                compute_chi_file(f, chi_dir, **kw)
            except Exception as e:
                print(f"  ERROR computing chi: {e}")
            finally:
                eps_ds.close()

    print(f"\nPipeline complete. Output in {output_dir}/")
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
        default=None,
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
        "--P-min", type=float, default=0.5, help="Minimum pressure [dbar] (default: 0.5)"
    )
    p.add_argument(
        "--W-min", type=float, default=0.3, help="Minimum fall rate [dbar/s] (default: 0.3)"
    )
    p.add_argument(
        "--direction",
        default="down",
        choices=["up", "down"],
        help="Profile direction (default: down)",
    )
    p.add_argument(
        "--min-duration", type=float, default=7.0, help="Minimum profile duration [s] (default: 7)"
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
        "-o", "--output", metavar="DIR", default=None, help="Output directory for epsilon .nc files"
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
        "--fft-length", type=int, default=256, help="FFT segment length [samples] (default: 256)"
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
        default="down",
        choices=["up", "down"],
        help="Profile direction (default: down)",
    )
    p.add_argument(
        "--no-goodman", action="store_true", help="Disable Goodman coherent noise removal"
    )
    p.add_argument(
        "--f-AA", type=float, default=98.0, help="Anti-aliasing filter cutoff [Hz] (default: 98)"
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
        "-o", "--output", metavar="DIR", default=None, help="Output directory for chi .nc files"
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
        "--fft-length", type=int, default=512, help="FFT segment length [samples] (default: 512)"
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
        default="down",
        choices=["up", "down"],
        help="Profile direction (default: down)",
    )
    p.add_argument(
        "--fp07-model",
        default="single_pole",
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
        default="mle",
        choices=["mle", "iterative"],
        help="Method 2 fitting: mle or iterative (default: mle)",
    )
    p.add_argument(
        "--spectrum-model",
        default="batchelor",
        choices=["batchelor", "kraichnan"],
        help="Theoretical spectrum model (default: batchelor)",
    )
    p.add_argument(
        "--f-AA", type=float, default=98.0, help="Anti-aliasing filter cutoff [Hz] (default: 98)"
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
        default=".",
        help="Base output directory (default: current directory)",
    )
    p.add_argument(
        "--direction",
        default="down",
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
        default=256,
        help="FFT length for epsilon [samples] (default: 256)",
    )
    p.add_argument(
        "--chi-fft-length",
        type=int,
        default=512,
        help="FFT length for chi [samples] (default: 512)",
    )
    p.add_argument(
        "--no-goodman",
        action="store_true",
        help="Disable Goodman coherent noise removal for epsilon",
    )
    p.add_argument(
        "--fp07-model",
        default="single_pole",
        choices=["single_pole", "double_pole"],
        help="FP07 transfer function model (default: single_pole)",
    )
    p.add_argument(
        "--spectrum-model",
        default="batchelor",
        choices=["batchelor", "kraichnan"],
        help="Theoretical spectrum model for chi (default: batchelor)",
    )
    p.add_argument(
        "--f-AA", type=float, default=98.0, help="Anti-aliasing filter cutoff [Hz] (default: 98)"
    )
    p.add_argument(
        "--salinity",
        type=float,
        default=None,
        help="Salinity [PSU] for viscosity (default: 35, fixed S)",
    )
    p.set_defaults(func=_cmd_pipeline)


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
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_info_parser(subparsers)
    _add_nc_parser(subparsers)
    _add_prof_parser(subparsers)
    _add_eps_parser(subparsers)
    _add_chi_parser(subparsers)
    _add_pipeline_parser(subparsers)

    args = parser.parse_args()
    args.func(args)
