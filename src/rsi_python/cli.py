"""
CLI entry points for rsi-python.

Commands:
    p2nc   — Convert .p files to NetCDF
    pinfo  — Print summary of .p file(s)
    p2prof — Extract profiles from .p or full-record .nc files
    p2eps  — Compute epsilon (TKE dissipation) from any pipeline stage
    p2chi  — Compute chi (thermal variance dissipation) from any pipeline stage
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


def p2nc():
    """Convert Rockland .p files to NetCDF4."""
    parser = argparse.ArgumentParser(
        prog="p2nc",
        description="Convert Rockland Scientific .p data files to NetCDF4.",
    )
    parser.add_argument(
        "files", nargs="+", metavar="FILE",
        help=".p file(s) or glob pattern(s) to convert",
    )
    parser.add_argument(
        "-o", "--output", metavar="PATH", default=None,
        help="Output file (single input) or directory (multiple inputs)",
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=1, metavar="N",
        help="Number of parallel workers (0 = all cores, default: 1)",
    )
    args = parser.parse_args()

    from rsi_python.convert import p_to_netcdf, convert_all

    p_files = _resolve_p_files(args.files)

    if len(p_files) == 1 and args.output and not Path(args.output).is_dir():
        _, nc_path = p_to_netcdf(p_files[0], args.output)
        print(f"Wrote {nc_path} ({nc_path.stat().st_size / 1e6:.1f} MB)")
    else:
        output_dir = Path(args.output) if args.output else None
        convert_all(p_files, output_dir, jobs=args.jobs)


def pinfo():
    """Print summary information about .p file(s)."""
    parser = argparse.ArgumentParser(
        prog="pinfo",
        description="Print summary of Rockland Scientific .p data files.",
    )
    parser.add_argument(
        "files", nargs="+", metavar="FILE",
        help=".p file(s) or glob pattern(s)",
    )
    args = parser.parse_args()

    from rsi_python.p_file import PFile

    p_files = _resolve_p_files(args.files)
    for i, pf_path in enumerate(p_files):
        if i > 0:
            print("\n" + "=" * 60 + "\n")
        pf = PFile(pf_path)
        pf.summary()


def p2prof():
    """Extract profiles from .p or full-record .nc files."""
    parser = argparse.ArgumentParser(
        prog="p2prof",
        description="Extract profiles from Rockland .p or full-record .nc files.",
    )
    parser.add_argument(
        "files", nargs="+", metavar="FILE",
        help=".p or .nc file(s) or glob pattern(s)",
    )
    parser.add_argument(
        "-o", "--output", metavar="DIR", default=None,
        help="Output directory for per-profile .nc files",
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=1, metavar="N",
        help="Parallel workers (0 = all cores, default: 1)",
    )
    parser.add_argument("--P-min", type=float, default=0.5,
                        help="Minimum pressure [dbar] (default: 0.5)")
    parser.add_argument("--W-min", type=float, default=0.3,
                        help="Minimum fall rate [dbar/s] (default: 0.3)")
    parser.add_argument("--direction", default="down", choices=["up", "down"],
                        help="Profile direction (default: down)")
    parser.add_argument("--min-duration", type=float, default=7.0,
                        help="Minimum profile duration [s] (default: 7)")
    args = parser.parse_args()

    from rsi_python.profile import extract_profiles, _extract_one

    files = _resolve_files(args.files, {".p", ".nc"})

    profile_kwargs = {
        "P_min": args.P_min,
        "W_min": args.W_min,
        "direction": args.direction,
        "min_duration": args.min_duration,
    }

    jobs = args.jobs
    if jobs == 0:
        jobs = os.cpu_count() or 1

    for f in files:
        output_dir = Path(args.output) if args.output else f.parent
        print(f"{f.name}:")
        if jobs == 1:
            extract_profiles(f, output_dir, **profile_kwargs)
        else:
            # For p2prof, each file is processed independently
            extract_profiles(f, output_dir, **profile_kwargs)


def p2eps():
    """Compute epsilon (TKE dissipation rate) from any pipeline stage."""
    parser = argparse.ArgumentParser(
        prog="p2eps",
        description="Compute TKE dissipation rate (epsilon) from VMP data.",
    )
    parser.add_argument(
        "files", nargs="+", metavar="FILE",
        help=".p, full-record .nc, or per-profile .nc file(s)",
    )
    parser.add_argument(
        "-o", "--output", metavar="DIR", default=None,
        help="Output directory for epsilon .nc files",
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=1, metavar="N",
        help="Parallel workers (0 = all cores, default: 1)",
    )
    parser.add_argument("--fft-length", type=int, default=256,
                        help="FFT segment length [samples] (default: 256)")
    parser.add_argument("--diss-length", type=int, default=None,
                        help="Dissipation window [samples] (default: 2*fft-length)")
    parser.add_argument("--overlap", type=int, default=None,
                        help="Window overlap [samples] (default: diss-length//2)")
    parser.add_argument("--speed", type=float, default=None,
                        help="Fixed profiling speed [m/s] (default: from dP/dt)")
    parser.add_argument("--direction", default="down", choices=["up", "down"],
                        help="Profile direction (default: down)")
    parser.add_argument("--no-goodman", action="store_true",
                        help="Disable Goodman coherent noise removal")
    parser.add_argument("--f-AA", type=float, default=98.0,
                        help="Anti-aliasing filter cutoff [Hz] (default: 98)")
    args = parser.parse_args()

    from rsi_python.dissipation import compute_diss_file, _compute_diss_one

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


def p2chi():
    """Compute chi (thermal variance dissipation rate) from any pipeline stage."""
    parser = argparse.ArgumentParser(
        prog="p2chi",
        description="Compute thermal variance dissipation rate (chi) from VMP data.",
    )
    parser.add_argument(
        "files", nargs="+", metavar="FILE",
        help=".p, full-record .nc, or per-profile .nc file(s)",
    )
    parser.add_argument(
        "-o", "--output", metavar="DIR", default=None,
        help="Output directory for chi .nc files",
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=1, metavar="N",
        help="Parallel workers (0 = all cores, default: 1)",
    )
    parser.add_argument("--fft-length", type=int, default=512,
                        help="FFT segment length [samples] (default: 512)")
    parser.add_argument("--diss-length", type=int, default=None,
                        help="Dissipation window [samples] (default: 3*fft-length)")
    parser.add_argument("--overlap", type=int, default=None,
                        help="Window overlap [samples] (default: diss-length//2)")
    parser.add_argument("--speed", type=float, default=None,
                        help="Fixed profiling speed [m/s] (default: from dP/dt)")
    parser.add_argument("--direction", default="down", choices=["up", "down"],
                        help="Profile direction (default: down)")
    parser.add_argument("--fp07-model", default="single_pole",
                        choices=["single_pole", "double_pole"],
                        help="FP07 transfer function model (default: single_pole)")
    parser.add_argument("--epsilon-dir", metavar="DIR", default=None,
                        help="Directory with epsilon .nc files from p2eps (Method 1). "
                             "If omitted, uses Method 2 (spectral fitting).")
    parser.add_argument("--fit-method", default="mle",
                        choices=["mle", "iterative"],
                        help="Method 2 fitting: mle (Ruddick 2000) or iterative "
                             "(Peterson & Fer 2014). Default: mle")
    parser.add_argument("--spectrum-model", default="batchelor",
                        choices=["batchelor", "kraichnan"],
                        help="Theoretical spectrum model (default: batchelor)")
    parser.add_argument("--f-AA", type=float, default=98.0,
                        help="Anti-aliasing filter cutoff [Hz] (default: 98)")
    args = parser.parse_args()

    from rsi_python.chi import compute_chi_file, _compute_chi_one

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

    # Load epsilon datasets if Method 1
    if args.epsilon_dir is not None:
        import xarray as xr
        eps_dir = Path(args.epsilon_dir)
        # Will be matched per-file below
        chi_kwargs["_epsilon_dir"] = eps_dir

    jobs = args.jobs
    if jobs == 0:
        jobs = os.cpu_count() or 1

    if jobs == 1:
        for f in files:
            output_dir = Path(args.output) if args.output else f.parent
            print(f"{f.name}:")

            # Match epsilon file if Method 1
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
