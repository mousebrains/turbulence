"""pyturb-cli: Jesse-compatible CLI for VMP turbulence processing.

Provides four subcommands matching Jesse's pyturb interface:
  p2nc   — convert .p files to NetCDF
  merge  — merge multiple NetCDF files
  eps    — compute epsilon (and gradT spectra)
  bin    — depth-bin profile results
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger("pyturb")


def _add_p2nc(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("p2nc", help="Convert .p files to NetCDF")
    p.add_argument("files", nargs="+", help=".p files or glob patterns")
    p.add_argument("-o", "--output", default=".", help="Output directory (default: .)")
    p.add_argument("--compress", action="store_true", default=True, dest="compress")
    p.add_argument("--no-compress", action="store_false", dest="compress")
    p.add_argument("--compression-level", type=int, default=4, help="zlib level 1-9 (default: 4)")
    p.add_argument("-n", "--n-workers", type=int, default=1, help="Parallel workers (default: 1)")
    p.add_argument("--min-file-size", type=int, default=100000, help="Skip files < N bytes")
    p.add_argument("-w", "--overwrite", action="store_true", default=False)
    p.add_argument("-W", "--no-overwrite", action="store_false", dest="overwrite")
    p.set_defaults(func=_run_p2nc)


def _add_merge(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("merge", help="Merge NetCDF files along time")
    p.add_argument("files", nargs="+", help=".nc files")
    p.add_argument("-o", "--output", required=True, help="Output file (required)")
    p.add_argument("--dry-run", action="store_true", help="Print summary without writing")
    p.add_argument("-w", "--overwrite", action="store_true", default=False)
    p.add_argument("-W", "--no-overwrite", action="store_false", dest="overwrite")
    p.set_defaults(func=_run_merge)


def _add_eps(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("eps", help="Compute epsilon and gradT spectra")
    p.add_argument("files", nargs="+", help=".p or .nc files")
    p.add_argument("-o", "--output", required=True, help="Output directory (required)")
    p.add_argument("-d", "--diss-len", type=float, default=4.0, help="Dissipation window [s]")
    p.add_argument("-f", "--fft-len", type=float, default=1.0, help="FFT segment [s]")
    p.add_argument("-s", "--min-speed", type=float, default=0.2, help="Min profiling speed [m/s]")
    p.add_argument(
        "--pressure-smoothing", type=float, default=0.25, help="Pressure smoothing [s]"
    )
    p.add_argument("-t", "--temperature", default="JAC_T", help="Temperature variable name")
    p.add_argument("--speed", default="W", help="Speed variable name")
    p.add_argument(
        "--direction", choices=["down", "up", "both"], default="down", help="Profile direction"
    )
    p.add_argument("--min-profile-pressure", type=float, default=0.0, help="Min pressure [dbar]")
    p.add_argument("--peaks-height", type=float, default=25.0, help="Peak height [dbar]")
    p.add_argument("--peaks-distance", type=int, default=200, help="Peak distance [samples]")
    p.add_argument("--peaks-prominence", type=float, default=25.0, help="Peak prominence [dbar]")
    p.add_argument("--despike-passes", type=int, default=6, help="Despike passes")
    p.add_argument("-a", "--aux", help="Auxiliary CTD NetCDF file")
    p.add_argument("--aux-lat", default="lat", help="Auxiliary latitude variable")
    p.add_argument("--aux-lon", default="lon", help="Auxiliary longitude variable")
    p.add_argument("--aux-temp", default="temperature", help="Auxiliary temperature variable")
    p.add_argument("--aux-sal", default="salinity", help="Auxiliary salinity variable")
    p.add_argument("--aux-dens", default="density", help="Auxiliary density variable")
    p.add_argument("--salinity", type=float, default=35.0, help="Salinity [PSU] (default: 35)")
    p.add_argument("-n", "--n-workers", type=int, default=1, help="Parallel workers")
    p.add_argument("-w", "--overwrite", action="store_true", default=False)
    p.add_argument("-W", "--no-overwrite", action="store_false", dest="overwrite")
    p.add_argument("--goodman", action="store_true", default=False, dest="goodman",
                   help="Enable Goodman coherent noise removal (off by default to match pyturb)")
    p.add_argument("--no-goodman", action="store_false", dest="goodman",
                   help="Disable Goodman cleaning (default)")
    p.add_argument("--aoa", type=float, default=None, help="Angle of attack (not implemented)")
    p.add_argument(
        "--pitch-correction", action="store_true", default=False, help="(not implemented)"
    )
    p.set_defaults(func=_run_eps)


def _add_bin(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("bin", help="Depth-bin profile results")
    p.add_argument("files", nargs="+", help="Profile .nc files (from eps)")
    p.add_argument("-o", "--output", default="binned_profiles.nc", help="Output file")
    p.add_argument("-b", "--bin-width", type=float, default=2.0, help="Bin width [m]")
    p.add_argument("--dmin", type=float, default=0.0, help="Min depth [m]")
    p.add_argument("--dmax", type=float, default=1000.0, help="Max depth [m]")
    p.add_argument("--lat", type=float, default=45.0, help="Latitude for P→depth")
    p.add_argument("-p", "--pressure", action="store_true", help="Bin by pressure instead of depth")
    p.add_argument(
        "-v",
        "--vars",
        default="eps_1,eps_2,W,temperature,salinity,density,nu,lat,lon",
        help="Variables to bin (comma-separated)",
    )
    p.add_argument("-n", "--n-workers", type=int, default=1, help="Parallel workers")
    p.set_defaults(func=_run_bin)


# ---------------------------------------------------------------------------
# Dispatch wrappers
# ---------------------------------------------------------------------------


def _run_p2nc(args: argparse.Namespace) -> None:
    from odas_tpw.pyturb.p2nc import run_p2nc

    run_p2nc(args)


def _run_merge(args: argparse.Namespace) -> None:
    from odas_tpw.pyturb.merge import run_merge

    run_merge(args)


def _run_eps(args: argparse.Namespace) -> None:
    if args.aoa is not None:
        logger.warning("--aoa is accepted but not implemented")
    if args.pitch_correction:
        logger.warning("--pitch-correction is accepted but not implemented")

    from odas_tpw.pyturb.eps import run_eps

    run_eps(args)


def _run_bin(args: argparse.Namespace) -> None:
    from odas_tpw.pyturb.bin import run_bin

    run_bin(args)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="pyturb-cli",
        description="Jesse-compatible VMP processing CLI backed by odas_tpw",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv)"
    )

    sub = parser.add_subparsers(dest="command")
    _add_p2nc(sub)
    _add_merge(sub)
    _add_eps(sub)
    _add_bin(sub)

    args = parser.parse_args(argv)

    # Configure logging
    level = logging.WARNING
    if args.verbose >= 2:
        level = logging.DEBUG
    elif args.verbose >= 1:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
