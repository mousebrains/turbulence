"""pyturb-cli merge: merge multiple NetCDF files along time."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import xarray as xr

from odas_tpw.pyturb._compat import check_overwrite

logger = logging.getLogger(__name__)


def run_merge(args: argparse.Namespace) -> None:
    """Execute the merge subcommand."""
    output_path = Path(args.output)

    if not args.dry_run and not check_overwrite(output_path, args.overwrite):
        logger.error(f"{output_path}: exists (use -w to overwrite)")
        return

    # Collect input files
    paths: list[Path] = []
    for pattern in args.files:
        p = Path(pattern)
        if p.is_file():
            paths.append(p)
        else:
            paths.extend(sorted(Path(".").glob(pattern)))

    if not paths:
        logger.error("No input files found")
        return

    datasets: list[xr.Dataset] = []
    for p in paths:
        try:
            ds = _open_nc(p)
            datasets.append(ds)
        except Exception as e:
            logger.error(f"{p.name}: {e}")

    if not datasets:
        logger.error("No valid datasets loaded")
        return

    # Sort by first time value
    def _first_time(ds: xr.Dataset) -> float:
        for dim in ("TIME", "time", "Time"):
            if dim in ds.dims or dim in ds.coords:
                vals = ds[dim].values
                if len(vals) > 0:
                    return float(vals[0])
        return 0.0

    datasets.sort(key=_first_time)

    if args.dry_run:
        print(f"Would merge {len(datasets)} files into {output_path}")
        total_time = sum(ds.sizes.get("TIME", ds.sizes.get("time", 0)) for ds in datasets)
        print(f"Total time samples: {total_time}")
        for i, ds in enumerate(datasets):
            t0 = _first_time(ds)
            n = ds.sizes.get("TIME", ds.sizes.get("time", 0))
            print(f"  [{i}] t0={t0:.6f}  n_time={n}")
        for ds in datasets:
            ds.close()
        return

    # Find the common time dimension name
    time_dim = "TIME"
    for dim_name in ("TIME", "time"):
        if dim_name in datasets[0].dims:
            time_dim = dim_name
            break

    merged = xr.concat(datasets, dim=time_dim)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_netcdf(output_path)

    n_time = merged.sizes.get(time_dim, 0)
    print(f"Merged {len(datasets)} files -> {output_path.name}  ({n_time} time samples)")

    merged.close()
    for ds in datasets:
        ds.close()


def _open_nc(path: Path) -> xr.Dataset:
    """Open a NetCDF file, preferring the L1_converted group if present."""
    import netCDF4

    # Check if there's an L1_converted group
    with netCDF4.Dataset(str(path), "r") as nc:
        has_l1 = "L1_converted" in nc.groups

    if has_l1:
        return xr.open_dataset(path, group="L1_converted")
    return xr.open_dataset(path)
