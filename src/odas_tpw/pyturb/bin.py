"""pyturb-cli bin: depth-bin profile results."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def run_bin(args: argparse.Namespace) -> None:
    """Execute the bin subcommand."""
    output_path = Path(args.output)

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

    var_names = [v.strip() for v in args.vars.split(",") if v.strip()]

    from odas_tpw.rsi.binning import bin_by_depth
    from odas_tpw.rsi.combine import combine_profiles

    binned_list: list[xr.Dataset] = []
    metadata_list: list[dict] = []

    log_mean_vars = {"eps_1", "eps_2", "eps_final", "epsilon", "chi", "chi_final", "epsi_final"}

    for filepath in sorted(paths):
        try:
            ds = xr.open_dataset(filepath)
        except Exception as e:
            logger.error(f"{filepath.name}: {e}")
            continue

        # Get pressure
        if "pressure" in ds:
            pres = ds["pressure"].values
        elif "P_mean" in ds:
            pres = ds["P_mean"].values
        else:
            logger.warning(f"{filepath.name}: no pressure variable, skipping")
            ds.close()
            continue

        # Convert pressure to depth if not using --pressure mode
        if not args.pressure:
            try:
                import gsw

                depth = -gsw.z_from_p(pres, args.lat)
            except ImportError:
                logger.warning("gsw not available, binning by pressure instead")
                depth = pres
        else:
            depth = pres

        # Collect variables
        values: dict[str, np.ndarray] = {}
        for vname in var_names:
            if vname in ds:
                values[vname] = ds[vname].values

        if not values:
            logger.warning(f"{filepath.name}: no matching variables, skipping")
            ds.close()
            continue

        binned = bin_by_depth(
            depth,
            values,
            bin_size=args.bin_width,
            pres_range=(args.dmin, args.dmax),
            log_mean_vars=log_mean_vars & set(var_names),
        )
        binned_list.append(binned)
        metadata_list.append({"source": filepath.name})
        ds.close()

    if not binned_list:
        logger.error("No profiles binned")
        return

    combined = combine_profiles(binned_list, profile_metadata=metadata_list)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_netcdf(output_path)

    n_prof = combined.sizes.get("profile", 0)
    n_bins = combined.sizes.get("depth_bin", 0)
    print(f"Binned {n_prof} profiles x {n_bins} depth bins -> {output_path.name}")
