"""pyturb-cli p2nc: convert .p files to NetCDF."""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from odas_tpw.pyturb._compat import check_overwrite

logger = logging.getLogger(__name__)


def _convert_one(p_path: Path, nc_path: Path, complevel: int) -> tuple[str, float]:
    """Worker: convert a single .p file."""
    from odas_tpw.rsi.convert import p_to_L1

    p_to_L1(p_path, nc_path, complevel=complevel)
    size_mb = nc_path.stat().st_size / 1e6
    return nc_path.name, size_mb


def run_p2nc(args: argparse.Namespace) -> None:
    """Execute the p2nc subcommand."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    complevel = args.compression_level if args.compress else 0

    # Collect and filter input files
    paths: list[Path] = []
    for pattern in args.files:
        p = Path(pattern)
        if p.is_file():
            paths.append(p)
        else:
            paths.extend(sorted(Path(".").glob(pattern)))

    work: list[tuple[Path, Path]] = []
    for p_path in paths:
        if p_path.stat().st_size < args.min_file_size:
            logger.info(f"Skipping {p_path.name}: too small ({p_path.stat().st_size} bytes)")
            continue
        nc_path = output_dir / p_path.with_suffix(".nc").name
        if not check_overwrite(nc_path, args.overwrite):
            logger.info(f"Skipping {nc_path.name}: exists (use -w to overwrite)")
            continue
        work.append((p_path, nc_path))

    if not work:
        logger.warning("No files to convert")
        return

    n_workers = args.n_workers
    if n_workers <= 1:
        for p_path, nc_path in work:
            try:
                name, size_mb = _convert_one(p_path, nc_path, complevel)
                print(f"{p_path.name} -> {name}  {size_mb:.1f} MB")
            except (OSError, ValueError, RuntimeError) as e:
                logger.error(f"{p_path.name}: {e}")
    else:
        print(f"Converting {len(work)} files with {n_workers} workers")
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_convert_one, p_path, nc_path, complevel): (p_path, nc_path)
                for p_path, nc_path in work
            }
            for future in as_completed(futures):
                p_path, nc_path = futures[future]
                try:
                    name, size_mb = future.result()
                    print(f"{p_path.name} -> {name}  {size_mb:.1f} MB")
                except (OSError, ValueError, RuntimeError) as e:
                    logger.error(f"{p_path.name}: {e}")
