# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""
NetCDF conversion for Rockland .p files.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rsi_python.p_file import PFile

import numpy as np


def p_to_netcdf(
    p_filepath: str | Path,
    nc_filepath: str | Path | None = None,
) -> tuple[PFile, Path]:
    """Convert a single .p file to NetCDF4.

    Parameters
    ----------
    p_filepath : str or Path
        Path to the .p file.
    nc_filepath : str or Path, optional
        Output path. Defaults to same name with .nc extension.

    Returns
    -------
    tuple of (PFile, Path)
        The parsed data object and the output path.
    """
    import netCDF4 as nc

    from rsi_python.p_file import PFile

    pf = PFile(p_filepath)
    if nc_filepath is None:
        nc_filepath = pf.filepath.with_suffix(".nc")
    nc_filepath = Path(nc_filepath)

    ds = nc.Dataset(str(nc_filepath), "w", format="NETCDF4")

    ds.title = f"VMP data from {pf.filepath.name}"
    ds.instrument_model = pf.config["instrument_info"].get("model", "")
    ds.instrument_sn = pf.config["instrument_info"].get("sn", "")
    ds.instrument_vehicle = pf.config["instrument_info"].get("vehicle", "")
    ds.operator = pf.config["cruise_info"].get("operator", "")
    ds.project = pf.config["cruise_info"].get("project", "")
    ds.start_time = pf.start_time.isoformat()
    ds.fs_fast = pf.fs_fast
    ds.fs_slow = pf.fs_slow
    ds.source_file = pf.filepath.name
    ds.history = f"Converted from .p file on {datetime.now(timezone.utc).isoformat()}"

    ds.createDimension("time_fast", len(pf.t_fast))
    ds.createDimension("time_slow", len(pf.t_slow))

    t_fast_var = ds.createVariable("t_fast", "f8", ("time_fast",), zlib=True)
    t_fast_var[:] = pf.t_fast
    t_fast_var.units = f"seconds since {pf.start_time.isoformat()}"
    t_fast_var.long_name = "time (fast channels)"

    t_slow_var = ds.createVariable("t_slow", "f8", ("time_slow",), zlib=True)
    t_slow_var[:] = pf.t_slow
    t_slow_var.units = f"seconds since {pf.start_time.isoformat()}"
    t_slow_var.long_name = "time (slow channels)"

    for ch_name, data in pf.channels.items():
        info = pf.channel_info[ch_name]
        dim = "time_fast" if pf.is_fast(ch_name) else "time_slow"
        var_name = ch_name.replace(" ", "_")
        var = ds.createVariable(var_name, "f4", (dim,), zlib=True)
        var[:] = data.astype(np.float32)
        var.units = info["units"]
        var.sensor_type = info["type"]
        var.long_name = ch_name

    ds.configuration_string = pf.config_str
    ds.close()
    return pf, nc_filepath


def _convert_one(args):
    """Worker function for parallel conversion. Must be top-level for pickling."""
    p_path, nc_path = args
    p_to_netcdf(p_path, nc_path)
    size_mb = nc_path.stat().st_size / 1e6
    return p_path.name, nc_path.name, size_mb


def convert_all(p_files: list[Path], output_dir: Path | None = None, jobs: int = 1) -> None:
    """Convert multiple .p files to NetCDF.

    Parameters
    ----------
    p_files : list of Path
        .p files to convert.
    output_dir : Path, optional
        Output directory. If None, each .nc goes next to its .p file.
    jobs : int
        Number of parallel workers. 0 means use all CPU cores.
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    work = []
    for pf_path in p_files:
        pf_path = Path(pf_path)
        if output_dir is not None:
            nc_path = output_dir / pf_path.with_suffix(".nc").name
        else:
            nc_path = pf_path.with_suffix(".nc")
        work.append((pf_path, nc_path))

    if jobs == 0:
        jobs = os.cpu_count() or 1

    if jobs == 1:
        for p_path, nc_path in work:
            print(f"{p_path.name} -> {nc_path.name} ... ", end="", flush=True)
            try:
                name, _, size_mb = _convert_one((p_path, nc_path))
                print(f"{size_mb:.1f} MB")
            except Exception as e:
                print(f"ERROR: {e}")
    else:
        print(f"Converting {len(work)} files with {jobs} workers")
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_convert_one, w): w for w in work}
            for future in as_completed(futures):
                p_path, nc_path = futures[future]
                try:
                    _, _, size_mb = future.result()
                    print(f"  {p_path.name} -> {nc_path.name}  {size_mb:.1f} MB")
                except Exception as e:
                    print(f"  {p_path.name}  ERROR: {e}")
