# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""CTD time-binning with GPS and seawater properties.

Bins slow (and optionally fast) scalar channels by time, computes
seawater properties per bin, interpolates GPS, and writes per-file
CTD NetCDF.

Uses BOTH up and down portions (full file, not profile-segmented).

Reference: Code/ctd2binned.m (219 lines)
"""

from pathlib import Path

import numpy as np
import xarray as xr

from odas_tpw.perturb.gps import GPSProvider
from odas_tpw.perturb.seawater import add_seawater_properties


def _time_bin(
    t: np.ndarray,
    data: dict[str, np.ndarray],
    bin_width: float,
    method: str = "mean",
    diagnostics: bool = False,
) -> dict[str, np.ndarray]:
    """Bin data arrays by time.

    Parameters
    ----------
    t : ndarray
        Time vector [s].
    data : dict of ndarray
        Data arrays to bin, keyed by name.
    bin_width : float
        Bin width [s].
    method : str
        "mean" or "median".
    diagnostics : bool
        If True, include n_samples and *_std per bin.

    Returns
    -------
    dict with 'bin_centers' and binned data arrays (+ diagnostics if requested).
    """
    t_min, t_max = np.nanmin(t), np.nanmax(t)
    bin_edges = np.arange(t_min, t_max + bin_width, bin_width)
    if len(bin_edges) < 2:
        bin_edges = np.array([t_min, t_min + bin_width])
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    n_bins = len(bin_centers)
    bin_idx = np.digitize(t, bin_edges) - 1  # 0-based
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    agg_func = np.nanmean if method == "mean" else np.nanmedian
    result = {"bin_centers": bin_centers}

    if diagnostics:
        n_samples = np.zeros(n_bins, dtype=int)
        for i in range(n_bins):
            n_samples[i] = np.sum(bin_idx == i)
        result["n_samples"] = n_samples

    for name, arr in data.items():
        binned = np.full(n_bins, np.nan)
        for i in range(n_bins):
            mask = bin_idx == i
            if np.any(mask):
                binned[i] = agg_func(arr[mask])
        result[name] = binned
        if diagnostics:
            std_arr = np.full(n_bins, np.nan)
            for i in range(n_bins):
                mask = bin_idx == i
                if np.sum(mask) > 1:
                    std_arr[i] = np.nanstd(arr[mask])
            result[f"{name}_std"] = std_arr

    return result


def ctd_bin_file(
    pf,
    gps: GPSProvider,
    output_dir: str | Path,
    *,
    bin_width: float = 0.5,
    T_name: str = "JAC_T",
    C_name: str = "JAC_C",
    variables: list[str] | None = None,
    method: str = "mean",
    diagnostics: bool = False,
) -> Path | None:
    """Time-bin CTD channels from a PFile and write to NetCDF.

    Parameters
    ----------
    pf : PFile
        Parsed .p file.
    gps : GPSProvider
        GPS provider for lat/lon interpolation.
    output_dir : str or Path
        Output directory.
    bin_width : float
        Time bin width [s].
    T_name, C_name : str
        Temperature and conductivity channel names.
    variables : list of str, optional
        Additional channels to bin (auto-detect if None).
    method : str
        Aggregation method ("mean" or "median").
    diagnostics : bool
        Write additional diagnostic variables.

    Returns
    -------
    Path or None
        Path to the output NetCDF, or None if insufficient data.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect channels to bin
    channels_to_bin = set()
    if T_name in pf.channels:
        channels_to_bin.add(T_name)
    if C_name in pf.channels:
        channels_to_bin.add(C_name)
    if "P" in pf.channels:
        channels_to_bin.add("P")

    if variables:
        for v in variables:
            if v in pf.channels:
                channels_to_bin.add(v)
    else:
        # Auto-detect scalar slow channels
        auto_names = ["DO", "Chlorophyll", "Turbidity", "Fluor"]
        for name in auto_names:
            if name in pf.channels:
                channels_to_bin.add(name)

    if not channels_to_bin:
        return None

    # Separate slow and fast channels
    slow_data = {}
    fast_data = {}
    for ch in channels_to_bin:
        if pf.is_fast(ch):
            fast_data[ch] = pf.channels[ch]
        else:
            slow_data[ch] = pf.channels[ch]

    # Bin slow channels by time
    binned = {}
    bin_centers = None
    if slow_data:
        result = _time_bin(pf.t_slow, slow_data, bin_width, method, diagnostics)
        bin_centers = result.pop("bin_centers")
        binned.update(result)

    if fast_data:
        result = _time_bin(pf.t_fast, fast_data, bin_width, method, diagnostics)
        if bin_centers is None:
            bin_centers = result.pop("bin_centers")
        else:
            result.pop("bin_centers")
        binned.update(result)

    if bin_centers is None or len(bin_centers) == 0:
        return None

    # GPS interpolation
    lat = gps.lat(bin_centers)
    lon = gps.lon(bin_centers)
    binned["lat"] = lat
    binned["lon"] = lon

    # Seawater properties (if T, C, P available)
    if T_name in binned and C_name in binned and "P" in binned:
        sw = add_seawater_properties(
            binned[T_name], binned[C_name], binned["P"], lat, lon
        )
        binned.update(sw)

    # Build xarray Dataset
    ds = xr.Dataset(
        {name: (["time"], arr) for name, arr in binned.items()},
        coords={"time": bin_centers},
    )
    ds.attrs["bin_width"] = bin_width
    ds.attrs["method"] = method
    ds.attrs["source_file"] = pf.filepath.name
    ds.attrs["Conventions"] = "CF-1.8"

    out_path = output_dir / f"{pf.filepath.stem}.nc"
    ds.to_netcdf(out_path)
    return out_path
