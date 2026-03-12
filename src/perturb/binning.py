# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Depth/time binning engine for profiles, dissipation, and chi.

Reference: Code/bin_by_real.m, Code/profile2binned.m, Code/diss2binned.m,
           Code/chi2binned.m
"""

from pathlib import Path

import numpy as np
import xarray as xr


def _bin_array(
    values: np.ndarray,
    coords: np.ndarray,
    bin_edges: np.ndarray,
    agg_func,
) -> tuple[np.ndarray, np.ndarray]:
    """Core binning: assign values to bins and aggregate.

    Parameters
    ----------
    values : ndarray, 1D
    coords : ndarray, 1D (same length as values)
    bin_edges : ndarray, 1D (n_bins + 1)
    agg_func : callable (e.g. np.nanmean, np.nanmedian)

    Returns
    -------
    binned : ndarray, shape (n_bins,)
    counts : ndarray, shape (n_bins,) — int
    """
    n_bins = len(bin_edges) - 1
    binned = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    idx = np.digitize(coords, bin_edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    for i in range(n_bins):
        mask = idx == i
        n = np.sum(mask)
        counts[i] = n
        if n > 0:
            binned[i] = agg_func(values[mask])

    return binned, counts


def _get_agg_func(aggregation: str):
    if aggregation == "median":
        return np.nanmedian
    return np.nanmean


def bin_by_depth(
    profile_files: list[Path],
    bin_width: float = 1.0,
    aggregation: str = "mean",
    diagnostics: bool = False,
) -> xr.Dataset:
    """Bin per-profile NetCDFs by depth into 2D (bin x profile).

    Parameters
    ----------
    profile_files : list of Path
        Paths to per-profile NetCDF files.
    bin_width : float
        Depth bin width [m].
    aggregation : str
        "mean" or "median".
    diagnostics : bool
        Include n_samples and *_std per bin.

    Returns
    -------
    xr.Dataset with dims (bin, profile).
    """
    agg = _get_agg_func(aggregation)
    datasets = [xr.open_dataset(f) for f in profile_files]

    # Determine global bin edges from all profiles
    all_depths = []
    for ds in datasets:
        if "depth" in ds:
            d = ds["depth"].values
            all_depths.extend(d[np.isfinite(d)])
        elif "P" in ds:
            d = ds["P"].values
            all_depths.extend(d[np.isfinite(d)])
    if not all_depths:
        return xr.Dataset()

    d_min = np.floor(np.min(all_depths) / bin_width) * bin_width
    d_max = np.ceil(np.max(all_depths) / bin_width) * bin_width
    bin_edges = np.arange(d_min, d_max + bin_width, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    n_bins = len(bin_centers)
    n_profiles = len(datasets)

    result_vars = {}
    # Determine depth coordinate for each profile
    for pi, ds in enumerate(datasets):
        if "depth" in ds:
            depth_var = "depth"
        elif "P" in ds:
            depth_var = "P"
        else:
            continue

        for vname in ds.data_vars:
            if vname == depth_var:
                continue
            arr = ds[vname].values
            depth = ds[depth_var].values
            if arr.ndim != 1 or len(arr) != len(depth):
                continue

            if vname not in result_vars:
                result_vars[vname] = np.full((n_bins, n_profiles), np.nan)
                if diagnostics:
                    result_vars[f"{vname}_std"] = np.full((n_bins, n_profiles), np.nan)

            binned, _counts = _bin_array(arr, depth, bin_edges, agg)
            result_vars[vname][:, pi] = binned

            if diagnostics:
                # Compute std per bin
                std_arr = np.full(n_bins, np.nan)
                idx = np.clip(np.digitize(depth, bin_edges) - 1, 0, n_bins - 1)
                for bi in range(n_bins):
                    mask = idx == bi
                    if np.sum(mask) > 1:
                        std_arr[bi] = np.nanstd(arr[mask])
                result_vars[f"{vname}_std"][:, pi] = std_arr

    for ds in datasets:
        ds.close()

    if diagnostics:
        result_vars["n_samples"] = np.zeros((n_bins, n_profiles), dtype=float)

    data_vars = {}
    for vname, arr in result_vars.items():
        data_vars[vname] = (["bin", "profile"], arr)

    return xr.Dataset(
        data_vars,
        coords={
            "bin": bin_centers,
            "profile": np.arange(n_profiles),
        },
    )


def bin_by_time(
    profile_files: list[Path],
    bin_width: float = 1.0,
    aggregation: str = "mean",
    diagnostics: bool = False,
) -> xr.Dataset:
    """Bin per-profile NetCDFs by time.

    Returns a concatenated dataset along a time dimension.
    """
    agg = _get_agg_func(aggregation)
    all_binned = []

    for f in profile_files:
        ds = xr.open_dataset(f)
        # Use the time coordinate
        for time_name in ("t_slow", "t_fast", "time"):
            if time_name in ds:
                t = ds[time_name].values
                break
        else:
            ds.close()
            continue

        t_min, t_max = np.nanmin(t), np.nanmax(t)
        bin_edges = np.arange(t_min, t_max + bin_width, bin_width)
        if len(bin_edges) < 2:
            bin_edges = np.array([t_min, t_min + bin_width])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        binned_data = {"time": bin_centers}
        for vname in ds.data_vars:
            arr = ds[vname].values
            if arr.ndim != 1 or len(arr) != len(t):
                continue
            b, _ = _bin_array(arr, t, bin_edges, agg)
            binned_data[vname] = b

        ds.close()
        if len(binned_data) > 1:
            bds = xr.Dataset(
                {k: (["time"], v) for k, v in binned_data.items() if k != "time"},
                coords={"time": binned_data["time"]},
            )
            all_binned.append(bds)

    if not all_binned:
        return xr.Dataset()
    return xr.concat(all_binned, dim="time")


def bin_diss(
    diss_files: list[Path],
    bin_width: float = 1.0,
    aggregation: str = "mean",
    method: str = "depth",
    diagnostics: bool = False,
) -> xr.Dataset:
    """Bin dissipation estimates by depth or time.

    Handles per-probe (e_1, e_2) and combined (epsilonMean) variables.
    """
    if method == "time":
        return bin_by_time(diss_files, bin_width, aggregation, diagnostics)
    return bin_by_depth(diss_files, bin_width, aggregation, diagnostics)


def bin_chi(
    chi_files: list[Path],
    bin_width: float = 1.0,
    aggregation: str = "mean",
    method: str = "depth",
    diagnostics: bool = False,
) -> xr.Dataset:
    """Bin chi estimates by depth or time."""
    if method == "time":
        return bin_by_time(chi_files, bin_width, aggregation, diagnostics)
    return bin_by_depth(chi_files, bin_width, aggregation, diagnostics)
