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
    bin_edges: np.ndarray | None = None,
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
    bin_edges : ndarray, optional
        Pre-computed bin edges. If provided, *bin_width* is ignored and the
        same edges are used regardless of *t*'s exact range — required when
        binning slow and fast channels in a single output (their last
        samples differ by up to ``1/fs_slow - 1/fs_fast`` seconds, which
        can flip one extra bin into existence and produce mismatched
        time dimensions downstream).

    Returns
    -------
    dict with 'bin_centers' and binned data arrays (+ diagnostics if requested).
    """
    if bin_edges is None:
        t_min, t_max = np.nanmin(t), np.nanmax(t)
        bin_edges = np.arange(t_min, t_max + bin_width, bin_width)
        if len(bin_edges) < 2:
            bin_edges = np.array([t_min, t_min + bin_width])
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    n_bins = len(bin_centers)
    bin_idx = np.digitize(t, bin_edges) - 1  # 0-based
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    result = {"bin_centers": bin_centers}

    if diagnostics:
        result["n_samples"] = np.bincount(bin_idx, minlength=n_bins).astype(int)

    if method == "median":
        # Sort once by bin index; each bin is then a contiguous slice. Avoids
        # the O(n_bins * n_samples) mask broadcasts of a naive ``bin_idx == i``
        # loop while still allowing per-bin nanmedian.
        order = np.argsort(bin_idx, kind="stable")
        sorted_idx = bin_idx[order]
        splits = np.searchsorted(sorted_idx, np.arange(n_bins + 1))
        for name, arr in data.items():
            sorted_arr = arr[order]
            binned = np.full(n_bins, np.nan)
            for i in range(n_bins):
                sl = sorted_arr[splits[i] : splits[i + 1]]
                if sl.size > 0:
                    binned[i] = np.nanmedian(sl)
            result[name] = binned
            if diagnostics:
                std_arr = np.full(n_bins, np.nan)
                for i in range(n_bins):
                    sl = sorted_arr[splits[i] : splits[i + 1]]
                    if sl.size > 1:
                        std_arr[i] = np.nanstd(sl)
                result[f"{name}_std"] = std_arr
        return result

    # Mean path — one O(N) bincount per channel, no per-bin Python loop.
    for name, arr in data.items():
        finite = np.isfinite(arr)
        idx_f = bin_idx[finite]
        arr_f = arr[finite]
        counts = np.bincount(idx_f, minlength=n_bins)
        sums = np.bincount(idx_f, weights=arr_f, minlength=n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            binned = np.where(counts > 0, sums / counts, np.nan)
        result[name] = binned
        if diagnostics:
            sums_sq = np.bincount(idx_f, weights=arr_f * arr_f, minlength=n_bins)
            with np.errstate(invalid="ignore", divide="ignore"):
                # Population std (ddof=0) to match np.nanstd default.
                mean_per_bin = np.where(counts > 0, sums / counts, 0.0)
                var = sums_sq / np.maximum(counts, 1) - mean_per_bin * mean_per_bin
                # Numerical safety: clip tiny negatives from cancellation.
                var = np.maximum(var, 0.0)
                std_arr = np.where(counts > 1, np.sqrt(var), np.nan)
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

    # Compute a single set of bin edges spanning both time vectors so slow
    # and fast channels share one time dimension in the output dataset.
    t_lo = np.nanmin(pf.t_slow if slow_data else pf.t_fast)
    t_hi = max(
        np.nanmax(pf.t_slow) if slow_data else -np.inf,
        np.nanmax(pf.t_fast) if fast_data else -np.inf,
    )
    bin_edges = np.arange(t_lo, t_hi + bin_width, bin_width)
    if len(bin_edges) < 2:
        bin_edges = np.array([t_lo, t_lo + bin_width])

    binned: dict[str, np.ndarray] = {}
    bin_centers: np.ndarray | None = None
    if slow_data:
        result = _time_bin(
            pf.t_slow, slow_data, bin_width, method, diagnostics, bin_edges=bin_edges
        )
        bin_centers = result.pop("bin_centers")
        binned.update(result)

    if fast_data:
        result = _time_bin(
            pf.t_fast, fast_data, bin_width, method, diagnostics, bin_edges=bin_edges
        )
        if bin_centers is None:
            bin_centers = result.pop("bin_centers")
        else:
            result.pop("bin_centers")
        binned.update(result)

    if bin_centers is None or len(bin_centers) == 0:
        return None

    # GPS interpolation — query in epoch seconds so the GPS provider's
    # absolute time axis lines up with bin_centers (which live on the
    # file-relative pf.t_slow timeline).
    epoch_offset = pf.start_time.timestamp() if hasattr(pf, "start_time") else 0.0
    lat = gps.lat(np.asarray(bin_centers, dtype=np.float64) + epoch_offset)
    lon = gps.lon(np.asarray(bin_centers, dtype=np.float64) + epoch_offset)
    binned["lat"] = lat
    binned["lon"] = lon

    # Seawater properties (if T, C, P available)
    if T_name in binned and C_name in binned and "P" in binned:
        sw = add_seawater_properties(binned[T_name], binned[C_name], binned["P"], lat, lon)
        binned.update(sw)

    # Promote bin_centers from file-relative seconds to epoch seconds so
    # concatenated CTD combos can be correctly ordered across files.  Add
    # CF time attrs so xarray's encoder serialises them as datetimes.
    time_epoch = np.asarray(bin_centers, dtype=np.float64) + epoch_offset

    # Build xarray Dataset
    ds = xr.Dataset(
        {name: (["time"], arr) for name, arr in binned.items()},
        coords={"time": time_epoch},
    )
    ds["time"].attrs["units"] = "seconds since 1970-01-01"
    ds["time"].attrs["calendar"] = "standard"
    ds["time"].attrs["standard_name"] = "time"
    ds["time"].attrs["long_name"] = "time bin centre"
    ds["time"].attrs["axis"] = "T"
    ds["time"].attrs["units_metadata"] = "leap_seconds: utc"
    ds.attrs["bin_width"] = bin_width
    ds.attrs["method"] = method
    ds.attrs["source_file"] = pf.filepath.name
    ds.attrs["Conventions"] = "CF-1.13, ACDD-1.3"

    from odas_tpw.perturb._nc_writer import write_dataset

    out_path = output_dir / f"{pf.filepath.stem}.nc"
    write_dataset(ds, out_path)
    return out_path
