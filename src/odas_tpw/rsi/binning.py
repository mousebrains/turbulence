"""Depth binning for dissipation and chi estimates (L5)."""

from __future__ import annotations

import numpy as np
import xarray as xr


def bin_by_depth(
    pres: np.ndarray,
    values: dict[str, np.ndarray],
    bin_size: float = 1.0,
    pres_range: tuple[float, float] | None = None,
    log_mean_vars: set[str] | None = None,
) -> xr.Dataset:
    """Bin estimates by depth (pressure).

    Parameters
    ----------
    pres : ndarray, shape (N,)
        Pressure per estimate [dbar].
    values : dict of str -> ndarray
        Variables to bin. Each value has shape (N,) or (M, N) where M is
        the number of probes. 2-D arrays are reduced across probes first
        (geometric mean for log_mean_vars, arithmetic mean otherwise).
    bin_size : float
        Bin size [dbar]. Default: 1.0.
    pres_range : tuple of (min, max), optional
        Pressure range for binning. Default: data range.
    log_mean_vars : set of str, optional
        Variables to average using geometric mean (log-normal).
        Default: {"epsilon", "chi", "chi_final", "epsi_final"}.

    Returns
    -------
    xr.Dataset with dimension ``depth_bin``.
    """
    if log_mean_vars is None:
        log_mean_vars = {"epsilon", "chi", "chi_final", "epsi_final"}

    pres = np.asarray(pres, dtype=np.float64)
    valid = np.isfinite(pres)
    if not valid.any():
        return xr.Dataset()

    if pres_range is None:
        p_min = np.floor(np.nanmin(pres) / bin_size) * bin_size
        p_max = np.ceil(np.nanmax(pres) / bin_size) * bin_size
    else:
        p_min, p_max = pres_range

    bin_edges = np.arange(p_min, p_max + bin_size, bin_size)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    n_bins = len(bin_centers)

    # Assign samples to bins
    bin_idx = np.digitize(pres, bin_edges) - 1  # 0-indexed
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    data_vars = {}
    for name, arr in values.items():
        arr = np.asarray(arr, dtype=np.float64)

        # Reduce 2-D arrays (probes x time) → 1-D
        if arr.ndim == 2:
            if name in log_mean_vars:
                with np.errstate(divide="ignore"):
                    arr = np.exp(np.nanmean(np.log(np.where(arr > 0, arr, np.nan)), axis=0))
            else:
                arr = np.nanmean(arr, axis=0)

        # Vectorize per-bin reduction with bincount.  Log-mean uses
        # log(x) accumulators with strictly-positive values; arithmetic
        # mean uses values directly.  Both paths drop NaN (and ≤0 for
        # log-mean) up front, then accumulate sums and counts in one
        # linear pass.
        if name in log_mean_vars:
            valid = np.isfinite(arr) & (arr > 0)
            log_vals = np.log(arr, where=valid, out=np.full_like(arr, np.nan))
            idx_v = bin_idx[valid]
            log_v = log_vals[valid]
            counts_v = np.bincount(idx_v, minlength=n_bins)
            sums_v = np.bincount(idx_v, weights=log_v, minlength=n_bins)
            with np.errstate(invalid="ignore", divide="ignore"):
                binned = np.where(
                    counts_v > 0, np.exp(sums_v / np.maximum(counts_v, 1)), np.nan
                )
        else:
            valid = np.isfinite(arr)
            idx_v = bin_idx[valid]
            vals_v = arr[valid]
            counts_v = np.bincount(idx_v, minlength=n_bins)
            sums_v = np.bincount(idx_v, weights=vals_v, minlength=n_bins)
            with np.errstate(invalid="ignore", divide="ignore"):
                binned = np.where(counts_v > 0, sums_v / np.maximum(counts_v, 1), np.nan)

        data_vars[name] = (["depth_bin"], binned)

    ds = xr.Dataset(
        data_vars,
        coords={"depth_bin": bin_centers},
    )
    ds.coords["depth_bin"].attrs.update(
        {
            "units": "dbar",
            "long_name": "depth bin center pressure",
            "standard_name": "sea_water_pressure",
            "positive": "down",
        }
    )
    ds.attrs["bin_size"] = bin_size

    return ds
