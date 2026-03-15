"""Combine binned profiles into a single dataset (L6)."""

from __future__ import annotations

import numpy as np
import xarray as xr


def combine_profiles(
    binned_datasets: list[xr.Dataset],
    profile_metadata: list[dict] | None = None,
) -> xr.Dataset:
    """Combine binned profiles into a multi-profile dataset.

    Aligns all profiles to a common depth grid (union of bin centers).
    Output dimensions: ``(profile, depth_bin)``. NaN where a profile
    doesn't cover a depth.

    Parameters
    ----------
    binned_datasets : list of xr.Dataset
        Per-profile binned datasets (from ``bin_by_depth``).
    profile_metadata : list of dict, optional
        Metadata for each profile (start_time, file, etc.).

    Returns
    -------
    xr.Dataset with dimensions ``(profile, depth_bin)``.
    """
    if not binned_datasets:
        return xr.Dataset()

    # Build common depth grid (union of all bin centers)
    all_depths = set()
    for ds in binned_datasets:
        if "depth_bin" in ds.coords:
            all_depths.update(ds.coords["depth_bin"].values.tolist())

    if not all_depths:
        return xr.Dataset()

    common_depths = np.sort(np.array(list(all_depths)))
    n_profiles = len(binned_datasets)
    n_depths = len(common_depths)

    # Collect all variable names
    var_names: set[str] = set()
    for ds in binned_datasets:
        var_names.update(ds.data_vars)

    data_vars = {}
    for var in sorted(var_names):
        combined = np.full((n_profiles, n_depths), np.nan)
        for i, ds in enumerate(binned_datasets):
            if var not in ds.data_vars:
                continue
            ds_depths = ds.coords["depth_bin"].values
            ds_vals = ds[var].values
            for j, d in enumerate(ds_depths):
                idx = np.searchsorted(common_depths, d)
                if idx < n_depths and np.isclose(common_depths[idx], d):
                    combined[i, idx] = ds_vals[j]
        data_vars[var] = (["profile", "depth_bin"], combined)

    coords = {
        "profile": np.arange(n_profiles),
        "depth_bin": common_depths,
    }

    result = xr.Dataset(data_vars, coords=coords)
    result.coords["depth_bin"].attrs.update(
        {
            "units": "dbar",
            "long_name": "depth bin center pressure",
            "standard_name": "sea_water_pressure",
            "positive": "down",
        }
    )
    result.coords["profile"].attrs["long_name"] = "profile number"

    # Attach metadata
    if profile_metadata:
        for i, meta in enumerate(profile_metadata):
            for key, val in meta.items():
                attr_name = f"profile_{i}_{key}"
                result.attrs[attr_name] = str(val)

    return result
