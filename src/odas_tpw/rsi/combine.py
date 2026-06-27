"""Combine binned profiles into a single dataset (L6)."""

from __future__ import annotations

import numpy as np
import xarray as xr

# Sub-bin tolerance (dbar) for collapsing ULP-distinct-but-equal bin centers.
# Far below any real bin spacing (>= ~0.1 dbar) yet far above float ULP noise
# (~1e-11 dbar at 100 dbar), so genuine bins stay distinct while accumulation
# error in per-profile np.arange grids is merged.
_DEPTH_ATOL = 1.0e-6


def _depth_key(d: float) -> int:
    """Quantize a depth-bin center onto the shared canonical grid."""
    return round(float(d) / _DEPTH_ATOL)


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

    # Build common depth grid (union of all bin centers). Each profile builds
    # its own np.arange center grid, so when bin_size is not binary-exact
    # (e.g. 0.1, 0.2) two profiles produce mathematically-equal centers that
    # differ by an ULP (0.35 vs 0.35000000000000003). Key the union on a
    # quantized center (sub-bin tolerance) so those collapse to one column
    # instead of splitting one physical depth into two half-NaN columns.
    canon: dict[int, float] = {}
    for ds in binned_datasets:
        if "depth_bin" in ds.coords:
            for d in ds.coords["depth_bin"].values.tolist():
                canon.setdefault(_depth_key(d), d)

    if not canon:
        return xr.Dataset()

    items = sorted(canon.items(), key=lambda kv: kv[1])
    common_depths = np.array([v for _, v in items])
    key_to_idx = {k: i for i, (k, _) in enumerate(items)}
    n_profiles = len(binned_datasets)
    n_depths = len(common_depths)

    # Collect all variable names
    var_names: set[str] = set()
    for ds in binned_datasets:
        var_names.update(str(v) for v in ds.data_vars)

    data_vars = {}
    for var in sorted(var_names):
        combined = np.full((n_profiles, n_depths), np.nan)
        for i, ds in enumerate(binned_datasets):
            if var not in ds.data_vars:
                continue
            ds_depths = ds.coords["depth_bin"].values
            ds_vals = ds[var].values
            for j, d in enumerate(ds_depths):
                idx = key_to_idx.get(_depth_key(d))
                if idx is not None:
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

    # CF/ACDD global attrs: the L6 headline product previously advertised no
    # convention conformance (empty .attrs), unlike every other writer, so a
    # CF/ACDD consumer could not validate it. L6 is a set of profiles on a
    # shared depth grid -> CF §9 featureType "profile" (audit: missing
    # Conventions/featureType on combined product).
    result.attrs["Conventions"] = "CF-1.13, ACDD-1.3"
    result.attrs["featureType"] = "profile"

    # Attach metadata
    if profile_metadata:
        for i, meta in enumerate(profile_metadata):
            for key, val in meta.items():
                attr_name = f"profile_{i}_{key}"
                result.attrs[attr_name] = str(val)

    return result
