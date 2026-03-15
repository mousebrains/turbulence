# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Multi-probe epsilon combining (mk_epsilon_mean).

Combines dissipation estimates from multiple shear probes using
the expected variance of ln(epsilon), iteratively removing probes
outside the 95% confidence interval, then computing the geometric mean.

Reference: Code/calc_diss_shear.m lines 238-299
"""

import warnings

import numpy as np
import xarray as xr


def mk_epsilon_mean(
    ds: xr.Dataset,
    epsilon_minimum: float = 1e-13,
) -> xr.Dataset:
    """Combine per-probe epsilon into a mean estimate with CI filtering.

    Port of Matlab ``mk_epsilon_mean``:
      1. Floor small values to NaN
      2. Kolmogorov length: L_K = (nu^3 / epsilon)^(1/4)
      3. Physical length: L = speed * diss_length / fs
      4. Normalized: L_hat = L / L_K
      5. var_ln_epsilon = 5.5 / (1 + (L_hat/4)^(7/9))
      6. 95% CI range: 1.96 * sqrt(2) * mean(sigma_ln_epsilon)
      7. Iteratively remove probes outside CI
      8. Geometric mean of surviving probes

    Parameters
    ----------
    ds : xr.Dataset
        Dataset from ``get_diss``.  Must contain:
        - ``e_1``, ``e_2``, ... (per-probe epsilon, 1D along 'time')
        - ``speed`` (1D along 'time')
        - ``nu`` (1D along 'time')
        - ``diss_length`` (scalar attribute or variable)
        - ``fs_fast`` (scalar attribute)
    epsilon_minimum : float
        Floor: values <= this are set to NaN.

    Returns
    -------
    xr.Dataset
        Input dataset with ``epsilonMean`` and ``epsilonLnSigma`` added.
    """
    ds = ds.copy()

    # Find epsilon probe variables — either separate e_1/e_2/... or 2D epsilon(probe, time)
    probe_names = sorted(str(k) for k in ds.data_vars if str(k).startswith("e_"))

    if not probe_names and "epsilon" in ds and "probe" in ds.dims:
        # Split 2D epsilon(probe, time) into per-probe variables
        for i in range(ds.sizes["probe"]):
            name = f"e_{i + 1}"
            ds[name] = ds["epsilon"].isel(probe=i)
            probe_names.append(name)

    if not probe_names:
        warnings.warn("No per-probe epsilon variables (e_1, e_2, ...) found")
        return ds

    n_time = ds.sizes.get("time", 0)
    if n_time == 0:
        return ds

    # Stack epsilon into 2D array [time x probes]
    epsilon = np.column_stack([ds[name].values.copy() for name in probe_names])

    # Floor small values
    epsilon[epsilon <= epsilon_minimum] = np.nan

    # Get parameters
    speed = ds["speed"].values if "speed" in ds else np.ones(n_time)
    nu = ds["nu"].values if "nu" in ds else np.full(n_time, 1e-6)

    if "diss_length" in ds.attrs:
        diss_length = float(ds.attrs["diss_length"])
    elif "diss_length" in ds:
        diss_length = float(ds["diss_length"].values)
    else:
        diss_length = 512.0

    if "fs_fast" in ds.attrs:
        fs = float(ds.attrs["fs_fast"])
    elif "fs_fast" in ds:
        fs = float(ds["fs_fast"].values)
    else:
        fs = 512.0

    # Kolmogorov length
    with np.errstate(invalid="ignore", divide="ignore"):
        L_K = (nu[:, np.newaxis] ** 3 / epsilon) ** 0.25

    # Physical length of the data segment
    L = speed * diss_length / fs  # [time]
    L = L[:, np.newaxis]  # [time x 1]

    # Normalized length
    with np.errstate(invalid="ignore", divide="ignore"):
        L_hat = L / L_K

    # Variance of ln(epsilon)
    with np.errstate(invalid="ignore"):
        var_ln_epsilon = 5.5 / (1.0 + (L_hat / 4.0) ** (7.0 / 9.0))
    sigma_ln_epsilon = np.sqrt(var_ln_epsilon)

    # Mean sigma across probes
    mu_sigma = np.nanmean(sigma_ln_epsilon, axis=1)  # [time]
    CF95_range = 1.96 * np.sqrt(2.0) * mu_sigma  # [time]

    # Iterative removal of probes outside 95% CI
    n_probes = epsilon.shape[1]
    for _ in range(n_probes - 1):
        with np.errstate(invalid="ignore"):
            min_e = np.nanmin(epsilon, axis=1)
            max_e = np.nanmax(epsilon, axis=1)
            ratio = np.abs(np.log(max_e) - np.log(min_e))

        outside = ratio > CF95_range
        if not np.any(outside):
            break

        # Set the maximum value to NaN for rows outside CI
        max_idx = np.nanargmax(epsilon, axis=1)
        for i in np.where(outside)[0]:
            epsilon[i, max_idx[i]] = np.nan

    # Geometric mean of surviving probes
    with np.errstate(invalid="ignore"):
        epsilon_mean = np.exp(np.nanmean(np.log(epsilon), axis=1))

    ds["epsilonMean"] = xr.DataArray(
        epsilon_mean,
        dims=["time"],
        attrs={"long_name": "combined epsilon (geometric mean)", "units": "W/kg"},
    )
    ds["epsilonLnSigma"] = xr.DataArray(
        mu_sigma,
        dims=["time"],
        attrs={"long_name": "sigma of ln(epsilon)", "units": ""},
    )

    return ds
