# May-2026, Claude and Pat Welch, pat@mousebrains.com
"""Multi-probe chi combining (mk_chi_mean).

Mirrors :func:`odas_tpw.processing.epsilon_combine.mk_epsilon_mean` for
chi: splits the 2-D ``chi(probe, time)`` from get_chi into per-probe
1-D ``chi_1``, ``chi_2``, ... variables, then combines them into
``chiMean`` (geometric mean of surviving probes) and ``chiLnSigma``
(expected ``sigma_ln(chi)``) using the same variance / 95% CI
machinery as the epsilon path. The 1-D split is what lets the depth
binner pick chi up; without it chi vanishes between chi_NN/ and
chi_combo_NN/.

The variance model is Lueck-style ``5.5 / (1 + (L_hat/4)^(7/9))``
borrowed from the dissipation pipeline. For chi this is a reasonable
first-order estimate when the FFT segments are long compared to the
Batchelor scale, but the chi-specific theoretical variance differs
from the shear case; refine here if a chi-tuned model becomes
available.
"""

import warnings

import numpy as np
import xarray as xr


def mk_chi_mean(
    ds: xr.Dataset,
    chi_minimum: float = 1e-13,
) -> xr.Dataset:
    """Combine per-probe chi into a mean estimate with CI filtering.

    Steps (mirroring :func:`mk_epsilon_mean`):

      1. If only ``chi(probe, time)`` is present, split into 1-D
         ``chi_1``, ``chi_2``, ...
      2. Floor small values (``<= chi_minimum``) to NaN.
      3. Kolmogorov length ``L_K = (nu^3 / epsilon_T)^(1/4)`` when
         ``epsilon_T`` is available, else fall back to the chi values
         themselves so the CI machinery still has something to work
         with.
      4. Physical segment length ``L = speed * diss_length / fs``.
      5. ``var_ln_chi = 5.5 / (1 + (L_hat/4)^(7/9))``; same Lueck-style
         expression used for epsilon.
      6. 95% CI range ``1.96 * sqrt(2) * mean(sigma_ln_chi)``.
      7. Iteratively remove the largest probe value from rows whose
         min/max log-spread exceeds the CI range.
      8. ``chiMean`` = geometric mean of surviving probes; ``chiLnSigma``
         = mean ``sigma_ln_chi`` across probes.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset from ``get_chi``. Should contain ``chi(probe, time)`` or
        per-probe ``chi_1``, ``chi_2``, ... 1-D vars; ``speed`` and
        ``nu`` along ``time``; ``diss_length`` / ``fs_fast`` as scalar
        attributes or variables; optionally ``epsilon_T(probe, time)``
        for the Kolmogorov length term.
    chi_minimum : float
        Floor — values at or below are set to NaN.

    Returns
    -------
    xr.Dataset
        Input dataset with ``chi_1``, ``chi_2``, ... promoted to 1-D
        vars (when only ``chi`` was 2-D), plus ``chiMean`` and
        ``chiLnSigma`` added.
    """
    ds = ds.copy()

    probe_names = sorted(str(k) for k in ds.data_vars if str(k).startswith("chi_"))

    if not probe_names and "chi" in ds and "probe" in ds.dims:
        for i in range(ds.sizes["probe"]):
            name = f"chi_{i + 1}"
            ds[name] = ds["chi"].isel(probe=i)
            probe_names.append(name)

    if not probe_names:
        warnings.warn("No per-probe chi variables (chi_1, chi_2, ...) found")
        return ds

    n_time = ds.sizes.get("time", 0)
    if n_time == 0:
        return ds

    chi = np.column_stack([ds[name].values.copy() for name in probe_names])

    chi[chi <= chi_minimum] = np.nan

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

    # Kolmogorov length: use epsilon_T(probe, time) when available; otherwise
    # fall back to chi itself so L_hat / sigma still vary smoothly with the
    # data — same shape constraint either way.
    if "epsilon_T" in ds and "probe" in ds["epsilon_T"].dims:
        eps_T = np.column_stack(
            [ds["epsilon_T"].isel(probe=i).values for i in range(ds.sizes["probe"])]
        )
    else:
        eps_T = chi

    with np.errstate(invalid="ignore", divide="ignore"):
        L_K = (nu[:, np.newaxis] ** 3 / eps_T) ** 0.25

    L = speed * diss_length / fs
    L = L[:, np.newaxis]

    with np.errstate(invalid="ignore", divide="ignore"):
        L_hat = L / L_K

    with np.errstate(invalid="ignore"):
        var_ln_chi = 5.5 / (1.0 + (L_hat / 4.0) ** (7.0 / 9.0))
    sigma_ln_chi = np.sqrt(var_ln_chi)

    mu_sigma = np.nanmean(sigma_ln_chi, axis=1)
    CF95_range = 1.96 * np.sqrt(2.0) * mu_sigma

    n_probes = chi.shape[1]
    for _ in range(n_probes - 1):
        with np.errstate(invalid="ignore"):
            min_c = np.nanmin(chi, axis=1)
            max_c = np.nanmax(chi, axis=1)
            ratio = np.abs(np.log(max_c) - np.log(min_c))

        outside = ratio > CF95_range
        if not np.any(outside):
            break

        max_idx = np.nanargmax(chi, axis=1)
        for i in np.where(outside)[0]:
            chi[i, max_idx[i]] = np.nan

    with np.errstate(invalid="ignore"):
        chi_mean = np.exp(np.nanmean(np.log(chi), axis=1))

    ds["chiMean"] = xr.DataArray(
        chi_mean,
        dims=["time"],
        attrs={"long_name": "combined chi (geometric mean)", "units": "K2 s-1"},
    )
    ds["chiLnSigma"] = xr.DataArray(
        mu_sigma,
        dims=["time"],
        attrs={"long_name": "sigma of ln(chi)", "units": ""},
    )

    return ds
