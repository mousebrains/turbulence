"""Compatibility helpers for pyturb-cli."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def seconds_to_samples(seconds: float, fs: float) -> int:
    """Convert a time duration to a sample count (made even).

    Matches Jesse's pyturb: ``n = int(seconds * fs)`` rounded to the nearest
    even number (no power-of-2 rounding).
    """
    n = int(seconds * fs)
    if n < 2:
        return 2
    # Make even
    return n if n % 2 == 0 else n + 1


def compute_window_parameters(
    fft_len_sec: float, diss_len_sec: float, fs: float
) -> tuple[int, int, int, int]:
    """Compute (n_fft, n_diss, fft_overlap, diss_overlap) matching Jesse's pyturb.

    Jesse's algorithm:
      n_fft  = int(fft_len * fs), made even
      n_diss = round(diss_len * fs), rounded UP to nearest multiple of n_fft
      fft_overlap = n_fft // 2
      diss_overlap = n_fft // 2
    """
    n_fft = seconds_to_samples(fft_len_sec, fs)
    n_diss_raw = round(diss_len_sec * fs)
    # Round up to nearest multiple of n_fft
    n_diss = n_fft if n_diss_raw <= n_fft else ((n_diss_raw + n_fft - 1) // n_fft) * n_fft
    fft_overlap = n_fft // 2
    diss_overlap = n_fft // 2
    return n_fft, n_diss, fft_overlap, diss_overlap


def check_overwrite(path: Path, overwrite: bool) -> bool:
    """Return True if *path* should be written (does not exist, or overwrite is set)."""
    if not path.exists():
        return True
    return overwrite


def load_auxiliary(
    path: str | Path,
    *,
    lat_var: str = "lat",
    lon_var: str = "lon",
    temp_var: str = "temperature",
    sal_var: str = "salinity",
    dens_var: str = "density",
) -> xr.Dataset:
    """Load an auxiliary CTD NetCDF and validate required variables exist."""
    ds = xr.open_dataset(path)
    for name, var in [
        ("lat", lat_var),
        ("lon", lon_var),
        ("temp", temp_var),
        ("sal", sal_var),
        ("dens", dens_var),
    ]:
        if var not in ds:
            raise KeyError(f"Auxiliary file missing variable '{var}' (for {name})")
    return ds


def rename_eps_dataset(
    l4,
    l3,
    l3_chi,
    *,
    aux_data: xr.Dataset | None = None,
    fs_fast: float = 512.0,
    salinity: float = 35.0,
) -> xr.Dataset:
    """Build an output Dataset with Jesse's variable naming convention.

    Parameters
    ----------
    l4 : L4Data
        Level-4 dissipation estimates.
    l3 : L3Data
        Level-3 shear wavenumber spectra.
    l3_chi : L3ChiData or None
        Level-3 temperature gradient spectra (may be None).
    aux_data : xr.Dataset, optional
        Auxiliary CTD data to merge.
    fs_fast : float
        Fast sampling rate for metadata.
    salinity : float
        Default salinity for viscosity computation.
    """
    from odas_tpw.scor160.ocean import density as calc_density
    from odas_tpw.scor160.ocean import visc35

    n_spec = l4.n_spectra
    if n_spec == 0:
        return xr.Dataset()

    # Scalar arrays
    pres = l4.pres
    speed = l4.pspd_rel
    temp = l3.temp

    # Viscosity and density
    nu = np.array([visc35(t) for t in temp])
    sal = np.full(n_spec, salinity)
    dens = np.array([calc_density(temp[i], sal[i], pres[i]) for i in range(n_spec)])

    # Frequency and wavenumber vectors (from first window)
    n_freq = l3.kcyc.shape[0]
    freq = np.arange(n_freq) * fs_fast / ((n_freq - 1) * 2)
    # Mean wavenumber across windows for coordinate
    k_mean = np.mean(l3.kcyc, axis=1)

    data_vars: dict = {
        "pressure": (["time"], pres, {"units": "dbar", "long_name": "mean pressure"}),
        "W": (["time"], speed, {"units": "m/s", "long_name": "profiling speed"}),
        "temperature": (
            ["time"], temp, {"units": "degree_Celsius", "long_name": "mean temperature"},
        ),
        "nu": (["time"], nu, {"units": "m2/s", "long_name": "kinematic viscosity"}),
        "salinity": (["time"], sal, {"units": "PSU", "long_name": "practical salinity"}),
        "density": (["time"], dens, {"units": "kg/m3", "long_name": "seawater density"}),
    }

    # Epsilon per probe
    for i in range(l4.n_shear):
        data_vars[f"eps_{i + 1}"] = (
            ["time"],
            l4.epsi[i],
            {"units": "W/kg", "long_name": f"TKE dissipation rate (probe {i + 1})"},
        )
        data_vars[f"k_max_{i + 1}"] = (
            ["time"],
            l4.kmax[i],
            {"units": "cpm", "long_name": f"upper integration limit (probe {i + 1})"},
        )
        data_vars[f"fom_{i + 1}"] = (
            ["time"],
            l4.fom[i],
            {"long_name": f"figure of merit (probe {i + 1})"},
        )
        data_vars[f"mad_{i + 1}"] = (
            ["time"],
            l4.mad[i],
            {"long_name": f"mean absolute deviation (probe {i + 1})"},
        )
        data_vars[f"method_{i + 1}"] = (
            ["time"],
            l4.method[i],
            {"long_name": f"estimation method (probe {i + 1}): 0=variance, 1=ISR"},
        )

    data_vars["eps_final"] = (
        ["time"],
        l4.epsi_final,
        {"units": "W/kg", "long_name": "combined TKE dissipation rate"},
    )

    # Shear spectra per probe
    for i in range(l3.n_shear):
        data_vars[f"S_sh{i + 1}"] = (
            ["frequency", "time"],
            l3.sh_spec_clean[i],
            {"units": "s-2/cpm", "long_name": f"cleaned shear spectrum (probe {i + 1})"},
        )

    # Temperature gradient spectra
    if l3_chi is not None and l3_chi.n_spectra > 0:
        for i in range(l3_chi.n_gradt):
            data_vars[f"S_gradT{i + 1}"] = (
                ["frequency", "time"],
                l3_chi.gradt_spec[i],
                {"units": "K2/m2/cpm", "long_name": f"gradT spectrum (probe {i + 1})"},
            )

    coords: dict = {
        "frequency": freq,
        "k": (["frequency"], k_mean, {"units": "cpm", "long_name": "wavenumber"}),
    }

    ds = xr.Dataset(data_vars, coords=coords)

    # Merge auxiliary data if provided
    if aux_data is not None:
        for var in aux_data.data_vars:
            if var not in ds:
                ds[var] = aux_data[var]

    return ds
