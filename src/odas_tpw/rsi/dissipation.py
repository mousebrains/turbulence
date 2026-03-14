# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Core epsilon (TKE dissipation rate) calculation.

Port of get_diss_odas.m from the ODAS MATLAB library.

References
----------
Lueck, R.G., 2022a: The statistics of oceanic turbulence measurements.
    Part 1: Shear variance and dissipation rates. J. Atmos. Oceanic
    Technol., 39, 1259-1276.
Lueck, R.G., 2022b: The statistics of oceanic turbulence measurements.
    Part 2: Shear spectra and a new spectral model. J. Atmos. Oceanic
    Technol., 39, 1273-1282.
Lueck, R.G., and 27 coauthors, 2024: Best practices recommendations for
    estimating dissipation rates from shear probes. Front. Mar. Sci.,
    11, 1334327.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import xarray as xr

if TYPE_CHECKING:
    from odas_tpw.rsi.p_file import PFile

from odas_tpw.rsi.helpers import load_channels, prepare_profiles
from odas_tpw.scor160.despike import despike
from odas_tpw.scor160.goodman import clean_shear_spec_batch
from odas_tpw.scor160.l4 import (
    E_ISR_THRESHOLD,
    _estimate_epsilon,
)
from odas_tpw.scor160.ocean import visc, visc35

# ---------------------------------------------------------------------------
# Named constants — citations inline
# ---------------------------------------------------------------------------

# Minimum profiling speed to avoid wavenumber singularity [m/s]
SPEED_MIN = 0.01

# Nuttall (1971) DOF correction for cosine-windowed overlapped FFTs
DOF_NUTTALL = 1.9

# Macoun & Lueck (1998) wavenumber correction: 1 + (K/MACOUN_LUECK_DENOM)^2
# Applied for K <= MACOUN_LUECK_K
MACOUN_LUECK_K = 150  # [cpm]
MACOUN_LUECK_DENOM = 48  # [cpm]

# Anti-aliasing safety margin (ODAS convention)
F_AA_MARGIN = 0.9


# ---------------------------------------------------------------------------
# Core dissipation calculation
# ---------------------------------------------------------------------------


def get_diss(
    source: "PFile | str | Path",
    fft_length: int = 256,
    diss_length: int | None = None,
    overlap: int | None = None,
    speed: float | None = None,
    direction: str = "down",
    goodman: bool = True,
    f_AA: float = 98.0,
    f_limit: float | None = None,
    fit_order: int = 3,
    despike_thresh: float = 8,
    despike_smooth: float = 0.5,
    salinity: npt.ArrayLike | None = None,
) -> list[xr.Dataset]:
    """Compute epsilon from any source.

    Parameters
    ----------
    source : PFile, str, or Path
        A PFile object, .p file path, full-record .nc,
        or per-profile .nc.
    fft_length : int
        FFT segment length in samples. Default: 256 (0.5 s at 512 Hz).
    diss_length : int or None
        Dissipation window in samples. Default: 2 * fft_length.
    overlap : int or None
        Window overlap in samples. Default: diss_length // 2.
    speed : float or None
        Profiling speed [m/s]. If None, computed from dP/dt.
    direction : str
        'up' or 'down' for speed sign convention.
    goodman : bool
        Apply Goodman coherent noise removal. Default: True.
    f_AA : float
        Anti-aliasing filter cutoff [Hz]. Default: 98.
    f_limit : float or None
        Maximum frequency for integration. Default: None (use f_AA).
    fit_order : int
        Polynomial order for spectral minimum fit. Default: 3.
    despike_thresh : float
        Despike threshold. Default: 8.
    despike_smooth : float
        Despike smoothing frequency [Hz]. Default: 0.5.
    salinity : float or array_like or None
        Practical salinity [PSU]. If provided, uses gsw-based viscosity
        instead of visc35. Scalar or array matching slow time series.

    Returns
    -------
    list of xarray.Dataset
        One Dataset per profile. Each contains:
        epsilon, K_max, speed, nu, T_mean, P_mean, spec_shear,
        spec_nasmyth, K, mad, fom, K_max_ratio.
    """
    if diss_length is None:
        diss_length = 2 * fft_length
    if overlap is None:
        overlap = diss_length // 2
    if f_limit is None:
        f_limit = np.inf

    data = load_channels(source)

    # Get shear and accel arrays
    shear_names = [s[0] for s in data["shear"]]
    shear_arrays = [s[1] for s in data["shear"]]
    accel_arrays = [a[1] for a in data["accel"]]
    n_shear = len(shear_arrays)
    n_accel = len(accel_arrays)

    if n_shear == 0:
        raise ValueError("No shear channels found")
    if goodman and n_accel == 0:
        raise ValueError("No accelerometer channels found (required for Goodman)")

    # Despike shear data
    fs_fast = data["fs_fast"]
    for i in range(n_shear):
        shear_arrays[i], _, _, _ = despike(
            shear_arrays[i],
            fs_fast,
            thresh=despike_thresh,
            smooth=despike_smooth,
        )

    # Look up vehicle-specific tau for speed smoothing
    from odas_tpw.rsi.profile import _VEHICLE_TAU

    vehicle = data.get("vehicle", "")
    tau = _VEHICLE_TAU.get(vehicle, 1.5)

    # Profile detection, speed, P/T interpolation, salinity
    prepared = prepare_profiles(data, speed, direction, salinity, tau=tau)
    if prepared is None:
        return []
    (profiles_slow, speed_fast, P_fast, T_fast, sal_fast, fs_fast, _fs_slow, ratio, t_fast) = (
        prepared
    )

    # Convert shear from piezo output to du/dz by dividing by speed^2,
    # matching ODAS odas_p2mat.m line 922.
    for i in range(n_shear):
        shear_arrays[i] = shear_arrays[i] / speed_fast**2

    # Effective AA cutoff
    f_AA_eff = F_AA_MARGIN * f_AA
    if f_limit < f_AA_eff:
        f_AA_eff = f_limit

    results = []
    for s_slow, e_slow in profiles_slow:
        s_fast = s_slow * ratio
        e_fast = min((e_slow + 1) * ratio, len(t_fast))

        # Build shear and accel matrices for this profile
        sh_mat = np.column_stack([s[s_fast:e_fast] for s in shear_arrays])
        ac_mat = np.column_stack([a[s_fast:e_fast] for a in accel_arrays])
        p_prof = P_fast[s_fast:e_fast]
        t_prof = T_fast[s_fast:e_fast]
        spd_prof = speed_fast[s_fast:e_fast]
        time_prof = t_fast[s_fast:e_fast]

        sal_prof = sal_fast[s_fast:e_fast] if isinstance(sal_fast, np.ndarray) else sal_fast

        ds = _compute_profile_diss(
            sh_mat,
            ac_mat,
            p_prof,
            t_prof,
            spd_prof,
            time_prof,
            shear_names,
            fs_fast,
            fft_length,
            diss_length,
            overlap,
            goodman,
            f_AA_eff,
            fit_order,
            salinity=sal_prof,
        )
        ds.attrs.update(data["metadata"])
        ds.attrs["history"] = f"Computed with microstructure-tpw on {datetime.now(UTC).isoformat()}"
        # CF time coordinate attributes
        start_time = data["metadata"].get("start_time", "")
        t_units = f"seconds since {start_time}" if start_time else "seconds"
        ds.coords["t"].attrs.update(
            {
                "standard_name": "time",
                "long_name": "time of dissipation estimate",
                "units": t_units,
                "calendar": "standard",
                "axis": "T",
            }
        )
        results.append(ds)

    return results


def _compute_profile_diss(
    shear,
    accel,
    P,
    T,
    speed,
    t_fast,
    shear_names,
    fs_fast,
    fft_length,
    diss_length,
    overlap,
    do_goodman,
    f_AA,
    fit_order,
    salinity=None,
):
    """Compute dissipation for all windows in a single profile.

    Iterates over dissipation windows, computes Goodman-cleaned shear
    wavenumber spectra, and estimates epsilon via the variance or ISR method.

    Parameters
    ----------
    shear : ndarray, shape (N, n_shear)
        Shear probe data [s⁻¹], already despiked and divided by speed².
    accel : ndarray, shape (N, n_accel)
        Accelerometer data for Goodman cleaning.
    P : ndarray, shape (N,)
        Pressure [dbar].
    T : ndarray, shape (N,)
        Temperature [°C].
    speed : ndarray, shape (N,)
        Profiling speed [m/s].
    t_fast : ndarray, shape (N,)
        Time vector [s].
    shear_names : list of str
        Probe names for coordinate labelling.
    fs_fast : float
        Sampling rate [Hz].
    fft_length : int
        FFT segment length [samples].
    diss_length : int
        Dissipation window length [samples].
    overlap : int
        Window overlap [samples].
    do_goodman : bool
        Whether to apply Goodman coherent noise removal.
    f_AA : float
        Effective anti-aliasing cutoff [Hz].
    fit_order : int
        Polynomial order for spectral minimum detection.
    salinity : float, ndarray, or None
        Practical salinity for viscosity calculation.

    Returns
    -------
    xarray.Dataset
        Contains epsilon, K_max, mad, fom, FM, K_max_ratio, spectra, etc.
    """
    N = shear.shape[0]
    n_shear = shear.shape[1]
    n_freq = fft_length // 2 + 1

    # Number of dissipation estimates
    if overlap >= diss_length:
        overlap = diss_length // 2
    step = diss_length - overlap
    n_est = max(1, 1 + (N - diss_length) // step)

    # Trim n_est if the last window would exceed the data
    while n_est > 0 and (n_est - 1) * step + diss_length > N:
        n_est -= 1

    # Degrees of freedom — Lueck (2022a,b)
    num_ffts = 2 * (diss_length // fft_length) - 1
    n_v = accel.shape[1] if do_goodman else 0
    dof_spec = DOF_NUTTALL * max(num_ffts - n_v, 1)

    # ---- Vectorized: extract all windows at once ----
    win_starts = np.arange(n_est) * step
    indices = win_starts[:, np.newaxis] + np.arange(diss_length)[np.newaxis, :]
    # shape: (n_est, diss_length, n_channels)
    sh_windows = shear[indices]
    ac_windows = accel[indices]

    # ---- Vectorized: batched CSD + Goodman ----
    if do_goodman:
        # clean_UU: (n_est, n_freq, n_shear, n_shear)
        P_sh_clean_all, F = clean_shear_spec_batch(
            ac_windows,
            sh_windows,
            fft_length,
            fs_fast,
        )
    else:
        from odas_tpw.scor160.spectral import csd_matrix_batch

        P_sh_all, F, _, _ = csd_matrix_batch(
            sh_windows,
            None,
            fft_length,
            fs_fast,
            overlap=fft_length // 2,
            detrend="linear",
        )
        P_sh_clean_all = np.real(P_sh_all)

    del sh_windows, ac_windows

    # ---- Vectorized: window means ----
    # Reshape for fast window-mean computation
    speed_windows = np.abs(speed[indices])  # (n_est, diss_length)
    speed_out = np.mean(speed_windows, axis=1)
    speed_out = np.maximum(speed_out, SPEED_MIN)

    T_out = np.mean(T[indices], axis=1)
    P_out = np.mean(P[indices], axis=1)
    t_out = np.mean(t_fast[indices], axis=1)

    # Viscosity per window
    if salinity is not None:
        if np.ndim(salinity) > 0:
            sal_windows = salinity[indices]
            sal_means = np.mean(sal_windows, axis=1)
        else:
            sal_means = np.full(n_est, float(salinity))
        nu_out = np.array([float(visc(T_out[i], sal_means[i], P_out[i])) for i in range(n_est)])
    else:
        nu_out = np.array([float(visc35(T_out[i])) for i in range(n_est)])

    # ---- Vectorized: frequency → wavenumber conversion + correction ----
    # K varies per window because speed varies: K[f, w] = F[f] / W[w]
    K_all = F[:, np.newaxis] / speed_out[np.newaxis, :]  # (n_freq, n_est)
    K_AA_all = f_AA / speed_out  # (n_est,)

    # Macoun-Lueck correction: (n_freq, n_est)
    correction = np.ones_like(K_all)
    mask = K_all <= MACOUN_LUECK_K
    correction[mask] = 1 + (K_all[mask] / MACOUN_LUECK_DENOM) ** 2

    # Convert CSD from frequency to wavenumber spectra and apply correction
    # P_sh_clean_all: (n_est, n_freq, n_sh, n_sh)
    # Multiply by W (per window) and correction (per freq, per window)
    wk_scale = speed_out[np.newaxis, :] * correction  # (n_freq, n_est)
    P_sh_clean_all = P_sh_clean_all * wk_scale.T[:, :, np.newaxis, np.newaxis]

    # Store F (same for all windows)
    F_out = np.broadcast_to(F[:, np.newaxis], (n_freq, n_est)).copy()
    K_out = K_all

    # ---- Per-window epsilon estimation (not vectorizable) ----
    epsilon = np.full((n_shear, n_est), np.nan)
    K_max_out = np.full((n_shear, n_est), np.nan)
    mad_out = np.full((n_shear, n_est), np.nan)
    fom_out = np.full((n_shear, n_est), np.nan)
    FM_out = np.full((n_shear, n_est), np.nan)
    K_max_ratio_out = np.full((n_shear, n_est), np.nan)
    method_out = np.zeros((n_shear, n_est), dtype=np.int8)
    spec_shear = np.full((n_shear, n_freq, n_est), np.nan)
    spec_nasmyth = np.full((n_shear, n_freq, n_est), np.nan)

    for idx in range(n_est):
        K = K_all[:, idx]
        K_AA = K_AA_all[idx]
        nu = nu_out[idx]

        for ci in range(n_shear):
            shear_spec = np.real(P_sh_clean_all[idx, :, ci, ci])

            e_4, k_max, mad, meth, fom_val, _var_res, nas_spec, K_max_ratio_val, FM_val = (
                _estimate_epsilon(
                    K,
                    shear_spec,
                    nu,
                    K_AA,
                    fit_order,
                    e_isr_threshold=E_ISR_THRESHOLD,
                    num_ffts=num_ffts,
                    n_v=n_v,
                )
            )
            epsilon[ci, idx] = e_4
            K_max_out[ci, idx] = k_max
            mad_out[ci, idx] = mad
            fom_out[ci, idx] = fom_val
            FM_out[ci, idx] = FM_val
            K_max_ratio_out[ci, idx] = K_max_ratio_val
            method_out[ci, idx] = meth
            spec_shear[ci, :, idx] = shear_spec
            spec_nasmyth[ci, :, idx] = nas_spec

    return _build_diss_dataset(
        epsilon=epsilon,
        K_max_out=K_max_out,
        mad_out=mad_out,
        fom_out=fom_out,
        FM_out=FM_out,
        K_max_ratio_out=K_max_ratio_out,
        method_out=method_out,
        speed_out=speed_out,
        nu_out=nu_out,
        P_out=P_out,
        T_out=T_out,
        t_out=t_out,
        spec_shear=spec_shear,
        spec_nasmyth=spec_nasmyth,
        K_out=K_out,
        F_out=F_out,
        shear_names=shear_names,
        fft_length=fft_length,
        diss_length=diss_length,
        overlap=overlap,
        fs_fast=fs_fast,
        dof_spec=dof_spec,
        do_goodman=do_goodman,
        f_AA=f_AA,
        fit_order=fit_order,
    )


def _build_diss_dataset(
    *,
    epsilon: np.ndarray,
    K_max_out: np.ndarray,
    mad_out: np.ndarray,
    fom_out: np.ndarray,
    FM_out: np.ndarray,
    K_max_ratio_out: np.ndarray,
    method_out: np.ndarray,
    speed_out: np.ndarray,
    nu_out: np.ndarray,
    P_out: np.ndarray,
    T_out: np.ndarray,
    t_out: np.ndarray,
    spec_shear: np.ndarray,
    spec_nasmyth: np.ndarray,
    K_out: np.ndarray,
    F_out: np.ndarray,
    shear_names: list[str],
    fft_length: int,
    diss_length: int,
    overlap: int,
    fs_fast: float,
    dof_spec: float,
    do_goodman: bool,
    f_AA: float,
    fit_order: int,
) -> xr.Dataset:
    """Build an xarray Dataset from epsilon estimation output arrays."""
    ds = xr.Dataset(
        {
            "epsilon": (
                ["probe", "time"],
                epsilon,
                {
                    "units": "W kg-1",
                    "long_name": "TKE dissipation rate",
                },
            ),
            "K_max": (
                ["probe", "time"],
                K_max_out,
                {
                    "units": "cpm",
                    "long_name": "upper wavenumber integration limit",
                },
            ),
            "mad": (
                ["probe", "time"],
                mad_out,
                {
                    "units": "1",
                    "long_name": "mean absolute deviation of spectral fit in log10 space",
                },
            ),
            "fom": (
                ["probe", "time"],
                fom_out,
                {
                    "units": "1",
                    "long_name": "figure of merit (observed/Nasmyth variance ratio)",
                },
            ),
            "FM": (
                ["probe", "time"],
                FM_out,
                {
                    "units": "1",
                    "long_name": "Lueck figure of merit (MAD * sqrt(dof))",
                    "comment": "FM < 1 for 97.5% of good spectra (Lueck, 2022a,b)",
                },
            ),
            "K_max_ratio": (
                ["probe", "time"],
                K_max_ratio_out,
                {
                    "units": "1",
                    "long_name": "K_max / K_95 spectral resolution ratio",
                },
            ),
            "method": (
                ["probe", "time"],
                method_out,
                {
                    "long_name": "spectral fitting method",
                    "flag_values": np.array([0, 1], dtype=np.int8),
                    "flag_meanings": "variance inertial_subrange",
                },
            ),
            "speed": (
                ["time"],
                speed_out,
                {
                    "units": "m s-1",
                    "long_name": "profiling speed",
                },
            ),
            "nu": (
                ["time"],
                nu_out,
                {
                    "units": "m2 s-1",
                    "long_name": "kinematic viscosity of sea water",
                },
            ),
            "P_mean": (
                ["time"],
                P_out,
                {
                    "units": "dbar",
                    "long_name": "mean sea water pressure",
                    "standard_name": "sea_water_pressure",
                    "positive": "down",
                },
            ),
            "T_mean": (
                ["time"],
                T_out,
                {
                    "units": "degree_Celsius",
                    "long_name": "mean sea water temperature",
                    "standard_name": "sea_water_temperature",
                },
            ),
            "spec_shear": (
                ["probe", "freq", "time"],
                spec_shear,
                {
                    "units": "s-2 cpm-1",
                    "long_name": "shear wavenumber spectrum (observed, cleaned)",
                },
            ),
            "spec_nasmyth": (
                ["probe", "freq", "time"],
                spec_nasmyth,
                {
                    "units": "s-2 cpm-1",
                    "long_name": "Nasmyth theoretical shear spectrum",
                },
            ),
            "K": (
                ["freq", "time"],
                K_out,
                {
                    "units": "cpm",
                    "long_name": "wavenumber (cycles per metre)",
                },
            ),
            "F": (
                ["freq", "time"],
                F_out,
                {
                    "units": "Hz",
                    "long_name": "frequency",
                },
            ),
        },
        coords={
            "probe": shear_names,
            "t": (["time"], t_out),
        },
        attrs={
            "Conventions": "CF-1.13",
            "fft_length": fft_length,
            "diss_length": diss_length,
            "overlap": overlap,
            "fs_fast": fs_fast,
            "dof_spec": dof_spec,
            "goodman": int(do_goodman),
            "f_AA": f_AA,
            "fit_order": fit_order,
        },
    )
    ds.coords["probe"].attrs["long_name"] = "shear probe name"
    return ds

    # _estimate_epsilon, _variance_correction, and _inertial_subrange
    # are now imported from odas_tpw.scor160.l4 at the top of this module.


# ---------------------------------------------------------------------------
# File-level processing
# ---------------------------------------------------------------------------


def compute_diss_file(
    source_path: str | Path,
    output_dir: str | Path,
    **diss_kwargs: Any,
) -> list[Path]:
    """Compute epsilon for one file and write NetCDF output(s).

    Parameters
    ----------
    source_path : str or Path
        Input file (.p, full-record .nc, or per-profile .nc).
    output_dir : str or Path
        Output directory for epsilon .nc files.
    **diss_kwargs
        Keyword arguments passed to get_diss.

    Returns
    -------
    list of Path
        Paths to output files written.
    """
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = get_diss(source_path, **diss_kwargs)

    output_paths = []
    for i, ds in enumerate(results):
        if len(results) == 1:
            out_name = f"{source_path.stem}_eps.nc"
        else:
            out_name = f"{source_path.stem}_prof{i + 1:03d}_eps.nc"
        out_path = output_dir / out_name
        ds.to_netcdf(out_path)
        output_paths.append(out_path)
        print(
            f"  {out_path.name}: {ds.sizes['time']} estimates, "
            f"P={float(ds.P_mean.min()):.0f}–{float(ds.P_mean.max()):.0f} dbar"  # noqa: RUF001
        )

    return output_paths


def _compute_diss_one(args: tuple) -> tuple[str, int]:
    """Worker for parallel dissipation computation."""
    source_path, output_dir, kwargs = args
    paths = compute_diss_file(source_path, output_dir, **kwargs)
    return str(source_path), len(paths)
