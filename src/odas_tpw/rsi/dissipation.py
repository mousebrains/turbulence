# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Core epsilon (TKE dissipation rate) calculation.

Port of get_diss_odas.m from the ODAS MATLAB library.

.. deprecated::
    ``get_diss`` and ``_compute_profile_diss`` are deprecated.
    Use :func:`odas_tpw.rsi.pipeline.run_pipeline` or the modular
    ``process_l2`` → ``process_l3`` → ``process_l4`` chain from
    :mod:`odas_tpw.scor160` instead.

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

import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import xarray as xr

if TYPE_CHECKING:
    from odas_tpw.rsi.p_file import PFile

from odas_tpw.rsi.helpers import _build_l1data_from_channels, load_channels, prepare_profiles
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
    warnings.warn(
        "get_diss() is deprecated. Use run_pipeline() or the modular "
        "process_l2 → process_l3 → process_l4 chain from odas_tpw.scor160 "
        "instead. See odas_tpw.rsi.pipeline.run_pipeline().",
        DeprecationWarning,
        stacklevel=2,
    )

    from odas_tpw.scor160.io import L2Params, L3Params
    from odas_tpw.scor160.l2 import process_l2
    from odas_tpw.scor160.l3 import process_l3

    if diss_length is None:
        diss_length = 2 * fft_length
    if overlap is None:
        overlap = diss_length // 2
    if f_limit is None:
        f_limit = np.inf

    data = load_channels(source)

    shear_names = [s[0] for s in data["shear"]]
    n_shear = len(shear_names)
    n_accel = len(data["accel"])

    if n_shear == 0:
        raise ValueError("No shear channels found")
    if goodman and n_accel == 0:
        raise ValueError("No accelerometer channels found (required for Goodman)")

    fs_fast = data["fs_fast"]

    from odas_tpw.rsi.profile import _VEHICLE_TAU

    vehicle = data.get("vehicle", "")
    tau = _VEHICLE_TAU.get(vehicle, 1.5)

    prepared = prepare_profiles(data, speed, direction, salinity, tau=tau)
    if prepared is None:
        return []
    (profiles_slow, speed_fast, P_fast, T_fast, sal_fast, fs_fast, _fs_slow, ratio, t_fast) = (
        prepared
    )

    f_AA_eff = F_AA_MARGIN * f_AA
    if f_limit < f_AA_eff:
        f_AA_eff = f_limit

    # Pipeline parameters — lenient section selection since profiles
    # are already detected by prepare_profiles
    l2_params = L2Params(
        HP_cut=0.25,
        despike_sh=np.array([despike_thresh, despike_smooth, 0.04]),
        despike_A=np.array([np.inf, 0.5, 0.04]),
        profile_min_W=0.05,
        profile_min_P=0.0,
        profile_min_duration=0.0,
        speed_tau=0.0,
    )
    l3_params = L3Params(
        fft_length=fft_length,
        diss_length=diss_length,
        overlap=overlap,
        HP_cut=0.25,
        fs_fast=fs_fast,
        goodman=goodman,
    )

    results = []
    for s_slow, e_slow in profiles_slow:
        s_fast = s_slow * ratio
        e_fast = min((e_slow + 1) * ratio, len(t_fast))

        sal_prof = sal_fast[s_fast:e_fast] if isinstance(sal_fast, np.ndarray) else sal_fast

        # Build L1Data (shear normalized by speed^2)
        l1 = _build_l1data_from_channels(
            data, s_fast, e_fast, speed_fast, P_fast, T_fast, direction,
        )

        # L2: HP filter + despike + section selection
        l2 = process_l2(l1, l2_params)

        # L3: wavenumber spectra (Goodman-cleaned, Macoun-Lueck corrected)
        l3 = process_l3(l2, l1, l3_params)

        if l3.n_spectra == 0:
            continue

        # L4: custom epsilon loop with salinity support and full output
        n_spec = l3.n_spectra
        n_freq = l3.n_wavenumber
        n_sh = l3.n_shear

        num_ffts = 2 * (diss_length // fft_length) - 1
        n_v = n_accel if goodman else 0
        dof_spec = DOF_NUTTALL * max(num_ffts - n_v, 1)

        epsilon = np.full((n_sh, n_spec), np.nan)
        K_max_out = np.full((n_sh, n_spec), np.nan)
        mad_out = np.full((n_sh, n_spec), np.nan)
        fom_out = np.full((n_sh, n_spec), np.nan)
        FM_out = np.full((n_sh, n_spec), np.nan)
        K_max_ratio_out = np.full((n_sh, n_spec), np.nan)
        method_out = np.zeros((n_sh, n_spec), dtype=np.int8)
        spec_shear = np.full((n_sh, n_freq, n_spec), np.nan)
        spec_nasmyth = np.full((n_sh, n_freq, n_spec), np.nan)

        # Viscosity per window (with salinity support)
        if salinity is not None:
            if np.ndim(sal_prof) > 0:
                sal_window = np.interp(l3.time, l1.time, sal_prof)
            else:
                sal_window = np.full(n_spec, float(sal_prof))
            nu_out = np.array([
                float(visc(l3.temp[j], sal_window[j], l3.pres[j]))
                for j in range(n_spec)
            ])
        else:
            nu_out = np.array([float(visc35(l3.temp[j])) for j in range(n_spec)])

        for j in range(n_spec):
            K = l3.kcyc[:, j]
            K_AA = f_AA_eff / max(l3.pspd_rel[j], SPEED_MIN)
            nu = nu_out[j]

            for ci in range(n_sh):
                shear_spec = l3.sh_spec_clean[ci, :, j]

                e_4, k_max, mad, meth, fom_val, _var_res, nas_spec, K_max_ratio_val, FM_val = (
                    _estimate_epsilon(
                        K, shear_spec, nu, K_AA, fit_order,
                        e_isr_threshold=E_ISR_THRESHOLD,
                        num_ffts=num_ffts, n_v=n_v,
                    )
                )
                epsilon[ci, j] = e_4
                K_max_out[ci, j] = k_max
                mad_out[ci, j] = mad
                fom_out[ci, j] = fom_val
                FM_out[ci, j] = FM_val
                K_max_ratio_out[ci, j] = K_max_ratio_val
                method_out[ci, j] = meth
                spec_shear[ci, :, j] = shear_spec
                spec_nasmyth[ci, :, j] = nas_spec

        # F vector (constant across windows)
        F = np.arange(n_freq) * fs_fast / fft_length
        F_out = np.broadcast_to(F[:, np.newaxis], (n_freq, n_spec)).copy()

        ds = _build_diss_dataset(
            epsilon=epsilon,
            K_max_out=K_max_out,
            mad_out=mad_out,
            fom_out=fom_out,
            FM_out=FM_out,
            K_max_ratio_out=K_max_ratio_out,
            method_out=method_out,
            speed_out=l3.pspd_rel,
            nu_out=nu_out,
            P_out=l3.pres,
            T_out=l3.temp,
            t_out=l3.time,
            spec_shear=spec_shear,
            spec_nasmyth=spec_nasmyth,
            K_out=l3.kcyc,
            F_out=F_out,
            shear_names=shear_names,
            fft_length=fft_length,
            diss_length=diss_length,
            overlap=overlap,
            fs_fast=fs_fast,
            dof_spec=dof_spec,
            do_goodman=goodman,
            f_AA=f_AA_eff,
            fit_order=fit_order,
        )
        ds.attrs.update(data["metadata"])
        ds.attrs["history"] = f"Computed with microstructure-tpw on {datetime.now(UTC).isoformat()}"
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
    from odas_tpw.rsi.helpers import _build_result_dataset

    variables = [
        ("epsilon", ["probe", "time"], epsilon, {"units": "W kg-1", "long_name": "TKE dissipation rate"}),
        ("K_max", ["probe", "time"], K_max_out, {"units": "cpm", "long_name": "upper wavenumber integration limit"}),
        ("mad", ["probe", "time"], mad_out, {"units": "1", "long_name": "mean absolute deviation of spectral fit in log10 space"}),
        ("fom", ["probe", "time"], fom_out, {"units": "1", "long_name": "figure of merit (observed/Nasmyth variance ratio)"}),
        ("FM", ["probe", "time"], FM_out, {"units": "1", "long_name": "Lueck figure of merit (MAD * sqrt(dof))", "comment": "FM < 1 for 97.5% of good spectra (Lueck, 2022a,b)"}),
        ("K_max_ratio", ["probe", "time"], K_max_ratio_out, {"units": "1", "long_name": "K_max / K_95 spectral resolution ratio"}),
        ("method", ["probe", "time"], method_out, {"long_name": "spectral fitting method", "flag_values": np.array([0, 1], dtype=np.int8), "flag_meanings": "variance inertial_subrange"}),
        ("speed", ["time"], speed_out, {"units": "m s-1", "long_name": "profiling speed"}),
        ("nu", ["time"], nu_out, {"units": "m2 s-1", "long_name": "kinematic viscosity of sea water"}),
        ("P_mean", ["time"], P_out, {"units": "dbar", "long_name": "mean sea water pressure", "standard_name": "sea_water_pressure", "positive": "down"}),
        ("T_mean", ["time"], T_out, {"units": "degree_Celsius", "long_name": "mean sea water temperature", "standard_name": "sea_water_temperature"}),
        ("spec_shear", ["probe", "freq", "time"], spec_shear, {"units": "s-2 cpm-1", "long_name": "shear wavenumber spectrum (observed, cleaned)"}),
        ("spec_nasmyth", ["probe", "freq", "time"], spec_nasmyth, {"units": "s-2 cpm-1", "long_name": "Nasmyth theoretical shear spectrum"}),
        ("K", ["freq", "time"], K_out, {"units": "cpm", "long_name": "wavenumber (cycles per metre)"}),
        ("F", ["freq", "time"], F_out, {"units": "Hz", "long_name": "frequency"}),
    ]
    return _build_result_dataset(
        variables, shear_names, t_out, "shear probe name",
        {
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
    from odas_tpw.rsi.helpers import write_profile_results

    source_path = Path(source_path)
    output_dir = Path(output_dir)
    results = get_diss(source_path, **diss_kwargs)
    return write_profile_results(results, source_path, output_dir, "eps")


def _compute_diss_one(args: tuple) -> tuple[str, int]:
    """Worker for parallel dissipation computation."""
    source_path, output_dir, kwargs = args
    paths = compute_diss_file(source_path, output_dir, **kwargs)
    return str(source_path), len(paths)
