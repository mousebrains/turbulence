# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Convenience wrappers for chi computation from PFile / NetCDF sources.

These functions bridge rsi instrument I/O with the pure chi
computation in chi.chi.  They handle data loading, profile
detection, and file-level orchestration.

.. deprecated::
    ``get_chi`` is deprecated.  Use :func:`odas_tpw.rsi.pipeline.run_pipeline`
    or the modular ``process_l2_chi`` → ``process_l3_chi`` → ``process_l4_chi_*``
    chain instead.
"""

from __future__ import annotations

import re
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import xarray as xr

if TYPE_CHECKING:
    from odas_tpw.rsi.p_file import PFile


def _compute_chi(
    source: PFile | str | Path,
    epsilon_ds: xr.Dataset | None = None,
    fft_length: int = 1024,
    diss_length: int | None = None,
    overlap: int | None = None,
    speed: float | None = None,
    direction: str = "auto",
    fp07_model: str = "single_pole",
    goodman: bool = True,
    f_AA: float = 98.0,
    fit_method: str = "iterative",
    spectrum_model: str = "kraichnan",
    salinity: npt.ArrayLike | None = None,
    vehicle: str | None = None,
    _pre_loaded: dict[str, Any] | None = None,
) -> list[xr.Dataset]:
    """Compute chi from temperature gradient spectra (internal, no deprecation warning).

    ``_pre_loaded`` is a private hook accepting the dict produced by
    :func:`_load_therm_channels` (i.e. channels + ``therm`` + ``diff_gains`` +
    ``therm_cal``) so callers like ``perturb.pipeline`` can avoid the
    redundant NC reads when diss and chi share the same source.
    """
    from odas_tpw.chi.chi import _spectrum_func
    from odas_tpw.chi.l2_chi import process_l2_chi
    from odas_tpw.chi.l3_chi import process_l3_chi
    from odas_tpw.chi.l4_chi import process_l4_chi_epsilon, process_l4_chi_fit
    from odas_tpw.rsi.helpers import _build_l1data_from_channels, prepare_profiles
    from odas_tpw.scor160.io import L2Params, L3Params
    from odas_tpw.scor160.l2 import process_l2

    if diss_length is None:
        diss_length = 4 * fft_length
    if overlap is None:
        overlap = diss_length // 2

    # Load channels including thermistor data
    data = _pre_loaded if _pre_loaded is not None else _load_therm_channels(source)

    therm_names = [t[0] for t in data["therm"]]
    n_therm = len(therm_names)
    diff_gains = data.get("diff_gains", [0.94] * n_therm)
    therm_cal = data.get("therm_cal", [{}] * n_therm)

    if n_therm == 0:
        raise ValueError("No thermistor gradient channels found")

    fs_fast = data["fs_fast"]
    if vehicle is None:
        vehicle = data.get("vehicle", "")

    prepared = prepare_profiles(data, speed, direction, salinity, vehicle=vehicle)
    if prepared is None:
        return []
    (profiles_slow, speed_fast, P_fast, T_fast, sal_fast, fs_fast, _fs_slow, ratio, t_fast) = (
        prepared
    )

    # Pipeline parameters
    l2_params = L2Params(
        HP_cut=0.25,
        despike_sh=np.array([np.inf, 0.5, 0.04]),  # don't despike shear for chi
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

    grad_func, _ = _spectrum_func(spectrum_model)

    results = []
    for s_slow, e_slow in profiles_slow:
        s_fast = s_slow * ratio
        e_fast = min((e_slow + 1) * ratio, len(t_fast))

        sal_prof = sal_fast[s_fast:e_fast] if isinstance(sal_fast, np.ndarray) else sal_fast

        # Build L1Data with temp_fast
        l1 = _build_l1data_from_channels(
            data,
            s_fast,
            e_fast,
            speed_fast,
            P_fast,
            T_fast,
            direction,
            therm_list=data["therm"],
            diff_gains=diff_gains,
        )

        # L2: section selection and speed
        l2 = process_l2(l1, l2_params)

        # L2_chi: despike temperature, HP-filter vib, compute gradient
        l2_chi = process_l2_chi(l1, l2)

        # L3_chi: temperature gradient spectra
        l3_chi = process_l3_chi(
            l2_chi,
            l3_params,
            fp07_model=fp07_model,
            salinity=sal_prof,
            therm_cal=therm_cal,
        )

        if l3_chi.n_spectra == 0:
            continue

        # L4_chi: chi estimation
        if epsilon_ds is not None:
            l4_diss = _epsilon_ds_to_l4data(epsilon_ds)
            l4_chi = process_l4_chi_epsilon(
                l3_chi,
                l4_diss,
                spectrum_model=spectrum_model,
                f_AA=f_AA,
            )
        else:
            l4_chi = process_l4_chi_fit(
                l3_chi,
                spectrum_model=spectrum_model,
                fit_method=fit_method,
                f_AA=f_AA,
            )

        # Build output dataset
        ds = _build_chi_ds_from_pipeline(
            l3_chi,
            l4_chi,
            therm_names,
            fft_length=fft_length,
            diss_length=diss_length,
            overlap=overlap,
            fs_fast=fs_fast,
            fp07_model=fp07_model,
            spectrum_model=spectrum_model,
            fit_method=fit_method,
            f_AA=f_AA,
            grad_func=grad_func,
        )
        ds.attrs.update(data["metadata"])
        ds.attrs["history"] = f"Computed with microstructure-tpw on {datetime.now(UTC).isoformat()}"
        start_time = data["metadata"].get("start_time", "")
        t_units = f"seconds since {start_time}" if start_time else "seconds"
        ds.coords["t"].attrs.update(
            {
                "standard_name": "time",
                "long_name": "time of chi estimate",
                "units": t_units,
                "calendar": "standard",
                "axis": "T",
            }
        )
        results.append(ds)

    return results


def get_chi(
    source: PFile | str | Path,
    epsilon_ds: xr.Dataset | None = None,
    fft_length: int = 1024,
    diss_length: int | None = None,
    overlap: int | None = None,
    speed: float | None = None,
    direction: str = "down",
    fp07_model: str = "single_pole",
    goodman: bool = True,
    f_AA: float = 98.0,
    fit_method: str = "iterative",
    spectrum_model: str = "kraichnan",
    salinity: npt.ArrayLike | None = None,
) -> list[xr.Dataset]:
    """Compute chi from temperature gradient spectra.

    .. deprecated::
        Use :func:`odas_tpw.rsi.pipeline.run_pipeline` or the modular
        ``process_l2_chi`` → ``process_l3_chi`` → ``process_l4_chi_*``
        chain instead.

    Parameters
    ----------
    source : PFile, str, or Path
        Input data (same multi-source support as get_diss).
    epsilon_ds : xarray.Dataset or None
        If provided: Method 1 — use known epsilon from shear probes.
        If None: Method 2 — fit Batchelor spectrum for kB.
    fft_length : int
        FFT segment length [samples], default 1024.
    diss_length : int or None
        Dissipation window length [samples], default 4 * fft_length.
    overlap : int or None
        Window overlap [samples], default diss_length // 2.
    speed : float or None
        Fixed profiling speed [m/s]. If None, from dP/dt.
    direction : str
        'up' or 'down' for speed sign convention.
    fp07_model : str
        'single_pole' or 'double_pole' for FP07 transfer function.
    goodman : bool
        Apply Goodman coherent noise removal using accelerometers (default True).
    f_AA : float
        Anti-aliasing filter cutoff [Hz].
    fit_method : str
        Method 2 only: 'mle' (Ruddick et al. 2000) or 'iterative'
        (Peterson & Fer 2014).
    spectrum_model : str
        'batchelor' or 'kraichnan'.
    salinity : float or array_like or None
        Practical salinity [PSU]. If provided, uses gsw-based viscosity
        instead of visc35. Scalar or array matching slow time series.

    Returns
    -------
    list of xarray.Dataset, one per profile, with variables:
        chi         (probe, time) — thermal dissipation rate [K^2/s]
        epsilon_T   (probe, time) — epsilon from T (Method 2) or input epsilon
        kB          (probe, time) — Batchelor wavenumber [cpm]
        K_max_T     (probe, time) — integration limit [cpm]
        fom         (probe, time) — figure of merit (obs/model variance)
        K_max_ratio (probe, time) — K_max / kB (spectral resolution)
        speed       (time) — profiling speed [m/s]
        nu          (time) — kinematic viscosity [m^2/s]
        T_mean      (time) — mean temperature [deg C]
        P_mean      (time) — mean pressure [dbar]
        spec_gradT  (probe, freq, time) — temperature gradient spectra
        spec_batch  (probe, freq, time) — fitted Batchelor spectra
        spec_noise  (probe, freq, time) — noise spectra (per-probe)
        K           (freq, time) — wavenumber vectors
    """
    warnings.warn(
        "get_chi() is deprecated. Use run_pipeline() or the modular "
        "process_l2_chi → process_l3_chi → process_l4_chi_* chain "
        "instead. See odas_tpw.rsi.pipeline.run_pipeline().",
        DeprecationWarning,
        stacklevel=2,
    )
    return _compute_chi(
        source,
        epsilon_ds=epsilon_ds,
        fft_length=fft_length,
        diss_length=diss_length,
        overlap=overlap,
        speed=speed,
        direction=direction,
        fp07_model=fp07_model,
        goodman=goodman,
        f_AA=f_AA,
        fit_method=fit_method,
        spectrum_model=spectrum_model,
        salinity=salinity,
    )


# ---------------------------------------------------------------------------
# Pipeline-to-Dataset helpers
# ---------------------------------------------------------------------------


def _epsilon_ds_to_l4data(epsilon_ds: xr.Dataset) -> Any:
    """Convert old-format epsilon xr.Dataset to L4Data for chi computation.

    If the dataset already carries ``epsilonMean`` (written by
    ``processing.epsilon_combine.mk_epsilon_mean``), that QC-filtered geometric
    mean is used as ``epsi_final``.  Otherwise we fall back to a plain
    ``nanmean`` across the per-probe ``epsilon`` array — the historical
    behaviour, retained for inputs produced by the lower-level rsi pipeline
    that does not run the multi-probe combine step.
    """
    from odas_tpw.scor160.io import L4Data

    eps_vals = epsilon_ds["epsilon"].values
    if eps_vals.ndim == 2:
        n_sh = eps_vals.shape[0]
    else:
        n_sh = 1
        eps_vals = eps_vals[np.newaxis]

    if "epsilonMean" in epsilon_ds:
        epsi_final = np.asarray(epsilon_ds["epsilonMean"].values, dtype=np.float64)
    elif n_sh > 1:
        epsi_final = np.nanmean(eps_vals, axis=0)
    else:
        epsi_final = eps_vals[0]

    times = epsilon_ds.coords["t"].values
    n = len(times)

    return L4Data(
        time=times,
        pres=epsilon_ds["P_mean"].values if "P_mean" in epsilon_ds else np.zeros(n),
        pspd_rel=epsilon_ds["speed"].values if "speed" in epsilon_ds else np.zeros(n),
        section_number=np.ones(n),
        epsi=eps_vals,
        epsi_final=epsi_final,
        epsi_flags=np.zeros((n_sh, n)),
        fom=np.zeros((n_sh, n)),
        mad=np.zeros((n_sh, n)),
        kmax=np.zeros((n_sh, n)),
        method=np.zeros((n_sh, n)),
        var_resolved=np.zeros((n_sh, n)),
    )


def _build_chi_ds_from_pipeline(
    l3_chi,
    l4_chi,
    therm_names: list[str],
    *,
    fft_length: int,
    diss_length: int,
    overlap: int,
    fs_fast: float,
    fp07_model: str,
    spectrum_model: str,
    fit_method: str,
    f_AA: float,
    grad_func,
) -> xr.Dataset:
    """Build old-format xr.Dataset from L3ChiData + L4ChiData."""
    n_spec = l3_chi.n_spectra
    n_gradt = l3_chi.n_gradt
    n_freq = l3_chi.n_wavenumber

    # DOF
    num_ffts = 2 * (diss_length // fft_length) - 1
    dof_spec = 1.9 * num_ffts

    # Reconstruct fitted Batchelor spectra
    spec_batch = np.full((n_gradt, n_freq, n_spec), np.nan)
    for j in range(n_spec):
        for ci in range(n_gradt):
            kB = l4_chi.kB[ci, j]
            chi_val = l4_chi.chi[ci, j]
            if np.isfinite(kB) and np.isfinite(chi_val):
                spec_batch[ci, :, j] = grad_func(l3_chi.kcyc[:, j], kB, chi_val)

    # F vector (constant across windows) — broadcast view is read-only
    # but suffices for xarray storage and NetCDF serialization.
    F_const = l3_chi.freq
    F_out = np.broadcast_to(F_const[:, np.newaxis], (n_freq, n_spec))

    return _build_chi_dataset(
        chi_out=l4_chi.chi,
        eps_out=l4_chi.epsilon_T,
        kB_out=l4_chi.kB,
        K_max_out=l4_chi.K_max,
        fom_out=l4_chi.fom,
        K_max_ratio_out=l4_chi.K_max_ratio,
        speed_out=l3_chi.pspd_rel,
        nu_out=l3_chi.nu,
        P_out=l3_chi.pres,
        T_out=l3_chi.temp,
        t_out=l3_chi.time,
        spec_gradT=l3_chi.gradt_spec,
        spec_batch=spec_batch,
        spec_noise_out=l3_chi.noise_spec,
        K_out=l3_chi.kcyc,
        F_out=F_out,
        therm_names=therm_names,
        fft_length=fft_length,
        diss_length=diss_length,
        overlap=overlap,
        fs_fast=fs_fast,
        dof_spec=dof_spec,
        fp07_model=fp07_model,
        spectrum_model=spectrum_model,
        fit_method=fit_method,
        f_AA=f_AA,
    )


def _build_chi_dataset(
    *,
    chi_out: np.ndarray,
    eps_out: np.ndarray,
    kB_out: np.ndarray,
    K_max_out: np.ndarray,
    fom_out: np.ndarray,
    K_max_ratio_out: np.ndarray,
    speed_out: np.ndarray,
    nu_out: np.ndarray,
    P_out: np.ndarray,
    T_out: np.ndarray,
    t_out: np.ndarray,
    spec_gradT: np.ndarray,
    spec_batch: np.ndarray,
    spec_noise_out: np.ndarray,
    K_out: np.ndarray,
    F_out: np.ndarray,
    therm_names: list[str],
    fft_length: int,
    diss_length: int,
    overlap: int,
    fs_fast: float,
    dof_spec: float,
    fp07_model: str,
    spectrum_model: str,
    fit_method: str,
    f_AA: float,
) -> xr.Dataset:
    """Build an xarray Dataset from chi estimation output arrays."""
    from odas_tpw.rsi.helpers import _build_result_dataset

    variables = [
        (
            "chi",
            ["probe", "time"],
            chi_out,
            {
                "units": "K2 s-1",
                "long_name": "thermal variance dissipation rate",
            },
        ),
        (
            "epsilon_T",
            ["probe", "time"],
            eps_out,
            {
                "units": "W kg-1",
                "long_name": "TKE dissipation rate from temperature",
            },
        ),
        (
            "kB",
            ["probe", "time"],
            kB_out,
            {
                "units": "cpm",
                "long_name": "Batchelor wavenumber",
            },
        ),
        (
            "K_max_T",
            ["probe", "time"],
            K_max_out,
            {
                "units": "cpm",
                "long_name": "upper wavenumber integration limit for chi",
            },
        ),
        (
            "fom",
            ["probe", "time"],
            fom_out,
            {
                "units": "1",
                "long_name": "figure of merit (observed/model variance ratio)",
            },
        ),
        (
            "K_max_ratio",
            ["probe", "time"],
            K_max_ratio_out,
            {
                "units": "1",
                "long_name": "K_max / kB spectral resolution ratio",
            },
        ),
        (
            "speed",
            ["time"],
            speed_out,
            {
                "units": "m s-1",
                "long_name": "profiling speed",
            },
        ),
        (
            "nu",
            ["time"],
            nu_out,
            {
                "units": "m2 s-1",
                "long_name": "kinematic viscosity of sea water",
            },
        ),
        (
            "P_mean",
            ["time"],
            P_out,
            {
                "units": "dbar",
                "long_name": "mean sea water pressure",
                "standard_name": "sea_water_pressure",
                "positive": "down",
            },
        ),
        (
            "T_mean",
            ["time"],
            T_out,
            {
                "units": "degree_Celsius",
                "long_name": "mean sea water temperature",
                "standard_name": "sea_water_temperature",
            },
        ),
        (
            "spec_gradT",
            ["probe", "freq", "time"],
            spec_gradT,
            {
                "units": "K2 m-1",
                "long_name": "temperature gradient wavenumber spectrum (observed)",
            },
        ),
        (
            "spec_batch",
            ["probe", "freq", "time"],
            spec_batch,
            {
                "units": "K2 m-1",
                "long_name": "fitted Batchelor temperature gradient spectrum",
            },
        ),
        (
            "spec_noise",
            ["probe", "freq", "time"],
            spec_noise_out,
            {
                "units": "K2 m-1",
                "long_name": "FP07 electronics noise spectrum",
            },
        ),
        (
            "K",
            ["freq", "time"],
            K_out,
            {
                "units": "cpm",
                "long_name": "wavenumber (cycles per metre)",
            },
        ),
        (
            "F",
            ["freq", "time"],
            F_out,
            {
                "units": "Hz",
                "long_name": "frequency",
            },
        ),
    ]
    return _build_result_dataset(
        variables,
        therm_names,
        t_out,
        "thermistor probe name",
        {
            "Conventions": "CF-1.13, ACDD-1.3",
            "fft_length": fft_length,
            "diss_length": diss_length,
            "overlap": overlap,
            "fs_fast": fs_fast,
            "dof_spec": dof_spec,
            "fp07_model": fp07_model,
            "spectrum_model": spectrum_model,
            "fit_method": fit_method,
            "f_AA": f_AA,
        },
    )


# ---------------------------------------------------------------------------
# Channel loading helpers
# ---------------------------------------------------------------------------


def _extract_therm_cal(ch_cfg: dict[str, Any]) -> dict[str, float]:
    """Extract thermistor calibration parameters from PFile channel config."""
    cal = {}
    for key in ("e_b", "b", "g", "beta_1", "beta_2", "adc_fs", "adc_bits", "T_0"):
        val = ch_cfg.get(key)
        if val is not None:
            cal[key] = float(val)
    # Map 'g' to 'gain' for noise_thermchannel
    if "g" in cal:
        cal["gain"] = cal.pop("g")
    return cal


def _load_therm_channels(source: PFile | str | Path) -> dict[str, Any]:
    """Load thermistor gradient channels from any source.

    Looks for channels matching T*_dT* pattern (pre-emphasized temperature),
    or falls back to T1/T2 channels for first-difference gradient.
    """
    from odas_tpw.rsi.helpers import load_channels
    from odas_tpw.rsi.p_file import PFile

    # First load standard channels (convert TypedDict → plain dict for extra keys)
    data: dict[str, Any] = dict(load_channels(source))

    therm: list[tuple[str, np.ndarray]] = []
    diff_gains: list[float] = []

    if isinstance(source, PFile):
        pf = source
    elif Path(source).suffix.lower() == ".p":
        pf = PFile(source)
    else:
        pf = None

    therm_cal: list[dict[str, float]] = []

    if pf is not None:
        from odas_tpw.rsi.helpers import DT_PATTERN, T_PATTERN

        for name in sorted(pf._fast_channels):
            if DT_PATTERN.match(name):
                therm.append((name, pf.channels[name]))
                # Get diff_gain from config
                ch_cfg: dict = next(
                    (ch for ch in pf.config["channels"] if ch.get("name") == name),
                    {},
                )
                dg = ch_cfg.get("diff_gain", "0.94")
                diff_gains.append(float(dg))
                # Extract calibration from base channel (T1 for T1_dT1)
                m = re.match(r"^(\w+)_d\1$", name)
                base_name = m.group(1) if m else name
                base_cfg: dict = next(
                    (ch for ch in pf.config["channels"] if ch.get("name") == base_name),
                    {},
                )
                therm_cal.append(_extract_therm_cal(base_cfg))

        # If no pre-emphasized channels, use T channels with first-difference
        if not therm:
            for name in sorted(pf._fast_channels):
                if T_PATTERN.match(name):
                    # Compute gradient via first-difference
                    T_data = pf.channels[name]
                    therm.append((name, T_data))
                    diff_gains.append(0.94)
                    ch_cfg = next(
                        (ch for ch in pf.config["channels"] if ch.get("name") == name),
                        {},
                    )
                    therm_cal.append(_extract_therm_cal(ch_cfg))
    else:
        # NetCDF source — look for gradient channels or T channels
        import netCDF4 as nc

        from odas_tpw.rsi.helpers import DT_PATTERN, T_PATTERN

        ds = nc.Dataset(str(source), "r")

        for vname in sorted(ds.variables.keys()):
            var = ds.variables[vname]
            if var.dimensions == ("time_fast",):
                arr = var[:].data.astype(np.float64)
                if DT_PATTERN.match(vname):
                    therm.append((vname, arr))
                    diff_gains.append(0.94)

        if not therm:
            for vname in sorted(ds.variables.keys()):
                var = ds.variables[vname]
                if var.dimensions == ("time_fast",) and T_PATTERN.match(vname):
                    therm.append((vname, var[:].data.astype(np.float64)))
                    diff_gains.append(0.94)

        ds.close()

    data["therm"] = therm
    data["diff_gains"] = diff_gains
    data["therm_cal"] = therm_cal if therm_cal else [{}] * len(therm)
    return data


# ---------------------------------------------------------------------------
# File-level processing
# ---------------------------------------------------------------------------


def compute_chi_file(
    source_path: str | Path,
    output_dir: str | Path,
    **chi_kwargs: Any,
) -> list[Path]:
    """Compute chi for one file and write NetCDF output(s).

    Parameters
    ----------
    source_path : str or Path
        Input file.
    output_dir : str or Path
        Output directory.
    **chi_kwargs
        Keyword arguments passed to get_chi.

    Returns
    -------
    list of Path
        Paths to output files written.
    """
    from odas_tpw.rsi.helpers import write_profile_results

    source_path = Path(source_path)
    output_dir = Path(output_dir)
    results = _compute_chi(source_path, **chi_kwargs)
    return write_profile_results(results, source_path, output_dir, "chi")


def _compute_chi_one(args: tuple) -> tuple[str, int]:
    """Worker for parallel chi computation.

    args is (source_path, output_dir, kwargs) or
           (source_path, output_dir, kwargs, epsilon_dir).
    When epsilon_dir is provided, the worker opens the matching epsilon
    file itself (xr.Datasets cannot be pickled across process boundaries).
    """
    if len(args) == 4:
        source_path, output_dir, kwargs, epsilon_dir = args
    else:
        source_path, output_dir, kwargs = args
        epsilon_dir = None

    if epsilon_dir is not None:
        eps_file = Path(epsilon_dir) / f"{Path(source_path).stem}_eps.nc"
        if eps_file.exists():
            import xarray as xr

            eps_ds = xr.open_dataset(eps_file)
            try:
                kwargs = dict(kwargs)
                kwargs["epsilon_ds"] = eps_ds
                paths = compute_chi_file(source_path, output_dir, **kwargs)
            finally:
                eps_ds.close()
        else:
            paths = compute_chi_file(source_path, output_dir, **kwargs)
    else:
        paths = compute_chi_file(source_path, output_dir, **kwargs)
    return str(source_path), len(paths)
