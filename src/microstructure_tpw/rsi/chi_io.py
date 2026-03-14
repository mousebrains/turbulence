# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Convenience wrappers for chi computation from PFile / NetCDF sources.

These functions bridge rsi instrument I/O with the pure chi
computation in chi.chi.  They handle data loading, profile
detection, and file-level orchestration.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import xarray as xr

if TYPE_CHECKING:
    from microstructure_tpw.rsi.p_file import PFile


def get_chi(
    source: PFile | str | Path,
    epsilon_ds: xr.Dataset | None = None,
    fft_length: int = 512,
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

    Parameters
    ----------
    source : PFile, str, or Path
        Input data (same multi-source support as get_diss).
    epsilon_ds : xarray.Dataset or None
        If provided: Method 1 — use known epsilon from shear probes.
        If None: Method 2 — fit Batchelor spectrum for kB.
    fft_length : int
        FFT segment length [samples], default 512.
    diss_length : int or None
        Dissipation window length [samples], default 3 * fft_length.
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
    from microstructure_tpw.chi.chi import _compute_profile_chi

    if diss_length is None:
        diss_length = 3 * fft_length
    if overlap is None:
        overlap = diss_length // 2

    # Load channels including thermistor data
    data = _load_therm_channels(source)

    therm_names = [t[0] for t in data["therm"]]
    therm_arrays = [t[1] for t in data["therm"]]
    n_therm = len(therm_arrays)
    diff_gains = data.get("diff_gains", [0.94] * n_therm)
    therm_cal = data.get("therm_cal", [{}] * n_therm)

    # Accelerometer arrays for scalar Goodman cleaning
    accel_arrays = [a[1] for a in data.get("accel", [])]
    if not goodman:
        accel_arrays = []

    if n_therm == 0:
        raise ValueError("No thermistor gradient channels found")

    # Profile detection, speed, P/T interpolation, salinity
    from microstructure_tpw.rsi.helpers import prepare_profiles
    from microstructure_tpw.rsi.profile import _VEHICLE_TAU

    vehicle = data.get("vehicle", "")
    tau = _VEHICLE_TAU.get(vehicle, 1.5)
    prepared = prepare_profiles(data, speed, direction, salinity, tau=tau)
    if prepared is None:
        return []
    (profiles_slow, speed_fast, P_fast, T_fast, sal_fast, fs_fast, _fs_slow, ratio, t_fast) = (
        prepared
    )

    f_AA_eff = 0.9 * f_AA

    results = []
    for s_slow, e_slow in profiles_slow:
        s_fast = s_slow * ratio
        e_fast = min((e_slow + 1) * ratio, len(t_fast))

        therm_prof = [arr[s_fast:e_fast] for arr in therm_arrays]
        accel_prof = [arr[s_fast:e_fast] for arr in accel_arrays] if accel_arrays else None
        p_prof = P_fast[s_fast:e_fast]
        t_prof = T_fast[s_fast:e_fast]
        spd_prof = speed_fast[s_fast:e_fast]
        time_prof = t_fast[s_fast:e_fast]

        sal_prof = sal_fast[s_fast:e_fast] if isinstance(sal_fast, np.ndarray) else sal_fast

        ds = _compute_profile_chi(
            therm_prof,
            therm_names,
            diff_gains,
            p_prof,
            t_prof,
            spd_prof,
            time_prof,
            fs_fast,
            fft_length,
            diss_length,
            overlap,
            f_AA_eff,
            fp07_model,
            spectrum_model,
            fit_method,
            epsilon_ds,
            salinity=sal_prof,
            therm_cal=therm_cal,
            accel_arrays=accel_prof,
        )
        ds.attrs.update(data["metadata"])
        ds.attrs["history"] = (
            f"Computed with microstructure-tpw on {datetime.now(UTC).isoformat()}"
        )
        # CF time coordinate attributes
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
    from microstructure_tpw.rsi.helpers import load_channels
    from microstructure_tpw.rsi.p_file import PFile

    # First load standard channels
    data = load_channels(source)

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
        # Look for pre-emphasized channels (T1_dT1, T2_dT2)
        dT_re = re.compile(r"^T\d+_dT\d+$")
        T_re = re.compile(r"^T\d+$")

        for name in sorted(pf._fast_channels):
            if dT_re.match(name):
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
                if T_re.match(name):
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

        ds = nc.Dataset(str(source), "r")
        dT_re = re.compile(r"^T\d+_dT\d+$")
        T_re = re.compile(r"^T\d+$")

        for vname in sorted(ds.variables.keys()):
            var = ds.variables[vname]
            if var.dimensions == ("time_fast",):
                arr = var[:].data.astype(np.float64)
                if dT_re.match(vname):
                    therm.append((vname, arr))
                    diff_gains.append(0.94)

        if not therm:
            for vname in sorted(ds.variables.keys()):
                var = ds.variables[vname]
                if var.dimensions == ("time_fast",) and T_re.match(vname):
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
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = get_chi(source_path, **chi_kwargs)

    output_paths = []
    for i, ds in enumerate(results):
        if len(results) == 1:
            out_name = f"{source_path.stem}_chi.nc"
        else:
            out_name = f"{source_path.stem}_prof{i + 1:03d}_chi.nc"
        out_path = output_dir / out_name
        ds.to_netcdf(out_path)
        output_paths.append(out_path)
        print(
            f"  {out_path.name}: {ds.sizes['time']} estimates, "
            f"P={float(ds.P_mean.min()):.0f}-{float(ds.P_mean.max()):.0f} dbar"
        )

    return output_paths


def _compute_chi_one(args: tuple) -> tuple[str, int]:
    """Worker for parallel chi computation."""
    source_path, output_dir, kwargs = args
    paths = compute_chi_file(source_path, output_dir, **kwargs)
    return str(source_path), len(paths)
