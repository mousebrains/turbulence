"""Full VMP data processing pipeline.

.P file → L1Data → L2 → L3 (shear spectra) → L4 (epsilon)
                       → L3_chi (temperature gradient spectra) → L4_chi
                       → L5 (depth binning) → L6 (combine profiles)
"""

from __future__ import annotations

import logging
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xarray as xr

from odas_tpw.chi.l2_chi import L2ChiParams, process_l2_chi
from odas_tpw.chi.l3_chi import process_l3_chi
from odas_tpw.chi.l4_chi import L4ChiData, process_l4_chi_epsilon, process_l4_chi_fit
from odas_tpw.rsi.adapter import pfile_to_l1data
from odas_tpw.rsi.binning import bin_by_depth
from odas_tpw.rsi.combine import combine_profiles
from odas_tpw.scor160.io import L2Params, L3Data, L3Params, L4Data
from odas_tpw.scor160.l2 import process_l2
from odas_tpw.scor160.l3 import process_l3
from odas_tpw.scor160.l4 import process_l4

logger = logging.getLogger(__name__)


def run_pipeline(
    p_files: list[Path],
    output_dir: Path,
    *,
    # Profile detection
    P_min: float = 0.5,
    W_min: float = 0.3,
    direction: str = "down",
    min_duration: float = 7.0,
    speed: float | None = None,
    speed_tau: float = 1.5,
    # L2 params
    HP_cut: float = 0.25,
    despike_thresh: float = 8.0,
    # Epsilon spectral params
    fft_length: int = 256,
    diss_length: int | None = None,
    overlap: int | None = None,
    f_AA: float = 98.0,
    fit_order: int = 3,
    salinity: float | None = None,
    # Chi cleaning params
    despike_T_thresh: float = 10.0,
    # Chi spectral params
    chi_fft_length: int = 512,
    chi_diss_length: int | None = None,
    chi_overlap: int | None = None,
    fp07_model: str = "single_pole",
    spectrum_model: str = "kraichnan",
    fit_method: str = "iterative",
    # Chain selection
    compute_chi_epsilon: bool = True,
    compute_chi_fit: bool = False,
    # Binning
    bin_size: float = 1.0,
    # Output
    goodman: bool = True,
) -> Path:
    """Run the full processing pipeline.

    Per .P file, per profile:
    1. pfile_to_l1data → L1Data
    2. process_l2 → L2Data
    3. process_l3 → L3Data (shear spectra)
    4. process_l4 → L4Data (epsilon)
    5. process_l3_chi → L3ChiData (temperature gradient spectra)
    6. process_l4_chi_epsilon or process_l4_chi_fit → L4ChiData
    7. bin_by_depth → L5
    8. combine_profiles → L6

    Parameters
    ----------
    p_files : list of Path
        Input .P files.
    output_dir : Path
        Base output directory.
    (all other params documented in plan)

    Returns
    -------
    Path
        Output directory.
    """
    from odas_tpw.rsi.p_file import PFile
    from odas_tpw.rsi.profile import _smooth_fall_rate, get_profiles

    if diss_length is None:
        diss_length = 2 * fft_length
    if overlap is None:
        overlap = diss_length // 2
    if chi_diss_length is None:
        chi_diss_length = 3 * chi_fft_length
    if chi_overlap is None:
        chi_overlap = chi_diss_length // 2

    if diss_length != chi_diss_length:
        warnings.warn(
            f"Epsilon and chi use different dissipation lengths "
            f"(diss_length={diss_length}, chi_diss_length={chi_diss_length}). "
            f"Depth-binned estimates will have different vertical resolution.",
            stacklevel=2,
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for p_path in p_files:
        logger.info("=" * 60)
        logger.info(p_path.name)
        logger.info("=" * 60)

        pf = PFile(p_path)
        pfile_dir = output_dir / p_path.stem
        pfile_dir.mkdir(parents=True, exist_ok=True)

        # Detect profiles on slow-rate pressure
        P_slow = pf.channels.get("P_dP", pf.channels.get("P"))
        W_slow = _smooth_fall_rate(P_slow, pf.fs_slow, tau=speed_tau)
        profiles = get_profiles(
            P_slow,
            W_slow,
            pf.fs_slow,
            P_min=P_min,
            W_min=W_min,
            direction=direction,
            min_duration=min_duration,
        )

        if not profiles:
            logger.warning("No profiles detected")
            continue

        logger.info(f"{len(profiles)} profile(s) detected")

        all_binned = []
        profile_metadata = []

        for prof_idx, (s_slow, e_slow) in enumerate(profiles):
            prof_num = prof_idx + 1
            prof_dir = pfile_dir / f"profile_{prof_num:03d}"
            prof_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Profile {prof_num}: slow samples {s_slow}-{e_slow}")

            binned = _process_profile(
                pf,
                prof_slice=(s_slow, e_slow),
                prof_num=prof_num,
                prof_dir=prof_dir,
                p_path=p_path,
                speed=speed,
                direction=direction,
                speed_tau=speed_tau,
                l2_params=L2Params(
                    HP_cut=HP_cut,
                    despike_sh=np.array([despike_thresh, 0.5, 0.04]),
                    despike_A=np.array([np.inf, 0.5, 0.04]),
                    profile_min_W=W_min,
                    profile_min_P=P_min,
                    profile_min_duration=min_duration,
                    speed_tau=speed_tau,
                ),
                l3_params=L3Params(
                    fft_length=fft_length,
                    diss_length=diss_length,
                    overlap=overlap,
                    HP_cut=HP_cut,
                    fs_fast=0,  # placeholder, overridden in _process_profile
                    goodman=goodman,
                ),
                f_AA=f_AA,
                fit_order=fit_order,
                salinity=salinity,
                despike_T_thresh=despike_T_thresh,
                chi_fft_length=chi_fft_length,
                chi_diss_length=chi_diss_length,
                chi_overlap=chi_overlap,
                fp07_model=fp07_model,
                spectrum_model=spectrum_model,
                fit_method=fit_method,
                compute_chi_epsilon=compute_chi_epsilon,
                compute_chi_fit=compute_chi_fit,
                bin_size=bin_size,
            )

            if binned is not None:
                all_binned.append(binned)

            profile_metadata.append(
                {
                    "source_file": p_path.name,
                    "profile_number": prof_num,
                    "s_slow": s_slow,
                    "e_slow": e_slow,
                }
            )

        # Step 8: Combine profiles
        if all_binned:
            combined = combine_profiles(all_binned, profile_metadata)
            combined_path = pfile_dir / "L6_combined.nc"
            combined.to_netcdf(combined_path)
            logger.info(f"Combined: {combined_path}")

    logger.info(f"Pipeline complete. Output: {output_dir}")
    return output_dir


def _process_profile(
    pf,
    prof_slice: tuple[int, int],
    prof_num: int,
    prof_dir: Path,
    p_path: Path,
    *,
    speed: float | None,
    direction: str,
    speed_tau: float,
    l2_params: L2Params,
    l3_params: L3Params,
    f_AA: float,
    fit_order: int,
    salinity: float | None,
    despike_T_thresh: float,
    chi_fft_length: int,
    chi_diss_length: int,
    chi_overlap: int,
    fp07_model: str,
    spectrum_model: str,
    fit_method: str,
    compute_chi_epsilon: bool,
    compute_chi_fit: bool,
    bin_size: float,
) -> xr.Dataset | None:
    """Process a single profile: L1→L2→L3→L4→chi→binning→NetCDF."""
    s_slow, e_slow = prof_slice

    # Step 1: L1Data
    l1 = pfile_to_l1data(
        pf,
        profile_slice=(s_slow, e_slow),
        speed=speed,
        direction=direction,
        speed_tau=speed_tau,
    )

    # Step 2: L2
    l2 = process_l2(l1, l2_params)

    # Step 3: L3 shear spectra (update fs_fast from actual L1)
    l3_params = L3Params(
        fft_length=l3_params.fft_length,
        diss_length=l3_params.diss_length,
        overlap=l3_params.overlap,
        HP_cut=l3_params.HP_cut,
        fs_fast=l1.fs_fast,
        goodman=l3_params.goodman,
    )
    l3 = process_l3(l2, l1, l3_params)

    if l3.n_spectra == 0:
        logger.warning("No valid spectral windows")
        return None

    # Step 4: L4 epsilon
    l4 = process_l4(
        l3,
        temp=l3.temp,
        f_AA=f_AA,
        fit_order=fit_order,
    )
    logger.info(f"Epsilon: {l4.n_spectra} estimates")

    # Step 5: L2_chi cleaning + chi spectra (if temperature data available)
    l3_chi = None
    l4_chi_eps = None
    l4_chi_fit_result = None

    if l1.has_temp_fast:
        l2_chi_params = L2ChiParams(
            HP_cut=l2_params.HP_cut,
            despike_T=np.array([despike_T_thresh, 0.5, 0.04]),
        )
        l2_chi = process_l2_chi(l1, l2, l2_chi_params)

        l3_chi_params = L3Params(
            fft_length=chi_fft_length,
            diss_length=chi_diss_length,
            overlap=chi_overlap,
            HP_cut=l2_params.HP_cut,
            fs_fast=l1.fs_fast,
            goodman=l3_params.goodman,
        )

        try:
            l3_chi = process_l3_chi(
                l2_chi,
                l3_chi_params,
                fp07_model=fp07_model,
                salinity=salinity,
            )
        except (ValueError, RuntimeError) as e:
            logger.error(f"Chi spectra error: {e}")

        if l3_chi is not None and l3_chi.n_spectra > 0:
            # Step 6a: Chi from epsilon (Method 1)
            if compute_chi_epsilon and l4.n_spectra > 0:
                try:
                    l4_chi_eps = process_l4_chi_epsilon(
                        l3_chi,
                        l4,
                        spectrum_model=spectrum_model,
                        f_AA=f_AA,
                    )
                    n_valid = np.sum(np.isfinite(l4_chi_eps.chi_final))
                    logger.info(f"Chi (Method 1): {n_valid} valid estimates")
                except (ValueError, RuntimeError) as e:
                    logger.error(f"Chi epsilon error: {e}")

            # Step 6b: Chi from fit (Method 2)
            if compute_chi_fit:
                try:
                    l4_chi_fit_result = process_l4_chi_fit(
                        l3_chi,
                        spectrum_model=spectrum_model,
                        fit_method=fit_method,
                        f_AA=f_AA,
                    )
                    n_valid = np.sum(np.isfinite(l4_chi_fit_result.chi_final))
                    logger.info(f"Chi (Method 2): {n_valid} valid estimates")
                except (ValueError, RuntimeError) as e:
                    logger.error(f"Chi fit error: {e}")

    # Step 7: Depth binning
    binned_parts = []

    if l4.n_spectra > 0:
        eps_bin = bin_by_depth(
            l4.pres,
            {"epsilon": l4.epsi_final, "P_mean": l4.pres, "speed": l4.pspd_rel},
            bin_size=bin_size,
        )
        binned_parts.append(eps_bin)

    if l4_chi_eps is not None and l4_chi_eps.n_spectra > 0:
        chi_bin = bin_by_depth(
            l4_chi_eps.pres,
            {"chi": l4_chi_eps.chi_final},
            bin_size=bin_size,
        )
        binned_parts.append(chi_bin)

    binned = None
    if binned_parts:
        binned = binned_parts[0]
        for extra in binned_parts[1:]:
            binned = binned.merge(extra, join="outer")
        binned.attrs["profile_number"] = prof_num
        binned.attrs["source_file"] = p_path.name
        binned.attrs["bin_size"] = bin_size

        binned.to_netcdf(prof_dir / "L5_binned.nc")

    # Write per-level NetCDF
    time_ref = pf.start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_l4_epsilon(l4, l3, prof_dir / "L4_epsilon.nc", pf)

    if l4_chi_eps is not None:
        _write_l4_chi(l4_chi_eps, prof_dir / "L4_chi_epsilon.nc", time_ref)
    if l4_chi_fit_result is not None:
        _write_l4_chi(l4_chi_fit_result, prof_dir / "L4_chi_fit.nc", time_ref)

    return binned


def _write_l4_epsilon(l4: L4Data, l3: L3Data, path: Path, pf) -> None:
    """Write L4 epsilon to NetCDF."""
    n_shear = l4.n_shear
    probe_names = [f"sh{i + 1}" for i in range(n_shear)]
    time_ref = pf.start_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    ds = xr.Dataset(
        {
            "epsilon": (
                ["probe", "time"],
                l4.epsi,
                {
                    "units": "W kg-1",
                    "long_name": "TKE dissipation rate per shear probe",
                },
            ),
            "epsilon_final": (
                ["time"],
                l4.epsi_final,
                {
                    "units": "W kg-1",
                    "long_name": "TKE dissipation rate (best estimate)",
                },
            ),
            "fom": (
                ["probe", "time"],
                l4.fom,
                {
                    "units": "1",
                    "long_name": "figure of merit",
                },
            ),
            "mad": (
                ["probe", "time"],
                l4.mad,
                {
                    "units": "1",
                    "long_name": "mean absolute deviation ratio",
                },
            ),
            "kmax": (
                ["probe", "time"],
                l4.kmax,
                {
                    "units": "cpm",
                    "long_name": "upper integration wavenumber",
                },
            ),
            "var_resolved": (
                ["probe", "time"],
                l4.var_resolved,
                {
                    "units": "1",
                    "long_name": "fraction of variance resolved",
                },
            ),
            "sea_water_pressure": (
                ["time"],
                l4.pres,
                {
                    "units": "dbar",
                    "standard_name": "sea_water_pressure",
                    "long_name": "mean pressure per window",
                },
            ),
            "pspd_rel": (
                ["time"],
                l4.pspd_rel,
                {
                    "units": "m s-1",
                    "long_name": "profiling speed",
                },
            ),
        },
        coords={
            "probe": probe_names,
            "time": (
                ["time"],
                l4.time,
                {
                    "units": f"seconds since {time_ref}",
                    "standard_name": "time",
                    "calendar": "standard",
                },
            ),
        },
        attrs={
            "Conventions": "CF-1.13",
            "source_file": str(pf.filepath.name),
            "history": f"Pipeline on {datetime.now(UTC).isoformat()}",
        },
    )
    ds.to_netcdf(path)


def _write_l4_chi(l4_chi: L4ChiData, path: Path, time_ref: str) -> None:
    """Write L4 chi to NetCDF."""
    n_gradt = l4_chi.n_gradt
    probe_names = [f"T{i + 1}" for i in range(n_gradt)]

    ds = xr.Dataset(
        {
            "chi": (
                ["probe", "time"],
                l4_chi.chi,
                {
                    "units": "K2 s-1",
                    "long_name": "thermal variance dissipation rate per probe",
                },
            ),
            "chi_final": (
                ["time"],
                l4_chi.chi_final,
                {
                    "units": "K2 s-1",
                    "long_name": "thermal variance dissipation rate (best estimate)",
                },
            ),
            "epsilon_T": (
                ["probe", "time"],
                l4_chi.epsilon_T,
                {
                    "units": "W kg-1",
                    "long_name": "TKE dissipation rate from chi",
                },
            ),
            "kB": (
                ["probe", "time"],
                l4_chi.kB,
                {
                    "units": "cpm",
                    "long_name": "Batchelor wavenumber",
                },
            ),
            "K_max": (
                ["probe", "time"],
                l4_chi.K_max,
                {
                    "units": "cpm",
                    "long_name": "upper integration wavenumber",
                },
            ),
            "fom": (
                ["probe", "time"],
                l4_chi.fom,
                {
                    "units": "1",
                    "long_name": "figure of merit",
                },
            ),
            "K_max_ratio": (
                ["probe", "time"],
                l4_chi.K_max_ratio,
                {
                    "units": "1",
                    "long_name": "K_max to Batchelor wavenumber ratio",
                },
            ),
            "sea_water_pressure": (
                ["time"],
                l4_chi.pres,
                {
                    "units": "dbar",
                    "standard_name": "sea_water_pressure",
                    "long_name": "mean pressure per window",
                },
            ),
            "pspd_rel": (
                ["time"],
                l4_chi.pspd_rel,
                {
                    "units": "m s-1",
                    "long_name": "profiling speed",
                },
            ),
        },
        coords={
            "probe": probe_names,
            "time": (
                ["time"],
                l4_chi.time,
                {
                    "units": f"seconds since {time_ref}",
                    "standard_name": "time",
                    "calendar": "standard",
                },
            ),
        },
        attrs={
            "Conventions": "CF-1.13",
            "method": l4_chi.method,
            "history": f"Pipeline on {datetime.now(UTC).isoformat()}",
        },
    )
    ds.to_netcdf(path)
