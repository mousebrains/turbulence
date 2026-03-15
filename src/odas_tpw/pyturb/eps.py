"""pyturb-cli eps: compute epsilon and temperature gradient spectra."""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from odas_tpw.pyturb._compat import (
    check_overwrite,
    compute_window_parameters,
    load_auxiliary,
    rename_eps_dataset,
)

logger = logging.getLogger(__name__)


def _process_one_file(
    filepath: Path,
    output_dir: Path,
    *,
    fft_length: int,
    diss_length: int,
    overlap: int,
    direction: str,
    min_speed: float,
    min_profile_pressure: float,
    peaks_height: float,
    peaks_distance: int,
    peaks_prominence: float,
    despike_passes: int,
    salinity: float,
    overwrite: bool,
    goodman: bool = False,
    aux_path: str | None = None,
    aux_kwargs: dict | None = None,
    pressure_smoothing: float = 0.25,
) -> list[Path]:
    """Process a single file: detect profiles, compute epsilon + gradT spectra."""
    from odas_tpw.pyturb._profind import find_profiles_peaks
    from odas_tpw.rsi.adapter import nc_to_l1data, pfile_to_l1data
    from odas_tpw.rsi.p_file import PFile
    from odas_tpw.scor160.io import L2Params, L3Params

    suffix = filepath.suffix.lower()
    stem = filepath.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load auxiliary data if provided
    aux_data = None
    if aux_path:
        aux_data = load_auxiliary(aux_path, **(aux_kwargs or {}))

    l2_params = L2Params(
        HP_cut=0.25,
        despike_sh=np.array([8.0, 0.5, 0.04]),
        despike_A=np.array([np.inf, 0.5, 0.04]),
        profile_min_W=min_speed,
        profile_min_P=min_profile_pressure,
        profile_min_duration=10.0,
        speed_tau=1.5,
    )

    l3_params = L3Params(
        fft_length=fft_length,
        diss_length=diss_length,
        overlap=overlap,
        HP_cut=0.25,
        fs_fast=512.0,  # will be overwritten below
        goodman=goodman,
    )

    output_paths: list[Path] = []

    if suffix == ".p":
        pf = PFile(filepath)
        fs_fast = pf.fs_fast
        fs_slow = pf.fs_slow
        l3_params.fs_fast = fs_fast

        # Detect profiles using peak-finding
        P_slow = pf.channels.get("P_dP", pf.channels.get("P"))
        if P_slow is None:
            logger.error(f"{filepath.name}: no pressure channel")
            return []

        profiles = find_profiles_peaks(
            P_slow,
            fs_slow,
            direction=direction,
            min_pressure=min_profile_pressure,
            peaks_height=peaks_height,
            peaks_distance=peaks_distance,
            peaks_prominence=peaks_prominence,
            min_speed=min_speed,
            smoothing_tau=pressure_smoothing,
        )

        if not profiles:
            logger.warning(f"{filepath.name}: no profiles detected")
            return []

        for pi, (s_slow, e_slow) in enumerate(profiles):
            out_path = output_dir / f"{stem}_p{pi:04d}.nc"
            if not check_overwrite(out_path, overwrite):
                logger.info(f"Skipping {out_path.name}: exists")
                continue

            # Build L1Data for this profile
            l1 = pfile_to_l1data(pf, profile_slice=(s_slow, e_slow), direction=direction)
            ds = _process_profile(l1, l2_params, l3_params, salinity, aux_data)
            if ds is not None:
                ds.attrs["source_file"] = filepath.name
                ds.attrs["profile_index"] = pi
                ds.to_netcdf(out_path)
                output_paths.append(out_path)
                logger.info(f"{out_path.name}: {ds.sizes.get('time', 0)} estimates")

    elif suffix == ".nc":
        # Load from NetCDF (may be full-record or single-profile)
        l1 = nc_to_l1data(filepath)
        fs_fast = l1.fs_fast
        fs_slow = l1.fs_slow
        l3_params.fs_fast = fs_fast

        if l1.pres.size == 0:
            logger.error(f"{filepath.name}: no pressure data")
            return []

        # If pressure range suggests a single profile, process directly
        pres_range = np.nanmax(l1.pres) - np.nanmin(l1.pres)
        if pres_range < peaks_height * 2:
            # Single profile — process entire file
            out_path = output_dir / f"{stem}_p0000.nc"
            if not check_overwrite(out_path, overwrite):
                return []
            ds = _process_profile(l1, l2_params, l3_params, salinity, aux_data)
            if ds is not None:
                ds.attrs["source_file"] = filepath.name
                ds.attrs["profile_index"] = 0
                ds.to_netcdf(out_path)
                output_paths.append(out_path)
        else:
            # Multiple profiles — use peak-finding on slow-rate pressure
            if l1.pres_slow.size > 0:
                P_for_peaks = l1.pres_slow
                fs_peaks = fs_slow if fs_slow > 0 else fs_fast / 8
            else:
                P_for_peaks = l1.pres
                fs_peaks = fs_fast

            profiles = find_profiles_peaks(
                P_for_peaks,
                fs_peaks,
                direction=direction,
                min_pressure=min_profile_pressure,
                peaks_height=peaks_height,
                peaks_distance=peaks_distance,
                peaks_prominence=peaks_prominence,
                min_speed=min_speed,
                smoothing_tau=pressure_smoothing,
            )
            if not profiles:
                logger.warning(f"{filepath.name}: no profiles detected")
                return []

            ratio = round(fs_fast / fs_peaks) if fs_peaks < fs_fast else 1
            for pi, (s_slow, e_slow) in enumerate(profiles):
                out_path = output_dir / f"{stem}_p{pi:04d}.nc"
                if not check_overwrite(out_path, overwrite):
                    continue

                # Slice L1Data
                s_fast = s_slow * ratio
                e_fast = min((e_slow + 1) * ratio, l1.n_time)

                from odas_tpw.scor160.io import L1Data

                l1_prof = L1Data(
                    time=l1.time[s_fast:e_fast],
                    pres=l1.pres[s_fast:e_fast],
                    shear=l1.shear[:, s_fast:e_fast] if l1.shear.size > 0 else l1.shear,
                    vib=l1.vib[:, s_fast:e_fast] if l1.vib.size > 0 else l1.vib,
                    vib_type=l1.vib_type,
                    fs_fast=l1.fs_fast,
                    f_AA=l1.f_AA,
                    vehicle=l1.vehicle,
                    profile_dir=direction,
                    time_reference_year=l1.time_reference_year,
                    pspd_rel=l1.pspd_rel[s_fast:e_fast] if l1.pspd_rel.size > 0 else l1.pspd_rel,
                    temp=l1.temp[s_fast:e_fast] if l1.temp.size >= e_fast else l1.temp,
                    temp_fast=(
                        l1.temp_fast[:, s_fast:e_fast]
                        if l1.temp_fast.ndim == 2 and l1.temp_fast.shape[1] >= e_fast
                        else l1.temp_fast
                    ),
                    diff_gains=l1.diff_gains,
                )

                ds = _process_profile(l1_prof, l2_params, l3_params, salinity, aux_data)
                if ds is not None:
                    ds.attrs["source_file"] = filepath.name
                    ds.attrs["profile_index"] = pi
                    ds.to_netcdf(out_path)
                    output_paths.append(out_path)
                    logger.info(f"{out_path.name}: {ds.sizes.get('time', 0)} estimates")
    else:
        logger.error(f"{filepath.name}: unsupported format {suffix}")

    return output_paths


def _process_profile(l1, l2_params, l3_params, salinity, aux_data):
    """Run the shear + chi processing chain on a single profile L1Data."""
    from odas_tpw.chi.l2_chi import process_l2_chi
    from odas_tpw.chi.l3_chi import process_l3_chi
    from odas_tpw.scor160.l2 import process_l2
    from odas_tpw.scor160.l3 import process_l3
    from odas_tpw.scor160.l4 import process_l4

    try:
        l2 = process_l2(l1, l2_params)
        l3 = process_l3(l2, l1, l3_params)
        l4 = process_l4(l3, temp=l3.temp, f_AA=l1.f_AA)
    except Exception as e:
        logger.error(f"Shear processing failed: {e}")
        return None

    if l4.n_spectra == 0:
        return None

    # Chi chain (temperature gradient spectra)
    l3_chi = None
    if l1.has_temp_fast:
        try:
            l2_chi = process_l2_chi(l1, l2)
            l3_chi = process_l3_chi(l2_chi, l3_params, salinity=salinity)
        except Exception as e:
            logger.warning(f"Chi processing failed (continuing without gradT): {e}")

    return rename_eps_dataset(
        l4,
        l3,
        l3_chi,
        aux_data=aux_data,
        fs_fast=l1.fs_fast,
        salinity=salinity,
    )


def run_eps(args: argparse.Namespace) -> None:
    """Execute the eps subcommand."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect input files
    paths: list[Path] = []
    for pattern in args.files:
        p = Path(pattern)
        if p.is_file():
            paths.append(p)
        else:
            paths.extend(sorted(Path(".").glob(pattern)))

    if not paths:
        logger.error("No input files found")
        return

    # Convert seconds to samples using Jesse's convention (no power-of-2).
    # Use 512 Hz as default (standard VMP); will be overridden per-file.
    fs_default = 512.0
    fft_length, diss_length, _fft_ovlp, diss_overlap = compute_window_parameters(
        args.fft_len, args.diss_len, fs_default
    )

    common_kwargs = dict(
        fft_length=fft_length,
        diss_length=diss_length,
        overlap=diss_overlap,
        goodman=args.goodman,
        direction=args.direction,
        min_speed=args.min_speed,
        min_profile_pressure=args.min_profile_pressure,
        peaks_height=args.peaks_height,
        peaks_distance=args.peaks_distance,
        peaks_prominence=args.peaks_prominence,
        despike_passes=args.despike_passes,
        salinity=args.salinity,
        overwrite=args.overwrite,
        pressure_smoothing=args.pressure_smoothing,
    )

    if args.aux:
        common_kwargs["aux_path"] = args.aux
        common_kwargs["aux_kwargs"] = {
            "lat_var": args.aux_lat,
            "lon_var": args.aux_lon,
            "temp_var": args.aux_temp,
            "sal_var": args.aux_sal,
            "dens_var": args.aux_dens,
        }

    if args.n_workers <= 1:
        for filepath in paths:
            try:
                out_paths = _process_one_file(filepath, output_dir, **common_kwargs)
                for op in out_paths:
                    print(f"  {op.name}")
            except Exception as e:
                logger.error(f"{filepath.name}: {e}")
    else:
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            futures = {
                pool.submit(_process_one_file, fp, output_dir, **common_kwargs): fp for fp in paths
            }
            for future in as_completed(futures):
                fp = futures[future]
                try:
                    out_paths = future.result()
                    for op in out_paths:
                        print(f"  {op.name}")
                except Exception as e:
                    logger.error(f"{fp.name}: {e}")
