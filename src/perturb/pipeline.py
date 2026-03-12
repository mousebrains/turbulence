# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Orchestration — per-file processing and full pipeline.

Reference: Code/process_P_files.m (233 lines), Code/mat2profile.m (170 lines)
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from perturb.config import merge_config, resolve_output_dir


def process_file(p_path: Path, config: dict, gps, output_dirs: dict) -> dict:
    """Process a single .p file through the full enhancement chain.

    Parameters
    ----------
    p_path : Path
        Path to the .p file.
    config : dict
        Full merged config (all sections).
    gps : GPSProvider
        GPS provider.
    output_dirs : dict
        Maps stage names to output directory Paths.

    Returns
    -------
    dict with output paths per stage.
    """
    from perturb.ct_align import ct_align
    from perturb.epsilon_combine import mk_epsilon_mean
    from perturb.fp07_cal import fp07_calibrate
    from rsi_python.dissipation import get_diss
    from rsi_python.p_file import PFile
    from rsi_python.profile import _smooth_fall_rate, get_profiles

    result = {"source": str(p_path), "profiles": [], "diss": [], "chi": []}

    try:
        pf = PFile(p_path)
    except Exception as exc:
        print(f"  ERROR loading {p_path.name}: {exc}")
        return result

    # ---- CTD fork (full file, both up and down) ----
    ctd_cfg = config.get("ctd", {})
    if ctd_cfg.get("enable", True) and "ctd" in output_dirs:
        try:
            from perturb.ctd import ctd_bin_file

            ctd_bin_file(
                pf, gps, output_dirs["ctd"],
                bin_width=ctd_cfg.get("bin_width", 0.5),
                T_name=ctd_cfg.get("T_name", "JAC_T"),
                C_name=ctd_cfg.get("C_name", "JAC_C"),
                variables=ctd_cfg.get("variables"),
                method=ctd_cfg.get("method", "mean"),
                diagnostics=ctd_cfg.get("diagnostics", False),
            )
        except Exception as exc:
            print(f"  ERROR CTD binning {p_path.name}: {exc}")

    # ---- Profile fork ----
    profiles_cfg = config.get("profiles", {})
    fp07_cfg = config.get("fp07", {})
    ct_cfg = config.get("ct", {})

    P_slow = pf.channels.get("P")
    if P_slow is None:
        print(f"  No pressure channel in {p_path.name}")
        return result

    W = _smooth_fall_rate(P_slow, pf.fs_slow)
    profiles = get_profiles(
        P_slow, W, pf.fs_slow,
        P_min=profiles_cfg.get("P_min", 0.5),
        W_min=profiles_cfg.get("W_min", 0.3),
        direction=profiles_cfg.get("direction", "down"),
        min_duration=profiles_cfg.get("min_duration", 7.0),
    )

    if not profiles:
        print(f"  No profiles in {p_path.name}")
        return result

    # FP07 calibration
    if fp07_cfg.get("calibrate", True):
        try:
            cal_result = fp07_calibrate(
                pf, profiles,
                reference=fp07_cfg.get("reference", "JAC_T"),
                order=fp07_cfg.get("order", 2),
                max_lag_seconds=fp07_cfg.get("max_lag_seconds", 10.0),
                must_be_negative=fp07_cfg.get("must_be_negative", True),
            )
            # Apply calibrated temperatures back to pf.channels
            for ch_name, cal_data in cal_result.get("channels", {}).items():
                pf.channels[ch_name] = cal_data
        except Exception as exc:
            print(f"  WARNING FP07 cal failed for {p_path.name}: {exc}")

    # CT alignment
    if ct_cfg.get("align", True):
        T_name = ct_cfg.get("T_name", "JAC_T")
        C_name = ct_cfg.get("C_name", "JAC_C")
        if T_name in pf.channels and C_name in pf.channels:
            try:
                C_aligned, _lag = ct_align(
                    pf.channels[T_name], pf.channels[C_name],
                    pf.fs_slow, profiles,
                )
                pf.channels[C_name] = C_aligned
            except Exception as exc:
                print(f"  WARNING CT align failed for {p_path.name}: {exc}")

    # Write per-profile NetCDFs
    from rsi_python.profile import extract_profiles

    if "profiles" in output_dirs:
        try:
            prof_paths = extract_profiles(pf, output_dirs["profiles"], **{
                k: v for k, v in profiles_cfg.items()
                if k in ("P_min", "W_min", "direction", "min_duration")
            })
            result["profiles"] = [str(p) for p in prof_paths]
        except Exception as exc:
            print(f"  ERROR extracting profiles {p_path.name}: {exc}")
            return result

    # Per-profile dissipation
    eps_cfg = config.get("epsilon", {})
    if "diss" in output_dirs and result["profiles"]:
        for prof_path in result["profiles"]:
            try:
                diss_results = get_diss(
                    prof_path,
                    **{k: v for k, v in eps_cfg.items()
                       if k not in (
                           "epsilon_minimum", "T_source", "T1_norm",
                           "T2_norm", "diagnostics",
                       )},
                )
                for ds in diss_results:
                    ds = mk_epsilon_mean(ds, eps_cfg.get("epsilon_minimum", 1e-13))
                    out_name = Path(prof_path).name
                    out_path = output_dirs["diss"] / out_name
                    ds.to_netcdf(out_path)
                    result["diss"].append(str(out_path))
            except Exception as exc:
                print(f"  ERROR diss for {Path(prof_path).name}: {exc}")

    # Per-profile chi (if enabled)
    chi_cfg = config.get("chi", {})
    if chi_cfg.get("enable", False) and "chi" in output_dirs and result["diss"]:
        try:
            from rsi_python.chi import get_chi
        except ImportError:
            pass
        else:
            for prof_path, diss_path in zip(result["profiles"], result["diss"]):
                try:
                    import xarray as xr

                    diss_ds = xr.open_dataset(diss_path)
                    chi_results = get_chi(
                        prof_path,
                        epsilon_ds=diss_ds,
                        **{k: v for k, v in chi_cfg.items()
                           if k not in ("enable", "diagnostics")},
                    )
                    diss_ds.close()
                    for chi_ds in chi_results:
                        out_name = Path(prof_path).name
                        out_path = output_dirs["chi"] / out_name
                        chi_ds.to_netcdf(out_path)
                        result["chi"].append(str(out_path))
                except Exception as exc:
                    print(f"  ERROR chi for {Path(prof_path).name}: {exc}")

    return result


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------


def run_trim(config: dict, p_files: list[Path] | None = None) -> list[Path]:
    """Trim corrupt final records from .p files."""
    from perturb.discover import find_p_files
    from perturb.trim import trim_p_file

    files_cfg = config.get("files", {})
    output_root = Path(files_cfg.get("output_root", "results/"))
    trim_dir = output_root / "trimmed"

    if p_files is None:
        root = files_cfg.get("p_file_root", "VMP/")
        pattern = files_cfg.get("p_file_pattern", "**/*.p")
        p_files = find_p_files(root, pattern)
    results = []
    for p in p_files:
        try:
            trimmed = trim_p_file(p, trim_dir)
            results.append(trimmed)
            print(f"  Trimmed: {trimmed.name}")
        except Exception as exc:
            print(f"  ERROR trimming {p.name}: {exc}")
    return results


def run_merge(config: dict) -> list[Path]:
    """Merge split .p files."""
    from perturb.discover import find_p_files
    from perturb.merge import find_mergeable_files, merge_p_files

    files_cfg = config.get("files", {})
    root = files_cfg.get("p_file_root", "VMP/")
    pattern = files_cfg.get("p_file_pattern", "**/*.p")
    output_root = Path(files_cfg.get("output_root", "results/"))
    merge_dir = output_root / "merged"

    p_files = find_p_files(root, pattern)
    chains = find_mergeable_files(p_files)
    results = []
    for chain in chains:
        try:
            merged = merge_p_files(chain, merge_dir)
            results.append(merged)
            print(f"  Merged {len(chain)} files -> {merged.name}")
        except Exception as exc:
            print(f"  ERROR merging: {exc}")
    return results


def _setup_output_dirs(config: dict) -> dict[str, Path]:
    """Set up versioned output directories based on config."""
    files_cfg = config.get("files", {})
    output_root = Path(files_cfg.get("output_root", "results/"))

    dirs = {}

    # Profiles directory — hash includes profiles + fp07 + ct + gps + bottom + top_trim
    profiles_params = merge_config("profiles", config.get("profiles"))
    dirs["profiles"] = resolve_output_dir(
        output_root, "profiles", "profiles", profiles_params,
    )

    # Diss directory — hash includes epsilon params + profiles upstream
    eps_params = merge_config("epsilon", config.get("epsilon"))
    dirs["diss"] = resolve_output_dir(
        output_root, "diss", "epsilon", eps_params,
        upstream=[("profiles", profiles_params)],
    )

    # Chi directory (if enabled)
    chi_cfg = config.get("chi", {})
    if chi_cfg.get("enable", False):
        chi_params = merge_config("chi", chi_cfg)
        dirs["chi"] = resolve_output_dir(
            output_root, "chi", "chi", chi_params,
            upstream=[("epsilon", eps_params)],
        )

    # CTD directory
    ctd_cfg = config.get("ctd", {})
    if ctd_cfg.get("enable", True):
        ctd_params = merge_config("ctd", ctd_cfg)
        dirs["ctd"] = resolve_output_dir(
            output_root, "ctd", "ctd", ctd_params,
        )

    return dirs


def run_pipeline(config: dict, p_files: list[Path] | None = None) -> None:
    """Run the full pipeline: discover -> trim -> merge -> process -> bin -> combo.

    Parameters
    ----------
    config : dict
        Full merged config (all sections).
    p_files : list of Path, optional
        Override file discovery with explicit file list.
    """
    from perturb.discover import find_p_files
    from perturb.gps import create_gps

    files_cfg = config.get("files", {})

    # Discover files
    if p_files is None:
        root = files_cfg.get("p_file_root", "VMP/")
        pattern = files_cfg.get("p_file_pattern", "**/*.p")
        p_files = find_p_files(root, pattern)

    if not p_files:
        print("No .p files found")
        return

    print(f"Found {len(p_files)} .p files")

    # Trim
    if files_cfg.get("trim", True):
        print("Trimming...")
        trimmed = run_trim(config, p_files)
        if trimmed:
            p_files = trimmed

    # Merge
    if files_cfg.get("merge", False):
        print("Merging...")
        run_merge(config)

    # Setup output directories
    output_dirs = _setup_output_dirs(config)

    # GPS provider
    gps_cfg = merge_config("gps", config.get("gps"))
    gps = create_gps(gps_cfg)

    # Parallel processing
    parallel_cfg = config.get("parallel", {})
    jobs = parallel_cfg.get("jobs", 1)
    if jobs == 0:
        jobs = os.cpu_count() or 1

    print(f"Processing {len(p_files)} files (jobs={jobs})...")

    if jobs == 1:
        for p_path in p_files:
            print(f"  Processing {p_path.name}...")
            process_file(p_path, config, gps, output_dirs)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(process_file, p, config, gps, output_dirs): p
                for p in p_files
            }
            for future in as_completed(futures):
                p = futures[future]
                try:
                    future.result()
                    print(f"  Done: {p.name}")
                except Exception as exc:
                    print(f"  ERROR {p.name}: {exc}")

    # Binning
    binning_cfg = config.get("binning", {})
    bin_method = binning_cfg.get("method", "depth")
    bin_width = binning_cfg.get("width", 1.0)
    aggregation = binning_cfg.get("aggregation", "mean")
    diagnostics = binning_cfg.get("diagnostics", False)

    output_root = Path(files_cfg.get("output_root", "results/"))

    from perturb.binning import bin_by_depth, bin_by_time, bin_chi, bin_diss

    # Bin profiles
    prof_ncs = sorted(output_dirs["profiles"].glob("*.nc")) if "profiles" in output_dirs else []
    if prof_ncs:
        print("Binning profiles...")
        binning_params = merge_config("binning", binning_cfg)
        prof_binned_dir = resolve_output_dir(
            output_root, "profiles_binned", "binning", binning_params,
            upstream=[("profiles", merge_config("profiles", config.get("profiles")))],
        )
        if bin_method == "depth":
            ds = bin_by_depth(prof_ncs, bin_width, aggregation, diagnostics)
        else:
            ds = bin_by_time(prof_ncs, bin_width, aggregation, diagnostics)
        if ds.data_vars:
            ds.to_netcdf(prof_binned_dir / "binned.nc")

    # Bin diss
    diss_ncs = sorted(output_dirs["diss"].glob("*.nc")) if "diss" in output_dirs else []
    if diss_ncs:
        print("Binning dissipation...")
        diss_width = binning_cfg.get("diss_width") or bin_width
        diss_agg = binning_cfg.get("diss_aggregation") or aggregation
        ds = bin_diss(diss_ncs, diss_width, diss_agg, bin_method, diagnostics)
        if ds.data_vars:
            diss_binned_dir = resolve_output_dir(
                output_root, "diss_binned", "binning",
                merge_config("binning", binning_cfg),
                upstream=[("epsilon", merge_config("epsilon", config.get("epsilon")))],
            )
            ds.to_netcdf(diss_binned_dir / "binned.nc")

    # Bin chi
    if "chi" in output_dirs:
        chi_ncs = sorted(output_dirs["chi"].glob("*.nc"))
        if chi_ncs:
            print("Binning chi...")
            chi_width = binning_cfg.get("chi_width") or binning_cfg.get("diss_width") or bin_width
            chi_agg = (
                binning_cfg.get("chi_aggregation")
                or binning_cfg.get("diss_aggregation")
                or aggregation
            )
            ds = bin_chi(chi_ncs, chi_width, chi_agg, bin_method, diagnostics)
            if ds.data_vars:
                chi_binned_dir = resolve_output_dir(
                    output_root, "chi_binned", "binning",
                    merge_config("binning", binning_cfg),
                    upstream=[("chi", merge_config("chi", config.get("chi")))],
                )
                ds.to_netcdf(chi_binned_dir / "binned.nc")

    # Combo assembly
    print("Assembling combo files...")
    # TODO: combo from binned directories once binned outputs are per-file

    print("Pipeline complete.")
