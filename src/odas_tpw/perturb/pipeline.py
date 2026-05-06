# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Orchestration — per-file processing and full pipeline.

Reference: Code/process_P_files.m (233 lines), Code/mat2profile.m (170 lines)
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from odas_tpw.perturb.config import merge_config, resolve_output_dir
from odas_tpw.perturb.logging_setup import (
    current_run_stamp,
    init_worker_logging,
    stage_log,
)

logger = logging.getLogger(__name__)


def _upstream_for(stage: str, config: dict) -> list[tuple[str, dict]]:
    """Return the full upstream parameter chain for *stage*.

    Each downstream stage's ``.params_sha256_*`` signature is the hash of
    the stage's own params plus every ancestor's params, so two runs that
    differ only in a deep upstream knob still resolve to different output
    directories.  Ancestors are listed deepest-first only by convention
    (the canonicaliser sorts on the section name anyway).

    Stages:

    * ``profiles``           — no upstream
    * ``diss``               — profiles
    * ``chi``                — epsilon, profiles  (chi → diss → profiles)
    * ``ctd``                — no upstream (full-file, doesn't use profiles)
    * ``profiles_binned``    — profiles
    * ``diss_binned``        — epsilon, profiles
    * ``chi_binned``         — chi, epsilon, profiles
    """
    profiles_p = merge_config("profiles", config.get("profiles"))
    eps_p = merge_config("epsilon", config.get("epsilon"))
    chi_p = merge_config("chi", config.get("chi"))
    ctd_p = merge_config("ctd", config.get("ctd"))

    chains: dict[str, list[tuple[str, dict]]] = {
        "profiles": [],
        "diss": [("profiles", profiles_p)],
        "chi": [("epsilon", eps_p), ("profiles", profiles_p)],
        "ctd": [],
        "profiles_binned": [("profiles", profiles_p)],
        "diss_binned": [("epsilon", eps_p), ("profiles", profiles_p)],
        "chi_binned": [("chi", chi_p), ("epsilon", eps_p), ("profiles", profiles_p)],
        # Combo stages have no own params — their hash is the binning step
        # plus everything upstream of that step.  Including ``binning``
        # itself in the chain matches the logical dependency, even though
        # it duplicates the same dict that gets passed as ``params``.
        "combo": [("profiles", profiles_p)],
        "diss_combo": [("epsilon", eps_p), ("profiles", profiles_p)],
        "chi_combo": [("chi", chi_p), ("epsilon", eps_p), ("profiles", profiles_p)],
        "ctd_combo": [("ctd", ctd_p)],
    }
    return chains[stage]


_TIME_SCALAR_ATTRS: dict[str, dict[str, str]] = {
    "stime": {
        "standard_name": "time",
        "long_name": "profile start time",
        "units_metadata": "leap_seconds: utc",
        "axis": "T",
    },
    "etime": {
        "standard_name": "time",
        "long_name": "profile end time",
        "units_metadata": "leap_seconds: utc",
    },
}
_LATLON_SCALAR_ATTRS: dict[str, dict[str, str]] = {
    "lat": {
        "units": "degrees_north",
        "standard_name": "latitude",
        "long_name": "profile latitude",
        "axis": "Y",
    },
    "lon": {
        "units": "degrees_east",
        "standard_name": "longitude",
        "long_name": "profile longitude",
        "axis": "X",
    },
}


def _scalars_to_dataarrays(scalars: dict[str, float]) -> dict:
    """Build xr.DataArray scalars matching what xr.open_dataset would yield.

    Returns ``{name: xr.DataArray}`` for whichever of ``lat/lon/stime/etime``
    are present in *scalars*. Time scalars are decoded to ``datetime64[ns]``
    with the encoding hints needed for round-trip consistency, exactly
    matching the open_dataset path that previously fed _copy_profile_scalars.
    """
    import numpy as np
    import xarray as xr

    out: dict = {}
    for name in ("stime", "etime"):
        if name in scalars:
            secs = float(scalars[name])
            # np.datetime64 rejects np.int64 — must be a Python int.
            # round(float) already returns a Python int, no extra cast needed.
            ns = round(secs * 1e9)
            arr = xr.DataArray(np.datetime64(ns, "ns"), attrs=_TIME_SCALAR_ATTRS[name])
            arr.encoding = {
                "units": "seconds since 1970-01-01",
                "calendar": "standard",
                "dtype": "float64",
            }
            out[name] = arr
    for name in ("lat", "lon"):
        if name in scalars:
            arr = xr.DataArray(np.float64(scalars[name]), attrs=_LATLON_SCALAR_ATTRS[name])
            out[name] = arr
    return out


def _copy_profile_scalars(
    prof_path: str | Path,
    target_ds,
    scalars_cache: dict[str, dict[str, float]] | None = None,
) -> None:
    """Copy CF §9 profile scalars (lat, lon, stime, etime) onto *target_ds*.

    The diss/chi per-profile NetCDFs share the same profile bounds as the
    source profile NetCDF, so they can carry the same lat/lon/time scalars.
    Downstream :func:`bin_by_depth` picks them up to populate per-profile
    1-D variables in the binned/combo output, which in turn unlocks ACDD
    ``geospatial_lat/lon_min/max`` and ``time_coverage_start/end``.

    *scalars_cache* maps profile path → raw scalar dict (as produced by
    :func:`extract_profiles` with ``return_scalars=True``). When supplied
    and the path is present, the scalars are reconstructed in memory and
    no NetCDF re-open is needed — saves ~10 ms * 166 profiles per file.
    """
    if scalars_cache is not None:
        scalars = scalars_cache.get(str(prof_path))
        if scalars is not None:
            for name, da in _scalars_to_dataarrays(scalars).items():
                target_ds[name] = da
            return

    import xarray as xr

    try:
        with xr.open_dataset(prof_path) as src:
            for sname in ("lat", "lon", "stime", "etime"):
                if sname in src.data_vars and src[sname].ndim == 0:
                    target_ds[sname] = src[sname]
    except Exception:
        # Profile file missing scalars — older runs / external NCs, fine.
        pass


def _adjust_profile_bounds(
    profiles: list[tuple[int, int]],
    pf,
    top_trim_cfg: dict,
    bottom_cfg: dict,
    file_label: str,
) -> list[tuple[int, int]]:
    """Push each profile's start forward (top_trim) and/or end backward (bottom).

    Both stages are skipped when their ``enable`` flag is False, so this
    returns the input list unchanged in that case.

    Top-trim removes propeller-wash / surface-instability samples by
    measuring shear/acceleration variance binned by depth and finding the
    depth where variance settles below a quantile threshold. Bottom-crash
    detection scans accelerometer variance from the deepest bin upward
    and flags a vibration spike near the seafloor.

    Both adjustments are clipped so the returned bounds always lie inside
    the original profile (no shrinking past the original end / before the
    original start). Any per-profile failure is logged and that profile
    is left unchanged — the rest of the file still processes.
    """
    import numpy as np

    do_top = top_trim_cfg.get("enable", False)
    do_bottom = bottom_cfg.get("enable", False)
    if not (do_top or do_bottom):
        return profiles

    P_slow = pf.channels.get("P")
    if P_slow is None:
        return profiles

    ratio = round(pf.fs_fast / pf.fs_slow)
    P_fast = pf.channels.get("P_fast")
    # Many .p files only carry P on the slow channel — interpolate to fast
    # rate when needed by repeating each slow sample ``ratio`` times.
    if P_fast is None or len(P_fast) != len(pf.t_fast):
        P_fast = np.repeat(P_slow, ratio)
        if len(P_fast) > len(pf.t_fast):
            P_fast = P_fast[: len(pf.t_fast)]
        elif len(P_fast) < len(pf.t_fast):
            P_fast = np.concatenate([P_fast, np.full(len(pf.t_fast) - len(P_fast), P_fast[-1])])

    top_kwargs = {k: v for k, v in top_trim_cfg.items() if k != "enable"}
    bottom_kwargs = {k: v for k, v in bottom_cfg.items() if k != "enable"}

    if do_top:
        from odas_tpw.processing.top_trim import compute_trim_depth
    if do_bottom:
        from odas_tpw.processing.bottom import detect_bottom_crash

    adjusted: list[tuple[int, int]] = []
    for pi, (s_slow, e_slow) in enumerate(profiles, 1):
        new_s = s_slow
        new_e = e_slow
        s_fast = s_slow * ratio
        e_fast = (e_slow + 1) * ratio
        depth_seg = P_fast[s_fast:e_fast]

        if do_top:
            try:
                fast_channels: dict = {}
                for name in ("sh1", "sh2", "Ax", "Ay"):
                    if name in pf.channels and pf.is_fast(name):
                        fast_channels[name] = pf.channels[name][s_fast:e_fast]
                trim_depth = compute_trim_depth(depth_seg, fast_channels, **top_kwargs)
                if trim_depth is not None and len(depth_seg) > 0:
                    # Find first slow index where pressure exceeds trim_depth.
                    P_seg = P_slow[s_slow : e_slow + 1]
                    above = np.where(P_seg >= trim_depth)[0]
                    if len(above) > 0:
                        push = int(above[0])
                        candidate = s_slow + push
                        if candidate < e_slow:
                            new_s = candidate
                            logger.info(
                                "%s prof%d: top_trim pushed start %.2f -> %.2f dbar",
                                file_label,
                                pi,
                                float(P_slow[s_slow]),
                                float(P_slow[new_s]),
                            )
            except Exception as exc:
                logger.warning("%s prof%d top_trim failed: %s", file_label, pi, exc)

        if do_bottom:
            try:
                # Build the vibration-channels dict from whatever fast accel/gyro
                # axes this instrument has. VMP-250 has Ax+Ay; a 3-axis IMU
                # would also contribute Az; a different platform might supply a
                # pre-aggregated ``vibration_rms`` channel.
                vibration: dict = {}
                for name in ("Ax", "Ay", "Az"):
                    ch = pf.channels.get(name)
                    if ch is not None and pf.is_fast(name):
                        vibration[name] = ch[s_fast:e_fast]
                if vibration:
                    bottom_depth = detect_bottom_crash(
                        depth_seg,
                        vibration,
                        pf.fs_fast,
                        **bottom_kwargs,
                    )
                    if bottom_depth is not None:
                        P_seg = P_slow[new_s : e_slow + 1]
                        below = np.where(P_seg >= bottom_depth)[0]
                        if len(below) > 0:
                            candidate = new_s + int(below[0]) - 1
                            if candidate > new_s:
                                new_e = candidate
                                logger.info(
                                    "%s prof%d: bottom pushed end %.2f -> %.2f dbar",
                                    file_label,
                                    pi,
                                    float(P_slow[e_slow]),
                                    float(P_slow[new_e]),
                                )
            except Exception as exc:
                logger.warning("%s prof%d bottom failed: %s", file_label, pi, exc)

        adjusted.append((new_s, new_e))

    return adjusted


def _nan_excluded_probes(ds, excluded_probes: list[str], file_label: str) -> None:
    """Set per-probe epsilon to NaN for shear probes named in *excluded_probes*.

    Mutates *ds* in place, so the subsequent mk_epsilon_mean call sees NaN for
    those probes and excludes them from the geometric mean (and, via the
    fallback path in chi, from chi Method 1). Both the 2-D ``epsilon``
    (probe x time) and any pre-existing 1-D ``e_N`` companion variables are
    masked. Unknown probe names are warned and ignored — typos in the
    config should be visible but not fatal.
    """
    import numpy as np

    if "epsilon" not in ds or "probe" not in ds.dims:
        return
    probe_names = [str(p) for p in ds["probe"].values]
    for probe in excluded_probes:
        if probe not in probe_names:
            logger.warning(
                "instruments override for %s: probe %r not present (have %s)",
                file_label,
                probe,
                probe_names,
            )
            continue
        idx = probe_names.index(probe)
        ds["epsilon"].values[idx, :] = np.nan
        # mk_epsilon_mean prefers e_N variables when present; mask them too
        # so any caller that round-trips through e_N sees the same result.
        e_name = f"e_{idx + 1}"
        if e_name in ds.data_vars:
            ds[e_name].values[:] = np.nan


def process_file(
    p_path: Path,
    config: dict,
    gps,
    output_dirs: dict,
    hotel_data=None,
    hotel_cfg=None,
    instrument_key: str | None = None,
) -> dict:
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
    instrument_key : str, optional
        Key into ``config["instruments"]`` for per-instrument overrides
        (e.g. ``"SN465"``). Falls back to ``p_path.parent.name`` if not
        supplied — but the caller normally passes this explicitly because
        the trim stage flattens the original ``<root>/<SN>/<file>.p``
        layout into a single ``trimmed/`` directory.

    Returns
    -------
    dict with output paths per stage.
    """
    from odas_tpw.perturb.fp07_cal import fp07_calibrate
    from odas_tpw.processing.ct_align import ct_align
    from odas_tpw.processing.epsilon_combine import mk_epsilon_mean
    from odas_tpw.rsi.dissipation import _compute_epsilon
    from odas_tpw.rsi.p_file import PFile
    from odas_tpw.rsi.profile import _smooth_fall_rate, get_profiles

    result: dict[str, Any] = {
        "source": str(p_path),
        "profiles": [],
        "diss": [],
        "chi": [],
    }

    try:
        pf = PFile(p_path)
    except Exception as exc:
        logger.error("loading %s: %s", p_path.name, exc)
        return result

    # ---- Hotel data injection ----
    if hotel_data is not None:
        from odas_tpw.perturb.hotel import interpolate_hotel

        hotel_channels = interpolate_hotel(hotel_data, pf, hotel_cfg or {})
        for name, data in hotel_channels.items():
            pf.channels[name] = data

    # Per-stage log files: each ``with stage_log(...)`` adds a FileHandler
    # for that stage's output dir so e.g. all CTD-binning records for ``a.p``
    # land in ``ctd_NN/a.log`` *and* the worker/run logs.  basename = stem so
    # ``a.p`` produces ``a.log``.
    log_basename = p_path.stem

    # ---- CTD fork (full file, both up and down) ----
    ctd_cfg = config.get("ctd", {})
    if ctd_cfg.get("enable", True) and "ctd" in output_dirs:
        with stage_log(output_dirs.get("ctd"), log_basename):
            try:
                from odas_tpw.perturb.ctd import ctd_bin_file

                ctd_bin_file(
                    pf,
                    gps,
                    output_dirs["ctd"],
                    bin_width=ctd_cfg.get("bin_width", 0.5),
                    T_name=ctd_cfg.get("T_name", "JAC_T"),
                    C_name=ctd_cfg.get("C_name", "JAC_C"),
                    variables=ctd_cfg.get("variables"),
                    method=ctd_cfg.get("method", "mean"),
                    diagnostics=ctd_cfg.get("diagnostics", False),
                )
            except Exception as exc:
                logger.error("CTD binning %s: %s", p_path.name, exc)

    # ---- Profile fork ----
    # Wraps profile detection, FP07 calibration, CT alignment, and the
    # extract_profiles write.  These all conceptually feed profiles_NN/.
    profiles_cfg = config.get("profiles", {})
    fp07_cfg = config.get("fp07", {})
    ct_cfg = config.get("ct", {})

    with stage_log(output_dirs.get("profiles"), log_basename):
        P_slow = pf.channels.get("P")
        if P_slow is None:
            logger.warning("No pressure channel in %s", p_path.name)
            return result

        W = _smooth_fall_rate(P_slow, pf.fs_slow)
        profiles = get_profiles(
            P_slow,
            W,
            pf.fs_slow,
            P_min=profiles_cfg.get("P_min", 0.5),
            W_min=profiles_cfg.get("W_min", 0.3),
            direction=profiles_cfg.get("direction", "down"),
            min_duration=profiles_cfg.get("min_duration", 7.0),
        )

        if not profiles:
            logger.warning("No profiles in %s", p_path.name)
            return result

        # Adjust profile bounds (top-trim removes prop-wash, bottom detects seafloor crash)
        profiles = _adjust_profile_bounds(
            profiles,
            pf,
            config.get("top_trim", {}),
            config.get("bottom", {}),
            p_path.name,
        )

        # FP07 calibration
        if fp07_cfg.get("calibrate", True):
            try:
                cal_result = fp07_calibrate(
                    pf,
                    profiles,
                    reference=fp07_cfg.get("reference", "JAC_T"),
                    order=fp07_cfg.get("order", 2),
                    max_lag_seconds=fp07_cfg.get("max_lag_seconds", 10.0),
                    must_be_negative=fp07_cfg.get("must_be_negative", True),
                )
                # Apply calibrated temperatures back to pf.channels
                for ch_name, cal_data in cal_result.get("channels", {}).items():
                    pf.channels[ch_name] = cal_data
            except Exception as exc:
                logger.warning("FP07 cal failed for %s: %s", p_path.name, exc)

        # CT alignment
        if ct_cfg.get("align", True):
            T_name = ct_cfg.get("T_name", "JAC_T")
            C_name = ct_cfg.get("C_name", "JAC_C")
            if T_name in pf.channels and C_name in pf.channels:
                try:
                    C_aligned, _lag = ct_align(
                        pf.channels[T_name],
                        pf.channels[C_name],
                        pf.fs_slow,
                        profiles,
                    )
                    pf.channels[C_name] = C_aligned
                except Exception as exc:
                    logger.warning("CT align failed for %s: %s", p_path.name, exc)

        # Write per-profile NetCDFs
        from odas_tpw.rsi.profile import extract_profiles

        prof_scalars_cache: dict[str, dict[str, float]] = {}
        if "profiles" in output_dirs:
            try:
                prof_paths, prof_scalars = extract_profiles(
                    pf,
                    output_dirs["profiles"],
                    profiles=profiles,
                    gps=gps,
                    return_scalars=True,
                )
                result["profiles"] = [str(p) for p in prof_paths]
                prof_scalars_cache = {
                    str(p): s for p, s in zip(prof_paths, prof_scalars)
                }
            except Exception as exc:
                logger.error("extracting profiles %s: %s", p_path.name, exc)
                return result

    # Per-profile dissipation
    eps_cfg = config.get("epsilon", {})
    inst_lookup = instrument_key if instrument_key is not None else p_path.parent.name
    instrument_cfg = config.get("instruments", {}).get(inst_lookup, {})
    excluded_probes = list(instrument_cfg.get("exclude_shear_probes", []))
    chi_cfg = config.get("chi", {})
    chi_enabled = bool(chi_cfg.get("enable", False)) and "chi" in output_dirs

    # Per-profile channels cache: filled here, consumed by the chi loop
    # below.  Holds ~1 MB / profile of fast-time numpy arrays — bounded by
    # the per-file profile count (~80 for SN465), so worker-RSS impact is
    # well under 100 MB.  Skipped entirely when chi is disabled.
    prof_data_cache: dict[str, dict[str, Any]] = {}

    if "diss" in output_dirs and result["profiles"]:
        with stage_log(output_dirs.get("diss"), log_basename):
            for prof_path in result["profiles"]:
                try:
                    pre_loaded: dict[str, Any] | None
                    if chi_enabled:
                        # Single NC pass that produces both channels and
                        # therm-gradient channels; reused below for chi.
                        from odas_tpw.rsi.chi_io import _load_therm_channels

                        pre_loaded = _load_therm_channels(prof_path)
                        prof_data_cache[prof_path] = pre_loaded
                    else:
                        pre_loaded = None
                    diss_results = _compute_epsilon(
                        prof_path,
                        **{
                            k: v
                            for k, v in eps_cfg.items()
                            if k
                            not in (
                                "epsilon_minimum",
                                "T_source",
                                "T1_norm",
                                "T2_norm",
                                "diagnostics",
                            )
                        },
                        _pre_loaded=pre_loaded,
                    )
                    for ds in diss_results:
                        if excluded_probes:
                            _nan_excluded_probes(ds, excluded_probes, p_path.name)
                        ds = mk_epsilon_mean(ds, eps_cfg.get("epsilon_minimum", 1e-13))
                        _copy_profile_scalars(prof_path, ds, prof_scalars_cache)
                        out_name = Path(prof_path).name
                        out_path = output_dirs["diss"] / out_name
                        ds.to_netcdf(out_path)
                        result["diss"].append(str(out_path))
                except Exception as exc:
                    logger.error("diss for %s: %s", Path(prof_path).name, exc)

    # Per-profile chi (if enabled)
    if chi_enabled and result["diss"]:
        try:
            from odas_tpw.rsi.chi_io import _compute_chi
        except ImportError:
            pass
        else:
            with stage_log(output_dirs.get("chi"), log_basename):
                for prof_path, diss_path in zip(result["profiles"], result["diss"]):
                    try:
                        import xarray as xr

                        diss_ds = xr.open_dataset(diss_path)
                        # Pop so the cache is released as we consume it.
                        pre_loaded = prof_data_cache.pop(prof_path, None)
                        chi_results = _compute_chi(
                            prof_path,
                            epsilon_ds=diss_ds,
                            **{
                                k: v
                                for k, v in chi_cfg.items()
                                if k not in ("enable", "diagnostics")
                            },
                            _pre_loaded=pre_loaded,
                        )
                        diss_ds.close()
                        for chi_ds in chi_results:
                            _copy_profile_scalars(prof_path, chi_ds, prof_scalars_cache)
                            out_name = Path(prof_path).name
                            out_path = output_dirs["chi"] / out_name
                            chi_ds.to_netcdf(out_path)
                            result["chi"].append(str(out_path))
                    except Exception as exc:
                        logger.error("chi for %s: %s", Path(prof_path).name, exc)

    return result


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------


def _trim_one(args: tuple) -> tuple[Path | None, Path, str | None]:
    """Worker: trim one .p file and return (trimmed_path, source_path, error).

    Designed to be pickled and dispatched to a ProcessPoolExecutor — must
    take a single argument and import its dependencies inside the call.
    """
    p, trim_dir = args
    try:
        from odas_tpw.perturb.trim import trim_p_file

        return trim_p_file(p, trim_dir), p, None
    except Exception as exc:
        return None, p, str(exc)


def run_trim(
    config: dict,
    p_files: list[Path] | None = None,
    *,
    jobs: int = 1,
) -> list[Path]:
    """Trim corrupt final records from .p files.

    *jobs* > 1 dispatches :func:`trim_p_file` calls onto a process pool
    so the per-file work overlaps. Each call is independent (different
    output path, no shared state), so process-level parallelism gives
    near-linear speedup until disk I/O saturates.
    """
    from odas_tpw.perturb.discover import find_p_files

    files_cfg = config.get("files", {})
    output_root = Path(files_cfg.get("output_root", "results/"))
    trim_dir = output_root / "trimmed"

    if p_files is None:
        root = files_cfg.get("p_file_root", "VMP/")
        pattern = files_cfg.get("p_file_pattern", "**/*.p")
        p_files = find_p_files(root, pattern)
    if not p_files:
        return []

    results: list[Path] = []
    if jobs <= 1 or len(p_files) <= 1:
        from odas_tpw.perturb.trim import trim_p_file

        for p in p_files:
            try:
                trimmed = trim_p_file(p, trim_dir)
                results.append(trimmed)
                logger.info("Trimmed: %s", trimmed.name)
            except Exception as exc:
                logger.error("trimming %s: %s", p.name, exc)
        return results

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(_trim_one, (p, trim_dir)): p for p in p_files}
        for future in as_completed(futures):
            trimmed_or_none, src, err = future.result()
            if err is not None:
                logger.error("trimming %s: %s", src.name, err)
                continue
            assert trimmed_or_none is not None  # narrow for the type checker
            results.append(trimmed_or_none)
            logger.info("Trimmed: %s", trimmed_or_none.name)
    return results


def run_merge(config: dict) -> list[Path]:
    """Merge split .p files."""
    from odas_tpw.perturb.discover import find_p_files
    from odas_tpw.perturb.merge import find_mergeable_files, merge_p_files

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
            logger.info("Merged %d files -> %s", len(chain), merged.name)
        except Exception as exc:
            logger.error("merging: %s", exc)
    return results


def _setup_output_dirs(config: dict) -> dict[str, Path]:
    """Set up versioned output directories based on config."""
    files_cfg = config.get("files", {})
    output_root = Path(files_cfg.get("output_root", "results/"))

    dirs = {}

    # Profiles directory — hash includes profiles + fp07 + ct + gps + bottom + top_trim
    profiles_params = merge_config("profiles", config.get("profiles"))
    dirs["profiles"] = resolve_output_dir(
        output_root,
        "profiles",
        "profiles",
        profiles_params,
        upstream=_upstream_for("profiles", config),
    )

    # Diss directory — hash includes epsilon params + profiles upstream
    eps_params = merge_config("epsilon", config.get("epsilon"))
    dirs["diss"] = resolve_output_dir(
        output_root,
        "diss",
        "epsilon",
        eps_params,
        upstream=_upstream_for("diss", config),
    )

    # Chi directory (if enabled)
    chi_cfg = config.get("chi", {})
    if chi_cfg.get("enable", False):
        chi_params = merge_config("chi", chi_cfg)
        dirs["chi"] = resolve_output_dir(
            output_root,
            "chi",
            "chi",
            chi_params,
            upstream=_upstream_for("chi", config),
        )

    # CTD directory
    ctd_cfg = config.get("ctd", {})
    if ctd_cfg.get("enable", True):
        ctd_params = merge_config("ctd", ctd_cfg)
        dirs["ctd"] = resolve_output_dir(
            output_root,
            "ctd",
            "ctd",
            ctd_params,
            upstream=_upstream_for("ctd", config),
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
    from odas_tpw.perturb.discover import find_p_files
    from odas_tpw.perturb.gps import create_gps

    files_cfg = config.get("files", {})

    # Discover files
    if p_files is None:
        root = files_cfg.get("p_file_root", "VMP/")
        pattern = files_cfg.get("p_file_pattern", "**/*.p")
        p_files = find_p_files(root, pattern)

    if not p_files:
        logger.warning("No .p files found")
        return

    logger.info("Found %d .p files", len(p_files))

    # Capture the original parent directory of each .p file before trim
    # flattens the layout (e.g. <root>/SN465/foo.p -> trimmed/foo.p). The
    # basename is preserved across trim/merge, so we can look the SN back
    # up by file name in the per-file processing loop below.
    original_parent_by_name = {p.name: p.parent.name for p in p_files}

    # Parallel processing
    parallel_cfg = config.get("parallel", {})
    jobs = parallel_cfg.get("jobs", 1)
    if jobs == 0:
        jobs = os.cpu_count() or 1

    # Trim
    if files_cfg.get("trim", True):
        logger.info("Trimming...")
        trimmed = run_trim(config, p_files, jobs=jobs)
        if trimmed:
            p_files = trimmed

    # Merge
    if files_cfg.get("merge", False):
        logger.info("Merging...")
        merged_files = run_merge(config)
        if merged_files:
            p_files = merged_files

    # Setup output directories
    output_dirs = _setup_output_dirs(config)

    # GPS provider
    gps_cfg = merge_config("gps", config.get("gps"))
    gps = create_gps(gps_cfg)

    # Hotel data
    hotel_cfg = merge_config("hotel", config.get("hotel"))
    hotel_data = None
    if hotel_cfg.get("enable", False) and hotel_cfg.get("file"):
        from odas_tpw.perturb.hotel import load_hotel

        hotel_data = load_hotel(
            hotel_cfg["file"],
            hotel_cfg.get("time_column", "time"),
            hotel_cfg.get("time_format", "auto"),
            hotel_cfg.get("channels", {}),
        )
        logger.info("Loaded hotel file: %d channels", len(hotel_data.channels))

    logger.info("Processing %d files (jobs=%d)...", len(p_files), jobs)

    def _instrument_key(p):
        # Trim flattens <root>/<SN>/<file>.p into trimmed/<file>.p, losing the
        # SN parent. Recover it from the original-paths map; fall back to
        # parent.name (works for un-trimmed runs).
        return original_parent_by_name.get(p.name, p.parent.name)

    if jobs == 1:
        for p_path in p_files:
            logger.info("Processing %s...", p_path.name)
            process_file(
                p_path,
                config,
                gps,
                output_dirs,
                hotel_data=hotel_data,
                hotel_cfg=hotel_cfg,
                instrument_key=_instrument_key(p_path),
            )
    else:
        # Spawn workers each get a per-pid log file inside <output_root>/logs/
        # so multi-process runs are diagnosable.  ``run_stamp`` is the parent
        # CLI's invocation timestamp so all workers share a prefix.
        output_root_path = Path(files_cfg.get("output_root", "results/"))
        worker_log_dir = output_root_path / "logs"
        run_stamp = current_run_stamp()
        with ProcessPoolExecutor(
            max_workers=jobs,
            initializer=init_worker_logging,
            initargs=(worker_log_dir, run_stamp),
        ) as executor:
            futures = {
                executor.submit(
                    process_file,
                    p,
                    config,
                    gps,
                    output_dirs,
                    hotel_data=hotel_data,
                    hotel_cfg=hotel_cfg,
                    instrument_key=_instrument_key(p),
                ): p
                for p in p_files
            }
            for future in as_completed(futures):
                p = futures[future]
                try:
                    future.result()
                    logger.info("Done: %s", p.name)
                except Exception as exc:
                    logger.error("%s: %s", p.name, exc)

    # Binning — three independent stages (profiles / diss / chi) that
    # share no data, so we run them concurrently on a thread pool.  Each
    # stage's heavy lifting is numpy + bincount + xarray-IO, which release
    # the GIL, so threads give us most of the wall-clock win without
    # nesting process pools or paying the ProcessPoolExecutor IPC cost on
    # whole xarray Datasets.
    binning_cfg = config.get("binning", {})
    bin_method = binning_cfg.get("method", "depth")
    bin_width = binning_cfg.get("width", 1.0)
    aggregation = binning_cfg.get("aggregation", "mean")
    diagnostics = binning_cfg.get("diagnostics", False)

    output_root = Path(files_cfg.get("output_root", "results/"))

    from odas_tpw.perturb.binning import bin_by_depth, bin_by_time, bin_chi, bin_diss

    prof_binned_dir: Path | None = None
    diss_binned_dir: Path | None = None
    chi_binned_dir: Path | None = None

    # Resolve binned-output dirs *before* the bin call so we can pass them
    # as ``log_dir`` and each input .p file's records land in
    # ``<binned_dir>/<stem>.log``.  This also means the binned dir is
    # created (with its signature) even if the bin call ends up emitting
    # no data — a small price for symmetric per-file logs.
    prof_ncs = sorted(output_dirs["profiles"].glob("*.nc")) if "profiles" in output_dirs else []
    if prof_ncs:
        binning_params = merge_config("binning", binning_cfg)
        prof_binned_dir = resolve_output_dir(
            output_root,
            "profiles_binned",
            "binning",
            binning_params,
            upstream=_upstream_for("profiles_binned", config),
        )

    diss_ncs = sorted(output_dirs["diss"].glob("*.nc")) if "diss" in output_dirs else []
    diss_width = binning_cfg.get("diss_width") or bin_width
    diss_agg = binning_cfg.get("diss_aggregation") or aggregation
    if diss_ncs:
        diss_binned_dir = resolve_output_dir(
            output_root,
            "diss_binned",
            "binning",
            merge_config("binning", binning_cfg),
            upstream=_upstream_for("diss_binned", config),
        )

    chi_ncs: list[Path] = []
    chi_width = binning_cfg.get("chi_width") or binning_cfg.get("diss_width") or bin_width
    chi_agg = (
        binning_cfg.get("chi_aggregation")
        or binning_cfg.get("diss_aggregation")
        or aggregation
    )
    if "chi" in output_dirs:
        chi_ncs = sorted(output_dirs["chi"].glob("*.nc"))
        if chi_ncs:
            chi_binned_dir = resolve_output_dir(
                output_root,
                "chi_binned",
                "binning",
                merge_config("binning", binning_cfg),
                upstream=_upstream_for("chi_binned", config),
            )

    # Stages run sequentially because each one now drives a per-profile
    # ``ProcessPoolExecutor`` of size *jobs*.  Running the 3 stages
    # concurrently on a thread pool would over-commit (3 * jobs workers
    # competing for the same cores) and cost more than it saves.

    if prof_ncs and prof_binned_dir is not None:
        logger.info("Binning profiles...")
        if bin_method == "depth":
            ds = bin_by_depth(
                prof_ncs,
                bin_width,
                aggregation,
                diagnostics,
                log_dir=prof_binned_dir,
                jobs=jobs,
            )
        else:
            ds = bin_by_time(
                prof_ncs, bin_width, aggregation, diagnostics, log_dir=prof_binned_dir
            )
        if ds.data_vars:
            ds.to_netcdf(prof_binned_dir / "binned.nc")

    if diss_ncs and diss_binned_dir is not None:
        logger.info("Binning dissipation...")
        ds = bin_diss(
            diss_ncs,
            diss_width,
            diss_agg,
            bin_method,
            diagnostics,
            log_dir=diss_binned_dir,
            jobs=jobs,
        )
        if ds.data_vars:
            ds.to_netcdf(diss_binned_dir / "binned.nc")

    if chi_ncs and chi_binned_dir is not None:
        logger.info("Binning chi...")
        ds = bin_chi(
            chi_ncs,
            chi_width,
            chi_agg,
            bin_method,
            diagnostics,
            log_dir=chi_binned_dir,
            jobs=jobs,
        )
        if ds.data_vars:
            ds.to_netcdf(chi_binned_dir / "binned.nc")

    # Combo assembly
    _run_combo(
        output_root,
        prof_binned_dir,
        diss_binned_dir,
        chi_binned_dir,
        output_dirs.get("ctd"),
        config.get("netcdf", {}),
        config=config,
        bin_method=bin_method,
    )

    logger.info("Pipeline complete.")


def _run_combo(
    output_root: Path,
    prof_binned_dir: Path | None,
    diss_binned_dir: Path | None,
    chi_binned_dir: Path | None,
    ctd_dir: Path | None,
    netcdf_attrs: dict,
    config: dict | None = None,
    bin_method: str = "depth",
) -> None:
    """Assemble combo NetCDFs from each populated binned directory.

    Each combo writes ``output_root/<name>_combo/combo.nc`` with CF/ACDD
    metadata applied via :mod:`odas_tpw.perturb.netcdf_schema`. Missing
    or empty input dirs are silently skipped — this is a best-effort
    publish step run after binning completes.

    *bin_method* selects the gluing strategy for profiles/diss/chi combos:

    * ``"depth"`` — vertical profilers (cast VMP, glider-mounted MR).
      Each per-file binned NetCDF has dims ``(bin, profile)``; the combo
      stacks them widthwise so the result is a profile-set indexed by
      time.  Produces CF ``featureType=profile``.
    * ``"time"`` — non-profiling instruments (mooring-mounted MR, AUV).
      Each per-file binned NetCDF has a ``time`` dimension; the combo
      concatenates them lengthwise and sorts by time, like the CTD/scalar
      combo.  Produces CF ``featureType=trajectory``.

    The ctd_combo path always uses ``time`` regardless — scalar sensor
    streams are inherently per-time, never per-depth.

    When *config* is provided, each combo dir also receives a
    ``.params_sha256_<hash>`` signature capturing the full upstream chain
    so the hash matches the corresponding binned/source dir.
    """
    from odas_tpw.perturb.combo import make_combo, make_ctd_combo
    from odas_tpw.perturb.netcdf_schema import CHI_SCHEMA, COMBO_SCHEMA, CTD_SCHEMA

    logger.info("Assembling combo files (method=%s)...", bin_method)

    binning_p = (
        merge_config("binning", config.get("binning") if config else None)
        if config is not None
        else None
    )

    def _resolve_dst(stage: str, section: str, params: dict | None) -> Path:
        """Versioned ``stage_NN/`` when config available; legacy fixed path otherwise."""
        if config is None or params is None:
            d = output_root / stage
            d.mkdir(parents=True, exist_ok=True)
            return d
        return resolve_output_dir(
            output_root, stage, section, params,
            upstream=_upstream_for(stage, config),
        )

    targets = [
        (prof_binned_dir, "combo", COMBO_SCHEMA, bin_method, make_combo),
        (diss_binned_dir, "diss_combo", COMBO_SCHEMA, bin_method, make_combo),
        (chi_binned_dir, "chi_combo", CHI_SCHEMA, bin_method, make_combo),
    ]
    for src, stage, schema, method, func in targets:
        if src is None or not src.exists():
            continue
        dst = _resolve_dst(stage, "binning", binning_p)
        with stage_log(dst, "combo"):
            try:
                out = func(src, dst, schema, netcdf_attrs=netcdf_attrs, method=method)
                if out is not None:
                    logger.info("Wrote %s", out)
            except Exception as exc:
                logger.error("combo %s: %s", dst.name, exc)

    if ctd_dir is not None and ctd_dir.exists():
        ctd_p = merge_config("ctd", config.get("ctd")) if config is not None else None
        ctd_combo_dir = _resolve_dst("ctd_combo", "ctd", ctd_p)
        with stage_log(ctd_combo_dir, "combo"):
            try:
                out = make_ctd_combo(
                    ctd_dir, ctd_combo_dir, CTD_SCHEMA, netcdf_attrs=netcdf_attrs
                )
                if out is not None:
                    logger.info("Wrote %s", out)
            except Exception as exc:
                logger.error("ctd combo: %s", exc)
