# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Orchestration — per-file processing and full pipeline.

Reference: Code/process_P_files.m (233 lines), Code/mat2profile.m (170 lines)
"""

import hashlib
import logging
import os
import re
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


_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_stem_part(part: str) -> str:
    """Return a filesystem-friendly stem segment."""
    safe = _SAFE_STEM_RE.sub("_", part).strip("._")
    return safe or "unnamed"


def _source_output_stem(path: Path, root: Path | str) -> str:
    """Return a unique, stable output stem for *path* relative to *root*."""
    path = Path(path)
    root = Path(root)
    try:
        rel = path.resolve().relative_to(root.resolve()).with_suffix("")
        parts = rel.parts
    except ValueError:
        digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
        parts = (path.stem, digest)
    return "__".join(_safe_stem_part(str(part)) for part in parts)


def _canonical_instruments_for_hash(instruments: dict | None) -> dict[str, Any]:
    """Normalize set-like instrument settings before hashing."""
    normalized: dict[str, Any] = {}
    for key, settings in sorted((instruments or {}).items()):
        if not isinstance(settings, dict):
            normalized[str(key)] = settings
            continue
        item = dict(settings)
        excludes = item.get("exclude_shear_probes")
        if isinstance(excludes, list):
            item["exclude_shear_probes"] = sorted(str(probe) for probe in excludes)
        normalized[str(key)] = item
    return normalized


def _upstream_for(stage: str, config: dict) -> list[tuple[str, dict]]:
    """Return the full upstream parameter chain for *stage*.

    Each downstream stage's ``.params_sha256_*`` signature is the hash of
    the stage's own params plus every ancestor's params, so two runs that
    differ only in a deep upstream knob still resolve to different output
    directories.  Ancestors are listed deepest-first only by convention
    (the canonicaliser sorts on the section name anyway).

    Stages:

    * ``profiles``           — file flow + profile-affecting preprocessing
    * ``diss``               — profiles + per-instrument probe exclusions
    * ``chi``                — epsilon, profiles, QC, and probe exclusions
    * ``ctd``                — file flow + GPS
    * binned/combo stages    — their source stage chain plus binning/netcdf
    """
    files_p = merge_config("files", config.get("files"))
    gps_p = merge_config("gps", config.get("gps"))
    hotel_p = merge_config("hotel", config.get("hotel"))
    speed_p = merge_config("speed", config.get("speed"))
    qc_p = merge_config("qc", config.get("qc"))
    fp07_p = merge_config("fp07", config.get("fp07"))
    ct_p = merge_config("ct", config.get("ct"))
    bottom_p = merge_config("bottom", config.get("bottom"))
    top_trim_p = merge_config("top_trim", config.get("top_trim"))
    profiles_p = merge_config("profiles", config.get("profiles"))
    eps_p = merge_config("epsilon", config.get("epsilon"))
    chi_p = merge_config("chi", config.get("chi"))
    ctd_p = merge_config("ctd", config.get("ctd"))
    netcdf_p = merge_config("netcdf", config.get("netcdf"))
    instruments_p = _canonical_instruments_for_hash(config.get("instruments"))

    profile_upstream = [
        ("files", files_p),
        ("gps", gps_p),
        ("hotel", hotel_p),
        ("speed", speed_p),
        ("qc", qc_p),
        ("fp07", fp07_p),
        ("ct", ct_p),
        ("bottom", bottom_p),
        ("top_trim", top_trim_p),
    ]
    profile_chain = [*profile_upstream, ("profiles", profiles_p)]
    diss_chain = [*profile_chain, ("instruments", instruments_p)]
    chi_chain = [*diss_chain, ("epsilon", eps_p)]
    ctd_chain = [
        ("files", files_p),
        ("gps", gps_p),
        ("hotel", hotel_p),
        ("speed", speed_p),
        ("qc", qc_p),
    ]

    chains: dict[str, list[tuple[str, dict]]] = {
        "profiles": profile_upstream,
        "diss": diss_chain,
        "chi": chi_chain,
        "ctd": ctd_chain,
        "profiles_binned": profile_chain,
        "diss_binned": [*diss_chain, ("epsilon", eps_p)],
        "chi_binned": [*chi_chain, ("chi", chi_p)],
        "combo": [*profile_chain, ("netcdf", netcdf_p)],
        "diss_combo": [*diss_chain, ("epsilon", eps_p), ("netcdf", netcdf_p)],
        "chi_combo": [*chi_chain, ("chi", chi_p), ("netcdf", netcdf_p)],
        "ctd_combo": [*ctd_chain, ("ctd", ctd_p), ("netcdf", netcdf_p)],
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


def _apply_fom_cut(ds, fom_max: float, file_label: str) -> int:
    """NaN per-probe per-segment values where ``fom >= fom_max``.

    Operates on the (probe, time) array set returned by the diss / chi
    stages. Mutates *ds* in place; returns the count of (probe, segment)
    pairs that were NaN'd.

    The FOM cut runs *before* ``mk_epsilon_mean`` / ``mk_chi_mean`` so
    the per-probe drops are reflected in the geometric-mean and
    log-sigma aggregates (a probe with bad FOM at one time step contributes
    nothing to the combined estimate at that step, while the other
    probe still does).

    Variables masked:
    - ``epsilon`` / ``chi`` (probe, time) — set NaN at flagged cells.
    - ``e_N`` / ``chi_N`` (time,) — set NaN at the time slots where
      probe ``N`` was flagged. ``mk_*_mean`` prefers these companions
      so they need to agree.

    Other per-probe metadata vars (``fom``, ``K_max``, ``kB`` etc.) are
    left untouched — they're useful even at flagged segments for QC
    and post-hoc diagnostics.
    """
    import numpy as np

    if "fom" not in ds or "probe" not in ds.dims:
        return 0
    fom = ds["fom"].values
    bad = np.isfinite(fom) & (fom >= fom_max)
    if not bad.any():
        return 0
    n_bad = int(bad.sum())
    n_probe = bad.shape[0]

    # 2-D (probe, time) value vars
    for v in ("epsilon", "chi", "epsilon_T"):
        if v in ds.data_vars and ds[v].shape == fom.shape:
            arr = ds[v].values.copy()
            arr[bad] = np.nan
            ds[v].values[...] = arr

    # 1-D companions used by mk_*_mean
    for i in range(n_probe):
        for prefix in ("e_", "chi_"):
            name = f"{prefix}{i + 1}"
            if name in ds.data_vars:
                arr = ds[name].values.copy()
                arr[bad[i, :]] = np.nan
                ds[name].values[...] = arr

    logger.info(
        "%s: fom_max=%g cut %d (probe,segment) cells",
        file_label, fom_max, n_bad,
    )
    return n_bad


def _time_epoch_seconds(da) -> Any:
    """Convert a time variable to float64 epoch seconds (ndarray).

    Handles both decoded (datetime64) and undecoded (float + CF
    "seconds since ..." units) representations, so arrays from different
    open modes can be paired on one time base.
    """
    import numpy as np

    vals = np.asarray(da.values)
    if np.issubdtype(vals.dtype, np.datetime64):
        return vals.astype("datetime64[ns]").astype("int64") / 1e9
    units = str(da.attrs.get("units", ""))
    if " since " in units:
        # np.datetime64 rejects timezone-aware strings; these stamps are UTC.
        origin = units.split(" since ", 1)[1].strip()
        origin = origin.replace("Z", "").replace("+00:00", "")
        origin_s = (
            np.datetime64(origin).astype("datetime64[ns]").astype("int64") / 1e9
        )
        return vals.astype(np.float64) + origin_s
    return vals.astype(np.float64)


def _add_mixing_quantities(
    chi_ds,
    diss_ds,
    prof_path,
    T_name: str = "JAC_T",
    C_name: str = "JAC_C",
    file_label: str = "",
):
    """Append derived mixing quantities to a per-profile chi dataset.

    Computes window-scale N2 and dT/dz from the profile's own slow CTD
    channels — with practical salinity from C/T/P via TEOS-10 when the
    conductivity channel exists, so N2 is fully constrained (unlike the
    rsi path, which may assume 35 PSU) — then K_T (Osborn-Cox), Gamma
    (Oakey 1982), and K_rho (Osborn 1980, Gamma_0 = 0.2), pairing
    epsilonMean from the matching diss dataset onto the chi window grid.

    Mutates and returns *chi_ds*.  No-op (with a log line) when the
    required inputs are missing.
    """
    import gsw
    import numpy as np
    import xarray as xr

    from odas_tpw.processing.mixing import (
        mixing_coefficients,
        pair_nearest,
        window_stratification,
    )

    if diss_ds is None or "epsilonMean" not in diss_ds or "chiMean" not in chi_ds:
        logger.info("mixing quantities skipped for %s: missing epsilon/chi", file_label)
        return chi_ds

    with xr.open_dataset(prof_path, decode_times=False) as prof:
        if any(v not in prof for v in ("t_slow", "P", T_name)):
            logger.info(
                "mixing quantities skipped for %s: profile lacks t_slow/P/%s",
                file_label,
                T_name,
            )
            return chi_ds
        t_slow = _time_epoch_seconds(prof["t_slow"])
        P = prof["P"].values.astype(np.float64)
        T = prof[T_name].values.astype(np.float64)
        lat = float(prof["lat"].values) if "lat" in prof else np.nan
        lon = float(prof["lon"].values) if "lon" in prof else np.nan
        if not np.isfinite(lat):
            lat = 0.0
        if not np.isfinite(lon):
            lon = 0.0

        if C_name in prof:
            C = prof[C_name].values.astype(np.float64)
            S = gsw.SP_from_C(C, T, P)
            sal_note = f"practical salinity from {C_name}/{T_name}/P (TEOS-10)"
        else:
            S = None
            sal_note = (
                "salinity assumed 35 PSU (no conductivity channel); N2 "
                "reflects temperature stratification only"
            )

    chi_t = _time_epoch_seconds(chi_ds["t"])
    try:
        half_w = 0.5 * float(chi_ds.attrs["diss_length"]) / float(chi_ds.attrs["fs_fast"])
    except (KeyError, TypeError, ValueError):
        logger.info("mixing quantities skipped for %s: no window attrs", file_label)
        return chi_ds

    strat = window_stratification(chi_t, half_w, t_slow, P, T, S=S, lat=lat, lon=lon)
    eps_on_chi = pair_nearest(
        _time_epoch_seconds(diss_ds["t"]), diss_ds["epsilonMean"].values, chi_t
    )
    mix = mixing_coefficients(
        eps_on_chi, chi_ds["chiMean"].values, strat.N2, strat.dTdz
    )

    var_specs = {
        "N2": (
            strat.N2,
            {
                "units": "s-2",
                "long_name": "buoyancy frequency squared (window scale)",
                "comment": (
                    "TEOS-10 (gsw.Nsquared) between the shallow- and deep-half "
                    f"means of each chi window; {sal_note}."
                ),
            },
        ),
        "dTdz": (
            strat.dTdz,
            {
                "units": "K m-1",
                "long_name": "background temperature gradient (positive down)",
                "comment": (
                    "Least-squares slope of in-situ temperature vs depth over "
                    "each chi window."
                ),
            },
        ),
        "K_T": (
            mix.K_T,
            {
                "units": "m2 s-1",
                "long_name": "Osborn-Cox eddy diffusivity of heat",
                "comment": (
                    "K_T = chi / (2*(dT/dz)^2), Osborn & Cox (1972), "
                    "doi:10.1080/03091927208236085. NaN where "
                    "|dT/dz| < 1e-4 K/m (well-mixed)."
                ),
            },
        ),
        "Gamma": (
            mix.Gamma,
            {
                "units": "1",
                "long_name": "mixing coefficient (measured)",
                "comment": (
                    "Gamma = N2*chi / (2*epsilon*(dT/dz)^2), Oakey (1982), "
                    "doi:10.1175/1520-0485(1982)012<0256:DOTROD>2.0.CO;2. "
                    "epsilonMean paired from the nearest dissipation window. "
                    "NaN where N2 < 1e-9 s-2 or |dT/dz| < 1e-4 K/m. "
                    "Canonical value ~0.2."
                ),
            },
        ),
        "K_rho": (
            mix.K_rho,
            {
                "units": "m2 s-1",
                "long_name": "Osborn diapycnal diffusivity (Gamma_0 = 0.2)",
                "comment": (
                    "K_rho = 0.2*epsilon/N2, Osborn (1980), "
                    "doi:10.1175/1520-0485(1980)010<0083:EOTLRO>2.0.CO;2. "
                    "Compare with K_T: agreement implies the measured Gamma "
                    "is near the canonical 0.2. NaN where N2 < 1e-9 s-2."
                ),
            },
        ),
    }
    for name, (arr, attrs) in var_specs.items():
        chi_ds[name] = xr.DataArray(arr, dims=["time"], attrs=attrs)
    n_gamma = int(np.sum(np.isfinite(mix.Gamma)))
    logger.info(
        "mixing quantities for %s: %d valid Gamma estimates", file_label, n_gamma
    )
    return chi_ds


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
    output_stem: str | None = None,
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
    output_stem : str, optional
        Unique stem for per-file outputs. Defaults to ``p_path.stem``.

    Returns
    -------
    dict with output paths per stage.
    """
    from odas_tpw.perturb.fp07_cal import fp07_calibrate
    from odas_tpw.perturb.qc_gate import apply_qc_to_dataset
    from odas_tpw.processing.chi_combine import mk_chi_mean
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
        from odas_tpw.perturb.hotel import merge_hotel_into_pfile

        merge_hotel_into_pfile(hotel_data, pf, hotel_cfg or {})

    # ---- Speed channel (downstream of .p load + hotel merge) ----
    # Inject ``speed_fast`` and ``W_slow`` so extract_profiles writes them
    # to every per-profile NetCDF, and downstream prepare_profiles uses
    # them instead of recomputing from P. Default method is "pressure",
    # which reproduces the historical VMP behaviour exactly.
    speed_cfg = merge_config("speed", config.get("speed"))
    if "P" in pf.channels:
        try:
            from odas_tpw.perturb.speed import compute_speed_for_pfile

            vehicle = pf.config.get("instrument_info", {}).get("vehicle", "").lower()
            speed_fast, W_slow = compute_speed_for_pfile(pf, speed_cfg, vehicle)

            pf.channels["speed_fast"] = speed_fast
            pf._fast_channels.add("speed_fast")
            pf.channel_info["speed_fast"] = {
                "units": "m s-1", "type": "computed",
                "name": "speed_fast",
            }
            pf.channels["W_slow"] = W_slow
            pf.channel_info["W_slow"] = {
                "units": "dbar s-1", "type": "computed",
                "name": "W_slow",
            }
        except Exception as exc:
            logger.warning(
                "speed channel computation failed for %s (method=%s): %s",
                p_path.name, speed_cfg.get("method", "pressure"), exc,
            )

    # ---- Internal QC rules (after hotel + speed so all channels exist) ----
    qc_rules_cfg = (config.get("qc") or {}).get("rules") or {}
    if qc_rules_cfg:
        try:
            from odas_tpw.perturb.qc_rules import (
                evaluate_rules,
                register_rule_channels,
            )

            evaluated = evaluate_rules(pf, qc_rules_cfg)
            register_rule_channels(pf, evaluated, qc_rules_cfg)
        except Exception as exc:
            logger.warning(
                "qc.rules evaluation failed for %s: %s", p_path.name, exc,
            )

    # Per-stage log files: each ``with stage_log(...)`` adds a FileHandler
    # for that stage's output dir so e.g. all CTD-binning records for ``a.p``
    # land in ``ctd_NN/a.log`` *and* the worker/run logs.  basename = stem so
    # ``a.p`` produces ``a.log``.
    output_stem = output_stem or p_path.stem
    log_basename = output_stem

    profiles_cfg = config.get("profiles", {})
    fp07_cfg = config.get("fp07", {})
    ct_cfg = config.get("ct", {})

    # ---- Profile detection + CT alignment ----
    # Runs BEFORE the CTD fork so the CTD product's salinity/density are
    # computed from time-aligned conductivity whenever profiles can be
    # detected (alignment needs the per-profile lag estimates).  Files
    # with no pressure or no detectable profiles still get a CTD product
    # from the unaligned conductivity.
    profiles: list[tuple[int, int]] = []
    P_slow = pf.channels.get("P")
    with stage_log(output_dirs.get("profiles"), log_basename):
        if P_slow is None:
            logger.warning("No pressure channel in %s", p_path.name)
        else:
            W = _smooth_fall_rate(P_slow, pf.fs_slow)
            # Resolve "auto" → vehicle default (e.g. slocum_glider → "glide").
            # ``scor160.profile.get_profiles`` doesn't know "auto" itself, so
            # without this it silently falls through to the "down" branch and
            # we lose every up-profile -- on a glider with MR-on-during-climb
            # only, that drops *all* the real flight data.
            from odas_tpw.rsi.vehicle import resolve_direction

            vehicle = pf.config.get("instrument_info", {}).get("vehicle", "").lower()
            direction = resolve_direction(
                profiles_cfg.get("direction", "auto"), vehicle,
            )
            profiles = get_profiles(
                P_slow,
                W,
                pf.fs_slow,
                P_min=profiles_cfg.get("P_min", 0.5),
                W_min=profiles_cfg.get("W_min", 0.3),
                direction=direction,
                min_duration=profiles_cfg.get("min_duration", 7.0),
            )
            if not profiles:
                logger.warning("No profiles in %s", p_path.name)

        # CT alignment (before the CTD fork; needs detected profiles)
        if profiles and ct_cfg.get("align", True):
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
                    output_stem=output_stem,
                )
            except Exception as exc:
                logger.error("CTD binning %s: %s", p_path.name, exc)

    if P_slow is None or not profiles:
        return result

    # ---- Profile fork ----
    # Wraps profile-bound adjustment, FP07 calibration, and the
    # extract_profiles write.  These all conceptually feed profiles_NN/.
    with stage_log(output_dirs.get("profiles"), log_basename):
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
                # Apply calibrated temperatures back to pf.channels —
                # both the base channels (T1, T2) and the recalibrated
                # deconvolved fast channels (T1_dT1, T2_dT2) that the
                # chi pipeline consumes.  Without the latter, chi keeps
                # the factory calibration and scales with the square of
                # the calibration slope error.
                for ch_name, cal_data in cal_result.get("channels", {}).items():
                    pf.channels[ch_name] = cal_data
                for ch_name, cal_data in cal_result.get("fast_channels", {}).items():
                    pf.channels[ch_name] = cal_data
            except Exception as exc:
                logger.warning("FP07 cal failed for %s: %s", p_path.name, exc)

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
                    output_stem=output_stem,
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
    diss_by_profile: dict[str, str] = {}

    # QC gate config — resolved once for both diss and chi loops below.
    qc_cfg = merge_config("qc", config.get("qc"))
    qc_enabled = bool(qc_cfg.get("enable", True))
    qc_drop_action = qc_cfg.get("drop_action", "nan")
    qc_eps_drop_from = list(qc_cfg.get("epsilon_drop_from") or [])
    qc_chi_drop_from = list(qc_cfg.get("chi_drop_from") or [])
    diss_length_samples = float(eps_cfg.get("diss_length") or 4 * eps_cfg.get("fft_length", 256))
    diss_length_seconds = diss_length_samples / float(pf.fs_fast)
    chi_diss_length_samples = float(
        chi_cfg.get("diss_length") or 4 * chi_cfg.get("fft_length", 512)
    )
    chi_diss_length_seconds = chi_diss_length_samples / float(pf.fs_fast)

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
                                "fom_max",
                                "diagnostics",
                            )
                        },
                        _pre_loaded=pre_loaded,
                    )
                    eps_fom_max = eps_cfg.get("fom_max")
                    for ds in diss_results:
                        if excluded_probes:
                            _nan_excluded_probes(ds, excluded_probes, p_path.name)
                        if eps_fom_max is not None:
                            _apply_fom_cut(ds, float(eps_fom_max), p_path.name)
                        ds = mk_epsilon_mean(ds, eps_cfg.get("epsilon_minimum", 1e-13))
                        if qc_enabled:
                            apply_qc_to_dataset(
                                ds, pf, qc_eps_drop_from, diss_length_seconds,
                                flag_var_name="qc_drop_epsilon",
                                value_vars=[
                                    "epsilonMean", "epsilonLnSigma",
                                    "e_1", "e_2", "epsilon",
                                ],
                                drop_action=qc_drop_action,
                            )
                        _copy_profile_scalars(prof_path, ds, prof_scalars_cache)
                        out_name = Path(prof_path).name
                        out_path = output_dirs["diss"] / out_name
                        ds.to_netcdf(out_path)
                        out_path_str = str(out_path)
                        result["diss"].append(out_path_str)
                        diss_by_profile[prof_path] = out_path_str
                except Exception as exc:
                    logger.error("diss for %s: %s", Path(prof_path).name, exc)

    # Per-profile chi (if enabled)
    if chi_enabled and result["diss"]:
        try:
            from odas_tpw.rsi.chi_io import _compute_chi
        except ImportError:
            pass
        else:
            chi_use_epsilon = bool(chi_cfg.get("use_epsilon", True))
            chi_fom_max = chi_cfg.get("fom_max")
            chi_kwargs = {
                k: v
                for k, v in chi_cfg.items()
                if k not in (
                    "enable", "chi_minimum", "diagnostics",
                    "use_epsilon", "fom_max", "mixing",
                )
            }
            with stage_log(output_dirs.get("chi"), log_basename):
                for prof_path in result["profiles"]:
                    diss_path = diss_by_profile.get(prof_path)
                    if diss_path is None:
                        logger.warning(
                            "chi skipped for %s: no matching diss output",
                            Path(prof_path).name,
                        )
                        prof_data_cache.pop(prof_path, None)
                        continue
                    diss_ds = None
                    try:
                        import xarray as xr

                        # Method 1 uses shear-probe epsilon to seed kB.
                        # Method 2 (use_epsilon=False) does a pure spectral
                        # fit and is appropriate when shear epsilon is
                        # contaminated (e.g. MR on a vibrating glider).
                        diss_ds = xr.open_dataset(diss_path) if chi_use_epsilon else None
                        # Pop so the cache is released as we consume it.
                        pre_loaded = prof_data_cache.pop(prof_path, None)
                        chi_results = _compute_chi(
                            prof_path,
                            epsilon_ds=diss_ds,
                            **chi_kwargs,
                            _pre_loaded=pre_loaded,
                        )
                        for chi_ds in chi_results:
                            if chi_fom_max is not None:
                                _apply_fom_cut(
                                    chi_ds, float(chi_fom_max), p_path.name,
                                )
                            chi_ds = mk_chi_mean(
                                chi_ds, chi_cfg.get("chi_minimum", 1e-13)
                            )
                            # Derived mixing quantities (Gamma, K_T, K_rho)
                            # with real salinity from the profile's own C/T/P
                            if chi_cfg.get("mixing", True):
                                mix_eps = (
                                    diss_ds
                                    if diss_ds is not None
                                    else xr.open_dataset(diss_path)
                                )
                                try:
                                    chi_ds = _add_mixing_quantities(
                                        chi_ds,
                                        mix_eps,
                                        prof_path,
                                        T_name=ct_cfg.get("T_name", "JAC_T"),
                                        C_name=ct_cfg.get("C_name", "JAC_C"),
                                        file_label=Path(prof_path).name,
                                    )
                                finally:
                                    if mix_eps is not diss_ds:
                                        mix_eps.close()
                            if qc_enabled:
                                apply_qc_to_dataset(
                                    chi_ds, pf, qc_chi_drop_from,
                                    chi_diss_length_seconds,
                                    flag_var_name="qc_drop_chi",
                                    value_vars=[
                                        "chiMean", "chiLnSigma",
                                        "chi_1", "chi_2", "chi",
                                        "epsilon_T",
                                    ],
                                    drop_action=qc_drop_action,
                                )
                            _copy_profile_scalars(prof_path, chi_ds, prof_scalars_cache)
                            out_name = Path(prof_path).name
                            out_path = output_dirs["chi"] / out_name
                            chi_ds.to_netcdf(out_path)
                            result["chi"].append(str(out_path))
                    except Exception as exc:
                        logger.error("chi for %s: %s", Path(prof_path).name, exc)
                    finally:
                        if diss_ds is not None:
                            diss_ds.close()

    return result


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------


def _configured_input_root(config: dict) -> Path:
    """Return the configured input root for preserving relative paths."""
    files_cfg = config.get("files", {})
    return Path(files_cfg.get("p_file_root", "VMP/"))


def _check_unique_outputs(destinations: list[Path]) -> None:
    """Raise if multiple inputs would write the same output path."""
    seen: dict[Path, int] = {}
    for dest in destinations:
        key = dest.resolve()
        seen[key] = seen.get(key, 0) + 1
    duplicates = [str(path) for path, count in seen.items() if count > 1]
    if duplicates:
        raise ValueError(
            "multiple input files map to the same output path: "
            + ", ".join(sorted(duplicates))
        )


def _check_unique_output_stems(stems: list[str]) -> None:
    """Raise if multiple processed inputs would share one output stem."""
    seen: dict[str, int] = {}
    for stem in stems:
        seen[stem] = seen.get(stem, 0) + 1
    duplicates = [stem for stem, count in seen.items() if count > 1]
    if duplicates:
        raise ValueError(
            "multiple input files map to the same output stem: "
            + ", ".join(sorted(duplicates))
        )


def _trim_one(args: tuple):
    """Worker: trim one .p file and return (trim_result, source_path, error).

    Designed to be pickled and dispatched to a ProcessPoolExecutor — must
    take a single argument and import its dependencies inside the call.
    The first element is a :class:`~odas_tpw.perturb.trim.TrimResult` on
    success (or ``None`` with an error string on failure).
    """
    p, trim_dir, root, force = args
    try:
        from odas_tpw.perturb.trim import trim_p_file

        return trim_p_file(p, trim_dir, root=root, force=force), p, None
    except Exception as exc:
        return None, p, str(exc)


def _trim_log_message(result) -> str:
    """Render the per-file trim log line, reporting what happened to the file.

    Distinguishes the three outcomes so a re-run is legible: a real trim
    reports the bytes dropped, a complete file reports that it was referenced
    in place (not copied), and an up-to-date trimmed output reports skipped.
    """
    name = result.dest.name
    if result.action == "trimmed":
        return (
            f"Trimmed: {name} (removed {result.bytes_removed} B — "
            f"incomplete final record of {result.record_size} B)"
        )
    if result.action == "skipped":
        return f"Trim: {name} (skipped — trimmed output already up to date)"
    return f"Trim: {name} (complete — referenced original in place, not copied)"


def _log_trim_summary(results: list, n_failed: int) -> None:
    """Emit one roll-up line with the per-action counts."""
    counts = {"trimmed": 0, "referenced": 0, "skipped": 0}
    for r in results:
        counts[r.action] = counts.get(r.action, 0) + 1
    logger.info(
        "Trim summary: %d trimmed, %d referenced in place, %d skipped (up to date), "
        "%d failed",
        counts["trimmed"], counts["referenced"], counts["skipped"], n_failed,
    )


def run_trim(
    config: dict,
    p_files: list[Path] | None = None,
    *,
    jobs: int = 1,
) -> list[Path]:
    """Trim corrupt final records from .p files; reference complete ones in place.

    *jobs* > 1 dispatches :func:`trim_p_file` calls onto a process pool
    so the per-file work overlaps. Each call is independent (different
    output path, no shared state), so process-level parallelism gives
    near-linear speedup until disk I/O saturates.

    Complete files are not copied — their original path is returned so
    downstream stages read them in place. Only genuinely-truncated files
    are written under ``<output_root>/trimmed/``. A trimmed file is skipped
    when an up-to-date output already exists (``files.force_trim: true``
    re-trims unconditionally). The returned list is the per-file path to
    use downstream (original for referenced files, trimmed path otherwise).
    """
    from odas_tpw.perturb.discover import find_p_files
    from odas_tpw.perturb.trim import trim_destination

    files_cfg = config.get("files", {})
    output_root = Path(files_cfg.get("output_root", "results/"))
    trim_dir = output_root / "trimmed"
    root = _configured_input_root(config)
    force = bool(files_cfg.get("force_trim", False))

    if p_files is None:
        root = _configured_input_root(config)
        pattern = files_cfg.get("p_file_pattern", "**/*.p")
        p_files = find_p_files(root, pattern)
    if not p_files:
        return []

    destinations = [trim_destination(p, trim_dir, root=root) for p in p_files]
    _check_unique_outputs(destinations)

    trim_results: list = []
    n_failed = 0
    if jobs <= 1 or len(p_files) <= 1:
        from odas_tpw.perturb.trim import trim_p_file

        for p in p_files:
            try:
                result = trim_p_file(p, trim_dir, root=root, force=force)
                trim_results.append(result)
                logger.info("%s", _trim_log_message(result))
            except Exception as exc:
                n_failed += 1
                logger.error("trimming %s: %s", p.name, exc)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(_trim_one, (p, trim_dir, root, force)): p
                for p in p_files
            }
            for future in as_completed(futures):
                result, src, err = future.result()
                if err is not None:
                    n_failed += 1
                    logger.error("trimming %s: %s", src.name, err)
                    continue
                assert result is not None  # narrow for the type checker
                trim_results.append(result)
                logger.info("%s", _trim_log_message(result))

    _log_trim_summary(trim_results, n_failed)
    return [r.dest for r in trim_results]


def run_merge(
    config: dict,
    p_files: list[Path] | None = None,
    *,
    include_singletons: bool = False,
    input_root: Path | str | None = None,
    merge_plan: list[tuple[Path, list[Path]]] | None = None,
) -> list[Path]:
    """Merge split .p files.

    By default, returns only newly merged outputs for CLI compatibility.
    When *include_singletons* is true, returns the complete post-merge file
    list: merged outputs plus untouched non-mergeable inputs.
    """
    from odas_tpw.perturb.discover import find_p_files
    from odas_tpw.perturb.merge import merge_p_files, plan_merge_outputs

    files_cfg = config.get("files", {})
    root = Path(input_root) if input_root is not None else _configured_input_root(config)
    output_root = Path(files_cfg.get("output_root", "results/"))
    merge_dir = output_root / "merged"

    if p_files is None:
        pattern = files_cfg.get("p_file_pattern", "**/*.p")
        p_files = find_p_files(root, pattern)
    if not p_files:
        return []

    plan = merge_plan if merge_plan is not None else plan_merge_outputs(
        p_files, merge_dir, root=root
    )
    planned_outputs = [
        output_path for output_path, chain in plan
        if include_singletons or len(chain) > 1
    ]
    _check_unique_outputs(planned_outputs)

    results: list[Path] = []
    for output_path, chain in plan:
        if len(chain) == 1:
            if include_singletons:
                results.append(chain[0])
            continue
        try:
            merged = merge_p_files(chain, merge_dir, root=root)
            results.append(merged)
            logger.info("Merged %d files -> %s", len(chain), merged.name)
        except Exception as exc:
            logger.error("merging: %s", exc)
            if include_singletons:
                results.extend(chain)
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


def _format_elapsed(seconds: float) -> str:
    """Human-readable elapsed time: ``12.3s`` under a minute, else ``1m 23s``."""
    if seconds >= 60:
        return f"{int(seconds // 60)}m {seconds % 60:02.0f}s"
    return f"{seconds:.1f}s"


def _done_message(name: str, result: Any) -> str:
    """Per-file completion line reporting products written and elapsed time.

    *result* is the dict returned by :func:`process_file` (``profiles`` /
    ``diss`` / ``chi`` lists, plus ``elapsed_s`` when run through
    :func:`_process_file_timed`). Reports the profile count headline plus the
    dissipation and chi counts (``0`` chi simply means chi was disabled) and
    the per-file wall-clock when available.
    """
    if isinstance(result, dict):
        n_prof = len(result.get("profiles", []))
        n_diss = len(result.get("diss", []))
        n_chi = len(result.get("chi", []))
        elapsed = result.get("elapsed_s")
    else:
        n_prof = n_diss = n_chi = 0
        elapsed = None
    elapsed_str = f" in {_format_elapsed(elapsed)}" if elapsed is not None else ""
    return (
        f"Done processing {name}: {n_prof} profiles "
        f"({n_diss} dissipation, {n_chi} chi){elapsed_str}"
    )


def _process_file_timed(*args, **kwargs) -> dict:
    """Run :func:`process_file`, recording its wall-clock as ``elapsed_s``.

    Wrapping rather than instrumenting process_file's several return points
    keeps the timing in one place. The elapsed seconds are measured inside the
    worker process, so they reflect true per-file processing time, not the
    time a file spent queued waiting for a free worker.
    """
    import time

    t0 = time.monotonic()
    result = process_file(*args, **kwargs)
    if isinstance(result, dict):
        result["elapsed_s"] = time.monotonic() - t0
    return result


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

    input_root = _configured_input_root(config)
    current_root = input_root
    instrument_key_by_path = {p.resolve(): p.parent.name for p in p_files}
    output_stem_by_path = {
        p.resolve(): _source_output_stem(p, current_root) for p in p_files
    }

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
            from odas_tpw.perturb.trim import trim_destination

            trim_dir = Path(files_cfg.get("output_root", "results/")) / "trimmed"
            # run_trim returns the trim-dir path for trimmed/skipped files and
            # the *original* path for complete (referenced) files. Recover the
            # per-file physical path and keep stems/instrument keys canonical
            # (derived from the original source under the input root) so a
            # referenced file and a trimmed file share one stable identity.
            trimmed_keys = {p.resolve() for p in trimmed}
            next_p_files: list[Path] = []
            next_instruments: dict[Path, str] = {}
            next_stems: dict[Path, str] = {}
            for source in p_files:
                src_key = source.resolve()
                dest = trim_destination(source, trim_dir, root=current_root)
                if dest.resolve() in trimmed_keys:
                    physical = dest          # rewritten (trimmed) or reused (skipped)
                elif src_key in trimmed_keys:
                    physical = source        # complete — referenced in place
                else:
                    continue                 # trim failed for this file; drop it
                key = physical.resolve()
                next_p_files.append(physical)
                next_instruments[key] = instrument_key_by_path.get(
                    src_key, source.parent.name
                )
                next_stems[key] = _source_output_stem(source, current_root)
            p_files = next_p_files
            instrument_key_by_path = next_instruments
            output_stem_by_path = next_stems
            # current_root stays the input root: referenced files (the common
            # case) live there, and trimmed files resolve via the stem map
            # above rather than a relative-path computation.

    # Merge
    if files_cfg.get("merge", False):
        logger.info("Merging...")
        from odas_tpw.perturb.merge import plan_merge_outputs

        merge_dir = Path(files_cfg.get("output_root", "results/")) / "merged"
        merge_plan = plan_merge_outputs(p_files, merge_dir, root=current_root)
        merged_files = run_merge(
            config,
            p_files,
            include_singletons=True,
            input_root=current_root,
            merge_plan=merge_plan,
        )
        if merged_files:
            merged_keys = {p.resolve() for p in merged_files}
            next_instruments = {}
            next_output_stems = {}
            for output_path, chain in merge_plan:
                output_key = output_path.resolve()
                if output_key in merged_keys:
                    source = chain[0]
                    next_instruments[output_key] = instrument_key_by_path.get(
                        source.resolve(), source.parent.name
                    )
                    if len(chain) > 1:
                        # Genuinely merged → new file under merge_dir.
                        next_output_stems[output_key] = _source_output_stem(
                            output_path, merge_dir
                        )
                    else:
                        # Passthrough singleton — keep the canonical stem the
                        # trim stage assigned (the physical file may sit under
                        # the input root or the trim dir).
                        next_output_stems[output_key] = output_stem_by_path.get(
                            output_key, _source_output_stem(output_path, current_root)
                        )
                    continue
                for source in chain:
                    source_key = source.resolve()
                    if source_key in merged_keys:
                        next_instruments[source_key] = instrument_key_by_path.get(
                            source_key, source.parent.name
                        )
                        next_output_stems[source_key] = output_stem_by_path.get(
                            source_key, _source_output_stem(source, current_root)
                        )
            p_files = merged_files
            instrument_key_by_path = next_instruments
            output_stem_by_path = next_output_stems
            current_root = merge_dir

    _check_unique_output_stems(
        [
            output_stem_by_path.get(
                p.resolve(), _source_output_stem(p, current_root)
            )
            for p in p_files
        ]
    )

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
        return instrument_key_by_path.get(p.resolve(), p.parent.name)

    def _output_stem(p):
        return output_stem_by_path.get(p.resolve(), _source_output_stem(p, current_root))

    if jobs == 1:
        for p_path in p_files:
            logger.info("Processing %s...", p_path.name)
            result = _process_file_timed(
                p_path,
                config,
                gps,
                output_dirs,
                hotel_data=hotel_data,
                hotel_cfg=hotel_cfg,
                instrument_key=_instrument_key(p_path),
                output_stem=_output_stem(p_path),
            )
            logger.info("%s", _done_message(p_path.name, result))
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
                    _process_file_timed,
                    p,
                    config,
                    gps,
                    output_dirs,
                    hotel_data=hotel_data,
                    hotel_cfg=hotel_cfg,
                    instrument_key=_instrument_key(p),
                    output_stem=_output_stem(p),
                ): p
                for p in p_files
            }
            for future in as_completed(futures):
                p = futures[future]
                try:
                    result = future.result()
                    logger.info("%s", _done_message(p.name, result))
                except Exception as exc:
                    logger.error("processing %s: %s", p.name, exc)

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
