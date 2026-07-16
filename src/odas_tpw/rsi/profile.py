# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Profile detection and extraction for vertical profilers.

Profile detection and smoothed fall-rate computation live in
:mod:`odas_tpw.scor160.profile` (instrument-independent).  This module
re-exports them for backward compatibility and adds RSI-specific I/O
(NetCDF extraction from PFile).
"""

import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np

if TYPE_CHECKING:
    from odas_tpw.rsi.p_file import PFile

# Re-export from scor160 — canonical implementations live there
from odas_tpw.scor160.profile import get_profiles, smooth_fall_rate

logger = logging.getLogger(__name__)

# Default longest slow-pressure gap [s] still treated as an isolated fill and
# linearly repaired. Longer non-finite spans are interpolated only so the
# fall-rate filter has a NaN-free array; profiles overlapping such a span are
# dropped rather than detected over fabricated pressure (audit / PR #79).
_MAX_REPAIR_GAP_S = 1.0

# Backward-compatible alias (previously underscore-prefixed)
_smooth_fall_rate = smooth_fall_rate


# Backward-compatible alias — computed from VEHICLE_ATTRIBUTES
def _build_vehicle_tau() -> dict[str, float]:
    from odas_tpw.rsi.vehicle import VEHICLE_ATTRIBUTES

    return {k: v[1] for k, v in VEHICLE_ATTRIBUTES.items()}


_VEHICLE_TAU = _build_vehicle_tau()


@overload
def extract_profiles(
    source: "PFile | str | Path",
    output_dir: str | Path,
    profiles: list[tuple[int, int]] | None = ...,
    gps: Any = ...,
    return_scalars: Literal[False] = ...,
    output_stem: str | None = ...,
    max_repair_gap_s: float = ...,
    extra_attrs: dict[str, Any] | None = ...,
    **profile_kwargs: Any,
) -> list[Path]: ...


@overload
def extract_profiles(
    source: "PFile | str | Path",
    output_dir: str | Path,
    profiles: list[tuple[int, int]] | None = ...,
    gps: Any = ...,
    *,
    return_scalars: Literal[True],
    output_stem: str | None = ...,
    max_repair_gap_s: float = ...,
    extra_attrs: dict[str, Any] | None = ...,
    **profile_kwargs: Any,
) -> tuple[list[Path], list[dict[str, float]]]: ...


def extract_profiles(
    source: "PFile | str | Path",
    output_dir: str | Path,
    profiles: list[tuple[int, int]] | None = None,
    gps: Any = None,
    return_scalars: bool = False,
    output_stem: str | None = None,
    max_repair_gap_s: float = _MAX_REPAIR_GAP_S,
    extra_attrs: dict[str, Any] | None = None,
    **profile_kwargs: Any,
) -> list[Path] | tuple[list[Path], list[dict[str, float]]]:
    """Extract profiles from a PFile or full-record NetCDF.

    Parameters
    ----------
    source : PFile, str, or Path
        A PFile object, path to .p file, or path to full-record .nc file.
    output_dir : Path
        Directory for per-profile NetCDF files.
    profiles : list of (start, end) tuples, optional
        Pre-computed profile bounds (slow-rate indices). If supplied, the
        internal call to :func:`get_profiles` is skipped — use this when
        the caller has already adjusted the bounds (e.g. via top-trim or
        bottom-crash detection).
    gps : GPSProvider, optional
        If supplied, lat/lon/stime/etime are written as scalar coordinate
        variables on each per-profile NetCDF for CF §9 profile featureType
        compliance.  ``gps`` must expose ``lat(t)`` and ``lon(t)`` taking a
        time array in *seconds since pf.start_time* (the same domain as
        ``pf.t_slow``).
    return_scalars : bool, default False
        When True, also return a list of dicts (parallel to paths) holding
        the per-profile scalar values written: ``{"lat", "lon", "stime",
        "etime"}`` (each missing if not written). The pipeline uses this
        to skip re-opening every profile NetCDF in the diss/chi steps.
    output_stem : str, optional
        Override the filename/global-title stem used for per-profile NetCDFs.
        When omitted, the source file stem is used.
    max_repair_gap_s : float, default 1.0
        Longest slow-pressure non-finite span [s] still treated as an
        isolated fill and linearly repaired. Longer gaps are interpolated
        only so the fall-rate filter has a NaN-free array; any profile
        overlapping such a fabricated span is dropped (relevant only to
        external/partial NetCDF inputs — raw ``.p`` pressure has no fills).
    extra_attrs : dict, optional
        Extra global attributes written to every per-profile NetCDF (after
        the source's own global attrs, so they win on collision). The
        perturb pipeline uses this for the speed provenance
        (``speed_method`` / ``speed_source``).
    **profile_kwargs
        Detection keyword arguments (P_min, W_min, direction, vehicle,
        min_duration). ``vehicle`` (default: from the source) resolves
        ``direction="auto"`` and the fall-rate smoothing tau; a
        missing/None ``W_min`` resolves from the direction (0.3 dbar/s
        free-fall, 0.05 glide/horizontal). The remaining kwargs pass
        through to :func:`get_profiles`.

    Returns
    -------
    list of Path
        Paths to per-profile NetCDF files written.
    list of dict
        Only when ``return_scalars=True``: per-profile raw scalar values
        (lat/lon as float64, stime/etime as epoch seconds float64).
    """
    import netCDF4 as nc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = _load_source(source)
    stem = output_stem or data["stem"]
    fs_slow = data["fs_slow"]
    fs_fast = data["fs_fast"]
    # Defect (audit M2 downstream): a single NaN in slow pressure (e.g. a
    # _FillValue gap from a partial/external NetCDF) is smeared across the
    # whole fall rate by smooth_fall_rate (gradient + zero-phase filtfilt),
    # silently dropping every profile. Repair isolated NaNs before detection;
    # long_gap flags fabricated long spans whose profiles are dropped below.
    max_gap = max(1, round(max_repair_gap_s * fs_slow))
    P_slow, long_gap = _repair_nans(data["P"], "P_slow", stem, max_gap)
    ratio = round(fs_fast / fs_slow)
    start_epoch_s = data.get("start_epoch_s")

    # Resolve vehicle-dependent detection parameters BEFORE get_profiles:
    # "auto" direction and a None W_min are meaningless to get_profiles (it
    # would silently treat "auto" as "down"), and the fall-rate smoothing tau
    # is a vehicle attribute (ODAS default_vehicle_attributes.ini) — a VMP
    # keeps tau=1.5 bit-identically, a Slocum gets 3.0.
    from odas_tpw.rsi.vehicle import resolve_detection

    vehicle = profile_kwargs.pop("vehicle", None)
    if vehicle is None:
        vehicle = data.get("vehicle", "")
    direction, tau, W_min = resolve_detection(
        profile_kwargs.pop("direction", "auto"),
        vehicle,
        W_min=profile_kwargs.pop("W_min", None),
    )

    # Compute smoothed fall rate from slow pressure
    W = _smooth_fall_rate(P_slow, fs_slow, tau=tau)

    if profiles is None:
        logger.info(
            "%s: profile detection direction=%s W_min=%g dbar/s (vehicle=%r)",
            stem,
            direction,
            W_min,
            vehicle,
        )
        profiles = get_profiles(
            P_slow, W, fs_slow, W_min=W_min, direction=direction, **profile_kwargs
        )
    if long_gap.any():
        # Drop profiles overlapping a fabricated long gap (their pressure /
        # boundaries are interpolated, not measured). Other casts survive.
        kept = [(s, e) for (s, e) in profiles if not long_gap[s : e + 1].any()]
        n_dropped = len(profiles) - len(kept)
        if n_dropped:
            logger.warning(
                "%s: dropped %d profile(s) overlapping a fabricated pressure gap",
                stem,
                n_dropped,
            )
        profiles = kept
    if not profiles:
        logger.warning(f"No profiles found in {stem}")
        return ([], []) if return_scalars else []

    # Lazy import: reaching perturb.atomic_io runs perturb/__init__ (config +
    # pipeline). Deferring it keeps that heavier perturb graph out of rsi at
    # module-load time (not a hard cycle — pipeline imports rsi.profile only
    # lazily — just a layering choice). Mirrors the netcdf_schema import below.
    from odas_tpw.perturb.atomic_io import tmp_sibling

    output_paths = []
    output_scalars: list[dict[str, float]] = []
    for pi, (s_slow, e_slow) in enumerate(profiles, 1):
        s_fast = s_slow * ratio
        # Clamp to the fast-axis length (matches _compute_epsilon): for a
        # full-record final profile (e_slow+1)*ratio can exceed len(t_fast) and
        # over-run the fast slices / size the time_fast dim too large.
        e_fast = min((e_slow + 1) * ratio, len(data["t_fast"]))
        s_slow_end = e_slow + 1

        prof_path = output_dir / f"{stem}_prof{pi:03d}.nc"

        # Write to a temp sibling and os.replace into place only after a clean
        # close, so a mid-write SMB/network drop on SeaChest never leaves a
        # readable partial *_profNNN.nc that the manifest then locks in on a
        # clean retry (identical source .p key -> skip). (#104 U5-2.)
        prof_tmp = tmp_sibling(prof_path)
        ds = nc.Dataset(str(prof_tmp), "w", format="NETCDF4")

        # Copy global attributes
        for attr in data["global_attrs"]:
            setattr(ds, attr, data["global_attrs"][attr])
        # Caller-supplied provenance (e.g. perturb's speed_method/speed_source)
        # wins over any same-named source attribute.
        if extra_attrs:
            for attr, value in extra_attrs.items():
                setattr(ds, attr, value)

        ds.Conventions = "CF-1.13, ACDD-1.3"
        ds.title = f"{stem} profile {pi}"
        ds.featureType = "profile"

        # Profile metadata
        ds.profile_number = pi
        ds.profile_start_index_slow = int(s_slow)
        ds.profile_end_index_slow = int(e_slow)
        ds.profile_P_start = float(P_slow[s_slow])
        ds.profile_P_end = float(P_slow[e_slow])
        ds.profile_duration_s = float((e_slow - s_slow) / fs_slow)
        ds.profile_mean_speed = float(np.mean(np.abs(W[s_slow:s_slow_end])))
        ds.history = f"Profile {pi} extracted on {datetime.now(UTC).isoformat()}"

        # Per-profile scalar coordinate variables (CF §9 profile featureType).
        # Keep these on the slow-rate timeline; gps must be in seconds since
        # the file's start_time, matching ``data["t_slow"]`` semantics.
        t_prof_s = float(data["t_slow"][s_slow])
        t_end_s = float(data["t_slow"][e_slow])
        scalars: dict[str, float] = {}
        if start_epoch_s is not None:
            stime_epoch = start_epoch_s + t_prof_s
            etime_epoch = start_epoch_s + t_end_s
            stime_var = ds.createVariable("stime", "f8", ())
            stime_var[...] = stime_epoch
            stime_var.units = "seconds since 1970-01-01"
            stime_var.standard_name = "time"
            stime_var.long_name = "profile start time"
            stime_var.calendar = "standard"
            stime_var.units_metadata = "leap_seconds: utc"
            stime_var.axis = "T"

            etime_var = ds.createVariable("etime", "f8", ())
            etime_var[...] = etime_epoch
            etime_var.units = "seconds since 1970-01-01"
            etime_var.standard_name = "time"
            etime_var.long_name = "profile end time"
            etime_var.calendar = "standard"
            etime_var.units_metadata = "leap_seconds: utc"
            scalars["stime"] = stime_epoch
            scalars["etime"] = etime_epoch
        if gps is not None:
            # GPSProvider expects epoch seconds — t_prof_s is file-relative.
            t_query = np.array([t_prof_s + (start_epoch_s or 0.0)])
            lat_val = float(np.asarray(gps.lat(t_query))[0])
            lon_val = float(np.asarray(gps.lon(t_query))[0])
            lat_var = ds.createVariable("lat", "f8", ())
            lat_var[...] = lat_val
            lat_var.units = "degrees_north"
            lat_var.standard_name = "latitude"
            lat_var.long_name = "profile latitude"
            lat_var.axis = "Y"
            lon_var = ds.createVariable("lon", "f8", ())
            lon_var[...] = lon_val
            lon_var.units = "degrees_east"
            lon_var.standard_name = "longitude"
            lon_var.long_name = "profile longitude"
            lon_var.axis = "X"
            scalars["lat"] = lat_val
            scalars["lon"] = lon_val

        n_fast = e_fast - s_fast
        n_slow = s_slow_end - s_slow
        ds.createDimension("time_fast", n_fast)
        ds.createDimension("time_slow", n_slow)

        # Time variables
        t_fast_var = ds.createVariable("t_fast", "f8", ("time_fast",), zlib=True)
        t_fast_var[:] = data["t_fast"][s_fast:e_fast]
        t_fast_var.units = data.get("t_fast_units", "seconds")
        t_fast_var.long_name = "time (fast channels)"
        t_fast_var.standard_name = "time"
        t_fast_var.calendar = "standard"
        t_fast_var.units_metadata = "leap_seconds: utc"
        t_fast_var.axis = "T"

        t_slow_var = ds.createVariable("t_slow", "f8", ("time_slow",), zlib=True)
        t_slow_var[:] = data["t_slow"][s_slow:s_slow_end]
        t_slow_var.units = data.get("t_slow_units", "seconds")
        t_slow_var.long_name = "time (slow channels)"
        t_slow_var.standard_name = "time"
        t_slow_var.calendar = "standard"
        t_slow_var.units_metadata = "leap_seconds: utc"
        t_slow_var.axis = "T"

        # Channel data — RSI's INI emits unit strings like ``umol_L-1`` and
        # ``deg`` that UDUNITS won't parse; canonicalise via the schema.
        from odas_tpw.perturb.netcdf_schema import COMBO_SCHEMA, canonicalize_units

        for ch_name, ch_data, dim, attrs in data["channels"]:
            trimmed = ch_data[s_fast:e_fast] if dim == "time_fast" else ch_data[s_slow:s_slow_end]
            var_name = ch_name.replace(" ", "_")
            var = ds.createVariable(var_name, "f4", (dim,), zlib=True)
            var[:] = trimmed.astype(np.float32)
            schema_entry = COMBO_SCHEMA.get(var_name, {})
            for k, v in attrs.items():
                # netCDF4 forbids setting these reserved attrs after creation
                # (e.g. _FillValue must be passed to createVariable). An
                # external/CF/ATOMIX source that declares a _FillValue would
                # otherwise crash the per-profile write with AttributeError.
                # The data is already NaN-filled (see _nc_filled), so dropping
                # _FillValue here is lossless.
                if k in _UNSETTABLE_NC_ATTRS:
                    continue
                if k == "units":
                    v = canonicalize_units(str(v))
                setattr(var, k, v)
            # Layer the canonical schema attrs (long_name, standard_name,
            # units_metadata, …) on top — schema wins where it has an opinion.
            for k, v in schema_entry.items():
                if k != "nc_name":
                    setattr(var, k, v)

        ds.fs_fast = float(fs_fast)
        ds.fs_slow = float(fs_slow)
        ds.close()
        # Publish atomically only after a complete close (U5-2): os.replace is
        # atomic within a filesystem, so prof_path never appears as a partial.
        # A mid-write crash leaves the hidden temp behind, but never a readable
        # partial product that the bin/combo manifest would lock in.
        os.replace(str(prof_tmp), str(prof_path))
        output_paths.append(prof_path)
        output_scalars.append(scalars)
        logger.info(
            f"Profile {pi}: P={P_slow[s_slow]:.1f}–{P_slow[e_slow]:.1f} dbar, "  # noqa: RUF001
            f"{(e_slow - s_slow) / fs_slow:.1f} s -> {prof_path.name}"
        )

    if return_scalars:
        return output_paths, output_scalars
    return output_paths


def _load_source(source: "PFile | str | Path") -> dict[str, Any]:
    """Load data from PFile, .p path, or .nc path into a common dict."""
    from odas_tpw.rsi.p_file import PFile

    if isinstance(source, PFile):
        return _load_from_pfile(source)

    source = Path(source)
    if source.suffix.lower() == ".p":
        return _load_from_pfile(PFile(source))
    elif source.suffix.lower() == ".nc":
        return _load_from_nc(source)
    else:
        raise ValueError(f"Unsupported file type: {source.suffix}")


def _load_from_pfile(pf: "PFile") -> dict[str, Any]:
    """Extract data dict from a PFile."""
    # Lazy imports: chi_io pulls xarray; helpers is only needed for the
    # channel-name patterns. Keeps profile.py light at module-load time.
    from odas_tpw.rsi.chi_io import (
        _DEFAULT_DIFF_GAIN,
        _extract_therm_cal,
        _therm_gradient_config,
    )
    from odas_tpw.rsi.helpers import DT_PATTERN, T_PATTERN
    from odas_tpw.rsi.p_file import instrument_sn

    channels = []
    for ch_name, ch_data in pf.channels.items():
        dim = "time_fast" if pf.is_fast(ch_name) else "time_slow"
        info = pf.channel_info[ch_name]
        # Honor a richer long_name/comment when the producer supplied one
        # (e.g. injected derived channels like N2/dTdz); otherwise fall back to
        # the bare channel name. The canonical schema still wins where it has an
        # opinion for known variables.
        attrs: dict[str, Any] = {
            "units": info["units"],
            "sensor_type": info["type"],
            "long_name": info.get("long_name", ch_name),
        }
        if info.get("comment"):
            attrs["comment"] = info["comment"]
        # Provenance (e.g. the in-situ FP07 calibration tag). Not a schema key,
        # so it survives apply_schema; binning carries it to the combo.
        if info.get("calibration"):
            attrs["calibration"] = info["calibration"]
        # FP07 electronics coefficients for the chi path (#131 m8): embed
        # diff_gain plus exactly _extract_therm_cal's output for the base
        # thermistor (including 'b' — noise_thermchannel's eta consumes it)
        # as float attrs on each pre-emphasized gradient channel, so chi
        # computed from a per-profile NetCDF uses the instrument's real
        # coefficients instead of generic defaults (SN479: diff_gain=0.912,
        # b=0.99861 vs the 0.94/1.0 fallbacks). NOTE: these attrs carry the
        # FACTORY calibration from the embedded config string; perturb's fp07
        # in-situ calibration may have rewritten the channel DATA in
        # pf.channels before this write. That is intended — the attrs
        # describe the electronics (noise floor, bilinear differentiator
        # gain), and factory coefficients still beat generic defaults.
        if dim == "time_fast" and DT_PATTERN.match(ch_name):
            diff_gain, therm_cal = _therm_gradient_config(pf.config, ch_name)
            attrs["diff_gain"] = diff_gain
            attrs.update(therm_cal)
        elif dim == "time_fast" and T_PATTERN.match(ch_name):
            # Plain fast T channel (no pre-emphasis): mirror the chi_io .p
            # branch's first-difference fallback exactly — diff_gain fixed at
            # the generic default, calibration from the channel's OWN config
            # section — so an instrument with no T*_dT* channels round-trips
            # through the per-profile NetCDF without a spurious "predates
            # their introduction" warning (W5-ii review F1).
            ch_cfg: dict = next(
                (ch for ch in pf.config["channels"] if ch.get("name") == ch_name),
                {},
            )
            attrs["diff_gain"] = _DEFAULT_DIFF_GAIN
            attrs.update(_extract_therm_cal(ch_cfg))
        channels.append((ch_name, ch_data, dim, attrs))

    global_attrs = {
        "Conventions": "CF-1.13, ACDD-1.3",
        "instrument_model": pf.config["instrument_info"].get("model", ""),
        "instrument_sn": instrument_sn(pf.config["instrument_info"]),
        "platform_type": pf.config["instrument_info"].get("vehicle", ""),
        "operator": pf.config["cruise_info"].get("operator", ""),
        "project": pf.config["cruise_info"].get("project", ""),
        "start_time": pf.start_time.isoformat(),
        "source_file": pf.filepath.name,
    }
    if hasattr(pf, "config_str"):
        global_attrs["configuration_string"] = pf.config_str

    return {
        "P": pf.channels["P"],
        "fs_fast": pf.fs_fast,
        "fs_slow": pf.fs_slow,
        "t_fast": pf.t_fast,
        "t_slow": pf.t_slow,
        "t_fast_units": f"seconds since {pf.start_time.isoformat()}",
        "t_slow_units": f"seconds since {pf.start_time.isoformat()}",
        "start_epoch_s": pf.start_time.timestamp(),
        "channels": channels,
        "global_attrs": global_attrs,
        "stem": pf.filepath.stem,
        "vehicle": pf.config["instrument_info"].get("vehicle", "").lower(),
    }


# netCDF4 reserved attributes that cannot be set with setattr after a variable
# is created (they must go through createVariable or are managed internally).
_UNSETTABLE_NC_ATTRS = frozenset(
    {"_FillValue", "_Netcdf4Coordinates", "_Netcdf4Dimid", "_Unsigned", "_ChunkSizes"}
)


def _nc_filled(var) -> np.ndarray:
    """Read a NetCDF variable as float64 with masked/_FillValue entries -> NaN.

    netCDF4 returns a *masked* array whenever a variable carries a ``_FillValue``
    or has unwritten elements. The previous ``var[:].data`` exposed the raw fill
    buffer (~9.97e36) instead of NaN, which silently poisoned pressure and
    fall-rate (a single fill produced a ~4.5e35 dbar/s spike that truncated the
    rest of the cast) and leaked into every channel. ``np.ma.filled`` converts
    masked entries to NaN and is a no-op on an already-unmasked ndarray, so this
    is correct for both the package's own fully-written files and external /
    partial (CF/ATOMIX) files that declare a ``_FillValue``.
    """
    return np.asarray(np.ma.filled(var[:].astype(np.float64), np.nan))


def _repair_nans(
    x: np.ndarray, name: str, stem: str, max_gap: int
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate NaNs (e.g. _FillValue gaps) over the sample index.

    Defect (audit M2 downstream): ``_nc_filled`` correctly turns a masked
    ``_FillValue`` into NaN, but a single NaN in the slow pressure is then
    smeared across the *entire* fall rate by ``smooth_fall_rate`` (gradient
    poisons two neighbors, zero-phase filtfilt poisons all samples), so a
    lone mid-cast fill silently drops every profile in the file. Repairing
    isolated NaNs by linear interpolation keeps the rest of the cast usable;
    a warning makes a poisoned/partial cast visible rather than silently lost.

    Every gap is interpolated so the fall-rate filter sees a NaN-free array,
    but a contiguous non-finite run longer than ``max_gap`` samples is
    *fabricated*, not repaired: linear interpolation across many missing
    samples invents pressure that must not define a profile (PR #79 review).
    The returned boolean mask flags those long-gap samples so the caller can
    drop any profile overlapping them.

    Returns ``(repaired, long_gap_mask)``.
    """
    long_gap = np.zeros(x.size, dtype=bool)
    bad = ~np.isfinite(x)
    n_bad = int(bad.sum())
    if n_bad == 0:
        return x, long_gap
    good = ~bad
    if good.sum() < 2:
        # Not enough finite samples to interpolate; leave as-is and warn loudly.
        logger.warning(
            "%s: %s has %d/%d non-finite samples; cannot repair", stem, name, n_bad, x.size
        )
        return x, long_gap
    # Flag contiguous non-finite runs longer than max_gap. Padding with a finite
    # sentinel at both ends turns run starts into +1 and ends into -1 in the diff.
    d = np.diff(np.concatenate(([0], bad.astype(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)  # exclusive
    n_long = 0
    for s, e in zip(starts, ends):
        if (e - s) > max_gap:
            long_gap[s:e] = True
            n_long += 1
    idx = np.arange(x.size, dtype=np.float64)
    repaired = x.copy()
    # np.interp clamps to the finite endpoints, so leading/trailing NaNs are
    # held flat rather than re-introducing NaN.
    repaired[bad] = np.interp(idx[bad], idx[good], x[good])
    if n_long:
        logger.warning(
            "%s: %s has %d non-finite span(s) longer than %d samples; interpolated for "
            "fall-rate only, profiles overlapping them will be dropped",
            stem,
            name,
            n_long,
            max_gap,
        )
    logger.warning("%s: interpolated %d/%d non-finite samples in %s", stem, n_bad, x.size, name)
    return repaired, long_gap


def _load_from_nc(nc_path: Path) -> dict[str, Any]:
    """Extract data dict from a full-record NetCDF file."""
    import netCDF4 as nc

    # Defect (audit io-hazard): the Dataset must be closed on every exit path,
    # not just the success path — the ValueError/AttributeError raises below
    # would otherwise leak the open file handle (and its lock).
    ds = nc.Dataset(str(nc_path), "r")
    stem = Path(nc_path).stem
    try:
        # Read global attributes
        global_attrs = {}
        for attr in ds.ncattrs():
            if attr != "configuration_string":
                global_attrs[attr] = getattr(ds, attr)
        if "configuration_string" in ds.ncattrs():
            global_attrs["configuration_string"] = ds.configuration_string

        # Attributes may be on root or in L1_converted group
        def _get_nc_attr(name: str, default=None):
            for src in [ds]:
                if name in {a for a in src.ncattrs()}:
                    return src.getncattr(name)
            if "L1_converted" in ds.groups:
                g = ds.groups["L1_converted"]
                if name in {a for a in g.ncattrs()}:
                    return g.getncattr(name)
            if default is not None:
                return default
            raise AttributeError(f"Attribute {name!r} not found in NetCDF file")

        fs_fast = float(_get_nc_attr("fs_fast"))
        fs_slow = float(_get_nc_attr("fs_slow"))

        # Time vectors may be at root or inside L1_converted group
        if "t_fast" in ds.variables:
            t_fast = _nc_filled(ds.variables["t_fast"])
            t_slow = _nc_filled(ds.variables["t_slow"])
        elif "L1_converted" in ds.groups:
            g = ds.groups["L1_converted"]
            # L1_converted uses TIME/TIME_SLOW dimension names
            t_fast = _nc_filled(g.variables["TIME"])
            t_slow = _nc_filled(g.variables["TIME_SLOW"])
        else:
            raise ValueError("No time variables found in NetCDF file")
        # Determine time units
        if "t_fast" in ds.variables:
            t_fast_units = (
                ds.variables["t_fast"].units
                if hasattr(ds.variables["t_fast"], "units")
                else "seconds"
            )
            t_slow_units = (
                ds.variables["t_slow"].units
                if hasattr(ds.variables["t_slow"], "units")
                else "seconds"
            )
        elif "L1_converted" in ds.groups:
            g = ds.groups["L1_converted"]
            t_fast_units = (
                g.variables["TIME"].units if hasattr(g.variables["TIME"], "units") else "days"
            )
            t_slow_units = (
                g.variables["TIME_SLOW"].units
                if hasattr(g.variables["TIME_SLOW"], "units")
                else "days"
            )
        else:
            t_fast_units = "seconds"
            t_slow_units = "seconds"

        # Pressure — need slow-rate pressure for profile detection
        if "P" in ds.variables:
            P = _nc_filled(ds.variables["P"])
        elif "L1_converted" in ds.groups:
            g = ds.groups["L1_converted"]
            if "PRES_SLOW" in g.variables:
                P = _nc_filled(g.variables["PRES_SLOW"])
            elif "PRES" in g.variables:
                P = _nc_filled(g.variables["PRES"])
            else:
                raise ValueError("No pressure variable found in NetCDF file")
        else:
            raise ValueError("No pressure variable found in NetCDF file")

        # Channels — scan both root and L1_converted group
        channels = []
        _seen = set()

        def _scan_vars(source, time_dim_fast, time_dim_slow):
            for vname in source.variables:
                if vname in _seen or vname in ("t_fast", "t_slow", "TIME", "TIME_SLOW"):
                    continue
                var = source.variables[vname]
                dims = var.dimensions
                if len(dims) != 1:
                    continue
                dim = dims[0]
                if dim == time_dim_fast:
                    mapped_dim = "time_fast"
                elif dim == time_dim_slow:
                    mapped_dim = "time_slow"
                else:
                    continue
                attrs = {}
                for a in var.ncattrs():
                    attrs[a] = getattr(var, a)
                channels.append((vname, _nc_filled(var), mapped_dim, attrs))
                _seen.add(vname)

        # Scan root-level variables (old format)
        _scan_vars(ds, "time_fast", "time_slow")
        # Scan L1_converted group (new ATOMIX format)
        if "L1_converted" in ds.groups:
            _scan_vars(ds.groups["L1_converted"], "TIME", "TIME_SLOW")
    finally:
        ds.close()

    # If time is in days, convert to seconds for consistency
    if "day" in t_fast_units.lower():
        # nanmin reference (not [0]): a _FillValue->NaN at index 0 of the TIME
        # axis would otherwise poison the whole converted vector.
        t_fast = (t_fast - np.nanmin(t_fast)) * 86400.0
        t_slow = (t_slow - np.nanmin(t_slow)) * 86400.0
        t_fast_units = "seconds"
        t_slow_units = "seconds"

    # Vehicle: explicit vehicle attr, then platform_type (what p_to_L1
    # writes), then a model-string heuristic — shared with the eps/chi NC
    # loader so both resolve the same vehicle for the same file.
    from odas_tpw.rsi.vehicle import vehicle_from_nc_attrs

    vehicle = vehicle_from_nc_attrs(global_attrs)

    return {
        "P": P.astype(np.float64),
        "fs_fast": fs_fast,
        "fs_slow": fs_slow,
        "t_fast": t_fast,
        "t_slow": t_slow,
        "t_fast_units": t_fast_units,
        "t_slow_units": t_slow_units,
        "channels": channels,
        "global_attrs": global_attrs,
        "stem": stem,
        "vehicle": vehicle,
    }


def _extract_one(args: tuple) -> tuple[str, int]:
    """Worker function for parallel profile extraction."""
    source_path, output_dir, kwargs = args
    paths = extract_profiles(source_path, output_dir, **kwargs)
    return str(source_path), len(paths)
