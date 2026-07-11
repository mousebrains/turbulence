# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Orchestration — per-file processing and full pipeline.

Reference: Code/process_P_files.m (233 lines), Code/mat2profile.m (170 lines)
"""

import contextlib
import hashlib
import json
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import xarray as xr

from odas_tpw.perturb.config import (
    config_dir_of,
    expand_config_dir,
    merge_config,
    resolve_output_dir,
)
from odas_tpw.perturb.config import (
    upstream_for as _upstream_for,
)
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
        # File is not under root. Use parent directory name to provide likely
        # uniqueness without relying on an absolute path that breaks cache portability.
        parts = (path.parent.name, path.stem)
    return "__".join(_safe_stem_part(str(part)) for part in parts)


# {stem}_prof{NNN}.nc — greedy stem so a stem containing "_prof" (or a current
# stem that is a prefix of an orphan's stem) still anchors on the final
# ``_prof{digits}.nc`` suffix, recovering the exact source stem.
_PROF_NC_RE = re.compile(r"^(?P<stem>.+)_prof\d+\.nc$")


def _prune_orphan_profile_ncs(stage_dir: Path, valid_stems: set[str]) -> int:
    """Delete per-profile NetCDFs whose source .p file is no longer present.

    The per-stage output dir is content-hashed on *config* and reused across
    runs, so its ``{stem}_prof{NNN}.nc`` files accumulate. The signature does
    not include the resolved input-file set, so re-running the same config with
    a changed set of .p files (one dropped, or an explicit reduced list) would
    leave orphaned NCs that the binning glob silently folds into the combos.

    Match the EXACT source stem (parse ``{stem}_prof{NNN}.nc`` and require
    ``stem in valid_stems``) rather than a name-prefix: a prefix test wrongly
    KEEPS an orphan whose stem has a current stem as a prefix (e.g. current
    ``a`` would protect a ``a_extra``-derived file). Only recognized profile
    NetCDFs are pruned, so files not matching the ``_prof{NNN}`` shape are left
    untouched and a file produced this run is never deleted. Returns the count.
    """
    if not stage_dir.exists():
        return 0
    pruned = 0
    for nc in stage_dir.glob("*_prof*.nc"):
        m = _PROF_NC_RE.match(nc.name)
        if m is None or m.group("stem") in valid_stems:
            continue  # not a recognized profile NC, or a current stem -> keep
        try:
            nc.unlink()
            pruned += 1
            logger.info("pruned orphaned %s (no source .p in current input set)", nc.name)
        except OSError as exc:
            logger.warning("could not prune %s: %s", nc.name, exc)
    return pruned


def _write_binned_or_clear(ds: "xr.Dataset", out_dir: Path, manifest: str | None = None) -> None:
    """Write *ds* to ``out_dir/binned.nc``, or remove a stale one if *ds* is empty.

    An empty re-run (same config but a reduced/changed input set) must not leave
    a prior run's ``binned.nc`` in place: ``resolve_output_dir`` hashes only the
    config, not the input file set, so the same dir is reused across input sets,
    and the combo glob would otherwise republish the stale file as current data
    (#56).

    *manifest* (a hash of the contributing per-file cache keys) is stored as the
    ``_input_manifest`` attribute so a later run can skip re-binning when the
    inputs are unchanged — keyed on content identity, NOT mtime (the data volume
    may be exFAT, whose 2 s mtime granularity makes mtime ordering unsafe).
    """
    out = out_dir / "binned.nc"
    if ds.data_vars:
        if manifest is not None:
            ds = ds.assign_attrs(_input_manifest=manifest)
        # Atomic write: to a temp file in the same dir, then os.replace into
        # place. A direct ds.to_netcdf(out) interrupted mid-payload (ENOSPC /
        # SMB drop / Ctrl-C) leaves a truncated binned.nc whose _input_manifest
        # header already validates as cache-current, permanently publishing
        # fill data with no error (audit r1-5). os.replace is atomic within a
        # filesystem, so a partial write never becomes the live file. (Mirrors
        # _write_marker and trim.py; the temp file shares out_dir's filesystem.)
        tmp = out.with_name(f".binned.nc.{os.getpid()}.tmp")
        try:
            ds.to_netcdf(tmp)
            os.replace(tmp, out)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
    elif out.exists():
        out.unlink()


def _atomic_to_netcdf(ds: "xr.Dataset", out_path: Path) -> None:
    """Write *ds* to *out_path* atomically (temp file + os.replace).

    A direct ds.to_netcdf(out_path) interrupted mid-payload (ENOSPC / SMB drop /
    Ctrl-C) or raising leaves a readable PARTIAL NetCDF on disk; the same run
    then bins it and the bin/combo manifest locks those fill-derived bins in, so
    a correct retry (identical filenames -> identical manifest) skips re-binning
    and permanently publishes the partial data (audit 2026-07-01). os.replace is
    atomic within a filesystem, so a partial write never becomes the live file.
    """
    tmp = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
    try:
        ds.to_netcdf(tmp)
        os.replace(tmp, out_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


_MANIFEST_ATTR = "_input_manifest"


def _inputs_manifest(
    nc_files: "list[Path]", file_cachekeys: dict[str, str], volatile_cfg: dict | None = None
) -> str:
    """A content-identity hash of a bin/combo's inputs: each input file paired
    with the per-file cache key of its source ``.p`` (or, for a binned input,
    that file's own manifest). mtime-immune, so it is safe on exFAT."""
    items: list[Any] = []
    for f in sorted(nc_files):
        m = _PROF_NC_RE.match(f.name)
        stem = m.group("stem") if m else f.stem
        items.append([f.name, file_cachekeys.get(stem, "?")])
    if volatile_cfg:
        items.append(["_volatile_cfg", volatile_cfg])
    return hashlib.sha256(
        json.dumps(items, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _output_is_current(nc_path: Path, manifest: str | None) -> bool:
    """True iff *manifest* is set, *nc_path* exists, and its ``_input_manifest``
    matches (None manifest → never current → always rebuild)."""
    return manifest is not None and _read_manifest_attr(nc_path) == manifest


def _read_manifest_attr(nc_path: Path) -> str | None:
    """The ``_input_manifest`` global attribute of *nc_path*, or None."""
    if not nc_path.exists():
        return None
    try:
        import xarray as xr

        with xr.open_dataset(nc_path) as ds:
            val = ds.attrs.get(_MANIFEST_ATTR)
            return str(val) if val is not None else None
    except (OSError, ValueError, KeyError):
        return None


def _stamp_manifest(nc_path: Path, manifest: str) -> None:
    """Add the ``_input_manifest`` global attribute to an already-written NC
    (header-only append — no data rewrite), so a combo can be skipped next run
    when its inputs are unchanged."""
    try:
        import netCDF4

        with netCDF4.Dataset(nc_path, "a") as nc:
            nc.setncattr(_MANIFEST_ATTR, manifest)
    except OSError as exc:
        logger.warning("could not stamp manifest on %s: %s", nc_path, exc)


def _prune_orphan_named_ncs(stage_dir: Path, valid_stems: set[str]) -> int:
    """Delete ``{stem}.nc`` per-file outputs whose source .p is no longer present.

    Sibling of :func:`_prune_orphan_profile_ncs` for products named exactly
    ``{stem}.nc`` (the CTD per-file outputs, ctd.py). Those carry no ``_prof``
    suffix, so the prefix-based pruner skips them and orphaned casts from a
    dropped .p file would leak into the CTD combo. Prune any ``*.nc`` whose stem
    is not in *valid_stems* (exact membership); a file produced this run always
    matches a current stem, so live data is never dropped. Returns the count.
    """
    if not stage_dir.exists():
        return 0
    pruned = 0
    for nc in stage_dir.glob("*.nc"):
        if nc.stem not in valid_stems:
            try:
                nc.unlink()
                pruned += 1
                logger.info(
                    "pruned orphaned CTD %s (no source .p in current input set)",
                    nc.name,
                )
            except OSError as exc:
                logger.warning("could not prune %s: %s", nc.name, exc)
    return pruned


# ---------------------------------------------------------------------------
# Per-file processing cache (incremental re-runs)
# ---------------------------------------------------------------------------
# A file's outputs are skipped on re-run when nothing that could change them has
# changed. The cache key folds: the input file's identity (size + mtime_ns of
# the trimmed/merged .p), external inputs referenced by path (hotel/GPS files),
# and the SIGNATURE HASHES of the output dirs the file targets — which already
# encode config + the engine-code fingerprint (see perturb.config). Keying on
# the signature hash, not the reusable ``{stage}_NN`` basename, prevents a stale
# marker matching a *different* config that happens to reuse a sequence number.
# A skip ALSO requires the recorded outputs to still exist on disk: a marker is
# a claim, the files are the truth — so a pruned/deleted output forces recompute.
_CACHE_DIRNAME = ".cache"
_SIG_PREFIX = ".params_sha256_"


def _file_fingerprint(p: Path) -> dict:
    """Cheap content-change proxy for a file: size + mtime binned to 2 s.

    mtime is floored to a 2 s boundary so the cache survives a copy to exFAT
    (2 s mtime granularity); the trade-off is that a same-size edit landing in
    the same 2 s bin is not re-detected (accepted file-identity model). A missing
    file yields ``{"missing": True}`` (no skip will ever match it, so it is
    reprocessed — and process_file then surfaces the real error)."""
    try:
        st = p.stat()
    except OSError:
        return {"missing": True}
    return {"size": st.st_size, "mtime_2s": int(st.st_mtime) // 2 * 2}


def _external_input_fingerprints(config: dict, hotel_cfg: dict | None) -> dict:
    """Fingerprint hotel/GPS files referenced *by path* in the config.

    Their *contents* can change ε/χ/position while the config (and thus every
    stage-dir hash) stays identical, so they must enter the per-file key or an
    edited hotel/GPS file would be silently ignored on re-run.
    """
    out: dict[str, dict] = {}
    cdir = config_dir_of(config)
    gps_file = merge_config("gps", config.get("gps")).get("file")
    for name, path in (("hotel", (hotel_cfg or {}).get("file")), ("gps", gps_file)):
        if path:
            fp = Path(expand_config_dir(path, cdir))
            out[name] = _file_fingerprint(fp) if fp.exists() else {"missing": True}
    return out


def _stage_signature_hashes(output_dirs: dict[str, Path]) -> dict[str, str]:
    """Each output dir's stored ``.params_sha256_<hash>`` value — the config +
    engine identity of that stage (collision-proof, unlike its ``{stage}_NN``
    basename)."""
    out: dict[str, str] = {}
    for stage, d in output_dirs.items():
        sigs = sorted(d.glob(f"{_SIG_PREFIX}*"))
        if sigs:
            out[stage] = sigs[0].name[len(_SIG_PREFIX) :]
    return out


def _file_cachekey(
    p: Path, ext_fp: dict, stage_hashes: dict[str, str], volatile_cfg: dict | None = None
) -> str:
    payload: dict[str, Any] = {
        "input": _file_fingerprint(p),
        "external": ext_fp,
        "stages": stage_hashes,
    }
    if volatile_cfg:
        payload["volatile"] = volatile_cfg
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _marker_path(output_root: Path, output_stem: str) -> Path:
    return output_root / _CACHE_DIRNAME / f"{output_stem}.json"


def _outputs_for_stem(
    output_dirs: dict[str, Path], output_stem: str, output_root: Path
) -> list[str]:
    """Relative paths of the NetCDFs a file produced: per-profile
    ``{stem}_prof*.nc`` (profiles/diss/chi) and ``{stem}.nc`` (ctd)."""
    rels: list[str] = []
    for stage, d in output_dirs.items():
        pattern = f"{output_stem}.nc" if stage == "ctd" else f"{output_stem}_prof*.nc"
        rels.extend(nc.relative_to(output_root).as_posix() for nc in d.glob(pattern))
    return sorted(rels)


def _list_present_outputs(output_dirs: dict[str, Path], output_root: Path) -> set[str]:
    """All output-NetCDF relpaths currently on disk, gathered with ONE directory
    listing per stage. The per-file output-exists check is then set membership
    rather than a ``stat`` per NC — which on a slow/network mount (SMB) turns
    thousands of round-trips into a handful (one ``glob`` per stage dir)."""
    present: set[str] = set()
    for d in output_dirs.values():
        if d.exists():
            present.update(nc.relative_to(output_root).as_posix() for nc in d.glob("*.nc"))
    return present


def _marker_is_current(marker: Path, cachekey: str, present_outputs: set[str]) -> bool:
    """True iff the marker exists, its cachekey matches, AND every recorded
    output is present (membership in the pre-listed *present_outputs* set, so a
    pruned/deleted/foreign output still forces recompute — no per-output stat)."""
    try:
        data = json.loads(marker.read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(data, dict) or data.get("cachekey") != cachekey:
        return False  # unreadable/malformed marker -> miss, never a crash
    outputs = data.get("outputs")
    if not isinstance(outputs, list):
        return False
    return all(rel in present_outputs for rel in outputs)


def _write_marker(
    output_root: Path, output_stem: str, cachekey: str, output_dirs: dict[str, Path]
) -> None:
    """Atomically record a file's cache marker after a successful process_file."""
    marker = _marker_path(output_root, output_stem)
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cachekey": cachekey,
        "outputs": _outputs_for_stem(output_dirs, output_stem, output_root),
    }
    tmp = marker.with_name(f"{marker.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, marker)  # atomic; concurrent workers write distinct stems


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
                # Use only the accelerometers (Ax, Ay) to find the prop-wash /
                # instrument-startup exit. On ARCTERX VMP data the accelerometers
                # are the only channels that settle at the true exit (~2-6 m): the
                # shear probes (sh1/sh2), inclinometers, and fall rate all stay
                # "elevated" deep because they respond to *ocean* turbulence the
                # instrument falls through, dragging the trim to 30-50 m. The
                # accelerometers capture the mechanical entry transient, which
                # ocean turbulence does not reproduce. (VMP only; MRs are trimmed
                # by a separate operation.)
                fast_channels: dict = {}
                for name in ("Ax", "Ay"):
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


def _apply_fom_cut(ds, fom_max: float, file_label: str, two_sided: bool = False) -> int:
    """NaN per-probe per-segment values where the fom fails its cut.

    With ``two_sided=False`` (epsilon): ``fom >= fom_max``.
    With ``two_sided=True`` (chi): ``fom >= fom_max`` OR ``fom <= 1/fom_max``.
    The chi fom is an obs/model VARIANCE RATIO that is bad far from 1.0 in either
    direction, so a model that overestimates observed variance (fom << 1) must be
    cut too, not only a high fom.

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
    if two_sided and fom_max > 0:
        bad = bad | (np.isfinite(fom) & (fom <= 1.0 / fom_max))
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
        file_label,
        fom_max,
        n_bad,
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
        origin_s = np.datetime64(origin).astype("datetime64[ns]").astype("int64") / 1e9
        return vals.astype(np.float64) + origin_s
    return vals.astype(np.float64)


def _compute_slow_stratification(pf, profiles, T_name, C_name, window):
    """Per-cast sorted N2/dT/dz on the full slow grid (NaN outside casts).

    Computes background stratification once per detected cast with
    :func:`profile_stratification` on a *window*-dbar scale, then interpolates
    each cast's coarse result onto the slow pressure grid. Returns
    ``(N2_full, dTdz_full)`` aligned with the slow channels, or ``(None, None)``
    when pressure/temperature are unavailable. Practical salinity comes from
    conductivity via TEOS-10 when present. Position defaults to 0/0 (a <0.3%
    gravity and negligible salinity-anomaly effect on N2).
    """
    import gsw
    import numpy as np

    from odas_tpw.processing.mixing import profile_stratification

    P = pf.channels.get("P")
    T = pf.channels.get(T_name)
    if P is None or T is None:
        return None, None
    P = np.asarray(P, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    C = pf.channels.get(C_name)
    # Only usable if conductivity is a slow channel aligned with P/T; a fast C
    # would be silently misaligned by the slow-index slice below.
    if C is not None and getattr(pf, "is_fast", lambda _n: False)(C_name):
        C = None
    C = np.asarray(C, dtype=np.float64) if C is not None else None

    n = len(P)
    N2_full = np.full(n, np.nan)
    dTdz_full = np.full(n, np.nan)
    for s, e in profiles:
        sl = slice(s, e + 1)
        P_c, T_c = P[sl], T[sl]
        S_c = gsw.SP_from_C(C[sl], T_c, P_c) if C is not None else None
        target_P, N2, dTdz = profile_stratification(P_c, T_c, S=S_c, window=window)
        good = np.isfinite(N2)
        if good.sum() >= 2:
            N2_full[sl] = np.interp(P_c, target_P[good], N2[good])
        good_d = np.isfinite(dTdz)
        if good_d.sum() >= 2:
            dTdz_full[sl] = np.interp(P_c, target_P[good_d], dTdz[good_d])
    return N2_full, dTdz_full


def _window_stratification_for_profile(
    prof_path, win_t, half_w, T_name: str, C_name: str, file_label: str
):
    """Sorted N2/dTdz at window times *win_t* from a profile's CTD channels.

    Returns ``(N2, dTdz, sal_note)`` evaluated on the *win_t* grid, or ``None``
    when the profile cannot be read or lacks t_slow/P/T. Practical salinity
    comes from the profile's own conductivity via TEOS-10 when present, else
    35 PSU is assumed (recorded in *sal_note*). Shared by the diss and chi
    products, each evaluating stratification at its own window scale (the diss
    path passes diss_length_seconds/2, the chi path 0.5*chi diss_length/fs), so
    the two coincide only when their diss_length/fft_length settings match.
    """
    import gsw
    import numpy as np
    import xarray as xr

    from odas_tpw.processing.mixing import sorted_stratification

    try:
        prof = xr.open_dataset(prof_path, decode_times=False)
    except Exception as exc:
        logger.info(
            "stratification skipped for %s: cannot read profile (%s)",
            file_label,
            exc,
        )
        return None
    with prof:
        if any(v not in prof for v in ("t_slow", "P", T_name)):
            logger.info(
                "stratification skipped for %s: profile lacks t_slow/P/%s",
                file_label,
                T_name,
            )
            return None
        t_slow = _time_epoch_seconds(prof["t_slow"])
        P = prof["P"].values.astype(np.float64)
        T = prof[T_name].values.astype(np.float64)
        lat = float(prof["lat"].values) if "lat" in prof else np.nan
        lon = float(prof["lon"].values) if "lon" in prof else np.nan
        lat = 0.0 if not np.isfinite(lat) else lat
        lon = 0.0 if not np.isfinite(lon) else lon
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
    strat = sorted_stratification(win_t, half_w, t_slow, P, T, S=S, lat=lat, lon=lon)
    return strat.N2, strat.dTdz, sal_note


def _attach_window_stratification(ds, prof_path, half_w, T_name, C_name, file_label, scale_name):
    """Attach sorted N2/dTdz to a window-grid dataset (e.g. diss). No-op on failure.

    *half_w* is the window half-width [s]; *scale_name* names the window in the
    variable comments (e.g. "dissipation").
    """
    import xarray as xr

    if "t" not in ds:
        return
    out = _window_stratification_for_profile(
        prof_path, _time_epoch_seconds(ds["t"]), half_w, T_name, C_name, file_label
    )
    if out is None:
        return
    N2, dTdz, sal_note = out
    ds["N2"] = xr.DataArray(
        N2,
        dims=["time"],
        attrs={
            "units": "s-2",
            "long_name": "buoyancy frequency squared (window scale)",
            "comment": (
                "TEOS-10 (gsw.Nsquared) between the shallow- and deep-half means "
                f"of each {scale_name} window after Thorpe-sorting to a stable "
                f"profile (overturns removed); {sal_note}."
            ),
        },
    )
    ds["dTdz"] = xr.DataArray(
        dTdz,
        dims=["time"],
        attrs={
            "units": "K m-1",
            "long_name": "background temperature gradient (positive down)",
            "comment": (
                "Least-squares slope of the Thorpe-sorted in-situ temperature vs "
                f"depth over each {scale_name} window."
            ),
        },
    )


def _add_mixing_quantities(
    chi_ds,
    diss_ds,
    prof_path,
    T_name: str = "JAC_T",
    C_name: str = "JAC_C",
    file_label: str = "",
    epsilon_provenance: str = "",
):
    """Append stratification and (when available) mixing quantities to chi.

    Always computes window-scale N2 and dT/dz from the profile's own slow CTD
    channels via the Thorpe-sorted (adiabatically leveled) method — with
    practical salinity from C/T/P via TEOS-10 when the conductivity channel
    exists, so N2 is fully constrained (unlike the rsi path, which may assume
    35 PSU). N2 and dT/dz depend only on the CTD profile, so they are written
    regardless of epsilon/chi.

    When epsilonMean (from the matching diss dataset) and chiMean are both
    present, additionally computes K_T (Osborn-Cox), Gamma (Oakey 1982), and
    K_rho (Osborn 1980, Gamma_0 = 0.2), pairing epsilonMean onto the chi grid.

    Mutates and returns *chi_ds*.  Returns it unchanged (with a log line) only
    when the profile lacks the t_slow/P/T channels or chi window attributes
    needed to compute stratification at all.
    """
    import numpy as np
    import xarray as xr

    from odas_tpw.processing.mixing import mixing_coefficients, pair_nearest

    # N2/dTdz are stratification quantities — they need only the CTD profile,
    # not epsilon or chi — so they are always computed and attached. Only the
    # derived coefficients (K_T, Gamma, K_rho) require epsilon and/or chi.
    have_mix = diss_ds is not None and "epsilonMean" in diss_ds and "chiMean" in chi_ds

    try:
        chi_t = _time_epoch_seconds(chi_ds["t"])
        half_w = 0.5 * float(chi_ds.attrs["diss_length"]) / float(chi_ds.attrs["fs_fast"])
    except (KeyError, TypeError, ValueError):
        logger.info("mixing quantities skipped for %s: no chi time/window attrs", file_label)
        return chi_ds

    strat_out = _window_stratification_for_profile(
        prof_path, chi_t, half_w, T_name, C_name, file_label
    )
    if strat_out is None:
        return chi_ds
    N2, dTdz, sal_note = strat_out

    var_specs = {
        "N2": (
            N2,
            {
                "units": "s-2",
                "long_name": "buoyancy frequency squared (window scale)",
                "comment": (
                    "TEOS-10 (gsw.Nsquared) between the shallow- and deep-half "
                    "means of each chi window after Thorpe-sorting to a stable "
                    f"profile (overturns removed); {sal_note}."
                ),
            },
        ),
        "dTdz": (
            dTdz,
            {
                "units": "K m-1",
                "long_name": "background temperature gradient (positive down)",
                "comment": (
                    "Least-squares slope of the Thorpe-sorted in-situ "
                    "temperature vs depth over each chi window."
                ),
            },
        ),
    }

    # K_T / Gamma / K_rho need epsilon and/or chi; only add them when available.
    if have_mix:
        eps_on_chi = pair_nearest(
            _time_epoch_seconds(diss_ds["t"]), diss_ds["epsilonMean"].values, chi_t
        )
        mix = mixing_coefficients(eps_on_chi, chi_ds["chiMean"].values, N2, dTdz)
        var_specs.update(
            {
                # The exact epsilon that entered Gamma/K_rho, stored for
                # traceability: consumers (e.g. perturb-plot gamma-scaling)
                # can recompute ratio quantities from the same pairing the
                # pipeline used instead of re-pairing against the diss
                # product (and possibly choosing different windows).
                "epsilon_paired": (
                    eps_on_chi,
                    {
                        "units": "W/kg",
                        "long_name": "epsilonMean paired onto the chi window grid",
                        "comment": (
                            "Nearest-window epsilonMean from the matching diss "
                            "dataset (processing.mixing.pair_nearest); the "
                            "epsilon used in this dataset's Gamma and K_rho. "
                            "NaN where no diss window paired." + epsilon_provenance
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
                            "Canonical value ~0.2." + epsilon_provenance
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
                            "is near the canonical 0.2. NaN where N2 < 1e-9 s-2 or "
                            "K_rho > 10 m2 s-1 (physically implausible diffusivity: "
                            "the unbounded near-floor-N2 artifact, or contaminated "
                            "near-surface windows where epsilon is itself spurious)."
                            + epsilon_provenance
                        ),
                    },
                ),
            }
        )

    for name, (arr, attrs) in var_specs.items():
        chi_ds[name] = xr.DataArray(arr, dims=["time"], attrs=attrs)
    if have_mix:
        n_gamma = int(np.sum(np.isfinite(var_specs["Gamma"][0])))
        logger.info("mixing quantities for %s: %d valid Gamma estimates", file_label, n_gamma)
    else:
        n_n2 = int(np.sum(np.isfinite(N2)))
        logger.info(
            "stratification for %s: %d valid N2 windows (no epsilon/chi coefficients)",
            file_label,
            n_n2,
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


def _profile_practical_salinity(prof_path, T_name: str, C_name: str):
    """Slow-time practical salinity for a profile NetCDF, or None.

    Derives per-slow-sample practical salinity [PSU] from the profile's own
    conductivity, in-situ temperature, and pressure via TEOS-10
    (``gsw.SP_from_C`` — the same conversion used for the derived mixing
    quantities), returning ``None`` when any of the C/T/P channels are
    absent. Used to resolve ``chi.salinity: measured`` to an array the chi
    viscosity path consumes instead of the fixed 35-PSU default.
    """
    import gsw
    import numpy as np
    import xarray as xr

    with xr.open_dataset(prof_path, decode_times=False) as ds:
        if any(v not in ds for v in ("P", T_name, C_name)):
            return None
        P = ds["P"].values.astype(np.float64)
        T = ds[T_name].values.astype(np.float64)
        C = ds[C_name].values.astype(np.float64)
    return gsw.SP_from_C(C, T, P)


_SEAWATER_UNITS = {
    "SP": "1",
    "SA": "g/kg",
    "CT": "degree_Celsius",
    "sigma0": "kg m-3",
    "rho": "kg m-3",
}


def _inject_seawater_properties(pf, gps, T_name: str, C_name: str) -> tuple[str, ...]:
    """Compute SP/SA/CT/sigma0/rho on the slow grid and inject as pf channels.

    For the profile product: the full-rate slow ``T_name``/``C_name``/``P`` feed
    TEOS-10 (:func:`add_seawater_properties`) *before* depth-binning, so the
    nonlinear conversions see unaveraged inputs; binning then averages the
    derived properties like any other channel. ``T_name`` is the in-situ CTD
    reference (e.g. JAC_T), untouched by FP07 recalibration. Per-slow-sample
    lat/lon come from *gps* (SA's small position dependence). Long names /
    canonical units are layered from ``COMBO_SCHEMA`` at write time, so only the
    minimal ``units``/``type`` channel_info is set here.

    Returns the names injected, or ``()`` when the C/T/P slow channels are
    missing or length-mismatched (e.g. a fast conductivity).
    """
    import numpy as np

    if not all(n in pf.channels for n in (T_name, C_name, "P")):
        return ()
    T = np.asarray(pf.channels[T_name], dtype=np.float64)
    C = np.asarray(pf.channels[C_name], dtype=np.float64)
    P = np.asarray(pf.channels["P"], dtype=np.float64)
    n_slow = len(pf.t_slow)
    if not (len(T) == len(C) == len(P) == n_slow):
        return ()

    from odas_tpw.perturb.seawater import add_seawater_properties

    epoch = pf.start_time.timestamp() + np.asarray(pf.t_slow, dtype=np.float64)
    sw = add_seawater_properties(T, C, P, gps.lat(epoch), gps.lon(epoch))
    for name in ("SP", "SA", "CT", "sigma0", "rho"):
        pf.channels[name] = sw[name]
        pf.channel_info[name] = {
            "units": _SEAWATER_UNITS[name],
            "type": "derived",
            "name": name,
        }
    return ("SP", "SA", "CT", "sigma0", "rho")


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
        result.setdefault("errors", []).append(f"load: {exc}")
        return result

    # ---- Hotel data injection ----
    if hotel_data is not None:
        from odas_tpw.perturb.hotel import merge_hotel_into_pfile

        merge_hotel_into_pfile(hotel_data, pf, hotel_cfg or {})

    # ---- Speed channel (downstream of .p load + hotel merge) ----
    # Inject ``speed_fast`` and ``W_slow`` so extract_profiles writes them
    # to every per-profile NetCDF, and downstream prepare_profiles uses
    # them instead of recomputing from P. Default method is "pressure",
    # which reproduces the historical VMP behavior exactly.
    speed_cfg = merge_config("speed", config.get("speed"))
    if "P" in pf.channels:
        try:
            from odas_tpw.perturb.speed import compute_speed_for_pfile

            vehicle = pf.config.get("instrument_info", {}).get("vehicle", "").lower()
            speed_fast, W_slow = compute_speed_for_pfile(pf, speed_cfg, vehicle)

            pf.channels["speed_fast"] = speed_fast
            pf._fast_channels.add("speed_fast")
            pf.channel_info["speed_fast"] = {
                "units": "m s-1",
                "type": "computed",
                "name": "speed_fast",
            }
            pf.channels["W_slow"] = W_slow
            pf.channel_info["W_slow"] = {
                "units": "dbar s-1",
                "type": "computed",
                "name": "W_slow",
                "long_name": "profiling rate (smoothed |dP/dt|)",
            }
        except Exception as exc:
            logger.warning(
                "speed channel computation failed for %s (method=%s): %s",
                p_path.name,
                speed_cfg.get("method", "pressure"),
                exc,
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
                "qc.rules evaluation failed for %s: %s",
                p_path.name,
                exc,
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
    direction = "down"  # resolved below when a pressure channel is present
    P_slow = pf.channels.get("P")
    with stage_log(output_dirs.get("profiles"), log_basename):
        if P_slow is None:
            logger.warning("No pressure channel in %s", p_path.name)
        else:
            # Resolve "auto" → vehicle default (e.g. slocum_glider → "glide").
            # ``scor160.profile.get_profiles`` doesn't know "auto" itself, so
            # without this it silently falls through to the "down" branch and
            # we lose every up-profile -- on a glider with MR-on-during-climb
            # only, that drops *all* the real flight data.
            from odas_tpw.rsi.vehicle import resolve_direction, resolve_tau

            vehicle = pf.config.get("instrument_info", {}).get("vehicle", "").lower()
            # Smooth the detection fall rate with the VEHICLE tau, matching ODAS
            # (odas_p2mat.m). The default 1.5 s is correct for a VMP but 2-40x
            # too fast for gliders/floats (slocum 3.0 s, argo 60.0 s), which made
            # W noisy at the W_min threshold and fragmented profile boundaries.
            W = _smooth_fall_rate(P_slow, pf.fs_slow, tau=resolve_tau(vehicle))
            direction = resolve_direction(
                profiles_cfg.get("direction", "auto"),
                vehicle,
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

    # ---- Background stratification (N2, dT/dz) on the slow grid ----
    # Injected as slow pf channels once, before the CTD and profile forks, so
    # extract_profiles writes them into every per-profile NetCDF and
    # ctd_bin_file bins them into the CTD product — all independent of eps/chi.
    strat_cfg = merge_config("stratification", config.get("stratification"))
    if profiles and bool(strat_cfg.get("enable", True)):
        try:
            N2_full, dTdz_full = _compute_slow_stratification(
                pf,
                profiles,
                ct_cfg.get("T_name", "JAC_T"),
                ct_cfg.get("C_name", "JAC_C"),
                float(strat_cfg.get("window", 2.0)),
            )
            if N2_full is not None:
                win = float(strat_cfg.get("window", 2.0))
                pf.channels["N2"] = N2_full
                pf.channel_info["N2"] = {
                    "units": "s-2",
                    "type": "derived",
                    "name": "N2",
                    "long_name": (
                        f"buoyancy frequency squared (background, {win:g}-dbar Thorpe-sorted)"
                    ),
                    "comment": (
                        "TEOS-10 N2 from the profile's own C/T/P over a "
                        f"{win:g}-dbar pressure window, Thorpe-sorted to a stable "
                        "profile. Background (profile/CTD) scale — distinct from "
                        "the dissipation/chi-window N2 in the diss/chi products."
                    ),
                }
                pf.channels["dTdz"] = dTdz_full
                pf.channel_info["dTdz"] = {
                    "units": "K m-1",
                    "type": "derived",
                    "name": "dTdz",
                    "long_name": (
                        "background temperature gradient (positive down, "
                        f"{win:g}-dbar Thorpe-sorted)"
                    ),
                    "comment": (
                        "Least-squares slope of the Thorpe-sorted in-situ "
                        f"temperature vs depth over a {win:g}-dbar pressure window."
                    ),
                }
        except Exception as exc:
            logger.warning("stratification failed for %s: %s", p_path.name, exc)

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
                    profiles=profiles,
                    direction=direction,
                )
            except Exception as exc:
                logger.error("CTD binning %s: %s", p_path.name, exc)
                result.setdefault("errors", []).append(f"ctd: {exc}")

    if P_slow is None or not profiles:
        return result

    # ---- Seawater properties (SP/SA/CT/sigma0/rho) for the profile product ----
    # Injected AFTER the CTD fork, which derives its own from post-bin T/C/P;
    # here they are computed on the full slow-rate T/C/P and depth-binned with
    # the profile, so both products stay self-consistent without conflicting.
    try:
        _inject_seawater_properties(
            pf, gps, ct_cfg.get("T_name", "JAC_T"), ct_cfg.get("C_name", "JAC_C")
        )
    except Exception as exc:
        logger.warning("seawater properties failed for %s: %s", p_path.name, exc)

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
                # Record the in-situ calibration as per-channel provenance so
                # the products are self-describing (and plots can label it).
                # Only channels actually recalibrated are tagged — if the
                # reference was missing, cal_result is empty and the channels
                # keep the factory calibration with no tag.
                cal_tag = (
                    f"in-situ (Steinhart-Hart order {fp07_cfg.get('order', 2)} "
                    f"vs {fp07_cfg.get('reference', 'JAC_T')})"
                )
                for ch_name, cal_data in cal_result.get("channels", {}).items():
                    pf.channels[ch_name] = cal_data
                    pf.channel_info.setdefault(ch_name, {})["calibration"] = cal_tag
                for ch_name, cal_data in cal_result.get("fast_channels", {}).items():
                    pf.channels[ch_name] = cal_data
                    pf.channel_info.setdefault(ch_name, {})["calibration"] = cal_tag
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
                prof_scalars_cache = {str(p): s for p, s in zip(prof_paths, prof_scalars)}
            except Exception as exc:
                logger.error("extracting profiles %s: %s", p_path.name, exc)
                result.setdefault("errors", []).append(f"profiles: {exc}")
                return result

    # Per-profile dissipation. Resolve via merge_config so the DEFAULTS (e.g.
    # epsilon.fft_length=256) — and any None values, which merge_config drops —
    # are the values actually handed to _compute_epsilon/_compute_chi and the
    # diss_length_seconds below. config.get() raw would let an omitted/null
    # fft_length fall to _compute_epsilon's own 1024 default (mis-sizing windows
    # and making the diss-dir provenance hash, which IS merge_config-based, lie).
    eps_cfg = merge_config("epsilon", config.get("epsilon"))
    inst_lookup = instrument_key if instrument_key is not None else p_path.parent.name
    instrument_cfg = config.get("instruments", {}).get(inst_lookup, {})
    excluded_probes = list(instrument_cfg.get("exclude_shear_probes", []))
    chi_cfg = merge_config("chi", config.get("chi"))
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
    strat_cfg = merge_config("stratification", config.get("stratification"))
    strat_enabled = bool(strat_cfg.get("enable", True))
    ct_T_name = ct_cfg.get("T_name", "JAC_T")
    ct_C_name = ct_cfg.get("C_name", "JAC_C")
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
                            n_probe = int(ds["probe"].size) if "probe" in ds.dims else 2
                            apply_qc_to_dataset(
                                ds,
                                pf,
                                qc_eps_drop_from,
                                diss_length_seconds,
                                flag_var_name="qc_drop_epsilon",
                                value_vars=[
                                    "epsilonMean",
                                    "epsilonLnSigma",
                                    *[f"e_{i + 1}" for i in range(n_probe)],
                                    "epsilon",
                                ],
                                drop_action=qc_drop_action,
                            )
                        _copy_profile_scalars(prof_path, ds, prof_scalars_cache)
                        if strat_enabled:
                            _attach_window_stratification(
                                ds,
                                prof_path,
                                diss_length_seconds / 2,
                                ct_T_name,
                                ct_C_name,
                                p_path.name,
                                "dissipation",
                            )
                        out_name = Path(prof_path).name
                        out_path = output_dirs["diss"] / out_name
                        _atomic_to_netcdf(ds, out_path)
                        out_path_str = str(out_path)
                        result["diss"].append(out_path_str)
                        diss_by_profile[prof_path] = out_path_str
                except Exception as exc:
                    logger.error("diss for %s: %s", Path(prof_path).name, exc)
                    result.setdefault("errors", []).append(f"diss {Path(prof_path).name}: {exc}")

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
                if k
                not in (
                    "enable",
                    "chi_minimum",
                    "diagnostics",
                    "use_epsilon",
                    "fom_max",
                    "mixing",
                    "salinity",
                )
            }
            # Resolve chi.salinity: "measured" -> per-profile practical salinity
            # from the profile's own C/T/P (TEOS-10); a number or None is passed
            # through (None -> fixed 35 PSU viscosity in process_l3_chi).
            chi_sal_cfg = chi_cfg.get("salinity")
            chi_use_measured_sal = (
                isinstance(chi_sal_cfg, str) and chi_sal_cfg.strip().lower() == "measured"
            )
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
                        if chi_use_measured_sal:
                            chi_sal = _profile_practical_salinity(
                                prof_path,
                                ct_cfg.get("T_name", "JAC_T"),
                                ct_cfg.get("C_name", "JAC_C"),
                            )
                            if chi_sal is None:
                                logger.warning(
                                    "chi salinity='measured' but %s lacks C/T/P; "
                                    "using fixed 35 PSU",
                                    Path(prof_path).name,
                                )
                        else:
                            chi_sal = chi_sal_cfg
                        chi_results = _compute_chi(
                            prof_path,
                            epsilon_ds=diss_ds,
                            salinity=chi_sal,
                            **chi_kwargs,
                            _pre_loaded=pre_loaded,
                        )
                        for chi_ds in chi_results:
                            if chi_fom_max is not None:
                                _apply_fom_cut(
                                    chi_ds,
                                    float(chi_fom_max),
                                    p_path.name,
                                    two_sided=True,
                                )
                            chi_ds = mk_chi_mean(chi_ds, chi_cfg.get("chi_minimum", 1e-13))
                            # Apply chi QC BEFORE deriving mixing quantities so a
                            # QC-dropped chi does not leak into K_T/Gamma (audit
                            # r1-4). Mixing then consumes the NaN'd chiMean, so
                            # K_T and Gamma (chi-derived) are NaN wherever chi was
                            # dropped. K_rho is epsilon-derived (0.2*eps/N2) and is
                            # correctly gated instead by the epsilon-side QC, via
                            # the NaN epsilonMean paired in by _add_mixing_quantities.
                            if qc_enabled:
                                n_cprobe = (
                                    int(chi_ds["probe"].size) if "probe" in chi_ds.dims else 2
                                )
                                apply_qc_to_dataset(
                                    chi_ds,
                                    pf,
                                    qc_chi_drop_from,
                                    chi_diss_length_seconds,
                                    flag_var_name="qc_drop_chi",
                                    value_vars=[
                                        "chiMean",
                                        "chiLnSigma",
                                        *[f"chi_{i + 1}" for i in range(n_cprobe)],
                                        "chi",
                                        "epsilon_T",
                                    ],
                                    drop_action=qc_drop_action,
                                )
                            # Derived mixing quantities (Gamma, K_T, K_rho)
                            # with real salinity from the profile's own C/T/P
                            if chi_cfg.get("mixing", True):
                                mix_eps = (
                                    diss_ds if diss_ds is not None else xr.open_dataset(diss_path)
                                )
                                # Gamma and K_rho are built from the shear-probe
                                # epsilonMean even under Method 2 (use_epsilon=False),
                                # the setting chosen precisely when shear epsilon is
                                # distrusted. Record that provenance in the attrs and
                                # warn so a downstream user is not misled.
                                if not chi_use_epsilon:
                                    eps_prov = (
                                        " PROVENANCE: chi.use_epsilon=false (Method 2), "
                                        "so chi is a pure spectral fit, but this epsilon "
                                        "is still the shear-probe epsilonMean (the source "
                                        "Method 2 is chosen to distrust); treat Gamma/K_rho "
                                        "accordingly."
                                    )
                                    logger.warning(
                                        "%s: chi.use_epsilon=false but mixing Gamma/K_rho "
                                        "still use shear epsilonMean; see variable attrs",
                                        Path(prof_path).name,
                                    )
                                else:
                                    eps_prov = ""
                                try:
                                    chi_ds = _add_mixing_quantities(
                                        chi_ds,
                                        mix_eps,
                                        prof_path,
                                        T_name=ct_cfg.get("T_name", "JAC_T"),
                                        C_name=ct_cfg.get("C_name", "JAC_C"),
                                        file_label=Path(prof_path).name,
                                        epsilon_provenance=eps_prov,
                                    )
                                finally:
                                    if mix_eps is not diss_ds:
                                        mix_eps.close()
                            _copy_profile_scalars(prof_path, chi_ds, prof_scalars_cache)
                            out_name = Path(prof_path).name
                            out_path = output_dirs["chi"] / out_name
                            _atomic_to_netcdf(chi_ds, out_path)
                            result["chi"].append(str(out_path))
                    except Exception as exc:
                        logger.error("chi for %s: %s", Path(prof_path).name, exc)
                        result.setdefault("errors", []).append(f"chi {Path(prof_path).name}: {exc}")
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
    root = expand_config_dir(files_cfg.get("p_file_root", "VMP/"), config_dir_of(config))
    return Path(root)


def _configured_output_root(config: dict) -> Path:
    """The configured output root, with any ``<CONFIG_DIR>`` token resolved."""
    files_cfg = config.get("files", {})
    root = expand_config_dir(files_cfg.get("output_root", "results/"), config_dir_of(config))
    return Path(root)


def _check_unique_outputs(destinations: list[Path]) -> None:
    """Raise if multiple inputs would write the same output path."""
    seen: dict[Path, int] = {}
    for dest in destinations:
        key = dest.resolve()
        seen[key] = seen.get(key, 0) + 1
    duplicates = [str(path) for path, count in seen.items() if count > 1]
    if duplicates:
        raise ValueError(
            "multiple input files map to the same output path: " + ", ".join(sorted(duplicates))
        )


def _check_unique_output_stems(stems: list[str]) -> None:
    """Raise if multiple processed inputs would share one output stem."""
    seen: dict[str, int] = {}
    for stem in stems:
        seen[stem] = seen.get(stem, 0) + 1
    duplicates = [stem for stem, count in seen.items() if count > 1]
    if duplicates:
        raise ValueError(
            "multiple input files map to the same output stem: " + ", ".join(sorted(duplicates))
        )


def _trim_one(args: tuple):
    """Worker: trim one .p file and return (trim_result, source_path, error).

    Designed to be pickled and dispatched to a ProcessPoolExecutor — must
    take a single argument and import its dependencies inside the call.
    The first element is a :class:`~odas_tpw.perturb.trim.TrimResult` on
    success (or ``None`` with an error string on failure).
    """
    p, trim_dir, root, force, cache_dir = args
    try:
        from odas_tpw.perturb.trim import trim_p_file

        return (
            trim_p_file(p, trim_dir, root=root, force=force, cache_dir=cache_dir),
            p,
            None,
        )
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
        "Trim summary: %d trimmed, %d referenced in place, %d skipped (up to date), %d failed",
        counts["trimmed"],
        counts["referenced"],
        counts["skipped"],
        n_failed,
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
    output_root = _configured_output_root(config)
    trim_dir = output_root / "trimmed"
    cache_dir = output_root / _CACHE_DIRNAME
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
                result = trim_p_file(p, trim_dir, root=root, force=force, cache_dir=cache_dir)
                trim_results.append(result)
                logger.info("%s", _trim_log_message(result))
            except Exception as exc:
                n_failed += 1
                logger.error("trimming %s: %s", p.name, exc)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(_trim_one, (p, trim_dir, root, force, cache_dir)): p
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
    output_root = _configured_output_root(config)
    merge_dir = output_root / "merged"

    if p_files is None:
        pattern = files_cfg.get("p_file_pattern", "**/*.p")
        p_files = find_p_files(root, pattern)
    if not p_files:
        return []

    plan = (
        merge_plan if merge_plan is not None else plan_merge_outputs(p_files, merge_dir, root=root)
    )
    planned_outputs = [
        output_path for output_path, chain in plan if include_singletons or len(chain) > 1
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
    output_root = _configured_output_root(config)

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

    # Chi directory (if enabled). merge_config so the enable decision matches
    # process_file's (which now resolves chi via merge_config).
    chi_cfg = merge_config("chi", config.get("chi"))
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
    dissipation and chi counts (``0`` chi usually means chi was disabled, but
    also when no chi outputs were produced — e.g. every profile errored or had
    no matching diss output) and the per-file wall-clock when available.
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


def _process_file_timed(*args, cachekey: str | None = None, **kwargs) -> dict:
    """Run :func:`process_file`, recording its wall-clock as ``elapsed_s``.

    Wrapping rather than instrumenting process_file's several return points
    keeps the timing in one place. The elapsed seconds are measured inside the
    worker process, so they reflect true per-file processing time, not the
    time a file spent queued waiting for a free worker.

    On success, writes the per-file cache marker (in the worker, so the real
    outputs are on disk) when *cachekey* is supplied — keeping marker-writing
    out of ``process_file`` itself so the mocked-``process_file`` tests are
    unaffected.
    """
    import time

    if cachekey is not None:
        output_dirs = args[3] if len(args) > 3 else kwargs.get("output_dirs")
        output_stem = kwargs.get("output_stem")
        if output_dirs and output_stem:
            output_root = Path(next(iter(output_dirs.values()))).parent
            # Invalidate the old marker immediately. If processing is interrupted mid-write,
            # we don't want a corrupted NetCDF to falsely validate against the old marker
            # on a subsequent run.
            with suppress(OSError):
                _marker_path(output_root, output_stem).unlink(missing_ok=True)

            # Unlink previous outputs for this stem so a failure doesn't leave
            # stale NetCDFs that get silently swept up by the combo binning glob.
            for stage, d in output_dirs.items():
                pattern = f"{output_stem}.nc" if stage == "ctd" else f"{output_stem}_prof*.nc"
                for nc_file in Path(d).glob(pattern):
                    with suppress(OSError):
                        nc_file.unlink()

    t0 = time.monotonic()
    result = process_file(*args, **kwargs)
    if isinstance(result, dict):
        result["elapsed_s"] = time.monotonic() - t0
    # Only cache a CLEAN run. process_file catches and swallows real failures
    # (PFile load, profile extraction, per-profile diss/chi, CTD binning),
    # returning an empty/partial result with ``errors`` set. Writing a marker
    # then would lock that incomplete output in as cache-valid until --force /
    # an input or config change. A legitimately-empty file (no profiles found)
    # has no ``errors`` and IS cached. The marker stays invalidated (unlinked
    # above) so the next run retries.
    if cachekey is not None and isinstance(result, dict) and not result.get("errors"):
        output_dirs = args[3] if len(args) > 3 else kwargs.get("output_dirs")
        output_stem = kwargs.get("output_stem")
        if output_dirs and output_stem:
            output_root = Path(next(iter(output_dirs.values()))).parent
            try:
                _write_marker(output_root, output_stem, cachekey, output_dirs)
            except OSError as exc:
                logger.warning("could not write cache marker for %s: %s", output_stem, exc)
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
        root = _configured_input_root(config)
        pattern = files_cfg.get("p_file_pattern", "**/*.p")
        p_files = find_p_files(root, pattern)

    if not p_files:
        logger.warning("No .p files found")
        return

    logger.info("Found %d .p files", len(p_files))

    input_root = _configured_input_root(config)
    current_root = input_root
    instrument_key_by_path = {p.resolve(): p.parent.name for p in p_files}
    output_stem_by_path = {p.resolve(): _source_output_stem(p, current_root) for p in p_files}

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

            trim_dir = _configured_output_root(config) / "trimmed"
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
                    physical = dest  # rewritten (trimmed) or reused (skipped)
                elif src_key in trimmed_keys:
                    physical = source  # complete — referenced in place
                else:
                    continue  # trim failed for this file; drop it
                key = physical.resolve()
                next_p_files.append(physical)
                next_instruments[key] = instrument_key_by_path.get(src_key, source.parent.name)
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

        merge_dir = _configured_output_root(config) / "merged"
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
                        next_output_stems[output_key] = _source_output_stem(output_path, merge_dir)
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
            output_stem_by_path.get(p.resolve(), _source_output_stem(p, current_root))
            for p in p_files
        ]
    )

    # Setup output directories
    output_dirs = _setup_output_dirs(config)
    output_root = _configured_output_root(config)

    # GPS provider (resolve any <CONFIG_DIR> in gps.file before it hits disk)
    cdir = config_dir_of(config)
    gps_cfg = merge_config("gps", config.get("gps"))
    if gps_cfg.get("file"):
        gps_cfg = {**gps_cfg, "file": expand_config_dir(gps_cfg["file"], cdir)}
    gps = create_gps(gps_cfg)

    # Hotel data
    hotel_cfg = merge_config("hotel", config.get("hotel"))
    hotel_data = None
    if hotel_cfg.get("enable", False) and hotel_cfg.get("file"):
        from odas_tpw.perturb.hotel import load_hotel

        hotel_data = load_hotel(
            expand_config_dir(hotel_cfg["file"], cdir),
            hotel_cfg.get("time_column", "time"),
            hotel_cfg.get("time_format", "auto"),
            hotel_cfg.get("channels", {}),
        )
        logger.info("Loaded hotel file: %d channels", len(hotel_data.channels))

    def _instrument_key(p):
        return instrument_key_by_path.get(p.resolve(), p.parent.name)

    def _output_stem(p):
        return output_stem_by_path.get(p.resolve(), _source_output_stem(p, current_root))

    # Per-file incremental cache: compute each file's cache key (input identity +
    # external inputs + the config/engine-hashed output dirs it targets) and skip
    # files whose marker still matches AND whose outputs still exist on disk.
    # ``--force`` (files.force) reprocesses everything. Every file's key — skipped
    # or not — also seeds the bin/combo manifest (``file_cachekeys``).
    force = bool(files_cfg.get("force", False))
    ext_fp = _external_input_fingerprints(config, hotel_cfg)
    stage_hashes = _stage_signature_hashes(output_dirs)

    # Fold in config keys that affect per-file output but are excluded from
    # directory signature hashes (so they cause an in-place rebuild).
    per_file_volatile: dict[str, Any] = {}
    if "hotel" in config:
        per_file_volatile["hotel"] = config["hotel"]
    if config.get("ctd", {}).get("diagnostics"):
        per_file_volatile["ctd_diagnostics"] = True

    # One directory listing per stage up front, so each file's output-exists
    # check is set membership instead of a stat per NetCDF (SMB-friendly).
    present_outputs = _list_present_outputs(output_dirs, output_root)
    file_cachekeys: dict[str, str] = {}
    to_process: list[tuple[Path, str]] = []
    n_skipped = 0
    for p in p_files:
        stem = _output_stem(p)
        cachekey = _file_cachekey(p, ext_fp, stage_hashes, volatile_cfg=per_file_volatile)
        file_cachekeys[stem] = cachekey
        if not force and _marker_is_current(
            _marker_path(output_root, stem), cachekey, present_outputs
        ):
            n_skipped += 1
        else:
            to_process.append((p, cachekey))

    if n_skipped:
        logger.info(
            "Processing %d file(s) (jobs=%d); %d up to date (skipped)",
            len(to_process),
            jobs,
            n_skipped,
        )
    else:
        logger.info("Processing %d files (jobs=%d)...", len(to_process), jobs)

    if jobs == 1:
        for p_path, cachekey in to_process:
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
                cachekey=cachekey,
            )
            logger.info("%s", _done_message(p_path.name, result))
    else:
        # Spawn workers each get a per-pid log file inside <output_root>/logs/
        # so multi-process runs are diagnosable.  ``run_stamp`` is the parent
        # CLI's invocation timestamp so all workers share a prefix.
        worker_log_dir = output_root / "logs"
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
                    cachekey=cachekey,
                ): p
                for p, cachekey in to_process
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

    from odas_tpw.perturb.binning import bin_by_depth, bin_by_time, bin_chi, bin_diss

    prof_binned_dir: Path | None = None
    diss_binned_dir: Path | None = None
    chi_binned_dir: Path | None = None

    # Drop per-profile NCs orphaned by a prior run on a different input set
    # before globbing, so the combos reflect exactly the current .p files
    # (new files are still picked up incrementally; only files no longer
    # present are removed). See _prune_orphan_profile_ncs.
    valid_stems = {_output_stem(p) for p in p_files}
    for _stage in ("profiles", "diss", "chi"):
        if _stage in output_dirs:
            _prune_orphan_profile_ncs(output_dirs[_stage], valid_stems)
    # CTD per-file outputs are named {stem}.nc (no _prof suffix), so prune them
    # with exact-stem matching before the CTD combo globs the dir.
    if "ctd" in output_dirs:
        _prune_orphan_named_ncs(output_dirs["ctd"], valid_stems)

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
        binning_cfg.get("chi_aggregation") or binning_cfg.get("diss_aggregation") or aggregation
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

    binning_volatile = {"diagnostics": diagnostics} if diagnostics else None

    if prof_ncs and prof_binned_dir is not None:
        manifest = _inputs_manifest(prof_ncs, file_cachekeys, volatile_cfg=binning_volatile)
        if not force and _output_is_current(prof_binned_dir / "binned.nc", manifest):
            logger.info("Binning profiles... up to date (skipped)")
        else:
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
            _write_binned_or_clear(ds, prof_binned_dir, manifest)

    if diss_ncs and diss_binned_dir is not None:
        manifest = _inputs_manifest(diss_ncs, file_cachekeys, volatile_cfg=binning_volatile)
        if not force and _output_is_current(diss_binned_dir / "binned.nc", manifest):
            logger.info("Binning dissipation... up to date (skipped)")
        else:
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
            _write_binned_or_clear(ds, diss_binned_dir, manifest)

    if chi_ncs and chi_binned_dir is not None:
        manifest = _inputs_manifest(chi_ncs, file_cachekeys, volatile_cfg=binning_volatile)
        if not force and _output_is_current(chi_binned_dir / "binned.nc", manifest):
            logger.info("Binning chi... up to date (skipped)")
        else:
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
            _write_binned_or_clear(ds, chi_binned_dir, manifest)

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
        force=force,
        file_cachekeys=file_cachekeys,
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
    force: bool = False,
    file_cachekeys: dict[str, str] | None = None,
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
            output_root,
            stage,
            section,
            params,
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
        # Propagate the source binned.nc's input manifest: skip re-assembling
        # the combo when its (unchanged) inputs already produced it. None when
        # the binned dir carries no manifest (e.g. the standalone `perturb bin`
        # path) → always rebuild, safely.
        manifest = _read_manifest_attr(src / "binned.nc")
        if not force and _output_is_current(dst / "combo.nc", manifest):
            logger.info("%s up to date (skipped)", dst.name)
            continue
        with stage_log(dst, "combo"):
            try:
                out = func(src, dst, schema, netcdf_attrs=netcdf_attrs, method=method)
                if out is not None:
                    if manifest is not None:
                        _stamp_manifest(Path(out), manifest)
                    logger.info("Wrote %s", out)
                else:
                    # Empty binned source -> no combo produced. Remove any stale
                    # combo.nc so a shrunk/zeroed input set does not leave the
                    # previous run's combo as the apparently-current product
                    # (mirrors _write_binned_or_clear on the binning side).
                    stale = dst / "combo.nc"
                    if stale.exists():
                        stale.unlink()
                        logger.info("Removed stale %s (no binned input)", stale)
            except Exception as exc:
                logger.error("combo %s: %s", dst.name, exc)

    if ctd_dir is not None and ctd_dir.exists():
        ctd_p = merge_config("ctd", config.get("ctd")) if config is not None else None
        ctd_combo_dir = _resolve_dst("ctd_combo", "ctd", ctd_p)
        ctd_manifest = (
            _inputs_manifest(sorted(ctd_dir.glob("*.nc")), file_cachekeys)
            if file_cachekeys is not None
            else None
        )
        if not force and _output_is_current(ctd_combo_dir / "combo.nc", ctd_manifest):
            logger.info("%s up to date (skipped)", ctd_combo_dir.name)
        else:
            with stage_log(ctd_combo_dir, "combo"):
                try:
                    out = make_ctd_combo(
                        ctd_dir, ctd_combo_dir, CTD_SCHEMA, netcdf_attrs=netcdf_attrs
                    )
                    if out is not None:
                        if ctd_manifest is not None:
                            _stamp_manifest(Path(out), ctd_manifest)
                        logger.info("Wrote %s", out)
                except Exception as exc:
                    logger.error("ctd combo: %s", exc)
