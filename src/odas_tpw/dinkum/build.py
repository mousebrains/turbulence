# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""Project Dinkum sensors onto a common time basis and write a hotel file.

The shape of the problem: a Slocum record carries only the sensors that
reported on that cycle, and there is no single clock. So a sensor's samples are
the rows where *it* is finite, timed by *its* clock (``time_sensor``), and
putting several sensors side by side means projecting each from its own
irregular sample times onto one shared basis.

Pipeline, per :func:`build_hotel`:

1. Sanitize the base time sensor -> the output time vector (finite, inside the
   valid range, sorted, deduped, strictly increasing). Native sample times, not
   a resampled grid: gaps stay gaps.
2. For each sensor: pair its finite values with its own time sensor's valid
   stamps, drop out-of-range values, collapse duplicate timestamps, then
   interpolate onto the output vector.
3. Apply ``scale``/``offset``, NaN across gaps wider than ``max_gap``, and
   write, with a provenance record of what each rule discarded.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from odas_tpw.dinkum.config import merge_config, normalize_sensors, required_sensor_names
from odas_tpw.dinkum.reader import load_dinkum
from odas_tpw.perturb.config import expand_config_dir

logger = logging.getLogger(__name__)

# Fallback time bounds when the YAML leaves them null. The floor rejects the
# 0.0 / tiny values Slocum writes for "this field never got set"; the ceiling
# rejects a clock that ran away forward (an unset RTC after a battery swap
# reads far in the future).
DEFAULT_MIN_TIME = 100.0
DEFAULT_MAX_TIME_HORIZON_DAYS = 365.0

_DEDUPE_METHODS = ("mean", "first", "last")  # same set as config.DEDUPE_METHODS


def _parse_time_bound(value: Any, label: str) -> float | None:
    """Accept epoch seconds or an ISO-8601 date string; ``None`` passes through."""
    if value is None:
        return None
    if isinstance(value, bool):  # bool is an int subclass; never a time
        raise ValueError(f"time.{label}={value!r}: expected a number or ISO-8601 date")
    if isinstance(value, (int, float)):
        v = float(value)
        if not np.isfinite(v):
            raise ValueError(f"time.{label}={value!r}: not finite")
        return v
    if isinstance(value, str):
        text = value.strip()
        try:
            # datetime.fromisoformat handles "Z" from Python 3.11 on.
            parsed = dt.datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(
                f"time.{label}={value!r}: not epoch seconds or an ISO-8601 date ({exc})"
            ) from exc
        if parsed.tzinfo is None:
            # A bare date means UTC here; guessing local time would shift the
            # bound by the operator's offset and silently reject good data.
            parsed = parsed.replace(tzinfo=dt.UTC)
        return parsed.timestamp()
    if isinstance(value, (dt.datetime, dt.date)):
        if isinstance(value, dt.datetime):
            d = value if value.tzinfo else value.replace(tzinfo=dt.UTC)
        else:
            d = dt.datetime(value.year, value.month, value.day, tzinfo=dt.UTC)
        return d.timestamp()
    raise ValueError(f"time.{label}={value!r}: expected a number or ISO-8601 date")


def resolve_time_bounds(
    min_value: Any = None, max_value: Any = None, *, now: float | None = None
) -> tuple[float, float]:
    """Resolve the valid time window to concrete epoch seconds.

    Either bound may be epoch seconds, an ISO-8601 date, or ``None``. ``None``
    falls back to :data:`DEFAULT_MIN_TIME` and *now* + 365 days. Pass *now* to
    make the fallback ceiling deterministic (tests, reproducible reruns).
    """
    lo = _parse_time_bound(min_value, "min_value")
    hi = _parse_time_bound(max_value, "max_value")
    if lo is None:
        lo = DEFAULT_MIN_TIME
    if hi is None:
        base = dt.datetime.now(dt.UTC).timestamp() if now is None else float(now)
        hi = base + DEFAULT_MAX_TIME_HORIZON_DAYS * 86400.0
    if not lo < hi:
        raise ValueError(f"time: min_value ({lo:g}) must be < max_value ({hi:g}) in epoch seconds")
    return lo, hi


def time_validity(t: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Boolean mask of usable timestamps: finite and inside ``[lo, hi]``."""
    t = np.asarray(t, dtype=np.float64)
    ok = np.isfinite(t)
    # Compare only where finite: a NaN comparison is False anyway, but doing it
    # under the mask keeps numpy from warning on all-NaN inputs.
    ok &= (t >= lo) & (t <= hi)
    return ok


def dedupe_samples(
    t: np.ndarray, v: np.ndarray, method: str
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse samples sharing a timestamp. Input must be sorted by ``t``.

    Public because ``fp07cal`` uses it too: one implementation of "what does a
    repeated timestamp mean" rather than one per subpackage that quietly
    disagree.
    """
    if t.size == 0:
        return t, v
    uniq, start_idx, counts = np.unique(t, return_index=True, return_counts=True)
    if uniq.size == t.size:
        return t, v  # already unique — the common case, no work
    if method == "first":
        return uniq, v[start_idx]
    if method == "last":
        return uniq, v[start_idx + counts - 1]
    # "mean": group-average. inverse indices give each sample its group.
    inv = np.searchsorted(uniq, t)
    sums = np.bincount(inv, weights=v, minlength=uniq.size)
    return uniq, sums / counts


def sanitize_time(
    t: np.ndarray, lo: float, hi: float, dedupe: str = "mean"
) -> tuple[np.ndarray, dict]:
    """Base time sensor -> the output time vector, plus a rejection tally.

    Returns ``(times, stats)`` where *times* is strictly increasing and
    *stats* records how many samples each rule removed.
    """
    if dedupe not in _DEDUPE_METHODS:
        raise ValueError(f"time.dedupe={dedupe!r}: not one of {list(_DEDUPE_METHODS)}")
    t = np.asarray(t, dtype=np.float64)
    n_total = t.size
    n_nan = int(np.sum(~np.isfinite(t)))
    ok = time_validity(t, lo, hi)
    n_out_of_range = int(n_total - n_nan - np.sum(ok))
    kept = np.sort(t[ok])
    n_valid = kept.size
    # np.unique sorts and dedupes in one pass; the output grid carries no
    # values, so the dedupe method is irrelevant here (all duplicates are the
    # same number). It matters for sensor values, in dedupe_samples.
    times = np.unique(kept)
    stats = {
        "n_total": n_total,
        "n_nan": n_nan,
        "n_out_of_range": n_out_of_range,
        "n_duplicate": int(n_valid - times.size),
        "n_kept": int(times.size),
    }
    return times, stats


def _interp(
    src_t: np.ndarray,
    src_v: np.ndarray,
    dst_t: np.ndarray,
    method: str,
    extrapolate: bool,
) -> np.ndarray:
    """Interpolate onto ``dst_t``; outside the source range -> NaN unless asked."""
    from scipy.interpolate import PchipInterpolator, interp1d

    if src_t.size < 2:
        # A single sample cannot define an interpolant. With extrapolation on,
        # holding it across the record is the documented behavior; otherwise
        # there is nothing to say.
        if src_t.size == 1 and extrapolate:
            return np.full(dst_t.shape, float(src_v[0]))
        return np.full(dst_t.shape, np.nan)

    if method == "pchip":
        f = PchipInterpolator(src_t, src_v, extrapolate=extrapolate)
        return np.asarray(f(dst_t), dtype=np.float64)
    fill: Any = "extrapolate" if extrapolate else np.nan
    f = interp1d(src_t, src_v, kind=method, bounds_error=False, fill_value=fill, assume_sorted=True)
    return np.asarray(f(dst_t), dtype=np.float64)


def _apply_max_gap(
    src_t: np.ndarray, dst_t: np.ndarray, values: np.ndarray, max_gap: float
) -> tuple[np.ndarray, int]:
    """NaN outputs whose bracketing source samples straddle a gap > ``max_gap``.

    Interpolating across a dropout produces a smooth line with no data under
    it. This blanks those stretches rather than publishing them.
    """
    if src_t.size < 2 or not np.isfinite(max_gap):
        return values, 0
    # For each output time, the source interval that contains it.
    right = np.searchsorted(src_t, dst_t, side="right")
    inside = (right >= 1) & (right < src_t.size)
    values = values.copy()
    n_blanked = 0
    if np.any(inside):
        idx = right[inside]
        gaps = src_t[idx] - src_t[idx - 1]
        too_wide = gaps > max_gap
        if np.any(too_wide):
            target = np.flatnonzero(inside)[too_wide]
            # An output time that coincides EXACTLY with a source sample is
            # measured, not interpolated, so a wide neighbouring gap must not
            # blank it. This is the common case rather than a corner: every
            # sensor riding the base clock lands on the output times exactly,
            # so without this a max_gap would NaN the real sample on each side
            # of every dropout.
            left_idx = np.searchsorted(src_t, dst_t[target], side="left")
            exact = (left_idx < src_t.size) & (
                src_t[np.minimum(left_idx, src_t.size - 1)] == dst_t[target]
            )
            target = target[~exact]
            # Only count values that were not already NaN.
            n_blanked = int(np.sum(np.isfinite(values[target])))
            values[target] = np.nan
    return values, n_blanked


def project_sensor(
    src_t: np.ndarray,
    src_v: np.ndarray,
    dst_t: np.ndarray,
    *,
    method: str = "linear",
    dedupe: str = "mean",
    extrapolate: bool = False,
    max_gap: float | None = None,
    valid_min: float | None = None,
    valid_max: float | None = None,
    time_lo: float = DEFAULT_MIN_TIME,
    time_hi: float = float("inf"),
) -> tuple[np.ndarray, dict]:
    """Project one sensor's samples onto ``dst_t``.

    *src_t* and *src_v* are the raw per-record arrays for the sensor's time
    sensor and the sensor itself (same length, NaN where absent). Returns
    ``(values_on_dst_t, stats)``.

    Order matters: the range check runs on the SOURCE samples, before
    interpolation, so an out-of-range spike is removed rather than being
    smeared into its neighbours.
    """
    src_t = np.asarray(src_t, dtype=np.float64)
    src_v = np.asarray(src_v, dtype=np.float64)
    if src_t.shape != src_v.shape:
        raise ValueError(
            f"time sensor and value arrays differ in length: {src_t.shape} vs {src_v.shape}"
        )
    n_total = src_v.size
    n_value_present = int(np.sum(np.isfinite(src_v)))

    # A sample is usable only if BOTH the value and its timestamp are good.
    usable = np.isfinite(src_v) & time_validity(src_t, time_lo, time_hi)
    n_bad_time = int(n_value_present - np.sum(usable))

    t = src_t[usable]
    v = src_v[usable]

    n_out_of_range = 0
    if valid_min is not None or valid_max is not None:
        lo = -np.inf if valid_min is None else float(valid_min)
        hi = np.inf if valid_max is None else float(valid_max)
        in_range = (v >= lo) & (v <= hi)
        n_out_of_range = int(np.sum(~in_range))
        t, v = t[in_range], v[in_range]

    order = np.argsort(t, kind="stable")
    t, v = t[order], v[order]
    n_before_dedupe = t.size
    t, v = dedupe_samples(t, v, dedupe)
    n_duplicate = int(n_before_dedupe - t.size)

    out = _interp(t, v, dst_t, method, extrapolate)
    n_blanked = 0
    if max_gap is not None:
        out, n_blanked = _apply_max_gap(t, dst_t, out, float(max_gap))

    stats = {
        "n_total": n_total,
        "n_value_present": n_value_present,
        "n_bad_time": n_bad_time,
        "n_out_of_range": n_out_of_range,
        "n_duplicate": n_duplicate,
        "n_source": int(t.size),
        "n_gap_blanked": n_blanked,
        "n_finite_out": int(np.sum(np.isfinite(out))),
    }
    return out, stats


def _resolve_paths(files_cfg: dict, config_dir: Path | None) -> list[Path]:
    """Expand ``files.root`` + ``files.patterns`` into a sorted file list."""
    root = Path(expand_config_dir(str(files_cfg.get("root", ".")), _cd(config_dir)))
    patterns = files_cfg.get("patterns") or ["*.[de]bd", "*.[de]cd"]
    if isinstance(patterns, str):
        patterns = [patterns]
    found: list[Path] = []
    seen: set[Path] = set()
    for pat in patterns:
        for p in sorted(root.glob(str(pat))):
            rp = p.resolve()
            if p.is_file() and rp not in seen:
                seen.add(rp)
                found.append(p)
    if not found:
        raise FileNotFoundError(
            f"No Dinkum files matched {patterns} under {root}. Check files.root "
            f"and files.patterns (compressed Slocum files end .dcd/.ecd, not .dbd/.ebd)."
        )
    return found


def build_hotel(
    config: dict,
    *,
    config_dir: Path | None = None,
    output: str | Path | None = None,
    now: float | None = None,
) -> Path:
    """Build a hotel NetCDF from Dinkum files per *config*.

    Parameters
    ----------
    config : dict
        Loaded YAML (see :mod:`odas_tpw.dinkum.config`).
    config_dir : Path, optional
        Directory the ``<CONFIG_DIR>`` token resolves to.
    output : path, optional
        Override ``files.output``.
    now : float, optional
        Epoch seconds used for the *now + 365 days* fallback ceiling. Pass it
        to make a run reproducible.

    Returns
    -------
    Path
        The written hotel file.
    """
    files_cfg = merge_config("files", config.get("files"))
    time_cfg = merge_config("time", config.get("time"))
    proj_cfg = merge_config("projection", config.get("projection"))
    nc_cfg = merge_config("netcdf", config.get("netcdf"))

    time_base = str(time_cfg.get("base"))
    sensors = normalize_sensors(config.get("sensors"), time_base)
    lo, hi = resolve_time_bounds(time_cfg.get("min_value"), time_cfg.get("max_value"), now=now)
    dedupe = str(time_cfg.get("dedupe", "mean"))
    default_method = str(proj_cfg.get("method", "linear"))
    extrapolate = bool(proj_cfg.get("extrapolate", False))
    default_gap = proj_cfg.get("max_gap")

    paths = _resolve_paths(files_cfg, config_dir)
    wanted = required_sensor_names(sensors, time_base)
    ds = load_dinkum(
        paths,
        backend=str(files_cfg.get("reader", "auto")),
        cache=expand_config_dir(files_cfg.get("cache"), _cd(config_dir)),
        sensors=wanted,
        skip_first_record=bool(files_cfg.get("skip_first_record", True)),
        repair=bool(files_cfg.get("repair", False)),
    )

    absent = [n for n in wanted if n not in ds.data_vars]
    if absent:
        raise KeyError(
            f"Sensor(s) not present in the input files: {absent}. "
            f"Run `dinkum-hotel sensors <files>` to list what they carry "
            f"(the files read here have {len(ds.data_vars)} sensors)."
        )

    def _col(name: str) -> np.ndarray:
        return np.asarray(ds[name].values, dtype=np.float64)

    times, base_stats = sanitize_time(_col(time_base), lo, hi, dedupe)
    # Fewer than two times is not a usable hotel file: perturb's loader skips
    # any channel whose time vector has < 2 samples (hotel.interpolate_hotel),
    # so a 1-sample file would merge as nothing at all. Fail here, loudly,
    # rather than shipping a file that silently contributes no channels.
    if times.size < 2:
        raise ValueError(
            f"Time base {time_base!r} yielded {times.size} valid sample(s) in "
            f"[{lo:g}, {hi:g}] epoch seconds — at least 2 are needed. "
            f"Of {base_stats['n_total']} records: {base_stats['n_nan']} non-finite, "
            f"{base_stats['n_out_of_range']} out of range, "
            f"{base_stats['n_duplicate']} duplicate. "
            f"Check time.base, time.min_value and time.max_value."
        )
    logger.info(
        "time base %s: %d records -> %d times (%d non-finite, %d out of range, %d duplicate)",
        time_base,
        base_stats["n_total"],
        base_stats["n_kept"],
        base_stats["n_nan"],
        base_stats["n_out_of_range"],
        base_stats["n_duplicate"],
    )

    data_vars: dict[str, xr.DataArray] = {}
    provenance: list[str] = []
    for src, opts in sensors.items():
        out_name = str(opts["name"])
        time_sensor = str(opts["time_sensor"])
        method = str(opts.get("method") or default_method)
        # normalize_sensors drops a per-sensor `max_gap: null`, so .get's
        # default is the global; an explicit per-sensor value wins.
        gap = opts.get("max_gap", default_gap)
        # normalize_sensors only sets a per-sensor dedupe for an explicit
        # method; a sensor inheriting a step-like GLOBAL method gets "last"
        # here for the same reason (a mean of two states is neither state).
        sensor_dedupe = str(
            opts.get("dedupe")
            or ("last" if method in ("previous", "next", "nearest", "zero") else dedupe)
        )
        values, stats = project_sensor(
            _col(time_sensor),
            _col(src),
            times,
            method=method,
            dedupe=sensor_dedupe,
            extrapolate=extrapolate,
            max_gap=None if gap is None else float(gap),
            valid_min=opts.get("valid_min"),
            valid_max=opts.get("valid_max"),
            time_lo=lo,
            time_hi=hi,
        )
        scale = float(opts.get("scale", 1.0))
        offset = float(opts.get("offset", 0.0))
        if scale != 1.0 or offset != 0.0:
            values = values * scale + offset

        if stats["n_source"] == 0:
            logger.warning(
                "%s: no usable samples (%d records, %d with a value, %d with a bad "
                "timestamp, %d out of range) — writing all-NaN",
                src,
                stats["n_total"],
                stats["n_value_present"],
                stats["n_bad_time"],
                stats["n_out_of_range"],
            )
        else:
            logger.info(
                "%s -> %s: %d source samples on %s, %d/%d finite after projection (%s)",
                src,
                out_name,
                stats["n_source"],
                time_sensor,
                stats["n_finite_out"],
                times.size,
                method,
            )

        attrs = {
            "units": str(opts.get("units") or ds[src].attrs.get("units", "") or ""),
            "dinkum_sensor": src,
            "dinkum_time_sensor": time_sensor,
            "projection_method": method,
            "dedupe": sensor_dedupe,
            "source_samples": stats["n_source"],
        }
        if opts.get("long_name"):
            attrs["long_name"] = str(opts["long_name"])
        if scale != 1.0 or offset != 0.0:
            attrs["comment"] = (
                f"value = {src} * {scale:g} + {offset:g} "
                f"(units converted from {ds[src].attrs.get('units', 'raw') or 'raw'})"
            )
        for key in ("valid_min", "valid_max"):
            if opts.get(key) is not None:
                attrs[f"qc_{key}"] = float(opts[key])
        if stats["n_gap_blanked"]:
            attrs["gap_blanked_samples"] = stats["n_gap_blanked"]

        data_vars[out_name] = xr.DataArray(values, dims=("time",), attrs=attrs)
        provenance.append(
            f"{src}->{out_name} on {time_sensor}: {stats['n_source']} samples, "
            f"{stats['n_bad_time']} bad-time, {stats['n_out_of_range']} out-of-range, "
            f"{stats['n_duplicate']} duplicate, {stats['n_gap_blanked']} gap-blanked"
        )

    # The time coordinate keeps the BASE SENSOR'S NAME, so the perturb side can
    # say `time_column: "sci_ctd41cp_timestamp"` and mean exactly what it says.
    out_ds = xr.Dataset(data_vars, coords={"time": times})
    out_ds = out_ds.rename({"time": time_base})
    out_ds[time_base].attrs.update(
        {
            "units": "seconds since 1970-01-01T00:00:00Z",
            "standard_name": "time",
            "long_name": f"{time_base} (Dinkum time basis)",
            "axis": "T",
            "calendar": "standard",
        }
    )

    out_ds.attrs.update(
        {
            "Conventions": "CF-1.13, ACDD-1.3",
            "featureType": "trajectory",
            "dinkum_time_base": time_base,
            "dinkum_time_min": lo,
            "dinkum_time_max": hi,
            "dinkum_time_dedupe": dedupe,
            "dinkum_projection_method": default_method,
            "dinkum_extrapolate": str(extrapolate),
            "dinkum_source_file_count": len(paths),
            "dinkum_source_files": ", ".join(p.name for p in paths[:50])
            + (" ..." if len(paths) > 50 else ""),
            "dinkum_reader": ds.attrs.get("dinkum_reader", "unknown"),
            "dinkum_time_base_records": base_stats["n_total"],
            "dinkum_time_base_rejected_nonfinite": base_stats["n_nan"],
            "dinkum_time_base_rejected_range": base_stats["n_out_of_range"],
            "dinkum_time_base_duplicates": base_stats["n_duplicate"],
            "dinkum_provenance": "; ".join(provenance),
            "date_created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "time_coverage_start": _iso(times[0]),
            "time_coverage_end": _iso(times[-1]),
        }
    )
    for key, val in nc_cfg.items():
        if val is not None:
            out_ds.attrs[key] = val
    if default_gap is not None:
        out_ds.attrs["dinkum_max_gap_s"] = float(default_gap)

    out_path = Path(expand_config_dir(str(output or files_cfg.get("output")), _cd(config_dir)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # CF-1.13 §2.5.1 forbids _FillValue on coordinates; xarray adds one to
    # float coords by default. Mirrors perturb's ctd/combo writers.
    encoding = {time_base: {"_FillValue": None}}
    out_ds.to_netcdf(out_path, encoding=encoding)
    logger.info("wrote %s: %d times x %d channels", out_path, times.size, len(data_vars))
    return out_path


def _cd(config_dir: Path | None) -> str | None:
    """expand_config_dir takes the config directory as a plain string."""
    return None if config_dir is None else str(config_dir)


def _iso(epoch: float) -> str:
    return dt.datetime.fromtimestamp(float(epoch), dt.UTC).isoformat(timespec="seconds")
