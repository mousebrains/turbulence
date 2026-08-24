# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""ERDDAP -> hotel file.

The fetch is the small half.  What matters here is that the QC and projection
are **not reimplemented**: ``build_hotel`` in :mod:`odas_tpw.dinkum.build` does
the sanitising, deduplication, projection onto one clock, gap-blanking and
provenance, and this module hands it a dataset instead of Dinkum files
(docs/erddap_access_DESIGN.md section 10.3).  The tree keeps two sanitisers,
not three.

What is genuinely ERDDAP's own, and lives here:

* **`_FillValue` masking.** Served as ``9.96921e+36`` and *not* masked on read
  -- it arrives as an ordinary finite float that any bound catches, but only if
  something applies one.
* **Chunked fetch with a cache**, and the three refresh modes.
* **Provenance**: ERDDAP already records the request URL and fetch time in its
  own ``history``, so that is preserved rather than duplicated, and the QC
  summary is appended to it.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Any

import numpy as np

from odas_tpw.dinkum.build import build_hotel
from odas_tpw.erddap import fetch as _fetch
from odas_tpw.erddap.config import merge_config, to_builder_config, validate
from odas_tpw.erddap.query import chunk_windows, count_url, iso, tabledap_url
from odas_tpw.perturb.config import expand_config_dir

logger = logging.getLogger(__name__)

__all__ = ["build", "fetch_chunks", "mask_fill_values", "plan_requests", "verify"]

# ERDDAP's conventional no-data marker. Present as an attribute on the served
# variables, but NOT applied on read: it comes through as a finite float.
DEFAULT_FILL = 9.96921e36
_FILL_ATOL = 1.0e30


def _cd(config_dir: Path | None) -> str | None:
    return None if config_dir is None else str(config_dir)


def plan_requests(config: dict, *, now: float | None = None) -> list[dict[str, Any]]:
    """Config -> the list of requests a build would issue.

    Returns one dict per chunk with ``url``, ``count_url`` and the window, so
    ``fetch --dry-run`` can print exactly what would be asked for without any
    I/O.  This is the piece most likely to be silently wrong, and it is
    testable by string comparison.
    """
    server = merge_config("server", config.get("server"))
    f = merge_config("fetch", config.get("fetch"))

    lo = _epoch(f.get("time_min"), "fetch.time_min")
    hi = _epoch(f.get("time_max"), "fetch.time_max")
    if lo is None:
        raise ValueError(
            "fetch.time_min: required. Without it the whole dataset is "
            "requested, which for a multi-month deployment is a very large "
            "download that will probably time out (design F5). Use "
            "`erddap-hotel info` to see the dataset's actual coverage."
        )
    if hi is None:
        hi = float(now if now is not None else dt.datetime.now(dt.UTC).timestamp())

    windows = chunk_windows(lo, hi, float(f.get("chunk_days") or 7.0))
    variables = list(f.get("variables") or [])
    constraints = list(f.get("constraints") or [])
    tvar = str(f.get("time_variable", "time"))
    out = []
    for i, win in enumerate(windows):
        last = i == len(windows) - 1
        out.append(
            {
                "window": win,
                "start": iso(win[0]),
                "end": iso(win[1]),
                "url": tabledap_url(
                    server["base_url"],
                    server["dataset_id"],
                    variables,
                    fmt=str(f.get("format", "nc")),
                    time_variable=tvar,
                    window=win,
                    last_window=last,
                    constraints=constraints,
                ),
                "count_url": count_url(
                    server["base_url"],
                    server["dataset_id"],
                    time_variable=tvar,
                    window=win,
                    last_window=last,
                    constraints=constraints,
                ),
            }
        )
    return out


def _epoch(value: Any, label: str) -> float | None:
    """Accept epoch seconds or an ISO-8601 date, as the rest of the tree does."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        raise ValueError(
            f"{label}={value!r}: not epoch seconds or an ISO-8601 date "
            "(e.g. 2021-10-01T00:00:00Z)"
        ) from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.timestamp()


def mask_fill_values(ds, variables: list[str], drop_zero: list[str] | None = None):
    """NaN out ``_FillValue`` (and, if asked, exact zeros) in place.

    The fill value is read from each variable's own attribute where present,
    falling back to ERDDAP's conventional 9.96921e+36.  It is compared with a
    tolerance because the served value is a float32 widened to float64, so an
    exact ``==`` misses it.

    *drop_zero* is the per-variable "0.0 means not sampled" rule.  It defaults
    to empty and should stay that way unless a dataset is shown to need it:
    measured over 5.4M rows every such row already has an unusable timestamp,
    and applied blanket the rule deletes real 0 degC polar water.
    """
    stats: dict[str, dict[str, int]] = {}
    zero_set = set(drop_zero or [])
    for name in variables:
        if name not in ds:
            continue
        values = np.asarray(ds[name].values, dtype=np.float64)
        declared = ds[name].attrs.get("_FillValue", ds[name].attrs.get("missing_value"))
        fill = float(np.asarray(declared).ravel()[0]) if declared is not None else DEFAULT_FILL
        # Relative tolerance, not equality: the value is served as float32 and
        # widened to float64, so `== 9.96921e+36` misses it. The magnitude
        # guard is a second net for an undeclared sentinel of a different
        # value -- nothing physical in a CTD feed is 1e30.
        bad = ~np.isfinite(values)
        if np.isfinite(fill):
            bad |= np.isclose(values, fill, rtol=1e-5, atol=0.0)
        bad |= np.abs(values) >= _FILL_ATOL
        n_fill = int(np.count_nonzero(bad & np.isfinite(values)))
        n_zero = 0
        if name in zero_set:
            is_zero = values == 0.0
            n_zero = int(np.count_nonzero(is_zero & ~bad))
            bad = bad | is_zero
        values[bad] = np.nan
        ds[name] = (ds[name].dims, values, dict(ds[name].attrs))
        stats[name] = {"fill_masked": n_fill, "zero_masked": n_zero}
        if n_fill or n_zero:
            logger.info(
                "%s: masked %d _FillValue%s", name, n_fill,
                f" and {n_zero} exact zeros" if n_zero else "",
            )
    return stats


def fetch_chunks(
    config: dict,
    *,
    config_dir: Path | None = None,
    now: float | None = None,
    offline: bool = False,
) -> tuple[list[Path], dict[str, Any]]:
    """Populate the cache and return the chunk files, newest-consistent order.

    Honours ``fetch.refresh``:

    ``never``
        use whatever is cached; fetch only what is missing.
    ``incremental`` (default)
        as ``never``, but also refetch the last ``overlap_chunks`` windows,
        because a ``-raw-delayed`` dataset can revise recent rows as well as
        extend them.
    ``always``
        refetch every window.
    """
    server = merge_config("server", config.get("server"))
    f = merge_config("fetch", config.get("fetch"))
    timeout = float(server["timeout_s"])
    retries = int(server["retries"])
    cache_dir = Path(expand_config_dir(str(server["cache"]), _cd(config_dir)))

    if offline:
        # The cache key includes the .das digest, and offline cannot compute
        # one. Reuse the digest the cache was last populated under; inventing
        # a placeholder would look under a key no online run ever wrote, so
        # every lookup would miss while the data sat right there.
        modified = None
        recalled = _fetch.recall_das_sha(cache_dir, server["dataset_id"])
        if recalled is None:
            raise _fetch.ErddapError(
                f"--offline, but {cache_dir} has never been populated for "
                f"{server['dataset_id']}. Run `erddap-hotel fetch` online once first."
            )
        das_sha = recalled
        logger.info("offline: using the cached dataset revision das=%s", das_sha[:12])
    else:
        das = _fetch.probe_das(server["base_url"], server["dataset_id"], timeout=timeout)
        das_sha, modified = _fetch.das_fingerprint(das)
        logger.info("dataset date_modified=%s das=%s", modified or "?", das_sha[:12])
        _fetch.remember_das_sha(cache_dir, server["dataset_id"], das_sha)

    plan = plan_requests(config, now=now)
    refresh = str(f.get("refresh", "incremental"))
    overlap = int(f.get("overlap_chunks", 1))
    n = len(plan)
    refetch_from = n if refresh == "never" else (0 if refresh == "always" else max(0, n - overlap))

    paths: list[Path] = []
    n_fetched = n_cached = n_empty = 0
    for i, req in enumerate(plan):
        dest = _fetch.cache_path(cache_dir, server["dataset_id"], req["url"], das_sha)
        if dest.exists() and i < refetch_from:
            paths.append(dest)
            n_cached += 1
            continue
        if offline:
            if dest.exists():
                paths.append(dest)
                n_cached += 1
                continue
            raise _fetch.ErddapError(
                f"--offline but chunk {req['start']}..{req['end']} is not cached "
                f"({dest}). Run `erddap-hotel fetch` first."
            )
        logger.info("fetching %s .. %s (%d/%d)", req["start"], req["end"], i + 1, n)
        try:
            expect = _fetch.count_rows(req["count_url"], timeout=timeout, retries=retries)
            if expect == 0:
                # F6b: a gap in a deployment is data, not a bug.
                logger.info("  no rows in this window -- skipping")
                n_empty += 1
                continue
            _fetch.fetch_to_file(
                req["url"], dest, timeout=timeout, retries=retries, expect_rows=expect
            )
        except _fetch.EmptyWindow:
            logger.info("  no rows in this window -- skipping")
            n_empty += 1
            continue
        paths.append(dest)
        n_fetched += 1

    if not paths:
        raise _fetch.ErddapError(
            f"no data in {n} window(s) between {plan[0]['start']} and {plan[-1]['end']}. "
            "Check fetch.time_min / time_max against the dataset's real coverage "
            "(`erddap-hotel info`)."
        )
    meta = {
        "das_sha256": das_sha,
        "date_modified": modified,
        "chunks_fetched": n_fetched,
        "chunks_cached": n_cached,
        "chunks_empty": n_empty,
        "requests": [r["url"] for r in plan],
    }
    logger.info(
        "%d chunk(s): %d fetched, %d cached, %d empty", n, n_fetched, n_cached, n_empty
    )
    return paths, meta


def _concat(paths: list[Path]):
    """Open the cached chunks as one table.

    ``decode_cf=False`` deliberately: we do our own epoch handling and want the
    raw numbers, and the CTD timestamp is a plain variable rather than the
    ``time`` axis, so letting xarray decode one and not the other invites F4.
    """
    import xarray as xr

    parts = []
    for p in paths:
        parts.append(xr.open_dataset(p, decode_times=False, decode_cf=False).load())

    # Each chunk's `history` ends with ITS OWN request URL and fetch time, so
    # the histories differ and `drop_conflicts` throws all of them away --
    # silently losing the upstream processing chain the design requires us to
    # preserve (section 5.3). Collect them first, deduplicated, keeping order.
    seen: dict[str, None] = {}
    for part in parts:
        for line in str(part.attrs.get("history", "")).splitlines():
            if line.strip():
                seen.setdefault(line.rstrip(), None)

    if len(parts) == 1:
        out = parts[0]
    else:
        dim = "row" if "row" in parts[0].dims else next(iter(parts[0].dims))
        out = xr.concat(parts, dim=dim, join="outer", combine_attrs="drop_conflicts")
    if seen:
        out.attrs["history"] = "\n".join(seen)
    return out


def build(
    config: dict,
    *,
    config_dir: Path | None = None,
    output: str | Path | None = None,
    now: float | None = None,
    offline: bool = False,
) -> Path:
    """Fetch (or reuse the cache), sanitise, and write the hotel file."""
    validate(config)
    server = merge_config("server", config.get("server"))
    f = merge_config("fetch", config.get("fetch"))
    qc = merge_config("qc", config.get("qc"))
    out_cfg = merge_config("output", config.get("output"))

    paths, meta = fetch_chunks(config, config_dir=config_dir, now=now, offline=offline)
    ds = _concat(paths)

    fill_stats = mask_fill_values(
        ds, list(f.get("variables") or []), list(qc.get("drop_zero_as_fill") or [])
    )

    upstream_history = str(ds.attrs.get("history", "")).strip()
    stamp = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    masked = "; ".join(
        f"{k}: {v['fill_masked']} fill"
        + (f", {v['zero_masked']} zero" if v["zero_masked"] else "")
        for k, v in fill_stats.items()
        if v["fill_masked"] or v["zero_masked"]
    )
    history = "\n".join(
        x
        for x in (
            upstream_history,
            f"{stamp}: erddap-hotel build: {len(paths)} chunk(s) "
            f"({meta['chunks_fetched']} fetched, {meta['chunks_cached']} cached, "
            f"{meta['chunks_empty']} empty); masked {masked or 'nothing'}",
        )
        if x
    )

    source_attrs = {
        "source": f"ERDDAP {server['protocol']} {server['base_url']}/{server['dataset_id']}",
        "history": history,
        "erddap_base_url": str(server["base_url"]),
        "erddap_dataset_id": str(server["dataset_id"]),
        "erddap_das_checksum": meta["das_sha256"],
        "erddap_refresh": str(f.get("refresh")),
        "erddap_chunk_days": float(f.get("chunk_days") or 7.0),
        "erddap_chunks": len(paths),
        "erddap_constraints": ", ".join(map(str, f.get("constraints") or [])) or "none",
        "erddap_request_urls": "\n".join(meta["requests"][:20])
        + ("\n..." if len(meta["requests"]) > 20 else ""),
    }
    if meta["date_modified"]:
        source_attrs["erddap_date_modified"] = meta["date_modified"]

    return build_hotel(
        to_builder_config(config),
        config_dir=config_dir,
        output=output or out_cfg.get("file"),
        now=now,
        dataset=ds,
        source_attrs=source_attrs,
    )


def verify(config: dict, hotel_file: Path | None = None) -> dict[str, Any]:
    """Has the dataset changed since the hotel file was built?

    Re-fetches only the ``.das`` -- a few tens of kB -- and compares its digest
    with the one recorded in the output.  This is the cheap answer to "is my
    result still reproducible", and it needs no data transfer.
    """
    import xarray as xr

    server = merge_config("server", config.get("server"))
    out_cfg = merge_config("output", config.get("output"))
    path = Path(hotel_file or out_cfg["file"])

    das = _fetch.probe_das(
        server["base_url"], server["dataset_id"], timeout=float(server["timeout_s"])
    )
    live_sha, live_modified = _fetch.das_fingerprint(das)

    if not path.exists():
        return {
            "hotel_file": str(path),
            "exists": False,
            "live_das_sha256": live_sha,
            "live_date_modified": live_modified,
            "changed": None,
        }
    with xr.open_dataset(path, decode_times=False) as ds:
        built_sha = ds.attrs.get("erddap_das_checksum")
        built_modified = ds.attrs.get("erddap_date_modified")
    return {
        "hotel_file": str(path),
        "exists": True,
        "built_das_sha256": built_sha,
        "built_date_modified": built_modified,
        "live_das_sha256": live_sha,
        "live_date_modified": live_modified,
        "changed": bool(built_sha) and built_sha != live_sha,
    }
