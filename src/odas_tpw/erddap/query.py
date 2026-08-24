# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""tabledap URL construction.

Pure string building, deliberately kept apart from the fetcher so it can be
tested by comparison without any I/O (``erddap-hotel fetch --dry-run`` prints
exactly what this returns).

The encoding rule, from the Rutgers notebook and confirmed against the live
server: the variable list and every constraint go through
``quote(..., safe="")``, so ``>=`` becomes ``%3E%3D``, ``,`` becomes ``%2C``
and the ``:`` in an ISO timestamp becomes ``%3A``.  Only the ``?`` introducing
the query and the ``&`` joining clauses stay literal.  ERDDAP will accept some
unencoded forms, but not all, and the failures are 400s that read like a
missing variable rather than a quoting bug.
"""

from __future__ import annotations

import datetime as dt
from urllib.parse import quote

__all__ = ["chunk_windows", "count_url", "das_url", "dds_url", "iso", "tabledap_url"]

_SEC_PER_DAY = 86400.0


def iso(epoch: float) -> str:
    """Epoch seconds -> the ``1970-01-01T00:00:00Z`` form ERDDAP expects."""
    return (
        dt.datetime.fromtimestamp(float(epoch), dt.UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def chunk_windows(
    time_min: float, time_max: float, chunk_days: float
) -> list[tuple[float, float]]:
    """Split ``[time_min, time_max]`` into half-open windows of *chunk_days*.

    Windows are ``[start, end)`` except the last, which is closed, so no row
    is fetched twice and none falls between two chunks.  The caller builds the
    constraint as ``time>=start`` and ``time<end`` (``<=`` on the final one).
    """
    if not (time_max > time_min):
        raise ValueError(f"time_max ({iso(time_max)}) must be after time_min ({iso(time_min)})")
    if not (chunk_days > 0):
        raise ValueError(f"chunk_days must be > 0, got {chunk_days!r}")
    step = float(chunk_days) * _SEC_PER_DAY
    out: list[tuple[float, float]] = []
    start = float(time_min)
    while start < time_max:
        out.append((start, min(start + step, float(time_max))))
        start += step
    return out


def tabledap_url(
    base_url: str,
    dataset_id: str,
    variables: list[str],
    *,
    fmt: str = "nc",
    time_variable: str = "time",
    window: tuple[float, float] | None = None,
    last_window: bool = False,
    constraints: list[str] | None = None,
) -> str:
    """Build one tabledap request URL.

    *window* is a ``(start, end)`` pair from :func:`chunk_windows`; the upper
    bound is exclusive unless *last_window*, which keeps the final row.
    """
    if not variables:
        raise ValueError("variables: at least one is required (ERDDAP 400s on an empty list)")
    root = base_url.rstrip("/")
    if not root.endswith("/tabledap"):
        root = f"{root}/tabledap"
    clauses = [quote(",".join(variables), safe="")]
    if window is not None:
        lo, hi = window
        clauses.append(quote(f"{time_variable}>={iso(lo)}", safe=""))
        clauses.append(quote(f"{time_variable}{'<=' if last_window else '<'}{iso(hi)}", safe=""))
    for extra in constraints or []:
        clauses.append(quote(str(extra), safe=""))
    return f"{root}/{dataset_id}.{fmt}?" + "&".join(clauses)


def das_url(base_url: str, dataset_id: str) -> str:
    """Metadata probe: attributes, units, valid ranges, ``date_modified``."""
    root = base_url.rstrip("/")
    if not root.endswith("/tabledap"):
        root = f"{root}/tabledap"
    return f"{root}/{dataset_id}.das"


def dds_url(base_url: str, dataset_id: str) -> str:
    """Type schema: which variables exist, and of what type."""
    return das_url(base_url, dataset_id)[:-4] + ".dds"


def count_url(
    base_url: str,
    dataset_id: str,
    *,
    time_variable: str = "time",
    window: tuple[float, float] | None = None,
    last_window: bool = False,
    constraints: list[str] | None = None,
) -> str:
    """A one-column ``.csv`` used only to count rows.

    Neither ``.dds`` (a type schema) nor ``.das`` (attributes) carries a row
    count, and ``.ncHeader`` 500s on this dataset, so counting means asking for
    one column and counting lines.  Used to detect a truncated ``.nc``: the
    body is served chunked with no ``Content-Length``, so a TCP-level cut is
    invisible by length alone.
    """
    return tabledap_url(
        base_url,
        dataset_id,
        [time_variable],
        fmt="csv",
        time_variable=time_variable,
        window=window,
        last_window=last_window,
        constraints=constraints,
    )
