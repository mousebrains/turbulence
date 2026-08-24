# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""Fetching from an ERDDAP server, with a cache.

stdlib ``urllib`` only -- no ``erddapy``, no ``requests``.  The fetch itself is
six lines; what this module adds is the four things ``urlopen`` does not give
you and whose absence corrupts a result quietly (docs/erddap_access_DESIGN.md
section 8):

* **A timeout.** ``urlopen`` defaults to none, so a hung server hangs the build
  forever.
* **Bounded retry**, on connection errors, 429 and 5xx *only*.  A 400 is a
  malformed request -- retrying it just asks the same wrong question again.
* **Validation before caching.** The body is checked for NetCDF magic bytes and
  actually opened before it is allowed into the cache, because ``.nc`` is
  served chunked with no ``Content-Length``: a TCP cut mid-body is invisible by
  length (F3).
* **An empty time window is not an error** (F6b).  ERDDAP answers ``404 Your
  query produced no matching results`` -- the same status as a wrong dataset
  ID -- and a gap in a deployment is data, not a bug.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "EmptyWindow",
    "ErddapError",
    "cache_path",
    "das_fingerprint",
    "fetch_bytes",
    "fetch_to_file",
    "normalize_das",
    "probe_das",
]

# CDF\x01 / CDF\x02 = classic NetCDF; \x89HDF = NetCDF-4 (HDF5). A proxy login
# page or an ERDDAP error body matches neither.
_NC_MAGIC = (b"CDF\x01", b"CDF\x02", b"\x89HDF")

# ERDDAP's "no rows matched" text. It arrives as a 404, which is also what a
# wrong dataset ID gives, so the status alone cannot tell them apart.
_EMPTY_RE = re.compile(r"produced no matching results|nRows\s*=\s*0", re.I)

_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class ErddapError(RuntimeError):
    """A request failed in a way the caller cannot paper over."""


class EmptyWindow(Exception):
    """The request was well-formed and matched no rows (F6b). Not an error."""


def _erddap_message(body: bytes) -> str:
    """Pull the human-readable line out of an ERDDAP error body."""
    text = body.decode("utf-8", "replace")
    m = re.search(r"Error\s*\{(.*?)\}", text, re.S)
    blob = m.group(1) if m else text
    for line in blob.splitlines():
        line = line.strip()
        if line and not line.startswith(("code=", "message=")):
            return line[:400]
        if line.startswith("message="):
            return line[len("message=") :].strip().strip('"')[:400]
    return text.strip()[:400] or "(no message in body)"


def fetch_bytes(
    url: str,
    *,
    timeout: float = 120.0,
    retries: int = 3,
    backoff: float = 2.0,
    _sleep=time.sleep,
) -> bytes:
    """GET *url*, returning the body.

    Raises :class:`EmptyWindow` for ERDDAP's "no matching results", and
    :class:`ErddapError` for anything else, with the server's own message
    rather than a bare status code.
    """
    last: Exception | None = None
    for attempt in range(int(retries) + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "odas-tpw/erddap-hotel"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return bytes(resp.read())
        except urllib.error.HTTPError as exc:
            body = b""
            with contextlib.suppress(Exception):
                body = exc.read()  # the body is a courtesy, not required
            msg = _erddap_message(body)
            if exc.code == 404 and _EMPTY_RE.search(body.decode("utf-8", "replace")):
                raise EmptyWindow(msg) from None
            if exc.code not in _RETRY_STATUS:
                # 400/404 and friends: a request bug. Asking again changes
                # nothing, and the retry only delays the useful error.
                raise ErddapError(f"HTTP {exc.code} from ERDDAP: {msg}\n  {url}") from None
            last = ErddapError(f"HTTP {exc.code} from ERDDAP: {msg}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = ErddapError(f"{type(exc).__name__}: {exc}")
        if attempt < int(retries):
            delay = backoff * (2**attempt)
            logger.warning("%s -- retry %d/%d in %.0fs", last, attempt + 1, retries, delay)
            _sleep(delay)
    raise ErddapError(f"gave up after {retries} retries: {last}\n  {url}")


def _looks_like_netcdf(body: bytes) -> bool:
    return any(body.startswith(m) for m in _NC_MAGIC)


def fetch_to_file(
    url: str,
    dest: Path,
    *,
    timeout: float = 120.0,
    retries: int = 3,
    expect_rows: int | None = None,
) -> int:
    """Fetch a NetCDF to *dest*, validating it before it is allowed to persist.

    Written to a sibling temp file and renamed only once it has passed, so a
    failed fetch can never leave a half-file in the cache for the next run to
    trust.  Returns the row count.
    """
    import xarray as xr

    body = fetch_bytes(url, timeout=timeout, retries=retries)
    if not _looks_like_netcdf(body):
        # F1: every probed ERDDAP error was non-2xx, but a proxy or captive
        # portal answers 200 with HTML, and that must not reach the cache.
        head = body[:200].decode("utf-8", "replace").strip().replace("\n", " ")
        raise ErddapError(
            f"response is not NetCDF (magic bytes {body[:4]!r}). "
            f"First 200 bytes: {head!r}\n  {url}"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(body)
    try:
        with xr.open_dataset(tmp, decode_times=False, decode_cf=False) as ds:
            rows = int(ds.sizes.get("row", ds.sizes.get("obs", 0)))
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        # F3: a body cut mid-header still starts with CDF\x01.
        raise ErddapError(f"downloaded NetCDF will not open ({exc})\n  {url}") from None
    if expect_rows is not None and rows != expect_rows:
        tmp.unlink(missing_ok=True)
        raise ErddapError(
            f"row count mismatch: the .nc holds {rows} rows but .csv reported "
            f"{expect_rows}. The response was probably truncated (it is served "
            f"chunked, with no Content-Length).\n  {url}"
        )
    tmp.replace(dest)
    return rows


def count_rows(url: str, *, timeout: float = 120.0, retries: int = 3) -> int:
    """Row count from a one-column ``.csv``: lines minus the two header rows.

    ERDDAP's CSV carries a name row and a units row.  Returns 0 for an empty
    window rather than raising, so a caller can use it as a cheap pre-check.
    """
    try:
        body = fetch_bytes(url, timeout=timeout, retries=retries)
    except EmptyWindow:
        return 0
    lines = [ln for ln in body.decode("utf-8", "replace").splitlines() if ln.strip()]
    return max(0, len(lines) - 2)


def probe_das(base_url: str, dataset_id: str, *, timeout: float = 120.0) -> str:
    """Fetch the ``.das``. Cheap, and the basis of both cache keying and verify."""
    from odas_tpw.erddap.query import das_url

    return fetch_bytes(das_url(base_url, dataset_id), timeout=timeout).decode("utf-8", "replace")


# ERDDAP appends a line to the served `.das`'s own `history` attribute
# recording WHEN YOU ASKED FOR IT and what you asked for:
#
#     2026-08-24T02:09:43Z (local files)
#     2026-08-24T02:09:43Z http://.../ru33-....das
#
# Those change on every single request, so hashing the `.das` verbatim gives a
# different digest each time -- which would defeat the cache key entirely (it
# could never hit) and make `verify` report CHANGED on an untouched dataset.
# Measured on Rutgers' server: two `.das` fetches one second apart differed in
# exactly these two lines and nothing else.
#
# The upstream provenance lines, which we DO want in the digest, are
# distinguishable: they read `<timestamp>Z: /path/to/script`, with a colon
# after the Z. ERDDAP's own request log has no colon.
_VOLATILE_DAS_LINE = re.compile(
    r"^\s*\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\s+(\(local files\)|https?://)"
)


def normalize_das(das: str) -> str:
    """Strip the per-request lines ERDDAP injects, so the digest is stable."""
    return "\n".join(ln for ln in das.splitlines() if not _VOLATILE_DAS_LINE.match(ln))


def das_fingerprint(das: str) -> tuple[str, str | None]:
    """``(sha256, date_modified)`` -- how a server-side revision is noticed.

    Both go into the cache key, so a reprocessed dataset misses the cache by
    construction rather than by anyone remembering to clear it.

    The digest is taken over :func:`normalize_das`, not the raw body: see the
    comment above for why hashing what the server sent would make every run a
    cache miss.
    """
    digest = hashlib.sha256(normalize_das(das).encode("utf-8")).hexdigest()
    m = re.search(r'date_modified\s+"([^"]+)"', das)
    return digest, (m.group(1) if m else None)


def cache_path(cache_dir: Path, dataset_id: str, url: str, das_sha: str) -> Path:
    """Cache filename for one chunk.

    Keyed on the full request URL *and* the ``.das`` digest, so both "different
    query" and "same query, revised dataset" are cache misses.
    """
    key = hashlib.sha256(f"{das_sha}\n{url}".encode()).hexdigest()[:16]
    return Path(cache_dir) / dataset_id / f"chunk_{key}.nc"
