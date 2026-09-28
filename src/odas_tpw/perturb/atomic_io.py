# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""Crash-atomic NetCDF writes for perturb products.

A direct ``ds.to_netcdf(path)`` (or a low-level ``netCDF4.Dataset(path, "w")``)
interrupted mid-payload — ENOSPC, an SMB/network drop on SeaChest, Ctrl-C, or any
raised exception — leaves a READABLE but PARTIAL NetCDF at the live path. The
perturb bin/combo manifest keys on the SOURCE ``.p`` cache keys, not the product
content, so a clean retry (identical filenames -> identical manifest) skips
re-assembly and permanently publishes the truncated file with no error.

Both helpers here write to a sibling ``.{name}.{pid}.tmp`` on the SAME filesystem
and ``os.replace`` it into place only after a clean write (``os.replace`` is
atomic within a filesystem), so a partial write never becomes the live file. The
temp is unlinked on any ``BaseException``.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import xarray as xr


def tmp_sibling(out_path: Path) -> Path:
    """A sibling temp path on the SAME filesystem as *out_path* (so ``os.replace``
    into place is atomic). PID-suffixed so concurrent workers never collide."""
    return out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")


def atomic_to_netcdf(ds: xr.Dataset, out_path: Path, **to_netcdf_kwargs: Any) -> None:
    """Write *ds* to *out_path* atomically (temp file + ``os.replace``).

    ``**to_netcdf_kwargs`` are forwarded to :meth:`xarray.Dataset.to_netcdf`
    (e.g. ``encoding=``). A partial or failed write never becomes the live file.
    """
    def _write() -> None:
        tmp = tmp_sibling(out_path)
        try:
            ds.to_netcdf(tmp, **to_netcdf_kwargs)
            os.replace(tmp, out_path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    # Safe to retry WHOLE: the temp is unlinked on any failure, so each attempt
    # starts from a clean temp and ``os.replace`` only ever publishes a file
    # that was written end to end.
    retry_transient_io(_write, what=str(out_path))


# --- transient network-filesystem faults -------------------------------------

_TRANSIENT_ATTEMPTS = 3
_TRANSIENT_BACKOFF = (0.5, 2.0)  # seconds; the fault clears in well under 1 s


def _is_transient_io(exc: BaseException) -> bool:
    """True for the netCDF/HDF5 faults that clear on a plain retry.

    NARROW ON PURPOSE. ``FileNotFoundError``, ``PermissionError`` and friends
    are ``OSError`` subclasses and are NOT transient — retrying them would hide
    a real bug behind a few seconds of delay. netCDF-C reports the SMB fault as
    errno -101 ("NetCDF: HDF error"); -51 ("NetCDF: Unknown file format") is the
    same fault caught a layer down, on a read that came back short.
    """
    if isinstance(
        exc, (FileNotFoundError, NotADirectoryError, IsADirectoryError, PermissionError)
    ):
        return False
    if not isinstance(exc, OSError):
        return False
    if exc.errno in (-101, -51):
        return True
    return "NetCDF:" in str(exc) or "HDF error" in str(exc)


def retry_transient_io(fn, *args, what: str = "", **kwargs):
    """Call *fn*, retrying a TRANSIENT netCDF/HDF5 failure a bounded number of times.

    On SeaChest (macOS smbfs) roughly 1 netCDF open in 25,000 fails with
    ``[Errno -101] NetCDF: HDF error`` and succeeds on a plain retry — verified
    by re-opening the failed paths afterwards, which read cleanly. Server-side
    logs are empty and HDF5 file locking is not enforced on that mount (a second
    writer is admitted), so this is the SMB client under concurrency, not a
    locking or server fault, and no NAS-side setting addresses it.

    Retries are WARNED, never silent: a rising retry count is the signal that the
    mount is degrading, and a silent retry would hide exactly that.
    """
    import logging
    import time

    for attempt in range(_TRANSIENT_ATTEMPTS):
        try:
            return fn(*args, **kwargs)
        except BaseException as exc:
            if not _is_transient_io(exc) or attempt == _TRANSIENT_ATTEMPTS - 1:
                raise
            delay = _TRANSIENT_BACKOFF[min(attempt, len(_TRANSIENT_BACKOFF) - 1)]
            logging.getLogger(__name__).warning(
                "transient I/O on %s (attempt %d/%d): %s — retrying in %.1fs",
                what or getattr(fn, "__name__", "?"),
                attempt + 1,
                _TRANSIENT_ATTEMPTS,
                exc,
                delay,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover
