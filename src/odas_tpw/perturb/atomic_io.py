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
    tmp = tmp_sibling(out_path)
    try:
        ds.to_netcdf(tmp, **to_netcdf_kwargs)
        os.replace(tmp, out_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
