# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""ERDDAP tabledap -> perturb hotel file.

The ERDDAP twin of :mod:`odas_tpw.dinkum`: where that reads Slocum binaries off
a disk, this fetches rows from a tabledap server.  Both write the same
artifact, and neither perturb nor ``fp07-cal`` knows the difference -- they
open a local NetCDF.

That indirection is the design, not an accident.  A URL in ``hotel.file`` would
leave perturb's skip-cache with a constant fingerprint, so a re-run that had
fetched genuinely new data would look unchanged and skip the work; a file on
disk gets a real size and mtime.  See ``docs/erddap_access_DESIGN.md``.

- :mod:`~odas_tpw.erddap.query` -- tabledap URL construction and chunking.
  Pure strings, no I/O.
- :mod:`~odas_tpw.erddap.fetch` -- stdlib ``urllib`` with timeout, bounded
  retry, response validation and a content-addressed cache.
- :mod:`~odas_tpw.erddap.config` -- the YAML schema, and its translation into
  the shared builder's schema.
- :mod:`~odas_tpw.erddap.build` -- ``_FillValue`` masking, then
  :func:`odas_tpw.dinkum.build.build_hotel` for the sanitising, projection and
  provenance.  The QC is shared, deliberately: the tree has two sanitisers, not
  three.
- :mod:`~odas_tpw.erddap.cli` -- the ``erddap-hotel`` command.
"""

from odas_tpw.erddap.build import build, fetch_chunks, mask_fill_values, plan_requests, verify
from odas_tpw.erddap.fetch import EmptyWindow, ErddapError
from odas_tpw.erddap.query import chunk_windows, das_url, tabledap_url

__all__ = [
    "EmptyWindow",
    "ErddapError",
    "build",
    "chunk_windows",
    "das_url",
    "fetch_chunks",
    "mask_fill_values",
    "plan_requests",
    "tabledap_url",
    "verify",
]
