# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Corrupt record trimming for .p files.

If the last record is incomplete (fractional), copy only complete records
to a new file.

Reference: Code/trim_P_files.m
"""

import contextlib
import json
import os
import struct
import tempfile
from pathlib import Path
from typing import NamedTuple

from odas_tpw.rsi.p_file import _H, HEADER_BYTES, HEADER_WORDS, _detect_endian


class TrimResult(NamedTuple):
    """Outcome of a single :func:`trim_p_file` call.

    Attributes
    ----------
    dest : Path
        The path downstream stages should read. For ``"trimmed"`` /
        ``"skipped"`` this is the rewritten file under the trim directory;
        for ``"referenced"`` it is the *original* source path, left in place.
    action : str
        One of:

        * ``"trimmed"``    — a fractional final record was dropped; a new
          truncated file was written to the trim directory.
        * ``"referenced"`` — the file was already complete, so it is used
          in place from its original location (no copy, no link).
        * ``"skipped"``    — an up-to-date trimmed output already existed;
          no work done.
    bytes_removed : int
        Bytes dropped from the incomplete final record. 0 unless
        *action* is ``"trimmed"``.
    record_size : int
        The .p record size in bytes, read from the header. 0 when the file
        was skipped (the header is not read on the skip path).
    """

    dest: Path
    action: str
    bytes_removed: int
    record_size: int


def trim_destination(source: Path, output_dir: Path, root: Path | str | None = None) -> Path:
    """Return the trim output path for *source*.

    When *root* is supplied and *source* lives underneath it, the relative
    directory structure is preserved under *output_dir*.  This prevents files
    such as ``SN001/cast.p`` and ``SN002/cast.p`` from overwriting each other.
    """
    source = Path(source)
    output_dir = Path(output_dir)
    if root is not None:
        try:
            rel = source.resolve().relative_to(Path(root).resolve())
        except ValueError:
            rel = Path(source.name)
        return output_dir / rel
    return output_dir / source.name


# ---------------------------------------------------------------------------
# Trim decision cache (incremental re-runs / slow mounts)
# ---------------------------------------------------------------------------
# A re-run otherwise re-opens and header-reads EVERY source .p to decide it
# needs no trimming — cheap locally, but minutes over an SMB/network mount where
# each open is a round-trip. The cache records, per source, a ``size + 2 s-mtime``
# fingerprint and whether the file was complete (referenced) or trimmed, so an
# unchanged file is decided from a single ``stat`` with no header read. Keyed on
# the source's path *relative to the trim dir* (portable across machines/mounts),
# and only trusted when its physical output is still present. ``force`` / a
# changed fingerprint re-reads the header. mtime is binned to 2 s so the cache
# survives a copy to exFAT (2 s mtime granularity).


def _trim_fingerprint(source: Path) -> dict | None:
    try:
        st = source.stat()
    except OSError:
        return None
    return {"size": st.st_size, "mtime_2s": int(st.st_mtime) // 2 * 2}


def _trim_marker(cache_dir: Path, dest: Path, output_dir: Path) -> Path:
    """Marker path mirroring the source's relative layout (collision-free,
    portable). *dest* is ``trim_destination(source, output_dir, root)``."""
    try:
        rel = dest.relative_to(output_dir).as_posix()
    except ValueError:
        rel = dest.name
    return cache_dir / "trim" / f"{rel}.json"


def _trim_cache_lookup(
    cache_dir: Path, source: Path, dest: Path, output_dir: Path
) -> "TrimResult | None":
    marker = _trim_marker(cache_dir, dest, output_dir)
    try:
        data = json.loads(marker.read_text())
    except (OSError, ValueError):
        return None
    if data.get("fp") != _trim_fingerprint(source):
        return None
    if data.get("referenced"):
        # Complete file, unchanged — reference the original, no header read.
        return TrimResult(source, "referenced", 0, int(data.get("record_size", 0)))
    # Previously trimmed — reuse the existing trimmed output only if it is still
    # present AND the right size. The size check preserves the original guard
    # against a truncated/corrupted/externally-replaced trimmed output (the
    # source fingerprint matched, but the *output* could have been disturbed).
    try:
        if dest.stat().st_size == data.get("dest_size"):
            return TrimResult(dest, "skipped", 0, 0)
    except OSError:
        pass  # missing dest -> re-trim
    return None


def _trim_cache_store(
    cache_dir: Path, source: Path, dest: Path, output_dir: Path, result: "TrimResult"
) -> None:
    fp = _trim_fingerprint(source)
    if fp is None:
        return
    marker = _trim_marker(cache_dir, dest, output_dir)
    payload = {
        "fp": fp,
        "referenced": result.action == "referenced",
        "record_size": result.record_size,
    }
    if result.action != "referenced":
        # Record the trimmed output's size so a later run can verify it wasn't
        # truncated/replaced before reusing it (see _trim_cache_lookup).
        try:
            payload["dest_size"] = dest.stat().st_size
        except OSError:
            return  # can't verify the output later -> don't cache it
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        tmp = marker.with_name(f"{marker.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, marker)
    except OSError:
        pass  # caching is best-effort; never fail a trim over a marker write


def trim_p_file(
    source: Path,
    output_dir: Path,
    root: Path | str | None = None,
    *,
    force: bool = False,
    cache_dir: Path | None = None,
) -> TrimResult:
    """Trim an incomplete final record from a .p file, or reference it in place.

    Complete files are *not* copied: the original is used in place
    (``action="referenced"``) so the trim directory only ever holds files
    that were genuinely truncated. This avoids duplicating large raw files
    and needs no filesystem-specific support (hard/soft links, reflinks),
    so it works identically on local disks, SMB/network mounts, and Windows.

    Parameters
    ----------
    source : Path
        Path to the original .p file.
    output_dir : Path
        Directory for any trimmed output file. Untouched for complete files.
    root : Path, optional
        Source root used to preserve relative paths under *output_dir*.
    force : bool, keyword-only, default False
        Re-trim even when an up-to-date trimmed output already exists. By
        default an existing trimmed file at least as new as *source* is
        reused (``action="skipped"``).

    Returns
    -------
    TrimResult
        See :class:`TrimResult`. ``dest`` is the original source for
        ``"referenced"`` files and the trim-directory path otherwise.

    Notes
    -----
    Skip guard (trimmed files only): the header is read every run, and a
    trimmed output is reused only when its size equals the freshly-computed
    trimmed size AND ``dest_mtime >= source_mtime`` (the trim-write path
    stamps a "now" mtime newer than the older source). A source modified
    after its trimmed output (newer mtime), repaired to completeness, or
    given a changed header geometry therefore re-trims/re-references rather
    than reusing a stale truncated copy. Complete files write nothing — they
    are re-referenced each run at the cost of a single 128-byte header read.
    """
    dest = trim_destination(source, output_dir, root=root)

    # Incremental skip: an unchanged source (fingerprint match) reuses last
    # run's decision from a single stat — no per-file header read/open, which is
    # what costs minutes over a slow mount. force_trim re-reads.
    if not force and cache_dir is not None:
        cached = _trim_cache_lookup(cache_dir, source, dest, output_dir)
        if cached is not None:
            return cached

    def _finish(result: TrimResult) -> TrimResult:
        if cache_dir is not None:
            _trim_cache_store(cache_dir, source, dest, output_dir, result)
        return result

    # Read the source header first (the same 128-byte read the 'referenced'
    # path already pays) so the skip decision is content-aware rather than
    # mtime-only. A previously-incomplete source that was later repaired (now
    # complete, or with changed header geometry) but carries an equal/older
    # mtime than the stale trimmed output must NOT be skipped.
    with open(source, "rb") as f:
        raw_hdr = f.read(HEADER_BYTES)
        if len(raw_hdr) < HEADER_BYTES:
            raise ValueError(f"{source.name}: file too small for header")

        endian = _detect_endian(raw_hdr, source)
        fmt = f"{endian}{HEADER_WORDS}H"
        words = struct.unpack(fmt, raw_hdr)
        header = {name: words[idx] for name, idx in _H.items()}

        header_size = header["header_size"]
        config_size = header["config_size"]
        record_size = header["record_size"]

        first_record_size = header_size + config_size

        f.seek(0, 2)
        file_size = f.tell()

    data_bytes = file_size - first_record_size
    if data_bytes < 0:
        raise ValueError(f"{source.name}: file smaller than first record")

    if record_size <= 0:
        raise ValueError(f"{source.name}: invalid record_size={record_size}")

    remainder = data_bytes % record_size
    if remainder == 0:
        # Complete file — reference the original in place, no copy. Reached
        # also when a once-incomplete source has since been completed, so its
        # stale trimmed output is deliberately not reused.
        return _finish(TrimResult(source, "referenced", 0, record_size))

    # Trim the fractional final record into the trim directory.
    n_complete = data_bytes // record_size
    trimmed_size = first_record_size + n_complete * record_size

    # Skip-if-up-to-date: a prior run wrote this exact trimmed output and the
    # source is unchanged. Require the dest SIZE to equal the freshly-computed
    # trimmed size (so a repaired or geometry-changed source re-trims rather
    # than reusing a stale truncated copy) on top of the nanosecond-mtime
    # guard. The trim-write path stamps a "now" mtime, so dest_mtime >=
    # source_mtime holds for an unchanged source.
    if (
        not force
        and dest.exists()
        and dest.stat().st_size == trimmed_size
        and dest.stat().st_mtime_ns >= source.stat().st_mtime_ns
    ):
        return _finish(TrimResult(dest, "skipped", 0, 0))

    dest.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file then atomically replace. If the dest resolves to the
    # source (files already under <root>/trimmed/, or a basename collision),
    # opening dest "wb" directly would truncate the source to 0 bytes BEFORE the
    # read — destroying it. A separate temp fd + os.replace keeps the source
    # intact until the atomic swap.
    fd, tmp = tempfile.mkstemp(dir=str(dest.parent), prefix=".trim-", suffix=".tmp")
    try:
        with open(source, "rb") as src_f, os.fdopen(fd, "wb") as dst_f:
            remaining = trimmed_size
            while remaining > 0:
                chunk = src_f.read(min(remaining, 1 << 20))
                if not chunk:
                    break
                dst_f.write(chunk)
                remaining -= len(chunk)
        os.replace(tmp, dest)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise

    return _finish(TrimResult(dest, "trimmed", remainder, record_size))
