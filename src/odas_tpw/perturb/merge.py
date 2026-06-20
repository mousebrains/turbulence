# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Merge split .p files that were rolled over due to file size limits.

Files are mergeable when they share the same configuration (config string
hash), endianness, and record size, and have sequential file numbers.

Reference: Code/merge_p_files.m
"""

import contextlib
import hashlib
import os
import shutil
import struct
import tempfile
from pathlib import Path

from odas_tpw.rsi.p_file import _H, HEADER_BYTES, HEADER_WORDS, _detect_endian


def _read_merge_info(path: Path) -> dict:
    """Read header fields needed for merge decisions."""
    with open(path, "rb") as f:
        raw_hdr = f.read(HEADER_BYTES)
        if len(raw_hdr) < HEADER_BYTES:
            raise ValueError(f"{path.name}: file too small for header")

        endian = _detect_endian(raw_hdr, path)
        fmt = f"{endian}{HEADER_WORDS}H"
        words = struct.unpack(fmt, raw_hdr)
        header = {name: words[idx] for name, idx in _H.items()}

        header_size = header["header_size"]
        config_size = header["config_size"]
        record_size = header["record_size"]
        file_number = header["file_number"]

        # Read config string to compute hash for matching
        f.seek(header_size)
        config_bytes = f.read(config_size)
        config_hash = hashlib.md5(config_bytes).hexdigest()

        f.seek(0, 2)
        file_size = f.tell()

    return {
        "path": path,
        "file_number": file_number,
        "endian": endian,
        "header_size": header_size,
        "config_size": config_size,
        "record_size": record_size,
        "config_hash": config_hash,
        "file_size": file_size,
    }


def _file_group_key(info: dict) -> tuple:
    """Key for grouping files that could potentially be merged.

    header_size and config_size are part of the key, not just record_size:
    the merged file is reparsed with the FIRST file's geometry (PFile._read
    slices every record as header_size//2 header words + data words), but the
    splice skips each continuation file's own header_size + config_size. If a
    chained file's header_size differed, its records would be silently
    mis-sliced. Keying on the full geometry keeps geometry-mismatched files in
    separate groups so they never chain.
    """
    return (
        info["config_hash"],
        info["endian"],
        info["record_size"],
        info["header_size"],
        info["config_size"],
    )


def find_mergeable_files(p_files: list[Path]) -> list[list[Path]]:
    """Identify chains of split files that should be merged.

    Files are chained when they share the same config hash, endianness,
    and record size, and have sequential file numbers.

    Parameters
    ----------
    p_files : list of Path
        .p file paths (typically from :func:`perturb.discover.find_p_files`).

    Returns
    -------
    list of list of Path
        Each inner list is a chain of files to merge (length >= 2).
        Single files (no merge partner) are not included.
    """
    if len(p_files) < 2:
        return []

    # Read headers
    infos = []
    for p in p_files:
        try:
            infos.append(_read_merge_info(p))
        except (ValueError, OSError):
            continue

    # Group by matching properties
    groups: dict[tuple, list[dict]] = {}
    for info in infos:
        key = _file_group_key(info)
        groups.setdefault(key, []).append(info)

    chains = []
    for group in groups.values():
        if len(group) < 2:
            continue

        # Sort by file number
        group.sort(key=lambda x: x["file_number"])

        # Build chains of sequential file numbers
        chain = [group[0]]
        for i in range(1, len(group)):
            if group[i]["file_number"] == chain[-1]["file_number"] + 1:
                chain.append(group[i])
            else:
                if len(chain) >= 2:
                    chains.append([info["path"] for info in chain])
                chain = [group[i]]
        if len(chain) >= 2:
            chains.append([info["path"] for info in chain])

    return chains


def merge_destination(
    chain: list[Path],
    output_dir: Path,
    root: Path | str | None = None,
) -> Path:
    """Return the merged output path for *chain*.

    The first file in the chain names the merged file.  When *root* is
    supplied and that first file lives underneath it, the relative directory
    structure is preserved under *output_dir*.
    """
    if not chain:
        raise ValueError("Empty chain")
    first = Path(chain[0])
    output_dir = Path(output_dir)
    if root is not None:
        try:
            rel = first.resolve().relative_to(Path(root).resolve())
        except ValueError:
            rel = Path(first.name)
        return output_dir / rel
    return output_dir / first.name


def plan_merge_outputs(
    p_files: list[Path],
    output_dir: Path,
    root: Path | str | None = None,
) -> list[tuple[Path, list[Path]]]:
    """Plan merge outputs for all inputs.

    Returns ``(output_path, source_chain)`` pairs.  Mergeable chains map to a
    new output path; non-mergeable singleton files map to themselves so the
    caller can keep processing them unchanged.
    """
    chains = find_mergeable_files(p_files)
    chain_members = {p.resolve() for chain in chains for p in chain}

    plan: list[tuple[Path, list[Path]]] = []
    for chain in chains:
        plan.append((merge_destination(chain, output_dir, root=root), chain))
    for p in p_files:
        if p.resolve() not in chain_members:
            plan.append((p, [p]))
    return plan


def merge_p_files(
    chain: list[Path],
    output_dir: Path,
    root: Path | str | None = None,
) -> Path:
    """Merge a chain of split .p files into a single file.

    The first file is copied in its entirety.  Subsequent files contribute
    only their data records (header + config are skipped).

    Parameters
    ----------
    chain : list of Path
        Ordered list of .p files to merge (first file is the base).
    output_dir : Path
        Directory for the merged output file.
    root : Path, optional
        Source root used to preserve relative paths under *output_dir*.

    Returns
    -------
    Path
        Path to the merged output file (named after the first file in the chain).
    """
    if not chain:
        raise ValueError("Empty chain")

    dest = merge_destination(chain, output_dir, root=root)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if len(chain) == 1:
        if dest.resolve() != chain[0].resolve():
            shutil.copy2(chain[0], dest)
        return dest

    # Build the merged file in a temp file then atomically replace. If dest
    # resolves to chain[0] (output_dir == the chain's own directory), opening
    # dest "wb" directly would truncate the base file to 0 bytes BEFORE it is
    # read — yielding a headerless, unparseable file and destroying the source.
    # Writing via a separate temp fd keeps every source intact until os.replace.
    fd, tmp = tempfile.mkstemp(dir=str(dest.parent), prefix=".merge-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as out_f:
            # Copy entire first file
            with open(chain[0], "rb") as in_f:
                shutil.copyfileobj(in_f, out_f)

            # Append data records from subsequent files
            for p in chain[1:]:
                info = _read_merge_info(p)
                skip = info["header_size"] + info["config_size"]
                with open(p, "rb") as in_f:
                    in_f.seek(skip)
                    shutil.copyfileobj(in_f, out_f)
        os.replace(tmp, dest)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise

    return dest
