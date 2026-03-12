# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Merge split .p files that were rolled over due to file size limits.

Files are mergeable when they share the same configuration (config string
hash), endianness, and record size, and have sequential file numbers.

Reference: Code/merge_p_files.m
"""

import hashlib
import shutil
import struct
from pathlib import Path

from rsi_python.p_file import _H, HEADER_BYTES, HEADER_WORDS, _detect_endian


def _read_merge_info(path: Path) -> dict:
    """Read header fields needed for merge decisions."""
    with open(path, "rb") as f:
        raw_hdr = f.read(HEADER_BYTES)
        if len(raw_hdr) < HEADER_BYTES:
            raise ValueError(f"{path.name}: file too small for header")

        endian = _detect_endian(raw_hdr)
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
    """Key for grouping files that could potentially be merged."""
    return (info["config_hash"], info["endian"], info["record_size"])


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


def merge_p_files(chain: list[Path], output_dir: Path) -> Path:
    """Merge a chain of split .p files into a single file.

    The first file is copied in its entirety.  Subsequent files contribute
    only their data records (header + config are skipped).

    Parameters
    ----------
    chain : list of Path
        Ordered list of .p files to merge (first file is the base).
    output_dir : Path
        Directory for the merged output file.

    Returns
    -------
    Path
        Path to the merged output file (named after the first file in the chain).
    """
    if not chain:
        raise ValueError("Empty chain")

    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / chain[0].name

    if len(chain) == 1:
        shutil.copy2(chain[0], dest)
        return dest

    with open(dest, "wb") as out_f:
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

    return dest
