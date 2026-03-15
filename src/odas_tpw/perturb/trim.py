# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Corrupt record trimming for .p files.

If the last record is incomplete (fractional), copy only complete records
to a new file.

Reference: Code/trim_P_files.m
"""

import shutil
import struct
from pathlib import Path

from odas_tpw.rsi.p_file import _H, HEADER_BYTES, HEADER_WORDS, _detect_endian


def trim_p_file(source: Path, output_dir: Path) -> Path:
    """Trim an incomplete final record from a .p file.

    Parameters
    ----------
    source : Path
        Path to the original .p file.
    output_dir : Path
        Directory for the trimmed output file.

    Returns
    -------
    Path
        Path to the trimmed file.  If no trimming was needed, the file is
        still copied to *output_dir* for consistency.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / source.name

    with open(source, "rb") as f:
        raw_hdr = f.read(HEADER_BYTES)
        if len(raw_hdr) < HEADER_BYTES:
            raise ValueError(f"{source.name}: file too small for header")

        endian = _detect_endian(raw_hdr)
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
        # No trimming needed — copy as-is
        shutil.copy2(source, dest)
        return dest

    # Trim the fractional final record
    n_complete = data_bytes // record_size
    trimmed_size = first_record_size + n_complete * record_size

    with open(source, "rb") as src_f, open(dest, "wb") as dst_f:
        dst_f.write(src_f.read(trimmed_size))

    return dest
