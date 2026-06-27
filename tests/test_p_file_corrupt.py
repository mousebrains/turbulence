"""Robustness tests for PFile against corrupted/truncated .p input.

Built by mutating a real file (tests/data/SN479_0006.p) so the header
layout stays authentic.
"""

import struct
import warnings
from pathlib import Path

import numpy as np
import pytest

from odas_tpw.rsi.p_file import PFile

SRC = Path(__file__).parent / "data" / "SN479_0006.p"


def _endian(raw: bytes) -> str:
    """Detect byte order the same way PFile does (header_size == 128)."""
    for fmt in ("<", ">"):
        if struct.unpack(f"{fmt}64H", raw[:128])[17] == 128:
            return fmt
    raise AssertionError("cannot detect fixture endianness")


def _header_words(raw: bytes) -> list[int]:
    return list(struct.unpack(f"{_endian(raw)}64H", raw[:128]))


def _patch_word(raw: bytes, index: int, value: int) -> bytes:
    """Replace 16-bit header word *index* (0-based) in file byte order."""
    out = bytearray(raw)
    out[2 * index : 2 * index + 2] = struct.pack(f"{_endian(raw)}H", value)
    return bytes(out)


@pytest.fixture(scope="module")
def src_bytes() -> bytes:
    return SRC.read_bytes()


class TestTruncatedFile:
    def test_partial_trailing_record_warns_and_reads(self, tmp_path, src_bytes):
        """A trailing partial record is dropped with a warning."""
        words = _header_words(src_bytes)
        header_size = words[17]  # sizes stored in bytes
        record_size = words[18]
        config_size = words[11]
        first = header_size + config_size
        n_records = (len(src_bytes) - first) // record_size
        assert n_records >= 2, "fixture too small"
        # Keep all-but-one full record plus half of the last one
        truncated = src_bytes[: first + (n_records - 1) * record_size + record_size // 2]
        p = tmp_path / "truncated.p"
        p.write_bytes(truncated)

        with pytest.warns(UserWarning, match="not an integer number of records"):
            pf = PFile(p)
        assert len(pf.t_slow) > 0
        assert np.all(np.isfinite(pf.channels["P"]))

    def test_no_data_records_raises(self, tmp_path, src_bytes):
        """Header+config only (no data records) raises ValueError."""
        words = _header_words(src_bytes)
        first = words[17] + words[11]  # sizes stored in bytes
        p = tmp_path / "empty.p"
        p.write_bytes(src_bytes[:first])
        with pytest.raises(ValueError, match="no data records"):
            PFile(p)

    def test_tiny_file_raises(self, tmp_path):
        """A file smaller than one header cannot be parsed.

        Regression: the short-header guard must raise ValueError (caught by
        every batch handler), not a raw struct.error that escapes them.
        """
        p = tmp_path / "tiny.p"
        p.write_bytes(b"\x00" * 32)
        with pytest.raises(ValueError, match="too small for header"):
            PFile(p)


class TestCorruptHeaderGeometry:
    """A corrupt matrix/record geometry must raise a clear ValueError naming
    the file, not a bare ZeroDivisionError or opaque numpy reshape error."""

    def test_zero_cols_raises_clear_error(self, tmp_path, src_bytes):
        """fast_cols == slow_cols == 0 -> ValueError, not ZeroDivisionError."""
        mutated = _patch_word(src_bytes, 28, 0)  # fast_cols
        mutated = _patch_word(mutated, 29, 0)  # slow_cols
        p = tmp_path / "zero_cols.p"
        p.write_bytes(mutated)
        with pytest.raises(ValueError, match="invalid matrix geometry"):
            PFile(p)

    def test_zero_rows_raises_clear_error(self, tmp_path, src_bytes):
        """n_rows == 0 -> ValueError, not ZeroDivisionError."""
        mutated = _patch_word(src_bytes, 30, 0)  # n_rows
        p = tmp_path / "zero_rows.p"
        p.write_bytes(mutated)
        with pytest.raises(ValueError, match="invalid matrix geometry"):
            PFile(p)

    def test_non_multiple_record_geometry_raises_clear_error(self, tmp_path, src_bytes):
        """data_words not a multiple of n_cols -> ValueError, not a numpy
        'cannot reshape array' error."""
        words = _header_words(src_bytes)
        header_size = words[17]
        record_size = words[18]
        n_cols = words[28] + words[29]
        data_words = (record_size - header_size) // 2
        assert data_words % n_cols == 0, "fixture should start well-formed"
        # Shrink record_size by 2 bytes so data_words drops by 1 and is no
        # longer a multiple of n_cols (n_cols > 1 for the SN479 fixture).
        assert n_cols > 1
        mutated = _patch_word(src_bytes, 18, record_size - 2)
        p = tmp_path / "bad_geom.p"
        p.write_bytes(mutated)
        with pytest.raises(ValueError, match="corrupt record geometry"):
            PFile(p)


class TestTimezoneSignedness:
    def test_negative_timezone_parsed(self, tmp_path, src_bytes):
        """A west-of-UTC timezone (stored two's complement) must not crash."""
        # timezone_min is header word 14
        mutated = _patch_word(src_bytes, 14, (-600) & 0xFFFF)  # UTC-10h
        p = tmp_path / "tz_west.p"
        p.write_bytes(mutated)
        pf = PFile(p)
        offset = pf.start_time.utcoffset()
        assert offset is not None
        assert offset.total_seconds() == -600 * 60

    def test_positive_timezone_unchanged(self, tmp_path, src_bytes):
        mutated = _patch_word(src_bytes, 14, 600)  # UTC+10h
        p = tmp_path / "tz_east.p"
        p.write_bytes(mutated)
        pf = PFile(p)
        offset = pf.start_time.utcoffset()
        assert offset is not None
        assert offset.total_seconds() == 600 * 60


class TestBaseline:
    def test_clean_file_no_warnings(self):
        """The unmodified fixture reads without any warnings."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pf = PFile(SRC)
        assert "sh1" in pf.channels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
