# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb.trim — corrupt record trimming."""

import struct

import pytest

from odas_tpw.perturb.trim import trim_p_file
from odas_tpw.rsi.p_file import _H, HEADER_BYTES, HEADER_WORDS


def _make_p_file(path, *, record_size=1024, config_size=256, n_records=3, extra_bytes=0):
    """Create a synthetic .p file with the given structure.

    Layout: [header_record] [data_record]*n_records [extra_bytes]
    header_record = 128-byte header + config_size bytes of config
    data_record = 128-byte header + (record_size - 128) bytes of data
    """
    header_size = HEADER_BYTES

    # Build a header with the fields trim.py reads
    words = [0] * HEADER_WORDS
    words[_H["header_size"]] = header_size
    words[_H["config_size"]] = config_size
    words[_H["record_size"]] = record_size
    words[_H["endian"]] = 1  # little-endian flag

    hdr_bytes = struct.pack(f"<{HEADER_WORDS}H", *words)
    config_bytes = b"\x00" * config_size

    # First record (header + config)
    data = hdr_bytes + config_bytes

    # Data records (each = record_size bytes)
    for _ in range(n_records):
        data += b"\x01" * record_size

    # Fractional record
    if extra_bytes > 0:
        data += b"\xff" * extra_bytes

    path.write_bytes(data)
    return path


class TestTrimPFile:
    def test_no_trim_needed(self, tmp_path):
        src = _make_p_file(tmp_path / "clean.p", n_records=3, extra_bytes=0)
        out_dir = tmp_path / "trimmed"
        dest = trim_p_file(src, out_dir)
        assert dest.exists()
        assert dest.stat().st_size == src.stat().st_size

    def test_trims_fractional_record(self, tmp_path):
        record_size = 1024
        src = _make_p_file(
            tmp_path / "frac.p",
            record_size=record_size,
            n_records=3,
            extra_bytes=50,
        )
        original_size = src.stat().st_size
        out_dir = tmp_path / "trimmed"
        dest = trim_p_file(src, out_dir)
        assert dest.exists()
        assert dest.stat().st_size == original_size - 50

    def test_output_in_correct_directory(self, tmp_path):
        src = _make_p_file(tmp_path / "test.p")
        out_dir = tmp_path / "output"
        dest = trim_p_file(src, out_dir)
        assert dest.parent == out_dir
        assert dest.name == "test.p"

    def test_creates_output_dir(self, tmp_path):
        src = _make_p_file(tmp_path / "test.p")
        out_dir = tmp_path / "deep" / "nested" / "dir"
        dest = trim_p_file(src, out_dir)
        assert dest.exists()

    def test_preserves_complete_records(self, tmp_path):
        record_size = 512
        config_size = 128
        n_records = 5
        src = _make_p_file(
            tmp_path / "multi.p",
            record_size=record_size,
            config_size=config_size,
            n_records=n_records,
            extra_bytes=100,
        )
        out_dir = tmp_path / "trimmed"
        dest = trim_p_file(src, out_dir)

        header_size = HEADER_BYTES
        first_record = header_size + config_size
        expected = first_record + n_records * record_size
        assert dest.stat().st_size == expected

    def test_file_too_small_raises(self, tmp_path):
        src = tmp_path / "tiny.p"
        src.write_bytes(b"\x00" * 10)
        with pytest.raises(ValueError, match="too small"):
            trim_p_file(src, tmp_path / "out")

    def test_file_smaller_than_first_record_raises(self, tmp_path):
        """File has a header that claims a large config_size, but is truncated."""
        # Build a 128-byte header that says config_size = 10000, then EOF
        words = [0] * HEADER_WORDS
        words[_H["header_size"]] = HEADER_BYTES
        words[_H["config_size"]] = 10000  # absurdly large
        words[_H["record_size"]] = 1024
        words[_H["endian"]] = 1

        src = tmp_path / "truncated.p"
        src.write_bytes(struct.pack(f"<{HEADER_WORDS}H", *words))  # only 128 bytes
        with pytest.raises(ValueError, match="smaller than first record"):
            trim_p_file(src, tmp_path / "out")

    def test_invalid_record_size_raises(self, tmp_path):
        """A record_size of 0 is rejected."""
        words = [0] * HEADER_WORDS
        words[_H["header_size"]] = HEADER_BYTES
        words[_H["config_size"]] = 16
        words[_H["record_size"]] = 0  # invalid
        words[_H["endian"]] = 1

        src = tmp_path / "bad_record.p"
        # 128 byte header + 16 bytes config = 144 bytes total, file_size >= first_record_size
        src.write_bytes(struct.pack(f"<{HEADER_WORDS}H", *words) + b"\x00" * 16)
        with pytest.raises(ValueError, match="invalid record_size"):
            trim_p_file(src, tmp_path / "out")
