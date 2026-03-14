# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb.merge — split file merging."""

import struct

from odas_tpw.perturb.merge import find_mergeable_files, merge_p_files
from odas_tpw.rsi.p_file import _H, HEADER_BYTES, HEADER_WORDS


def _make_p_file(
    path,
    *,
    file_number=1,
    record_size=512,
    config_size=128,
    config_content=b"[root]\nversion=1\n",
    n_records=2,
    data_byte=0x01,
):
    """Create a synthetic .p file for merge testing."""
    header_size = HEADER_BYTES

    words = [0] * HEADER_WORDS
    words[_H["header_size"]] = header_size
    words[_H["config_size"]] = config_size
    words[_H["record_size"]] = record_size
    words[_H["file_number"]] = file_number
    words[_H["endian"]] = 1  # little-endian

    hdr_bytes = struct.pack(f"<{HEADER_WORDS}H", *words)

    # Pad config content to config_size
    cfg = config_content[:config_size].ljust(config_size, b"\x00")

    data = hdr_bytes + cfg
    for _ in range(n_records):
        data += bytes([data_byte]) * record_size

    path.write_bytes(data)
    return path


class TestFindMergeableFiles:
    def test_sequential_files_detected(self, tmp_path):
        f1 = _make_p_file(tmp_path / "SN479_0001.p", file_number=1)
        f2 = _make_p_file(tmp_path / "SN479_0002.p", file_number=2)
        chains = find_mergeable_files([f1, f2])
        assert len(chains) == 1
        assert len(chains[0]) == 2

    def test_non_sequential_not_merged(self, tmp_path):
        f1 = _make_p_file(tmp_path / "SN479_0001.p", file_number=1)
        f3 = _make_p_file(tmp_path / "SN479_0003.p", file_number=3)
        chains = find_mergeable_files([f1, f3])
        assert len(chains) == 0

    def test_different_config_not_merged(self, tmp_path):
        f1 = _make_p_file(
            tmp_path / "SN479_0001.p", file_number=1, config_content=b"config_A"
        )
        f2 = _make_p_file(
            tmp_path / "SN479_0002.p", file_number=2, config_content=b"config_B"
        )
        chains = find_mergeable_files([f1, f2])
        assert len(chains) == 0

    def test_different_record_size_not_merged(self, tmp_path):
        f1 = _make_p_file(tmp_path / "SN479_0001.p", file_number=1, record_size=512)
        f2 = _make_p_file(tmp_path / "SN479_0002.p", file_number=2, record_size=1024)
        chains = find_mergeable_files([f1, f2])
        assert len(chains) == 0

    def test_single_file_no_chain(self, tmp_path):
        f1 = _make_p_file(tmp_path / "SN479_0001.p", file_number=1)
        chains = find_mergeable_files([f1])
        assert len(chains) == 0

    def test_three_file_chain(self, tmp_path):
        f1 = _make_p_file(tmp_path / "SN479_0001.p", file_number=1)
        f2 = _make_p_file(tmp_path / "SN479_0002.p", file_number=2)
        f3 = _make_p_file(tmp_path / "SN479_0003.p", file_number=3)
        chains = find_mergeable_files([f1, f2, f3])
        assert len(chains) == 1
        assert len(chains[0]) == 3

    def test_two_separate_chains(self, tmp_path):
        f1 = _make_p_file(
            tmp_path / "A_0001.p", file_number=1, config_content=b"cfg_A"
        )
        f2 = _make_p_file(
            tmp_path / "A_0002.p", file_number=2, config_content=b"cfg_A"
        )
        f3 = _make_p_file(
            tmp_path / "B_0001.p", file_number=1, config_content=b"cfg_B"
        )
        f4 = _make_p_file(
            tmp_path / "B_0002.p", file_number=2, config_content=b"cfg_B"
        )
        chains = find_mergeable_files([f1, f2, f3, f4])
        assert len(chains) == 2


class TestMergePFiles:
    def test_merge_two_files(self, tmp_path):
        record_size = 512
        config_size = 128
        n_records = 2

        f1 = _make_p_file(
            tmp_path / "SN479_0001.p",
            file_number=1,
            record_size=record_size,
            config_size=config_size,
            n_records=n_records,
            data_byte=0xAA,
        )
        f2 = _make_p_file(
            tmp_path / "SN479_0002.p",
            file_number=2,
            record_size=record_size,
            config_size=config_size,
            n_records=n_records,
            data_byte=0xBB,
        )

        out_dir = tmp_path / "merged"
        merged = merge_p_files([f1, f2], out_dir)

        assert merged.exists()
        assert merged.name == "SN479_0001.p"

        # Expected size: first file in full + data records from second file
        first_record = HEADER_BYTES + config_size
        f1_size = first_record + n_records * record_size
        f2_data = n_records * record_size
        expected_size = f1_size + f2_data
        assert merged.stat().st_size == expected_size

    def test_merged_data_concatenated(self, tmp_path):
        record_size = 256
        config_size = 128

        f1 = _make_p_file(
            tmp_path / "file_0001.p",
            file_number=1,
            record_size=record_size,
            config_size=config_size,
            n_records=1,
            data_byte=0xAA,
        )
        f2 = _make_p_file(
            tmp_path / "file_0002.p",
            file_number=2,
            record_size=record_size,
            config_size=config_size,
            n_records=1,
            data_byte=0xBB,
        )

        out_dir = tmp_path / "merged"
        merged = merge_p_files([f1, f2], out_dir)

        content = merged.read_bytes()
        first_record = HEADER_BYTES + config_size

        # First file's data record
        rec1_start = first_record
        rec1 = content[rec1_start : rec1_start + record_size]
        assert all(b == 0xAA for b in rec1)

        # Second file's data record (appended after first file)
        rec2_start = first_record + record_size
        rec2 = content[rec2_start : rec2_start + record_size]
        assert all(b == 0xBB for b in rec2)

    def test_output_directory_created(self, tmp_path):
        f1 = _make_p_file(tmp_path / "f.p", file_number=1)
        out_dir = tmp_path / "deep" / "nested"
        merged = merge_p_files([f1], out_dir)
        assert merged.exists()
