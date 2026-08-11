# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb.merge — split file merging."""

import struct
import warnings
from datetime import datetime, timedelta
from typing import ClassVar

import pytest

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
    header_size=HEADER_BYTES,
    start=None,
    header_version=0,
    clock_hz=0,
    fast_cols=0,
    slow_cols=0,
    n_rows=0,
):
    """Create a synthetic .p file for merge testing.

    ``start`` (a datetime), ``clock_hz`` and the matrix-geometry words are
    optional: they default to zero, which leaves the file's start time and
    duration uncomputable, so continuity checks are skipped exactly as before
    these parameters existed. Pass them to exercise the gap logic.
    """
    words = [0] * HEADER_WORDS
    words[_H["header_size"]] = header_size
    words[_H["config_size"]] = config_size
    words[_H["record_size"]] = record_size
    words[_H["file_number"]] = file_number
    words[_H["endian"]] = 1  # little-endian
    words[_H["header_version"]] = header_version
    words[_H["clock_hz"]] = clock_hz
    words[_H["fast_cols"]] = fast_cols
    words[_H["slow_cols"]] = slow_cols
    words[_H["n_rows"]] = n_rows
    if start is not None:
        words[_H["year"]] = start.year
        words[_H["month"]] = start.month
        words[_H["day"]] = start.day
        words[_H["hour"]] = start.hour
        words[_H["minute"]] = start.minute
        words[_H["second"]] = start.second
        words[_H["millisecond"]] = start.microsecond // 1000

    hdr_bytes = struct.pack(f"<{HEADER_WORDS}H", *words)
    # Pad the header region out to header_size (config starts at header_size).
    hdr_region = hdr_bytes.ljust(header_size, b"\x00")

    # Pad config content to config_size
    cfg = config_content[:config_size].ljust(config_size, b"\x00")

    data = hdr_region + cfg
    for _ in range(n_records):
        data += bytes([data_byte]) * record_size

    path.write_bytes(data)
    return path


class TestWriteTimeGapAmbiguity:
    """TN-051 (rev. 2026-01-12) s2.4.5: a pre-6.2 header stamp is the time a
    record was WRITTEN, and the RDL can buffer minutes of records first, so a
    timestamp gap between sequential files is not evidence of a data gap. Such
    a pair must not be merged on that evidence — but the ambiguity is reported
    rather than swallowed.
    """

    # 192 words/record / 1 column = 192 scans; 2 records at 512 Hz = 0.75 s.
    _GEOM: ClassVar[dict] = dict(
        record_size=512, clock_hz=512, fast_cols=1, slow_cols=0, n_rows=1, n_records=2
    )
    _DURATION = 0.75

    def _pair(self, tmp_path, gap_s, version):
        t0 = datetime(2025, 1, 15, 12, 0, 0)
        f1 = _make_p_file(
            tmp_path / "SN479_0001.p",
            file_number=1,
            start=t0,
            header_version=version,
            **self._GEOM,
        )
        f2 = _make_p_file(
            tmp_path / "SN479_0002.p",
            file_number=2,
            start=t0 + timedelta(seconds=self._DURATION + gap_s),
            header_version=version,
            **self._GEOM,
        )
        return [f1, f2]

    def test_v61_ambiguous_gap_warns_and_does_not_merge(self, tmp_path):
        files = self._pair(tmp_path, gap_s=60.0, version=0x0601)
        with pytest.warns(UserWarning, match="may be a buffering artifact"):
            chains = find_mergeable_files(files)
        assert chains == [], "an unverifiable gap must not be spliced together"

    def test_v63_ambiguous_gap_is_silent(self, tmp_path):
        """v6.2+ timestamps are calculated from the DAQ clock, so a gap is a
        gap — nothing to flag."""
        files = self._pair(tmp_path, gap_s=60.0, version=0x0603)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            chains = find_mergeable_files(files)
        assert chains == []

    def test_v61_large_gap_is_silent(self, tmp_path):
        """Beyond the few minutes TN-051 describes, the gap is taken as real."""
        files = self._pair(tmp_path, gap_s=3600.0, version=0x0601)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            chains = find_mergeable_files(files)
        assert chains == []

    def test_v61_contiguous_files_still_merge(self, tmp_path):
        files = self._pair(tmp_path, gap_s=0.0, version=0x0601)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            chains = find_mergeable_files(files)
        assert len(chains) == 1 and len(chains[0]) == 2


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
        f1 = _make_p_file(tmp_path / "SN479_0001.p", file_number=1, config_content=b"config_A")
        f2 = _make_p_file(tmp_path / "SN479_0002.p", file_number=2, config_content=b"config_B")
        chains = find_mergeable_files([f1, f2])
        assert len(chains) == 0

    def test_different_record_size_not_merged(self, tmp_path):
        f1 = _make_p_file(tmp_path / "SN479_0001.p", file_number=1, record_size=512)
        f2 = _make_p_file(tmp_path / "SN479_0002.p", file_number=2, record_size=1024)
        chains = find_mergeable_files([f1, f2])
        assert len(chains) == 0

    def test_different_header_size_not_merged(self, tmp_path):
        """Same config/record_size but different header_size must NOT chain:
        the merged file is reparsed with the first file's header geometry, so a
        mismatched continuation would be silently mis-sliced (audit #55)."""
        f1 = _make_p_file(tmp_path / "SN479_0001.p", file_number=1, header_size=HEADER_BYTES)
        f2 = _make_p_file(tmp_path / "SN479_0002.p", file_number=2, header_size=HEADER_BYTES + 32)
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
        f1 = _make_p_file(tmp_path / "A_0001.p", file_number=1, config_content=b"cfg_A")
        f2 = _make_p_file(tmp_path / "A_0002.p", file_number=2, config_content=b"cfg_A")
        f3 = _make_p_file(tmp_path / "B_0001.p", file_number=1, config_content=b"cfg_B")
        f4 = _make_p_file(tmp_path / "B_0002.p", file_number=2, config_content=b"cfg_B")
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

    def test_dest_equals_chain0_does_not_destroy_base(self, tmp_path):
        """When the merge dest resolves to chain[0] (output_dir == the chain's
        own dir), the base file must NOT be truncated to 0 bytes before it's
        read — the merged file is built via a temp + atomic replace (M-11)."""
        record_size, config_size, n = 512, 128, 2
        f1 = _make_p_file(tmp_path / "SN_0001.p", file_number=1,
                          record_size=record_size, config_size=config_size,
                          n_records=n, data_byte=0xAA)
        f2 = _make_p_file(tmp_path / "SN_0002.p", file_number=2,
                          record_size=record_size, config_size=config_size,
                          n_records=n, data_byte=0xBB)
        f2_size = f2.stat().st_size
        merged = merge_p_files([f1, f2], tmp_path, root=tmp_path)  # dest == f1
        assert merged.resolve() == f1.resolve()
        first_record = HEADER_BYTES + config_size
        expected = (first_record + n * record_size) + (n * record_size)
        assert merged.stat().st_size == expected   # base intact + f2 data appended
        assert f2.stat().st_size == f2_size         # continuation untouched
        # The base header/config survived (not a headerless 0xBB-only file).
        head = merged.read_bytes()[first_record:first_record + record_size]
        assert all(b == 0xAA for b in head)

    def test_empty_chain_raises(self, tmp_path):
        import pytest

        with pytest.raises(ValueError, match="Empty chain"):
            merge_p_files([], tmp_path / "out")

    def test_fractional_trailing_record_dropped(self, tmp_path):
        """A continuation with a fractional trailing record must not shift the
        merged geometry (audit #96). Only complete data records are appended, so
        the merged size stays an integer number of records relative to the base.

        On the OLD code the leftover bytes were copied verbatim, leaving the
        merged file mis-aligned (size != base_full + N*record_size)."""
        record_size, config_size, n = 512, 128, 2
        f1 = _make_p_file(tmp_path / "SN_0001.p", file_number=1,
                          record_size=record_size, config_size=config_size,
                          n_records=n, data_byte=0xAA)
        f2 = _make_p_file(tmp_path / "SN_0002.p", file_number=2,
                          record_size=record_size, config_size=config_size,
                          n_records=n, data_byte=0xBB)
        # Append a partial (sub-record) tail to the continuation, as an
        # un-trimmed mid-record rollover would leave behind.
        with open(f2, "ab") as fh:
            fh.write(b"\xCC" * (record_size // 3))

        out_dir = tmp_path / "merged"
        merged = merge_p_files([f1, f2], out_dir)

        first_record = HEADER_BYTES + config_size
        f1_full = first_record + n * record_size
        expected = f1_full + n * record_size  # only complete f2 records
        assert merged.stat().st_size == expected
        # Merged data region is an integer number of records of base geometry.
        assert (merged.stat().st_size - first_record) % record_size == 0
        # No 0xCC partial-record bytes leaked into the merged file.
        assert b"\xCC" not in merged.read_bytes()

    def test_base_fractional_trailing_record_dropped(self, tmp_path):
        """A fractional tail on the BASE file is likewise trimmed so the
        continuation's records stay aligned to base geometry (audit #96)."""
        record_size, config_size, n = 256, 128, 1
        f1 = _make_p_file(tmp_path / "B_0001.p", file_number=1,
                          record_size=record_size, config_size=config_size,
                          n_records=n, data_byte=0xAA)
        f2 = _make_p_file(tmp_path / "B_0002.p", file_number=2,
                          record_size=record_size, config_size=config_size,
                          n_records=n, data_byte=0xBB)
        with open(f1, "ab") as fh:
            fh.write(b"\xCC" * 17)  # sub-record junk on the base

        merged = merge_p_files([f1, f2], tmp_path / "merged")

        first_record = HEADER_BYTES + config_size
        expected = (first_record + n * record_size) + n * record_size
        assert merged.stat().st_size == expected
        assert b"\xCC" not in merged.read_bytes()


# ---------------------------------------------------------------------------
# Discovery edge cases
# ---------------------------------------------------------------------------


class TestFindMergeableFilesEdges:
    def test_too_small_for_header_skipped(self, tmp_path):
        """A file too small for a header is skipped, not raised."""
        good = _make_p_file(tmp_path / "good_0001.p", file_number=1)
        good2 = _make_p_file(tmp_path / "good_0002.p", file_number=2)
        # Bad file: too small for HEADER_BYTES → _read_merge_info raises → skipped
        bad = tmp_path / "bad.p"
        bad.write_bytes(b"\x00" * 10)
        chains = find_mergeable_files([good, bad, good2])
        # The good chain still forms even though bad raised
        assert len(chains) == 1
        assert len(chains[0]) == 2

    def test_non_sequential_break_finalizes_chain(self, tmp_path):
        """File numbers 1, 2, 5 → finalize [1,2] when 5 breaks the run."""
        f1 = _make_p_file(tmp_path / "f_0001.p", file_number=1)
        f2 = _make_p_file(tmp_path / "f_0002.p", file_number=2)
        f5 = _make_p_file(tmp_path / "f_0005.p", file_number=5)
        chains = find_mergeable_files([f1, f2, f5])
        # Only one chain (1,2) because 5 breaks the sequence and is alone
        assert len(chains) == 1
        assert len(chains[0]) == 2

    def test_two_chains_within_one_group(self, tmp_path):
        """Same config but two non-overlapping sequences: 1,2 then 5,6."""
        f1 = _make_p_file(tmp_path / "f_0001.p", file_number=1)
        f2 = _make_p_file(tmp_path / "f_0002.p", file_number=2)
        f5 = _make_p_file(tmp_path / "f_0005.p", file_number=5)
        f6 = _make_p_file(tmp_path / "f_0006.p", file_number=6)
        chains = find_mergeable_files([f1, f2, f5, f6])
        assert len(chains) == 2
        # Both chains have 2 files
        assert all(len(c) == 2 for c in chains)


# ---------------------------------------------------------------------------
# _read_merge_info errors
# ---------------------------------------------------------------------------


class TestReadMergeInfoErrors:
    def test_too_small_raises(self, tmp_path):
        import pytest

        from odas_tpw.perturb.merge import _read_merge_info

        bad = tmp_path / "tiny.p"
        bad.write_bytes(b"\x00" * 10)
        with pytest.raises(ValueError, match="too small"):
            _read_merge_info(bad)
