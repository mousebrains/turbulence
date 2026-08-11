"""Robustness tests for PFile against corrupted/truncated .p input.

Built by mutating a real file (tests/data/SN479_0006.p) so the header
layout stays authentic.
"""

import struct
import warnings
from pathlib import Path

import numpy as np
import pytest

from odas_tpw.rsi.p_file import PFile, diagnose_daq_clock

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


class TestMatrixDisagreesWithHeader:
    """TN-051 (rev. 2026-01-12) note 6: header words 29/30/31 ARE the address
    matrix dimensions, but the matrix VALUES come from the config string. When
    the two disagree the data block is still reshaped on the header's n_cols
    while channel columns are looked up in the config matrix — silently
    reading the wrong multiplexer slots. It must raise instead.
    """

    def test_column_count_mismatch_raises(self, tmp_path, src_bytes):
        """Header says 9 columns, config matrix has 10 -> ValueError."""
        words = _header_words(src_bytes)
        assert words[28] + words[29] == 10, "fixture geometry changed"
        mutated = _patch_word(src_bytes, 29, words[29] - 1)  # slow_cols 2 -> 1
        p = tmp_path / "narrow_matrix.p"
        p.write_bytes(mutated)
        with pytest.raises(ValueError, match=r"config \[matrix\] is 8x10 .*declares 8x9"):
            PFile(p)

    def test_row_count_mismatch_raises(self, tmp_path, src_bytes):
        """Header says 4 rows, config matrix has 8 -> ValueError.

        A short row count would break the slow-channel extraction stride.
        """
        words = _header_words(src_bytes)
        assert words[30] == 8, "fixture geometry changed"
        mutated = _patch_word(src_bytes, 30, 4)
        p = tmp_path / "short_matrix.p"
        p.write_bytes(mutated)
        with pytest.raises(ValueError, match=r"config \[matrix\] is 8x10 .*declares 4x10"):
            PFile(p)

    def test_record_not_whole_matrix_cycles_raises(self, tmp_path, src_bytes):
        """TN-051 s2.3: a data block is a multiple of the FULL matrix size.

        A record holding a fractional multiplexer cycle would shift the cycle
        phase at every record boundary, mis-assigning slow-channel samples.
        The pre-existing guard only caught a non-multiple of n_cols.
        """
        words = _header_words(src_bytes)
        header_size, record_size = words[17], words[18]
        n_cols, n_rows = words[28] + words[29], words[30]
        data_words = (record_size - header_size) // 2
        assert data_words % n_cols == 0 and (data_words // n_cols) % n_rows == 0
        # Drop exactly one scan: still a whole number of columns, no longer a
        # whole number of n_rows-row cycles.
        mutated = _patch_word(src_bytes, 18, record_size - 2 * n_cols)
        assert ((data_words - n_cols) // n_cols) % n_rows != 0
        p = tmp_path / "partial_cycle.p"
        p.write_bytes(mutated)
        with pytest.raises(ValueError, match="whole number of 8-row address-matrix cycles"):
            PFile(p)


class TestDroppedRecords:
    """Header word 2 is a running record counter. A forward jump means records
    are missing from the file — a silent time shift, since sample timestamps
    come from the sample index, not from the record headers.
    """

    @staticmethod
    def _record_offsets(raw):
        w = _header_words(raw)
        first, record_size = w[17] + w[11], w[18]
        n = (len(raw) - first) // record_size
        return first, record_size, n

    def _renumber(self, raw, numbers):
        """Rewrite word 2 of each data record from *numbers*."""
        first, record_size, n = self._record_offsets(raw)
        out = bytearray(raw)
        for i, value in zip(range(n), numbers):
            off = first + i * record_size + 2  # word 2 -> byte offset 2
            out[off : off + 2] = struct.pack(f"{_endian(raw)}H", value)
        return bytes(out)

    def test_gap_in_record_numbers_warns(self, tmp_path, src_bytes):
        _f, _rs, n = self._record_offsets(src_bytes)
        # 1..20 then resume at 31: records 21-30 (ten of them) never arrived.
        numbers = list(range(1, 21)) + list(range(31, 31 + n - 20))
        p = tmp_path / "dropped.p"
        p.write_bytes(self._renumber(src_bytes, numbers))
        with pytest.warns(UserWarning, match=r"skip 10 record\(s\) across 1 gap"):
            PFile(p)

    def test_counter_reset_is_not_a_gap(self, tmp_path, src_bytes):
        """A reset is a splice boundary (our own merge output, or a uint16
        wrap), not lost data."""
        _f, _rs, n = self._record_offsets(src_bytes)
        half = n // 2
        numbers = list(range(1, half + 1)) + list(range(1, n - half + 1))
        p = tmp_path / "spliced.p"
        p.write_bytes(self._renumber(src_bytes, numbers))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            PFile(p)

    def test_unpopulated_counter_is_silent(self, tmp_path, src_bytes):
        _f, _rs, n = self._record_offsets(src_bytes)
        p = tmp_path / "zeros.p"
        p.write_bytes(self._renumber(src_bytes, [0] * n))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            PFile(p)

    def test_clean_fixture_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            PFile(SRC)


class TestDaqClockDiagnosis:
    """TN-051 (rev. 2026-01-12) s2.4.3: v6.1/v6.2 firmware put a count-by-one
    error in the DAQ clock. The note publishes no correction, so this is a
    diagnostic only — fs_fast must keep the header's value.
    """

    def test_sn479_v61_flagged(self):
        """The real fixture: divisor 9376 where 9375 gives exactly 5120 Hz."""
        d = diagnose_daq_clock(5119.454, 10, 0x0601)
        assert d is not None
        assert (d["count"], d["expected_count"]) == (9376, 9375)
        assert d["corrected_fs_fast"] == pytest.approx(512.0)
        assert d["ppm"] == pytest.approx(106.65, abs=0.01)

    def test_fs_fast_is_not_corrected(self):
        """The diagnosis must not leak into the time base."""
        pf = PFile(SRC)
        assert pf.clock_diagnosis is not None
        assert pf.fs_fast == pytest.approx(511.9454), "fs must stay as recorded"
        assert pf.t_fast[1] - pf.t_fast[0] == pytest.approx(1 / 511.9454)

    @pytest.mark.parametrize(
        "f_clock,n_cols,version,why",
        [
            (48e6 / 9375, 10, 0x0601, "count already gives the exact rate"),
            (48e6 / 10416, 9, 0x0601, "requested rate not an exact divisor"),
            (5119.454, 10, 0x0603, "v6.3 firmware fixed the bug"),
            (5119.454, 10, 0x0600, "v6.0 predates it"),
            (1234.567, 10, 0x0601, "not an integer divisor of the clock"),
            (0.0, 10, 0x0601, "degenerate clock"),
        ],
    )
    def test_not_flagged(self, f_clock, n_cols, version, why):
        assert diagnose_daq_clock(f_clock, n_cols, version) is None, why


class TestBadBufferMarkers:
    """TN-051 (rev. 2026-01-12) s3.2: from v6.1 the RDL replaces missing or
    erroneous channel data with -32753 in place of the sample. Detection keys
    on run length because the sentinel is unlikely, not impossible: rail-riding
    channels in the real SN479 files produce isolated coincidences, while real
    dropouts there run 63-64 samples.
    """

    @staticmethod
    def _geometry(raw: bytes) -> dict:
        w = _header_words(raw)
        header_size, record_size = w[17], w[18]
        n_cols = w[28] + w[29]
        return {
            "first": header_size + w[11],
            "header_size": header_size,
            "record_size": record_size,
            "n_cols": n_cols,
            "n_rows": w[30],
            "scans_per_record": ((record_size - header_size) // 2) // n_cols,
        }

    @classmethod
    def _inject(cls, raw: bytes, cells) -> bytes:
        """Write the sentinel into (cycle, matrix_row, col) cells."""
        g = cls._geometry(raw)
        out = bytearray(raw)
        word = struct.pack(f"{_endian(raw)}h", -32753)
        for cycle, row, col in cells:
            scan = cycle * g["n_rows"] + row
            rec, scan_in_rec = divmod(scan, g["scans_per_record"])
            off = (
                g["first"]
                + rec * g["record_size"]
                + g["header_size"]
                + 2 * (scan_in_rec * g["n_cols"] + col)
            )
            out[off : off + 2] = word
        return bytes(out)

    def _slow_cell(self, raw):
        """A matrix cell holding an address sampled exactly once per cycle.

        Uniqueness matters: an address occupying several cells is scanned as
        several series, so its run positions have no single index space and
        ``spans`` is deliberately left None.
        """
        pf = PFile(SRC)
        for row, col in zip(*np.nonzero(pf.matrix >= 0)):
            addr = int(pf.matrix[row, col])
            if np.count_nonzero(pf.matrix == addr) == 1:
                return int(row), int(col), addr
        raise AssertionError("fixture has no uniquely-addressed cell")

    def test_run_on_slow_channel_warns(self, tmp_path, src_bytes):
        row, col, _addr = self._slow_cell(src_bytes)
        mutated = self._inject(src_bytes, [(c, row, col) for c in range(100, 108)])
        p = tmp_path / "dropout.p"
        p.write_bytes(mutated)

        with pytest.warns(UserWarning, match="RDL bad-buffer markers"):
            pf = PFile(p)
        confirmed = pf.bad_buffer_report["confirmed"]
        assert len(confirmed) == 1
        (found,) = confirmed.values()
        assert found["n_runs"] == 1
        assert found["longest_run"] == 8
        assert found["n_samples"] == 8
        assert found["spans"] == [(100, 8)]

    def test_isolated_hits_do_not_warn(self, tmp_path, src_bytes):
        """Scattered single samples are what ordinary rail-riding data looks
        like; reporting them as dropouts would fire on most real files."""
        row, col, _addr = self._slow_cell(src_bytes)
        mutated = self._inject(src_bytes, [(c, row, col) for c in (100, 400, 900)])
        p = tmp_path / "isolated.p"
        p.write_bytes(mutated)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pf = PFile(p)
        assert pf.bad_buffer_report["confirmed"] == {}
        assert sum(pf.bad_buffer_report["isolated"].values()) == 3

    def test_run_on_fast_channel_measured_in_sample_order(self, tmp_path, src_bytes):
        """A fast channel's samples run down its whole column, so a run must
        not be split into per-matrix-row fragments by scan interleaving."""
        pf_src = PFile(SRC)
        col = next(
            c
            for c in range(pf_src.n_cols)
            if len(set(pf_src.matrix[:, c].tolist())) == 1
        )
        g = self._geometry(src_bytes)
        # 8 consecutive scans in one column == one run of 8 for that channel,
        # spread across all 8 matrix rows of a single cycle.
        cells = [(100, row, col) for row in range(g["n_rows"])]
        p = tmp_path / "fast_dropout.p"
        p.write_bytes(self._inject(src_bytes, cells))

        with pytest.warns(UserWarning, match="RDL bad-buffer markers"):
            pf = PFile(p)
        (found,) = pf.bad_buffer_report["confirmed"].values()
        assert found["n_runs"] == 1, "run was fragmented by matrix row"
        assert found["longest_run"] == g["n_rows"]

    def test_v60_file_is_not_scanned(self, tmp_path, src_bytes):
        """v6.0 uses header word 16 + the channel-255 special character; the
        sentinel has no meaning there, so scanning would be pure false alarm."""
        row, col, _addr = self._slow_cell(src_bytes)
        mutated = self._inject(src_bytes, [(c, row, col) for c in range(100, 120)])
        mutated = _patch_word(mutated, 10, 0x0600)  # claim v6.0
        p = tmp_path / "v60.p"
        p.write_bytes(mutated)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pf = PFile(p)
        assert pf.bad_buffer_report == {}

    def test_spans_locate_the_samples(self, tmp_path, src_bytes):
        """The reported span must index the channel's own extracted samples."""
        row, col, addr = self._slow_cell(src_bytes)
        mutated = self._inject(src_bytes, [(c, row, col) for c in range(50, 56)])
        p = tmp_path / "spans.p"
        p.write_bytes(mutated)

        with pytest.warns(UserWarning, match="RDL bad-buffer markers"):
            pf = PFile(p, deconvolve=False)
        name, found = next(iter(pf.bad_buffer_report["confirmed"].items()))
        assert found["address"] == addr
        start, length = found["spans"][0]
        raw = pf.channels_raw[name]
        assert np.all(raw[start : start + length].astype(np.int64) == -32753)

    def test_clean_fixture_reports_nothing(self):
        pf = PFile(SRC)
        assert pf.bad_buffer_report == {}


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
