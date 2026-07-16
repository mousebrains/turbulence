"""Tests for RSI P-file record extraction."""

from __future__ import annotations

import struct
import sys
from datetime import timedelta
from pathlib import Path

import pytest

from odas_tpw.rsi.p_file import _H, HEADER_WORDS, PFile, extract_pfile_segment

SAMPLE_FILE = Path(__file__).parent / "data" / "SN479_0006.p"


def _write_synthetic_pfile(path, *, n_records: int = 6, record_size: int = 160):
    """Write a minimal P-file-like byte stream for cutp tests."""
    header_size = 128
    config = b"[instrument_info]\nmodel=test\n"
    words = [0] * HEADER_WORDS
    words[_H["config_size"]] = len(config)
    words[_H["header_size"]] = header_size
    words[_H["record_size"]] = record_size
    words[_H["n_records_written"]] = n_records
    words[_H["endian"]] = 1
    header = struct.pack("<64H", *words)
    records = [bytes([i]) * record_size for i in range(n_records)]
    path.write_bytes(header + config + b"".join(records))
    return header + config, records


def test_extract_pfile_segment_copies_header_config_and_selected_records(tmp_path):
    first_record, records = _write_synthetic_pfile(tmp_path / "source.p", n_records=6)
    source = tmp_path / "source.p"
    dest = tmp_path / "segment.p"

    result = extract_pfile_segment(source, dest, start_record=2, n_records=3)

    assert result == dest
    assert dest.read_bytes() == first_record + b"".join(records[2:5])


def test_extract_pfile_segment_refuses_existing_output_without_overwrite(tmp_path):
    _write_synthetic_pfile(tmp_path / "source.p")
    source = tmp_path / "source.p"
    dest = tmp_path / "segment.p"
    dest.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        extract_pfile_segment(source, dest, start_record=0, n_records=1)

    extract_pfile_segment(source, dest, start_record=1, n_records=1, overwrite=True)
    assert dest.read_bytes().endswith(bytes([1]) * 160)


@pytest.mark.parametrize(
    ("start_record", "n_records", "message"),
    [
        (-1, 1, "start_record must be >= 0"),
        (0, 0, "n_records must be >= 1"),
        (6, 1, "out of range"),
        (5, 2, "only 1 complete record is available"),
    ],
)
def test_extract_pfile_segment_validates_record_range(
    tmp_path, start_record, n_records, message
):
    _write_synthetic_pfile(tmp_path / "source.p", n_records=6)

    with pytest.raises(ValueError, match=message):
        extract_pfile_segment(
            tmp_path / "source.p",
            tmp_path / "segment.p",
            start_record=start_record,
            n_records=n_records,
        )


def test_extract_pfile_segment_rejects_records_smaller_than_record_header(tmp_path):
    _write_synthetic_pfile(tmp_path / "source.p", record_size=64)

    with pytest.raises(ValueError, match="invalid record_size=64"):
        extract_pfile_segment(tmp_path / "source.p", tmp_path / "segment.p")


def test_extract_pfile_segment_from_fixture_opens_as_pfile(tmp_path):
    if not SAMPLE_FILE.exists():
        pytest.skip("Test data not available")

    dest = extract_pfile_segment(SAMPLE_FILE, tmp_path / "segment.p", start_record=1, n_records=3)

    pf = PFile(dest)
    assert pf.header["header_size"] == 128
    assert len(pf._record_headers) == 3


def test_extract_pfile_segment_nonzero_start_advances_timestamp(tmp_path):
    """A segment cut at start_record N carries absolute time N records forward."""
    if not SAMPLE_FILE.exists():
        pytest.skip("Test data not available")

    src = PFile(SAMPLE_FILE)
    dest = extract_pfile_segment(SAMPLE_FILE, tmp_path / "segment.p", start_record=5, n_records=2)
    seg = PFile(dest)
    assert seg.start_time == src.start_time + timedelta(seconds=5 * src.recsize)

    # Only the timestamp words (year..millisecond, words 3-9) may differ in
    # record 0; everything else — including the config — is copied verbatim.
    first_record_size = src.header["header_size"] + src.header["config_size"]
    src_r0 = SAMPLE_FILE.read_bytes()[:first_record_size]
    dst_r0 = dest.read_bytes()[:first_record_size]
    ts_lo, ts_hi = _H["year"] * 2, (_H["millisecond"] + 1) * 2
    assert dst_r0[:ts_lo] == src_r0[:ts_lo]
    assert dst_r0[ts_hi:] == src_r0[ts_hi:]


def test_extract_pfile_segment_timestamp_crosses_minute_boundary(tmp_path):
    """Offsets >= 60 s force a minute rollover regardless of the source clock."""
    if not SAMPLE_FILE.exists():
        pytest.skip("Test data not available")

    src = PFile(SAMPLE_FILE)
    dest = extract_pfile_segment(SAMPLE_FILE, tmp_path / "segment.p", start_record=61, n_records=1)
    seg = PFile(dest)
    expected = src.start_time + timedelta(seconds=61 * src.recsize)
    assert seg.start_time == expected
    assert seg.start_time.minute != src.start_time.minute  # boundary actually crossed


def test_extract_pfile_segment_invalid_header_date_left_unchanged(tmp_path):
    """A year-0 (startup) clock cannot be advanced: warn and copy verbatim."""
    first_record, records = _write_synthetic_pfile(tmp_path / "source.p", n_records=6)

    with pytest.warns(UserWarning, match="valid calendar date"):
        dest = extract_pfile_segment(
            tmp_path / "source.p", tmp_path / "segment.p", start_record=3, n_records=1
        )
    assert dest.read_bytes() == first_record + records[3]


def test_rsi_cli_cutp_writes_segment(monkeypatch, tmp_path):
    first_record, records = _write_synthetic_pfile(tmp_path / "source.p", n_records=5)
    dest = tmp_path / "segment.p"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rsi-tpw",
            "cutp",
            str(tmp_path / "source.p"),
            "-o",
            str(dest),
            "--start",
            "1",
            "--n-records",
            "2",
        ],
    )

    from odas_tpw.rsi.cli import main

    main()
    assert dest.read_bytes() == first_record + b"".join(records[1:3])


@pytest.mark.parametrize("force_flag", ["--force", "--overwrite"])
def test_rsi_cli_cutp_overwrite_flags(monkeypatch, tmp_path, force_flag):
    first_record, records = _write_synthetic_pfile(tmp_path / "source.p", n_records=4)
    dest = tmp_path / "segment.p"
    dest.write_bytes(b"existing")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rsi-tpw",
            "cutp",
            str(tmp_path / "source.p"),
            "-o",
            str(dest),
            "--start",
            "2",
            "--n-records",
            "1",
            force_flag,
        ],
    )

    from odas_tpw.rsi.cli import main

    main()
    assert dest.read_bytes() == first_record + records[2]
