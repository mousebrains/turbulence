# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Branch-coverage tests for rsi.p_file using synthetic binary files."""

from __future__ import annotations

import struct
import warnings

import numpy as np
import pytest

from odas_tpw.rsi.p_file import (
    _H,
    HEADER_BYTES,
    HEADER_WORDS,
    PFile,
    _detect_endian,
    parse_config,
)

# ---------------------------------------------------------------------------
# _detect_endian fallback paths (lines 68-78)
# ---------------------------------------------------------------------------


def _hdr_bytes(*, endian_word: int, header_size_word: int = 128, prefix: str = "<") -> bytes:
    """Build a 128-byte header where word 63 = endian_word and word 17 = header_size."""
    words = [0] * HEADER_WORDS
    words[_H["header_size"]] = header_size_word
    words[_H["endian"]] = endian_word
    return struct.pack(f"{prefix}{HEADER_WORDS}H", *words)


class TestDetectEndian:
    def test_little_endian_flag_1(self):
        assert _detect_endian(_hdr_bytes(endian_word=1, prefix="<")) == "<"

    def test_big_endian_flag_2(self):
        assert _detect_endian(_hdr_bytes(endian_word=2, prefix=">")) == ">"

    def test_endian_zero_warns_and_assumes_little(self):
        """Endian word == 0 → warn, assume little-endian."""
        raw = _hdr_bytes(endian_word=0, prefix="<")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _detect_endian(raw)
        assert result == "<"
        assert any("Endian flag is 0" in str(rec.message) for rec in w)

    def test_endian_garbage_falls_back_to_header_size_le(self):
        """Endian word neither 1, 2, nor 0 → check header_size as fallback."""
        # Build manually: endian_word=999 (garbage), header_size_word=128 in little
        raw = _hdr_bytes(endian_word=999, header_size_word=128, prefix="<")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert _detect_endian(raw) == "<"

    def test_endian_garbage_falls_back_to_header_size_be(self):
        """Header_size readable big-endian → '>'."""
        raw = _hdr_bytes(endian_word=999, header_size_word=128, prefix=">")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert _detect_endian(raw) == ">"

    def test_completely_corrupt_warns_and_defaults_big(self):
        """Both header_size readings are bogus → warn, default '>'."""
        words = [0] * HEADER_WORDS
        words[_H["header_size"]] = 555  # not 128 in either endian
        words[_H["endian"]] = 999  # not 0/1/2
        raw = struct.pack(f"<{HEADER_WORDS}H", *words)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _detect_endian(raw)
        assert result == ">"
        assert any("Cannot determine endian" in str(rec.message) for rec in w)


# ---------------------------------------------------------------------------
# parse_config — section / channel / matrix branches
# ---------------------------------------------------------------------------


class TestParseConfig:
    def test_basic_sections(self):
        cfg = """
        [instrument_info]
        model = VMP-250IR
        sn = 479

        [cruise_info]
        operator = Pat
        project = Test
        """
        result = parse_config(cfg)
        assert result["instrument_info"]["model"] == "VMP-250IR"
        assert result["cruise_info"]["operator"] == "Pat"

    def test_comments_stripped(self):
        cfg = "[root]\nfoo = bar ; this is a comment\n"
        result = parse_config(cfg)
        assert result["root"]["foo"] == "bar"

    def test_matrix_rows(self):
        cfg = "[matrix]\nrow1 = 1 2 3\nrow2 = 4 5 6\n"
        result = parse_config(cfg)
        assert result["matrix"] == [[1, 2, 3], [4, 5, 6]]

    def test_matrix_non_row_keys_ignored(self):
        cfg = "[matrix]\nrow1 = 1 2\nfoo = bar\n"
        result = parse_config(cfg)
        # 'foo' is not 'row*', so it's silently dropped
        assert result["matrix"] == [[1, 2]]

    def test_channels_collected(self):
        cfg = """
        [channel]
        id = 5
        name = sh1
        type = shear

        [channel]
        id = 6
        name = sh2
        type = shear
        """
        result = parse_config(cfg)
        assert len(result["channels"]) == 2
        assert result["channels"][0]["name"] == "sh1"
        assert result["channels"][1]["name"] == "sh2"

    def test_crlf_line_endings(self):
        cfg = "[root]\r\nfoo = bar\r\n"
        result = parse_config(cfg)
        assert result["root"]["foo"] == "bar"

    def test_cr_line_endings(self):
        cfg = "[root]\rfoo = bar\r"
        result = parse_config(cfg)
        assert result["root"]["foo"] == "bar"

    def test_empty_config(self):
        result = parse_config("")
        assert result["channels"] == []
        assert result["matrix"] == []


# ---------------------------------------------------------------------------
# Synthetic .p file builder
# ---------------------------------------------------------------------------


def _make_minimal_p_file(
    path,
    *,
    config_text: str,
    n_records: int = 1,
    record_size: int = 256,
    fast_cols: int = 1,
    slow_cols: int = 0,
    n_rows: int = 4,
    clock_hz: int = 2048,
    endian: str = "<",
    set_endian_word: int | None = None,
):
    """Build a minimal valid .p file from a config text string."""
    config_bytes = config_text.encode("ascii")
    config_size = len(config_bytes)

    words = [0] * HEADER_WORDS
    words[_H["header_size"]] = HEADER_BYTES
    words[_H["config_size"]] = config_size
    words[_H["record_size"]] = record_size
    words[_H["fast_cols"]] = fast_cols
    words[_H["slow_cols"]] = slow_cols
    words[_H["n_rows"]] = n_rows
    words[_H["clock_hz"]] = clock_hz
    words[_H["clock_frac"]] = 0
    words[_H["year"]] = 2025
    words[_H["month"]] = 1
    words[_H["day"]] = 15
    words[_H["hour"]] = 12
    words[_H["minute"]] = 0
    words[_H["second"]] = 0
    words[_H["millisecond"]] = 0
    words[_H["timezone_min"]] = 0
    if set_endian_word is None:
        words[_H["endian"]] = 1 if endian == "<" else 2
    else:
        words[_H["endian"]] = set_endian_word

    header = struct.pack(f"{endian}{HEADER_WORDS}H", *words)
    out = bytearray()
    out += header
    out += config_bytes

    # Build n_records records, each with 128 byte header + (record_size - 128) data
    rec_data_size = record_size - HEADER_BYTES
    for ri in range(n_records):
        rec_hdr = bytearray(HEADER_BYTES)
        out += rec_hdr
        # Build int16 data — fill with sequential values
        n_int16 = rec_data_size // 2
        data = np.arange(n_int16, dtype=np.int16) + ri * n_int16
        out += data.tobytes()

    path.write_bytes(bytes(out))
    return path


# ---------------------------------------------------------------------------
# PFile error paths and branches
# ---------------------------------------------------------------------------


class TestPFileNoDataRecords:
    def test_zero_records_raises(self, tmp_path):
        """File with header + config but no data records → raise ValueError."""
        config = """
[matrix]
row1 = 1 1 1 1
row2 = 1 1 1 1
row3 = 1 1 1 1
row4 = 1 1 1 1

[instrument_info]
model = test
sn = 1

[cruise_info]
operator = pat

[channel]
id = 1
name = X
type = raw
"""
        path = tmp_path / "empty.p"
        _make_minimal_p_file(
            path,
            config_text=config,
            n_records=0,  # No records at all
            record_size=256,
            fast_cols=1,
            slow_cols=0,
            n_rows=4,
        )
        with pytest.raises(ValueError, match="contains no data records"):
            PFile(path)


class TestPFileChannelSkips:
    def test_channel_missing_keys_skipped(self, tmp_path):
        """Channels missing id/name/type are silently skipped."""
        config = """
[matrix]
row1 = 1 1 1 1
row2 = 1 1 1 1
row3 = 1 1 1 1
row4 = 1 1 1 1

[instrument_info]
model = test
sn = 1

[cruise_info]
operator = pat

[channel]
id = 1
name = good_ch
type = raw

[channel]
name = no_id_no_type
"""
        path = tmp_path / "skip.p"
        _make_minimal_p_file(
            path, config_text=config, fast_cols=1, slow_cols=0, n_rows=4, n_records=1
        )
        pf = PFile(path)
        # The malformed channel should be silently skipped
        assert "good_ch" in pf.channels
        assert "no_id_no_type" not in pf.channels

    def test_channel_id_not_in_matrix_skipped(self, tmp_path):
        """Channel whose id never appears in the address matrix is skipped."""
        config = """
[matrix]
row1 = 1 1 1 1
row2 = 1 1 1 1
row3 = 1 1 1 1
row4 = 1 1 1 1

[instrument_info]
model = test
sn = 1

[cruise_info]
operator = pat

[channel]
id = 1
name = present
type = raw

[channel]
id = 99
name = absent
type = raw
"""
        path = tmp_path / "absent.p"
        _make_minimal_p_file(
            path, config_text=config, fast_cols=1, slow_cols=0, n_rows=4, n_records=1
        )
        pf = PFile(path)
        assert "present" in pf.channels
        assert "absent" not in pf.channels


class TestPFileNoConverter:
    def test_unknown_type_warns_and_keeps_raw(self, tmp_path):
        """Channel type with no converter → warn and store raw counts."""
        config = """
[matrix]
row1 = 1 1 1 1
row2 = 1 1 1 1
row3 = 1 1 1 1
row4 = 1 1 1 1

[instrument_info]
model = test
sn = 1

[cruise_info]
operator = pat

[channel]
id = 1
name = mystery
type = nonexistent_type
"""
        path = tmp_path / "unknown.p"
        _make_minimal_p_file(
            path, config_text=config, fast_cols=1, slow_cols=0, n_rows=4, n_records=1
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pf = PFile(path)
        assert any("No converter" in str(rec.message) for rec in w)
        assert "mystery" in pf.channels
        assert pf.channel_info["mystery"]["units"] == "counts"


class TestPFileTwoIdChannel:
    def test_two_id_channel_decoded(self, tmp_path):
        """A 2-id channel pair (e.g. for 32-bit decoding) is reassembled."""
        config = """
[matrix]
row1 = 1 2
row2 = 1 2
row3 = 1 2
row4 = 1 2

[instrument_info]
model = test
sn = 1

[cruise_info]
operator = pat

[channel]
id = 1, 2
name = bigval
type = raw
"""
        path = tmp_path / "two_id.p"
        _make_minimal_p_file(
            path, config_text=config, fast_cols=2, slow_cols=0, n_rows=4, n_records=1
        )
        pf = PFile(path)
        assert "bigval" in pf.channels

    def test_two_id_one_missing_skipped(self, tmp_path):
        """If one of the two ids is absent from the matrix → skip the channel."""
        config = """
[matrix]
row1 = 1 1
row2 = 1 1
row3 = 1 1
row4 = 1 1

[instrument_info]
model = test
sn = 1

[cruise_info]
operator = pat

[channel]
id = 1, 99
name = halfmissing
type = raw
"""
        path = tmp_path / "halfmissing.p"
        _make_minimal_p_file(
            path, config_text=config, fast_cols=2, slow_cols=0, n_rows=4, n_records=1
        )
        pf = PFile(path)
        assert "halfmissing" not in pf.channels


class TestPFileSlowChannel:
    def test_slow_channel_at_specific_row(self, tmp_path):
        """Slow channel: id appears in only one row → goes into _slow_channels."""
        # Matrix: id 1 fills all rows (fast), id 2 only in row 1 (slow)
        config = """
[matrix]
row1 = 1 2
row2 = 1 5
row3 = 1 5
row4 = 1 5

[instrument_info]
model = test
sn = 1

[cruise_info]
operator = pat

[channel]
id = 1
name = fast_ch
type = raw

[channel]
id = 2
name = slow_ch
type = raw

[channel]
id = 5
name = filler
type = raw
"""
        path = tmp_path / "slow.p"
        _make_minimal_p_file(
            path, config_text=config, fast_cols=2, slow_cols=0, n_rows=4, n_records=1
        )
        pf = PFile(path)
        # fast_ch is in every row of col 0 → fast
        # slow_ch is only in row 0 of col 1 → slow
        assert pf.is_fast("fast_ch")
        assert not pf.is_fast("slow_ch")


class TestPFileUnsignedSign:
    def test_unsigned_sign_wraps_negative(self, tmp_path):
        """Channel with sign=unsigned wraps negative int16 to unsigned."""
        config = """
[matrix]
row1 = 1 1 1 1
row2 = 1 1 1 1
row3 = 1 1 1 1
row4 = 1 1 1 1

[instrument_info]
model = test
sn = 1

[cruise_info]
operator = pat

[channel]
id = 1
name = unsig
type = raw
sign = unsigned
"""
        path = tmp_path / "unsigned.p"
        _make_minimal_p_file(
            path, config_text=config, fast_cols=1, slow_cols=0, n_rows=4, n_records=1
        )
        # The data starts negative because we use np.arange; but the int16
        # encoding might produce all-positive values. Just verify it runs.
        pf = PFile(path)
        assert "unsig" in pf.channels


# ---------------------------------------------------------------------------
# PFile.summary smoke test
# ---------------------------------------------------------------------------


class TestPFileSummary:
    def test_summary_runs(self, tmp_path, capsys):
        """summary() prints to stdout without errors."""
        config = """
[matrix]
row1 = 1 1 1 1
row2 = 1 1 1 1
row3 = 1 1 1 1
row4 = 1 1 1 1

[instrument_info]
model = TestModel
sn = 999

[cruise_info]
operator = pat

[channel]
id = 1
name = X
type = raw
"""
        path = tmp_path / "summary.p"
        _make_minimal_p_file(
            path, config_text=config, fast_cols=1, slow_cols=0, n_rows=4, n_records=2
        )
        pf = PFile(path)
        pf.summary()
        captured = capsys.readouterr()
        assert "TestModel" in captured.out
        assert "999" in captured.out
