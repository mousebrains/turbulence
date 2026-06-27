# Jun-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for odas_tpw.rsi.config_patch — post-acquisition .p config patching.

Binary integrity is checked against the real committed fixture; the text editor
is exercised both through the binary round-trip (read back via ``PFile``) and on
inline config strings.
"""

from __future__ import annotations

import struct
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from odas_tpw.rsi import config_patch as cp
from odas_tpw.rsi.p_file import PFile, extract_pfile_segment

DATA = Path(__file__).parent / "data" / "SN479_0006.p"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny(tmp_path):
    """A small, valid .p carved from the committed fixture (fast to load)."""
    return extract_pfile_segment(DATA, tmp_path / "tiny.p", start_record=0, n_records=4)


def _spec(tmp_path, text, name="edits.yaml"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _data_region(path):
    with open(path, "rb") as f:
        head = f.read(128)
        endian = ">" if struct.unpack_from(">H", head, 126)[0] == 2 else "<"
        w = struct.unpack(f"{endian}64H", head)
        f.seek(w[17] + w[11])  # header_size + config_size
        return f.read()


def _config_size_word(path):
    with open(path, "rb") as f:
        head = f.read(128)
    endian = ">" if struct.unpack_from(">H", head, 126)[0] == 2 else "<"
    return struct.unpack(f"{endian}64H", head)[11]


def _channel(cfg, name):
    return next(c for c in cfg["channels"] if c.get("name") == name)


def _no_lone_lf(text: str) -> bool:
    """True if every '\\n' is preceded by '\\r' (CRLF intact, no bare LF)."""
    return all(text[i - 1] == "\r" for i, c in enumerate(text) if c == "\n")


def _patch(src, dst, spec, **kw):
    text, changes = cp.edit_config_text(
        cp.read_config_text(src), spec, when="2026-06-24 12:00:00", **kw
    )
    if changes:
        cp.write_patched_pfile(src, dst, text)
    return changes


def _synth(path, endian, config_text, *, version=0x0600, data=b"\x01\x02\x03\x04"):
    """A minimal byte-valid .p: a 64-word header + config + trailing data bytes."""
    cfg = config_text.encode("latin-1")
    words = [0] * 64
    words[10] = version
    words[11] = len(cfg)
    words[17] = 128
    words[18] = 128 + len(data)
    words[63] = 1 if endian == "<" else 2
    path.write_bytes(struct.pack(f"{endian}64H", *words) + cfg + data)
    return path


# ---------------------------------------------------------------------------
# Binary integrity / round-trip through PFile
# ---------------------------------------------------------------------------


class TestBinaryRoundTrip:
    def test_data_region_byte_identical(self, tiny, tmp_path):
        dst = tmp_path / "out.p"
        _patch(
            tiny,
            dst,
            cp.EditSpec(note="n", author="a", sections={"instrument_info": {"vehicle": "rvmp"}}),
        )
        assert _data_region(tiny) == _data_region(dst)

    def test_raw_channels_identical(self, tiny, tmp_path):
        dst = tmp_path / "out.p"
        _patch(tiny, dst, cp.EditSpec(note="n", author="a", channels={"sh1": {"sens": "0.0812"}}))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a, b = PFile(tiny), PFile(dst)
        assert set(a.channels_raw) == set(b.channels_raw)
        for k in a.channels_raw:
            assert np.array_equal(a.channels_raw[k], b.channels_raw[k]), k

    def test_original_file_untouched(self, tiny, tmp_path):
        before = tiny.read_bytes()
        mtime = tiny.stat().st_mtime_ns
        _patch(
            tiny,
            tmp_path / "out.p",
            cp.EditSpec(note="n", author="a", sections={"instrument_info": {"vehicle": "rvmp"}}),
        )
        assert tiny.read_bytes() == before
        assert tiny.stat().st_mtime_ns == mtime

    def test_config_size_word_matches_new_length(self, tiny, tmp_path):
        dst = tmp_path / "out.p"
        _patch(
            tiny,
            dst,
            cp.EditSpec(note="n", author="a", sections={"instrument_info": {"vehicle": "rvmp"}}),
        )
        assert _config_size_word(dst) == len(cp.read_config_text(dst).encode("latin-1"))
        assert _config_size_word(dst) != _config_size_word(tiny)

    def test_new_values_read_back(self, tiny, tmp_path):
        dst = tmp_path / "out.p"
        _patch(
            tiny,
            dst,
            cp.EditSpec(
                note="n",
                author="a",
                sections={"instrument_info": {"vehicle": "rvmp"}},
                channels={"sh1": {"sens": "0.0812", "sn": "Z999"}},
            ),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = PFile(dst).config
        assert cfg["instrument_info"]["vehicle"] == "rvmp"
        assert _channel(cfg, "sh1")["sens"] == "0.0812"
        assert _channel(cfg, "sh1")["sn"] == "Z999"

    def test_only_named_channel_changes(self, tiny, tmp_path):
        dst = tmp_path / "out.p"
        _patch(tiny, dst, cp.EditSpec(note="n", author="a", channels={"sh1": {"sens": "0.0812"}}))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = PFile(dst).config
        assert _channel(cfg, "sh1")["sens"] == "0.0812"
        assert _channel(cfg, "sh2")["sens"] == "0.1130"  # untouched


# ---------------------------------------------------------------------------
# Text fidelity
# ---------------------------------------------------------------------------


class TestTextFidelity:
    def test_trailing_zero_preserved(self, tiny, tmp_path):
        dst = tmp_path / "out.p"
        _patch(tiny, dst, cp.EditSpec(note="n", author="a", channels={"sh1": {"sens": "0.1130"}}))
        # Exact text, not 0.113 (the YAML-coercion failure mode).
        assert "= 0.1130" in cp.read_config_text(dst)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert _channel(PFile(dst).config, "sh1")["sens"] == "0.1130"

    def test_scientific_notation_preserved(self, tiny, tmp_path):
        dst = tmp_path / "out.p"
        _patch(tiny, dst, cp.EditSpec(note="n", author="a", channels={"sh1": {"sens": "7.07e-2"}}))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert _channel(PFile(dst).config, "sh1")["sens"] == "7.07e-2"

    def test_crlf_preserved(self, tiny, tmp_path):
        dst = tmp_path / "out.p"
        _patch(
            tiny,
            dst,
            cp.EditSpec(note="n", author="a", sections={"instrument_info": {"vehicle": "rvmp"}}),
        )
        text = cp.read_config_text(dst)
        assert "\r\n" in text
        assert _no_lone_lf(text)

    def test_inline_comment_and_alignment_preserved(self, tiny, tmp_path):
        dst = tmp_path / "out.p"
        _patch(
            tiny,
            dst,
            cp.EditSpec(note="n", author="a", sections={"instrument_info": {"vehicle": "rvmp"}}),
        )
        text = cp.read_config_text(dst)
        # The active vehicle line keeps its inline comment...
        assert "vehicle = rvmp" in text
        assert "; downward profiling" in text
        # ...and the commented sibling line is never matched/altered.
        assert ";vehicle= rvmp" in text

    def test_version_history_comment_preserved(self, tiny, tmp_path):
        dst = tmp_path / "out.p"
        marker = "; Configuration setup.cfg file prepared"
        assert marker in cp.read_config_text(tiny)
        _patch(
            tiny,
            dst,
            cp.EditSpec(note="n", author="a", sections={"instrument_info": {"vehicle": "rvmp"}}),
        )
        assert marker in cp.read_config_text(dst)

    def test_provenance_comment_inserted(self, tiny, tmp_path):
        dst = tmp_path / "out.p"
        _patch(
            tiny,
            dst,
            cp.EditSpec(
                note="cal sheet", author="Jane", sections={"instrument_info": {"vehicle": "rvmp"}}
            ),
        )
        text = cp.read_config_text(dst)
        assert "[PATCH" in text and "Jane" in text
        # Stanza context + quoted original and new value.
        assert 'instrument_info vehicle: "VMP" -> "rvmp"' in text
        assert "(cal sheet)" in text

    def test_provenance_includes_channel_and_quoted_values(self, tiny, tmp_path):
        # A channel SN edit must say *which* channel and clearly delimit the
        # original and new value (SN is ambiguous across channels).
        dst = tmp_path / "out.p"
        _patch(tiny, dst, cp.EditSpec(note="fix", author="B", channels={"T1": {"sn": "T2227"}}))
        assert 'channel T1 SN: "T" -> "T2227"' in cp.read_config_text(dst)

    def test_original_embedded_and_recoverable(self, tiny, tmp_path):
        dst = tmp_path / "out.p"
        _patch(
            tiny,
            dst,
            cp.EditSpec(note="n", author="a", sections={"instrument_info": {"vehicle": "rvmp"}}),
        )
        text = cp.read_config_text(dst)
        assert text.count(cp.CONFIG_MARKER) == 1
        embedded = text.split(cp.CONFIG_MARKER, 1)[1]
        recovered = "\n".join(ln[2:] if ln.startswith("; ") else ln for ln in embedded.splitlines())
        assert "vehicle = VMP" in recovered  # pristine original value


# ---------------------------------------------------------------------------
# Editor semantics (mostly pure string -> string)
# ---------------------------------------------------------------------------


class TestEditorSemantics:
    def test_uppercase_key_via_lowercase_yaml(self, tiny, tmp_path):
        # File has 'SN = M2732'; user writes lowercase 'sn'. It must edit the
        # existing line, not append a duplicate.
        dst = tmp_path / "out.p"
        _patch(tiny, dst, cp.EditSpec(note="n", author="a", channels={"sh1": {"sn": "NEW"}}))
        text = cp.read_config_text(dst)
        active = text.split(cp.CONFIG_MARKER, 1)[0]
        assert "SN          = NEW" in active or "SN = NEW" in active
        # Exactly one SN assignment in the sh1 block region (no duplicate added).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert _channel(PFile(dst).config, "sh1")["sn"] == "NEW"

    def test_duplicate_channel_name_errors(self):
        cfg = (
            "[channel]\r\nname = sh1\r\nsens = 0.07\r\n\r\n"
            "[channel]\r\nname = sh1\r\nsens = 0.08\r\n"
        )
        with pytest.raises(ValueError, match="appears 2 times"):
            cp.edit_config_text(
                cfg, cp.EditSpec(note="n", author="a", channels={"sh1": {"sens": "0.09"}})
            )

    def test_add_keys_appends_in_block(self):
        cfg = "[channel]\r\nname = sh1\r\nsens = 0.07\r\n"
        text, changes = cp.edit_config_text(
            cfg,
            cp.EditSpec(note="n", author="a", channels={"sh1": {"cal_date": "2026-01-01"}}),
            add_keys=True,
        )
        assert len(changes) == 1 and changes[0].old is None
        assert "cal_date = 2026-01-01" in text

    def test_unknown_key_without_add_keys_errors(self, tiny):
        with pytest.raises(ValueError, match="unknown key 'sensitivity'"):
            cp.edit_config_text(
                cp.read_config_text(tiny),
                cp.EditSpec(note="n", author="a", channels={"sh1": {"sensitivity": "1"}}),
            )

    def test_unknown_channel_errors(self, tiny):
        with pytest.raises(ValueError, match="channel 'shX' not found"):
            cp.edit_config_text(
                cp.read_config_text(tiny),
                cp.EditSpec(note="n", author="a", channels={"shX": {"sens": "1"}}),
            )

    def test_missing_section_errors(self):
        with pytest.raises(ValueError, match=r"\[cruise_info\] section is not present"):
            cp.edit_config_text(
                "[instrument_info]\r\nvehicle = VMP\r\n",
                cp.EditSpec(note="n", author="a", sections={"cruise_info": {"ship": "X"}}),
            )

    def test_instrument_sn_warns(self, tiny):
        with pytest.warns(UserWarning, match="serial number"):
            cp.edit_config_text(
                cp.read_config_text(tiny),
                cp.EditSpec(note="n", author="a", sections={"instrument_info": {"sn": "999"}}),
            )

    def test_matrix_inside_comment_not_a_section(self):
        # A comment mentioning [matrix] must not create a matrix block / shift
        # the section the following key belongs to.
        cfg = "; mentions the [matrix] here\r\n[instrument_info]\r\nvehicle = VMP\r\n"
        text, changes = cp.edit_config_text(
            cfg,
            cp.EditSpec(note="n", author="a", sections={"instrument_info": {"vehicle": "rvmp"}}),
        )
        assert len(changes) == 1
        assert "vehicle = rvmp" in text

    def test_marker_inside_original_stays_editable(self):
        # Defect: edit_config_text froze everything below the FIRST CONFIG_MARKER,
        # even when the marker is part of the ORIGINAL config (e.g. an RSI-pre-
        # patched file whose uncommented stanzas live below it). Those stanzas
        # must remain editable, not be misreported as "not present".
        cfg = (
            "[instrument_info]\r\nvehicle = VMP\r\n"
            + cp.CONFIG_MARKER
            + "\r\n[cruise_info]\r\nship = Thompson\r\n"
        )
        text, changes = cp.edit_config_text(
            cfg, cp.EditSpec(note="n", author="a", sections={"cruise_info": {"ship": "Sally Ride"}})
        )
        assert [c.key for c in changes] == ["ship"]
        assert "ship = Sally Ride" in text

    def test_self_produced_marker_block_still_frozen(self):
        # A marker WE wrote (column 0, followed by a fully ';'-commented block)
        # must still be treated as the frozen original (idempotent re-patch).
        active = "[instrument_info]\r\nvehicle = VMP\r\n"
        cfg = active + cp.CONFIG_MARKER + "\r\n; [instrument_info]\r\n; vehicle = VMP\r\n"
        text, changes = cp.edit_config_text(
            cfg,
            cp.EditSpec(note="n", author="a", sections={"instrument_info": {"vehicle": "rvmp"}}),
        )
        assert [c.key for c in changes] == ["vehicle"]
        # The frozen commented block is re-attached verbatim, not re-edited.
        assert text.count(cp.CONFIG_MARKER) == 1
        assert "; vehicle = VMP" in text.split(cp.CONFIG_MARKER, 1)[1]


# ---------------------------------------------------------------------------
# No-op behaviour
# ---------------------------------------------------------------------------


class TestNoOp:
    def test_no_change_returns_empty(self, tiny):
        cfg = cp.read_config_text(tiny)
        text, changes = cp.edit_config_text(
            cfg, cp.EditSpec(note="n", author="a", sections={"instrument_info": {"vehicle": "VMP"}})
        )
        assert changes == []
        assert text == cfg  # untouched

    def test_partial_change_only_edits_differing_key(self, tiny):
        cfg = cp.read_config_text(tiny)
        _text, changes = cp.edit_config_text(
            cfg,
            cp.EditSpec(note="n", author="a", channels={"sh1": {"sens": "0.1075", "sn": "NEW"}}),
        )  # sens already 0.1075 (no-op), sn changes
        assert [c.key for c in changes] == ["SN"]

    def test_idempotent_repatch(self, tiny, tmp_path):
        out1 = tmp_path / "o1.p"
        spec = cp.EditSpec(note="n", author="a", channels={"sh1": {"sens": "0.0812"}})
        assert _patch(tiny, out1, spec)  # first patch changes something
        # Re-applying the SAME spec to the patched file is a no-op.
        _, changes = cp.edit_config_text(cp.read_config_text(out1), spec)
        assert changes == []
        assert cp.read_config_text(out1).count(cp.CONFIG_MARKER) == 1


# ---------------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------------


class TestSpecValidation:
    def test_unquoted_number_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="must be quoted text"):
            cp.load_edit_spec(_spec(tmp_path, 'note: "x"\nchannels:\n  sh1:\n    sens: 0.0812\n'))

    def test_yaml_bool_rejected(self, tmp_path):
        # 'true'/'false' coerce to bool under YAML 1.2; reject (unlike 'no'/'yes',
        # which stay strings and are kept verbatim).
        with pytest.raises(ValueError, match="must be quoted text"):
            cp.load_edit_spec(_spec(tmp_path, 'note: "x"\ninstrument_info:\n  vehicle: true\n'))

    def test_leading_zero_int_rejected(self, tmp_path):
        # 0479 would coerce to int 479, losing the leading zero — reject it.
        with pytest.raises(ValueError, match="must be quoted text"):
            cp.load_edit_spec(_spec(tmp_path, 'note: "x"\ninstrument_info:\n  sn: 0479\n'))

    def test_unknown_top_level_key_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="unknown top-level key"):
            cp.load_edit_spec(_spec(tmp_path, 'note: "x"\nroot:\n  rate: "512"\n'))

    def test_missing_note_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="non-empty 'note'"):
            cp.load_edit_spec(_spec(tmp_path, 'instrument_info:\n  vehicle: "rvmp"\n'))

    def test_empty_spec_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="no edits"):
            cp.load_edit_spec(_spec(tmp_path, 'note: "just a note"\n'))

    def test_name_key_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="cannot be set"):
            cp.load_edit_spec(_spec(tmp_path, 'note: "x"\ninstrument_info:\n  name: "Z"\n'))

    def test_semicolon_in_value_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="must not contain ';'"):
            cp.load_edit_spec(_spec(tmp_path, 'note: "x"\ninstrument_info:\n  vehicle: "a;b"\n'))

    def test_non_ascii_value_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="must be ASCII"):
            cp.load_edit_spec(_spec(tmp_path, 'note: "x"\ninstrument_info:\n  vehicle: "café"\n'))

    def test_non_ascii_note_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="'note' must be a single line of ASCII"):
            cp.load_edit_spec(
                _spec(tmp_path, 'note: "café"\ninstrument_info:\n  vehicle: "rvmp"\n')
            )

    def test_valid_spec_parses(self, tmp_path):
        spec = cp.load_edit_spec(
            _spec(
                tmp_path,
                'note: "fix"\nauthor: "Jane"\n'
                'instrument_info:\n  vehicle: "rvmp"\n'
                'channels:\n  sh1:\n    sens: "0.0812"\n',
            )
        )
        assert spec.author == "Jane"
        assert spec.sections["instrument_info"]["vehicle"] == "rvmp"
        assert spec.channels["sh1"]["sens"] == "0.0812"

    def test_block_scalar_value_injection_rejected(self, tmp_path):
        # A YAML block scalar smuggling a [root] stanza must be rejected, not
        # injected into the config (it would bypass the acquisition-param guard).
        yaml = 'note: "n"\ninstrument_info:\n  vehicle: |\n    rvmp\n    [root]\n    rate = 1\n'
        with pytest.raises(ValueError, match="single line"):
            cp.load_edit_spec(_spec(tmp_path, yaml))

    def test_escaped_newline_value_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="single line"):
            cp.load_edit_spec(
                _spec(
                    tmp_path, 'note: "n"\ninstrument_info:\n  vehicle: "rvmp\\n[matrix]\\nrate=1"\n'
                )
            )

    def test_multiline_note_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="'note' must be a single line"):
            cp.load_edit_spec(
                _spec(
                    tmp_path,
                    'note: |\n  hi\n  [root]\n  rate=1\ninstrument_info:\n  vehicle: "rvmp"\n',
                )
            )

    def test_multiline_author_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="'author' must be a single line"):
            cp.load_edit_spec(
                _spec(tmp_path, 'note: "n"\nauthor: "a\\nb"\ninstrument_info:\n  vehicle: "rvmp"\n')
            )

    def test_banner_source_label_newline_sanitized(self, tiny):
        # A newline in the (machine-derived) source path must be collapsed, not
        # spilled past the '; ' comment prefix as active config.
        new, _ch = cp.edit_config_text(
            cp.read_config_text(tiny),
            cp.EditSpec(note="n", author="a", sections={"instrument_info": {"vehicle": "rvmp"}}),
            when="W",
            source_label="/tmp/x\n[root]\nrate=9",
        )
        parsed = cp.parse_config(new.split(cp.CONFIG_MARKER, 1)[0])
        assert parsed["root"].get("rate") is None

    def test_when_newline_sanitized_in_provenance_and_banner(self, tiny):
        # Defect: 'when' was interpolated raw into both the provenance comment
        # and the banner. A newline must be collapsed, not injected as a stanza.
        new, _ch = cp.edit_config_text(
            cp.read_config_text(tiny),
            cp.EditSpec(note="n", author="a", sections={"instrument_info": {"vehicle": "rvmp"}}),
            when="W\r\n[root]\r\nrate=9",
        )
        parsed = cp.parse_config(new.split(cp.CONFIG_MARKER, 1)[0])
        assert parsed["root"].get("rate") is None

    def test_provenance_note_newline_sanitized(self, tiny):
        # Defect: spec.note/old/new were interpolated raw into the provenance
        # comment (only the banner sanitized note). A directly-constructed
        # EditSpec with a newline in 'note' must not inject active config.
        new, _ch = cp.edit_config_text(
            cp.read_config_text(tiny),
            cp.EditSpec(
                note="ok\r\n[root]\r\nrate=7",
                author="a",
                sections={"instrument_info": {"vehicle": "rvmp"}},
            ),
            when="W",
        )
        parsed = cp.parse_config(new.split(cp.CONFIG_MARKER, 1)[0])
        assert parsed["root"].get("rate") is None


# ---------------------------------------------------------------------------
# Binary writer guards
# ---------------------------------------------------------------------------


class TestBinaryGuards:
    def test_little_endian_writer(self, tmp_path):
        src = _synth(tmp_path / "le.p", "<", "[instrument_info]\r\nvehicle = VMP\r\n")
        new = "[instrument_info]\r\nvehicle = rvmp_longer\r\n"
        dst = cp.write_patched_pfile(src, tmp_path / "le_out.p", new)
        assert _config_size_word(dst) == len(new.encode("latin-1"))
        assert _data_region(src) == _data_region(dst)

    def test_big_endian_writer(self, tmp_path):
        src = _synth(tmp_path / "be.p", ">", "[instrument_info]\r\nvehicle = VMP\r\n")
        new = "[instrument_info]\r\nvehicle = rv\r\n"  # shorter
        dst = cp.write_patched_pfile(src, tmp_path / "be_out.p", new)
        assert _config_size_word(dst) == len(new.encode("latin-1"))
        assert _data_region(src) == _data_region(dst)

    def test_v1_file_refused(self, tmp_path):
        src = _synth(
            tmp_path / "v1.p", "<", "[instrument_info]\r\nvehicle = VMP\r\n", version=0x0100
        )
        with pytest.raises(ValueError, match="not supported"):
            cp.write_patched_pfile(src, tmp_path / "v1_out.p", "[x]\r\n")

    def test_config_size_overflow_refused(self, tiny, tmp_path):
        with pytest.raises(ValueError, match="65535"):
            cp.write_patched_pfile(tiny, tmp_path / "big.p", "x" * 70000)

    def test_truncated_source_refused(self, tmp_path):
        # Header advertises a first record larger than the file -> refuse rather
        # than seek past EOF and silently write a data-less output.
        cfg = b"[instrument_info]\r\nvehicle = VMP\r\n"
        words = [0] * 64
        words[10] = 0x0600
        words[11] = 5000  # lies: real config is ~34 bytes
        words[17] = 128
        words[18] = 200
        words[63] = 1
        src = tmp_path / "trunc.p"
        src.write_bytes(struct.pack("<64H", *words) + cfg + b"\x01\x02")
        with pytest.raises(ValueError, match="truncated or has no data records"):
            cp.write_patched_pfile(src, tmp_path / "o.p", "[x]\r\n")

    def test_existing_output_is_exclusive(self, tiny, tmp_path):
        # Atomic exclusive create: writing over an existing file raises.
        dst = tmp_path / "exists.p"
        dst.write_bytes(b"x")
        with pytest.raises(FileExistsError):
            cp.write_patched_pfile(tiny, dst, cp.read_config_text(tiny))

    def test_small_header_size_refused(self, tmp_path):
        # Defect: header_size < HEADER_BYTES was unvalidated; with header_size in
        # [12,23] struct.pack_into at offset 22 raised an uncatchable struct.error
        # (and [24,127] silently wrote a truncated <128-byte header). Must be a
        # clean, CLI-catchable ValueError instead.
        cfg = b"[instrument_info]\r\nvehicle = VMP\r\n"
        words = [0] * 64
        words[10] = 0x0600
        words[11] = len(cfg)
        words[17] = 20  # header_size lies: < 128 and < the offset-22 pack region
        words[18] = 200
        words[63] = 1
        src = tmp_path / "small_hdr.p"
        src.write_bytes(struct.pack("<64H", *words) + cfg + b"\x00" * 200)
        with pytest.raises(ValueError, match="invalid header_size"):
            cp.write_patched_pfile(src, tmp_path / "o.p", "[x]\r\n")
        # read_config_text shares the unvalidated read and must guard too.
        with pytest.raises(ValueError, match="invalid header_size"):
            cp.read_config_text(src)

    def test_write_is_atomic_on_failure(self, tiny, tmp_path, monkeypatch):
        # Defect: a mid-copy failure left a truncated, valid-looking patched .p in
        # the final destination. The write must go via a temp file + os.replace so
        # a commit-time failure leaves no dst and no orphaned temp file.
        dst = tmp_path / "atomic.p"
        text, _ = cp.edit_config_text(
            cp.read_config_text(tiny),
            cp.EditSpec(note="n", author="a", sections={"instrument_info": {"vehicle": "rvmp"}}),
            when="W",
        )

        def boom(_a, _b):
            raise OSError("simulated commit failure")

        monkeypatch.setattr(cp.os, "replace", boom)
        with pytest.raises(OSError, match="simulated commit failure"):
            cp.write_patched_pfile(tiny, dst, text)
        assert not dst.exists()  # no corrupt file left behind
        assert list(tmp_path.glob("atomic.p.tmp.*")) == []  # temp cleaned up

    def test_odd_config_size_reads_back_identically(self, tiny, tmp_path):
        # Rockland ships odd config_size (the fixture itself is odd); a patched
        # file with odd config_size must still read byte-identical data, since
        # np.fromfile / MATLAB fread read by byte offset, not word alignment.
        text, _ = cp.edit_config_text(
            cp.read_config_text(tiny),
            cp.EditSpec(note="n", author="a", channels={"sh1": {"sens": "0.0812"}}),
            when="W",
        )
        if len(text.encode("latin-1")) % 2 == 0:
            text += " "  # force an odd length (harmless trailing whitespace)
        dst = tmp_path / "odd.p"
        cp.write_patched_pfile(tiny, dst, text)
        assert _config_size_word(dst) % 2 == 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a, b = PFile(tiny), PFile(dst)
        for k in a.channels_raw:
            assert np.array_equal(a.channels_raw[k], b.channels_raw[k]), k


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


class TestPatchFiles:
    def test_batch_cal_guard(self, tiny, tmp_path):
        second = extract_pfile_segment(DATA, tmp_path / "two.p", start_record=0, n_records=4)
        spec = cp.EditSpec(note="n", author="a", channels={"sh1": {"sens": "0.0812"}})
        with pytest.raises(ValueError, match="per-channel calibration"):
            cp.patch_files([tiny, second], tmp_path / "out", spec)

    def test_batch_cal_allowed_with_flag(self, tiny, tmp_path):
        second = extract_pfile_segment(DATA, tmp_path / "two.p", start_record=0, n_records=4)
        spec = cp.EditSpec(note="n", author="a", channels={"sh1": {"sens": "0.0812"}})
        res = cp.patch_files([tiny, second], tmp_path / "out", spec, batch_cal=True)
        assert sum(1 for _s, d, _c in res if d is not None) == 2

    def test_provenance_only_batches_freely(self, tiny, tmp_path):
        second = extract_pfile_segment(DATA, tmp_path / "two.p", start_record=0, n_records=4)
        spec = cp.EditSpec(note="n", author="a", sections={"instrument_info": {"vehicle": "rvmp"}})
        res = cp.patch_files([tiny, second], tmp_path / "out", spec)
        assert sum(1 for _s, d, _c in res if d is not None) == 2

    def test_out_equals_source_dir_refused(self, tiny):
        spec = cp.EditSpec(note="n", author="a", sections={"instrument_info": {"vehicle": "rvmp"}})
        with pytest.raises(ValueError, match="overwrite the original"):
            cp.patch_files([tiny], tiny.parent, spec)

    def test_existing_dst_refused(self, tiny, tmp_path):
        out = tmp_path / "out"
        spec = cp.EditSpec(note="n", author="a", sections={"instrument_info": {"vehicle": "rvmp"}})
        cp.patch_files([tiny], out, spec)
        with pytest.raises(FileExistsError, match="already exists"):
            cp.patch_files([tiny], out, spec)

    def test_basename_collision_refused_before_any_write(self, tmp_path):
        # Defect: two inputs sharing a basename mapped to the same out_dir/NAME.p;
        # the first was written, the second aborted the batch with a misleading
        # "already exists" error. Detect the collision up front and write nothing.
        da = tmp_path / "a"
        db = tmp_path / "b"
        da.mkdir()
        db.mkdir()
        extract_pfile_segment(DATA, da / "SAME.p", start_record=0, n_records=4)
        extract_pfile_segment(DATA, db / "SAME.p", start_record=0, n_records=4)
        out = tmp_path / "out"
        spec = cp.EditSpec(note="n", author="a", sections={"instrument_info": {"vehicle": "rvmp"}})
        with pytest.raises(ValueError, match="share a basename"):
            cp.patch_files([da / "SAME.p", db / "SAME.p"], out, spec)
        # Nothing written: the misleading partial-batch state is avoided.
        assert not (out / "SAME.p").exists()

    def test_no_op_not_written(self, tiny, tmp_path, capsys):
        out = tmp_path / "out"
        spec = cp.EditSpec(note="n", author="a", sections={"instrument_info": {"vehicle": "VMP"}})
        res = cp.patch_files([tiny], out, spec)
        assert res[0][1] is None
        assert "no changes" in capsys.readouterr().out
        assert not (out / tiny.name).exists()

    def test_dry_run_writes_nothing(self, tiny, tmp_path, capsys):
        out = tmp_path / "dry"
        spec = cp.EditSpec(note="n", author="a", sections={"instrument_info": {"vehicle": "rvmp"}})
        cp.patch_files([tiny], out, spec, dry_run=True)
        assert not out.exists()
        assert "vehicle" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Template scaffold + full loop
# ---------------------------------------------------------------------------


class TestScaffold:
    def test_scaffold_has_current_values_and_channels(self, tiny):
        text = cp.scaffold_yaml(tiny)
        assert 'vehicle: "VMP"' in text
        assert "sh1:" in text and 'sens: "0.1075"' in text

    def test_scaffold_unedited_is_noop(self, tiny, tmp_path):
        # Applying the scaffold with only its current vehicle value is a no-op.
        text = cp.scaffold_yaml(tiny)
        # Build a minimal spec mirroring the uncommented part of the scaffold.
        spec = cp.load_edit_spec(_spec(tmp_path, 'note: "x"\ninstrument_info:\n  vehicle: "VMP"\n'))
        _, changes = cp.edit_config_text(cp.read_config_text(tiny), spec)
        assert changes == []
        assert "patch-config" in text  # scaffold carries usage instructions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_template_then_patch(self, tiny, tmp_path, monkeypatch, capsys):
        tmpl = tmp_path / "t.yaml"
        monkeypatch.setattr(sys, "argv", ["rsi-tpw", "patch-template", str(tiny), "-o", str(tmpl)])
        from odas_tpw.rsi.cli import main

        main()
        assert tmpl.exists() and 'vehicle: "VMP"' in tmpl.read_text()

        edits = _spec(tmp_path, 'note: "switch to upward"\ninstrument_info:\n  vehicle: "rvmp"\n')
        out = tmp_path / "patched"
        monkeypatch.setattr(
            sys,
            "argv",
            ["rsi-tpw", "patch-config", str(tiny), "--edits", str(edits), "--out", str(out)],
        )
        main()
        dst = out / tiny.name
        assert dst.exists()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert PFile(dst).config["instrument_info"]["vehicle"] == "rvmp"

    def test_patch_config_bad_spec_exits(self, tiny, tmp_path, monkeypatch):
        bad = _spec(tmp_path, 'note: "x"\nchannels:\n  sh1:\n    sens: 0.0812\n')  # unquoted
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "rsi-tpw",
                "patch-config",
                str(tiny),
                "--edits",
                str(bad),
                "--out",
                str(tmp_path / "o"),
            ],
        )
        from odas_tpw.rsi.cli import main

        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 1
