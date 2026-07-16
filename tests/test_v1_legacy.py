# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for the ODAS header-version-1 legacy reader / v1->v6 translator (#141).

Covers: PFile version dispatch (v6 untouched, v1 translated in memory, loud
refusal otherwise), the old-dialect setup parser, INI synthesis round-trip,
byte-level translation vs the vendor convert_header contract, sbt/sbc
converters, the shear-sens hard error, the type-based sbt reference-
temperature candidacy, the v1 refusal guards in the six positional readers,
and golden per-channel hashes pinning the v6 path (plan delta F11).
"""

import hashlib
import struct
import sys
import warnings
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from odas_tpw.rsi.channels import convert_sbc, convert_sbt, convert_shear
from odas_tpw.rsi.p_file import (
    _H,
    HEADER_BYTES,
    HEADER_WORDS,
    PFile,
    extract_pfile_segment,
    parse_config,
    read_config_string,
)
from odas_tpw.rsi.setup_v1 import (
    discover_setup_candidates,
    load_setup_file,
    parse_setup_v1,
)
from odas_tpw.rsi.v1_translate import (
    synthesize_ini,
    translate_v1_bytes,
    translate_v1_to_v6,
)

DATA = Path(__file__).parent / "data"

# ---------------------------------------------------------------------------
# Synthetic v1 file builder
# ---------------------------------------------------------------------------

# 4x4 matrix exercising: sp_char, Sea-Bird pair (16/17), pressure, shear
# (fast column), T1_dT1 (fast column), T1 double-slot (decimate-to-slow, F3),
# and gnd double-slot (warning-exempt).
V1_MATRIX = [
    [255, 16, 8, 5],
    [10, 17, 8, 5],
    [4, 0, 8, 5],
    [4, 0, 8, 5],
]
V1_RECORD_SIZE = 128 + 2 * 32  # 16 scans... (32 words = 8 scans of 4 cols)
V1_SCANS_PER_RECORD = (V1_RECORD_SIZE - HEADER_BYTES) // 2 // 4  # = 8 -> 2 cycles

SETUP_TEXT = """\
# synthetic v1 setup file (old dialect)
rate:     512
recsize:  1
no-fast:  2
no-slow:  2
profile:  vertical
prefix:   SYN_
channel: 10,Pres,4.46,0.049994,1.341e-8,0
channel: 16,SBT1E,4.36461798e-3,6.35619211e-4,2.12747658e-5,1.87173415e-6,1000.0,24e6,128
channel: 17,SBT1O,4.36461798e-3,6.35619211e-4,2.12747658e-5,1.87173415e-6,1000.0,24e6,128
sh1_sens: 0.0893
matrix: 255 16 8 5
matrix: 10 17 8 5
matrix: 4 0 8 5
matrix: 4 0 8 5
"""

SBT_COEFS = (4.36461798e-3, 6.35619211e-4, 2.12747658e-5, 1.87173415e-6, 1000.0, 24e6, 128)


def _v1_header(
    rec_no: int,
    *,
    endian: str = "<",
    version: int = 1,
    bad_buffer: bool = False,
) -> bytes:
    words = [0] * HEADER_WORDS
    words[_H["file_number"]] = 7
    words[_H["record_number"]] = rec_no
    words[_H["year"]], words[_H["month"]], words[_H["day"]] = 2013, 5, 20
    words[_H["hour"]], words[_H["minute"]], words[_H["second"]] = 3, 34, 40
    words[_H["millisecond"]] = 187
    words[_H["header_version"]] = version
    words[_H["config_size"]] = 0
    words[_H["buffer_status"]] = 1 if bad_buffer else 0
    words[_H["header_size"]] = HEADER_BYTES
    words[_H["record_size"]] = V1_RECORD_SIZE
    words[_H["n_records_written"]] = rec_no
    words[_H["clock_hz"]] = 2048
    words[_H["clock_frac"]] = 0
    words[_H["fast_cols"]] = 2
    words[_H["slow_cols"]] = 2
    words[_H["n_rows"]] = 4
    words[_H["endian"]] = 1 if endian == "<" else 2
    return struct.pack(f"{endian}{HEADER_WORDS}H", *words)


def make_v1_file(
    path: Path,
    *,
    n_data_records: int = 3,
    endian: str = "<",
    version: int = 1,
    bad_buffer_record: int | None = None,
    extra_tail_bytes: int = 0,
) -> Path:
    """Write a synthetic v1 .p file with deterministic per-address samples."""
    out = bytearray()
    out += _v1_header(0, endian=endian, version=version)
    m = np.array(V1_MATRIX, dtype=endian + "u2").ravel()
    block = bytearray(V1_RECORD_SIZE - HEADER_BYTES)
    block[: m.nbytes] = m.tobytes()
    out += block
    scan = 0
    for rec in range(1, n_data_records + 1):
        out += _v1_header(rec, endian=endian, bad_buffer=(rec == bad_buffer_record))
        words = []
        for _ in range(V1_SCANS_PER_RECORD):
            cyc, r = divmod(scan, 4)
            sh1 = 100 + scan  # fast: one sample per scan row
            t1d = -50 + scan
            if r == 0:
                # sbt low word: (10 << 16) | (29640 + cyc) ~ 685000 counts
                # -> f ~ 4.48 kHz -> ~16.5 degC (plausible ocean T, passes QC)
                words += [32752, 29640 + cyc, sh1, t1d]
            elif r == 1:
                words += [2000 + cyc, 10, sh1, t1d]  # P counts; sbt high word
            else:
                words += [7000 + 10 * cyc + r, 11 + r, sh1, t1d]  # T1 slots; gnd
            scan += 1
        out += np.array(words, dtype=endian + "i2").tobytes()
    if extra_tail_bytes:
        out += b"\x00" * extra_tail_bytes
    path.write_bytes(bytes(out))
    return path


@pytest.fixture
def v1_dir(tmp_path):
    """tmp dir holding a synthetic v1 file + its setup.txt sibling."""
    (tmp_path / "setup.txt").write_text(SETUP_TEXT)
    make_v1_file(tmp_path / "SYN_001.p")
    return tmp_path


def _expected_channels(n_cycles: int) -> dict[str, np.ndarray]:
    """Hand-computed physical values for the synthetic file's channels."""
    P_counts = 2000 + np.arange(n_cycles)
    w32 = (10 << 16) | (29640 + np.arange(n_cycles))
    g, h, i, j, f0, f_ref, n_per = SBT_COEFS
    f = n_per * f_ref / w32
    x = np.log(f0 / f)
    sbt = 1.0 / (g + h * x + i * x**2 + j * x**3) - 273.15
    sh_counts = 100 + np.arange(4 * n_cycles)
    return {
        "sp_char": np.full(n_cycles, 32752.0),
        "P": 4.46 + 0.049994 * P_counts + 1.341e-8 * P_counts**2,
        "SBT1": sbt,
        "sh1": (5.0 / 2**16) * sh_counts / (2 * np.sqrt(2) * 1.0 * 0.0893),
        "T1_dT1": -50.0 + np.arange(4 * n_cycles),
        # T1 occupies slow rows 3 and 4 (0-based 2, 3); extraction keeps the
        # FIRST occurrence (row index 2) per cycle -> 7000 + 10*cyc + 2.
        "T1": 7000.0 + 10 * np.arange(n_cycles) + 2,
        # gnd: first occurrence is (row 2, col 1) 0-based -> 11 + r with r=2.
        "Gnd": np.full(n_cycles, 13.0),
    }


# ---------------------------------------------------------------------------
# Version dispatch (plan delta F1)
# ---------------------------------------------------------------------------


class TestVersionDispatch:
    def test_v1_file_reads_via_translation(self, v1_dir):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pf = PFile(v1_dir / "SYN_001.p")
        assert pf.translated_from_v1 is True
        assert pf.setup_file_source == str(v1_dir / "setup.txt")
        assert pf.fs_fast == pytest.approx(512.0)
        assert pf.fs_slow == pytest.approx(128.0)
        n_cycles = 3 * V1_SCANS_PER_RECORD // 4
        for name, want in _expected_channels(n_cycles).items():
            np.testing.assert_allclose(pf.channels[name], want, rtol=0, atol=1e-9, err_msg=name)

    def test_v1_decimate_to_slow_warns(self, v1_dir):
        """T1 occupies 2 slow slots/cycle; the k-occurrence path warns (F3)."""
        with pytest.warns(UserWarning, match="'T1' .*appears 2x per scan"):
            PFile(v1_dir / "SYN_001.p")

    def test_v6_fixture_untouched(self):
        pf = PFile(DATA / "SN479_0006.p")
        assert pf.translated_from_v1 is False
        assert pf.setup_file_source is None

    @pytest.mark.parametrize("version", [2, 3, 0x0300, 5, 0x0500])
    def test_unsupported_versions_refused(self, tmp_path, version):
        p = make_v1_file(tmp_path / "bad.p", version=version)
        with pytest.raises(ValueError, match="unsupported ODAS header version"):
            PFile(p)
        # The message names the raw word AND the decoded major.minor.
        with pytest.raises(ValueError, match=rf"word 11 = {version} = v{version >> 8}\."):
            PFile(p)

    def test_v1_without_setup_file_errors(self, tmp_path):
        p = make_v1_file(tmp_path / "orphan.p")
        with pytest.raises(ValueError, match="needs an external setup file"):
            PFile(p)

    def test_explicit_setup_file_kwarg(self, v1_dir, tmp_path):
        alt = tmp_path / "elsewhere" / "mysetup.txt"
        alt.parent.mkdir()
        alt.write_text(SETUP_TEXT.replace("0.0893", "0.1000"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pf = PFile(v1_dir / "SYN_001.p", setup_file=alt)
        assert pf.setup_file_source == str(alt)
        n_cycles = 3 * V1_SCANS_PER_RECORD // 4
        sh_counts = 100 + np.arange(4 * n_cycles)
        np.testing.assert_allclose(
            pf.channels["sh1"], (5.0 / 2**16) * sh_counts / (2 * np.sqrt(2) * 0.1), rtol=1e-12
        )

    def test_missing_sens_is_loud_error(self, tmp_path):
        """A v1 shear channel with no sens anywhere errors, never defaults (F2)."""
        (tmp_path / "setup.txt").write_text(SETUP_TEXT.replace("sh1_sens: 0.0893\n", ""))
        p = make_v1_file(tmp_path / "SYN_001.p")
        with (
            pytest.raises(ValueError, match=r"'sens' missing"),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore")
            PFile(p)

    def test_bad_buffer_warning_parity(self, v1_dir):
        """Header word 16 flags survive translation and warn like on v6 files."""
        make_v1_file(v1_dir / "SYN_002.p", bad_buffer_record=2)
        with pytest.warns(UserWarning, match="1/3 records.*bad-buffer"):
            PFile(v1_dir / "SYN_002.p")

    def test_partial_trailing_record_warns(self, v1_dir):
        make_v1_file(v1_dir / "SYN_003.p", extra_tail_bytes=10)
        with pytest.warns(UserWarning, match="not an integer number of records"):
            pf = PFile(v1_dir / "SYN_003.p")
        assert len(pf.t_slow) == 3 * V1_SCANS_PER_RECORD // 4

    def test_big_endian_v1_warns_but_reads(self, tmp_path):
        (tmp_path / "setup.txt").write_text(SETUP_TEXT)
        p = make_v1_file(tmp_path / "SYN_BE.p", endian=">")
        with pytest.warns(UserWarning, match="big-endian"):
            pf = PFile(p)
        n_cycles = 3 * V1_SCANS_PER_RECORD // 4
        np.testing.assert_allclose(
            pf.channels["P"], _expected_channels(n_cycles)["P"], rtol=0, atol=1e-9
        )

    def test_summary_notes_translation(self, v1_dir, capsys):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            PFile(v1_dir / "SYN_001.p").summary()
        out = capsys.readouterr().out
        assert "translated to v6 in memory" in out
        assert "setup.txt" in out


# ---------------------------------------------------------------------------
# Old-dialect setup parser
# ---------------------------------------------------------------------------


class TestParseSetupV1:
    def test_corpus_shape(self):
        cfg = parse_setup_v1(SETUP_TEXT)
        assert cfg["root"]["rate"] == "512"
        assert cfg["root"]["recsize"] == "1"
        assert cfg["matrix"] == V1_MATRIX
        assert cfg["instrument_info"] == {"vehicle": "vmp"}
        by_name = {c["name"]: c for c in cfg["channels"]}
        assert by_name["P"]["type"] == "poly"
        assert by_name["P"]["coef0"] == "4.46"
        assert by_name["P"]["coef2"] == "1.341e-8"
        assert "coef3" not in by_name["P"]  # trailing zero stripped
        assert by_name["P"]["units"] == "[dBar]"
        assert by_name["SBT1"]["type"] == "sbt"
        assert by_name["SBT1"]["id"] == "16,17"
        assert by_name["SBT1"]["coef4"] == "1000.0"
        assert by_name["SBT1"]["coef6"] == "128"
        assert by_name["sh1"] == {
            "id": "8",
            "name": "sh1",
            "type": "shear",
            "adc_fs": "5.0",
            "adc_bits": "16",
            "diff_gain": "1.0",
            "sens": "0.0893",
        }
        # Thermistors stay raw counts (F8 policy).
        assert by_name["T1"]["type"] == "raw"
        assert by_name["T1_dT1"]["type"] == "raw"
        assert by_name["sp_char"]["type"] == "raw"
        assert by_name["Gnd"]["type"] == "gnd"

    def test_whitespace_and_comma_values(self):
        cfg = parse_setup_v1("rate: 512\nmatrix: 1\t2  3\nchannel: 1,Ax,10,20\n")
        assert cfg["matrix"] == [[1, 2, 3]]
        ax = cfg["channels"][0]  # addresses sorted: 1 (Ax), 2, 3
        assert (ax["name"], ax["coef0"], ax["coef1"]) == ("Ax", "10", "20")

    def test_comment_and_blank_lines_skipped(self):
        cfg = parse_setup_v1("# comment\n\nrate: 512\nmatrix: 0\n")
        assert cfg["root"] == {"rate": "512"}

    def test_no_matrix_raises(self):
        with pytest.raises(ValueError, match="no 'matrix:' rows"):
            parse_setup_v1("rate: 512\n")

    def test_sens_override_beats_extension_key(self):
        cfg = parse_setup_v1(SETUP_TEXT, sens_overrides={"sh1": 0.5})
        sh1 = next(c for c in cfg["channels"] if c["name"] == "sh1")
        assert float(sh1["sens"]) == 0.5

    def test_sens_override_unknown_channel_raises(self):
        with pytest.raises(ValueError, match="non-shear/unknown"):
            parse_setup_v1(SETUP_TEXT, sens_overrides={"sh9": 0.5})

    def test_unknown_address_kept_raw_with_warning(self):
        with pytest.warns(UserWarning, match="address 42"):
            cfg = parse_setup_v1("matrix: 42\n")
        assert cfg["channels"] == [{"id": "42", "name": "ch42", "type": "raw"}]

    def test_thermistor_coefficients_ignored_with_warning(self):
        with pytest.warns(UserWarning, match="thermistor coefficients.*ignored"):
            cfg = parse_setup_v1("matrix: 4\nchannel: 4,T1,-46,0.99856,11\n")
        assert cfg["channels"][0]["type"] == "raw"

    def test_instrument_extension_keys(self):
        cfg = parse_setup_v1("model: VMP-2000\nsn: 002\nmatrix: 0\n")
        assert cfg["instrument_info"]["model"] == "VMP-2000"
        assert cfg["instrument_info"]["sn"] == "002"
        assert cfg["instrument_info"]["vehicle"] == "vmp"

    def test_seabird_pair_needs_seven_coefficients(self):
        with pytest.raises(ValueError, match="needs 7 coefficients"):
            parse_setup_v1("matrix: 16 17\nchannel: 16,SBT1E,1,2,3\nchannel: 17,SBT1O,1,2,3\n")


# ---------------------------------------------------------------------------
# INI synthesis round-trip (pivot requirement)
# ---------------------------------------------------------------------------


class TestIniRoundTrip:
    def test_parse_config_reproduces_setup_dict(self):
        cfg = parse_setup_v1(SETUP_TEXT)
        ini = synthesize_ini(cfg, {"translated_from": "odas_v1"})
        back = parse_config(ini)
        assert back["matrix"] == cfg["matrix"]
        assert back["channels"] == cfg["channels"]
        assert back["instrument_info"] == cfg["instrument_info"]
        assert back["root"]["translated_from"] == "odas_v1"
        for k, v in cfg["root"].items():
            assert back["root"][k] == v

    def test_config_command_output_is_reparseable(self, v1_dir):
        """The synthesized INI embedded in a translated file parses cleanly."""
        dst, meta = translate_v1_to_v6(v1_dir / "SYN_001.p", v1_dir / "out" / "SYN_001.p")
        assert read_config_string(dst) == meta["config_str"]


# ---------------------------------------------------------------------------
# Byte-level translation (vendor convert_header contract)
# ---------------------------------------------------------------------------


class TestTranslator:
    def test_header_words_vendor_contract(self, v1_dir):
        src = v1_dir / "SYN_001.p"
        blob, meta = translate_v1_bytes(src)
        old = struct.unpack(f"<{HEADER_WORDS}H", src.read_bytes()[:HEADER_BYTES])
        new = struct.unpack(f"<{HEADER_WORDS}H", blob[:HEADER_BYTES])
        assert new[10] == 6 << 8  # header_version -> v6.0
        assert new[11] == len(meta["config_str"].encode("ascii"))  # config_size
        assert new[12] == 0  # product_id stays legacy
        assert new[13] == 0  # build number
        for idx in range(HEADER_WORDS):
            if idx not in (10, 11, 12, 13):
                assert new[idx] == old[idx], f"header word {idx + 1} changed"

    def test_data_records_verbatim(self, v1_dir):
        """All data records (headers included) are copied byte-for-byte —
        deliberate deviation from patch_setupstr.m, which clobbers record 1's
        header with a copy of record 0's."""
        src = v1_dir / "SYN_001.p"
        blob, _ = translate_v1_bytes(src)
        raw = src.read_bytes()
        config_size = struct.unpack_from("<H", blob, 11 * 2)[0]
        assert blob[HEADER_BYTES + config_size :] == raw[V1_RECORD_SIZE:]

    def test_disk_and_memory_routes_identical(self, v1_dir):
        src = v1_dir / "SYN_001.p"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pf_mem = PFile(src)
            dst, _ = translate_v1_to_v6(src, v1_dir / "out" / "SYN_001.p")
            pf_disk = PFile(dst)
        assert pf_disk.translated_from_v1 is False
        assert set(pf_mem.channels) == set(pf_disk.channels)
        for name in pf_mem.channels:
            np.testing.assert_array_equal(pf_mem.channels[name], pf_disk.channels[name])
        np.testing.assert_array_equal(pf_mem.t_fast, pf_disk.t_fast)
        assert pf_mem.start_time == pf_disk.start_time

    def test_translated_file_works_with_v6_tools(self, v1_dir):
        """Zero special-casing: cutp and patch-template paths accept it."""
        from odas_tpw.rsi.config_patch import read_config_text

        dst, _ = translate_v1_to_v6(v1_dir / "SYN_001.p", v1_dir / "out" / "SYN_001.p")
        assert "translated_from = odas_v1" in read_config_text(dst)
        seg = extract_pfile_segment(dst, v1_dir / "out" / "cut.p", n_records=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pf = PFile(seg)
        assert len(pf.t_slow) == 2 * V1_SCANS_PER_RECORD // 4

    def test_refuses_v6_input(self, tmp_path):
        with pytest.raises(ValueError, match="not a header-v1 file"):
            translate_v1_bytes(DATA / "SN479_0006.p")

    def test_refuses_overwrite_without_force(self, v1_dir):
        dst = v1_dir / "out" / "SYN_001.p"
        translate_v1_to_v6(v1_dir / "SYN_001.p", dst)
        with pytest.raises(FileExistsError):
            translate_v1_to_v6(v1_dir / "SYN_001.p", dst)
        translate_v1_to_v6(v1_dir / "SYN_001.p", dst, overwrite=True)

    def test_refuses_in_place(self, v1_dir):
        with pytest.raises(ValueError, match="in place"):
            translate_v1_to_v6(v1_dir / "SYN_001.p", v1_dir / "SYN_001.p")

    def test_matrix_mismatch_refused(self, v1_dir):
        bad = SETUP_TEXT.replace("matrix: 10 17 8 5", "matrix: 10 17 8 7")
        (v1_dir / "setup.txt").write_text(bad)
        with pytest.raises(ValueError, match="does not match the binary matrix"):
            translate_v1_bytes(v1_dir / "SYN_001.p")

    def test_rate_mismatch_refused(self, v1_dir):
        (v1_dir / "setup.txt").write_text(SETUP_TEXT.replace("rate:     512", "rate:     1024"))
        with pytest.raises(ValueError, match="rate mismatch"):
            translate_v1_bytes(v1_dir / "SYN_001.p")

    def test_sens_flag_beats_setup_extension(self, v1_dir):
        _blob, meta = translate_v1_bytes(v1_dir / "SYN_001.p", sens={"sh1": 0.25})
        assert meta["sens_source"] == "--sens override"
        assert "sens = 0.25" in meta["config_str"]

    def test_setup_discovery_precedence(self, tmp_path):
        """setup.txt outranks SetUp*.txt outranks setup*.cfg; one level up."""
        d = tmp_path / "casts"
        d.mkdir()
        make_v1_file(d / "SYN_001.p")
        (tmp_path / "SetUp2013.txt").write_text(SETUP_TEXT)
        cands = discover_setup_candidates(d / "SYN_001.p")
        assert [c.name for c in cands] == ["SetUp2013.txt"]  # parent dir, pattern 2
        (d / "setup_alt.cfg").write_text("[matrix]\nrow1 = 0\n")
        (d / "setup.txt").write_text(SETUP_TEXT)
        cands = discover_setup_candidates(d / "SYN_001.p")
        assert [c.name for c in cands] == ["setup.txt", "setup_alt.cfg", "SetUp2013.txt"]

    def test_ini_dialect_candidate_sniffed(self, tmp_path):
        """A .cfg in the INI dialect routes to parse_config (delta F7)."""
        ini_setup = tmp_path / "setup_modern.cfg"
        ini_setup.write_text(
            "rate = 512\nno-fast = 2\nno-slow = 2\nrecsize = 1\n"
            "[matrix]\n"
            "row1 = 255 16 8 5\nrow2 = 10 17 8 5\nrow3 = 4 0 8 5\nrow4 = 4 0 8 5\n"
            "[channel]\nid = 10\nname = P\ntype = poly\ncoef0 = 4.46\n"
            "coef1 = 0.049994\ncoef2 = 1.341e-8\nunits = [dBar]\n"
        )
        cfg, dialect = load_setup_file(ini_setup)
        assert dialect == "ini"
        assert cfg["matrix"] == V1_MATRIX
        # And it drives a translation when passed explicitly.
        p = make_v1_file(tmp_path / "SYN_001.p")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _blob, meta = translate_v1_bytes(p, setup_file=ini_setup)
        assert meta["setup_file"] == str(ini_setup)

    def test_cross_candidate_disagreement_warns(self, v1_dir):
        (v1_dir / "setup_wrong.cfg").write_text(
            "[matrix]\nrow1 = 1 2 3\n"
            "[channel]\nid = 10\nname = P\ntype = poly\ncoef0 = 0.449\n"
            "coef1 = 0.11629\ncoef2 = -4.729e-8\n"
        )
        with pytest.warns(UserWarning, match="setup_wrong.cfg disagrees with setup.txt"):
            translate_v1_bytes(v1_dir / "SYN_001.p")

    def test_record0_nonzero_padding_warns(self, v1_dir):
        src = v1_dir / "SYN_001.p"
        raw = bytearray(src.read_bytes())
        raw[V1_RECORD_SIZE - 2 : V1_RECORD_SIZE] = b"\x01\x00"  # last pad word
        bad = v1_dir / "SYN_pad.p"
        bad.write_bytes(bytes(raw))
        with pytest.warns(UserWarning, match="non-zero words beyond the address matrix"):
            translate_v1_bytes(bad)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCliV1to6:
    def test_batch_translate_with_error_containment(self, v1_dir, tmp_path, monkeypatch, capsys):
        from odas_tpw.rsi.cli import main

        make_v1_file(v1_dir / "SYN_004.p")
        # A v6 file mixed in must fail loudly but not stop the batch.
        (v1_dir / "notv1.p").write_bytes((DATA / "SN479_0006.p").read_bytes())
        out = tmp_path / "translated"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "rsi-tpw",
                "v1to6",
                str(v1_dir / "SYN_001.p"),
                str(v1_dir / "SYN_004.p"),
                str(v1_dir / "notv1.p"),
                "-o",
                str(out),
                "--sens",
                "sh1=0.0893",
            ],
        )
        with pytest.raises(SystemExit) as exc, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            main()
        assert exc.value.code == 1  # the v6 file failed
        captured = capsys.readouterr()
        assert (out / "SYN_001.p").exists()
        assert (out / "SYN_004.p").exists()
        assert not (out / "notv1.p").exists()
        assert "not a header-v1 file" in captured.err
        assert "sens: --sens override" in captured.out

    def test_bad_sens_argument(self, monkeypatch, capsys):
        from odas_tpw.rsi.cli import main

        monkeypatch.setattr(
            sys, "argv", ["rsi-tpw", "v1to6", "x.p", "-o", "y", "--sens", "sh1:0.1"]
        )
        with pytest.raises(SystemExit):
            main()
        assert "NAME=VALUE" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# sbt / sbc converters (SPEC §2.4 items 3-4)
# ---------------------------------------------------------------------------


class TestSeaBirdConverters:
    PARAMS_SBT: ClassVar[dict[str, str]] = {
        "name": "SBT1",
        "coef0": "4.36461798e-3",
        "coef1": "6.35619211e-4",
        "coef2": "2.12747658e-5",
        "coef3": "1.87173415e-6",
        "coef4": "1000.0",
        "coef5": "24e6",
        "coef6": "128",
    }
    PARAMS_SBC: ClassVar[dict[str, str]] = {
        "name": "SBC1",
        "coef0": "-9.92054954e0",
        "coef1": "0",
        "coef2": "1.59292963e0",
        "coef3": "-1.24096695e-3",
        "coef4": "1.93245653e-4",
        "coef5": "24e6",
        "coef6": "128",
    }

    def test_sbt_hand_computed(self):
        w = np.array([6.0e5, 8.0e5, 1.2e6])
        T, units = convert_sbt(w, self.PARAMS_SBT)
        f = 128 * 24e6 / w
        x = np.log(1000.0 / f)
        want = (
            1.0
            / (4.36461798e-3 + 6.35619211e-4 * x + 2.12747658e-5 * x**2 + 1.87173415e-6 * x**3)
            - 273.15
        )
        np.testing.assert_allclose(T, want, rtol=1e-14)
        assert units == "deg_C"

    def test_sbc_hand_computed(self):
        w = np.array([3.0e5, 4.0e5])
        C, units = convert_sbc(w, self.PARAMS_SBC)
        f = 128 * 24e6 / w / 1000.0
        want = -9.92054954 + 1.59292963 * f**2 - 1.24096695e-3 * f**3 + 1.93245653e-4 * f**4
        np.testing.assert_allclose(C, want, rtol=1e-14)
        assert units == "mS_cm-1"

    def test_zero_count_yields_nan(self):
        T, _ = convert_sbt(np.array([0.0, 6.0e5]), self.PARAMS_SBT)
        C, _ = convert_sbc(np.array([0.0, 3.0e5]), self.PARAMS_SBC)
        assert np.isnan(T[0]) and np.isfinite(T[1])
        assert np.isnan(C[0]) and np.isfinite(C[1])


class TestShearSensHardError:
    def test_convert_shear_missing_sens_raises(self):
        with pytest.raises(ValueError, match="'sens' missing"):
            convert_shear(np.array([1.0]), {"name": "sh1", "diff_gain": "1.0"})

    def test_convert_shear_empty_sens_raises(self):
        with pytest.raises(ValueError, match="'sens' missing"):
            convert_shear(np.array([1.0]), {"name": "sh1", "diff_gain": "1.0", "sens": ""})


# ---------------------------------------------------------------------------
# Type-based sbt reference-temperature candidacy (delta F8)
# ---------------------------------------------------------------------------


class TestSbtTemperatureCandidate:
    def test_sbt_appended_after_jac_t_by_type(self):
        from odas_tpw.rsi.helpers import temperature_candidates

        n = 50
        channels = {
            "SBT1": np.full(n, 20.0),
            "JAC_T": np.full(n, 21.0),
            "T1": np.full(n, 22.0),
            "sh1": np.zeros(4 * n),
        }
        types = {"SBT1": "sbt", "JAC_T": "jac_t", "T1": "raw", "sh1": "shear"}
        assert temperature_candidates(channels, n, types) == ["T1", "JAC_T", "SBT1"]
        # Without types the sbt channel is invisible (name-based only).
        assert temperature_candidates(channels, n) == ["T1", "JAC_T"]

    def test_counts_valued_t1_falls_through_to_sbt(self, v1_dir):
        """On the synthetic v1 file, raw-counts T1 fails QC and auto lands on
        the Sea-Bird channel (F8: the whole point of the type-based tail)."""
        from odas_tpw.rsi.helpers import load_channels

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = load_channels(v1_dir / "SYN_001.p")
        assert data["metadata"]["temperature_source"] == "SBT1"
        assert data["metadata"]["translated_from"] == "odas_v1"
        assert data["metadata"]["setup_file_source"] == str(v1_dir / "setup.txt")


# ---------------------------------------------------------------------------
# Refusal guards in the six positional readers (delta F5)
# ---------------------------------------------------------------------------


class TestV1RefusalGuards:
    REMEDY = "rsi-tpw v1to6"

    def test_extract_pfile_segment(self, v1_dir, tmp_path):
        with pytest.raises(ValueError, match=self.REMEDY):
            extract_pfile_segment(v1_dir / "SYN_001.p", tmp_path / "cut.p")

    def test_read_config_string(self, v1_dir):
        with pytest.raises(ValueError, match=self.REMEDY):
            read_config_string(v1_dir / "SYN_001.p")

    def test_perturb_trim(self, v1_dir, tmp_path):
        from odas_tpw.perturb.trim import trim_p_file

        with pytest.raises(ValueError, match=self.REMEDY):
            trim_p_file(v1_dir / "SYN_001.p", tmp_path / "trimmed")

    def test_perturb_merge(self, v1_dir):
        from odas_tpw.perturb.merge import _read_merge_info

        with pytest.raises(ValueError, match=self.REMEDY):
            _read_merge_info(v1_dir / "SYN_001.p")

    def test_sensor_inventory(self, v1_dir):
        from odas_tpw.rsi.sensor_inventory import _read_header_and_config

        with pytest.raises(ValueError, match=self.REMEDY):
            _read_header_and_config(v1_dir / "SYN_001.p")

    def test_config_patch_read(self, v1_dir):
        from odas_tpw.rsi.config_patch import read_config_text

        with pytest.raises(ValueError, match=self.REMEDY):
            read_config_text(v1_dir / "SYN_001.p")

    def test_config_patch_write(self, v1_dir, tmp_path):
        from odas_tpw.rsi.config_patch import write_patched_pfile

        with pytest.raises(ValueError, match=self.REMEDY):
            write_patched_pfile(v1_dir / "SYN_001.p", tmp_path / "out.p", "[root]\n")


# ---------------------------------------------------------------------------
# Golden per-channel hashes: the v6 path is provably untouched (delta F11)
# ---------------------------------------------------------------------------

# Computed on the W7 integration base (main + #133 + #134 merges) BEFORE any
# v1 work, via sha256(channel_array.tobytes())[:16]. None of these fixtures
# carries sbt/sbc-type channels, so the new converters cannot move them.
GOLDEN_V6_HASHES = {
    "SN479_0006": {
        "Ax": "a61d427617b6918d",
        "Ay": "f95155be380604fe",
        "Chlorophyll": "8f6de426dfb5c362",
        "DO": "4f0cd67e10051f53",
        "DO_T": "91f04f25c6bcb22b",
        "Gnd": "dbd9fbfa4b22d3f1",
        "Incl_T": "e5da6d1ac433be60",
        "Incl_X": "86fe2ce2667efeef",
        "Incl_Y": "a1fe378e076828d0",
        "JAC_C": "20b55409f370c7d1",
        "JAC_T": "93ca7dd6123e20f5",
        "P": "62c511ab5fc327cf",
        "PV": "384a5e055fdc1645",
        "P_dP": "62c511ab5fc327cf",
        "T1": "ec57f5d2d542a527",
        "T1_dT1": "b1ae40c0abf510cf",
        "T2": "5a739072c354307e",
        "T2_dT2": "24f4ba20f2daa940",
        "Turbidity": "d721e274eb51f15a",
        "V_Bat": "82c684bcba5505e7",
        "sh1": "b2f348fdeab765f6",
        "sh2": "92c3001eb5bdb1df",
    },
    "MR_SL435": {
        "Ax": "862cdaf2b0556bdf",
        "Ay": "8da7192fb79dbcf6",
        "EMC_Cur": "8ee2bd803c6c6c8e",
        "Gnd": "a93b0d91ca597e58",
        "Incl_T": "e95eade6809608c3",
        "Incl_X": "3e1484dc25c0b128",
        "Incl_Y": "6f7ebc5744586e28",
        "P": "26c58fc2f60e583a",
        "PV": "cf2da47866cf44af",
        "P_dP": "26c58fc2f60e583a",
        "T1": "011369160998d579",
        "T1_dT1": "0e6e0c9cdf1a99e0",
        "T2": "bcffc31f135b2702",
        "T2_dT2": "940750d3b8ef5890",
        "U_EM": "842347c50f8333a8",
        "V_Bat": "78a3b5c41de3796b",
        "sh1": "121f054917d7b835",
        "sh2": "4369b359b21c1dac",
    },
    "VMP142_bench": {
        "Ax": "f30c93cd04db040d",
        "Ay": "b4e644e20d9d4e02",
        "Gnd": "51be6287dc7c573e",
        "Incl_T": "b3ce9bf2afa25565",
        "Incl_X": "26714cb5b7964f5c",
        "Incl_Y": "69c6a375cba3efb8",
        "JAC_C": "a6ecc2db3089703b",
        "JAC_T": "fe351014dd41c00a",
        "P": "f9a4dd36c8cd5bb1",
        "PV": "df7d1ad0f905d869",
        "P_dP": "f9a4dd36c8cd5bb1",
        "T1": "d28affcc80485e0e",
        "T1_dT1": "2e27e779538386de",
        "T2": "56cf7f61cbf3e359",
        "T2_dT2": "255fe622fc09ecef",
        "V_Bat": "85923f4e93a27d58",
        "sh1": "18d09bc85df4a7f1",
        "sh2": "e2558f062dfe7eaf",
    },
}


@pytest.mark.parametrize("fixture", sorted(GOLDEN_V6_HASHES))
def test_v6_golden_channel_hashes(fixture):
    """Bit-identical v6 regression: every channel of every committed v6
    fixture hashes exactly as on the pre-#141 integration base."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pf = PFile(DATA / f"{fixture}.p")
    got = {
        name: hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]
        for name, arr in pf.channels.items()
    }
    assert got == GOLDEN_V6_HASHES[fixture]
