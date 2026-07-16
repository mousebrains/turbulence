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


def make_v1_profile_file(path: Path, *, seconds: float = 14.0) -> Path:
    """A profile-shaped synthetic v1 file: steady ~1 dbar/s descent with
    noisy shear — long enough for profile detection and epsilon windows."""
    rng = np.random.default_rng(42)
    n_cycles = int(seconds * 128)  # fs_slow = 128 Hz for this geometry
    n_scans = n_cycles * 4
    n_records = n_scans // V1_SCANS_PER_RECORD
    out = bytearray()
    out += _v1_header(0)
    m = np.array(V1_MATRIX, dtype="<u2").ravel()
    block = bytearray(V1_RECORD_SIZE - HEADER_BYTES)
    block[: m.nbytes] = m.tobytes()
    out += block
    sh = rng.integers(-150, 150, size=n_scans + V1_SCANS_PER_RECORD)
    scan = 0
    for rec in range(1, n_records + 1):
        out += _v1_header(rec)
        words = []
        for _ in range(V1_SCANS_PER_RECORD):
            cyc, r = divmod(scan, 4)
            if r == 0:
                # sp_char; sbt even word (small wiggle: an exactly-constant
                # temperature trips the railed-sensor QC)
                words += [32752, 29640 + (cyc % 7), int(sh[scan]), 0]
            elif r == 1:
                # ~1 dbar/s descent: +0.15625 P counts/cycle at ~0.05 dbar/count
                words += [200 + int(cyc * 0.15625), 10, int(sh[scan]), 0]
            else:
                words += [7000, 11, int(sh[scan]), 0]  # T1 slots; gnd
            scan += 1
        out += np.array(words, dtype="<i2").tobytes()
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

    @pytest.mark.parametrize(
        "bad", ["sh1=nan,sh2=inf", "sh1=nan", "sh2=inf", "sh1=-inf", "sh1=0", "sh2=-0.05"]
    )
    def test_nonfinite_sens_argument_rejected(self, bad, monkeypatch, capsys):
        """--sens must be finite and positive: 'nan' bypasses sign checks
        (NaN compares False -> all-NaN shear) and 'inf' zeroes the shear."""
        from odas_tpw.rsi.cli import main

        monkeypatch.setattr(sys, "argv", ["rsi-tpw", "v1to6", "x.p", "-o", "y", "--sens", bad])
        with pytest.raises(SystemExit):
            main()
        assert "finite positive" in capsys.readouterr().err


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

    @pytest.mark.parametrize(
        "bad", ["0.0893,", "abc", "  ", "0", "-0.08", "nan", "NaN", "inf", "-inf", "Infinity"]
    )
    def test_convert_shear_unusable_sens_raises(self, bad):
        """A PRESENT-but-unusable sens (stray trailing comma, typo, zero,
        negative, NaN, infinite) must raise, never fall through to a
        fabricated 1.0 — epsilon would be silently wrong by sens^-2 (~125x on
        real probes). NaN is the nasty one: every comparison with NaN is
        False, so a bare `sens <= 0` check waves it through into all-NaN
        shear; +inf sails through a sign check into all-zero shear."""
        with pytest.raises(ValueError, match="sens"):
            convert_shear(
                np.array([1.0]), {"name": "sh1", "diff_gain": "1.0", "sens": bad}
            )

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), 0.0, -0.1])
    def test_sens_override_api_rejects_nonfinite(self, bad):
        """The Python-API override path (translate_v1_bytes(sens=...) ->
        parse_setup_v1) is guarded like the CLI parser and convert_shear."""
        with pytest.raises(ValueError, match="finite positive"):
            parse_setup_v1(SETUP_TEXT, sens_overrides={"sh1": bad})

    def test_synthesize_ini_refuses_unroundtrippable_value(self):
        """parse_config strips ';' as an inline comment: a value carrying one
        cannot round-trip and must be refused, not silently corrupted."""
        from odas_tpw.rsi.v1_translate import synthesize_ini

        cfg = {
            "matrix": [[0, 1]],
            "channels": [],
            "instrument_info": {},
            "root": {"disk": "c:\\data;archive"},
        }
        with pytest.raises(ValueError, match="round-trip"):
            synthesize_ini(cfg, {})

    def test_synthesize_ini_guards_ordered_channel_fields(self):
        """The ordered id/name/type channel fields go through the same
        round-trip guard: a legacy pair name carrying ';' must refuse to
        synthesize, not survive to a parse_config silent truncation."""
        from odas_tpw.rsi.v1_translate import synthesize_ini

        setup = (
            "matrix: 16 17\n"
            "channel: 16,SBT1;wrongE,1,2,3,4,5,6,7\n"
            "channel: 17,SBT1;wrongO,1,2,3,4,5,6,7\n"
        )
        cfg = parse_setup_v1(setup)
        assert cfg["channels"][0]["name"] == "SBT1;wrong"  # parser keeps it
        with pytest.raises(ValueError, match="round-trip"):
            synthesize_ini(cfg, {})


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
# Complete translation provenance on derived products (#141 review P2)
# ---------------------------------------------------------------------------


class TestV1ProvenanceOnProducts:
    """The COMPLETE provenance set (V1_PROVENANCE_KEYS — incl. setup_file_md5
    and sens_source, the audit trail for the sens^-2 epsilon scaling) must
    land on epsilon products from (a) an on-disk translated file, (b) a
    direct raw-v1 read, and (c) the per-profile NC route."""

    def _assert_provenance(self, attrs, *, src_name: str, sens_source: str) -> None:
        from odas_tpw.rsi.p_file import V1_PROVENANCE_KEYS

        missing = [k for k in V1_PROVENANCE_KEYS if k not in attrs]
        assert not missing, f"missing provenance attrs: {missing}"
        assert attrs["translated_from"] == "odas_v1"
        assert attrs["v1_source_file"] == src_name
        assert str(attrs["setup_file_source"]).endswith("setup.txt")
        assert len(str(attrs["setup_file_md5"])) == 32
        assert sens_source in str(attrs["sens_source"])
        assert "v1to6" in str(attrs["translator"])
        assert str(attrs["translated_on"])

    @pytest.fixture
    def profile_dir(self, tmp_path):
        (tmp_path / "setup.txt").write_text(SETUP_TEXT)
        make_v1_profile_file(tmp_path / "PROF_001.p")
        return tmp_path

    def test_epsilon_attrs_from_translated_file(self, profile_dir, tmp_path):
        import xarray as xr

        from odas_tpw.rsi.dissipation import compute_diss_file

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dst, meta = translate_v1_to_v6(
                profile_dir / "PROF_001.p",
                tmp_path / "out" / "PROF_001.p",
                sens={"sh1": 0.08},
            )
            paths = compute_diss_file(dst, tmp_path / "eps_a", fft_length=256, goodman=False)
        assert paths
        with xr.open_dataset(paths[0]) as ds:
            self._assert_provenance(
                ds.attrs, src_name="PROF_001.p", sens_source="--sens override"
            )
            assert ds.attrs["setup_file_md5"] == meta["setup_md5"]

    def test_epsilon_attrs_from_raw_v1_read(self, profile_dir, tmp_path):
        """Direct raw-v1 read (sens via the setup-file extension keys): the
        in-memory route publishes the identical provenance set."""
        import xarray as xr

        from odas_tpw.rsi.dissipation import compute_diss_file

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            paths = compute_diss_file(
                profile_dir / "PROF_001.p", tmp_path / "eps_b", fft_length=256, goodman=False
            )
        assert paths
        with xr.open_dataset(paths[0]) as ds:
            self._assert_provenance(ds.attrs, src_name="PROF_001.p", sens_source="_sens keys")

    def test_epsilon_attrs_via_per_profile_nc(self, profile_dir, tmp_path):
        """.p -> per-profile NC -> epsilon: the NC global attrs and the
        NC-branch metadata allowlist both carry the complete set."""
        import xarray as xr

        from odas_tpw.rsi.dissipation import compute_diss_file
        from odas_tpw.rsi.profile import extract_profiles

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            extract_profiles(profile_dir / "PROF_001.p", tmp_path / "prof")
            prof_ncs = sorted((tmp_path / "prof").glob("*prof*.nc"))
            assert prof_ncs, "no per-profile NC produced"
            paths = compute_diss_file(
                prof_ncs[0], tmp_path / "eps_c", fft_length=256, goodman=False
            )
        assert paths
        with xr.open_dataset(paths[0]) as ds:
            self._assert_provenance(ds.attrs, src_name="PROF_001.p", sens_source="_sens keys")

    def test_load_channels_metadata_complete(self, profile_dir):
        from odas_tpw.rsi.helpers import load_channels
        from odas_tpw.rsi.p_file import V1_PROVENANCE_KEYS

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = load_channels(profile_dir / "PROF_001.p")
        for key in V1_PROVENANCE_KEYS:
            assert key in data["metadata"], key


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
# v1 work. Two layers, split by what is platform-stable:
#  - RAW demuxed counts (deconvolve=False -> pure integer demux) hash exactly
#    on every OS/Python: this pins the dispatch/record-tiling/demux surface
#    the v1 work touches, bit-for-bit.
#  - CONVERTED channels involve transcendental libm calls (thermistor log,
#    Butterworth deconvolution) whose last-ulp differs across platforms, so
#    exact hashes are NOT portable (they broke CI on ubuntu/windows). They
#    are pinned instead by (nanmean, nanmin, nanmax) at rtol=1e-8 — platform
#    noise is ~1e-15 relative, while any conversion regression (wrong
#    coefficient, formula, channel routing) moves these far beyond 1e-8.
# None of these fixtures carries sbt/sbc-type channels, so the new
# converters cannot move them.
GOLDEN_V6_RAW_HASHES = {
    "MR_SL435": {
        "Ax": "596375b644308bd6",
        "Ay": "76ee2f7bad34853e",
        "EMC_Cur": "2f38651863e3cc02",
        "Gnd": "a1acf47935d76283",
        "Incl_T": "4d6dd2dfdd2ae3eb",
        "Incl_X": "c060927bd9ede51d",
        "Incl_Y": "357bdef3f4499c53",
        "P": "597440b5cc5af030",
        "PV": "be9d0104e15ee0d4",
        "P_dP": "0be2c084d990fef1",
        "T1": "b117abfef732d903",
        "T1_dT1": "e29eae1d16aac372",
        "T2": "12b20793b88f9a81",
        "T2_dT2": "c9dd233e7a30d518",
        "U_EM": "9d68b759bc5fc113",
        "V_Bat": "146eb13f9240970b",
        "sh1": "b3b8b218046ed064",
        "sh2": "aeafcb2125b7c24f",
    },
    "SN479_0006": {
        "Ax": "b7474f5f77b0415e",
        "Ay": "1fa055a822117127",
        "Chlorophyll": "4e0ec947b34248e7",
        "DO": "bd88b978aaf3ed3f",
        "DO_T": "c6a2d10a24c1b4d7",
        "Gnd": "1ea50be7e8167463",
        "Incl_T": "73fb39facf234d5b",
        "Incl_X": "3deb4d44be616085",
        "Incl_Y": "0e5f2b0c7c1fa934",
        "JAC_C": "422b62fab62ec5ee",
        "JAC_T": "5154654b928ab776",
        "P": "b154c93ea3b9d9d8",
        "PV": "34019042cb73934a",
        "P_dP": "84b4ec973ee51c68",
        "T1": "de915377413aa116",
        "T1_dT1": "6a48e83ddde4bc8f",
        "T2": "9befa5a08939d2ff",
        "T2_dT2": "7908ff8dfee8e4d2",
        "Turbidity": "028a41ea4f142e3d",
        "V_Bat": "2875323333466b14",
        "sh1": "86f5f0266590e67e",
        "sh2": "cfbdbdbc275153d8",
    },
    "VMP142_bench": {
        "Ax": "bce7385cdb2add1b",
        "Ay": "f5e6f5c04aa75308",
        "Gnd": "c10dbfc803f2d291",
        "Incl_T": "c97ce52801dbbb4b",
        "Incl_X": "a00f1bf86857c51f",
        "Incl_Y": "b51198c7baae332f",
        "JAC_C": "c2c9db4e798eebd6",
        "JAC_T": "434db0bdf8b9393b",
        "P": "03514c8cfbdc651e",
        "PV": "bb9a5cb85d2b266d",
        "P_dP": "9348660d4e20060c",
        "T1": "a7698b640619a04f",
        "T1_dT1": "30ec1dbb54c96cfc",
        "T2": "6a7c1cebc3c0b96e",
        "T2_dT2": "9a4f090bfedd78c0",
        "V_Bat": "83207e58d3264006",
        "sh1": "77de3929d31d33cd",
        "sh2": "649b5be587ce1e9c",
    },
}

GOLDEN_V6_STATS = {
    "MR_SL435": {
        "Ax": (-1.4130859375, -3384.0, 3368.0),
        "Ay": (-2.45732421875, -2020.0, 1697.0),
        "EMC_Cur": (-0.03642034301757813, -0.793875, 0.723),
        "Gnd": (-0.9953125, -2.0, 0.0),
        "Incl_T": (13.0, 13.0, 13.0),
        "Incl_X": (11.224531250000002, 10.875, 11.525),
        "Incl_Y": (-20.909960937500003, -26.375, -16.05),
        "P": (506.2075301469202, 505.46664752242043, 506.6902011702802),
        "PV": (3.8819456054687507, 3.881625, 3.88225),
        "P_dP": (506.2075301469202, 505.46664752242043, 506.6902011702802),
        "T1": (12.096820415826603, 12.095993418224282, 12.098329765050607),
        "T1_dT1": (12.096821178056981, 12.095993418224282, 12.098342333671326),
        "T2": (11.109274872349648, 11.106009263127135, 11.111500484441933),
        "T2_dT2": (11.109276994249996, 11.105999046959653, 11.111511022270236),
        "U_EM": (0.13549874154593752, -0.2659174, 0.15097912219999998),
        "V_Bat": (13.6215244140625, 13.575, 13.665625),
        "sh1": (-9.269801734550487e-05, -0.616276157293797, 0.5014575028053113),
        "sh2": (-0.0005741897618813179, -0.23150099810766922, 0.3506524743373511),
    },
    "SN479_0006": {
        "Ax": (-2.0428292410714284, -6410.0, 6292.0),
        "Ay": (-2.0635734437003967, -15032.0, 17705.0),
        "Chlorophyll": (0.27238653297510507, -1.2878584340000003, 1.7220515029999994),
        "DO": (218.97191499255953, 197.7, 228.7),
        "DO_T": (24.79482682291667, 16.556, 28.729999999999997),
        "Gnd": (-1.4685639880952381, -3.0, 0.0),
        "Incl_T": (23.679123883928625, 22.870000000000005, 24.280000000000086),
        "Incl_X": (-30.354417007688493, -90.0, 2.1750000000000003),
        "Incl_Y": (58.59761672247024, 7.2250000000000005, 90.0),
        "JAC_C": (52.67294093271342, 44.13009425495208, 57.51683679477821),
        "JAC_T": (24.812081204233255, 16.471109912590247, 28.739318840562923),
        "P": (113.6230423843735, 0.22480541080417638, 262.52192592897546),
        "PV": (3.855852767702133, 3.8305000000000002, 3.875375),
        "P_dP": (113.6230423843735, 0.22480541080417638, 262.52192592897546),
        "T1": (23.92485570795937, 15.762638141907246, 27.741034264235623),
        "T1_dT1": (23.92485664792897, 15.76222182673638, 27.741034264235623),
        "T2": (32.58799599530332, 24.149816607348725, 36.55255866404195),
        "T2_dT2": (32.587996482371466, 24.149753781713457, 36.55255866404195),
        "Turbidity": (0.8369726211018556, -0.6400297487078213, 6.247696848686701),
        "V_Bat": (14.56040331643725, 14.456249999999999, 14.594999999999999),
        "sh1": (-0.25838922629818684, -6.977641875684132, 6.847716046538951),
        "sh2": (-0.2500851945673939, -6.78764058890531, 6.660627768251408),
    },
    "VMP142_bench": {
        "Ax": (7.5885633680555555, -4468.0, 4437.0),
        "Ay": (8.995442708333334, -5545.0, 6654.0),
        "Gnd": (7.0578125, 6.0, 8.0),
        "Incl_T": (18.32609548611114, 17.700000000000045, 18.639999999999986),
        "Incl_X": (31.85534722222222, 31.8, 31.900000000000002),
        "Incl_Y": (0.23493489583333335, 0.17500000000000002, 0.30000000000000004),
        "JAC_C": (0.022591772065860806, 0.01730732799970599, 0.02777614159700873),
        "JAC_T": (21.746821094470032, 21.65854341324288, 21.781292978578747),
        "P": (0.30736272589596597, 0.3053025990938696, 0.30793836512702),
        "PV": (4.0106680338541665, 4.0105, 4.010875),
        "P_dP": (0.30736272589596597, 0.3053025990938696, 0.30793836512702),
        "T1": (16.1569150871382, 16.156904844929784, 16.157311548123346),
        "T1_dT1": (16.156915066536943, 16.15690450258313, 16.157311548123346),
        "T2": (16.145748038894958, 16.145745181298878, 16.14591898407184),
        "T2_dT2": (16.145748026781277, 16.145745181298878, 16.14591898407184),
        "V_Bat": (16.545872829861114, 16.476875, 16.5575),
        "sh1": (0.002504827736571818, 0.0010244566918550186, 0.004097826767420074),
        "sh2": (0.00275363126114523, 0.0012347569505908953, 0.004321649327068134),
    },
}


@pytest.mark.parametrize("fixture", sorted(GOLDEN_V6_RAW_HASHES))
def test_v6_golden_raw_hashes(fixture):
    """Bit-identical v6 demux regression (platform-stable integer arrays)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pf = PFile(DATA / f"{fixture}.p", deconvolve=False)
    got = {
        name: hashlib.sha256(
            str(np.ascontiguousarray(arr).dtype).encode()
            + np.ascontiguousarray(arr).tobytes()
        ).hexdigest()[:16]
        for name, arr in pf.channels_raw.items()
    }
    assert got == GOLDEN_V6_RAW_HASHES[fixture]


@pytest.mark.parametrize("fixture", sorted(GOLDEN_V6_STATS))
def test_v6_golden_channel_stats(fixture):
    """Converted-channel v6 regression at platform-safe tolerance."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pf = PFile(DATA / f"{fixture}.p")
    assert sorted(pf.channels) == sorted(GOLDEN_V6_STATS[fixture])
    for name, (m, lo, hi) in GOLDEN_V6_STATS[fixture].items():
        arr = np.asarray(pf.channels[name], dtype=np.float64)
        got = (float(np.nanmean(arr)), float(np.nanmin(arr)), float(np.nanmax(arr)))
        np.testing.assert_allclose(got, (m, lo, hi), rtol=1e-8, atol=0,
                                   err_msg=f"{fixture}:{name}")


# Order-sensitive converted-channel goldens (review P3b): whole-array stats
# alone cannot see a one-sample roll (nanmean/min/max are order-blind) or a
# single interior NaN (nanmean moves ~1/N). tests/data/v6_golden_converted.npz
# commits, per fixture/channel: the exact SHAPE, a sha256 of the finite-mask
# BYTES (pins NaN placement sample-exactly; bools are platform-stable ints),
# and a ~256-point DECIMATED float array compared via allclose (never byte
# hashes of floats — libm last-ulp noise is ~1e-15 relative across platforms,
# far below the 1e-8 tolerance, while a roll shifts each sampled point by the
# local sample-to-sample signal difference, orders of magnitude above it).
_GOLDEN_CONVERTED_NPZ = DATA / "v6_golden_converted.npz"


@pytest.mark.parametrize("fixture", sorted(GOLDEN_V6_STATS))
def test_v6_golden_converted_arrays(fixture):
    """Shape + finite-mask + decimated-value v6 regression (order-sensitive)."""
    ref = np.load(_GOLDEN_CONVERTED_NPZ)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pf = PFile(DATA / f"{fixture}.p")
    ref_names = sorted({k.split("|")[1] for k in ref.files if k.startswith(f"{fixture}|")})
    assert sorted(pf.channels) == ref_names
    for name in ref_names:
        arr = np.asarray(pf.channels[name], dtype=np.float64)
        # (i) exact shape
        assert list(arr.shape) == ref[f"{fixture}|{name}|shape"].tolist(), name
        # (ii) exact finite-mask bytes: catches any injected/moved NaN
        mask = np.ascontiguousarray(np.isfinite(arr))
        got_hash = hashlib.sha256(mask.tobytes()).hexdigest()[:16]
        assert got_hash == str(ref[f"{fixture}|{name}|maskhash"]), name
        # (iii) decimated whole-array comparison: order-sensitive, catches
        # rolls/shifts that channel-wide stats cannot. atol is scaled to the
        # channel magnitude so exact zeros compare exactly.
        want = ref[f"{fixture}|{name}|dec"]
        got = arr[:: max(1, arr.size // 256)]
        scale = float(np.nanmax(np.abs(want))) if want.size else 0.0
        np.testing.assert_allclose(
            got,
            want,
            rtol=1e-8,
            atol=1e-8 * max(scale, 1e-30),
            equal_nan=True,
            err_msg=f"{fixture}:{name}",
        )
