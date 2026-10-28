"""PFile on mis-framed .p files: a record of the wrong length (refuse or repair).

Built by damaging real files the way MicroRider SN 156 (2022) damaged 28 of
its 29: one record short (33 of 64 matrices), or a record that swallowed the
next one's header and part of its data (98 or 67 matrices), or junk words.
"""

import struct
import warnings
from pathlib import Path

import numpy as np
import pytest

from odas_tpw.rsi.p_file import PFile

DATA = Path(__file__).parent / "data"
MR = DATA / "MR_SL685_climb.p"  # v6.3, records numbered from 121 (a cutp segment)
VMP = DATA / "SN479_0006.p"  # v6.1, numbered from 1


def _endian(raw: bytes) -> str:
    return ">" if struct.unpack(">64H", raw[:128])[17] == 128 else "<"


def _geometry(raw: bytes):
    w = struct.unpack(f"{_endian(raw)}64H", raw[:128])
    assert w[17] == 128
    header, first, record = w[17], w[17] + w[11], w[18]
    matrix = w[30] * (w[28] + w[29]) * 2
    return header, first, record, matrix


def short(raw: bytes, k: int, keep: int = 33) -> bytes:
    """Data record k (0-based) keeps only its first *keep* matrices."""
    header, first, record, matrix = _geometry(raw)
    a = first + k * record + header + keep * matrix
    return raw[:a] + raw[first + (k + 1) * record :]


def swallow(raw: bytes, k: int, lost: int = 30) -> bytes:
    """Record k+1 loses its header and its first *lost* matrices into record k."""
    header, first, record, matrix = _geometry(raw)
    a = first + (k + 1) * record
    return raw[:a] + raw[a + header + lost * matrix :]


def junk(raw: bytes, k: int, n_words: int = 41) -> bytes:
    """*n_words* junk words appended to record k's data."""
    _, first, record, _ = _geometry(raw)
    a = first + (k + 1) * record
    return raw[:a] + bytes(range(1, 2 * n_words + 1)) + raw[a:]


def set_record_numbers(raw: bytes, numbers) -> bytes:
    """Overwrite header word 2 of the data records found at their true positions."""
    header, first, record, _ = _geometry(raw)
    out = bytearray(raw)
    sig = raw[first + 17 * 2 : first + 19 * 2]
    pos, i = first, 0
    while pos + header <= len(out) and i < len(numbers):
        if out[pos + 34 : pos + 38] == sig:
            out[pos + 2 : pos + 4] = struct.pack(f"{_endian(raw)}H", numbers[i] % 65536)
            i += 1
            pos += record
        else:
            pos += 2
    return bytes(out)


def load(raw: bytes, tmp_path, name="damaged.p", **kw) -> PFile:
    p = tmp_path / name
    p.write_bytes(raw)
    return PFile(p, deconvolve=False, **kw)


@pytest.fixture(scope="module")
def mr_bytes():
    return MR.read_bytes()


@pytest.fixture(scope="module")
def mr_clean():
    return PFile(MR, deconvolve=False)


def _assert_repaired(pf, clean, slot0, n_slots):
    """Every channel equals the clean file outside slots [slot0, slot0+n_slots),
    and the bad-buffer report covers exactly those slots on every channel."""
    assert pf.t_fast.size == clean.t_fast.size
    assert pf.t_slow.size == clean.t_slow.size
    scans = pf.header["record_size"] // 2 - pf.header["header_size"] // 2
    scans //= pf.n_cols
    matrices = scans // pf.n_rows
    confirmed = pf.bad_buffer_report["confirmed"]
    for name, x in clean.channels_raw.items():
        y = pf.channels_raw[name]
        per = scans if x.size == clean.t_fast.size else matrices
        lo, hi = slot0 * per, (slot0 + n_slots) * per
        keep = np.ones(x.size, dtype=bool)
        keep[lo:hi] = False
        np.testing.assert_array_equal(y[keep], x[keep], err_msg=name)
        assert confirmed[name]["spans"] == [(lo, hi - lo)], name
        assert confirmed[name]["rate"] == ("fast" if per == scans else "slow"), name


class TestRepair:
    @pytest.mark.parametrize(
        "damage, slot0, n",
        [
            (lambda r: short(r, 9), 9, 1),  # 33 matrices
            (lambda r: swallow(r, 9), 9, 2),  # 98 matrices + #11 header lost
            (lambda r: swallow(r, 9, lost=61), 9, 2),  # 67 matrices
            (lambda r: junk(r, 9), 9, 1),
            (lambda r: short(r, 0), 0, 1),  # the first data record itself
        ],
        ids=["short", "swallow98", "swallow67", "junk", "first-record"],
    )
    def test_intact_samples_are_untouched(self, tmp_path, mr_bytes, mr_clean, damage, slot0, n):
        with pytest.warns(UserWarning, match="mis-framed") as rec:
            pf = load(damage(mr_bytes), tmp_path)
        _assert_repaired(pf, mr_clean, slot0, n)
        assert pf.framing_report["rebuilt_slots"] == [(slot0, n)]
        # The fixed-size symptoms are gone: no word-16 flags, record-number
        # gaps or "non-integer record count" from the rebuilt stream.
        text = " ".join(str(w.message) for w in rec)
        assert "flagged bad-buffer" not in text
        assert "skip" not in text
        assert "integer number of records" not in text

    def test_time_base_is_the_clean_files(self, tmp_path, mr_bytes, mr_clean):
        with pytest.warns(UserWarning, match="mis-framed"):
            pf = load(swallow(mr_bytes, 40), tmp_path)
        assert pf.start_time == mr_clean.start_time
        np.testing.assert_array_equal(pf.t_fast, mr_clean.t_fast)

    def test_two_damaged_spots(self, tmp_path, mr_bytes, mr_clean):
        raw = short(swallow(mr_bytes, 100), 20)
        with pytest.warns(UserWarning, match="mis-framed"):
            pf = load(raw, tmp_path)
        assert pf.framing_report["rebuilt_slots"] == [(20, 1), (100, 2)]
        assert pf.framing_report["missing_record_numbers"] == [121 + 101]
        assert [r["record_number"] for r in pf.framing_report["irregular_records"]] == [
            121 + 20,
            121 + 100,
        ]

    def test_rebuilt_slots_are_interpolated_not_zero(self, tmp_path, mr_bytes, mr_clean):
        """Filling with interpolation keeps the deconvolution filters free of a
        step; zeros would ring through the neighbouring records."""
        with pytest.warns(UserWarning, match="mis-framed"):
            pf = load(short(mr_bytes, 50), tmp_path)
        scans = pf.t_fast.size // 150
        p = pf.channels_raw["P"]
        span = slice(50 * scans // pf.n_rows, 51 * scans // pf.n_rows)
        lo, hi = p[span.start - 1], p[span.stop]
        assert np.all(p[span] >= min(lo, hi)) and np.all(p[span] <= max(lo, hi))

    def test_deconvolution_outside_the_gap(self, tmp_path, mr_bytes):
        """Shear is not deconvolved and is untouched; deconvolved T and P move
        only by the global regression fit (measured 3.5e-5 C, 0.35 mdbar)."""
        clean = PFile(MR)
        p = tmp_path / "d.p"
        p.write_bytes(short(mr_bytes, 70))
        with pytest.warns(UserWarning, match="mis-framed"):
            pf = PFile(p)
        scans = pf.t_fast.size // 150
        keep = np.ones(pf.t_fast.size, dtype=bool)
        keep[70 * scans : 71 * scans] = False
        np.testing.assert_array_equal(pf.channels["sh1"][keep], clean.channels["sh1"][keep])
        for name, tol in (("T1_dT1", 1e-4), ("P", 1e-3)):
            x, y = clean.channels[name], pf.channels[name]
            t = clean.t_fast if x.size == clean.t_fast.size else clean.t_slow
            out = (t < 70 * 0.99994 - 5) | (t > 71 * 0.99994 + 5)
            assert np.max(np.abs(x[out] - y[out])) < tol, name

    def test_trailing_partial_after_damage_is_dropped(self, tmp_path, mr_bytes, mr_clean):
        raw = short(mr_bytes, 30)[:-1000]
        with pytest.warns(UserWarning, match="mis-framed"):
            pf = load(raw, tmp_path)
        assert pf.framing_report["trailing_records_dropped"] == 1
        assert pf.t_slow.size == mr_clean.t_slow.size - mr_clean.t_slow.size // 150

    def test_remnant_shorter_than_a_header_keeps_the_last_record(
        self, tmp_path, mr_bytes, mr_clean
    ):
        raw = short(mr_bytes, 30) + b"\x00" * 100
        with pytest.warns(UserWarning, match="mis-framed"):
            pf = load(raw, tmp_path)
        assert pf.t_slow.size == mr_clean.t_slow.size
        assert pf.framing_report["trailing_words_dropped"] == 50
        assert pf.framing_report["trailing_records_dropped"] == 0

    def test_counter_from_zero(self, tmp_path, mr_bytes, mr_clean):
        raw = set_record_numbers(mr_bytes, list(range(150)))
        with pytest.warns(UserWarning, match="mis-framed"):
            pf = load(short(raw, 60), tmp_path)
        _assert_repaired(pf, mr_clean, 60, 1)

    def test_little_endian(self, tmp_path, mr_bytes):
        """The same file byte-swapped to little-endian repairs identically."""
        _, first, _, _ = _geometry(mr_bytes)
        h = np.frombuffer(mr_bytes[:128], ">u2").astype("<u2")
        h[63] = 1
        data = np.frombuffer(mr_bytes[first:], ">u2").astype("<u2")
        le = h.tobytes() + mr_bytes[128:first] + data.tobytes()
        clean = load(le, tmp_path, name="le.p")
        with pytest.warns(UserWarning, match="mis-framed"):
            pf = load(short(le, 9), tmp_path, name="le_damaged.p")
        assert pf.endian == "<"
        _assert_repaired(pf, clean, 9, 1)

    def test_markers_and_rebuilt_slots_together(self, tmp_path):
        """v6.1 sentinel runs elsewhere in a mis-framed file stay reported,
        beside the rebuilt slots."""
        clean = PFile(VMP, deconvolve=False)
        col = int(np.flatnonzero(np.all(clean.matrix == clean.matrix[0], axis=0))[-1])
        raw = _inject_sentinels(VMP.read_bytes(), clean, record=300, column=col, n=20)
        with pytest.warns(UserWarning, match="mis-framed"):
            pf = load(swallow(raw, 100), tmp_path)
        assert pf.framing_report["rebuilt_slots"] == [(100, 2)]
        name = next(n for n, i in pf.bad_buffer_report["confirmed"].items() if len(i["spans"]) == 2)
        spans = pf.bad_buffer_report["confirmed"][name]["spans"]
        per = pf.t_fast.size // pf._record_headers.shape[0]
        assert spans[0] == (100 * per, 2 * per)
        assert spans[1][1] >= 20

    def test_uint16_record_number_wrap(self, tmp_path, mr_bytes, mr_clean):
        raw = set_record_numbers(mr_bytes, list(range(65500, 65500 + 150)))
        with pytest.warns(UserWarning, match="mis-framed"):
            pf = load(short(raw, 60), tmp_path)  # numbers wrap at slot 36
        _assert_repaired(pf, mr_clean, 60, 1)

    def test_vmp_v61(self, tmp_path):
        raw = VMP.read_bytes()
        clean = PFile(VMP, deconvolve=False)
        with pytest.warns(UserWarning, match="mis-framed"):
            pf = load(swallow(raw, 200), tmp_path)
        _assert_repaired(pf, clean, 200, 2)


def _inject_sentinels(raw: bytes, pf: PFile, *, record: int, column: int, n: int) -> bytes:
    """Write the v6.1 bad-buffer sentinel into *n* consecutive scans of a fast
    column, starting at the first scan of data record *record* (0-based)."""
    header, first, size, _ = _geometry(raw)
    out = bytearray(raw)
    for k in range(n):
        o = first + record * size + header + 2 * (k * pf.n_cols + column)
        out[o : o + 2] = struct.pack(f"{_endian(raw)}h", -32753)
    return bytes(out)


class TestRefuse:
    def test_policy_refuse(self, tmp_path, mr_bytes):
        with pytest.raises(ValueError, match=r"mis-framed.*refuse"):
            load(short(mr_bytes, 9), tmp_path, framing="refuse")

    def test_unknown_policy(self):
        with pytest.raises(ValueError, match="framing must be one of"):
            PFile(MR, framing="ignore")

    def test_record_numbers_that_do_not_increase(self, tmp_path, mr_bytes):
        raw = set_record_numbers(mr_bytes, [7] * 150)
        with pytest.raises(ValueError, match="record numbers"):
            load(short(raw, 9), tmp_path)

    def test_damage_before_the_first_header(self, tmp_path, mr_bytes):
        _, first, _, _ = _geometry(mr_bytes)
        raw = mr_bytes[:first] + mr_bytes[first + 128 :]  # first header lost
        with pytest.raises(ValueError, match="before its first data-record header"):
            load(raw, tmp_path)

    def test_counter_reset_is_not_a_wrap(self, tmp_path, mr_bytes):
        """A large backward step (a merge splice, a corrupt number) is refused,
        not unwrapped into a phantom gap of tens of thousands of records."""
        numbers = list(range(40000, 40000 + 150))
        numbers[80:] = range(1, 71)
        raw = set_record_numbers(mr_bytes, numbers)
        with pytest.raises(ValueError, match="record numbers"):
            load(short(raw, 9), tmp_path)

    def test_no_intact_record(self, tmp_path):
        raw = (DATA / "MR_SL435.p").read_bytes()
        for k in range(19, -1, -1):
            raw = short(raw, k)
        with pytest.raises(ValueError, match="no data record is intact"):
            load(raw, tmp_path)

    def test_mostly_damage(self, tmp_path, mr_bytes):
        raw = mr_bytes
        for k in range(140, 0, -3):  # two of every three slots
            raw = swallow(raw, k)
        with pytest.raises(ValueError, match="refused rather than"):
            load(raw, tmp_path)


class TestUnaffected:
    @pytest.mark.parametrize(
        "name", ["MR_SL685_climb.p", "SN479_0006.p", "MR_SL435.p", "VMP142_bench.p"]
    )
    def test_clean_files_take_the_fast_path(self, name):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            pf = PFile(DATA / name)
        assert pf.framing_report == {}
        assert not any("mis-framed" in str(w.message) for w in rec)

    def test_refuse_policy_reads_a_clean_file(self):
        assert PFile(MR, framing="refuse").framing_report == {}

    def test_one_damaged_header_keeps_the_fixed_size_read(self, tmp_path, mr_bytes, mr_clean):
        """A bit flip in one data header is not mis-framing: main read the file
        correctly, so must this."""
        _, first, size, _ = _geometry(mr_bytes)
        raw = bytearray(mr_bytes)
        o = first + 50 * size + 2 * 18  # record_size word of data record 51
        raw[o : o + 2] = struct.pack(">H", 1234)
        with pytest.warns(UserWarning, match="header of data record"):
            pf = load(bytes(raw), tmp_path)
        assert pf.framing_report == {}
        for name, x in mr_clean.channels_raw.items():
            np.testing.assert_array_equal(pf.channels_raw[name], x, err_msg=name)

    def test_patch_setupstr_keeps_the_sentinel_scan(self, tmp_path):
        """With record 0 rewritten as v6.0 by patch_setupstr, the v6.1 data
        records still carry sentinels, and they must still be found."""
        clean = PFile(VMP, deconvolve=False)
        col = int(np.flatnonzero(np.all(clean.matrix == clean.matrix[0], axis=0))[-1])
        raw = bytearray(_inject_sentinels(VMP.read_bytes(), clean, record=300, column=col, n=20))
        raw[20:22] = struct.pack(">H", 0x0600)
        with pytest.warns(UserWarning, match="RDL bad-buffer markers"):
            pf = load(bytes(raw), tmp_path)
        assert pf.bad_buffer_report["confirmed"]

    def test_patch_setupstr_version_mismatch(self, tmp_path):
        """ODAS patch_setupstr rewrites record 0 as v6.0 and copies the data
        records verbatim, so a patched v6.1 file differs from its data headers
        in the version word -- 94 of 308 sampled SeaChest files do. That is not
        mis-framing."""
        raw = bytearray(VMP.read_bytes())
        raw[20:22] = struct.pack(">H", 0x0600)
        p = tmp_path / "patched.p"
        p.write_bytes(bytes(raw))
        pf = PFile(p, deconvolve=False)
        clean = PFile(VMP, deconvolve=False)
        assert pf.framing_report == {}
        for name, x in clean.channels_raw.items():
            np.testing.assert_array_equal(pf.channels_raw[name], x, err_msg=name)
