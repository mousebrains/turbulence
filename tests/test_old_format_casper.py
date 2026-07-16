# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""Old-format (2013-2017 CASPER-era) MicroRider .p support — issue #131.

Synthetic-file tests (the CASPER corpus itself lives on an external volume and
is validated manually) for:

* M11 — a natively-fast base channel (fast T1 + fast T1_dT1 columns of a 4x10
  matrix) is reclassified slow after deconvolution, so ``is_fast()`` matches
  the stored slow-length array and NetCDF conversion no longer crashes.
* M12 — a duplicate-sampled slow pair (P/P_dP twice per scan) is deconvolved
  at the rate of the DECIMATED array (fs_slow), not the 2x matrix-occurrence
  rate, and the existing decimation warning still fires.
* m1 — the ``serial_num`` dialect is honored wherever the instrument SN is
  read (summary, helpers metadata, NetCDF attrs, sensor inventory).
* m5 — ``accel`` with ``coef0 = 0`` / ``coef1 = 1`` rewrites to ``piezo``
  (setupstr.m:478-495 parity, EXACT-string trigger), routing Ax/Ay to the
  VIB role in counts; a genuinely calibrated accel stays ACC in m/s².
* m6 — ``[cruise info]`` (and any unknown section) is kept, normalized to
  ``cruise_info``; config_patch agrees on the normalized name.
* m7 — ``id_even``/``id_odd`` synthesizes the 2-id join identically to an
  explicit ``id = even,odd``; matrix addresses with no usable [channel]
  section warn (address 255, the RSI special character, is exempt).
"""

from __future__ import annotations

import io
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

import odas_tpw.rsi.p_file as p_file_mod
from odas_tpw.rsi.adapter import pfile_to_l1data
from odas_tpw.rsi.helpers import load_channels
from odas_tpw.rsi.p_file import PFile, instrument_sn, parse_config
from tests.test_p_file_branches import _make_minimal_p_file

SN479_FILE = Path(__file__).parent / "data" / "SN479_0006.p"

# ---------------------------------------------------------------------------
# Shared synthetic configs
# ---------------------------------------------------------------------------

# CAS_001-006 layout: 4x10 matrix; T1 and T1_dT1 BOTH full fast columns;
# P / P_dP one occurrence per scan; unused cells hold the RSI special
# address 255 (present in the real CASPER matrices with no section).
_THERM_COEFFS = """adc_fs = 4.096
adc_bits = 16
a = 0
b = 1
g = 6
e_b = 0.682
t_0 = 289
beta_1 = 3000
"""

_CASPER_FAST_T1_CONFIG = f"""
[instrument_info]
model = MR_1000_LP
serial_num = 134

[cruise info]
boat = Elakha

[matrix]
row1 = 10 255 2 3 255 255 255 255 255 255
row2 = 11 255 2 3 255 255 255 255 255 255
row3 = 255 255 2 3 255 255 255 255 255 255
row4 = 255 255 2 3 255 255 255 255 255 255

[channel]
id = 10
name = P
type = poly
coef0 = 0
coef1 = 0.01
units = [dBar]

[channel]
id = 11
name = P_dP
type = poly
diff_gain = 0.9

[channel]
id = 2
name = T1
type = therm
{_THERM_COEFFS}
[channel]
id = 3
name = T1_dT1
type = therm
diff_gain = 0.94
"""


def _make_casper_fast_t1(tmp_path):
    """4x10-matrix file with natively-fast T1 and T1_dT1 (M11 layout)."""
    path = tmp_path / "casper_fast_t1.p"
    # data_words = (928-128)/2 = 400 = 40 scans of 10 -> 10 matrix passes
    # per record; 2 records -> n_slow = 20, n_fast = 80.
    return _make_minimal_p_file(
        path,
        config_text=_CASPER_FAST_T1_CONFIG,
        n_records=2,
        record_size=928,
        fast_cols=8,
        slow_cols=2,
        n_rows=4,
        clock_hz=5120,
    )


# ---------------------------------------------------------------------------
# M11 — natively-fast base channel reclassified after deconvolution
# ---------------------------------------------------------------------------


class TestFastBaseReclassified:
    def test_base_is_slow_after_load(self, tmp_path):
        pf = PFile(_make_casper_fast_t1(tmp_path))
        n_slow, n_fast = len(pf.t_slow), len(pf.t_fast)
        assert (n_slow, n_fast) == (20, 80)
        # The base holds the slow-rate view and must be CLASSIFIED slow;
        # the full fast-rate deconvolved signal lives in T1_dT1.
        assert pf.is_fast("T1") is False
        assert "T1" in pf._slow_channels
        assert len(pf.channels["T1"]) == n_slow
        assert pf.is_fast("T1_dT1") is True
        assert len(pf.channels["T1_dT1"]) == n_fast

    def test_nc_conversion_succeeds(self, tmp_path):
        """The M11 repro: p_to_netcdf crashed on main with a broadcast error
        (slow-length dT/dt divided by fast-length speed in the gradT builder)."""
        netCDF4 = pytest.importorskip("netCDF4")
        from odas_tpw.rsi.convert import p_to_netcdf

        p_path = _make_casper_fast_t1(tmp_path)
        nc_path = tmp_path / "casper_fast_t1.nc"
        p_to_netcdf(p_path, nc_path)
        ds = netCDF4.Dataset(str(nc_path))
        try:
            # m1: the serial_num dialect reaches the NetCDF attribute.
            assert ds.instrument_sn == "134"
            assert "GRADT" in ds.groups["L1_converted"].variables
        finally:
            ds.close()


# ---------------------------------------------------------------------------
# M12 — duplicate-sampled slow pair deconvolved at the decimated array's rate
# ---------------------------------------------------------------------------

_CASPER_DUP_P_CONFIG = """
[instrument_info]
model = MR_1000_LP
serial_num = 134

[matrix]
row1 = 10 255 255 255 255 255 255 255 255 255
row2 = 11 255 255 255 255 255 255 255 255 255
row3 = 10 255 255 255 255 255 255 255 255 255
row4 = 11 255 255 255 255 255 255 255 255 255

[channel]
id = 10
name = P
type = poly
coef0 = 0
coef1 = 0.01
units = [dBar]

[channel]
id = 11
name = P_dP
type = poly
diff_gain = 0.9
"""


class TestDuplicateSampledDeconvolutionRate:
    def test_deconvolve_receives_fs_slow(self, tmp_path, monkeypatch):
        calls: list[tuple[int, float]] = []
        real_deconvolve = p_file_mod.deconvolve

        def spy(X, X_dX, fs, diff_gain):
            calls.append((len(X_dX), fs))
            return real_deconvolve(X, X_dX, fs, diff_gain)

        monkeypatch.setattr("odas_tpw.rsi.p_file.deconvolve", spy)

        path = tmp_path / "casper_dup_p.p"
        _make_minimal_p_file(
            path,
            config_text=_CASPER_DUP_P_CONFIG,
            n_records=2,
            record_size=928,
            fast_cols=8,
            slow_cols=2,
            n_rows=4,
            clock_hz=5120,
        )
        # The existing decimation warning must still flag the data loss.
        with pytest.warns(UserWarning, match="appears 2x per scan matrix"):
            pf = PFile(path)

        n_slow = len(pf.t_slow)
        # Extraction kept the FIRST occurrence per scan -> the stored P_dP is
        # at fs_slow, and deconvolve must see that rate, not the matrix-count
        # 2x fs_slow that mis-blends P and P_dP.
        assert calls == [(n_slow, pytest.approx(pf.fs_slow))]
        assert pf.is_fast("P_dP") is False
        assert len(pf.channels["P_dP"]) == n_slow
        assert len(pf.channels["P"]) == n_slow


# ---------------------------------------------------------------------------
# m1 — serial_num dialect
# ---------------------------------------------------------------------------


class TestSerialNumDialect:
    def test_instrument_sn_helper(self):
        assert instrument_sn({"sn": "479"}) == "479"
        assert instrument_sn({"serial_num": "134"}) == "134"
        # Modern key wins when both are present; empty values fall through.
        assert instrument_sn({"sn": "479", "serial_num": "134"}) == "479"
        assert instrument_sn({"sn": "", "serial_num": "134"}) == "134"
        assert instrument_sn({}) == ""

    def test_summary_reports_serial_num(self, tmp_path):
        pf = PFile(_make_casper_fast_t1(tmp_path))
        buf = io.StringIO()
        with redirect_stdout(buf):
            pf.summary()
        assert "SN 134" in buf.getvalue()

    def test_helpers_metadata_sn(self, tmp_path):
        pf = PFile(_make_casper_fast_t1(tmp_path))
        data = load_channels(pf)
        assert data["metadata"]["sn"] == "134"

    def test_sensor_inventory_platform_sn(self):
        from odas_tpw.rsi.sensor_inventory import _platform_from_instrument

        vehicle, platform_sn = _platform_from_instrument(
            {"vehicle": "slocum_glider", "serial_num": "134"}
        )
        assert (vehicle, platform_sn) == ("slocum_glider", "134")


# ---------------------------------------------------------------------------
# m5 — accel(0,1) -> piezo rewrite (setupstr.m parity)
# ---------------------------------------------------------------------------


class TestAccelToPiezo:
    def test_parse_config_exact_string_trigger(self):
        cfg = """
[channel]
id = 1
name = Ax
type = accel
coef0 = 0
coef1 = 1
"""
        result = parse_config(cfg)
        assert result["channels"][0]["type"] == "piezo"

    @pytest.mark.parametrize(
        "coef0, coef1",
        [
            ("0.0", "1"),  # float-equal but not string-equal: must NOT flip
            ("0", "1.0"),
            ("0.1", "2.5"),  # a real accelerometer calibration
        ],
    )
    def test_no_flip_without_exact_strings(self, coef0, coef1):
        cfg = f"""
[channel]
id = 1
name = Ax
type = accel
coef0 = {coef0}
coef1 = {coef1}
"""
        result = parse_config(cfg)
        assert result["channels"][0]["type"] == "accel"

    def test_no_flip_when_coefs_missing(self):
        cfg = "[channel]\nid = 1\nname = Ax\ntype = accel\ncoef0 = 0\n"
        result = parse_config(cfg)
        assert result["channels"][0]["type"] == "accel"

    def _make_vib_file(self, tmp_path, *, real_accel: bool):
        """4x4-matrix file: slow P/T1 plus fast Ax/Ay.

        ``real_accel=False``: Ax and Ay are CASPER-style accel(0,1) → piezo.
        ``real_accel=True``: Ay carries a genuine accel calibration instead.
        """
        ay_cal = "coef0 = 0.1\ncoef1 = 2.5" if real_accel else "coef0 = 0\ncoef1 = 1"
        config = f"""
[instrument_info]
model = MR_1000_LP
serial_num = 134

[matrix]
row1 = 10 1 2 255
row2 = 11 1 2 255
row3 = 255 1 2 255
row4 = 255 1 2 255

[channel]
id = 10
name = P
type = poly
coef0 = 0
coef1 = 0.01
units = [dBar]

[channel]
id = 11
name = T1
type = therm
{_THERM_COEFFS}
[channel]
id = 1
name = Ax
type = accel
coef0 = 0
coef1 = 1

[channel]
id = 2
name = Ay
type = accel
{ay_cal}
"""
        path = tmp_path / f"vib_{real_accel}.p"
        # data_words = 64 -> 16 scans of 4 per record; 4 records -> n_slow 16.
        return _make_minimal_p_file(
            path,
            config_text=config,
            n_records=4,
            record_size=256,
            fast_cols=3,
            slow_cols=1,
            n_rows=4,
            clock_hz=2048,
        )

    def test_pfile_piezo_counts_and_adapter_vib(self, tmp_path):
        pf = PFile(self._make_vib_file(tmp_path, real_accel=False))
        assert pf.channel_info["Ax"] == {"units": "counts", "type": "piezo"}
        assert pf.channel_info["Ay"] == {"units": "counts", "type": "piezo"}

        l1 = pfile_to_l1data(pf, speed=0.5)
        assert l1.vib_type == "VIB"
        assert l1.n_vib == 2

        # Goodman coherent-noise input: the name-based accel selection in
        # helpers still hands the piezo channels to the epsilon path.
        data = load_channels(pf)
        assert [name for name, _ in data["accel"]] == ["Ax", "Ay"]

    def test_real_accel_mixed_hardware(self, tmp_path):
        """Mixed hardware: Ax accel(0,1)→piezo alongside a genuinely
        calibrated Ay accel. The vibration stack is the union of both
        (Goodman must keep every coherent reference), name-sorted, with the
        ACC label because a true accelerometer is present."""
        pf = PFile(self._make_vib_file(tmp_path, real_accel=True))
        assert pf.channel_info["Ax"] == {"units": "counts", "type": "piezo"}
        assert pf.channel_info["Ay"] == {"units": "m_s-2", "type": "accel"}

        l1 = pfile_to_l1data(pf, speed=0.5)
        assert l1.vib_type == "ACC"
        assert l1.n_vib == 2
        # Deterministic name-sorted order: row 0 = Ax (piezo), row 1 = Ay.
        np.testing.assert_array_equal(l1.vib[0], pf.channels["Ax"])
        np.testing.assert_array_equal(l1.vib[1], pf.channels["Ay"])

    @pytest.mark.skipif(not SN479_FILE.exists(), reason="SN479_0006.p test data not available")
    def test_sn479_adapter_vib(self):
        """Pins the F4 harmonization: modern configs declare Ax/Ay type=piezo,
        so the adapter now routes them as VIB (matching convert.py's
        _classify_channels and the L1-NC round trip), no longer ACC-by-name."""
        pf = PFile(SN479_FILE)
        l1 = pfile_to_l1data(pf, speed=0.6)
        assert l1.vib_type == "VIB"
        assert l1.n_vib == 2


# ---------------------------------------------------------------------------
# m6 — [cruise info] and unknown sections are kept
# ---------------------------------------------------------------------------


class TestSectionNormalization:
    def test_cruise_info_with_space(self):
        cfg = "[cruise info]\nboat = Elakha\ncaptain = Mike Kriz\n"
        result = parse_config(cfg)
        assert result["cruise_info"]["boat"] == "Elakha"
        assert result["cruise_info"]["captain"] == "Mike Kriz"

    def test_unknown_section_kept(self):
        cfg = "[Some Section]\nfoo = bar\n"
        result = parse_config(cfg)
        assert result["some_section"]["foo"] == "bar"

    def test_channels_section_name_does_not_clobber_list(self):
        """A stray '[channels]' stanza must not replace the channel list."""
        cfg = "[channels]\nfoo = bar\n\n[channel]\nid = 1\nname = X\ntype = raw\n"
        result = parse_config(cfg)
        assert isinstance(result["channels"], list)
        assert result["channels"][0]["name"] == "X"

    def test_pfile_keeps_cruise_info(self, tmp_path):
        pf = PFile(_make_casper_fast_t1(tmp_path))
        assert pf.config["cruise_info"] == {"boat": "Elakha"}

    def test_config_patch_edits_spaced_section(self):
        from odas_tpw.rsi.config_patch import EditSpec, edit_config_text

        cfg = "[cruise info]\nboat = Elakha\n"
        spec = EditSpec(
            note="rename boat",
            author="test",
            sections={"cruise_info": {"boat": "Oceanus"}},
        )
        new_text, changes = edit_config_text(cfg, spec, when="2026-07-15")
        assert len(changes) == 1
        assert "boat = Oceanus" in new_text


# ---------------------------------------------------------------------------
# m7 — id_even/id_odd dialect + orphan matrix-address warning
# ---------------------------------------------------------------------------

_TWO_ID_MATRIX = """
[matrix]
row1 = 1 2
row2 = 1 2
row3 = 1 2
row4 = 1 2

[instrument_info]
model = test
sn = 1
"""

_EXPLICIT_TWO_ID = (
    _TWO_ID_MATRIX
    + """
[channel]
id = 1, 2
name = bigval
type = raw
"""
)

_EVEN_ODD_TWO_ID = (
    _TWO_ID_MATRIX
    + """
[channel]
id_even = 1
id_odd = 2
name = bigval
type = raw
"""
)


class TestIdEvenOddDialect:
    def test_even_odd_matches_explicit_id(self, tmp_path):
        # Non-trivial words in both columns so a swapped even/odd order
        # would change the joined 32-bit value.
        n_int16 = (256 - 128) // 2
        rec = np.arange(n_int16, dtype=np.int16)
        rec[1::2] = -1  # high (odd) words -> 0xFFFF

        p_explicit = tmp_path / "explicit.p"
        p_evenodd = tmp_path / "evenodd.p"
        for path, cfg in [(p_explicit, _EXPLICIT_TWO_ID), (p_evenodd, _EVEN_ODD_TWO_ID)]:
            _make_minimal_p_file(
                path,
                config_text=cfg,
                fast_cols=2,
                slow_cols=0,
                n_rows=4,
                n_records=1,
                record_data=rec,
            )
        pf_explicit = PFile(p_explicit)
        pf_evenodd = PFile(p_evenodd)
        assert "bigval" in pf_evenodd.channels
        np.testing.assert_array_equal(
            pf_evenodd.channels["bigval"], pf_explicit.channels["bigval"]
        )

    def test_orphan_matrix_address_warns(self, tmp_path):
        config = """
[matrix]
row1 = 1 9
row2 = 1 9
row3 = 1 9
row4 = 1 9

[channel]
id = 1
name = X
type = raw
"""
        path = tmp_path / "orphan.p"
        _make_minimal_p_file(
            path, config_text=config, fast_cols=2, slow_cols=0, n_rows=4, n_records=1
        )
        with pytest.warns(UserWarning, match=r"matrix address\(es\) 9 have no usable"):
            PFile(path)

    def test_address_255_exempt(self, tmp_path):
        """The RSI special character 255 lives in old matrices with no section
        (read_odas.m inserts a synthetic ch255) — it must not warn."""
        config = """
[matrix]
row1 = 1 255
row2 = 1 255
row3 = 1 255
row4 = 1 255

[channel]
id = 1
name = X
type = raw
"""
        path = tmp_path / "with255.p"
        _make_minimal_p_file(
            path, config_text=config, fast_cols=2, slow_cols=0, n_rows=4, n_records=1
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            PFile(path)
        assert not any("no usable" in str(rec.message) for rec in w)
