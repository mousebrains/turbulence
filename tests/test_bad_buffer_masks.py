"""RDL bad-buffer dropouts -> epsilon/chi repair and masks (TN-051 s3.2).

A gap is repaired only if BOTH hold: the channel is consumed as a slowly-varying
scalar (speed, pressure, reference T/C) rather than as a spectrum, and the gap
is at most bad_buffer.MAX_INTERP_S long. Shear, vibration and the FP07
thermistors never qualify -- their high-frequency content IS the measurement.
The duration test is on TIME, not sample count, because the RDL always loses the
same 64-sample buffer: 0.125 s of a fast channel but 1.0 s of a slow one.

The unit tests pin the grading, the repair and the dependency scoping; the
integration tests inject the sentinel into real .p files and check what
actually comes out of epsilon and chi.
"""

import struct
import warnings
from pathlib import Path

import numpy as np
import pytest

from odas_tpw.rsi import bad_buffer as bb
from odas_tpw.rsi.p_file import BAD_BUFFER_SENTINEL, PFile

SRC = Path(__file__).parent / "data" / "SN479_0006.p"
MR_SRC = Path(__file__).parent / "data" / "MR_SL685_climb.p"


def _endian(raw: bytes) -> str:
    for fmt in ("<", ">"):
        if struct.unpack(f"{fmt}64H", raw[:128])[17] == 128:
            return fmt
    raise AssertionError("cannot detect fixture endianness")


def _geometry(raw: bytes) -> dict:
    w = list(struct.unpack(f"{_endian(raw)}64H", raw[:128]))
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


def _inject(raw: bytes, cells) -> bytes:
    """Write the sentinel into (cycle, matrix_row, col) cells."""
    g = _geometry(raw)
    out = bytearray(raw)
    word = struct.pack(f"{_endian(raw)}h", BAD_BUFFER_SENTINEL)
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


def _ch_ids(pf: PFile) -> dict:
    out = {}
    for ch in pf.config["channels"]:
        name = ch.get("name")
        if not name:
            continue
        ids = [int(v) for v in str(ch.get("id", "")).split() if v.strip().lstrip("-").isdigit()]
        if ids:
            out[name] = ids
    return out


def _cell(pf: PFile, channel: str) -> tuple[int, int]:
    """(row, col) of a channel's first matrix occurrence -- what _read keeps."""
    where = np.where(pf.matrix == _ch_ids(pf)[channel][0])
    return int(where[0][0]), int(where[1][0])


def _inject_fast_run(raw: bytes, pf: PFile, channel: str, start_cycle: int, n_samples: int):
    """A run of *n_samples* consecutive samples of a FAST channel.

    A fast channel's samples run down its whole matrix column, so one cycle
    contributes n_rows of them.
    """
    _row, col = _cell(pf, channel)
    g = _geometry(raw)
    n_cycles = n_samples // g["n_rows"]
    cells = [
        (start_cycle + c, row, col) for c in range(n_cycles) for row in range(g["n_rows"])
    ]
    return _inject(raw, cells)


def _inject_slow_run(raw: bytes, pf: PFile, channel: str, start_cycle: int, n_samples: int):
    """A run of *n_samples* consecutive samples of a SLOW channel (one per cycle)."""
    row, col = _cell(pf, channel)
    return _inject(raw, [(start_cycle + i, row, col) for i in range(n_samples)])


@pytest.fixture(scope="module")
def src_bytes() -> bytes:
    return SRC.read_bytes()


@pytest.fixture(scope="module")
def clean_pf() -> PFile:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PFile(SRC)


@pytest.fixture(scope="module")
def profile_cycle(clean_pf) -> int:
    """A matrix cycle inside the fixture's detected profile.

    Computed, not hard-coded: a dropout outside the profile is trimmed away
    before windowing and would make these tests silently vacuous. One slow
    sample == one matrix cycle, so the slow index is the cycle.
    """
    from odas_tpw.rsi.helpers import load_channels, prepare_profiles

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = load_channels(SRC)
        prepared = prepare_profiles(data, None, "down", None)
    profiles_slow = prepared[0]
    assert profiles_slow, "fixture must contain a detectable profile"
    s_slow, e_slow = profiles_slow[0]
    return (s_slow + e_slow) // 2


# ---------------------------------------------------------------------------
# Unit: grading short vs long
# ---------------------------------------------------------------------------


class TestGrading:
    """Two tests gate a repair: the channel's ROLE, then the gap's DURATION."""

    def test_measurement_channels_are_never_interpolated(self):
        """Shear, vibration and the FP07s are consumed as spectra -- their
        high-frequency content is the measurement, so a smooth ramp across the
        gap fabricates spectral content rather than repairing it."""
        spans = [(100, 8)]  # 0.016 s: far below the duration threshold
        report = {
            "confirmed": {
                name: {"rate": "fast", "spans": spans}
                for name in ("sh1", "sh2", "T1_dT1", "T1", "Ax", "Ay")
            }
        }
        masks = bb.sample_masks(report, 8000, 1000, fs_fast=512.0, fs_slow=64.0)
        for name, mask in masks.items():
            assert set(np.unique(mask[100:108])) == {bb.DROPPED}, (
                f"{name} must never be interpolated"
            )

    def test_context_channels_are_interpolated_when_short(self):
        """Speed, pressure and the CT pair are read as slowly-varying scalars."""
        report = {
            "confirmed": {
                name: {"rate": "slow", "spans": [(10, 7)]}  # 0.11 s @ 64 Hz
                for name in sorted(bb.INTERPOLATABLE_CHANNELS)
            }
        }
        masks = bb.sample_masks(report, 8000, 1000, fs_fast=512.0, fs_slow=64.0)
        assert set(masks) == set(bb.INTERPOLATABLE_CHANNELS)
        for name, mask in masks.items():
            assert set(np.unique(mask[10:17])) == {bb.INTERPOLATED}, name

    def test_shear_is_not_in_the_interpolatable_set(self):
        for name in ("sh1", "sh2", "T1", "T1_dT1", "T2_dT2", "Ax", "Ay", "Az"):
            assert name not in bb.INTERPOLATABLE_CHANNELS

    def test_a_long_gap_drops_even_in_a_context_channel(self):
        """1.0 s of U_EM is one RDL buffer at 64 Hz -- too long to bridge."""
        report = {"confirmed": {"U_EM": {"rate": "slow", "spans": [(100, 64)]}}}
        masks = bb.sample_masks(report, 8000, 1000, fs_fast=512.0, fs_slow=64.0)
        assert set(np.unique(masks["U_EM"][100:164])) == {bb.DROPPED}

    def test_boundary_is_inclusive(self):
        report = {
            "confirmed": {
                "P": {"rate": "fast", "spans": [(0, 128)]},  # exactly 0.25 s
                "U_EM": {"rate": "fast", "spans": [(200, 129)]},  # just over
            }
        }
        masks = bb.sample_masks(report, 8000, 1000, fs_fast=512.0, fs_slow=64.0)
        assert masks["P"][0] == bb.INTERPOLATED
        assert masks["U_EM"][200] == bb.DROPPED

    def test_threshold_is_configurable(self):
        report = {"confirmed": {"P": {"rate": "fast", "spans": [(0, 64)]}}}
        strict = bb.sample_masks(
            report, 8000, 1000, fs_fast=512.0, fs_slow=64.0, max_interp_s=0.01
        )
        assert strict["P"][0] == bb.DROPPED

    def test_interpolatable_set_is_configurable(self):
        report = {"confirmed": {"sh1": {"rate": "fast", "spans": [(0, 8)]}}}
        opened = bb.sample_masks(
            report, 8000, 1000, fs_fast=512.0, fs_slow=64.0, interpolatable={"sh1"}
        )
        assert opened["sh1"][0] == bb.INTERPOLATED

    def test_uses_the_declared_rate_not_the_length(self):
        """Deconvolution moves a base channel between axes, so the array length
        cannot identify the axis the scan ran on."""
        report = {
            "confirmed": {
                "slowly": {"rate": "slow", "spans": [(3, 4)]},
                "fastly": {"rate": "fast", "spans": [(5, 2)]},
            }
        }  # names outside INTERPOLATABLE_CHANNELS: grade is irrelevant here
        masks = bb.sample_masks(report, 800, 100, fs_fast=512.0, fs_slow=64.0)
        assert masks["slowly"].size == 100
        assert masks["fastly"].size == 800

    def test_clips_a_span_running_past_the_end(self):
        report = {"confirmed": {"x": {"rate": "slow", "spans": [(98, 50)]}}}
        masks = bb.sample_masks(report, 800, 100, fs_fast=512.0, fs_slow=64.0)
        assert (masks["x"][98:] != bb.CLEAN).all()
        assert int((masks["x"] != bb.CLEAN).sum()) == 2


class TestRepair:
    def test_interpolates_across_a_short_gap(self):
        values = np.arange(100, dtype=np.float64)
        mask = np.zeros(100, dtype=np.int8)
        mask[40:45] = bb.INTERPOLATED
        values_bad = values.copy()
        values_bad[40:45] = -32753.0
        out = bb.repair(values_bad, mask)
        assert out[40:45] == pytest.approx(values[40:45])
        assert out[:40] == pytest.approx(values[:40])

    def test_leaves_dropped_samples_alone(self):
        """Those windows are rejected; inventing data would hide the rejection."""
        values = np.zeros(50)
        values[10:20] = -32753.0
        mask = np.zeros(50, dtype=np.int8)
        mask[10:20] = bb.DROPPED
        out = bb.repair(values, mask)
        assert np.array_equal(out, values)

    def test_does_not_mutate_the_input(self):
        """The arrays are views onto PFile.channels, shared with every other
        consumer."""
        values = np.array([0.0, 1.0, -32753.0, 3.0])
        mask = np.array([0, 0, bb.INTERPOLATED, 0], dtype=np.int8)
        out = bb.repair(values, mask)
        assert values[2] == -32753.0
        assert out[2] == pytest.approx(2.0)

    def test_endpoint_run_holds_flat_rather_than_extrapolating(self):
        values = np.array([-32753.0, -32753.0, 5.0, 6.0])
        mask = np.array([bb.INTERPOLATED, bb.INTERPOLATED, 0, 0], dtype=np.int8)
        out = bb.repair(values, mask)
        assert out[0] == pytest.approx(5.0)
        assert out[1] == pytest.approx(5.0)

    def test_all_bad_channel_is_left_untouched(self):
        values = np.full(10, -32753.0)
        mask = np.full(10, bb.INTERPOLATED, dtype=np.int8)
        assert np.array_equal(bb.repair(values, mask), values)

    def test_length_mismatch_is_a_no_op(self):
        values = np.arange(10.0)
        assert bb.repair(values, np.zeros(5, dtype=np.int8)) is values


# ---------------------------------------------------------------------------
# Unit: the dependency scoping
# ---------------------------------------------------------------------------


class TestSpeedDependency:
    """A dropout only matters where the channel actually feeds the estimate."""

    def _masks(self, grade):
        m = np.zeros(1000, dtype=np.int8)
        m[100:164] = grade
        return {"U_EM": m}

    def _common(self, masks):
        return dict(
            masks=masks,
            probe_names=["sh1", "sh2"],
            shared_names=[],
            t_fast=np.arange(8000) / 512.0,
            t_slow=np.arange(1000) / 64.0,
            fs_fast=512.0,
        )

    def test_flight_model_ignores_u_em_dropout(self):
        """Pat's caveat: under a flight model U_EM is not a speed input, so a
        U_EM dropout must not reject anything."""
        common = self._common(self._masks(bb.DROPPED))
        em, em_prov = bb.probe_masks(speed_names=bb.SPEED_INPUT_CHANNELS["em"], **common)
        flight, flight_prov = bb.probe_masks(
            speed_names=bb.SPEED_INPUT_CHANNELS["flight"], **common
        )
        assert (em == bb.DROPPED).any(), "must reject when the speed came from U_EM"
        assert em_prov == {"U_EM": "speed"}
        assert not flight.any(), "U_EM is not a flight-model input"
        assert flight_prov == {}

    def test_flight_model_still_depends_on_pressure(self):
        """The flight model reads W_slow, hence P, so a P dropout DOES count."""
        assert "P" in bb.SPEED_INPUT_CHANNELS["flight"]
        assert "P" in bb.SPEED_INPUT_CHANNELS["pressure"]
        assert "U_EM" not in bb.SPEED_INPUT_CHANNELS["flight"]

    def test_fixed_and_constant_speed_depend_on_nothing(self):
        assert bb.SPEED_INPUT_CHANNELS["constant"] == ()
        assert bb.speed_channels({"speed_channels": ""}) == ()

    def test_speed_channels_reads_the_stamp_over_the_method(self):
        """The stamp is authoritative: prepare_profiles knows which branch won,
        the method name alone does not (a precomputed speed_fast reads nothing
        from this file even though speed_method may say 'em')."""
        assert bb.speed_channels({"speed_channels": "U_EM", "speed_method": "flight"}) == (
            "U_EM",
        )
        assert bb.speed_channels({"speed_method": "em"}) == ("U_EM",)
        assert bb.speed_channels({}) == ()
        assert bb.speed_channels(None) == ()

    def test_unknown_method_counts_nothing_rather_than_guessing(self):
        assert bb.speed_channels({"speed_method": "hotel:speed"}) == ()


class TestProbeIsolation:
    def test_shear_dropout_affects_only_its_own_probe(self):
        n = 4096
        masks = {"sh1": np.zeros(n, dtype=np.int8)}
        masks["sh1"][100:164] = bb.DROPPED
        out, prov = bb.probe_masks(
            masks=masks,
            probe_names=["sh1", "sh2"],
            shared_names=[],
            t_fast=np.arange(n) / 512.0,
            t_slow=np.arange(n // 8) / 64.0,
            fs_fast=512.0,
        )
        assert int((out[0] == bb.DROPPED).sum()) == 64
        assert not out[1].any()
        assert prov == {"sh1": "probe"}

    def test_vibration_dropout_affects_every_probe_under_goodman(self):
        """Goodman mixes the vibration reference into every shear spectrum."""
        n = 4096
        masks = {"Ax": np.zeros(n, dtype=np.int8)}
        masks["Ax"][10:80] = bb.DROPPED
        out, prov = bb.probe_masks(
            masks=masks,
            probe_names=["sh1", "sh2"],
            shared_names=["Ax"],  # what _build_l1data passes when goodman=True
            t_fast=np.arange(n) / 512.0,
            t_slow=np.arange(n // 8) / 64.0,
            fs_fast=512.0,
        )
        assert int((out[0] == bb.DROPPED).sum()) == 70
        assert int((out[1] == bb.DROPPED).sum()) == 70
        assert prov == {"Ax": "shared"}

    def test_pre_emphasized_probe_inherits_its_base_channel(self):
        """Deconvolution couples T1 and T1_dT1: a dropout in either poisons the
        reconstruction."""
        masks = {"T1": np.zeros(512, dtype=np.int8)}
        masks["T1"][40:48] = bb.DROPPED
        out, prov = bb.probe_masks(
            masks=masks,
            probe_names=["T1_dT1"],
            shared_names=[],
            t_fast=np.arange(4096) / 512.0,
            t_slow=np.arange(512) / 64.0,
            fs_fast=512.0,
        )
        assert out[0].any()
        assert prov == {"T1": "probe"}

    def test_dropped_beats_interpolated_when_channels_combine(self):
        n = 1024
        masks = {
            "sh1": np.zeros(n, dtype=np.int8),
            "Ax": np.zeros(n, dtype=np.int8),
        }
        masks["sh1"][100:200] = bb.INTERPOLATED
        masks["Ax"][150:250] = bb.DROPPED
        out, _ = bb.probe_masks(
            masks=masks,
            probe_names=["sh1"],
            shared_names=["Ax"],
            t_fast=np.arange(n) / 512.0,
            t_slow=np.arange(n // 8) / 64.0,
            fs_fast=512.0,
        )
        assert out[0][120] == bb.INTERPOLATED
        assert out[0][160] == bb.DROPPED  # overlap resolves to the worse grade
        assert out[0][220] == bb.DROPPED


class TestMaskMechanics:
    def test_expand_to_fast_covers_the_interpolation_stencil(self):
        t_slow = np.arange(10) / 1.0
        t_fast = np.arange(90) / 10.0
        m = np.zeros(10, dtype=np.int8)
        m[5] = bb.DROPPED
        fast = bb.expand_to_fast(m, t_slow, t_fast)
        # np.interp at t in (t_slow[4], t_slow[6]) reads the bad sample.
        assert fast[np.searchsorted(t_fast, 4.5)] == bb.DROPPED
        assert fast[np.searchsorted(t_fast, 5.5)] == bb.DROPPED
        assert fast[np.searchsorted(t_fast, 3.0)] == bb.CLEAN

    def test_expand_to_fast_refuses_a_mismatched_mask(self):
        out = bb.expand_to_fast(np.ones(7, dtype=np.int8), np.arange(10.0), np.arange(20.0))
        assert not out.any()

    def test_dilate_widens_only_dropped_runs(self):
        """An interpolated sample carries a plausible value before anything
        downstream sees it, so it no longer smears through the filters."""
        m = np.zeros(100, dtype=np.int8)
        m[50:52] = bb.DROPPED
        assert int((bb.dilate(m, before=5, after=3) == bb.DROPPED).sum()) == 10

        m2 = np.zeros(100, dtype=np.int8)
        m2[50:52] = bb.INTERPOLATED
        assert int((bb.dilate(m2, before=5, after=3) != bb.CLEAN).sum()) == 2

    def test_dilate_clips_at_the_edges(self):
        m = np.zeros(10, dtype=np.int8)
        m[0] = m[9] = bb.DROPPED
        assert int((bb.dilate(m, before=4, after=4) == bb.DROPPED).sum()) == 10

    def test_window_fractions_selects_a_grade(self):
        mask = np.zeros((2, 100), dtype=np.int8)
        mask[0, 10:20] = bb.DROPPED
        mask[0, 30:35] = bb.INTERPOLATED
        dropped = bb.window_fractions(mask, np.array([0, 50]), 50, bb.DROPPED)
        interp = bb.window_fractions(mask, np.array([0, 50]), 50, bb.INTERPOLATED)
        assert dropped[0, 0] == pytest.approx(0.2)
        assert interp[0, 0] == pytest.approx(0.1)
        assert dropped[0, 1] == 0.0
        assert dropped[1].sum() == 0.0

    def test_span_round_trip_keeps_the_grade(self):
        m = np.zeros(200, dtype=np.int8)
        m[10:20] = bb.INTERPOLATED
        m[150:151] = bb.DROPPED
        text = bb.encode_spans(m)
        assert text == "10:10:1,150:1:2"
        assert np.array_equal(bb.decode_spans(text, 200), m)

    def test_abutting_grades_are_not_merged(self):
        m = np.zeros(50, dtype=np.int8)
        m[10:20] = bb.INTERPOLATED
        m[20:30] = bb.DROPPED
        assert np.array_equal(bb.decode_spans(bb.encode_spans(m), 50), m)

    def test_decode_skips_malformed_entries(self):
        out = bb.decode_spans("bogus,,5:3:1,9:x:2,-1:4:2,7:2:9", 20)
        assert int((out != bb.CLEAN).sum()) == 3
        assert (out[5:8] == bb.INTERPOLATED).all()

    def test_encode_empty(self):
        assert bb.encode_spans(np.zeros(10, dtype=np.int8)) == ""


# ---------------------------------------------------------------------------
# Integration: real files with injected dropouts
# ---------------------------------------------------------------------------

# One RDL buffer on a 512 Hz fast channel = 0.125 s -> repairable.
SHORT_FAST = 64
# 0.5 s -> past MAX_INTERP_S, unrepairable.
LONG_FAST = 256


class TestShortGapByChannelRole:
    """A short gap is repaired only in a context channel; in a measurement
    channel it still rejects the window."""

    def test_short_shear_gap_is_rejected_not_interpolated(
        self, src_bytes, clean_pf, tmp_path, profile_cycle
    ):
        """0.125 s is short in time, but shear is consumed as a spectrum:
        bridging it would substitute a smooth ramp for real variance inside the
        band being fitted."""
        from odas_tpw.rsi.dissipation import _compute_epsilon

        p = tmp_path / "short_sh1.p"
        p.write_bytes(_inject_fast_run(src_bytes, clean_pf, "sh1", profile_cycle, SHORT_FAST))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = _compute_epsilon(p)

        hit = [ds for ds in results if float(ds["bad_buffer_fraction"].max()) > 0]
        assert hit, "a short shear gap must still reject its windows"
        ds = hit[0]
        assert float(ds["interpolated_fraction"].max()) == 0.0, (
            "shear must never be interpolated"
        )
        i = [str(v) for v in ds["probe"].values].index("sh1")
        frac = ds["bad_buffer_fraction"].values
        assert np.all(np.isnan(ds["epsilon"].values[i][frac[i] > 0]))

    def test_short_thermistor_gap_is_rejected_not_interpolated(
        self, src_bytes, clean_pf, tmp_path, profile_cycle
    ):
        from odas_tpw.rsi.chi_io import _compute_chi

        p = tmp_path / "short_T1.p"
        p.write_bytes(
            _inject_fast_run(src_bytes, clean_pf, "T1_dT1", profile_cycle, SHORT_FAST)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = _compute_chi(p)
        hit = [ds for ds in results if float(ds["bad_buffer_fraction"].max()) > 0]
        assert hit, "a short FP07 gap must still reject its windows"
        assert float(hit[0]["interpolated_fraction"].max()) == 0.0

    def test_short_speed_gap_is_interpolated_and_the_estimate_kept(self, tmp_path):
        """The case this repair exists for: U_EM feeds a window-mean speed, so
        bridging 0.11 s of it perturbs a scalar and nothing else."""
        from odas_tpw.rsi.dissipation import _compute_epsilon

        raw = MR_SRC.read_bytes()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pf = PFile(MR_SRC)
        p = tmp_path / "short_u_em.p"
        # 7 slow samples at 64 Hz = 0.11 s, the run length the archive scan
        # actually found on glider U_EM channels.
        p.write_bytes(_inject_slow_run(raw, pf, "U_EM", len(pf.t_slow) // 2, 7))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = _compute_epsilon(p, speed_method="em")

        assert results
        assert sum(int((ds["bad_buffer_fraction"].values > 0).sum()) for ds in results) == 0
        touched = sum(
            int((ds["interpolated_fraction"].values > 0).sum()) for ds in results
        )
        assert touched > 0, "a short U_EM gap should be repaired, not ignored"
        for ds in results:
            hit = ds["interpolated_fraction"].values > 0
            assert np.isfinite(ds["epsilon"].values[hit]).all(), (
                "a repaired window must still yield epsilon"
            )

    def test_repair_removes_the_sentinel_from_the_loaded_channel(self, tmp_path):
        """The sentinel converts to 3.52 m/s through the aem1g_d calibration, so
        the repair has to land before anything consumes the channel."""
        from odas_tpw.rsi.helpers import load_channels

        raw = MR_SRC.read_bytes()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pf = PFile(MR_SRC)
        start = len(pf.t_slow) // 2
        p = tmp_path / "u_em_load.p"
        p.write_bytes(_inject_slow_run(raw, pf, "U_EM", start, 7))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bad = PFile(p)
            data = load_channels(p)

        span_start, span_len = bad.bad_buffer_report["confirmed"]["U_EM"]["spans"][0]
        gap = data["U_EM"][span_start : span_start + span_len]
        clean = np.asarray(pf.channels["U_EM"])
        lo, hi = np.nanmin(clean), np.nanmax(clean)
        assert np.isfinite(gap).all()
        assert gap.min() >= lo and gap.max() <= hi, (
            "repaired speed must sit inside the real channel's range"
        )
        # And the raw sentinel value is gone.
        assert np.abs(gap - np.asarray(bad.channels["U_EM"])[span_start]).max() > 0


class TestGoodmanCoupling:
    """A vibration dropout is not a measurement loss, it is a REFERENCE loss:
    Goodman regresses the shear spectra against the vibration spectra, so a
    corrupted accelerometer degrades the cleaning for every probe -- and only
    matters at all when Goodman is running."""

    @staticmethod
    def _accel_dropout(src_bytes, pf, tmp_path, cycle):
        p = tmp_path / "accel.p"
        p.write_bytes(_inject_fast_run(src_bytes, pf, "Ax", cycle, LONG_FAST))
        return p

    def test_accel_dropout_rejects_every_probe_under_goodman(
        self, src_bytes, clean_pf, tmp_path, profile_cycle
    ):
        from odas_tpw.rsi.dissipation import _compute_epsilon

        p = self._accel_dropout(src_bytes, clean_pf, tmp_path, profile_cycle)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = _compute_epsilon(p, goodman=True)
        hit = [ds for ds in results if float(ds["bad_buffer_fraction"].max()) > 0]
        assert hit
        frac = hit[0]["bad_buffer_fraction"].values
        # Every probe, not just one: the vibration reference is shared.
        assert (frac.max(axis=1) > 0).all()
        assert float(hit[0]["interpolated_fraction"].max()) == 0.0

    def test_accel_dropout_is_irrelevant_without_goodman(
        self, src_bytes, clean_pf, tmp_path, profile_cycle
    ):
        """With Goodman off the vibration channels are never read, so the same
        dropout must cost nothing."""
        from odas_tpw.rsi.dissipation import _compute_epsilon

        p = self._accel_dropout(src_bytes, clean_pf, tmp_path, profile_cycle)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = _compute_epsilon(p, goodman=False)
        assert results
        assert sum(int((ds["bad_buffer_fraction"].values > 0).sum()) for ds in results) == 0


class TestEpsilonLongGap:
    @staticmethod
    def _long(src_bytes, pf, tmp_path, cycle, channel="sh1"):
        p = tmp_path / f"long_{channel}.p"
        p.write_bytes(_inject_fast_run(src_bytes, pf, channel, cycle, LONG_FAST))
        return p

    def test_detected_and_graded_as_dropped(
        self, src_bytes, clean_pf, tmp_path, profile_cycle
    ):
        p = self._long(src_bytes, clean_pf, tmp_path, profile_cycle)
        with pytest.warns(UserWarning, match="RDL bad-buffer markers"):
            pf = PFile(p)
        found = pf.bad_buffer_report["confirmed"]["sh1"]
        assert found["rate"] == "fast"
        start, length = found["spans"][0]
        assert length == LONG_FAST
        raw = pf.channels_raw["sh1"]
        assert np.all(np.asarray(raw[start : start + length]).astype(np.int64) == -32753)

    def test_epsilon_rejected_only_on_the_affected_probe(
        self, src_bytes, clean_pf, tmp_path, profile_cycle
    ):
        from odas_tpw.rsi.dissipation import _compute_epsilon

        p = self._long(src_bytes, clean_pf, tmp_path, profile_cycle)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = _compute_epsilon(p)

        hit = [ds for ds in results if float(ds["bad_buffer_fraction"].max()) > 0]
        assert hit, "a 0.5 s gap should reject some window"
        ds = hit[0]
        frac = ds["bad_buffer_fraction"].values
        eps = ds["epsilon"].values
        names = [str(v) for v in ds["probe"].values]
        i = names.index("sh1")
        assert np.all(np.isnan(eps[i][frac[i] > 0]))
        for j, name in enumerate(names):
            if name == "sh1":
                continue
            assert frac[j].max() == 0, f"{name} does not depend on sh1"
            assert np.isfinite(eps[j][frac[i] > 0]).any()

    def test_mask_bad_buffers_false_keeps_the_estimates(
        self, src_bytes, clean_pf, tmp_path, profile_cycle
    ):
        from odas_tpw.rsi.dissipation import _compute_epsilon

        p = self._long(src_bytes, clean_pf, tmp_path, profile_cycle)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kept = _compute_epsilon(p, mask_bad_buffers=False)
        ds = next(d for d in kept if float(d["bad_buffer_fraction"].max()) > 0)
        frac = ds["bad_buffer_fraction"].values
        i = [str(v) for v in ds["probe"].values].index("sh1")
        assert np.isfinite(ds["epsilon"].values[i][frac[i] > 0]).any(), (
            "the fraction must still be reported, but nothing NaN'd"
        )

    def test_masks_survive_the_per_profile_netcdf(
        self, src_bytes, clean_pf, tmp_path, profile_cycle
    ):
        """The prof -> eps route must reject the same windows as the direct
        route, or the handling silently stops at the file boundary."""
        from odas_tpw.rsi.dissipation import _compute_epsilon
        from odas_tpw.rsi.profile import extract_profiles

        p = self._long(src_bytes, clean_pf, tmp_path, profile_cycle)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            direct = _compute_epsilon(p)
            prof_paths = extract_profiles(p, tmp_path / "prof")
            via_nc = [ds for path in prof_paths for ds in _compute_epsilon(path)]

        assert prof_paths and via_nc
        direct_bad = sum(int((ds["bad_buffer_fraction"].values > 0).sum()) for ds in direct)
        nc_bad = sum(int((ds["bad_buffer_fraction"].values > 0).sum()) for ds in via_nc)
        assert direct_bad > 0
        assert nc_bad == direct_bad, "NetCDF round trip lost the dropout"


class TestCleanFile:
    def test_clean_file_reports_zero_and_changes_nothing(self):
        from odas_tpw.rsi.dissipation import _compute_epsilon

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = _compute_epsilon(SRC)
        assert results
        for ds in results:
            assert float(ds["bad_buffer_fraction"].max()) == 0.0
            assert float(ds["interpolated_fraction"].max()) == 0.0


class TestGliderSpeedCaveat:
    """End to end on a real v6.3 glider MicroRider: the same U_EM dropout must
    reject under speed_method='em' and be ignored entirely under 'flight'."""

    @staticmethod
    @pytest.fixture(scope="class")
    def u_em_dropout(tmp_path_factory):
        raw = MR_SRC.read_bytes()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pf = PFile(MR_SRC)
        # 64 slow samples at 64 Hz = 1.0 s, past MAX_INTERP_S.
        mutated = _inject_slow_run(raw, pf, "U_EM", len(pf.t_slow) // 2, 64)
        p = tmp_path_factory.mktemp("mr") / "u_em_dropout.p"
        p.write_bytes(mutated)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            check = PFile(p)
        assert check.bad_buffer_report["confirmed"]["U_EM"]["rate"] == "slow"
        return p

    def _run(self, path, method):
        from odas_tpw.rsi.dissipation import _compute_epsilon

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _compute_epsilon(path, speed_method=method)

    def test_em_speed_rejects_the_windows(self, u_em_dropout):
        results = self._run(u_em_dropout, "em")
        assert results
        assert sum(int((ds["bad_buffer_fraction"].values > 0).sum()) for ds in results) > 0
        assert all(ds.attrs.get("speed_channels") == "U_EM" for ds in results)

    def test_flight_speed_ignores_them(self, u_em_dropout):
        results = self._run(u_em_dropout, "flight")
        assert results
        assert sum(int((ds["bad_buffer_fraction"].values > 0).sum()) for ds in results) == 0
        assert sum(int((ds["interpolated_fraction"].values > 0).sum()) for ds in results) == 0
        assert all(
            ds.attrs.get("speed_channels") == "P,Incl_X,Incl_Y" for ds in results
        )
        assert all(np.isfinite(ds["epsilon"].values).any() for ds in results)


class TestChiMasking:
    def test_long_thermistor_gap_rejects_chi_before_chi_final(
        self, src_bytes, clean_pf, tmp_path, profile_cycle
    ):
        from odas_tpw.rsi.chi_io import _compute_chi

        p = tmp_path / "long_T1.p"
        p.write_bytes(
            _inject_fast_run(src_bytes, clean_pf, "T1_dT1", profile_cycle, LONG_FAST)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = _compute_chi(p)
        hit = [ds for ds in results if float(ds["bad_buffer_fraction"].max()) > 0]
        assert hit, "the injected dropout should reject some chi window"
        ds = hit[0]
        frac = ds["bad_buffer_fraction"].values
        i = [str(v) for v in ds["probe"].values].index("T1_dT1")
        contaminated = frac[i] > 0
        assert np.all(np.isnan(ds["chi"].values[i][contaminated]))
        # epsilon_T and var_resolved go with it, so nothing downstream can
        # resurrect a rejected window through them.
        assert np.all(np.isnan(ds["epsilon_T"].values[i][contaminated]))
        assert np.all(np.isnan(ds["var_resolved"].values[i][contaminated]))

    def test_chi_final_drops_a_fully_masked_window(self):
        """chi_final is formed inside L4 (it is not written to the product), so
        the rejection has to happen before it -- otherwise a contaminated probe
        is averaged back into the reported chi."""
        from odas_tpw.chi.l3_chi import L3ChiData
        from odas_tpw.chi.l4_chi import _process_l4_chi

        n_spec, n_gradt, n_freq = 3, 2, 8
        l3 = L3ChiData(
            time=np.arange(n_spec, dtype=float),
            pres=np.full(n_spec, 10.0),
            temp=np.full(n_spec, 10.0),
            pspd_rel=np.full(n_spec, 0.7),
            section_number=np.ones(n_spec),
            nu=np.full(n_spec, 1.3e-6),
            kappa_T=np.full(n_spec, 1.4e-7),
            kcyc=np.tile(np.linspace(1, 100, n_freq)[:, None], (1, n_spec)),
            freq=np.linspace(1, 100, n_freq),
            gradt_spec=np.full((n_gradt, n_freq, n_spec), 1e-6),
            noise_spec=np.full((n_gradt, n_freq, n_spec), 1e-12),
            H2=np.ones((n_spec, n_freq)),
            tau0=np.full(n_spec, 1e-3),
            bad_fraction=np.zeros((n_gradt, n_spec)),
            interp_fraction=np.zeros((n_gradt, n_spec)),
        )
        l3.bad_fraction[:, 1] = 0.01  # window 1 unrepairable on every probe

        def chi_func(j, ci, *_args, **_kw):
            return (1e-8, 1e-9, 50.0, 40.0, 1.0, 0.9, 0.95)

        out = _process_l4_chi(l3, chi_func, "epsilon", 98.0)
        assert np.all(np.isnan(out.chi[:, 1]))
        assert np.isnan(out.chi_final[1]), "chi_final averaged a rejected probe back in"
        assert np.isfinite(out.chi_final[[0, 2]]).all()

        kept = _process_l4_chi(l3, chi_func, "epsilon", 98.0, mask_bad_buffers=False)
        assert np.isfinite(kept.chi_final).all()

    def test_accumulated_interpolation_rejects_the_window(self):
        """Every gap short enough to repair, but too much of the window
        repaired: the variance loss is bounded by rejecting it anyway."""
        from odas_tpw.chi.l3_chi import L3ChiData
        from odas_tpw.chi.l4_chi import _process_l4_chi
        from odas_tpw.scor160.io import BAD_MAX_INTERP_FRACTION

        n_spec, n_gradt, n_freq = 3, 1, 8
        l3 = L3ChiData(
            time=np.arange(n_spec, dtype=float),
            pres=np.full(n_spec, 10.0),
            temp=np.full(n_spec, 10.0),
            pspd_rel=np.full(n_spec, 0.7),
            section_number=np.ones(n_spec),
            nu=np.full(n_spec, 1.3e-6),
            kappa_T=np.full(n_spec, 1.4e-7),
            kcyc=np.tile(np.linspace(1, 100, n_freq)[:, None], (1, n_spec)),
            freq=np.linspace(1, 100, n_freq),
            gradt_spec=np.full((n_gradt, n_freq, n_spec), 1e-6),
            noise_spec=np.full((n_gradt, n_freq, n_spec), 1e-12),
            H2=np.ones((n_spec, n_freq)),
            tau0=np.full(n_spec, 1e-3),
            bad_fraction=np.zeros((n_gradt, n_spec)),
            interp_fraction=np.zeros((n_gradt, n_spec)),
        )
        l3.interp_fraction[0, 0] = BAD_MAX_INTERP_FRACTION / 2  # tolerable
        l3.interp_fraction[0, 1] = BAD_MAX_INTERP_FRACTION * 2  # too much

        def chi_func(j, ci, *_args, **_kw):
            return (1e-8, 1e-9, 50.0, 40.0, 1.0, 0.9, 0.95)

        out = _process_l4_chi(l3, chi_func, "epsilon", 98.0)
        assert np.isfinite(out.chi[0, 0]), "a lightly repaired window is usable"
        assert np.isnan(out.chi[0, 1]), "a heavily repaired window is not"
        assert np.isfinite(out.chi[0, 2])
