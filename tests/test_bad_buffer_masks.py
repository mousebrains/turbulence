"""RDL bad-buffer dropouts -> epsilon/chi masks (TN-051 rev. 2026-01-12 s3.2).

The unit tests pin the dependency logic in odas_tpw.rsi.bad_buffer; the
integration tests inject the sentinel into a real .p file and check that the
estimates which overlap it are actually rejected, that a clean probe is not,
and that the masking survives the per-profile NetCDF round trip.
"""

import struct
import warnings
from pathlib import Path

import numpy as np
import pytest

from odas_tpw.rsi import bad_buffer as bb
from odas_tpw.rsi.p_file import BAD_BUFFER_SENTINEL, PFile

SRC = Path(__file__).parent / "data" / "SN479_0006.p"


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


def _fast_column(pf: PFile, channel: str) -> int:
    """Matrix column of a fast channel (its address fills the whole column)."""
    ids = next(
        info["ids"] for name, info in _ch_config(pf).items() if name == channel
    )
    where = np.where(pf.matrix == int(ids[0]))
    return int(where[1][0])


def _ch_config(pf: PFile) -> dict:
    """Channel name -> {'ids': [...]} from the parsed config."""
    out = {}
    for ch in pf.config["channels"]:
        name = ch.get("name")
        if not name:
            continue
        ids = [int(v) for v in str(ch.get("id", "")).split() if v.strip().lstrip("-").isdigit()]
        if ids:
            out[name] = {"ids": ids}
    return out


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
    before windowing and would make these tests silently vacuous (the first
    draft injected at cycle 200, which sits 48k samples ahead of the cast).
    One slow sample == one matrix cycle, so the slow index is the cycle.
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
# Unit: the dependency logic
# ---------------------------------------------------------------------------


class TestSpeedDependency:
    """The whole point of the module: a dropout only masks what it feeds."""

    def test_flight_model_ignores_u_em_dropout(self):
        """Pat's caveat: under a flight model U_EM is not a speed input, so a
        U_EM dropout must not reject any estimate."""
        n_slow, ratio = 1000, 8
        t_slow = np.arange(n_slow) / 64.0
        t_fast = np.arange(n_slow * ratio) / 512.0
        masks = {"U_EM": np.zeros(n_slow, dtype=bool)}
        masks["U_EM"][100:164] = True

        common = dict(
            masks=masks,
            probe_names=["sh1", "sh2"],
            shared_names=[],
            t_fast=t_fast,
            t_slow=t_slow,
            fs_fast=512.0,
        )
        em, em_prov = bb.probe_masks(
            speed_names=bb.SPEED_INPUT_CHANNELS["em"], **common
        )
        flight, flight_prov = bb.probe_masks(
            speed_names=bb.SPEED_INPUT_CHANNELS["flight"], **common
        )

        assert em.any(), "a U_EM dropout must mask when the speed came from U_EM"
        assert em_prov == {"U_EM": "speed"}
        assert not flight.any(), "U_EM is not a flight-model input"
        assert flight_prov == {}

    def test_flight_model_still_depends_on_pressure(self):
        """The flight model reads W_slow (from P), so a P dropout DOES mask."""
        assert "P" in bb.SPEED_INPUT_CHANNELS["flight"]
        assert "P" in bb.SPEED_INPUT_CHANNELS["pressure"]
        assert "U_EM" not in bb.SPEED_INPUT_CHANNELS["flight"]

    def test_fixed_and_constant_speed_depend_on_nothing(self):
        assert bb.SPEED_INPUT_CHANNELS["constant"] == ()
        assert bb.speed_channels({"speed_channels": ""}) == ()

    def test_speed_channels_reads_the_stamp_over_the_method(self):
        """The stamp is authoritative: prepare_profiles knows which branch won,
        the method name alone does not (a precomputed speed_fast reads nothing
        from this file even though speed_method says 'em')."""
        assert bb.speed_channels({"speed_channels": "U_EM", "speed_method": "flight"}) == (
            "U_EM",
        )
        assert bb.speed_channels({"speed_method": "em"}) == ("U_EM",)
        assert bb.speed_channels({}) == ()
        assert bb.speed_channels(None) == ()

    def test_unknown_method_masks_nothing_rather_than_guessing(self):
        assert bb.speed_channels({"speed_method": "hotel:speed"}) == ()


class TestProbeIsolation:
    def test_shear_dropout_masks_only_its_own_probe(self):
        n = 4096
        t_fast = np.arange(n) / 512.0
        masks = {"sh1": np.zeros(n, dtype=bool)}
        masks["sh1"][100:164] = True
        out, prov = bb.probe_masks(
            masks=masks,
            probe_names=["sh1", "sh2"],
            shared_names=[],
            t_fast=t_fast,
            t_slow=np.arange(n // 8) / 64.0,
            fs_fast=512.0,
        )
        assert out[0].sum() == 64
        assert out[1].sum() == 0
        assert prov == {"sh1": "probe"}

    def test_vibration_dropout_masks_every_probe_under_goodman(self):
        """Goodman mixes the vibration reference into every shear spectrum."""
        n = 4096
        masks = {"Ax": np.zeros(n, dtype=bool)}
        masks["Ax"][10:80] = True
        out, prov = bb.probe_masks(
            masks=masks,
            probe_names=["sh1", "sh2"],
            shared_names=["Ax"],  # what _build_l1data passes when goodman=True
            t_fast=np.arange(n) / 512.0,
            t_slow=np.arange(n // 8) / 64.0,
            fs_fast=512.0,
        )
        assert out[0].sum() == 70
        assert out[1].sum() == 70
        assert prov == {"Ax": "shared"}

    def test_pre_emphasized_probe_inherits_its_base_channel(self):
        """Deconvolution couples T1 and T1_dT1: a dropout in either poisons the
        reconstruction, so the mask must cover both."""
        n_slow, n_fast = 512, 4096
        masks = {"T1": np.zeros(n_slow, dtype=bool)}
        masks["T1"][40:48] = True
        out, prov = bb.probe_masks(
            masks=masks,
            probe_names=["T1_dT1"],
            shared_names=[],
            t_fast=np.arange(n_fast) / 512.0,
            t_slow=np.arange(n_slow) / 64.0,
            fs_fast=512.0,
        )
        assert out[0].any()
        assert prov == {"T1": "probe"}


class TestMaskMechanics:
    def test_sample_masks_uses_the_declared_rate(self):
        """After deconvolution a slow base channel is stored at the fast rate,
        so the length cannot identify the detection axis — the report's
        declared rate must win."""
        report = {
            "confirmed": {
                "slowly": {"rate": "slow", "spans": [(3, 4)]},
                "fastly": {"rate": "fast", "spans": [(5, 2)]},
            }
        }
        masks = bb.sample_masks(report, n_fast=800, n_slow=100)
        assert masks["slowly"].size == 100
        assert masks["fastly"].size == 800

    def test_sample_masks_clips_a_span_running_past_the_end(self):
        report = {"confirmed": {"x": {"rate": "slow", "spans": [(98, 50)]}}}
        masks = bb.sample_masks(report, n_fast=800, n_slow=100)
        assert masks["x"][98:].all()
        assert masks["x"].sum() == 2

    def test_expand_to_fast_covers_the_interpolation_stencil(self):
        t_slow = np.arange(10) / 1.0
        t_fast = np.arange(90) / 10.0
        m = np.zeros(10, dtype=bool)
        m[5] = True
        fast = bb.expand_to_fast(m, t_slow, t_fast)
        # np.interp at t in (t_slow[4], t_slow[6]) reads the bad sample.
        assert fast[np.searchsorted(t_fast, 4.5)]
        assert fast[np.searchsorted(t_fast, 5.5)]
        assert not fast[np.searchsorted(t_fast, 3.0)]

    def test_expand_to_fast_refuses_a_mismatched_mask(self):
        out = bb.expand_to_fast(np.ones(7, dtype=bool), np.arange(10.0), np.arange(20.0))
        assert not out.any()

    def test_dilate_widens_runs(self):
        m = np.zeros(100, dtype=bool)
        m[50:52] = True
        assert bb.dilate(m, before=5, after=3).sum() == 10
        assert bb.dilate(m, 0, 0) is m

    def test_dilate_clips_at_the_edges(self):
        m = np.zeros(10, dtype=bool)
        m[0] = m[9] = True
        assert bb.dilate(m, before=4, after=4).sum() == 10

    def test_window_fractions(self):
        mask = np.zeros((2, 100), dtype=bool)
        mask[0, 10:20] = True
        frac = bb.window_fractions(mask, np.array([0, 50]), 50)
        assert frac[0, 0] == pytest.approx(0.2)
        assert frac[0, 1] == 0.0
        assert frac[1].sum() == 0.0

    def test_span_round_trip(self):
        m = np.zeros(200, dtype=bool)
        m[10:20] = True
        m[150:151] = True
        text = bb.encode_spans(m)
        assert text == "10:10,150:1"
        assert np.array_equal(bb.decode_spans(text, 200), m)

    def test_decode_skips_malformed_entries(self):
        out = bb.decode_spans("bogus,,5:3,9:x,-1:4", 20)
        assert out.sum() == 3
        assert out[5:8].all()

    def test_encode_empty(self):
        assert bb.encode_spans(np.zeros(10, dtype=bool)) == ""


# ---------------------------------------------------------------------------
# Integration: a real file with an injected dropout
# ---------------------------------------------------------------------------


class TestEpsilonMasking:
    @staticmethod
    def _inject_shear(src_bytes, pf, tmp_path, start_cycle, channel="sh1", n=64):
        col = _fast_column(pf, channel)
        g = _geometry(src_bytes)
        cells = [
            (start_cycle + c, row, col)
            for c in range(n // g["n_rows"])
            for row in range(g["n_rows"])
        ]
        p = tmp_path / f"dropout_{channel}.p"
        p.write_bytes(_inject(src_bytes, cells))
        return p

    def test_dropout_is_detected_on_the_shear_channel(
        self, src_bytes, clean_pf, tmp_path, profile_cycle
    ):
        p = self._inject_shear(src_bytes, clean_pf, tmp_path, profile_cycle)
        with pytest.warns(UserWarning, match="RDL bad-buffer markers"):
            pf = PFile(p)
        confirmed = pf.bad_buffer_report["confirmed"]
        assert "sh1" in confirmed
        assert confirmed["sh1"]["rate"] == "fast"
        start, length = confirmed["sh1"]["spans"][0]
        raw = pf.channels_raw["sh1"]
        assert np.all(np.asarray(raw[start : start + length]).astype(np.int64) == -32753)

    def test_epsilon_rejected_only_on_the_affected_probe(
        self, src_bytes, clean_pf, tmp_path, profile_cycle
    ):
        from odas_tpw.rsi.dissipation import _compute_epsilon

        p = self._inject_shear(src_bytes, clean_pf, tmp_path, profile_cycle)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = _compute_epsilon(p)
        assert results, "fixture should yield at least one profile"

        hit = [ds for ds in results if float(ds["bad_buffer_fraction"].max()) > 0]
        assert hit, "the injected dropout should land in some dissipation window"
        ds = hit[0]
        frac = ds["bad_buffer_fraction"].values
        eps = ds["epsilon"].values
        names = [str(v) for v in ds["probe"].values]
        i = names.index("sh1")
        assert frac[i].max() > 0
        assert np.all(np.isnan(eps[i][frac[i] > 0])), "contaminated epsilon must be NaN"
        # The clean probe keeps its estimates in exactly those windows.
        for j, name in enumerate(names):
            if name == "sh1":
                continue
            assert frac[j].max() == 0, f"{name} does not depend on sh1"
            assert np.isfinite(eps[j][frac[i] > 0]).any()

    def test_mask_bad_buffers_false_keeps_the_estimates(
        self, src_bytes, clean_pf, tmp_path, profile_cycle
    ):
        from odas_tpw.rsi.dissipation import _compute_epsilon

        p = self._inject_shear(src_bytes, clean_pf, tmp_path, profile_cycle)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kept = _compute_epsilon(p, mask_bad_buffers=False)
        ds = next(d for d in kept if float(d["bad_buffer_fraction"].max()) > 0)
        frac = ds["bad_buffer_fraction"].values
        eps = ds["epsilon"].values
        i = [str(v) for v in ds["probe"].values].index("sh1")
        assert np.isfinite(eps[i][frac[i] > 0]).any(), (
            "the fraction must still be reported, but nothing NaN'd"
        )

    def test_clean_file_reports_zero_fraction_and_no_nans_from_masking(self, tmp_path):
        from odas_tpw.rsi.dissipation import _compute_epsilon

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = _compute_epsilon(SRC)
        assert results
        for ds in results:
            assert float(ds["bad_buffer_fraction"].max()) == 0.0

    def test_masks_survive_the_per_profile_netcdf(
        self, src_bytes, clean_pf, tmp_path, profile_cycle
    ):
        """The prof -> eps route must reject the same windows as the direct
        route, or the masking silently stops at the file boundary."""
        from odas_tpw.rsi.dissipation import _compute_epsilon
        from odas_tpw.rsi.profile import extract_profiles

        p = self._inject_shear(src_bytes, clean_pf, tmp_path, profile_cycle)
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


class TestChiMasking:
    def test_thermistor_dropout_rejects_chi_before_chi_final(
        self, src_bytes, clean_pf, tmp_path, profile_cycle
    ):
        from odas_tpw.rsi.chi_io import _compute_chi

        col = _fast_column(clean_pf, "T1_dT1")
        g = _geometry(src_bytes)
        cells = [
            (profile_cycle + c, row, col) for c in range(8) for row in range(g["n_rows"])
        ]
        p = tmp_path / "dropout_T1.p"
        p.write_bytes(_inject(src_bytes, cells))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = _compute_chi(p)
        assert results
        hit = [ds for ds in results if float(ds["bad_buffer_fraction"].max()) > 0]
        assert hit, "the injected dropout should land in some chi window"
        ds = hit[0]
        frac = ds["bad_buffer_fraction"].values
        chi = ds["chi"].values
        i = [str(v) for v in ds["probe"].values].index("T1_dT1")
        contaminated = frac[i] > 0
        assert np.all(np.isnan(chi[i][contaminated]))
        # epsilon_T and var_resolved are NaN'd with chi, so no downstream
        # consumer can resurrect a rejected window through them.
        assert np.all(np.isnan(ds["epsilon_T"].values[i][contaminated]))
        assert np.all(np.isnan(ds["var_resolved"].values[i][contaminated]))

    def test_chi_final_drops_a_fully_masked_window(self):
        """chi_final is formed inside L4 (it is not written to the product), so
        the rejection has to happen before it — otherwise a contaminated probe
        gets averaged back into the reported chi."""
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
        )
        # Window 1 contaminated on every probe; windows 0 and 2 clean.
        l3.bad_fraction[:, 1] = 0.01

        def chi_func(j, ci, *_args, **_kw):
            return (1e-8, 1e-9, 50.0, 40.0, 1.0, 0.9, 0.95)

        out = _process_l4_chi(l3, chi_func, "epsilon", 98.0)
        assert np.all(np.isnan(out.chi[:, 1]))
        assert np.isnan(out.chi_final[1]), "chi_final averaged a rejected probe back in"
        assert np.isfinite(out.chi_final[[0, 2]]).all()

        kept = _process_l4_chi(l3, chi_func, "epsilon", 98.0, mask_bad_buffers=False)
        assert np.isfinite(kept.chi_final).all()
