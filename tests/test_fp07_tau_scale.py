"""Per-thermistor FP07 time constant (``fp07_tau_scale``).

Two FP07 beads on one instrument can have materially different response.
Because chi goes as the temperature gradient SQUARED, and because the assumed
tau drives the variance extrapolation wherever the Batchelor rolloff is only
partly resolved, that shows up as a large, persistent chi disagreement between
the two probes -- 1.77x on ARCTERX-2022 SN194 and 0.72x on SN428, in opposite
directions. These tests pin the knob that lets each bead carry its own tau.
"""

from pathlib import Path

import numpy as np
import pytest

from odas_tpw.chi.fp07 import fp07_tau, fp07_transfer
from odas_tpw.chi.l2_chi import L2ChiData
from odas_tpw.chi.l3_chi import L3ChiData, process_l3_chi
from odas_tpw.perturb.config import _validate_instruments, canonical_instruments_for_hash
from odas_tpw.rsi.chi_io import resolve_tau_scales
from odas_tpw.scor160.io import L3Params

FS = 512.0
N = 4096


def _l2(n_temp=2, speed=1.0):
    """Minimal two-thermistor L2ChiData with a broadband gradient signal."""
    rng = np.random.default_rng(20260906)
    return L2ChiData(
        time=np.arange(N) / FS,
        pres=np.linspace(10.0, 60.0, N),
        temp=np.full(N, 15.0),
        temp_fast=np.full((n_temp, N), 15.0) + rng.standard_normal((n_temp, N)) * 1e-3,
        gradt=rng.standard_normal((n_temp, N)) * 1e-3,
        vib=rng.standard_normal((2, N)) * 1e-4,
        pspd_rel=np.full(N, speed),
        # section 0 is "excluded"; use 1 so the windows are actually processed.
        section_number=np.ones(N, dtype=int),
        diff_gains=[0.94] * n_temp,
        fs_fast=FS,
    )


def _params():
    return L3Params(
        fft_length=256,
        diss_length=1024,
        overlap=512,
        fs_fast=FS,
        HP_cut=1.0,
        goodman=False,
    )


def _l3(tau_scale=None):
    return process_l3_chi(_l2(), _params(), tau_scale=tau_scale)


class TestResolveTauScales:
    def test_bead_name_matches_gradient_channel(self):
        assert resolve_tau_scales(["T1_dT1", "T2_dT2"], {"T1": 0.3, "T2": 0.75}) == [0.3, 0.75]

    def test_gradient_channel_name_also_accepted(self):
        assert resolve_tau_scales(["T1_dT1", "T2_dT2"], {"T2_dT2": 1.4}) == [1.0, 1.4]

    def test_absent_mapping_is_none_not_ones(self):
        # None lets process_l3_chi take its untouched default path.
        assert resolve_tau_scales(["T1_dT1"], None) is None
        assert resolve_tau_scales(["T1_dT1"], {}) is None

    @pytest.mark.parametrize("bad", [{"T3": 1.0}, {"T": 1.0}, {"": 1.0}])
    def test_unmatched_key_raises_rather_than_silently_ignoring(self, bad):
        # A typo must not leave a probe uncorrected while the config claims
        # otherwise: chi would be wrong by the square of the response error
        # with no downstream symptom.
        with pytest.raises(ValueError, match="matched"):
            resolve_tau_scales(["T1_dT1", "T2_dT2"], bad)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_non_positive_or_non_finite_raises(self, bad):
        with pytest.raises(ValueError, match="finite and > 0"):
            resolve_tau_scales(["T1_dT1"], {"T1": bad})


class TestL3PerProbeTau:
    def test_default_is_bit_identical_to_no_scaling(self):
        """The whole point: an unset knob must not perturb existing results."""
        a, b = _l3(None), _l3([1.0, 1.0])
        np.testing.assert_array_equal(a.tau0, b.tau0)
        np.testing.assert_array_equal(a.H2, b.H2)
        np.testing.assert_array_equal(a.gradt_spec, b.gradt_spec)

    def test_tau_is_per_probe_shaped(self):
        l3 = _l3()
        assert l3.tau0.shape == (2, l3.n_spectra)
        assert l3.H2.shape == (2, l3.n_spectra, l3.freq.size)

    def test_scale_multiplies_the_model_tau(self):
        l3 = _l3([0.3, 0.75])
        expected = fp07_tau(l3.pspd_rel, model="lueck")
        np.testing.assert_allclose(l3.tau0[0], 0.3 * expected, rtol=1e-12)
        np.testing.assert_allclose(l3.tau0[1], 0.75 * expected, rtol=1e-12)

    def test_speed_dependence_survives_scaling(self):
        """A scale factor on tau(U), not a fixed tau -- so U-dependence remains."""
        fast = process_l3_chi(_l2(speed=1.5), _params(), tau_scale=[0.5, 1.0])
        slow = process_l3_chi(_l2(speed=0.5), _params(), tau_scale=[0.5, 1.0])
        assert np.median(fast.tau0[0]) < np.median(slow.tau0[0])

    def test_H2_matches_the_scaled_tau(self):
        l3 = _l3([0.3, 1.0])
        np.testing.assert_allclose(
            l3.H2[0, 0], fp07_transfer(l3.freq, l3.tau0[0, 0]), rtol=1e-12
        )

    def test_a_faster_bead_gets_less_attenuation(self):
        l3 = _l3([0.3, 1.0])
        # Smaller tau -> higher corner -> |H|^2 nearer 1 at every frequency.
        assert np.all(l3.H2[0, 0] >= l3.H2[1, 0] - 1e-15)
        assert l3.H2[0, 0][-1] > l3.H2[1, 0][-1]

    @pytest.mark.parametrize("bad", [[1.0], [1.0, 1.0, 1.0], [1.0, 0.0], [1.0, -2.0]])
    def test_bad_tau_scale_raises(self, bad):
        with pytest.raises(ValueError, match="tau_scale"):
            _l3(bad)


class TestLegacyLayoutStillReadable:
    """An L3ChiData built by older code (or a test) keeps working."""

    def test_accessors_accept_shared_layout(self):
        n_spec, n_freq = 3, 5
        l3 = L3ChiData(
            time=np.zeros(n_spec),
            pres=np.zeros(n_spec),
            temp=np.zeros(n_spec),
            pspd_rel=np.ones(n_spec),
            section_number=np.zeros(n_spec),
            nu=np.ones(n_spec),
            kappa_T=np.ones(n_spec),
            kcyc=np.ones((n_freq, n_spec)),
            freq=np.arange(n_freq, dtype=float),
            gradt_spec=np.ones((2, n_freq, n_spec)),
            noise_spec=np.zeros((2, n_freq, n_spec)),
            H2=np.tile(np.linspace(1.0, 0.5, n_freq), (n_spec, 1)),
            tau0=np.full(n_spec, 0.007),
        )
        # Both probes see the same shared value, as before.
        assert l3.tau0_for(0, 1) == pytest.approx(0.007)
        assert l3.tau0_for(1, 1) == pytest.approx(0.007)
        np.testing.assert_array_equal(l3.H2_for(0, 2), l3.H2_for(1, 2))

    def test_accessors_select_per_probe_layout(self):
        l3 = _l3([0.3, 1.0])
        assert l3.tau0_for(0, 0) != l3.tau0_for(1, 0)
        assert not np.array_equal(l3.H2_for(0, 0), l3.H2_for(1, 0))


class TestPerturbConfigSurface:
    def test_valid_block_accepted(self):
        _validate_instruments({"SN194": {"fp07_tau_scale": {"T1": 0.30, "T2": 0.75}}})

    @pytest.mark.parametrize(
        "bad",
        [
            {"S": {"fp07_tau_scale": {"T1": 0}}},
            {"S": {"fp07_tau_scale": {"T1": -1}}},
            {"S": {"fp07_tau_scale": {"T1": "x"}}},
            {"S": {"fp07_tau_scale": {"T1": True}}},
            {"S": {"fp07_tau_scale": [1.0]}},
            {"S": {"fp07_tau_scale": {1: 1.0}}},
        ],
    )
    def test_invalid_block_raises(self, bad):
        with pytest.raises(ValueError, match="fp07_tau_scale"):
            _validate_instruments(bad)

    def test_key_order_does_not_change_the_hash(self):
        a = canonical_instruments_for_hash({"S": {"fp07_tau_scale": {"T2": 0.75, "T1": 0.3}}})
        b = canonical_instruments_for_hash({"S": {"fp07_tau_scale": {"T1": 0.3, "T2": 0.75}}})
        assert a == b

    def test_different_tau_changes_the_hash(self):
        """A different tau is a different chi; it must not reuse the old dir."""
        a = canonical_instruments_for_hash({"S": {"fp07_tau_scale": {"T1": 0.3}}})
        b = canonical_instruments_for_hash({"S": {"fp07_tau_scale": {"T1": 0.4}}})
        assert a != b


class TestTauProvenanceOnTheProduct:
    """A chi file must be traceable to the tau that produced it.

    The multipliers are deployment-specific fits, not anything derivable from
    the .p file, so without this attribute the number is unrecoverable from the
    product alone -- and "a different tau is different chi".
    """

    @pytest.mark.skipif(
        not Path("/Volumes/SeaChest/ARCTERX/2022/Interior/VMP/Data/SN194").is_dir(),
        reason="needs the ARCTERX-2022 corpus",
    )
    def test_applied_scales_are_written_and_absent_when_unused(self):
        from odas_tpw.rsi.chi_io import _compute_chi

        src = "/Volumes/SeaChest/ARCTERX/2022/Interior/VMP/Data/SN194/ARC1A032.P"
        kw = dict(fft_length=512, diss_length=2048, spectrum_model="kraichnan")

        with_tau = _compute_chi(src, fp07_tau_scale={"T1": 0.30, "T2": 0.75}, **kw)
        a = with_tau[0].attrs
        assert a["fp07_tau_scale_T1_dT1"] == pytest.approx(0.30)
        assert a["fp07_tau_scale_T2_dT2"] == pytest.approx(0.75)
        assert a["fp07_tau_model"] == "lueck"

        plain = _compute_chi(src, **kw)
        assert not [k for k in plain[0].attrs if k.startswith("fp07_tau_scale_")]
        assert "fp07_tau_model" not in plain[0].attrs

    @pytest.mark.skipif(
        not Path("/Volumes/SeaChest/ARCTERX/2022/Interior/VMP/Data/SN194").is_dir(),
        reason="needs the ARCTERX-2022 corpus",
    )
    def test_an_unscaled_probe_records_no_misleading_one(self):
        from odas_tpw.rsi.chi_io import _compute_chi

        src = "/Volumes/SeaChest/ARCTERX/2022/Interior/VMP/Data/SN194/ARC1A032.P"
        out = _compute_chi(
            src, fp07_tau_scale={"T1": 0.30}, fft_length=512, diss_length=2048
        )
        a = out[0].attrs
        assert "fp07_tau_scale_T1_dT1" in a
        assert "fp07_tau_scale_T2_dT2" not in a
