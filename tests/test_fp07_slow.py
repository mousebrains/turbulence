"""A bead's slow parallel response path, divided out of its chi (chi/fp07_slow.py, #252)."""

import copy
import datetime as dt
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from odas_tpw.chi import fp07_slow as S
from odas_tpw.chi.batchelor import batchelor_kB, kraichnan_grad
from odas_tpw.chi.fp07 import fp07_tau, fp07_transfer
from odas_tpw.chi.l3_chi import L3ChiData
from odas_tpw.chi.l4_chi import process_l4_chi_epsilon, process_l4_chi_fit
from odas_tpw.scor160.io import L4Data

BETA, TAU_MS = 0.124, 110.0  # T2005's common path (issue #252)
SPAN = {"beta": BETA, "tau_s_ms": TAU_MS, "first": "2021-06-29T18:00:00Z",
        "last": "2022-06-20T08:30:00Z"}


def _span(beta=BETA, tau_ms=TAU_MS):
    return S.SlowSpan("T2005", beta, tau_ms, 0.0, 1.0)


# --- the response ------------------------------------------------------------


class TestResponse:
    def test_closed_form_is_the_complex_response(self):
        f = np.geomspace(1e-3, 300, 200)
        h = (1 - BETA) + BETA / (1 + 1j * 2 * np.pi * f * TAU_MS / 1000)
        np.testing.assert_allclose(S.slow_h2(f, BETA, TAU_MS / 1000), np.abs(h) ** 2, rtol=1e-12)

    def test_limits(self):
        assert S.slow_h2(0.0, BETA, 0.11) == pytest.approx(1.0, abs=1e-15)
        assert S.slow_h2(1e9, BETA, 0.11) == pytest.approx((1 - BETA) ** 2, rel=1e-12)

    def test_monotone_decreasing(self):
        e = S.slow_h2(np.linspace(0, 256, 1000), BETA, 0.11)
        assert np.all(np.diff(e) < 0)

    def test_beta_zero_is_the_identity(self):
        np.testing.assert_array_equal(S.slow_h2(np.linspace(0, 256, 129), 0.0, 0.11), 1.0)

    def test_guard_prediction(self):
        sp = _span()
        lo, hi = S.slow_h2(np.array([0.25, 2.0]), BETA, 0.11)
        assert sp.ln_change() == pytest.approx(np.log(hi / lo))
        assert sp.ln_change() == pytest.approx(-0.159, abs=1e-3)  # the coordinator's figure


# --- configuration -----------------------------------------------------------


class TestParse:
    def test_one_mapping_or_a_list(self):
        one = S.parse_slow_response({"T2005": SPAN})
        two = S.parse_slow_response({"T2005": [SPAN]})
        assert one == two
        (sp,) = one["T2005"]
        assert sp.beta == BETA and sp.tau_s == pytest.approx(0.110)
        assert sp.first == dt.datetime(2021, 6, 29, 18, tzinfo=dt.UTC).timestamp()

    def test_none_and_empty_are_no_correction(self):
        assert S.parse_slow_response(None) == {}
        assert S.parse_slow_response({}) == {}

    def test_spans_sorted_and_disjoint(self):
        a = dict(SPAN, first="2022-01-01", last="2022-02-01")
        b = dict(SPAN, first="2021-07-01", last="2021-08-01")
        spans = S.parse_slow_response({"T2005": [a, b]})["T2005"]
        assert spans[0].first < spans[1].first
        c = dict(SPAN, first="2021-07-15", last="2021-09-01")  # overlaps b
        with pytest.raises(ValueError, match="overlap"):
            S.parse_slow_response({"T2005": [b, c]})

    @pytest.mark.parametrize(
        ("bad", "match"),
        [
            ({"T2": SPAN, "x": 1}, "not an FP07 bead serial"),
            ({"2005": SPAN}, "not an FP07 bead serial"),
            ({"T2005": dict(SPAN, beta=1.0)}, r"beta must be in \[0, 1\)"),
            ({"T2005": dict(SPAN, beta=-0.1)}, r"beta must be in \[0, 1\)"),
            ({"T2005": dict(SPAN, beta=True)}, "beta must be a number"),
            ({"T2005": dict(SPAN, beta=float("nan"))}, "beta must be finite"),
            ({"T2005": dict(SPAN, tau_s_ms=0)}, "tau_s_ms must be > 0"),
            ({"T2005": dict(SPAN, tau_s_ms="110")}, "tau_s_ms must be a number"),
            ({"T2005": {k: v for k, v in SPAN.items() if k != "last"}}, "missing"),
            ({"T2005": dict(SPAN, last=None)}, "missing"),
            ({"T2005": dict(SPAN, tau_s=0.11)}, "unknown key"),
            ({"T2005": dict(SPAN, first="2022-07-01")}, "must be before last"),
            ({"T2005": dict(SPAN, first="last june")}, "not an ISO-8601 time"),
            ({"T2005": []}, "non-empty list"),
            ({"T2005": 0.11}, "non-empty list"),
            (["T2005"], "must be a mapping"),
        ],
    )
    def test_refused(self, bad, match):
        with pytest.raises(ValueError, match=match):
            S.parse_slow_response(bad)

    def test_quoted_and_unquoted_times_hash_alike(self):
        from ruamel.yaml import YAML

        quoted = YAML().load(
            'T2005: {beta: 0.124, tau_s_ms: 110, first: "2021-06-29T18:00:00Z", '
            'last: "2022-06-20"}'
        )
        bare = YAML().load(
            "T2005: {beta: 0.124, tau_s_ms: 110.0, first: 2021-06-29T18:00:00Z, last: 2022-06-20}"
        )
        assert not isinstance(bare["T2005"]["last"], str)  # YAML made a date of it
        assert S.canonical_for_hash(quoted) == S.canonical_for_hash(bare)
        import json

        json.dumps(S.canonical_for_hash(bare))  # no date object reaches json

    def test_config_form_round_trips(self):
        plain = S.config_form({"T2005": SPAN})
        assert S.parse_slow_response(plain) == S.parse_slow_response({"T2005": SPAN})
        assert isinstance(plain["T2005"][0]["first"], str)

    def test_a_fraction_of_a_second_is_applied_as_hashed(self):
        # The workers get config_form; it must not truncate what the hash kept.
        cfg = {"T2005": dict(SPAN, last="2022-06-20T08:30:00.900Z")}
        (hashed,) = S.canonical_for_hash(cfg)["T2005"]
        assert hashed["last"] == pytest.approx(1655713800.9, abs=1e-6)
        (applied,) = S.parse_slow_response(S.config_form(cfg))["T2005"]
        assert applied.last == pytest.approx(hashed["last"], abs=1e-6)
        assert S.select({"T2005": [applied]}, ["T1", "T2005"], 1655713800.5).applied
        assert applied.span_text().endswith("08:30:00.900000+00:00")
        # whole seconds still print as seconds
        assert S.config_form({"T2005": SPAN})["T2005"][0]["last"] == "2022-06-20T08:30:00+00:00"


class TestSelect:
    SPANS = S.parse_slow_response({"T2005": SPAN})
    T_IN = dt.datetime(2021, 7, 2, tzinfo=dt.UTC).timestamp()
    T_OUT = dt.datetime(2022, 7, 4, tzinfo=dt.UTC).timestamp()

    def test_bead_inside_its_span(self):
        sel = S.select(self.SPANS, ["T1592", "T2005"], self.T_IN)
        assert list(sel.applied) == [1] and not sel.outside
        assert sel.serials == ["T1592", "T2005"]

    def test_bead_outside_every_span(self):
        sel = S.select(self.SPANS, ["T1592", "T2005"], self.T_OUT)
        assert not sel.applied and sel.outside == {1: "T2005"}

    def test_file_without_the_bead_is_untouched(self):
        sel = S.select(self.SPANS, ["T1592", "T1593"], self.T_IN)
        assert not sel.applied and not sel.outside
        assert sel.serials == ["T1592", "T1593"]  # still known, for the run summary

    def test_bead_on_two_channels_inside_its_span_is_refused(self):
        with pytest.raises(ValueError, match="on 2 thermistor channels"):
            S.select(self.SPANS, ["T2005", "T2005"], self.T_IN)

    def test_bead_on_two_channels_outside_every_span_is_only_recorded(self):
        # A copy-paste placeholder serial on both beads: nothing would be
        # applied, so the file keeps its chi.
        sel = S.select(self.SPANS, ["T2005", "T2005"], self.T_OUT)
        assert not sel.applied and sel.outside == {0: "T2005", 1: "T2005"}

    def test_span_is_inclusive(self):
        (sp,) = self.SPANS["T2005"]
        assert S.select(self.SPANS, ["T1", "T2005"], sp.first).applied
        assert S.select(self.SPANS, ["T1", "T2005"], sp.last).applied


# --- chi: synthetic spectra --------------------------------------------------

FS, NFFT, W, NU, KAPPA, EPS, DOF = 512.0, 256, 0.7, 1.2e-6, 1.4e-7, 1e-9, 13.3


def _synth(slow: bool, n: int = 300, seed: int = 1) -> tuple[L3ChiData, np.ndarray, float]:
    """Two probes seeing one Kraichnan field with chi spanning the detection limit.

    Probe 2's bead has the slow path when *slow*; the chi-square draws are the
    same either way, so the pair of datasets differs only by the bead.
    """
    rng = np.random.default_rng(seed)
    F = np.arange(NFFT // 2 + 1) * FS / NFFT
    K = F / W
    kB = float(batchelor_kB(EPS, NU, KAPPA))
    tau0 = float(fp07_tau(W, model="lueck"))
    H2 = fp07_transfer(F, tau0)
    e = S.slow_h2(F, BETA, TAU_MS / 1000) if slow else np.ones_like(F)
    chi_true = np.logspace(-10.5, -7, n)  # S/N at the peak ~0.3 .. 1000
    N = np.full(F.size, 3.4e-6)
    X = rng.chisquare(DOF, (2, F.size, n)) / DOF
    spec = np.empty((2, F.size, n))
    for j in range(n):
        B = kraichnan_grad(K, kB, chi_true[j], kappa_T=KAPPA)
        spec[0, :, j] = (B * H2 + N) * X[0, :, j]
        spec[1, :, j] = (B * H2 * e + N) * X[1, :, j]
    l3 = L3ChiData(
        time=np.arange(n, dtype=float), pres=np.linspace(10, 60, n), temp=np.full(n, 20.0),
        pspd_rel=np.full(n, W), section_number=np.ones(n), nu=np.full(n, NU),
        kappa_T=np.full(n, KAPPA), kcyc=np.tile(K[:, None], (1, n)), freq=F,
        gradt_spec=spec, noise_spec=np.tile(N[None, :, None], (2, 1, n)),
        H2=np.tile(H2[None, None, :], (2, n, 1)), tau0=np.full((2, n), tau0),
        diff_gains=[0.94, 0.94], dof_spec=np.full(n, DOF),
    )
    return l3, chi_true, kB


def _l4eps(l3):
    n = l3.n_spectra
    return L4Data(time=l3.time.copy(), pres=l3.pres.copy(), pspd_rel=l3.pspd_rel.copy(),
                  section_number=np.ones(n), epsi=np.full((2, n), EPS),
                  epsi_final=np.full(n, EPS), epsi_flags=np.zeros((2, n)),
                  fom=np.ones((2, n)), mad=np.zeros((2, n)), kmax=np.full((2, n), 50.0),
                  method=np.zeros((2, n)), var_resolved=np.ones((2, n)))


def _method1(l3):
    return process_l4_chi_epsilon(l3, _l4eps(l3))


def _corrected(l3):
    out = copy.deepcopy(l3)
    sel = S.SlowSelection(applied={1: _span()})
    S.apply_to_l3(out, sel, 2)
    return out


def _med_ln(x, truth):
    ok = np.isfinite(x)
    return float(np.median(np.log(x[ok] / truth[ok])))


@pytest.fixture(scope="module")
def synth():
    slow, chi_true, kB = _synth(slow=True)
    ref, _, _ = _synth(slow=False)
    return slow, ref, chi_true, kB


class TestChiRecoversTruth:
    """Moderate S/N spanning the detection limit, both methods."""

    @pytest.mark.parametrize("method", ["epsilon", "fit"])
    def test_corrected_matches_a_bead_without_the_path(self, synth, method):
        slow, ref, chi_true, kB = synth
        run = _method1 if method == "epsilon" else process_l4_chi_fit
        unc, cor, base = run(slow), run(_corrected(slow)), run(ref)
        # The reference is the same field seen by a bead WITHOUT the path: what
        # the estimator gives with a healthy bead, its own small bias included.
        b = _med_ln(base.chi[1], chi_true)
        assert _med_ln(cor.chi[1], chi_true) == pytest.approx(b, abs=0.02)
        # not vacuous: uncorrected, the bead reads ~20-30% low
        assert _med_ln(unc.chi[1], chi_true) < b - 0.2
        if method == "fit":
            # Method 2 fits kB on the corrected spectrum: kB and epsilon_T match
            # the healthy bead's to one step of the MLE's fine kB grid, on which
            # their medians are quantized (~1.5% in kB: ln 0.014 in kB, 0.055 in
            # epsilon_T ~ kB^4; seeds 1-5 land 0 or 1 step apart). There is no
            # reliable "uncorrected is off" check for kB: with bin 1 out of the
            # band (issue #191) the first bin used is 4 Hz, where this path's
            # |H_slow|^2 is already 0.79 of an asymptotic 0.77, a nearly flat
            # factor that chi's amplitude carries (the chi check above), not
            # the spectral shape kB is fitted to.
            kb_ref = _med_ln(base.kB[1], np.full_like(chi_true, kB))
            eps_ref = _med_ln(base.epsilon_T[1], np.full_like(chi_true, EPS))
            assert _med_ln(cor.kB[1], np.full_like(chi_true, kB)) == pytest.approx(kb_ref,
                                                                                   abs=0.02)
            assert _med_ln(cor.epsilon_T[1], np.full_like(chi_true, EPS)) == pytest.approx(
                eps_ref, abs=0.08)

    def test_method2_mle_divides_once(self, synth):
        """The MLE path takes the factor once: applied twice, chi reads high."""
        slow, ref, chi_true, _kB = synth

        def run(l3):
            return process_l4_chi_fit(l3, fit_method="mle")

        cor = _corrected(slow)
        twice = _corrected(slow)
        once = twice.slow_factor[1]
        twice.slow_factor[1] = lambda F: once(F) ** 2
        b = _med_ln(run(ref).chi[1], chi_true)
        # measured: ref +0.025, corrected +0.034, uncorrected -0.228, twice +0.297
        assert _med_ln(run(cor).chi[1], chi_true) == pytest.approx(b, abs=0.03)
        assert _med_ln(run(slow).chi[1], chi_true) < b - 0.15
        assert _med_ln(run(twice).chi[1], chi_true) > b + 0.15

    def test_method1_invariants(self, synth):
        """Dividing S and N by one e keeps every bin's S/N (Method 1: kB fixed)."""
        slow = synth[0]
        unc, cor = _method1(slow), _method1(_corrected(slow))
        # the other probe is untouched, bit for bit
        for name in ("chi", "kB", "K_max", "fom", "K_max_ratio", "epsilon_T", "mad_shape"):
            np.testing.assert_array_equal(getattr(cor, name)[0], getattr(unc, name)[0])
        # the gates, the band and K_max are unchanged
        np.testing.assert_array_equal(np.isfinite(cor.chi), np.isfinite(unc.chi))
        np.testing.assert_array_equal(cor.K_max, unc.K_max)
        np.testing.assert_array_equal(cor.K_max_ratio, unc.K_max_ratio)
        np.testing.assert_array_equal(cor.chi_qc_fallback, unc.chi_qc_fallback)
        # chi rises by a weighted mean of 1/e over the band: within
        # [1/e(first bin), 1/e(inf) = (1-beta)^-2]
        r = cor.chi[1] / unc.chi[1]
        r = r[np.isfinite(r)]
        lo = 1 / _span().factor(slow.freq[1:2])[0]
        assert np.all(r >= lo - 1e-12) and np.all(r <= (1 - BETA) ** -2 + 1e-12)
        # fom moves only because e is not flat in band (0.85 -> 0.77): by < 0.2%
        with np.errstate(invalid="ignore"):
            fr = cor.fom[1] / unc.fom[1]
        assert np.nanmax(np.abs(fr - 1)) < 2e-3

    def test_no_factor_is_byte_identical(self, synth):
        slow = synth[0]
        none = copy.deepcopy(slow)
        none.slow_factor = [None, None]
        a, b = _method1(slow), _method1(none)
        for name in ("chi", "kB", "fom", "K_max_ratio", "chi_final"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))

    def test_beta_zero_changes_nothing(self, synth):
        slow = synth[0]
        zero = copy.deepcopy(slow)
        S.apply_to_l3(zero, S.SlowSelection(applied={1: _span(beta=0.0)}), 2)
        np.testing.assert_array_equal(_method1(zero).chi, _method1(slow).chi)

    def test_l3_is_left_untouched(self, synth):
        slow = synth[0]
        cor = _corrected(slow)
        for name in ("gradt_spec", "noise_spec", "H2"):
            np.testing.assert_array_equal(getattr(cor, name), getattr(slow, name))


# --- the guard ----------------------------------------------------------------


def _gradients(beta=BETA, tau=TAU_MS / 1000, seconds=1400.0, seed=0):
    """(fast bead, slow bead) gradient series seeing one red temperature field."""
    from scipy.signal import lfilter

    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    T = np.cumsum(rng.standard_normal(n)) * 1e-3
    a = np.exp(-1 / (FS * tau))
    T2 = (1 - beta) * T + beta * lfilter([1 - a], [1, -a], T)
    g = lambda x: np.diff(x, append=x[-1]) * FS / W  # noqa: E731
    return g(T), g(T2)


class TestGuard:
    def test_recovers_the_path_and_a_swap_flips_its_sign(self):
        fast, slow = _gradients()
        sec = np.ones(fast.size)
        pred = _span().ln_change()
        lo, hi = S.window_log_ratios(slow, fast, sec, FS)
        assert lo.size >= S.SLOW_GUARD_MIN_WINDOWS
        assert S.guard_delta(lo, hi) == pytest.approx(pred, abs=0.01)
        assert S.guard_verdict(S.guard_delta(lo, hi), pred, lo.size) == "ok"
        lo, hi = S.window_log_ratios(fast, slow, sec, FS)  # the probes swapped
        assert S.guard_delta(lo, hi) == pytest.approx(-pred, abs=0.01)
        assert S.guard_verdict(S.guard_delta(lo, hi), pred, lo.size).startswith("WARNING")

    def test_a_flat_gain_is_not_a_slow_path(self):
        fast, _ = _gradients()
        lo, hi = S.window_log_ratios(1.3 * fast, fast, np.ones(fast.size), FS)
        assert S.guard_delta(lo, hi) == pytest.approx(0.0, abs=1e-9)

    def test_windows_follow_sections_and_finite_samples(self):
        fast, slow = _gradients(seconds=100.0)
        sec = np.ones(fast.size)
        sec[: int(10 * FS)] = 0  # excluded
        sec[int(50 * FS):] = 2  # a second profile: 50 s -> one 32-s and one 18-s window
        slow = slow.copy()
        slow[int(95 * FS)] = np.nan  # the last 5 s cannot join it
        lo, _ = S.window_log_ratios(slow, fast, sec, FS)
        # section 1: 40 s -> 32 + 8 (dropped, < 12 s); section 2: 45 s -> 32 + 13
        assert lo.size == 3

    def test_too_few_windows_give_no_verdict(self):
        v = S.guard_verdict(-0.6, -0.16, S.SLOW_GUARD_MIN_WINDOWS - 1)
        assert v.startswith("no verdict")
        assert S.guard_verdict(float("nan"), -0.16, 100).startswith("no verdict")


def _sections(fast, slow, n_sections=1):
    """chi_io's (l2_chi, l3_chi) pairs, reduced to what the guard reads."""
    out = []
    for part in np.array_split(np.arange(fast.size), n_sections):
        l2 = SimpleNamespace(gradt=np.vstack([fast[part], slow[part]]),
                             section_number=np.ones(part.size), fs_fast=FS)
        out.append((l2, SimpleNamespace()))
    return out


class TestApplySlowResponse:
    NAMES = ("T1_dT1", "T2_dT2")
    SPANS = S.parse_slow_response({"T2005": SPAN})
    T_IN = dt.datetime(2021, 7, 2, tzinfo=dt.UTC).timestamp()

    def _apply(self, serials, fast, slow, names=None, t=None):
        from odas_tpw.rsi.chi_io import _apply_slow_response

        names = list(names or self.NAMES)
        sel = S.select(self.SPANS, serials, self.T_IN if t is None else t)
        secs = _sections(fast, slow)
        if len(names) != 2:
            for l2, _ in secs:
                l2.gradt = np.vstack([l2.gradt] + [l2.gradt[:1]] * (len(names) - 2))
        return _apply_slow_response(sel, self.SPANS, secs, names, "x.p"), secs

    def test_records_itself_and_attaches_the_factor(self):
        fast, slow = _gradients()
        attrs, secs = self._apply(["T1592", "T2005"], fast, slow)
        a = S.ATTR
        assert attrs[f"{a}_applied"] == 1
        assert attrs[f"{a}_bead_T2_dT2"] == "T2005"
        assert attrs[f"{a}_beta_T2_dT2"] == BETA and attrs[f"{a}_tau_s_ms_T2_dT2"] == TAU_MS
        assert attrs[f"{a}_span_T2_dT2"] == "2021-06-29T18:00:00+00:00..2022-06-20T08:30:00+00:00"
        assert attrs[f"{a}_guard_T2_dT2"] == "ok"
        assert attrs[f"{a}_delta_predicted_T2_dT2"] == pytest.approx(_span().ln_change())
        assert attrs[f"{a}_guard_windows_T2_dT2"] == attrs[f"{a}_guard_lnr_lo_T2_dT2"].size
        assert not any(k.endswith("T1_dT1") for k in attrs)
        (_l2, l3), = secs
        assert l3.slow_factor[0] is None
        np.testing.assert_allclose(l3.slow_factor[1](np.array([0.0, 1e9])),
                                   [1.0, (1 - BETA) ** 2])

    def test_swapped_probe_warns_but_still_applies(self, caplog):
        fast, slow = _gradients()
        with caplog.at_level(logging.WARNING, logger="odas_tpw.rsi.chi_io"):
            # T2005 is configured on T2 but the slow bead is really on T1
            attrs, secs = self._apply(["T1592", "T2005"], slow, fast)
        assert attrs[f"{S.ATTR}_applied"] == 1
        assert attrs[f"{S.ATTR}_guard_T2_dT2"].startswith("WARNING")
        assert attrs[f"{S.ATTR}_delta_measured_T2_dT2"] > 0  # the sign flipped
        assert "fp07_slow_response T2005 on T2_dT2" in caplog.text
        assert secs[0][1].slow_factor[1] is not None

    def test_outside_every_span_is_recorded_not_applied(self):
        fast, slow = _gradients()
        t = dt.datetime(2022, 7, 4, tzinfo=dt.UTC).timestamp()
        attrs, secs = self._apply(["T1592", "T2005"], fast, slow, t=t)
        a = S.ATTR
        assert attrs[f"{a}_applied"] == 0
        assert "outside every configured span" in attrs[f"{a}_reason"]
        assert f"{a}_beta_T2_dT2" not in attrs
        # the measured change is still recorded, for QC
        assert attrs[f"{a}_delta_measured_T2_dT2"] == pytest.approx(_span().ln_change(), abs=0.01)
        assert not hasattr(secs[0][1], "slow_factor")

    def test_no_bead_records_only_the_beads_it_carries(self):
        fast, slow = _gradients(seconds=60.0)
        attrs, secs = self._apply(["T1592", "T1593"], fast, slow)
        assert attrs == {f"{S.ATTR}_serials": "T1592/T1593"}
        assert not hasattr(secs[0][1], "slow_factor")

    def test_same_bead_on_both_channels_outside_its_span_is_recorded(self):
        fast, slow = _gradients(seconds=60.0)
        t = dt.datetime(2022, 7, 4, tzinfo=dt.UTC).timestamp()
        attrs, secs = self._apply(["T2005", "T2005"], fast, slow, t=t)
        assert attrs[f"{S.ATTR}_applied"] == 0
        assert attrs[f"{S.ATTR}_bead_T1_dT1"] == attrs[f"{S.ATTR}_bead_T2_dT2"] == "T2005"
        assert not hasattr(secs[0][1], "slow_factor")

    def test_guard_skipped_without_exactly_two_thermistors(self):
        fast, slow = _gradients(seconds=60.0)
        attrs, _ = self._apply(["T1592", "T2005", "T1"], fast, slow,
                               names=["T1_dT1", "T2_dT2", "T3_dT3"])
        assert attrs[f"{S.ATTR}_applied"] == 1
        assert np.isnan(attrs[f"{S.ATTR}_delta_measured_T2_dT2"])
        assert attrs[f"{S.ATTR}_guard_T2_dT2"].startswith("skipped: the guard needs exactly two")


# --- chi_io on a real file -------------------------------------------------------

PROFILE_FILE = Path(__file__).parent / "data" / "SN479_0006.p"
P_T0 = dt.datetime(2025, 1, 16, 1, 57, 57, tzinfo=dt.UTC)  # the fixture's start
IN_SPAN = {"beta": BETA, "tau_s_ms": TAU_MS, "first": "2025-01-01", "last": "2025-02-01"}


def _pfile(t1="T1592", t2="T2005"):
    """The SN479 fixture with real bead serials (its config writes 'T')."""
    from odas_tpw.rsi.p_file import PFile

    pf = PFile(PROFILE_FILE)
    for ch in pf.config["channels"]:
        if ch.get("name") == "T1":
            ch["sn"] = t1
        elif ch.get("name") == "T2":
            ch["sn"] = t2
    return pf


@pytest.fixture(scope="module")
def patched(tmp_path_factory):
    """The fixture rewritten on disk with real bead serials, T2005 on T2 under
    ``VMP/`` and on T1 under ``VMPswap/`` (same file name in both)."""
    if not PROFILE_FILE.exists():
        pytest.skip("SN479 fixture missing")
    from odas_tpw.rsi.config_patch import (
        EditSpec,
        edit_config_text,
        read_config_text,
        write_patched_pfile,
    )

    root = tmp_path_factory.mktemp("slow_patched")
    for d, (t1, t2) in {"VMP": ("T1592", "T2005"), "VMPswap": ("T2005", "T1592")}.items():
        (root / d).mkdir()
        spec = EditSpec(note="test", author="test", channels={"T1": {"sn": t1}, "T2": {"sn": t2}})
        new, _ = edit_config_text(read_config_text(PROFILE_FILE), spec, add_keys=True)
        write_patched_pfile(PROFILE_FILE, root / d / PROFILE_FILE.name, new)
    return root


def _run(pf, **kw):
    from odas_tpw.rsi.chi_io import _compute_chi

    out = _compute_chi(pf, fft_length=1024, goodman=False, **kw)
    assert out
    return out[0]


@pytest.fixture(scope="module")
def plain():
    if not PROFILE_FILE.exists():
        pytest.skip("SN479 fixture missing")
    return _run(_pfile())


def _slow_attrs(ds):
    return {k for k in ds.attrs if k.startswith(S.ATTR)}


@pytest.mark.skipif(not PROFILE_FILE.exists(), reason="SN479 fixture missing")
class TestChiIo:
    def test_applied_to_the_bead_only(self, plain):
        ds = _run(_pfile(), fp07_slow_response={"T2005": IN_SPAN})
        assert ds.attrs[f"{S.ATTR}_applied"] == 1
        assert ds.attrs[f"{S.ATTR}_bead_T2_dT2"] == "T2005"
        a, b = plain["chi"].values, ds["chi"].values
        np.testing.assert_array_equal(b[0], a[0])  # the other probe, bit for bit
        r = b[1] / a[1]
        r = r[np.isfinite(r)]
        assert r.size >= 3 and np.median(r) > 1.2
        # the stored spectra stay raw
        np.testing.assert_array_equal(ds["spec_gradT"].values, plain["spec_gradT"].values)
        np.testing.assert_array_equal(ds["spec_noise"].values, plain["spec_noise"].values)

    def test_bead_on_the_other_channel_follows_it(self, plain):
        ds = _run(_pfile(t1="T2005", t2="T1592"), fp07_slow_response={"T2005": IN_SPAN})
        assert ds.attrs[f"{S.ATTR}_bead_T1_dT1"] == "T2005"
        np.testing.assert_array_equal(ds["chi"].values[1], plain["chi"].values[1])

    def test_without_the_bead_the_product_is_unchanged(self, plain):
        ds = _run(_pfile(), fp07_slow_response={"T9999": IN_SPAN})
        # only the beads it does carry, for a run summary's unmatched-key line
        assert _slow_attrs(ds) == {f"{S.ATTR}_serials"}
        assert ds.attrs[f"{S.ATTR}_serials"] == "T1592/T2005"
        for name in plain.data_vars:
            np.testing.assert_array_equal(ds[name].values, plain[name].values)
        assert set(ds.attrs) == set(plain.attrs) | {f"{S.ATTR}_serials"}

    def test_outside_its_span_the_product_is_unchanged_but_says_so(self, plain):
        later = dict(IN_SPAN, first="2025-02-01", last="2025-03-01")
        ds = _run(_pfile(), fp07_slow_response={"T2005": later})
        assert ds.attrs[f"{S.ATTR}_applied"] == 0
        assert "outside every configured span" in ds.attrs[f"{S.ATTR}_reason"]
        np.testing.assert_array_equal(ds["chi"].values, plain["chi"].values)

    def test_bead_on_two_channels_is_an_error_inside_its_span(self):
        with pytest.raises(ValueError, match="on 2 thermistor channels"):
            _run(_pfile(t1="T2005", t2="T2005"), fp07_slow_response={"T2005": IN_SPAN})

    def test_bead_on_two_channels_outside_its_span_keeps_the_file(self, plain):
        later = dict(IN_SPAN, first="2025-02-01", last="2025-03-01")
        ds = _run(_pfile(t1="T2005", t2="T2005"), fp07_slow_response={"T2005": later})
        assert ds.attrs[f"{S.ATTR}_applied"] == 0
        np.testing.assert_array_equal(ds["chi"].values, plain["chi"].values)

    @pytest.mark.parametrize("other", [{"fp07_tau_scale": {"T1": 0.5}},
                                       {"fp07_response": ["x.nc"]}])
    def test_never_with_another_response_correction(self, other):
        with pytest.raises(ValueError, match="mutually exclusive"):
            _run(_pfile(), fp07_slow_response={"T2005": IN_SPAN}, **other)

    def test_a_bad_entry_fails_before_the_file_is_read(self):
        from odas_tpw.rsi.chi_io import _compute_chi

        with pytest.raises(ValueError, match="tau_s_ms"):
            _compute_chi("does-not-exist.p", fp07_slow_response={"T2005": dict(IN_SPAN,
                                                                               tau_s_ms=-1)})


def test_an_unidentifiable_file_is_refused(tmp_path):
    # No configuration_string: the beads cannot be read, and the bead may well
    # be there. That is an error, never "no bead on this file".
    import netCDF4

    from odas_tpw.rsi.chi_io import _select_slow_response

    src = tmp_path / "noconfig.nc"
    with netCDF4.Dataset(src, "w") as ds:
        ds.title = "no configuration_string"
    data = {"metadata": {"start_time": "2021-07-02T01:00:00+00:00"}}
    with pytest.raises(ValueError, match="cannot be identified"):
        _select_slow_response(S.parse_slow_response({"T2005": SPAN}), data, src,
                              ["T1_dT1", "T2_dT2"])


def test_a_naive_start_time_is_utc(monkeypatch):
    from odas_tpw.rsi import chi_io

    monkeypatch.setattr(chi_io, "_pair_identity", lambda src, names: ("SN194", ["T1", "T2005"]))
    spans = S.parse_slow_response({"T2005": dict(SPAN, first="2021-07-02T00:00:00Z")})
    data = {"metadata": {"start_time": "2021-07-02T00:00:00"}}  # no zone: the .p clock is UTC
    assert chi_io._select_slow_response(spans, data, "x.p", ["T1_dT1", "T2_dT2"]).applied


# --- perturb configuration and hashing ----------------------------------------


class TestPerturbConfig:
    def test_accepted_and_validated_at_load(self):
        from odas_tpw.perturb.config import _validate_instruments

        _validate_instruments({"SN194": {"fp07_slow_response": {"T2005": SPAN}}})
        with pytest.raises(ValueError, match=r"instruments\.SN194\.fp07_slow_response\.T2005"):
            _validate_instruments({"SN194": {"fp07_slow_response": {"T2005": dict(SPAN, beta=2)}}})

    @pytest.mark.parametrize("other", [{"fp07_tau_scale": {"T1": 0.3}},
                                       {"fp07_response": "a.nc"}])
    def test_never_with_another_response_correction(self, other):
        from odas_tpw.perturb.config import _validate_instruments

        with pytest.raises(ValueError, match="mutually exclusive"):
            _validate_instruments({"SN194": {"fp07_slow_response": {"T2005": SPAN}, **other}})

    def test_empty_is_unset(self):
        from odas_tpw.perturb.config import _validate_instruments, canonical_instruments_for_hash

        _validate_instruments({"SN194": {"fp07_slow_response": {}, "fp07_tau_scale": {"T1": 0.3}}})
        chi = canonical_instruments_for_hash({"SN194": {"fp07_slow_response": {}}},
                                             frozenset({"fp07_slow_response"}))
        assert chi == {}

    def _dump(self, cfg, stage):
        import json

        from odas_tpw.perturb.config import upstream_for

        return json.dumps(upstream_for(stage, cfg), sort_keys=True)

    def test_values_version_chi_and_never_epsilon(self):
        base = {"chi": {"enable": True}}
        a = {**base, "instruments": {"SN194": {"fp07_slow_response": {"T2005": SPAN}}}}
        b = {**base, "instruments": {"SN194": {"fp07_slow_response": {
            "T2005": dict(SPAN, beta=0.10)}}}}
        c = {**base, "instruments": {"SN194": {"fp07_slow_response": {
            "T2005": dict(SPAN, last="2022-06-21")}}}}
        assert self._dump(a, "diss") == self._dump(base, "diss")
        chis = {self._dump(x, "chi") for x in (base, a, b, c)}
        assert len(chis) == 4
        assert self._dump(a, "chi_combo") != self._dump(base, "chi_combo")

    def test_key_order_and_time_spelling_do_not(self):
        from ruamel.yaml import YAML

        a = YAML().load(
            "SN194:\n  fp07_slow_response:\n"
            "    T2005: {beta: 0.124, tau_s_ms: 110, first: 2021-06-29T18:00:00Z,"
            " last: 2022-06-20T08:30:00Z}\n"
            "    T1593: {beta: 0.05, tau_s_ms: 90, first: 2019-01-01, last: 2019-12-31}\n"
        )
        b = YAML().load(
            "SN194:\n  fp07_slow_response:\n"
            "    T1593: {tau_s_ms: 90.0, beta: 0.05, last: '2019-12-31T00:00:00+00:00',"
            " first: '2019-01-01'}\n"
            "    T2005: {last: '2022-06-20T08:30:00Z', first: '2021-06-29T18:00:00+00:00',"
            " tau_s_ms: 110, beta: 0.124}\n"
        )
        assert self._dump({"instruments": a}, "chi") == self._dump({"instruments": b}, "chi")

    def test_template_example_parses(self):
        from ruamel.yaml import YAML

        from odas_tpw.perturb.config import _TEMPLATE, _validate_instruments

        # The whole commented example is one config (tests/test_probe_exclusion
        # validates the rest of it); here, the slow-response entry.
        tail = _TEMPLATE.split("# Example:\n", 1)[1]
        body = "\n".join(ln[2:] for ln in tail.splitlines() if ln.startswith("# "))
        inst = YAML().load(body)["instruments"]
        _validate_instruments({"VMP": inst["VMP"]})
        (sp,) = S.parse_slow_response(inst["VMP"]["fp07_slow_response"])["T2005"]
        assert (sp.beta, sp.tau_s_ms) == (BETA, TAU_MS)
        assert sp.span_text() == "2021-06-29T18:00:00+00:00..2022-06-20T08:30:00+00:00"


TWO_KEYS = {"T2005": SPAN, "T9999": SPAN}  # one bead key that matches nothing


class TestPerturbNotices:
    def _rec(self, **kw):
        rec = {"instrument": "SN194", "configured": ["T2005"], "profiles": 3, "seen": [],
               "outside": [], "pool": {}, "file_guard": [], "guard": []}
        rec.update(kw)
        return rec

    T_IN = dt.datetime(2021, 7, 2, tzinfo=dt.UTC).timestamp()
    T_OUT = dt.datetime(2022, 7, 4, tzinfo=dt.UTC).timestamp()

    def _file(self, serials, t, *, swapped=False, seconds=60.0, n_profiles=1):
        """process_file's record for one .p file, built by the REAL path: chi_io's
        attributes on each profile, the tally, then the file-level verdict."""
        from odas_tpw.perturb.pipeline import _slow_guard_file_verdicts, _tally_slow_response
        from odas_tpw.rsi.chi_io import _apply_slow_response

        spans = S.parse_slow_response(TWO_KEYS)
        rec = self._rec(instrument="VMP", configured=sorted(TWO_KEYS), profiles=0)
        fast, slow = _gradients(seconds=seconds)
        if swapped:  # the slow bead is really on T1
            fast, slow = slow, fast
        for k, (l2, l3) in enumerate(_sections(fast, slow, n_profiles)):
            attrs = _apply_slow_response(S.select(spans, serials, t), spans, [(l2, l3)],
                                         ["T1_dT1", "T2_dT2"], "f.p")
            _tally_slow_response(rec, attrs, f"f_prof{k + 1:03d}.nc")
        _slow_guard_file_verdicts(rec, "f.p")
        return rec

    def test_tally_pools_a_files_profiles(self):
        # 43 windows split over 6 profiles: no profile has a verdict, the file does
        rec = self._file(["T1592", "T2005"], self.T_IN, swapped=True, seconds=1400.0,
                         n_profiles=6)
        assert rec["profiles"] == 6 and rec["seen"] == ["T1592", "T2005"]
        assert "pool" not in rec
        ((bead, ch, measured, predicted, n, verdict),) = rec["file_guard"]
        assert (bead, ch) == ("T2005", "T2_dT2") and n >= S.SLOW_GUARD_MIN_WINDOWS
        assert measured > 0 > predicted and verdict.startswith("WARNING")
        assert rec["guard"] == [["T2005", "T2_dT2", measured, predicted, n]]

    def test_notices(self):
        from odas_tpw.perturb.pipeline import _slow_response_notices

        outcomes = {
            "VMP__a": self._file(["T1592", "T2005"], self.T_IN, swapped=True, seconds=1400.0),
            "VMP__b": self._file(["T1592", "T2005"], self.T_OUT),
            "VMP__c": self._file(["T1592", "T1593"], self.T_IN),
            "VMP__d": self._file(["T1592", "T2005"], self.T_IN),
        }
        assert outcomes["VMP__c"]["seen"] == ["T1592", "T1593"]  # no bead, still known
        inst = {"VMP": {"fp07_slow_response": TWO_KEYS}}
        text = "\n".join(_slow_response_notices(inst, {"VMP"}, outcomes))
        assert ("fp07_slow_response.T9999 matched no thermistor in the 4 chi profile(s) of 4 "
                "file(s) (beads seen: T1592, T1593, T2005)") in text
        assert "fp07_slow_response.T2005 matched no" not in text
        assert "bead T2005 (instruments.VMP) was on 1 file(s) outside every" in text
        assert "configured span, which were NOT corrected: VMP__b" in text
        assert "in 1 file(s) (instruments.VMP) the bead's own 0.25->2 Hz" in text
        assert "Files: VMP__a" in text
        assert "too few windows for a guard verdict" in text and "unnoticed: VMP__d" in text
        # a key matching no input file is _instrument_key_notices'; no chi, nothing checked
        assert _slow_response_notices(inst, {"SN194"}, outcomes) == []
        assert _slow_response_notices(inst, {"VMP"}, {}) == []

    def test_the_file_record_survives_the_cache(self, tmp_path):
        from odas_tpw.perturb.pipeline import _read_slow_sidecar, _write_slow_sidecar

        rec = self._file(["T1592", "T2005"], self.T_IN, swapped=True, seconds=1400.0)
        _write_slow_sidecar(tmp_path, "VMP__a", rec, "a.p")
        back = _read_slow_sidecar(tmp_path, "VMP__a")
        assert back["cached"] is True and back["file"] == "a.p"
        assert back["software_version"] and back["guard_limit"] == S.SLOW_GUARD_MAX
        for key in ("instrument", "configured", "profiles", "seen", "outside", "guard"):
            assert back[key] == rec[key]
        assert back["file_guard"][0][5].startswith("WARNING")
        assert _read_slow_sidecar(tmp_path, "nope") is None
        assert _read_slow_sidecar(None, "VMP__a") is None


# --- rsi-tpw -----------------------------------------------------------------


class TestRsi:
    def test_unset_leaves_every_existing_hash_alone(self):
        from odas_tpw.rsi.config import canonicalize

        assert "fp07_slow_response" not in canonicalize("chi", {})
        assert canonicalize("chi", {}) == canonicalize("chi", {"fp07_slow_response": None})

    def test_set_versions_the_output_directory(self, tmp_path):
        from odas_tpw.rsi.cli import _chi_response_params
        from odas_tpw.rsi.config import resolve_output_dir

        def name(merged):
            _kw, hp = _chi_response_params(merged, [])
            return resolve_output_dir(tmp_path, "chi", "chi", hp).name

        plain = name({})
        a = name({"fp07_slow_response": {"T2005": SPAN}})
        b = name({"fp07_slow_response": {"T2005": dict(SPAN, tau_s_ms=150)}})
        assert len({plain, a, b}) == 3
        unquoted = name({"fp07_slow_response": {"T2005": dict(
            SPAN, first=dt.datetime(2021, 6, 29, 18, tzinfo=dt.UTC))}})
        assert unquoted == a

    def test_kwargs_are_plain_and_parse_back(self):
        from odas_tpw.rsi.cli import _chi_response_params

        kw, hp = _chi_response_params({"fp07_slow_response": {"T2005": SPAN}}, [])
        assert S.parse_slow_response(kw["fp07_slow_response"]) == S.parse_slow_response(
            {"T2005": SPAN})
        assert isinstance(hp["fp07_slow_response"]["T2005"][0]["first"], float)

    def test_refused_beside_fp07_response(self, tmp_path):
        from odas_tpw.rsi.cli import _chi_response_params

        with pytest.raises(ValueError, match="mutually exclusive"):
            _chi_response_params({"fp07_slow_response": {"T2005": SPAN}}, ["a.nc"])

    def test_validated_at_load(self, tmp_path):
        from odas_tpw.rsi.config import load_config

        cfg = tmp_path / "rsi.yaml"
        cfg.write_text("chi:\n  fp07_slow_response:\n    T2005: {beta: 0.124}\n",
                       encoding="utf-8")
        with pytest.raises(ValueError, match="missing"):
            load_config(cfg)

    @pytest.mark.skipif(not PROFILE_FILE.exists(), reason="SN479 fixture missing")
    def test_pipeline_refuses_rather_than_ignores_it(self, tmp_path, monkeypatch, capsys):
        import sys

        import yaml

        from odas_tpw.rsi.cli import main

        cfg = tmp_path / "rsi.yaml"
        cfg.write_text(yaml.safe_dump({"chi": {"fp07_slow_response": {"T2005": SPAN}}}),
                       encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["rsi-tpw", "-c", str(cfg), "pipeline",
                                          str(PROFILE_FILE), "-o", str(tmp_path / "out")])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 2
        assert "does not apply fp07_slow_response" in capsys.readouterr().err


    def test_end_of_run_warnings(self, patched, tmp_path, monkeypatch, capsys):
        import sys

        import yaml

        from odas_tpw.rsi.cli import main

        cfg = tmp_path / "rsi.yaml"
        cfg.write_text(yaml.safe_dump({"chi": {"fp07_slow_response": {
            "T2005": IN_SPAN, "T9999": IN_SPAN}}}), encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["rsi-tpw", "-c", str(cfg), "chi",
                                          str(patched / "VMP" / PROFILE_FILE.name),
                                          "-o", str(tmp_path / "out")])
        main()
        err = capsys.readouterr().err
        assert ("WARNING: fp07_slow_response.T9999 matched no thermistor in 1 input(s) "
                "(beads seen: T1592, T2005)") in err
        assert "fp07_slow_response.T2005 matched" not in err
        assert "too few guard windows" in err  # one ~8-minute fixture is < 40 windows

    def test_per_profile_inputs_get_one_line(self):
        from odas_tpw.rsi.cli import _slow_response_run_warnings

        summ = {"serials": ["T1592", "T2005"], "outside": [],
                "applied": [{"bead": "T2005", "channel": "T2_dT2", "verdict": "no verdict: 1",
                             "measured": -0.2, "predicted": -0.159, "n_windows": 1,
                             "lo": [], "hi": []}]}
        names = [Path("c_prof001.nc"), Path("c_prof002.nc")]
        lines = _slow_response_run_warnings({"T2005": SPAN}, {n.name: summ for n in names},
                                            names)
        assert len(lines) == 1 and "per-profile NetCDFs" in lines[0]
        assert _slow_response_run_warnings({"T2005": SPAN}, {"a.p": None}, [Path("a.p")]) == []


# --- perturb end to end ----------------------------------------------------------


@pytest.mark.skipif(not PROFILE_FILE.exists(), reason="SN479 fixture missing")
def test_perturb_end_to_end(patched, tmp_path):
    """Two instrument directories holding same-named files, the bead on T2 in one
    and on T1 in the other: jobs=1, then a cached rerun."""
    import xarray as xr

    from odas_tpw.perturb import pipeline as pl

    out = tmp_path / "out"
    files = [patched / d / PROFILE_FILE.name for d in ("VMP", "VMPswap")]

    def run(inst):
        return pl.run_pipeline({
            "files": {"p_file_root": str(patched), "output_root": str(out), "trim": False,
                      "merge": False},
            "chi": {"enable": True}, "ctd": {"enable": False}, "parallel": {"jobs": 1},
            "instruments": inst}, p_files=files)

    assert run({}).ok
    slow = {"fp07_slow_response": {"T2005": IN_SPAN, "T9999": IN_SPAN}}
    first = run({"VMP": slow, "VMPswap": slow})
    assert first.ok
    dirs = {d.name for d in out.iterdir()}
    # chi and what embeds it re-versioned; diss and profiles reused
    assert {"chi_01", "chi_binned_01", "chi_combo_01"} <= dirs
    assert "diss_01" not in dirs and "profiles_01" not in dirs
    for d, probe in (("VMP", 1), ("VMPswap", 0)):
        stem = f"{d}__{PROFILE_FILE.stem}"
        with (xr.open_dataset(out / "chi_00" / f"{stem}_prof001.nc") as a,
              xr.open_dataset(out / "chi_01" / f"{stem}_prof001.nc") as b):
            np.testing.assert_array_equal(b["chi"].values[1 - probe], a["chi"].values[1 - probe])
            r = np.nanmedian(b["chi"].values[probe] / a["chi"].values[probe])
            assert 1.2 < r <= (1 - BETA) ** -2 + 0.05
            assert b.attrs[f"{S.ATTR}_serials"] == ("T1592/T2005" if d == "VMP"
                                                    else "T2005/T1592")
        assert (out / "chi_01" / f"{stem}{'.fp07_slow_response.json'}").exists()
    slow_lines = [n for n in first.notices if n.startswith("FP07 slow response")]
    text = "\n".join(slow_lines)
    assert "(beads seen: T1592, T2005)" in text
    # keyed by output stem: same-named files in two directories are both there
    assert "unnoticed: VMP__SN479_0006" in text and "unnoticed: VMPswap__SN479_0006" in text
    # a rerun skips the files, and the recorded outcome says the same again
    again = run({"VMP": slow, "VMPswap": slow})
    assert [n for n in again.notices if n.startswith("FP07 slow response")] == slow_lines


# --- perturb-diag --------------------------------------------------------------


def test_diag_model_carries_the_factor():
    from odas_tpw.perturb.diag.render import _fp07_h2

    K = np.linspace(1, 100, 50)
    W_ = 0.8
    attrs = {"fp07_model": "single_pole", f"{S.ATTR}_beta_T2_dT2": BETA,
             f"{S.ATTR}_tau_s_ms_T2_dT2": TAU_MS}
    base = fp07_transfer(K * W_, fp07_tau(W_, model="lueck"))
    np.testing.assert_allclose(_fp07_h2(K, W_, attrs, probe="T2_dT2"),
                               base * S.slow_h2(K * W_, BETA, TAU_MS / 1000))
    # the other probe stays on the model
    np.testing.assert_allclose(_fp07_h2(K, W_, attrs, probe="T1_dT1"), base)
