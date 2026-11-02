# Nov-2026, Claude and Pat Welch, pat@mousebrains.com
"""perturb's per-file FP07 fit refuses a curve no FP07 has (issue #257).

CASPER 2015 East doug fitted order 2 over 0.16-2 K of temperature in 88 of
272 channels: ``beta_1 < 0`` or ``t_0`` outside 250-320 K, the parabola's
vertex inside the data in 57, and the converted temperature folding back
there. perturb now refuses a curve no FP07 has, over the L range the curve is
APPLIED to: not NTC (``fp07-cal``'s #242), a local beta outside twice the FP07
sheet's band (a per-file fit also absorbs whatever scales the configured L:
Taiwan 2013 husker fits ~4300-4600 K in every file), or a local beta changing faster than
2%/K (curvature no bead has). An order > 1 curve that fails is
refitted at order 1; one that still fails keeps the factory coefficients.

The synthetic beads are physical (``tests/fp07_bead.py``). A reference with a
quadratic error ``k*(T - Tm)**2`` gives the order-2 fit the curvature
``dT_ref/dT = 1 + 2k(T - Tm)``, a vertex at ``Tm - 1/(2k)``, while leaving the
order-1 slope unbiased (the error is even in the signal): that is the
CASPER signature, made deterministic.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from odas_tpw.perturb.fp07_cal import _not_an_fp07, fp07_calibrate
from odas_tpw.perturb.pipeline import _fp07_notices
from tests.fp07_bead import BETA_K, T0_K, fp07_counts
from tests.test_perturb_fp07_cal_defects import _CFG, _PF

N = 4000
TM = 12.0


def _pf(T_probe: np.ndarray, T_ref: np.ndarray) -> _PF:
    n = T_probe.size
    return _PF(
        channels={"T1": np.zeros(n), "JAC_T": T_ref, "P": np.linspace(0, 50, n)},
        channels_raw={"T1": fp07_counts(T_probe)},
        config={"channels": [dict(_CFG)]},
        channel_info={"T1": {"type": "therm"}, "JAC_T": {"type": "jac_t"},
                      "P": {"type": "pres"}},
    )


def _signal(amp: float, n: int = N) -> np.ndarray:
    return TM + amp * np.sin(2 * np.pi * np.arange(n) / 640.0)


def _physical_curve() -> np.ndarray:
    """``1/T_K = 1/T0 + L/beta``, the beads' own curve, as Steinhart-Hart coefficients."""
    return np.array([1.0 / T0_K, 1.0 / BETA_K])


def _fit(pf, profiles, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fp07_calibrate(pf, profiles, **kw)
    return out, [str(w.message) for w in caught]


class TestTheRule:
    def test_an_fp07_curve_passes(self):
        assert _not_an_fp07(_physical_curve(), (-0.3, 0.3)) == ""

    def test_a_real_beads_curvature_passes(self):
        """beta_2 = 2.5e5, SN479's: a bead's local beta barely moves."""
        c = np.array([1.0 / T0_K, 1.0 / BETA_K, 1.0 / 2.5e5])
        assert _not_an_fp07(c, (-0.3, 0.3)) == ""

    def test_a_slope_no_fp07_has(self):
        """A local beta of 610 K: what a probe seeing a fifth of the gradient fits."""
        c = np.array([1.0 / T0_K, 5.0 / BETA_K])
        assert "a slope no FP07 has" in _not_an_fp07(c, (-0.3, 0.3))

    def test_a_steady_off_sheet_level_is_not_refused(self):
        """A steady 4500 K (Taiwan 2013 husker, every file) is a calibration the
        per-file fit exists to make; 7000 K is outside even that."""
        assert _not_an_fp07(np.array([1.0 / T0_K, 1.0 / 4500.0]), (-0.3, 0.3)) == ""
        assert "a slope no FP07 has" in _not_an_fp07(
            np.array([1.0 / T0_K, 1.0 / 7000.0]), (-0.3, 0.3)
        )

    def test_a_vertex_inside_the_range_is_not_ntc(self):
        """d(1/T)/dL = 1/beta + 2*a2*L vanishes at L = -1/(2*a2*beta)."""
        a2 = 1.0 / (2 * BETA_K * 0.1)  # vertex at L = -0.1
        c = np.array([1.0 / T0_K, 1.0 / BETA_K, a2])
        assert "not NTC" in _not_an_fp07(c, (-0.2, 0.2))

    @pytest.mark.parametrize(("per_K", "ok"), [(0.01, True), (0.05, False)])
    def test_curvature_no_bead_has(self, per_K, ok):
        """Over +-0.1 in L (about 5 K) the local beta ratio is ~1 + 0.4*a2*beta:
        a2 is chosen for a curvature of 1%/K (passes) or 5%/K (fails)."""
        L_range = (-0.1, 0.1)
        span_K = float(np.ptp(1.0 / (1.0 / T0_K + np.array(L_range) / BETA_K)))
        a2 = per_K * span_K / (0.4 * BETA_K)
        why = _not_an_fp07(np.array([1.0 / T0_K, 1.0 / BETA_K, a2]), L_range)
        assert (why == "") is ok, why
        if not ok:
            assert "curvature fitted to noise" in why


class TestTheFit:
    def test_a_physical_bead_is_fitted_at_its_order(self):
        """The common case is untouched: no fallback, no refusal."""
        T = _signal(2.0)
        out, _ = _fit(_pf(T, T), [(10, N - 20)], order=2)
        assert "T1" in out["coefficients"] and len(out["coefficients"]["T1"]) == 3
        assert out["order_fallback"] == {} and out["factory"] == {}
        assert out["info"]["T1"]["order"] == 2

    def test_an_implausible_order_2_falls_back_to_order_1(self):
        """0.2 K of range against a reference with a curved error, the CASPER
        regime: the order-2 curve folds inside the data, order 1 is an FP07's."""
        T = _signal(0.1)
        k = 10.0  # vertex at TM - 0.05 K, inside the data
        out, msgs = _fit(_pf(T, T + k * (T - TM) ** 2), [(10, N - 20)], order=2)
        assert "T1" in out["order_fallback"], out["factory"]
        assert "not an FP07 curve" in out["order_fallback"]["T1"]
        assert len(out["coefficients"]["T1"]) == 2
        assert out["info"]["T1"]["order"] == 1
        assert "T1" in out["channels"]
        assert any("Using the order-1 fit" in m for m in msgs)
        # ...and the order-1 curve is the bead's: its slope within 1%.
        assert out["coefficients"]["T1"][1] == pytest.approx(1.0 / BETA_K, rel=0.01)

    def test_no_fp07_curve_at_any_order_keeps_the_factory_coefficients(self):
        T = _signal(0.4)
        out, msgs = _fit(_pf(TM + 5.0 * (T - TM), T), [(10, N - 20)], order=2)
        assert "T1" not in out["coefficients"] and "T1" not in out["channels"]
        assert out["factory"]["T1"].startswith("fit is not an FP07 curve")
        assert "order 2:" in out["factory"]["T1"] and "order 1:" in out["factory"]["T1"]
        assert any("Leaving the factory calibration" in m for m in msgs)

    def test_order_1_requested_is_not_refitted(self):
        T = _signal(0.4)
        out, _ = _fit(_pf(TM + 5.0 * (T - TM), T), [(10, N - 20)], order=1)
        assert "T1" in out["factory"] and "order 2:" not in out["factory"]["T1"]
        assert out["order_fallback"] == {}

    def test_the_check_covers_where_the_curve_is_applied(self, monkeypatch):
        """A curve fitted on one cast is applied to every cast the products use:
        the gate is handed the L range of all of them, not of the fit's."""
        import odas_tpw.perturb.fp07_cal as F

        half = N // 2
        T = np.concatenate([_signal(0.4, half), np.linspace(TM, TM - 3.0, N - half)])
        pf = _pf(T, T)
        fit_cast, other_cast = (10, half - 20), (half, N - 1)
        seen: list = []
        real = F._not_an_fp07
        monkeypatch.setattr(F, "_not_an_fp07", lambda c, r: seen.append(r) or real(c, r))
        L_cold = BETA_K * (1.0 / (TM - 3.0 + 273.15) - 1.0 / T0_K)  # colder: larger L
        _fit(pf, [fit_cast], order=2)
        assert seen[-1][1] < L_cold - 0.01  # the fit's own cast only
        _fit(pf, [fit_cast], order=2, range_profiles=[fit_cast, other_cast])
        assert seen[-1][1] >= L_cold - 1e-3

    def test_a_steady_off_sheet_level_is_fitted(self):
        """Husker's case: a steady bead whose configured L is scaled (local beta
        ~4300-4600 K in every file) keeps its order-2 fit."""
        T = TM + 3.0 * np.sin(2 * np.pi * np.arange(N) / 640.0)
        pf = _pf(T, T)
        pf.channels_raw["T1"] = fp07_counts(T, beta_K=4500.0)
        out, _ = _fit(pf, [(10, N - 20)], order=2)
        assert out["order_fallback"] == {} and out["factory"] == {}
        assert out["info"]["T1"]["order"] == 2

    def test_info_records_the_range_checked(self):
        T = _signal(2.0)
        out, _ = _fit(_pf(T, T), [(10, N - 20)], order=2)
        lo, hi = out["info"]["T1"]["L_range_checked"]
        L = BETA_K * (1.0 / (T + 273.15) - 1.0 / T0_K)
        assert lo <= L.min() + 1e-3 and hi >= L.max() - 1e-3


class TestTheRunSummary:
    def test_an_order_1_fallback_is_named(self):
        """Grouped by cause: the numbers in the parentheses differ per file."""
        why = "the order-2 fit is not an FP07 curve: curvature no bead has (local beta {} K)"
        outcomes = {
            "a.p": {"fitted": ["T1", "T2"], "factory": {},
                    "order_fallback": {"T1": why.format("2900-3600")}},
            "b.p": {"fitted": ["T1", "T2"], "factory": {},
                    "order_fallback": {"T1": why.format("1.2e+04-1.5e+04")}},
            "c.p": {"fitted": ["T1", "T2"], "factory": {}},
        }
        lines = _fp07_notices(outcomes)
        assert lines[0].startswith("FP07 T1: fitted at ORDER 1 in 2 file(s) of 3 fitted")
        assert ("(2 of 2): the order-N fit is not an FP07 curve: curvature no bead has."
                in lines[0])  # one cause; "FP07" is not a number
        assert lines[1] == "FP07 T1 order-1 files: a.p, b.p"
        assert len(lines) == 2  # T2 fitted everywhere at the configured order

    def test_a_refusal_is_a_factory_notice(self):
        why = "fit is not an FP07 curve at order 1: a slope no FP07 has (local beta 610 K)"
        lines = _fp07_notices({"a.p": {"fitted": [], "factory": {"T1": why}}})
        assert "FACTORY coefficients kept in 1 file(s)" in lines[0]
        assert "fit is not an FP07 curve at order N: a slope no FP07 has." in lines[0]


class TestThePipeline:
    """process_file on SN479, its order-2 curves made to fail the rule."""

    def test_a_fallback_reaches_the_record_and_the_product(self, tmp_path, monkeypatch):
        from tests.test_noise_model_insitu_coeffs import (
            BEAD_COEFFS_SOURCE_ATTR,
            P_FILE,
            _run,
            _var_attrs,
        )

        if not P_FILE.exists():
            pytest.skip("SN479_0006.p test data not available")
        import odas_tpw.perturb.fp07_cal as F

        real = F._not_an_fp07
        monkeypatch.setattr(
            F, "_not_an_fp07",
            lambda c, r: "forced for the test" if len(c) > 2 else real(c, r),
        )
        res, dirs, fit = _run(tmp_path, P_FILE, {"calibrate": True, "order": 2}, monkeypatch)
        assert res["fp07"]["fitted"] == ["T1", "T2"]
        assert set(res["fp07"]["order_fallback"]) == {"T1", "T2"}
        assert all(len(fit["coefficients"][b]) == 2 for b in ("T1", "T2"))
        prof = sorted(dirs["profiles"].glob("*.nc"))[0]
        tag = "in-situ (Steinhart-Hart order 1 vs JAC_T)"
        for base in ("T1", "T2"):
            for name in (base, f"{base}_d{base}"):
                assert _var_attrs(prof, name)["calibration"] == tag, name
            attrs = _var_attrs(prof, f"{base}_d{base}")
            assert attrs[BEAD_COEFFS_SOURCE_ATTR] == tag
            assert "beta_2" not in attrs  # the noise model has the order-1 curve too

    def test_sn479_is_fitted_at_its_order(self, tmp_path, monkeypatch):
        """A real VMP with 15-28 degC of range: nothing falls back."""
        from tests.test_noise_model_insitu_coeffs import P_FILE, _run

        if not P_FILE.exists():
            pytest.skip("SN479_0006.p test data not available")
        res, _dirs, fit = _run(tmp_path, P_FILE, {"calibrate": True, "order": 2}, monkeypatch)
        assert res["fp07"]["order_fallback"] == {} and res["fp07"]["factory"] == {}
        assert all(len(fit["coefficients"][b]) == 3 for b in ("T1", "T2"))
