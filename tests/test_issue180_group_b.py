# Sep-2026, Claude and Pat Welch, pat@mousebrains.com
"""Negative controls for issue #180 group B: QC propagation and provenance.

F01, F02, F04, F05, F07 — the findings about which windows survive to the
mixing products, and whether a consumer can tell how they got there.
"""

from __future__ import annotations

import re
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from odas_tpw.chi.l4_chi import _compute_chi_final, chi_final_and_fallback
from odas_tpw.processing.chi_combine import mk_chi_mean
from odas_tpw.processing.mixing import MAX_PAIR_DT, pair_nearest
from odas_tpw.rsi.speed import compute_speed_for_pfile
from odas_tpw.scor160.io import L3Data
from odas_tpw.scor160.l4 import _compute_epsi_final, _compute_flags, process_l4
from odas_tpw.scor160.nasmyth import nasmyth_grid


def _one_window_l3(interp_fraction: float = 0.0, bad_fraction: float = 0.0) -> L3Data:
    K = np.linspace(0, 256, 513)
    spec = nasmyth_grid(1e-7, 1e-6, K)
    return L3Data(
        time=np.array([4.0]),
        pres=np.array([20.0]),
        temp=np.array([10.0]),
        pspd_rel=np.array([0.5]),
        section_number=np.array([1]),
        kcyc=K[:, None],
        sh_spec=spec[None, :, None],
        sh_spec_clean=spec[None, :, None],
        bad_fraction=np.full((1, 1), bad_fraction),
        interp_fraction=np.full((1, 1), interp_fraction),
    )


class TestF01BadBufferAtL4:
    """process_l4 must enforce the same bad-buffer ceiling as its two siblings."""

    def test_half_interpolated_window_is_rejected(self):
        l3 = _one_window_l3(interp_fraction=0.5)
        with pytest.warns(UserWarning, match="bad-buffer"):
            l4 = process_l4(l3, num_ffts=7)
        # Pre-fix: epsi_flags [[0]], epsilon 1.327e-7 -- repaired samples are
        # finite, so FFT finiteness could not see the declared 5% ceiling.
        assert not np.isfinite(l4.epsi_final[0])
        assert l4.epsi_flags[0, 0] == 255

    def test_the_reason_survives_into_the_product(self):
        """A masked window must not be indistinguishable from an unusable one."""
        l3 = _one_window_l3(interp_fraction=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            l4 = process_l4(l3, num_ffts=7)
        assert l4.interp_fraction is not None
        assert l4.interp_fraction[0, 0] == pytest.approx(0.5)
        assert l4.bad_fraction is not None
        assert l4.bad_fraction[0, 0] == pytest.approx(0.0)

    def test_tolerable_interpolation_is_kept(self):
        l3 = _one_window_l3(interp_fraction=0.01)
        l4 = process_l4(l3, num_ffts=7)
        assert np.isfinite(l4.epsi_final[0])

    def test_opt_out_restores_the_old_behaviour(self):
        l3 = _one_window_l3(interp_fraction=0.5)
        l4 = process_l4(l3, num_ffts=7, mask_bad_buffers=False)
        assert np.isfinite(l4.epsi_final[0])


class TestF02RatioTestEligibility:
    """A probe that already failed must not be the ratio test's yardstick."""

    def test_failed_low_probe_no_longer_kills_the_passing_one(self):
        eps = np.array([[1e-12], [1e-7]])
        flags = _compute_flags(
            eps,
            np.array([[10.0], [0.5]]),  # probe 1 fails FOM, probe 2 passes
            np.ones((2, 1)),
            1.15,
            0.5,
            sigma_ln=np.full((2, 1), 0.2),
            method=np.zeros((2, 1)),
        )
        # Pre-fix: flags [1, 4] and a NaN combined value -- the window was lost
        # although the second probe had no failure of its own.
        assert flags[0, 0] == 1  # still flagged for its own FOM
        assert flags[1, 0] == 0  # no longer flagged by the failed probe
        assert _compute_epsi_final(eps, flags)[0] == pytest.approx(1e-7)

    def test_two_healthy_probes_still_trip_the_ratio_test(self):
        """The test must not have been switched off, only re-referenced."""
        eps = np.array([[1e-12], [1e-7]])
        flags = _compute_flags(
            eps,
            np.array([[0.5], [0.5]]),  # both pass FOM
            np.ones((2, 1)),
            1.15,
            0.5,
            sigma_ln=np.full((2, 1), 0.2),
            method=np.zeros((2, 1)),
        )
        assert flags[0, 0] == 0  # the minimum is never flagged
        assert flags[1, 0] == 4

    def test_all_probes_failed_degrades_to_the_old_candidate_set(self):
        """With nothing eligible, still report mutual inconsistency."""
        eps = np.array([[1e-12], [1e-7]])
        flags = _compute_flags(
            eps,
            np.array([[10.0], [10.0]]),  # both fail FOM
            np.ones((2, 1)),
            1.15,
            0.5,
            sigma_ln=np.full((2, 1), 0.2),
            method=np.zeros((2, 1)),
        )
        assert flags[0, 0] == 1
        assert flags[1, 0] == 5  # 1 (FOM) + 4 (ratio)

    def test_under_resolved_probe_is_also_ineligible(self):
        """Flag order matters: bits 1/2/8/16 are computed before the ratio test."""
        eps = np.array([[1e-12], [1e-7]])
        flags = _compute_flags(
            eps,
            np.array([[0.5], [0.5]]),
            np.array([[0.1], [0.9]]),  # probe 1 under-resolved
            1.15,
            0.5,
            sigma_ln=np.full((2, 1), 0.2),
            method=np.zeros((2, 1)),  # variance method: bit 16 applies
        )
        assert flags[0, 0] == 16
        assert flags[1, 0] == 0


class TestF04ChiSoftQCFallback:
    """An all-fail chi window must be distinguishable, and must not drive mixing."""

    def test_flag_marks_the_fallback(self):
        chi = np.array([[1e-7], [4e-7]])
        fom = np.full((2, 1), 100.0)
        kmr = np.full((2, 1), 0.01)
        res = chi_final_and_fallback(chi, fom, kmr)
        # The value is still reported -- no window is lost...
        assert res.chi_final[0] == pytest.approx(2e-7)
        # ...but it now says how it got there.
        assert bool(res.qc_fallback[0])
        # Backward-compatible wrapper is unchanged.
        assert _compute_chi_final(chi, fom, kmr)[0] == pytest.approx(2e-7)

    def test_passing_window_is_not_flagged(self):
        chi = np.array([[1e-7], [4e-7]])
        fom = np.full((2, 1), 1.0)
        kmr = np.full((2, 1), 0.8)
        res = chi_final_and_fallback(chi, fom, kmr)
        assert not bool(res.qc_fallback[0])

    def test_empty_window_is_not_a_fallback(self):
        """No finite probe at all is a missing window, not a rescued one."""
        chi = np.array([[np.nan], [-1.0]])
        res = chi_final_and_fallback(chi, np.ones((2, 1)), np.full((2, 1), 0.8))
        assert not np.isfinite(res.chi_final[0])
        assert not bool(res.qc_fallback[0])

    def test_mk_chi_mean_exports_the_flag(self):
        n = 3
        ds = xr.Dataset(
            {
                # The perturb shape: chi/fom/K_max_ratio on (probe, time).
                "chi": (("probe", "time"), np.array([[1e-7] * n, [4e-7] * n])),
                "fom": (("probe", "time"), np.array([[1.0, 100.0, 1.0], [1.0, 100.0, 100.0]])),
                "K_max_ratio": (("probe", "time"), np.full((2, n), 0.8)),
            },
            coords={"time": np.arange(float(n)), "probe": np.arange(2)},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = mk_chi_mean(ds, fom_limit=1.15, k_max_ratio_min=0.5)
        assert "chiQCFallback" in out
        np.testing.assert_array_equal(out["chiQCFallback"].values, [0, 1, 0])

    def test_rsi_mixing_excludes_the_fallback_by_default(self):
        from odas_tpw.rsi.pipeline import _qc_chi_final

        chi = np.array([[1e-7], [4e-7]])
        fom = np.full((2, 1), 100.0)
        kmr = np.full((2, 1), 0.01)
        assert not np.isfinite(_qc_chi_final(chi, fom, kmr)[0])
        assert _qc_chi_final(chi, fom, kmr, use_qc_fallback=True)[0] == pytest.approx(2e-7)


class TestF05PairingTolerance:
    """QC gaps must not redefine the source cadence."""

    def test_gap_does_not_widen_the_tolerance(self):
        t = np.arange(1001.0)
        v = np.full(t.size, np.nan)
        v[0] = 1e-7
        v[-1] = 1e-5
        # Pre-fix: returned the estimate at t=0, 500 s away, because the median
        # spacing of the two SURVIVING estimates is 1000 s.
        assert not np.isfinite(pair_nearest(t, v, np.array([500.0]))[0])

    def test_normal_cadence_still_pairs(self):
        src_t = np.arange(0.0, 80.0, 8.0)
        src_v = np.full(src_t.size, 1e-7)
        out = pair_nearest(src_t, src_v, np.array([7.0, 9.0]))
        assert np.all(np.isfinite(out))

    def test_gap_does_not_shadow_an_adjacent_valid_estimate(self):
        """The behaviour this replaced must survive: a NaN neighbour is skipped."""
        src_t = np.arange(0.0, 40.0, 8.0)
        src_v = np.array([1e-7, np.nan, 3e-7, 4e-7, 5e-7])
        out = pair_nearest(src_t, src_v, np.array([8.0]))
        assert np.isfinite(out[0])

    def test_automatic_tolerance_is_capped(self):
        src_t = np.array([0.0, 10000.0, 20000.0])
        src_v = np.full(3, 1e-7)
        out = pair_nearest(src_t, src_v, np.array([5000.0]))
        assert not np.isfinite(out[0])
        assert MAX_PAIR_DT == 120.0

    def test_explicit_max_dt_is_the_callers_choice(self):
        src_t = np.array([0.0, 10000.0, 20000.0])
        src_v = np.full(3, 1e-7)
        out = pair_nearest(src_t, src_v, np.array([5000.0]), max_dt=1e5)
        assert np.isfinite(out[0])


def _em_pfile(em: np.ndarray) -> SimpleNamespace:
    slow = np.arange(0.0, 100.0, 0.25)
    fast = np.arange(0.0, 100.0, 0.0625)
    return SimpleNamespace(
        channels={"P": 20.0 + 0.1 * slow, "U_EM": em},
        fs_slow=4.0,
        fs_fast=16.0,
        t_fast=fast,
        t_slow=slow,
    )


class TestF07MeasuredSpeedCoverage:
    """One surviving sample is not a measured speed record."""

    def test_single_em_sample_is_refused(self):
        em = np.full(400, np.nan)
        em[100] = 0.3
        # Pre-fix: 1/400 finite slow samples produced 1600/1600 finite fast
        # samples, all 0.3 m/s, with provenance "em".
        with pytest.raises(ValueError, match=re.escape("covers 0.2% of the record")):
            compute_speed_for_pfile(_em_pfile(em), {"method": "em"}, vehicle="vmp")

    def test_complete_record_keeps_the_historical_provenance_token(self):
        _, _, src = compute_speed_for_pfile(
            _em_pfile(np.full(400, 0.3)), {"method": "em"}, vehicle="vmp"
        )
        assert src == "em"

    def test_partial_record_declares_its_imputation(self):
        em = np.full(400, 0.3)
        em[50:80] = np.nan  # 7.5 s hole, 92.5% coverage
        _, _, src = compute_speed_for_pfile(_em_pfile(em), {"method": "em"}, vehicle="vmp")
        assert src.startswith("em(")
        assert "cov=0.925" in src

    def test_long_gap_is_refused_even_at_good_coverage(self):
        em = np.full(400, 0.3)
        em[100:380] = np.nan  # 70 s hole, still 30% coverage
        with pytest.raises(ValueError, match="interior gap"):
            compute_speed_for_pfile(_em_pfile(em), {"method": "em"}, vehicle="vmp")

    def test_max_gap_is_configurable(self):
        em = np.full(400, 0.3)
        em[100:380] = np.nan
        with pytest.raises(ValueError, match="covers"):
            # Coverage still fails; raise both to accept.
            compute_speed_for_pfile(
                _em_pfile(em), {"method": "em", "max_gap_s": 1000.0}, vehicle="vmp"
            )
