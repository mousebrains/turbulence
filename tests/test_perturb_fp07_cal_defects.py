# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""Regressions for the defects the per-file FP07 calibration used to carry.

Each test here corresponds to a way ``fp07_calibrate`` could previously emit
confident, wrong coefficients.  They are separate from the existing suite
because that suite pins behaviour these fix.
"""

import warnings
from datetime import UTC, datetime

import numpy as np
import pytest

from odas_tpw.perturb.fp07_cal import (
    _lowpass_filter,
    _polyfit_centered,
    _reference_interval,
    fp07_calibrate,
)

_CFG = {
    "name": "T1", "type": "therm", "e_b": "2.5", "a": "0", "b": "1",
    "g": "1", "adc_fs": "5", "adc_bits": "16",
}


class _PF:
    """PFile-like stub, slow-rate only."""

    def __init__(self, channels, channels_raw, config, channel_info):
        self.channels = channels
        self.channels_raw = channels_raw
        self.config = config
        self.channel_info = channel_info
        self.fs_fast = 512.0
        self.fs_slow = 64.0
        n = len(next(iter(channels.values())))
        self.t_slow = np.arange(n) / self.fs_slow
        self.t_fast = np.arange(n) / self.fs_slow
        self.start_time = datetime.fromtimestamp(1.7e9, tz=UTC)

    def is_fast(self, name):
        return False


def _make(n=500, counts=None, ref=None, cfg=None):
    base = 2.0 * np.sin(2.0 * np.pi * np.arange(n) / 128.0)
    rng = np.random.default_rng(0)
    T_ref = 10.0 + base + rng.standard_normal(n) * 0.005 if ref is None else ref
    raw = 5000.0 - 800.0 * base + rng.standard_normal(n) * 2 if counts is None else counts
    return _PF(
        channels={"T1": np.zeros(n), "JAC_T": T_ref, "P": np.linspace(0, 50, n)},
        channels_raw={"T1": raw},
        config={"channels": [dict(cfg or _CFG)]},
        channel_info={"T1": {"type": "therm"}, "JAC_T": {"type": "jac_t"},
                      "P": {"type": "pres"}},
    )


# --- A4: one implementation of L, and no substituted defaults --------------
def test_missing_bridge_parameter_is_refused_not_defaulted():
    """A guessed e_b makes the fit's L disagree with the reader's, silently."""
    cfg = {k: v for k, v in _CFG.items() if k != "e_b"}
    pf = _make(cfg=cfg)
    with pytest.warns(UserWarning, match="missing bridge parameter"):
        out = fp07_calibrate(pf, [(10, 480)])
    assert "T1" not in out["coefficients"]


# --- A12: a lag has to correlate with something ----------------------------
def test_uncorrelated_reference_is_refused():
    """The old code fitted a fabricated reference at r ~ 0.02 without complaint."""
    n = 500
    # A reference that is a smooth ramp -- what interpolating across a gap
    # produces -- against a probe carrying real structure.
    pf = _make(n=n, ref=19.0 + np.linspace(0.0, 0.25, n))
    with pytest.warns(UserWarning, match="lag correlation gate"):
        out = fp07_calibrate(pf, [(10, 480)])
    assert "T1" not in out["coefficients"]
    assert "T1" not in out["channels"]


def test_correlated_reference_is_accepted_and_reports_its_evidence():
    pf = _make()
    out = fp07_calibrate(pf, [(10, 480)], order=1)
    assert "T1" in out["coefficients"]
    info = out["info"]["T1"]
    assert abs(info["median_corr"]) >= 0.5
    assert info["n_profiles"] >= 1
    assert np.isfinite(info["condition_number"])


# --- A7: clipped samples are not silently regressed ------------------------
def test_fully_railed_channel_is_refused():
    """Counts outside the bridge range give a constant L that looks ordinary."""
    n = 500
    base = 2.0 * np.sin(2.0 * np.pi * np.arange(n) / 128.0)
    # 30000 counts with this config puts |Z| at ~1.8, hard against the 0.6 rail.
    pf = _make(n=n, counts=30000.0 - 800.0 * base)
    with pytest.warns(UserWarning, match="bridge rail"):
        out = fp07_calibrate(pf, [(10, 480)])
    assert "T1" not in out["coefficients"]


# --- A3: conditioning ------------------------------------------------------
def test_centered_fit_is_well_conditioned_and_exact():
    """Centering must improve conditioning without changing the curve."""
    rng = np.random.default_rng(3)
    L = -0.12 + 0.004 * rng.standard_normal(4000)
    truth = np.array([1.0 / 288.0, 1.0 / 3100.0, 1.0 / 2.5e5])
    target = truth[0] + truth[1] * L + truth[2] * L**2

    coeffs, cond = _polyfit_centered(L, target, 2)
    raw = np.linalg.cond(np.column_stack([L**i for i in range(3)]))
    assert cond < raw / 100.0
    # And the recovered polynomial still evaluates to the same curve.
    got = coeffs[0] + coeffs[1] * L + coeffs[2] * L**2
    np.testing.assert_allclose(got, target, rtol=1e-9, atol=1e-14)


# --- A2: bandwidth matching -------------------------------------------------
def test_reference_interval_is_inferred_from_an_interpolated_array():
    fs = 64.0
    n = 1280
    rng = np.random.default_rng(4)
    knots = np.arange(0, n + 1, 64, dtype=float)      # 1 Hz on a 64 Hz grid
    T_ref = np.interp(np.arange(n), knots, rng.standard_normal(knots.size))
    assert _reference_interval(T_ref, fs) == pytest.approx(1.0, rel=0.2)


def test_reference_interval_of_a_fast_reference_is_the_grid_rate():
    fs = 64.0
    T_ref = np.random.default_rng(5).standard_normal(1000)
    assert _reference_interval(T_ref, fs) == pytest.approx(1.0 / fs)


def test_non_jac_lowpass_actually_matches_the_reference():
    """fs/3 on a 64 Hz grid is ~21 Hz -- no filtering, dressed up as matched."""
    fs = 64.0
    n = 1280
    rng = np.random.default_rng(6)
    fp07 = rng.standard_normal(n)
    knots = np.arange(0, n + 1, 64, dtype=float)
    T_ref = np.interp(np.arange(n), knots, rng.standard_normal(knots.size))
    matched = _lowpass_filter(fp07, "SBE_T", fs, np.full(n, 0.5), [(0, n - 1)],
                              T_ref=T_ref)
    assert np.var(matched) < 0.5 * np.var(fp07)


def test_jac_branch_is_unchanged():
    """The VMP path has a vendor relation and must not be disturbed."""
    fs = 64.0
    n = 1000
    fp07 = np.random.default_rng(7).standard_normal(n)
    out = _lowpass_filter(fp07, "JAC_T", fs, np.full(n, 0.5), [(0, n - 1)])
    assert out.shape == fp07.shape
    assert np.var(out) < np.var(fp07)


# --- review finding on PR #151 ---------------------------------------------
def test_a_rejected_profile_reaches_neither_the_lag_nor_the_fit():
    """"Contributes nothing" has to mean nothing.

    min_corr used to exclude a profile only from the median lag; the
    regression still iterated the original `profiles` list, so one accepted
    profile laundered every rejected one into the coefficients and the rail
    fraction.
    """
    import odas_tpw.perturb.fp07_cal as F

    n = 400
    rng = np.random.default_rng(0)
    base = 2.0 * np.sin(2.0 * np.pi * np.arange(n) / 128.0)
    # First profile: the probe tracks the reference. Second: the reference is a
    # smooth fabricated ramp, exactly what interpolating across a gap produces.
    T_ref = np.empty(n)
    T_ref[:200] = 10.0 + base[:200] + rng.standard_normal(200) * 0.005
    T_ref[200:] = 19.0 + np.linspace(0.0, 0.25, n - 200)
    raw = 5000.0 - 800.0 * base + rng.standard_normal(n) * 2
    pf = _PF(
        channels={"T1": np.zeros(n), "JAC_T": T_ref, "P": np.linspace(0, 50, n)},
        channels_raw={"T1": raw},
        config={"channels": [dict(_CFG)]},
        channel_info={"T1": {"type": "therm"}, "JAC_T": {"type": "jac_t"},
                      "P": {"type": "pres"}},
    )

    seen: dict = {}
    original = F._polyfit_centered

    def spy(L, target, order):
        seen["n"] = L.size
        return original(L, target, order)

    F._polyfit_centered = spy
    try:
        # No warning here: one profile IS accepted, so the channel is fitted.
        out = F.fp07_calibrate(pf, [(5, 195), (205, 395)], order=1)
    finally:
        F._polyfit_centered = original

    info = out["info"]["T1"]
    assert info["n_profiles"] == 1
    assert info["n_profiles_rejected"] == 1
    # The accepted profile is 191 samples; both together would be 382.
    assert seen["n"] <= 191, (
        f"the regression saw {seen['n']} samples -- the rejected profile leaked in"
    )


# --- Review round 2: bandwidth matching on the default (pchip) path ---------
def _pchip_reference(n: int, fs: float, seed: int = 7) -> tuple[np.ndarray, float]:
    """A 1 Hz reference merged onto the grid with pchip, as perturb does."""
    from scipy.interpolate import PchipInterpolator

    rng = np.random.default_rng(seed)
    knots_t = np.arange(0, n / fs + 1.0, 1.0)
    T_ref = PchipInterpolator(knots_t, rng.standard_normal(knots_t.size))(
        np.arange(n) / fs
    )
    return T_ref, 1.0


def test_pchip_merged_reference_is_filtered_via_explicit_interval():
    fs = 64.0
    n = 1280
    rng = np.random.default_rng(8)
    fp07 = rng.standard_normal(n)
    T_ref, dt = _pchip_reference(n, fs)
    # Inference alone finds no knots in a pchip merge...
    assert _reference_interval(T_ref, fs) == pytest.approx(1.0 / fs)
    # ...but the explicit interval engages the filter at 0.5 Hz.
    matched = _lowpass_filter(fp07, "SBE_T", fs, np.full(n, 0.5), [(0, n - 1)],
                              T_ref=T_ref, reference_interval=dt)
    assert np.var(matched) < 0.1 * np.var(fp07)


def test_fallback_warns_when_interval_unknown_and_inference_fails():
    fs = 64.0
    n = 1280
    fp07 = np.random.default_rng(9).standard_normal(n)
    T_ref, _ = _pchip_reference(n, fs)
    with pytest.warns(UserWarning, match="NOT low-pass filtered"):
        out = _lowpass_filter(fp07, "SBE_T", fs, np.full(n, 0.5), [(0, n - 1)],
                              T_ref=T_ref)
    np.testing.assert_array_equal(out, fp07)


def test_fp07_calibrate_passes_reference_interval_through():
    """End to end: pchip reference + explicit interval -> no warning, filtered fit."""
    n = 500
    pf = _make(n=n)
    pf.channels["SBE_T"] = pf.channels.pop("JAC_T")   # non-JAC -> bandwidth path
    # Without the interval the fallback inference fails on this reference
    # (no linear-merge knots) and says so.
    with pytest.warns(UserWarning, match="NOT low-pass filtered"):
        fp07_calibrate(pf, [(10, 480)], reference="SBE_T", order=1)
    # With it, silence and a fit.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        out = fp07_calibrate(
            pf, [(10, 480)], reference="SBE_T", order=1, reference_interval=1.0
        )
    assert "T1" in out["coefficients"]


def test_reference_interval_survives_a_nan_run():
    fs = 64.0
    n = 64 * 200
    rng = np.random.default_rng(10)
    knots = np.arange(0, n + 1, 64, dtype=float)
    T_ref = np.interp(np.arange(n), knots, rng.standard_normal(knots.size))
    T_ref[3000:6000] = np.nan
    assert _reference_interval(T_ref, fs) == pytest.approx(1.0, rel=0.2)


def test_reference_interval_survives_a_single_spike():
    fs = 64.0
    n = 64 * 200
    rng = np.random.default_rng(11)
    knots = np.arange(0, n + 1, 64, dtype=float)
    T_ref = np.interp(np.arange(n), knots, rng.standard_normal(knots.size))
    T_ref[5000] += 5.0
    assert _reference_interval(T_ref, fs) == pytest.approx(1.0, rel=0.2)
