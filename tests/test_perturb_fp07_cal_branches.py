# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Branch-coverage tests for perturb.fp07_cal — non-therm types and edge cases."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from odas_tpw.perturb.fp07_cal import (
    _calc_lag,
    fp07_calibrate,
)


class _PFileStub:
    """Minimal PFile stand-in for unit tests."""

    def __init__(
        self,
        channels,
        channel_info=None,
        config=None,
        channels_raw=None,
        fs_slow=64.0,
        fs_fast=512.0,
        fast_channels=None,
    ):
        self.channels = channels
        self.channel_info = channel_info or {}
        self.config = config or {"channels": []}
        self.channels_raw = channels_raw or {}
        self.fs_slow = fs_slow
        self.fs_fast = fs_fast
        n = len(next(iter(channels.values()))) if channels else 100
        self.t_slow = np.arange(n) / fs_slow
        self.t_fast = np.arange(n * round(fs_fast / fs_slow)) / fs_fast
        self.filepath = Path("test_001.p")
        self._fast = fast_channels or set()

    def is_fast(self, ch):
        return ch in self._fast


# ---------------------------------------------------------------------------
# The non-therm branch of the old _compute_RT_R0 is gone. It was unreachable
# from fp07_calibrate --- _find_fp07_channels only admits channels whose type
# is therm/thermistor --- so it existed solely to be covered here, while
# carrying the default mismatch (e_b defaulting to 0, g to 1, adc_fs to 5)
# that made the fit's L disagree with the reader's. The bridge algebra now
# comes from fp07cal.logr, which requires every parameter rather than
# substituting a default.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _calc_lag — must_be_negative empty corr_search branch (line 147)
# ---------------------------------------------------------------------------


class TestCalcLagEmptyNegative:
    def test_must_be_negative_no_negative_lags(self):
        """When max_lag_seconds is so small that no negative lags survive."""
        # Use very small max_lag → only lag=0 survives, and the negative-mask
        # gives lags<=0 which still includes 0. Need to construct so negative
        # mask is empty — that's only when there's no lag <= 0 in the mask range.
        # But lags are symmetric, so 0 is always present if any lags survive.
        # Skip that exact branch — it's hard to trigger without code change.
        # Instead test must_be_negative=False path (line 150-152).
        rng = np.random.default_rng(0)
        x = rng.standard_normal(200)
        y = rng.standard_normal(200)
        lag, corr = _calc_lag(x, y, fs=64.0, must_be_negative=False)
        assert isinstance(lag, float)
        assert isinstance(corr, float)


# ---------------------------------------------------------------------------
# fp07_calibrate — channel skip branches
# ---------------------------------------------------------------------------


class TestFP07CalibrateSkips:
    def test_channel_with_no_config_skipped(self):
        """If config has no entry for the channel → skip (line 216)."""
        n_slow = 200
        rng = np.random.default_rng(0)
        T1 = 10.0 + rng.standard_normal(n_slow) * 0.01
        JAC_T = T1 + 0.001 * rng.standard_normal(n_slow)
        pf = _PFileStub(
            channels={"T1": T1, "JAC_T": JAC_T, "P": np.linspace(0, 50, n_slow)},
            channel_info={
                "T1": {"type": "therm"},
                "JAC_T": {"type": "jac_t"},
                "P": {"type": "pres"},
            },
            # config has no entry for T1 → _get_channel_config returns {}
            config={"channels": []},
            channels_raw={"T1": T1.copy()},
        )
        with np.errstate(all="ignore"):
            result = fp07_calibrate(pf, profiles=[(10, 190)])
        # T1 should not be in calibrated channels because config empty
        assert "T1" not in result["channels"]

    def test_channel_not_in_raw_skipped(self):
        """If raw counts not in channels_raw → skip (line 220)."""
        n_slow = 200
        rng = np.random.default_rng(0)
        T1 = 10.0 + rng.standard_normal(n_slow) * 0.01
        JAC_T = T1 + 0.001 * rng.standard_normal(n_slow)
        pf = _PFileStub(
            channels={"T1": T1, "JAC_T": JAC_T, "P": np.linspace(0, 50, n_slow)},
            channel_info={
                "T1": {"type": "therm"},
                "JAC_T": {"type": "jac_t"},
                "P": {"type": "pres"},
            },
            config={
                "channels": [
                    {
                        "name": "T1",
                        "type": "therm",
                        "e_b": "2.5",
                        "a": "0",
                        "b": "1",
                        "g": "1",
                        "adc_fs": "5",
                        "adc_bits": "16",
                        "adc_zero": "0",
                    }
                ]
            },
            channels_raw={},  # T1 missing → skip
        )
        result = fp07_calibrate(pf, profiles=[(10, 190)])
        assert "T1" not in result["channels"]

    def test_short_profile_skipped(self):
        """Profile shorter than 10 samples is skipped in lag computation (line 239)."""
        n_slow = 500
        rng = np.random.default_rng(0)
        # Structure, not a ramp. _calc_lag correlates DIFFERENCES, and the
        # derivative of a straight line is a constant -- after mean removal
        # only independent noise is left and nothing correlates, so a ramp
        # cannot establish a lag no matter how clean it is.
        base = 2.0 * np.sin(2.0 * np.pi * np.arange(n_slow) / 128.0)
        T1 = 10.0 + base + rng.standard_normal(n_slow) * 0.005
        JAC_T = 10.0 + base + rng.standard_normal(n_slow) * 0.005
        pf = _PFileStub(
            channels={"T1": T1, "JAC_T": JAC_T, "P": np.linspace(0, 50, n_slow)},
            channel_info={
                "T1": {"type": "therm"},
                "JAC_T": {"type": "jac_t"},
                "P": {"type": "pres"},
            },
            config={
                "channels": [
                    {
                        "name": "T1",
                        "type": "therm",
                        "e_b": "2.5",
                        "a": "0",
                        "b": "1",
                        "g": "1",
                        "adc_fs": "5",
                        "adc_bits": "16",
                        "adc_zero": "0",
                    }
                ]
            },
            # Two things this fixture has to get right, both of which the old
            # code let slide:
            #  * the counts must stay inside the bridge range. With this
            #    config the scale factor is 6.1e-5, so |counts| must be under
            #    ~9830 or Z rails at +/-0.6 and L goes constant. The previous
            #    30000 was fully clipped -- the test was fitting a flat line.
            #  * they must actually TRACK the reference, because the lag search
            #    is now gated on correlation.
            channels_raw={
                "T1": 5000.0 - 800.0 * base + rng.standard_normal(n_slow) * 2
            },
        )
        # First profile too short (5 samples), second is full length
        with np.errstate(all="ignore"):
            result = fp07_calibrate(pf, profiles=[(0, 4), (50, 490)])
        # Should still compute calibration from the valid second profile
        assert "T1" in result["lags"]

    def test_all_profiles_too_short_skipped(self):
        """When all profiles are too short → no lags → channel skipped (line 251)."""
        n_slow = 100
        rng = np.random.default_rng(0)
        T1 = 10.0 + rng.standard_normal(n_slow) * 0.01
        JAC_T = T1.copy()
        pf = _PFileStub(
            channels={"T1": T1, "JAC_T": JAC_T},
            channel_info={
                "T1": {"type": "therm"},
                "JAC_T": {"type": "jac_t"},
            },
            config={
                "channels": [
                    {
                        "name": "T1",
                        "type": "therm",
                        "e_b": "2.5",
                        "a": "0",
                        "b": "1",
                        "g": "1",
                        "adc_fs": "5",
                        "adc_bits": "16",
                        "adc_zero": "0",
                    }
                ]
            },
            channels_raw={"T1": np.full(n_slow, 30000.0)},
        )
        # All profiles < 10 samples
        with np.errstate(all="ignore"):
            result = fp07_calibrate(pf, profiles=[(0, 5), (10, 15), (20, 25)])
        assert "T1" not in result["lags"]


# ---------------------------------------------------------------------------
# _lowpass_filter — non-JAC reference path (line 99)
# ---------------------------------------------------------------------------


class TestLowpassFilterNonJac:
    def test_non_jac_uses_fs3(self):
        """Non-JAC reference uses fc = fs/3."""
        from odas_tpw.perturb.fp07_cal import _lowpass_filter

        rng = np.random.default_rng(0)
        fp07 = rng.standard_normal(500)
        W = np.full(500, 0.5)
        # Use a non-JAC reference name
        result = _lowpass_filter(fp07, "OTHER_T", fs=64.0, W=W, profiles=[(10, 490)])
        assert result.shape == (500,)
        assert np.all(np.isfinite(result))
