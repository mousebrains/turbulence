# May-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb.qc_rules — declarative range-check QC."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from odas_tpw.perturb.qc_rules import (
    evaluate_rules,
    register_rule_channels,
)


class _StubPF:
    def __init__(self, *, t_slow, channels=None, fast_channels=None,
                 channel_info=None, fs_slow=64.0, fs_fast=512.0):
        self.t_slow = np.asarray(t_slow, dtype=np.float64)
        self.channels = channels or {}
        self.channel_info = channel_info or {}
        self._fast_channels = set(fast_channels or [])
        self.fs_slow = fs_slow
        self.fs_fast = fs_fast


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    def test_unknown_option_raises(self):
        pf = _StubPF(t_slow=np.arange(5))
        with pytest.raises(ValueError, match="unknown options"):
            evaluate_rules(pf, {"r": {"channel": "speed_fast", "bogus": 1}})

    def test_missing_channel_field_raises(self):
        pf = _StubPF(t_slow=np.arange(5))
        with pytest.raises(ValueError, match="'channel' is required"):
            evaluate_rules(pf, {"r": {"min": 0.0}})

    def test_non_power_of_2_bit_raises(self):
        pf = _StubPF(t_slow=np.arange(5),
                     channels={"speed_fast": np.zeros(5)})
        with pytest.raises(ValueError, match="power of 2"):
            evaluate_rules(pf, {"r": {"channel": "speed_fast", "bit": 3}})

    def test_oversized_bit_raises(self):
        pf = _StubPF(t_slow=np.arange(5),
                     channels={"speed_fast": np.zeros(5)})
        with pytest.raises(ValueError, match="power of 2"):
            evaluate_rules(pf, {"r": {"channel": "speed_fast", "bit": 256}})


# ---------------------------------------------------------------------------
# Range conditions
# ---------------------------------------------------------------------------


class TestRangeConditions:
    def test_min_only(self):
        pf = _StubPF(t_slow=np.arange(5),
                     channels={"speed_fast": np.array([0.01, 0.10, 0.20, 0.05, 0.30])})
        out = evaluate_rules(
            pf, {"speed_oor": {"channel": "speed_fast", "min": 0.05, "bit": 8}}
        )
        # min=0.05 → flag where x < 0.05.
        np.testing.assert_array_equal(out["speed_oor"], [8, 0, 0, 0, 0])

    def test_max_only(self):
        pf = _StubPF(t_slow=np.arange(5),
                     channels={"speed_fast": np.array([0.5, 1.0, 1.5, 2.0, 0.5])})
        out = evaluate_rules(
            pf, {"speed_oor": {"channel": "speed_fast", "max": 1.5, "bit": 8}}
        )
        np.testing.assert_array_equal(out["speed_oor"], [0, 0, 0, 8, 0])

    def test_abs_max(self):
        pf = _StubPF(t_slow=np.arange(5),
                     channels={"Incl_X": np.array([0, 30, -50, 60, -10])})
        out = evaluate_rules(
            pf, {"pitch": {"channel": "Incl_X", "abs_max": 45, "bit": 16}}
        )
        np.testing.assert_array_equal(out["pitch"], [0, 0, 16, 16, 0])

    def test_min_and_max_combined(self):
        pf = _StubPF(t_slow=np.arange(5),
                     channels={"speed_fast": np.array([0.01, 0.20, 0.50, 1.0, 2.0])})
        out = evaluate_rules(
            pf, {"sp": {"channel": "speed_fast", "min": 0.05, "max": 1.5, "bit": 8}}
        )
        np.testing.assert_array_equal(out["sp"], [8, 0, 0, 0, 8])

    def test_nans_count_as_flagged(self):
        pf = _StubPF(t_slow=np.arange(4),
                     channels={"x": np.array([1.0, np.nan, 2.0, np.inf])})
        out = evaluate_rules(
            pf, {"r": {"channel": "x", "min": 0.0, "max": 10.0, "bit": 1}}
        )
        # NaN and Inf both fail the np.isfinite gate even though they
        # don't trip min/max directly.
        assert out["r"][1] == 1
        assert out["r"][3] == 1


# ---------------------------------------------------------------------------
# Pseudo-name pitch / roll auto-detect
# ---------------------------------------------------------------------------


class TestInclinometerAutoPick:
    def test_pitch_picks_larger_amplitude_axis(self):
        n = 100
        # Incl_Y has bigger swing → "pitch" axis.
        pf = _StubPF(
            t_slow=np.arange(n),
            channels={
                "Incl_X": np.full(n, 1.0),                  # near constant
                "Incl_Y": np.linspace(-30.0, 30.0, n),     # 60° swing
            },
        )
        out = evaluate_rules(
            pf, {"pitch": {"channel": "pitch", "abs_max": 25, "bit": 16}}
        )
        # If pitch resolved to Incl_Y, |y|>25 fires at the ends.
        assert out["pitch"][0] == 16
        assert out["pitch"][-1] == 16
        # Around the middle, |y| < 25 → not flagged.
        assert out["pitch"][n // 2] == 0

    def test_roll_picks_other_axis(self):
        n = 50
        pf = _StubPF(
            t_slow=np.arange(n),
            channels={
                "Incl_X": np.full(n, 8.0),                  # constant 8°
                "Incl_Y": np.linspace(-30.0, 30.0, n),     # bigger
            },
        )
        out = evaluate_rules(
            pf, {"roll": {"channel": "roll", "abs_max": 5, "bit": 32}}
        )
        # Roll = Incl_X = 8°, |8| > 5 → all samples flagged.
        np.testing.assert_array_equal(out["roll"], np.full(n, 32, dtype=np.uint8))

    def test_pitch_skipped_when_inclinometer_missing(self, caplog):
        pf = _StubPF(t_slow=np.arange(5))  # no Incl channels
        with caplog.at_level(logging.WARNING):
            out = evaluate_rules(
                pf, {"pitch": {"channel": "pitch", "abs_max": 45, "bit": 16}}
            )
        assert "pitch" not in out
        assert any("requires Incl_X and Incl_Y" in r.message for r in caplog.records)

    def test_outlier_does_not_dominate_axis_pick(self):
        """Regression test: sl684-style spike in roll axis must not flip pitch.

        Real-data shape from RIOT sl684/MR433 deployment: a brief sensor
        saturation drove ``Incl_X`` (roll) to -90°, giving a min-max spread
        of ~155°. ``Incl_Y`` (the true pitch axis) had a typical glide
        cycle spread of ~99°. Naive ``nanmax - nanmin`` flipped the axis
        choice and produced a 92% pitch_oor flag rate.
        """
        n = 200
        # Pitch on Incl_Y: realistic ±25° glide cycle.
        incl_y = 25.0 * np.sin(2 * np.pi * np.arange(n) / 50.0)
        # Roll on Incl_X: ±5° normal flight + one -90° saturation spike.
        incl_x = 5.0 * np.sin(2 * np.pi * np.arange(n) / 30.0)
        incl_x[3] = -90.0  # the offender — only one sample
        pf = _StubPF(
            t_slow=np.arange(n),
            channels={"Incl_X": incl_x, "Incl_Y": incl_y},
        )
        # With min/max axis picking, Incl_X spread (~95°) > Incl_Y (~50°),
        # pitch would resolve to Incl_X and roll_oor (|x|>10°) would flag
        # nearly everything because pitch swings hit ±25°. With the
        # percentile-spread heuristic, the spike falls outside (1, 99) so
        # Incl_Y wins as pitch and roll resolves to Incl_X correctly.
        out = evaluate_rules(
            pf,
            {
                "pitch_oor": {"channel": "pitch", "abs_max": 45, "bit": 16},
                "roll_oor": {"channel": "roll", "abs_max": 10, "bit": 32},
            },
        )
        # Pitch on Incl_Y, |y|<=25 < 45 → no pitch flags.
        assert out["pitch_oor"].sum() == 0
        # Roll on Incl_X: only the -90° spike trips |x|>10 (plus a few
        # samples near the sinusoid extremes that legitimately exceed 10°
        # in principle, but with amplitude 5° they don't). The -90° spike
        # is the only flagged sample.
        assert out["roll_oor"][3] == 32
        assert int((out["roll_oor"] != 0).sum()) == 1

    def test_amplitude_quantile_yaml_override(self):
        """``amplitude_quantile`` per-rule controls the spread window.

        Using (0, 100) (i.e. nanmin/nanmax) reproduces the broken behavior:
        the outlier-dominated axis wins. This guards against a future
        regression that hard-codes the percentiles.
        """
        n = 200
        incl_y = 25.0 * np.sin(2 * np.pi * np.arange(n) / 50.0)
        incl_x = 5.0 * np.sin(2 * np.pi * np.arange(n) / 30.0)
        incl_x[3] = -90.0
        pf = _StubPF(
            t_slow=np.arange(n),
            channels={"Incl_X": incl_x, "Incl_Y": incl_y},
        )
        # Force min/max picking: Incl_X spread ~95° > Incl_Y ~50°, so
        # pitch incorrectly resolves to Incl_X. With aoa+pitch swings of
        # ±25 on the wrong axis, plenty of |pitch|>20 samples will fire.
        out = evaluate_rules(
            pf,
            {
                "pitch_oor": {
                    "channel": "pitch", "abs_max": 20, "bit": 16,
                    "amplitude_quantile": [0.0, 100.0],
                },
            },
        )
        # With wrong axis (Incl_X, ±5° normal + spike), |x|>20 only fires
        # at the spike → exactly one flagged sample.
        assert out["pitch_oor"][3] == 16
        assert int((out["pitch_oor"] != 0).sum()) == 1

    def test_amplitude_quantile_invalid_range_raises(self):
        pf = _StubPF(
            t_slow=np.arange(5),
            channels={"Incl_X": np.zeros(5), "Incl_Y": np.zeros(5)},
        )
        with pytest.raises(ValueError, match="amplitude_quantile"):
            evaluate_rules(
                pf,
                {"pitch": {"channel": "pitch", "abs_max": 45, "bit": 16,
                           "amplitude_quantile": [50.0, 50.0]}},
            )
        with pytest.raises(ValueError, match="amplitude_quantile"):
            evaluate_rules(
                pf,
                {"pitch": {"channel": "pitch", "abs_max": 45, "bit": 16,
                           "amplitude_quantile": [-1.0, 99.0]}},
            )
        with pytest.raises(ValueError, match="amplitude_quantile"):
            evaluate_rules(
                pf,
                {"pitch": {"channel": "pitch", "abs_max": 45, "bit": 16,
                           "amplitude_quantile": [1.0]}},
            )


# ---------------------------------------------------------------------------
# Missing channels warn-and-skip
# ---------------------------------------------------------------------------


class TestWarnAndSkip:
    def test_missing_channel_warns_no_raise(self, caplog):
        pf = _StubPF(t_slow=np.arange(5))
        with caplog.at_level(logging.WARNING):
            out = evaluate_rules(
                pf, {"sp": {"channel": "speed_fast", "min": 0.05, "bit": 8}}
            )
        assert out == {}
        assert any("not on pf.channels" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Fast-rate channels are coarsened to slow
# ---------------------------------------------------------------------------


class TestFastRateCoarsening:
    def test_max_pool_to_slow(self):
        # ratio = 4 → 16 fast samples → 4 slow samples
        n_slow = 4
        n_fast = 16
        speed_fast = np.zeros(n_fast)
        speed_fast[5] = 2.0  # one bad sample inside slow bin index 1
        pf = _StubPF(
            t_slow=np.arange(n_slow),
            channels={"speed_fast": speed_fast},
            fast_channels=["speed_fast"],
            fs_slow=4.0, fs_fast=16.0,
        )
        out = evaluate_rules(
            pf, {"sp": {"channel": "speed_fast", "max": 1.5, "bit": 8}}
        )
        # Slow bins: [0..3], [4..7] (← contains the spike), [8..11], [12..15].
        np.testing.assert_array_equal(out["sp"], [0, 8, 0, 0])


# ---------------------------------------------------------------------------
# register_rule_channels
# ---------------------------------------------------------------------------


class TestPitchWConsistencyRule:
    """Tests for the ``pitch_w_consistency`` rule type — flag samples
    where pitch direction and dP/dt sign disagree."""

    def _make_pf(self, *, pitch, P, fs_slow=1.0):
        n = len(pitch)
        # Add a tiny ramp to Incl_Y so the auto-picker has nonzero spread
        # to compare against Incl_X (also zero); otherwise the tie-break
        # picks Incl_X = roll, not Incl_Y = pitch.
        pitch_arr = np.asarray(pitch, dtype=np.float64) + np.linspace(-0.1, 0.1, n)
        return _StubPF(
            t_slow=np.arange(n),
            channels={
                "Incl_X": np.zeros(n),  # roll axis ~ flat
                "Incl_Y": pitch_arr,
                "P":      np.asarray(P, dtype=np.float64),
            },
            fs_slow=fs_slow, fs_fast=fs_slow * 8,
        )

    def test_consistent_climb_not_flagged(self):
        """Nose-up MR (Incl_Y < 0) while ascending (P decreasing) is fine."""
        # P decreasing: 100, 90, 80, ... ascending. Pitch = -20 (MR nose-up).
        n = 60
        P = 100.0 - np.arange(n) * 0.1   # dP/dt = -0.1 dbar/s -> ascending
        pitch = np.full(n, -20.0)
        pf = self._make_pf(pitch=pitch, P=P)
        out = evaluate_rules(pf, {
            "fc": {"type": "pitch_w_consistency", "bit": 64,
                   "pitch_min_deg": 5.0, "W_min_dbar_per_s": 0.02},
        })
        # Edge transients aside, interior should be unflagged.
        assert (out["fc"][10:-10] == 0).all()

    def test_consistent_dive_not_flagged(self):
        """Nose-down (Incl_Y > 0) while descending (P increasing) is fine."""
        n = 60
        P = np.arange(n) * 0.1
        pitch = np.full(n, +20.0)
        pf = self._make_pf(pitch=pitch, P=P)
        out = evaluate_rules(pf, {
            "fc": {"type": "pitch_w_consistency", "bit": 64,
                   "pitch_min_deg": 5.0, "W_min_dbar_per_s": 0.02},
        })
        assert (out["fc"][10:-10] == 0).all()

    def test_stalled_glider_flagged(self):
        """Nose-up but sinking — flag set."""
        n = 60
        P = np.arange(n) * 0.5         # rapid descent (W ~ +0.5 dbar/s)
        pitch = np.full(n, -25.0)      # but pitched nose-up
        pf = self._make_pf(pitch=pitch, P=P)
        out = evaluate_rules(pf, {
            "fc": {"type": "pitch_w_consistency", "bit": 64,
                   "pitch_min_deg": 5.0, "W_min_dbar_per_s": 0.02},
        })
        # Interior samples (skip filter edge transients) should all be flagged.
        assert (out["fc"][20:-20] != 0).all()

    def test_below_pitch_threshold_skipped(self):
        """|pitch| < pitch_min_deg → don't flag (noise zone around level)."""
        n = 60
        P = np.arange(n) * 0.5         # descending
        pitch = np.full(n, -2.0)       # nose-up but only -2°, below threshold
        pf = self._make_pf(pitch=pitch, P=P)
        out = evaluate_rules(pf, {
            "fc": {"type": "pitch_w_consistency", "bit": 64,
                   "pitch_min_deg": 5.0, "W_min_dbar_per_s": 0.02},
        })
        assert (out["fc"] == 0).all()

    def test_below_w_threshold_skipped(self):
        """|W| < W_min_dbar_per_s → don't flag (stationary)."""
        n = 60
        P = np.full(n, 100.0)          # not moving
        pitch = np.full(n, -25.0)      # nose-up
        pf = self._make_pf(pitch=pitch, P=P)
        out = evaluate_rules(pf, {
            "fc": {"type": "pitch_w_consistency", "bit": 64,
                   "pitch_min_deg": 5.0, "W_min_dbar_per_s": 0.02},
        })
        assert (out["fc"] == 0).all()

    def test_missing_inclinometer_warns_and_skips(self, caplog):
        pf = _StubPF(
            t_slow=np.arange(10),
            channels={"P": np.arange(10, dtype=np.float64)},
        )
        with caplog.at_level(logging.WARNING):
            out = evaluate_rules(pf, {
                "fc": {"type": "pitch_w_consistency", "bit": 64},
            })
        assert "fc" not in out
        assert any("Incl_X and Incl_Y" in r.message for r in caplog.records)

    def test_missing_P_channel_warns_and_skips(self, caplog):
        pf = _StubPF(
            t_slow=np.arange(10),
            channels={
                "Incl_X": np.zeros(10),
                "Incl_Y": np.full(10, -25.0),
            },
        )
        with caplog.at_level(logging.WARNING):
            out = evaluate_rules(pf, {
                "fc": {"type": "pitch_w_consistency", "bit": 64},
            })
        assert "fc" not in out
        assert any("requires P" in r.message for r in caplog.records)

    def test_unknown_type_raises(self):
        pf = _StubPF(t_slow=np.arange(5))
        with pytest.raises(ValueError, match="unknown type"):
            evaluate_rules(pf, {"fc": {"type": "bogus", "bit": 1}})

    def test_unknown_option_per_type(self):
        """Range-only options on a consistency rule should raise."""
        pf = _StubPF(t_slow=np.arange(5),
                     channels={"Incl_X": np.zeros(5), "Incl_Y": np.full(5, -25.0),
                               "P": np.arange(5.0)})
        with pytest.raises(ValueError, match="unknown options"):
            evaluate_rules(pf, {
                "fc": {"type": "pitch_w_consistency", "bit": 64,
                       "channel": "P"},   # 'channel' is range-only
            })


class TestRegisterRuleChannels:
    def test_registers_arrays_and_flag_attrs(self):
        pf = _StubPF(t_slow=np.arange(5))
        rule_arrays = {"sp": np.array([0, 8, 0, 0, 0], dtype=np.uint8)}
        rules = {"sp": {"channel": "speed_fast", "max": 1.5, "bit": 8}}
        register_rule_channels(pf, rule_arrays, rules)

        assert "sp" in pf.channels
        info = pf.channel_info["sp"]
        assert info["type"] == "qc_rule"
        assert info["flag_meanings"] == "sp"
        assert info["flag_masks"] == [8]
