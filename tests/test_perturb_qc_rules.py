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
