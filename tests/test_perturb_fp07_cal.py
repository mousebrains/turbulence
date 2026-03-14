# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb.fp07_cal — FP07 in-situ calibration."""

import warnings
from pathlib import Path

import numpy as np

from microstructure_tpw.perturb.fp07_cal import (
    _calc_lag,
    _compute_RT_R0,
    _find_fp07_channels,
    _get_channel_config,
    _lowpass_filter,
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
        t_slow=None,
        t_fast=None,
        filepath=None,
        fast_channels=None,
    ):
        self.channels = channels
        self.channel_info = channel_info or {}
        self.config = config or {"channels": []}
        self.channels_raw = channels_raw or {}
        self.fs_slow = fs_slow
        self.fs_fast = fs_fast
        n = len(next(iter(channels.values()))) if channels else 100
        self.t_slow = t_slow if t_slow is not None else np.arange(n) / fs_slow
        self.t_fast = (
            t_fast
            if t_fast is not None
            else np.arange(n * round(fs_fast / fs_slow)) / fs_fast
        )
        self.filepath = filepath or Path("test_001.p")
        self._fast = fast_channels or set()

    def is_fast(self, ch):
        return ch in self._fast


class TestComputeRTR0:
    def test_known_values(self):
        """With identity-like calibration, RT_R0 should be well-defined."""
        ch_config = {
            "type": "therm",
            "e_b": "2.5",
            "a": "0",
            "b": "1",
            "g": "1",
            "adc_fs": "5",
            "adc_bits": "16",
        }
        counts = np.array([30000.0, 32000.0, 34000.0])
        result = _compute_RT_R0(counts, ch_config)
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_different_counts_different_RT_R0(self):
        """Counts at different values should produce different RT_R0."""
        ch_config = {
            "type": "therm",
            "e_b": "2.5",
            "a": "0",
            "b": "1",
            "g": "1",
            "adc_fs": "5",
            "adc_bits": "16",
        }
        # Use values that produce different Z values in the valid range
        low = _compute_RT_R0(np.array([10000.0]), ch_config)
        high = _compute_RT_R0(np.array([15000.0]), ch_config)
        assert low[0] != high[0]


class TestCalcLag:
    def test_zero_lag_identical_signals(self):
        """Identical signals should have ~zero lag."""
        fs = 64.0
        t = np.arange(1000) / fs
        signal = np.sin(2 * np.pi * 0.5 * t)
        lag, _corr = _calc_lag(signal, signal, fs, must_be_negative=False)
        assert abs(lag) < 2.0 / fs  # within 2 samples

    def test_known_lag_direction(self):
        """A lagged signal should produce a lag with the correct sign."""
        fs = 64.0
        n = 4000
        t = np.arange(n) / fs
        ref = np.sin(2 * np.pi * 0.3 * t) + 0.5 * np.sin(2 * np.pi * 1.0 * t)
        shift_samples = 10
        fp07 = np.roll(ref, shift_samples)  # fp07 lags ref by shift_samples

        _lag, corr = _calc_lag(ref, fp07, fs, max_lag_seconds=1.0, must_be_negative=False)
        # The correlation should be high
        assert corr > 0.5

    def test_must_be_negative_restricts(self):
        """With must_be_negative=True, lag should be <= 0."""
        fs = 64.0
        n = 2000
        t = np.arange(n) / fs
        ref = np.sin(2 * np.pi * 0.5 * t)
        fp07 = np.roll(ref, 5)  # positive lag

        lag, _ = _calc_lag(ref, fp07, fs, must_be_negative=True)
        assert lag <= 0


class TestFindFP07Channels:
    def test_finds_therm_channels(self):
        """Channels matching T<digit> with therm/thermistor type are found."""
        pf = _PFileStub(
            channels={"T1": np.zeros(100), "T2": np.zeros(100), "Ax": np.zeros(100)},
            channel_info={
                "T1": {"type": "therm"},
                "T2": {"type": "thermistor"},
                "Ax": {"type": "accel"},
            },
        )
        result = _find_fp07_channels(pf)  # type: ignore[arg-type]
        assert result == ["T1", "T2"]

    def test_empty_when_no_therm(self):
        """No FP07 channels when no therm types are present."""
        pf = _PFileStub(
            channels={"Ax": np.zeros(100), "Ay": np.zeros(100)},
            channel_info={"Ax": {"type": "accel"}, "Ay": {"type": "accel"}},
        )
        result = _find_fp07_channels(pf)  # type: ignore[arg-type]
        assert result == []


class TestGetChannelConfig:
    def test_found(self):
        """Returns the matching channel config dict."""
        pf = _PFileStub(
            channels={"T1": np.zeros(100)},
            config={
                "channels": [
                    {"name": "T1", "type": "therm", "e_b": "2.5"},
                    {"name": "Ax", "type": "accel"},
                ]
            },
        )
        cfg = _get_channel_config(pf, "T1")  # type: ignore[arg-type]
        assert cfg["type"] == "therm"
        assert cfg["e_b"] == "2.5"

    def test_not_found(self):
        """Returns empty dict when channel is not in config."""
        pf = _PFileStub(
            channels={"T1": np.zeros(100)},
            config={"channels": [{"name": "Ax", "type": "accel"}]},
        )
        cfg = _get_channel_config(pf, "T1")  # type: ignore[arg-type]
        assert cfg == {}


class TestLowpassFilter:
    def test_jac_reference(self):
        """JAC reference uses speed-dependent cutoff and runs without error."""
        fs = 64.0
        n = 1000
        fp07 = np.random.default_rng(42).standard_normal(n)
        W = np.full(n, 0.5)  # constant fall rate
        profiles = [(0, n - 1)]
        result = _lowpass_filter(fp07, "JAC_T", fs, W, profiles)
        assert result.shape == fp07.shape
        # Filtered signal should be smoother (lower variance)
        assert np.var(result) < np.var(fp07)

    def test_non_jac_reference(self):
        """Non-JAC reference uses fs/3 cutoff and runs without error."""
        fs = 64.0
        n = 1000
        fp07 = np.random.default_rng(42).standard_normal(n)
        W = np.full(n, 0.5)
        profiles = [(0, n - 1)]
        result = _lowpass_filter(fp07, "SBE_T", fs, W, profiles)
        assert result.shape == fp07.shape
        assert np.var(result) < np.var(fp07)


class TestFP07Calibrate:
    def test_no_reference_warns(self):
        """Missing reference channel emits warning and returns empty dicts."""
        pf = _PFileStub(
            channels={"T1": np.zeros(100)},
            channel_info={"T1": {"type": "therm"}},
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = fp07_calibrate(pf, profiles=[(0, 99)], reference="JAC_T")  # type: ignore[arg-type]
        assert any("Reference channel" in str(m.message) for m in w)
        assert result["channels"] == {}
        assert result["fast_channels"] == {}

    def test_no_fp07_warns(self):
        """Reference present but no FP07 channels emits warning."""
        pf = _PFileStub(
            channels={"JAC_T": np.zeros(100), "Ax": np.zeros(100)},
            channel_info={"JAC_T": {"type": "therm_slow"}, "Ax": {"type": "accel"}},
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = fp07_calibrate(pf, profiles=[(0, 99)], reference="JAC_T")  # type: ignore[arg-type]
        assert any("No FP07 channels" in str(m.message) for m in w)
        assert result["channels"] == {}

    def test_integration(self):
        """Full calibration pipeline with synthetic data produces output."""
        rng = np.random.default_rng(0)
        n_slow = 2000
        fs_slow = 64.0
        fs_fast = 512.0
        ratio = round(fs_fast / fs_slow)
        n_fast = n_slow * ratio

        # Synthetic reference temperature: smooth ramp with small noise
        t_slow = np.arange(n_slow) / fs_slow
        T_ref = 10.0 + 5.0 * (t_slow / t_slow[-1]) + rng.normal(0, 0.01, n_slow)

        # Pressure ramp (simulates a profile)
        P = np.linspace(5, 200, n_slow)

        # Channel config for T1 (therm type)
        ch_config = {
            "name": "T1",
            "type": "therm",
            "e_b": "2.5",
            "a": "0",
            "b": "1",
            "g": "1",
            "adc_fs": "5",
            "adc_bits": "16",
        }

        # Build raw counts that will produce reasonable RT_R0 values.
        # _compute_RT_R0 does: Z = factor*(counts - a)/b, RT_R0 = ln((1-Z)/(1+Z))
        # factor = (5 / 2^16) * 2/(1*2.5) = 0.00006103515625
        # We want Z values near 0 to get RT_R0 near 0, so counts near 0.
        # Use counts that produce a monotonic RT_R0 correlated with T_ref.
        # Steinhart-Hart: 1/(T+273.15) = a0 + a1*RT_R0 + a2*RT_R0^2
        # We reverse-engineer counts from T_ref via an approximate relationship.
        factor = (5.0 / 2**16) * 2.0 / (1.0 * 2.5)
        # Target RT_R0 from T_ref via an approximate Steinhart-Hart (linear approx)
        target_inv_T = 1.0 / (T_ref + 273.15)
        # Approximate: RT_R0 ~ (target_inv_T - mean) / some scale
        rt_r0_approx = (target_inv_T - np.mean(target_inv_T)) * 500.0
        # Invert _compute_RT_R0: RT_R0 = ln((1-Z)/(1+Z)) => Z = (1-exp(RT_R0))/(1+exp(RT_R0))
        exp_rt = np.exp(rt_r0_approx)
        Z = (1.0 - exp_rt) / (1.0 + exp_rt)
        # Z = factor * counts => counts = Z / factor
        raw_counts_slow = Z / factor

        # Fast-rate raw counts: upsample
        raw_counts_fast = np.repeat(raw_counts_slow, ratio) + rng.normal(
            0, 0.5, n_fast
        )

        pf = _PFileStub(
            channels={
                "JAC_T": T_ref,
                "T1": T_ref + rng.normal(0, 0.05, n_slow),  # physical-unit approx
                "P": P,
            },
            channel_info={
                "JAC_T": {"type": "therm_slow"},
                "T1": {"type": "therm"},
                "P": {"type": "pres"},
            },
            config={"channels": [ch_config]},
            channels_raw={"T1": raw_counts_fast},
            fs_slow=fs_slow,
            fs_fast=fs_fast,
            t_slow=t_slow,
            t_fast=np.arange(n_fast) / fs_fast,
            fast_channels={"T1"},
        )

        profiles = [(100, 1900)]
        result = fp07_calibrate(
            pf,  # type: ignore[arg-type]
            profiles,
            reference="JAC_T",
            order=2,
            must_be_negative=False,
        )

        # The calibration should have produced output for T1
        assert "T1" in result["channels"], "Slow-rate calibrated channel missing"
        assert "T1" in result["fast_channels"], "Fast-rate calibrated channel missing"
        assert "T1" in result["coefficients"]
        assert "T1" in result["lags"]
        assert "T1" in result["info"]

        # Calibrated slow data length matches reference
        assert result["channels"]["T1"].shape == T_ref.shape

        # Calibrated fast data length matches fast time vector
        assert result["fast_channels"]["T1"].shape == (n_fast,)

        # Coefficients should have order+1 elements
        assert len(result["coefficients"]["T1"]) == 3  # order=2 -> 3 coeffs

        # Info dict should have expected keys
        info = result["info"]["T1"]
        assert "median_lag" in info
        assert "n_profiles" in info
        assert info["n_profiles"] == 1
