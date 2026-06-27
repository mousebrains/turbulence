# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for the deconvolve module (pre-emphasis deconvolution)."""

import numpy as np
from scipy.signal import butter, lfilter

from odas_tpw.rsi.deconvolve import _interp_if_required, deconvolve


class TestDeconvolveKnownSignal:
    """Deconvolve a known pre-emphasized sinusoid and verify recovery."""

    def test_recover_sinusoid(self):
        """Apply first-order high-pass (pre-emphasis), then deconvolve to recover."""
        fs = 512.0
        diff_gain = 0.006  # ~26 Hz cutoff, typical FP07
        N = 4096
        t = np.arange(N) / fs

        # Low-frequency sinusoid well within passband
        freq = 2.0  # Hz
        X_true = 10.0 + 0.5 * np.sin(2 * np.pi * freq * t)

        # Simulate pre-emphasis: first-order high-pass via differentiator
        f_c = 1.0 / (2.0 * np.pi * diff_gain)
        b, a = butter(1, f_c / (fs / 2.0))
        X_dX = lfilter(b, a, X_true)

        result = deconvolve(X_true, X_dX, fs, diff_gain)

        assert result.shape == X_dX.shape
        # Trim transient at the start and compare
        trim = int(0.5 * fs)
        np.testing.assert_allclose(result[trim:], X_true[trim:], rtol=0.05)


class TestDeconvolveXNone:
    """Deconvolve with no reference signal (X=None)."""

    def test_returns_correct_length(self):
        fs = 512.0
        diff_gain = 0.006
        N = 2048
        X_dX = np.random.default_rng(42).standard_normal(N)

        result = deconvolve(None, X_dX, fs, diff_gain)

        assert isinstance(result, np.ndarray)
        assert result.shape == (N,)
        assert np.all(np.isfinite(result))


class TestDeconvolveEmpty:
    """Deconvolve with empty input."""

    def test_empty_returns_empty(self):
        result = deconvolve(None, np.array([]), 512.0, 0.006)

        assert isinstance(result, np.ndarray)
        assert len(result) == 0


class TestDeconvolveInterpolation:
    """Deconvolve when X is shorter than X_dX (triggers interpolation)."""

    def test_x_shorter_triggers_interp(self):
        fs = 512.0
        diff_gain = 0.006
        N_fast = 4096
        N_slow = N_fast // 8  # simulate 8:1 fast/slow ratio

        t_fast = np.arange(N_fast) / fs
        X_true = 15.0 + 0.3 * np.sin(2 * np.pi * 1.0 * t_fast)

        # Pre-emphasize
        f_c = 1.0 / (2.0 * np.pi * diff_gain)
        b, a = butter(1, f_c / (fs / 2.0))
        X_dX = lfilter(b, a, X_true)

        # Create slow-rate version of X (subsampled)
        X_slow = X_true[::8]
        assert len(X_slow) == N_slow

        result = deconvolve(X_slow, X_dX, fs, diff_gain)

        assert result.shape == (N_fast,)
        assert np.all(np.isfinite(result))
        # After transient, should approximate the original signal
        trim = int(0.5 * fs)
        np.testing.assert_allclose(result[trim:], X_true[trim:], rtol=0.10)


class TestDeconvolveSignInversion:
    """Exercise the inverted-derivative-circuit branch (p[0] < -0.5).

    Regression guard for the in-place sign correction at deconvolve.py:66-68,
    which negates X_dX when the pre-emphasis derivative circuit is wired with
    inverted polarity (ODAS deconvolve.m:188 ``if p(1)<-0.5, X_dX=-X_dX``).
    Previously this branch (and its int16 -> float64 promotion guard) had no
    test coverage; a flipped threshold or a dropped promotion would silently
    produce sign-flipped output.
    """

    def test_inverted_signal_recovered(self):
        """An inverted pre-emphasized signal is un-inverted and recovered.

        With X_dX = -lfilter(...), polyfit(X_true, X_dX) has slope ~-1 (< -0.5),
        so the branch fires; the recovered signal must match X_true, not its
        negation.
        """
        fs = 512.0
        diff_gain = 0.006
        N = 4096
        t = np.arange(N) / fs
        X_true = 10.0 + 0.5 * np.sin(2 * np.pi * 2.0 * t)

        f_c = 1.0 / (2.0 * np.pi * diff_gain)
        b, a = butter(1, f_c / (fs / 2.0))
        # Inverted derivative circuit: pre-emphasized signal is negated.
        X_dX_inverted = -lfilter(b, a, X_true)

        # Confirm this input actually triggers the p[0] < -0.5 branch.
        assert np.polyfit(X_true, X_dX_inverted, 1)[0] < -0.5

        result = deconvolve(X_true, X_dX_inverted, fs, diff_gain)

        assert result.shape == X_dX_inverted.shape
        assert np.all(np.isfinite(result))
        trim = int(0.5 * fs)
        # Recovers X_true (un-inverted), NOT -X_true.
        np.testing.assert_allclose(result[trim:], X_true[trim:], rtol=0.05)

    def test_int16_negation_promotes_to_float64(self):
        """The branch promotes int16 to float64 before negating.

        ``-np.int16(-32768)`` wraps back to -32768 (two's-complement overflow),
        so the ``.astype(np.float64)`` in the negation is load-bearing.  Build
        an anti-correlated int16 X_dX containing -32768 and assert the result
        is finite and float-typed (a missing promotion corrupts the extreme
        sample and poisons the lfilter output).
        """
        fs = 512.0
        diff_gain = 0.006
        X = np.linspace(0.0, 1000.0, 512)
        # Anti-correlated int16 signal -> polyfit slope < -0.5 (branch fires).
        X_dX = (-X).astype(np.int16)
        X_dX[0] = np.int16(-32768)  # extreme that wraps under a naive negate

        # Sanity: the branch must actually fire for this input.
        assert np.polyfit(X, X_dX.astype(np.float64), 1)[0] < -0.5
        # Sanity: naive int16 negate would NOT recover +32768 (the bug guarded).
        with np.errstate(over="ignore"):
            assert (-X_dX[0]).astype(np.int16) == np.int16(-32768)

        result = deconvolve(X, X_dX, fs, diff_gain)

        assert result.shape == X_dX.shape
        assert np.issubdtype(result.dtype, np.floating)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# _interp_if_required
# ---------------------------------------------------------------------------


class TestInterpIfRequiredNone:
    """_interp_if_required with X=None returns None."""

    def test_none_returns_none(self):
        X_dX = np.arange(100, dtype=float)
        result = _interp_if_required(None, X_dX, 512.0)
        assert result is None


class TestInterpIfRequiredMatchingLengths:
    """_interp_if_required with matching lengths returns X unchanged."""

    def test_same_length_returns_same(self):
        X = np.linspace(0, 10, 100)
        X_dX = np.ones(100)
        result = _interp_if_required(X, X_dX, 512.0)
        np.testing.assert_array_equal(result, X)


class TestInterpIfRequiredDifferentLengths:
    """_interp_if_required with different lengths returns interpolated array."""

    def test_interpolated_correct_length(self):
        N_slow = 64
        N_fast = 512
        X = np.linspace(5.0, 25.0, N_slow)
        X_dX = np.ones(N_fast)

        result = _interp_if_required(X, X_dX, 512.0)

        assert result is not None
        assert len(result) == N_fast
        assert np.all(np.isfinite(result))
        # Endpoints should be close to the original range
        np.testing.assert_allclose(result[0], X[0], rtol=0.01)
        np.testing.assert_allclose(result[-1], X[-1], rtol=0.02)

    def test_single_element_x_returns_unchanged(self):
        """A single-element X array should be returned as-is (len <= 1 guard)."""
        X = np.array([42.0])
        X_dX = np.ones(512)
        result = _interp_if_required(X, X_dX, 512.0)
        np.testing.assert_array_equal(result, X)
