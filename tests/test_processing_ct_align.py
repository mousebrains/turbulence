# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for processing.ct_align — CT sensor alignment."""

import numpy as np

from odas_tpw.processing.ct_align import ct_align


class TestCTAlign:
    def test_zero_lag_identical(self):
        """Identical signals should give ~zero lag."""
        fs = 64.0
        n = 2000
        t = np.arange(n) / fs
        T = np.sin(2 * np.pi * 0.5 * t) + 10.0
        C = T.copy()
        profiles = [(100, 1900)]

        _C_aligned, lag = ct_align(T, C, fs, profiles)
        assert abs(lag) < 2.0 / fs

    def test_shifted_signal_has_nonzero_lag(self):
        """A shifted signal should produce a nonzero lag."""
        fs = 64.0
        n = 3000
        t = np.arange(n) / fs
        T = np.sin(2 * np.pi * 0.3 * t) + 0.5 * np.sin(2 * np.pi * 1.0 * t)
        shift = 10  # samples — large enough shift to be clearly detectable
        C = np.roll(T, shift)
        profiles = [(200, 2800)]

        _C_aligned, lag = ct_align(T, C, fs, profiles)
        assert lag != 0.0

    def test_no_profiles_returns_copy(self):
        T = np.array([1.0, 2.0, 3.0])
        C = np.array([4.0, 5.0, 6.0])
        C_aligned, lag = ct_align(T, C, 64.0, [])
        np.testing.assert_array_equal(C_aligned, C)
        assert lag == 0.0

    def test_output_shape_matches(self):
        fs = 64.0
        n = 1000
        T = np.random.randn(n)
        C = np.random.randn(n)
        profiles = [(100, 900)]
        C_aligned, _ = ct_align(T, C, fs, profiles)
        assert C_aligned.shape == C.shape

    def test_short_profile_continues(self):
        """A profile shorter than 10 samples is skipped, not crashed on."""
        fs = 64.0
        T = np.linspace(0, 1, 100)
        C = T.copy()
        # First profile too short (< 10 samples), second is fine
        profiles = [(0, 5), (10, 90)]
        C_aligned, lag = ct_align(T, C, fs, profiles)
        assert C_aligned.shape == C.shape
        assert np.isfinite(lag)

    def test_all_profiles_too_short(self):
        """When every profile is too short, return identity copy."""
        fs = 64.0
        T = np.linspace(0, 1, 50)
        C = T.copy() * 0.5
        # Every profile <10 samples → per_profile stays empty → fallback return
        profiles = [(0, 4), (10, 14), (20, 24)]
        C_aligned, lag = ct_align(T, C, fs, profiles)
        np.testing.assert_array_equal(C_aligned, C)
        assert lag == 0.0

    def test_flatlined_channel_does_not_return_max_lag(self):
        """A constant (flatlined) C channel carries no alignment info; it must
        not yield the maximally-wrong -max_lag_seconds (audit #42)."""
        fs = 64.0
        n = 2000
        t = np.arange(n) / fs
        T = np.sin(2 * np.pi * 0.5 * t) + 10.0
        C = np.full(n, 5.0)  # flatlined conductivity
        profiles = [(100, 1900)]
        max_lag_seconds = 0.5
        C_aligned, lag = ct_align(T, C, fs, profiles, max_lag_seconds=max_lag_seconds)
        # The buggy path returned lag == -max_lag_seconds; the fix skips the
        # degenerate profile, so no usable correlation -> no shift.
        assert lag == 0.0
        np.testing.assert_array_equal(C_aligned, C)

    def test_weighted_median_multi_profile_consensus(self):
        """The rewritten weighted median (searchsorted to half-weight, #41)
        returns the consensus lag when multiple profiles agree."""
        fs = 64.0
        n = 3000
        t = np.arange(n) / fs
        base = np.sin(2 * np.pi * 0.3 * t) + 0.5 * np.sin(2 * np.pi * 1.1 * t)
        T = base
        C = np.roll(base, 5)  # both profiles share a 5-sample shift
        profiles = [(100, 1400), (1500, 2900)]
        _C_aligned, lag = ct_align(T, C, fs, profiles)
        assert abs(lag) == 5.0 / fs  # consensus magnitude, sign per convention
