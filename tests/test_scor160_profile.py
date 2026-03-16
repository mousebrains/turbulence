# Tests for odas_tpw.scor160.profile
"""Unit tests for profile detection from pressure time series."""

import numpy as np
import pytest

from odas_tpw.scor160.profile import get_profiles, smooth_fall_rate

# ---------------------------------------------------------------------------
# smooth_fall_rate
# ---------------------------------------------------------------------------


class TestSmoothFallRateBasic:
    """Core behaviour of smooth_fall_rate."""

    def test_linear_ramp_constant_speed(self):
        """Linear pressure increase should give nearly constant fall rate."""
        fs = 64.0
        duration = 30.0  # long enough for filter to settle
        t = np.arange(0, duration, 1 / fs)
        speed = 0.7  # dbar/s
        P = speed * t

        W = smooth_fall_rate(P, fs)

        # Interior points (avoid filter edge effects) should be close to speed
        interior = slice(int(5 * fs), int(25 * fs))
        np.testing.assert_allclose(W[interior], speed, atol=0.01)

    def test_constant_pressure_zero_speed(self):
        """Constant pressure should give zero fall rate everywhere."""
        fs = 64.0
        P = 10.0 * np.ones(int(20 * fs))

        W = smooth_fall_rate(P, fs)
        np.testing.assert_allclose(W, 0.0, atol=1e-12)

    def test_output_shape_matches_input(self):
        fs = 64.0
        P = np.linspace(0, 50, int(10 * fs))
        W = smooth_fall_rate(P, fs)
        assert W.shape == P.shape


class TestSmoothFallRateTau:
    """Effect of the tau smoothing parameter."""

    def test_larger_tau_smoother(self):
        """Larger tau should produce a smoother (lower-variance) output."""
        rng = np.random.default_rng(42)
        fs = 64.0
        t = np.arange(0, 60, 1 / fs)
        P = 0.6 * t + 0.05 * rng.standard_normal(len(t))

        W_small = smooth_fall_rate(P, fs, tau=0.5)
        W_large = smooth_fall_rate(P, fs, tau=5.0)

        # Compare interior only to avoid filter edge transients
        interior = slice(int(15 * fs), int(45 * fs))
        assert np.std(W_large[interior]) < np.std(W_small[interior])

    def test_small_tau_tracks_faster(self):
        """Smaller tau should respond faster to a step change in speed."""
        fs = 64.0
        n = int(30 * fs)
        P = np.zeros(n)
        # First half: speed 0.5 dbar/s, second half: 1.5 dbar/s
        mid = n // 2
        P[:mid] = 0.5 * np.arange(mid) / fs
        P[mid:] = P[mid - 1] + 1.5 * np.arange(n - mid) / fs

        W_fast = smooth_fall_rate(P, fs, tau=0.3)
        W_slow = smooth_fall_rate(P, fs, tau=5.0)

        # Shortly after the transition, fast should be closer to 1.5
        check = mid + int(1.0 * fs)  # 1 s after transition
        assert abs(W_fast[check] - 1.5) < abs(W_slow[check] - 1.5)


class TestSmoothFallRateEdge:
    """Edge cases."""

    def test_very_short_array_raises(self):
        """Array shorter than filtfilt padlen should raise ValueError."""
        fs = 64.0
        P = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="padlen"):
            smooth_fall_rate(P, fs)


# ---------------------------------------------------------------------------
# get_profiles
# ---------------------------------------------------------------------------


def _make_dive(fs, P_surface, P_bottom, speed, pause_before=0, pause_after=0):
    """Build a synthetic dive: optional pause, ramp down, optional pause.

    Returns (P, W) arrays.
    """
    n_pause_before = int(pause_before * fs)
    n_pause_after = int(pause_after * fs)
    duration = (P_bottom - P_surface) / speed
    n_dive = int(duration * fs)

    segments_P = []
    segments_W = []

    if n_pause_before > 0:
        segments_P.append(P_surface * np.ones(n_pause_before))
        segments_W.append(np.zeros(n_pause_before))

    segments_P.append(np.linspace(P_surface, P_bottom, n_dive))
    segments_W.append(speed * np.ones(n_dive))

    if n_pause_after > 0:
        segments_P.append(P_bottom * np.ones(n_pause_after))
        segments_W.append(np.zeros(n_pause_after))

    return np.concatenate(segments_P), np.concatenate(segments_W)


class TestGetProfilesSingleProfile:
    """Detection of a single profiling segment."""

    def test_single_downward_profile(self):
        fs = 64.0
        P, W = _make_dive(
            fs, P_surface=0.0, P_bottom=50.0, speed=0.7, pause_before=5, pause_after=5
        )

        profiles = get_profiles(P, W, fs, P_min=0.5, W_min=0.3, min_duration=5)
        assert len(profiles) == 1
        s, e = profiles[0]
        assert s < e
        # Start should be after surface pause; end before bottom pause
        assert P[s] > 0.5
        assert W[s] >= 0.3

    def test_single_upward_profile(self):
        fs = 64.0
        # Build a "rise" by reversing the pressure ramp
        P, W = _make_dive(
            fs, P_surface=0.0, P_bottom=50.0, speed=0.7, pause_before=5, pause_after=5
        )
        P = P[::-1]  # deep → shallow
        W = -W[::-1]  # negative = upward

        profiles = get_profiles(P, W, fs, P_min=0.5, W_min=0.3, direction="up", min_duration=5)
        assert len(profiles) == 1
        s, _e = profiles[0]
        # During the upward segment W is negative, but function internally flips
        assert P[s] > 0.5


class TestGetProfilesMultiple:
    """Detection of multiple profiles separated by pauses."""

    def test_two_dives(self):
        fs = 64.0
        P1, W1 = _make_dive(fs, 0.0, 40.0, 0.8, pause_before=3, pause_after=5)
        P2, W2 = _make_dive(fs, 0.0, 60.0, 0.6, pause_before=5, pause_after=3)
        P = np.concatenate([P1, P2])
        W = np.concatenate([W1, W2])

        profiles = get_profiles(P, W, fs, P_min=0.5, W_min=0.3, min_duration=5)
        assert len(profiles) == 2
        # Profiles should not overlap
        assert profiles[0][1] < profiles[1][0]


class TestGetProfilesNoDetection:
    """Scenarios where no profiles should be detected."""

    def test_pressure_below_threshold(self):
        fs = 64.0
        P, W = _make_dive(fs, 0.0, 0.4, 0.7, pause_before=2, pause_after=2)
        # All pressures below P_min=0.5
        profiles = get_profiles(P, W, fs, P_min=0.5, W_min=0.3, min_duration=5)
        assert profiles == []

    def test_speed_below_threshold(self):
        fs = 64.0
        P, W = _make_dive(fs, 0.0, 50.0, 0.1, pause_before=2, pause_after=2)
        # Speed 0.1 < W_min=0.3
        profiles = get_profiles(P, W, fs, P_min=0.5, W_min=0.3, min_duration=5)
        assert profiles == []

    def test_empty_arrays(self):
        profiles = get_profiles([], [], fs=64.0)
        assert profiles == []


class TestGetProfilesFiltering:
    """Threshold and duration filtering."""

    def test_min_duration_excludes_short_segment(self):
        fs = 64.0
        # A 3-second dive is too short for min_duration=5
        P_short, W_short = _make_dive(fs, 0.0, 3.0, 1.0)
        # A 20-second dive should be kept
        P_long, W_long = _make_dive(fs, 0.0, 20.0, 1.0)

        # Separate them with a pause
        gap_P = np.zeros(int(3 * fs))
        gap_W = np.zeros(int(3 * fs))

        P = np.concatenate([P_short, gap_P, P_long])
        W = np.concatenate([W_short, gap_W, W_long])

        profiles = get_profiles(P, W, fs, P_min=0.5, W_min=0.3, min_duration=5)
        assert len(profiles) == 1
        s, e = profiles[0]
        # The surviving profile should be the long one
        assert (e - s) / fs >= 5

    def test_p_min_threshold(self):
        """Raising P_min should shrink or eliminate detected profiles."""
        fs = 64.0
        P, W = _make_dive(fs, 0.0, 10.0, 0.7, pause_before=3, pause_after=3)

        prof_low = get_profiles(P, W, fs, P_min=1.0, W_min=0.3, min_duration=3)
        prof_high = get_profiles(P, W, fs, P_min=8.0, W_min=0.3, min_duration=3)

        # Higher P_min admits fewer samples → shorter or no profile
        if prof_high:
            span_low = prof_low[0][1] - prof_low[0][0]
            span_high = prof_high[0][1] - prof_high[0][0]
            assert span_high < span_low
        else:
            # High threshold eliminated the profile entirely
            assert prof_high == []

    def test_w_min_threshold(self):
        """Raising W_min should shrink or eliminate detected profiles."""
        fs = 64.0
        P, W = _make_dive(fs, 0.0, 50.0, 0.5, pause_before=3, pause_after=3)

        prof_easy = get_profiles(P, W, fs, P_min=0.5, W_min=0.2, min_duration=5)
        prof_hard = get_profiles(P, W, fs, P_min=0.5, W_min=0.6, min_duration=5)

        # W=0.5 < W_min=0.6, so the strict threshold should reject
        assert len(prof_easy) >= 1
        assert prof_hard == []


class TestGetProfilesReturnTypes:
    """Verify return shapes and types."""

    def test_returns_list_of_int_tuples(self):
        fs = 64.0
        P, W = _make_dive(fs, 0.0, 50.0, 0.7, pause_before=3, pause_after=3)
        profiles = get_profiles(P, W, fs, min_duration=5)
        assert isinstance(profiles, list)
        for s, e in profiles:
            assert isinstance(s, int)
            assert isinstance(e, int)

    def test_indices_within_bounds(self):
        fs = 64.0
        P, W = _make_dive(fs, 0.0, 50.0, 0.7, pause_before=3, pause_after=3)
        profiles = get_profiles(P, W, fs, min_duration=5)
        for s, e in profiles:
            assert 0 <= s < len(P)
            assert 0 <= e < len(P)


# ---------------------------------------------------------------------------
# Glide and horizontal directions
# ---------------------------------------------------------------------------


class TestGlideDirection:
    """direction='glide' detects both up and down segments."""

    def test_glide_finds_both_up_and_down(self):
        """Synthetic yo-yo data: down then up → both segments found."""
        fs = 64.0
        # Down segment: 0 → 50 dbar
        P_down, W_down = _make_dive(fs, 0.0, 50.0, 0.7, pause_before=3, pause_after=3)
        # Up segment: 50 → 0 dbar (reversed)
        P_up = P_down[::-1]
        W_up = -W_down[::-1]
        # Concatenate: down then up with a gap
        gap = int(3 * fs)
        P = np.concatenate([P_down, np.full(gap, 50.0), P_up])
        W = np.concatenate([W_down, np.zeros(gap), W_up])

        profiles = get_profiles(P, W, fs, P_min=0.5, W_min=0.3, direction="glide", min_duration=5)
        assert len(profiles) == 2
        # First segment should be down (earlier indices), second up
        assert profiles[0][0] < profiles[1][0]

    def test_glide_sorted_by_start_index(self):
        """Glide results should be sorted by start index."""
        fs = 64.0
        P_down, W_down = _make_dive(fs, 0.0, 30.0, 0.8, pause_before=2, pause_after=2)
        P_up = P_down[::-1]
        W_up = -W_down[::-1]
        P = np.concatenate([P_down, P_up])
        W = np.concatenate([W_down, W_up])

        profiles = get_profiles(P, W, fs, P_min=0.5, W_min=0.3, direction="glide", min_duration=3)
        starts = [s for s, _ in profiles]
        assert starts == sorted(starts)


class TestHorizontalDirection:
    """direction='horizontal' detects segments regardless of W sign."""

    def test_horizontal_finds_segments(self):
        """Horizontal mode uses abs(W) ≥ W_min."""
        fs = 64.0
        # Simulate instrument moving through water: positive W (moving forward)
        P, W = _make_dive(fs, 0.0, 50.0, 0.7, pause_before=3, pause_after=3)
        profiles = get_profiles(
            P, W, fs, P_min=0.5, W_min=0.3, direction="horizontal", min_duration=5
        )
        assert len(profiles) >= 1

    def test_horizontal_finds_negative_w(self):
        """Horizontal should also detect segments with negative W."""
        fs = 64.0
        P, W = _make_dive(fs, 0.0, 50.0, 0.7, pause_before=3, pause_after=3)
        # Flip W sign — horizontal should still detect
        profiles = get_profiles(
            P, -W, fs, P_min=0.5, W_min=0.3, direction="horizontal", min_duration=5
        )
        assert len(profiles) >= 1
