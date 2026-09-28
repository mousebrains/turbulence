# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for processing.bottom — bottom crash detection."""

import warnings
from typing import ClassVar

import numpy as np
import pytest

from odas_tpw.processing.bottom import (
    bottom_anchor_depth,
    detect_bottom_crash,
    detect_bottom_fallrate,
)


class TestDetectBottomCrash:
    def test_crash_detected(self):
        """Synthetic profile with acceleration spike near bottom."""
        n = 5000
        depth = np.linspace(0, 100, n)
        Ax = np.random.randn(n) * 0.01
        Ay = np.random.randn(n) * 0.01
        # Add a big spike near the bottom
        crash_idx = int(0.95 * n)
        Ax[crash_idx - 50 : crash_idx + 50] = 10.0
        Ay[crash_idx - 50 : crash_idx + 50] = 10.0

        bottom = detect_bottom_crash(
            depth, {"Ax": Ax, "Ay": Ay}, fs=512.0, vibration_factor=3.0
        )
        assert bottom is not None
        assert bottom > 80.0

    def test_reports_sample_mean_within_flagged_bin(self):
        """The crash depth is the MEAN of the spike bin's real samples: close to
        the bin center for an interior bin (so it is not under-read like the
        shallow edge, #22), but — unlike the geometric center — guaranteed never
        to fall below the deepest sample, so the caller can always trim it
        (audit 2026-06-25 M4)."""
        np.random.seed(42)
        n = 5000
        depth = np.linspace(10.0, 50.0, n)
        accel = np.random.randn(n) * 0.01
        # A noisy spike confined to the depth bin [38, 42) (center 40). With the
        # defaults depth_minimum=10, depth_window=4 the edges are [10,14,...,50].
        spike = (depth >= 38.0) & (depth < 42.0)
        accel[spike] = np.random.randn(int(spike.sum())) * 5.0
        bottom = detect_bottom_crash(
            depth, {"vibration_rms": accel}, fs=512.0, vibration_factor=4.0
        )
        assert bottom is not None
        assert 38.0 <= bottom < 42.0  # within the flagged bin's samples
        assert abs(bottom - 40.0) < 0.05  # ~the center for a full interior bin
        assert bottom <= float(np.nanmax(depth))  # never below the deepest sample

    def test_no_crash(self):
        """Smooth profile — no crash."""
        n = 5000
        depth = np.linspace(0, 100, n)
        Ax = np.random.randn(n) * 0.01
        Ay = np.random.randn(n) * 0.01

        bottom = detect_bottom_crash(
            depth, {"Ax": Ax, "Ay": Ay}, fs=512.0, vibration_factor=100.0
        )
        assert bottom is None

    def test_midcolumn_spike_is_not_a_crash(self):
        """Audit r1-3: a vibration spike far above the deepest sample is
        mid-column contamination, not a bottom crash — the deep cast is kept
        and a warning is emitted (no silent truncation)."""
        import warnings as _w

        n = 8000
        depth = np.linspace(0.0, 230.0, n)  # cast keeps descending to 230 m
        accel = np.random.default_rng(3).standard_normal(n) * 0.01
        spike = (depth >= 98.0) & (depth < 104.0)  # transient ~100 m up
        accel[spike] = np.random.default_rng(4).standard_normal(int(spike.sum())) * 8.0
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            bottom = detect_bottom_crash(
                depth, {"vibration_rms": accel}, fs=512.0, vibration_factor=4.0
            )
        assert bottom is None, f"mid-column spike truncated the cast at {bottom} m"
        assert any("mid-column" in str(w.message) for w in caught)

    def test_near_bottom_spike_still_detected(self):
        """A spike within proximity_bins of the deepest sample is a real crash."""
        n = 8000
        depth = np.linspace(0.0, 230.0, n)
        accel = np.random.default_rng(5).standard_normal(n) * 0.01
        spike = depth >= 228.0  # within ~2 m of the 230 m bottom
        accel[spike] = np.random.default_rng(6).standard_normal(int(spike.sum())) * 8.0
        bottom = detect_bottom_crash(
            depth, {"vibration_rms": accel}, fs=512.0, vibration_factor=4.0
        )
        assert bottom is not None and bottom > 220.0

    def test_proximity_bins_zero_requires_deepest_bin(self):
        """proximity_bins=0 accepts only the single deepest sampled bin."""
        n = 8000
        depth = np.linspace(0.0, 100.0, n)
        accel = np.random.default_rng(7).standard_normal(n) * 0.01
        # Spike two bins above the bottom (bins are 4 m wide from depth_minimum).
        spike = (depth >= 90.0) & (depth < 94.0)
        accel[spike] = np.random.default_rng(8).standard_normal(int(spike.sum())) * 8.0
        assert detect_bottom_crash(
            depth, {"vibration_rms": accel}, fs=512.0, vibration_factor=4.0,
            proximity_bins=0,
        ) is None

    def test_shallow_profile_returns_none(self):
        """Profile too shallow — below depth_minimum."""
        n = 1000
        depth = np.linspace(0, 5, n)
        Ax = np.random.randn(n) * 10.0
        Ay = np.random.randn(n) * 10.0

        bottom = detect_bottom_crash(
            depth, {"Ax": Ax, "Ay": Ay}, fs=512.0, depth_minimum=10.0
        )
        assert bottom is None

    def test_empty_channel_dict_returns_none(self):
        depth = np.linspace(0, 100, 1000)
        assert detect_bottom_crash(depth, {}, fs=512.0) is None

    def test_all_nan_depth_returns_none(self):
        """An all-NaN depth segment must return None, not crash on
        np.arange(..., NaN, ...) (audit round-2)."""
        n = 1000
        depth = np.full(n, np.nan)
        Ax = np.random.randn(n) * 10.0
        Ay = np.random.randn(n) * 10.0
        assert detect_bottom_crash(depth, {"Ax": Ax, "Ay": Ay}, fs=512.0) is None

    def test_all_nan_depth_emits_no_runtime_warning(self):
        """The all-NaN-depth path must be silent. np.errstate does NOT suppress
        nanmax's 'All-NaN slice encountered' RuntimeWarning, so the prior
        errstate-only guard let it escape as log noise (and callers that promote
        RuntimeWarning to errors would crash). Short-circuiting before nanmax
        keeps the path quiet (audit 2026-06-26)."""
        import warnings

        n = 1000
        depth = np.full(n, np.nan)
        Ax = np.random.randn(n) * 10.0
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            # Fails on the old code: nanmax raises "All-NaN slice encountered".
            assert detect_bottom_crash(depth, {"Ax": Ax}, fs=512.0) is None

    def test_single_channel_pre_aggregated(self):
        """One pre-computed magnitude channel works just like Ax/Ay."""
        n = 5000
        depth = np.linspace(0, 100, n)
        Ax = np.random.randn(n) * 0.01
        Ay = np.random.randn(n) * 0.01
        crash_idx = int(0.95 * n)
        Ax[crash_idx - 50 : crash_idx + 50] = 10.0
        Ay[crash_idx - 50 : crash_idx + 50] = 10.0
        rms = np.sqrt(Ax**2 + Ay**2)

        # sqrt(rms**2) == rms, so passing it as a single channel is identical
        # to passing the components.
        b1 = detect_bottom_crash(
            depth, {"vibration": rms}, fs=512.0, vibration_factor=3.0
        )
        b2 = detect_bottom_crash(
            depth, {"Ax": Ax, "Ay": Ay}, fs=512.0, vibration_factor=3.0
        )
        assert b1 == b2

    def test_three_axis_accelerometer(self):
        """Az contributes when present (3-axis IMU case)."""
        n = 5000
        depth = np.linspace(0, 100, n)
        Ax = np.random.randn(n) * 0.01
        Ay = np.random.randn(n) * 0.01
        Az = np.random.randn(n) * 0.01
        # Spike Az at the bottom but leave Ax/Ay quiet — only the 3-axis
        # call should detect it; the 2-axis call should not.
        crash_idx = int(0.95 * n)
        Az[crash_idx - 50 : crash_idx + 50] = 10.0

        b_xy = detect_bottom_crash(
            depth, {"Ax": Ax, "Ay": Ay}, fs=512.0, vibration_factor=3.0
        )
        b_xyz = detect_bottom_crash(
            depth, {"Ax": Ax, "Ay": Ay, "Az": Az}, fs=512.0, vibration_factor=3.0
        )
        assert b_xy is None
        assert b_xyz is not None
        assert b_xyz > 80.0

    def test_mismatched_channel_length_returns_none(self):
        depth = np.linspace(0, 100, 1000)
        Ax = np.zeros(500)
        assert detect_bottom_crash(depth, {"Ax": Ax}, fs=512.0) is None

    def test_too_few_bins_returns_none(self):
        """When max_depth - depth_minimum < bin_size, len(bins) < 2."""
        depth = np.linspace(0, 12, 200)  # max_depth=12 is just barely above min
        Ax = np.random.randn(200) * 0.01
        # depth_window=10, depth_minimum=10 → bins = arange(10, 22, 10) = [10, 20]
        # len < 2 only when window>max-min. Use window=20 with max=12 → bins = [10] only
        result = detect_bottom_crash(
            depth, {"Ax": Ax}, fs=64.0, depth_window=20.0, depth_minimum=10.0
        )
        assert result is None

    def test_too_few_valid_bins_returns_none(self):
        """If only 1-2 bins have enough samples for a finite std, return None."""
        # Tiny profile with just 3 points, all near the surface
        depth = np.array([10.5, 11.0, 11.5, 12.0])
        Ax = np.array([0.1, 0.2, 0.3, 0.4])
        # Most bins empty → < 3 valid stds → return None (line 100)
        result = detect_bottom_crash(
            depth, {"Ax": Ax}, fs=64.0, depth_window=4.0, depth_minimum=10.0
        )
        assert result is None


class TestUnwiredKnobWarnings:
    """#104 U4-F2: speed_factor / median_factor / vibration_frequency are
    accepted for backward compatibility but not implemented; tuning one must
    warn rather than be a silent no-op."""

    depth: ClassVar = np.array([10.5, 11.0, 11.5, 12.0])
    channels: ClassVar = {"Ax": np.array([0.1, 0.2, 0.3, 0.4])}

    def test_default_values_do_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning fails
            detect_bottom_crash(self.depth, self.channels, fs=64.0)

    @pytest.mark.parametrize(
        "kwargs, needle",
        [
            ({"speed_factor": 0.5}, "speed_factor"),
            ({"median_factor": 2.0}, "median_factor"),
            ({"vibration_frequency": 8}, "vibration_frequency"),
        ],
    )
    def test_non_default_knob_warns(self, kwargs, needle):
        with pytest.warns(UserWarning, match=needle):
            detect_bottom_crash(self.depth, self.channels, fs=64.0, **kwargs)


def _descent(fs=64.0, w=1.0, z0=0.5, z_stop=20.0, tail_s=2.0, decel_m=0.08):
    """A free fall at `w` to `z_stop`, decelerating over the last `decel_m`."""
    t_fall = (z_stop - decel_m - z0) / w
    t1 = np.arange(0.0, t_fall, 1.0 / fs)
    d1 = z0 + w * t1
    # linear ramp of speed to zero over decel_m
    n2 = max(round(decel_m / w * fs), 2)
    frac = np.linspace(1.0, 0.0, n2)
    d2 = d1[-1] + np.cumsum(frac * w / fs)
    d3 = np.full(int(tail_s * fs), d2[-1])
    return np.concatenate([d1, d2, d3])


class TestDetectBottomFallrate:
    def test_finds_end_of_free_fall(self):
        fs = 64.0
        d = _descent(fs=fs, z_stop=20.0, decel_m=0.08)
        z = detect_bottom_fallrate(d, fs)
        assert z is not None
        # end of free fall is ~decel_m above the stop, within a couple samples
        assert 19.85 < z < 20.0

    def test_fires_where_vibration_has_no_spike(self):
        """No accel channel at all, yet the bottom is found — the point of it."""
        z = detect_bottom_fallrate(_descent(), 64.0)
        assert z is not None

    def test_no_descent_returns_none(self):
        assert detect_bottom_fallrate(np.full(600, 7.0), 64.0) is None

    def test_too_short_returns_none(self):
        assert detect_bottom_fallrate(np.array([1.0, 2.0, 3.0]), 64.0) is None

    def test_all_nan_returns_none(self):
        assert detect_bottom_fallrate(np.full(600, np.nan), 64.0) is None

    def test_span_below_min_returns_none(self):
        d = _descent(z_stop=4.0)
        assert detect_bottom_fallrate(d, 64.0, min_span=8.0) is None

    def test_bad_fs_returns_none(self):
        assert detect_bottom_fallrate(_descent(), 0.0) is None
        assert detect_bottom_fallrate(_descent(), np.nan) is None

    def test_fraction_must_be_open_unit_interval(self):
        for bad in (0.0, 1.0, -0.5, 1.5):
            with pytest.raises(ValueError, match="fraction"):
                detect_bottom_fallrate(_descent(), 64.0, fraction=bad)

    def test_lower_fraction_sits_deeper(self):
        """Monotonicity: a laxer threshold is crossed closer to the stop."""
        d = _descent(decel_m=0.4)
        z90 = detect_bottom_fallrate(d, 64.0, fraction=0.9)
        z50 = detect_bottom_fallrate(d, 64.0, fraction=0.5)
        assert z50 > z90

    def test_smoothing_inflates_the_answer(self):
        """The measured deceleration length grows with the smoother — the
        reason smooth_s is documented as not a free parameter."""
        d = _descent(decel_m=0.08, z_stop=20.0)
        z_sharp = detect_bottom_fallrate(d, 64.0, smooth_s=0.0)
        z_smooth = detect_bottom_fallrate(d, 64.0, smooth_s=0.5)
        assert z_smooth < z_sharp  # smoothing pushes the answer shallower

    def test_picks_longest_descent_of_several(self):
        fs = 64.0
        short = _descent(fs=fs, z_stop=12.0, tail_s=1.0)
        long_ = _descent(fs=fs, z_stop=40.0, tail_s=1.0)
        d = np.concatenate([short, np.full(int(2 * fs), 0.3), long_])
        z = detect_bottom_fallrate(d, fs)
        assert z is not None and z > 35.0


class TestBottomAnchorDepth:
    def test_subtracts_height(self):
        assert bottom_anchor_depth(20.0, 0.15) == pytest.approx(19.85)

    def test_zero_height_is_the_bottom(self):
        assert bottom_anchor_depth(20.0, 0.0) == pytest.approx(20.0)

    def test_none_passes_through(self):
        assert bottom_anchor_depth(None, 0.15) is None

    def test_nan_bottom_passes_through(self):
        assert bottom_anchor_depth(np.nan, 0.15) is None

    def test_negative_height_rejected(self):
        with pytest.raises(ValueError, match="height"):
            bottom_anchor_depth(20.0, -0.1)

    def test_nonfinite_height_rejected(self):
        with pytest.raises(ValueError, match="height"):
            bottom_anchor_depth(20.0, np.nan)
