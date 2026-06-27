# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for processing.bottom — bottom crash detection."""

import numpy as np

from odas_tpw.processing.bottom import detect_bottom_crash


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
