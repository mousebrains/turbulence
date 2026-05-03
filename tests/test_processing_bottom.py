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
