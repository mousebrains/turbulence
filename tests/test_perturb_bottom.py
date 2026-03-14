# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb.bottom — bottom crash detection."""

import numpy as np

from odas_tpw.perturb.bottom import detect_bottom_crash


class TestDetectBottomCrash:
    def test_crash_detected(self):
        """Synthetic profile with acceleration spike near bottom."""
        n = 5000
        depth = np.linspace(0, 100, n)
        Ax = np.random.randn(n) * 0.01
        Ay = np.random.randn(n) * 0.01
        # Add a big spike near the bottom
        crash_idx = int(0.95 * n)
        Ax[crash_idx - 50:crash_idx + 50] = 10.0
        Ay[crash_idx - 50:crash_idx + 50] = 10.0

        bottom = detect_bottom_crash(depth, Ax, Ay, fs=512.0, vibration_factor=3.0)
        assert bottom is not None
        assert bottom > 80.0

    def test_no_crash(self):
        """Smooth profile — no crash."""
        n = 5000
        depth = np.linspace(0, 100, n)
        Ax = np.random.randn(n) * 0.01
        Ay = np.random.randn(n) * 0.01

        bottom = detect_bottom_crash(depth, Ax, Ay, fs=512.0, vibration_factor=100.0)
        assert bottom is None

    def test_shallow_profile_returns_none(self):
        """Profile too shallow — below depth_minimum."""
        n = 1000
        depth = np.linspace(0, 5, n)
        Ax = np.random.randn(n) * 10.0
        Ay = np.random.randn(n) * 10.0

        bottom = detect_bottom_crash(depth, Ax, Ay, fs=512.0, depth_minimum=10.0)
        assert bottom is None
