"""Unit tests for ``odas_tpw.perturb.speed.compute_speed_for_pfile``."""

from __future__ import annotations

import numpy as np
import pytest

from odas_tpw.perturb.speed import compute_speed_for_pfile


class _StubPF:
    """Minimal duck-type of a PFile for the speed module."""

    def __init__(self, *, P, t_slow, t_fast, fs_slow, fs_fast,
                 U_EM=None, Incl_X=None, Incl_Y=None):
        self.channels = {"P": np.asarray(P, dtype=np.float64)}
        if U_EM is not None:
            self.channels["U_EM"] = np.asarray(U_EM, dtype=np.float64)
        if Incl_X is not None:
            self.channels["Incl_X"] = np.asarray(Incl_X, dtype=np.float64)
        if Incl_Y is not None:
            self.channels["Incl_Y"] = np.asarray(Incl_Y, dtype=np.float64)
        self.t_slow = np.asarray(t_slow, dtype=np.float64)
        self.t_fast = np.asarray(t_fast, dtype=np.float64)
        self.fs_slow = float(fs_slow)
        self.fs_fast = float(fs_fast)


@pytest.fixture
def vmp_descent():
    """Linear descent: 0.5 m/s for 10 s, slow 64 Hz, fast 512 Hz."""
    fs_slow, fs_fast = 64.0, 512.0
    n_slow = int(10 * fs_slow)
    n_fast = int(10 * fs_fast)
    t_slow = np.arange(n_slow) / fs_slow
    t_fast = np.arange(n_fast) / fs_fast
    P = 0.5 * t_slow                      # dbar, 0.5 m/s
    return _StubPF(P=P, t_slow=t_slow, t_fast=t_fast,
                   fs_slow=fs_slow, fs_fast=fs_fast)


@pytest.fixture
def glider_with_em(vmp_descent):
    """Same descent + a known U_EM channel."""
    pf = vmp_descent
    pf.channels["U_EM"] = np.full_like(pf.channels["P"], 0.32)
    return pf


@pytest.fixture
def glider_with_incl(vmp_descent):
    """Same descent + inclinometer (pitch=-30°, roll=2°)."""
    pf = vmp_descent
    pf.channels["Incl_X"] = np.full_like(pf.channels["P"], -30.0)
    pf.channels["Incl_Y"] = np.full_like(pf.channels["P"], 2.0)
    return pf


# ---------------------------------------------------------------------------
# Method dispatch
# ---------------------------------------------------------------------------


class TestPressureMethod:
    def test_pressure_default_returns_vertical_speed(self, vmp_descent):
        """Default ``pressure`` method recovers the imposed 0.5 m/s descent."""
        speed_fast, W_slow = compute_speed_for_pfile(vmp_descent, {}, vehicle="vmp")
        # Trim filter transients at the array ends.
        assert speed_fast.shape == vmp_descent.t_fast.shape
        np.testing.assert_allclose(np.median(speed_fast[1000:-1000]), 0.5, atol=0.02)
        np.testing.assert_allclose(np.median(np.abs(W_slow[100:-100])), 0.5, atol=0.02)

    def test_speed_cutout_floor(self):
        """Stationary pressure → speed clamped to ``speed_cutout``."""
        n_slow, n_fast = 640, 5120
        pf = _StubPF(
            P=np.zeros(n_slow), t_slow=np.arange(n_slow) / 64.0,
            t_fast=np.arange(n_fast) / 512.0, fs_slow=64.0, fs_fast=512.0,
        )
        speed_fast, _ = compute_speed_for_pfile(
            pf, {"method": "pressure", "speed_cutout": 0.07}, vehicle="vmp",
        )
        np.testing.assert_allclose(speed_fast, 0.07, atol=1e-12)


class TestEMMethod:
    def test_em_returns_u_em(self, glider_with_em):
        speed_fast, W_slow = compute_speed_for_pfile(
            glider_with_em, {"method": "em"}, vehicle="slocum_glider",
        )
        # Constant U_EM=0.32 → fast-rate output should sit at 0.32 (after
        # smoothing settle).
        np.testing.assert_allclose(np.median(speed_fast[1000:-1000]), 0.32, atol=1e-3)
        # W_slow is independent of method — still the |dP/dt| of the descent.
        np.testing.assert_allclose(np.median(np.abs(W_slow[100:-100])), 0.5, atol=0.02)

    def test_em_takes_abs(self, vmp_descent):
        """Negative U_EM (stall noise) is mapped to |U_EM|."""
        pf = vmp_descent
        pf.channels["U_EM"] = np.full_like(pf.channels["P"], -0.4)
        speed_fast, _ = compute_speed_for_pfile(
            pf, {"method": "em"}, vehicle="slocum_glider",
        )
        np.testing.assert_allclose(np.median(speed_fast[1000:-1000]), 0.4, atol=1e-3)

    def test_em_missing_channel_raises(self, vmp_descent):
        with pytest.raises(ValueError, match="U_EM is missing"):
            compute_speed_for_pfile(
                vmp_descent, {"method": "em"}, vehicle="slocum_glider",
            )


class TestFlightMethod:
    def test_flight_recovers_along_axis_speed(self, glider_with_incl):
        """For pitch=-30°, roll=2°, AoA=3°: U = |W| / (sin(27°)cos(2°))."""
        speed_fast, _ = compute_speed_for_pfile(
            glider_with_incl,
            {"method": "flight", "aoa_deg": 3.0},
            vehicle="slocum_glider",
        )
        expected = 0.5 / (np.sin(np.deg2rad(27.0)) * np.cos(np.deg2rad(2.0)))
        np.testing.assert_allclose(
            np.median(speed_fast[1000:-1000]), expected, atol=0.02,
        )

    def test_flight_picks_pitch_axis_by_amplitude(self, vmp_descent):
        """Pitch on Incl_Y (larger amplitude) → recovered the same."""
        pf = vmp_descent
        n = pf.channels["P"].size
        # Roll: tiny ±0.5° wobble. Pitch: −30° ± 1° gentle drift.
        pf.channels["Incl_X"] = 0.5 * np.sin(2 * np.pi * np.arange(n) / n)
        pf.channels["Incl_Y"] = -30.0 + np.sin(2 * np.pi * np.arange(n) / n)
        speed_fast, _ = compute_speed_for_pfile(
            pf, {"method": "flight", "aoa_deg": 3.0}, vehicle="slocum_glider",
        )
        # Pitch axis = Incl_Y (range ~2°), roll ≈ Incl_X (range ~1°).
        # Effective sin path ≈ sin(27°)·cos(0°) on average.
        expected = 0.5 / np.sin(np.deg2rad(27.0))
        np.testing.assert_allclose(
            np.median(speed_fast[1000:-1000]), expected, rtol=0.05,
        )

    def test_flight_default_aoa_is_3deg(self, glider_with_incl):
        """Omitting ``aoa_deg`` uses ODAS default 3°."""
        a, _ = compute_speed_for_pfile(
            glider_with_incl, {"method": "flight"}, vehicle="slocum_glider",
        )
        b, _ = compute_speed_for_pfile(
            glider_with_incl, {"method": "flight", "aoa_deg": 3.0},
            vehicle="slocum_glider",
        )
        np.testing.assert_allclose(a, b)

    def test_flight_missing_inclinometer_raises(self, vmp_descent):
        with pytest.raises(ValueError, match="Incl_X and Incl_Y"):
            compute_speed_for_pfile(
                vmp_descent, {"method": "flight"}, vehicle="slocum_glider",
            )


class TestConstantMethod:
    def test_constant_uses_value(self, vmp_descent):
        speed_fast, _ = compute_speed_for_pfile(
            vmp_descent, {"method": "constant", "value": 0.42}, vehicle="vmp",
        )
        np.testing.assert_allclose(speed_fast, 0.42)

    def test_constant_value_required(self, vmp_descent):
        with pytest.raises(ValueError, match="speed.value is null"):
            compute_speed_for_pfile(
                vmp_descent, {"method": "constant", "value": None}, vehicle="vmp",
            )


class TestUnknownMethod:
    def test_raises(self, vmp_descent):
        with pytest.raises(ValueError, match="Unknown speed.method"):
            compute_speed_for_pfile(
                vmp_descent, {"method": "magic"}, vehicle="vmp",
            )
