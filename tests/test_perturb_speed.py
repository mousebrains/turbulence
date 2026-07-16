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

    def test_glider_pressure_method_warns(self, vmp_descent):
        """Audit r2-3: the pressure method on a glide vehicle warns that
        |dP/dt| is the vertical (not through-water) speed and epsilon is
        biased — matching the rsi path, which perturb otherwise bypasses."""
        with pytest.warns(UserWarning, match="strongly biased"):
            compute_speed_for_pfile(
                vmp_descent, {"method": "pressure"}, vehicle="slocum_glider",
            )

    def test_vmp_pressure_method_does_not_warn(self, vmp_descent):
        """A VMP's vertical speed IS its through-water speed — no warning."""
        import warnings as _w

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            compute_speed_for_pfile(
                vmp_descent, {"method": "pressure"}, vehicle="vmp",
            )
        assert not any("strongly biased" in str(w.message) for w in caught)


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
        """For pitch=-30°, AoA=3°: U = |W| / sin(33°) (glide path steeper
        than pitch by the attack angle; ODAS odas_p2mat.m convention)."""
        speed_fast, _ = compute_speed_for_pfile(
            glider_with_incl,
            {"method": "flight", "aoa_deg": 3.0},
            vehicle="slocum_glider",
        )
        expected = 0.5 / np.sin(np.deg2rad(33.0))
        np.testing.assert_allclose(
            np.median(speed_fast[1000:-1000]), expected, atol=0.02,
        )

    def test_flight_aoa_steepens_glide_path(self, glider_with_incl):
        """Sign regression for issue #131 M6: a larger angle of attack
        makes the glide path STEEPER (|pitch| + aoa), so the recovered
        along-path speed must DECREASE with aoa. The pre-fix code
        subtracted aoa, which reversed this ordering (and biased epsilon
        ~2.4x low at Slocum pitch through the ~U^-4 leverage)."""
        u0, _ = compute_speed_for_pfile(
            glider_with_incl, {"method": "flight", "aoa_deg": 0.0},
            vehicle="slocum_glider",
        )
        u6, _ = compute_speed_for_pfile(
            glider_with_incl, {"method": "flight", "aoa_deg": 6.0},
            vehicle="slocum_glider",
        )
        m0 = float(np.median(u0[1000:-1000]))
        m6 = float(np.median(u6[1000:-1000]))
        assert m6 < m0, f"aoa=6 speed {m6} should be below aoa=0 speed {m0}"
        # And quantitatively: sin(30°)/sin(36°) ratio.
        np.testing.assert_allclose(
            m6 / m0, np.sin(np.deg2rad(30.0)) / np.sin(np.deg2rad(36.0)),
            rtol=0.02,
        )

    def test_flight_uem_crosscheck_warns_on_disagreement(self, glider_with_incl):
        """A U_EM channel disagreeing with the flight speed by >20% warns."""
        pf = glider_with_incl
        pf.channels["U_EM"] = np.full_like(pf.channels["P"], 0.4)  # flight ~0.92
        with pytest.warns(UserWarning, match="disagrees with U_EM"):
            compute_speed_for_pfile(
                pf, {"method": "flight", "aoa_deg": 3.0},
                vehicle="slocum_glider",
            )

    def test_flight_uem_crosscheck_silent_when_consistent(self, glider_with_incl):
        import warnings as _w

        pf = glider_with_incl
        consistent = 0.5 / np.sin(np.deg2rad(33.0))
        pf.channels["U_EM"] = np.full_like(pf.channels["P"], consistent)
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            compute_speed_for_pfile(
                pf, {"method": "flight", "aoa_deg": 3.0},
                vehicle="slocum_glider",
            )
        assert not any("disagrees with U_EM" in str(w.message) for w in caught)

    def test_flight_picks_pitch_axis_by_amplitude(self, vmp_descent):
        """Pitch on Incl_Y (larger amplitude) → recovered the same."""
        pf = vmp_descent
        n = pf.channels["P"].size
        # Roll: tiny +/-0.5 deg wobble. Pitch: -30 deg +/- 1 deg gentle drift.
        pf.channels["Incl_X"] = 0.5 * np.sin(2 * np.pi * np.arange(n) / n)
        pf.channels["Incl_Y"] = -30.0 + np.sin(2 * np.pi * np.arange(n) / n)
        speed_fast, _ = compute_speed_for_pfile(
            pf, {"method": "flight", "aoa_deg": 3.0}, vehicle="slocum_glider",
        )
        # Pitch axis = Incl_Y (range ~2°), roll ≈ Incl_X (range ~1°).
        # Effective sin path ≈ sin(30° + 3°) on average.
        expected = 0.5 / np.sin(np.deg2rad(33.0))
        np.testing.assert_allclose(
            np.median(speed_fast[1000:-1000]), expected, rtol=0.05,
        )

    def test_flight_outlier_does_not_flip_pitch_axis(self, vmp_descent):
        """Single saturation spike on the roll axis must not flip pitch.

        Real-data shape from RIOT sl684: ``Incl_X`` (true roll) had a brief
        -90° saturation that gave it a larger min-max spread than the
        steady ±25° pitch swing on ``Incl_Y``. With the percentile-spread
        heuristic (default 1..99), the spike falls outside the window and
        ``Incl_Y`` is correctly chosen as pitch.
        """
        pf = vmp_descent
        n = pf.channels["P"].size
        # Pitch on Incl_Y: realistic ±25° glide.
        pf.channels["Incl_Y"] = 25.0 * np.sin(2 * np.pi * np.arange(n) / 1024)
        # Roll on Incl_X: ±5° flight + one -90° saturation spike.
        roll = 5.0 * np.sin(2 * np.pi * np.arange(n) / 600)
        roll[10] = -90.0
        pf.channels["Incl_X"] = roll
        speed_fast, _ = compute_speed_for_pfile(
            pf, {"method": "flight", "aoa_deg": 3.0}, vehicle="slocum_glider",
        )
        # With the right axis (|pitch| median ~18°+aoa, min-pitch gate on)
        # the median speed is ~1.4 m/s; a fooled pick (pitch read off the
        # ±5° roll axis) yields ~3.6 m/s. Band chosen to separate the two
        # (measured 1.41 vs 3.58 under the current formula).
        median_speed = float(np.median(speed_fast[1000:-1000]))
        assert 0.8 < median_speed < 3.0, (
            f"axis pick likely fooled by outlier: median speed {median_speed}"
        )

    def test_flight_amplitude_quantile_yaml_override(self, vmp_descent):
        """``amplitude_quantile`` from the speed cfg is threaded into the picker.

        Same outlier scenario as above. With the default (1, 99), pitch is
        correctly Incl_Y. With (0, 100) (i.e. nanmin/nanmax), the -90°
        spike on Incl_X dominates and pitch flips to the wrong axis,
        producing a markedly different speed estimate. This guards
        against a future regression that forgets to thread the YAML
        option through.
        """
        pf = vmp_descent
        n = pf.channels["P"].size
        pf.channels["Incl_Y"] = 25.0 * np.sin(2 * np.pi * np.arange(n) / 1024)
        roll = 5.0 * np.sin(2 * np.pi * np.arange(n) / 600)
        roll[10] = -90.0
        pf.channels["Incl_X"] = roll

        good, _ = compute_speed_for_pfile(
            pf,
            {"method": "flight", "aoa_deg": 3.0,
             "amplitude_quantile": [1.0, 99.0]},
            vehicle="slocum_glider",
        )
        bad, _ = compute_speed_for_pfile(
            pf,
            {"method": "flight", "aoa_deg": 3.0,
             "amplitude_quantile": [0.0, 100.0]},
            vehicle="slocum_glider",
        )
        # The two should disagree substantially: same data, different
        # picker. If the option is silently ignored, the medians coincide.
        assert not np.allclose(
            np.median(good[1000:-1000]),
            np.median(bad[1000:-1000]),
            rtol=0.1,
        ), "amplitude_quantile cfg appears to be ignored"

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
        with pytest.raises(ValueError, match=r"speed\.value is null"):
            compute_speed_for_pfile(
                vmp_descent, {"method": "constant", "value": None}, vehicle="vmp",
            )


class TestUnknownMethod:
    def test_raises(self, vmp_descent):
        with pytest.raises(ValueError, match=r"Unknown speed\.method"):
            compute_speed_for_pfile(
                vmp_descent, {"method": "magic"}, vehicle="vmp",
            )


def test_speed_model_is_shared_not_duplicated():
    """perturb.speed re-exports rsi.speed — one implementation, no divergent copy.

    The rsi pipeline (adapter.pfile_to_l1data) reuses the same function, so this
    guards against someone re-adding a separate copy under perturb later.
    """
    from odas_tpw.perturb.speed import compute_speed_for_pfile as perturb_fn
    from odas_tpw.rsi.speed import compute_speed_for_pfile as rsi_fn

    assert perturb_fn is rsi_fn


class TestFlightParameterValidation:
    """#132 review [P3]: invalid aoa must error, not silently produce the
    speed cutout (pitch=-30, aoa=-40 previously returned 0.05 m/s)."""

    @pytest.mark.parametrize("bad_aoa", [-40.0, -0.1, float("nan"), float("inf")])
    def test_invalid_aoa_raises(self, glider_with_incl, bad_aoa):
        with pytest.raises(ValueError, match="aoa_deg"):
            compute_speed_for_pfile(
                glider_with_incl,
                {"method": "flight", "aoa_deg": bad_aoa},
                vehicle="slocum_glider",
            )

    @pytest.mark.parametrize("bad_gate", [-1.0, float("nan")])
    def test_invalid_min_pitch_raises(self, glider_with_incl, bad_gate):
        with pytest.raises(ValueError, match="min_pitch_deg"):
            compute_speed_for_pfile(
                glider_with_incl,
                {"method": "flight", "aoa_deg": 3.0, "min_pitch_deg": bad_gate},
                vehicle="slocum_glider",
            )

    def test_zero_aoa_is_valid(self, glider_with_incl):
        speed, _ = compute_speed_for_pfile(
            glider_with_incl,
            {"method": "flight", "aoa_deg": 0.0},
            vehicle="slocum_glider",
        )
        np.testing.assert_allclose(
            np.median(speed[1000:-1000]), 0.5 / np.sin(np.deg2rad(30.0)), atol=0.02
        )
