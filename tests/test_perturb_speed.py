"""Unit tests for ``odas_tpw.perturb.speed.compute_speed_for_pfile``."""

from __future__ import annotations

import warnings
from pathlib import Path

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


class _StubPFWithGrid(_StubPF):
    """_StubPF plus the ``is_fast`` grid registration a real PFile carries
    (updated by ``merge_hotel_into_pfile`` for hotel channels)."""

    def __init__(self, *args, fast_channels=(), **kwargs):
        super().__init__(*args, **kwargs)
        self._fast_channels = set(fast_channels)

    def is_fast(self, name):
        return name in self._fast_channels


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
        speed_fast, W_slow, _ = compute_speed_for_pfile(vmp_descent, {}, vehicle="vmp")
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
        speed_fast, _, _ = compute_speed_for_pfile(
            pf, {"method": "pressure", "speed_cutout": 0.07}, vehicle="vmp",
        )
        np.testing.assert_allclose(speed_fast, 0.07, atol=1e-12)

    @pytest.mark.parametrize("vehicle", ["slocum_glider", "sea_glider", "sea_explorer"])
    def test_glider_pressure_method_is_refused(self, vmp_descent, vehicle):
        """|dP/dt| is a glider's VERTICAL rate: the pressure method is an error
        there, not a warning (Pat, 2026-11-04). Without U_EM the message points
        to the flight model, and only to it."""
        from odas_tpw.rsi.speed import PressureSpeedOnGliderError

        with pytest.raises(PressureSpeedOnGliderError) as err:
            compute_speed_for_pfile(vmp_descent, {"method": "pressure"}, vehicle=vehicle)
        msg = str(err.value)
        assert vehicle in msg and "speed.method: flight" in msg
        assert "speed.method: em" not in msg

    def test_glider_default_method_is_refused(self, vmp_descent):
        """The DEFAULT method is pressure, so a glider config with no speed
        block is refused too: the RIOT 2026 shakedown case."""
        from odas_tpw.rsi.speed import PressureSpeedOnGliderError

        with pytest.raises(PressureSpeedOnGliderError):
            compute_speed_for_pfile(vmp_descent, {}, vehicle="slocum_glider")

    def test_glider_with_em_is_pointed_at_em(self, glider_with_em):
        """With a U_EM channel the message names method em and em_scale, and
        flight as the alternative."""
        from odas_tpw.rsi.speed import PressureSpeedOnGliderError

        with pytest.raises(PressureSpeedOnGliderError) as err:
            compute_speed_for_pfile(
                glider_with_em, {"method": "pressure"}, vehicle="slocum_glider",
            )
        msg = str(err.value)
        assert "speed.method: em" in msg and "speed.em_scale" in msg
        assert "speed.method: flight" in msg

    @pytest.mark.parametrize("bad", [{"tau": 0.001}, {"tau": -1}, {"speed_cutout": "x"}])
    def test_glider_refused_before_anything_else_can_raise(self, vmp_descent, bad):
        """A malformed speed setting raises an ordinary error, which perturb
        would treat as a recoverable failure of the default method and fall
        back to |dP/dt|: the glider refusal must come first (review of 1cfc488)."""
        from odas_tpw.rsi.speed import PressureSpeedOnGliderError

        with pytest.raises(PressureSpeedOnGliderError):
            compute_speed_for_pfile(
                vmp_descent, {"method": "pressure", **bad}, vehicle="slocum_glider",
            )

    @pytest.mark.parametrize("has_u_em, remedy", [(True, "speed.method: em"),
                                                  (False, "speed.method: flight")])
    def test_remedy_survives_the_run_summary(self, has_u_em, remedy):
        """The run summary clips each line at 200 characters; the remedy must
        be inside them, after a realistic file prefix."""
        from odas_tpw.perturb.pipeline import _summary_lines
        from odas_tpw.rsi.speed import pressure_speed_on_glider_message

        msg = pressure_speed_on_glider_message("slocum_glider", has_u_em)
        (line,) = _summary_lines("file error:  ", [f"RIOTSHAKE_MR435_0123.p: speed: {msg}"])
        assert remedy in line

    def test_glider_refusal_is_a_value_error(self):
        """Callers that catch ValueError for a bad speed configuration still do."""
        from odas_tpw.rsi.speed import PressureSpeedOnGliderError

        assert issubclass(PressureSpeedOnGliderError, ValueError)

    @pytest.mark.parametrize("vehicle", ["auv", "stand"])
    def test_horizontal_pressure_method_still_warns(self, vmp_descent, vehicle):
        """Only gliders are refused; another horizontal platform keeps the
        audit r2-3 warning, and still gets |dP/dt|."""
        with pytest.warns(UserWarning, match="strongly biased"):
            speed_fast, _, source = compute_speed_for_pfile(
                vmp_descent, {"method": "pressure"}, vehicle=vehicle,
            )
        assert source == "pressure" and np.isfinite(speed_fast).all()

    def test_glider_explicit_methods_unaffected(self, glider_with_em):
        """em on a glider is not touched by the refusal."""
        _, _, source = compute_speed_for_pfile(
            glider_with_em, {"method": "em"}, vehicle="slocum_glider",
        )
        assert source == "em"

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
        speed_fast, W_slow, _ = compute_speed_for_pfile(
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
        speed_fast, _, _ = compute_speed_for_pfile(
            pf, {"method": "em"}, vehicle="slocum_glider",
        )
        np.testing.assert_allclose(np.median(speed_fast[1000:-1000]), 0.4, atol=1e-3)

    def test_em_missing_channel_raises(self, vmp_descent):
        with pytest.raises(ValueError, match="U_EM is missing"):
            compute_speed_for_pfile(
                vmp_descent, {"method": "em"}, vehicle="slocum_glider",
            )

    def test_em_all_nan_raises_not_floor(self, vmp_descent):
        """An all-NaN U_EM (dead/disconnected flowmeter) must ERROR: the
        _slow_to_fast all-NaN fill would otherwise publish a constant
        0.05 m/s with provenance 'em' — missing telemetry indistinguishable
        from a real 0.05 m/s speed (PR #139 P1)."""
        pf = vmp_descent
        pf.channels["U_EM"] = np.full_like(pf.channels["P"], np.nan)
        with pytest.raises(ValueError, match=r"no finite samples.*speed_cutout floor"):
            compute_speed_for_pfile(pf, {"method": "em"}, vehicle="slocum_glider")

    def test_em_partial_nan_ok(self, glider_with_em):
        """em rejects only ZERO finite samples — gaps are bridged."""
        pf = glider_with_em
        pf.channels["U_EM"][100:200] = np.nan
        speed_fast, _, _ = compute_speed_for_pfile(
            pf, {"method": "em"}, vehicle="slocum_glider",
        )
        assert np.isfinite(speed_fast).all()
        np.testing.assert_allclose(np.median(speed_fast[1000:-1000]), 0.32, atol=1e-3)


class TestFlightMethod:
    def test_flight_recovers_along_axis_speed(self, glider_with_incl):
        """For pitch=-30°, AoA=3°: U = |W| / sin(33°) (glide path steeper
        than pitch by the attack angle; ODAS odas_p2mat.m convention)."""
        speed_fast, _, _ = compute_speed_for_pfile(
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
        u0, _, _ = compute_speed_for_pfile(
            glider_with_incl, {"method": "flight", "aoa_deg": 0.0},
            vehicle="slocum_glider",
        )
        u6, _, _ = compute_speed_for_pfile(
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
        speed_fast, _, _ = compute_speed_for_pfile(
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
        speed_fast, _, _ = compute_speed_for_pfile(
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

        good, _, _ = compute_speed_for_pfile(
            pf,
            {"method": "flight", "aoa_deg": 3.0,
             "amplitude_quantile": [1.0, 99.0]},
            vehicle="slocum_glider",
        )
        bad, _, _ = compute_speed_for_pfile(
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
        a, _, _ = compute_speed_for_pfile(
            glider_with_incl, {"method": "flight"}, vehicle="slocum_glider",
        )
        b, _, _ = compute_speed_for_pfile(
            glider_with_incl, {"method": "flight", "aoa_deg": 3.0},
            vehicle="slocum_glider",
        )
        np.testing.assert_allclose(a, b)

    def test_flight_missing_inclinometer_raises(self, vmp_descent):
        with pytest.raises(ValueError, match="Incl_X and Incl_Y"):
            compute_speed_for_pfile(
                vmp_descent, {"method": "flight"}, vehicle="slocum_glider",
            )

    def test_flight_zero_finite_raises_not_floor(self, vmp_descent):
        """Pitch below min_pitch_deg for the WHOLE record (level flight /
        all-inflection) → flight model all-NaN → ERROR, not a constant
        0.05 m/s published with provenance 'flight' (PR #139 P1)."""
        pf = vmp_descent
        # |pitch| - aoa(3°) = 0 everywhere → sin path 0 < sin(min_pitch) → NaN
        pf.channels["Incl_X"] = np.full_like(pf.channels["P"], 2.0)
        pf.channels["Incl_Y"] = np.full_like(pf.channels["P"], 1.0)
        with pytest.raises(ValueError, match=r"no finite samples.*speed_cutout floor"):
            compute_speed_for_pfile(pf, {"method": "flight"}, vehicle="slocum_glider")

    def test_flight_partial_nan_below_min_pitch_ok(self, vmp_descent):
        """The em/flight guard is ZERO-finite, not hotel's 50% rule: flight
        legitimately NaNs samples below min_pitch_deg at dive/climb
        inflections, so a mostly-NaN cast with real steady-glide stretches
        must still compute (asymmetry documented in speed.py)."""
        pf = vmp_descent
        n = pf.channels["P"].size
        pitch = np.full(n, 1.0)          # below min_pitch → NaN in the model
        pitch[: n // 4] = -30.0          # a real glide stretch (25% of cast)
        pf.channels["Incl_Y"] = pitch
        pf.channels["Incl_X"] = np.full(n, 2.0)
        speed_fast, _, _ = compute_speed_for_pfile(
            pf, {"method": "flight"}, vehicle="slocum_glider",
        )
        assert np.isfinite(speed_fast).all()


class TestConstantMethod:
    def test_constant_uses_value(self, vmp_descent):
        speed_fast, _, _ = compute_speed_for_pfile(
            vmp_descent, {"method": "constant", "value": 0.42}, vehicle="vmp",
        )
        np.testing.assert_allclose(speed_fast, 0.42)

    def test_constant_value_required(self, vmp_descent):
        with pytest.raises(ValueError, match=r"speed\.value is null"):
            compute_speed_for_pfile(
                vmp_descent, {"method": "constant", "value": None}, vehicle="vmp",
            )

    def test_constant_non_finite_value_raises(self, vmp_descent):
        """A NaN/inf speed.value must ERROR (PR #139 P1): max(nan, cutout)
        propagates NaN, which downstream would be floored to 0.05 m/s and
        published with provenance 'constant:nan'."""
        for bad in (float("nan"), float("inf")):
            with pytest.raises(ValueError, match=r"is not finite"):
                compute_speed_for_pfile(
                    vmp_descent, {"method": "constant", "value": bad}, vehicle="vmp",
                )


class TestHotelMethod:
    """speed.method='hotel': consume a hotel-merged channel (#131 M10)."""

    def test_slow_grid_channel_by_length(self, vmp_descent):
        """A slow-grid hotel channel (duck-typed source without is_fast:
        length matching) is |·|/interp/smoothed to fast rate."""
        pf = vmp_descent
        pf.channels["speed"] = np.full_like(pf.channels["P"], 0.35)
        speed_fast, W_slow, src = compute_speed_for_pfile(
            pf, {"method": "hotel"}, vehicle="slocum_glider",
        )
        assert src == "hotel:speed"
        assert speed_fast.shape == pf.t_fast.shape
        np.testing.assert_allclose(np.median(speed_fast[1000:-1000]), 0.35, atol=1e-3)
        # W_slow is independent of method — still the |dP/dt| of the descent.
        np.testing.assert_allclose(np.median(np.abs(W_slow[100:-100])), 0.5, atol=0.02)

    def test_fast_grid_channel_via_is_fast(self):
        """The default hotel fast_channels puts 'speed' on the FAST grid;
        the grid comes from pf.is_fast, and the fast branch mirrors
        _slow_to_fast (NaN-interp + Butterworth + floor, no regrid)."""
        fs_slow, fs_fast = 64.0, 512.0
        n_slow, n_fast = 640, 5120
        pf = _StubPFWithGrid(
            P=0.5 * np.arange(n_slow) / fs_slow,
            t_slow=np.arange(n_slow) / fs_slow,
            t_fast=np.arange(n_fast) / fs_fast,
            fs_slow=fs_slow, fs_fast=fs_fast,
            fast_channels={"speed"},
        )
        speed = np.full(n_fast, 0.28)
        speed[100:200] = np.nan  # interior gap: interpolated, not floored
        pf.channels["speed"] = speed
        speed_fast, _, src = compute_speed_for_pfile(
            pf, {"method": "hotel"}, vehicle="slocum_glider",
        )
        assert src == "hotel:speed"
        assert np.isfinite(speed_fast).all()
        np.testing.assert_allclose(np.median(speed_fast[1000:-1000]), 0.28, atol=1e-3)
        # The gap must be bridged with neighboring data, not the 0.05 floor.
        np.testing.assert_allclose(speed_fast[100:200], 0.28, atol=1e-2)

    def test_is_fast_preferred_over_length(self):
        """A channel registered fast but with slow-grid length must error:
        the merge's grid registration wins over length matching."""
        fs_slow, fs_fast = 64.0, 512.0
        n_slow, n_fast = 640, 5120
        pf = _StubPFWithGrid(
            P=0.5 * np.arange(n_slow) / fs_slow,
            t_slow=np.arange(n_slow) / fs_slow,
            t_fast=np.arange(n_fast) / fs_fast,
            fs_slow=fs_slow, fs_fast=fs_fast,
            fast_channels={"speed"},
        )
        pf.channels["speed"] = np.full(n_slow, 0.3)  # slow length, fast-registered
        with pytest.raises(ValueError, match=r"does not match the fast grid"):
            compute_speed_for_pfile(pf, {"method": "hotel"}, vehicle="slocum_glider")

    def test_hotel_var_selects_channel(self, vmp_descent):
        pf = vmp_descent
        pf.channels["u_thru"] = np.full_like(pf.channels["P"], 0.42)
        speed_fast, _, src = compute_speed_for_pfile(
            pf, {"method": "hotel", "hotel_var": "u_thru"}, vehicle="slocum_glider",
        )
        assert src == "hotel:u_thru"
        np.testing.assert_allclose(np.median(speed_fast[1000:-1000]), 0.42, atol=1e-3)

    def test_takes_abs(self, vmp_descent):
        """Negative telemetry (sign convention) is mapped to |speed|."""
        pf = vmp_descent
        pf.channels["speed"] = np.full_like(pf.channels["P"], -0.35)
        speed_fast, _, _ = compute_speed_for_pfile(
            pf, {"method": "hotel"}, vehicle="slocum_glider",
        )
        np.testing.assert_allclose(np.median(speed_fast[1000:-1000]), 0.35, atol=1e-3)

    def test_missing_channel_raises(self, vmp_descent):
        """Missing hotel channel is an error naming hotel_var and the
        hotel.channels remedy — no fall-back."""
        with pytest.raises(ValueError, match=r"'speed' is not present.*hotel\.channels"):
            compute_speed_for_pfile(
                vmp_descent, {"method": "hotel"}, vehicle="slocum_glider",
            )

    def test_missing_custom_var_named_in_error(self, vmp_descent):
        with pytest.raises(ValueError, match=r"'u_thru' is not present"):
            compute_speed_for_pfile(
                vmp_descent, {"method": "hotel", "hotel_var": "u_thru"},
                vehicle="slocum_glider",
            )

    def test_neither_grid_raises(self, vmp_descent):
        pf = vmp_descent
        pf.channels["speed"] = np.full(17, 0.3)  # matches neither grid
        with pytest.raises(ValueError, match=r"matching neither the fast grid"):
            compute_speed_for_pfile(pf, {"method": "hotel"}, vehicle="slocum_glider")

    def test_mostly_nan_raises_not_floor(self, vmp_descent):
        """< 50% finite must ERROR — an explicitly requested hotel speed is
        never silently replaced by the 0.05 m/s speed_cutout floor (F6)."""
        pf = vmp_descent
        speed = np.full_like(pf.channels["P"], 0.35)
        speed[: int(0.6 * len(speed))] = np.nan  # 60% NaN
        pf.channels["speed"] = speed
        with pytest.raises(ValueError, match=r"finite.*speed_cutout floor"):
            compute_speed_for_pfile(pf, {"method": "hotel"}, vehicle="slocum_glider")

    def test_all_nan_raises_not_floor(self, vmp_descent):
        """All-NaN (dead external feed) would historically become a constant
        0.05 m/s via _slow_to_fast's fallback; for hotel it is an error."""
        pf = vmp_descent
        pf.channels["speed"] = np.full_like(pf.channels["P"], np.nan)
        with pytest.raises(ValueError, match=r"0\.0% finite"):
            compute_speed_for_pfile(pf, {"method": "hotel"}, vehicle="slocum_glider")


class TestSourceReturn:
    """Third return value: the provenance vocabulary (#131 W1b/F17)."""

    def test_source_vocabulary(self, vmp_descent):
        pf = vmp_descent
        _, _, src = compute_speed_for_pfile(pf, {}, vehicle="vmp")
        assert src == "pressure"
        _, _, src = compute_speed_for_pfile(
            pf, {"method": "constant", "value": 0.4}, vehicle="vmp",
        )
        assert src == "constant:0.4"
        pf.channels["U_EM"] = np.full_like(pf.channels["P"], 0.3)
        _, _, src = compute_speed_for_pfile(
            pf, {"method": "em"}, vehicle="slocum_glider",
        )
        assert src == "em"
        # MR convention: Incl_Y ~ pitch (larger swing), Incl_X mostly roll.
        n = pf.channels["P"].size
        pf.channels["Incl_X"] = np.full_like(pf.channels["P"], 2.0)
        pf.channels["Incl_Y"] = -30.0 + np.sin(2 * np.pi * np.arange(n) / n)
        _, _, src = compute_speed_for_pfile(
            pf, {"method": "flight"}, vehicle="slocum_glider",
        )
        assert src == "flight"


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
        speed, _, _ = compute_speed_for_pfile(
            glider_with_incl,
            {"method": "flight", "aoa_deg": 0.0},
            vehicle="slocum_glider",
        )
        np.testing.assert_allclose(
            np.median(speed[1000:-1000]), 0.5 / np.sin(np.deg2rad(30.0)), atol=0.02
        )


class TestEMScale:
    """speed.em_scale: speed = k * |U_EM| (Merckelbach et al. 2019 k = 0.93)."""

    NOTE = "test factor"

    def test_identity_is_bit_identical(self, glider_with_em):
        pf = glider_with_em
        pf.channels["U_EM"] = 0.32 + 0.05 * np.sin(np.arange(pf.channels["P"].size) / 50.0)
        base, _, src0 = compute_speed_for_pfile(pf, {"method": "em"}, vehicle="slocum_glider")
        for k in (1.0, 1, None):
            s, _, src = compute_speed_for_pfile(
                pf, {"method": "em", "em_scale": k}, vehicle="slocum_glider",
            )
            np.testing.assert_array_equal(s, base)
            assert src == src0 == "em"

    def test_scales_speed_exactly(self, glider_with_em):
        """Interp and smoothing are linear, so off the floor the product is k x."""
        pf = glider_with_em
        pf.channels["U_EM"] = 0.32 + 0.05 * np.sin(np.arange(pf.channels["P"].size) / 50.0)
        base, _, _ = compute_speed_for_pfile(pf, {"method": "em"}, vehicle="slocum_glider")
        s, W, src = compute_speed_for_pfile(
            pf,
            {"method": "em", "em_scale": 0.9, "em_correction_note": self.NOTE},
            vehicle="slocum_glider",
        )
        np.testing.assert_allclose(s, 0.9 * base, rtol=1e-12)
        assert src == "em[k=0.9]"
        # W_slow is |dP/dt| and does not move.
        _, W0, _ = compute_speed_for_pfile(pf, {"method": "em"}, vehicle="slocum_glider")
        np.testing.assert_array_equal(W, W0)

    def test_scales_abs_of_negative_u_em(self, vmp_descent):
        pf = vmp_descent
        pf.channels["U_EM"] = np.full_like(pf.channels["P"], -0.4)
        s, _, _ = compute_speed_for_pfile(
            pf,
            {"method": "em", "em_scale": 0.85, "em_correction_note": self.NOTE},
            vehicle="slocum_glider",
        )
        np.testing.assert_allclose(np.median(s[1000:-1000]), 0.34, atol=1e-3)

    def test_imputation_suffix_kept(self, glider_with_em):
        pf = glider_with_em
        pf.channels["U_EM"][100:200] = np.nan
        _, _, src = compute_speed_for_pfile(
            pf,
            {"method": "em", "em_scale": 0.93, "em_correction_note": self.NOTE},
            vehicle="slocum_glider",
        )
        assert src.startswith("em[k=0.93](cov=")

    def test_warns_when_pushed_under_cutout(self, glider_with_em):
        pf = glider_with_em
        pf.channels["U_EM"] = np.full_like(pf.channels["P"], 0.052)
        with pytest.warns(UserWarning, match=r"below speed_cutout"):
            compute_speed_for_pfile(
                pf,
                {"method": "em", "em_scale": 0.9, "em_correction_note": self.NOTE},
                vehicle="slocum_glider",
            )

    @pytest.mark.parametrize("bad", [0.0, -0.9, float("nan"), float("inf"), "0.9", True])
    def test_invalid_value_raises(self, glider_with_em, bad):
        with pytest.raises(ValueError, match=r"speed\.em_scale"):
            compute_speed_for_pfile(
                glider_with_em,
                {"method": "em", "em_scale": bad, "em_correction_note": self.NOTE},
                vehicle="slocum_glider",
            )

    @pytest.mark.parametrize("note", [None, "", "   "])
    def test_note_required(self, glider_with_em, note):
        with pytest.raises(ValueError, match=r"needs speed\.em_correction_note"):
            compute_speed_for_pfile(
                glider_with_em,
                {"method": "em", "em_scale": 0.9, "em_correction_note": note},
                vehicle="slocum_glider",
            )

    @pytest.mark.parametrize("method", ["pressure", "flight", "constant", "hotel"])
    def test_non_em_method_is_an_error(self, glider_with_incl, method):
        """Decided 2026-09-15 (docs/uem_correction_PLAN.md §6): a factor that
        would silently do nothing is an error. The text is asserted on purpose."""
        expected = (
            f"speed.em_scale is set but speed.method is {method!r}. The U_EM "
            "correction is applied only when speed comes from U_EM (method 'em'); "
            f"with method {method!r} it would be silently ignored and the products "
            "would not be corrected. Remove that key, or set speed.method: em."
        )
        with pytest.raises(ValueError) as exc:
            compute_speed_for_pfile(
                glider_with_incl,
                {"method": method, "value": 0.3, "em_scale": 0.9,
                 "em_correction_note": self.NOTE},
                vehicle="slocum_glider",
            )
        assert str(exc.value) == expected

    def test_identity_under_other_method_is_fine(self, glider_with_incl):
        compute_speed_for_pfile(
            glider_with_incl, {"method": "flight", "em_scale": 1.0}, vehicle="slocum_glider",
        )


class TestEMScaleConfig:
    """perturb config: load-time validation and the stage signature."""

    def _write(self, tmp_path, speed_yaml):
        p = tmp_path / "c.yaml"
        p.write_text("speed:\n" + speed_yaml)
        return p

    def test_default_and_template(self, tmp_path):
        from odas_tpw.perturb import config as cfg

        assert cfg.DEFAULTS["speed"]["em_scale"] == 1.0
        assert cfg.DEFAULTS["speed"]["em_correction_note"] is None
        template = cfg.generate_template(tmp_path / "t.yaml").read_text()
        assert "em_scale: 1.0" in template
        assert "em_correction_note: null" in template

    def test_identity_leaves_the_signature_unchanged(self):
        from odas_tpw.perturb import config as cfg

        for spec in ({}, {"method": "em"}, {"method": "flight", "aoa_deg": 4.0}):
            view = cfg.canonicalize("speed", spec)
            assert "em_scale" not in view and "em_correction_note" not in view
            h = cfg.compute_hash("speed", spec)
            assert cfg.compute_hash("speed", {**spec, "em_scale": 1.0}) == h
            assert cfg.compute_hash("speed", {**spec, "em_scale": None}) == h
            # A note without a factor is inert too: it documents a correction
            # that is not applied, so it must not re-version anything.
            assert cfg.compute_hash("speed", {**spec, "em_correction_note": "x"}) == h
            assert (
                cfg.compute_hash("speed", {**spec, "em_scale": 1, "em_correction_note": "x"}) == h
            )
        em = {"method": "em"}
        scaled = {"method": "em", "em_scale": 0.9, "em_correction_note": "x"}
        assert cfg.compute_hash("speed", scaled) != cfg.compute_hash("speed", em)
        assert cfg.compute_hash("speed", scaled) != cfg.compute_hash(
            "speed", {**scaled, "em_scale": 0.93}
        )

    def test_process_file_refuses_a_factor_pressure_would_ignore(self, tmp_path):
        # process_file can be handed a config that never went through
        # load_config (Python callers). Its speed stage turns a failure under
        # the default method "pressure" into a warning and |dP/dt|, so the
        # check must run before it, and raise.
        from odas_tpw.perturb.pipeline import process_file

        config = {"speed": {"em_scale": 0.9, "em_correction_note": "x"}}
        with pytest.raises(ValueError, match=r"speed\.method is 'pressure'"):
            process_file(tmp_path / "never_read.p", config, None, {})

    def test_loads_a_valid_factor(self, tmp_path):
        from odas_tpw.perturb import config as cfg

        p = self._write(
            tmp_path, '  method: "em"\n  em_scale: 0.9\n  em_correction_note: "Merckelbach"\n'
        )
        assert cfg.load_config(p)["speed"]["em_scale"] == 0.9

    @pytest.mark.parametrize(
        ("body", "match"),
        [
            ('  method: "flight"\n  em_scale: 0.9\n  em_correction_note: "x"\n',
             r"speed\.method is 'flight'"),
            ("  em_scale: 0.9\n  em_correction_note: \"x\"\n", r"speed\.method is 'pressure'"),
            ('  method: "em"\n  em_scale: 0.9\n', r"needs speed\.em_correction_note"),
            ('  method: "em"\n  em_scale: -1\n  em_correction_note: "x"\n', r"> 0"),
            ('  method: "em"\n  em_scale: "0.9"\n  em_correction_note: "x"\n',
             r"expected a number"),
        ],
    )
    def test_fails_at_load(self, tmp_path, body, match):
        from odas_tpw.perturb import config as cfg

        with pytest.raises(ValueError, match=match):
            cfg.load_config(self._write(tmp_path, body))

    def test_out_of_range_warns_at_load(self, tmp_path):
        from odas_tpw.perturb import config as cfg

        p = self._write(tmp_path, '  method: "em"\n  em_scale: 0.6\n  em_correction_note: "x"\n')
        with pytest.warns(UserWarning, match=r"outside \[0\.7, 1\.3\]"):
            cfg.load_config(p)


class TestGliderPressureRefusedInPerturb:
    """process_file aborts a glider file under the pressure method instead of
    falling back to |dP/dt| (the fallback that hid the RIOT 2026 defect)."""

    FIXTURE = Path(__file__).parent / "data" / "MR_SL435.p"

    def _run(self, tmp_path, speed):
        from odas_tpw.perturb.pipeline import process_file

        if not self.FIXTURE.exists():
            pytest.skip("MR_SL435.p test data not available")
        cfg = {"speed": speed} if speed is not None else {}
        dirs = {k: tmp_path / k for k in ("profiles", "diss", "chi")}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return process_file(self.FIXTURE, cfg, None, dirs), dirs

    @pytest.mark.parametrize("speed", [None, {"method": "pressure"}])
    def test_aborted_with_the_reason(self, tmp_path, speed):
        result, dirs = self._run(tmp_path, speed)
        errors = result.get("errors") or []
        assert len(errors) == 1 and errors[0].startswith(
            "speed: speed.method 'pressure' is refused on a glider")
        assert "slocum_glider" in errors[0] and "speed.method: em" in errors[0]
        assert result["profiles"] == []
        assert not any(dirs["profiles"].glob("*.nc"))

    def test_malformed_tau_does_not_reopen_the_fallback(self, tmp_path):
        """speed.tau = 0.001 used to fail inside the stage first, which the
        default method's warn-and-continue turned into |dP/dt| products."""
        result, _ = self._run(tmp_path, {"tau": 0.001})
        errors = result.get("errors") or []
        assert len(errors) == 1 and "refused on a glider" in errors[0]
        assert result["profiles"] == []

    def test_em_is_not_aborted(self, tmp_path):
        result, _ = self._run(tmp_path, {"method": "em"})
        assert not any(str(e).startswith("speed:") for e in result.get("errors") or [])
