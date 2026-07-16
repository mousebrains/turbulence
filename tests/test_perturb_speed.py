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
        """For pitch=-30°, roll=2°, AoA=3°: U = |W| / (sin(27°)cos(2°))."""
        speed_fast, _, _ = compute_speed_for_pfile(
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
        # Roll: tiny +/-0.5 deg wobble. Pitch: -30 deg +/- 1 deg gentle drift.
        pf.channels["Incl_X"] = 0.5 * np.sin(2 * np.pi * np.arange(n) / n)
        pf.channels["Incl_Y"] = -30.0 + np.sin(2 * np.pi * np.arange(n) / n)
        speed_fast, _, _ = compute_speed_for_pfile(
            pf, {"method": "flight", "aoa_deg": 3.0}, vehicle="slocum_glider",
        )
        # Pitch axis = Incl_Y (range ~2°), roll ≈ Incl_X (range ~1°).
        # Effective sin path ≈ sin(27°)·cos(0°) on average.
        expected = 0.5 / np.sin(np.deg2rad(27.0))
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
        # If the picker was fooled, pitch≈±5° and U = |W|/sin(2°) is huge
        # (>10 m/s), or hits the floor. With the right axis the median
        # speed should be order |W|/sin(~22°) ~ 1.3 m/s.
        median_speed = float(np.median(speed_fast[1000:-1000]))
        assert 0.5 < median_speed < 5.0, (
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
