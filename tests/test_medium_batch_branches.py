# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Medium-batch branch tests — scor160/l2, despike, pyturb/_profind."""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# scor160/l2.py — _filtfilt_nan all-NaN early-out + empty data in _hp_filter
# ---------------------------------------------------------------------------


class TestL2NanFilter:
    def test_filtfilt_nan_all_nan_returns_copy(self):
        """_filtfilt_nan with an all-NaN array short-circuits to x.copy()."""
        from scipy.signal import butter

        from odas_tpw.scor160.l2 import _filtfilt_nan

        b, a = butter(1, 0.1)
        x = np.full(100, np.nan)
        result = _filtfilt_nan(b, a, x)
        assert np.all(np.isnan(result))
        assert result is not x  # copy, not original

    def test_filtfilt_nan_with_partial_nan(self):
        """_filtfilt_nan interpolates internal NaN and restores them after filtering."""
        from scipy.signal import butter

        from odas_tpw.scor160.l2 import _filtfilt_nan

        b, a = butter(1, 0.1)
        x = np.linspace(0.0, 10.0, 200)
        x[50:55] = np.nan
        result = _filtfilt_nan(b, a, x)
        # NaN positions stay NaN
        assert np.all(np.isnan(result[50:55]))
        # Other positions are finite
        assert np.all(np.isfinite(result[:50]))

    def test_hp_filter_empty_data(self):
        """_hp_filter on empty array returns a copy without computing anything."""
        from odas_tpw.scor160.l2 import _hp_filter

        empty = np.array([], dtype=np.float64)
        result = _hp_filter(empty, fs=512.0, f_hp=0.25)
        assert result.size == 0

    def test_hp_filter_zero_fhp(self):
        """f_hp <= 0 → return data.copy() (no filtering applied)."""
        from odas_tpw.scor160.l2 import _hp_filter

        data = np.linspace(0.0, 1.0, 100)
        result = _hp_filter(data, fs=512.0, f_hp=0.0)
        np.testing.assert_array_equal(result, data)


# ---------------------------------------------------------------------------
# scor160/l2.py — _select_sections edge cases
# ---------------------------------------------------------------------------


class TestL2SelectSections:
    def test_horizontal_uses_abs_speed(self):
        """direction='horizontal' → abs(speed) >= min_speed criterion."""
        from odas_tpw.scor160.l2 import _select_sections

        n = 1000
        # Negative-going speed for a horizontal profiler
        speed = np.full(n, -0.5)
        pres = np.full(n, 10.0)
        sections = _select_sections(
            speed, pres, fs=64.0, min_speed=0.2, min_pressure=5.0,
            min_duration=1.0, direction="horizontal",
        )
        # Negative speed but |speed| >= 0.2 → all in section 1
        assert sections[100] > 0
        assert np.all(sections[(speed > -100)] == sections[100])  # all same sec_id

    def test_good_at_start_and_end(self):
        """When the section spans the array boundaries, the start/end fix-ups run."""
        from odas_tpw.scor160.l2 import _select_sections

        n = 1000
        speed = np.full(n, 0.5)  # entirely above min_speed
        pres = np.full(n, 10.0)
        sections = _select_sections(
            speed, pres, fs=64.0, min_speed=0.2, min_pressure=5.0,
            min_duration=1.0, direction="down",
        )
        # Section starts at 0 and ends at n
        assert sections[0] > 0
        assert sections[-1] > 0


# ---------------------------------------------------------------------------
# scor160/despike.py — boundary spikes (no before / no after data)
# ---------------------------------------------------------------------------


class TestDespikeBoundary:
    def test_spike_at_array_start(self):
        """A spike at the very start covers the `not good[0]` branch (line 137)."""
        from odas_tpw.scor160.despike import despike

        rng = np.random.default_rng(42)
        fs = 512.0
        n = 10000
        signal = 0.01 * rng.standard_normal(n)
        signal[20] = 50.0  # near-start spike
        result = despike(signal, fs)
        assert result.n_passes >= 1

    def test_spike_at_array_end(self):
        """A spike at the very end covers the `not good[-1]` branch (line 139)."""
        from odas_tpw.scor160.despike import despike

        rng = np.random.default_rng(43)
        fs = 512.0
        n = 10000
        signal = 0.01 * rng.standard_normal(n)
        signal[-20] = 50.0  # near-end spike
        result = despike(signal, fs)
        assert result.n_passes >= 1


# ---------------------------------------------------------------------------
# pyturb/_profind.py — high-frequency cutoff clamp + no maxima/minima paths
# ---------------------------------------------------------------------------


class TestProfindEdges:
    def test_smoothing_tau_too_small_clamps_fc(self):
        """Tiny smoothing_tau with low fs → f_c >= nyquist → clamps to nyquist*0.9."""
        from odas_tpw.pyturb._profind import find_profiles_peaks

        # smoothing_tau=0.1 → f_c = 0.68/0.1 = 6.8 Hz, nyquist=2.0 → f_c clamped
        fs = 4.0
        n = 600
        t = np.arange(n) / fs
        pressure = 50.0 * np.abs(np.sin(np.pi * t / 60.0))
        profiles = find_profiles_peaks(
            pressure, fs, direction="down",
            peaks_height=10.0, peaks_distance=20, peaks_prominence=10.0,
            smoothing_tau=0.1,
        )
        # Should still complete without error
        assert isinstance(profiles, list)

    def test_no_profiles_with_minima_only(self):
        """Pressure with negative-going local minima but no rising peaks."""
        from odas_tpw.pyturb._profind import find_profiles_peaks

        # Monotonically decreasing pressure → maxima will be at start only
        # which won't satisfy peaks_height
        pressure = np.linspace(50.0, 0.0, 600)
        profiles = find_profiles_peaks(
            pressure, 64.0, peaks_height=100.0, peaks_distance=20,
            peaks_prominence=10.0,
        )
        assert profiles == []

    def test_min_pressure_filters_short_drop(self):
        """Profile with insufficient pressure drop is filtered out."""
        from odas_tpw.pyturb._profind import find_profiles_peaks

        fs = 64.0
        n = int(60 * fs)
        t = np.arange(n) / fs
        # Small-amplitude pressure swing
        pressure = 30.0 * np.abs(np.sin(2 * np.pi * t / 60.0))
        # min_pressure=100 is bigger than the actual swing → should drop the profile
        profiles = find_profiles_peaks(
            pressure, fs, direction="down",
            peaks_height=10.0, peaks_distance=100, peaks_prominence=5.0,
            min_pressure=100.0,
        )
        assert profiles == []
