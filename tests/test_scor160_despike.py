# Tests for odas_tpw.scor160.despike
"""Unit tests for iterative spike removal."""

import warnings

import numpy as np
import pytest

from odas_tpw.scor160.despike import _single_despike, despike


class TestDespikeCleanSignal:
    """Despike should be a no-op on clean signals."""

    def test_no_spikes_found(self):
        rng = np.random.default_rng(42)
        fs = 512.0
        t = np.arange(0, 10, 1 / fs)
        signal = 0.01 * rng.standard_normal(len(t))
        y, spikes, n_passes, frac = despike(signal, fs)
        assert len(spikes) == 0
        assert n_passes == 0
        assert frac == 0.0
        np.testing.assert_array_equal(y, signal)

    def test_constant_signal(self):
        fs = 512.0
        signal = np.ones(5000)
        y, spikes, _n_passes, _frac = despike(signal, fs)
        assert len(spikes) == 0
        np.testing.assert_array_equal(y, signal)

    def test_nan_input_reports_zero_fraction(self):
        """filtfilt disables detection on NaN input; the no-op must report
        fraction 0, not count unchanged NaN positions as changed (#36)."""
        fs = 512.0
        rng = np.random.default_rng(1)
        signal = 0.01 * rng.standard_normal(5000)
        signal[1234] = np.nan  # one dropout
        with pytest.warns(RuntimeWarning, match="non-finite"):
            _y, _spikes, n_passes, frac = despike(signal, fs)
        assert n_passes == 0  # no spikes detectable through the NaN
        assert frac == 0.0  # NaN != NaN must not inflate the fraction


class TestDespikeEdgeCases:
    """Regression tests for short-input and degenerate-region handling."""

    @pytest.mark.parametrize("length", [1, 2, 3, 5])
    def test_short_input_does_not_crash(self, length):
        """Short sections must not raise scipy's 'len(x) > padlen' ValueError.

        With the MATLAB padlen (3) plus a too-short guard, despike degrades to
        a no-op instead of crashing. The old scipy-default padlen (6) raised
        ValueError on a length-2 (padded-to-6) input. (#25/#49)
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = despike(np.zeros(length), 512.0)
        assert result.y.shape == (length,)
        assert result.n_passes == 0
        assert len(result.spike_indices) == 0

    def test_nan_input_emits_warning(self):
        """A non-finite sample silently disables ALL spike detection via
        filtfilt NaN propagation; despike must warn so the no-op is
        observable rather than silently passing real spikes through. (#26)
        """
        rng = np.random.default_rng(2)
        signal = 0.01 * rng.standard_normal(10000)
        signal[3000] = 50.0  # a genuine spike that will NOT be removed
        signal[7000] = np.nan  # one dropout disables detection globally
        with pytest.warns(RuntimeWarning, match="non-finite"):
            y, spikes, n_passes, _frac = despike(signal, 512.0)
        # Detection is disabled, so the real spike survives (documents the
        # data-integrity hazard the warning now surfaces).
        assert len(spikes) == 0
        assert n_passes == 0
        assert y[3000] == 50.0

    def test_finite_input_emits_no_nan_warning(self):
        """Clean finite input must not raise the non-finite RuntimeWarning."""
        rng = np.random.default_rng(3)
        signal = 0.01 * rng.standard_normal(5000)
        signal[2500] = 50.0
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            despike(signal, 512.0)  # must not raise

    def test_unfillable_region_yields_nan_not_zero(self):
        """When a bad region has no valid neighbors on either side, the
        replacement must be NaN (matching MATLAB sum([])/length([])=NaN) so
        the data loss is detectable, not a fabricated finite 0.0. (#27/#50)
        """
        dv = np.zeros(10)
        dv[5] = 50.0
        # A removal width N far larger than the array marks the whole
        # (padded) array bad, leaving no valid neighbor on either side.
        y, _spikes = _single_despike(dv, thresh=8.0, smooth=0.5, fs=512.0, N=1000)
        assert np.all(np.isnan(y))
        assert not np.any(y == 0.0)  # old code injected a finite 0.0 here


class TestDespikeSyntheticSpikes:
    """Despike should detect and remove synthetic spikes."""

    def test_single_large_spike(self):
        rng = np.random.default_rng(123)
        fs = 512.0
        n = 10000
        signal = 0.01 * rng.standard_normal(n)
        spike_loc = 5000
        signal[spike_loc] = 50.0  # enormous spike

        y, spikes, n_passes, frac = despike(signal, fs)
        assert spike_loc in spikes
        assert n_passes >= 1
        assert frac > 0
        # Spike should be replaced — output at that location should be much smaller
        assert abs(y[spike_loc]) < 1.0

    def test_multiple_spikes(self):
        rng = np.random.default_rng(456)
        fs = 512.0
        n = 20000
        signal = 0.01 * rng.standard_normal(n)
        spike_locs = [3000, 8000, 15000]
        for loc in spike_locs:
            signal[loc] = 40.0

        _y, spikes, _n_passes, frac = despike(signal, fs)
        # All spike locations should be detected
        for loc in spike_locs:
            assert loc in spikes
        assert frac > 0

    def test_negative_spike(self):
        rng = np.random.default_rng(789)
        fs = 512.0
        n = 10000
        signal = 0.01 * rng.standard_normal(n)
        signal[4000] = -40.0

        y, spikes, _, _ = despike(signal, fs)
        assert 4000 in spikes
        assert abs(y[4000]) < 1.0

    def test_output_shape_matches_input(self):
        rng = np.random.default_rng(111)
        fs = 512.0
        n = 8000
        signal = rng.standard_normal(n)
        y, _spikes, _n_passes, _frac = despike(signal, fs)
        assert y.shape == signal.shape

    def test_fraction_bounds(self):
        rng = np.random.default_rng(222)
        fs = 512.0
        n = 10000
        signal = 0.01 * rng.standard_normal(n)
        signal[5000] = 50.0
        _, _, _, frac = despike(signal, fs)
        assert 0 <= frac <= 1


class TestDespikeParameters:
    """Test parameter effects."""

    def test_high_threshold_ignores_spikes(self):
        rng = np.random.default_rng(333)
        fs = 512.0
        n = 10000
        signal = 0.01 * rng.standard_normal(n)
        signal[5000] = 5.0  # moderate spike

        _, spikes_normal, _, _ = despike(signal.copy(), fs, thresh=8)
        _, spikes_high, _, _ = despike(signal.copy(), fs, thresh=1000)
        # Very high threshold should find fewer/no spikes
        assert len(spikes_high) <= len(spikes_normal)

    def test_max_passes_limit(self):
        rng = np.random.default_rng(444)
        fs = 512.0
        n = 10000
        signal = 0.01 * rng.standard_normal(n)
        # Add many spikes
        for i in range(100, n - 100, 200):
            signal[i] = 30.0

        _, _, n_passes, _ = despike(signal, fs, max_passes=2)
        assert n_passes <= 2

    def test_custom_N(self):
        rng = np.random.default_rng(555)
        fs = 512.0
        n = 10000
        signal = 0.01 * rng.standard_normal(n)
        signal[5000] = 50.0

        _y, spikes, _, _ = despike(signal, fs, N=10)
        assert len(spikes) > 0


class TestDespikeReturnTypes:
    """Verify return types."""

    def test_return_tuple(self):
        rng = np.random.default_rng(666)
        fs = 512.0
        result = despike(rng.standard_normal(5000), fs)
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_spike_indices_dtype(self):
        rng = np.random.default_rng(777)
        fs = 512.0
        signal = 0.01 * rng.standard_normal(10000)
        signal[5000] = 50.0
        _, spikes, _, _ = despike(signal, fs)
        assert spikes.dtype == np.intp

    def test_spike_indices_sorted(self):
        rng = np.random.default_rng(888)
        fs = 512.0
        signal = 0.01 * rng.standard_normal(10000)
        signal[3000] = 50.0
        signal[7000] = -50.0
        _, spikes, _, _ = despike(signal, fs)
        assert np.all(np.diff(spikes) >= 0)
