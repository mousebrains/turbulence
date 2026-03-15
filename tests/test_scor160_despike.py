# Tests for odas_tpw.scor160.despike
"""Unit tests for iterative spike removal."""

import numpy as np

from odas_tpw.scor160.despike import despike


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
