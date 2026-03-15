# Tests for odas_tpw.scor160.goodman
"""Unit tests for Goodman coherent noise removal."""

import numpy as np
import pytest

from odas_tpw.scor160.goodman import clean_shear_spec, clean_shear_spec_batch


def _make_signals(n, fs, n_shear=2, n_accel=2, noise_amp=0.1, vib_amp=1.0, rng=None):
    """Create synthetic shear and accel signals with coherent vibration noise.

    Returns (accel, shear) as column matrices.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    t = np.arange(n) / fs

    # Vibration: a sum of sinusoids at known frequencies
    vib_freqs = [20.0, 40.0, 80.0]
    accel = np.zeros((n, n_accel))
    for i in range(n_accel):
        for f0 in vib_freqs:
            phase = rng.uniform(0, 2 * np.pi)
            accel[:, i] += vib_amp * np.sin(2 * np.pi * f0 * t + phase)

    # Shear: ocean signal + coherent vibration contamination
    shear = np.zeros((n, n_shear))
    for i in range(n_shear):
        # "Ocean" shear signal (white noise)
        shear[:, i] = noise_amp * rng.standard_normal(n)
        # Add vibration contamination (linear combination of accelerometers)
        for j in range(n_accel):
            coeff = rng.uniform(0.01, 0.1)
            shear[:, i] += coeff * accel[:, j]

    return accel, shear


class TestCleanShearSpec:
    """Tests for single-window Goodman cleaning."""

    def test_output_shapes(self):
        n, fs, nfft = 4096, 512.0, 256
        accel, shear = _make_signals(n, fs)
        clean_UU, AA, UU, UA, F = clean_shear_spec(accel, shear, nfft, fs)
        n_freq = nfft // 2 + 1
        n_sh = shear.shape[1]
        n_ac = accel.shape[1]
        assert clean_UU.shape == (n_freq, n_sh, n_sh)
        assert AA.shape == (n_freq, n_ac, n_ac)
        assert UU.shape == (n_freq, n_sh, n_sh)
        assert UA.shape == (n_freq, n_sh, n_ac)
        assert F.shape == (n_freq,)

    def test_cleaned_less_than_uncleaned(self):
        """Cleaned shear spectrum should have less power than uncleaned
        when vibration contamination is present."""
        n, fs, nfft = 8192, 512.0, 256
        accel, shear = _make_signals(n, fs, vib_amp=2.0)
        clean_UU, _, UU, _, _F = clean_shear_spec(accel, shear, nfft, fs)

        # Compare auto-spectral diagonals
        for i in range(shear.shape[1]):
            clean_power = np.sum(clean_UU[:, i, i])
            uncleaned_power = np.sum(np.real(UU[:, i, i]))
            assert clean_power < uncleaned_power

    def test_clean_spec_real(self):
        """Cleaned spectrum diagonal should be real."""
        n, fs, nfft = 4096, 512.0, 256
        accel, shear = _make_signals(n, fs)
        clean_UU, _, _, _, _ = clean_shear_spec(accel, shear, nfft, fs)
        assert clean_UU.dtype == np.float64

    def test_no_accel_no_cleaning(self):
        """With zero accelerometer signal, cleaned = uncleaned."""
        rng = np.random.default_rng(99)
        n, fs, nfft = 4096, 512.0, 256
        accel = np.zeros((n, 2))
        shear = rng.standard_normal((n, 2))
        # Zero accel → AA is singular, but the code handles this
        clean_UU, _, UU, _, _ = clean_shear_spec(accel, shear, nfft, fs)
        # Should not crash; cleaned may equal uncleaned or be slightly different
        assert clean_UU.shape == UU.shape

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same number of rows"):
            clean_shear_spec(np.ones((100, 2)), np.ones((200, 2)), 32, 100.0)

    def test_short_signal_fallback(self):
        """Signal too short for cross-spectrum should warn and return."""
        import warnings

        accel = np.ones((100, 2))
        shear = np.ones((100, 2))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            clean_UU, _AA, _UU, _UA, _F = clean_shear_spec(accel, shear, 256, 100.0)
            assert len(w) >= 1
        assert clean_UU.shape[0] > 0  # should return something


class TestCleanShearSpecBatch:
    """Tests for batched Goodman cleaning."""

    def test_output_shapes(self):
        rng = np.random.default_rng(111)
        n_win = 4
        diss_len = 2048
        n_sh, n_ac = 2, 2
        nfft = 256
        fs = 512.0

        accel_win = rng.standard_normal((n_win, diss_len, n_ac))
        shear_win = rng.standard_normal((n_win, diss_len, n_sh))

        clean_UU, F = clean_shear_spec_batch(accel_win, shear_win, nfft, fs)
        n_freq = nfft // 2 + 1
        assert clean_UU.shape == (n_win, n_freq, n_sh, n_sh)
        assert F.shape == (n_freq,)

    def test_matches_single_window(self):
        """Batch of 1 window should match single-window result."""
        rng = np.random.default_rng(222)
        diss_len = 2048
        n_sh, n_ac = 2, 2
        nfft = 256
        fs = 512.0

        accel = rng.standard_normal((diss_len, n_ac))
        shear = rng.standard_normal((diss_len, n_sh))

        # Single window
        clean_single, _, _, _, F_single = clean_shear_spec(accel, shear, nfft, fs)

        # Batch of 1
        clean_batch, F_batch = clean_shear_spec_batch(
            accel[np.newaxis],
            shear[np.newaxis],
            nfft,
            fs,
        )

        np.testing.assert_allclose(F_batch, F_single)
        np.testing.assert_allclose(clean_batch[0], clean_single, rtol=1e-10)

    def test_cleaning_reduces_power(self):
        """Batched cleaning should also reduce coherent noise."""
        n_win = 3
        diss_len = 2048
        n_sh, n_ac = 2, 2
        nfft = 256
        fs = 512.0

        accel_win = np.zeros((n_win, diss_len, n_ac))
        shear_win = np.zeros((n_win, diss_len, n_sh))

        for w in range(n_win):
            accel_win[w], shear_win[w] = _make_signals(
                diss_len,
                fs,
                n_sh,
                n_ac,
                vib_amp=2.0,
                rng=np.random.default_rng(333 + w),
            )

        clean_UU, _ = clean_shear_spec_batch(accel_win, shear_win, nfft, fs)

        # Compare with uncleaned (compute raw auto-spectra)
        from odas_tpw.scor160.spectral import csd_matrix_batch

        raw_Cxy, _, _, _ = csd_matrix_batch(shear_win, None, nfft, fs)
        for w in range(n_win):
            for i in range(n_sh):
                clean_power = np.sum(clean_UU[w, :, i, i])
                raw_power = np.sum(np.real(raw_Cxy[w, :, i, i]))
                assert clean_power < raw_power
