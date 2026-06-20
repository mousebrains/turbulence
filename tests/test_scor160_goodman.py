# Tests for odas_tpw.scor160.goodman
"""Unit tests for Goodman coherent noise removal."""

import numpy as np
import pytest

from odas_tpw.scor160.goodman import (
    _sanitize_autospectra,
    clean_shear_spec,
    clean_shear_spec_batch,
)


class TestSanitizeAutospectra:
    def test_floors_negative_and_nans_nonfinite(self):
        """Diagonal: finite negatives floor to 0; non-finite (NaN from the
        singular fallback, or inf from a solve that didn't raise) -> NaN;
        positive diagonals and off-diagonal cross-spectra untouched (#84/#83)."""
        # (n_freq=3, n_sh=2, n_sh=2)
        m = np.array([
            [[1.0, 5.0], [5.0, -2.0]],        # freq 0: [1,1]=-2 negative -> 0
            [[np.nan, 7.0], [7.0, 4.0]],      # freq 1: [0,0]=NaN stays NaN
            [[np.inf, 9.0], [9.0, 6.0]],      # freq 2: [0,0]=inf -> NaN (portable)
        ])
        out = _sanitize_autospectra(m.copy())
        assert out[0, 1, 1] == 0.0
        assert np.isnan(out[1, 0, 0])
        assert np.isnan(out[2, 0, 0])         # inf -> NaN
        assert out[0, 0, 0] == 1.0            # positive diagonal kept
        assert out[2, 1, 1] == 6.0
        assert out[0, 0, 1] == 5.0            # off-diagonal kept
        assert out[2, 0, 1] == 9.0

    def test_batched_shape_supported(self):
        """Works on the batched (n_win, n_freq, n_sh, n_sh) shape too."""
        m = np.ones((2, 3, 2, 2))
        m[1, 2, 0, 0] = -1.0
        out = _sanitize_autospectra(m.copy())
        assert out[1, 2, 0, 0] == 0.0
        assert (out[0] == 1.0).all()


class TestNanSingularBins:
    def test_singular_aa_bins_nan_regardless_of_solver(self):
        """A bin whose AA is exactly singular (cond == inf) is NaN'd via the
        condition number, independent of whether np.linalg.solve raised (the
        portability fix for the macOS/py3.14 LAPACK build, #83)."""
        from odas_tpw.scor160.goodman import _nan_singular_bins

        clean = np.ones((3, 1, 1))                 # 3 freq bins, n_sh=1
        AA = np.tile(np.eye(2)[None], (3, 1, 1))   # well-conditioned default
        AA[1] = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank-1 -> singular
        out = _nan_singular_bins(clean.copy(), AA)
        assert np.isnan(out[1, 0, 0])              # singular bin NaN'd
        assert out[0, 0, 0] == 1.0                 # well-conditioned bins kept
        assert out[2, 0, 0] == 1.0

    def test_nonfinite_aa_does_not_crash_svd(self):
        """A NaN/inf in AA must not feed np.linalg.cond's SVD (which raises
        'SVD did not converge' on the whole batch); treat it as a bad bin
        (audit round-2 M-3 regression)."""
        from odas_tpw.scor160.goodman import _nan_singular_bins

        clean = np.ones((3, 1, 1))
        AA = np.tile(np.eye(2)[None], (3, 1, 1))
        AA[1, 0, 0] = np.nan                       # one NaN in bin 1's AA
        out = _nan_singular_bins(clean.copy(), AA)  # must not raise
        assert np.isnan(out[1, 0, 0])              # NaN-AA bin flagged
        assert out[0, 0, 0] == 1.0
        assert out[2, 0, 0] == 1.0

    def test_clean_shear_spec_batch_survives_nan_accel(self):
        """End-to-end: a single NaN accelerometer sample must not crash the
        batched Goodman path; the contaminated bins come back NaN, not an
        exception (M-3)."""
        rng = np.random.default_rng(7)
        accel = rng.standard_normal((2, 1024, 2))
        shear = rng.standard_normal((2, 1024, 2))
        accel[0, 500, 0] = np.nan                  # one corrupted vibration sample
        clean_UU, _F = clean_shear_spec_batch(accel, shear, 256, 512.0)  # no raise
        assert clean_UU.shape == (2, 129, 2, 2)
        assert np.isnan(clean_UU[0]).any()         # window 0 has NaN-flagged bins
        assert np.isfinite(clean_UU[1]).any()      # clean window still usable


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

        # Compare auto-spectral diagonals. nansum: a singular-AA frequency bin
        # is now NaN'd (uncleanable) rather than left as the raw uncleaned UU
        # (#83), so exclude it from the power comparison.
        for i in range(shear.shape[1]):
            clean_power = np.nansum(clean_UU[:, i, i])
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
                # nansum: singular-AA bins are NaN'd (uncleanable), not raw UU (#83).
                clean_power = np.nansum(clean_UU[w, :, i, i])
                raw_power = np.sum(np.real(raw_Cxy[w, :, i, i]))
                assert clean_power < raw_power
