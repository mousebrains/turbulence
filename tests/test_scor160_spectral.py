# Tests for odas_tpw.scor160.spectral
"""Unit tests for cross-spectral density estimation."""

import numpy as np
import pytest

from odas_tpw.scor160.spectral import csd_matrix, csd_matrix_batch, csd_odas


class TestCsdOdasAutoSpectrum:
    """Auto-spectrum tests for csd_odas."""

    def test_white_noise_flat_spectrum(self):
        """White noise should produce an approximately flat spectrum."""
        rng = np.random.default_rng(42)
        fs = 512.0
        n = 8192
        nfft = 256
        x = rng.standard_normal(n)

        Cxy, F, Cxx, Cyy = csd_odas(x, None, nfft, fs)
        assert Cxx is None
        assert Cyy is None
        assert F[0] == 0.0
        assert F[-1] == pytest.approx(fs / 2)
        assert len(F) == nfft // 2 + 1

        # Spectrum should be roughly flat (within a factor of 5 of the mean)
        mean_level = np.mean(Cxy[1:-1])
        assert np.all(Cxy[1:-1] > mean_level / 5)
        assert np.all(Cxy[1:-1] < mean_level * 5)

    def test_sinusoid_peak(self):
        """Spectrum of a sinusoid should peak at its frequency."""
        fs = 512.0
        n = 8192
        nfft = 256
        f0 = 50.0
        t = np.arange(n) / fs
        x = np.sin(2 * np.pi * f0 * t)

        Cxy, F, _, _ = csd_odas(x, None, nfft, fs)
        peak_idx = np.argmax(Cxy)
        assert F[peak_idx] == pytest.approx(f0, abs=fs / nfft)

    def test_parseval_theorem(self):
        """Integral of PSD should approximate signal variance."""
        rng = np.random.default_rng(99)
        fs = 512.0
        n = 16384
        nfft = 512
        x = rng.standard_normal(n)

        Cxy, F, _, _ = csd_odas(x, None, nfft, fs)
        spectral_var = np.trapezoid(Cxy, F)
        time_var = np.var(x)
        # Should match within ~20% (Welch smoothing introduces some bias)
        assert spectral_var == pytest.approx(time_var, rel=0.2)

    def test_auto_spectrum_real(self):
        """Auto-spectrum should be real-valued."""
        rng = np.random.default_rng(11)
        x = rng.standard_normal(2048)
        Cxy, _, _, _ = csd_odas(x, None, 256, 100.0)
        assert Cxy.dtype in (np.float64, np.float32)

    def test_auto_spectrum_nonnegative(self):
        """Auto-spectrum should be non-negative."""
        rng = np.random.default_rng(22)
        x = rng.standard_normal(4096)
        Cxy, _, _, _ = csd_odas(x, None, 256, 100.0)
        assert np.all(Cxy >= 0)


class TestCsdOdasCrossSpectrum:
    """Cross-spectrum tests for csd_odas."""

    def test_cross_of_same_signal_equals_auto(self):
        """csd_odas(x, x) should detect identity and return auto-spectrum."""
        rng = np.random.default_rng(33)
        x = rng.standard_normal(4096)
        Cauto, F_auto, _, _ = csd_odas(x, None, 256, 100.0)
        Ccross, F_cross, Cxx, Cyy = csd_odas(x, x, 256, 100.0)
        # When x==y, the code detects this and computes auto-spectrum
        np.testing.assert_allclose(Ccross, Cauto, rtol=1e-10)
        assert Cxx is None  # treated as auto
        assert Cyy is None

    def test_cross_spectrum_returns_all(self):
        rng = np.random.default_rng(44)
        x = rng.standard_normal(4096)
        y = rng.standard_normal(4096)
        Cxy, F, Cxx, Cyy = csd_odas(x, y, 256, 100.0)
        assert Cxy.dtype == np.complex128
        assert Cxx is not None
        assert Cyy is not None
        assert len(Cxy) == len(F)

    def test_coherent_signals(self):
        """Cross-spectrum of coherent signals should have high coherence."""
        fs = 100.0
        n = 8192
        nfft = 256
        f0 = 10.0
        t = np.arange(n) / fs
        x = np.sin(2 * np.pi * f0 * t)
        y = 0.5 * np.sin(2 * np.pi * f0 * t + 0.3)

        Cxy, F, Cxx, Cyy = csd_odas(x, y, nfft, fs)
        # Coherence at f0 should be near 1
        peak_idx = np.argmax(Cxx)
        coherence = np.abs(Cxy[peak_idx]) ** 2 / (Cxx[peak_idx] * Cyy[peak_idx])
        assert coherence > 0.95


class TestCsdOdasEdgeCases:
    """Edge cases and error handling."""

    def test_too_short_signal_raises(self):
        x = np.ones(100)
        with pytest.raises(ValueError, match="at least 2\\*nfft"):
            csd_odas(x, None, 256, 100.0)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            csd_odas(np.ones(1024), np.ones(512), 256, 100.0)

    def test_detrend_options(self):
        rng = np.random.default_rng(55)
        x = rng.standard_normal(4096)
        for method in ("none", "constant", "linear"):
            Cxy, F, _, _ = csd_odas(x, None, 256, 100.0, detrend=method)
            assert len(Cxy) == 129  # 256/2+1


class TestCsdMatrix:
    """Tests for multi-channel csd_matrix."""

    def test_auto_spectral_matrix_shape(self):
        rng = np.random.default_rng(66)
        n_ch = 3
        n = 4096
        nfft = 256
        x = rng.standard_normal((n, n_ch))
        Cxy, F, _, _ = csd_matrix(x, None, nfft, 100.0)
        assert Cxy.shape == (nfft // 2 + 1, n_ch, n_ch)

    def test_auto_diagonal_matches_single(self):
        """Diagonal of auto-spectral matrix should match single-channel PSD."""
        rng = np.random.default_rng(77)
        n = 4096
        nfft = 256
        fs = 100.0
        ch0 = rng.standard_normal(n)
        ch1 = rng.standard_normal(n)
        x = np.column_stack([ch0, ch1])

        Cmatrix, _, _, _ = csd_matrix(x, None, nfft, fs)
        Csingle, _, _, _ = csd_odas(ch0, None, nfft, fs)

        np.testing.assert_allclose(
            np.real(Cmatrix[:, 0, 0]), Csingle, rtol=1e-10,
        )

    def test_cross_spectral_matrix_shape(self):
        rng = np.random.default_rng(88)
        n = 4096
        nfft = 256
        n_x, n_y = 2, 3
        x = rng.standard_normal((n, n_x))
        y = rng.standard_normal((n, n_y))
        Cxy, F, Cxx, Cyy = csd_matrix(x, y, nfft, 100.0)
        assert Cxy.shape == (nfft // 2 + 1, n_x, n_y)
        assert Cxx.shape == (nfft // 2 + 1, n_x, n_x)
        assert Cyy.shape == (nfft // 2 + 1, n_y, n_y)

    def test_hermitian_symmetry(self):
        """Auto-spectral matrix should be Hermitian: Cxy[f,i,j] = conj(Cxy[f,j,i])."""
        rng = np.random.default_rng(99)
        n = 4096
        nfft = 256
        x = rng.standard_normal((n, 3))
        Cxy, _, _, _ = csd_matrix(x, None, nfft, 100.0)
        for f in range(Cxy.shape[0]):
            np.testing.assert_allclose(
                Cxy[f], np.conj(Cxy[f].T), atol=1e-12,
            )

    def test_too_short_raises(self):
        x = np.ones((100, 2))
        with pytest.raises(ValueError, match="at least 2\\*nfft"):
            csd_matrix(x, None, 256, 100.0)


class TestCsdMatrixBatch:
    """Tests for batched cross-spectral density."""

    def test_matches_single_window(self):
        """Batch with 1 window should match csd_matrix."""
        rng = np.random.default_rng(111)
        nfft = 128
        diss_len = 512
        n_x = 2
        fs = 100.0

        data = rng.standard_normal((diss_len, n_x))
        x_win = data[np.newaxis, :, :]  # (1, diss_len, n_x)

        Cbatch, F_batch, _, _ = csd_matrix_batch(x_win, None, nfft, fs)
        Csingle, F_single, _, _ = csd_matrix(data, None, nfft, fs)

        assert Cbatch.shape == (1,) + Csingle.shape
        np.testing.assert_allclose(Cbatch[0], Csingle, rtol=1e-10)
        np.testing.assert_allclose(F_batch, F_single)

    def test_batch_shape(self):
        rng = np.random.default_rng(222)
        n_win = 5
        nfft = 128
        diss_len = 512
        n_x, n_y = 2, 3
        fs = 100.0

        x_win = rng.standard_normal((n_win, diss_len, n_x))
        y_win = rng.standard_normal((n_win, diss_len, n_y))

        Cxy, F, Cxx, Cyy = csd_matrix_batch(x_win, y_win, nfft, fs)
        n_freq = nfft // 2 + 1
        assert Cxy.shape == (n_win, n_freq, n_x, n_y)
        assert Cxx.shape == (n_win, n_freq, n_x, n_x)
        assert Cyy.shape == (n_win, n_freq, n_y, n_y)

    def test_auto_batch_shape(self):
        rng = np.random.default_rng(333)
        n_win = 3
        nfft = 128
        diss_len = 512
        n_x = 2
        fs = 100.0

        x_win = rng.standard_normal((n_win, diss_len, n_x))
        Cxy, F, Cxx, Cyy = csd_matrix_batch(x_win, None, nfft, fs)
        n_freq = nfft // 2 + 1
        assert Cxy.shape == (n_win, n_freq, n_x, n_x)
        assert Cxx is None
        assert Cyy is None

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="3-D"):
            csd_matrix_batch(np.ones((10, 2)), None, 8, 100.0)

    def test_too_short_diss_raises(self):
        with pytest.raises(ValueError, match="too short"):
            csd_matrix_batch(np.ones((2, 10, 1)), None, 64, 100.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="incompatible"):
            csd_matrix_batch(
                np.ones((3, 512, 2)),
                np.ones((4, 512, 2)),
                128, 100.0,
            )
