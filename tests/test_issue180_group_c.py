# Sep-2026, Claude and Pat Welch, pat@mousebrains.com
"""Negative controls for issue #180 group C: the chi estimator's own integrals.

F03 has two halves and they are tested separately:

  (a) the non-detection gate had uncontrolled window-level false positives --
      554 of 1000 pure-noise windows produced a finite chi, 393 of them inside
      the two-sided FOM band, which cannot catch it because the model amplitude
      is fitted to the same observations;
  (b) ``np.trapezoid`` over a non-contiguous SELECTION integrates the subset
      grid, bridging every excluded interval with a straight line.

The Monte-Carlo test MEASURES the achieved false-positive rate rather than
asserting a derived one: Welch bins from overlapped segments are correlated, so
no clean analytic rate exists (see chi.detection_floor).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from odas_tpw.chi.chi import (
    _band_slice,
    _below_detection,
    _chi_from_epsilon,
    detection_floor,
)
from odas_tpw.chi.fp07 import fp07_transfer
from odas_tpw.scor160.spectral import csd_matrix_batch

FS, SPEED, NFFT, NSAMP = 512.0, 0.5, 1024, 4096
DOF = 1.9 * 7  # seven overlapping FFTs, nothing removed by Goodman


def _noise_setup():
    K = np.fft.rfftfreq(NFFT, 1 / FS) / SPEED
    noise = np.full(K.size, 1e-6)
    H = fp07_transfer(K * SPEED, 0.01)
    return K, noise, H


class TestF03bBridgedIntegration:
    """Integrating the subset grid counts unobserved area as measured."""

    def test_band_slice_spans_first_to_last(self):
        mask = np.array([False, True, False, False, True, False])
        assert _band_slice(mask) == slice(1, 5)

    def test_empty_mask_is_an_empty_band(self):
        assert _band_slice(np.zeros(5, dtype=bool)) == slice(0, 0)

    def test_reviewers_isolated_peaks_reproduce_the_53x_bridge(self):
        """The reviewer's exact construction, pinned as the thing being fixed."""
        K = np.fft.rfftfreq(NFFT, 1 / FS) / SPEED
        noise = np.full(K.size, 1e-6)
        s = noise.copy()
        peaks = np.array([1, 80, 160])
        s[peaks] = 3e-6
        excess = s - noise
        subset = np.trapezoid(excess[peaks], K[peaks])
        assert subset / np.trapezoid(excess, K) == pytest.approx(53.0, rel=0.02)

    def test_isolated_peaks_are_not_bridged(self):
        """Away from the array edges the contiguous band recovers the TRUE area
        exactly, while the subset grid over-counts by more than an order of
        magnitude."""
        K = np.fft.rfftfreq(NFFT, 1 / FS) / SPEED
        noise = np.full(K.size, 1e-6)
        s = noise.copy()
        peaks = np.array([20, 80, 160])
        s[peaks] = 3e-6
        excess = s - noise
        truth = np.trapezoid(excess, K)
        # Selection spans [19, 161] so every excess triangle lies strictly
        # INSIDE the band; a peak sitting on a band edge legitimately loses its
        # outer half, which is trapezoid behaviour and not the defect.
        selected = np.isin(np.arange(K.size), np.concatenate([peaks, [19, 161]]))
        subset = np.trapezoid(excess[selected], K[selected])
        band = _band_slice(selected)
        contiguous = np.trapezoid(excess[band], K[band])
        assert contiguous == pytest.approx(truth, rel=1e-12)
        assert subset / truth > 20.0

    def test_contiguous_selection_is_unchanged(self):
        """The fix must be a no-op when the selection has no interior hole."""
        K = np.linspace(0.0, 100.0, 401)
        y = np.exp(-((K - 40.0) ** 2) / 200.0)
        mask = (K >= 20.0) & (K <= 60.0)
        band = _band_slice(mask)
        assert np.trapezoid(y[band], K[band]) == pytest.approx(
            np.trapezoid(y[mask], K[mask]), rel=1e-12
        )


class TestF03aDetectionFloor:
    """The floor must scale with the searched bandwidth and the window's DOF."""

    def test_unknown_dof_keeps_the_old_fixed_floor(self):
        assert detection_floor(500, 0.0) == 3
        assert detection_floor(500, 0.0, min_points=6) == 6

    def test_floor_grows_with_bandwidth(self):
        assert detection_floor(400, DOF) > detection_floor(100, DOF)

    def test_floor_grows_as_dof_falls(self):
        """Fewer FFTs -> looser bins -> more of them clear 2x noise by chance."""
        assert detection_floor(200, 1.9 * 3) > detection_floor(200, 1.9 * 20)

    def test_below_detection_uses_the_floor(self):
        K, noise, _ = _noise_setup()
        spec = noise.copy()
        spec[10:14] = 5e-6  # 4 bins above 2x noise: passes 3, fails the dof floor
        K_AA = 88.2 / SPEED
        assert not _below_detection(spec, noise, K, K_AA)
        assert _below_detection(spec, noise, K, K_AA, dof=DOF)

    def test_noise_only_false_positive_rate_is_measured(self):
        """1000 pure-noise windows with the correct floor supplied and zero
        thermal signal. Pre-#180: 554 finite. This pins the MEASURED rate; the
        bound is loose because overlapped Welch bins are correlated and the
        floor is calibrated, not derived."""
        K, noise, H = _noise_setup()
        rng = np.random.default_rng(414)
        n_finite = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(10):
                x = rng.normal(
                    0, np.sqrt(noise[1] * FS / (2 * SPEED)), size=(100, NSAMP, 1)
                )
                spectra = (
                    csd_matrix_batch(x, None, NFFT, FS, detrend="linear").Cxy[:, :, 0, 0].real
                    * SPEED
                )
                for sp in spectra:
                    fit = _chi_from_epsilon(
                        sp, K, 1e-7, 1e-6, noise, H, 0.01, fp07_transfer, 88.2, SPEED,
                        "kraichnan", dof=DOF,
                    )
                    n_finite += bool(np.isfinite(fit.chi))
        assert n_finite <= 30, f"noise-only false positives {n_finite}/1000 (was 554)"

    def test_a_real_signal_still_detects(self):
        """The floor must not simply reject everything."""
        K, noise, H = _noise_setup()
        rng = np.random.default_rng(99)
        spec = noise * (1.0 + 30.0 * np.exp(-((K - 20.0) ** 2) / 400.0))
        spec = spec * rng.chisquare(DOF, size=K.size) / DOF
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = _chi_from_epsilon(
                spec, K, 1e-7, 1e-6, noise, H, 0.01, fp07_transfer, 88.2, SPEED,
                "kraichnan", dof=DOF,
            )
        assert np.isfinite(fit.chi) and fit.chi > 0
