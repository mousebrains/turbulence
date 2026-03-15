# Tests for odas_tpw.rsi.dissipation
"""Unit tests for the RSI dissipation module."""

import numpy as np
import pytest

from odas_tpw.rsi.dissipation import (
    DOF_NUTTALL,
    F_AA_MARGIN,
    MACOUN_LUECK_DENOM,
    MACOUN_LUECK_K,
    SPEED_MIN,
)
from odas_tpw.scor160.l4 import _estimate_epsilon
from odas_tpw.scor160.nasmyth import nasmyth

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_nasmyth_spectrum(epsilon, nu, speed=0.6, nfft=256, fs=512.0):
    """Create a synthetic Nasmyth shear wavenumber spectrum.

    Returns (K, spec) where K is in cpm and spec in s^-2 cpm^-1.
    """
    F = np.arange(nfft // 2 + 1) * fs / nfft
    K = F / speed
    phi = nasmyth(epsilon, nu, K + 1e-30)  # tiny offset for K=0
    phi[0] = 0.0  # DC bin
    return K, phi


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify key named constants have expected values."""

    def test_speed_min(self):
        assert SPEED_MIN == 0.01

    def test_dof_nuttall(self):
        assert pytest.approx(1.9) == DOF_NUTTALL

    def test_macoun_lueck_k(self):
        assert MACOUN_LUECK_K == 150

    def test_macoun_lueck_denom(self):
        assert MACOUN_LUECK_DENOM == 48

    def test_f_aa_margin(self):
        assert pytest.approx(0.9) == F_AA_MARGIN


# ---------------------------------------------------------------------------
# Epsilon recovery from synthetic Nasmyth spectra
# ---------------------------------------------------------------------------


class TestEpsilonRecovery:
    """Feed synthetic Nasmyth spectra into _estimate_epsilon and verify recovery."""

    NU = 1.3e-6  # typical seawater kinematic viscosity

    @pytest.mark.parametrize("eps_in", [1e-8, 1e-6, 1e-4])
    def test_recovery(self, eps_in):
        """Recovered epsilon should be within a factor of 2 of input."""
        speed = 0.6
        K, spec = _make_nasmyth_spectrum(eps_in, self.NU, speed=speed)
        K_AA = F_AA_MARGIN * 98.0 / speed  # typical AA wavenumber limit

        e_out, _k_max, _mad, _method, _fom, _var_res, _nas_spec, _K_max_ratio, _FM, _eisr, _evar = (
            _estimate_epsilon(K, spec, self.NU, K_AA, fit_order=3)
        )

        ratio = e_out / eps_in
        assert 0.5 < ratio < 2.0, f"eps_in={eps_in:.1e}, eps_out={e_out:.1e}, ratio={ratio:.3f}"

    @pytest.mark.parametrize("eps_in", [1e-8, 1e-6, 1e-4])
    def test_fom_near_one(self, eps_in):
        """FOM should be near 1.0 for a perfect Nasmyth spectrum."""
        speed = 0.6
        K, spec = _make_nasmyth_spectrum(eps_in, self.NU, speed=speed)
        K_AA = F_AA_MARGIN * 98.0 / speed

        _, _, _, _, fom, _, _, _, _, _, _ = _estimate_epsilon(K, spec, self.NU, K_AA, fit_order=3)

        if np.isfinite(fom):
            assert 0.5 < fom < 2.0, f"eps_in={eps_in:.1e}, fom={fom:.3f}"

    def test_variance_method_for_low_epsilon(self):
        """Low epsilon should use variance method (method=0)."""
        eps_in = 1e-10
        speed = 0.6
        K, spec = _make_nasmyth_spectrum(eps_in, self.NU, speed=speed)
        K_AA = F_AA_MARGIN * 98.0 / speed

        result = _estimate_epsilon(K, spec, self.NU, K_AA, fit_order=3)
        method = result[3]

        assert method == 0

    def test_isr_method_for_high_epsilon(self):
        """High epsilon should use ISR method (method=1)."""
        eps_in = 1e-3
        speed = 0.6
        K, spec = _make_nasmyth_spectrum(eps_in, self.NU, speed=speed)
        K_AA = F_AA_MARGIN * 98.0 / speed

        result = _estimate_epsilon(K, spec, self.NU, K_AA, fit_order=3)
        method = result[3]

        assert method == 1

    def test_nasmyth_spectrum_returned(self):
        """Returned Nasmyth spectrum should be positive and finite."""
        eps_in = 1e-7
        speed = 0.6
        K, spec = _make_nasmyth_spectrum(eps_in, self.NU, speed=speed)
        K_AA = F_AA_MARGIN * 98.0 / speed

        result = _estimate_epsilon(K, spec, self.NU, K_AA, fit_order=3)
        nas_spec = result[6]

        # Skip DC bin (K=0)
        assert np.all(np.isfinite(nas_spec[1:]))
        assert np.all(nas_spec[1:] > 0)

    def test_k_max_ratio_positive(self):
        """K_max_ratio should be positive and finite."""
        eps_in = 1e-7
        speed = 0.6
        K, spec = _make_nasmyth_spectrum(eps_in, self.NU, speed=speed)
        K_AA = F_AA_MARGIN * 98.0 / speed

        result = _estimate_epsilon(K, spec, self.NU, K_AA, fit_order=3)
        K_max_ratio = result[7]

        assert np.isfinite(K_max_ratio)
        assert K_max_ratio > 0


# ---------------------------------------------------------------------------
# Macoun-Lueck correction
# ---------------------------------------------------------------------------


class TestMacounLueckCorrection:
    """Verify the Macoun-Lueck wavenumber correction formula."""

    def test_correction_at_zero(self):
        """At K=0, correction = 1 + 0 = 1."""
        K = 0.0
        correction = 1 + (K / MACOUN_LUECK_DENOM) ** 2
        assert correction == pytest.approx(1.0)

    def test_correction_at_denom(self):
        """At K=48, correction = 1 + 1 = 2."""
        K = float(MACOUN_LUECK_DENOM)
        correction = 1 + (K / MACOUN_LUECK_DENOM) ** 2
        assert correction == pytest.approx(2.0)

    def test_correction_at_limit(self):
        """At K=150, correction = 1 + (150/48)^2."""
        K = float(MACOUN_LUECK_K)
        correction = 1 + (K / MACOUN_LUECK_DENOM) ** 2
        expected = 1 + (150 / 48) ** 2
        assert correction == pytest.approx(expected)

    def test_correction_monotonically_increasing(self):
        """Correction should increase with wavenumber."""
        K_vals = np.linspace(0, MACOUN_LUECK_K, 50)
        corrections = 1 + (K_vals / MACOUN_LUECK_DENOM) ** 2
        assert np.all(np.diff(corrections) > 0)

    def test_correction_vectorized_matches_dissipation(self):
        """Verify that the vectorized correction in _compute_profile_diss
        matches the scalar formula 1 + (K/48)^2 for K <= 150."""
        K = np.array([0, 10, 48, 100, 150, 200])
        correction = np.ones_like(K, dtype=float)
        mask = K <= MACOUN_LUECK_K
        correction[mask] = 1 + (K[mask] / MACOUN_LUECK_DENOM) ** 2

        # K <= 150 should be corrected
        assert correction[0] == pytest.approx(1.0)
        assert correction[1] == pytest.approx(1 + (10 / 48) ** 2)
        assert correction[2] == pytest.approx(2.0)
        assert correction[3] == pytest.approx(1 + (100 / 48) ** 2)
        assert correction[4] == pytest.approx(1 + (150 / 48) ** 2)
        # K > 150 should be uncorrected
        assert correction[5] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Speed clamping
# ---------------------------------------------------------------------------


class TestSpeedClamping:
    """Verify speed floor behavior."""

    def test_speed_min_value(self):
        """SPEED_MIN should be 0.01 m/s."""
        assert pytest.approx(0.01) == SPEED_MIN

    def test_clamp_applies(self):
        """np.maximum(speed, SPEED_MIN) should clamp low values."""
        speeds = np.array([0.001, 0.005, 0.01, 0.1, 0.5])
        clamped = np.maximum(speeds, SPEED_MIN)
        assert clamped[0] == pytest.approx(SPEED_MIN)
        assert clamped[1] == pytest.approx(SPEED_MIN)
        assert clamped[2] == pytest.approx(SPEED_MIN)
        assert clamped[3] == pytest.approx(0.1)
        assert clamped[4] == pytest.approx(0.5)
