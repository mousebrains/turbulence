# Tests for odas_tpw.scor160.l4
"""Unit tests for L3→L4 dissipation estimation."""

import numpy as np
import pytest

from odas_tpw.scor160.io import L3Data, L4Data
from odas_tpw.scor160.l4 import (
    DEFAULT_FOM_LIMIT,
    DEFAULT_VAR_RESOLVED_LIMIT,
    E_ISR_THRESHOLD,
    EPSILON_FLOOR,
    ISOTROPY_FACTOR,
    K_LIMIT_MAX,
    _compute_epsi_final,
    _compute_flags,
    _variance_correction,
    _variance_resolved_fraction,
    process_l4,
)
from odas_tpw.scor160.nasmyth import nasmyth


def _make_nasmyth_spectrum(
    epsilon, nu, speed=0.6, nfft=256, fs=512.0, noise=0.0, rng=None,
):
    """Create a synthetic Nasmyth-shaped shear wavenumber spectrum.

    Returns (K, spec) arrays shaped like scor160 convention.
    """
    F = np.arange(nfft // 2 + 1) * fs / nfft
    K = F / speed
    phi = nasmyth(epsilon, nu, K + 1e-30)  # add tiny offset for K=0
    phi[0] = 0.0  # DC bin
    if noise > 0 and rng is not None:
        phi = phi * (1 + noise * rng.standard_normal(len(phi)))
        phi = np.maximum(phi, 0)
    return K, phi


def _make_l3_from_nasmyth(
    n_spec=10, n_shear=2, epsilon=1e-8, nu=1.3e-6,
    speed=0.6, nfft=256, fs=512.0, noise=0.0,
):
    """Create a synthetic L3Data with Nasmyth-shaped spectra."""
    rng = np.random.default_rng(42)
    n_freq = nfft // 2 + 1

    kcyc = np.zeros((n_freq, n_spec))
    sh_spec = np.zeros((n_shear, n_freq, n_spec))
    sh_spec_clean = np.zeros((n_shear, n_freq, n_spec))

    for j in range(n_spec):
        K, phi = _make_nasmyth_spectrum(epsilon, nu, speed, nfft, fs, noise, rng)
        kcyc[:, j] = K
        for i in range(n_shear):
            sh_spec[i, :, j] = phi
            sh_spec_clean[i, :, j] = phi

    return L3Data(
        time=np.linspace(0, 1, n_spec),
        pres=np.linspace(10, 50, n_spec),
        temp=np.full(n_spec, 10.0),
        pspd_rel=np.full(n_spec, speed),
        section_number=np.ones(n_spec),
        kcyc=kcyc,
        sh_spec=sh_spec,
        sh_spec_clean=sh_spec_clean,
    )


class TestProcessL4:
    """Tests for the main process_l4 function."""

    def test_output_type(self):
        l3 = _make_l3_from_nasmyth()
        l4 = process_l4(l3)
        assert isinstance(l4, L4Data)

    def test_output_shapes(self):
        n_spec, n_sh = 10, 2
        l3 = _make_l3_from_nasmyth(n_spec=n_spec, n_shear=n_sh)
        l4 = process_l4(l3)
        assert l4.epsi.shape == (n_sh, n_spec)
        assert l4.epsi_final.shape == (n_spec,)
        assert l4.epsi_flags.shape == (n_sh, n_spec)
        assert l4.fom.shape == (n_sh, n_spec)
        assert l4.mad.shape == (n_sh, n_spec)
        assert l4.kmax.shape == (n_sh, n_spec)
        assert l4.method.shape == (n_sh, n_spec)
        assert l4.var_resolved.shape == (n_sh, n_spec)

    def test_properties(self):
        l3 = _make_l3_from_nasmyth(n_spec=5, n_shear=2)
        l4 = process_l4(l3)
        assert l4.n_spectra == 5
        assert l4.n_shear == 2

    def test_nasmyth_recovery_low_epsilon(self):
        """For a pure Nasmyth spectrum, recovered epsilon should be close to input."""
        eps_in = 1e-9
        nu = 1.3e-6
        l3 = _make_l3_from_nasmyth(n_spec=5, epsilon=eps_in, nu=nu)
        l4 = process_l4(l3, temp=np.full(5, 10.0))

        for i in range(l4.n_shear):
            for j in range(l4.n_spectra):
                ratio = l4.epsi[i, j] / eps_in
                # Should recover within a factor of 2
                assert 0.5 < ratio < 2.0, f"probe {i}, spec {j}: ratio={ratio:.3f}"

    def test_nasmyth_recovery_high_epsilon(self):
        """High epsilon should use ISR method and still recover accurately."""
        eps_in = 1e-4
        nu = 1.3e-6
        l3 = _make_l3_from_nasmyth(n_spec=5, epsilon=eps_in, nu=nu)
        l4 = process_l4(l3, temp=np.full(5, 10.0))

        for i in range(l4.n_shear):
            for j in range(l4.n_spectra):
                ratio = l4.epsi[i, j] / eps_in
                assert 0.3 < ratio < 3.0

    def test_variance_method_used_low_eps(self):
        """Low epsilon should use variance method (method=0)."""
        l3 = _make_l3_from_nasmyth(epsilon=1e-10)
        l4 = process_l4(l3)
        assert np.all(l4.method == 0)

    def test_isr_method_used_high_eps(self):
        """High epsilon should use ISR method (method=1)."""
        l3 = _make_l3_from_nasmyth(epsilon=1e-3)
        l4 = process_l4(l3)
        assert np.all(l4.method == 1)

    def test_fom_near_one_for_nasmyth(self):
        """FOM should be near 1.0 for a clean Nasmyth spectrum."""
        l3 = _make_l3_from_nasmyth(epsilon=1e-8)
        l4 = process_l4(l3)
        for i in range(l4.n_shear):
            for j in range(l4.n_spectra):
                if np.isfinite(l4.fom[i, j]):
                    assert 0.5 < l4.fom[i, j] < 2.0

    def test_empty_l3(self):
        """Empty L3 should produce empty L4."""
        l3 = _make_l3_from_nasmyth(n_spec=0)
        l4 = process_l4(l3)
        assert l4.n_spectra == 0
        assert l4.epsi.shape == (2, 0)

    def test_metadata_propagated(self):
        l3 = _make_l3_from_nasmyth()
        l4 = process_l4(l3)
        np.testing.assert_array_equal(l4.time, l3.time)
        np.testing.assert_array_equal(l4.pres, l3.pres)
        np.testing.assert_array_equal(l4.section_number, l3.section_number)


class TestComputeFlags:
    """Tests for QC flag computation."""

    def test_all_good(self):
        epsi = np.full((2, 5), 1e-8)
        fom = np.full((2, 5), 1.0)
        var_res = np.full((2, 5), 0.8)
        flags = _compute_flags(epsi, fom, var_res, 1.15, 0.5)
        assert np.all(flags == 0)

    def test_fom_flag(self):
        epsi = np.full((2, 5), 1e-8)
        fom = np.full((2, 5), 1.5)  # exceeds 1.15
        var_res = np.full((2, 5), 0.8)
        flags = _compute_flags(epsi, fom, var_res, 1.15, 0.5)
        assert np.all(flags == 1)

    def test_var_resolved_flag(self):
        epsi = np.full((2, 5), 1e-8)
        fom = np.full((2, 5), 1.0)
        var_res = np.full((2, 5), 0.3)  # below 0.5
        flags = _compute_flags(epsi, fom, var_res, 1.15, 0.5)
        assert np.all(flags == 16)

    def test_both_flags(self):
        epsi = np.full((2, 5), 1e-8)
        fom = np.full((2, 5), 1.5)
        var_res = np.full((2, 5), 0.3)
        flags = _compute_flags(epsi, fom, var_res, 1.15, 0.5)
        assert np.all(flags == 17)  # 1 + 16

    def test_nan_epsilon(self):
        epsi = np.full((2, 5), np.nan)
        fom = np.full((2, 5), 1.0)
        var_res = np.full((2, 5), 0.8)
        flags = _compute_flags(epsi, fom, var_res, 1.15, 0.5)
        assert np.all(flags == 255)


class TestComputeEpsiFinal:
    """Tests for EPSI_FINAL computation."""

    def test_geometric_mean_good_probes(self):
        epsi = np.array([[1e-8, 1e-7], [1e-6, 1e-5]])
        flags = np.zeros((2, 2))
        final = _compute_epsi_final(epsi, flags)
        # Geometric mean of columns
        expected_0 = np.exp(np.mean(np.log([1e-8, 1e-6])))
        expected_1 = np.exp(np.mean(np.log([1e-7, 1e-5])))
        assert final[0] == pytest.approx(expected_0, rel=1e-10)
        assert final[1] == pytest.approx(expected_1, rel=1e-10)

    def test_one_flagged_probe(self):
        """If one probe is flagged, use only the good probe."""
        epsi = np.array([[1e-8], [1e-6]])
        flags = np.array([[0.0], [1.0]])  # probe 1 flagged
        final = _compute_epsi_final(epsi, flags)
        assert final[0] == pytest.approx(1e-8)

    def test_all_flagged_fallback(self):
        """If all probes flagged, fall back to geometric mean of all finite."""
        epsi = np.array([[1e-8], [1e-6]])
        flags = np.array([[1.0], [1.0]])
        final = _compute_epsi_final(epsi, flags)
        expected = np.exp(np.mean(np.log([1e-8, 1e-6])))
        assert final[0] == pytest.approx(expected, rel=1e-10)


class TestVarianceCorrection:
    """Tests for iterative variance correction."""

    def test_correction_increases_epsilon(self):
        """Variance correction should increase epsilon (compensating for truncation)."""
        e3 = 1e-8
        K_upper = 50.0
        nu = 1.3e-6
        e4 = _variance_correction(e3, K_upper, nu)
        assert e4 >= e3

    def test_well_resolved_no_change(self):
        """For very high K_upper (well resolved), correction should be small."""
        e3 = 1e-8
        nu = 1.3e-6
        ks = (e3 / nu**3) ** 0.25
        K_upper = ks  # up to Kolmogorov scale
        e4 = _variance_correction(e3, K_upper, nu)
        assert e4 == pytest.approx(e3, rel=0.1)

    def test_convergence(self):
        """Should converge (not oscillate or diverge)."""
        e3 = 1e-7
        K_upper = 30.0
        nu = 1.3e-6
        e4 = _variance_correction(e3, K_upper, nu)
        assert np.isfinite(e4)
        assert e4 > 0


class TestVarianceResolvedFraction:
    """Tests for variance resolved fraction."""

    def test_high_kmax_high_fraction(self):
        """High K_max → most variance resolved."""
        eps = 1e-8
        nu = 1.3e-6
        ks = (eps / nu**3) ** 0.25
        vr = _variance_resolved_fraction(ks, eps, nu)
        assert vr > 0.9

    def test_low_kmax_low_fraction(self):
        """Very low K_max → little variance resolved."""
        vr = _variance_resolved_fraction(5.0, 1e-8, 1.3e-6)
        assert vr < 0.5

    def test_bounded_zero_one(self):
        """Fraction should always be in [0, 1]."""
        for eps in [1e-12, 1e-8, 1e-4]:
            for K_upper in [1, 10, 50, 200]:
                vr = _variance_resolved_fraction(K_upper, eps, 1.3e-6)
                assert 0 <= vr <= 1


class TestConstants:
    """Verify key constants are set correctly."""

    def test_epsilon_floor(self):
        assert EPSILON_FLOOR == 1e-15

    def test_isr_threshold(self):
        assert E_ISR_THRESHOLD == 1.5e-5

    def test_isotropy_factor(self):
        assert ISOTROPY_FACTOR == 7.5

    def test_k_limit_max(self):
        assert K_LIMIT_MAX == 150

    def test_default_fom_limit(self):
        assert DEFAULT_FOM_LIMIT == 1.15

    def test_default_var_resolved_limit(self):
        assert DEFAULT_VAR_RESOLVED_LIMIT == 0.5
