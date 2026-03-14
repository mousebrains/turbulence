# Tests for odas_tpw.scor160.l3
"""Unit tests for L2→L3 spectral processing."""

import numpy as np
import pytest

from odas_tpw.scor160.io import L1Data, L2Data, L3Data, L3Params
from odas_tpw.scor160.l3 import process_l3


def _make_l1_l2(
    n_time=20000,
    n_shear=2,
    n_vib=2,
    fs=512.0,
):
    """Create paired L1 and L2 data for L3 processing tests."""
    rng = np.random.default_rng(42)
    time = np.arange(n_time) / fs / 86400

    l1 = L1Data(
        time=time,
        pres=np.linspace(10, 50, n_time),
        shear=rng.standard_normal((n_shear, n_time)) * 0.05,
        vib=rng.standard_normal((n_vib, n_time)) * 0.01,
        vib_type="ACC",
        fs_fast=fs,
        f_AA=98.0,
        vehicle="vmp",
        profile_dir="down",
        time_reference_year=2024,
        temp=np.full(n_time, 10.0),
    )

    l2 = L2Data(
        time=time,
        shear=rng.standard_normal((n_shear, n_time)) * 0.05,
        vib=rng.standard_normal((n_vib, n_time)) * 0.01,
        vib_type="ACC",
        pspd_rel=np.full(n_time, 0.6),
        section_number=np.ones(n_time),  # all data in section 1
    )

    return l1, l2


def _make_params(fs=512.0):
    return L3Params(
        fft_length=256,
        diss_length=2048,
        overlap=1024,
        HP_cut=0.25,
        fs_fast=fs,
        goodman=True,
    )


class TestProcessL3:
    """Tests for the main process_l3 function."""

    def test_output_type(self):
        l1, l2 = _make_l1_l2()
        params = _make_params()
        l3 = process_l3(l2, l1, params)
        assert isinstance(l3, L3Data)

    def test_output_shapes(self):
        l1, l2 = _make_l1_l2()
        params = _make_params()
        l3 = process_l3(l2, l1, params)

        n_freq = params.fft_length // 2 + 1
        n_spec = l3.n_spectra
        assert n_spec > 0
        assert l3.kcyc.shape == (n_freq, n_spec)
        assert l3.sh_spec.shape == (2, n_freq, n_spec)
        assert l3.sh_spec_clean.shape == (2, n_freq, n_spec)
        assert l3.time.shape == (n_spec,)
        assert l3.pres.shape == (n_spec,)
        assert l3.temp.shape == (n_spec,)
        assert l3.pspd_rel.shape == (n_spec,)
        assert l3.section_number.shape == (n_spec,)

    def test_n_spectra_expected(self):
        """Number of spectra should match expected from windowing."""
        n_time = 20000
        l1, l2 = _make_l1_l2(n_time=n_time)
        params = _make_params()
        l3 = process_l3(l2, l1, params)

        sec_len = n_time
        diss_step = params.diss_length - params.overlap
        expected_n = (sec_len - params.diss_length) // diss_step + 1
        assert l3.n_spectra == expected_n

    def test_wavenumber_grid(self):
        """Wavenumber grid should be f/W where W is the mean speed."""
        l1, l2 = _make_l1_l2()
        params = _make_params()
        l3 = process_l3(l2, l1, params)

        fs = params.fs_fast
        nfft = params.fft_length
        F = np.arange(nfft // 2 + 1) * fs / nfft
        W = l3.pspd_rel[0]
        expected_k = F / W
        np.testing.assert_allclose(l3.kcyc[:, 0], expected_k)

    def test_spectra_positive(self):
        """Shear spectra should be non-negative (auto-spectra)."""
        l1, l2 = _make_l1_l2()
        params = _make_params()
        l3 = process_l3(l2, l1, params)
        assert np.all(l3.sh_spec >= 0)

    def test_section_numbers_propagated(self):
        l1, l2 = _make_l1_l2()
        params = _make_params()
        l3 = process_l3(l2, l1, params)
        assert np.all(l3.section_number == 1)

    def test_pressure_in_range(self):
        l1, l2 = _make_l1_l2()
        params = _make_params()
        l3 = process_l3(l2, l1, params)
        assert np.all(l3.pres >= 10)
        assert np.all(l3.pres <= 50)

    def test_speed_preserved(self):
        l1, l2 = _make_l1_l2()
        params = _make_params()
        l3 = process_l3(l2, l1, params)
        np.testing.assert_allclose(l3.pspd_rel, 0.6, atol=0.01)

    def test_properties(self):
        l1, l2 = _make_l1_l2()
        params = _make_params()
        l3 = process_l3(l2, l1, params)
        assert l3.n_shear == 2
        assert l3.n_wavenumber == 129


class TestProcessL3NoGoodman:
    """Test L3 processing without Goodman cleaning."""

    def test_no_goodman(self):
        l1, l2 = _make_l1_l2()
        params = _make_params()
        params.goodman = False
        l3 = process_l3(l2, l1, params)
        # Without Goodman, clean == raw
        np.testing.assert_array_equal(l3.sh_spec, l3.sh_spec_clean)


class TestProcessL3NoSections:
    """Test with no valid sections."""

    def test_empty_output(self):
        l1, l2 = _make_l1_l2()
        l2.section_number = np.zeros(l2.section_number.shape)  # no sections
        params = _make_params()
        l3 = process_l3(l2, l1, params)
        assert l3.n_spectra == 0
        assert l3.time.shape == (0,)

    def test_short_section(self):
        """Section shorter than diss_length should produce no spectra."""
        l1, l2 = _make_l1_l2(n_time=1000)
        params = _make_params()
        params.diss_length = 2048
        l3 = process_l3(l2, l1, params)
        assert l3.n_spectra == 0


class TestProcessL3NoVib:
    """Test L3 with no vibration channels."""

    def test_no_vib(self):
        l1, l2 = _make_l1_l2(n_vib=0)
        l1.vib = np.zeros((0, l1.n_time))
        l2.vib = np.zeros((0, l2.time.shape[0]))
        params = _make_params()
        l3 = process_l3(l2, l1, params)
        # Without vibration, clean == raw
        np.testing.assert_array_equal(l3.sh_spec, l3.sh_spec_clean)
