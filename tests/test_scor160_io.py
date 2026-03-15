# Tests for odas_tpw.scor160.io
"""Unit tests for dataclasses and I/O utilities."""

import numpy as np

from odas_tpw.scor160.io import (
    L1Data,
    L2Data,
    L2Params,
    L3Data,
    L3Params,
    L4Data,
    L4Params,
)


def _make_l1(n_time=10000, n_shear=2, n_vib=2, fs=512.0):
    """Create a minimal L1Data for testing."""
    return L1Data(
        time=np.linspace(0, n_time / fs / 86400, n_time),
        pres=np.linspace(5, 100, n_time),
        shear=np.zeros((n_shear, n_time)),
        vib=np.zeros((n_vib, n_time)),
        vib_type="ACC",
        fs_fast=fs,
        f_AA=98.0,
        vehicle="vmp",
        profile_dir="down",
        time_reference_year=2024,
    )


class TestL1Data:
    def test_properties(self):
        l1 = _make_l1()
        assert l1.n_shear == 2
        assert l1.n_vib == 2
        assert l1.n_time == 10000

    def test_has_speed_empty(self):
        l1 = _make_l1()
        assert not l1.has_speed

    def test_has_speed_nonempty(self):
        l1 = _make_l1()
        l1.pspd_rel = np.ones(l1.n_time)
        assert l1.has_speed

    def test_default_optional_arrays(self):
        l1 = _make_l1()
        assert l1.time_slow.size == 0
        assert l1.pres_slow.size == 0
        assert l1.pitch.size == 0
        assert l1.roll.size == 0
        assert l1.temp.size == 0
        assert l1.fs_slow == 0.0


class TestL2Params:
    def test_construction(self):
        params = L2Params(
            HP_cut=0.25,
            despike_sh=np.array([8.0, 0.5, 0.04]),
            despike_A=np.array([np.inf, 0.5, 0.04]),
            profile_min_W=0.1,
            profile_min_P=1.0,
            profile_min_duration=10.0,
            speed_tau=1.5,
        )
        assert params.HP_cut == 0.25
        assert params.speed_tau == 1.5


class TestL2Data:
    def test_construction(self):
        n = 1000
        l2 = L2Data(
            time=np.zeros(n),
            shear=np.zeros((2, n)),
            vib=np.zeros((2, n)),
            vib_type="ACC",
            pspd_rel=np.ones(n),
            section_number=np.ones(n),
        )
        assert l2.shear.shape == (2, n)


class TestL3Params:
    def test_construction(self):
        params = L3Params(
            fft_length=256,
            diss_length=2048,
            overlap=1024,
            HP_cut=0.25,
            fs_fast=512.0,
            goodman=True,
        )
        assert params.fft_length == 256
        assert params.goodman is True


class TestL3Data:
    def test_properties(self):
        n_wn, n_spec, n_sh = 129, 20, 2
        l3 = L3Data(
            time=np.zeros(n_spec),
            pres=np.zeros(n_spec),
            temp=np.zeros(n_spec),
            pspd_rel=np.ones(n_spec),
            section_number=np.ones(n_spec),
            kcyc=np.zeros((n_wn, n_spec)),
            sh_spec=np.zeros((n_sh, n_wn, n_spec)),
            sh_spec_clean=np.zeros((n_sh, n_wn, n_spec)),
        )
        assert l3.n_spectra == n_spec
        assert l3.n_wavenumber == n_wn
        assert l3.n_shear == n_sh


class TestL4Params:
    def test_construction(self):
        params = L4Params(
            fft_length=256,
            diss_length=2048,
            overlap=1024,
            fs_fast=512.0,
            fit_order=3,
            f_AA=98.0,
            FOM_limit=1.15,
            variance_resolved_limit=0.5,
        )
        assert params.fit_order == 3


class TestL4Data:
    def test_properties(self):
        n_sh, n_spec = 2, 30
        l4 = L4Data(
            time=np.zeros(n_spec),
            pres=np.zeros(n_spec),
            pspd_rel=np.ones(n_spec),
            section_number=np.ones(n_spec),
            epsi=np.full((n_sh, n_spec), 1e-8),
            epsi_final=np.full(n_spec, 1e-8),
            epsi_flags=np.zeros((n_sh, n_spec)),
            fom=np.ones((n_sh, n_spec)),
            mad=np.full((n_sh, n_spec), 0.1),
            kmax=np.full((n_sh, n_spec), 50.0),
            method=np.zeros((n_sh, n_spec)),
            var_resolved=np.ones((n_sh, n_spec)),
        )
        assert l4.n_spectra == n_spec
        assert l4.n_shear == n_sh
