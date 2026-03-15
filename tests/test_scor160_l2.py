# Tests for odas_tpw.scor160.l2
"""Unit tests for L1→L2 processing: section selection, HP filter, despike."""

import numpy as np
import pytest

from odas_tpw.scor160.io import L1Data, L2Data, L2Params
from odas_tpw.scor160.l2 import process_l2


def _make_l1(
    n_time=30000,
    n_shear=2,
    n_vib=2,
    fs=512.0,
    p_start=5.0,
    p_end=60.0,
    profile_dir="down",
    add_speed=False,
):
    """Create a synthetic L1Data with a realistic-looking downward profile."""
    rng = np.random.default_rng(42)
    time = np.arange(n_time) / fs / 86400  # days

    # Pressure ramp (simulating descent)
    pres = np.linspace(p_start, p_end, n_time)

    # Shear: small random signal + a couple of spikes
    shear = rng.standard_normal((n_shear, n_time)) * 0.05

    # Vibration: small signal
    vib = rng.standard_normal((n_vib, n_time)) * 0.01

    l1 = L1Data(
        time=time,
        pres=pres,
        shear=shear,
        vib=vib,
        vib_type="ACC",
        fs_fast=fs,
        f_AA=98.0,
        vehicle="vmp",
        profile_dir=profile_dir,
        time_reference_year=2024,
    )
    if add_speed:
        l1.pspd_rel = np.full(n_time, 0.6)
    return l1


def _make_params():
    return L2Params(
        HP_cut=0.25,
        despike_sh=np.array([8.0, 0.5, 0.04]),
        despike_A=np.array([np.inf, 0.5, 0.04]),
        profile_min_W=0.05,
        profile_min_P=2.0,
        profile_min_duration=5.0,
        speed_tau=1.5,
    )


class TestProcessL2:
    """Tests for the main process_l2 function."""

    def test_output_type(self):
        l1 = _make_l1()
        params = _make_params()
        l2 = process_l2(l1, params)
        assert isinstance(l2, L2Data)

    def test_output_shapes(self):
        n = 30000
        l1 = _make_l1(n_time=n)
        params = _make_params()
        l2 = process_l2(l1, params)
        assert l2.time.shape == (n,)
        assert l2.shear.shape == (2, n)
        assert l2.vib.shape == (2, n)
        assert l2.pspd_rel.shape == (n,)
        assert l2.section_number.shape == (n,)

    def test_sections_found(self):
        """With a smooth descent, at least one section should be found."""
        l1 = _make_l1(p_start=5, p_end=60)
        params = _make_params()
        l2 = process_l2(l1, params)
        assert l2.section_number.max() >= 1

    def test_speed_positive_downward(self):
        """For a downward profiler, speed within sections should be positive."""
        l1 = _make_l1(profile_dir="down")
        params = _make_params()
        l2 = process_l2(l1, params)
        in_section = l2.section_number > 0
        if in_section.any():
            assert np.mean(l2.pspd_rel[in_section]) > 0

    def test_precomputed_speed_used(self):
        """When L1 has pspd_rel, it should be used instead of computing from pressure."""
        l1 = _make_l1(add_speed=True)
        params = _make_params()
        l2 = process_l2(l1, params)
        # Speed should be close to the pre-set value (after LP filtering)
        in_section = l2.section_number > 0
        if in_section.any():
            mean_spd = np.mean(l2.pspd_rel[in_section])
            assert mean_spd == pytest.approx(0.6, abs=0.1)

    def test_hp_filter_removes_dc(self):
        """HP filter should remove DC offset from shear."""
        l1 = _make_l1()
        # Add a large DC offset to shear
        l1.shear += 10.0
        params = _make_params()
        l2 = process_l2(l1, params)
        # After HP filtering, mean should be near zero
        in_section = l2.section_number > 0
        if in_section.any():
            for i in range(l1.n_shear):
                assert abs(np.mean(l2.shear[i, in_section])) < 1.0

    def test_time_preserved(self):
        l1 = _make_l1()
        params = _make_params()
        l2 = process_l2(l1, params)
        np.testing.assert_array_equal(l2.time, l1.time)

    def test_vib_type_preserved(self):
        l1 = _make_l1()
        params = _make_params()
        l2 = process_l2(l1, params)
        assert l2.vib_type == "ACC"


class TestSectionSelection:
    """Test section selection criteria."""

    def test_no_sections_shallow(self):
        """If all pressure is below min_pressure, no sections selected."""
        l1 = _make_l1(p_start=0.0, p_end=0.5)
        params = _make_params()
        params.profile_min_P = 2.0
        l2 = process_l2(l1, params)
        assert l2.section_number.max() == 0

    def test_no_sections_short_duration(self):
        """Very short record shouldn't produce sections."""
        l1 = _make_l1(n_time=100)
        params = _make_params()
        params.profile_min_duration = 10.0
        l2 = process_l2(l1, params)
        assert l2.section_number.max() == 0


class TestDespiking:
    """Test that despiking modifies spiked signals."""

    def test_spike_removed(self):
        l1 = _make_l1()
        # Add a large spike in the middle of the record
        l1.shear[0, 15000] = 100.0
        params = _make_params()
        l2 = process_l2(l1, params)
        # After despiking, the spike should be substantially reduced
        assert abs(l2.shear[0, 15000]) < 50.0

    def test_inf_threshold_skips_despike(self):
        """Setting threshold to inf should skip despiking."""
        l1 = _make_l1()
        l1.shear[0, 15000] = 100.0
        params = _make_params()
        params.despike_sh = np.array([np.inf, 0.5, 0.04])
        l2 = process_l2(l1, params)
        # HP filter will change the value but it shouldn't be despiked
        # Just verify it runs without error
        assert l2.shear.shape == l1.shear.shape


class TestNoVibration:
    """Test with empty vibration data."""

    def test_no_vib_channels(self):
        l1 = _make_l1(n_vib=0)
        l1.vib = np.zeros((0, l1.n_time))
        params = _make_params()
        l2 = process_l2(l1, params)
        assert l2.vib.shape[0] == 0
