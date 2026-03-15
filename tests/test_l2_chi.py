# Tests for odas_tpw.chi.l2_chi
"""Unit tests for L2 chi processing: despike, HP-filter vibration, temperature gradient."""

import numpy as np
import pytest

from odas_tpw.chi.l2_chi import L2ChiData, L2ChiParams, process_l2_chi
from odas_tpw.scor160.io import L1Data, L2Data


def _make_l1(
    n_time=4096,
    n_temp=2,
    n_vib=2,
    fs=512.0,
    include_temp_fast=True,
    include_temp_slow=True,
    diff_gains=None,
):
    """Create a synthetic L1Data with fast temperature channels."""
    rng = np.random.default_rng(42)
    time = np.arange(n_time) / fs / 86400  # days
    pres = np.linspace(5.0, 50.0, n_time)
    shear = rng.standard_normal((2, n_time)) * 0.05
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
        profile_dir="down",
        time_reference_year=2024,
    )

    if include_temp_fast:
        # Smooth temperature with small noise
        base = np.linspace(15.0, 10.0, n_time)
        temp_fast = np.empty((n_temp, n_time))
        for i in range(n_temp):
            temp_fast[i] = base + rng.standard_normal(n_time) * 0.001
        l1.temp_fast = temp_fast
        l1.diff_gains = diff_gains if diff_gains else [0.94] * n_temp

    if include_temp_slow:
        l1.temp = np.linspace(15.0, 10.0, n_time)

    return l1


def _make_l2(n_time=4096, n_sections=1, speed=0.6):
    """Create a synthetic L2Data with known section numbers."""
    section_number = np.zeros(n_time)
    if n_sections >= 1:
        # Assign bulk of the record to section 1
        sec_start = n_time // 10
        sec_end = n_time * 9 // 10
        section_number[sec_start:sec_end] = 1.0
    if n_sections >= 2:
        mid = (sec_start + sec_end) // 2
        section_number[mid:sec_end] = 2.0

    return L2Data(
        time=np.arange(n_time) / 512.0 / 86400,
        shear=np.zeros((2, n_time)),
        vib=np.zeros((2, n_time)),
        vib_type="ACC",
        pspd_rel=np.full(n_time, speed),
        section_number=section_number,
    )


# ---- L2ChiData dataclass ----


class TestL2ChiData:
    """Test the L2ChiData dataclass creation and properties."""

    def test_fields(self):
        n = 1000
        data = L2ChiData(
            time=np.arange(n, dtype=float),
            pres=np.arange(n, dtype=float),
            temp=np.zeros(n),
            temp_fast=np.zeros((2, n)),
            gradt=np.zeros((2, n)),
            vib=np.zeros((3, n)),
            pspd_rel=np.ones(n),
            section_number=np.zeros(n),
            diff_gains=[0.94, 0.94],
            fs_fast=512.0,
        )
        assert data.n_temp == 2
        assert data.n_vib == 3
        assert data.n_time == n
        assert data.fs_fast == 512.0
        assert data.diff_gains == [0.94, 0.94]

    def test_n_temp_1d_returns_zero(self):
        """If temp_fast is 1-D, n_temp should be 0."""
        data = L2ChiData(
            time=np.arange(10, dtype=float),
            pres=np.arange(10, dtype=float),
            temp=np.zeros(10),
            temp_fast=np.zeros(10),  # 1-D
            gradt=np.zeros((1, 10)),
            vib=np.zeros((2, 10)),
            pspd_rel=np.ones(10),
            section_number=np.zeros(10),
            diff_gains=[0.94],
            fs_fast=512.0,
        )
        assert data.n_temp == 0


# ---- L2ChiParams dataclass ----


class TestL2ChiParams:
    """Test default parameter values."""

    def test_defaults(self):
        p = L2ChiParams()
        assert p.HP_cut == 0.25
        np.testing.assert_array_equal(p.despike_T, [10.0, 0.5, 0.04])

    def test_custom_values(self):
        p = L2ChiParams(HP_cut=1.0, despike_T=np.array([5.0, 1.0, 0.1]))
        assert p.HP_cut == 1.0
        assert p.despike_T[0] == 5.0


# ---- process_l2_chi ----


class TestProcessL2Chi:
    """Tests for the main process_l2_chi function."""

    def test_output_type(self):
        l1 = _make_l1()
        l2 = _make_l2()
        result = process_l2_chi(l1, l2)
        assert isinstance(result, L2ChiData)

    def test_output_shapes(self):
        n = 4096
        l1 = _make_l1(n_time=n, n_temp=2, n_vib=2)
        l2 = _make_l2(n_time=n)
        result = process_l2_chi(l1, l2)
        assert result.time.shape == (n,)
        assert result.pres.shape == (n,)
        assert result.temp.shape == (n,)
        assert result.temp_fast.shape == (2, n)
        assert result.gradt.shape == (2, n)
        assert result.vib.shape == (2, n)
        assert result.pspd_rel.shape == (n,)
        assert result.section_number.shape == (n,)

    def test_no_temp_fast_raises(self):
        """Should raise ValueError when L1 has no temp_fast."""
        l1 = _make_l1(include_temp_fast=False)
        l2 = _make_l2()
        with pytest.raises(ValueError, match="no temp_fast"):
            process_l2_chi(l1, l2)

    def test_section_number_preserved(self):
        """Output section_number should match L2Data input."""
        l1 = _make_l1()
        l2 = _make_l2(n_sections=2)
        result = process_l2_chi(l1, l2)
        np.testing.assert_array_equal(result.section_number, l2.section_number)

    def test_section_number_is_copy(self):
        """Output section_number should be a copy, not a reference."""
        l1 = _make_l1()
        l2 = _make_l2()
        result = process_l2_chi(l1, l2)
        assert result.section_number is not l2.section_number

    def test_default_params_used(self):
        """Passing params=None should use L2ChiParams defaults."""
        l1 = _make_l1()
        l2 = _make_l2()
        result = process_l2_chi(l1, l2, params=None)
        assert isinstance(result, L2ChiData)

    def test_diff_gains_preserved(self):
        l1 = _make_l1(diff_gains=[0.91, 0.95])
        l2 = _make_l2()
        result = process_l2_chi(l1, l2)
        assert result.diff_gains == [0.91, 0.95]

    def test_diff_gains_default_when_empty(self):
        """When L1 has no diff_gains, should default to [0.94] * n_temp."""
        l1 = _make_l1(n_temp=2)
        l1.diff_gains = []
        l2 = _make_l2()
        result = process_l2_chi(l1, l2)
        assert result.diff_gains == [0.94, 0.94]

    def test_fs_fast_preserved(self):
        l1 = _make_l1(fs=1024.0)
        l2 = _make_l2()
        result = process_l2_chi(l1, l2)
        assert result.fs_fast == 1024.0

    def test_time_and_pres_are_copies(self):
        """Output time and pres should be copies of L1 data."""
        l1 = _make_l1()
        l2 = _make_l2()
        result = process_l2_chi(l1, l2)
        np.testing.assert_array_equal(result.time, l1.time)
        np.testing.assert_array_equal(result.pres, l1.pres)
        assert result.time is not l1.time
        assert result.pres is not l1.pres


class TestDespiking:
    """Test that despiking modifies spiked temperature signals."""

    def test_spike_reduced(self):
        """A large spike within a section should be reduced by despiking."""
        n = 4096
        l1 = _make_l1(n_time=n, n_temp=1)
        l2 = _make_l2(n_time=n, n_sections=1)

        # Insert a spike in the section interior
        spike_idx = n // 2
        l1.temp_fast[0, spike_idx] += 5.0  # huge spike
        original_val = l1.temp_fast[0, spike_idx].copy()

        result = process_l2_chi(l1, l2)
        # After despiking, the spike should be substantially reduced
        assert abs(result.temp_fast[0, spike_idx]) < abs(original_val)

    def test_inf_threshold_skips_despike(self):
        """With threshold=inf, despiking should be skipped."""
        n = 4096
        l1 = _make_l1(n_time=n, n_temp=1)
        l2 = _make_l2(n_time=n)

        spike_idx = n // 2
        l1.temp_fast[0, spike_idx] += 5.0
        original = l1.temp_fast[0, spike_idx].copy()

        params = L2ChiParams(despike_T=np.array([np.inf, 0.5, 0.04]))
        result = process_l2_chi(l1, l2, params=params)
        # Value should be unchanged since despike was skipped
        assert result.temp_fast[0, spike_idx] == pytest.approx(original, abs=1e-10)


class TestVibrationHPFilter:
    """Test HP filtering of vibration channels."""

    def test_dc_removed(self):
        """HP filter should remove DC component from vibration."""
        n = 4096
        l1 = _make_l1(n_time=n, n_vib=2)
        l1.vib += 10.0  # add DC offset
        l2 = _make_l2(n_time=n)

        result = process_l2_chi(l1, l2)
        # After HP filtering, mean should be near zero (ignore edges)
        interior = slice(n // 4, 3 * n // 4)
        for vi in range(2):
            assert abs(np.mean(result.vib[vi, interior])) < 1.0

    def test_no_vib_channels(self):
        """Empty vibration should pass through without error."""
        n = 4096
        l1 = _make_l1(n_time=n, n_vib=0)
        l1.vib = np.zeros((0, n))
        l2 = _make_l2(n_time=n)

        result = process_l2_chi(l1, l2)
        assert result.vib.shape == (0, n)

    def test_hp_cut_zero_skips_filter(self):
        """With HP_cut=0, vibration should not be filtered."""
        n = 4096
        l1 = _make_l1(n_time=n, n_vib=1)
        l1.vib[:] = 5.0
        l2 = _make_l2(n_time=n)

        params = L2ChiParams(HP_cut=0.0)
        result = process_l2_chi(l1, l2, params=params)
        np.testing.assert_array_almost_equal(result.vib[0], 5.0)


class TestTemperatureGradient:
    """Test temperature gradient computation."""

    def test_gradient_sign(self):
        """Decreasing temperature with depth should give negative gradient."""
        n = 4096
        l1 = _make_l1(n_time=n, n_temp=1)
        # Perfectly linear decreasing temperature: 15 -> 10 C
        l1.temp_fast[0] = np.linspace(15.0, 10.0, n)
        l2 = _make_l2(n_time=n, speed=0.6)

        params = L2ChiParams(despike_T=np.array([np.inf, 0.5, 0.04]))
        result = process_l2_chi(l1, l2, params=params)

        # Interior gradient should be negative (temp decreasing with depth)
        interior = slice(n // 4, 3 * n // 4)
        assert np.mean(result.gradt[0, interior]) < 0

    def test_gradient_magnitude(self):
        """Gradient dT/dz should be consistent with fs * diff(T) / speed."""
        n = 4096
        fs = 512.0
        speed = 0.5
        l1 = _make_l1(n_time=n, n_temp=1, fs=fs)
        # Linear ramp: dT/dt = (10-15)/((n-1)/fs) = -5 * fs/(n-1) [C/s]
        # dT/dz = dT/dt / speed
        l1.temp_fast[0] = np.linspace(15.0, 10.0, n)
        l2 = _make_l2(n_time=n, speed=speed)

        params = L2ChiParams(despike_T=np.array([np.inf, 0.5, 0.04]))
        result = process_l2_chi(l1, l2, params=params)

        dTdt = -5.0 * fs / (n - 1)  # C/s
        expected_gradt = dTdt / speed
        interior = slice(n // 4, 3 * n // 4)
        assert np.mean(result.gradt[0, interior]) == pytest.approx(expected_gradt, rel=0.01)

    def test_speed_floor(self):
        """Speed values near zero should be floored to 0.01 m/s."""
        n = 4096
        l1 = _make_l1(n_time=n, n_temp=1)
        l1.temp_fast[0] = np.linspace(15.0, 10.0, n)
        l2 = _make_l2(n_time=n, speed=0.001)  # below floor

        params = L2ChiParams(despike_T=np.array([np.inf, 0.5, 0.04]))
        result = process_l2_chi(l1, l2, params=params)

        # Gradient should be finite (not inf from dividing by near-zero speed)
        assert np.all(np.isfinite(result.gradt))


class TestTempSlow:
    """Test slow-rate temperature selection for viscosity."""

    def test_uses_slow_temp_when_available(self):
        n = 4096
        l1 = _make_l1(n_time=n, include_temp_slow=True)
        l2 = _make_l2(n_time=n)
        result = process_l2_chi(l1, l2)
        np.testing.assert_array_equal(result.temp, l1.temp)

    def test_falls_back_to_temp_fast(self):
        """Without slow temp, should use first fast temperature channel."""
        n = 4096
        l1 = _make_l1(n_time=n, include_temp_slow=False)
        l2 = _make_l2(n_time=n)
        result = process_l2_chi(l1, l2)
        # Should equal the despiked temp_fast[0]
        np.testing.assert_array_equal(result.temp, result.temp_fast[0])
