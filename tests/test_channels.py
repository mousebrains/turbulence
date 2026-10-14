# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for channel conversion functions."""

import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from odas_tpw.rsi.channels import (
    CONVERTERS,
    convert_accel,
    convert_aem1g_a,
    convert_aem1g_d,
    convert_alec_emc,
    convert_aroft_o2,
    convert_aroft_t,
    convert_inclt,
    convert_inclxy,
    convert_jac_c,
    convert_jac_t,
    convert_magn,
    convert_piezo,
    convert_poly,
    convert_raw,
    convert_sbc,
    convert_sbt,
    convert_shear,
    convert_therm,
    convert_voltage,
)


class TestConvertTherm:
    def test_synthetic_counts_to_temperature(self):
        """Synthetic raw counts with known Steinhart-Hart params -> temperature in 0-30 deg C."""
        params = {
            "a": "0",
            "b": "1",
            "adc_fs": "4.096",
            "adc_bits": "16",
            "g": "6.0",
            "e_b": "0.68",
            "t_0": "289.0",
            "beta_1": "3000",
        }
        # Mid-range counts should give a temperature in the ocean range
        data = np.array([0.0, 1000.0, 5000.0, 10000.0, 15000.0])
        T, units = convert_therm(data, params)
        assert units == "deg_C"
        assert T.shape == data.shape
        # All values should be finite
        assert np.all(np.isfinite(T))
        # Temperatures should be in a physically plausible range
        assert np.all(T > -10)
        assert np.all(T < 50)

    def test_beta_preferred_over_beta_1(self):
        """When a config supplies BOTH beta and beta_1, ODAS
        (convert_odas.m:501) uses `beta` first; our code must match that
        precedence, not prefer beta_1 (#4)."""
        base = {
            "a": "0", "b": "1", "adc_fs": "4.096", "adc_bits": "16",
            "g": "6.0", "e_b": "0.68", "t_0": "289.0",
        }
        data = np.array([1000.0, 5000.0, 10000.0])
        # Distinct beta vs beta_1 so the choice is observable.
        both = convert_therm(data, {**base, "beta": "3000", "beta_1": "3500"})[0]
        beta_only = convert_therm(data, {**base, "beta": "3000"})[0]
        beta1_only = convert_therm(data, {**base, "beta_1": "3500"})[0]
        # Result with both present must equal the beta-only result (beta wins).
        np.testing.assert_allclose(both, beta_only, rtol=1e-12)
        assert not np.allclose(both, beta1_only)


class TestConvertShear:
    def test_zero_mean_symmetric(self):
        """Zero-mean raw data → shear output symmetric around zero."""
        params = {
            "adc_fs": "4.096",
            "adc_bits": "16",
            "diff_gain": "1",
            "sens": "1",
            "adc_zero": "0",
            "sig_zero": "0",
        }
        data = np.array([-100.0, -50.0, 0.0, 50.0, 100.0])
        shear, units = convert_shear(data, params)
        # Units stay UDUNITS-valid "s-1" (CF compliance); the pre-normalization
        # caveat is carried in the sh1/sh2 ``comment`` attr instead (#104 U1-1).
        assert units == "s-1"
        # Should be antisymmetric around zero
        np.testing.assert_allclose(shear[2], 0.0, atol=1e-10)
        np.testing.assert_allclose(shear[0], -shear[4], rtol=1e-10)
        np.testing.assert_allclose(shear[1], -shear[3], rtol=1e-10)


class TestConvertPoly:
    def test_known_polynomial(self):
        """Known polynomial coeffs → matches np.polyval."""
        params = {
            "coef0": "1.0",
            "coef1": "2.0",
            "coef2": "-0.001",
            "units": "[dbar]",
        }
        data = np.array([0.0, 100.0, 1000.0, 10000.0])
        phys, units = convert_poly(data, params)
        # coef0 + coef1*x + coef2*x^2, polyval wants [coef2, coef1, coef0]
        expected = np.polyval([-0.001, 2.0, 1.0], data)
        np.testing.assert_allclose(phys, expected)
        assert units == "dbar"

    def test_no_coefficients(self):
        """No coefficients → passthrough."""
        data = np.array([1.0, 2.0, 3.0])
        phys, units = convert_poly(data, {})
        np.testing.assert_array_equal(phys, data)
        assert units == "counts"


class TestConvertThermBeta3:
    def test_beta3_path(self):
        """Thermistor conversion with beta_2 and beta_3 set."""
        params = {
            "a": "0",
            "b": "1",
            "adc_fs": "4.096",
            "adc_bits": "16",
            "g": "6.0",
            "e_b": "0.68",
            "t_0": "289.0",
            "beta_1": "3000",
            "beta_2": "50000",
            "beta_3": "1000000",
        }
        data = np.array([0.0, 1000.0, 5000.0, 10000.0, 15000.0])
        T, units = convert_therm(data, params)
        assert units == "deg_C"
        assert np.all(np.isfinite(T))
        # With beta_2 and beta_3, temperatures should still be in plausible range
        assert np.all(T > -20)
        assert np.all(T < 60)

    def test_beta2_only(self):
        """Thermistor conversion with beta_2 but no beta_3."""
        params = {
            "a": "0",
            "b": "1",
            "adc_fs": "4.096",
            "adc_bits": "16",
            "g": "6.0",
            "e_b": "0.68",
            "t_0": "289.0",
            "beta_1": "3000",
            "beta_2": "50000",
        }
        data = np.array([0.0, 1000.0, 5000.0])
        T, units = convert_therm(data, params)
        assert units == "deg_C"
        assert np.all(np.isfinite(T))


class TestConvertRaw:
    def test_passthrough(self):
        """Raw converter is identity."""
        data = np.array([42.0, -1.0, 0.0, 999.0])
        result, units = convert_raw(data, {})
        np.testing.assert_array_equal(result, data)
        assert units == "counts"


class TestConvertVoltage:
    def test_known_values(self):
        """Hand-calculated voltage conversion: (adc_zero + data * adc_fs / 2^adc_bits) / gain."""
        params = {
            "adc_fs": "4.096",
            "adc_bits": "16",
            "g": "2.0",
            "adc_zero": "0",
        }
        # data=32768 -> (0 + 32768 * 4.096 / 65536) / 2 = (2.048) / 2 = 1.024
        data = np.array([0.0, 32768.0, 65536.0])
        result, units = convert_voltage(data, params)
        assert units == "V"
        # data=0: (0 + 0) / 2 = 0
        np.testing.assert_allclose(result[0], 0.0, atol=1e-12)
        # data=32768: (0 + 32768 * 4.096 / 65536) / 2 = 2.048 / 2 = 1.024
        np.testing.assert_allclose(result[1], 1.024, rtol=1e-10)
        # data=65536: (0 + 65536 * 4.096 / 65536) / 2 = 4.096 / 2 = 2.048
        np.testing.assert_allclose(result[2], 2.048, rtol=1e-10)

    def test_nonzero_adc_zero(self):
        """Non-zero adc_zero shifts the output."""
        params = {
            "adc_fs": "4.096",
            "adc_bits": "16",
            "g": "1.0",
            "adc_zero": "0.5",
        }
        data = np.array([0.0])
        result, units = convert_voltage(data, params)
        assert units == "V"
        # (0.5 + 0) / 1.0 = 0.5
        np.testing.assert_allclose(result[0], 0.5, atol=1e-12)

    def test_defaults_warn_and_use_16bit(self):
        """A missing adc_bits warns (was a silent 2**0=1 mis-scale by 2**16) and
        falls back to a 16-bit ADC default (#28)."""
        import warnings

        data = np.array([3.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result, units = convert_voltage(data, {})
        assert units == "V"
        # (0 + 3 * 1 / 2^16) / 1 = 3 / 65536
        np.testing.assert_allclose(result[0], 3.0 / 65536.0, rtol=1e-10)
        assert any("adc_bits" in str(wi.message) for wi in w)


class TestConvertPiezo:
    def test_subtract_offset(self):
        """Piezo subtracts a_0 from data."""
        params = {"a_0": "100"}
        data = np.array([150.0, 100.0, 50.0, 0.0])
        result, units = convert_piezo(data, params)
        assert units == "counts"
        np.testing.assert_array_equal(result, np.array([50.0, 0.0, -50.0, -100.0]))

    def test_default_a0_zero(self):
        """With no a_0 param, default is 0 so output equals input."""
        data = np.array([42.0, -10.0])
        result, units = convert_piezo(data, {})
        assert units == "counts"
        np.testing.assert_array_equal(result, data)

    def test_negative_values(self):
        """Negative input values are handled correctly."""
        params = {"a_0": "-50"}
        data = np.array([-100.0, 0.0, 100.0])
        result, units = convert_piezo(data, params)
        assert units == "counts"
        # data - (-50) = data + 50
        np.testing.assert_array_equal(result, np.array([-50.0, 50.0, 150.0]))


class TestConvertInclXY:
    def test_positive_value(self):
        """Positive value below 2^13 passes through adis unchanged."""
        # raw=100, adis leaves it as 100 (no status bits set, < 2^13)
        # result = 0.025 * 100 + 0 = 2.5
        params = {"coef0": "0", "coef1": "0.025"}
        data = np.array([100.0])
        result, units = convert_inclxy(data, params)
        assert units == "deg"
        np.testing.assert_allclose(result[0], 0.025 * 100, rtol=1e-10)

    def test_negative_twos_complement(self):
        """Value >= 2^13 wraps via 14-bit two's complement."""
        # raw=8192 → adis: 8192 >= 2^13 → 8192 - 2^14 = -8192
        # result = 0.025 * (-8192) + 0 = -204.8
        params = {"coef0": "0", "coef1": "0.025"}
        data = np.array([8192.0])
        result, units = convert_inclxy(data, params)
        assert units == "deg"
        np.testing.assert_allclose(result[0], 0.025 * (-8192), rtol=1e-10)

    def test_negative_raw_passthrough(self):
        """Negative raw values in normal range pass through adis unchanged."""
        # raw=-3064 → none of the adis conditions fire → stays -3064
        # result = 0.025 * (-3064) = -76.6
        params = {"coef0": "0", "coef1": "0.025"}
        data = np.array([-3064.0])
        result, units = convert_inclxy(data, params)
        assert units == "deg"
        np.testing.assert_allclose(result[0], 0.025 * (-3064), rtol=1e-10)

    def test_bit15_new_data_cleared(self):
        """Bit 15 (new-data flag) is cleared by adis."""
        # raw < -2^14: e.g. -20000 → add 2^15 = -20000+32768 = 12768
        # 12768 >= 2^13 → wraps: 12768 - 16384 = -3616
        params = {"coef0": "0", "coef1": "0.025"}
        data = np.array([-20000.0])
        result, units = convert_inclxy(data, params)
        assert units == "deg"
        np.testing.assert_allclose(result[0], 0.025 * (-3616), rtol=1e-10)

    def test_zero_input(self):
        """Zero raw → zero degrees."""
        params = {"coef0": "0", "coef1": "0.025"}
        data = np.array([0.0])
        result, units = convert_inclxy(data, params)
        assert units == "deg"
        np.testing.assert_allclose(result[0], 0.0, atol=1e-12)

    def test_custom_coefficients(self):
        """Custom coef0 and coef1."""
        params = {"coef0": "10.0", "coef1": "0.05"}
        data = np.array([40.0])  # adis: 40 < 2^13, passthrough
        result, units = convert_inclxy(data, params)
        assert units == "deg"
        np.testing.assert_allclose(result[0], 0.05 * 40 + 10.0, rtol=1e-10)

    def test_defaults_warn_on_missing(self):
        """Missing coefficients WARN (ODAS errors; we default+warn, #104 U1-3)
        while still applying coef0=0, coef1=0.025."""
        data = np.array([8.0])  # adis: passthrough
        with pytest.warns(UserWarning, match="inclxy: calibration coefficient"):
            result, units = convert_inclxy(data, {})
        assert units == "deg"
        np.testing.assert_allclose(result[0], 0.025 * 8, rtol=1e-10)

    def test_supplied_coefficients_do_not_warn(self):
        """Explicit coef0/coef1 present → no missing-calibration warning."""
        import warnings

        data = np.array([8.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning fails the test
            result, _ = convert_inclxy(data, {"coef0": "0", "coef1": "0.025"})
        np.testing.assert_allclose(result[0], 0.025 * 8, rtol=1e-10)


class TestConvertInclT:
    def test_positive_value(self):
        """Positive value with default coefficients (coef0=624, coef1=-0.47)."""
        # raw=100 → adis passthrough (100 < 2^13)
        # result = -0.47 * 100 + 624 = -47 + 624 = 577.0
        data = np.array([100.0])
        result, units = convert_inclt(data, {})
        assert units == "deg_C"
        np.testing.assert_allclose(result[0], -0.47 * 100 + 624, rtol=1e-10)

    def test_negative_twos_complement(self):
        """Value >= 2^13 wraps via 14-bit two's complement."""
        # raw=8192 → adis: 8192 - 16384 = -8192
        # result = -0.47 * (-8192) + 624 = 3850.24 + 624 = 4474.24
        data = np.array([8192.0])
        result, units = convert_inclt(data, {})
        assert units == "deg_C"
        np.testing.assert_allclose(result[0], -0.47 * (-8192) + 624, rtol=1e-10)

    def test_zero_input(self):
        """Zero raw → coef0 value (624 by default)."""
        data = np.array([0.0])
        result, units = convert_inclt(data, {})
        assert units == "deg_C"
        np.testing.assert_allclose(result[0], 624.0, atol=1e-12)

    def test_custom_coefficients(self):
        """Custom coef0 and coef1 override defaults."""
        params = {"coef0": "500", "coef1": "-0.5"}
        data = np.array([40.0])  # adis: passthrough
        result, units = convert_inclt(data, params)
        assert units == "deg_C"
        np.testing.assert_allclose(result[0], -0.5 * 40 + 500, rtol=1e-10)


class TestConvertJacC:
    def test_known_values(self):
        """JAC conductivity with known polynomial coefficients."""
        # data encodes two 16-bit values packed into a 32-bit word:
        #   i_part = floor(data / 2^16), v_part = mod(data, 2^16)
        #   ratio = i_part / v_part
        # For data = 3 * 2^16 + 2 = 196610: i_part=3, v_part=2, ratio=1.5
        # poly: c*ratio^2 + b*ratio + a = 0.1*(1.5)^2 + 2.0*(1.5) + 1.0 = 0.225 + 3.0 + 1.0 = 4.225
        params = {"a": "1.0", "b": "2.0", "c": "0.1"}
        data = np.array([3 * 2**16 + 2])
        result, units = convert_jac_c(data, params)
        assert units == "mS_cm-1"
        np.testing.assert_allclose(result[0], 4.225, rtol=1e-10)

    def test_zero_v_part_is_nan(self):
        """A zero voltage part is corrupt -> flagged NaN, not fabricated as
        ratio=i/1 which looks like real conductivity (audit round-2)."""
        params = {"a": "0", "b": "0", "c": "0"}
        data = np.array([5 * 2**16])  # i_part=5, v_part=0
        result, units = convert_jac_c(data, params)
        assert units == "mS_cm-1"
        assert np.isnan(result[0])

    def test_zero_v_part_with_nonzero_coeffs_is_nan(self):
        params = {"a": "0", "b": "0", "c": "1.0"}
        data = np.array([4.0 * 2**16])  # v_part=0
        result, units = convert_jac_c(data, params)
        assert units == "mS_cm-1"
        assert np.isnan(result[0])

    def test_equal_parts(self):
        """When i_part equals v_part, ratio is 1."""
        # data = 2 * 2^16 + 2 = 131074: i_part=2, v_part=2, ratio=1
        # poly: 0.5*1 + 1.0*1 + 0.25 = 1.75
        params = {"a": "0.25", "b": "1.0", "c": "0.5"}
        data = np.array([2 * 2**16 + 2])
        result, units = convert_jac_c(data, params)
        assert units == "mS_cm-1"
        np.testing.assert_allclose(result[0], 1.75, rtol=1e-10)


class TestConvertJacT:
    def test_positive_values(self):
        """JAC temperature with positive (non-wrapping) data and known 5th-order polynomial."""
        # d = 100 (positive, no wrapping)
        # poly: f*d^5 + e*d^4 + d_coef*d^3 + c*d^2 + b*d + a
        # With simple coefficients: a=1, b=0.01, rest=0
        # result = 0.01*100 + 1 = 2.0
        params = {"a": "1.0", "b": "0.01", "c": "0", "d": "0", "e": "0", "f": "0"}
        data = np.array([100.0])
        result, units = convert_jac_t(data, params)
        assert units == "deg_C"
        np.testing.assert_allclose(result[0], 2.0, rtol=1e-10)

    def test_unsigned_wrapping(self):
        """Negative values wrap to unsigned: d[d<0] += 2^16."""
        # d = -1 -> wraps to 65535
        # With a=0, b=1, c-f=0: result = 65535
        params = {"a": "0", "b": "1.0", "c": "0", "d": "0", "e": "0", "f": "0"}
        data = np.array([-1.0])
        result, units = convert_jac_t(data, params)
        assert units == "deg_C"
        np.testing.assert_allclose(result[0], 65535.0, rtol=1e-10)

    def test_mixed_positive_negative(self):
        """Mix of positive and negative values; only negative ones wrap."""
        params = {"a": "0", "b": "1.0", "c": "0", "d": "0", "e": "0", "f": "0"}
        data = np.array([10.0, -10.0, 0.0])
        result, units = convert_jac_t(data, params)
        assert units == "deg_C"
        np.testing.assert_allclose(result[0], 10.0, rtol=1e-10)
        np.testing.assert_allclose(result[1], 65536 - 10.0, rtol=1e-10)
        np.testing.assert_allclose(result[2], 0.0, atol=1e-12)

    def test_full_polynomial(self):
        """Full 5th-order polynomial with non-zero coefficients."""
        params = {
            "a": "1.0",
            "b": "2.0",
            "c": "3.0",
            "d": "4.0",
            "e": "5.0",
            "f": "6.0",
        }
        d = 2.0
        # polyval([f, e, d, c, b, a], 2.0)
        # = 6*32 + 5*16 + 4*8 + 3*4 + 2*2 + 1 = 192 + 80 + 32 + 12 + 4 + 1 = 321
        expected = np.polyval([6.0, 5.0, 4.0, 3.0, 2.0, 1.0], d)
        data = np.array([d])
        result, units = convert_jac_t(data, params)
        assert units == "deg_C"
        np.testing.assert_allclose(result[0], expected, rtol=1e-10)

    def test_does_not_mutate_input(self):
        """Ensure the input array is not modified (copy semantics)."""
        data = np.array([-5.0, 10.0])
        original = data.copy()
        params = {"a": "0", "b": "1.0", "c": "0", "d": "0", "e": "0", "f": "0"}
        convert_jac_t(data, params)
        np.testing.assert_array_equal(data, original)


class TestConvertAroftO2:
    def test_positive_values(self):
        """Mid-range values (clear of the deglitch rails) divided by 100."""
        data = np.array([25000.0, 10000.0, 200.0])
        result, units = convert_aroft_o2(data, {})
        assert units == "umol_L-1"
        np.testing.assert_allclose(result[0], 250.0, rtol=1e-10)
        np.testing.assert_allclose(result[1], 100.0, rtol=1e-10)
        np.testing.assert_allclose(result[2], 2.0, rtol=1e-10)

    def test_rs232_deglitch(self):
        """RS232 rail glitches (<=50 or >=2^16-50) hold the previous sample;
        the first sample is never replaced (ODAS parity, #104 U1-2)."""
        # 30 (<=50) and 65535 (>=65486) are glitches -> take preceding value;
        # a run of consecutive glitches all inherit the last good value.
        data = np.array([10000.0, 30.0, 20000.0, 65535.0, 40.0])
        result, _ = convert_aroft_o2(data, {})
        # unsigned holds: [10000, 10000, 20000, 20000, 20000] -> /100
        np.testing.assert_allclose(result, [100.0, 100.0, 200.0, 200.0, 200.0], rtol=1e-10)

    def test_rs232_deglitch_keeps_first_sample(self):
        """A leading glitch is kept (ODAS loops from the 2nd sample)."""
        data = np.array([30.0, 10000.0])
        result, _ = convert_aroft_o2(data, {})
        np.testing.assert_allclose(result, [0.3, 100.0], rtol=1e-10)

    def test_unsigned_wrapping(self):
        """Negative values wrap to unsigned before dividing."""
        # d = -1 -> 65535 -> 65535 / 100 = 655.35
        data = np.array([-1.0])
        result, units = convert_aroft_o2(data, {})
        assert units == "umol_L-1"
        np.testing.assert_allclose(result[0], 655.35, rtol=1e-10)

    def test_mixed_sign(self):
        """Mix of positive and negative."""
        data = np.array([200.0, -100.0])
        result, units = convert_aroft_o2(data, {})
        assert units == "umol_L-1"
        np.testing.assert_allclose(result[0], 2.0, rtol=1e-10)
        np.testing.assert_allclose(result[1], (65536 - 100) / 100.0, rtol=1e-10)

    def test_does_not_mutate_input(self):
        """Ensure the input array is not modified."""
        data = np.array([-5.0, 10.0])
        original = data.copy()
        convert_aroft_o2(data, {})
        np.testing.assert_array_equal(data, original)


class TestConvertAroftT:
    def test_positive_values(self):
        """Mid-range values (clear of the deglitch rails): d/1000 - 5."""
        data = np.array([10000.0, 5000.0, 8000.0])
        result, units = convert_aroft_t(data, {})
        assert units == "deg_C"
        np.testing.assert_allclose(result[0], 5.0, rtol=1e-10)
        np.testing.assert_allclose(result[1], 0.0, atol=1e-12)
        np.testing.assert_allclose(result[2], 3.0, rtol=1e-10)

    def test_rs232_deglitch(self):
        """RS232 rail glitches hold the previous sample (#104 U1-2)."""
        data = np.array([10000.0, 20.0, 8000.0])
        result, _ = convert_aroft_t(data, {})
        # unsigned holds: [10000, 10000, 8000] -> /1000 - 5
        np.testing.assert_allclose(result, [5.0, 5.0, 3.0], rtol=1e-10)

    def test_unsigned_wrapping(self):
        """Negative values wrap to unsigned before scaling."""
        # d = -1 -> 65535 -> 65535/1000 - 5 = 65.535 - 5 = 60.535
        data = np.array([-1.0])
        result, units = convert_aroft_t(data, {})
        assert units == "deg_C"
        np.testing.assert_allclose(result[0], 60.535, rtol=1e-10)

    def test_mixed_sign(self):
        """Mix of positive and negative values."""
        data = np.array([15000.0, -500.0])
        result, units = convert_aroft_t(data, {})
        assert units == "deg_C"
        np.testing.assert_allclose(result[0], 10.0, rtol=1e-10)
        np.testing.assert_allclose(result[1], (65536 - 500) / 1000.0 - 5.0, rtol=1e-10)

    def test_does_not_mutate_input(self):
        """Ensure the input array is not modified."""
        data = np.array([-5.0, 10.0])
        original = data.copy()
        convert_aroft_t(data, {})
        np.testing.assert_array_equal(data, original)


class TestConvertGnd:
    def test_gnd_is_raw_alias(self):
        """CONVERTERS['gnd'] is convert_raw; should be a passthrough."""
        assert CONVERTERS["gnd"] is convert_raw

    def test_gnd_passthrough(self):
        """Calling via CONVERTERS dict returns data unchanged with 'counts' units."""
        data = np.array([42.0, -1.0, 0.0, 999.0])
        result, units = CONVERTERS["gnd"](data, {})
        np.testing.assert_array_equal(result, data)
        assert units == "counts"

    def test_all_expected_keys_present(self):
        """CONVERTERS dict has entries for all converter types."""
        expected = {
            "therm",
            "shear",
            "poly",
            "voltage",
            "piezo",
            "inclxy",
            "inclt",
            "jac_c",
            "jac_t",
            "raw",
            "aroft_o2",
            "aroft_t",
            "gnd",
            "aem1g_a",
            "aem1g_d",
            "alec_emc",
            "jac_emc",
        }
        assert expected.issubset(set(CONVERTERS.keys()))


class TestConvertAem1gA:
    def test_known_values(self):
        """Hand-calculated analog EM current meter conversion."""
        params = {
            "adc_fs": "4.096",
            "adc_bits": "16",
            "adc_zero": "2.048",
            "a": "0",       # cm/s offset → /100 → 0 m/s
            "b": "100",     # cm/s per V → /100 → 1 m/s per V
            "bias": "0",
        }
        # V = adc_zero + data * (adc_fs / 2^adc_bits)
        #   = 2.048 + 0 * (4.096/65536) = 2.048
        # V = a/100 + b/100 * V = 0 + 1.0 * 2.048 = 2.048
        # result = V - bias = 2.048
        data = np.array([0.0])
        result, units = convert_aem1g_a(data, params)
        assert units == "m_s-1"
        np.testing.assert_allclose(result[0], 2.048, rtol=1e-10)

    def test_with_bias(self):
        """Bias is subtracted from the final result."""
        params = {
            "adc_fs": "4.096",
            "adc_bits": "16",
            "adc_zero": "2.048",
            "a": "0",
            "b": "100",
            "bias": "2.048",
        }
        data = np.array([0.0])
        result, units = convert_aem1g_a(data, params)
        assert units == "m_s-1"
        np.testing.assert_allclose(result[0], 0.0, atol=1e-12)

    def test_nonzero_data(self):
        """Non-zero data counts produce correct voltage-to-velocity."""
        params = {
            "adc_fs": "4.096",
            "adc_bits": "16",
            "adc_zero": "0",
            "a": "0",
            "b": "100",
            "bias": "0",
        }
        # V = 0 + 32768 * (4.096/65536) = 32768 * 6.25e-5 = 2.048
        # V = 0 + 1.0 * 2.048 = 2.048
        data = np.array([32768.0])
        result, units = convert_aem1g_a(data, params)
        assert units == "m_s-1"
        np.testing.assert_allclose(result[0], 2.048, rtol=1e-10)

    def test_defaults(self):
        """Default parameters produce finite output."""
        data = np.array([0.0, 1000.0, -1000.0])
        result, units = convert_aem1g_a(data, {})
        assert units == "m_s-1"
        assert np.all(np.isfinite(result))

    def test_cm_to_m_conversion(self):
        """Coefficients a and b are divided by 100 (cm/s → m/s)."""
        params = {
            "adc_fs": "4.096",
            "adc_bits": "16",
            "adc_zero": "0",
            "a": "50",      # 50 cm/s → 0.5 m/s
            "b": "200",     # 200 cm/s per V → 2.0 m/s per V
            "bias": "0",
        }
        data = np.array([0.0])
        # V = 0 + 0 = 0
        # V = 0.5 + 2.0 * 0 = 0.5
        result, units = convert_aem1g_a(data, params)
        assert units == "m_s-1"
        np.testing.assert_allclose(result[0], 0.5, rtol=1e-10)


class TestConvertAem1gD:
    def test_known_values(self):
        """Digital EM current meter with known coefficients."""
        params = {"a": "0", "b": "100"}  # b/100 = 1.0
        data = np.array([500.0])
        # d = 500 (positive, no wrapping)
        # result = 0 + 1.0 * 500 = 500.0
        result, units = convert_aem1g_d(data, params)
        assert units == "m_s-1"
        np.testing.assert_allclose(result[0], 500.0, rtol=1e-10)

    def test_unsigned_wrapping(self):
        """Negative values wrap to unsigned before conversion."""
        params = {"a": "0", "b": "100"}  # b/100 = 1.0
        data = np.array([-1.0])
        # d = -1 -> 65535
        # result = 0 + 1.0 * 65535 = 65535.0
        result, units = convert_aem1g_d(data, params)
        assert units == "m_s-1"
        np.testing.assert_allclose(result[0], 65535.0, rtol=1e-10)

    def test_cm_to_m_conversion(self):
        """Coefficients a and b are divided by 100 (cm/s → m/s)."""
        params = {"a": "10", "b": "1"}
        # a/100 = 0.1, b/100 = 0.01
        data = np.array([100.0])
        # result = 0.1 + 0.01 * 100 = 0.1 + 1.0 = 1.1
        result, units = convert_aem1g_d(data, params)
        assert units == "m_s-1"
        np.testing.assert_allclose(result[0], 1.1, rtol=1e-10)

    def test_does_not_mutate_input(self):
        """Ensure the input array is not modified."""
        data = np.array([-5.0, 10.0])
        original = data.copy()
        convert_aem1g_d(data, {"a": "0", "b": "100"})
        np.testing.assert_array_equal(data, original)

    def test_defaults(self):
        """Default parameters produce finite output."""
        data = np.array([0.0, 1000.0, -1000.0])
        result, units = convert_aem1g_d(data, {})
        assert units == "m_s-1"
        assert np.all(np.isfinite(result))


class TestConvertAlecEmc:
    def test_known_values(self):
        """Alec EMC: coef1 * data + coef0."""
        params = {"coef0": "0.5", "coef1": "0.01"}
        data = np.array([100.0])
        # result = 0.01 * 100 + 0.5 = 1.5
        result, units = convert_alec_emc(data, params)
        assert units == "m_s-1"
        np.testing.assert_allclose(result[0], 1.5, rtol=1e-10)

    def test_zero_data(self):
        """Zero data → just the offset."""
        params = {"coef0": "2.0", "coef1": "0.5"}
        data = np.array([0.0])
        result, units = convert_alec_emc(data, params)
        assert units == "m_s-1"
        np.testing.assert_allclose(result[0], 2.0, rtol=1e-10)

    def test_negative_data(self):
        """Negative data values are handled correctly."""
        params = {"coef0": "0", "coef1": "1"}
        data = np.array([-50.0, 0.0, 50.0])
        result, units = convert_alec_emc(data, params)
        assert units == "m_s-1"
        np.testing.assert_allclose(result, [-50.0, 0.0, 50.0], rtol=1e-10)

    def test_defaults(self):
        """Default coef0=0, coef1=1: passthrough."""
        data = np.array([42.0])
        result, units = convert_alec_emc(data, {})
        assert units == "m_s-1"
        np.testing.assert_allclose(result[0], 42.0, rtol=1e-10)


class TestJacEmcAlias:
    def test_jac_emc_is_aem1g_a(self):
        """jac_emc is a deprecated alias for aem1g_a."""
        assert CONVERTERS["jac_emc"] is convert_aem1g_a


class TestConvertAccel:
    """Linear (DC-response) accelerometer, odas_accel_internal port."""

    def test_known_values(self):
        from odas_tpw.rsi.channels import convert_accel

        # x = data*adc_fs/2^bits + adc_zero - sig_zero; phys = 9.81*(x-coef0)/coef1
        params = {
            "adc_fs": "4.096",
            "adc_bits": "16",
            "adc_zero": "0",
            "sig_zero": "0",
            "coef0": "0.5",
            "coef1": "2.0",
        }
        data = np.array([16000.0])
        x = 16000.0 * 4.096 / 2**16
        expected = 9.81 * (x - 0.5) / 2.0
        result, units = convert_accel(data, params)
        assert units == "m_s-2"
        np.testing.assert_allclose(result[0], expected, rtol=1e-12)

    def test_zero_input_with_offsets(self):
        from odas_tpw.rsi.channels import convert_accel

        params = {
            "adc_fs": "1",
            "adc_bits": "0",
            "adc_zero": "1.0",
            "sig_zero": "0.25",
            "coef0": "0",
            "coef1": "1",
        }
        result, _ = convert_accel(np.array([0.0]), params)
        # x = 0*1/1 + 1.0 - 0.25 = 0.75 -> 9.81*0.75
        np.testing.assert_allclose(result[0], 9.81 * 0.75, rtol=1e-12)

    def test_missing_coefficients_warn(self):
        from odas_tpw.rsi.channels import convert_accel

        with pytest.warns(UserWarning, match="coef1"):
            convert_accel(np.array([1.0]), {"coef0": "0"})


class TestConvertMagn:
    """Magnetometer, odas_magn_internal port."""

    def test_known_values(self):
        from odas_tpw.rsi.channels import convert_magn

        params = {"coef0": "100.0", "coef1": "4.0"}
        result, units = convert_magn(np.array([500.0, 100.0]), params)
        assert units == "uT"
        np.testing.assert_allclose(result, [(500.0 - 100.0) / 4.0, 0.0], rtol=1e-12)

    def test_missing_coefficients_warn(self):
        from odas_tpw.rsi.channels import convert_magn

        with pytest.warns(UserWarning, match="coef0"):
            convert_magn(np.array([1.0]), {})

    def test_registered_in_converters(self):
        from odas_tpw.rsi.channels import convert_accel, convert_magn

        assert CONVERTERS["accel"] is convert_accel
        assert CONVERTERS["magn"] is convert_magn


# RSI's JAC-CT board on the MicroRider MR-SE SN 240 (SeaExplorer SEA025, 2017).
# Coefficients are that file's config, verbatim; the expected values are ODAS
# v4.5.1 convert_odas.m on the same counts (computed in MATLAB, 2026-10-13).
_RSIJAC_C_PARAMS = {
    "type": "rsijac_c",
    "division_factor": "2^28",
    "cell_constant": "2.692219e-2",
    "gain": "10.88/100",
    "y_offset": "2.87e-7",
    "a": "0",
    "b": "1",
}
_RSIJAC_T_PARAMS = {
    "type": "rsijac_t",
    "scale_factor": "2^28",
    "t_0": "288.4393",
    "beta_1": "3905.963",
    "beta_2": "3.115573e5",
    "beta_3": "1.265211e7",
}


class TestConfigNumber:
    """ODAS evals config values; RSI writes some as arithmetic. MATLAB's rules."""

    def test_matlab_arithmetic(self):
        from odas_tpw.rsi.channels import config_number

        assert config_number("2^28") == 2.0**28
        assert config_number("10.88/100") == pytest.approx(0.1088, rel=1e-15)
        assert config_number(" -2.87e-7 ") == -2.87e-7
        assert config_number("(1+2)*3") == 9.0
        assert config_number("1e-3") == 1e-3
        assert config_number("3.0e5") == 3.0e5
        assert config_number(".5") == 0.5

    def test_matlab_power_rules(self):
        from odas_tpw.rsi.channels import config_number

        assert config_number("-2^2") == -4.0  # ^ binds tighter than unary minus
        assert config_number("2^3^2") == 64.0  # left-associative, as MATLAB
        assert config_number("2^-3") == 0.125  # signed exponent
        assert config_number("2*-3") == -6.0

    def test_default_only_when_absent(self):
        from odas_tpw.rsi.channels import config_number

        assert config_number(None, 5.0) == 5.0
        assert config_number("", 5.0) == 5.0
        assert config_number("0", 5.0) == 0.0
        with pytest.raises(ValueError, match=r"JAC_T: t_0 is required and missing"):
            config_number(None, key="t_0", channel="JAC_T")

    def test_rejects_anything_but_arithmetic(self):
        from odas_tpw.rsi.channels import config_number

        for bad in ("__import__('os')", "a + 1", "2 ** 28", "1e5)", "[1]", "0x10", "1_000",
                    "True", "INF", "NaNa", "pi", "2^", "((1)", "0,09"):
            with pytest.raises(ValueError, match="not a numeric expression"):
                config_number(bad, 1.0, key="k", channel="JAC_C")

    def test_non_finite_results_are_value_errors(self):
        from odas_tpw.rsi.channels import ConfigNumberNotFinite, config_number

        for bad in ("1/0", "0^-1", "10^400", "(-8)^(1/3)", "(" * 150 + "1" + ")" * 150):
            with pytest.raises(ValueError):
                config_number(bad, 1.0)
        # MATLAB's NaN/Inf are expressions; their values are what is refused,
        # with a type that tells them apart from a malformed value.
        for bad in ("NaN", "nan", "Inf", "-inf", "1/0", "10^400", "Inf-Inf"):
            with pytest.raises(ConfigNumberNotFinite):
                config_number(bad, 1.0)
        with pytest.raises(ValueError) as info:
            config_number("0,09", 1.0)
        assert not isinstance(info.value, ConfigNumberNotFinite)


    def test_whitespace_around_operators_not_inside_operands(self):
        """MATLAB refuses ``1 2``; deleting the space would read it as 12."""
        from odas_tpw.rsi.channels import config_number

        assert config_number("2 ^ 28") == 2.0**28
        assert config_number(" 10.88 / 100\t") == config_number("10.88/100")
        assert config_number("- 5") == -5.0
        for bad in ("1 2", "2 .5", "1e 5", "0.0 767"):
            with pytest.raises(ValueError, match="not a numeric expression"):
                config_number(bad, key="sens", channel="sh1")


# Every converter's numeric coefficients, with representative values.  Each key
# is read through config_number: arithmetic is evaluated as ODAS evaluates it,
# and a present-but-malformed value raises instead of becoming the default.
_THERM = {"t_0": "289", "beta_1": "3000", "beta_2": "5e5", "beta_3": "1e7", "g": "6",
          "e_b": "0.68", "a": "1", "b": "1.001", "adc_fs": "4.096", "adc_bits": "16"}
_COEFFICIENT_CASES = [
    (convert_therm, _THERM, [1000.0, -2000.0]),
    (convert_shear, {"sens": "0.1", "diff_gain": "0.94", "adc_fs": "4.096",
                     "adc_bits": "16", "adc_zero": "0.01", "sig_zero": "0.002"}, [1000.0]),
    (convert_poly, {"coef0": "1.5", "coef1": "0.25", "coef2": "1e-4"}, [10.0, 2000.0]),
    (convert_voltage, {"adc_fs": "4.096", "adc_bits": "16", "g": "2", "adc_zero": "0.1"},
     [1000.0]),
    (convert_piezo, {"a_0": "12"}, [100.0]),
    (convert_accel, {"adc_zero": "0.1", "adc_fs": "4.096", "adc_bits": "16",
                     "sig_zero": "0.05", "coef0": "0.2", "coef1": "0.5"}, [1000.0]),
    (convert_magn, {"coef0": "3", "coef1": "0.7"}, [100.0]),
    (convert_inclxy, {"coef0": "0.5", "coef1": "0.025"}, [100.0]),
    (convert_inclt, {"coef0": "624", "coef1": "-0.47"}, [100.0]),
    (convert_jac_c, {"a": "0.1", "b": "30", "c": "-2"}, [3.0 * 2**16 + 5]),
    (convert_jac_t, {"a": "-5", "b": "1e-3", "c": "1e-8", "d": "1e-12", "e": "1e-17",
                     "f": "1e-22"}, [20000.0]),
    (convert_sbt, {"coef0": "4.4e-3", "coef1": "6.4e-4", "coef2": "2.3e-5",
                   "coef3": "2.2e-6", "coef4": "1000", "coef5": "24e6", "coef6": "128"},
     [614400.0]),
    (convert_sbc, {"coef0": "-4", "coef1": "4.9", "coef2": "-1e-3", "coef3": "2e-4",
                   "coef4": "1e-6", "coef5": "24e6", "coef6": "128"}, [384000.0]),
    (convert_aem1g_a, {"adc_fs": "4.096", "adc_bits": "16", "adc_zero": "2.048",
                       "a": "-150", "b": "75", "bias": "0.01"}, [1000.0]),
    (convert_aem1g_d, {"a": "-150", "b": "0.02"}, [1000.0]),
    (convert_alec_emc, {"coef0": "0.1", "coef1": "0.002"}, [1000.0]),
]
_COEFFICIENT_KEYS = [
    pytest.param(conv, params, data, key, id=f"{conv.__name__}-{key}")
    for conv, params, data in _COEFFICIENT_CASES
    for key in params
]


class TestEveryCoefficientReadsLikeOdas:
    """No converter reads arithmetic, or a malformed value, as its default.

    ``_safe_float`` used to turn ``2^28`` -- or ``0,5`` -- into 0 or the
    caller's default without a word, in thirteen converters.
    """

    @pytest.mark.parametrize(("conv", "params", "data", "key"), _COEFFICIENT_KEYS)
    def test_arithmetic_is_evaluated(self, conv, params, data, key):
        # Two values, so the test cannot pass by an old fallback default
        # happening to equal the table's value (voltage adc_bits = 16 did).
        for x in (float(params[key]), 1.25 * float(params[key])):
            expr = f"{2 * x!r}/2"  # exact: doubling and halving are lossless
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # the arithmetic warning; tested below
                plain, units = conv(np.array(data), {**params, key: repr(x)})
                evaluated, units2 = conv(np.array(data), {**params, key: expr})
            np.testing.assert_array_equal(evaluated, plain)
            assert units2 == units

    @pytest.mark.parametrize(("conv", "params", "data", "key"), _COEFFICIENT_KEYS)
    def test_malformed_value_raises_naming_the_key(self, conv, params, data, key):
        # Quoted or as "<channel>: <key> =": a bare \ba\b matched the article
        # "a" in the message's own advice.
        with pytest.raises(ValueError, match=rf"'{key}'|: {key} = "):
            conv(np.array(data), {**params, key: "0,5"})

    def test_every_numeric_converter_is_covered(self):
        """A converter added later must join the table (raw/gnd/aroft read no
        coefficients; rsijac has its own ODAS golden tests)."""
        covered = {conv for conv, _, _ in _COEFFICIENT_CASES}
        exempt = {"raw", "gnd", "aroft_o2", "aroft_t", "rsijac_c", "rsijac_t", "jac_emc"}
        missing = {t for t, f in CONVERTERS.items() if f not in covered and t not in exempt}
        assert not missing

    def test_absent_keys_keep_their_defaults(self):
        """Only present values changed: absent ones default exactly as before."""
        assert convert_inclt(np.array([0.0]), {})[0][0] == 624.0
        assert convert_piezo(np.array([7.0]), {})[0][0] == 7.0
        np.testing.assert_array_equal(
            convert_alec_emc(np.array([2.0]), {})[0], np.array([2.0])
        )


class TestOneParserForEveryReader:
    """Readers outside the converters parse the same keys the same way."""

    def test_chi_thermistor_calibration_evaluates_arithmetic(self):
        from odas_tpw.rsi.channels import config_number
        from odas_tpw.rsi.chi_io import _extract_therm_cal

        cal = _extract_therm_cal({"name": "T1", "t_0": "273.15+16", "beta_1": "3000",
                                  "g": "6", "e_b": "0.68", "b": ""})
        assert cal["T_0"] == config_number("273.15+16")
        assert "b" not in cal  # empty is absent, as in the converters
        with pytest.raises(ValueError, match=r"T1: beta_1"):
            _extract_therm_cal({"name": "T1", "beta_1": "3,000"})

    def test_chi_noise_model_and_converter_read_t0_identically(self):
        """Before, convert_therm refused ``t_0 = 273.15+16`` while the chi path
        crashed on it; neither can now read it differently from the other."""
        from odas_tpw.rsi.chi_io import _extract_therm_cal

        params = {**_THERM, "t_0": "273.15+16"}
        plain = convert_therm(np.array([1000.0]), {**_THERM, "t_0": repr(273.15 + 16)})[0]
        np.testing.assert_array_equal(convert_therm(np.array([1000.0]), params)[0], plain)
        assert _extract_therm_cal(params)["T_0"] == 273.15 + 16

    def test_gradient_diff_gain(self):
        from odas_tpw.rsi.chi_io import _therm_gradient_config

        cfg = {"channels": [{"name": "T1_dT1", "diff_gain": "0.94*1"},
                            {"name": "T1", "t_0": "289"}]}
        diff_gain, cal = _therm_gradient_config(cfg, "T1_dT1")
        assert diff_gain == 0.94
        assert cal["T_0"] == 289.0

    def test_shear_sheet_check_reads_sens_like_the_converter(self):
        from odas_tpw.rsi.shear_cal import _to_float

        assert _to_float("0.0767") == 0.0767
        assert _to_float("7.67/100") == 7.67 / 100
        for bad in ("", "nan", "inf", "0,0767", None):
            assert _to_float(bad) is None


class TestConfigNumberEdges:
    """Corners the adversarial review of the parser found."""

    def test_matlab_precedence_of_a_signed_exponent(self):
        """Checked in MATLAB on 2026-10-13: a sign after ``^`` takes only the
        next operand, and ``^`` stays left-associative across it."""
        from odas_tpw.rsi.channels import config_number

        assert config_number("2^-3^2") == 0.015625  # (2^-3)^2, not 2^-(3^2)
        assert config_number("2^-3^-2") == 64.0
        assert config_number("2^+3^2") == 64.0
        assert config_number("2*-3^2") == -18.0

    def test_space_inside_an_exponent_is_refused(self):
        from odas_tpw.rsi.channels import config_number

        for bad in ("1e -5", "1e- 5", "1e+ 5", "1E -5", "2.5e +3"):
            with pytest.raises(ValueError, match="not a numeric expression"):
                config_number(bad, key="sens", channel="sh1")
        assert config_number("1e-5 - 1") == 1e-5 - 1  # a space around an operator is fine

    def test_numeric_types_keep_their_exact_value(self):
        from odas_tpw.rsi.channels import ConfigNumberNotFinite, config_number

        assert config_number(np.float32(0.1)) == float(np.float32(0.1))  # not str() = 0.1
        assert config_number(np.int64(16)) == 16.0
        with pytest.raises(ConfigNumberNotFinite):
            config_number(10**400)
        for flag in (True, np.bool_(False)):  # a flag is not a coefficient
            with pytest.raises(ValueError):
                config_number(flag)

    def test_non_finite_type_reaches_the_converters_caller(self):
        from odas_tpw.rsi.channels import ConfigNumberNotFinite

        with pytest.raises(ConfigNumberNotFinite, match="'g'"):
            convert_therm(np.array([1000.0]), {**_THERM, "g": "NaN"})

    def test_blank_is_absent(self):
        """``key =`` with nothing after it parses to ''; it is treated as absent."""
        from odas_tpw.rsi.channels import config_number

        assert config_number("  ", 0.94) == 0.94
        with pytest.raises(ValueError, match="required and missing"):
            config_number("", key="diff_gain", channel="T1_dT1")

    def test_blank_legacy_beta_is_absent(self):
        """As in chi's noise model and fp07cal: a blank ``beta`` defers to beta_1."""
        data = np.array([1000.0])
        np.testing.assert_array_equal(
            convert_therm(data, {**_THERM, "beta": ""})[0], convert_therm(data, _THERM)[0]
        )

    def test_blank_poly_coefficient_is_refused(self):
        """ODAS evals '' to [] and drops the term, shifting every higher order
        down one; neither 0 nor a shift is safe to guess."""
        with pytest.raises(ValueError, match="'coef1' is present but empty"):
            convert_poly(np.array([1.0]), {"name": "P", "coef0": "1", "coef1": "", "coef2": "3"})

    def test_blank_required_coefficient_warns_like_a_missing_one(self):
        blank = {**_THERM, "g": ""}
        absent = {k: v for k, v in _THERM.items() if k != "g"}
        with pytest.warns(UserWarning, match="'g' missing"):
            a = convert_therm(np.array([1000.0]), blank)[0]
        with pytest.warns(UserWarning, match="'g' missing"):
            b = convert_therm(np.array([1000.0]), absent)[0]
        np.testing.assert_array_equal(a, b)


class TestArithmeticWarns:
    """The only arithmetic outside JAC-CT in 19,064 real configs is a typo."""

    def test_typo_shaped_coefficient_warns_with_its_value(self):
        params = {**_THERM, "name": "T1", "beta_1": "3-1315.08"}  # SUNRISE 2021 SN 412
        with pytest.warns(UserWarning, match=r"T1: beta_1 = '3-1315.08' is arithmetic.*-1312\.08"):
            convert_therm(np.array([1000.0]), params)

    def test_plain_numbers_do_not_warn(self):
        params = {**_THERM, "name": "T1", "beta_1": " +3.0e3 ", "t_0": "289.", "b": ".9991"}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            convert_therm(np.array([1000.0]), params)

    def test_jac_ct_arithmetic_is_rsis_own_and_silent(self):
        from odas_tpw.rsi.channels import convert_rsijac_c

        params = {"name": "JAC_C", "division_factor": "2^28", "gain": "10.88/100",
                  "cell_constant": "1/10"}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            convert_rsijac_c(np.array([3.0e8]), params)


class TestBetaInfinity:
    """``beta_2 = Inf`` (Rockland's Zissou tutorial VMP_024.p) is no term, as in ODAS."""

    def test_converter_reads_inf_as_omitted(self):
        with_inf = {**_THERM, "beta_2": "Inf"}
        with_inf.pop("beta_3")
        omitted = {k: v for k, v in with_inf.items() if k != "beta_2"}
        data = np.array([1000.0, -2000.0])
        np.testing.assert_array_equal(
            convert_therm(data, with_inf)[0], convert_therm(data, omitted)[0]
        )

    def test_inf_beta_2_still_applies_beta_3(self):
        """ODAS nests beta_3 under beta_2's isfield, not under its value:
        beta_2 = Inf is the same as 1e30 (no quadratic term), beta_3 kept."""
        data = np.array([1000.0, -2000.0])
        np.testing.assert_array_equal(
            convert_therm(data, {**_THERM, "beta_2": "Inf"})[0],
            convert_therm(data, {**_THERM, "beta_2": "1e30"})[0],
        )
        no_cubic = convert_therm(data, {**_THERM, "beta_2": "Inf", "beta_3": "inf"})[0]
        assert not np.array_equal(convert_therm(data, {**_THERM, "beta_2": "Inf"})[0], no_cubic)

    def test_negative_infinity_is_no_term_and_nan_is_refused(self):
        data = np.array([1000.0])
        np.testing.assert_array_equal(
            convert_therm(data, {**_THERM, "beta_2": "-Inf"})[0],
            convert_therm(data, {**_THERM, "beta_2": "Inf"})[0],
        )
        with pytest.raises(ValueError, match="beta_2"):
            convert_therm(data, {**_THERM, "beta_2": "NaN"})

    def test_noise_model_and_fp07cal_agree(self):
        from odas_tpw.fp07cal.logr import config_to_coeffs
        from odas_tpw.rsi.chi_io import _extract_therm_cal

        cfg = {**_THERM, "name": "T1", "beta_2": "Inf"}
        assert "beta_2" not in _extract_therm_cal(cfg)
        coeffs = config_to_coeffs(cfg)
        assert coeffs[2] == 0.0 and coeffs[3] == 1.0 / 1e7


class TestEveryOtherReader:
    """Readers outside rsi/channels.py, each tested on its own so that
    reverting any one of them to ``float()`` fails a test."""

    def test_fp07cal_bridge_and_coefficients(self):
        from odas_tpw.fp07cal.logr import BridgeParams, config_to_coeffs

        cfg = {**_THERM, "name": "T1", "adc_bits": "2^4", "t_0": "273.15+16"}
        assert BridgeParams.from_channel_config(cfg, "T1").adc_bits == 16
        assert config_to_coeffs(cfg)[0] == 1.0 / (273.15 + 16)
        with pytest.raises(ValueError, match="non-numeric bridge parameter"):
            BridgeParams.from_channel_config({**cfg, "g": "6,0"}, "T1")

    def test_fp07cal_noise_model_honors_legacy_beta(self):
        """convert_therm reads ``beta`` before ``beta_1``; so must chi's noise model."""
        from odas_tpw.rsi.chi_io import _extract_therm_cal

        assert _extract_therm_cal({**_THERM, "beta": "3100"})["beta_1"] == 3100.0

    def test_sensor_inventory_compares_coefficients_by_value(self):
        from odas_tpw.rsi.sensor_inventory import _norm

        assert _norm("diff_gain", "94/100") == _norm("diff_gain", "0.94")
        assert _norm("adc_bits", "2^4") == 16
        assert _norm("sens", "nan") == "nan"  # non-finite keeps its text
        # A date is not arithmetic: 2019-06-26 and 2019-07-25 both "equal" 1987.
        assert _norm("cal_date", "2019-06-26") != _norm("cal_date", "2019-07-25")

    def test_diff_gain_audit(self):
        from odas_tpw.rsi.diff_gain import collect_gains

        cfg = {"instrument_info": {"sn": "479"},
               "channels": [{"name": "sh1", "diff_gain": "94/100"}]}
        uses, errors = collect_gains([Path("x.p")], lambda p: ({}, cfg))
        assert not errors and uses[0].diff_gain == 0.94


def _patched_copy(src, dst, channel, key, value):
    """*src* with ``key`` of the [channel] section named *channel* set to *value*."""
    from odas_tpw.rsi.config_patch import read_config_text, write_patched_pfile

    blocks = re.split(r"(?m)^(?=\[)", read_config_text(src))
    hits = 0
    for i, b in enumerate(blocks):
        if re.search(rf"(?m)^name\s*=\s*{re.escape(channel)}\s*$", b):
            blocks[i], n = re.subn(rf"(?m)^({key}\s*=\s*)\S+", rf"\g<1>{value}", b)
            hits += n
    assert hits == 1, (channel, key, hits)
    return write_patched_pfile(src, dst, "".join(blocks))


@pytest.fixture
def arithmetic_diff_gain_copy(sample_p_file, tmp_path):
    """SN479_0006.p with every ``diff_gain`` rewritten as exact arithmetic."""
    from odas_tpw.rsi.config_patch import read_config_text, write_patched_pfile

    def double_half(m):
        return f"{m.group(1)}{2 * float(m.group(2))!r}/2"

    text, n = re.subn(
        r"(?m)^(diff_gain\s*=\s*)([0-9.]+)", double_half, read_config_text(sample_p_file)
    )
    assert n == 5  # sh1, sh2, T1_dT1, T2_dT2, P_dP
    return write_patched_pfile(sample_p_file, tmp_path / "arithmetic.p", text)


class TestArithmeticDiffGainEndToEnd:
    """A real file whose every diff_gain is arithmetic reads identically."""

    def test_deconvolution(self, sample_p_file, arithmetic_diff_gain_copy):
        from odas_tpw.rsi.p_file import PFile

        a, b = PFile(sample_p_file), PFile(arithmetic_diff_gain_copy)
        assert set(a.channels) == set(b.channels)
        for name in a.channels:
            np.testing.assert_array_equal(b.channels[name], a.channels[name], err_msg=name)

    def test_l1_adapter(self, sample_p_file, arithmetic_diff_gain_copy):
        from odas_tpw.rsi.adapter import pfile_to_l1data
        from odas_tpw.rsi.p_file import PFile

        a = pfile_to_l1data(PFile(sample_p_file), speed=0.6)
        b = pfile_to_l1data(PFile(arithmetic_diff_gain_copy), speed=0.6)
        assert b.diff_gains == a.diff_gains and len(a.diff_gains) == 2

    def test_bad_buffer_repair_gains(self, sample_p_file, arithmetic_diff_gain_copy, monkeypatch):
        from odas_tpw.rsi import bad_buffer
        from odas_tpw.rsi.helpers import load_channels
        from odas_tpw.rsi.p_file import PFile

        # The file has no RDL dropouts; a clean mask is enough to reach the
        # diff_gain read on the repair path.
        monkeypatch.setattr(
            bad_buffer, "sample_masks",
            lambda report, n_fast, n_slow, **kw: {"sh1": np.zeros(n_fast, dtype=np.int8)},
        )
        a = load_channels(PFile(sample_p_file))["diff_gain_by_name"]
        b = load_channels(PFile(arithmetic_diff_gain_copy))["diff_gain_by_name"]
        assert b == a and len(a) == 5

    def test_adapter_bad_buffer_gains(self, sample_p_file, arithmetic_diff_gain_copy,
                                      monkeypatch):
        """The adapter's second diff_gain read, on its bad-buffer path."""
        from odas_tpw.rsi import bad_buffer
        from odas_tpw.rsi.adapter import pfile_to_l1data
        from odas_tpw.rsi.p_file import PFile

        seen = []

        def spy(**kw):
            seen.append(dict(kw["diff_gains"]))
            return np.zeros((len(kw["probe_names"]), len(kw["t_fast"])), dtype=bool), {}

        monkeypatch.setattr(
            bad_buffer, "sample_masks",
            lambda report, n_fast, n_slow, **kw: {"sh1": np.zeros(n_fast, dtype=np.int8)},
        )
        monkeypatch.setattr(bad_buffer, "probe_masks", spy)
        pfile_to_l1data(PFile(sample_p_file), speed=0.6)
        pfile_to_l1data(PFile(arithmetic_diff_gain_copy), speed=0.6)
        assert seen and seen[-1] == seen[0] and len(seen[0]) == 5

    def test_pre_emphasis_gain_warns_once_under_its_own_name(self, arithmetic_diff_gain_copy):
        """No converter reads a T1_dT1 diff_gain; deconvolution's read warns."""
        from odas_tpw.rsi.p_file import PFile

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            PFile(arithmetic_diff_gain_copy)
        hits = [x for x in w if str(x.message).startswith("T1_dT1: diff_gain")]
        assert len(hits) == 1
        assert hits[0].filename.endswith("p_file.py")  # the caller, not the parser

    @pytest.mark.parametrize("value", ["0", "-0.954", "0.9-12"])
    def test_non_positive_pre_emphasis_gain_names_the_channel(self, sample_p_file, tmp_path,
                                                              value):
        from odas_tpw.rsi.p_file import PFile

        bad = _patched_copy(sample_p_file, tmp_path / "bad.p", "T1_dT1", "diff_gain", value)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match=r"T1_dT1: diff_gain = .* must be positive"):
                PFile(bad)

    def test_one_arithmetic_coefficient_warns_once(self, sample_p_file, tmp_path):
        """T1_dT1 is converted with T1's stanza; only T1's conversion warns."""
        from odas_tpw.rsi.p_file import PFile

        bad = _patched_copy(sample_p_file, tmp_path / "beta.p", "T1", "beta_1", "3000+100")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            PFile(bad)
        hits = [str(x.message) for x in w if "is arithmetic" in str(x.message)]
        assert len(hits) == 1 and hits[0].startswith("T1: beta_1 = '3000+100'")

    def test_viewer(self, sample_p_file, arithmetic_diff_gain_copy):
        import matplotlib

        matplotlib.use("Agg")
        from odas_tpw.rsi.mixing_look import MixingLookViewer
        from odas_tpw.rsi.p_file import PFile

        a = MixingLookViewer(PFile(sample_p_file), fft_length=256, f_AA=98.0)
        b = MixingLookViewer(PFile(arithmetic_diff_gain_copy), fft_length=256, f_AA=98.0)
        assert b.diff_gains == a.diff_gains and len(a.diff_gains) == 2


class TestConvertRsijac:
    def test_temperature_matches_odas(self):
        from odas_tpw.rsi.channels import convert_rsijac_t

        counts = np.array([175842112.0, 268435456.0, 447612843.0])
        result, units = convert_rsijac_t(counts, _RSIJAC_T_PARAMS)
        assert units == "deg_C"
        np.testing.assert_allclose(
            result, [24.5399056830441, 15.2893, 4.72883771486283], rtol=0, atol=1e-11
        )

    def test_unit_count_is_t0(self):
        # ln(1) = 0, so a count equal to the division factor reads T_0.
        from odas_tpw.rsi.channels import convert_rsijac_t

        result, _ = convert_rsijac_t(np.array([2.0**28]), _RSIJAC_T_PARAMS)
        assert result[0] == pytest.approx(288.4393 - 273.15, abs=1e-12)

    def test_reads_division_factor_not_scale_factor(self):
        # ODAS ignores scale_factor; a different value there must change nothing.
        from odas_tpw.rsi.channels import convert_rsijac_t

        counts = np.array([175842112.0])
        base, _ = convert_rsijac_t(counts, _RSIJAC_T_PARAMS)
        other, _ = convert_rsijac_t(counts, {**_RSIJAC_T_PARAMS, "scale_factor": "2^20"})
        np.testing.assert_array_equal(base, other)

    def test_nonpositive_count_is_nan(self):
        from odas_tpw.rsi.channels import convert_rsijac_t

        result, _ = convert_rsijac_t(np.array([0.0, -5.0]), _RSIJAC_T_PARAMS)
        assert np.all(np.isnan(result))

    def test_conductivity_matches_odas(self):
        from odas_tpw.rsi.channels import convert_rsijac_c

        counts = np.array([346350063.0, 268435456.0, 217810216.0])
        result, units = convert_rsijac_c(counts, _RSIJAC_C_PARAMS)
        assert units == "mS_cm-1"
        np.testing.assert_allclose(
            result, [52.1426398880382, 40.4126532796923, 32.7910607425665], rtol=0, atol=1e-11
        )

    def test_conductivity_needs_cell_constant(self):
        from odas_tpw.rsi.channels import convert_rsijac_c

        params = {k: v for k, v in _RSIJAC_C_PARAMS.items() if k != "cell_constant"}
        with pytest.raises(ValueError, match="cell_constant is required"):
            convert_rsijac_c(np.array([2.0**28]), params)

    # Non-trivial a, b and division_factor, with gain and y_offset OMITTED so
    # ODAS's defaults apply. Expected values: ODAS v4.5.1 convert_odas.m on a
    # config string with exactly these stanzas (MATLAB, 2026-10-13). These pin
    # the order of operations (b*x + a after the division) and both defaults.
    def test_conductivity_polynomial_and_defaults_match_odas(self):
        from odas_tpw.rsi.channels import convert_rsijac_c

        params = {"name": "JAC_C", "division_factor": "2^27", "cell_constant": "2.7e-2",
                  "a": "0.01", "b": "1.002"}
        counts = np.array([346350063.0, 268435456.0, 217810216.0])
        result, _ = convert_rsijac_c(counts, params)
        np.testing.assert_allclose(
            result, [104.595784594346, 81.1566348148148, 65.9269798287512], rtol=0, atol=1e-10
        )

    def test_temperature_polynomial_matches_odas(self):
        from odas_tpw.rsi.channels import convert_rsijac_t

        params = {"name": "JAC_T", "division_factor": "2^27", "t_0": "288.4393",
                  "beta_1": "3905.963", "beta_2": "3.115573e5", "beta_3": "1.265211e7",
                  "a": "-0.003", "b": "0.998"}
        counts = np.array([175842112.0, 268435456.0, 447612843.0])
        result, _ = convert_rsijac_t(counts, params)
        np.testing.assert_allclose(
            result, [9.71797099107766, 1.19484829557075, -8.55265411850701], rtol=0, atol=1e-10
        )

    @pytest.mark.parametrize("key", ["t_0", "beta_1", "beta_2", "beta_3", "division_factor"])
    def test_temperature_zero_divisor_is_refused(self, key):
        from odas_tpw.rsi.channels import convert_rsijac_t

        params = {**_RSIJAC_T_PARAMS, "name": "JAC_T", key: "0"}
        with pytest.raises(ValueError, match=rf"JAC_T: {key} = 0 is a divisor"):
            convert_rsijac_t(np.array([2.0**28]), params)

    @pytest.mark.parametrize("key", ["cell_constant", "division_factor"])
    def test_conductivity_zero_divisor_is_refused(self, key):
        from odas_tpw.rsi.channels import convert_rsijac_c

        params = {**_RSIJAC_C_PARAMS, "name": "JAC_C", key: "0"}
        with pytest.raises(ValueError, match=rf"JAC_C: {key} = 0 is a divisor"):
            convert_rsijac_c(np.array([2.0**28]), params)

    def test_expression_coefficients_reach_provenance(self):
        # CalRecorder used float() and silently dropped "2^28" and "10.88/100".
        from odas_tpw.rsi.channels import CalRecorder, convert_rsijac_c

        rec = CalRecorder({**_RSIJAC_C_PARAMS, "name": "JAC_C"})
        convert_rsijac_c(np.array([2.0**28]), rec)
        cal = rec.calibration()
        assert cal["division_factor"] == 2.0**28
        assert cal["gain"] == pytest.approx(0.1088, rel=1e-15)
        assert cal["cell_constant"] == 2.692219e-2

    def test_rsi_tpw_nc_files_them_as_the_ctd(self):
        # convert._classify_channels used to know only jac_c/jac_t, so these
        # were written as supplementary channels and COND/TEMP_CTD came out empty.
        from odas_tpw.rsi.convert import _classify_channels

        class StubPF:
            def __init__(self):
                self.channels = {"JAC_C": object(), "JAC_T": object()}
                self.channel_info = {"JAC_C": {"type": "rsijac_c"}, "JAC_T": {"type": "rsijac_t"}}

            def is_fast(self, name):
                return False

        ch = _classify_channels(StubPF())
        assert ch["cond_ctd"] == ["JAC_C"]
        assert ch["temp_ctd"] == "JAC_T"
        assert "JAC_C" not in ch["supplementary"] and "JAC_T" not in ch["supplementary"]

    def test_registered_in_converters(self):
        from odas_tpw.rsi.channels import convert_rsijac_c, convert_rsijac_t

        assert CONVERTERS["rsijac_c"] is convert_rsijac_c
        assert CONVERTERS["rsijac_t"] is convert_rsijac_t
