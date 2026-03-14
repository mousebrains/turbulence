# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for channel conversion functions."""

import numpy as np

from microstructure_tpw.rsi.channels import (
    CONVERTERS,
    convert_aroft_o2,
    convert_aroft_t,
    convert_inclt,
    convert_inclxy,
    convert_jac_c,
    convert_jac_t,
    convert_piezo,
    convert_poly,
    convert_raw,
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


class TestSafeFloat:
    def test_invalid_string(self):
        from microstructure_tpw.rsi.channels import _safe_float

        assert _safe_float("invalid") == 0.0

    def test_invalid_string_custom_default(self):
        from microstructure_tpw.rsi.channels import _safe_float

        assert _safe_float("invalid", default=42.0) == 42.0

    def test_none(self):
        from microstructure_tpw.rsi.channels import _safe_float

        assert _safe_float(None) == 0.0

    def test_none_custom_default(self):
        from microstructure_tpw.rsi.channels import _safe_float

        assert _safe_float(None, default=-1.0) == -1.0

    def test_valid_string(self):
        from microstructure_tpw.rsi.channels import _safe_float

        assert _safe_float("3.14") == 3.14


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

    def test_defaults(self):
        """Default params: adc_fs=1, adc_bits=0, g=1, adc_zero=0."""
        data = np.array([3.0])
        result, units = convert_voltage(data, {})
        assert units == "V"
        # (0 + 3 * 1 / 2^0) / 1 = 3.0
        np.testing.assert_allclose(result[0], 3.0, atol=1e-12)


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
    def test_positive_14bit(self):
        """Positive 14-bit two's complement value with known coefficients."""
        # Raw value where right-shift 2 gives a small positive number.
        # raw=100 -> val = 100 >> 2 = 25 (positive, < 2^13=8192)
        # With default coef1=0.025, coef0=0: result = 0.025 * 25 + 0 = 0.625
        params = {"coef0": "0", "coef1": "0.025"}
        data = np.array([100.0])
        result, units = convert_inclxy(data, params)
        assert units == "deg"
        np.testing.assert_allclose(result[0], 0.025 * 25, rtol=1e-10)

    def test_negative_twos_complement(self):
        """Negative value via two's complement wrapping (val >= 2^13)."""
        # We need val = raw >> 2 >= 8192, so raw >= 8192 * 4 = 32768
        # raw=32768 -> val = 32768 >> 2 = 8192 -> wraps: 8192 - 16384 = -8192
        # result = 0.025 * (-8192) + 0 = -204.8
        params = {"coef0": "0", "coef1": "0.025"}
        data = np.array([32768.0])
        result, units = convert_inclxy(data, params)
        assert units == "deg"
        np.testing.assert_allclose(result[0], 0.025 * (-8192), rtol=1e-10)

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
        data = np.array([40.0])  # val = 40 >> 2 = 10
        result, units = convert_inclxy(data, params)
        assert units == "deg"
        np.testing.assert_allclose(result[0], 0.05 * 10 + 10.0, rtol=1e-10)

    def test_defaults(self):
        """Default coefficients: coef0=0, coef1=0.025."""
        data = np.array([8.0])  # val = 8 >> 2 = 2
        result, units = convert_inclxy(data, {})
        assert units == "deg"
        np.testing.assert_allclose(result[0], 0.025 * 2, rtol=1e-10)


class TestConvertInclT:
    def test_positive_value(self):
        """Positive 14-bit value with default coefficients (coef0=624, coef1=-0.47)."""
        # raw=100 -> val = 100 >> 2 = 25
        # result = -0.47 * 25 + 624 = -11.75 + 624 = 612.25
        data = np.array([100.0])
        result, units = convert_inclt(data, {})
        assert units == "deg_C"
        np.testing.assert_allclose(result[0], -0.47 * 25 + 624, rtol=1e-10)

    def test_negative_twos_complement(self):
        """Negative two's complement wrapping."""
        # raw=32768 -> val = 8192 -> wraps to -8192
        # result = -0.47 * (-8192) + 624 = 3850.24 + 624 = 4474.24
        data = np.array([32768.0])
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
        data = np.array([40.0])  # val = 40 >> 2 = 10
        result, units = convert_inclt(data, params)
        assert units == "deg_C"
        np.testing.assert_allclose(result[0], -0.5 * 10 + 500, rtol=1e-10)


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

    def test_zero_v_part_protection(self):
        """When v_part is 0, it is replaced by 1 to avoid division by zero."""
        # data = 5 * 2^16 + 0 = 327680: i_part=5, v_part=0 -> v_part=1, ratio=5
        # poly: 0*25 + 0*5 + 0 = 0
        params = {"a": "0", "b": "0", "c": "0"}
        data = np.array([5 * 2**16])
        result, units = convert_jac_c(data, params)
        assert units == "mS_cm-1"
        assert np.all(np.isfinite(result))

    def test_zero_v_part_with_nonzero_coeffs(self):
        """Zero v_part with non-trivial coefficients uses ratio = i_part/1."""
        # data = 4 * 2^16: i_part=4, v_part=0 -> 1, ratio=4
        # poly: 1*16 + 0*4 + 0 = 16
        params = {"a": "0", "b": "0", "c": "1.0"}
        data = np.array([4.0 * 2**16])
        result, units = convert_jac_c(data, params)
        assert units == "mS_cm-1"
        np.testing.assert_allclose(result[0], 16.0, rtol=1e-10)

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
        """Positive values divided by 100."""
        data = np.array([25000.0, 10000.0, 0.0])
        result, units = convert_aroft_o2(data, {})
        assert units == "umol_L-1"
        np.testing.assert_allclose(result[0], 250.0, rtol=1e-10)
        np.testing.assert_allclose(result[1], 100.0, rtol=1e-10)
        np.testing.assert_allclose(result[2], 0.0, atol=1e-12)

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
        """Positive values: d/1000 - 5."""
        data = np.array([10000.0, 5000.0, 0.0])
        result, units = convert_aroft_t(data, {})
        assert units == "deg_C"
        np.testing.assert_allclose(result[0], 5.0, rtol=1e-10)
        np.testing.assert_allclose(result[1], 0.0, atol=1e-12)
        np.testing.assert_allclose(result[2], -5.0, rtol=1e-10)

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
        }
        assert expected.issubset(set(CONVERTERS.keys()))
