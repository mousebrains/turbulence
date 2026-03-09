# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for channel conversion functions."""

import numpy as np

from rsi_python.channels import (
    convert_poly,
    convert_raw,
    convert_shear,
    convert_therm,
)


class TestConvertTherm:
    def test_synthetic_counts_to_temperature(self):
        """Synthetic raw counts with known Steinhart-Hart params → temperature in 0–30 °C."""
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
        from rsi_python.channels import _safe_float

        assert _safe_float("invalid") == 0.0

    def test_invalid_string_custom_default(self):
        from rsi_python.channels import _safe_float

        assert _safe_float("invalid", default=42.0) == 42.0

    def test_none(self):
        from rsi_python.channels import _safe_float

        assert _safe_float(None) == 0.0

    def test_none_custom_default(self):
        from rsi_python.channels import _safe_float

        assert _safe_float(None, default=-1.0) == -1.0

    def test_valid_string(self):
        from rsi_python.channels import _safe_float

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
