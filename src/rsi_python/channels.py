# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""
Channel conversion functions: raw counts -> physical units.

Ported from the ODAS MATLAB Library convert_odas.m.
"""

from typing import Any

import numpy as np


def _safe_float(s: Any, default: float = 0.0) -> float:
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


def convert_therm(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """FP07 thermistor: Steinhart-Hart via half-bridge."""
    a = _safe_float(params.get("a", "0"))
    b = _safe_float(params.get("b", "1"))
    adc_fs = _safe_float(params.get("adc_fs", "4.096"))
    adc_bits = _safe_float(params.get("adc_bits", "16"))
    G = _safe_float(params.get("g", "6.0"))
    E_B = _safe_float(params.get("e_b", "0.68"))
    T_0 = _safe_float(params.get("t_0", "289.0"))
    beta_1 = _safe_float(params.get("beta_1", "3000"))
    beta_2 = params.get("beta_2")

    Z = ((data - a) / b) * (adc_fs / 2**adc_bits) * 2 / (G * E_B)
    Z = np.clip(Z, -0.6, 0.6)
    R_ratio = (1 - Z) / (1 + Z)
    log_R = np.log(R_ratio)

    inv_T = 1.0 / T_0 + (1.0 / beta_1) * log_R
    if beta_2 is not None:
        inv_T += (1.0 / _safe_float(beta_2)) * log_R**2
        beta_3 = params.get("beta_3")
        if beta_3 is not None:
            inv_T += (1.0 / _safe_float(beta_3)) * log_R**3
    return 1.0 / inv_T - 273.15, "deg_C"


def convert_shear(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Shear probe."""
    adc_fs = _safe_float(params.get("adc_fs", "4.096"))
    adc_bits = _safe_float(params.get("adc_bits", "16"))
    diff_gain = _safe_float(params.get("diff_gain", "1"))
    sens = _safe_float(params.get("sens", "1"))
    adc_zero = _safe_float(params.get("adc_zero", "0"))
    sig_zero = _safe_float(params.get("sig_zero", "0"))
    phys = (adc_fs / 2**adc_bits) * data + (adc_zero - sig_zero)
    phys = phys / (2 * np.sqrt(2) * diff_gain * sens)
    return phys, "s-1"


def convert_poly(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Polynomial conversion (pressure, etc.)."""
    coeffs = []
    for i in range(10):
        key = f"coef{i}"
        if key in params:
            coeffs.append(_safe_float(params[key]))
        else:
            break
    if not coeffs:
        return data, "counts"
    phys = np.polyval(coeffs[::-1], data)
    units = params.get("units", "").strip("[]")
    return phys, units or "unknown"


def convert_voltage(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    adc_fs = _safe_float(params.get("adc_fs", "1"))
    adc_bits = _safe_float(params.get("adc_bits", "0"))
    gain = _safe_float(params.get("g", "1"))
    adc_zero = _safe_float(params.get("adc_zero", "0"))
    phys = (adc_zero + data * adc_fs / 2**adc_bits) / gain
    return phys, "V"


def convert_piezo(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    a_0 = _safe_float(params.get("a_0", "0"))
    return data - a_0, "counts"


def convert_inclxy(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """ADIS inclinometer X or Y."""
    raw = data.astype(np.int32)
    val = (raw >> 2).astype(np.float64)
    val[val >= 2**13] -= 2**14
    coef0 = _safe_float(params.get("coef0", "0"))
    coef1 = _safe_float(params.get("coef1", "0.025"))
    return coef1 * val + coef0, "deg"


def convert_inclt(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """ADIS inclinometer temperature."""
    raw = data.astype(np.int32)
    val = (raw >> 2).astype(np.float64)
    val[val >= 2**13] -= 2**14
    coef0 = _safe_float(params.get("coef0", "624"))
    coef1 = _safe_float(params.get("coef1", "-0.47"))
    return coef1 * val + coef0, "deg_C"


def convert_jac_c(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """JAC conductivity (32-bit combined channel)."""
    i_part = np.floor(data / 2**16)
    v_part = np.mod(data, 2**16)
    v_part[v_part == 0] = 1
    ratio = i_part / v_part
    a = _safe_float(params.get("a"))
    b = _safe_float(params.get("b"))
    c = _safe_float(params.get("c"))
    return np.polyval([c, b, a], ratio), "mS_cm-1"


def convert_jac_t(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """JAC temperature."""
    d = data.copy()
    d[d < 0] = d[d < 0] + 2**16
    a = _safe_float(params.get("a"))
    b = _safe_float(params.get("b"))
    c = _safe_float(params.get("c"))
    d_coef = _safe_float(params.get("d"))
    e = _safe_float(params.get("e"))
    f = _safe_float(params.get("f"))
    return np.polyval([f, e, d_coef, c, b, a], d), "deg_C"


def convert_raw(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    return data, "counts"


def convert_aroft_o2(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """RINKO FT dissolved oxygen."""
    d = data.copy()
    d[d < 0] = d[d < 0] + 2**16
    return d / 100.0, "umol_L-1"


def convert_aroft_t(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """RINKO FT temperature."""
    d = data.copy()
    d[d < 0] = d[d < 0] + 2**16
    return d / 1000.0 - 5.0, "deg_C"


CONVERTERS = {
    "therm": convert_therm,
    "shear": convert_shear,
    "poly": convert_poly,
    "voltage": convert_voltage,
    "piezo": convert_piezo,
    "inclxy": convert_inclxy,
    "inclt": convert_inclt,
    "jac_c": convert_jac_c,
    "jac_t": convert_jac_t,
    "raw": convert_raw,
    "aroft_o2": convert_aroft_o2,
    "aroft_t": convert_aroft_t,
    "gnd": convert_raw,
}
