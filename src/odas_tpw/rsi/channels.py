# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""
Channel conversion functions: raw counts -> physical units.

Ported from the ODAS MATLAB Library convert_odas.m.
"""

from typing import Any

import numpy as np


def _safe_float(s: Any, default: float = 0.0) -> float:
    """Convert *s* to float, returning *default* on failure."""
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


def _adis_14bit(data: np.ndarray) -> np.ndarray:
    """Extract 14-bit data from ADIS16209 inclinometer words.

    Matches ODAS ``adis.m``: clears status bits (15 = new-data,
    14 = error), then applies 14-bit two's complement.  The
    inclination X/Y channels use 14-bit signed data; the
    temperature channel uses 12-bit unsigned, which passes through
    the two's complement step unchanged.
    """
    val = data.copy().astype(np.float64)
    # Bit 15 set → new-data flag.  Clear it.
    mask = val < -(2**14)
    val[mask] += 2**15
    # Bit 14 set → error flag.  Clear it.
    mask = val >= 2**14
    val[mask] -= 2**14
    # Two's complement for the upper half of the 14-bit range.
    mask = val >= 2**13
    val[mask] -= 2**14
    return val


def _unsigned_16bit(data: np.ndarray) -> np.ndarray:
    """Convert signed int16 to unsigned by wrapping negative values.

    The caller may pass already-unsigned data (the PFile loader's
    upstream wrap step views int16 channels as uint16 in place); in that
    case the negative-mask is empty and the array is returned as-is
    (with a copy to keep the existing contract that callers may safely
    mutate the result).
    """
    if data.dtype.kind == "u":
        return data.copy()
    if data.dtype.kind == "i" and data.dtype.itemsize <= 2:
        # Native +2**16 would overflow int16; promote first.
        d = data.astype(np.int32)
    else:
        d = data.copy()
    d[d < 0] += 2**16
    return d


def convert_therm(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """FP07 thermistor via Steinhart-Hart equation and half-bridge circuit.

    Uses coefficients T_0, beta_1 (and optional beta_2, beta_3) from the
    channel config.  Matches ODAS ``convert_odas.m`` therm path.
    """
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
    """Shear probe: raw counts to velocity shear [s⁻¹].

    Formula: ``(adc_fs / 2^adc_bits * data + offset) / (2*sqrt(2)*diff_gain*sens)``.
    """
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
    """Polynomial conversion (pressure, etc.): coef0 + coef1*x + coef2*x² + …"""
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
    """Generic voltage channel: ``(adc_zero + data * adc_fs / 2^adc_bits) / gain`` → Volts."""
    adc_fs = _safe_float(params.get("adc_fs", "1"))
    adc_bits = _safe_float(params.get("adc_bits", "0"))
    gain = _safe_float(params.get("g", "1"))
    adc_zero = _safe_float(params.get("adc_zero", "0"))
    phys = (adc_zero + data * adc_fs / 2**adc_bits) / gain
    return phys, "V"


def convert_piezo(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Piezo accelerometer: subtract zero-offset ``a_0``."""
    a_0 = _safe_float(params.get("a_0", "0"))
    return data - a_0, "counts"


def convert_inclxy(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """ADIS inclinometer X or Y: 14-bit two's complement → degrees."""
    val = _adis_14bit(data)
    coef0 = _safe_float(params.get("coef0", "0"))
    coef1 = _safe_float(params.get("coef1", "0.025"))
    return coef1 * val + coef0, "deg"


def convert_inclt(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """ADIS inclinometer temperature: 14-bit two's complement → °C."""
    val = _adis_14bit(data)
    coef0 = _safe_float(params.get("coef0", "624"))
    coef1 = _safe_float(params.get("coef1", "-0.47"))
    return coef1 * val + coef0, "deg_C"


def convert_jac_c(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """JAC conductivity: ratio of I/V parts from 32-bit combined word → mS/cm."""
    i_part = np.floor(data / 2**16)
    v_part = np.mod(data, 2**16)
    v_part[v_part == 0] = 1
    ratio = i_part / v_part
    a = _safe_float(params.get("a"))
    b = _safe_float(params.get("b"))
    c = _safe_float(params.get("c"))
    return np.polyval([c, b, a], ratio), "mS_cm-1"


def convert_jac_t(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """JAC temperature: unsigned 16-bit wrapping + 5th-order polynomial → °C."""
    d = _unsigned_16bit(data)
    a = _safe_float(params.get("a"))
    b = _safe_float(params.get("b"))
    c = _safe_float(params.get("c"))
    d_coef = _safe_float(params.get("d"))
    e = _safe_float(params.get("e"))
    f = _safe_float(params.get("f"))
    return np.polyval([f, e, d_coef, c, b, a], d), "deg_C"


def convert_raw(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Passthrough: return raw counts unchanged."""
    return data, "counts"


def convert_aroft_o2(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """RINKO FT dissolved oxygen: unsigned 16-bit wrapping / 100 → µmol/L."""
    d = _unsigned_16bit(data)
    return d / 100.0, "umol_L-1"


def convert_aroft_t(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """RINKO FT temperature: unsigned 16-bit wrapping / 1000 - 5 -> deg C."""
    d = _unsigned_16bit(data)
    return d / 1000.0 - 5.0, "deg_C"


def convert_aem1g_a(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """AEM1-G electromagnetic current meter, analog output → m/s.

    Matches ODAS ``convert_odas.m`` ``odas_aem1g_a_internal``:
      1. Convert raw counts to voltage: ``V = adc_zero + data * (adc_fs / 2^adc_bits)``
      2. Apply calibration: ``V = a/100 + b/100 * V``  (cm/s → m/s)
      3. Subtract bias: ``physical = V - bias``
    """
    adc_fs = _safe_float(params.get("adc_fs", "4.096"))
    adc_bits = _safe_float(params.get("adc_bits", "16"))
    adc_zero = _safe_float(params.get("adc_zero", str(adc_fs / 2)))
    a = _safe_float(params.get("a", "0")) / 100.0
    b = _safe_float(params.get("b", "1")) / 100.0
    bias = _safe_float(params.get("bias", "0"))
    V = adc_zero + data * (adc_fs / 2**adc_bits)
    V = a + b * V
    return V - bias, "m_s-1"


def convert_aem1g_d(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """AEM1-G electromagnetic current meter, digital/RS232 output → m/s.

    Matches ODAS ``convert_odas.m`` ``odas_aem1g_d_internal``:
      1. Convert signed int16 to unsigned (wrap negatives)
      2. Apply calibration: ``physical = a/100 + b/100 * data``  (cm/s → m/s)
    """
    d = _unsigned_16bit(data)
    a = _safe_float(params.get("a", "0")) / 100.0
    b = _safe_float(params.get("b", "1")) / 100.0
    return a + b * d, "m_s-1"


def convert_alec_emc(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Alec Electronics electromagnetic current meter → m/s.

    Matches ODAS ``convert_odas.m`` ``odas_Alec_EMC_internal``:
      ``physical = coef1 * data + coef0``
    """
    coef0 = _safe_float(params.get("coef0", "0"))
    coef1 = _safe_float(params.get("coef1", "1"))
    return np.polyval([coef1, coef0], data), "m_s-1"


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
    "aem1g_a": convert_aem1g_a,
    "aem1g_d": convert_aem1g_d,
    "alec_emc": convert_alec_emc,
    "jac_emc": convert_aem1g_a,  # deprecated alias
}
