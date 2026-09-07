# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""
Channel conversion functions: raw counts -> physical units.

Ported from the ODAS MATLAB Library convert_odas.m.
"""

import warnings
from typing import Any

import numpy as np


def _safe_float(s: Any, default: float = 0.0) -> float:
    """Convert *s* to float, returning *default* on failure."""
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


# --- Calibration provenance ----------------------------------------------

# Structural config keys that a converter reads but that are not calibration
# coefficients: ``name`` (error messages and the range check) and ``units``
# (convert_poly's declared output unit).  Everything else a converter reads and
# that parses as a finite number is provenance.
#
# Both are normally non-numeric, so ``float()`` would reject them anyway — but
# that is an accident of their usual values, not a rule.  A channel named "2"
# is legal in an RSI config and would otherwise be recorded as a coefficient
# called "name".  Listing only the keys that are actually read keeps the set
# honest: a longer speculative list looks like protection while testing as
# dead code.
_NON_CAL_KEYS = frozenset({"name", "units"})


class CalRecorder(dict[str, Any]):
    """A channel-config dict that remembers which keys a converter read.

    Provenance by observation rather than by a hand-maintained table: the
    recorded set is, by construction, exactly the keys the converter consulted
    and found, so it cannot drift out of step with the conversion the way a
    parallel "coefficients for sensor type X" list would.  Add a coefficient
    to a converter and it documents itself; stop reading one and the attribute
    disappears on its own.

    Keys probed but *absent* are not recorded.  :func:`convert_poly` walks
    ``coef0``..``coef9`` to find the end of the polynomial, and a missing
    ``coef7`` is not provenance — it is the loop's exit condition.

    The idea is Jesse Cusack's (pyturb writes the coefficients it used onto
    each variable); the recording is ours, because our converters take the
    config as a plain dict and we would rather not maintain the mapping twice.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._used: set[str] = set()

    def __getitem__(self, key: str) -> Any:
        value = super().__getitem__(key)  # raises before recording if absent
        self._used.add(key)
        return value

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        if dict.__contains__(self, key):
            self._used.add(key)
        return dict.get(self, key, default)

    def __contains__(self, key: object) -> bool:
        present = super().__contains__(key)
        if present and isinstance(key, str):
            self._used.add(key)
        return present

    def calibration(self) -> dict[str, float]:
        """The numeric coefficients this conversion took FROM THE CONFIG.

        Not every number that entered the arithmetic: a key the config omits,
        where the converter falls back to a documented default (``adc_fs``,
        ``adc_bits``), is absent here.  That is the more useful of the two
        readings — it separates what the instrument declared from what we
        assumed on its behalf — but it does mean "no ``cal_adc_fs``" reads as
        "the config was silent", not "no ADC scaling was applied".
        """
        out: dict[str, float] = {}
        for key in sorted(self._used):
            if key in _NON_CAL_KEYS:
                continue
            try:
                value = float(dict.__getitem__(self, key))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                out[key] = value
        return out


def _require_float(params: dict[str, Any], key: str, default: float, sensor: str) -> float:
    """Like _safe_float, but warn loudly when a sensor-specific calibration
    coefficient is missing from the channel config.

    ODAS convert_odas.m errors outright in this situation; silently
    substituting a generic default would produce plausible-looking but
    wrong physical units (e.g. a missing shear ``sens`` scales epsilon by
    the square of the sensitivity error).
    """
    if key not in params or params.get(key) in (None, ""):
        warnings.warn(
            f"{sensor}: calibration coefficient '{key}' missing from channel "
            f"config; using default {default} — physical units are suspect",
            stacklevel=3,
        )
        return default
    return _safe_float(params[key], default)


def _parse_finite_float(
    raw: Any,
    params: dict[str, Any],
    key: str,
    default: float,
    sensor: str,
    positive: bool,
    nonzero: bool = False,
) -> float:
    """Parse a PRESENT calibration coefficient, or raise.

    ``_safe_float`` swallows a parse failure and returns the caller's generic
    default, so a present-but-malformed coefficient is indistinguishable from a
    real calibration.  A stray decimal comma (``diff_gain = 0,09``) silently
    replaced 0.09 with 1.0, scaling shear by 0.09x and shear VARIANCE — hence
    epsilon before its iterative correction — by 0.0081x, with no warning at all
    (issue #180 F06).  Someone wrote a value here; we cannot honour it, so we
    refuse rather than invent one.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"{sensor} channel {params.get('name', '?')}: calibration coefficient "
            f"'{key}' is present but unparseable ({raw!r}); refusing to substitute "
            f"the default {default} and fabricate physical units. Fix the value in "
            "the instrument config (a decimal comma is the usual cause), or remove "
            "the key entirely to accept the documented default."
        ) from None
    if not np.isfinite(value) or (positive and value <= 0) or (nonzero and value == 0):
        if positive:
            need = "a finite positive number"
        elif nonzero:
            need = "a finite non-zero number"
        else:
            need = "a finite number"
        extra = ""
        if nonzero and value == 0:
            # The Steinhart-Hart betas are RECIPROCALS: zero is an infinite
            # term, not a deleted one. 1e30 is bit-identical to omitting the key.
            extra = (
                f" ('{key}' is a reciprocal coefficient: zero means an INFINITE "
                "term, not a deleted one — use 1e30, or remove the key)"
            )
        raise ValueError(
            f"{sensor} channel {params.get('name', '?')}: calibration coefficient "
            f"'{key}' is {raw!r}; must be {need}{extra}. Refusing to fabricate "
            "physical units."
        )
    return value


def _require_finite_float(
    params: dict[str, Any],
    key: str,
    default: float,
    sensor: str,
    *,
    positive: bool = False,
    nonzero: bool = False,
) -> float:
    """Strict :func:`_require_float`: missing warns, malformed raises.

    Missing stays a warning — legacy corpora genuinely omit keys and the
    caller's default is the documented fallback.  See :func:`_parse_finite_float`
    for why present-but-wrong must not be defaulted.
    """
    if key not in params or params.get(key) in (None, ""):
        warnings.warn(
            f"{sensor}: calibration coefficient '{key}' missing from channel "
            f"config; using default {default} — physical units are suspect",
            stacklevel=3,
        )
        return default
    return _parse_finite_float(params[key], params, key, default, sensor, positive, nonzero)


def _optional_finite_float(
    params: dict[str, Any],
    key: str,
    default: float,
    sensor: str,
    *,
    positive: bool = False,
    nonzero: bool = False,
) -> float:
    """Strict parse for a key with a universal default (ADC scaling, offsets).

    Absent is silent — ``adc_fs``/``adc_bits`` carry instrument-independent
    defaults and warning on every channel that omits them would be noise.
    Present-but-malformed still raises: these scale every converted sample.
    """
    if key not in params or params.get(key) in (None, ""):
        return default
    return _parse_finite_float(params[key], params, key, default, sensor, positive, nonzero)


def _adis_14bit(data: np.ndarray) -> np.ndarray:
    """Extract 14-bit data from ADIS16209 inclinometer words.

    Matches ODAS ``adis.m``: clears status bits (15 = new-data,
    14 = error), then applies 14-bit two's complement.  The
    inclination X/Y channels use 14-bit signed data; the
    temperature channel uses 12-bit unsigned, which passes through
    the two's complement step unchanged.

    Note: the strict ``< -2**14`` test (kept to match ``adis.m:48``)
    mis-decodes error-flagged words 0xC000-0xDFFF (bit14+bit13 set). This is a
    faithful port of an ODAS quirk; it touches only the auxiliary inclinometer
    channels, never shear/temperature, and no such words occur in the ARCTERX
    data (verified across all 29 files). Left as-is for ODAS parity (#1).
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


def _deglitch_rs232(d: np.ndarray) -> np.ndarray:
    """Forward-fill obvious RS232-receiver glitches (RINKO FT sensors).

    Faithful port of the "Temporary fix of a buggy RS232 receiver" in ODAS
    ``convert_odas.m`` (odas_aroft_o2_internal / odas_aroft_t_internal): any
    sample sitting at/near the unsigned rails (``<= 50`` or ``>= 2^16 - 50``)
    is replaced by the preceding sample. The first sample is never replaced
    (ODAS loops from index 2, 1-based), and a run of consecutive glitches all
    inherit the last good value. Applied to the UNSIGNED-wrapped values, i.e.
    after :func:`_unsigned_16bit`, matching the ODAS ordering. (#104 U1-2.)
    """
    out = np.asarray(d, dtype=np.float64).copy()
    if out.size == 0:
        return out
    glitch = (out <= 50) | (out >= 2**16 - 50)
    glitch[0] = False  # ODAS keeps the first sample regardless
    # Standard forward-fill: map each glitch to the last non-glitch index.
    idx = np.where(glitch, 0, np.arange(out.size))
    np.maximum.accumulate(idx, out=idx)
    return out[idx]


def convert_therm(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """FP07 thermistor via Steinhart-Hart equation and half-bridge circuit.

    Uses coefficients T_0, beta_1 (and optional beta_2, beta_3) from the
    channel config.  Matches ODAS ``convert_odas.m`` therm path.
    """
    a = _optional_finite_float(params, "a", 0.0, "therm")
    b = _optional_finite_float(params, "b", 1.0, "therm", nonzero=True)
    adc_fs = _optional_finite_float(params, "adc_fs", 4.096, "therm", positive=True)
    adc_bits = _optional_finite_float(params, "adc_bits", 16.0, "therm", positive=True)
    # g (gain) and e_b (bridge excitation) scale the bridge resistance and thus
    # temperature directly — warn if the config omits them rather than silently
    # using a default that may not match the instrument (as t_0 already does).
    # Strict on a present-but-malformed value: every one of these divides or
    # scales the Steinhart-Hart result (issue #180 F06).
    G = _require_finite_float(params, "g", 6.0, "therm", positive=True)
    E_B = _require_finite_float(params, "e_b", 0.68, "therm", positive=True)
    T_0 = _require_finite_float(params, "t_0", 289.0, "therm", positive=True)
    # `beta` and `beta_1` are mutually exclusive alternatives for the linear
    # Steinhart-Hart term (beta = legacy single-coeff form, beta_1 = newer
    # multi-coeff form; beta_2/beta_3 are additive higher-order terms applied
    # regardless). ODAS convert_odas.m:501-507 checks `beta` FIRST; match that
    # precedence rather than preferring beta_1 (#4).
    if "beta" in params:
        beta_1 = _parse_finite_float(
            params["beta"], params, "beta", 3000.0, "therm", False, nonzero=True
        )
    elif "beta_1" in params:
        beta_1 = _parse_finite_float(
            params["beta_1"], params, "beta_1", 3000.0, "therm", False, nonzero=True
        )
    else:
        beta_1 = _require_finite_float(params, "beta_1", 3000.0, "therm", nonzero=True)
    beta_2 = params.get("beta_2")

    Z = ((data - a) / b) * (adc_fs / 2**adc_bits) * 2 / (G * E_B)
    Z = np.clip(Z, -0.6, 0.6)
    R_ratio = (1 - Z) / (1 + Z)
    log_R = np.log(R_ratio)

    inv_T = 1.0 / T_0 + (1.0 / beta_1) * log_R
    # beta_2/beta_3 are RECIPROCALS in the Steinhart-Hart sum, so a malformed
    # value defaulting to 0.0 raised ZeroDivisionError from inside an arithmetic
    # expression rather than naming the bad coefficient. The documented way to
    # omit a term is a huge value (1e30), not zero — see CLAUDE.md.
    if beta_2 not in (None, ""):
        inv_T += (
            1.0 / _parse_finite_float(beta_2, params, "beta_2", 1e30, "therm", False, nonzero=True)
        ) * log_R**2
        beta_3 = params.get("beta_3")
        if beta_3 not in (None, ""):
            inv_T += (
                1.0
                / _parse_finite_float(beta_3, params, "beta_3", 1e30, "therm", False, nonzero=True)
            ) * log_R**3
    return 1.0 / inv_T - 273.15, "deg_C"


# --- Plausible-range checks on calibration coefficients -------------------
#
# _parse_finite_float fails closed on a value that cannot be a number at all.
# It cannot catch a value that parses cleanly, is finite and positive, and is
# still physically impossible — the un-filled ``sens = 1.0`` placeholder being
# the canonical case: it converts without complaint and scales epsilon by
# sens^-2, roughly 200x on a real probe.  The idea (and the shear-sensitivity
# bounds) come from Jesse Cusack's pyturb ``_pfile/convert.py``.
#
# Bounds are checked against our own corpus rather than adopted on faith.  The
# shear inventory (15998 channel-configs across ARCTERX, SUNRISE, RIOT, CASPER,
# Taiwan, ASTRAL, Keck and goflow; /Volumes/SeaChest/Shear Inventory/
# shear_sensors.csv, read 2026-09-07) gives:
#
#   sens       min 0.041   max 0.123   median 0.1041   67 distinct values
#   diff_gain  min 0.09    max 1.01    median 0.941    28 distinct values
#
# so [0.03, 0.15] V*s/m brackets every real probe we have ever deployed with
# ~35% margin at the low end and ~22% at the high end, and fires on none of
# them.
_SHEAR_SENS_MIN = 0.03
_SHEAR_SENS_MAX = 0.15

# diff_gain deliberately gets a much wider bound.  The corpus is BIMODAL: 2236
# of 15998 rows (14%) sit at 0.090-0.099 and the remaining 13762 at 0.905-1.01,
# and the low band is a real population rather than corruption.  What separates
# them is the SAMPLING RATE: every 1024 Hz instrument is low (MR 330, 429 —
# high-energy/tidal builds), every 512 Hz instrument is high (13 of them).  The
# comparison that isolates it is same-model: MR1000RDL-EM reads 0.927/0.941 on
# the 512 Hz gliders SN 433/435 and 0.099/0.094 on the 1024 Hz SN 429.
#
# The ~10x gain step is NOT the 2x rate step, so the rate marks a different
# differentiator build, not a scaling law; vehicle class is confounded with it
# and is not the driver.  Either way a tight band around the upper mode would
# flag a seventh of every shear channel we own, and an alarm that fires on a
# seventh of the corpus is an alarm that gets muted.  This converter sees only
# the CHANNEL config — not the rate, not the vehicle — so it could not apply a
# class-aware bound even if we wanted one.
#
# Telling "0.09 is this instrument's real differentiator" from "0.09 is a typo"
# therefore needs an instrument-keyed comparison, which is what ``rsi-tpw
# sensors --diff-gain`` (odas_tpw.rsi.diff_gain) exists to do.  SN 132 is the
# one instrument the rate does NOT explain: 511.95 Hz on a 10-column
# vmp-250-IR, the same sampling configuration as SN 412/465/479 at 0.937-0.99
# (issue #178).  The bound here is only a floor/ceiling for a value that cannot
# be a differentiator gain at all — it clears the observed extremes by roughly
# a factor of 9 on each side.
_SHEAR_DIFF_GAIN_MIN = 0.01
_SHEAR_DIFF_GAIN_MAX = 10.0


def _check_plausible_range(
    value: float,
    key: str,
    sensor: str,
    lo: float,
    hi: float,
    units: str,
    name: str,
) -> None:
    """Warn when a well-formed calibration coefficient is outside its range.

    Deliberately a warning, not an error.  The value parsed, so it may be a
    real coefficient for hardware we have not seen; refusing to convert would
    make an unfamiliar-but-valid instrument unreadable.  A warning is
    recoverable, an exception is not, and the operator is the one who can tell
    the two cases apart.
    """
    if lo <= value <= hi:
        return
    warnings.warn(
        f"{sensor} channel {name}: calibration coefficient '{key}'={value:g} is "
        f"outside the plausible range [{lo:g}, {hi:g}] {units}. The value parsed "
        "cleanly, so it is used as-is and physical units follow from it — but "
        "check the instrument config: an un-filled placeholder or a "
        "mis-transcribed coefficient looks exactly like this.",
        stacklevel=3,
    )


def convert_shear(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Shear probe: raw counts to velocity shear [s⁻¹].

    Formula: ``(adc_fs / 2^adc_bits * data + offset) / (2*sqrt(2)*diff_gain*sens)``.

    A missing probe sensitivity ``sens`` is a hard error (ODAS convert_odas.m
    parity — it errors outright): a fabricated default of 1.0 would produce
    plausible-looking shear that scales epsilon by sens⁻², and legacy v1
    corpora genuinely lack sens in their setup files (issue #141).
    """
    adc_fs = _optional_finite_float(params, "adc_fs", 4.096, "shear", positive=True)
    adc_bits = _optional_finite_float(params, "adc_bits", 16.0, "shear", positive=True)
    # Strict: diff_gain divides the shear amplitude, so a malformed value that
    # silently became 1.0 rescaled shear variance by (diff_gain)^2 with no
    # warning — 123x on a real 0.09 probe (issue #180 F06).
    diff_gain = _require_finite_float(params, "diff_gain", 1.0, "shear", positive=True)
    _check_plausible_range(
        diff_gain,
        "diff_gain",
        "shear",
        _SHEAR_DIFF_GAIN_MIN,
        _SHEAR_DIFF_GAIN_MAX,
        "(dimensionless)",
        str(params.get("name", "?")),
    )
    # Strict parse: a present-but-unparseable sens (stray comma, typo) must
    # not fall through _safe_float's default — that silently fabricates
    # sens=1.0 and scales epsilon by sens^-2 (~125x on real probes).
    # `not np.isfinite(...)` (rather than `<= 0` alone) also rejects
    # sens="nan" — every comparison with NaN is False, so it would sail
    # through a sign check into all-NaN shear — and sens="inf" (zero shear).
    raw_sens = params.get("sens")
    sens: float | None
    try:
        sens = float(raw_sens) if raw_sens not in (None, "") else None
    except (TypeError, ValueError):
        sens = None
    if sens is None or not np.isfinite(sens) or sens <= 0:
        problem = (
            "missing from the channel config"
            if raw_sens in (None, "")
            else f"unusable ({raw_sens!r}; must be a finite positive number)"
        )
        raise ValueError(
            f"shear channel {params.get('name', '?')}: calibration coefficient "
            f"'sens' {problem}; refusing to fabricate "
            "physical shear. Inject the probe sensitivity with 'rsi-tpw "
            "patch-config' (--add-keys), or — for legacy v1 corpora — "
            "re-translate with 'rsi-tpw v1to6 --sens' or add "
            "'sh1_sens:'/'sh2_sens:' keys to the setup file (issue #141)."
        )
    _check_plausible_range(
        sens,
        "sens",
        "shear",
        _SHEAR_SENS_MIN,
        _SHEAR_SENS_MAX,
        "V*s/m",
        str(params.get("name", "?")),
    )
    adc_zero = _optional_finite_float(params, "adc_zero", 0.0, "shear")
    sig_zero = _optional_finite_float(params, "sig_zero", 0.0, "shear")
    phys = (adc_fs / 2**adc_bits) * data + (adc_zero - sig_zero)
    phys = phys / (2 * np.sqrt(2) * diff_gain * sens)
    # NB: this is the ODAS intermediate — still MISSING the /speed^2 fall-rate
    # normalization that makes it true physical shear (applied exactly once
    # downstream in helpers.py + adapter). The units stay the UDUNITS-valid
    # "s-1" so CF-declared per-profile files remain compliant; the
    # pre-normalization caveat is surfaced CF-legally via the sh1/sh2 ``comment``
    # attribute (set during channel conversion in PFile._read), not baked into
    # ``units``. (#104 U1-1.)
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
    # adc_fs/adc_bits set the absolute voltage scale; a missing adc_bits used to
    # default to 0 -> 2**0 = 1, silently mis-scaling the result by 2**16. Route
    # through _require_float so omission warns loudly (default 16-bit ADC).
    adc_fs = _require_float(params, "adc_fs", 1.0, "voltage")
    adc_bits = _require_float(params, "adc_bits", 16.0, "voltage")
    gain = _safe_float(params.get("g", "1"))
    adc_zero = _safe_float(params.get("adc_zero", "0"))
    phys = (adc_zero + data * adc_fs / 2**adc_bits) / gain
    return phys, "V"


def convert_piezo(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Piezo accelerometer: subtract zero-offset ``a_0``."""
    a_0 = _safe_float(params.get("a_0", "0"))
    return data - a_0, "counts"


def convert_accel(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Linear (DC-response) accelerometer → m/s².

    Matches ODAS ``convert_odas.m`` ``odas_accel_internal``:
      ``x = data*adc_fs/2^adc_bits + adc_zero - sig_zero``
      ``physical = 9.81 * (x - coef0) / coef1``

    Used by older VMPs with calibrated accelerometers (in place of the
    piezo type, whose output stays in counts).
    """
    adc_zero = _safe_float(params.get("adc_zero", "0"))
    # ODAS odas_accel_internal defaults adc_bits to 0 (2**0 = 1) and adc_fs to 1
    # for the counts-based coef0/coef1 calibration used by legacy calibrated
    # accelerometers that omit the ADC params. (Unlike convert_voltage, whose
    # ODAS default IS 16 — do NOT copy that here: the 2026-06-19 audit
    # disconfirmed a 16 default for accel, and PR #74's 16.0 mis-scaled such
    # configs by 2**16. SN479 configs carry adc_bits=16 explicitly, so this
    # default does not change the campaign output.)
    adc_fs = _safe_float(params.get("adc_fs", "1"))
    adc_bits = _safe_float(params.get("adc_bits", "0"))
    sig_zero = _safe_float(params.get("sig_zero", "0"))
    coef0 = _require_float(params, "coef0", 0.0, "accel")
    coef1 = _require_float(params, "coef1", 1.0, "accel")
    x = data * adc_fs / 2**adc_bits + adc_zero - sig_zero
    return 9.81 * (x - coef0) / coef1, "m_s-2"


def convert_magn(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Magnetometer → µT.

    Matches ODAS ``convert_odas.m`` ``odas_magn_internal``:
      ``physical = (data - coef0) / coef1``
    """
    coef0 = _require_float(params, "coef0", 0.0, "magn")
    coef1 = _require_float(params, "coef1", 1.0, "magn")
    return (data - coef0) / coef1, "uT"


def convert_inclxy(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """ADIS inclinometer X or Y: 14-bit two's complement → degrees."""
    val = _adis_14bit(data)
    # ODAS odas_inclxy_internal has no default for coef1 — MATLAB errors on a
    # missing scale rather than fabricating one. Route through _require_float so
    # a missing coefficient WARNS (house style: shear/voltage/accel/magn) instead
    # of silently substituting the ADIS16209 nominal 0.025 deg/LSB. (#104 U1-3.)
    coef0 = _require_float(params, "coef0", 0.0, "inclxy")
    coef1 = _require_float(params, "coef1", 0.025, "inclxy")
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
    # NaN (not 1) for a zero voltage part: a v_part==0 sample is corrupt, so
    # flag it rather than fabricating a finite ratio that looks like real data.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(v_part == 0, np.nan, i_part / v_part)
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


def convert_sbt(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Sea-Bird SBE3 temperature from a 32-bit period count → °C.

    Matches ODAS ``convert_odas.m`` ``odas_sbt_internal``:
      ``f[Hz] = coef6 * coef5 / w``  (n_periods * f_ref / count)
      ``x = ln(coef4 / f)``          (ln(f0/f))
      ``T = 1/(coef0 + coef1·x + coef2·x² + coef3·x³) - 273.15``

    The input is the joined even/odd 32-bit word (the PFile 2-id join).
    Verified to ≤ 6.4e-10 °C against the 2013 Taiwan ground truth (issue
    #141). Deviation from ODAS: a zero count (corrupt sample) yields NaN
    instead of a division blow-up.
    """
    g = _require_float(params, "coef0", 0.0, "sbt")
    h = _require_float(params, "coef1", 0.0, "sbt")
    i = _require_float(params, "coef2", 0.0, "sbt")
    j = _require_float(params, "coef3", 0.0, "sbt")
    f0 = _require_float(params, "coef4", 1000.0, "sbt")
    f_ref = _require_float(params, "coef5", 24e6, "sbt")
    n_periods = _require_float(params, "coef6", 128.0, "sbt")
    w = np.asarray(data, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(w != 0, n_periods * f_ref / w, np.nan)
        x = np.log(f0 / f)
        T = 1.0 / np.polyval([j, i, h, g], x) - 273.15
    return T, "deg_C"


def convert_sbc(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Sea-Bird SBE4 conductivity from a 32-bit period count → mS/cm.

    Matches ODAS ``convert_odas.m`` ``odas_sbc_internal``:
      ``f[kHz] = coef6 * coef5 / w / 1000``
      ``C = coef0 + coef1·f + coef2·f² + coef3·f³ + coef4·f⁴``

    No thermal-expansion/compressibility correction (vendor comment: done
    separately). Output is mS/cm, directly compatible with ``gsw.SP_from_C``.
    Verified to ≤ 1.8e-9 mS/cm against the 2013 Taiwan ground truth (issue
    #141). Deviation from ODAS: a zero count yields NaN.
    """
    c0 = _require_float(params, "coef0", 0.0, "sbc")
    c1 = _require_float(params, "coef1", 0.0, "sbc")
    c2 = _require_float(params, "coef2", 0.0, "sbc")
    c3 = _require_float(params, "coef3", 0.0, "sbc")
    c4 = _require_float(params, "coef4", 0.0, "sbc")
    f_ref = _require_float(params, "coef5", 24e6, "sbc")
    n_periods = _require_float(params, "coef6", 128.0, "sbc")
    w = np.asarray(data, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(w != 0, n_periods * f_ref / w / 1000.0, np.nan)
        C = np.polyval([c4, c3, c2, c1, c0], f)
    return C, "mS_cm-1"


def convert_aroft_o2(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """RINKO FT dissolved oxygen: unsigned 16-bit wrapping / 100 → µmol/L."""
    d = _deglitch_rs232(_unsigned_16bit(data))
    return d / 100.0, "umol_L-1"


def convert_aroft_t(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """RINKO FT temperature: unsigned 16-bit wrapping / 1000 - 5 -> deg C."""
    d = _deglitch_rs232(_unsigned_16bit(data))
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
    "accel": convert_accel,
    "magn": convert_magn,
    "inclxy": convert_inclxy,
    "inclt": convert_inclt,
    "jac_c": convert_jac_c,
    "jac_t": convert_jac_t,
    "sbt": convert_sbt,
    "sbc": convert_sbc,
    "raw": convert_raw,
    "aroft_o2": convert_aroft_o2,
    "aroft_t": convert_aroft_t,
    "gnd": convert_raw,
    "aem1g_a": convert_aem1g_a,
    "aem1g_d": convert_aem1g_d,
    "alec_emc": convert_alec_emc,
    "jac_emc": convert_aem1g_a,  # deprecated alias
}
