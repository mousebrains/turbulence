# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""
Channel conversion functions: raw counts -> physical units.

Ported from the ODAS MATLAB Library convert_odas.m.
"""

import re
import warnings
from typing import Any

import numpy as np

_NUMBER_RE = re.compile(r"(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
# Whitespace inside an operand: "1 2", "2 .5", "1e 5", "1e -5", "1e- 5".
_SPLIT_OPERAND_RE = re.compile(r"[\w.]\s+[\w.]|[\d.][eE]\s+[+-]|[\d.][eE][+-]\s+\d")
# A value that is just one decimal number: anything else is arithmetic.
_PLAIN_NUMBER_RE = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
# MATLAB's infinity, which ODAS reads as a beta of Inf: a Steinhart-Hart term of 1/Inf = 0.
_INFINITY_RE = re.compile(r"\s*[+-]?(?:Inf|inf)\s*")
# MATLAB's non-finite constants, as eval reads them (it has no NAN or INF).
_NONFINITE_RE = re.compile(r"(?:NaN|nan|Inf|inf)(?!\w)")
_MAX_EXPR_LEN = 200


class ConfigNumberNotFinite(ValueError):
    """A coefficient that is a valid expression but is NaN, infinite, or overflows.

    A subclass, so a caller can tell ``NaN`` from ``0,09`` by type; every
    caller that only needs "unusable" can keep catching ``ValueError``.
    """


def config_number(
    s: Any,
    default: float | None = None,
    *,
    key: str = "value",
    channel: str = "?",
    warn_arithmetic: bool = False,
) -> float:
    """A channel coefficient from the config string, read the way ODAS reads it.

    ODAS ``convert_odas.m`` passes every channel parameter through MATLAB
    ``eval``, so a coefficient may be written as arithmetic, and RSI's own
    configs do it (``division_factor = 2^28``, ``gain = 10.88/100``).
    ``float()`` cannot read those.  Every reader of a channel coefficient in
    this package goes through this one function -- the converters and their
    provenance record, the chi noise model's thermistor calibration,
    deconvolution's ``diff_gain``, ``fp07cal``'s bridge parameters and
    coefficients, the inventories -- so no two of them read a written value
    differently.  (ODAS itself does not manage that: ``deconvolve.m`` reads
    ``diff_gain`` with ``str2double``, which turns arithmetic into NaN.)  What
    a reader does about an ABSENT key is its own business and still differs on
    purpose: ``convert_therm`` defaults ``t_0`` to 289.0 K, the noise model to
    289.3 K, and ``fp07cal`` refuses to calibrate without it.

    A small parser with MATLAB's rules, not ``eval``: decimal numbers (with an
    optional exponent), parentheses, ``+ - * /`` and ``^`` for power.  As in
    MATLAB, ``^`` binds tighter than a unary sign (``-2^2`` is -4), is
    LEFT-associative (``2^3^2`` is 64) and may take a signed exponent
    (``2^-3``).  Whitespace may sit around operators but not inside an
    operand: ``1 2`` is refused, as MATLAB refuses it, rather than read as 12.
    MATLAB's ``NaN`` and ``Inf`` are expressions too, but a non-finite result
    is refused with :class:`ConfigNumberNotFinite`.  Anything else -- other
    names (``pi``), hex, underscores, a decimal comma, Python's ``**`` -- is
    refused as not a numeric expression.  Every failure is a ``ValueError``
    naming *channel* and *key*.  ``default`` applies only when *s* is absent
    (``None`` or blank); with no default an absent value is an error.

    *warn_arithmetic* warns when a value is evaluated as arithmetic rather than
    read as one number.  RSI writes JAC-CT coefficients that way on purpose;
    everywhere else in our corpus the one arithmetic value is a typo
    (SUNRISE 2021 SN 412 ``beta_1 = 3-1315.08``), which ODAS evaluates without
    a word.  The converters set it; readers that re-read a key a converter
    already read do not, so a value warns once.

    Root-section values (``recsize``, ``no-fast``) are not channel
    coefficients: ODAS reads those with ``str2double``, and so do we.
    """
    where = f"{channel}: {key}"
    if s is None or (isinstance(s, str) and not s.strip()):
        if default is None:
            raise ValueError(f"{where} is required and missing from the channel config")
        return default
    if isinstance(s, (int, float, np.integer, np.floating)) and not isinstance(
        s, (bool, np.bool_)
    ):
        try:
            value = float(s)  # a numpy float32 keeps its exact value, not its str()
        except OverflowError as exc:
            raise ConfigNumberNotFinite(f"{where} = {s!r} is too large") from exc
    else:

        def fail() -> ValueError:
            return ValueError(f"{where} = {s!r} is not a numeric expression")

        text = str(s).strip()
        if _SPLIT_OPERAND_RE.search(text):
            raise fail()
        text = re.sub(r"\s+", "", text)
        if len(text) > _MAX_EXPR_LEN:
            raise ValueError(f"{where} = {s!r} is too long to be a coefficient")
        pos = 0

        def peek() -> str:
            return text[pos] if pos < len(text) else ""

        def primary() -> float:
            nonlocal pos
            if peek() == "(":
                pos += 1
                v = expr()
                if peek() != ")":
                    raise fail()
                pos += 1
                return v
            m = _NUMBER_RE.match(text, pos) or _NONFINITE_RE.match(text, pos)
            if m is None:
                raise fail()
            pos = m.end()
            return float(m.group())

        def signed() -> float:
            # A unary sign applies to the whole power: -2^2 = -(2^2).
            nonlocal pos
            sign = 1.0
            while peek() in "+-" and peek():
                if peek() == "-":
                    sign = -sign
                pos += 1
            return sign * power()

        def power() -> float:
            nonlocal pos
            v = primary()
            while peek() == "^":
                pos += 1
                exp_sign = 1.0
                while peek() in "+-" and peek():
                    if peek() == "-":
                        exp_sign = -exp_sign
                    pos += 1
                v = v ** (exp_sign * primary())  # left-associative, as MATLAB
            return v

        def term() -> float:
            nonlocal pos
            v = signed()
            while peek() in "*/" and peek():
                op = peek()
                pos += 1
                rhs = signed()
                v = v * rhs if op == "*" else v / rhs
            return v

        def expr() -> float:
            nonlocal pos
            v = term()
            while peek() in "+-" and peek():
                op = peek()
                pos += 1
                rhs = term()
                v = v + rhs if op == "+" else v - rhs
            return v

        try:
            value = expr()
        except (ZeroDivisionError, OverflowError) as exc:
            raise ConfigNumberNotFinite(
                f"{where} = {s!r} does not evaluate to a finite number"
            ) from exc
        except RecursionError as exc:
            raise ValueError(f"{where} = {s!r} is nested too deeply") from exc
        if pos != len(text):
            raise fail()
        if isinstance(value, complex):
            raise ValueError(f"{where} = {s!r} is complex")
        if warn_arithmetic and np.isfinite(value) and not _PLAIN_NUMBER_RE.fullmatch(text):
            warnings.warn(
                f"{where} = {s!r} is arithmetic; evaluated as ODAS evaluates it, to "
                f"{value:.10g}. RSI writes JAC-CT coefficients this way; elsewhere "
                "it has been a typo. Check the instrument config.",
                stacklevel=2,
                skip_file_prefixes=(__file__,),  # point at the caller, not the parser
            )
    if not np.isfinite(value):
        raise ConfigNumberNotFinite(f"{where} = {s!r} does not evaluate to a finite number")
    return float(value)


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
            # Through the converters' own parser, so an arithmetic
            # coefficient ("2^28", "10.88/100") is recorded as its value.
            try:
                out[key] = config_number(dict.__getitem__(self, key), key=key)
            except ValueError:
                continue
        return out


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

    Never a default for a value someone wrote.  A parser that returned the
    caller's default on failure made a present-but-malformed coefficient
    indistinguishable from a real calibration: a stray decimal comma
    (``diff_gain = 0,09``) silently replaced 0.09 with 1.0, scaling shear by
    0.09x and shear VARIANCE — hence epsilon before its iterative correction —
    by 0.0081x, with no warning at all (issue #180 F06).  The same parser read
    ODAS-style arithmetic (``2^28``) as the default too.  Parsing is
    :func:`config_number`'s, so arithmetic is evaluated as ODAS evaluates it and
    anything else is refused.
    """
    need = (
        "a finite positive number"
        if positive
        else "a finite non-zero number"
        if nonzero
        else "a finite number"
    )
    try:
        value = config_number(
            raw, key=key, channel=str(params.get("name", "?")), warn_arithmetic=True
        )
    except ConfigNumberNotFinite:
        raise ConfigNumberNotFinite(
            f"{sensor} channel {params.get('name', '?')}: calibration coefficient "
            f"'{key}' is {raw!r}; must be {need}. Refusing to fabricate physical units."
        ) from None
    except ValueError as exc:
        raise ValueError(
            f"{sensor} channel {params.get('name', '?')}: calibration coefficient "
            f"'{key}' is present but unparseable ({raw!r}: {exc}); refusing to "
            f"substitute the default {default} and fabricate physical units. Fix "
            "the value in the instrument config (a decimal comma is the usual "
            "cause), or remove the key entirely to accept the documented default."
        ) from None
    if (positive and value <= 0) or (nonzero and value == 0):
        extra = ""
        if nonzero and value == 0 and key.startswith("beta"):
            # The Steinhart-Hart betas are RECIPROCALS: zero is an infinite
            # term, not a deleted one. 1e30 is bit-identical to omitting the key.
            extra = (
                f" ('{key}' is a reciprocal coefficient: zero means an INFINITE "
                "term, not a deleted one — use 1e30, or remove the key)"
            )
        elif nonzero and value == 0:
            extra = f" ('{key}' is a divisor)"
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
    """A sensor-specific coefficient: missing warns, malformed raises.

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
    if params.get("beta") not in (None, ""):  # a blank `beta` is absent
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
        if not _is_infinity(beta_2):  # ODAS: 1/Inf = 0, no quadratic term
            inv_T += (
                1.0
                / _parse_finite_float(beta_2, params, "beta_2", 1e30, "therm", False, nonzero=True)
            ) * log_R**2
        beta_3 = params.get("beta_3")
        if beta_3 not in (None, "") and not _is_infinity(beta_3):
            inv_T += (
                1.0
                / _parse_finite_float(beta_3, params, "beta_3", 1e30, "therm", False, nonzero=True)
            ) * log_R**3
    return 1.0 / inv_T - 273.15, "deg_C"


def _is_infinity(raw: Any) -> bool:
    """``beta_2 = Inf``: Rockland's own tutorial file writes it; ODAS reads it as
    a term of 1/Inf = 0, i.e. no term, the same as omitting the key."""
    if isinstance(raw, (float, np.floating)):
        return bool(np.isinf(raw))
    return isinstance(raw, str) and _INFINITY_RE.fullmatch(raw) is not None


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
# of 15998 rows (14%) sit at 0.090-0.099 and the remaining 13762 at 0.905-1.01.
# That is not corruption — it is one rung of a deliberate Rockland design
# ladder.  To work in higher-energy environments they DOUBLE the sampling
# frequency (capturing more of the Nasmyth spectrum) and DROP the differential
# gain by a factor of ten to keep the differentiator out of saturation:
#
#     fs_fast    diff_gain    in our corpus
#      512 Hz      ~0.95      13 instruments (0.905-1.01)
#     1024 Hz      ~0.095     MR 330, 429 (tidal builds); VMP 132 at 512 Hz
#     2048 Hz      ~0.0095    exists at Rockland; none in our portfolio
#
# (Design rationale from Pat Welch, who works directly with Rockland; the rate
# and gain columns are measured from our own file headers and configs.)
#
# The same-model comparison isolates it from vehicle and model: MR1000RDL-EM
# reads 0.927/0.941 on the 512 Hz gliders SN 433/435 and 0.099/0.094 on the
# 1024 Hz SN 429.  So a tight band around the upper mode would flag a seventh
# of every shear channel we own, and an alarm that fires on a seventh of the
# corpus is an alarm that gets muted.  This converter sees only the CHANNEL
# config — not the rate — so it cannot select the right rung itself.
#
# The floor sits below the 2048 Hz rung ON PURPOSE.  0.01 would have been the
# obvious choice from the observed data alone, and it would warn on every
# channel of the first 2048 Hz instrument we ever read (~0.0095) — a check that
# fires on correct hardware the day it arrives.  0.001 clears that rung by ~9x,
# the same margin the ceiling gives the top one.
#
# Telling "0.09 is this instrument's real differentiator" from "0.09 is a typo"
# needs an instrument-keyed comparison, which is what ``rsi-tpw sensors
# --diff-gain`` (odas_tpw.rsi.diff_gain) exists to do — and in the end only a
# measurement settles it.  SN 132 shows why the rate cannot select the rung
# either: it samples at 511.9454 Hz yet 0.09 is on the RIGHT rung.  It
# was converted to a 1024 Hz tidal build in 2021 and back, and its analog chain
# evidently stayed on the tidal rung (bench-noise fit 0.10-0.13; epsilon with
# 0.09 matches SN 465's, cast nearby from a second boat, at ~0.9-1.2x, 95% CIs
# within ~0.7-1.7x).  0.09 versus ~0.12 is still open.  See issue #178.
_SHEAR_DIFF_GAIN_MIN = 0.001
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
    # not fall through to a default — that silently fabricates sens=1.0 and
    # scales epsilon by sens^-2 (~125x on real probes). config_number refuses
    # sens="nan" — every comparison with NaN is False, so it would sail through
    # a sign check into all-NaN shear — and sens="inf" (zero shear).
    raw_sens = params.get("sens")
    sens: float | None
    try:
        sens = (
            config_number(
                raw_sens, key="sens", channel=str(params.get("name", "?")), warn_arithmetic=True
            )
            if raw_sens not in (None, "")
            else None
        )
    except ValueError:
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
            if params[key] in (None, ""):
                # ODAS evals '' to [] and the concatenation DROPS the term,
                # shifting every higher order down one: refuse, don't guess.
                raise ValueError(
                    f"poly channel {params.get('name', '?')}: coefficient '{key}' is "
                    "present but empty; give it a value or remove it."
                )
            coeffs.append(_parse_finite_float(params[key], params, key, 0.0, "poly", False))
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
    # default to 0 -> 2**0 = 1, silently mis-scaling the result by 2**16, so
    # omission warns loudly (default 16-bit ADC).
    adc_fs = _require_finite_float(params, "adc_fs", 1.0, "voltage")
    adc_bits = _require_finite_float(params, "adc_bits", 16.0, "voltage")
    gain = _optional_finite_float(params, "g", 1.0, "voltage", nonzero=True)
    adc_zero = _optional_finite_float(params, "adc_zero", 0.0, "voltage")
    phys = (adc_zero + data * adc_fs / 2**adc_bits) / gain
    return phys, "V"


def convert_piezo(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Piezo accelerometer: subtract zero-offset ``a_0``."""
    a_0 = _optional_finite_float(params, "a_0", 0.0, "piezo")
    return data - a_0, "counts"


def convert_accel(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Linear (DC-response) accelerometer → m/s².

    Matches ODAS ``convert_odas.m`` ``odas_accel_internal``:
      ``x = data*adc_fs/2^adc_bits + adc_zero - sig_zero``
      ``physical = 9.81 * (x - coef0) / coef1``

    Used by older VMPs with calibrated accelerometers (in place of the
    piezo type, whose output stays in counts).
    """
    adc_zero = _optional_finite_float(params, "adc_zero", 0.0, "accel")
    # ODAS odas_accel_internal defaults adc_bits to 0 (2**0 = 1) and adc_fs to 1
    # for the counts-based coef0/coef1 calibration used by legacy calibrated
    # accelerometers that omit the ADC params. (Unlike convert_voltage, whose
    # ODAS default IS 16 — do NOT copy that here: the 2026-06-19 audit
    # disconfirmed a 16 default for accel, and PR #74's 16.0 mis-scaled such
    # configs by 2**16. SN479 configs carry adc_bits=16 explicitly, so this
    # default does not change the campaign output.)
    adc_fs = _optional_finite_float(params, "adc_fs", 1.0, "accel")
    adc_bits = _optional_finite_float(params, "adc_bits", 0.0, "accel")
    sig_zero = _optional_finite_float(params, "sig_zero", 0.0, "accel")
    coef0 = _require_finite_float(params, "coef0", 0.0, "accel")
    coef1 = _require_finite_float(params, "coef1", 1.0, "accel", nonzero=True)
    x = data * adc_fs / 2**adc_bits + adc_zero - sig_zero
    return 9.81 * (x - coef0) / coef1, "m_s-2"


def convert_magn(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Magnetometer → µT.

    Matches ODAS ``convert_odas.m`` ``odas_magn_internal``:
      ``physical = (data - coef0) / coef1``
    """
    coef0 = _require_finite_float(params, "coef0", 0.0, "magn")
    coef1 = _require_finite_float(params, "coef1", 1.0, "magn", nonzero=True)
    return (data - coef0) / coef1, "uT"


def convert_inclxy(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """ADIS inclinometer X or Y: 14-bit two's complement → degrees."""
    val = _adis_14bit(data)
    # ODAS odas_inclxy_internal has no default for coef1 — MATLAB errors on a
    # missing scale rather than fabricating one. Route through
    # _require_finite_float so a missing coefficient WARNS (house style:
    # shear/voltage/accel/magn) instead of silently substituting the ADIS16209
    # nominal 0.025 deg/LSB. (#104 U1-3.)
    coef0 = _require_finite_float(params, "coef0", 0.0, "inclxy")
    coef1 = _require_finite_float(params, "coef1", 0.025, "inclxy")
    return coef1 * val + coef0, "deg"


def convert_inclt(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """ADIS inclinometer temperature: 14-bit two's complement → °C."""
    val = _adis_14bit(data)
    coef0 = _optional_finite_float(params, "coef0", 624.0, "inclt")
    coef1 = _optional_finite_float(params, "coef1", -0.47, "inclt")
    return coef1 * val + coef0, "deg_C"


def convert_jac_c(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """JAC conductivity: ratio of I/V parts from 32-bit combined word → mS/cm."""
    i_part = np.floor(data / 2**16)
    v_part = np.mod(data, 2**16)
    # NaN (not 1) for a zero voltage part: a v_part==0 sample is corrupt, so
    # flag it rather than fabricating a finite ratio that looks like real data.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(v_part == 0, np.nan, i_part / v_part)
    a = _optional_finite_float(params, "a", 0.0, "jac_c")
    b = _optional_finite_float(params, "b", 0.0, "jac_c")
    c = _optional_finite_float(params, "c", 0.0, "jac_c")
    return np.polyval([c, b, a], ratio), "mS_cm-1"


def convert_jac_t(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """JAC temperature: unsigned 16-bit wrapping + 5th-order polynomial → °C."""
    d = _unsigned_16bit(data)
    a, b, c, d_coef, e, f = (
        _optional_finite_float(params, k, 0.0, "jac_t") for k in ("a", "b", "c", "d", "e", "f")
    )
    return np.polyval([f, e, d_coef, c, b, a], d), "deg_C"


def _nonzero(value: float, key: str, channel: str) -> float:
    """A divisor: zero would give inf or a ZeroDivisionError downstream."""
    if value == 0.0:
        raise ValueError(f"{channel}: {key} = 0 is a divisor and cannot be zero")
    return value


def convert_rsijac_c(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """RSI's JAC-CT board, conductivity: Q3.28 fixed point → mS/cm.

    Matches ODAS ``convert_odas.m`` ``odas_rsijac_c_internal`` (WID, 2017-05):
    ``counts = input / division_factor``, then ``b * counts + a``, then
    ``(counts * gain - y_offset) / cell_constant`` [S/m], times 10.  The defaults
    are ODAS's; ``cell_constant`` has none and is required.  Distinct from
    ``jac_c``, the JFE Advantech readout, which packs a current/voltage pair.
    Unlike ODAS, a zero or non-finite coefficient raises instead of yielding inf.
    """
    ch = str(params.get("name", "?"))

    def num(key: str, default: float | None = None) -> float:
        return config_number(params.get(key), default, key=key, channel=ch)

    df = _nonzero(num("division_factor", 2.0**28), "division_factor", ch)
    gain = num("gain", 10.88 / 100)
    offset = num("y_offset", 2.86e-7)
    a = num("a", 0.0)
    b = num("b", 1.0)
    cell = _nonzero(num("cell_constant"), "cell_constant", ch)
    counts = b * (np.asarray(data, dtype=np.float64) / df) + a
    return (counts * gain - offset) / cell * 10.0, "mS_cm-1"


def convert_rsijac_t(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """RSI's JAC-CT board, temperature: Q3.28 fixed point → °C.

    Matches ODAS ``convert_odas.m`` ``odas_rsijac_t_internal``: ``counts =
    input / division_factor``, ``b * counts + a``, then the cubic
    Steinhart-Hart form ``1/T = 1/T_0 + ln(c)/beta_1 + ln(c)^2/beta_2 +
    ln(c)^3/beta_3``.  ODAS reads ``division_factor`` (default 2^28) and
    ignores the ``scale_factor`` key RSI's SeaExplorer configs carry; so does
    this.  A non-positive count has no logarithm and becomes NaN.  ``T_0`` and
    the three betas are required and must be non-zero (each is a divisor);
    ODAS would return -273.15 C or inf instead.
    """
    ch = str(params.get("name", "?"))

    def num(key: str, default: float | None = None) -> float:
        return config_number(params.get(key), default, key=key, channel=ch)

    df = _nonzero(num("division_factor", 2.0**28), "division_factor", ch)
    a = num("a", 0.0)
    b = num("b", 1.0)
    t0 = _nonzero(num("t_0"), "t_0", ch)
    b1 = _nonzero(num("beta_1"), "beta_1", ch)
    b2 = _nonzero(num("beta_2"), "beta_2", ch)
    b3 = _nonzero(num("beta_3"), "beta_3", ch)
    counts = b * (np.asarray(data, dtype=np.float64) / df) + a
    with np.errstate(divide="ignore", invalid="ignore"):
        lc = np.log(np.where(counts > 0, counts, np.nan))
        kelvin = 1.0 / (1.0 / t0 + lc / b1 + lc**2 / b2 + lc**3 / b3)
    return kelvin - 273.15, "deg_C"


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
    g = _require_finite_float(params, "coef0", 0.0, "sbt")
    h = _require_finite_float(params, "coef1", 0.0, "sbt")
    i = _require_finite_float(params, "coef2", 0.0, "sbt")
    j = _require_finite_float(params, "coef3", 0.0, "sbt")
    f0 = _require_finite_float(params, "coef4", 1000.0, "sbt", positive=True)
    f_ref = _require_finite_float(params, "coef5", 24e6, "sbt", positive=True)
    n_periods = _require_finite_float(params, "coef6", 128.0, "sbt", positive=True)
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
    c0 = _require_finite_float(params, "coef0", 0.0, "sbc")
    c1 = _require_finite_float(params, "coef1", 0.0, "sbc")
    c2 = _require_finite_float(params, "coef2", 0.0, "sbc")
    c3 = _require_finite_float(params, "coef3", 0.0, "sbc")
    c4 = _require_finite_float(params, "coef4", 0.0, "sbc")
    f_ref = _require_finite_float(params, "coef5", 24e6, "sbc", positive=True)
    n_periods = _require_finite_float(params, "coef6", 128.0, "sbc", positive=True)
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
    adc_fs = _optional_finite_float(params, "adc_fs", 4.096, "aem1g_a", positive=True)
    adc_bits = _optional_finite_float(params, "adc_bits", 16.0, "aem1g_a", positive=True)
    adc_zero = _optional_finite_float(params, "adc_zero", adc_fs / 2, "aem1g_a")
    a = _optional_finite_float(params, "a", 0.0, "aem1g_a") / 100.0
    b = _optional_finite_float(params, "b", 1.0, "aem1g_a") / 100.0
    bias = _optional_finite_float(params, "bias", 0.0, "aem1g_a")
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
    a = _optional_finite_float(params, "a", 0.0, "aem1g_d") / 100.0
    b = _optional_finite_float(params, "b", 1.0, "aem1g_d") / 100.0
    return a + b * d, "m_s-1"


def convert_alec_emc(data: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Alec Electronics electromagnetic current meter → m/s.

    Matches ODAS ``convert_odas.m`` ``odas_Alec_EMC_internal``:
      ``physical = coef1 * data + coef0``
    """
    coef0 = _optional_finite_float(params, "coef0", 0.0, "alec_emc")
    coef1 = _optional_finite_float(params, "coef1", 1.0, "alec_emc")
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
    "rsijac_c": convert_rsijac_c,
    "rsijac_t": convert_rsijac_t,
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
