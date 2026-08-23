# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""The bridge algebra: raw thermistor counts -> ``L = ln(R_T/R_0)``.

This is deliberately the ONLY implementation of ``L`` on the calibration side.
``perturb/fp07_cal.py`` carries its own copy whose defaults disagree with the
reader's (``rsi/channels.py::convert_therm``) --- ``e_b`` defaults to 0 there
and 0.68 here, ``g`` to 1 there and 6.0 here, ``adc_fs`` to 5 there and 4.096
here.  A config missing any of those silently produces a different ``L`` in the
fit than the reader will later compute from the very coefficients the fit
emitted, and the calibration is wrong with no symptom.

So this module refuses to guess: every bridge parameter is **required**, and a
missing one is a hard error naming the channel.  A calibration is not the place
for a fallback default.

See ``docs/fp07_insitu_calibration_PLAN.md`` findings A4 and A7.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ODAS ``convert_odas.m`` clips the bridge output to +/-0.6 before forming the
# resistance ratio; |Z| approaching 1 means the sample is outside the bridge's
# range and ln((1-Z)/(1+Z)) diverges.  We reproduce the clip so ``L`` matches
# the reader exactly, but --- unlike the reader --- we also report WHICH
# samples were clipped so the fit can drop them instead of fitting a wrong
# ``L`` that looks perfectly ordinary downstream.
Z_CLIP = 0.6

_REQUIRED = ("a", "b", "g", "e_b", "adc_fs", "adc_bits")


@dataclass(frozen=True)
class BridgeParams:
    """The half-bridge constants that define ``L`` for one thermistor channel.

    These are recorded verbatim in the coefficient record: a Steinhart-Hart
    coefficient set is meaningless without the ``L`` definition it was fitted
    against, so an apply-time mismatch in any of these must be detectable.
    """

    a: float
    b: float
    g: float
    e_b: float
    adc_fs: float
    adc_bits: int

    @classmethod
    def from_channel_config(cls, cfg: dict, channel: str) -> BridgeParams:
        """Extract from a ``PFile`` channel config stanza; missing key -> error."""
        missing = [k for k in _REQUIRED if cfg.get(k) in (None, "")]
        if missing:
            raise ValueError(
                f"channel {channel!r}: config is missing bridge parameter(s) "
                f"{missing}. In-situ calibration will not substitute defaults — "
                f"the fitted coefficients would not reproduce in the reader. "
                f"Patch the config first (rsi-tpw patch)."
            )
        try:
            vals = {k: float(cfg[k]) for k in _REQUIRED}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"channel {channel!r}: non-numeric bridge parameter: {exc}") from exc
        if vals["b"] == 0:
            raise ValueError(f"channel {channel!r}: bridge parameter b == 0")
        if vals["g"] * vals["e_b"] == 0:
            raise ValueError(
                f"channel {channel!r}: g*e_b == 0 (g={vals['g']}, e_b={vals['e_b']}); "
                f"the bridge scaling would collapse"
            )
        return cls(
            a=vals["a"],
            b=vals["b"],
            g=vals["g"],
            e_b=vals["e_b"],
            adc_fs=vals["adc_fs"],
            adc_bits=int(vals["adc_bits"]),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "a": self.a,
            "b": self.b,
            "g": self.g,
            "e_b": self.e_b,
            "adc_fs": self.adc_fs,
            "adc_bits": float(self.adc_bits),
        }


def log_r(counts: np.ndarray, bp: BridgeParams) -> tuple[np.ndarray, np.ndarray]:
    """``L = ln(R_T/R_0)`` from raw counts, plus a per-sample clipped mask.

    Identical algebra to ``rsi/channels.py::convert_therm`` so that patching the
    emitted coefficients into the config reproduces the fit exactly.

    Returns ``(L, clipped)``.  ``clipped`` is True where the bridge output hit
    the +/-0.6 rail; those samples carry a wrong ``L`` and must be excluded from
    the fit rather than silently regressed (finding A7).
    """
    counts = np.asarray(counts, dtype=np.float64)
    z = ((counts - bp.a) / bp.b) * (bp.adc_fs / 2.0**bp.adc_bits) * 2.0 / (bp.g * bp.e_b)
    clipped = np.abs(z) >= Z_CLIP
    z = np.clip(z, -Z_CLIP, Z_CLIP)
    return np.log((1.0 - z) / (1.0 + z)), clipped


def temperature(L: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Apply Steinhart-Hart coefficients: ``1/T_K = sum a_i L^i`` -> degC."""
    L = np.asarray(L, dtype=np.float64)
    inv_T = np.zeros_like(L)
    for i, a in enumerate(np.asarray(coeffs, dtype=np.float64)):
        inv_T = inv_T + a * L**i
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(inv_T > 0, 1.0 / inv_T - 273.15, np.nan)


def coeffs_to_config(coeffs: np.ndarray) -> dict[str, float]:
    """``[a0, a1, ...]`` -> ``{t_0, beta_1, ...}``.

    Exact and lossless: ``convert_therm`` evaluates
    ``1/T_K = 1/t_0 + (1/beta_1)L + (1/beta_2)L^2 + (1/beta_3)L^3``, so the
    mapping is termwise reciprocal.  A zero coefficient has no representation
    (``beta_i = inf``) and is an error rather than a silently dropped term.
    """
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if coeffs.size < 2 or coeffs.size > 4:
        raise ValueError(f"expected 2..4 coefficients (order 1..3), got {coeffs.size}")
    if not np.all(np.isfinite(coeffs)) or np.any(coeffs == 0):
        raise ValueError(f"coefficients must be finite and non-zero: {coeffs.tolist()}")
    out = {"t_0": 1.0 / coeffs[0]}
    for i, c in enumerate(coeffs[1:], start=1):
        out[f"beta_{i}"] = 1.0 / c
    return out


def config_to_coeffs(cfg: dict) -> np.ndarray:
    """``{t_0, beta|beta_1, ...}`` -> ``[a0, a1, ...]``.

    Mirrors ``convert_therm``'s key precedence: the legacy ``beta`` is checked
    BEFORE ``beta_1``, so a config carrying both is read the way the reader
    reads it (finding A5 --- patching ``beta_1`` on such a file is a no-op).
    """
    t_0 = float(cfg["t_0"])
    if "beta" in cfg and cfg["beta"] not in (None, ""):
        beta_1 = float(cfg["beta"])
    else:
        beta_1 = float(cfg["beta_1"])
    out = [1.0 / t_0, 1.0 / beta_1]
    for key in ("beta_2", "beta_3"):
        val = cfg.get(key)
        if val is None or val == "":
            break
        out.append(1.0 / float(val))
    return np.array(out, dtype=np.float64)


def live_beta_key(cfg: dict) -> str:
    """Which key the reader will actually use for the linear term (A5)."""
    return "beta" if cfg.get("beta") not in (None, "") else "beta_1"
