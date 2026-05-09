# May-2026, Claude and Pat Welch, pat@mousebrains.com
"""Declarative internal QC rules.

Evaluates simple range checks against the merged ``pf.channels`` and
emits synthetic uint8 channels that drop into the same per-segment
``compute_segment_drop`` machinery as hotel-injected QC channels.

YAML schema (under the perturb config's ``qc:`` block):

    qc:
      rules:
        speed_oor:                 # rule name = synthetic channel name
          channel: speed_fast
          min: 0.05                # any of {min, max, abs_max} can be set
          max: 1.5
          bit: 8                   # bit set in the uint8 output
        pitch_oor:
          channel: pitch           # pseudo-name → auto-pick from Incl_X/Y
          abs_max: 45              # |angle| > 45 deg → flag
          bit: 16
        roll_oor:
          channel: roll            # pseudo-name → "the other" inclinometer
          abs_max: 10
          bit: 32

Behavior
--------
- A rule's ``channel`` is read from ``pf.channels``. Pseudo-names
  ``"pitch"`` and ``"roll"`` auto-pick from ``Incl_X`` / ``Incl_Y`` by
  amplitude — same heuristic as the flight-model speed; pitch is the
  axis with the larger swing, roll is the other.
- Missing channels are warned-and-skipped rather than fatal: one config
  can target multiple instrument types where some channels don't exist.
- Fast-rate channels are max-pooled to slow rate so the qc_gate can
  uniformly consume bool-on-pf.t_slow arrays.
- Each rule writes a ``uint8`` channel with the named bit set wherever
  the condition is met, plus CF ``flag_meanings`` / ``flag_masks`` so
  ``compute_segment_drop`` can lift them into the per-segment bitfield.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Maximum bit value we accept (uint8 fits 8 distinct flags).
_MAX_BIT = 1 << 7

# Range rule keys (the original / default rule type).
_RANGE_OPTION_KEYS = frozenset({
    "type", "channel", "min", "max", "abs_max", "bit", "amplitude_quantile",
})

# pitch_w_consistency rule keys.
_FLIGHT_CONSISTENCY_OPTION_KEYS = frozenset({
    "type", "bit", "amplitude_quantile",
    "pitch_min_deg", "W_min_dbar_per_s",
})

# Default low / high percentiles for inclinometer-axis amplitude detection.
# 1..99 strips the outlier tails (e.g. surface tumbles or brief sensor
# saturation spikes that can dominate ``nanmax - nanmin``) while still
# capturing the bulk of the in-flight swing.
_DEFAULT_AMPLITUDE_Q: tuple[float, float] = (1.0, 99.0)


def _resolve_inclinometer_axis(
    pf: Any,
    want: str,
    amplitude_q: tuple[float, float] = _DEFAULT_AMPLITUDE_Q,
) -> str | None:
    """Return ``Incl_X`` or ``Incl_Y`` (whichever is pitch / roll on this
    instrument) by percentile-spread amplitude. ``None`` when neither
    inclinometer is present.

    The spread is the high minus the low percentile of each axis
    (default 99-1). Percentiles keep brief outliers — e.g. a surface
    tumble or sensor saturation that briefly drives one axis to
    -90 deg — from masquerading as the high-amplitude flight axis.
    Pitch is then the axis with the larger spread (the glider rotates
    around its body axis during dive / climb cycles); the other axis
    is roll.

    Same heuristic is shared with :mod:`odas_tpw.perturb.speed`'s
    flight model so a single deployment's QC and speed estimate agree
    on which axis is which.
    """
    iX = pf.channels.get("Incl_X")
    iY = pf.channels.get("Incl_Y")
    if iX is None or iY is None:
        return None
    lo_q, hi_q = float(amplitude_q[0]), float(amplitude_q[1])
    rx = float(np.nanpercentile(iX, hi_q) - np.nanpercentile(iX, lo_q))
    ry = float(np.nanpercentile(iY, hi_q) - np.nanpercentile(iY, lo_q))
    pitch_name, roll_name = ("Incl_X", "Incl_Y") if rx >= ry else ("Incl_Y", "Incl_X")
    return pitch_name if want == "pitch" else roll_name


def _coarsen_to_slow(flag: np.ndarray, ratio: int) -> np.ndarray:
    """Max-pool a fast-rate boolean to slow rate. ``ratio = fs_fast/fs_slow``."""
    n_slow = flag.size // ratio
    if n_slow == 0:
        return np.zeros(0, dtype=bool)
    return np.asarray(flag[: n_slow * ratio].reshape(n_slow, ratio).any(axis=1))


def _validate_bit(name: str, bit_val: int) -> None:
    if bit_val < 1 or bit_val > _MAX_BIT or bit_val & (bit_val - 1):
        raise ValueError(
            f"qc.rules.{name}: 'bit' must be a power of 2 in 1..{_MAX_BIT}, "
            f"got {bit_val}"
        )


def _validate_amplitude_q(name: str, aq) -> tuple[float, float]:
    if (not hasattr(aq, "__len__") or len(aq) != 2
            or not (0.0 <= float(aq[0]) < float(aq[1]) <= 100.0)):
        raise ValueError(
            f"qc.rules.{name}.amplitude_quantile: must be a pair "
            f"[lo, hi] with 0 <= lo < hi <= 100, got {aq!r}"
        )
    return (float(aq[0]), float(aq[1]))


def _evaluate_range(pf: Any, name: str, cfg: dict, n_slow: int,
                    ratio: int) -> np.ndarray | None:
    """Per-channel min/max/abs_max range check. Original / default rule type."""
    ch_name = cfg.get("channel")
    if not ch_name:
        raise ValueError(f"qc.rules.{name}: 'channel' is required")

    if ch_name in ("pitch", "roll"):
        aq = cfg.get("amplitude_quantile") or _DEFAULT_AMPLITUDE_Q
        aq = _validate_amplitude_q(name, aq)
        resolved = _resolve_inclinometer_axis(pf, ch_name, aq)
        if resolved is None:
            logger.warning(
                "qc.rules.%s: %r requires Incl_X and Incl_Y; skipping",
                name, ch_name,
            )
            return None
        actual = resolved
    else:
        actual = ch_name

    if actual not in pf.channels:
        logger.warning(
            "qc.rules.%s: channel %r not on pf.channels; skipping",
            name, actual,
        )
        return None

    arr = np.asarray(pf.channels[actual], dtype=np.float64)
    flag = np.zeros(arr.shape, dtype=bool)
    if "min" in cfg:
        flag |= arr < float(cfg["min"])
    if "max" in cfg:
        flag |= arr > float(cfg["max"])
    if "abs_max" in cfg:
        flag |= np.abs(arr) > float(cfg["abs_max"])
    # NaN samples are flagged too — they're as bad as out-of-range for QC.
    flag |= ~np.isfinite(arr)

    # Coarsen fast-rate channels to slow so qc_gate consumes uniformly.
    if actual in pf._fast_channels and arr.size != n_slow:
        flag = _coarsen_to_slow(flag, ratio)
    return np.asarray(flag, dtype=bool)


def _evaluate_pitch_w_consistency(pf: Any, name: str, cfg: dict,
                                  n_slow: int) -> np.ndarray | None:
    """Flag samples where pitch direction and dP/dt sign disagree.

    Healthy glide has nose-up + ascending or nose-down + descending; a
    stalled glider can be pitched up while still sinking (or vice
    versa) and shouldn't contribute turbulence estimates.

    The check operates on:

    - ``pitch``: auto-resolved Incl_Y / Incl_X (sign convention of the
      raw channel — *no* normalisation, since we only test sign vs W).
    - ``W``: signed dP/dt at slow rate (positive = depth increasing =
      sinking), computed from ``pf.channels["P"]`` with the same
      smoothing as :func:`smooth_fall_rate` in the speed module.

    Within ±``pitch_min_deg`` of level or ±``W_min_dbar_per_s`` of
    stationary, samples are *not* flagged — too close to a sign-zero
    crossing to call confidently.

    YAML schema::

        flight_consistency:
          type: pitch_w_consistency
          pitch_min_deg: 5.0
          W_min_dbar_per_s: 0.02
          bit: 64
    """
    from odas_tpw.scor160.profile import smooth_fall_rate

    aq = cfg.get("amplitude_quantile") or _DEFAULT_AMPLITUDE_Q
    aq = _validate_amplitude_q(name, aq)
    pitch_name = _resolve_inclinometer_axis(pf, "pitch", aq)
    if pitch_name is None:
        logger.warning(
            "qc.rules.%s: pitch_w_consistency requires Incl_X and Incl_Y; skipping",
            name,
        )
        return None
    if "P" not in pf.channels:
        logger.warning(
            "qc.rules.%s: pitch_w_consistency requires P; skipping",
            name,
        )
        return None

    pitch = np.asarray(pf.channels[pitch_name], dtype=np.float64)
    P = np.asarray(pf.channels["P"], dtype=np.float64)
    fs_slow = float(pf.fs_slow)
    W = smooth_fall_rate(P, fs_slow)  # signed, dbar/s

    pitch_min = float(cfg.get("pitch_min_deg", 5.0))
    w_min = float(cfg.get("W_min_dbar_per_s", 0.02))

    # Signs disagree (one positive, one negative) and both are clearly
    # nonzero (outside the noise zone around level / stationary).
    flag = (pitch * W < 0) & (np.abs(pitch) >= pitch_min) & (np.abs(W) >= w_min)
    flag |= ~np.isfinite(pitch) | ~np.isfinite(W)
    return np.asarray(flag, dtype=bool)


# Dispatch table. New rule types register here.
_RULE_TYPES: dict[str, tuple[frozenset, Any]] = {
    "range": (_RANGE_OPTION_KEYS, _evaluate_range),
    "pitch_w_consistency": (
        _FLIGHT_CONSISTENCY_OPTION_KEYS, _evaluate_pitch_w_consistency,
    ),
}


def evaluate_rules(pf: Any, rules: dict[str, dict] | None) -> dict[str, np.ndarray]:
    """Evaluate the rules block against ``pf.channels``.

    Returns a dict ``{rule_name: uint8_array}`` keyed by the YAML rule
    name. Each array sits on ``pf.t_slow`` with the rule's ``bit`` set
    wherever its condition is met.

    Rule type is selected via ``type:`` (default ``range`` for backwards
    compat). See :data:`_RULE_TYPES` for registered evaluators.

    Rules referencing a missing channel are warned-and-skipped (the
    output dict simply omits them — downstream ``drop_from`` lookups
    treat absent channels the same as channels containing only zeros).
    """
    if not rules:
        return {}

    out: dict[str, np.ndarray] = {}
    ratio = max(1, round(float(pf.fs_fast) / float(pf.fs_slow)))
    n_slow = len(pf.t_slow)

    for name, raw in rules.items():
        cfg = dict(raw or {})
        rtype = str(cfg.get("type", "range"))
        if rtype not in _RULE_TYPES:
            raise ValueError(
                f"qc.rules.{name}: unknown type {rtype!r}. "
                f"Valid: {sorted(_RULE_TYPES)}"
            )
        valid_keys, evaluator = _RULE_TYPES[rtype]
        unknown = set(cfg) - valid_keys
        if unknown:
            raise ValueError(
                f"qc.rules.{name} (type={rtype}): unknown options {sorted(unknown)}. "
                f"Valid: {sorted(valid_keys)}"
            )
        bit_val = int(cfg.get("bit", 1))
        _validate_bit(name, bit_val)

        if rtype == "range":
            flag = evaluator(pf, name, cfg, n_slow, ratio)
        else:
            flag = evaluator(pf, name, cfg, n_slow)
        if flag is None:
            continue

        if flag.size != n_slow:
            # Pad / truncate to slow length — keeps qc_gate slicing valid.
            if flag.size < n_slow:
                pad = np.zeros(n_slow - flag.size, dtype=bool)
                flag = np.concatenate((flag, pad))
            else:
                flag = flag[:n_slow]
        out[name] = flag.astype(np.uint8) * np.uint8(bit_val)

    return out


def register_rule_channels(pf: Any, rule_arrays: dict[str, np.ndarray],
                           rules: dict[str, dict]) -> None:
    """Register evaluated rule arrays on ``pf`` so the qc_gate sees them.

    Adds each rule's array to ``pf.channels`` and writes a
    ``channel_info`` entry with CF ``flag_meanings`` / ``flag_masks``
    that ``compute_segment_drop`` lifts into the per-segment bitfield.
    """
    for name, arr in rule_arrays.items():
        bit_val = int(rules[name].get("bit", 1))
        pf.channels[name] = arr
        pf.channel_info[name] = {
            "units": "",
            "type": "qc_rule",
            "name": name,
            "flag_meanings": name,
            "flag_masks": [bit_val],
        }
