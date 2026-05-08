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
_RULE_OPTION_KEYS = frozenset({"channel", "min", "max", "abs_max", "bit"})


def _resolve_inclinometer_axis(pf: Any, want: str) -> str | None:
    """Return ``Incl_X`` or ``Incl_Y`` (whichever is pitch / roll on this
    instrument) by ``np.nanmax - np.nanmin`` amplitude. ``None`` when
    neither inclinometer is present.

    Same heuristic as :mod:`odas_tpw.perturb.speed`'s flight model: the
    axis with the larger swing is pitch (the glider rotates more around
    its body axis than rolls during normal flight); the other is roll.
    """
    iX = pf.channels.get("Incl_X")
    iY = pf.channels.get("Incl_Y")
    if iX is None or iY is None:
        return None
    rx = float(np.nanmax(iX) - np.nanmin(iX))
    ry = float(np.nanmax(iY) - np.nanmin(iY))
    pitch_name, roll_name = ("Incl_X", "Incl_Y") if rx >= ry else ("Incl_Y", "Incl_X")
    return pitch_name if want == "pitch" else roll_name


def _coarsen_to_slow(flag: np.ndarray, ratio: int) -> np.ndarray:
    """Max-pool a fast-rate boolean to slow rate. ``ratio = fs_fast/fs_slow``."""
    n_slow = flag.size // ratio
    if n_slow == 0:
        return np.zeros(0, dtype=bool)
    return np.asarray(flag[: n_slow * ratio].reshape(n_slow, ratio).any(axis=1))


def evaluate_rules(pf: Any, rules: dict[str, dict] | None) -> dict[str, np.ndarray]:
    """Evaluate the rules block against ``pf.channels``.

    Returns a dict ``{rule_name: uint8_array}`` keyed by the YAML rule
    name. Each array sits on ``pf.t_slow`` with the rule's ``bit`` set
    wherever its condition is met.

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
        unknown = set(cfg) - _RULE_OPTION_KEYS
        if unknown:
            raise ValueError(
                f"qc.rules.{name}: unknown options {sorted(unknown)}. "
                f"Valid: {sorted(_RULE_OPTION_KEYS)}"
            )
        ch_name = cfg.get("channel")
        if not ch_name:
            raise ValueError(f"qc.rules.{name}: 'channel' is required")
        bit_val = int(cfg.get("bit", 1))
        if bit_val < 1 or bit_val > _MAX_BIT or bit_val & (bit_val - 1):
            raise ValueError(
                f"qc.rules.{name}: 'bit' must be a power of 2 in 1..{_MAX_BIT}, "
                f"got {bit_val}"
            )

        # Resolve channel name (pseudo-name auto-detect for pitch / roll).
        if ch_name in ("pitch", "roll"):
            resolved = _resolve_inclinometer_axis(pf, ch_name)
            if resolved is None:
                logger.warning(
                    "qc.rules.%s: %r requires Incl_X and Incl_Y; skipping",
                    name, ch_name,
                )
                continue
            actual = resolved
        else:
            actual = ch_name

        if actual not in pf.channels:
            logger.warning(
                "qc.rules.%s: channel %r not on pf.channels; skipping",
                name, actual,
            )
            continue

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
