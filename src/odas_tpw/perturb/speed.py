# May-2026, Pat Welch, pat@mousebrains.com
"""Through-water speed for the perturb pipeline.

The implementation now lives in :mod:`odas_tpw.rsi.speed` so the lower-level
``rsi`` pipeline can share the exact same speed model (pressure / em / flight /
constant) without a backwards dependency on ``perturb``. This module re-exports
it for backward compatibility.
"""

from __future__ import annotations

from odas_tpw.rsi.speed import compute_speed_for_pfile

__all__ = ["compute_speed_for_pfile"]
