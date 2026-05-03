# May-2026, Claude and Pat Welch, pat@mousebrains.com
"""Instrument-agnostic profile-level processing.

Operates on raw numpy arrays (and a thin layer of xarray for
mk_epsilon_mean) — no PFile, no NetCDF I/O, no perturb-specific config.
Suitable for any vertical profiler whose pipeline can hand over the
right arrays.

- :func:`top_trim.compute_trim_depth` — depth where shear / acceleration
  variance settles below a quantile threshold (prop-wash exit).
- :func:`bottom.detect_bottom_crash` — depth where vibration spikes near
  the deepest bin (seafloor impact).
- :func:`ct_align.ct_align` — cross-correlate diff(T) vs diff(C) and
  shift C by the median lag across profiles.
- :func:`epsilon_combine.mk_epsilon_mean` — Lueck (2022) iterative 95% CI
  geometric mean of multi-probe dissipation estimates.
"""

from odas_tpw.processing.bottom import detect_bottom_crash
from odas_tpw.processing.ct_align import ct_align
from odas_tpw.processing.epsilon_combine import mk_epsilon_mean
from odas_tpw.processing.top_trim import compute_trim_depth, compute_trim_depths

__all__ = [
    "compute_trim_depth",
    "compute_trim_depths",
    "ct_align",
    "detect_bottom_crash",
    "mk_epsilon_mean",
]
