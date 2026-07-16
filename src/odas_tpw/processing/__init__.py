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
- :func:`chi_combine.mk_chi_mean` — same iterative 95% CI machinery for
  multi-probe thermal-variance dissipation chi.
- :func:`mixing.window_stratification` / :func:`mixing.mixing_coefficients`
  — background N² and dT/dz per dissipation window, and the derived
  mixing quantities K_T (Osborn-Cox), Gamma (Oakey), K_rho (Osborn).
- :func:`thorpe.window_thorpe` / :func:`thorpe.patch_n2` — per-window
  Thorpe displacements/scale L_T and the overturn-weighted "patch"
  stratification (Smyth et al. 2001), plus Ozmidov/R_OT/Re_b/Cox helpers.
- :func:`probe_consistency.annotate_probe_consistency` — cross-probe
  median-ratio consistency diagnostic (per-pair attrs + two-tier
  logger warning; issue #131).
"""

from odas_tpw.processing.bottom import detect_bottom_crash
from odas_tpw.processing.chi_combine import mk_chi_mean
from odas_tpw.processing.ct_align import ct_align
from odas_tpw.processing.epsilon_combine import mk_epsilon_mean
from odas_tpw.processing.mixing import (
    mixing_coefficients,
    pair_nearest,
    window_stratification,
)
from odas_tpw.processing.probe_consistency import (
    PairStat,
    annotate_probe_consistency,
    lueck_ln_sigma,
    probe_pair_stats,
)
from odas_tpw.processing.thorpe import (
    cox_number,
    ozmidov,
    patch_n2,
    r_ot,
    reynolds_buoyancy,
    window_thorpe,
)
from odas_tpw.processing.top_trim import compute_trim_depth, compute_trim_depths

__all__ = [
    "PairStat",
    "annotate_probe_consistency",
    "compute_trim_depth",
    "compute_trim_depths",
    "cox_number",
    "ct_align",
    "detect_bottom_crash",
    "lueck_ln_sigma",
    "mixing_coefficients",
    "mk_chi_mean",
    "mk_epsilon_mean",
    "ozmidov",
    "pair_nearest",
    "patch_n2",
    "probe_pair_stats",
    "r_ot",
    "reynolds_buoyancy",
    "window_stratification",
    "window_thorpe",
]
