# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""FP07 in-situ calibration: extracting reusable Steinhart-Hart coefficients.

A pre-pipeline tool.  It reads raw ``.p`` files plus a CTD reference, fits one
coefficient set for the whole deployment from whatever yos actually carried a
reference, and reports whether that set is stable over time.

See ``docs/fp07_insitu_calibration_PLAN.md``.
"""

from odas_tpw.fp07cal.fit import FitResult, fit_calibration, residual_breakdown
from odas_tpw.fp07cal.gradient_lag import gradient_lag, hysteresis_lag
from odas_tpw.fp07cal.lag import LagResult, highpass, pressure_offset, temperature_lag
from odas_tpw.fp07cal.logr import BridgeParams, coeffs_to_config, log_r, temperature
from odas_tpw.fp07cal.pairs import (
    PairConfig,
    PairSet,
    build_pairs,
    build_pairs_multi,
    estimate_clock_offset,
    estimate_lag,
)
from odas_tpw.fp07cal.series import (
    ProbeSeries,
    ReferenceSeries,
    load_hotel_reference,
    load_probe_series,
    sanitize_reference,
)
from odas_tpw.fp07cal.stability import (
    Block,
    StabilityResult,
    blocked_offsets,
    corroborates,
    drift_fit,
    t1_t2_series,
)

__all__ = [
    "Block",
    "BridgeParams",
    "FitResult",
    "LagResult",
    "PairConfig",
    "PairSet",
    "ProbeSeries",
    "ReferenceSeries",
    "StabilityResult",
    "blocked_offsets",
    "build_pairs",
    "build_pairs_multi",
    "coeffs_to_config",
    "corroborates",
    "drift_fit",
    "estimate_clock_offset",
    "estimate_lag",
    "fit_calibration",
    "gradient_lag",
    "highpass",
    "hysteresis_lag",
    "load_hotel_reference",
    "load_probe_series",
    "log_r",
    "pressure_offset",
    "residual_breakdown",
    "sanitize_reference",
    "t1_t2_series",
    "temperature",
    "temperature_lag",
]
