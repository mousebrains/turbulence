# Nov-2026, Claude and Pat Welch, pat@mousebrains.com
"""Raw counts a physical FP07 bead gives for a temperature series.

perturb's per-file fit refuses a curve no FP07 has (issue #257): it must be NTC,
with a local beta ``1/(d(1/T_K)/dL)`` within 2x of ``fp07cal.prior.FP07_BETA_K``
that changes by at most 2%/K (``perturb.fp07_cal._not_an_fp07``).  The
synthetic thermistors some tests used to build (``5000 - 800*T`` counts, or an L
scaled by 500) had a local beta of -8580 K or 500 K, so they are generated here
from a real bead's curve instead: ``1/T_K = 1/T0 + L/beta``, inverted through
the half-bridge that ``fp07cal.logr.log_r`` reads.
"""

from __future__ import annotations

import numpy as np

BETA_K = 3050.0  # inside the FP07 sheet's 2995-3105 K
T0_K = 288.15  # L = 0 at 15 degC

# The bridge the perturb FP07 tests configure.
BRIDGE = {"a": 0.0, "b": 1.0, "g": 1.0, "e_b": 2.5, "adc_fs": 5.0, "adc_bits": 16}


def fp07_counts(
    T_degC: np.ndarray, bridge: dict | None = None, beta_K: float = BETA_K
) -> np.ndarray:
    """Counts whose ``log_r`` L gives *T_degC* through ``1/T_K = 1/T0 + L/beta``.

    *beta_K* other than the bead's stands for a bead whose configured L is
    scaled (Taiwan 2013 husker: ~4300-4600 K, cause unverified).
    """
    p = {**BRIDGE, **(bridge or {})}
    L = beta_K * (1.0 / (np.asarray(T_degC, dtype=np.float64) + 273.15) - 1.0 / T0_K)
    z = (1.0 - np.exp(L)) / (1.0 + np.exp(L))  # inverse of L = ln((1-z)/(1+z))
    scale = 2.0 ** int(p["adc_bits"]) / float(p["adc_fs"]) * float(p["g"]) * float(p["e_b"]) / 2.0
    return float(p["a"]) + float(p["b"]) * z * scale
