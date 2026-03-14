# Chi (thermal variance dissipation) calculations
#
# Computes chi from temperature microstructure spectra using
# Batchelor/Kraichnan spectral fitting (Method 2) or from known
# epsilon via Dillon & Caldwell (Method 1).

from odas_tpw.chi.batchelor import (
    KAPPA_T,
    Q_BATCHELOR,
    Q_KRAICHNAN,
    batchelor_grad,
    batchelor_kB,
    kraichnan_grad,
)
from odas_tpw.chi.fp07 import fp07_double_pole, fp07_tau, fp07_transfer
from odas_tpw.chi.l2_chi import L2ChiData, L2ChiParams, process_l2_chi
from odas_tpw.chi.l3_chi import L3ChiData, process_l3_chi
from odas_tpw.chi.l4_chi import L4ChiData, process_l4_chi_epsilon, process_l4_chi_fit

__all__ = [
    "KAPPA_T",
    "Q_BATCHELOR",
    "Q_KRAICHNAN",
    "L2ChiData",
    "L2ChiParams",
    "L3ChiData",
    "L4ChiData",
    "batchelor_grad",
    "batchelor_kB",
    "fp07_double_pole",
    "fp07_tau",
    "fp07_transfer",
    "kraichnan_grad",
    "process_l2_chi",
    "process_l3_chi",
    "process_l4_chi_epsilon",
    "process_l4_chi_fit",
]
