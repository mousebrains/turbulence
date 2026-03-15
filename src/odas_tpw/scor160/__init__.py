# SCOR/ATOMIX benchmark processing (scor160-tpw)
#
# Reproduces L2, L3, and L4 results from the ATOMIX shear-probe
# benchmark datasets following the best-practices recommendations
# in Lueck et al. (2024), doi:10.3389/fmars.2024.1334327.

from odas_tpw.scor160.io import (
    AtomixData,
    L1Data,
    L2Data,
    L2Params,
    L3Data,
    L3Params,
    L4Data,
    read_atomix,
)
from odas_tpw.scor160.l2 import process_l2
from odas_tpw.scor160.l3 import process_l3
from odas_tpw.scor160.l4 import process_l4
from odas_tpw.scor160.profile import get_profiles, smooth_fall_rate

__all__ = [
    "AtomixData",
    "L1Data",
    "L2Data",
    "L2Params",
    "L3Data",
    "L3Params",
    "L4Data",
    "get_profiles",
    "process_l2",
    "process_l3",
    "process_l4",
    "read_atomix",
    "smooth_fall_rate",
]
