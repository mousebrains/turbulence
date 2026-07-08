# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""``perturb-diag chi`` — interactive thermal-dissipation (chi) inspector.

The chi sibling of the epsilon inspector: overview pcolor of the combined mean
and per-probe chi over cast x depth.  Click a cell to drill into that estimate's
temperature-gradient wavenumber spectra (Batchelor/Kraichnan fits + FP07 noise
floor), the profile's chi diagnostics, and the raw instrument-diagnostics row.
"""

from __future__ import annotations

import argparse

from odas_tpw.perturb.diag import _common, render
from odas_tpw.perturb.diag.data import load_chi_profile_file

# Overview panels: (combo variable, mathtext title). Mirrors the epsilon layout.
_FIELDS: list[tuple[str, str]] = [
    ("chiMean", r"$\langle\chi\rangle$"),
    ("chi_1", r"$\chi_1$"),
    ("chi_2", r"$\chi_2$"),
]


def add_arguments(p: argparse.ArgumentParser) -> None:
    """Register CLI flags for the chi diagnostics subcommand on *p*."""
    _common.add_common_args(p)


def run(args: argparse.Namespace) -> str:
    """Launch the interactive inspector (or write a snapshot with --out)."""
    return _common.build_and_run(
        args,
        combo_stage="chi_combo",
        profile_stage="chi",
        fields=_FIELDS,
        qc_var="qc_drop_chi",
        loader=load_chi_profile_file,
        spectra_fn=render.draw_chi_spectra,
        strip_fn=render.draw_chi_strip,
        cbar_label=r"$\chi$  (K$^2$ s$^{-1}$)",
    )
