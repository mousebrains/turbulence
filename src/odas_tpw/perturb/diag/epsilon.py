# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""``perturb-diag epsilon`` — interactive dissipation inspector.

Overview: pcolor of the combined mean and per-probe epsilon over cast x depth.
Click a cell (or arrow-key around) to drill into that dissipation estimate's
shear wavenumber spectra (with Nasmyth fits and K_max), the profile's
dissipation diagnostics, and — from the profiles product — a row of raw
instrument diagnostics (inclinometer, accelerometer, shear probe, temperature
gradient) vs depth.
"""

from __future__ import annotations

import argparse

from odas_tpw.perturb.diag import _common, render
from odas_tpw.perturb.diag.data import load_profile_file

# Overview panels: (combo variable, mathtext title). Matches the Matlab
# diss_inspector's epsilonMean / e_1 / e_2 layout.
_FIELDS: list[tuple[str, str]] = [
    ("epsilonMean", r"$\langle\epsilon\rangle$"),
    ("e_1", r"$\epsilon_1$"),
    ("e_2", r"$\epsilon_2$"),
]


def add_arguments(p: argparse.ArgumentParser) -> None:
    """Register CLI flags for the epsilon diagnostics subcommand on *p*."""
    _common.add_common_args(p)


def run(args: argparse.Namespace) -> str:
    """Launch the interactive inspector (or write a snapshot with --out)."""
    return _common.build_and_run(
        args,
        combo_stage="diss_combo",
        profile_stage="diss",
        fields=_FIELDS,
        qc_var="qc_drop_epsilon",
        loader=load_profile_file,
        spectra_fn=render.draw_eps_spectra,
        strip_fn=render.draw_diss_strip,
        cbar_label=r"$\epsilon$  (W kg$^{-1}$)",
    )
