# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""``perturb-diag mixing`` — interactive mixing-coefficient inspector.

Overview: K_rho, K_T and Gamma over cast x depth (from chi_combo, populated by a
mixing run with CTD).  The drill-down is *adaptive* — it follows the physics of
the clicked cell: K_rho is set by shear (epsilon), so clicking a K_rho cell
shows that cast's shear spectrum (from the diss file); K_T is set by the
temperature gradient (chi), so a K_T cell shows its Batchelor spectrum (from the
chi file); Gamma couples both, so it points at the two.  Below the spectra, a
strip of epsilon / chi / N^2 / dT-dz / K_T / K_rho / Gamma vs depth, and the raw
instrument-diagnostics row.
"""

from __future__ import annotations

import argparse
import os

from odas_tpw.perturb import resolve
from odas_tpw.perturb.diag import _common, render
from odas_tpw.perturb.diag.data import (
    MixingCellSource,
    apply_sections,
    load_overview,
)
from odas_tpw.perturb.diag.inspector import DiagInspector

# Overview panels: (chi_combo variable, mathtext title). Each has its own units
# (m^2/s, m^2/s, dimensionless), so the inspector scales them independently.
_FIELDS: list[tuple[str, str]] = [
    ("K_rho", r"$K_\rho$"),
    ("K_T", r"$K_T$"),
    ("Gamma", r"$\Gamma$"),
]


def add_arguments(p: argparse.ArgumentParser) -> None:
    """Register CLI flags for the mixing diagnostics subcommand on *p*."""
    _common.add_common_args(p)


def run(args: argparse.Namespace) -> str:
    """Launch the interactive inspector (or write a snapshot with --out)."""
    args.root = resolve.require_root(args)

    combo_dir = resolve.resolve_for_args(args, "chi_combo")
    if combo_dir is None:
        raise SystemExit(f"No chi_combo dir under {args.root}")
    combo_path = os.path.join(combo_dir, "combo.nc")

    chi_dir = resolve.resolve_for_args(args, "chi", optional=True, conflict_ok=True)
    diss_dir = resolve.resolve_for_args(args, "diss", optional=True, conflict_ok=True)
    if chi_dir is None or diss_dir is None:
        raise SystemExit(
            f"perturb-diag mixing needs both per-profile chi_NN and diss_NN dirs "
            f"under {args.root} for the adaptive drill-down"
        )

    data = load_overview(
        combo_path, tuple(n for n, _ in _FIELDS),
        qc_var="qc_drop_chi", apply_qc=args.apply_qc,
    )
    data = apply_sections(data, args.sections, args.select)

    profiles_dir = _common.resolve_profiles_dir(args)
    source = MixingCellSource(diss_dir, chi_dir, profiles_dir=profiles_dir)
    title = args.title or os.path.basename(os.path.normpath(args.root))

    inspector = DiagInspector(
        data, source,
        field_specs=_FIELDS,
        spectra_fn=render.draw_mixing_spectra,
        strip_fn=render.draw_mixing_strip,
        n_strip=7,
        diag_fn=render.draw_diag_strip if profiles_dir else None,
        n_diag=render.DIAG_PANEL_COUNT if profiles_dir else 0,
        title=title,
        per_field_clim=True,  # K_rho/K_T/Gamma differ in units — scale each alone
        clim=tuple(args.clim) if args.clim else None,
        gap_seconds=args.gap_seconds,
    )

    if args.out:
        out = str(args.out)
        inspector.snapshot(out)
        print(f"Wrote {out}")
        return out
    inspector.show()
    return ""
