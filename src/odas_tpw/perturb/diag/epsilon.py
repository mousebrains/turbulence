# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""``perturb-diag epsilon`` — interactive dissipation inspector.

Overview: pcolor of the combined mean and per-probe epsilon over cast x depth.
Click a cell (or arrow-key around) to drill into that dissipation estimate's
shear wavenumber spectra (with Nasmyth fits and K_max) and the profile's
dissipation diagnostics.
"""

from __future__ import annotations

import argparse
import os

from odas_tpw.perturb import resolve
from odas_tpw.perturb.diag import render
from odas_tpw.perturb.diag.data import (
    EpsilonCellSource,
    apply_sections,
    load_overview,
)
from odas_tpw.perturb.diag.inspector import DiagInspector

# Overview panels: (combo variable, mathtext title). Matches the Matlab
# diss_inspector's epsilonMean / e_1 / e_2 layout.
_FIELDS: list[tuple[str, str]] = [
    ("epsilonMean", r"$\langle\epsilon\rangle$"),
    ("e_1", r"$\epsilon_1$"),
    ("e_2", r"$\epsilon_2$"),
]


def add_arguments(p: argparse.ArgumentParser) -> None:
    """Register CLI flags for the epsilon diagnostics subcommand on *p*."""
    p.add_argument("--root", default=None,
                   help="perturb output root (e.g. results/). "
                        "Required unless --config is given.")
    resolve.add_resolve_args(p)
    p.add_argument("--title", default=None,
                   help="figure title prefix (default: basename of --root)")
    p.add_argument("--sections", default=None,
                   help="sections YAML (perturb-plot format): narrow the overview "
                        "to the selected section(s)' UTC start/stop window. The "
                        "xaxis method is ignored (the overview is always cast x "
                        "depth).")
    p.add_argument("--select", action="append", default=None, metavar="NAME",
                   help="show only the named section(s) from --sections "
                        "(repeatable, or comma-separated). Default: every section "
                        "in the file. Only valid together with --sections.")
    p.add_argument("--clim", type=float, nargs=2, default=None,
                   metavar=("LOG10_MIN", "LOG10_MAX"),
                   help="epsilon color limits as log10(W/kg) "
                        "(default: 1/99%% quantile of the data)")
    p.add_argument("--gap-seconds", type=float, default=600,
                   help="split casts into clusters when the gap exceeds this "
                        "(default 10 min)")
    p.add_argument("--apply-qc", dest="apply_qc", action="store_true",
                   default=True,
                   help="NaN overview bins where qc_drop_epsilon is set (default)")
    p.add_argument("--no-qc", dest="apply_qc", action="store_false",
                   help="ignore qc_drop_epsilon and show raw values")
    p.add_argument("--out", default=None,
                   help="save a static snapshot of a representative cell to this "
                        "PNG instead of opening the interactive window")


def run(args: argparse.Namespace) -> str:
    """Launch the interactive inspector (or write a snapshot with --out)."""
    args.root = resolve.require_root(args)

    combo_dir = resolve.resolve_for_args(args, "diss_combo")
    if combo_dir is None:
        raise SystemExit(f"No diss_combo dir under {args.root}")
    combo_path = os.path.join(combo_dir, "combo.nc")

    # The per-profile diss dir holds the spectra; it is required here (unlike
    # perturb-plot, where it only supplies title metadata).
    diss_dir = resolve.resolve_for_args(args, "diss", optional=True, conflict_ok=True)
    if diss_dir is None:
        raise SystemExit(
            f"No per-profile diss_NN dir under {args.root}; "
            "perturb-diag needs it for the spectra drill-down"
        )

    data = load_overview(
        combo_path, tuple(n for n, _ in _FIELDS), apply_qc=args.apply_qc
    )
    data = apply_sections(data, args.sections, args.select)
    source = EpsilonCellSource(diss_dir)
    title = args.title or os.path.basename(os.path.normpath(args.root))

    inspector = DiagInspector(
        data, source,
        field_specs=_FIELDS,
        spectra_fn=render.draw_eps_spectra,
        strip_fn=render.draw_diss_strip,
        n_strip=5,
        title=title,
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
