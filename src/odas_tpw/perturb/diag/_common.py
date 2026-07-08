# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""Shared CLI wiring for ``perturb-diag`` product inspectors.

``epsilon`` and ``chi`` are the same inspector over a different product: a
combined-mean + per-probe overview whose cells drill into per-profile spectra.
Both register the same flags (:func:`add_common_args`) and run the same builder
(:func:`build_and_run`), differing only in the stage names, overview fields, QC
flag, per-profile loader and renderers.  (``mixing`` drills adaptively into two
products and has its own builder.)
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable

from odas_tpw.perturb import resolve
from odas_tpw.perturb.diag import render
from odas_tpw.perturb.diag.data import (
    EpsilonCellSource,
    apply_sections,
    load_overview,
)
from odas_tpw.perturb.diag.inspector import DiagInspector, SpectraFn, StripFn


def add_common_args(p: argparse.ArgumentParser) -> None:
    """Register the flags shared by every ``perturb-diag`` product subcommand."""
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
                   help="color limits as log10 of the field's units "
                        "(default: 1/99%% quantile of the data)")
    p.add_argument("--gap-seconds", type=float, default=600,
                   help="split casts into clusters when the gap exceeds this "
                        "(default 10 min)")
    p.add_argument("--apply-qc", dest="apply_qc", action="store_true",
                   default=True,
                   help="NaN overview bins flagged by the product's QC drop "
                        "(default)")
    p.add_argument("--no-qc", dest="apply_qc", action="store_false",
                   help="ignore the QC drop flag and show raw values")
    p.add_argument("--out", default=None,
                   help="save a static snapshot of a representative cell to this "
                        "PNG instead of opening the interactive window")
    p.add_argument("--diag", dest="diag", action="store_true", default=True,
                   help="show the raw instrument-diagnostics row "
                        "(inclinometer/accel/shear/temp-gradient) below the "
                        "drill-down (default; needs the profiles_NN dir)")
    p.add_argument("--no-diag", dest="diag", action="store_false",
                   help="hide the raw instrument-diagnostics row")


def resolve_profiles_dir(args: argparse.Namespace) -> str | None:
    """The profiles dir for the raw-diagnostics row, or None (with a note)."""
    if not args.diag:
        return None
    profiles_dir = resolve.resolve_for_args(
        args, "profiles", optional=True, conflict_ok=True
    )
    if profiles_dir is None:
        print("no profiles_NN dir found; raw diagnostics row disabled")
    return profiles_dir


def build_and_run(
    args: argparse.Namespace,
    *,
    combo_stage: str,
    profile_stage: str,
    fields: list[tuple[str, str]],
    qc_var: str,
    loader: Callable[[str], object],
    spectra_fn: SpectraFn,
    strip_fn: StripFn,
    cbar_label: str,
) -> str:
    """Build and launch a single-product inspector (epsilon/chi)."""
    args.root = resolve.require_root(args)

    combo_dir = resolve.resolve_for_args(args, combo_stage)
    if combo_dir is None:
        raise SystemExit(f"No {combo_stage} dir under {args.root}")
    combo_path = os.path.join(combo_dir, "combo.nc")

    prof_dir = resolve.resolve_for_args(
        args, profile_stage, optional=True, conflict_ok=True
    )
    if prof_dir is None:
        raise SystemExit(
            f"No per-profile {profile_stage}_NN dir under {args.root}; "
            "perturb-diag needs it for the spectra drill-down"
        )

    data = load_overview(
        combo_path, tuple(n for n, _ in fields),
        qc_var=qc_var, apply_qc=args.apply_qc,
    )
    data = apply_sections(data, args.sections, args.select)

    profiles_dir = resolve_profiles_dir(args)
    source = EpsilonCellSource(prof_dir, profiles_dir=profiles_dir, loader=loader)
    title = args.title or os.path.basename(os.path.normpath(args.root))

    inspector = DiagInspector(
        data, source,
        field_specs=fields,
        spectra_fn=spectra_fn,
        strip_fn=strip_fn,
        n_strip=5,
        diag_fn=render.draw_diag_strip if profiles_dir else None,
        n_diag=render.DIAG_PANEL_COUNT if profiles_dir else 0,
        title=title,
        cbar_label=cbar_label,
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
