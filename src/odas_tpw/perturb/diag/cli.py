# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""``perturb-diag`` console-script entry point.

Interactive diagnostics for a perturb run's outputs — the sibling of
``perturb-plot`` (static sections).  Each product lives in its own module under
:mod:`odas_tpw.perturb.diag` and is wired in here as a subcommand so
``perturb-diag --help`` is the single source of truth.

Adding a new diagnostic
-----------------------
1. Create ``odas_tpw/perturb/diag/<name>.py`` exposing ``add_arguments(parser)``
   and ``run(args) -> str``.
2. Register it in :data:`_SUBCOMMANDS` below.
"""

from __future__ import annotations

import argparse

from odas_tpw.perturb.diag import chi, epsilon, mixing

# Map subcommand name -> (description, add_arguments, run).
_SUBCOMMANDS = {
    "epsilon": (
        "Interactive dissipation inspector: click a cast x depth epsilon cell "
        "to see its shear spectra (Nasmyth + K_max) and profile diagnostics.",
        epsilon.add_arguments,
        epsilon.run,
    ),
    "chi": (
        "Interactive thermal-dissipation inspector: click a cast x depth chi "
        "cell to see its temperature-gradient spectra (Batchelor + noise floor) "
        "and profile diagnostics.",
        chi.add_arguments,
        chi.run,
    ),
    "mixing": (
        "Interactive mixing inspector: click a K_rho / K_T / Gamma cell for the "
        "spectrum that sets it (shear for K_rho, temperature gradient for K_T) "
        "and a strip of epsilon/chi/N2/dTdz/K_T/K_rho/Gamma vs depth.",
        mixing.add_arguments,
        mixing.run,
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="perturb-diag",
        description="Interactive diagnostics for perturb runs.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="subcommand")
    for name, (descr, add_args, _run) in _SUBCOMMANDS.items():
        sp = sub.add_parser(name, help=descr, description=descr)
        add_args(sp)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _, _, run_fn = _SUBCOMMANDS[args.command]
    run_fn(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
