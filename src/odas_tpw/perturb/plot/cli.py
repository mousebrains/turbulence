# May-2026, Claude and Pat Welch, pat@mousebrains.com
"""``perturb-plot`` console-script entry point.

Top-level dispatcher for the figures produced from a perturb run's
aggregated outputs. Every plot lives in its own sibling module under
:mod:`odas_tpw.perturb.plot` and is wired in here as a subcommand so
``perturb-plot --help`` is the single source of truth.

Adding a new plot
-----------------
1. Create ``odas_tpw/perturb/plot/<name>.py`` exposing
   ``add_arguments(parser)`` and ``run(args) -> str`` (returns the
   output path).
2. Register it in :data:`_SUBCOMMANDS` below.
"""

from __future__ import annotations

import argparse

from odas_tpw.perturb.plot import eps_chi

# Map subcommand name -> (description, add_arguments, run).
_SUBCOMMANDS = {
    "eps-chi": (
        "Pcolor of log10(epsilon), log10(chi) and log10(chi/epsilon) "
        "vs depth and cast number.",
        eps_chi.add_arguments,
        eps_chi.run,
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="perturb-plot",
        description="Aggregated-output plotting for perturb runs.",
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
