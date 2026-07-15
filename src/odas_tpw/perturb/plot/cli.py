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

from odas_tpw.perturb.plot import (
    eps_chi,
    figure,
    gamma_scaling,
    overview,
    profiles,
    scalar,
)

# Map subcommand name -> (description, add_arguments, run).
_SUBCOMMANDS = {
    "figure": (
        "Render many figures from one YAML spec (presets scalar/profiles/epsilon/"
        "chi/mixing/eps-chi), resolving directories from a perturb config.",
        figure.add_arguments,
        figure.run,
    ),
    "eps-chi": (
        "Pcolor of log10(epsilon), log10(chi) and log10(chi/epsilon) "
        "vs depth and cast number.",
        eps_chi.add_arguments,
        eps_chi.run,
    ),
    "overview": (
        "Per-section overview: epsilon and chi (two full-width rows) over a "
        "context row chosen by --bottom (ctd: dT/dz, CT, salinity; mixing: "
        "K_rho, K_T, Gamma), all vs depth and the section x-axis.",
        overview.add_arguments,
        overview.run,
    ),
    "scalar": (
        "Depth-vs-x scalar sections (T/S/density/...) from the CTD combo, "
        "with time/latitude/longitude/distance x-axis methods.",
        scalar.add_arguments,
        scalar.run,
    ),
    "profiles": (
        "Depth-vs-x sections of binned slow channels (T1/T2/N2/dTdz) per cast.",
        profiles.PROFILES.add_arguments,
        profiles.PROFILES.run,
    ),
    "epsilon": (
        "Depth-vs-x sections of binned epsilon (dissipation) per cast.",
        profiles.EPSILON.add_arguments,
        profiles.EPSILON.run,
    ),
    "chi": (
        "Depth-vs-x sections of binned chi (thermal-variance dissipation) per cast.",
        profiles.CHI.add_arguments,
        profiles.CHI.run,
    ),
    "mixing": (
        "Depth-vs-x sections of binned mixing (K_T/Gamma/K_rho) per cast.",
        profiles.MIXING.add_arguments,
        profiles.MIXING.run,
    ),
    "gamma-scaling": (
        "Mixing-efficiency scaling scatter (Lewin et al. 2025 Fig. 5): "
        "Gamma vs R_OT, Re_b, and (with --adcp) Ri_g per window, plus a "
        "Thorpe-route comparison figure.",
        gamma_scaling.add_arguments,
        gamma_scaling.run,
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
    from odas_tpw._completion import enable_argcomplete

    parser = build_parser()
    enable_argcomplete(parser)
    args = parser.parse_args(argv)
    _, _, run_fn = _SUBCOMMANDS[args.command]
    run_fn(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
