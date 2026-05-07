# May-2026, Claude and Pat Welch, pat@mousebrains.com
"""Aggregated-output plotting for perturb runs.

The :mod:`odas_tpw.perturb.plot` subpackage hosts the figures produced
from the ``*_combo_NN/`` outputs of a perturb pipeline. They are
exposed as subcommands of the ``perturb-plot`` console script (see
``pyproject.toml``); ``perturb-plot --help`` lists what is available.

Modules
-------
- :mod:`odas_tpw.perturb.plot.layout` — shared depth/cast layout helpers.
- :mod:`odas_tpw.perturb.plot.eps_chi` — pcolor of log10(epsilon),
  log10(chi) and log10(chi/epsilon) vs depth and cast number.
- :mod:`odas_tpw.perturb.plot.cli` — argparse dispatcher.
"""
