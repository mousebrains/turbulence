# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""Interactive diagnostics for perturb outputs (``perturb-diag``).

Sibling of :mod:`odas_tpw.perturb.plot`, which produces static PNG sections.
Where ``perturb-plot`` renders and saves, ``perturb-diag`` opens a single
interactive matplotlib window: a time x depth pcolor overview whose cells,
when clicked, drive per-cell spectra and per-profile diagnostic panels drawn
in the same figure.

Layout mirrors ``plot/``: one module per product exposing ``add_arguments``
and ``run``, wired into :mod:`odas_tpw.perturb.diag.cli`.
"""
