# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""Dinkum Binary Data (Slocum glider) -> perturb hotel file.

Slocum gliders write Dinkum Binary Data files: ``*.dbd``/``*.ebd`` (flight /
science) and their LZ4-compressed ``*.dcd``/``*.ecd`` counterparts. Each record
carries only the sensors that reported on that cycle; everything else is absent
(NaN). There is no single clock — the flight computer stamps ``m_present_time``,
the science computer stamps ``sci_m_present_time``, and the CTD's own print
arrival is ``sci_ctd41cp_timestamp``.

This subpackage turns a pile of those files into one hotel NetCDF that
:mod:`odas_tpw.perturb.hotel` can merge onto an instrument's clock:

- :mod:`~odas_tpw.dinkum.reader` — read DBD files into an xarray Dataset via
  whichever backend is available (``xarray-dbd``, the ``dbd2netCDF`` binary, or
  a pre-converted NetCDF).
- :mod:`~odas_tpw.dinkum.config` — the YAML schema (sensors, time base,
  projection).
- :mod:`~odas_tpw.dinkum.build` — time sanitizing, per-sensor time attribution,
  projection onto the common base, and the NetCDF write.
- :mod:`~odas_tpw.dinkum.cli` — the ``dinkum-hotel`` command.
"""

from odas_tpw.dinkum.build import (
    build_hotel,
    project_sensor,
    resolve_time_bounds,
    sanitize_time,
    time_validity,
)
from odas_tpw.dinkum.reader import available_backends, load_dinkum, sensor_inventory

__all__ = [
    "available_backends",
    "build_hotel",
    "load_dinkum",
    "project_sensor",
    "resolve_time_bounds",
    "sanitize_time",
    "sensor_inventory",
    "time_validity",
]
