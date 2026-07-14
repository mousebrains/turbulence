# odas-tpw: Ocean microstructure turbulence processing (ODAS heritage)
#
# Subpackages:
#   rsi         — Rockland Scientific instrument I/O (.p files, NetCDF, profiles)
#   chi         — Chi (thermal variance dissipation) calculation
#   scor160     — ATOMIX shear-probe benchmark processing (L1-L4)
#   processing  — Instrument-agnostic algorithms (top_trim, bottom, ct_align, etc.)
#   perturb     — Full processing pipeline (trim, merge, calibrate, compute, bin)

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("microstructure-tpw")
except PackageNotFoundError:  # pragma: no cover - running from an uninstalled source tree
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
