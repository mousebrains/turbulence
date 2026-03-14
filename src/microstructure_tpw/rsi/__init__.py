# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""microstructure-tpw.rsi — Read Rockland Scientific .P files and convert to NetCDF."""

from importlib.metadata import version

__version__ = version("microstructure-tpw")

from microstructure_tpw.rsi.chi_io import get_chi
from microstructure_tpw.rsi.convert import convert_all, p_to_L1, p_to_netcdf
from microstructure_tpw.rsi.dissipation import get_diss
from microstructure_tpw.rsi.p_file import PFile, parse_config
from microstructure_tpw.rsi.profile import extract_profiles, get_profiles

__all__ = [
    "PFile",
    "convert_all",
    "extract_profiles",
    "get_chi",
    "get_diss",
    "get_profiles",
    "p_to_L1",
    "p_to_netcdf",
    "parse_config",
]
