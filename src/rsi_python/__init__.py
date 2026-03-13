# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""rsi-python — Read Rockland Scientific .P files and convert to NetCDF."""

from importlib.metadata import version

__version__ = version("rsi-python")

from rsi_python.chi_io import get_chi
from rsi_python.convert import convert_all, p_to_L1, p_to_netcdf
from rsi_python.p_file import PFile, parse_config
from rsi_python.profile import extract_profiles, get_profiles

__all__ = [
    "PFile",
    "convert_all",
    "extract_profiles",
    "get_chi",
    "get_profiles",
    "p_to_L1",
    "p_to_netcdf",
    "parse_config",
]
