# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""rsi-python — Tools for Rockland Scientific microprofiler data."""

from importlib.metadata import version

__version__ = version("rsi-python")

from rsi_python.batchelor import batchelor_grad, batchelor_kB
from rsi_python.chi import get_chi
from rsi_python.convert import convert_all, p_to_netcdf
from rsi_python.dissipation import get_diss, load_channels
from rsi_python.nasmyth import nasmyth
from rsi_python.ocean import buoyancy_freq, density, visc, visc35
from rsi_python.p_file import PFile, parse_config
from rsi_python.profile import extract_profiles, get_profiles
from rsi_python.quick_look import quick_look

__all__ = [
    "PFile",
    "parse_config",
    "p_to_netcdf",
    "convert_all",
    "visc35",
    "visc",
    "density",
    "buoyancy_freq",
    "nasmyth",
    "get_profiles",
    "extract_profiles",
    "get_diss",
    "load_channels",
    "get_chi",
    "batchelor_grad",
    "batchelor_kB",
    "quick_look",
]
