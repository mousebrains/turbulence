# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""rsi-python — Tools for Rockland Scientific microprofiler data."""

from importlib.metadata import version

__version__ = version("rsi-python")

from rsi_python.batchelor import batchelor_grad, batchelor_kB
from rsi_python.chi import get_chi
from rsi_python.convert import convert_all, p_to_netcdf
from rsi_python.diss_look import diss_look
from rsi_python.dissipation import get_diss, load_channels
from rsi_python.nasmyth import nasmyth, nasmyth_grid
from rsi_python.ocean import buoyancy_freq, density, visc, visc35
from rsi_python.p_file import PFile, parse_config
from rsi_python.profile import extract_profiles, get_profiles
from rsi_python.quick_look import quick_look
from rsi_python.shear_noise import noise_shearchannel as shear_noise

__all__ = [
    "PFile",
    "batchelor_grad",
    "batchelor_kB",
    "buoyancy_freq",
    "convert_all",
    "density",
    "diss_look",
    "extract_profiles",
    "get_chi",
    "get_diss",
    "get_profiles",
    "load_channels",
    "nasmyth",
    "nasmyth_grid",
    "p_to_netcdf",
    "parse_config",
    "quick_look",
    "shear_noise",
    "visc",
    "visc35",
]
