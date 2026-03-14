# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""perturb — Batch processing pipeline for Rockland VMP/MicroRider data."""

from importlib.metadata import version

from microstructure_tpw.perturb.config import DEFAULTS, generate_template, load_config, merge_config
from microstructure_tpw.perturb.pipeline import process_file, run_pipeline

__version__ = version("microstructure-tpw")

__all__ = [
    "DEFAULTS",
    "generate_template",
    "load_config",
    "merge_config",
    "process_file",
    "run_pipeline",
]
