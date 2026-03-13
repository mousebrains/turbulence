# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""perturb — Batch processing pipeline for Rockland VMP/MicroRider data."""

from importlib.metadata import version

from perturb.config import DEFAULTS, generate_template, load_config, merge_config
from perturb.pipeline import process_file, run_pipeline

__version__ = version("rsi-python")

__all__ = [
    "DEFAULTS",
    "generate_template",
    "load_config",
    "merge_config",
    "process_file",
    "run_pipeline",
]
