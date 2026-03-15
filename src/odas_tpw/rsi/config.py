# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""YAML configuration file support for rsi-tpw.

Provides loading, validation, three-way merge (defaults <- config <- CLI),
deterministic hashing, and sequential output directory management.

All config management logic lives in :mod:`odas_tpw.config_base.ConfigManager`;
this module defines rsi-specific DEFAULTS, instantiates the manager, and
re-exports the methods as module-level functions for backward compatibility.
"""

from pathlib import Path

from odas_tpw.config_base import ConfigManager, _normalize_value  # noqa: F401

# ---------------------------------------------------------------------------
# Canonical defaults — one dict per processing section
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, dict] = {
    "profiles": {
        "P_min": 0.5,
        "W_min": 0.3,
        "direction": "down",
        "min_duration": 7.0,
    },
    "epsilon": {
        "fft_length": 256,
        "diss_length": None,
        "overlap": None,
        "speed": None,
        "direction": "down",
        "goodman": True,
        "f_AA": 98.0,
        "f_limit": None,
        "fit_order": 3,
        "despike_thresh": 8,
        "despike_smooth": 0.5,
        "salinity": None,
    },
    "chi": {
        "fft_length": 512,
        "diss_length": None,
        "overlap": None,
        "speed": None,
        "direction": "down",
        "fp07_model": "single_pole",
        "goodman": True,
        "f_AA": 98.0,
        "fit_method": "iterative",
        "spectrum_model": "kraichnan",
        "salinity": None,
    },
}

_mgr = ConfigManager(DEFAULTS)

# Re-export manager methods as module-level functions
load_config = _mgr.load_config
validate_config = _mgr.validate_config
merge_config = _mgr.merge_config
canonicalize = _mgr.canonicalize
compute_hash = _mgr.compute_hash
resolve_output_dir = _mgr.resolve_output_dir
write_signature = _mgr.write_signature
write_resolved_config = _mgr.write_resolved_config

# Also keep _VALID_SECTIONS for any direct references
_VALID_SECTIONS = _mgr.valid_sections


# ---------------------------------------------------------------------------
# Template generation
# ---------------------------------------------------------------------------

_TEMPLATE = """\
# rsi-tpw configuration
# Values shown are the defaults. Uncomment and edit to customize.
# CLI flags override values in this file.

profiles:
  P_min: 0.5            # minimum pressure [dbar]
  W_min: 0.3            # minimum fall rate [dbar/s]
  direction: down       # profile direction: up or down
  min_duration: 7.0     # minimum profile duration [s]

epsilon:
  fft_length: 256       # FFT segment length [samples]
  diss_length: null     # dissipation window [samples] (null = 2 * fft_length)
  overlap: null         # window overlap [samples] (null = diss_length // 2)
  speed: null           # profiling speed [m/s] (null = from dP/dt)
  direction: down       # profile direction: up or down
  goodman: true         # Goodman coherent noise removal
  f_AA: 98.0            # anti-aliasing filter cutoff [Hz]
  f_limit: null         # upper frequency limit [Hz] (null = f_AA)
  fit_order: 3          # polynomial fit order for Nasmyth integration
  despike_thresh: 8     # despike threshold [MAD]
  despike_smooth: 0.5   # despike smoothing window [s]
  salinity: null        # salinity [PSU] (null = 35, fixed S)

chi:
  fft_length: 512       # FFT segment length [samples]
  diss_length: null     # dissipation window [samples] (null = 3 * fft_length)
  overlap: null         # window overlap [samples] (null = diss_length // 2)
  speed: null           # profiling speed [m/s] (null = from dP/dt)
  direction: down       # profile direction: up or down
  fp07_model: single_pole  # FP07 transfer function: single_pole or double_pole
  goodman: true         # Goodman coherent noise removal
  f_AA: 98.0            # anti-aliasing filter cutoff [Hz]
  fit_method: iterative # Method 2 fitting: iterative or mle
  spectrum_model: kraichnan  # theoretical spectrum: batchelor or kraichnan
  salinity: null        # salinity [PSU] (null = 35, fixed S)
"""


def generate_template(path: str | Path) -> Path:
    """Write a fully-commented template configuration file."""
    path = Path(path)
    path.write_text(_TEMPLATE)
    return path
