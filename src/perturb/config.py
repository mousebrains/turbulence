# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""YAML configuration file support for perturb.

Provides loading, validation, three-way merge (defaults <- config <- CLI),
deterministic hashing, and sequential output directory management.

Structural copy of rsi_python/config.py, referencing perturb's own DEFAULTS
(13 sections).  Imports only ``_normalize_value`` from rsi_python.config
since it is a pure helper.
"""

import hashlib
import json
from pathlib import Path

from ruamel.yaml import YAML

from rsi_python.config import _normalize_value

# ---------------------------------------------------------------------------
# Canonical defaults — one dict per processing section
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, dict] = {
    "files": {
        "p_file_root": "VMP/",
        "p_file_pattern": "**/*.p",
        "output_root": "results/",
        "trim": True,
        "merge": False,
    },
    "gps": {
        "source": "nan",
        "lat": None,
        "lon": None,
        "file": None,
        "time_col": "t",
        "lat_col": "lat",
        "lon_col": "lon",
        "max_time_diff": 60,
    },
    "hotel": {
        "enable": False,
        "file": None,
        "time_column": "time",
        "time_format": "auto",
        "channels": {},
        "fast_channels": ["speed", "P"],
        "interpolation": "pchip",
    },
    "profiles": {
        "P_min": 0.5,
        "W_min": 0.3,
        "direction": "down",
        "min_duration": 7.0,
        "diagnostics": False,
    },
    "fp07": {
        "calibrate": True,
        "order": 2,
        "max_lag_seconds": 10,
        "reference": "JAC_T",
        "must_be_negative": True,
    },
    "ct": {
        "align": True,
        "T_name": "JAC_T",
        "C_name": "JAC_C",
    },
    "bottom": {
        "enable": False,
        "depth_window": 4.0,
        "depth_minimum": 10.0,
        "speed_factor": 0.3,
        "median_factor": 1.0,
        "vibration_frequency": 16,
        "vibration_factor": 4.0,
    },
    "top_trim": {
        "enable": False,
        "dz": 0.5,
        "min_depth": 1.0,
        "max_depth": 50.0,
        "quantile": 0.6,
    },
    "epsilon": {
        "fft_length": 256,
        "diss_length": None,
        "overlap": None,
        "goodman": True,
        "f_AA": 98.0,
        "f_limit": None,
        "fit_order": 3,
        "despike_thresh": 8,
        "despike_smooth": 0.5,
        "salinity": None,
        "epsilon_minimum": 1e-13,
        "T_source": None,
        "T1_norm": 1.0,
        "T2_norm": 1.0,
        "diagnostics": False,
    },
    "chi": {
        "enable": False,
        "fft_length": 512,
        "diss_length": None,
        "overlap": None,
        "fp07_model": "single_pole",
        "goodman": True,
        "f_AA": 98.0,
        "fit_method": "iterative",
        "spectrum_model": "kraichnan",
        "salinity": None,
        "diagnostics": False,
    },
    "ctd": {
        "enable": True,
        "bin_width": 0.5,
        "T_name": "JAC_T",
        "C_name": "JAC_C",
        "variables": None,
        "method": "mean",
        "diagnostics": False,
    },
    "binning": {
        "method": "depth",
        "width": 1.0,
        "aggregation": "mean",
        "diss_width": None,
        "diss_aggregation": None,
        "chi_width": None,
        "chi_aggregation": None,
        "diagnostics": False,
    },
    "netcdf": {
        "title": None,
        "summary": None,
        "institution": None,
        "creator_name": None,
        "creator_email": None,
        "creator_url": None,
        "creator_type": None,
        "creator_institution": None,
        "contributor_name": None,
        "contributor_role": None,
        "project": None,
        "program": None,
        "id": None,
        "naming_authority": None,
        "source": None,
        "platform": None,
        "platform_vocabulary": None,
        "instrument": None,
        "instrument_vocabulary": None,
        "processing_level": None,
        "license": None,
        "references": None,
        "comment": None,
        "acknowledgement": None,
        "date_created": None,
        "date_modified": None,
        "date_issued": None,
        "geospatial_lat_min": None,
        "geospatial_lat_max": None,
        "geospatial_lon_min": None,
        "geospatial_lon_max": None,
        "geospatial_vertical_min": None,
        "geospatial_vertical_max": None,
        "geospatial_vertical_positive": "down",
        "time_coverage_start": None,
        "time_coverage_end": None,
        "time_coverage_duration": None,
        "time_coverage_resolution": None,
        "keywords": None,
        "keywords_vocabulary": None,
        "standard_name_vocabulary": "CF Standard Name Table v83",
        "Conventions": "CF-1.8, ACDD-1.3",
        "history": None,
    },
    "parallel": {
        "jobs": 1,
    },
}

# Keys excluded from hashing — toggling diagnostics should not create new output dirs
_HASH_EXCLUDE_KEYS = frozenset({"diagnostics"})

# All valid section names
_VALID_SECTIONS = frozenset(DEFAULTS)

# ---------------------------------------------------------------------------
# Load / validate
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> dict[str, dict]:
    """Load a YAML config file and return as plain dict of dicts.

    Parameters
    ----------
    path : str or Path
        Path to YAML configuration file.

    Returns
    -------
    dict mapping section names to parameter dicts.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file contains unknown sections or keys.
    """
    yaml = YAML()
    with open(path, encoding="utf-8") as fh:
        raw = yaml.load(fh)
    if raw is None:
        return {}
    config = {str(k): (dict(v) if v is not None else {}) for k, v in raw.items()}
    validate_config(config)
    return config


def validate_config(config: dict[str, dict]) -> None:
    """Raise ValueError if *config* has unknown sections or keys."""
    for section, params in config.items():
        if section not in _VALID_SECTIONS:
            raise ValueError(
                f"Unknown config section: {section!r}. "
                f"Valid sections: {sorted(_VALID_SECTIONS)}"
            )
        valid_keys = set(DEFAULTS[section])
        unknown = set(params) - valid_keys
        if unknown:
            raise ValueError(
                f"Unknown key(s) in [{section}]: {sorted(unknown)}. "
                f"Valid keys: {sorted(valid_keys)}"
            )


# ---------------------------------------------------------------------------
# Three-way merge
# ---------------------------------------------------------------------------


def merge_config(
    section: str,
    file_values: dict | None = None,
    cli_overrides: dict | None = None,
) -> dict:
    """Merge defaults <- config-file values <- CLI overrides.

    None values in either *file_values* or *cli_overrides* are treated as
    "not specified" and do not mask earlier layers.

    Returns a clean kwargs dict with all None values removed (so the
    downstream function receives only the parameters that were explicitly set
    or have non-None defaults).
    """
    if section not in DEFAULTS:
        raise ValueError(f"Unknown section: {section!r}")

    merged = dict(DEFAULTS[section])

    if file_values:
        for k, v in file_values.items():
            if k in merged and v is not None:
                merged[k] = v

    if cli_overrides:
        for k, v in cli_overrides.items():
            if k in merged and v is not None:
                merged[k] = v

    # Strip None values so downstream functions use their own defaults
    return {k: v for k, v in merged.items() if v is not None}


# ---------------------------------------------------------------------------
# Canonicalization and hashing
# ---------------------------------------------------------------------------


def _canonicalize_section(section: str, params: dict) -> dict:
    """Canonicalize a single section's parameters into a normalized dict.

    Keys in ``_HASH_EXCLUDE_KEYS`` (e.g. ``diagnostics``) are omitted so
    toggling them does not change the hash.
    """
    base = dict(DEFAULTS[section])
    for k, v in params.items():
        if k in base and v is not None:
            base[k] = v
    return {
        k: _normalize_value(v)
        for k, v in sorted(base.items())
        if k not in _HASH_EXCLUDE_KEYS
    }


def canonicalize(
    section: str,
    params: dict,
    upstream: list[tuple[str, dict]] | None = None,
) -> str:
    """Produce a deterministic JSON string for hashing.

    Overlays *params* on the section defaults, normalizes types, and
    returns compact sorted JSON.

    Parameters
    ----------
    section : str
        The primary config section (e.g. "epsilon", "chi").
    params : dict
        Resolved parameters for the primary section.
    upstream : list of (section, params) tuples, optional
        Upstream sections whose parameters contributed to this result.
        The canonical output includes all sections keyed by name, ensuring
        that changing an upstream parameter produces a different hash.
    """
    sections = {}
    if upstream:
        for up_section, up_params in upstream:
            sections[up_section] = _canonicalize_section(up_section, up_params)
    sections[section] = _canonicalize_section(section, params)
    return json.dumps(sections, sort_keys=True, separators=(",", ":"))


def compute_hash(
    section: str,
    params: dict,
    upstream: list[tuple[str, dict]] | None = None,
) -> str:
    """SHA-256 hex digest of the canonical representation."""
    canonical = canonicalize(section, params, upstream=upstream)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Output directory management
# ---------------------------------------------------------------------------


def resolve_output_dir(
    base: str | Path,
    prefix: str,
    section: str,
    params: dict,
    upstream: list[tuple[str, dict]] | None = None,
) -> Path:
    """Find or create a sequential output directory matching *params*.

    Scans ``base/{prefix}_NN/`` directories for a hash-matching signature file.
    If found, returns that directory.  Otherwise creates the next sequential
    directory and writes the signature file.
    """
    base = Path(base)
    target_hash = compute_hash(section, params, upstream=upstream)

    # Scan existing sequential directories
    max_seq = -1
    import glob as globmod

    pattern = str(base / f"{prefix}_[0-9][0-9]")
    for d in sorted(globmod.glob(pattern)):
        dp = Path(d)
        try:
            seq = int(dp.name.split("_")[-1])
        except ValueError:
            continue
        max_seq = max(max_seq, seq)
        if not dp.is_dir():
            continue
        sig_file = dp / f".params_sha256_{target_hash}"
        if sig_file.exists():
            return dp

    # No match — create next sequential directory
    next_seq = max_seq + 1
    new_dir = base / f"{prefix}_{next_seq:02d}"
    new_dir.mkdir(parents=True, exist_ok=True)
    write_signature(new_dir, section, params, upstream=upstream)
    return new_dir


def write_signature(
    directory: Path,
    section: str,
    params: dict,
    upstream: list[tuple[str, dict]] | None = None,
) -> Path:
    """Write a ``.params_sha256_<hash>`` signature file with canonical JSON contents."""
    h = compute_hash(section, params, upstream=upstream)
    canonical = canonicalize(section, params, upstream=upstream)
    sig_file = directory / f".params_sha256_{h}"
    sig_file.write_text(canonical)
    return sig_file


def write_resolved_config(
    directory: Path,
    section: str,
    params: dict,
    upstream: list[tuple[str, dict]] | None = None,
) -> Path:
    """Write a human-readable ``config.yaml`` with the resolved parameters."""
    yaml = YAML()
    yaml.default_flow_style = False

    data = {}

    if upstream:
        for up_section, up_params in upstream:
            resolved = dict(DEFAULTS[up_section])
            for k, v in up_params.items():
                if k in resolved and v is not None:
                    resolved[k] = v
            data[up_section] = resolved

    resolved = dict(DEFAULTS[section])
    for k, v in params.items():
        if k in resolved and v is not None:
            resolved[k] = v
    data[section] = resolved

    out = directory / "config.yaml"
    with open(out, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh)
    return out


# ---------------------------------------------------------------------------
# Template generation
# ---------------------------------------------------------------------------

_TEMPLATE = """\
# perturb configuration
# Values shown are the defaults. Uncomment and edit to customize.
# CLI flags override values in this file.

files:
  p_file_root: "VMP/"
  p_file_pattern: "**/*.p"
  output_root: "results/"
  trim: true
  merge: false

gps:
  source: "nan"           # nan | fixed | csv | netcdf
  lat: null
  lon: null
  file: null
  time_col: "t"
  lat_col: "lat"
  lon_col: "lon"
  max_time_diff: 60       # warning threshold [s]

hotel:
  enable: false
  file: null              # path to hotel file (CSV, NetCDF, or .mat)
  time_column: "time"     # time column/variable name
  time_format: "auto"     # auto | seconds | epoch | iso
  channels: {}            # hotel_col → output_name (empty = all)
  fast_channels:          # channels interpolated onto fast time axis
    - speed
    - P
  interpolation: "pchip"  # pchip | linear

profiles:
  P_min: 0.5              # minimum pressure [dbar]
  W_min: 0.3              # minimum fall rate [dbar/s]
  direction: "down"       # profile direction: up or down
  min_duration: 7.0       # minimum profile duration [s]
  diagnostics: false      # write T1_raw, C_raw, W, cal attrs

fp07:
  calibrate: true
  order: 2                # Steinhart-Hart polynomial order
  max_lag_seconds: 10     # max cross-correlation lag [s]
  reference: "JAC_T"      # reference temperature channel
  must_be_negative: true  # expect negative lag (falling VMP)

ct:
  align: true
  T_name: "JAC_T"         # temperature channel for alignment
  C_name: "JAC_C"         # conductivity channel for alignment

bottom:
  enable: false
  depth_window: 4.0       # depth window for crash detection [m]
  depth_minimum: 10.0     # minimum depth to search [m]
  speed_factor: 0.3       # speed reduction factor
  median_factor: 1.0      # acceleration std dev filter factor
  vibration_frequency: 16 # vibration binning frequency [Hz]
  vibration_factor: 4.0   # vibration std dev acceptance factor

top_trim:
  enable: false
  dz: 0.5                 # depth bin size [m]
  min_depth: 1.0          # minimum search depth [m]
  max_depth: 50.0         # maximum search depth [m]
  quantile: 0.6           # quantile threshold for trim detection

epsilon:
  fft_length: 256         # FFT segment length [samples]
  diss_length: null       # dissipation window [samples] (null = 2 * fft_length)
  overlap: null           # window overlap [samples] (null = diss_length // 2)
  goodman: true           # Goodman coherent noise removal
  f_AA: 98.0              # anti-aliasing filter cutoff [Hz]
  f_limit: null           # upper frequency limit [Hz] (null = f_AA)
  fit_order: 3            # polynomial fit order for Nasmyth integration
  despike_thresh: 8       # despike threshold [MAD]
  despike_smooth: 0.5     # despike smoothing window [s]
  salinity: null          # salinity [PSU] (null = 35, fixed S)
  epsilon_minimum: 1.0e-13  # floor for small epsilon values
  T_source: null          # temperature source for viscosity (null = blend T1/T2)
  T1_norm: 1.0            # T1 blending weight
  T2_norm: 1.0            # T2 blending weight
  diagnostics: false      # write spec_shear_raw, spike_count, probe_mask

chi:
  enable: false           # chi is optional, separate stage after diss
  fft_length: 512         # FFT segment length [samples]
  diss_length: null       # dissipation window [samples] (null = 3 * fft_length)
  overlap: null           # window overlap [samples] (null = diss_length // 2)
  fp07_model: "single_pole"  # FP07 transfer function model
  goodman: true           # Goodman coherent noise removal
  f_AA: 98.0              # anti-aliasing filter cutoff [Hz]
  fit_method: "iterative" # Method 2 fitting: iterative or mle
  spectrum_model: "kraichnan"  # theoretical spectrum: batchelor or kraichnan
  salinity: null          # salinity [PSU] (null = 35, fixed S)
  diagnostics: false      # write spec_noise, fp07_transfer, gradT_raw

ctd:
  enable: true
  bin_width: 0.5          # time bin width [s]
  T_name: "JAC_T"         # temperature channel
  C_name: "JAC_C"         # conductivity channel
  variables: null         # channels to bin (null = auto-detect)
  method: "mean"          # aggregation method
  diagnostics: false      # write n_samples, *_std per bin

binning:
  method: "depth"         # depth | time
  width: 1.0              # bin width [m or s]
  aggregation: "mean"     # mean | median
  diss_width: null        # override bin width for diss
  diss_aggregation: null  # override aggregation for diss
  chi_width: null         # override bin width for chi
  chi_aggregation: null   # override aggregation for chi
  diagnostics: false      # write n_samples, *_std per bin

netcdf:
  title: null
  summary: null
  institution: null
  creator_name: null
  creator_email: null
  creator_url: null
  creator_type: null
  creator_institution: null
  contributor_name: null
  contributor_role: null
  project: null
  program: null
  id: null
  naming_authority: null
  source: null
  platform: null
  platform_vocabulary: null
  instrument: null
  instrument_vocabulary: null
  processing_level: null
  license: null
  references: null
  comment: null
  acknowledgement: null
  date_created: null       # auto-filled if null
  date_modified: null      # auto-filled if null
  date_issued: null
  geospatial_lat_min: null # auto-filled from data if null
  geospatial_lat_max: null
  geospatial_lon_min: null
  geospatial_lon_max: null
  geospatial_vertical_min: null
  geospatial_vertical_max: null
  geospatial_vertical_positive: "down"
  time_coverage_start: null  # auto-filled from data if null
  time_coverage_end: null
  time_coverage_duration: null
  time_coverage_resolution: null
  keywords: null
  keywords_vocabulary: null
  standard_name_vocabulary: "CF Standard Name Table v83"
  Conventions: "CF-1.8, ACDD-1.3"
  history: null            # auto-filled with processing log

parallel:
  jobs: 1
"""


def generate_template(path: str | Path) -> Path:
    """Write a fully-commented template configuration file."""
    path = Path(path)
    path.write_text(_TEMPLATE, encoding="utf-8")
    return path
