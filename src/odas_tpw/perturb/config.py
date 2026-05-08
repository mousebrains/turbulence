# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""YAML configuration file support for perturb.

Provides loading, validation, three-way merge (defaults <- config <- CLI),
deterministic hashing, and sequential output directory management.

All config management logic lives in :mod:`odas_tpw.config_base.ConfigManager`;
this module defines perturb-specific DEFAULTS (13 sections), instantiates the
manager, and re-exports the methods as module-level functions.
"""

from pathlib import Path

from odas_tpw.config_base import ConfigManager

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
        "chi_minimum": 1.0e-13,
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
    "speed": {
        "method": "pressure",   # pressure | em | flight | constant
        "value": None,           # m/s, only for method="constant"
        "aoa_deg": 3.0,          # angle of attack, only for method="flight"
        "min_pitch_deg": 5.0,    # flight: skip |pitch+aoa| below this (deg)
        "speed_cutout": 0.05,    # m/s floor applied to fast-rate speed
        "tau": None,             # smoothing time constant; null = vehicle default
    },
    # Per-segment QC gate. Each "_drop_from" entry names a hotel-injected
    # channel (uint8 bitfield or boolean) sampled by time over the segment's
    # window; if any sample is nonzero, the segment is flagged. The
    # qc_drop_epsilon / qc_drop_chi variables on the diss / chi NetCDFs are
    # always written (CF flag attrs preserved from the source channel); when
    # drop_action is "nan" the corresponding epsilon / chi values are also
    # NaN'd so default plots and combos exclude them.
    "qc": {
        "enable": True,
        "drop_action": "nan",     # nan | flag_only
        "epsilon_drop_from": [],  # e.g. ["q_drop_epsilon"]
        "chi_drop_from": [],      # e.g. ["q_drop_chi"]
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
        "publisher_name": None,
        "publisher_email": None,
        "publisher_url": None,
        "publisher_type": None,
        "publisher_institution": None,
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
        "standard_name_vocabulary": "CF Standard Name Table v89",
        "Conventions": "CF-1.13, ACDD-1.3",
        "history": None,
    },
    "parallel": {
        "jobs": 1,
    },
    # Per-instrument overrides keyed by serial-number identifier (matched
    # against the parent directory of each .p file). Allowed inner keys:
    #   exclude_shear_probes : list[str]  (e.g. ["sh1"] or ["sh2"])
    "instruments": {},
}

# Keys excluded from hashing — toggling diagnostics should not create new output dirs
_HASH_EXCLUDE_KEYS = frozenset({"diagnostics"})

# Sections whose inner keys are user-defined (e.g. instrument serials).
# ConfigManager skips strict unknown-key validation for these; per-section
# inner-schema validation happens in a wrapper below.
_DYNAMIC_KEY_SECTIONS = frozenset({"instruments"})

_INSTRUMENT_VALID_KEYS = frozenset({"exclude_shear_probes"})

_mgr = ConfigManager(
    DEFAULTS,
    hash_exclude_keys=_HASH_EXCLUDE_KEYS,
    dynamic_key_sections=_DYNAMIC_KEY_SECTIONS,
)


def _validate_instruments(instruments: dict) -> None:
    """Validate the inner structure of the dynamic ``instruments`` section."""
    for sn, settings in instruments.items():
        if not isinstance(settings, dict):
            raise ValueError(
                f"instruments.{sn}: must be a mapping, got {type(settings).__name__}"
            )
        unknown = set(settings) - _INSTRUMENT_VALID_KEYS
        if unknown:
            raise ValueError(
                f"Unknown key(s) in instruments.{sn}: {sorted(unknown)}. "
                f"Valid keys: {sorted(_INSTRUMENT_VALID_KEYS)}"
            )
        excludes = settings.get("exclude_shear_probes", [])
        if not isinstance(excludes, list) or not all(isinstance(p, str) for p in excludes):
            raise ValueError(
                f"instruments.{sn}.exclude_shear_probes: must be a list of strings, "
                f"got {excludes!r}"
            )


def _load_config_with_instruments_check(path: str | Path) -> dict[str, dict]:
    """Load and validate a perturb config, including the instruments section."""
    config = _mgr.load_config(path)
    _validate_instruments(config.get("instruments", {}))
    return config


def _validate_config_with_instruments_check(config: dict[str, dict]) -> None:
    """Validate a perturb config including the instruments section."""
    _mgr.validate_config(config)
    _validate_instruments(config.get("instruments", {}))

# Re-export manager methods as module-level functions
load_config = _load_config_with_instruments_check
validate_config = _validate_config_with_instruments_check
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
  channels: {}            # source-keyed selection. {} = take every source
                          # variable. Each value can be:
                          #   ~ (null) / {}     : include with defaults
                          #   "new_name"        : rename only
                          #   {name, interp,    : per-variable options
                          #    scale, offset,   #   (any subset)
                          #    units, fast}
  fast_channels:          # default fast set when 'fast' is unset per-var
    - speed
    - P
  interpolation: "pchip"  # default kind: pchip | linear | nearest |
                          # previous | next | zero | slinear | quadratic
                          # | cubic. Per-variable interp wins.

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
  diss_length: null       # dissipation window [samples] (null = 4 * fft_length)
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
  diss_length: null       # dissipation window [samples] (null = 4 * fft_length)
  overlap: null           # window overlap [samples] (null = diss_length // 2)
  fp07_model: "single_pole"  # FP07 transfer function model
  goodman: true           # Goodman coherent noise removal
  f_AA: 98.0              # anti-aliasing filter cutoff [Hz]
  fit_method: "iterative" # Method 2 fitting: iterative or mle
  spectrum_model: "kraichnan"  # theoretical spectrum: batchelor or kraichnan
  salinity: null          # salinity [PSU] (null = 35, fixed S)
  chi_minimum: 1.0e-13    # floor for mk_chi_mean (values <= go to NaN)
  diagnostics: false      # write spec_noise, fp07_transfer, gradT_raw

ctd:
  enable: true
  bin_width: 0.5          # time bin width [s]
  T_name: "JAC_T"         # temperature channel
  C_name: "JAC_C"         # conductivity channel
  variables: null         # channels to bin (null = auto-detect)
  method: "mean"          # aggregation method
  diagnostics: false      # write n_samples, *_std per bin

speed:
  # Through-water speed source. Computed AFTER hotel merge so the
  # method has access to both .p-file and hotel channels.
  method: "pressure"      # pressure | em | flight | constant
  # pressure : ODAS smoothed |dP/dt|. Correct for VMP. (default)
  # em       : use the U_EM channel from the .p file (MicroRider EM
  #            flowmeter). Errors out if U_EM is missing.
  # flight   : |W| / (sin(|pitch|-aoa)*cos|roll|), pitch axis auto-
  #            picked from Incl_X/Incl_Y by amplitude.
  # constant : use the scalar in `value`.
  value: null             # m/s, only when method="constant"
  aoa_deg: 3.0            # angle of attack [deg], for method="flight"
  min_pitch_deg: 5.0      # flight: drop samples with |pitch|-aoa < this
  speed_cutout: 0.05      # m/s floor applied to fast-rate speed
  tau: null               # smoothing tau [s]; null = vehicle default
                          # (vmp/xmp 1.5, slocum_glider 3.0, ...)

qc:
  enable: true
  drop_action: "nan"      # nan | flag_only
                          # 'nan' NaNs e_*/epsilonMean (and chi_*/chiMean)
                          # for flagged segments; 'flag_only' leaves the
                          # values untouched. The qc_drop_* bitfield is
                          # always written so plots can audit / mask.
  epsilon_drop_from: []   # hotel channel names OR'd over each diss
                          # segment's time window. Each named channel
                          # should be a uint8 CF bitfield (or 0/1).
                          # e.g. ["q_drop_epsilon"]
  chi_drop_from: []       # same, for chi segments

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
  publisher_name: null
  publisher_email: null
  publisher_url: null
  publisher_type: null
  publisher_institution: null
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
  standard_name_vocabulary: "CF Standard Name Table v89"
  Conventions: "CF-1.13, ACDD-1.3"
  history: null            # auto-filled with processing log

parallel:
  jobs: 1

# Per-instrument overrides. The key matches the parent directory of each .p
# file (e.g. ARCTERX/VMP/SN465 -> SN465). Use this to suppress shear probes
# whose amplifier or sensor is known to be bad — the named probe is NaN'd
# out before mk_epsilon_mean, so it is excluded from the multi-probe
# epsilonMean and from chi Method 1 (which uses epsilonMean).
instruments: {}
# Example:
# instruments:
#   SN465:
#     exclude_shear_probes: ["sh2"]
"""


def generate_template(path: str | Path) -> Path:
    """Write a fully-commented template configuration file."""
    path = Path(path)
    path.write_text(_TEMPLATE, encoding="utf-8")
    return path
