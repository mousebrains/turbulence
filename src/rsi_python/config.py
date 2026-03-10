# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""YAML configuration file support for rsi-tpw.

Provides loading, validation, three-way merge (defaults ← config ← CLI),
deterministic hashing, and sequential output directory management.
"""

import hashlib
import json
from pathlib import Path

from ruamel.yaml import YAML

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
        "goodman": False,
        "f_AA": 98.0,
        "fit_method": "mle",
        "spectrum_model": "kraichnan",
        "salinity": None,
    },
}

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
    with open(path) as fh:
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
                f"Unknown config section: {section!r}. Valid sections: {sorted(_VALID_SECTIONS)}"
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
    """Merge defaults ← config-file values ← CLI overrides.

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


def _normalize_value(v):
    """Normalize a single value for deterministic JSON encoding."""
    if v is None:
        return None
    # bool check must come before int (bool is a subclass of int)
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if v == int(v):
            return int(v)
        return round(v, 10)
    if isinstance(v, str):
        return v
    return v


def _canonicalize_section(section: str, params: dict) -> dict:
    """Canonicalize a single section's parameters into a normalized dict."""
    base = dict(DEFAULTS[section])
    for k, v in params.items():
        if k in base and v is not None:
            base[k] = v
    return {k: _normalize_value(v) for k, v in sorted(base.items())}


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
        For example, chi depends on epsilon, so pass
        ``upstream=[("epsilon", eps_params)]`` to include epsilon
        parameters in the hash.  The canonical output includes all
        sections keyed by name, ensuring that changing an upstream
        parameter produces a different hash.
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
    """SHA-256 hex digest of the canonical representation.

    Parameters
    ----------
    section : str
        The primary config section.
    params : dict
        Resolved parameters for the primary section.
    upstream : list of (section, params) tuples, optional
        Upstream sections to include in the hash (see `canonicalize`).
    """
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

    Parameters
    ----------
    base : str or Path
        Parent directory (e.g. ``results/``).
    prefix : str
        Directory prefix (e.g. ``"eps"``, ``"chi"``, ``"prof"``).
    section : str
        Config section name for hashing.
    params : dict
        Resolved parameters.
    upstream : list of (section, params) tuples, optional
        Upstream sections to include in the hash.

    Returns
    -------
    Path to the (possibly new) output directory.
    """
    base = Path(base)
    target_hash = compute_hash(section, params, upstream=upstream)

    # Scan existing sequential directories
    max_seq = -1
    import glob as globmod

    pattern = str(base / f"{prefix}_[0-9][0-9]")
    for d in sorted(globmod.glob(pattern)):
        dp = Path(d)
        # Track sequence number from any entry (file or dir) to avoid name collisions
        try:
            seq = int(dp.name.split("_")[-1])
        except ValueError:
            continue
        max_seq = max(max_seq, seq)
        if not dp.is_dir():
            continue
        # Check for matching signature file
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

    # Include upstream sections first (in order)
    if upstream:
        for up_section, up_params in upstream:
            resolved = dict(DEFAULTS[up_section])
            for k, v in up_params.items():
                if k in resolved and v is not None:
                    resolved[k] = v
            data[up_section] = resolved

    # Primary section
    resolved = dict(DEFAULTS[section])
    for k, v in params.items():
        if k in resolved and v is not None:
            resolved[k] = v
    data[section] = resolved

    out = directory / "config.yaml"
    with open(out, "w") as fh:
        yaml.dump(data, fh)
    return out


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
  goodman: false        # Goodman coherent noise removal
  f_AA: 98.0            # anti-aliasing filter cutoff [Hz]
  fit_method: mle       # Method 2 fitting: mle or iterative
  spectrum_model: kraichnan  # theoretical spectrum: batchelor or kraichnan
  salinity: null        # salinity [PSU] (null = 35, fixed S)
"""


def generate_template(path: str | Path) -> Path:
    """Write a fully-commented template configuration file."""
    path = Path(path)
    path.write_text(_TEMPLATE)
    return path
