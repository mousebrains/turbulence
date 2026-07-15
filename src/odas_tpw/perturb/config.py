# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""YAML configuration file support for perturb.

Provides loading, validation, three-way merge (defaults <- config <- CLI),
deterministic hashing, and sequential output directory management.

All config management logic lives in :mod:`odas_tpw.config_base.ConfigManager`;
this module defines perturb-specific DEFAULTS (17 sections), instantiates the
manager, and re-exports the methods as module-level functions.
"""

import functools
import hashlib
import importlib.metadata
import os
import warnings
from pathlib import Path
from typing import Any

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
        "force_trim": False,
        "force": False,
        "merge": False,
        # Absolute directory of the loaded YAML, stamped by load_config so path
        # values written with the <CONFIG_DIR> token can be resolved relative to
        # the config file. None when the config was not loaded from a file.
        # Hash-excluded (see _HASH_EXCLUDE_KEYS): it is mount-specific, so the
        # portable <CONFIG_DIR> token — not this absolute path — is what feeds
        # the stage-dir signatures.
        "config_dir": None,
    },
    "gps": {
        "source": "nan",
        "lat": None,
        "lon": None,
        "file": None,
        "time_col": None,  # null = source default ("t" CSV, "time" NetCDF)
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
        "noise_factor": 2.0,
    },
    "epsilon": {
        # Durations (seconds) are the primary interface — instruments sample
        # at different rates (standard VMP-250: 512 Hz; coastal/high-energy
        # units: 1-2 kHz), while the physical constraints on the windows are
        # durations x speed (Lueck et al. 2024, doi:10.3389/fmars.2024.1334327).
        # Sample-count keys (fft_length/diss_length/overlap), when set, are an
        # exact-control override and win over the duration keys.
        "fft_sec": 1.0,
        "diss_sec": None,  # None = 4 x fft
        "overlap_sec": None,  # None = half the dissipation window
        "fft_length": None,
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
        "fom_max": None,  # null = no FOM cut. e.g. 2.0 NaNs each
        # per-probe (e_N, epsilon[probe,:]) cell
        # whose figure-of-merit fom[probe,seg]
        # >= fom_max BEFORE mk_epsilon_mean, so
        # bad probes drop out of the geomean
        # individually.
        "diagnostics": False,
    },
    "chi": {
        "enable": False,
        # Same duration-first interface as [epsilon] above.
        "fft_sec": 1.0,
        "diss_sec": None,
        "overlap_sec": None,
        "fft_length": None,
        "diss_length": None,
        "overlap": None,
        "fp07_model": "single_pole",
        "goodman": True,
        "f_AA": 98.0,
        "use_epsilon": True,  # Method 1 (chi from shear epsilon).
        # Set false for Method 2 spectral fit -- a
        # MR on a vibrating glider has unreliable
        # epsilon and should not seed chi from it.
        "fit_method": "iterative",  # Only used when use_epsilon=False
        "spectrum_model": "kraichnan",
        "salinity": None,
        "mixing": True,  # Derived mixing quantities (N2, dTdz,
        # K_T, Gamma, K_rho) on the chi grid,
        # with salinity from the profile's own
        # C/T/P (TEOS-10).
        "chi_minimum": 1.0e-13,
        "spectral_qc": True,  # Soft fom + K_max_ratio QC on chiMean
        # (hence K_T/Gamma), matching the rsi
        # chi_final. Never drops a window (falls
        # back to all probes when none pass). Set
        # false to publish unfiltered chiMean.
        "fom_max": None,  # null = no FOM cut. An OPTIONAL, more
        # aggressive HARD per-probe pre-cut (can NaN a
        # whole window); independent of spectral_qc.
        # Same mechanism as epsilon.fom_max.
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
        "method": "pressure",  # pressure | em | flight | constant
        "value": None,  # m/s, only for method="constant"
        "aoa_deg": 3.0,  # angle of attack, only for method="flight"
        "min_pitch_deg": 5.0,  # flight: skip |pitch+aoa| below this (deg)
        "speed_cutout": 0.05,  # m/s floor applied to fast-rate speed
        "tau": None,  # smoothing time constant; null = vehicle default
        "amplitude_quantile": [1.0, 99.0],  # for flight pitch-axis auto-pick
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
        "drop_action": "nan",  # nan | flag_only
        "epsilon_drop_from": [],  # e.g. ["q_drop_epsilon"]
        "chi_drop_from": [],  # e.g. ["q_drop_chi"]
        # Internal range-check rules. Each entry produces a synthetic
        # uint8 channel and can be referenced by *_drop_from.
        # See odas_tpw.perturb.qc_rules for the per-entry schema.
        "rules": {},
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
    # Background stratification (N2, dT/dz) written to the diss and profile
    # products, independent of epsilon/chi. Computed with the Thorpe-sorted
    # (adiabatically leveled) method. diss uses its dissipation window; the
    # profile product uses the configurable ``window`` below. NOT written to the
    # CTD product (profile-only; the CTD spans the whole up/down trajectory).
    "stratification": {
        "enable": True,
        "window": 2.0,  # background vertical window [dbar] for the
        # profile product
        "salinity": None,  # N2 salinity source: null = conductivity (else 35),
        # a number = fixed PSU, "measured" = C/T/P (TEOS-10),
        # "hotel"/"hotel:<var>" = a hotel-injected salinity channel
    },
    "parallel": {
        "jobs": 1,
    },
    # Per-instrument overrides keyed by serial-number identifier (matched
    # against the parent directory of each .p file). Allowed inner keys:
    #   exclude_shear_probes : list[str]  (e.g. ["sh1"] or ["sh2"])
    "instruments": {},
}

# Keys excluded from hashing — runtime toggles that must NOT create new output
# dirs: ``diagnostics`` (extra plots), and the cache-bypass flags ``force`` /
# ``force_trim`` (they re-run work into the SAME dirs, so they cannot change the
# dir identity or --force would defeat itself by recomputing into fresh dirs).
_HASH_EXCLUDE_KEYS = frozenset({"diagnostics", "force", "force_trim", "config_dir"})

# Sections whose inner keys are user-defined (e.g. instrument serials).
# ConfigManager skips strict unknown-key validation for these; per-section
# inner-schema validation happens in a wrapper below.
_DYNAMIC_KEY_SECTIONS = frozenset({"instruments"})

_INSTRUMENT_VALID_KEYS = frozenset({"exclude_shear_probes"})

# Numeric dependencies whose version can change ε/χ/N² outputs even with no
# change to our own source — folded into the engine fingerprint so a dep
# upgrade also invalidates cached results.
_ENGINE_DEPS = ("numpy", "scipy", "gsw", "netCDF4", "xarray", "pandas")
# Subpackages that cannot affect processing numerics; excluded from the source
# hash so editing plots/standalone tools doesn't invalidate cached science.
_FINGERPRINT_EXCLUDE = ("perturb/plot/", "pyturb/")
_ENGINE_OVERRIDE_ENV = "ODAS_TPW_ENGINE_FINGERPRINT"


@functools.lru_cache(maxsize=1)
def engine_fingerprint() -> str:
    """Hash of the processing *code + key numeric deps*.

    Folded into every stage signature so a change that could alter outputs
    yields new ``{stage}_NN`` dirs (recompute) while an unchanged engine reuses
    them — making "outputs already exist in this dir" a *safe* cache signal.
    Covers our own ``.py`` source (excluding plot/pyturb) plus the installed
    versions of the distribution and key numeric deps.

    Override with ``$ODAS_TPW_ENGINE_FINGERPRINT`` to pin the value — to force
    cache reuse across a code change you KNOW can't affect numerics, or for
    deterministic tests.
    """
    override = os.environ.get(_ENGINE_OVERRIDE_ENV)
    if override:
        return override

    h = hashlib.sha256()
    for dist in ("microstructure-tpw", *_ENGINE_DEPS):
        try:
            version = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            version = "?"
        h.update(f"{dist}={version}\n".encode())

    pkg_root = Path(__file__).resolve().parent.parent  # .../odas_tpw
    n_py = 0
    for py in sorted(pkg_root.rglob("*.py")):
        rel = py.relative_to(pkg_root).as_posix()
        if any(rel.startswith(excl) for excl in _FINGERPRINT_EXCLUDE):
            continue
        try:
            data = py.read_bytes()
        except OSError:
            continue
        h.update(rel.encode())
        h.update(b"\0")
        h.update(hashlib.sha256(data).digest())
        n_py += 1
    if n_py == 0:
        warnings.warn(
            "engine_fingerprint: no .py source found under "
            f"{pkg_root} (compiled/zip install?); the processing cache is keyed "
            "on dependency versions only and will NOT detect source edits — set "
            f"${_ENGINE_OVERRIDE_ENV} to manage cache invalidation manually.",
            stacklevel=2,
        )
    return h.hexdigest()


_WINDOW_KEY_PAIRS = (
    ("fft_length", "fft_sec"),
    ("diss_length", "diss_sec"),
    ("overlap", "overlap_sec"),
)

_WINDOW_SECTIONS = ("epsilon", "chi")


class _PerturbConfigManager(ConfigManager):
    """ConfigManager with perturb's window-duration key semantics.

    In [epsilon]/[chi] a duration key (fft_sec/diss_sec/overlap_sec) is
    dropped from every section VIEW — the canonical form that is hashed
    into stage-directory signatures, and the resolved config.yaml written
    into each stage dir — when it is inert: null, or overridden by a
    non-null sample-count twin. A legacy config that pins fft_length
    therefore keeps a byte-identical canonical form (and signature) across
    this feature; only a GOVERNING duration key changes hashes.
    """

    def _section_view_postprocess(self, section: str, mapping: dict) -> dict:
        if section in _WINDOW_SECTIONS:
            for samples_key, sec_key in _WINDOW_KEY_PAIRS:
                if sec_key in mapping and (
                    mapping[sec_key] is None or mapping.get(samples_key) is not None
                ):
                    del mapping[sec_key]
        return mapping

    def validate_config(self, config: dict) -> None:
        """Structural validation plus fs-independent window-duration checks.

        Durations are validated at LOAD time so a sign typo (fft_sec: -1)
        or an inverted window pair fails before any stage directory is
        created, rather than aborting mid-run inside per-file processing.
        """
        super().validate_config(config)
        for section in _WINDOW_SECTIONS:
            params = config.get(section) or {}
            for _, sec_key in _WINDOW_KEY_PAIRS:
                val = params.get(sec_key)
                if val is None:
                    continue
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    raise ValueError(
                        f"{section}.{sec_key}: expected a number of seconds, got {val!r}"
                    )
                if not (val > 0 and val == val and val != float("inf")):
                    raise ValueError(
                        f"{section}.{sec_key}: must be a positive finite "
                        f"number of seconds, got {val!r}"
                    )
            fft_s, diss_s = params.get("fft_sec"), params.get("diss_sec")
            if (
                isinstance(fft_s, (int, float))
                and isinstance(diss_s, (int, float))
                and diss_s < fft_s
            ):
                raise ValueError(
                    f"{section}: diss_sec ({diss_s}) is shorter than fft_sec ({fft_s})"
                )


_mgr = _PerturbConfigManager(
    DEFAULTS,
    hash_exclude_keys=_HASH_EXCLUDE_KEYS,
    dynamic_key_sections=_DYNAMIC_KEY_SECTIONS,
    engine_fingerprint=engine_fingerprint,
)


def _validate_instruments(instruments: dict) -> None:
    """Validate the inner structure of the dynamic ``instruments`` section."""
    for sn, settings in instruments.items():
        if not isinstance(settings, dict):
            raise ValueError(f"instruments.{sn}: must be a mapping, got {type(settings).__name__}")
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


CONFIG_DIR_TOKEN = "<CONFIG_DIR>"


def expand_config_dir(value: Any, config_dir: str | None) -> Any:
    """Resolve a leading ``<CONFIG_DIR>`` token against *config_dir*.

    A path written as ``<CONFIG_DIR>/sub`` in a YAML config is made absolute
    relative to the config file's own directory, so a config plus its data tree
    can be moved (or mounted at a different point) without editing paths. Values
    without the token — and non-strings — pass through unchanged. Raises
    ``ValueError`` if the token is used but no config directory is known (e.g. a
    config assembled in memory rather than loaded from a file).
    """
    # Accept str or os.PathLike (a Path holding the token must not slip through
    # the isinstance check and reach the filesystem verbatim). None/numbers and
    # any non-token value pass through unchanged, preserving the original type.
    if isinstance(value, str):
        text = value
    elif isinstance(value, os.PathLike):
        text = os.fspath(value)
    else:
        return value
    if not text.startswith(CONFIG_DIR_TOKEN):
        return value
    if not config_dir:
        raise ValueError(
            f"{CONFIG_DIR_TOKEN} in path {text!r} requires a config loaded from "
            f"a file (no config directory is known)"
        )
    # Split the remainder on either separator and re-join with the OS separator,
    # so a config's forward-slash paths yield a clean native path on Windows too
    # (a bare os.path.join would leave the internal "/" -> a mixed "C:\x/y").
    parts = [seg for seg in text[len(CONFIG_DIR_TOKEN) :].replace("\\", "/").split("/") if seg]
    return os.path.join(config_dir, *parts) if parts else config_dir


def config_dir_of(config: dict) -> str | None:
    """The absolute directory of the loaded config, or None (see load_config)."""
    return (config.get("files") or {}).get("config_dir")


def _load_config_with_instruments_check(path: str | Path) -> dict[str, dict]:
    """Load and validate a perturb config, including the instruments section."""
    config = _mgr.load_config(path)
    _validate_instruments(config.get("instruments", {}))
    # Stamp the config's own directory so <CONFIG_DIR> path tokens resolve
    # relative to it. Excluded from every stage hash (_HASH_EXCLUDE_KEYS) so the
    # same config on a different mount still matches its existing output dirs.
    # An empty config file loads to {} (shared contract, and it has no paths to
    # resolve), so it is left untouched.
    if config:
        config.setdefault("files", {})["config_dir"] = str(Path(path).resolve().parent)
    return config


def _validate_config_with_instruments_check(config: dict[str, dict]) -> None:
    """Validate a perturb config including the instruments section."""
    _mgr.validate_config(config)
    _validate_instruments(config.get("instruments", {}))


# Re-export manager methods as module-level functions
load_config = _load_config_with_instruments_check
validate_config = _validate_config_with_instruments_check


def merge_config(
    section: str,
    file_values: dict | None = None,
    cli_overrides: dict | None = None,
) -> dict:
    """merge defaults <- file <- CLI, then drop inert window-duration keys.

    In the [epsilon]/[chi] sections an explicit sample-count key
    (fft_length/diss_length/overlap) wins over its duration twin
    (fft_sec/diss_sec/overlap_sec); the duration key is then inert and is
    removed so it cannot perturb the stage-directory signature — a legacy
    config that pins fft_length keeps a bit-identical signature.
    """
    merged = _mgr.merge_config(section, file_values, cli_overrides)
    if section in ("epsilon", "chi"):
        for samples_key, sec_key in _WINDOW_KEY_PAIRS:
            if samples_key in merged and sec_key in merged:
                del merged[sec_key]
    return merged


def resolve_window_config(cfg: dict, fs: float, *, section: str = "epsilon") -> dict:
    """Resolve fft/diss/overlap windows to integer sample counts at rate *fs*.

    *cfg* is a merge_config-merged [epsilon] or [chi] section. Duration keys
    (seconds) are converted via *fs* and rounded to the nearest even sample
    count; explicit sample keys pass through unchanged (they won at merge
    time). Defaults: fft = 1 s, dissipation window = 4 x fft, overlap =
    downstream default (half the window). Returns a copy with concrete
    integer ``fft_length``/``diss_length`` (and ``overlap`` when
    determined) and the ``*_sec`` keys removed, suitable for splatting into
    the compute functions.
    """
    if not (fs and fs > 0):
        raise ValueError(f"{section}: invalid sampling rate {fs!r}")

    import math

    def seconds(key: str) -> float | None:
        val = cfg.get(key)
        if val is None:
            return None
        try:
            sec = float(val)
        except (TypeError, ValueError):
            raise ValueError(
                f"{section}.{key}: expected a number of seconds, got {val!r}"
            ) from None
        if not (math.isfinite(sec) and sec > 0):
            raise ValueError(
                f"{section}.{key}: must be a positive finite number of seconds, got {val!r}"
            )
        return sec

    def even(x: float) -> int:
        return max(2, 2 * round(x / 2.0))

    out = {k: v for k, v in cfg.items() if k not in ("fft_sec", "diss_sec", "overlap_sec")}
    fft = cfg.get("fft_length")
    if fft is None:
        fft = even(fs * (seconds("fft_sec") or 1.0))
    fft = int(fft)
    diss = cfg.get("diss_length")
    if diss is None:
        diss_sec = seconds("diss_sec")
        diss = even(fs * diss_sec) if diss_sec is not None else 4 * fft
    diss = int(diss)
    if diss < fft:
        raise ValueError(
            f"{section}: dissipation window ({diss} samples) is shorter than "
            f"the FFT segment ({fft} samples)"
        )
    overlap = cfg.get("overlap")
    if overlap is None:
        overlap_sec = seconds("overlap_sec")
        if overlap_sec is not None:
            overlap = even(fs * overlap_sec)
    if overlap is not None:
        overlap = int(overlap)
        if overlap >= diss:
            raise ValueError(
                f"{section}: overlap ({overlap} samples) must be smaller than "
                f"the dissipation window ({diss} samples)"
            )
        out["overlap"] = overlap
    out["fft_length"] = fft
    out["diss_length"] = diss
    return out


canonicalize = _mgr.canonicalize
compute_hash = _mgr.compute_hash
resolve_output_dir = _mgr.resolve_output_dir
write_signature = _mgr.write_signature
write_resolved_config = _mgr.write_resolved_config


# ---------------------------------------------------------------------------
# Per-stage signature inputs (relocated here from pipeline.py so the read-only
# plot-time resolver in resolve.py can reproduce a stage's signature without
# importing the heavy pipeline module). The pipeline imports these back.
# ---------------------------------------------------------------------------

# The section whose params are passed *positionally* to resolve_output_dir for
# each stage. NOT always the stage name (e.g. `diss` hashes under `epsilon`,
# every combo under `binning`/`ctd`) — mirrors the resolve_output_dir() calls
# in pipeline.py. The discriminating detail lives in the upstream chain.
_STAGE_SECTION: dict[str, str] = {
    "profiles": "profiles",
    "diss": "epsilon",
    "chi": "chi",
    "ctd": "ctd",
    "profiles_binned": "binning",
    "diss_binned": "binning",
    "chi_binned": "binning",
    "combo": "binning",
    "diss_combo": "binning",
    "chi_combo": "binning",
    "ctd_combo": "ctd",
}

#: Stages a caller (e.g. the plot resolver) may resolve to a ``{stage}_NN`` dir.
STAGES: frozenset[str] = frozenset(_STAGE_SECTION)


def canonical_instruments_for_hash(instruments: dict | None) -> dict[str, Any]:
    """Normalize set-like instrument settings before hashing."""
    normalized: dict[str, Any] = {}
    # Sort by str(key): instrument serials mix int (unquoted `465:`) and str keys.
    for key, settings in sorted((instruments or {}).items(), key=lambda kv: str(kv[0])):
        if not isinstance(settings, dict):
            normalized[str(key)] = settings
            continue
        item = dict(settings)
        excludes = item.get("exclude_shear_probes")
        if isinstance(excludes, list):
            item["exclude_shear_probes"] = sorted(str(probe) for probe in excludes)
        normalized[str(key)] = item
    return normalized


def upstream_for(stage: str, config: dict) -> list[tuple[str, dict]]:
    """Return the full upstream parameter chain for *stage*.

    Each downstream stage's ``.params_sha256_*`` signature is the hash of the
    stage's own params plus every ancestor's params, so two runs that differ
    only in a deep upstream knob still resolve to different output directories.
    """
    files_p = merge_config("files", config.get("files"))
    gps_p = merge_config("gps", config.get("gps"))
    hotel_p = merge_config("hotel", config.get("hotel"))
    speed_p = merge_config("speed", config.get("speed"))
    qc_p = merge_config("qc", config.get("qc"))
    fp07_p = merge_config("fp07", config.get("fp07"))
    ct_p = merge_config("ct", config.get("ct"))
    bottom_p = merge_config("bottom", config.get("bottom"))
    top_trim_p = merge_config("top_trim", config.get("top_trim"))
    profiles_p = merge_config("profiles", config.get("profiles"))
    eps_p = merge_config("epsilon", config.get("epsilon"))
    chi_p = merge_config("chi", config.get("chi"))
    ctd_p = merge_config("ctd", config.get("ctd"))
    netcdf_p = merge_config("netcdf", config.get("netcdf"))
    strat_p = merge_config("stratification", config.get("stratification"))
    instruments_p = canonical_instruments_for_hash(config.get("instruments"))

    profile_upstream = [
        ("files", files_p),
        ("gps", gps_p),
        ("hotel", hotel_p),
        ("speed", speed_p),
        ("qc", qc_p),
        ("fp07", fp07_p),
        ("ct", ct_p),
        ("bottom", bottom_p),
        ("top_trim", top_trim_p),
        # N2/dT/dz are injected onto the profile/diss products and gated by
        # stratification.{enable,window}, so they must re-version those outputs.
        ("stratification", strat_p),
    ]
    profile_chain = [*profile_upstream, ("profiles", profiles_p)]
    diss_chain = [*profile_chain, ("instruments", instruments_p)]
    chi_chain = [*diss_chain, ("epsilon", eps_p)]
    # CTD salinity/density come from CT-aligned conductivity (depends on ct.* and
    # the detected profiles) — so ct and profiles must be hashed. The CTD product
    # does NOT carry the background N2/dT/dz (those are profile-only), so
    # stratification.* deliberately does not version it.
    ctd_chain = [
        ("files", files_p),
        ("gps", gps_p),
        ("hotel", hotel_p),
        ("speed", speed_p),
        ("qc", qc_p),
        ("ct", ct_p),
        ("profiles", profiles_p),
    ]

    chains: dict[str, list[tuple[str, dict]]] = {
        "profiles": profile_upstream,
        "diss": diss_chain,
        "chi": chi_chain,
        "ctd": ctd_chain,
        "profiles_binned": profile_chain,
        "diss_binned": [*diss_chain, ("epsilon", eps_p)],
        "chi_binned": [*chi_chain, ("chi", chi_p)],
        "combo": [*profile_chain, ("netcdf", netcdf_p)],
        "diss_combo": [*diss_chain, ("epsilon", eps_p), ("netcdf", netcdf_p)],
        "chi_combo": [*chi_chain, ("chi", chi_p), ("netcdf", netcdf_p)],
        "ctd_combo": [*ctd_chain, ("ctd", ctd_p), ("netcdf", netcdf_p)],
    }
    return chains[stage]


def stage_signature(stage: str, config: dict) -> tuple[str, dict, list[tuple[str, dict]]]:
    """The exact ``(section, params, upstream)`` triple the pipeline passes to
    ``resolve_output_dir`` for *stage* — the single source of truth shared by
    the pipeline and the plot resolver."""
    section = _STAGE_SECTION[stage]
    params = merge_config(section, config.get(section))
    return section, params, upstream_for(stage, config)


# Also keep _VALID_SECTIONS for any direct references
_VALID_SECTIONS = _mgr.valid_sections


# ---------------------------------------------------------------------------
# Template generation
# ---------------------------------------------------------------------------

_TEMPLATE = """\
# perturb configuration
# Values shown are the defaults. Uncomment and edit to customize.
# CLI flags override values in this file.
#
# Paths (p_file_root, output_root, gps.file, hotel.file) may begin with the
# token <CONFIG_DIR>, which expands to this file's own directory — so a config
# plus its data tree can be moved or mounted elsewhere without editing paths
# (e.g. p_file_root: <CONFIG_DIR>/VMP). Relative paths without the token stay
# relative to the current working directory, as before. The token — not the
# resolved absolute path — is what feeds the cache signatures, so a remount
# reuses existing output directories.

files:
  p_file_root: "VMP/"
  p_file_pattern: "**/*.p"
  output_root: "results/"
  trim: true
  force_trim: false         # re-trim even when output is already up to date
  merge: false              # size-limit rollovers only; fuses independent casts otherwise (#97)

gps:
  source: "nan"           # nan | fixed | csv | netcdf
  lat: null
  lon: null
  file: null
  time_col: null          # null = source default ("t" for csv, "time" for netcdf)
  lat_col: "lat"
  lon_col: "lon"
  max_time_diff: 60       # warn when extrapolating > this [s] outside GPS coverage

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
                          #    scale, offset,   #   (any subset). replace:true
                          #    units, fast,     #   lets a hotel channel
                          #    replace}         #   overwrite a native one
                          #                     #   (else refused; pair with
                          #                     #   fast:false for slow P).
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
  diagnostics: false      # RESERVED / not yet implemented (would write T1_raw,
                          # C_raw, W, cal attrs). Currently a no-op for profiles.

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
  speed_factor: 0.3       # UNUSED (reserved; tuning has no effect)
  median_factor: 1.0      # UNUSED (reserved; tuning has no effect)
  vibration_frequency: 16 # UNUSED (reserved; tuning has no effect)
  vibration_factor: 4.0   # vibration std dev acceptance factor

top_trim:
  enable: false
  dz: 0.5                 # depth bin size [m]
  min_depth: 1.0          # minimum search depth [m]
  max_depth: 50.0         # maximum search depth [m]
  quantile: 0.6           # quantile of per-bin std taken as settled background
  noise_factor: 2.0       # std > noise_factor*background == still in prop wash

epsilon:
  # Windows are specified as DURATIONS (seconds) and converted per instrument
  # via its sampling rate (standard VMP-250: 512 Hz; coastal units: 1-2 kHz),
  # so one config serves a mixed fleet. Choosing fft_sec is a sandwich
  # (Lueck et al. 2024, Front. Mar. Sci. 11:1334327,
  # doi:10.3389/fmars.2024.1334327):
  #  - MINIMUM: the spectrum must resolve its peak, so the lowest resolved
  #    wavenumber k_l = 1/(fft_sec*W) must be <= ~0.5 cpm for low epsilon
  #    (<~1e-9 W/kg), ~1 cpm for moderate (<~1e-7), ~2 cpm for high.
  #  - MAXIMUM: the FFT span fft_sec*W must not exceed the profiler's body
  #    length (larger eddies advect the whole body; their shear is
  #    attenuated, not measured).
  # For a ~1-m VMP-250 at ~1 m/s the bounds meet at fft_sec = 1.0, and
  # epsilon <~1e-9 cannot resolve the peak on that platform. Slower/longer
  # platforms (MicroRider on a glider) shift both bounds; see
  # docs/perturb/dissipation_length.md.
  fft_sec: 1.0            # FFT segment duration [s]
  diss_sec: null          # dissipation window [s] (null = 4 * fft_sec);
                          # longer windows lower the statistical uncertainty
                          # (sigma_lnE, Lueck 2022 doi:10.1175/JTECH-D-21-0051.1)
                          # at the cost of vertical resolution and patch mixing
  overlap_sec: null       # window overlap [s] (null = half the window)
  fft_length: null        # EXPERT override [samples]; wins over fft_sec
  diss_length: null       # EXPERT override [samples]; wins over diss_sec
  overlap: null           # EXPERT override [samples]; wins over overlap_sec
  goodman: true           # Goodman coherent noise removal
  f_AA: 98.0              # anti-aliasing filter cutoff [Hz]
  f_limit: null           # upper frequency limit [Hz] (null = f_AA)
  fit_order: 3            # polynomial fit order for Nasmyth integration
  despike_thresh: 8       # despike threshold (rectified-HP / LP-envelope ratio, not MAD)
  despike_smooth: 0.5     # despike envelope low-pass cutoff [Hz]
  salinity: null          # viscosity salinity: null = fixed 35; a number =
                          # that fixed PSU; "measured" = per-profile from
                          # C/T/P (TEOS-10, needs conductivity); "hotel" (or
                          # "hotel:<var>") = a hotel-injected salinity channel
                          # (default var "salinity") — for gliders/MRs with no
                          # onboard conductivity but a hotel CTD feed
  epsilon_minimum: 1.0e-13  # floor for small epsilon values
  T_source: null          # temperature source for viscosity (null = blend T1/T2)
  T1_norm: 1.0            # T1 blending weight
  T2_norm: 1.0            # T2 blending weight
  fom_max: null           # null = NO spectral-fit QC on epsilon. IMPORTANT: unlike
                          # the rsi run_pipeline path (which applies the full ATOMIX
                          # flag set -- FM>1.15, var_resolved<0.5, despike limits --
                          # via _compute_flags), the perturb pipeline masks per-probe
                          # epsilon ONLY via this cut. With null, epsilonMean / binned
                          # / combo include estimates ATOMIX QC would reject. Set e.g.
                          # 2.0 to NaN each per-probe cell (e_N, epsilon[probe,:])
                          # whose variance-ratio FOM >= fom_max BEFORE mk_epsilon_mean,
                          # so bad probes drop out of the geomean individually. (Note
                          # this thresholds the variance-ratio `fom`, not the Lueck
                          # `FM` the file also carries.)
  diagnostics: false      # RESERVED / not yet implemented (would write
                          # spec_shear_raw, spike_count, probe_mask). No-op for epsilon.

chi:
  enable: false           # chi is optional, separate stage after diss
  fft_sec: 1.0            # FFT segment duration [s]; same sandwich
                          # constraints as epsilon.fft_sec above
  diss_sec: null          # dissipation window [s] (null = 4 * fft_sec)
  overlap_sec: null       # window overlap [s] (null = half the window)
  fft_length: null        # EXPERT override [samples]; wins over fft_sec
  diss_length: null       # EXPERT override [samples]; wins over diss_sec
  overlap: null           # EXPERT override [samples]; wins over overlap_sec
  fp07_model: "single_pole"  # FP07 transfer function model
  goodman: true           # Goodman coherent noise removal
  f_AA: 98.0              # anti-aliasing filter cutoff [Hz]
  use_epsilon: true       # true  = Method 1: chi from shear-probe epsilon.
                          # false = Method 2: spectral fit (uses fit_method).
                          # Set false for instruments where shear epsilon is
                          # unreliable (e.g. MR on a glider with thruster /
                          # surface vibration contaminating shear probes).
  fit_method: "iterative" # Method 2 fitting: iterative or mle (ignored if
                          # use_epsilon=true)
  spectrum_model: "kraichnan"  # theoretical spectrum: batchelor or kraichnan
  salinity: null          # viscosity salinity [PSU]: null = fixed 35; a
                          # number = that fixed S; "measured" = per-profile
                          # from JAC_C/JAC_T/P (TEOS-10); "hotel" (or
                          # "hotel:<var>") = a hotel-injected salinity channel
  mixing: true            # derived mixing quantities (N2, dTdz, K_T, Gamma,
                          # K_rho) on the chi grid; N2 salinity follows the
                          # stratification.salinity setting
  chi_minimum: 1.0e-13    # floor for mk_chi_mean (values <= go to NaN)
  fom_max: null           # null = no FOM cut. Same per-probe mechanism as
                          # epsilon.fom_max but on chi NCs (NaN's chi[probe,
                          # seg] / chi_N where fom[probe, seg] >= fom_max).
  diagnostics: false      # RESERVED / not yet implemented (would write
                          # spec_noise, fp07_transfer, gradT_raw). No-op for chi.

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
  amplitude_quantile: [1.0, 99.0]  # flight: percentile spread used to
                                   # auto-pick the pitch axis from
                                   # Incl_X/Incl_Y. 1..99 strips outliers
                                   # (surface tumbles, sensor saturation
                                   # spikes) that can dominate min/max.

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
  rules: {}               # internal range-check rules; each entry synthesizes a
                          # uint8 channel referenceable by *_drop_from. See
                          # odas_tpw.perturb.qc_rules for the per-entry schema.

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

stratification:
  enable: true            # write N2 and dT/dz (Thorpe-sorted) to the diss and
                          # profile products, independent of eps/chi. Not on the
                          # CTD product (profile-only; CTD spans up+down).
  window: 2.0             # background vertical window [dbar] for the profile
                          # product (diss uses its dissipation window)
  salinity: null          # N2 salinity source: null = conductivity via TEOS-10
                          # (else 35 PSU); a number = that fixed PSU; "measured"
                          # = C/T/P (TEOS-10); "hotel" (or "hotel:<var>") = a
                          # hotel-injected salinity channel (default "salinity")

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
