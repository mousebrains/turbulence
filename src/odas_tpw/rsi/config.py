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
        "W_min": None,  # null = auto: 0.3 free-fall, 0.05 glide/horizontal
        "direction": "auto",
        "vehicle": None,
        "min_duration": 7.0,
    },
    "epsilon": {
        "fft_length": 1024,
        "diss_length": None,
        "overlap": None,
        "speed": None,
        "speed_method": None,  # null = pressure |dP/dt|; em | flight
        "aoa_deg": 3.0,
        "direction": "auto",
        "vehicle": None,
        "W_min": None,  # null = auto: 0.3 free-fall, 0.05 glide/horizontal
        "goodman": True,
        "f_AA": 98.0,
        "f_limit": None,
        "fit_order": 3,
        "despike_thresh": 8,
        "despike_smooth": 0.5,
        # Shear despike removal half-width [s]. At the 0.04 default the key is
        # dropped from the canonical view, so every existing epsilon hash is
        # unchanged; any other value re-versions the output.
        "despike_half_width": 0.04,
        # Subtract shear interference locked to the .p data record
        # (rsi.record_locked). Dropped from the canonical view while false.
        "record_locked_shear": False,
        "salinity": None,
        "temperature": "auto",
        "conductivity": "auto",
    },
    "chi": {
        "fft_length": 1024,
        "diss_length": None,
        "overlap": None,
        "speed": None,
        "speed_method": None,  # null = pressure |dP/dt|; em | flight
        "aoa_deg": 3.0,
        "direction": "auto",
        "vehicle": None,
        "W_min": None,  # null = auto: 0.3 free-fall, 0.05 glide/horizontal
        "fp07_model": "single_pole",
        "goodman": True,
        "f_AA": 98.0,
        "fit_method": "iterative",
        "spectrum_model": "kraichnan",
        # Lowest FFT bin chi uses, DC = bin 0 (chi.chi.FIRST_BIN, issue #191).
        # At 1, the band before the key existed, it is dropped from the
        # canonical view so those hashes are unchanged.
        "first_bin": 2,
        "salinity": None,
        "temperature": "auto",
        "conductivity": "auto",
        # Measured per-bead FP07 corrections (fp07-pair fit). The CLI hashes
        # their CONTENT digests here; unset, the key is dropped from the
        # canonical view so every existing chi hash is unchanged.
        "fp07_response": None,
        # A bead's slow response path, {bead: {beta, tau_s_ms, first, last}}
        # (chi.fp07_slow, issue #252). Hashed with spans as epoch seconds;
        # unset, dropped from the canonical view like fp07_response.
        "fp07_slow_response": None,
    },
}

# chi keys whose unset (None) value is left out of the canonical view, so the
# hash of every chi run from before the key existed is unchanged.
_CHI_DROP_WHEN_UNSET = ("fp07_response", "fp07_slow_response")


class _RsiConfigManager(ConfigManager):
    def _section_view_postprocess(self, section: str, mapping: dict) -> dict:
        if section == "chi":
            return {
                k: v for k, v in mapping.items()
                if not (k in _CHI_DROP_WHEN_UNSET and v is None)
                and not (k == "first_bin" and v == 1)
            }
        if section == "epsilon":
            # At the values that reproduce the code from before each key
            # existed, the key is left out, so those hashes are unchanged.
            inert = {"despike_half_width": 0.04, "record_locked_shear": False}
            return {k: v for k, v in mapping.items() if k not in inert or v != inert[k]}
        return mapping

    def validate_config(self, config: dict) -> None:
        """Unknown sections/keys, plus the epsilon despike/record-locked values.

        Checked at load time so a bad value fails once, before the output
        directory is created, rather than once per file inside the workers.
        """
        super().validate_config(config)
        from odas_tpw.rsi.dissipation import validate_epsilon_options

        validate_epsilon_options(config.get("epsilon") or {})
        first_bin = (config.get("chi") or {}).get("first_bin")
        if first_bin is not None:
            from odas_tpw.chi.chi import validate_first_bin

            validate_first_bin(first_bin, where="[chi] first_bin")
        slow = (config.get("chi") or {}).get("fp07_slow_response")
        if slow is not None:
            from odas_tpw.chi.fp07_slow import parse_slow_response

            parse_slow_response(slow, where="[chi] fp07_slow_response")


_mgr = _RsiConfigManager(DEFAULTS)

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
  W_min: null           # minimum fall rate [dbar/s] (null = auto: 0.3
                        # free-fall, 0.05 glide/horizontal)
  direction: auto       # profile direction: auto, up, down, glide, horizontal
  vehicle: null         # vehicle override (null = from .p file)
  min_duration: 7.0     # minimum profile duration [s]

epsilon:
  fft_length: 1024      # FFT segment length [samples]
  diss_length: null     # dissipation window [samples] (null = 4 * fft_length)
  overlap: null         # window overlap [samples] (null = diss_length // 2)
  speed: null           # fixed profiling speed [m/s] (null = speed_method)
  speed_method: null    # through-water speed model: null = pressure |dP/dt|;
                        # em (U_EM flowmeter); flight (|W|/sin(|pitch|-aoa))
  aoa_deg: 3.0          # angle of attack [deg] for speed_method: flight
  direction: auto       # profile direction: auto, up, down, glide, horizontal
  vehicle: null         # vehicle override (null = from .p file)
  W_min: null           # minimum fall rate [dbar/s] (null = auto: 0.3
                        # free-fall, 0.05 glide/horizontal)
  goodman: true         # Goodman coherent noise removal
  f_AA: 98.0            # anti-aliasing filter cutoff [Hz]
  f_limit: null         # upper frequency limit [Hz] (null = f_AA)
  fit_order: 3          # polynomial fit order for Nasmyth integration
  despike_thresh: 8     # despike threshold (rectified-HP / LP-envelope ratio, not MAD)
  despike_smooth: 0.5   # despike envelope low-pass cutoff [Hz]
  despike_half_width: 0.04  # despike removal half-width [s]: blanks half of it
                        # before each spike and all of it after (0.04 = ~60 ms
                        # per spike, 0.01 = ~14 ms); in (0, 0.5]
  record_locked_shear: false  # subtract shear interference that sits at fixed
                        # positions in every .p data record (an electrical
                        # pulse the despiker misses below its threshold);
                        # corrects only samples where the per-profile record
                        # mean is significant. Needs a .p file or a profile
                        # NetCDF from rsi-tpw prof
  salinity: null        # PSU value, or "measured" (JAC/selected C+T) (null = 35, fixed S)
  temperature: auto     # reference temperature for viscosity: auto = first
                        # plausible of T1..Tn, T, JAC_T; a channel name (QC
                        # failure warns but proceeds); or a number = constant
                        # reference temperature [degC]
  conductivity: auto    # conductivity channel for salinity "measured"
                        # (auto = JAC_C when present)

chi:
  fft_length: 1024      # FFT segment length [samples]
  diss_length: null     # dissipation window [samples] (null = 4 * fft_length)
  overlap: null         # window overlap [samples] (null = diss_length // 2)
  speed: null           # fixed profiling speed [m/s] (null = speed_method)
  speed_method: null    # through-water speed model: null = pressure |dP/dt|;
                        # em (U_EM flowmeter); flight (|W|/sin(|pitch|-aoa))
  aoa_deg: 3.0          # angle of attack [deg] for speed_method: flight
  direction: auto       # profile direction: auto, up, down, glide, horizontal
  vehicle: null         # vehicle override (null = from .p file)
  W_min: null           # minimum fall rate [dbar/s] (null = auto: 0.3
                        # free-fall, 0.05 glide/horizontal)
  fp07_model: single_pole  # FP07 transfer function: single_pole or double_pole
  goodman: true         # Goodman coherent noise removal
  f_AA: 98.0            # anti-aliasing filter cutoff [Hz]
  fit_method: iterative # Method 2 fitting: iterative or mle
  spectrum_model: kraichnan  # theoretical spectrum: batchelor or kraichnan
  first_bin: 2          # lowest FFT bin chi uses (DC = 0). The per-segment
                        # linear detrend leaves bin 1 ~30% low on a red
                        # spectrum (issue #191); 1 reproduces a product
                        # made before bin 1 was dropped
  salinity: null        # PSU value, or "measured" (JAC/selected C+T) (null = 35, fixed S)
  temperature: auto     # reference temperature for viscosity/kappa_T: auto =
                        # first plausible of T1..Tn, T, JAC_T; a channel name
                        # (QC failure warns but proceeds); or a number =
                        # constant reference temperature [degC]
  conductivity: auto    # conductivity channel for salinity "measured"
                        # (auto = JAC_C when present)
  fp07_response: null   # per-bead FP07 corrections from `fp07-pair fit` (a
                        # list of files; rsi-tpw chi only -- pipeline refuses
                        # it). See docs/fp07pair/runbook.md
  fp07_slow_response: null  # a bead's slow response path divided out of its
                        # spectra, keyed by bead serial with a mandatory UTC
                        # span (rsi-tpw chi only; never with fp07_response):
                        #   T2005: {beta: 0.124, tau_s_ms: 110,
                        #           first: "2021-06-29T18:00:00Z",
                        #           last: "2022-06-20T08:30:00Z"}
                        # Calibrated temperature only. See chi/fp07_slow.py
"""


def generate_template(path: str | Path) -> Path:
    """Write a fully-commented template configuration file."""
    path = Path(path)
    # encoding is explicit: the default is the platform codepage, so on
    # Windows a non-ASCII character (the template has an em dash) is written
    # as cp1252 and load_config, which reads utf-8, then raises
    # UnicodeDecodeError on a file we just generated.
    path.write_text(_TEMPLATE, encoding="utf-8")
    return path
