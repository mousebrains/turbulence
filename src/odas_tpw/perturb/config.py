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
import math
import os
import warnings
from pathlib import Path
from typing import Any

from odas_tpw.config_base import ConfigManager
from odas_tpw.perturb.probe_exclusion import parse_exclusions

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
        # REQUIRED when hotel.enable is true. None is not a default value here
        # -- it is the "you have not said" state, and interpolate_hotel raises
        # on it. Interpolating across an unbounded gap manufactures data that
        # ct / ctd / stratification / salinity:"measured" / epsilon.T_source all
        # read as measurement, so the limit has to be a decision, not a default.
        # The opt-out is the explicit string "unlimited".
        "max_gap": None,
        # Flipped: an edge-held constant outside the source's coverage
        # correlates with nothing and is not a measurement.
        "extrapolate": False,
        # [s] ADDED to the hotel file's timestamps to put them on the
        # instrument's clock. A MicroRider takes its time from the science
        # computer once, at file open, and then free-runs, so two
        # unsynchronised clocks is the normal case. `fp07-cal fit` measures the
        # offset from pressure against pressure (clock_offset_s in
        # coefficients.json) with no thermal physics involved.
        #
        # SIGN: clock_offset_s is positive when the hotel timestamps run AHEAD
        # of the instrument's, so set time_offset = -clock_offset_s. Only a
        # MEASURED one: a record whose clock_source says the 0 was asserted
        # (fp07-cal's reference.same_clock: true) measured nothing.
        # 0.0 means "the two clocks agree", which is an assertion, not a
        # default to leave unexamined: on osu684 the offset was +5.06 s, and at
        # a 0.27 dbar/s climb that puts every merged CTD sample 1.4 dbar from
        # where it was taken.
        "time_offset": 0.0,
    },
    "profiles": {
        "P_min": 0.5,
        "W_min": None,  # null = auto: 0.3 free-fall, 0.05 glide/horizontal
        "direction": "auto",  # auto = vehicle-resolved (glide for a glider)
        "min_duration": 7.0,
        "diagnostics": False,
    },
    "fp07": {
        "calibrate": True,
        "order": 2,
        "max_lag_seconds": 10,
        "reference": "JAC_T",
        "must_be_negative": True,
        "min_corr": 0.5,
        "reference_interval": None,
        "allow_hotel_reference": False,
    },
    "ct": {
        "align": True,
        "T_name": "JAC_T",
        "C_name": "JAC_C",
    },
    "bottom": {
        "enable": False,
        # "vibration" = the historical binned-vibration detector.
        # "fallrate"  = the depth at which free fall ends, from the pressure
        #               record. Measured on SUNRISE 2022 Point Sur (134
        #               descents) the vibration detector reported the bottom a
        #               median 1.28 m above where the instrument actually
        #               stopped; fall-rate put it at 0.086 m, a 14.9x gain, and
        #               fired on 100% of descents against 97%.
        # "both"      = fall-rate for the datum, vibration as a cross-check.
        "method": "vibration",
        "depth_window": 4.0,
        "depth_minimum": 10.0,
        "speed_factor": 0.3,
        "median_factor": 1.0,
        "vibration_frequency": 16,
        "vibration_factor": 4.0,
        # Was silently unsettable before 2026-09-22: absent from these
        # defaults, merge_config dropped it with no error and no effect, so the
        # one parameter that actually gates detection could not be tuned while
        # the three that do nothing at least warned.
        "proximity_bins": 2,
        # Clearance ABOVE the detected bottom for the deepest analysis window.
        # 0.15 m clears the deceleration (free fall ends a median 0.086-0.109 m
        # above the stop, p90 0.13 m, on two ships) and keeps the retained
        # record entirely PRE-IMPACT, so a silty seabed's resuspension cloud
        # does not yet exist in it. It need not include the probe-to-crasher
        # offset: the crasher ring leads the shear probes by 6-8 inches, so the
        # deepest pre-impact sample already sits that far above the seabed.
        "height": 0.15,
        "fallrate_fraction": 0.9,
        "fallrate_smooth_s": 0.05,
    },
    "top_trim": {
        "enable": False,
        "dz": 0.5,
        "min_depth": 1.0,
        "max_depth": 50.0,
        "quantile": 0.6,
        "noise_factor": 2.0,
        # The fall-rate residual is ON by default: Ax/Ay on a Rockland VMP
        # are piezo VIBRATION sensors, blind to prop wash, and on their own
        # they trim to ~3 m while two decades of wash epsilon survive.
        # "max" goes with it -- under a median the two blind channels
        # outvote the one that can see. Only affects configs that already
        # set top_trim.enable.
        "combine": "max",
        "use_fall_rate_residual": True,
        "residual_smooth_m": 6.0,
        # Inclinometer voter, OFF by default. In a tow-yo the VMP is launched
        # nearly parallel to the surface and pitches down as it falls, so the
        # shear probes see a mean CROSS-FLOW until it is vertical and epsilon
        # (~ shear^2/U^4) is meaningless, not merely contaminated. Measured on
        # SUNRISE 2021 Walton Smith SN 194, 773 descents: Incl_Y starts at a
        # median 20.2 deg and reaches vertical at 3.9 m, but anywhere in
        # 2.7-5.7 m -- so a flat floor cannot do this job and a per-cast test
        # can. Default OFF because it is VMP-SPECIFIC: it relies on Incl_Y
        # being +90 = nose down, which is a VMP convention and NOT a
        # MicroRider one (there Incl_Y is roughly pitch).
        "use_inclinometer": False,
        "incl_channel": "Incl_Y",
        "incl_threshold_deg": 85.0,
        "incl_relax_deg": 5.0,
        "incl_hold_m": 1.0,
        # Reaching vertical is not the same as the flow having settled:
        # epsilon is still 6.7x interior at the vertical depth, 2.2x one metre
        # below, 1.2x two metres below.
        "incl_margin_m": 2.0,
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
        # Window layout. "top" is the historical behaviour: windows start at
        # the top of the section and march down, so the bottom remainder is
        # DISCARDED -- which on a profiler that crashes into the seabed throws
        # away the boundary layer. "bottom" pins the deepest window's lower
        # edge to the end of the section and marches up.
        "anchor": "top",
        # A shorter dissipation window for the near-bottom zone. Epsilon is
        # attributed to a window's CENTRE, so halving the window halves the
        # distance from the seabed to the deepest estimate. Costs DOF: with
        # fft_sec 0.5 s, a 2.0 s window averages 7 FFTs and a 1.0 s window 3.
        # Only the DISSIPATION length may differ -- fft_length is shared,
        # because one product carries one wavenumber axis.
        "bbl_diss_sec": None,  # None = no separate BBL windows
        "bbl_overlap_sec": None,  # None = half the BBL window
        "bbl_extent_sec": None,  # how far above the seabed they apply
        "bbl_diss_length": None,
        "bbl_overlap": None,
        "bbl_extent": None,
        "goodman": True,
        "f_AA": 98.0,
        "f_limit": None,
        "fit_order": 3,
        "despike_thresh": 8,
        "despike_smooth": 0.5,
        # Shear despike removal half-width [s] (rsi.dissipation
        # DESPIKE_HALF_WIDTH). At the 0.04 default it is dropped from the
        # canonical view, so existing signatures are unchanged.
        "despike_half_width": 0.04,
        # Subtract shear interference locked to the .p data record
        # (rsi.record_locked), per profile, in the diss stage. Dropped from the
        # canonical view while false; true re-versions diss and, through
        # Method 1's epsilon, chi -- but not profiles, whose data it leaves
        # alone.
        "record_locked_shear": False,
        "salinity": None,
        "epsilon_minimum": 1e-13,
        "T_source": None,  # reference temperature for seawater
        # properties (epsilon viscosity; chi
        # viscosity/kappa_T — one knob serves both
        # stages). null = "auto" QC chain; a channel
        # name; or a number = constant reference
        # temperature [degC].
        "spectral_qc": False,  # ATOMIX-style FM + var_resolved cut on
        # per-probe epsilon. Default FALSE while chi.spectral_qc
        # defaults True — a deliberate, documented asymmetry:
        # enabling it moves ~20% of probe-windows and it is not
        # yet validated against the ATOMIX benchmark.
        "FM_max": 1.15,  # bit 1. The MAD-based FM, NOT the
        # variance-ratio `fom` that fom_max thresholds.
        "var_resolved_min": 0.5,  # bit 16, method == 0 only
        "pair_policy": "keep_both",  # keep_both | drop_high | flag_only
        # Two-probe windows only; >=3 probes stay with
        # mk_epsilon_mean's symmetric outlier rule.
        "pair_limit": 2.7718585822512662,  # 1.96*sqrt(2); the FULL
        # coefficient of mean(sigma_ln), matching
        # scor160.l4.DEFAULT_DISS_RATIO_LIMIT exactly
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
        # anchor: top | bottom, same meaning as [epsilon]. Chi needs its OWN
        # key because the two are laid out independently: with chi left at
        # "top" while epsilon is "bottom", a 2.0 s window puts chi's deepest
        # center ~1.2 m above the seabed, so a bottom-up product has a proper
        # BBL epsilon and NO BBL chi (measured on SUNRISE 2021/Pelican/SN194:
        # chi occupancy 0.0% at 0.62 m and 0.4% at 0.88 m against epsilon's
        # 95.7%). Gamma and K_T inherit that hole.
        "anchor": "top",
        # The bbl_* twins mirror [epsilon] exactly. They became possible once
        # L3ChiData.dof_spec was made PER WINDOW: a shorter near-bed window
        # averages fewer FFT segments, and that value gates l4_chi's
        # non-detection floor, so a single file-wide DOF would mis-gate exactly
        # the near-bed estimates the BBL analysis exists to measure.
        "bbl_diss_sec": None,
        "bbl_overlap_sec": None,
        "bbl_extent_sec": None,
        "bbl_diss_length": None,
        "bbl_overlap": None,
        "bbl_extent": None,
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
        # K_T, Gamma, K_rho) on the chi grid.
        # N2 salinity follows
        # stratification.salinity: the profile's
        # own C/T/P (TEOS-10) by default, or a
        # fixed value / hotel-injected channel.
        "chi_minimum": 1.0e-13,
        "spectral_qc": True,  # Soft fom + K_max_ratio QC on chiMean
        # (hence K_T/Gamma), matching the rsi
        # chi_final. Never drops a window (falls
        # back to all probes when none pass). Set
        # false to publish unfiltered chiMean.
        "mixing_use_qc_fallback": False,  # May windows kept ONLY by that
        # fallback (chiQCFallback == 1: no probe
        # passed) feed K_T/Gamma/K_rho? They are
        # always reported in chiMean; true restores
        # the pre-#180 behaviour of also building
        # mixing coefficients from them.
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
        "method": "pressure",  # pressure | em | flight | constant | hotel
        "value": None,  # m/s, only for method="constant"
        "hotel_var": "speed",  # merged channel name, only for method="hotel"
        "aoa_deg": 3.0,  # angle of attack, only for method="flight"
        "min_pitch_deg": 5.0,  # flight: skip |pitch| below this (deg)
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
        # depth | time | altitude.  "altitude" bins on HEIGHT ABOVE THE SEABED
        # using each profile's `bottom_depth` datum. It is NOT a relabelling of
        # the depth axis: within one SUNRISE leg the seabed moves from 21.5 to
        # 34.5 m (p10-p90), so a fixed depth bin sits 12 m above the bed in one
        # cast and 1 m above it in the next. The conversion must happen before
        # the bin assignment or the boundary layer is averaged away. Requires
        # [bottom] enabled, since the datum is written at profile extraction.
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
    # products, independent of epsilon/chi. Computed from each window's parcels
    # after reordering them (``sort``). diss uses its dissipation window; the
    # profile product uses the configurable ``window`` below. NOT written to the
    # CTD product (profile-only; the CTD spans the whole up/down trajectory).
    "stratification": {
        "enable": True,
        "window": 2.0,  # background vertical window [dbar] for the
        # profile product
        "salinity": None,  # N2 salinity source: null = conductivity (else 35),
        # a number = fixed PSU, "measured" = C/T/P (TEOS-10),
        # "hotel"/"hotel:<var>" = a hotel-injected salinity channel
        "sort": "temperature",  # parcel order within each window:
        # "temperature" (conservative temperature alone; immune to salinity
        # noise; the default since 2026-10), "density" (Thorpe-sorted; the
        # default before), or "none" (as measured)
        "dTdz_source": "ct",  # "ct" = ct.T_name; "fp07" = the FP07
        # thermistors (mean of per-probe slopes; N2 stays on the CT pair)
    },
    "parallel": {
        "jobs": 1,
    },
    # Per-instrument overrides keyed by serial-number identifier (matched
    # against the parent directory of each .p file). Allowed inner keys:
    #   exclude_shear_probes : list[str]  (e.g. ["sh1"] or ["sh2"])
    #   exclude              : [{probe, P_min, P_max, start, stop, reason}]
    #                          one probe masked over a pressure/time window
    #                          (perturb/probe_exclusion.py, issue #190)
    #   fp07_tau_scale       : {probe: multiplier}
    #   fp07_response        : path | [paths] from `fp07-pair fit`
    #                          (mutually exclusive with fp07_tau_scale)
    #   fp07_slow_response   : {bead: {beta, tau_s_ms, first, last} | [...]}
    #                          a bead's slow response path (chi/fp07_slow.py,
    #                          issue #252); exclusive with both of the above
    "instruments": {},
}

# Keys excluded from hashing — runtime toggles that must NOT create new output
# dirs: ``diagnostics`` (extra plots), and the cache-bypass flags ``force`` /
# ``force_trim`` (they re-run work into the SAME dirs, so they cannot change the
# dir identity or --force would defeat itself by recomputing into fresh dirs).
# ``fp07.allow_hotel_reference`` only decides whether a run may START; a run it
# allows writes exactly what it wrote before the key existed, so hashing it
# would re-version every product of every deployment for nothing.
_HASH_EXCLUDE_KEYS = frozenset(
    {"diagnostics", "force", "force_trim", "config_dir", "allow_hotel_reference"}
)

# Sections whose inner keys are user-defined (e.g. instrument serials).
# ConfigManager skips strict unknown-key validation for these; per-section
# inner-schema validation happens in a wrapper below.
_DYNAMIC_KEY_SECTIONS = frozenset({"instruments"})

_INSTRUMENT_VALID_KEYS = frozenset(
    {"exclude_shear_probes", "exclude", "fp07_tau_scale", "fp07_response", "fp07_slow_response"}
)

# Which `instruments.<SN>.*` keys actually reach each stage. The block is hashed
# into a stage's signature so a change re-versions its output -- but not every
# key touches every stage. `fp07_tau_scale` is consumed only by the chi
# computation, so letting it version the DISS directory forces a full epsilon
# recompute that cannot change a single value (verified bit-identical over
# 34820 epsilon / FM / fom / var_resolved values on ARCTERX-2022).
# `exclude_shear_probes` does reach diss, and reaches chi transitively through
# Method 1's epsilonMean, so it belongs to both. So does `exclude`, but only
# entry by entry (_EXCLUDE_PRODUCTS_BY_STAGE).
_INSTRUMENT_KEYS_BY_STAGE: dict[str, frozenset[str]] = {
    "diss": frozenset({"exclude_shear_probes", "exclude"}),
    "chi": frozenset(
        {"exclude_shear_probes", "exclude", "fp07_tau_scale", "fp07_response", "fp07_slow_response"}
    ),
}

# Which `exclude` entries reach each stage, by the product their probe is masked
# in. A shear entry masks diss, and reaches chi through epsilonMean (Method 1,
# epsilon_paired, K_rho, Gamma; every chi file records it). A thermistor entry
# masks chi only and cannot change a diss byte, so it does not version diss.
_EXCLUDE_PRODUCTS_BY_STAGE: dict[str, frozenset[str]] = {
    "diss": frozenset({"diss"}),
    "chi": frozenset({"diss", "chi"}),
}

# Numeric dependencies whose version can change ε/χ/N² outputs even with no
# change to our own source — folded into the engine fingerprint so a dep
# upgrade also invalidates cached results.
_ENGINE_DEPS = ("numpy", "scipy", "gsw", "netCDF4", "xarray", "pandas")
# Code that cannot affect processing numerics; excluded from the source hash so
# editing plots, standalone tools or the --version and provenance-stamp code
# doesn't invalidate cached science. The flip side: a fix to the stamps
# themselves reaches existing products only on a run with --force.
_FINGERPRINT_EXCLUDE = ("perturb/plot/", "_version_info.py")
_ENGINE_OVERRIDE_ENV = "ODAS_TPW_ENGINE_FINGERPRINT"


@functools.lru_cache(maxsize=1)
def engine_fingerprint() -> str:
    """Hash of the processing *code + key numeric deps*.

    Folded into every stage signature so a change that could alter outputs
    yields new ``{stage}_NN`` dirs (recompute) while an unchanged engine reuses
    them — making "outputs already exist in this dir" a *safe* cache signal.
    Covers our own ``.py`` source (excluding plot/) plus the installed
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
    ("bbl_diss_length", "bbl_diss_sec"),
    ("bbl_overlap", "bbl_overlap_sec"),
    ("bbl_extent", "bbl_extent_sec"),
)

_WINDOW_SECTIONS = ("epsilon", "chi")

# stratification.dTdz_source values (validated at load).
DTDZ_SOURCES = ("ct", "fp07")


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
        if section == "stratification":
            # A key is dropped only when it selects what the code did before the
            # key existed, so those configs keep their canonical form (and
            # stage signature) and their products stay valid. That is
            # sort: density, NOT the current default: products made before the
            # default became "temperature" are density-sorted, so a config that
            # leaves sort unset (or null) must hash differently from them and be
            # reprocessed. dTdz_source: ct is both the old behavior and the
            # default.
            if mapping.get("sort") is None:
                mapping["sort"] = DEFAULTS["stratification"]["sort"]
            if mapping["sort"] == "density":
                mapping.pop("sort")
            if mapping.get("dTdz_source") in ("ct", None):
                mapping.pop("dTdz_source", None)
        if section in _WINDOW_SECTIONS:
            for samples_key, sec_key in _WINDOW_KEY_PAIRS:
                if sec_key in mapping and (
                    mapping[sec_key] is None or mapping.get(samples_key) is not None
                ):
                    del mapping[sec_key]
            # Same rule for the bottom-up layout keys: a config that does not
            # use them must keep the canonical form -- and therefore the
            # stage-directory signature -- it had before they existed, so that
            # adding the feature does not silently re-version every campaign's
            # products. A config that DOES set one changes the hash, as it must.
            if mapping.get("anchor") == "top":
                mapping.pop("anchor", None)
            # Only an inert (None) sample key may be dropped. Popping all three
            # whenever bbl_diss_length was None also dropped a GOVERNING
            # bbl_overlap/bbl_extent beside a bbl_diss_sec window, so changing
            # it left the hash -- and the cached product -- unchanged.
            if mapping.get("bbl_diss_length") is None:
                for k in ("bbl_diss_length", "bbl_overlap", "bbl_extent"):
                    if mapping.get(k) is None:
                        mapping.pop(k, None)
        if section == "epsilon":
            # At the values that reproduce the code from before each key
            # existed (0.04 was hard-wired), the key is left out.
            for key, inert in (("despike_half_width", 0.04), ("record_locked_shear", False)):
                if key in mapping and mapping[key] == inert:
                    del mapping[key]
        return mapping

    def validate_config(self, config: dict) -> None:
        """Structural validation plus fs-independent window-duration checks.

        Durations are validated at LOAD time so a sign typo (fft_sec: -1)
        or an inverted window pair fails before any stage directory is
        created, rather than aborting mid-run inside per-file processing.
        """
        super().validate_config(config)
        # binning.method is routed by membership tests, and every one of them
        # falls back to TIME binning. An unrecognised value therefore produces a
        # complete, writable, silently wrong product rather than an error, so it
        # has to be rejected here at load time.
        # Imported lazily: binning pulls numpy/xarray, and building the CLI
        # parser must not (tests/test_perturb_mk_sections.py pins that).
        from odas_tpw.perturb.binning import BIN_METHODS

        bin_method = (config.get("binning") or {}).get("method", "depth")
        if bin_method not in BIN_METHODS:
            raise ValueError(
                f"binning.method: expected one of {sorted(BIN_METHODS)}, "
                f"got {bin_method!r}"
            )
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
            # The window layout is validated here too: section_groups would
            # otherwise raise on a bad anchor once per profile, and a BBL window
            # under two FFT segments passes resolve_window_config only to fail
            # inside csd_matrix_batch, leaving an empty stage directory.
            anchor = params.get("anchor", "top")
            if anchor not in ("top", "bottom"):
                raise ValueError(
                    f"{section}.anchor: expected 'top' or 'bottom', got {anchor!r}"
                )
            bbl_s = params.get("bbl_diss_sec")
            if (
                isinstance(fft_s, (int, float))
                and isinstance(bbl_s, (int, float))
                and bbl_s < 2 * fft_s
            ):
                raise ValueError(
                    f"{section}: bbl_diss_sec ({bbl_s}) must hold at least two FFT "
                    f"segments (2 x fft_sec = {2 * fft_s})"
                )
            uses_bbl = (
                params.get("bbl_diss_sec") is not None
                or params.get("bbl_diss_length") is not None
            )
            if uses_bbl and anchor != "bottom":
                warnings.warn(
                    f"{section}: bbl_* keys take effect only with anchor: bottom; "
                    "they are ignored with anchor: top",
                    UserWarning,
                    stacklevel=2,
                )
        from odas_tpw.rsi.dissipation import validate_epsilon_options

        validate_epsilon_options(config.get("epsilon") or {})
        # stratification.sort / dTdz_source pick a code path by membership, so a
        # typo would otherwise fall through to a default silently.
        strat = config.get("stratification") or {}
        if isinstance(strat, dict):
            from odas_tpw.processing.mixing import SORT_MODES

            sort = strat.get("sort")
            if sort is not None and sort not in SORT_MODES:
                raise ValueError(
                    f"stratification.sort: expected one of {list(SORT_MODES)}, got {sort!r}"
                )
            src = strat.get("dTdz_source", "ct")
            if src is not None and src not in DTDZ_SOURCES:
                raise ValueError(
                    f"stratification.dTdz_source: expected one of {list(DTDZ_SOURCES)}, "
                    f"got {src!r}"
                )
        # hotel.max_gap is required when the hotel merge is enabled; validate
        # it (and every per-channel override) here so a missing value fails
        # once at load time rather than once per file inside the worker pool.
        hotel = config.get("hotel") or {}
        if isinstance(hotel, dict) and hotel.get("enable", False):
            from odas_tpw.perturb.hotel import resolve_gap_settings

            resolve_gap_settings({**DEFAULTS["hotel"], **hotel})


_mgr = _PerturbConfigManager(
    DEFAULTS,
    hash_exclude_keys=_HASH_EXCLUDE_KEYS,
    dynamic_key_sections=_DYNAMIC_KEY_SECTIONS,
    engine_fingerprint=engine_fingerprint,
)


def _validate_instruments(instruments: dict) -> None:
    """Validate the inner structure of the dynamic ``instruments`` section."""
    for sn, settings in instruments.items():
        # process_file looks a file's settings up by its raw directory name, a
        # string, so a key YAML loads as anything else -- `465:` is an int,
        # `2017-10-24:` a date -- can never match and would be ignored silently.
        if not isinstance(sn, str):
            raise ValueError(
                f"instruments: key {sn!r} loaded as {type(sn).__name__}, so it can never "
                f"match an instrument key, which is the raw directory name of the .p "
                f"files (a string); quote it ('{sn}')"
            )
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
        # Windowed exclusions: every entry is checked here, at load, so a typo
        # fails before any stage directory exists. Whether each probe is on
        # the instrument can only be known from its data, at run time.
        parse_exclusions(sn, settings.get("exclude"))
        tau_scale = settings.get("fp07_tau_scale", {})
        if not isinstance(tau_scale, dict):
            raise ValueError(
                f"instruments.{sn}.fp07_tau_scale: must be a mapping of "
                f"thermistor name -> multiplier, got {tau_scale!r}"
            )
        for probe, value in tau_scale.items():
            if not isinstance(probe, str):
                raise ValueError(
                    f"instruments.{sn}.fp07_tau_scale: keys must be thermistor "
                    f"names (strings), got {probe!r}"
                )
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"instruments.{sn}.fp07_tau_scale.{probe}: must be a number, "
                    f"got {value!r}"
                )
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"instruments.{sn}.fp07_tau_scale.{probe}: must be finite and "
                    f"> 0, got {value!r}"
                )
        response = settings.get("fp07_response")
        if response is not None:
            items = [response] if isinstance(response, (str, os.PathLike)) else response
            if (
                not isinstance(items, list)
                or not items
                or not all(isinstance(x, (str, os.PathLike)) for x in items)
            ):
                raise ValueError(
                    f"instruments.{sn}.fp07_response: must be a path or a non-empty list "
                    f"of paths to fp07-pair corrections, got {response!r}"
                )
            if tau_scale:
                raise ValueError(
                    f"instruments.{sn}: fp07_response and fp07_tau_scale are mutually "
                    "exclusive. The correction is measured relative to the bead pair, so "
                    "stacking it on a tau scale corrects the relative part twice (on "
                    "ARCTERX-2022 SN194 it would invert the T1/T2 disagreement). Remove "
                    "fp07_tau_scale, and expect chi to move by its geometric mean."
                )
        slow = settings.get("fp07_slow_response")
        if slow is not None:
            # Imported here, not at module top: building the CLI parser must
            # not pull numpy in (see probe_exclusion).
            from odas_tpw.chi.fp07_slow import parse_slow_response

            if parse_slow_response(slow, where=f"instruments.{sn}.fp07_slow_response"):
                stacked = [k for k in ("fp07_tau_scale", "fp07_response") if settings.get(k)]
                if stacked:
                    raise ValueError(
                        f"instruments.{sn}: fp07_slow_response and {stacked[0]} are mutually "
                        "exclusive. Both were fitted on spectra that already contain the slow "
                        "path (ARCTERX-2022's tau scales; an fp07-pair correction's ln e), so "
                        "stacking corrects the bead twice. Remove "
                        f"{stacked[0]}; an fp07-pair fit on slow-corrected spectra is future work."
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
        _resolve_fp07_response(config)
    return config


def _resolve_fp07_response(config: dict) -> None:
    """Make every ``instruments.<SN>.fp07_response`` path absolute, and check each file.

    Like every other perturb path: ``<CONFIG_DIR>/...`` resolves against the
    config's own directory (so a config and its corrections move together),
    and a relative path without the token against the working directory. Each
    file is opened now -- a missing or hand-edited correction fails at load,
    not in a worker halfway through a run.
    """
    from odas_tpw.chi.fp07_pair import ResponseCorrection

    cdir = config_dir_of(config)
    for sn, settings in (config.get("instruments") or {}).items():
        if not isinstance(settings, dict) or settings.get("fp07_response") is None:
            continue
        raw = settings["fp07_response"]
        items = [raw] if isinstance(raw, (str, os.PathLike)) else list(raw)
        resolved = []
        for item in items:
            path = Path(str(expand_config_dir(item, cdir)))
            if not path.exists():
                raise ValueError(f"instruments.{sn}.fp07_response: {path} does not exist")
            ResponseCorrection.from_netcdf(path)
            resolved.append(str(path.resolve()))
        settings["fp07_response"] = resolved


_DIGEST_CACHE: dict[tuple[str, int, int], str] = {}


def _fp07_response_digest(path: str) -> str:
    """Semantic digest of a correction file, cached by (path, size, mtime)."""
    from odas_tpw.chi.fp07_pair import ResponseCorrection

    st = os.stat(path)
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _DIGEST_CACHE:
        _DIGEST_CACHE[key] = ResponseCorrection.from_netcdf(path).digest()
    return _DIGEST_CACHE[key]


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

    out = {
        k: v
        for k, v in cfg.items()
        if k
        not in (
            "fft_sec",
            "diss_sec",
            "overlap_sec",
            "bbl_diss_sec",
            "bbl_overlap_sec",
            "bbl_extent_sec",
        )
    }
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

    # Near-bottom windows. Resolved the same way, then checked against the
    # main window: a BBL window LONGER than the ordinary one would defeat the
    # purpose, and one shorter than the FFT cannot be transformed at all.
    bbl = cfg.get("bbl_diss_length")
    if bbl is None:
        bbl_sec = seconds("bbl_diss_sec")
        bbl = even(fs * bbl_sec) if bbl_sec is not None else None
    if bbl is not None:
        bbl = int(bbl)
        if bbl < 2 * fft:
            raise ValueError(
                f"{section}: BBL window ({bbl} samples) must hold at least two "
                f"FFT segments ({2 * fft} samples); csd_matrix_batch refuses "
                "anything shorter"
            )
        if bbl > diss:
            raise ValueError(
                f"{section}: BBL window ({bbl} samples) is longer than the "
                f"ordinary dissipation window ({diss} samples); the BBL window "
                "exists to buy vertical resolution near the seabed, so it must "
                "be shorter"
            )
        out["bbl_diss_length"] = bbl

        bbl_ov = cfg.get("bbl_overlap")
        if bbl_ov is None:
            ov_sec = seconds("bbl_overlap_sec")
            bbl_ov = even(fs * ov_sec) if ov_sec is not None else bbl // 2
        bbl_ov = int(bbl_ov)
        if bbl_ov >= bbl:
            raise ValueError(
                f"{section}: BBL overlap ({bbl_ov} samples) must be smaller "
                f"than the BBL window ({bbl} samples)"
            )
        out["bbl_overlap"] = bbl_ov

        ext = cfg.get("bbl_extent")
        if ext is None:
            ext_sec = seconds("bbl_extent_sec")
            ext = even(fs * ext_sec) if ext_sec is not None else bbl
        ext = int(ext)
        if ext < bbl:
            raise ValueError(
                f"{section}: BBL extent ({ext} samples) is shorter than one "
                f"BBL window ({bbl} samples), so no BBL window would fit"
            )
        out["bbl_extent"] = ext
    else:
        out.pop("bbl_diss_length", None)
        out.pop("bbl_overlap", None)
        out.pop("bbl_extent", None)
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


def canonical_instruments_for_hash(
    instruments: dict | None,
    keys: frozenset[str] | None = None,
    exclude_products: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Normalize set-like instrument settings before hashing.

    *keys* restricts the projection to the inner keys a given stage actually
    consumes (see :data:`_INSTRUMENT_KEYS_BY_STAGE`). An instrument whose
    settings are ALL irrelevant to that stage is dropped entirely rather than
    left as an empty mapping, so a config carrying only a chi-side knob hashes
    identically to one carrying no ``instruments`` at all -- which is what makes
    the existing diss output reusable instead of needlessly re-versioned.

    *exclude_products* does the same inside ``exclude``: only the entries whose
    probe is masked in one of those products are kept (see
    :data:`_EXCLUDE_PRODUCTS_BY_STAGE`), each with its original entry number.
    """
    normalized: dict[str, Any] = {}
    # Sort by str(key): instrument serials mix int (unquoted `465:`) and str keys.
    for key, settings in sorted((instruments or {}).items(), key=lambda kv: str(kv[0])):
        if not isinstance(settings, dict):
            normalized[str(key)] = settings
            continue
        item = dict(settings)
        if keys is not None:
            item = {k: v for k, v in item.items() if k in keys}
            if not item:
                continue
        excludes = item.get("exclude_shear_probes")
        if isinstance(excludes, list):
            item["exclude_shear_probes"] = sorted(str(probe) for probe in excludes)
        # Values normalized (30 == 30.0; quoted == unquoted timestamp; T1_dT1
        # == T1) and each entry carries its number, which names its attributes
        # in the product. reason stays in: it is written into the product too.
        if "exclude" in item:
            canon = [
                e.hash_form()
                for e in parse_exclusions(key, item["exclude"])
                if exclude_products is None or e.product in exclude_products
            ]
            if canon:
                item["exclude"] = canon
            else:
                # `exclude: []` (or null), or entries this stage cannot see,
                # change nothing here: hash it as absent, so the existing output
                # is reused instead of re-versioned.
                del item["exclude"]
                if keys is not None and not item:
                    continue
        # Key order must not change the hash, but the VALUES must: a different
        # tau is different chi, so it has to land in a new stage directory.
        tau_scale = item.get("fp07_tau_scale")
        if isinstance(tau_scale, dict):
            item["fp07_tau_scale"] = {
                str(k): float(v) for k, v in sorted(tau_scale.items(), key=lambda kv: str(kv[0]))
            }
        # A correction is hashed by its CONTENT, never its path: moving or
        # re-timestamping the file leaves chi where it is; refitting it to
        # different numbers re-versions chi (and chi_binned/chi_combo, which
        # embed the chi chain) but not epsilon.
        response = item.get("fp07_response")
        if response is not None:
            paths = [response] if isinstance(response, (str, os.PathLike)) else list(response)
            item["fp07_response"] = sorted(_fp07_response_digest(str(p)) for p in paths)
        # A bead's slow path: beads sorted, numbers as floats, spans as epoch
        # seconds (a quoted and an unquoted time hash alike, and no YAML date
        # reaches json). Empty hashes as absent, like `exclude: []`.
        if "fp07_slow_response" in item:
            from odas_tpw.chi.fp07_slow import canonical_for_hash

            slow = canonical_for_hash(
                item["fp07_slow_response"], where=f"instruments.{key}.fp07_slow_response"
            )
            if slow:
                item["fp07_slow_response"] = slow
            else:
                del item["fp07_slow_response"]
                if keys is not None and not item:
                    continue
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
    instruments_raw = config.get("instruments")
    instruments_diss = canonical_instruments_for_hash(
        instruments_raw,
        _INSTRUMENT_KEYS_BY_STAGE["diss"],
        _EXCLUDE_PRODUCTS_BY_STAGE["diss"],
    )
    instruments_chi = canonical_instruments_for_hash(
        instruments_raw,
        _INSTRUMENT_KEYS_BY_STAGE["chi"],
        _EXCLUDE_PRODUCTS_BY_STAGE["chi"],
    )

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
    diss_chain = [*profile_chain, ("instruments", instruments_diss)]
    # chi re-states `instruments` with the chi-side keys included, so changing
    # fp07_tau_scale re-versions chi WITHOUT re-versioning diss.
    chi_chain = [*profile_chain, ("instruments", instruments_chi), ("epsilon", eps_p)]
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
  max_gap: null           # [s] REQUIRED when enable is true. Where the two
                          # bracketing SOURCE samples are farther apart than
                          # this, the output is NaN instead of a straight line
                          # ruled across the hole.
                          #
                          # There is no safe default: the right limit is the
                          # sensor's own rate (~30 s for a 1 Hz CTD, minutes for
                          # a flight-state variable). Leaving it unset is an
                          # error, because interpolating across an unbounded gap
                          # manufactures data that ct, ctd, stratification,
                          # salinity:"measured" and epsilon.T_source all read as
                          # measurement.
                          #
                          # To deliberately keep the old interpolate-across-
                          # anything behaviour, write the string "unlimited".
                          # Per-variable max_gap wins.
  extrapolate: false      # NaN outside the source's own time range rather than
                          # holding the first/last value. Per-variable wins.

profiles:
  P_min: 0.5              # minimum pressure [dbar]
  W_min: null             # minimum fall rate [dbar/s] (null = auto: 0.3
                          # free-fall, 0.05 glide/horizontal)
  direction: "auto"       # profile direction: auto | up | down | glide |
                          # horizontal (auto = vehicle-resolved: down for a
                          # VMP, glide = up + down for a glider)
  min_duration: 7.0       # minimum profile duration [s]
  diagnostics: false      # RESERVED / not yet implemented (would write T1_raw,
                          # C_raw, W, cal attrs). Currently a no-op for profiles.

fp07:
  calibrate: true
  order: 2                # Steinhart-Hart polynomial order
  max_lag_seconds: 10     # max cross-correlation lag [s]
  reference: "JAC_T"      # reference temperature channel
  must_be_negative: true  # expect negative lag (falling VMP)
  min_corr: 0.5           # min |r| of the lag cross-correlation for a profile
                          # to count; a channel with none above it keeps the
                          # factory calibration
  reference_interval: null  # native sample interval [s] of a non-JAC
                          # reference (sets the FP07 low-pass to its Nyquist).
                          # null = taken from the hotel merge, else inferred
  allow_hotel_reference: false  # a reference from the HOTEL file (a glider
                          # CTD) is refused unless this is true. This fit is
                          # per .p FILE: a file the CTD did not cover keeps the
                          # factory coefficients while its neighbors are
                          # fitted. `fp07-cal` (deployment-pooled, pre-pipeline,
                          # then calibrate: false) avoids that; with true, read
                          # the run summary's FP07 notice.

ct:
  align: true
  T_name: "JAC_T"         # temperature channel for alignment
  C_name: "JAC_C"         # conductivity channel for alignment

# binning.method: altitude bins on height above the seabed instead of depth --
# the right vertical coordinate for bottom-boundary-layer work, and it needs
# [bottom] enabled so each profile carries a `bottom_depth` datum.

bottom:
  enable: false
  # vibration | fallrate | both.  "fallrate" takes the bottom from the collapse
  # of the fall rate in the pressure record instead of binned vibration std.
  # On SUNRISE 2022 Point Sur (134 descents) the vibration detector put the
  # bottom a median 1.28 m ABOVE where the instrument actually stopped, because
  # it reports the MEAN depth of a depth_window-wide bin; fall-rate put it at
  # 0.086 m and fired on 100% of descents against 97%.
  method: vibration
  depth_window: 4.0       # depth window for crash detection [m]
  depth_minimum: 10.0     # minimum depth to search [m]
  speed_factor: 0.3       # UNUSED (reserved; tuning has no effect)
  median_factor: 1.0      # UNUSED (reserved; tuning has no effect)
  vibration_frequency: 16 # UNUSED (reserved; tuning has no effect)
  vibration_factor: 4.0   # vibration std dev acceptance factor
  proximity_bins: 2       # accept a crash only this many bins from the deepest
  height: 0.15            # clearance ABOVE the detected bottom [m]; keeps the
                          # retained record pre-impact and above the
                          # deceleration (free fall ends ~0.09-0.11 m up)
  fallrate_fraction: 0.9  # fraction of terminal fall rate = end of free fall.
                          # Do NOT raise toward 1.0: at 0.98-0.99 ordinary
                          # fall-rate variability crosses it metres too high.
  fallrate_smooth_s: 0.05 # boxcar before differencing [s]; NOT a free
                          # parameter -- a wider one inflates the answer

top_trim:
  enable: false
  dz: 0.5                 # depth bin size [m]
  min_depth: 1.0          # minimum search depth [m]
  max_depth: 50.0         # maximum search depth [m]
  quantile: 0.6           # quantile of per-bin std taken as settled background
  noise_factor: 2.0       # std > noise_factor*background == still in prop wash

epsilon:
  # anchor: top | bottom.  "top" starts the first window at the top of the
  # section and marches down, DISCARDING the bottom remainder -- on a profiler
  # that crashes into the seabed that is the boundary layer. "bottom" pins the
  # deepest window's lower edge to the end of the section and marches up.
  anchor: top
  # A shorter dissipation window for the near-bottom zone. Epsilon belongs to a
  # window's CENTRE, so halving the window halves the distance from the seabed
  # to the deepest estimate. Only the DISSIPATION length may differ -- one
  # product carries one wavenumber axis, so fft_length is shared. The cost is
  # DOF: with fft_sec 0.5, a 2.0 s window averages 7 FFTs, a 1.0 s window 3.
  bbl_diss_sec: null      # null = no separate near-bottom windows
  bbl_overlap_sec: null   # null = half the BBL window
  bbl_extent_sec: null    # how far above the seabed the short windows apply
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
  despike_half_width: 0.04  # despike removal half-width [s], in (0, 0.5]: the
                          # despiker blanks half of it before each detected
                          # spike and all of it after, so 0.04 replaces ~60 ms
                          # per spike and 0.01 ~14 ms. Narrow it for a narrow
                          # electrical pulse; a probe strike needs the default.
  record_locked_shear: false  # subtract shear interference at fixed positions
                          # in every .p data record (electrical pickup the
                          # despiker misses below despike_thresh). Per profile,
                          # at the samples where the quiet-record mean of the
                          # high-passed shear is >= 5 standard errors, each
                          # record's local (+-16 record) mean is subtracted;
                          # nothing on clean shear. Re-versions diss and chi,
                          # not profiles.
                          # See the record_locked_* attrs on the diss files.
  salinity: null          # viscosity salinity: null = fixed 35; a number =
                          # that fixed PSU; "measured" = per-profile from
                          # C/T/P (TEOS-10, needs conductivity); "hotel" (or
                          # "hotel:<var>") = a hotel-injected salinity channel
                          # (default var "salinity") — for gliders/MRs with no
                          # onboard conductivity but a hotel CTD feed
  epsilon_minimum: 1.0e-13  # floor for small epsilon values
  T_source: null          # reference temperature for seawater properties
                          # (viscosity for epsilon; viscosity and kappa_T for
                          # chi — one knob serves both stages). null/"auto" =
                          # first plausible of T1, T2, ..., T, JAC_T (railed/
                          # implausible channels are skipped with a warning);
                          # a channel name (e.g. "T2", "JAC_T", or a hotel
                          # temperature channel) = use that channel (a QC
                          # failure warns but proceeds); a number = constant
                          # reference temperature [degC] (like ODAS
                          # constant_temp)
  spectral_qc: false      # ATOMIX-style per-probe cut. Default FALSE, while
                          # chi.spectral_qc defaults TRUE -- a deliberate,
                          # documented asymmetry: enabling this moves ~20% of
                          # probe-windows and it is not yet validated against
                          # the ATOMIX benchmark. Implements bits 1 and 16 of
                          # scor160.l4._compute_flags and NOTHING else; bits 2
                          # and 8 (despike fraction/passes) need diagnostics the
                          # diss product does not carry. Hence `spectral_qc`,
                          # not `atomix_qc` -- it cannot be the full set.
                          # When no probe survives a window, the window is
                          # DROPPED (NaN), matching rsi's _compute_epsi_final
                          # rather than chi.spectral_qc's never-drop fallback:
                          # a finite-but-wrong epsilon rescales Method-1 chi
                          # ~linearly while the chi fom stays ~1, so nothing
                          # downstream could reject it.
  FM_max: 1.15            # bit 1. The MAD-based FM -- NOT the variance-ratio
                          # `fom` that fom_max below thresholds. Different
                          # statistics, only weakly related.
  var_resolved_min: 0.5   # bit 16, applied to method == 0 ONLY. The gate is
                          # not optional: an ISR fit never integrates the
                          # dissipation range, so a low resolved fraction is
                          # expected rather than diagnostic, and ATOMIX exempts
                          # it. Ungated on one real corpus this rejected 3.51%
                          # instead of 0.18%. With no `method` variable the
                          # criterion is skipped, not guessed.
  pair_policy: keep_both  # bit 4, TWO-PROBE windows only. mk_epsilon_mean's
                          # outlier rule needs >= 3 probes -- with two, neither
                          # is identifiable as the outlier and always dropping
                          # the maximum biases epsilonMean low -- so it declines
                          # to act and ATOMIX bit 4 does. Both are defensible,
                          # so the choice is exposed:
                          #   keep_both  : perturb's behaviour (default)
                          #   drop_high  : ATOMIX's -- keeps the MINIMUM, which
                          #                makes a low junk probe authoritative
                          #   flag_only  : count it, mask nothing
                          # >= 3 probes stay with mk_epsilon_mean, so two rules
                          # never contest the same window.
  pair_limit: 2.7718585822512662
                          # The FULL coefficient of mean(sigma_ln) -- i.e.
                          # 1.96*sqrt(2), identical to scor160's
                          # DEFAULT_DISS_RATIO_LIMIT, so a value can be moved
                          # between the two unchanged. (mk_epsilon_mean spells
                          # the same threshold as 1.96*sqrt(2)*mu_sigma.)
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
                          # (the reference temperature for viscosity/kappa_T
                          # comes from epsilon.T_source — one knob serves
                          # both stages)
  fft_sec: 1.0            # FFT segment duration [s]; same sandwich
                          # constraints as epsilon.fft_sec above
  diss_sec: null          # dissipation window [s] (null = 4 * fft_sec)
  overlap_sec: null       # window overlap [s] (null = half the window)
  fft_length: null        # EXPERT override [samples]; wins over fft_sec
  diss_length: null       # EXPERT override [samples]; wins over diss_sec
  overlap: null           # EXPERT override [samples]; wins over overlap_sec
  anchor: top             # top | bottom, as epsilon.anchor. chi lays out its
                          # own windows, so a bottom-up product needs BOTH set
  bbl_diss_sec: null      # near-bottom windows, as epsilon.bbl_*; each must
  bbl_overlap_sec: null   # hold two FFT segments, and they take effect only
  bbl_extent_sec: null    # with anchor: bottom
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
  mixing_use_qc_fallback: false
                          # May a window kept ONLY by chi's soft spectral-QC
                          # fallback (chiQCFallback == 1 -- NO probe inside the
                          # fom band with enough K_max_ratio) feed K_T / Gamma /
                          # K_rho?  false (default) still reports chiMean for
                          # those windows but leaves the mixing coefficients
                          # NaN: a diffusivity built from a chi that failed
                          # every spectral test is a finite number with no
                          # evidence behind it, and before issue #180 F04
                          # nothing downstream could tell it apart.  true
                          # restores the old behaviour.
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
  # Through-water speed source. Computed AFTER the hotel merge, so
  # method "hotel" can consume a hotel-merged channel.
  method: "pressure"      # pressure | em | flight | constant | hotel
  # pressure : ODAS smoothed |dP/dt|. Correct for VMP. (default)
  # em       : use the U_EM channel from the .p file (MicroRider EM
  #            flowmeter). Errors out if U_EM is missing.
  # flight   : |W| / sin(|pitch|+aoa) (glide path steeper than pitch;
  #            ODAS convention), pitch axis auto-picked from
  #            Incl_X/Incl_Y by amplitude.
  # constant : use the scalar in `value`.
  # hotel    : use the hotel-merged channel named by `hotel_var` (map a
  #            source variable onto it via hotel.channels, e.g.
  #            hotel.channels.m_speed: "speed"). Errors out when the
  #            channel is missing or mostly non-finite — it never
  #            silently falls back to the speed_cutout floor.
  value: null             # m/s, only when method="constant"
  hotel_var: "speed"      # merged channel name, only when method="hotel"
  aoa_deg: 3.0            # angle of attack [deg], for method="flight"
  min_pitch_deg: 5.0      # flight: drop samples with |pitch| < this
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
  enable: true            # write N2 and dT/dz (background) to the diss and
                          # profile products, independent of eps/chi. Not on the
                          # CTD product (profile-only; CTD spans up+down).
  window: 2.0             # background vertical window [dbar] for the profile
                          # product (diss uses its dissipation window)
  salinity: null          # N2 salinity source: null = conductivity via TEOS-10
                          # (else 35 PSU); a number = that fixed PSU; "measured"
                          # = C/T/P (TEOS-10); "hotel" (or "hotel:<var>") = a
                          # hotel-injected salinity channel (default "salinity")
  sort: temperature       # how each window's parcels are reordered before its
                          # gradients are taken. temperature (the default since
                          # 2026-10) = conservative temperature alone, monotonic
                          # along the trend of a fit over five windows (as
                          # measured where that is unresolved): immune to
                          # salinity noise, which scrambles a density sort (dT/dz
                          # low, N2 high), but it also removes real
                          # density-compensated inversions. density =
                          # Thorpe-sorted to static stability (adiabatic
                          # leveling; the default before). none = as measured.
                          # With a per-sample salinity, N2 carries sort_kappa
                          # (density- over temperature-sorted dT/dz) and a
                          # warning fires below 0.8.
  dTdz_source: ct         # ct = ct.T_name. fp07 = the FP07 thermistors (in-situ
                          # calibrated when fp07.calibrate ran), slow-rate,
                          # temperature-sorted (or unsorted with sort: none), mean
                          # of per-probe slopes. The bead that measured chi then
                          # sets the gradient chi is normalized by, so a bead gain
                          # error cancels in K_T and Gamma. N2 stays on the CT pair.

parallel:
  jobs: 1

# Per-instrument overrides. The key matches the parent directory of each .p
# file (e.g. ARCTERX/VMP/SN465 -> SN465). Use this to suppress shear probes
# whose amplifier or sensor is known to be bad — the named probe is NaN'd
# out before mk_epsilon_mean, so it is excluded from the multi-probe
# epsilonMean and from chi Method 1 (which uses epsilonMean).
#
# exclude does the same to ONE probe -- a shear probe (sh1, sh2) or a
# thermistor (T1, T2) -- over a window: masked where P_min <= P_mean < P_max
# [dbar] and start <= t < stop [ISO-8601, UTC], each window's own stored
# P_mean and center time. An omitted bound is unbounded (no bounds = the whole
# deployment); reason is required and is written into the product with the
# number of windows masked. A shear entry reaches epsilonMean, chi Method 1,
# epsilon_paired, K_rho and Gamma; a thermistor entry reaches chiMean,
# chiLnSigma, K_T and Gamma, never epsilon. A probe the data does not carry is
# an error wherever the entry's window selects a window; the run summary
# reports what each entry masked.
#
# fp07_tau_scale multiplies the FP07 time constant per thermistor. tau is
# otherwise a single model (fp07_tau: 'lueck' for single_pole, 'goto' for
# double_pole) shared by every bead on the instrument, and two beads on one
# probe head can genuinely differ. The symptom is a persistent chi
# disagreement between T1 and T2 that VARIES with K_max_ratio (K_max/kB) and
# trends toward 1 as more of the Batchelor rolloff is resolved -- a flat gain
# error would be K_max_ratio-independent. Keys take the bead name (T1) or the
# gradient channel (T1_dT1); an unmatched key raises rather than being
# ignored. The value scales the MODEL, so its speed dependence survives.
# Caveat: this makes the two beads agree with each other, not with the truth --
# anchor the absolute level separately (e.g. where K_max_ratio > 1.2), and note
# fom cannot tell you which bead to trust (it can sit at ~1.000 for both).
#
# fp07_response is the measured alternative to fp07_tau_scale (never both on
# one instrument): `fp07-pair fit 'results/chi_00/*.nc' -o fp07pair/` measures
# each bead's response relative to its partner per time segment; list EVERY
# file it writes (including *_notapplied.nc). A file matching none is an error.
# See docs/fp07pair/runbook.md.
#
# fp07_slow_response divides a bead's slow parallel response path,
# |(1-beta) + beta/(1 + i 2 pi f tau_s)|^2, out of its observed and noise
# spectra (issue #252; T2005 on SN194 loses ~12% of its amplitude through a
# ~110 ms path). Keyed by BEAD SERIAL as the .p configuration names it, not by
# channel, with a mandatory span [first, last] (UTC, inclusive) tested
# against each file's start_time; a list gives several spans. A file without
# the bead is untouched; a file with it outside every span is left uncorrected
# and named in the run summary. Needs calibrated temperature (fp07.calibrate:
# true, or fp07-cal patched files): the path is relative to the DC level the
# calibration fixes. Never with fp07_tau_scale or fp07_response, which were
# both fitted on spectra that already contain the path. Each chi file records
# its own 0.25 -> 2 Hz T-pair ratio change beside the prediction and warns
# when they disagree by more than 0.10 (a swapped probe, a faded path).
instruments: {}
# Example:
# instruments:
#   SN465:
#     exclude_shear_probes: ["sh2"]
#   SN194:
#     fp07_tau_scale: {T1: 0.30, T2: 0.75}
#     exclude:
#       - probe: sh2       # pressure-related fault: bad in a depth window
#         P_min: 30        # dbar, inclusive
#         P_max: 70        # dbar, exclusive
#         start: "2017-10-09T00:00:00Z"   # inclusive
#         stop: "2017-10-24T00:00:00Z"    # exclusive
#         reason: "pressure-related fault, window opens ~10-09"
#       - probe: sh2       # ...then fails outright at every depth
#         start: "2017-10-24T00:00:00Z"
#         reason: "probe failed; e_2/e_1 = 0.37-0.82 at all depths"
#   SN428:
#     fp07_response:
#       - <CONFIG_DIR>/fp07pair/fp07pair_SN428_T2232-T2234_20220322T1450Z.nc
#   VMP:                   # .p files directly under VMP/; never on a key with
#     fp07_slow_response:  # fp07_tau_scale or fp07_response
#       T2005: {beta: 0.124, tau_s_ms: 110,
#               first: "2021-06-29T18:00:00Z", last: "2022-06-20T08:30:00Z"}
"""


def generate_template(path: str | Path) -> Path:
    """Write a fully-commented template configuration file."""
    path = Path(path)
    path.write_text(_TEMPLATE, encoding="utf-8")
    return path
