# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""YAML configuration for the ERDDAP -> hotel builder.

Six sections: ``server`` (where), ``fetch`` (what to download and how to chunk
it), ``qc`` (what to throw away), ``sensors`` (units and per-variable clocks),
``time`` (the common basis) and ``output``.  ``netcdf`` carries output metadata.

The schema is the one already documented in
``examples/rutgers_erddap/erddap-hotel.yaml``; that file is the reference, and
a test loads it to keep the two from drifting.

``sensors`` has user-defined keys (ERDDAP variable names), so it is a
dynamic-key section and its inner structure is checked in :func:`validate`
rather than by the generic unknown-key check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from odas_tpw.config_base import ConfigManager
from odas_tpw.dinkum.config import DEDUPE_METHODS, SENSOR_OPTION_KEYS

REFRESH_MODES = frozenset({"never", "incremental", "always"})
FORMATS = frozenset({"nc", "ncCF"})

DEFAULTS: dict[str, dict] = {
    "server": {
        "base_url": None,
        "protocol": "tabledap",
        "dataset_id": None,
        "timeout_s": 120.0,
        "retries": 3,
        "cache": "<CONFIG_DIR>/erddap_cache",
    },
    "fetch": {
        "variables": [],
        # The variable the server-side constraint applies to. NOT necessarily
        # the clock you build on: on Rutgers' raw trajectory datasets ERDDAP's
        # `time` axis has long_name "m_present_time", the flight computer's
        # clock. See time.base.
        "time_variable": "time",
        "time_min": None,
        "time_max": None,
        "chunk_days": 7.0,
        "constraints": [],
        "format": "nc",
        # never | incremental | always. A mission in progress is APPENDED to,
        # so the default picks up what landed since the last run instead of
        # re-fetching the whole trajectory.
        "refresh": "incremental",
        # A -raw-delayed dataset can REVISE recent rows, not only extend them,
        # so an incremental run refetches this much of the tail.
        "overlap_chunks": 1,
    },
    "qc": {
        # The output clock. Every variable is attributed to its own time
        # sensor and projected onto this one, exactly as dinkum-hotel does.
        "time_base": None,
        "time_min": None,
        "time_max": None,
        "dedupe": "mean",
        # Per-variable "0.0 means not sampled". Deliberately empty by default:
        # measured over 5.4M rows the timestamp bounds already remove every
        # such row, and applied blanket it would delete a real 0 degC sample.
        "drop_zero_as_fill": [],
        # variable -> [min, max]. Dynamic keys.
        "valid_range": {},
    },
    "sensors": {},
    "time": {
        "base": None,  # falls back to qc.time_base
        "max_gap": None,
        "extrapolate": False,
        "method": "linear",
    },
    "output": {"file": "<CONFIG_DIR>/hotel.nc"},
    "netcdf": {
        "title": None,
        "summary": None,
        "institution": None,
        "source": None,
        "platform": None,
        "comment": None,
        "creator_name": None,
        "creator_email": None,
    },
}

# Only `sensors` is a dynamic-key SECTION. `qc.valid_range` also has
# user-defined keys, but it is a nested dict under a fixed key, so the
# unknown-key check never reaches inside it; validate() checks its shape.
_manager = ConfigManager(DEFAULTS, dynamic_key_sections=frozenset({"sensors"}))

load_config = _manager.load_config
validate_config = _manager.validate_config
merge_config = _manager.merge_config


def validate(config: dict) -> None:
    """Check the cross-section invariants a schema check cannot express.

    Raises ``ValueError`` naming the key, because a config error found at
    request-build time surfaces as an opaque ERDDAP 400.
    """
    server = merge_config("server", config.get("server"))
    fetch = merge_config("fetch", config.get("fetch"))
    qc = merge_config("qc", config.get("qc"))
    time_cfg = merge_config("time", config.get("time"))

    for key in ("base_url", "dataset_id"):
        if not server.get(key):
            raise ValueError(f"server.{key}: required")
    if server.get("protocol") != "tabledap":
        raise ValueError(
            f"server.protocol={server.get('protocol')!r}: only 'tabledap' is "
            "implemented. griddap serves grids, not the row-per-sample tables "
            "a hotel file is built from."
        )
    if float(server.get("timeout_s") or 0) <= 0:
        raise ValueError("server.timeout_s: must be > 0 (urlopen defaults to no timeout)")

    variables = list(fetch.get("variables") or [])
    if not variables:
        raise ValueError(
            "fetch.variables: list at least one. Nothing not listed is "
            "downloaded; run `erddap-hotel info` to see what the dataset carries."
        )
    if fetch.get("refresh") not in REFRESH_MODES:
        raise ValueError(f"fetch.refresh={fetch.get('refresh')!r}: not in {sorted(REFRESH_MODES)}")
    if fetch.get("format") not in FORMATS:
        raise ValueError(f"fetch.format={fetch.get('format')!r}: not in {sorted(FORMATS)}")
    if float(fetch.get("chunk_days") or 0) <= 0:
        raise ValueError("fetch.chunk_days: must be > 0")
    if int(fetch.get("overlap_chunks") or 0) < 0:
        raise ValueError("fetch.overlap_chunks: must be >= 0")

    base = time_cfg.get("base") or qc.get("time_base")
    if not base:
        raise ValueError("time.base (or qc.time_base): required -- the output clock")
    if base not in variables:
        raise ValueError(
            f"time.base={base!r} is not in fetch.variables {variables}. "
            "The clock has to be downloaded like any other column."
        )
    if qc.get("dedupe") not in DEDUPE_METHODS:
        raise ValueError(f"qc.dedupe={qc.get('dedupe')!r}: not in {sorted(DEDUPE_METHODS)}")

    for name, rng in (qc.get("valid_range") or {}).items():
        if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
            raise ValueError(f"qc.valid_range[{name!r}]: must be [min, max]")
        if float(rng[0]) >= float(rng[1]):
            raise ValueError(f"qc.valid_range[{name!r}]: min ({rng[0]}) >= max ({rng[1]})")
        if name not in variables:
            raise ValueError(
                f"qc.valid_range[{name!r}]: not in fetch.variables, so it is never downloaded"
            )
    for name in qc.get("drop_zero_as_fill") or []:
        if name not in variables:
            raise ValueError(
                f"qc.drop_zero_as_fill: {name!r} is not in fetch.variables, so it is "
                "never downloaded"
            )

    gap = time_cfg.get("max_gap")
    if gap is not None and float(gap) <= 0:
        raise ValueError(f"time.max_gap={gap!r}: must be > 0 seconds (or null for no limit)")

    for src, opts in (config.get("sensors") or {}).items():
        if not isinstance(opts, dict):
            continue
        unknown = set(opts) - SENSOR_OPTION_KEYS
        if unknown:
            raise ValueError(
                f"sensors[{src!r}]: unknown option(s) {sorted(unknown)}. "
                f"Valid: {sorted(SENSOR_OPTION_KEYS)}"
            )
        if src not in variables:
            raise ValueError(
                f"sensors[{src!r}]: not in fetch.variables, so it is never downloaded"
            )


def to_builder_config(config: dict) -> dict[str, Any]:
    """Translate this schema into the one :func:`dinkum.build.build_hotel` takes.

    The two configs describe the same operation with different words -- the QC
    bounds live under ``qc`` here and under ``time`` there, the gap limit under
    ``time`` here and ``projection`` there.  Mapping once, in one place, is
    what lets the ERDDAP front end reuse the tested builder rather than
    reimplement projection and gap-blanking (design section 10.3).
    """
    fetch = merge_config("fetch", config.get("fetch"))
    qc = merge_config("qc", config.get("qc"))
    time_cfg = merge_config("time", config.get("time"))
    base = str(time_cfg.get("base") or qc.get("time_base"))

    sensors: dict[str, dict] = {}
    ranges = qc.get("valid_range") or {}
    for src in fetch.get("variables") or []:
        if src == base:
            continue
        opts = dict((config.get("sensors") or {}).get(src) or {})
        if src in ranges and "valid_min" not in opts and "valid_max" not in opts:
            opts["valid_min"], opts["valid_max"] = float(ranges[src][0]), float(ranges[src][1])
        opts.setdefault("time_sensor", base)
        sensors[src] = opts

    return {
        # `files` is inert: build_hotel never consults it when handed a dataset.
        "files": {},
        "time": {
            "base": base,
            "min_value": qc.get("time_min"),
            "max_value": qc.get("time_max"),
            "dedupe": qc.get("dedupe", "mean"),
        },
        "projection": {
            "method": time_cfg.get("method", "linear"),
            "max_gap": time_cfg.get("max_gap"),
            "extrapolate": bool(time_cfg.get("extrapolate", False)),
        },
        "sensors": sensors,
        "netcdf": merge_config("netcdf", config.get("netcdf")),
    }


_TEMPLATE = '''# erddap-hotel — build a perturb hotel file from an ERDDAP tabledap dataset.
#
#   erddap-hotel info   -c erddap-hotel.yaml   # look before you download
#   erddap-hotel fetch  -c erddap-hotel.yaml   # populate the cache only
#   erddap-hotel build  -c erddap-hotel.yaml   # fetch (cached) + QC -> hotel.nc
#   erddap-hotel verify -c erddap-hotel.yaml   # has the dataset changed?
#
# A worked, fully-annotated version for a Rutgers Slocum deployment is in
# examples/rutgers_erddap/. Paths may use <CONFIG_DIR>, which resolves to this
# file's own directory.

server:
  base_url: "https://example.org/erddap"
  protocol: "tabledap"
  dataset_id: "GLIDER-YYYYMMDDTHHMM-trajectory-raw-delayed"
  # Prefer a RAW product. A QC'd one typically renames every variable, carries
  # derived fields, exposes no flags — and if it has gap-filled, perturb's
  # hotel.max_gap is defeated at source: the gaps it exists to refuse are no
  # longer visible as gaps.
  timeout_s: 120         # urlopen defaults to NO timeout; a hung server would
                         # otherwise hang the build forever
  retries: 3             # connection errors, 429 and 5xx only — never a 400
  cache: "<CONFIG_DIR>/erddap_cache"

fetch:
  variables:             # pushed to the server: nothing not listed is fetched
    - sci_ctd41cp_timestamp
    - sci_water_temp
    - sci_water_cond
    - sci_water_pressure
  time_variable: "time"  # the variable the server-side CONSTRAINT applies to.
                         # Not necessarily the clock you build on — see
                         # qc.time_base.
  time_min: "2024-01-01T00:00:00Z"   # required: without it the whole dataset
  time_max: null                     # is requested. null max = up to now.
  chunk_days: 7          # one request per window: bounds each request, lets a
                         # partial failure retry cheaply, keeps cache entries small
  constraints:
    - "distinct()"       # server-side exact-duplicate removal. Halves the
                         # duplicate problem before download; does not solve it
                         # (rows sharing a stamp but differing in a NaN are not
                         # duplicates to the server), so qc below still runs.
  format: "nc"           # nc | ncCF
  refresh: "incremental" # never | incremental | always. A mission in progress
                         # is APPENDED to, so the default picks up what landed
                         # since the last run.
  overlap_chunks: 1      # refetch this much of the tail: a delayed-mode dataset
                         # can REVISE recent rows, not only extend them

qc:
  time_base: "sci_ctd41cp_timestamp"
  # Check what your server's `time` axis actually is before using it here. On
  # Rutgers raw trajectory datasets it has long_name "m_present_time" — the
  # FLIGHT computer's clock — and building on it smears every CTD sample by the
  # science-to-flight latency.
  time_min: "2024-01-01T00:00:00Z"   # epoch seconds or an ISO-8601 date
  time_max: "2025-01-01T00:00:00Z"
  # Bounds on the TIME variable do most of the work: a "did not sample this
  # cycle" row usually has an unusable timestamp as well as a 0.0 value. Set
  # BOTH — garbage stamps like 1.19e+103 need the upper bound.
  dedupe: "mean"         # mean | first | last, for rows sharing a stamp
  drop_zero_as_fill: []  # per-variable "0.0 means not sampled". Leave EMPTY
                         # unless measured otherwise: applied blanket it deletes
                         # real 0 degC polar water.
  valid_range:           # mirror the server's declared valid_min/valid_max.
                         # _FillValue (typically 9.96921e+36) is NOT masked on
                         # read — it arrives as an ordinary finite float, and
                         # any bound at all catches it.
    sci_water_temp: [-5.0, 40.0]
    sci_water_cond: [0.0, 10.0]
    sci_water_pressure: [0.0, 2000.0]

sensors:
  # UNIT CONVERSION HAPPENS HERE, ONCE. The perturb side must not re-apply it:
  # scaling in both places multiplies by ten twice.
  sci_water_temp:
    time_sensor: "sci_ctd41cp_timestamp"
    units: "degree_Celsius"
  sci_water_cond:
    time_sensor: "sci_ctd41cp_timestamp"
    units: "mS/cm"
    scale: 10.0          # if served as "S m-1"; check with `erddap-hotel info`
  sci_water_pressure:
    time_sensor: "sci_ctd41cp_timestamp"
    units: "dbar"
    scale: 10.0          # if served as "bar"; check with `erddap-hotel info`

time:
  base: "sci_ctd41cp_timestamp"
  method: "linear"
  max_gap: 30.0          # [s] NaN the output where bracketing samples are
                         # farther apart than this, rather than ruling a
                         # straight line across a dropout
  extrapolate: false

output:
  file: "<CONFIG_DIR>/hotel.nc"

netcdf:
  title: null
  summary: null
  institution: null
'''


def generate_template(path: str | Path) -> Path:
    """Write a fully-commented template configuration file."""
    path = Path(path)
    # encoding is explicit: the default is the platform codepage, so on
    # Windows a non-ASCII character (the template has an em dash) is written
    # as cp1252 and load_config, which reads utf-8, then raises
    # UnicodeDecodeError on a file we just generated.
    path.write_text(_TEMPLATE, encoding="utf-8")
    return path
