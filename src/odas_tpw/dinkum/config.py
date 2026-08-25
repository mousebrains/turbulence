# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""YAML configuration for the Dinkum -> hotel converter.

Four sections: ``files`` (what to read, with what), ``time`` (the common time
basis and its sanity range), ``projection`` (how sensors are put onto that
basis), and ``sensors`` (what to extract). ``netcdf`` carries output metadata.

``sensors`` has user-defined keys (the Slocum sensor names), so it is declared
as a dynamic-key section and its inner structure is validated in
:func:`normalize_sensors` instead of by the generic unknown-key check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from odas_tpw.config_base import ConfigManager

# Interpolation kinds accepted by projection.method and per-sensor method.
# Deliberately the same vocabulary as odas_tpw.perturb.hotel, so a method name
# means the same thing on both sides of the hotel file.
INTERP_KINDS = frozenset(
    {
        "pchip",
        "linear",
        "nearest",
        "previous",
        "next",
        "zero",
        "slinear",
        "quadratic",
        "cubic",
    }
)

# Per-sensor option keys. Mirrors perturb.hotel's _CHANNEL_OPTION_KEYS where the
# meaning matches (name/scale/offset/units), and adds the ones that only make
# sense while building (time_sensor, method, valid_min/max, max_gap).
SENSOR_OPTION_KEYS = frozenset(
    {
        "name",
        "time_sensor",
        "method",
        "transform",
        "scale",
        "offset",
        "units",
        "long_name",
        "valid_min",
        "valid_max",
        "max_gap",
        "dedupe",
    }
)

# Non-linear per-sensor conversions, applied to the SOURCE samples before the
# range check and before interpolation. `scale`/`offset` cannot express these,
# and — unlike an affine map — they do not commute with interpolation, so they
# cannot be deferred to the output the way scale/offset is.
#
# nmea_degrees
#     Slocum reports every geographic coordinate in NMEA ``ddmm.mmmm``:
#     ``m_lat = 2015.61159`` is 20 deg 15.61159 min = 20.260193 deg N, NOT
#     20.1561 deg. There is no scale factor that converts it. Interpolating
#     the raw form is also wrong: across a whole minute the raw value steps by
#     40.02 (2059.99 -> 2100.01) where the true position moved 0.0003 deg, so
#     any sample straddling the boundary would be ruled straight across a
#     cliff. Convert first, then interpolate.
TRANSFORMS = frozenset({"nmea_degrees"})

# Dedupe strategies accepted by time.dedupe and per-sensor dedupe.
DEDUPE_METHODS = frozenset({"mean", "first", "last"})

# Step-like projections: a duplicate-timestamp group must collapse to the
# value that was actually in force, never to a mean the sensor never reported.
_STATE_METHODS = frozenset({"previous", "next", "nearest", "zero"})

DEFAULTS: dict[str, dict] = {
    "files": {
        # Directory the patterns are relative to. The <CONFIG_DIR> token
        # resolves to the directory holding the YAML, as in perturb.
        "root": "<CONFIG_DIR>",
        # Globs, in order. Flight and science files are normally both listed:
        # the flight file carries m_present_time / pitch / speed, the science
        # file carries the CTD.
        "patterns": ["*.[de]bd", "*.[de]cd"],
        # Sensor-list cache directory. Slocum files reference their sensor
        # list by hash; a file whose hash is not cached CANNOT be decoded and
        # is silently skipped by both readers. Effectively required.
        "cache": None,
        "output": "<CONFIG_DIR>/hotel.nc",
        "reader": "auto",  # auto | xarray-dbd | dbd2netcdf | netcdf
        "skip_first_record": True,
        "repair": False,
        # Glob patterns (relative to root) to leave out. A deployment usually
        # has one or two files the glider never finished writing.
        "exclude": [],
        # How many files the reader may fail to decode before the build gives
        # up. 0 is the right default -- a cache miss silently drops a whole
        # mission segment, and a short hotel file looks exactly like a
        # complete one. Raise it only for files you have LOOKED AT and know
        # are junk.
        "max_skipped": 0,
    },
    "time": {
        # The common time basis every sensor is projected onto. Commonly
        # m_present_time (flight computer), sci_m_present_time (science
        # computer), or sci_ctd41cp_timestamp (arrival of the CTD's print).
        "base": "sci_ctd41cp_timestamp",
        # Validity window for ANY time sensor (the base and per-sensor
        # overrides alike). Each accepts epoch seconds or an ISO-8601 date
        # string. null falls back to: min 100 s, max now + 365 days.
        "min_value": None,
        "max_value": None,
        # How to collapse samples sharing one timestamp. Slocum repeats the
        # last CTD timestamp on rows the CTD did not refresh, so duplicates
        # are routine and must go: the interpolators either raise (pchip) or
        # produce infinite slopes (linear). Per-sensor `dedupe` overrides;
        # sensors projected with previous/next/nearest/zero default to
        # "last" (the state actually in force), not the global value.
        "dedupe": "mean",  # mean | first | last
    },
    "projection": {
        "method": "linear",  # default; per-sensor `method` wins
        # NaN the output wherever the bracketing source samples are further
        # apart than this [s] — i.e. do not draw a straight line across a
        # dropout and present it as data. null = no gap limit.
        "max_gap": None,
        # Outside a sensor's own first/last sample the value is NaN. Whether
        # perturb holds or NaNs the edges at merge time is its own decision
        # (hotel.max_gap / hotel.extrapolate; see #150); holding here too
        # would bake a constant into the archive.
        "extrapolate": False,
    },
    # Dynamic keys: Slocum sensor names -> options (see SENSOR_OPTION_KEYS).
    "sensors": {},
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

_manager = ConfigManager(DEFAULTS, dynamic_key_sections=frozenset({"sensors"}))

load_config = _manager.load_config
validate_config = _manager.validate_config
merge_config = _manager.merge_config


def normalize_sensors(sensors_cfg: dict | None, time_base: str) -> dict[str, dict]:
    """Validate and fill in the ``sensors`` block.

    Each value may be ``None``/``{}`` (all defaults), a string (rename), or a
    dict of :data:`SENSOR_OPTION_KEYS`. Returns source_name -> fully-specified
    option dict with ``name`` and ``time_sensor`` always present.

    Per-sensor ``max_gap: null`` means *inherit the global* — the key is
    dropped so :func:`build_hotel` falls back to ``projection.max_gap``.
    Per-sensor ``dedupe`` defaults to ``"last"`` for the step-like methods
    (previous/next/nearest/zero) and to the global ``time.dedupe`` otherwise;
    an explicit value always wins.

    Raises ``ValueError`` on an unknown option, a bad interpolation method, a
    bad dedupe, a non-positive ``max_gap``, an inverted
    ``valid_min``/``valid_max``, or an output name that collides with
    another sensor or with the time base.
    """
    if not sensors_cfg:
        raise ValueError(
            "sensors: is empty — list at least one Slocum sensor to extract. "
            "Run `dinkum-hotel sensors <files>` to see what the files carry."
        )
    out: dict[str, dict] = {}
    for src, val in sensors_cfg.items():
        src = str(src)
        if val is None or val == {}:
            opts: dict[str, Any] = {}
        elif isinstance(val, str):
            opts = {"name": val}
        elif isinstance(val, dict):
            unknown = set(val) - SENSOR_OPTION_KEYS
            if unknown:
                raise ValueError(
                    f"sensors[{src!r}]: unknown option(s) {sorted(unknown)}. "
                    f"Valid: {sorted(SENSOR_OPTION_KEYS)}"
                )
            opts = dict(val)
        else:
            raise ValueError(
                f"sensors[{src!r}]: must be a string, mapping, or null; got {type(val).__name__}"
            )

        method = opts.get("method")
        if method is not None and method not in INTERP_KINDS:
            raise ValueError(f"sensors[{src!r}].method={method!r}: not in {sorted(INTERP_KINDS)}")
        xform = opts.get("transform")
        if xform is not None and xform not in TRANSFORMS:
            raise ValueError(
                f"sensors[{src!r}].transform={xform!r}: not in {sorted(TRANSFORMS)}"
            )
        if "max_gap" in opts and opts["max_gap"] is None:
            del opts["max_gap"]  # null -> inherit projection.max_gap
        gap = opts.get("max_gap")
        if gap is not None and not (float(gap) > 0):
            raise ValueError(f"sensors[{src!r}].max_gap={gap!r}: must be > 0 seconds")
        dd = opts.get("dedupe")
        if dd is not None and dd not in DEDUPE_METHODS:
            raise ValueError(f"sensors[{src!r}].dedupe={dd!r}: not in {sorted(DEDUPE_METHODS)}")
        if dd is None and method in _STATE_METHODS:
            opts["dedupe"] = "last"
        vmin, vmax = opts.get("valid_min"), opts.get("valid_max")
        if vmin is not None and vmax is not None and float(vmin) >= float(vmax):
            raise ValueError(f"sensors[{src!r}]: valid_min ({vmin}) >= valid_max ({vmax})")

        opts.setdefault("name", src)
        opts.setdefault("time_sensor", time_base)
        out[src] = opts

    # Two sensors resolving to one output name is not a rename, it is a
    # collision: build_hotel stores each result as data_vars[out_name], so the
    # later physical sensor silently overwrites the earlier one while the log
    # and the provenance still list both. Refuse rather than pick a winner.
    # The time base is seeded in: it becomes the output's time coordinate, so
    # a sensor named after it would collide inside xarray at write time.
    by_out: dict[str, list[str]] = {time_base: [f"time.base={time_base!r}"]}
    for src, opts in out.items():
        by_out.setdefault(str(opts["name"]), []).append(src)
    collisions = {name: srcs for name, srcs in by_out.items() if len(srcs) > 1}
    if collisions:
        detail = "; ".join(
            f"{name!r} <- {sorted(srcs)}" for name, srcs in sorted(collisions.items())
        )
        raise ValueError(
            f"sensors: two or more sensors resolve to the same output name "
            f"({detail}). Each output variable can hold one sensor, and the "
            f"time base's name is the time coordinate; give them distinct "
            f"`name` values."
        )
    return out


def required_sensor_names(sensors: dict[str, dict], time_base: str) -> list[str]:
    """Every sensor the read must include: data sensors plus all time sensors.

    Restricting the read to the data sensors alone would drop the time sensor
    a channel is attributed to, which is unrecoverable later.
    """
    names = set(sensors) | {time_base}
    names |= {str(o["time_sensor"]) for o in sensors.values()}
    return sorted(names)


_TEMPLATE = """\
# dinkum-hotel configuration — Slocum Dinkum Binary Data -> perturb hotel file.
#
# Every sensor listed below is projected onto ONE common time basis
# (time.base), and the result is written as a hotel NetCDF that
# perturb's `hotel:` block reads directly. The output time variable keeps
# the name of the base sensor, so the perturb side reads:
#
#     hotel:
#       file: "hotel.nc"
#       time_column: "sci_ctd41cp_timestamp"
#       time_format: "epoch"
#
# Paths may use <CONFIG_DIR>, which resolves to this file's own directory.

files:
  root: "<CONFIG_DIR>"      # directory the patterns below are relative to
  patterns:                 # globs, in order; flight and science both
    - "*.[de]bd"            #   uncompressed dbd/ebd
    - "*.[de]cd"            #   LZ4-compressed dcd/ecd
  cache: null               # sensor-list cache directory. NOT optional in
                            # practice: a Slocum file whose sensor-list hash
                            # is not in the cache cannot be decoded and is
                            # skipped, so a wrong/empty cache yields "decoded
                            # 0 records".
  output: "<CONFIG_DIR>/hotel.nc"
  reader: "auto"            # auto | xarray-dbd | dbd2netcdf | netcdf
                            # auto: NetCDF inputs -> netcdf; else xarray-dbd
                            # if importable; else the dbd2netCDF binary.
  skip_first_record: true   # the first record of a file is routinely partial
  repair: false             # attempt recovery of corrupt records

time:
  # The common time basis. The three that matter on a Slocum:
  #   m_present_time         flight computer clock (every flight record)
  #   sci_m_present_time     science computer clock (every science record)
  #   sci_ctd41cp_timestamp  when the CTD's print ARRIVED at the science
  #                          computer — the right basis for CTD channels
  base: "sci_ctd41cp_timestamp"

  # Valid range for any time sensor. Each accepts epoch seconds or an
  # ISO-8601 date; null falls back to 100 s and (now + 365 days).
  # Pin both for a reproducible rerun — the now-relative default moves.
  min_value: null           # e.g. 100  or  "2025-01-15T00:00:00Z"
  max_value: null           # e.g.      "2025-04-01T00:00:00Z"

  dedupe: "mean"            # mean | first | last — collapse samples sharing
                            # one timestamp. Slocum repeats the last CTD
                            # timestamp on rows the CTD did not refresh, and
                            # duplicates make pchip raise / linear go infinite.
                            # Sensors projected with previous/next/nearest/
                            # zero use "last" unless they say otherwise.

projection:
  method: "linear"          # default projection; per-sensor `method` wins.
                            # linear | pchip | nearest | previous | next |
                            # zero | slinear | quadratic | cubic
                            # Use "previous" (zero-order hold) for flight
                            # state / discrete flags, where interpolating
                            # between states invents values that never held.
  max_gap: null             # [s] NaN the output where the bracketing source
                            # samples are farther apart than this, instead of
                            # ruling a straight line across a dropout
  extrapolate: false        # outside a sensor's own range -> NaN

sensors:
  # source_name:            # include with all defaults
  # source_name: "new_name" # rename only
  # source_name:            # full form
  #   name: "new_name"      #   output variable name (default: same)
  #   time_sensor: "..."    #   which clock stamps THIS sensor
  #                         #   (default: time.base)
  #   method: "previous"    #   projection override
  #   transform: nmea_degrees
  #                         #   non-linear conversion applied to the SOURCE
  #                         #   samples, before the range check and before
  #                         #   interpolation. Only "nmea_degrees" so far:
  #                         #   Slocum ddmm.mmmm -> decimal degrees, which
  #                         #   `scale` cannot express (m_lat 2015.61159 is
  #                         #   20.260193 degN, not 20.1561) and which must
  #                         #   NOT be interpolated in its raw form (the raw
  #                         #   value steps 40.02 across a whole minute).
  #                         #   Use it on m_lat/m_lon, m_gps_lat/m_gps_lon,
  #                         #   c_wpt_lat/c_wpt_lon.
  #   scale: 10.0           #   value = raw * scale + offset, applied to the
  #                         #   OUTPUT (affine, so it commutes with interp)
  #   offset: 0.0
  #   units: "dbar"         #   CF units for the output
  #   long_name: "..."
  #   valid_min: -5.0       #   values outside -> NaN BEFORE projecting.
  #   valid_max: 45.0       #   In TRANSFORMED units when `transform` is set,
  #                         #   but before `scale`/`offset`.
  #   max_gap: 30.0         #   per-sensor gap limit [s] (null = global)
  #   dedupe: "last"        #   mean | first | last (default: "last" for
  #                         #   previous/next/nearest/zero, else time.dedupe)

  sci_water_temp:
    units: "degree_Celsius"
    valid_min: -5.0
    valid_max: 45.0
  sci_water_cond:
    scale: 10.0             # Slocum reports S/m; perturb/gsw want mS/cm
    units: "mS/cm"
    valid_min: 0.0
    valid_max: 70.0
  sci_water_pressure:
    scale: 10.0             # Slocum reports bar; everything else wants dbar
    units: "dbar"
    valid_min: -2.0
    valid_max: 2000.0

  # Flight channels ride the flight clock, so name it explicitly. "previous"
  # holds each commanded/state value until it actually changes.
  # m_lat:
  #   name: "lat"
  #   time_sensor: "m_present_time"
  #   transform: nmea_degrees   # ddmm.mmmm -> decimal degrees
  #   units: "degrees_north"
  #   valid_min: -90.0          # degrees: the range check sees the transform
  #   valid_max: 90.0
  # m_lon:
  #   name: "lon"
  #   time_sensor: "m_present_time"
  #   transform: nmea_degrees
  #   units: "degrees_east"
  #   valid_min: -180.0
  #   valid_max: 180.0
  # m_pitch:
  #   units: "rad"
  #   time_sensor: "m_present_time"
  # m_speed:
  #   name: "speed"
  #   units: "m s-1"
  #   time_sensor: "m_present_time"

netcdf:
  title: null
  summary: null
  institution: null
  source: null
  platform: null
  comment: null
  creator_name: null
  creator_email: null
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
