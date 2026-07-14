# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""Inventory microstructure sensors across a tree of Rockland ``.p`` files.

Point this at the top of a directory tree; it finds every ``.p`` file below it
and, for each sensor serial number, summarizes:

  * the date range over which the sensor was used,
  * the number of files it appears in,
  * the platform(s) it was mounted on (vehicle type + instrument SN), and
  * whether its calibration parameters changed across those uses.

Two sensor kinds are understood today and the registry is extensible:

  * **shear** probes  — config channel ``type`` ``shear`` / ``xmp_shear``
    (tracks adc_fs, adc_bits, diff_gain, sens, cal_date), and
  * **fp07** fast thermistors — config channel ``type`` ``therm``
    (tracks adc_fs, adc_bits, a, b, g, e_b, beta_1, beta_2, t_0, cal_date).

Only the 128-byte binary header and the embedded INI configuration string of
each file are read (never the data records), so the scan is fast even over
large trees.  The header supplies the timestamp; the config supplies the sensor
channels and their per-sensor parameters.

Sensors are grouped by serial number: a probe that moves between ports
(sh1↔sh2) or platforms is still one entry.  A caveat follows from that — if a
config leaves the serial number blank or uses a placeholder shared by physically
distinct sensors (e.g. FP07 ``sn = T`` on both T1 and T2), those sensors are
merged under one SN and their differing calibrations show up as "CHANGED".

Usage
-----
    python -m odas_tpw.rsi.sensor_inventory /path/to/tree [more paths ...]
    python -m odas_tpw.rsi.sensor_inventory VMP/ --shear --csv probes.csv
    python -m odas_tpw.rsi.sensor_inventory VMP/ --all --verbose
"""

from __future__ import annotations

import argparse
import csv
import glob as globmod
import math
import re
import struct
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import TextIO

from odas_tpw.rsi.p_file import (
    HEADER_BYTES,
    _detect_endian,
    _parse_header,
    parse_config,
)


@dataclass(frozen=True)
class SensorKind:
    """A recognizable microstructure sensor family.

    Attributes
    ----------
    key : str
        Short identifier / CLI selector (e.g. ``"shear"``).
    label : str
        Human-readable singular label (e.g. ``"Shear probe"``).
    types : frozenset[str]
        Config channel ``type`` values (lower-case) that denote this sensor.
    params : tuple[str, ...]
        Per-sensor calibration parameters tracked for change, in display order.
    """

    key: str
    label: str
    types: frozenset[str]
    params: tuple[str, ...]


# Registry of known sensor kinds.  Add new kinds here; --all covers every entry.
SENSOR_KINDS: dict[str, SensorKind] = {
    "shear": SensorKind(
        key="shear",
        label="Shear probe",
        types=frozenset({"shear", "xmp_shear"}),
        params=("adc_fs", "adc_bits", "diff_gain", "sens", "cal_date"),
    ),
    "fp07": SensorKind(
        key="fp07",
        label="FP07 thermistor",
        types=frozenset({"therm"}),
        params=("adc_fs", "adc_bits", "a", "b", "g", "e_b", "beta_1", "beta_2", "t_0", "cal_date"),
    ),
}

# Pre-emphasized derivative channels (T1_dT1, P_dP, ...) share a base channel's
# ``type`` but carry only diff_gain, not the sensor's SN / calibration.  They are
# the same physical sensor, so exclude them from the inventory.  Mirrors the
# X_dX pattern matched in p_file.PFile._apply_deconvolution.
_PREEMPH_RE = re.compile(r"^(\w+)_d\1$")


def resolve_kinds(shear: bool = False, fp07: bool = False, want_all: bool = False) -> list[str]:
    """Map the --shear/--fp07/--all flags to an ordered list of kind keys.

    With nothing selected (or --all), every registered kind is returned so the
    tool inventories everything by default.
    """
    if want_all or not (shear or fp07):
        return list(SENSOR_KINDS)
    kinds = []
    if shear:
        kinds.append("shear")
    if fp07:
        kinds.append("fp07")
    return kinds


# ---------------------------------------------------------------------------
# Lightweight header + config reader (no data records)
# ---------------------------------------------------------------------------


def _read_header_and_config(path: Path) -> tuple[dict, dict]:
    """Return (parsed header, parsed config) reading only record 0 of *path*."""
    with open(path, "rb") as f:
        raw_hdr = f.read(HEADER_BYTES)
        if len(raw_hdr) < HEADER_BYTES:
            raise ValueError(f"{path.name}: file too small for header")
        with warnings.catch_warnings():
            # Over a large tree, MR/glider files whose clock had not yet been set
            # at power-up carry a zeroed header (endian flag 0), which makes
            # _detect_endian warn on potentially thousands of files.  The warning
            # is irrelevant to this tool -- serial numbers come from the ASCII
            # config block, which is endian-independent -- so silence it to keep
            # the run readable.
            warnings.simplefilter("ignore")
            endian = _detect_endian(raw_hdr, path)
            header = _parse_header(raw_hdr, endian)

        header_size = int(header["header_size"])
        config_size = int(header["config_size"])
        if header_size < HEADER_BYTES:
            raise ValueError(f"{path.name}: invalid header_size={header_size}")
        if config_size < 0:
            raise ValueError(f"{path.name}: invalid config_size={config_size}")

        f.seek(header_size)
        config_str = f.read(config_size).decode("ascii", errors="replace")
    return header, parse_config(config_str)


def _start_time_utc(header: dict, config: dict) -> datetime | None:
    """Recording start time in UTC, or None if the header carries no valid clock.

    Mirrors PFile._read: the header timestamp marks the END of record 0, so the
    ODAS-consistent data start time subtracts the record duration (``recsize``,
    default 1.0 s).  Absolute alignment is immaterial for a date-range summary,
    but staying consistent with the rest of the codebase avoids surprises.

    An MR/glider that powers up before the host sets its clock writes a zeroed
    (or garbage) date — year 0, month 36123, etc.  Those still carry a valid
    ASCII config with real probe serial numbers, so rather than discard the
    file we return None here and let the caller inventory it without a date.
    """
    h = header
    tz_min = h["timezone_min"] - 2**16 if h["timezone_min"] >= 2**15 else h["timezone_min"]
    root = config.get("root", {}) if isinstance(config, dict) else {}
    raw = root.get("recsize", root.get("recordduration"))
    try:
        recsize = float(raw) if raw is not None else 1.0
    except (TypeError, ValueError):
        recsize = 1.0
    if not math.isfinite(recsize):  # a corrupt 'recsize = 1e999' -> inf -> timedelta OverflowError
        recsize = 1.0
    # Everything that can raise on a garbage header must stay inside the guard so
    # a single bad file returns None (undated) rather than aborting the batch:
    # out-of-range date fields and tz -> ValueError; a datetime at the floor of
    # the range minus recsize, or a huge recsize -> OverflowError.
    try:
        st = datetime(
            h["year"],
            h["month"],
            h["day"],
            h["hour"],
            h["minute"],
            h["second"],
            h["millisecond"] * 1000,
            tzinfo=timezone(timedelta(minutes=tz_min)),
        )
        st -= timedelta(seconds=recsize)
        return st.astimezone(UTC)
    except (ValueError, OverflowError):
        return None


@dataclass
class SensorUse:
    """One sensor channel as configured in one .p file."""

    kind: str  # SensorKind.key, e.g. "shear"
    path: Path
    start_time: datetime | None  # None when the file's clock was unset
    vehicle: str
    platform_sn: str
    channel: str  # e.g. "sh1", "T1"
    sensor_sn: str  # e.g. "M2732"
    params: dict[str, str]  # kind.params -> raw string value


def _platform_from_instrument(inst: dict) -> tuple[str, str]:
    """(vehicle, platform_sn) from an instrument_info block, tolerant of gaps.

    Strip BEFORE falling back, so a whitespace-only field (truthy but empty)
    still degrades to model/"?" instead of collapsing to "" — otherwise the
    ("", sn) platform key would split from the ("?", sn) of a genuinely-missing
    field.  A microrider-on-glider leaves vehicle set ("slocum_glider"); a file
    whose host had not populated instrument_info yet yields ("?", "?").
    """
    vehicle = inst.get("vehicle", "").strip() or inst.get("model", "").strip() or "?"
    platform_sn = inst.get("sn", "").strip() or "?"
    return vehicle, platform_sn


def scan_file(path: Path, kinds: list[str]) -> list[SensorUse]:
    """Extract every sensor use of the requested *kinds* from a single .p file.

    Returns one :class:`SensorUse` per matching (non pre-emphasis) channel.
    Raises on unreadable/corrupt headers so the caller can report the file.
    """
    header, config = _read_header_and_config(path)
    st = _start_time_utc(header, config)

    vehicle, platform_sn = _platform_from_instrument(config.get("instrument_info", {}))

    # Channel type -> owning kind, for the requested kinds only.
    type_to_kind: dict[str, SensorKind] = {}
    for key in kinds:
        for ctype in SENSOR_KINDS[key].types:
            type_to_kind[ctype] = SENSOR_KINDS[key]

    uses: list[SensorUse] = []
    for ch in config.get("channels", []):
        name = (ch.get("name") or "").strip()
        if _PREEMPH_RE.match(name):
            continue
        kind = type_to_kind.get(ch.get("type", "").strip().lower())
        if kind is None:
            continue
        sensor_sn = (ch.get("sn") or "").strip() or "(no SN)"
        uses.append(
            SensorUse(
                kind=kind.key,
                path=path,
                start_time=st,
                vehicle=vehicle,
                platform_sn=platform_sn,
                channel=name or "?",
                sensor_sn=sensor_sn,
                params={p: (ch.get(p) or "").strip() for p in kind.params},
            )
        )
    return uses


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _norm(key: str, value: str):
    """Normalize a parameter value for change detection.

    Numeric params compare by value so cosmetic formatting differences
    ("0.933" vs "0.9330") are not mistaken for a real change; non-numeric or
    unparseable values (e.g. cal_date) fall back to the stripped string.

    Non-finite floats from a corrupt config (nan/inf) fall back to the exact
    string too — otherwise two identical "nan" values would compare unequal
    (a false CHANGED) and two different overflowing values ("1e400"/"1e999",
    both inf) would merge (a false constant).
    """
    if value == "":
        return ""
    try:
        f = float(value)
    except ValueError:
        return value
    if not math.isfinite(f):
        return value
    # int() only after the finite check, so int(float("inf")) can't raise
    # OverflowError here (this runs in build_inventory, which has no guard).
    return int(f) if key == "adc_bits" else f


def _new_value_slot() -> dict:
    """Per-distinct-value accumulator: display text, files, and date range."""
    return {"display": "", "files": set(), "first": None, "last": None}


def _new_param_map() -> defaultdict:
    """param key -> (normalized value -> value slot), created lazily."""
    return defaultdict(lambda: defaultdict(_new_value_slot))


@dataclass
class SensorAgg:
    """Accumulated record for one sensor serial number."""

    files: set[Path] = field(default_factory=set)
    first: datetime | None = None
    last: datetime | None = None
    # platform (vehicle, sn) -> set of channel names it was wired to
    platforms: dict[tuple[str, str], set[str]] = field(default_factory=lambda: defaultdict(set))
    # param key -> normalized value -> {display, files, first, last}
    params: defaultdict = field(default_factory=_new_param_map)

    def add(self, use: SensorUse) -> None:
        self.files.add(use.path)
        st = use.start_time
        if st is not None:
            if self.first is None or st < self.first:
                self.first = st
            if self.last is None or st > self.last:
                self.last = st
        self.platforms[(use.vehicle, use.platform_sn)].add(use.channel)
        for key, raw in use.params.items():
            slot = self.params[key][_norm(key, raw)]
            if not slot["display"]:  # first-seen wins → deterministic (files sorted)
                slot["display"] = raw if raw != "" else "(blank)"
            slot["files"].add(use.path)
            if st is not None:
                if slot["first"] is None or st < slot["first"]:
                    slot["first"] = st
                if slot["last"] is None or st > slot["last"]:
                    slot["last"] = st


def build_inventory(uses: list[SensorUse]) -> dict[str, dict[str, SensorAgg]]:
    """Group uses into ``{kind_key: {serial_number: SensorAgg}}``."""
    inventory: dict[str, dict[str, SensorAgg]] = defaultdict(lambda: defaultdict(SensorAgg))
    for use in uses:
        inventory[use.kind][use.sensor_sn].add(use)
    return inventory


def collect_uses(
    files: list[Path], kinds: list[str]
) -> tuple[list[SensorUse], list[tuple[Path, str]], int]:
    """Scan *files*, returning (uses, errors, n_files_with_no_matching_sensor)."""
    uses: list[SensorUse] = []
    errors: list[tuple[Path, str]] = []
    n_no_sensor = 0
    for fp in files:
        try:
            found = scan_file(fp, kinds)
        except (OSError, ValueError, KeyError, OverflowError, struct.error) as exc:
            errors.append((fp, str(exc)))
            continue
        if not found:
            n_no_sensor += 1
        uses.extend(found)
    return uses, errors, n_no_sensor


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def _collect_from(p: Path, found: set[Path]) -> None:
    """Add every .p/.P file at or under *p* (a directory or a single file)."""
    if p.is_dir():
        for pat in ("*.p", "*.P"):
            found.update(q for q in p.rglob(pat) if q.is_file())
    elif p.is_file():
        found.add(p)


def iter_pfiles(paths: list[Path]) -> list[Path]:
    """All ``.p``/``.P`` files under the given files/directories, de-duplicated.

    Each argument may be a directory (scanned recursively), a single file, or a
    glob pattern.  A path that does not exist is passed to ``glob`` so that a
    quoted or shell-unexpanded pattern (e.g. ``'VMP/*.p'``) works the same way it
    does for the other rsi-tpw subcommands.
    """
    found: set[Path] = set()
    for p in paths:
        if p.exists():
            _collect_from(p, found)
            continue
        matches = globmod.glob(str(p), recursive=True)
        if not matches:
            print(f"warning: {p} matched no files or directories; skipping", file=sys.stderr)
            continue
        for m in matches:
            _collect_from(Path(m), found)
    return sorted(found)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_range(first: datetime | None, last: datetime | None) -> str:
    if first is None:
        return "(no valid timestamp)"
    a = first.strftime("%Y-%m-%d %H:%M")
    if last is None or last == first:
        return f"{a} UTC"
    b = last.strftime("%H:%M" if first.date() == last.date() else "%Y-%m-%d %H:%M")
    return f"{a} → {b} UTC"


def _fmt_platform(vehicle: str, platform_sn: str, channels: set[str]) -> str:
    """Human platform label, tolerant of the empty instrument_info a file writes
    before its host populates it (both vehicle and SN come through as "?")."""
    chans = ", ".join(sorted(channels))
    if vehicle == "?" and platform_sn == "?":
        where = "unknown platform"
    elif platform_sn == "?":
        where = f"{vehicle} (SN unknown)"
    elif vehicle == "?":
        where = f"SN {platform_sn}"
    else:
        where = f"{vehicle} SN {platform_sn}"
    return f"{where} (as {chans})"


# Acquisition/ADC settings dropped from the one-line --compact view: they are
# instrument config shared across probes, not part of a probe's own calibration.
_COMPACT_ADC_PARAMS = frozenset({"adc_fs", "adc_bits"})


def _compact_value(agg: SensorAgg, key: str) -> str:
    """The value of *key* for a compact line: the single value, or the distinct
    values joined oldest-first by ``→`` when the parameter changed across uses."""
    values = agg.params.get(key, {})
    if not values:
        return "(none)"
    slots = sorted(values.values(), key=lambda s: s["first"] or datetime.max.replace(tzinfo=UTC))
    displays = [s["display"] for s in slots]
    return displays[0] if len(displays) == 1 else "→".join(displays)


def _compact_line(kind: SensorKind, sn: str, agg: SensorAgg) -> str:
    """One-line probe summary: ``<sn> #<files> <param>: <value> ... used: <range>``.

    Omits the ADC settings, the platform, and the ``(constant)`` markers of the
    full report; a changed parameter shows its values joined by ``→``.
    """
    parts = [sn, f"#{len(agg.files)}"]
    parts += [
        f"{key}: {_compact_value(agg, key)}"
        for key in kind.params
        if key not in _COMPACT_ADC_PARAMS
    ]
    parts.append(f"used: {_fmt_range(agg.first, agg.last).removesuffix(' UTC')}")
    return " ".join(parts)


def print_report(
    inventory: dict[str, dict[str, SensorAgg]],
    kinds: list[str],
    verbose: bool = False,
    compact: bool = False,
    stream: TextIO | None = None,
) -> None:
    out = stream if stream is not None else sys.stdout
    for kind_key in kinds:
        probes = inventory.get(kind_key)
        if not probes:
            continue
        kind = SENSOR_KINDS[kind_key]
        heading = f"{kind.label}s ({len(probes)})"
        print(heading, file=out)
        print("=" * len(heading), file=out)
        print(file=out)

        if compact:
            for sn in sorted(probes):
                print(_compact_line(kind, sn, probes[sn]), file=out)
            print(file=out)
            continue

        width = max(len(p) for p in kind.params) + 1
        for sn in sorted(probes):
            agg = probes[sn]
            print(f"{kind.label} {sn}", file=out)
            print(f"  files:      {len(agg.files)}", file=out)
            print(f"  date range: {_fmt_range(agg.first, agg.last)}", file=out)

            plats = [
                _fmt_platform(veh, psn, channels)
                for (veh, psn), channels in sorted(agg.platforms.items())
            ]
            print(f"  platforms:  {'; '.join(plats)}", file=out)

            for key in kind.params:
                values = agg.params.get(key, {})
                if len(values) <= 1:
                    disp = next(iter(values.values()))["display"] if values else "(none)"
                    print(f"  {key + ':':{width}s} {disp}  (constant)", file=out)
                else:
                    print(f"  {key + ':':{width}s} CHANGED", file=out)
                    # Oldest-first so the sequence of changes reads naturally.
                    for slot in sorted(
                        values.values(),
                        key=lambda s: s["first"] or datetime.max.replace(tzinfo=UTC),
                    ):
                        n = len(slot["files"])
                        unit = "file" if n == 1 else "files"
                        print(
                            f"      {slot['display']:<12s} {n:3d} {unit:5s} "
                            f"{_fmt_range(slot['first'], slot['last'])}",
                            file=out,
                        )
                        if verbose:
                            for fp in sorted(slot["files"]):
                                print(f"          {fp}", file=out)
            print(file=out)


def write_csv(uses: list[SensorUse], out: Path, kinds: list[str]) -> None:
    """One row per sensor channel per file — the raw data behind the summary.

    Wide format: the union of the selected kinds' parameter columns, blank where
    a parameter does not apply to that sensor kind.
    """
    param_cols: list[str] = []
    for key in kinds:
        for p in SENSOR_KINDS[key].params:
            if p not in param_cols:
                param_cols.append(p)
    cols = ["sensor", "sn", "file", "start_time_utc", "vehicle", "platform_sn", "channel"]
    cols += param_cols

    undated = datetime.min.replace(tzinfo=UTC)  # sort undated uses first, deterministically
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for u in sorted(
            uses, key=lambda u: (u.kind, u.sensor_sn, u.start_time or undated, str(u.path))
        ):
            w.writerow(
                [
                    u.kind,
                    u.sensor_sn,
                    str(u.path),
                    u.start_time.strftime("%Y-%m-%dT%H:%M:%SZ") if u.start_time else "",
                    u.vehicle,
                    u.platform_sn,
                    u.channel,
                    *(u.params.get(c, "") for c in param_cols),
                ]
            )


def run(
    paths: list[Path],
    kinds: list[str],
    csv_out: Path | None = None,
    verbose: bool = False,
    compact: bool = False,
    stream: TextIO | None = None,
) -> int:
    """Scan *paths* for the requested sensor *kinds* and print a summary.

    Returns a process exit code: 0 = ok, 1 = no .p files found / unwritable CSV
    target / every file failed to parse.
    """
    out = stream if stream is not None else sys.stdout

    # Fail fast on an unwritable CSV target BEFORE the (potentially long) scan,
    # rather than crashing with a raw traceback after all the work is done.
    if csv_out is not None and (csv_out.is_dir() or not csv_out.parent.is_dir()):
        print(f"Error: cannot write CSV to {csv_out}", file=sys.stderr)
        return 1

    files = iter_pfiles(paths)
    if not files:
        print("No .p files found.", file=sys.stderr)
        return 1

    uses, errors, n_no_sensor = collect_uses(files, kinds)
    inventory = build_inventory(uses)

    n_parsed = len(files) - len(errors)
    kind_labels = ", ".join(SENSOR_KINDS[k].label.lower() + "s" for k in kinds)
    print(
        f"Sensor inventory ({kind_labels}) — {len(files)} .p file(s): "
        f"{n_parsed} parsed, {len(errors)} error(s)",
        file=out,
    )
    if n_no_sensor:
        print(f"  ({n_no_sensor} file(s) had no matching sensor channels)", file=out)
    n_undated = len({u.path for u in uses if u.start_time is None})
    if n_undated:
        print(
            f"  ({n_undated} file(s) had no valid clock — e.g. MR/glider startup — "
            "inventoried without a date)",
            file=out,
        )
    print(file=out)

    print_report(inventory, kinds, verbose=verbose, compact=compact, stream=out)

    if errors:
        print("Errors:", file=out)
        for fp, msg in errors:
            print(f"  {fp}: {msg}", file=out)
        print(file=out)

    if csv_out is not None:
        try:  # backstop for a race/permission change since the pre-scan check
            write_csv(uses, csv_out, kinds)
        except OSError as exc:
            print(f"Error: could not write CSV to {csv_out}: {exc}", file=sys.stderr)
            return 1
        print(f"Wrote {len(uses)} rows to {csv_out}", file=out)

    # Non-zero only when the scan wholly failed (files present, all errored),
    # so a script can distinguish "nothing worked" from "found no sensors".
    return 1 if errors and not uses else 0


# ---------------------------------------------------------------------------
# CLI (python -m odas_tpw.rsi.sensor_inventory)
# ---------------------------------------------------------------------------


def build_arg_parser(prog: str = "sensor_inventory") -> argparse.ArgumentParser:
    """argparse parser shared by ``__main__`` and the ``rsi-tpw sensors`` command."""
    ap = argparse.ArgumentParser(
        prog=prog,
        description="Inventory microstructure sensors across a tree of Rockland .p files.",
    )
    ap.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Directories (scanned recursively), .p files, or glob patterns",
    )
    ap.add_argument("--shear", action="store_true", help="Inventory shear probes")
    ap.add_argument("--fp07", action="store_true", help="Inventory FP07 thermistors")
    ap.add_argument(
        "--all",
        dest="want_all",
        action="store_true",
        help="Inventory every sensor kind (shear + fp07; the default if none is given)",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        metavar="PATH",
        help="Write a per-(file,channel) CSV table (overwritten if it exists)",
    )
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="List the individual files behind each changed parameter value",
    )
    ap.add_argument(
        "--compact",
        action="store_true",
        help="One line per probe: SN, file count, calibration, and date range",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    kinds = resolve_kinds(args.shear, args.fp07, args.want_all)
    return run(args.paths, kinds, csv_out=args.csv, verbose=args.verbose, compact=args.compact)


if __name__ == "__main__":
    raise SystemExit(main())
