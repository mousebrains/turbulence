# Sep-2026, Claude and Pat Welch, pat@mousebrains.com
"""Differential-gain audit, keyed by INSTRUMENT rather than by sensor.

``diff_gain`` is the pre-emphasis differential gain of an instrument's own
amplifier chain. It is not a property of the probe screwed into it, which is
why :mod:`odas_tpw.rsi.sensor_inventory` cannot report it usefully: that tool
is keyed on sensor serial, and it deliberately drops the ``X_dX``
pre-emphasized channels (``T1_dT1``, ``P_dP``) because they are the same
physical sensor as their base channel. So the thermistor and pressure gains are
invisible there, and the shear gains are attributed to whichever probe happened
to be installed.

The stakes are the shear sensitivity's. The conversion is

    shear = (adc_fs / 2^adc_bits * counts + offset) / (2*sqrt(2)*diff_gain*sens)

so epsilon goes as ``(diff_gain * sens)^-2`` and chi as ``diff_gain^-2``: a 2%
gain error is 4% in both, and an order-of-magnitude error is two orders of
magnitude in epsilon. On ARCTERX-2023 one VMP carried ``diff_gain = 0.09`` on
both shear channels where every sibling in the same cruise carried 0.92-0.99,
and nothing in the processing chain noticed.

Two things this reports that a per-sensor view cannot:

* **Outliers against the fleet.** A value far from the corpus median for the
  same channel name is flagged. This is deliberately relative rather than a
  hardcoded band, because the plausible range differs by channel class
  (differentiator channels sit near 1, ``P_dP`` near 20) and hardcoding either
  would be a guess.
* **Changes over time on one instrument.** Gains follow the ELECTRONICS, so
  they change when an instrument is rebuilt -- e.g. a Persistor CF2 VMP
  upgraded to RDL hardware. Such a change is expected and informative, not an
  error, so it is reported as a transition with dates rather than flagged.

Grouping is on the SERIAL NUMBER ALONE, deliberately. A rebuild changes the
``model`` string too (SN 142 became ``VMP250IR_RDL``, SN 479 ``VMP250IR_RT``),
so keying on ``(sn, model)`` would split the instrument in two and suppress
exactly the transition this exists to surface. The models seen are reported
alongside the gains instead.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# A value this far from the corpus median for the same channel is called out.
# Wide on purpose: real fleet spread is a few percent, and the failure this
# exists to catch is an order of magnitude.
OUTLIER_FACTOR = 2.0
# Below this many instruments the median is not worth trusting, so no flagging.
MIN_INSTRUMENTS_FOR_OUTLIER = 3
# Serial used when a config carries no instrument identity at all. Kept out of
# the fleet median: an unattributable gain cannot cast an instrument's vote,
# and merging every such file into one bucket would both fabricate an
# "instrument" and let one real unit's gains count twice.
UNKNOWN_SN = "?"


@dataclass
class GainUse:
    """One (file, channel) differential gain observation."""

    path: Path
    instrument_sn: str
    model: str
    vehicle: str
    channel: str
    diff_gain: float
    raw: str
    when: datetime | None

    @property
    def key(self) -> str:
        """Value identity for change detection: the PARSED number.

        Keying on the raw string would report ``0.95`` -> ``0.950`` as a
        hardware rebuild, which the report explicitly tells the reader to read
        as real electronics history.
        """
        return repr(self.diff_gain)


@dataclass
class GainAgg:
    """Every observation of one channel on one instrument."""

    values: dict[str, list[GainUse]] = field(default_factory=lambda: defaultdict(list))

    @property
    def n_files(self) -> int:
        return sum(len(v) for v in self.values.values())

    @property
    def changed(self) -> bool:
        return len(self.values) > 1

    def timeline(self) -> list[tuple[str, datetime | None, datetime | None, int]]:
        """(displayed value, first, last, n) per distinct value, in time order."""
        out = []
        for uses in self.values.values():
            times = [u.when for u in uses if u.when is not None]
            first = min(times) if times else None
            last = max(times) if times else None
            out.append((uses[0].raw, first, last, len(uses)))
        return sorted(out, key=lambda r: (r[1] is None, r[1] or datetime.min))


def collect_gains(
    pfiles: list[Path], reader: Callable[[Path], tuple[dict, dict]]
) -> tuple[list[GainUse], list[tuple[Path, str]]]:
    """Read ``diff_gain`` from every channel of every file.

    *reader* is a ``path -> (header, config)`` callable (injected so this module
    does not duplicate :mod:`sensor_inventory`'s header handling, and so tests
    can supply a stub).
    """
    from odas_tpw.rsi.p_file import instrument_sn

    uses: list[GainUse] = []
    errors: list[tuple[Path, str]] = []
    for p in pfiles:
        try:
            header, config = reader(p)
        except Exception as exc:
            errors.append((p, f"{type(exc).__name__}: {exc}"))
            continue
        inst = config.get("instrument_info", {}) if isinstance(config, dict) else {}
        if isinstance(inst, list):
            inst = inst[0] if inst else {}
        # instrument_sn also reads CASPER-era `serial_num`; going straight to
        # `sn` would collapse every such file into one fabricated instrument.
        sn = str(instrument_sn(inst) or UNKNOWN_SN).strip() or UNKNOWN_SN
        model = str(inst.get("model", "?")).strip() or "?"
        vehicle = str(inst.get("vehicle", "?")).strip() or "?"
        when = _safe_time(header, config)
        for ch in config.get("channels", []) or []:
            raw = ch.get("diff_gain")
            name = str(ch.get("name", "?")).strip()
            if raw is None or name == "?":
                continue
            try:
                val = float(str(raw).strip())
            except (TypeError, ValueError):
                errors.append((p, f"{name}: unparseable diff_gain {raw!r}"))
                continue
            # A non-finite gain parses fine but poisons every median it enters
            # (statistics.median propagates NaN), which would silently disable
            # outlier detection for that channel across the whole fleet.
            if not math.isfinite(val):
                errors.append((p, f"{name}: non-finite diff_gain {raw!r}"))
                continue
            uses.append(GainUse(p, sn, model, vehicle, name, val, str(raw).strip(), when))
    return uses, errors


def _safe_time(header: dict, config: dict) -> datetime | None:
    from odas_tpw.rsi.sensor_inventory import _start_time_utc

    try:
        return _start_time_utc(header, config)
    except Exception:
        return None


def build(uses: list[GainUse]) -> dict[str, dict[str, GainAgg]]:
    """instrument SN -> channel -> aggregate.

    Keyed on the serial ALONE; see the module docstring for why the model
    string must not be part of the key.
    """
    out: dict[str, dict[str, GainAgg]] = defaultdict(lambda: defaultdict(GainAgg))
    for u in uses:
        out[u.instrument_sn][u.channel].values[u.key].append(u)
    return out


def models_for(uses: list[GainUse], sn: str) -> list[str]:
    """Distinct model strings seen for one serial, in first-seen order."""
    seen: dict[str, datetime | None] = {}
    for u in uses:
        if u.instrument_sn == sn and u.model not in seen:
            seen[u.model] = u.when
    return sorted(seen, key=lambda m: (seen[m] is None, seen[m] or datetime.min))


def channel_medians(uses: list[GainUse]) -> dict[str, tuple[float, int]]:
    """channel name -> (median gain, number of distinct instruments seen).

    One vote per instrument, not per file: a campaign with a thousand files
    from one unit must not drown out the rest of the fleet. Files with no
    instrument identity are excluded -- they cannot cast a vote, and lumping
    them together would let one unit count twice.
    """
    per_ch: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for u in uses:
        if u.instrument_sn == UNKNOWN_SN:
            continue
        per_ch[u.channel][u.instrument_sn].append(u.diff_gain)
    out = {}
    for ch, by_sn in per_ch.items():
        per_instrument = [statistics.median(v) for v in by_sn.values()]
        out[ch] = (statistics.median(per_instrument), len(by_sn))
    return out


def invalid_gains(uses: list[GainUse]) -> list[tuple[str, str, float]]:
    """(SN, channel, value) for gains that are impossible regardless of fleet.

    ``diff_gain <= 0`` divides the shear conversion by zero or flips its sign.
    It needs no fleet median to be wrong, and must never be quietly skipped
    just because it cannot form a ratio.
    """
    seen: set[tuple[str, str, str]] = set()
    out = []
    for u in uses:
        if u.diff_gain <= 0:
            k = (u.instrument_sn, u.channel, u.key)
            if k not in seen:
                seen.add(k)
                out.append((u.instrument_sn, u.channel, u.diff_gain))
    return sorted(out)


def unchecked_channels(uses: list[GainUse]) -> list[tuple[str, int]]:
    """(channel, n instruments) for channels with too few instruments to check.

    The outlier gate is per CHANNEL, so a channel only one or two units carry
    goes unchecked even when the scan as a whole has plenty of instruments.
    Reporting "no outliers" without saying so would be a false all-clear.
    """
    med = channel_medians(uses)
    return sorted((ch, n) for ch, (_, n) in med.items() if n < MIN_INSTRUMENTS_FOR_OUTLIER)


def outliers(
    uses: list[GainUse], factor: float = OUTLIER_FACTOR
) -> list[tuple[str, str, float, float]]:
    """(SN, channel, value, fleet median) for every flagged gain.

    Flags only when at least :data:`MIN_INSTRUMENTS_FOR_OUTLIER` instruments
    contribute to that CHANNEL's median, so a thinly-covered channel never
    invents an outlier. Channels skipped for that reason are reported by
    :func:`unchecked_channels` rather than passing silently.
    """
    med = channel_medians(uses)
    seen: set[tuple[str, str, str]] = set()
    out = []
    for u in uses:
        m, n_inst = med.get(u.channel, (None, 0))
        if m is None or n_inst < MIN_INSTRUMENTS_FOR_OUTLIER or m <= 0 or u.diff_gain <= 0:
            continue
        ratio = u.diff_gain / m
        if ratio > factor or ratio < 1.0 / factor:
            key = (u.instrument_sn, u.channel, u.key)
            if key not in seen:
                seen.add(key)
                out.append((u.instrument_sn, u.channel, u.diff_gain, m))
    return sorted(out, key=lambda r: (r[0], r[1]))


def _fmt_dt(d: datetime | None) -> str:
    return d.strftime("%Y-%m-%d") if d else "no clock"


def _sn_sort(sn: str) -> tuple:
    return (0, int(sn)) if sn.isdigit() else (1, sn)


def format_report(uses: list[GainUse], errors: list[tuple[Path, str]]) -> str:
    """Human-readable audit."""
    if not uses:
        if errors:
            head: list[str] = [
                "No differential gains found — because no file could be read, "
                "not because the files carry no diff_gain.",
                "",
                f"{len(errors)} file(s) could not be read:",
            ]
            for p, msg in errors[:10]:
                head.append(f"  {p.name}: {msg}")
            if len(errors) > 10:
                head.append(f"  ... and {len(errors) - 10} more")
            return "\n".join(head) + "\n"
        return "No differential gains found (no .p file carried a diff_gain).\n"

    inv = build(uses)
    med = channel_medians(uses)
    flagged = {(sn, ch) for sn, ch, _, _ in outliers(uses)}
    bad = {(sn, ch) for sn, ch, _ in invalid_gains(uses)}
    lines: list[str] = []
    lines.append(
        f"Differential gains — {len(uses)} channel observations across "
        f"{len({u.path for u in uses})} file(s), {len(inv)} instrument(s)"
    )
    lines.append("")
    lines.append("diff_gain belongs to the INSTRUMENT's amplifier chain, not to the probe.")
    lines.append("epsilon goes as (diff_gain*sens)^-2 and chi as diff_gain^-2, so a 2% error")
    lines.append("is 4% in both.")
    lines.append("")

    for sn in sorted(inv, key=_sn_sort):
        chans = inv[sn]
        all_uses = [u for c in chans.values() for v in c.values.values() for u in v]
        times = [u.when for u in all_uses if u.when]
        models = models_for(uses, sn)
        label = f"SN {sn}" if sn != UNKNOWN_SN else "SN unknown (no instrument_info)"
        lines.append(f"{label}   model {' → '.join(models)}")
        lines.append(
            f"  {len({u.path for u in all_uses})} file(s)   "
            f"{_fmt_dt(min(times) if times else None)} → "
            f"{_fmt_dt(max(times) if times else None)}"
        )
        if sn == UNKNOWN_SN:
            lines.append("  (excluded from the fleet median: no instrument to attribute it to)")
        for ch in sorted(chans):
            tl = chans[ch].timeline()
            mark = ""
            if (sn, ch) in bad:
                mark = "  ** INVALID (<= 0) **"
            elif (sn, ch) in flagged:
                mark = "  ** OUTLIER **"
            if len(tl) == 1:
                raw = tl[0][0]
                m, n_ch = med.get(ch, (None, 0))
                ctx = f"   fleet median {m:g} over {n_ch} instrument(s)" if m else ""
                lines.append(f"    {ch:10s} {raw:>8s}{mark}{ctx}")
            else:
                lines.append(f"    {ch:10s} CHANGED over time:{mark}")
                for raw, f_, l_, n in tl:
                    lines.append(f"      {raw:>8s}   {_fmt_dt(f_)} → {_fmt_dt(l_)}   ({n} file(s))")
                lines.append(
                    "      (gains follow the electronics; a change is expected when an "
                    "instrument is rebuilt)"
                )
        lines.append("")

    bad_rows = invalid_gains(uses)
    if bad_rows:
        lines.append("INVALID GAINS")
        lines.append("=" * 13)
        lines.append("A gain <= 0 divides the shear conversion by zero or flips its sign.")
        for sn, ch, val in bad_rows:
            lines.append(f"  SN {sn:>5s} {ch:10s} {val:g}")
        lines.append("")

    out = outliers(uses)
    if out:
        lines.append("OUTLIERS")
        lines.append("=" * 8)
        lines.append(
            f"Flagged where a gain differs from the fleet median for the same channel "
            f"by more than {OUTLIER_FACTOR:g}x."
        )
        lines.append("")
        for sn, ch, val, m in out:
            lines.append(
                f"  SN {sn:>5s} {ch:10s} {val:g}  vs fleet median {m:g}  ({val / m:.2f}x)"
            )
        lines.append("")
        lines.append("  A gain wrong by a factor f puts epsilon out by f^-2. Confirm against the")
        lines.append("  instrument's Rockland record before using its dissipation.")
    else:
        lines.append(
            f"No outliers among the channels that could be checked "
            f"(within {OUTLIER_FACTOR:g}x of the fleet median)."
        )

    skipped = unchecked_channels(uses)
    if skipped:
        lines.append("")
        lines.append(
            f"NOT CHECKED — these channels are carried by fewer than "
            f"{MIN_INSTRUMENTS_FOR_OUTLIER} instruments, so no fleet median is "
            f"trustworthy and no outlier can be ruled out:"
        )
        for ch, n in skipped:
            lines.append(f"  {ch:10s} {n} instrument(s)")

    if errors:
        lines.append("")
        lines.append(f"{len(errors)} file(s) could not be read:")
        for p, msg in errors[:10]:
            lines.append(f"  {p.name}: {msg}")
        if len(errors) > 10:
            lines.append(f"  ... and {len(errors) - 10} more")
    return "\n".join(lines) + "\n"


def write_csv(uses: list[GainUse], out: Path) -> None:
    """Per-(file, channel) table."""
    import csv

    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["file", "instrument_sn", "model", "vehicle", "channel", "diff_gain", "start_time"]
        )
        for u in sorted(uses, key=lambda u: (u.instrument_sn, str(u.path), u.channel)):
            w.writerow(
                [
                    str(u.path),
                    u.instrument_sn,
                    u.model,
                    u.vehicle,
                    u.channel,
                    u.raw,
                    u.when.isoformat() if u.when else "",
                ]
            )


def run(
    pfiles: list[Path],
    reader: Callable[[Path], tuple[dict, dict]],
    csv_path: Path | None = None,
) -> tuple[str, int, bool]:
    """Audit *pfiles*.

    Returns ``(report, n_problems, csv_written)``. ``n_problems`` counts both
    fleet outliers and gains that are invalid outright, since both are reasons
    for ``--diff-gain-strict`` to fail.
    """
    uses, errors = collect_gains(pfiles, reader)
    csv_written = False
    if csv_path is not None and uses:
        try:
            write_csv(uses, csv_path)
            csv_written = True
        except OSError as exc:
            # Never throw away the audit because the side-car table failed.
            errors = [*errors, (csv_path, f"could not write CSV: {exc}")]
    n_problems = len(outliers(uses)) + len(invalid_gains(uses))
    return format_report(uses, errors), n_problems, csv_written
