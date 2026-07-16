# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""Check shear-probe sensitivities in ``.p`` files against Rockland cal sheets.

A Rockland *Shear Probe Calibration Report* (PDF) records the probe serial
number, its sensitivity (config ``sens`` / *S*), the calibration date and —
when the probe has been calibrated before — the *previous* calibration date
and sensitivity.  This module parses a directory of those sheets, builds a
per-probe calibration timeline, and reports where a ``.p`` file's configured
``sens`` disagrees with the calibration that was in effect when the file was
recorded.

The calibration-sheet directory is an external, user-supplied path and is
never part of this repository.  PDF text extraction uses :mod:`pypdf`, an
**optional** dependency installed via the ``cal`` extra
(``pip install 'microstructure-tpw[cal]'``); it is imported lazily so the rest
of the ``sensors`` command works without it.

Sensitivity model — **hold-previous** (current default): the sensitivity applied
to an observation is that of the most recent calibration on or before the
observation's date; an observation before the earliest known calibration clamps
to that earliest value (and is flagged ``before-earliest``).  Linear
interpolation between calibration dates is deliberately *not* done yet;
:class:`CalTimeline` carries a ``mode`` so it can be added later without
touching callers.

Text-parsing is separated from PDF reading so the parser is unit-testable on
plain strings (see :func:`parse_sheet_text`), independent of any PDF.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:  # avoid a runtime import cycle (sensor_inventory imports us lazily)
    from odas_tpw.rsi.sensor_inventory import SensorUse


logger = logging.getLogger(__name__)


class CalDependencyError(RuntimeError):
    """Raised when reading calibration PDFs but the ``cal`` extra is missing."""


# ---------------------------------------------------------------------------
# Sheet text parsing (no PDF involved — testable on plain strings)
# ---------------------------------------------------------------------------

_NUM = r"([0-9]*\.?[0-9]+)"
_DATE = r"(\d{4})[/_.-](\d{1,2})[/_.-](\d{1,2})"

# "Probe SN: M1458".  The capture is a letter-prefix + digits, plus an optional
# hyphenated suffix (e.g. "M1254-2") — but NOT a bare trailing letter, so the
# glued page-2 plot caption "...ProbeSN:M1254U=0.703m/s..." yields M1254, not
# M1254U (the fall-rate variable U must not become part of the serial).
_SN_RE = re.compile(r"Probe\s*SN\s*:?\s*([A-Za-z]{0,3}\d+(?:-\w+)?)", re.I)
# "Sensitivity (sens or S): 0.0777 m2Vs-2"  — the "(sens or S)" pins it to the
# current value and away from "Previous Sensitivity:" and the "S(0) = ..." line.
_SENS_RE = re.compile(r"Sensitivity\s*\(sens\s*or\s*S\)\s*:?\s*" + _NUM, re.I)
_PREV_SENS_RE = re.compile(r"Previous\s+Sensitivity\s*:?\s*" + _NUM, re.I)
_DATE_RE = re.compile(_DATE)
# "M1458_2026_06_19].pdf" -> ("M1458", 2026, 6, 19); tolerant of trailing junk.
_FNAME_RE = re.compile(r"([A-Za-z]?\d[\w-]*?)[_-](\d{4})[_-](\d{1,2})[_-](\d{1,2})")


def _parse_date_parts(y: str, m: str, d: str) -> date | None:
    try:
        return date(int(y), int(m), int(d))
    except ValueError:
        return None


@dataclass
class CalSheet:
    """The calibration facts parsed from one sheet.

    ``prev_*`` are ``None`` for a probe's first-ever calibration (the sheet then
    simply omits the "Previous ..." lines, as Rockland's do).
    """

    sn: str | None
    sensitivity: float | None
    cal_date: date | None
    prev_sensitivity: float | None = None
    prev_cal_date: date | None = None
    source: str = ""

    def is_usable(self) -> bool:
        return self.sn is not None and self.sensitivity is not None and self.cal_date is not None

    def points(self) -> list[CalPoint]:
        """Every ``(date, sensitivity)`` calibration point this sheet supplies."""
        pts: list[CalPoint] = []
        if self.cal_date is not None and self.sensitivity is not None:
            pts.append(CalPoint(self.cal_date, self.sensitivity, self.source))
        if self.prev_cal_date is not None and self.prev_sensitivity is not None:
            pts.append(
                CalPoint(self.prev_cal_date, self.prev_sensitivity, f"{self.source} (previous)")
            )
        return pts


def parse_filename(name: str) -> tuple[str, date | None] | None:
    """``(serial, date)`` from a ``M<sn>_<YYYY>_<MM>_<DD>.pdf`` filename, or None.

    Tolerant of a trailing stray character before the extension (one real sheet
    is named ``M1458_2026_06_19].pdf``).
    """
    m = _FNAME_RE.search(Path(name).stem)
    if not m:
        return None
    return m.group(1).upper(), _parse_date_parts(m.group(2), m.group(3), m.group(4))


def parse_sheet_text(text: str, source: str = "") -> CalSheet:
    """Parse the labeled fields out of a sheet's extracted text.

    Line-oriented and order-independent so it tolerates ``pypdf``'s two-column
    interleaving and spacing quirks.  "Previous ..." lines are matched before
    their non-previous counterparts so the current values are not overwritten.
    """
    sn: str | None = None
    sens: float | None = None
    cal_date: date | None = None
    prev_sens: float | None = None
    prev_cal_date: date | None = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        if sn is None:
            m = _SN_RE.search(line)
            if m:
                sn = m.group(1).upper()

        if prev_sens is None:
            m = _PREV_SENS_RE.search(line)
            if m:
                prev_sens = float(m.group(1))
        if sens is None and "previous" not in line.lower():
            m = _SENS_RE.search(line)
            if m:
                sens = float(m.group(1))

        # Dates: skip the "Recommended re-calibration" line; keep previous vs current apart.
        low = line.lower()
        if "recommend" in low:
            continue
        if "previous" in low and "calibration date" in low and prev_cal_date is None:
            m = _DATE_RE.search(line)
            if m:
                prev_cal_date = _parse_date_parts(m.group(1), m.group(2), m.group(3))
        elif "calibration date" in low and cal_date is None:
            m = _DATE_RE.search(line)
            if m:
                cal_date = _parse_date_parts(m.group(1), m.group(2), m.group(3))

    # A physically-valid sensitivity is positive; treat 0/negative (a misparse or
    # a corrupt sheet) as missing so it can't seed a bogus timeline point or a
    # divide-by-zero downstream.
    if sens is not None and sens <= 0:
        sens = None
    if prev_sens is not None and prev_sens <= 0:
        prev_sens = None

    return CalSheet(sn, sens, cal_date, prev_sens, prev_cal_date, source=source)


# ---------------------------------------------------------------------------
# PDF text extraction (lazy pypdf — the only part that needs the 'cal' extra)
# ---------------------------------------------------------------------------


def extract_pdf_text(path: Path) -> str:
    """Return the concatenated text of every page of *path* via ``pypdf``.

    Raises :class:`CalDependencyError` if ``pypdf`` is not installed.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise CalDependencyError(
            "reading calibration sheets requires the 'cal' extra: "
            "pip install 'microstructure-tpw[cal]'"
        ) from exc
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


# ---------------------------------------------------------------------------
# Per-probe calibration timeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalPoint:
    date: date
    sensitivity: float
    source: str


@dataclass
class CalTimeline:
    """A probe's calibration points over time, and the sensitivity lookup."""

    sn: str
    points: list[CalPoint]  # sorted ascending by date, de-duplicated
    mode: str = "hold-previous"  # future: "interpolate"

    def sensitivity_at(self, obs: date) -> tuple[float, CalPoint, str]:
        """``(sensitivity, governing point, status)`` for an observation date.

        ``status`` is ``"in-effect"`` when a calibration on or before *obs*
        governs, or ``"before-earliest"`` when *obs* precedes every known
        calibration and the earliest is clamped to.
        """
        if not self.points:
            raise ValueError(f"CalTimeline for {self.sn!r} has no calibration points")
        prior = [p for p in self.points if p.date <= obs]
        if prior:
            gov = max(prior, key=lambda p: p.date)
            return gov.sensitivity, gov, "in-effect"
        earliest = min(self.points, key=lambda p: p.date)
        return earliest.sensitivity, earliest, "before-earliest"


def load_cal_dir(cal_dir: Path) -> tuple[dict[str, CalTimeline], list[str]]:
    """Parse every ``*.pdf`` in *cal_dir* into ``{serial: CalTimeline}``.

    Points for a probe are merged across all its sheets and de-duplicated by
    date (a newer sheet's "previous" point usually repeats an older sheet's
    current point).  Returns ``(timelines, warnings)``; *warnings* lists sheets
    that could not be parsed or that carry conflicting values.  Raises
    :class:`CalDependencyError` if ``pypdf`` is missing.
    """
    warnings: list[str] = []
    per_sn: dict[str, dict[date, CalPoint]] = {}

    pdfs = sorted(cal_dir.glob("*.pdf"))
    if not pdfs:
        warnings.append(f"no *.pdf calibration sheets found in {cal_dir}")

    for pdf in pdfs:
        text = extract_pdf_text(pdf)  # CalDependencyError propagates (fail fast, once)
        sheet = parse_sheet_text(text, source=pdf.name)

        # The filename (M<sn>_<YYYY>_<MM>_<DD>.pdf) also encodes the serial and
        # calibration date, so use it to fill anything the PDF text didn't yield
        # (a garbled SN line still leaves a usable sheet), and to cross-check the
        # rest.  Sensitivity has no filename source, so the sheet still needs it.
        fn = parse_filename(pdf.name)
        if fn is not None:
            fn_sn, fn_date = fn
            if sheet.sn is None:
                sheet.sn = fn_sn
            elif fn_sn != sheet.sn:
                warnings.append(
                    f"{pdf.name}: filename SN {fn_sn} != sheet SN {sheet.sn}; using sheet SN"
                )
            if fn_date is not None:
                if sheet.cal_date is None:
                    sheet.cal_date = fn_date
                elif fn_date != sheet.cal_date:
                    warnings.append(
                        f"{pdf.name}: filename date {fn_date} != sheet date {sheet.cal_date}"
                    )

        if not sheet.is_usable():
            missing = [
                f for f, v in (("SN", sheet.sn), ("sensitivity", sheet.sensitivity),
                               ("cal_date", sheet.cal_date)) if v is None
            ]
            warnings.append(f"{pdf.name}: could not parse {', '.join(missing)}; skipped")
            continue

        assert sheet.sn is not None  # is_usable() guarantees it (for type-checkers)
        dated = per_sn.setdefault(sheet.sn, {})
        for pt in sheet.points():
            existing = dated.get(pt.date)
            if existing is not None and abs(existing.sensitivity - pt.sensitivity) > 1e-9:
                warnings.append(
                    f"{sheet.sn}: conflicting sensitivity for {pt.date} "
                    f"({existing.sensitivity} vs {pt.sensitivity}); kept {existing.sensitivity}"
                )
            else:
                dated.setdefault(pt.date, pt)

    timelines = {
        sn: CalTimeline(sn, sorted(pts.values(), key=lambda p: p.date))
        for sn, pts in per_sn.items()
    }
    return timelines, warnings


# ---------------------------------------------------------------------------
# Checking .p-file shear probes against the timelines
# ---------------------------------------------------------------------------


@dataclass
class ObsCheck:
    """One shear-probe observation compared to its calibration."""

    sn: str
    channel: str
    file: Path
    obs_date: date | None
    configured: float
    expected: float
    governing: CalPoint
    status: str  # "in-effect" | "before-earliest"

    @property
    def abs_diff(self) -> float:
        """Signed ``configured - expected``, in sensitivity units (the flag metric)."""
        return self.configured - self.expected

    @property
    def pct_diff(self) -> float:
        """Signed percent difference — reported for context, not the flag metric."""
        if self.expected == 0:  # defensive; parse rejects non-positive sensitivities
            return float("inf")
        return (self.configured - self.expected) / self.expected * 100.0


@dataclass
class CheckSummary:
    """Outcome of checking a set of shear uses against the calibration sheets."""

    checks: list[ObsCheck]
    n_checked: int  # observations actually compared
    no_sheet: set[str]  # probe SNs with no matching sheet
    n_no_sens: int  # observations whose configured sens was blank/unparseable
    n_undated: int  # observations with no usable clock

    def mismatches(self, tol: float) -> list[ObsCheck]:
        """Checks whose |configured - expected| exceeds *tol* (sensitivity units)."""
        return [c for c in self.checks if abs(c.abs_diff) > tol]


def _to_float(raw: str) -> float | None:
    try:
        f = float(raw)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def check_uses(uses: list[SensorUse], timelines: dict[str, CalTimeline]) -> CheckSummary:
    """Compare every shear :class:`SensorUse` to its probe's calibration timeline."""
    checks: list[ObsCheck] = []
    no_sheet: set[str] = set()
    n_no_sens = 0
    n_undated = 0

    for use in uses:
        if use.kind != "shear":
            continue
        # Sheet timelines are keyed upper-case; the .p-config serial can be any
        # case, so normalize the lookup or a case difference silently reads as
        # "no calibration sheet".
        tl = timelines.get(use.sensor_sn.upper())
        if tl is None:
            no_sheet.add(use.sensor_sn)
            continue
        configured = _to_float(use.params.get("sens", ""))
        if configured is None:
            n_no_sens += 1
            continue
        if use.start_time is None:
            n_undated += 1
            continue
        obs_date = use.start_time.date()
        expected, gov, status = tl.sensitivity_at(obs_date)
        checks.append(
            ObsCheck(
                use.sensor_sn, use.channel, use.path, obs_date, configured, expected, gov, status
            )
        )

    return CheckSummary(checks, len(checks), no_sheet, n_no_sens, n_undated)


# ---------------------------------------------------------------------------
# Reporting (only mismatches, per the design)
# ---------------------------------------------------------------------------


def _fmt_tol(x: float) -> str:
    """Format an absolute tolerance plainly (``0.00005``, not ``5e-05``)."""
    if x == 0:
        return "0"
    return f"{x:.10f}".rstrip("0").rstrip(".")


def _fmt_group(key: tuple, obs: list[ObsCheck]) -> list[str]:
    sn, configured, expected, status, gov_date = key
    dates = sorted(o.obs_date for o in obs if o.obs_date is not None)
    when = ""
    if dates:
        when = f"{dates[0]}" if dates[0] == dates[-1] else f"{dates[0]}…{dates[-1]}"
    channels = ", ".join(sorted({o.channel for o in obs}))
    delta = configured - expected
    pct = float("inf") if expected == 0 else delta / expected * 100.0
    tag = " [before earliest cal]" if status == "before-earliest" else ""
    n = len(obs)
    unit = "file" if n == 1 else "files"
    return [
        f"  {sn}  configured sens {configured:g}  vs  calibration {expected:g} "
        f"(cal {gov_date}){tag}  →  Δ {delta:+.4f} ({pct:+.1f}%)",
        f"      {n} {unit}, obs {when}, as {channels}",
    ]


def format_check(
    summary: CheckSummary,
    cal_dir: Path,
    tol: float,
    load_warnings: list[str] | None = None,
) -> list[str]:
    """Render the calibration-check section (mismatches only) as text lines.

    *tol* is an absolute sensitivity difference (same units as ``sens``), not a
    percentage — a configured ``sens`` is flagged when it differs from the
    in-effect calibration by more than *tol*.
    """
    lines: list[str] = []
    heading = "Shear calibration check"
    lines.append(heading)
    lines.append("=" * len(heading))
    lines.append(f"cal-dir: {cal_dir}   tolerance: ±{_fmt_tol(tol)} (sensitivity units)")

    n_probes_checked = len({c.sn for c in summary.checks})
    lines.append(
        f"checked {summary.n_checked} shear observation(s) across "
        f"{n_probes_checked} probe(s) with a matching sheet"
    )
    if summary.no_sheet:
        lines.append(
            f"no calibration sheet for {len(summary.no_sheet)} probe(s): "
            f"{', '.join(sorted(summary.no_sheet))}"
        )
    if summary.n_no_sens:
        lines.append(f"{summary.n_no_sens} observation(s) had no configured sensitivity (skipped)")
    if summary.n_undated:
        lines.append(f"{summary.n_undated} observation(s) had no usable clock (skipped)")
    for w in load_warnings or []:
        lines.append(f"warning: {w}")
    lines.append("")

    mism = summary.mismatches(tol)
    if summary.n_checked == 0:
        lines.append(
            "No shear observations were checked "
            "(no probe matched a calibration sheet with a usable clock and sens)."
        )
        lines.append("")
        return lines
    if not mism:
        lines.append(
            f"No mismatches: all {summary.n_checked} checked observation(s) agree "
            f"with their calibration within ±{_fmt_tol(tol)}."
        )
        lines.append("")
        return lines

    grouped: dict[tuple, list[ObsCheck]] = {}
    for c in mism:
        key = (c.sn, c.configured, c.expected, c.status, c.governing.date)
        grouped.setdefault(key, []).append(c)

    lines.append(f"{len(mism)} mismatching observation(s):")
    for key in sorted(grouped, key=lambda k: (k[0], str(k[4]))):
        lines.extend(_fmt_group(key, grouped[key]))
    lines.append("")
    return lines


def print_check(
    summary: CheckSummary,
    cal_dir: Path,
    tol: float,
    load_warnings: list[str] | None = None,
    stream: TextIO | None = None,
) -> None:
    import sys

    out = stream if stream is not None else sys.stdout
    for line in format_check(summary, cal_dir, tol, load_warnings):
        print(line, file=out)


# ---------------------------------------------------------------------------
# Tracked sensitivity registry (shear_sensitivities.csv)
# ---------------------------------------------------------------------------

CSV_FIELDS = ["serial", "cal_date", "sens", "units", "source", "sheet", "recal_due", "notes"]
CSV_UNITS = "V/(m^2 s^-2)"
_PREV_NOTE = "previous-calibration entry on this sheet"


@dataclass
class CsvUpdateStats:
    sheets_parsed: int = 0
    sheets_failed: int = 0
    added: int = 0
    upgraded: int = 0
    unchanged: int = 0
    conflicts: list[str] = field(default_factory=list)


def update_sensitivity_csv(cal_dir: Path, csv_path: Path | None = None) -> CsvUpdateStats:
    """Merge every calibration sheet in *cal_dir* into the CSV registry.

    Idempotent: re-running adds nothing. Merge rules —
    - key = (serial, cal_date, sens): an existing row with the same key is
      kept (``manual`` rows are never touched), EXCEPT that a sheet's own
      "current" entry replaces a previous-calibration attestation of the same
      point (better provenance: it carries the sheet id and recal date).
    - same (serial, cal_date) with a DIFFERENT sens is kept side by side and
      reported as a conflict — conflicting records must stay visible, never
      be silently dropped.
    """
    import csv as _csv

    cal_dir = Path(cal_dir)
    if csv_path is None:
        csv_path = cal_dir / "shear_sensitivities.csv"
    stats = CsvUpdateStats()

    rows: list[dict[str, str]] = []
    if csv_path.exists():
        with csv_path.open(newline="") as f:
            for row in _csv.DictReader(f):
                rows.append({k: (row.get(k) or "") for k in CSV_FIELDS})

    def _key(r: dict[str, str]) -> tuple[str, str, str]:
        return (r["serial"], r["cal_date"], r["sens"])

    index = {_key(r): i for i, r in enumerate(rows)}

    def _merge(new: dict[str, str]) -> None:
        k = _key(new)
        if k in index:
            old = rows[index[k]]
            # Upgrade a previous-entry attestation to the point's own sheet.
            if old["notes"] == _PREV_NOTE and new["notes"] != _PREV_NOTE:
                rows[index[k]] = new
                stats.upgraded += 1
            else:
                stats.unchanged += 1
            return
        clash = [
            r for r in rows
            if r["serial"] == new["serial"] and r["cal_date"] == new["cal_date"]
        ]
        if clash:
            stats.conflicts.append(
                f"{new['serial']} {new['cal_date']}: sens {new['sens']} "
                f"({new['sheet'] or new['source']}) vs existing "
                + ", ".join(f"{r['sens']} ({r['sheet'] or r['source']})" for r in clash)
            )
        rows.append(new)
        index[k] = len(rows) - 1
        stats.added += 1

    for pdf in sorted(cal_dir.glob("*.pdf")):
        try:
            sheet = parse_sheet_text(extract_pdf_text(pdf), source=pdf.name)
        except (CalDependencyError, OSError, ValueError) as e:
            stats.sheets_failed += 1
            logger.warning("%s: could not parse calibration sheet: %s", pdf.name, e)
            continue
        if sheet.sn is None or sheet.sensitivity is None or sheet.cal_date is None:
            stats.sheets_failed += 1
            logger.warning("%s: sheet lacks serial/sensitivity/date; skipped", pdf.name)
            continue
        stats.sheets_parsed += 1
        recal = getattr(sheet, "recal_due", None)  # present once the staleness guard lands
        _merge({
            "serial": sheet.sn,
            "cal_date": sheet.cal_date.isoformat(),
            "sens": f"{sheet.sensitivity}",
            "units": CSV_UNITS,
            "source": "sheet",
            "sheet": pdf.name,
            "recal_due": recal.isoformat() if recal else "",
            "notes": "",
        })
        if sheet.prev_sensitivity is not None and sheet.prev_cal_date is not None:
            _merge({
                "serial": sheet.sn,
                "cal_date": sheet.prev_cal_date.isoformat(),
                "sens": f"{sheet.prev_sensitivity}",
                "units": CSV_UNITS,
                "source": "sheet",
                "sheet": pdf.name,
                "recal_due": "",
                "notes": _PREV_NOTE,
            })

    rows.sort(key=lambda r: (r["serial"], r["cal_date"], r["source"]))
    with csv_path.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return stats
