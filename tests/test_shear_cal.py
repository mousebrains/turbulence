# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for shear-probe calibration-sheet parsing, timelines, and checking."""

from __future__ import annotations

import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from odas_tpw.rsi import sensor_inventory as si
from odas_tpw.rsi import shear_cal as sc

# The one real Rockland sheet committed for end-to-end pypdf extraction tests
# (page 1 only, image-stripped via ghostscript to keep it lean).
CAL_PDF = Path(__file__).parent / "data" / "shear_cal_M1458_2026_06_19.pdf"

# Text as pypdf actually extracts it from a Rockland sheet WITH a previous cal
# (note the missing space after the sensitivity colon and the mangled unit).
M1458_TEXT = """
Shear Probe Calibration Report
Probe SN: M1458
Calibration Results
Sensitivity (sens or S):0.0777 V
Calibration Date: 2026/06/19
Recommended re-calibration: 2027/06/19
Previous Calibration Date: 2021/04/27
Previous Sensitivity: 0.0679
S(0) = 0.0777
"""

# A sheet with NO previous calibration (first-ever); the "Previous ..." lines
# are simply absent, plus an unrelated "Pressure Test Date" to not confuse.
M3039_TEXT = """
Probe SN: M3039
Sensitivity (sens or S): 0.1189 V
Calibration Date: 2024/07/09
Recommended re-calibration: 2025/07/09
Pressure Test Date: 2024/07/05
"""


# ---------------------------------------------------------------------------
# Text parsing
# ---------------------------------------------------------------------------


def test_parse_sheet_with_previous():
    s = sc.parse_sheet_text(M1458_TEXT, source="M1458.pdf")
    assert s.sn == "M1458"
    assert s.sensitivity == pytest.approx(0.0777)
    assert s.cal_date == date(2026, 6, 19)
    assert s.prev_sensitivity == pytest.approx(0.0679)
    assert s.prev_cal_date == date(2021, 4, 27)
    assert s.is_usable()
    # Two calibration points come out of one sheet.
    pts = s.points()
    assert {(p.date, p.sensitivity) for p in pts} == {
        (date(2026, 6, 19), 0.0777),
        (date(2021, 4, 27), 0.0679),
    }


def test_parse_sheet_without_previous():
    s = sc.parse_sheet_text(M3039_TEXT, source="M3039.pdf")
    assert s.sn == "M3039"
    assert s.sensitivity == pytest.approx(0.1189)
    assert s.cal_date == date(2024, 7, 9)
    assert s.prev_sensitivity is None
    assert s.prev_cal_date is None
    assert [(p.date, p.sensitivity) for p in s.points()] == [(date(2024, 7, 9), 0.1189)]


def test_parse_does_not_confuse_recommended_or_previous_dates():
    s = sc.parse_sheet_text(M1458_TEXT)
    # 2027 is the *recommended* re-calibration; must not be taken as the cal date.
    assert s.cal_date == date(2026, 6, 19)
    # current sensitivity is not the "Previous Sensitivity" value
    assert s.sensitivity == pytest.approx(0.0777)


def test_parse_unusable_when_fields_missing():
    s = sc.parse_sheet_text("Probe SN: M9999\n(no calibration numbers here)")
    assert s.sn == "M9999"
    assert not s.is_usable()


@pytest.mark.parametrize(
    "name,sn,d",
    [
        ("M1458_2026_06_19].pdf", "M1458", date(2026, 6, 19)),  # stray ']'
        ("M3039_2024_07_09.pdf", "M3039", date(2024, 7, 9)),
        ("M2502_2026_06_19.pdf", "M2502", date(2026, 6, 19)),
    ],
)
def test_parse_filename(name, sn, d):
    got = sc.parse_filename(name)
    assert got == (sn, d)


def test_parse_filename_none_when_no_match():
    assert sc.parse_filename("not-a-sheet.pdf") is None


# ---------------------------------------------------------------------------
# Timeline / hold-previous lookup
# ---------------------------------------------------------------------------


def _timeline(sn="M1458"):
    pts = [
        sc.CalPoint(date(2021, 4, 27), 0.0679, "old"),
        sc.CalPoint(date(2026, 6, 19), 0.0777, "new"),
    ]
    return sc.CalTimeline(sn, sorted(pts, key=lambda p: p.date))


def test_sensitivity_hold_previous_between_cals():
    s, gov, status = _timeline().sensitivity_at(date(2023, 1, 1))
    assert s == pytest.approx(0.0679)  # the *previous* value, not interpolated
    assert status == "in-effect"
    assert gov.date == date(2021, 4, 27)


def test_sensitivity_on_and_after_latest_cal():
    tl = _timeline()
    assert tl.sensitivity_at(date(2026, 6, 19))[0] == pytest.approx(0.0777)  # exact date
    assert tl.sensitivity_at(date(2027, 1, 1))[0] == pytest.approx(0.0777)  # after latest


def test_sensitivity_before_earliest_clamps_and_flags():
    s, gov, status = _timeline().sensitivity_at(date(2019, 1, 1))
    assert s == pytest.approx(0.0679)
    assert status == "before-earliest"
    assert gov.date == date(2021, 4, 27)


# ---------------------------------------------------------------------------
# Checking .p-file uses against the timelines
# ---------------------------------------------------------------------------


def _use(sn, sens, day=0, *, kind="shear", channel="sh1", dated=True):
    st = datetime(2023, 1, 1, tzinfo=UTC) + timedelta(days=day) if dated else None
    return si.SensorUse(
        kind=kind,
        path=Path(f"f{day}.p"),
        start_time=st,
        vehicle="VMP",
        platform_sn="479",
        channel=channel,
        sensor_sn=sn,
        params={"sens": sens},
    )


def test_check_flags_mismatch_beyond_tolerance():
    tls = {"M1458": _timeline()}
    # obs in 2023 -> in-effect sens is 0.0679; a config of 0.0777 is +14.4%.
    summary = sc.check_uses([_use("M1458", "0.0777")], tls)
    assert summary.n_checked == 1
    assert not summary.no_sheet
    m = summary.mismatches(1.0)
    assert len(m) == 1
    assert m[0].expected == pytest.approx(0.0679)
    assert m[0].pct_diff == pytest.approx(14.43, abs=0.05)


def test_check_no_mismatch_within_tolerance():
    tls = {"M1458": _timeline()}
    summary = sc.check_uses([_use("M1458", "0.0679")], tls)  # matches in-effect cal
    assert summary.n_checked == 1
    assert summary.mismatches(1.0) == []


def test_check_counts_no_sheet_undated_and_blank():
    tls = {"M1458": _timeline()}
    uses = [
        _use("M9999", "0.10"),  # no sheet
        _use("M1458", "", day=1),  # blank sens
        _use("M1458", "0.0777", day=2, dated=False),  # undated
        _use("M1458", "0.0777", kind="fp07"),  # not a shear use -> ignored
    ]
    summary = sc.check_uses(uses, tls)
    assert summary.no_sheet == {"M9999"}
    assert summary.n_no_sens == 1
    assert summary.n_undated == 1
    assert summary.n_checked == 0


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_format_check_reports_only_mismatches():
    tls = {"M1458": _timeline()}
    summary = sc.check_uses([_use("M1458", "0.0777")], tls)
    text = "\n".join(sc.format_check(summary, Path("/cal"), 1.0))
    assert "Shear calibration check" in text
    assert "M1458" in text
    assert "+14.4%" in text


def test_format_check_clean_pass_message():
    tls = {"M1458": _timeline()}
    summary = sc.check_uses([_use("M1458", "0.0679")], tls)
    text = "\n".join(sc.format_check(summary, Path("/cal"), 1.0))
    assert "No mismatches" in text


def test_format_check_marks_before_earliest():
    tls = {"M1458": _timeline()}
    # obs in 2019 (before earliest 2021 cal) with a mismatching config
    u = _use("M1458", "0.0777")
    u.start_time = datetime(2019, 1, 1, tzinfo=UTC)
    summary = sc.check_uses([u], tls)
    text = "\n".join(sc.format_check(summary, Path("/cal"), 1.0))
    assert "before earliest cal" in text


# ---------------------------------------------------------------------------
# PDF extraction (needs pypdf; skip if the 'cal' extra is absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CAL_PDF.exists(), reason="calibration fixture PDF missing")
def test_load_cal_dir_end_to_end(tmp_path):
    pytest.importorskip("pypdf")
    # Point at a dir with only the fixture so glob is deterministic.
    d = tmp_path / "sheets"
    d.mkdir()
    (d / CAL_PDF.name).write_bytes(CAL_PDF.read_bytes())

    tls, warns = sc.load_cal_dir(d)
    assert warns == []
    assert set(tls) == {"M1458"}
    pts = {(p.date, round(p.sensitivity, 4)) for p in tls["M1458"].points}
    assert pts == {(date(2021, 4, 27), 0.0679), (date(2026, 6, 19), 0.0777)}
    # hold-previous through the real timeline
    assert tls["M1458"].sensitivity_at(date(2023, 6, 1))[0] == pytest.approx(0.0679)


def test_extract_pdf_text_missing_pypdf(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "pypdf", None)  # force ImportError on `from pypdf ...`
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    with pytest.raises(sc.CalDependencyError):
        sc.extract_pdf_text(pdf)


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_run_cal_dir_not_a_directory(tmp_path, capsys):
    p = Path(__file__).parent / "data" / "SN479_0006.p"
    if not p.exists():
        pytest.skip("SN479 fixture missing")
    # A real .p file gets run() past the "no files" check, so a missing cal_dir
    # reaches the is_dir() guard.
    code = si.run([p], si.resolve_kinds(shear=True), cal_dir=tmp_path / "does_not_exist")
    assert code == 1
    assert "not a directory" in capsys.readouterr().err


def test_run_reports_no_sheet_for_unmatched_probe(tmp_path, capsys):
    pytest.importorskip("pypdf")
    p = Path(__file__).parent / "data" / "SN479_0006.p"
    if not p.exists():
        pytest.skip("SN479 fixture missing")
    d = tmp_path / "sheets"
    d.mkdir()
    (d / CAL_PDF.name).write_bytes(CAL_PDF.read_bytes())
    code = si.run([p], si.resolve_kinds(shear=True), cal_dir=d)
    out = capsys.readouterr().out
    assert "Shear calibration check" in out
    # SN479's probes are M2732/M2746 -> no sheet, nothing checked
    assert "no calibration sheet" in out
    assert code == 0
