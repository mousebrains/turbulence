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


def test_parse_recal_due():
    assert sc.parse_sheet_text(M1458_TEXT).recal_due == date(2027, 6, 19)
    assert sc.parse_sheet_text(M3039_TEXT).recal_due == date(2025, 7, 9)


def test_parse_recal_due_ignores_dateless_recommendation_lines():
    # Real sheets carry dateless prose ("Rockland recommends re-calibrating
    # shear probes ...", "Frequent re-calibration is strongly ..."); only the
    # anchored 're-?calibration' line WITH a date may set recal_due.
    text = (
        "Probe SN: M1111\n"
        "Sensitivity (sens or S): 0.08\n"
        "Frequent re-calibration is strongly\n"
        "Rockland recommends re-calibrating shear probes annually\n"
        "Calibration Date: 2024/01/02\n"
    )
    s = sc.parse_sheet_text(text)
    assert s.recal_due is None
    assert s.cal_date == date(2024, 1, 2)


def test_parse_recal_due_rejects_prose_glued_to_stray_date():
    """pypdf line-gluing can merge a prose recommendation with an unrelated
    date; the label-anchored regex must not take it, and a recal_due on/before
    the calibration date is discarded as a mis-parse."""
    text = (
        "Probe SN: M2222\n"
        "Sensitivity (sens or S): 0.08\n"
        "Frequent re-calibration is strongly recommended. Date: 2016/04/19\n"
        "Calibration Date: 2024/01/02\n"
    )
    s = sc.parse_sheet_text(text)
    assert s.recal_due is None
    assert s.cal_date == date(2024, 1, 2)

    # Even a label-anchored line whose date is on/before cal_date is rejected.
    text2 = (
        "Probe SN: M3333\n"
        "Sensitivity (sens or S): 0.08\n"
        "Calibration Date: 2024/01/02\n"
        "Recommended re-calibration: 2023/01/02\n"
    )
    s2 = sc.parse_sheet_text(text2)
    assert s2.recal_due is None
    assert s2.cal_date == date(2024, 1, 2)


def test_points_carry_recal_due_on_current_only():
    s = sc.parse_sheet_text(M1458_TEXT)
    by_date = {p.date: p for p in s.points()}
    assert by_date[date(2026, 6, 19)].recal_due == date(2027, 6, 19)
    # The sheet never states a recal date for the PREVIOUS calibration.
    assert by_date[date(2021, 4, 27)].recal_due is None


def test_parse_sn_not_swallowed_from_glued_caption():
    # A page-2 plot caption glues the serial to the fall-rate variable, e.g.
    # "...ProbeSN:M1254U=0.703m/s...".  The serial must come out as M1254, not
    # M1254U (the trailing 'U' belongs to the U=... fall rate, not the probe).
    s = sc.parse_sheet_text("Calibrationdate:2026-04-28 ProbeSN:M1254U=0.703 ms!1")
    assert s.sn == "M1254"


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
    s, gov, status, _stale = _timeline().sensitivity_at(date(2023, 1, 1))
    assert s == pytest.approx(0.0679)  # the *previous* value, not interpolated
    assert status == "in-effect"
    assert gov.date == date(2021, 4, 27)


def test_sensitivity_on_and_after_latest_cal():
    tl = _timeline()
    assert tl.sensitivity_at(date(2026, 6, 19))[0] == pytest.approx(0.0777)  # exact date
    assert tl.sensitivity_at(date(2027, 1, 1))[0] == pytest.approx(0.0777)  # after latest


def test_sensitivity_before_earliest_clamps_and_flags():
    s, gov, status, stale = _timeline().sensitivity_at(date(2019, 1, 1))
    assert s == pytest.approx(0.0679)
    assert status == "before-earliest"
    assert gov.date == date(2021, 4, 27)
    assert stale is None  # before-earliest is never additionally marked stale


def test_sensitivity_at_empty_timeline_raises():
    tl = sc.CalTimeline("M0000", [])
    with pytest.raises(ValueError, match="no calibration points"):
        tl.sensitivity_at(date(2023, 1, 1))


# ---------------------------------------------------------------------------
# Staleness (m2): recommended-recal date, or the max-age fallback
# ---------------------------------------------------------------------------


def test_cal_staleness_uses_sheet_recal_date():
    gov = sc.CalPoint(date(2024, 7, 9), 0.1189, "s", recal_due=date(2025, 7, 9))
    assert sc.cal_staleness(gov, date(2025, 7, 9)) is None  # on the due date: fresh
    st = sc.cal_staleness(gov, date(2025, 8, 1))
    assert st is not None
    assert st.recal_due == date(2025, 7, 9)
    assert st.max_age_months is None
    assert st.months_old == 12
    assert "recal was recommended by 2025-07-09" in st.describe()
    assert "verify no newer sheet exists" in st.describe()


def test_cal_staleness_fallback_max_age_only_without_sheet_line():
    gov = sc.CalPoint(date(2024, 1, 15), 0.1, "s")  # sheet lacked the recal line
    assert sc.cal_staleness(gov, date(2025, 1, 15)) is None  # exactly 12 months: fresh
    st = sc.cal_staleness(gov, date(2025, 1, 16))
    assert st is not None
    assert st.recal_due is None
    assert st.max_age_months == 12
    assert "older than the 12-month max age" in st.describe()
    # The CLI age is only the fallback — a wider age keeps it fresh ...
    assert sc.cal_staleness(gov, date(2025, 1, 16), max_age_months=24) is None
    # ... but never overrides a sheet that DOES carry the line.
    with_line = sc.CalPoint(date(2024, 1, 15), 0.1, "s", recal_due=date(2025, 1, 15))
    assert sc.cal_staleness(with_line, date(2025, 1, 16), max_age_months=24) is not None


def test_sensitivity_at_reports_staleness():
    pts = [sc.CalPoint(date(2021, 4, 27), 0.0679, "old", recal_due=date(2022, 4, 27))]
    tl = sc.CalTimeline("M1458", pts)
    assert tl.sensitivity_at(date(2021, 6, 1))[3] is None  # fresh
    stale = tl.sensitivity_at(date(2023, 1, 1))[3]
    assert stale is not None
    assert stale.recal_due == date(2022, 4, 27)


def test_check_uses_counts_stale_and_honors_max_age():
    tls = {"M1458": _timeline()}  # points carry no recal_due -> fallback age
    # obs 2023-01-01, governing cal 2021-04-27 -> 20 whole months old
    summary = sc.check_uses([_use("M1458", "0.0679")], tls)
    assert summary.n_stale == 1
    assert summary.checks[0].stale is not None
    assert summary.checks[0].stale.months_old == 20
    # A 48-month fallback age keeps it fresh.
    summary = sc.check_uses([_use("M1458", "0.0679")], tls, max_age_months=48)
    assert summary.n_stale == 0


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
    # obs in 2023 -> in-effect sens is 0.0679; a config of 0.0777 differs by
    # +0.0098 in absolute sensitivity units (well beyond the 0.00005 tolerance).
    summary = sc.check_uses([_use("M1458", "0.0777")], tls)
    assert summary.n_checked == 1
    assert not summary.no_sheet
    m = summary.mismatches(0.00005)
    assert len(m) == 1
    assert m[0].expected == pytest.approx(0.0679)
    assert m[0].abs_diff == pytest.approx(0.0098)
    assert m[0].pct_diff == pytest.approx(14.43, abs=0.05)


def test_check_no_mismatch_within_tolerance():
    tls = {"M1458": _timeline()}
    # A 0.00004 drift (below the 0.00005 tolerance) is not flagged even though it
    # would be a nonzero percent difference.
    summary = sc.check_uses([_use("M1458", "0.06794")], tls)  # in-effect cal is 0.0679
    assert summary.n_checked == 1
    assert summary.mismatches(0.00005) == []


def test_check_lookup_is_case_insensitive():
    # Sheet timelines are keyed upper-case; a .p config with a lower-case serial
    # must still match (else it silently reads as "no calibration sheet").
    tls = {"M1458": _timeline()}
    summary = sc.check_uses([_use("m1458", "0.0679")], tls)
    assert summary.n_checked == 1
    assert summary.no_sheet == set()


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
    text = "\n".join(sc.format_check(summary, Path("/cal"), 0.00005))
    assert "Shear calibration check" in text
    assert "M1458" in text
    # Absolute delta is the flag metric; the tolerance renders plainly (no 5e-05).
    assert "Δ +0.0098" in text
    assert "±0.00005 (sensitivity units)" in text
    assert "+14.4%" in text  # percent still shown for context


def test_format_check_clean_pass_message():
    tls = {"M1458": _timeline()}
    summary = sc.check_uses([_use("M1458", "0.0679")], tls)
    text = "\n".join(sc.format_check(summary, Path("/cal"), 0.00005))
    assert "No mismatches" in text
    assert "within ±0.00005" in text


def test_format_check_nothing_checked_message():
    # A probe with no matching sheet means nothing was checked; the report must
    # say so rather than claiming "all 0 observation(s) agree" (reads as a pass).
    tls = {"M1458": _timeline()}
    summary = sc.check_uses([_use("M9999", "0.10")], tls)
    assert summary.n_checked == 0
    text = "\n".join(sc.format_check(summary, Path("/cal"), 0.00005))
    assert "No shear observations were checked" in text
    assert "all 0" not in text


def test_format_check_marks_before_earliest():
    tls = {"M1458": _timeline()}
    # obs in 2019 (before earliest 2021 cal) with a mismatching config
    u = _use("M1458", "0.0777")
    u.start_time = datetime(2019, 1, 1, tzinfo=UTC)
    summary = sc.check_uses([u], tls)
    text = "\n".join(sc.format_check(summary, Path("/cal"), 0.00005))
    assert "before earliest cal" in text
    assert "stale" not in text  # before-earliest is never additionally stale


def test_format_check_annotates_stale_mismatch_rows():
    tls = {"M1458": _timeline()}  # no recal line on the points -> 12-month fallback
    # obs 2023-01-01 governed by the 2021-04-27 cal (20 months old), mismatching
    summary = sc.check_uses([_use("M1458", "0.0777")], tls)
    text = "\n".join(sc.format_check(summary, Path("/cal"), 0.00005))
    assert "[cal 20 months old at use; older than the 12-month max age" in text
    assert "verify no newer sheet exists" in text
    assert "1 of 1 checked observation(s) governed by stale calibrations" in text


def test_format_check_stale_annotation_quotes_recal_date():
    pts = [sc.CalPoint(date(2021, 4, 27), 0.0679, "old", recal_due=date(2022, 4, 27))]
    tls = {"M1458": sc.CalTimeline("M1458", pts)}
    summary = sc.check_uses([_use("M1458", "0.0777")], tls)
    text = "\n".join(sc.format_check(summary, Path("/cal"), 0.00005))
    assert "recal was recommended by 2022-04-27" in text


def test_format_check_no_mismatch_reports_stale_count():
    tls = {"M1458": _timeline()}
    summary = sc.check_uses([_use("M1458", "0.0679")], tls)  # agrees, but stale
    text = "\n".join(sc.format_check(summary, Path("/cal"), 0.00005))
    assert "No mismatches" in text
    assert "(1 observation(s) governed by stale calibrations)" in text


def test_format_check_fresh_observation_no_stale_noise():
    tls = {"M1458": _timeline()}
    u = _use("M1458", "0.0679")
    u.start_time = datetime(2021, 6, 1, tzinfo=UTC)  # 1 month after the 2021 cal
    summary = sc.check_uses([u], tls)
    text = "\n".join(sc.format_check(summary, Path("/cal"), 0.00005))
    assert "No mismatches" in text
    assert "stale" not in text


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
    # The real sheet's "Recommended re-calibration: 2027/06/19" is parsed onto
    # the current calibration point (m2 staleness guard).
    by_date = {p.date: p for p in tls["M1458"].points}
    assert by_date[date(2026, 6, 19)].recal_due == date(2027, 6, 19)
    assert by_date[date(2021, 4, 27)].recal_due is None


@pytest.mark.parametrize("old_name", ["M1458_2021_04_27.pdf", "Z-M1458_2021_04_27.pdf"])
def test_load_cal_dir_dedup_prefers_recal_due(monkeypatch, tmp_path, old_name):
    """A newer sheet's "previous" point repeats an older sheet's current point
    WITHOUT its recommended-recal date; whichever sheet order glob yields, the
    merged timeline must keep the copy that carries recal_due."""
    old_sheet = (
        "Probe SN: M1458\n"
        "Sensitivity (sens or S): 0.0679 V\n"
        "Calibration Date: 2021/04/27\n"
        "Recommended re-calibration: 2022/04/27\n"
    )
    texts = {old_name: old_sheet, "M1458_2026_06_19.pdf": M1458_TEXT}
    monkeypatch.setattr(sc, "extract_pdf_text", lambda p: texts[p.name])
    d = tmp_path / "sheets"
    d.mkdir()
    for name in texts:
        (d / name).write_bytes(b"%PDF-1.4\n")

    tls, _warns = sc.load_cal_dir(d)
    by_date = {p.date: p for p in tls["M1458"].points}
    assert by_date[date(2021, 4, 27)].recal_due == date(2022, 4, 27)
    assert by_date[date(2026, 6, 19)].recal_due == date(2027, 6, 19)


def test_load_cal_dir_fills_sn_and_date_from_filename(monkeypatch, tmp_path):
    # PDF text with a garbled/absent "Probe SN:" line is still usable: the serial
    # (and, if missing, the date) come from the M<sn>_<Y>_<M>_<D>.pdf filename.
    text = "Sensitivity (sens or S): 0.0777 V\nCalibration Date: 2026/06/19\n"
    monkeypatch.setattr(sc, "extract_pdf_text", lambda p: text)
    d = tmp_path / "sheets"
    d.mkdir()
    (d / "M1458_2026_06_19.pdf").write_bytes(b"%PDF-1.4\n")

    tls, warns = sc.load_cal_dir(d)
    assert set(tls) == {"M1458"}
    assert warns == []
    assert tls["M1458"].sensitivity_at(date(2026, 6, 19))[0] == pytest.approx(0.0777)


def test_load_cal_dir_warns_on_empty_dir(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    tls, warns = sc.load_cal_dir(d)  # no PDFs -> pypdf never touched
    assert tls == {}
    assert any("no *.pdf" in w for w in warns)


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
    # The --cal-dir directory check is a pre-scan fail-fast, so it fires before
    # any .p file is read (no fixture needed).
    code = si.run(
        [tmp_path / "x.p"], si.resolve_kinds(shear=True), cal_dir=tmp_path / "does_not_exist"
    )
    assert code == 1
    assert "not a directory" in capsys.readouterr().err


def test_run_warns_when_cal_dir_but_shear_not_scanned(tmp_path, capsys):
    # --cal-dir with an fp07-only scan checks nothing; the user must be warned.
    d = tmp_path / "sheets"
    d.mkdir()
    si.run([tmp_path / "x.p"], si.resolve_kinds(fp07=True), cal_dir=d)
    assert "shear channels are not being scanned" in capsys.readouterr().err


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


# ---------------------------------------------------------------------------
# --cal-strict exit codes (m3).  The synthetic-sheet texts stand in for the
# gitignored microstructure_sensors/ directory: extract_pdf_text is
# monkeypatched, so no pypdf (and no real PDF) is needed.  MR_SL435.p has
# sh1 = M3038 configured with sens 0.1041 on 2025-02-12.
# ---------------------------------------------------------------------------

MR_FIXTURE = Path(__file__).parent / "data" / "MR_SL435.p"

M3038_MISMATCH_TEXT = """
Probe SN: M3038
Sensitivity (sens or S): 0.0950 V
Calibration Date: 2024/07/09
Recommended re-calibration: 2025/07/09
"""

M3038_MATCH_TEXT = """
Probe SN: M3038
Sensitivity (sens or S): 0.1041 V
Calibration Date: 2024/07/09
Recommended re-calibration: 2025/07/09
"""


def _synthetic_cal_dir(monkeypatch, tmp_path, text):
    monkeypatch.setattr(sc, "extract_pdf_text", lambda p: text)
    d = tmp_path / "sheets"
    d.mkdir()
    (d / "M3038_2024_07_09.pdf").write_bytes(b"%PDF-1.4\n")
    return d


def test_run_cal_mismatch_default_is_report_only(monkeypatch, tmp_path, capsys):
    if not MR_FIXTURE.exists():
        pytest.skip("MR fixture missing")
    d = _synthetic_cal_dir(monkeypatch, tmp_path, M3038_MISMATCH_TEXT)
    code = si.run([MR_FIXTURE], si.resolve_kinds(shear=True), cal_dir=d)
    assert code == 0  # default behavior unchanged: mismatches reported, exit 0
    assert "mismatching observation(s)" in capsys.readouterr().out


def test_run_cal_strict_exits_3_on_mismatch(monkeypatch, tmp_path, capsys):
    if not MR_FIXTURE.exists():
        pytest.skip("MR fixture missing")
    d = _synthetic_cal_dir(monkeypatch, tmp_path, M3038_MISMATCH_TEXT)
    code = si.run([MR_FIXTURE], si.resolve_kinds(shear=True), cal_dir=d, cal_strict=True)
    assert code == 3
    # The report is still fully printed before the strict exit.
    assert "mismatching observation(s)" in capsys.readouterr().out


def test_run_cal_strict_exit_0_when_within_tolerance(monkeypatch, tmp_path):
    if not MR_FIXTURE.exists():
        pytest.skip("MR fixture missing")
    d = _synthetic_cal_dir(monkeypatch, tmp_path, M3038_MATCH_TEXT)
    code = si.run([MR_FIXTURE], si.resolve_kinds(shear=True), cal_dir=d, cal_strict=True)
    assert code == 0


def test_run_cal_strict_requires_cal_dir(capsys):
    code = si.run([Path("x.p")], si.resolve_kinds(shear=True), cal_strict=True)
    assert code == 1
    assert "requires --cal-dir" in capsys.readouterr().err


def test_sensor_inventory_main_wires_cal_strict(monkeypatch, tmp_path):
    """build_arg_parser must carry --cal-strict / --cal-max-age-months (F6)."""
    if not MR_FIXTURE.exists():
        pytest.skip("MR fixture missing")
    d = _synthetic_cal_dir(monkeypatch, tmp_path, M3038_MISMATCH_TEXT)
    code = si.main(
        [str(MR_FIXTURE), "--shear", "--cal-dir", str(d), "--cal-strict",
         "--cal-max-age-months", "6"]
    )
    assert code == 3


def test_rsi_cli_sensors_wires_cal_strict(monkeypatch, tmp_path):
    """The rsi-tpw sensors parser must carry the same flags (F6)."""
    if not MR_FIXTURE.exists():
        pytest.skip("MR fixture missing")
    d = _synthetic_cal_dir(monkeypatch, tmp_path, M3038_MISMATCH_TEXT)
    monkeypatch.setattr(
        sys,
        "argv",
        ["rsi-tpw", "sensors", str(MR_FIXTURE), "--shear", "--cal-dir", str(d), "--cal-strict"],
    )
    from odas_tpw.rsi.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 3
