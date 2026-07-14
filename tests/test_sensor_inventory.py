# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for the microstructure sensor inventory tool and its rsi-tpw wiring."""

import csv
import io
import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from odas_tpw.rsi import sensor_inventory as si

# The tracked fixture (SN479 VMP, ARCTERX Wake) carries shear probes M2732 (sh1)
# and M2746 (sh2) plus FP07 thermistors on T1/T2 with the placeholder sn "T".
#
# MR_FIXTURE is a trimmed real glider-microrider file (ARCTERX 2025 Interior,
# osu685) carrying shear M3038/M3039 and FP07 T2811/T2813 on vehicle
# "slocum_glider" SN 435 — distinct SNs, so no placeholder collision.
MR_FIXTURE = Path(__file__).parent / "data" / "MR_SL435.p"

# STARTUP_FIXTURE is a trimmed real VMP-142 first-of-deployment file whose clock
# was not yet set: header year 0 and empty instrument_info, but a valid config
# with shear M1254/M1218 and FP07 T2007/T1592. Exercises the undated path.
STARTUP_FIXTURE = Path(__file__).parent / "data" / "VMP142_startup_noclock.p"


# ---------------------------------------------------------------------------
# Kind selection
# ---------------------------------------------------------------------------


def test_resolve_kinds_default_is_all():
    assert si.resolve_kinds() == list(si.SENSOR_KINDS)
    assert si.resolve_kinds(want_all=True) == list(si.SENSOR_KINDS)


def test_resolve_kinds_individual_and_combined():
    assert si.resolve_kinds(shear=True) == ["shear"]
    assert si.resolve_kinds(fp07=True) == ["fp07"]
    assert si.resolve_kinds(shear=True, fp07=True) == ["shear", "fp07"]


def test_resolve_kinds_all_overrides_individual():
    # --all wins over a specific flag rather than duplicating or dropping kinds.
    assert si.resolve_kinds(shear=True, want_all=True) == list(si.SENSOR_KINDS)


# ---------------------------------------------------------------------------
# scan_file
# ---------------------------------------------------------------------------


def test_scan_file_shear(sample_p_file):
    uses = si.scan_file(sample_p_file, ["shear"])
    by_sn = {u.sensor_sn: u for u in uses}
    assert set(by_sn) == {"M2732", "M2746"}

    sh1 = by_sn["M2732"]
    assert sh1.kind == "shear"
    assert sh1.channel == "sh1"
    assert sh1.vehicle == "VMP"
    assert sh1.platform_sn == "479"
    assert sh1.params["sens"] == "0.1075"
    assert sh1.params["diff_gain"] == "0.954"
    assert sh1.start_time.tzinfo is UTC


def test_scan_file_fp07_excludes_preemphasis(sample_p_file):
    """Only the base T1/T2 channels count; T1_dT1/T2_dT2 are pre-emphasis."""
    uses = si.scan_file(sample_p_file, ["fp07"])
    channels = sorted(u.channel for u in uses)
    assert channels == ["T1", "T2"]
    assert all(not c.channel.endswith(("_dT1", "_dT2")) for c in uses)
    t1 = next(u for u in uses if u.channel == "T1")
    assert t1.kind == "fp07"
    assert t1.params["a"] == "-16.3"
    assert t1.params["b"] == "0.99861"


def test_scan_file_all_covers_both_kinds(sample_p_file):
    kinds = si.resolve_kinds(want_all=True)
    got = {(u.kind, u.channel) for u in si.scan_file(sample_p_file, kinds)}
    assert ("shear", "sh1") in got
    assert ("shear", "sh2") in got
    assert ("fp07", "T1") in got
    assert ("fp07", "T2") in got


def test_scan_file_mr_glider():
    """Real microrider-on-glider fixture: non-VMP vehicle, distinct FP07 SNs."""
    uses = si.scan_file(MR_FIXTURE, si.resolve_kinds(want_all=True))
    assert {u.sensor_sn for u in uses if u.kind == "shear"} == {"M3038", "M3039"}
    assert {u.sensor_sn for u in uses if u.kind == "fp07"} == {"T2811", "T2813"}
    u0 = uses[0]
    assert u0.vehicle == "slocum_glider"
    assert u0.platform_sn == "435"
    assert u0.start_time is not None


# ---------------------------------------------------------------------------
# Aggregation / change detection
# ---------------------------------------------------------------------------


def _mk(day, kind, sn, params, *, vehicle="VMP", psn="479", channel="sh1"):
    base = datetime(2025, 1, 14, tzinfo=UTC)
    return si.SensorUse(
        kind=kind,
        path=f"f{day}.p",
        start_time=base + timedelta(days=day),
        vehicle=vehicle,
        platform_sn=psn,
        channel=channel,
        sensor_sn=sn,
        params=params,
    )


def test_build_inventory_groups_by_kind_and_sn(sample_p_file):
    kinds = si.resolve_kinds(want_all=True)
    uses, errors, n_no = si.collect_uses([sample_p_file], kinds)
    assert errors == []
    assert n_no == 0
    inv = si.build_inventory(uses)
    assert set(inv["shear"]) == {"M2732", "M2746"}
    assert inv["shear"]["M2732"].files == {sample_p_file}
    # FP07 placeholder SN "T" merges T1 and T2 under one entry.
    assert "T" in inv["fp07"]
    assert inv["fp07"]["T"].platforms[("VMP", "479")] == {"T1", "T2"}


def test_change_detection_flags_real_changes_not_formatting():
    uses = [
        _mk(0, "shear", "M1", {"sens": "0.1075", "diff_gain": "0.954"}),
        _mk(1, "shear", "M1", {"sens": "0.1075", "diff_gain": "0.9540"}),  # cosmetic
        _mk(5, "shear", "M1", {"sens": "0.1120", "diff_gain": "0.954"}),  # real change
    ]
    agg = si.build_inventory(uses)["shear"]["M1"]
    assert len(agg.params["diff_gain"]) == 1  # 0.954 == 0.9540
    assert len(agg.params["sens"]) == 2  # 0.1075 -> 0.1120


def test_norm_numeric_vs_string():
    assert si._norm("diff_gain", "0.954") == si._norm("diff_gain", "0.9540")
    assert si._norm("sens", "0.1075") != si._norm("sens", "0.1120")
    assert si._norm("adc_bits", "16") == si._norm("adc_bits", "16.0")
    assert si._norm("diff_gain", "1e-3") == si._norm("diff_gain", "0.001")
    # "0" is numeric (0.0) and must stay distinct from missing ("")
    assert si._norm("sens", "0") != si._norm("sens", "")
    # cal_date is not numeric -> compared as string
    assert si._norm("cal_date", "2025-01-05") == "2025-01-05"


def test_norm_non_finite_falls_back_to_string():
    # nan/inf from a corrupt config must not raise and must not false-merge/change.
    assert si._norm("sens", "nan") == si._norm("sens", "nan")  # identical nan -> equal
    assert si._norm("sens", "1e400") != si._norm("sens", "1e999")  # both inf -> stay distinct
    assert si._norm("adc_bits", "inf") == "inf"  # int(float("inf")) would raise; must not


def test_cross_platform_grouping_by_serial():
    """A probe reused on a different platform stays one entry with both platforms."""
    uses = [
        _mk(0, "shear", "M9", {"sens": "0.1"}, vehicle="VMP", psn="479", channel="sh1"),
        _mk(9, "shear", "M9", {"sens": "0.1"}, vehicle="slocum_glider", psn="435", channel="sh2"),
    ]
    agg = si.build_inventory(uses)["shear"]["M9"]
    assert set(agg.platforms) == {("VMP", "479"), ("slocum_glider", "435")}
    assert len(agg.files) == 2


# ---------------------------------------------------------------------------
# Unset-clock startup files (MR/glider powered up before the host set the time)
# ---------------------------------------------------------------------------


def test_start_time_none_on_unset_clock():
    good = {
        "year": 2025, "month": 1, "day": 14, "hour": 15,
        "minute": 30, "second": 16, "millisecond": 445, "timezone_min": 0,
    }  # fmt: skip
    assert si._start_time_utc(good, {}) is not None
    zeroed = {
        "year": 0, "month": 36123, "day": 23115, "hour": 0,
        "minute": 0, "second": 0, "millisecond": 0, "timezone_min": 0,
    }  # fmt: skip
    assert si._start_time_utc(zeroed, {}) is None


def test_start_time_utc_concrete_value_with_recsize_and_tz():
    # Header time is the END of record 0; data start = header time - recsize.
    # Local 12:00:00.500 at +60 min == 11:00:00.500 UTC, minus 2 s recsize.
    h = {
        "year": 2025, "month": 6, "day": 15, "hour": 12,
        "minute": 0, "second": 0, "millisecond": 500, "timezone_min": 60,
    }  # fmt: skip
    st = si._start_time_utc(h, {"root": {"recsize": "2.0"}})
    assert st == datetime(2025, 6, 15, 10, 59, 58, 500000, tzinfo=UTC)

    # Negative offset is stored two's-complement: 65236 -> -300 min (UTC-5).
    h_west = {**h, "timezone_min": 2**16 - 300, "millisecond": 0}
    st_west = si._start_time_utc(h_west, {})  # default recsize 1.0 s
    assert st_west == datetime(2025, 6, 15, 16, 59, 59, tzinfo=UTC)


def test_start_time_utc_robust_to_overflow():
    base = {
        "year": 2025, "month": 1, "day": 1, "hour": 0,
        "minute": 0, "second": 0, "millisecond": 0, "timezone_min": 0,
    }  # fmt: skip
    # A garbage 'recsize = 1e999' -> inf must not raise; it falls back to 1.0 s.
    assert si._start_time_utc(base, {"root": {"recsize": "1e999"}}) == datetime(
        2024, 12, 31, 23, 59, 59, tzinfo=UTC
    )
    # A valid datetime at the range floor minus recsize underflows -> None, not a crash.
    floor = {**base, "year": 1}
    assert si._start_time_utc(floor, {}) is None


def test_undated_use_is_inventoried_without_date():
    use = si.SensorUse(
        kind="shear",
        path="startup.p",
        start_time=None,
        vehicle="?",
        platform_sn="?",
        channel="sh1",
        sensor_sn="M9",
        params={"sens": "0.1", "cal_date": ""},
    )
    agg = si.build_inventory([use])["shear"]["M9"]
    assert agg.files == {"startup.p"}  # still counted
    assert agg.first is None and agg.last is None  # no date contributed
    assert agg.platforms[("?", "?")] == {"sh1"}
    assert si._fmt_range(agg.first, agg.last) == "(no valid timestamp)"


def test_fmt_platform_handles_missing_info():
    assert si._fmt_platform("?", "?", {"sh1"}) == "unknown platform (as sh1)"
    assert si._fmt_platform("VMP", "?", {"sh1"}) == "VMP (SN unknown) (as sh1)"
    assert si._fmt_platform("?", "435", {"sh1"}) == "SN 435 (as sh1)"  # vehicle unknown, SN known
    assert (
        si._fmt_platform("slocum_glider", "435", {"sh2", "sh1"})
        == "slocum_glider SN 435 (as sh1, sh2)"
    )


def test_platform_from_instrument_fallbacks():
    assert si._platform_from_instrument({"vehicle": "VMP", "sn": "479"}) == ("VMP", "479")
    # whitespace-only vehicle falls through to model, not "" (which would split grouping)
    assert si._platform_from_instrument({"vehicle": "  ", "model": "MR1000", "sn": "1"}) == (
        "MR1000",
        "1",
    )
    assert si._platform_from_instrument({}) == ("?", "?")
    assert si._platform_from_instrument({"sn": "435"}) == ("?", "435")


# ---------------------------------------------------------------------------
# Report rendering (print_report) — the headline CHANGED / verbose behavior
# ---------------------------------------------------------------------------


def _render(uses, kinds, verbose=False, compact=False):
    buf = io.StringIO()
    si.print_report(si.build_inventory(uses), kinds, verbose=verbose, compact=compact, stream=buf)
    return buf.getvalue()


def _use(path, start, sn="M2499", params=None, vehicle="slocum_glider", psn="433", channel="sh1"):
    """A SensorUse with an explicit path + timestamp (finer control than _mk)."""
    return si.SensorUse(
        kind="shear",
        path=path,
        start_time=start,
        vehicle=vehicle,
        platform_sn=psn,
        channel=channel,
        sensor_sn=sn,
        params=params or {},
    )


def test_print_report_changed_lists_values_oldest_first():
    uses = [
        _mk(0, "shear", "M1", {"sens": "0.10", "diff_gain": "0.95"}),
        _mk(1, "shear", "M1", {"sens": "0.10", "diff_gain": "0.95"}),
        _mk(9, "shear", "M1", {"sens": "0.12", "diff_gain": "0.95"}),  # sens changed later
    ]
    text = _render(uses, ["shear"])
    assert "Shear probes (1)" in text
    # unchanged parameter renders as constant
    dg_line = next(ln for ln in text.splitlines() if ln.strip().startswith("diff_gain:"))
    assert "0.95" in dg_line and "(constant)" in dg_line
    # changed parameter renders CHANGED with both values, oldest first, right counts
    assert "CHANGED" in text
    assert text.index("0.10") < text.index("0.12")
    assert "2 files" in text and "1 file" in text


def test_print_report_verbose_lists_the_files(tmp_path):
    uses = [
        _mk(0, "shear", "M1", {"sens": "0.10"}),
        _mk(9, "shear", "M1", {"sens": "0.12"}),
    ]
    plain = _render(uses, ["shear"], verbose=False)
    verbose = _render(uses, ["shear"], verbose=True)
    assert "f0.p" not in plain  # non-verbose does not list files
    assert "f0.p" in verbose and "f9.p" in verbose  # verbose lists them under each value


# ---------------------------------------------------------------------------
# Compact one-line-per-probe rendering (--compact)
# ---------------------------------------------------------------------------

_CAL = {
    "adc_fs": "4.096", "adc_bits": "16",
    "diff_gain": "0.927", "sens": "0.0817", "cal_date": "",
}  # fmt: skip


def _compact_lines(uses):
    text = _render(uses, ["shear"], compact=True)
    return [ln for ln in text.splitlines() if ln and ln[0] == "M"]


def test_compact_line_exact_shape():
    # Reproduces the requested target line exactly (two same-day files → HH:MM range).
    uses = [
        _use("a.p", datetime(2025, 1, 30, 1, 20, tzinfo=UTC), params=_CAL),
        _use("b.p", datetime(2025, 1, 30, 2, 40, tzinfo=UTC), params=_CAL),
    ]
    (line,) = _compact_lines(uses)
    assert line == (
        "M2499 #2 diff_gain: 0.927 sens: 0.0817 cal_date: (blank) used: 2025-01-30 01:20 → 02:40"
    )


def test_compact_omits_adc_platform_and_constant_markers():
    (line,) = _compact_lines([_use("a.p", datetime(2025, 1, 1, tzinfo=UTC), params=_CAL)])
    assert "adc_fs" not in line and "adc_bits" not in line  # ADC settings dropped
    assert "slocum_glider" not in line and "SN 433" not in line  # platform dropped
    assert "(constant)" not in line


def test_compact_joins_changed_values_oldest_first():
    p1 = {**_CAL, "diff_gain": "0.95"}
    p2 = {**_CAL, "diff_gain": "0.97"}
    uses = [
        _use("a.p", datetime(2025, 1, 1, tzinfo=UTC), params=p1),
        _use("b.p", datetime(2025, 1, 2, tzinfo=UTC), params=p2),
    ]
    (line,) = _compact_lines(uses)
    assert "diff_gain: 0.95→0.97" in line  # changed → arrow-joined, oldest first
    assert "sens: 0.0817 " in line  # unchanged param stays a single value


def test_compact_undated_probe():
    (line,) = _compact_lines([_use("a.p", None, params=_CAL)])
    assert line.endswith("used: (no valid timestamp)")


def test_cli_sensors_compact(monkeypatch, tmp_path, sample_p_file, capsys):
    p = tmp_path / "x.p"
    shutil.copy(sample_p_file, p)
    monkeypatch.setattr(sys, "argv", ["rsi-tpw", "sensors", "--shear", "--compact", str(p)])
    from odas_tpw.rsi.cli import main

    main()
    out = capsys.readouterr().out
    assert "date range:" not in out  # not the multi-line block
    assert any(ln.startswith("M2732") and "used:" in ln for ln in out.splitlines())


# ---------------------------------------------------------------------------
# File discovery + error handling
# ---------------------------------------------------------------------------


def test_iter_pfiles_walks_tree_finds_uppercase_and_ignores_others(tmp_path, sample_p_file):
    (tmp_path / "sub").mkdir()
    a = tmp_path / "a.p"
    b = tmp_path / "sub" / "b.P"  # uppercase .P must be found (rglob is case-sensitive on Linux)
    shutil.copy(sample_p_file, a)
    shutil.copy(sample_p_file, b)
    (tmp_path / "notes.txt").write_text("ignore me")
    assert si.iter_pfiles([tmp_path]) == sorted([a, b])


def test_iter_pfiles_dedups_overlapping_inputs(tmp_path, sample_p_file):
    a = tmp_path / "a.p"
    shutil.copy(sample_p_file, a)
    # dir + the same file explicitly + a directory-relative duplicate → exactly one
    assert si.iter_pfiles([tmp_path, a, tmp_path]) == [a]


def test_iter_pfiles_expands_glob_patterns(tmp_path, sample_p_file):
    a = tmp_path / "a.p"
    b = tmp_path / "b.p"
    shutil.copy(sample_p_file, a)
    shutil.copy(sample_p_file, b)
    (tmp_path / "c.txt").write_text("nope")
    # A non-existent literal that is a glob pattern is expanded (like sibling subcommands).
    assert si.iter_pfiles([tmp_path / "*.p"]) == sorted([a, b])


def test_iter_pfiles_warns_on_nonmatch(tmp_path, capsys):
    assert si.iter_pfiles([tmp_path / "nope*.p"]) == []
    assert "matched no files" in capsys.readouterr().err


def test_collect_uses_reports_bad_file_without_aborting(tmp_path, sample_p_file):
    good = tmp_path / "good.p"
    shutil.copy(sample_p_file, good)
    bad = tmp_path / "bad.p"
    bad.write_bytes(b"too short")  # < 128-byte header
    files = si.iter_pfiles([tmp_path])
    uses, errors, _ = si.collect_uses(files, si.resolve_kinds(want_all=True))
    assert any(u.path == good for u in uses)
    assert len(errors) == 1 and errors[0][0] == bad
    assert "too small for header" in errors[0][1]


# ---------------------------------------------------------------------------
# CSV + end-to-end run
# ---------------------------------------------------------------------------


def test_run_writes_csv_and_returns_zero(tmp_path, sample_p_file, capsys):
    csv_path = tmp_path / "probes.csv"
    code = si.run([sample_p_file], si.resolve_kinds(want_all=True), csv_out=csv_path)
    assert code == 0
    out = capsys.readouterr().out
    assert "Shear probes" in out
    assert "FP07 thermistors" in out
    assert "M2732" in out

    lines = csv_path.read_text().splitlines()
    header = lines[0].split(",")
    for col in ("sensor", "sn", "channel", "sens", "a", "cal_date"):
        assert col in header
    # 2 shear + 2 fp07 channels in one file
    assert len(lines) == 1 + 4
    assert any(",M2732," in row or row.split(",")[1] == "M2732" for row in lines[1:])


def test_csv_value_placement_and_blank_fill(tmp_path, sample_p_file):
    csv_path = tmp_path / "p.csv"
    si.run([sample_p_file], si.resolve_kinds(want_all=True), csv_out=csv_path, stream=io.StringIO())
    rows = list(csv.DictReader(csv_path.open()))

    shear_row = next(r for r in rows if r["sn"] == "M2732")
    assert shear_row["sensor"] == "shear"
    assert shear_row["channel"] == "sh1"
    assert shear_row["sens"] == "0.1075"  # value lands in the right column
    assert shear_row["a"] == ""  # fp07-only column is blank on a shear row

    fp07_row = next(r for r in rows if r["sensor"] == "fp07")
    assert fp07_row["a"] != ""  # fp07 carries its 'a' coefficient
    assert fp07_row["sens"] == ""  # shear-only column is blank on an fp07 row


def test_run_no_files_returns_one(tmp_path, capsys):
    code = si.run([tmp_path], si.resolve_kinds(want_all=True))
    assert code == 1


def test_run_returns_one_when_all_files_error(tmp_path, capsys):
    bad = tmp_path / "bad.p"
    bad.write_bytes(b"too short")
    code = si.run([bad], si.resolve_kinds(want_all=True))
    assert code == 1  # files present but none parsed -> non-zero
    assert "Errors:" in capsys.readouterr().out


def test_run_returns_one_on_unwritable_csv(tmp_path, sample_p_file, capsys):
    # csv target is a directory -> fail fast before scanning, clean error, exit 1
    code = si.run([sample_p_file], ["shear"], csv_out=tmp_path)
    assert code == 1
    assert "cannot write CSV" in capsys.readouterr().err


def test_run_reports_undated_startup_file(capsys):
    """End-to-end on the real no-clock startup fixture: inventoried, not dropped."""
    code = si.run([STARTUP_FIXTURE], si.resolve_kinds(want_all=True))
    assert code == 0
    out = capsys.readouterr().out
    assert "no valid clock" in out  # summary note
    assert "M1254" in out and "M1218" in out  # probes still found
    assert "unknown platform" in out  # empty instrument_info
    assert "(no valid timestamp)" in out  # date range for undated probe


def test_run_reports_no_sensor_files(monkeypatch, tmp_path, sample_p_file, capsys):
    f = tmp_path / "x.p"
    shutil.copy(sample_p_file, f)
    monkeypatch.setattr(si, "scan_file", lambda p, k: [])  # force "no matching sensors"
    code = si.run([f], ["shear"])
    assert code == 0  # a clean scan that simply found nothing is not an error
    assert "had no matching sensor channels" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# rsi-tpw CLI wiring
# ---------------------------------------------------------------------------


def test_cli_sensors_shear(monkeypatch, tmp_path, sample_p_file, capsys):
    p = tmp_path / "x.p"
    shutil.copy(sample_p_file, p)
    monkeypatch.setattr(sys, "argv", ["rsi-tpw", "sensors", "--shear", str(tmp_path)])
    from odas_tpw.rsi.cli import main

    main()
    out = capsys.readouterr().out
    assert "Shear probes" in out
    assert "M2732" in out
    assert "FP07 thermistors" not in out  # --shear only


def test_cli_sensors_csv_and_default_all(monkeypatch, tmp_path, sample_p_file, capsys):
    p = tmp_path / "x.p"
    shutil.copy(sample_p_file, p)
    csv_path = tmp_path / "out.csv"
    # No kind flag -> defaults to all kinds.
    monkeypatch.setattr(sys, "argv", ["rsi-tpw", "sensors", str(p), "--csv", str(csv_path)])
    from odas_tpw.rsi.cli import main

    main()
    out = capsys.readouterr().out
    assert "Shear probes" in out and "FP07 thermistors" in out
    assert csv_path.exists()


def test_cli_sensors_no_files_exits_one(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["rsi-tpw", "sensors", str(tmp_path)])
    from odas_tpw.rsi.cli import main

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
