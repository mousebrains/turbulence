# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for the microstructure sensor inventory tool and its rsi-tpw wiring."""

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
    # cal_date is not numeric -> compared as string
    assert si._norm("cal_date", "2025-01-05") == "2025-01-05"


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
    assert (
        si._fmt_platform("slocum_glider", "435", {"sh2", "sh1"})
        == "slocum_glider SN 435 (as sh1, sh2)"
    )


# ---------------------------------------------------------------------------
# File discovery + error handling
# ---------------------------------------------------------------------------


def test_iter_pfiles_walks_tree_and_dedups(tmp_path, sample_p_file):
    (tmp_path / "sub").mkdir()
    a = tmp_path / "a.p"
    b = tmp_path / "sub" / "b.p"
    shutil.copy(sample_p_file, a)
    shutil.copy(sample_p_file, b)
    (tmp_path / "notes.txt").write_text("ignore me")
    found = si.iter_pfiles([tmp_path])
    assert found == sorted([a, b])


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


def test_run_no_files_returns_one(tmp_path, capsys):
    code = si.run([tmp_path], si.resolve_kinds(want_all=True))
    assert code == 1


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
