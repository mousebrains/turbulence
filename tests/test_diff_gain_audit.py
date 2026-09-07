"""The differential-gain audit (`rsi-tpw sensors --diff-gain`).

`diff_gain` belongs to an instrument's amplifier chain, not to the probe bolted
into it, and epsilon goes as `(diff_gain*sens)^-2` while chi goes as
`diff_gain^-2` — the same leverage as the shear sensitivity. The sensor
inventory cannot report it usefully: it is keyed on sensor serial, and it drops
the `X_dX` pre-emphasized channels because they are the same physical sensor as
their base. Hence a separate, instrument-keyed pass.

The case that motivated it: one ARCTERX-2023 VMP carried `diff_gain = 0.09` on
both shear channels where its three siblings in the same cruise carried
0.92-0.99, which would inflate its epsilon by ~110x. Nothing noticed.
"""

from datetime import datetime
from pathlib import Path

import pytest

from odas_tpw.rsi.diff_gain import (
    MIN_INSTRUMENTS_FOR_OUTLIER,
    build,
    channel_medians,
    collect_gains,
    format_report,
    invalid_gains,
    models_for,
    outliers,
    run,
    unchecked_channels,
    write_csv,
)

CHANNELS = ("sh1", "sh2", "T1_dT1", "T2_dT2", "P_dP")


def _cfg(sn, model, gains, when=(2023, 5, 10)):
    """A stub (header, config) pair in the shape _read_header_and_config returns."""
    return (
        {"_when": datetime(*when)},
        {
            "instrument_info": {"sn": sn, "model": model, "vehicle": "vmp"},
            "channels": [{"name": n, "diff_gain": g} for n, g in gains.items()],
        },
    )


def _reader(table):
    def read(path):
        if isinstance(table[path], Exception):
            raise table[path]
        return table[path]

    return read


def _fleet(**overrides):
    """Three healthy instruments, so the fleet median is meaningful."""
    good = {"sh1": "0.941", "sh2": "0.922", "T1_dT1": "0.908", "T2_dT2": "0.926", "P_dP": "20.74"}
    t = {}
    for i, sn in enumerate(("409", "428", "465")):
        g = dict(good)
        g.update(overrides.get(sn, {}))
        t[Path(f"/x/{sn}.p")] = _cfg(sn, "VMP250IR_RDL", g, when=(2023, 5, 10 + i))
    return t


@pytest.fixture(autouse=True)
def _no_clock(monkeypatch):
    """Take the stub's time rather than parsing a real ODAS header."""
    monkeypatch.setattr(
        "odas_tpw.rsi.sensor_inventory._start_time_utc",
        lambda header, config: header.get("_when"),
    )


class TestCollection:
    def test_reads_every_channel_with_a_gain(self):
        t = _fleet()
        uses, errors = collect_gains(list(t), _reader(t))
        assert not errors
        assert len(uses) == 3 * len(CHANNELS)
        assert {u.channel for u in uses} == set(CHANNELS)

    def test_includes_the_preemphasized_channels(self):
        """The whole point: the sensor inventory drops these."""
        t = _fleet()
        uses, _ = collect_gains(list(t), _reader(t))
        assert {"T1_dT1", "T2_dT2", "P_dP"} <= {u.channel for u in uses}

    def test_a_bad_file_does_not_stop_the_scan(self):
        t = _fleet()
        t[Path("/x/bad.p")] = ValueError("truncated")
        uses, errors = collect_gains(list(t), _reader(t))
        assert len(errors) == 1
        assert len(uses) == 3 * len(CHANNELS)

    def test_unparseable_gain_is_reported_not_silently_dropped(self):
        t = _fleet()
        t[Path("/x/w.p")] = _cfg("999", "vmp-250", {"sh1": "not-a-number"})
        uses, errors = collect_gains(list(t), _reader(t))
        assert any("unparseable" in m for _, m in errors)
        assert not any(u.instrument_sn == "999" for u in uses)


class TestOutlierDetection:
    def test_catches_an_order_of_magnitude_error(self):
        """The SN132 case: 0.09 where the fleet sits near 0.93."""
        t = _fleet(**{"409": {"sh1": "0.09", "sh2": "0.09"}})
        uses, _ = collect_gains(list(t), _reader(t))
        got = outliers(uses)
        assert {(sn, ch) for sn, ch, _, _ in got} == {("409", "sh1"), ("409", "sh2")}

    def test_a_few_percent_of_spread_is_not_flagged(self):
        """Real fleet spread must not produce noise."""
        t = _fleet(**{"409": {"sh1": "0.90"}, "465": {"sh1": "0.99"}})
        uses, _ = collect_gains(list(t), _reader(t))
        assert outliers(uses) == []

    def test_no_flagging_below_the_instrument_threshold(self):
        """One instrument cannot be its own outlier -- and the report must SAY so."""
        t = {Path("/x/a.p"): _cfg("132", "vmp-250-IR", {"sh1": "0.09", "sh2": "0.09"})}
        uses, _ = collect_gains(list(t), _reader(t))
        assert outliers(uses) == []
        r = format_report(uses, [])
        assert "NOT CHECKED" in r, "a false all-clear"
        assert "sh1" in r

    def test_a_thinly_carried_channel_is_reported_not_silently_passed(self):
        """The gate is per CHANNEL. With plenty of instruments overall but only
        two carrying P_dP, a 10x P_dP error cannot be flagged -- and saying
        'no outliers' without qualification would be a false all-clear."""
        t = _fleet()
        for sn in ("409", "428"):
            t[Path(f"/x/{sn}.p")][1]["channels"] = [
                c for c in t[Path(f"/x/{sn}.p")][1]["channels"] if c["name"] != "P_dP"
            ]
        t[Path("/x/465.p")][1]["channels"] = [
            {**c, "diff_gain": "2.0"} if c["name"] == "P_dP" else c
            for c in t[Path("/x/465.p")][1]["channels"]
        ]
        uses, _ = collect_gains(list(t), _reader(t))
        assert ("P_dP", 1) in unchecked_channels(uses)
        r = format_report(uses, [])
        assert "NOT CHECKED" in r and "P_dP" in r

    def test_median_is_per_instrument_not_per_file(self):
        """A campaign with many files from one unit must not swamp the fleet."""
        t = _fleet()
        for i in range(50):
            t[Path(f"/x/spam{i}.p")] = _cfg("409", "VMP250IR_RDL", {"sh1": "0.09"})
        uses, _ = collect_gains(list(t), _reader(t))
        med, n_inst = channel_medians(uses)["sh1"]
        assert med == pytest.approx(0.941)  # unmoved by 50 junk files
        assert n_inst == 3

    def test_high_side_outliers_are_caught_too(self):
        t = _fleet(**{"409": {"P_dP": "2.0"}})
        uses, _ = collect_gains(list(t), _reader(t))
        assert any(ch == "P_dP" for _, ch, _, _ in outliers(uses))

    def test_a_nan_gain_cannot_poison_the_fleet_median(self):
        """statistics.median propagates NaN, and every ratio against NaN is
        False -- so one corrupt config would silently disable detection for
        that channel fleet-wide. It must be rejected at parse time."""
        t = _fleet(**{"409": {"sh1": "0.09"}})
        t[Path("/x/nan.p")] = _cfg("500", "VMP250IR_RDL", {"sh1": "nan"})
        uses, errors = collect_gains(list(t), _reader(t))
        assert any("non-finite" in m for _, m in errors)
        med, _ = channel_medians(uses)["sh1"]
        assert med == pytest.approx(0.941)
        assert any(ch == "sh1" for _, ch, _, _ in outliers(uses)), "detection survived"

    def test_a_zero_gain_is_reported_not_silently_skipped(self):
        """diff_gain = 0 divides the shear conversion by zero -- the worst
        possible value. It forms no ratio, so it must be caught separately."""
        t = _fleet(**{"409": {"sh1": "0"}})
        uses, _ = collect_gains(list(t), _reader(t))
        assert ("409", "sh1", 0.0) in invalid_gains(uses)
        assert "INVALID" in format_report(uses, [])


class TestTimeline:
    def test_a_change_is_recorded_with_dates_not_flagged(self):
        """Gains follow the electronics; a rebuild legitimately changes them."""
        t = {
            Path("/x/old.p"): _cfg("142", "vmp-250", {"sh1": "0.96"}, when=(2019, 5, 28)),
            Path("/x/new.p"): _cfg("142", "vmp-250", {"sh1": "0.953"}, when=(2021, 6, 22)),
        }
        uses, _ = collect_gains(list(t), _reader(t))
        agg = build(uses)["142"]["sh1"]
        assert agg.changed
        tl = agg.timeline()
        assert [r[0] for r in tl] == ["0.96", "0.953"]  # time order
        assert tl[0][1] == datetime(2019, 5, 28)
        assert "CHANGED over time" in format_report(uses, [])

    def test_a_constant_gain_is_not_reported_as_changed(self):
        t = _fleet()
        uses, _ = collect_gains(list(t), _reader(t))
        assert not build(uses)["428"]["sh1"].changed

    def test_a_rebuild_that_changes_the_model_STILL_shows_the_transition(self):
        """The headline feature, and the reason grouping is on the serial alone.

        A real CF2 -> RDL upgrade changes the `model` string as well as the
        gains (SN 142 became VMP250IR_RDL, SN 479 VMP250IR_RT). Keying on
        (sn, model) would split the instrument in two and suppress exactly the
        transition this tool exists to surface.
        """
        t = {
            Path("/x/a.p"): _cfg("194", "vmp-250", {"sh1": "0.98"}, when=(2022, 3, 21)),
            Path("/x/b.p"): _cfg("194", "VMP250IR", {"sh1": "0.976"}, when=(2025, 2, 1)),
        }
        uses, _ = collect_gains(list(t), _reader(t))
        assert set(build(uses)) == {"194"}, "one instrument, not two"
        assert build(uses)["194"]["sh1"].changed
        assert models_for(uses, "194") == ["vmp-250", "VMP250IR"]
        r = format_report(uses, [])
        assert "CHANGED over time" in r
        assert "vmp-250 → VMP250IR" in r

    def test_trailing_zeros_are_not_a_hardware_rebuild(self):
        """0.95 and 0.950 are the same gain; reporting a transition would
        invent electronics history the docs tell the reader to trust."""
        t = {
            Path("/x/a.p"): _cfg("194", "vmp-250", {"sh1": "0.95"}, when=(2022, 3, 21)),
            Path("/x/b.p"): _cfg("194", "vmp-250", {"sh1": "0.950"}, when=(2023, 3, 21)),
        }
        uses, _ = collect_gains(list(t), _reader(t))
        assert not build(uses)["194"]["sh1"].changed


class TestReportAndRun:
    def test_report_names_the_outlier_and_its_leverage(self):
        t = _fleet(**{"409": {"sh1": "0.09"}})
        uses, _ = collect_gains(list(t), _reader(t))
        r = format_report(uses, [])
        assert "OUTLIER" in r and "0.09" in r and "f^-2" in r

    def test_run_returns_the_problem_count(self):
        t = _fleet(**{"409": {"sh1": "0.09", "sh2": "0.09"}})
        report, n_problems, wrote = run(list(t), _reader(t))
        assert n_problems == 2
        assert not wrote
        assert "OUTLIERS" in report

    def test_run_is_clean_on_a_healthy_fleet(self):
        t = _fleet()
        report, n_problems, _ = run(list(t), _reader(t))
        assert n_problems == 0
        assert "No outliers" in report

    def test_empty_input_is_not_an_error(self):
        report, n_problems, _ = run([], _reader({}))
        assert n_problems == 0
        assert "No differential gains" in report

    def test_all_files_unreadable_says_so_rather_than_no_gains(self):
        t = {Path("/x/a.p"): ValueError("truncated"), Path("/x/b.p"): ValueError("v1")}
        report, _n, _ = run(list(t), _reader(t))
        assert "could not be read" in report
        # It must actively disclaim the wrong cause, not just omit it.
        assert "not because the files carry no diff_gain" in report

    def test_an_unwritable_csv_does_not_destroy_the_report(self, tmp_path):
        t = _fleet()
        target = tmp_path / "sub" / "x.csv"  # parent does not exist
        report, _n, wrote = run(list(t), _reader(t), target)
        assert not wrote
        assert "Differential gains" in report
        assert "could not write CSV" in report

    def test_csv_is_not_claimed_when_nothing_was_written(self, tmp_path):
        _report, _n, wrote = run([], _reader({}), tmp_path / "x.csv")
        assert not wrote
        assert not (tmp_path / "x.csv").exists()

    def test_csv_round_trips(self, tmp_path):
        t = _fleet()
        uses, _ = collect_gains(list(t), _reader(t))
        out = tmp_path / "g.csv"
        write_csv(uses, out)
        rows = out.read_text().strip().split("\n")
        assert rows[0].startswith("file,instrument_sn,model")
        assert len(rows) == 1 + len(uses)


class TestCLIWiring:
    def test_the_flags_exist_and_default_off(self):
        from odas_tpw.rsi.sensor_inventory import build_arg_parser

        a = build_arg_parser().parse_args(["x"])
        assert a.diff_gain is False
        assert a.diff_gain_csv is None
        assert a.diff_gain_strict is False
        b = build_arg_parser().parse_args(["x", "--diff-gain", "--diff-gain-strict"])
        assert b.diff_gain and b.diff_gain_strict

    def test_rsi_tpw_sensors_exposes_the_same_flags(self):
        """The two front ends must not drift apart.

        `rsi-tpw sensors` and `python -m odas_tpw.rsi.sensor_inventory` define
        their flags separately, so a knob added to one and not the other is a
        silent divergence.
        """
        import argparse

        from odas_tpw.rsi.cli import _add_sensors_parser
        from odas_tpw.rsi.sensor_inventory import build_arg_parser

        top = argparse.ArgumentParser()
        _add_sensors_parser(top.add_subparsers(dest="command"))
        a = top.parse_args(["sensors", "x", "--diff-gain", "--diff-gain-strict"])
        assert a.diff_gain and a.diff_gain_strict
        assert a.diff_gain_csv is None

        standalone = {x.dest for x in build_arg_parser()._actions}
        via_cli = set(vars(top.parse_args(["sensors", "x"])))
        for knob in ("diff_gain", "diff_gain_csv", "diff_gain_strict"):
            assert knob in standalone, f"{knob} missing from sensor_inventory"
            assert knob in via_cli, f"{knob} missing from rsi-tpw sensors"

    def test_min_instruments_constant_is_sane(self):
        assert MIN_INSTRUMENTS_FOR_OUTLIER >= 2
