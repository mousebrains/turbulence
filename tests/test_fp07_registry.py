# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""The FP07 calibration sheet (``microstructure_sensors/fp07_calibrations.yaml``).

The load-bearing test is the drift guard: the committed ``.md`` and ``.csv``
views must be exactly what the YAML renders, or the sheet a person reads is not
the one the code believes.  The rest pin the strict loader (each defect raises
its own named error), the prior rule, and the chain from a sheet entry through
the ``.p`` config form to the reader's temperature.
"""

import copy
import csv
import io
import json
import math
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from odas_tpw.fp07cal import registry as reg
from odas_tpw.fp07cal.cli import main as fp07cal_main
from odas_tpw.fp07cal.logr import BridgeParams, config_to_coeffs, log_r
from odas_tpw.rsi.channels import convert_therm

REPO_ROOT = Path(__file__).resolve().parents[1]
SHEET = REPO_ROOT / "microstructure_sensors" / "fp07_calibrations.yaml"


# ------------------------------------------------------------- drift guard
def test_committed_views_are_current():
    """The committed .md and .csv are exactly what the YAML renders."""
    stale = reg.stale_views(SHEET)
    assert not stale, (
        f"{', '.join(p.name for p in stale)} out of date with {SHEET.name}: "
        f"run `fp07-cal registry render` and commit the result"
    )


def test_repository_sheet_is_valid_and_seeded():
    sheet = reg.load(SHEET)
    assert len(sheet.calibrations) >= 22
    assert {"T1592", "T1593", "T2005", "T2811", "T2813"} <= set(sheet.beads())
    for c in sheet.calibrations:
        assert np.all(np.isfinite(c.poly))
        lo, hi = c.valid_T
        assert lo < hi


def test_every_entry_reproduces_through_the_reader():
    """Sheet entry -> .p config strings -> convert_therm gives the entry's own T(L).

    This threads the reciprocal convention end to end: the sheet stores what a
    config carries, and the reader is what will evaluate it.
    """
    sheet = reg.load(SHEET)
    counts = np.linspace(-6000.0, 6000.0, 25)
    for c in sheet.calibrations:
        cfg = {k: repr(v) for k, v in c.coefficients.items()}
        cfg.update({k: repr(v) for k, v in c.bridge.items()})
        np.testing.assert_array_equal(config_to_coeffs(cfg), c.poly)
        T_reader, units = convert_therm(counts, cfg)
        assert units == "deg_C"
        bp = BridgeParams(**{k: (int(v) if k == "adc_bits" else v) for k, v in c.bridge.items()})
        L, _clipped = log_r(counts, bp)
        np.testing.assert_allclose(T_reader, c.temperature(L), rtol=0, atol=1e-9)


def test_rendering_is_deterministic_and_order_independent():
    data = _load_yaml(SHEET)
    a = reg.parse(copy.deepcopy(data))
    shuffled = copy.deepcopy(data)
    shuffled["calibrations"] = list(reversed(shuffled["calibrations"]))
    b = reg.parse(shuffled)
    assert reg.render_markdown(a) == reg.render_markdown(b)
    assert reg.render_csv(a) == reg.render_csv(b)
    assert reg.render_markdown(a) == reg.render_markdown(reg.parse(copy.deepcopy(data)))


def test_csv_is_lossless():
    sheet = reg.load(SHEET)
    rows = list(csv.DictReader(io.StringIO(reg.render_csv(sheet))))
    assert len(rows) == len(sheet.calibrations)
    assert tuple(rows[0]) == reg.CSV_COLUMNS
    for row, c in zip(rows, sheet.calibrations, strict=True):
        assert (row["bead"], row["deployment"]) == c.key
        for k, v in c.coefficients.items():
            assert float(row[k]) == v  # bit-exact
        missing = [k for k in reg.BETA_KEYS if k not in c.coefficients]
        assert all(row[k] == "" for k in missing)


# -------------------------------------------------------------- the loader
TRANSFER = {
    "offset_mK": 15, "shape_mK": 5, "shape_span_K": [5, 10], "cross_instrument_mK": 10,
    "settling_mK": [5, 10], "settling_days": [7, 14],
    "source": "https://example.invalid/189",
    "evidence": {"offset": "o", "shape": "s", "cross_instrument": "c", "settling": "t"},
    "reference_caveat": "one reference",
}


def _entry(bead="T100", deployment="Test 2024", serial="7", channel="T1",
           first="2024-01-01T00:00:00Z", last="2024-01-10T00:00:00Z", **over):
    e = {
        "bead": bead,
        "deployment": deployment,
        "instrument": {"model": "vmp-250", "serial": serial, "channel": channel},
        "first": first,
        "last": last,
        "reference": {"kind": "JAC CT", "serial": "23", "cal_date": "2016-02-15"},
        "fit": {
            "method": "m", "pooled": True, "n_files": 3, "n_profiles": 30, "n_pairs": 3000,
            "order": 2, "order_choice": "auto",
            "held_out_by_order_mK": {1: 9.0, 2: 4.0, 3: 6.0},
            "robust": True, "geometry": False, "lag_s": 0.1, "reference_pole_s": 0.17,
            "software_version": "test",
        },
        "coefficients": {"t_0": 285.0, "beta_1": 3050.0, "beta_2": 280000.0},
        "bridge": {"a": -9.7, "b": 0.99878, "g": 6.0, "e_b": 0.68229, "adc_fs": 4.096,
                   "adc_bits": 16},
        "uncertainty": {"fit_rms_mK": 3.0, "held_out_mK": 4.0, "block_sd_mK": 1.0,
                        "n_blocks": 6},
        "valid_T_degC": [18.0, 28.0],
    }
    e.update(over)
    return e


def _sheet(*entries):
    return {
        "schema": reg.SCHEMA,
        "transfer": copy.deepcopy(TRANSFER),
        "methods": {"m": "a test method"},
        "calibrations": list(entries) if entries else [_entry()],
    }


def test_minimal_sheet_parses():
    sheet = reg.parse(_sheet())
    (c,) = sheet.calibrations
    assert c.key == ("T100", "Test 2024")
    assert c.cal_age_years == pytest.approx((date(2024, 1, 1) - date(2016, 2, 15)).days / 365.25)
    np.testing.assert_allclose(c.poly, [1 / 285.0, 1 / 3050.0, 1 / 280000.0])


@pytest.mark.parametrize("where", ["top", "entry", "fit", "instrument", "uncertainty"])
def test_unknown_key_is_refused(where):
    data = _sheet()
    target = {
        "top": data,
        "entry": data["calibrations"][0],
        "fit": data["calibrations"][0]["fit"],
        "instrument": data["calibrations"][0]["instrument"],
        "uncertainty": data["calibrations"][0]["uncertainty"],
    }[where]
    target["oder"] = 2
    with pytest.raises(reg.UnknownKeyError, match="oder"):
        reg.parse(data)


@pytest.mark.parametrize("path", [("bridge", "e_b"), ("fit", "lag_s"), ("coefficients", "t_0"),
                                  (None, "valid_T_degC"), ("reference", "cal_date")])
def test_missing_key_is_refused(path):
    data = _sheet()
    e = data["calibrations"][0]
    section, key = path
    del (e if section is None else e[section])[key]
    with pytest.raises(reg.MissingKeyError, match=key):
        reg.parse(data)


@pytest.mark.parametrize("bad", ["T", "1592", "t1592", "T1592A", 1592, "T01592", "T0"])
def test_bad_bead_serial_is_refused(bad):
    with pytest.raises(reg.SerialPatternError):
        reg.parse(_sheet(_entry(bead=bad)))


def test_unquoted_instrument_serial_is_refused():
    with pytest.raises(reg.SerialPatternError, match="quoted"):
        reg.parse(_sheet(_entry(serial=194)))


def test_unquoted_reference_serial_is_refused():
    """Unquoted, YAML would read JAC "0079" as the number 79."""
    e = _entry()
    e["reference"]["serial"] = 79
    with pytest.raises(reg.SerialPatternError, match="0079"):
        reg.parse(_sheet(e))


def test_bad_channel_is_refused():
    with pytest.raises(reg.SerialPatternError):
        reg.parse(_sheet(_entry(channel="sh1")))


def test_first_after_last_is_refused():
    with pytest.raises(reg.DateOrderError, match="after last"):
        reg.parse(_sheet(_entry(first="2024-02-01T00:00:00Z", last="2024-01-01T00:00:00Z")))


def test_reference_calibrated_after_the_deployment_is_refused():
    e = _entry()
    e["reference"]["cal_date"] = "2024-06-01"
    with pytest.raises(reg.DateOrderError, match="cal_date"):
        reg.parse(_sheet(e))


def test_a_bead_in_two_places_at_once_is_refused():
    a = _entry(deployment="A")
    b = _entry(deployment="B", serial="8", first="2024-01-05T00:00:00Z",
               last="2024-01-20T00:00:00Z")
    with pytest.raises(reg.DateOrderError, match="two calibrations at once"):
        reg.parse(_sheet(a, b))


def test_a_channel_carrying_two_beads_at_once_is_refused():
    a = _entry(bead="T100", deployment="A")
    b = _entry(bead="T200", deployment="A", first="2024-01-05T00:00:00Z",
               last="2024-01-20T00:00:00Z")
    with pytest.raises(reg.DateOrderError, match="SN 7 T1"):
        reg.parse(_sheet(a, b))


def test_order_must_match_the_beta_terms():
    e = _entry()
    e["fit"]["order"] = 3
    e["fit"]["held_out_by_order_mK"] = {1: 9.0, 2: 6.0, 3: 4.0}
    with pytest.raises(reg.OrderMismatchError, match="2 beta"):
        reg.parse(_sheet(e))


def test_beta_terms_must_be_consecutive():
    e = _entry()
    e["coefficients"] = {"t_0": 285.0, "beta_1": 3050.0, "beta_3": 1e7}
    with pytest.raises(reg.OrderMismatchError, match="consecutive"):
        reg.parse(_sheet(e))


def test_a_neutralised_term_does_not_belong_in_the_sheet():
    """beta_2 = 1e30 is how a .p config switches a term off; here the key is just absent."""
    e = _entry()
    e["coefficients"]["beta_2"] = 1e30
    with pytest.raises(reg.OrderMismatchError, match="leave the key out"):
        reg.parse(_sheet(e))


def test_a_zero_beta_is_refused():
    e = _entry()
    e["coefficients"]["beta_2"] = 0.0
    with pytest.raises(reg.RegistryError, match="non-zero"):
        reg.parse(_sheet(e))


def test_auto_order_must_be_the_least_held_out():
    e = _entry()
    e["fit"]["held_out_by_order_mK"] = {1: 3.0, 2: 4.0, 3: 6.0}
    with pytest.raises(reg.OrderMismatchError, match="beats"):
        reg.parse(_sheet(e))
    e["fit"]["order_choice"] = "fixed"
    reg.parse(_sheet(e))  # a forced order may lose on held-out error


def test_held_out_must_be_the_chosen_orders():
    e = _entry()
    e["uncertainty"]["held_out_mK"] = 9.0
    with pytest.raises(reg.OrderMismatchError, match="chosen order"):
        reg.parse(_sheet(e))


def test_duplicate_bead_and_deployment_is_refused():
    a = _entry()
    b = _entry(serial="8", first="2025-01-01T00:00:00Z", last="2025-01-10T00:00:00Z")
    with pytest.raises(reg.DuplicateEntryError, match="T100 / Test 2024"):
        reg.parse(_sheet(a, b))


def test_block_scatter_needs_three_blocks():
    e = _entry()
    e["uncertainty"]["n_blocks"] = 2
    with pytest.raises(reg.RegistryError, match="null"):
        reg.parse(_sheet(e))
    e["uncertainty"]["block_sd_mK"] = None
    reg.parse(_sheet(e))


def test_unknown_flag_and_method_are_refused():
    with pytest.raises(reg.RegistryError, match="unknown flag"):
        reg.parse(_sheet(_entry(flags=["fast-decent"])))
    e = _entry()
    e["fit"]["method"] = "nope"
    with pytest.raises(reg.RegistryError, match="methods"):
        reg.parse(_sheet(e))


def test_a_null_reference_field_needs_a_note():
    e = _entry()
    e["reference"]["cal_date"] = None
    with pytest.raises(reg.MissingKeyError, match="note"):
        reg.parse(_sheet(e))
    e["reference"]["note"] = "not in the data"
    assert reg.parse(_sheet(e)).calibrations[0].cal_age_years is None


def test_wrong_schema_is_refused():
    data = _sheet()
    data["schema"] = "fp07-calibrations/0"
    with pytest.raises(reg.RegistryError, match="schema"):
        reg.parse(data)


def test_geometry_true_without_values_is_refused():
    e = _entry()
    e["fit"]["geometry"] = True
    with pytest.raises(reg.RegistryError, match="dz_m"):
        reg.parse(_sheet(e))
    e["fit"]["geometry"] = {"dz_m": 0.66, "tau_s": 2.1}
    assert reg.parse(_sheet(e)).calibrations[0].fit.geometry == {"dz_m": 0.66, "tau_s": 2.1}


def test_instrument_serials_are_compared_as_numbers():
    """"0007" and "7" are one instrument: stored as 7, and caught by the channel check."""
    assert reg.parse(_sheet(_entry(serial="0007"))).calibrations[0].instrument.serial == "7"
    a = _entry(bead="T100", deployment="A", serial="7")
    b = _entry(bead="T200", deployment="B", serial="0007", first="2024-01-05T00:00:00Z",
               last="2024-01-20T00:00:00Z")
    with pytest.raises(reg.DateOrderError, match="SN 7 T1"):
        reg.parse(_sheet(a, b))


def test_duplicate_labels_ignore_case_and_spacing():
    a = _entry(deployment="Test 2024")
    b = _entry(deployment="test  2024", serial="8", first="2025-01-01T00:00:00Z",
               last="2025-01-10T00:00:00Z")
    with pytest.raises(reg.DuplicateEntryError):
        reg.parse(_sheet(a, b))


def test_t0_in_celsius_is_refused():
    e = _entry()
    e["coefficients"]["t_0"] = 11.85
    with pytest.raises(reg.PlausibilityError, match="KELVIN"):
        reg.parse(_sheet(e))


def test_negative_beta_1_is_refused():
    e = _entry()
    e["coefficients"]["beta_1"] = -3050.0
    with pytest.raises(reg.PlausibilityError, match="beta_1"):
        reg.parse(_sheet(e))


@pytest.mark.parametrize("rng", [[-60.0, 28.0], [18.0, 90.0]])
def test_a_range_beyond_the_bridge_clip_is_refused(rng):
    """|Z| <= 0.6 means |L| < ln(1.6/0.4); a range needing more cannot be real."""
    with pytest.raises(reg.PlausibilityError, match=r"1\.386"):
        reg.parse(_sheet(_entry(valid_T_degC=rng)))


def test_a_curve_that_rises_with_L_is_refused():
    e = _entry()
    e["fit"]["order"] = 3
    e["fit"]["held_out_by_order_mK"] = {1: 9.0, 2: 6.0, 3: 4.0}
    e["coefficients"] = {"t_0": 285.0, "beta_1": 3050.0, "beta_2": 1e12, "beta_3": -1000.0}
    with pytest.raises(reg.PlausibilityError, match="dT/dL"):
        reg.parse(_sheet(e))


def test_block_sd_null_needs_fewer_than_three_blocks():
    e = _entry()
    e["uncertainty"]["block_sd_mK"] = None
    with pytest.raises(reg.MissingKeyError, match="block_sd_mK"):
        reg.parse(_sheet(e))


def test_auto_order_needs_its_held_out_errors():
    e = _entry()
    del e["fit"]["held_out_by_order_mK"]
    with pytest.raises(reg.MissingKeyError, match="held_out_by_order_mK"):
        reg.parse(_sheet(e))
    e["fit"]["order_choice"] = "fixed"
    reg.parse(_sheet(e))


def test_reference_caveat_is_required():
    data = _sheet()
    del data["transfer"]["reference_caveat"]
    with pytest.raises(reg.MissingKeyError, match="reference_caveat"):
        reg.parse(data)


def test_prose_line_breaks_cannot_split_a_table_row():
    """A YAML `|` block keeps its newlines; rendered raw, they would break the table."""
    data = _sheet()
    data["transfer"]["evidence"]["offset"] = "first line\nsecond | line\n"
    data["methods"]["m"] = "one\ntwo"
    md = reg.render_markdown(reg.parse(data))
    assert "| first line second \\| line |" in md
    assert "| `m` | one two |" in md


def test_yaml_dates_unquoted_or_quoted(tmp_path):
    """ruamel turns an unquoted date into a date object; both spellings load."""
    data = _sheet()
    text = _dump_yaml(data)
    text = text.replace("cal_date: '2016-02-15'", "cal_date: 2016-02-15")
    assert "cal_date: 2016-02-15" in text
    p = tmp_path / "s.yaml"
    p.write_text(text)
    assert reg.load(p).calibrations[0].reference.cal_date == date(2016, 2, 15)


# ---------------------------------------------------------------- the prior
def _history(**flags):
    """One bead on SN 7 T1 twice (B wider but flagged), then moved to SN 8 T2.

    A: 2021, SN 7 T1, 18-28 degC (10 K), unflagged
    B: 2022, SN 7 T1, 14-29 degC (15 K), flagged fast-descent
    C: 2025, SN 8 T2, 20-25 degC (5 K), unflagged, the most recent
    """
    return reg.parse(_sheet(
        _entry(deployment="A", first="2021-01-01T00:00:00Z", last="2021-01-10T00:00:00Z",
               flags=flags.get("A", [])),
        _entry(deployment="B", first="2022-01-01T00:00:00Z", last="2022-01-10T00:00:00Z",
               valid_T_degC=[14.0, 29.0], flags=flags.get("B", ["fast-descent"]),
               coefficients={"t_0": 285.01, "beta_1": 3051.0, "beta_2": 281000.0}),
        _entry(deployment="C", serial="8", channel="T2", first="2025-01-01T00:00:00Z",
               last="2025-01-10T00:00:00Z", valid_T_degC=[20.0, 25.0],
               flags=flags.get("C", []),
               coefficients={"t_0": 285.02, "beta_1": 3052.0, "beta_2": 282000.0}),
    ))


def test_prior_takes_the_widest_unflagged_calibration():
    """B is wider but fast-descent; C is newer but narrower: A wins."""
    p = _history().prior_for("T100")
    assert p.source.deployment == "A"
    assert not p.fallback
    assert p.coefficients == {"t_0": 285.0, "beta_1": 3050.0, "beta_2": 280000.0}
    assert p.fixed_terms == {"beta_1": 3050.0, "beta_2": 280000.0}
    np.testing.assert_allclose(p.poly, [1 / 285.0, 1 / 3050.0, 1 / 280000.0])
    assert p.t_0_uncertainty_mK == 15
    assert p.shape_mK == p.sheet_shape_mK == 5
    assert not p.cross_instrument
    assert p.bridge["e_b"] == 0.68229


@pytest.mark.parametrize("flag", ["short", "fast-descent", "episode"])
def test_disqualifying_flags(flag):
    """With B already out (fast-descent), flagging A leaves only C."""
    p = _history(A=[flag]).prior_for("T100")
    assert p.source.deployment == "C"
    assert not p.fallback


@pytest.mark.parametrize("flag", ["lag-gate-overridden", "export-gate-overridden",
                                  "reference-edited", "screened"])
def test_other_flags_do_not_disqualify(flag):
    assert _history(A=[flag]).prior_for("T100").source.deployment == "A"
    p = _history(B=[flag]).prior_for("T100")  # B is the widest once unflagged
    assert p.source.deployment == "B"
    assert any(f"flagged: {flag}" in n for n in p.notes)


def test_prior_reports_the_most_recent_calibrations_offset():
    sheet = _history()
    p = sheet.prior_for("T100")
    assert p.recent.deployment == "C"
    assert p.recent_offset_T == pytest.approx(22.5)  # center of 20-25, common to A and C
    expected = reg.curve_difference(p.recent, p.source, 22.5)
    assert p.recent_offset_mK == pytest.approx(expected)
    assert abs(expected) > 1.0  # C really differs from A
    assert any(f"{expected:+.1f} mK" in n and "other electronics" in n for n in p.notes)
    q = sheet.prior_for("T100", as_of="2021-12-31")
    assert q.recent is q.source and q.recent_offset_mK is None
    assert any("also the most recent" in n for n in q.notes)


def test_curve_difference_is_zero_against_itself_and_antisymmetric():
    sheet = _history()
    a, c = (next(x for x in sheet.calibrations if x.deployment == d) for d in "AC")
    assert reg.curve_difference(a, a, 22.0) == pytest.approx(0.0, abs=1e-9)
    assert reg.curve_difference(c, a, 22.0) == pytest.approx(
        -reg.curve_difference(a, c, 22.0), abs=0.01)


def test_no_unflagged_calibration_falls_back_and_widens_the_shape():
    p = _history(A=["short"], B=["episode"], C=["short"]).prior_for("T100")
    assert p.fallback
    assert p.source.deployment == "B"  # the widest of all
    assert p.shape_mK == pytest.approx(math.hypot(5, p.source.uncertainty.held_out_mK))
    assert p.notes[0].startswith("No unflagged calibration")


def test_prior_as_of_uses_only_what_had_ended():
    sheet = _history()
    assert sheet.prior_for("T100", as_of="2024-12-31").source.deployment == "A"
    assert sheet.prior_for("T100", as_of=date(2021, 1, 10)).source.deployment == "A"
    with pytest.raises(reg.RegistryError, match="had ended by"):
        sheet.prior_for("T100", as_of="2021-01-09T23:59:59Z")
    # Before C ended only A and B count; flag A too and nothing unflagged is left.
    p = _history(A=["short"]).prior_for("T100", as_of="2024-12-31")
    assert p.fallback and p.source.deployment == "B"


def test_prior_prefers_the_target_instrument():
    sheet = _history()
    p = sheet.prior_for("T100", instrument="8")
    assert p.source.deployment == "C"
    assert not p.cross_instrument
    assert any("assumes T2" in n for n in p.notes)  # channel not given
    assert any("widest-range unflagged calibration is A" in n for n in p.notes)
    for spelling in ("SN 8", "sn8", "SN008", "0008", 8):
        assert sheet.prior_for("T100", instrument=spelling).source.deployment == "C"
    assert sheet.prior_for("T100", instrument="8", channel="t2").source.deployment == "C"


def test_prior_channel_without_instrument_is_an_error():
    with pytest.raises(reg.RegistryError, match="target instrument"):
        _history().prior_for("T100", channel="T1")


def test_prior_on_new_electronics_adds_the_cross_instrument_term():
    sheet = _history()
    p = sheet.prior_for("T100", instrument="9")
    assert p.source.deployment == "A"  # the bead's widest unflagged
    assert p.cross_instrument
    assert p.t_0_uncertainty_mK == pytest.approx(math.hypot(15, 10))
    assert any("one observation" in n for n in p.notes)
    # Same instrument, other channel: other bridge constants, so also cross.
    q = sheet.prior_for("T100", instrument="8", channel="T1")
    assert q.cross_instrument
    assert q.source.deployment == "A"
    # A flagged calibration on the target does not beat an unflagged one elsewhere.
    r = _history(C=["short"]).prior_for("T100", instrument="8")
    assert r.source.deployment == "A" and r.cross_instrument


def test_prior_reports_the_union_and_extrapolation():
    p = _history().prior_for("T100")
    assert p.valid_T == (18.0, 28.0)
    assert p.union_T == ((14.0, 29.0),)
    assert any("extrapolated" in n for n in p.notes)
    # The flagged wider calibration is never offered as "the widest-range" one.
    assert not any("widest-range" in n and "B (" in n for n in p.notes)


def test_repository_priors_follow_the_rule():
    """Pat's decision (2026-10-28): widest unflagged, else a flagged fallback."""
    sheet = reg.load(SHEET)
    assert sheet.prior_for("T1592").source.deployment == "ARCTERX-2022 Interior"
    assert sheet.prior_for("T2005").source.deployment == "ARCTERX-2022 Interior"
    p = sheet.prior_for("T1592", instrument="142", channel="T2")
    assert p.source.deployment == "ARCTERX-2025 Interior" and not p.cross_instrument
    for bead in sheet.beads():
        p = sheet.prior_for(bead)
        bad = set(reg.DISQUALIFYING_FLAGS)
        assert p.fallback == all(set(c.flags) & bad for c in sheet.history(bead))
        if not p.fallback:
            assert not set(p.source.flags) & bad
    assert sheet.prior_for("T2810").fallback  # its only calibration has an episode


def test_prior_config_lines_round_trip_exactly():
    sheet = reg.load(SHEET)
    for bead in sheet.beads():
        p = sheet.prior_for(bead)
        parsed = {k.strip(): float(v) for k, v in (ln.split("=") for ln in p.config_lines())}
        assert parsed == p.coefficients


def test_prior_for_an_unknown_bead_is_an_error():
    with pytest.raises(reg.RegistryError, match="T999"):
        _history().prior_for("T999")


def test_module_prior_for_reads_a_path():
    p = reg.prior_for("T2811", path=SHEET)
    assert p.source.instrument.serial == "435"
    json.dumps(p.to_dict())  # plain data all the way down


# ---------------------------------------------------------------------- CLI
def _copy_sheet(tmp_path):
    dst = tmp_path / reg.YAML_NAME
    shutil.copy(SHEET, dst)
    return dst


def test_cli_render_then_check(tmp_path, capsys):
    path = _copy_sheet(tmp_path)
    assert fp07cal_main(["registry", "check", "-r", str(path)]) == 1
    assert "fp07-cal registry render" in capsys.readouterr().err
    assert fp07cal_main(["registry", "render", "-r", str(path)]) == 0
    assert path.with_suffix(".md").read_text() == SHEET.with_suffix(".md").read_text()
    assert fp07cal_main(["registry", "check", "-r", str(path)]) == 0
    assert "both views current" in capsys.readouterr().out
    md = path.with_suffix(".md")
    md.write_text(md.read_text() + "hand edit\n")
    assert fp07cal_main(["registry", "check", "-r", str(path)]) == 1
    assert "stale" in capsys.readouterr().err


def test_cli_check_reports_an_invalid_sheet(tmp_path, capsys):
    path = _copy_sheet(tmp_path)
    path.write_text(path.read_text().replace("bead: T1592", "bead: T", 1))
    assert fp07cal_main(["registry", "check", "-r", str(path)]) == 1
    assert "invalid" in capsys.readouterr().err


def test_cli_show(capsys):
    assert fp07cal_main(["registry", "show", "T1592", "-r", str(SHEET),
                         "--instrument", "194", "--channel", "T1"]) == 0
    out = capsys.readouterr().out
    assert "Prior: from ARCTERX-2022 Interior" in out
    assert "widest-range unflagged calibration on the target" in out
    assert "t_0 = " in out
    assert "legacy `beta`" in out
    # By the end of 2021, SUNRISE 2019 (fast-descent) is excluded and SUNRISE 2021
    # Pelican (2.8 K) is wider than WaltonSmith (1.9 K).
    assert fp07cal_main(["registry", "show", "T1592", "-r", str(SHEET), "--json",
                         "--as-of", "2021-12-31"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["prior"]["source"]["deployment"] == "SUNRISE 2021 Pelican"
    assert doc["prior"]["most_recent"]["deployment"] == "SUNRISE 2021 WaltonSmith"
    assert fp07cal_main(["registry", "show", "T2810", "-r", str(SHEET)]) == 0
    assert "NONE IS UNFLAGGED" in capsys.readouterr().out
    assert fp07cal_main(["registry", "show", "T9", "-r", str(SHEET)]) == 1


# ------------------------------------------------------------------ helpers
def _load_yaml(path):
    from ruamel.yaml import YAML

    return YAML(typ="safe", pure=True).load(Path(path).read_text())


def _dump_yaml(data):
    from ruamel.yaml import YAML

    y = YAML(typ="safe", pure=True)
    y.default_flow_style = False
    buf = io.StringIO()
    y.dump(data, buf)
    return buf.getvalue()
