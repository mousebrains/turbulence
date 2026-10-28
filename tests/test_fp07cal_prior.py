# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""``fp07-cal fit`` with a bead's prior (``fit.prior``) and ``fp07-cal registry add``.

Synthetic deployments are built from a KNOWN curve (T1592's ARCTERX-2022
Interior shape), so the offset-only fit can be checked against truth: over a
narrow range it should win the held-out comparison and recover ``t_0``; given
a shape the data can see is wrong it should lose; given another bead's prior
the QC must flag it.  The rest pin the config's strictness, the refusals, the
record, the schema's ``prior`` order choice, and the append-only ``add``.
"""

import copy
import difflib
import hashlib
import io
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from ruamel.yaml import YAML

from odas_tpw.fp07cal import registry as reg
from odas_tpw.fp07cal.cli import TEMPLATE, _load_config, run_calibration
from odas_tpw.fp07cal.cli import main as fp07cal_main
from odas_tpw.fp07cal.fit import (
    fit_calibration,
    held_out_error,
    held_out_errors,
    select_order,
    shipped_fit,
)
from odas_tpw.fp07cal.geometry import joint_fit, local_dTdz
from odas_tpw.fp07cal.lag import temperature_lag
from odas_tpw.fp07cal.pairs import PairConfig
from odas_tpw.fp07cal.patch import build_edits
from odas_tpw.fp07cal.prior import PriorError, parse_prior_config, plausibility
from odas_tpw.fp07cal.synth import DEFAULT_BRIDGE, SynthConfig, make_deployment

REPO_ROOT = Path(__file__).resolve().parents[1]
SHEET = REPO_ROOT / "microstructure_sensors" / "fp07_calibrations.yaml"

TRUE = {"t_0": 283.1245701203119, "beta_1": 3047.8872920200442,
        "beta_2": 302761.4588270863, "beta_3": 1700539.9960895304}
POLY = np.array([1 / TRUE["t_0"], 1 / TRUE["beta_1"], 1 / TRUE["beta_2"], 1 / TRUE["beta_3"]])
SHAPE = {k: v for k, v in TRUE.items() if k != "t_0"}
BRIDGE = {"a": DEFAULT_BRIDGE.a, "b": DEFAULT_BRIDGE.b, "g": DEFAULT_BRIDGE.g,
          "e_b": DEFAULT_BRIDGE.e_b, "adc_fs": DEFAULT_BRIDGE.adc_fs,
          "adc_bits": DEFAULT_BRIDGE.adc_bits}
TRANSFER = {"offset_mK": 15, "shape_mK": 5, "shape_span_K": [5, 10], "cross_instrument_mK": 10,
            "settling_mK": [5, 10], "settling_days": [7, 14], "source": "test",
            "evidence": {"offset": "o", "shape": "s", "cross_instrument": "c", "settling": "t"},
            "reference_caveat": "one reference"}


def _t0_shifted(dT_K: float, t_0: float = TRUE["t_0"]) -> float:
    """A t_0 whose curve reads dT_K warmer at L = 0 (same shape)."""
    return 1.0 / (1.0 / t_0 - dT_K / t_0**2)


def _entry(bead="T100", deployment="Prior 2022", coefficients=None, serial="7",
           channel="T1", first="2022-03-01T00:00:00Z", last="2022-03-10T00:00:00Z",
           valid=(18.0, 28.35), bridge=None, **over):
    e = {
        "bead": bead, "deployment": deployment,
        "instrument": {"model": "vmp-250", "serial": serial, "channel": channel},
        "first": first, "last": last,
        "reference": {"kind": "JAC CT", "serial": "23", "cal_date": "2016-02-15"},
        "fit": {"method": "m", "pooled": True, "n_files": 3, "n_profiles": 30, "n_pairs": 3000,
                "order": 3, "order_choice": "fixed", "robust": True, "geometry": False,
                "lag_s": 0.1, "reference_pole_s": 0.7, "software_version": "test"},
        "coefficients": dict(coefficients or {**TRUE, "t_0": _t0_shifted(0.008)}),
        "bridge": dict(bridge or BRIDGE),
        "uncertainty": {"fit_rms_mK": 3.0, "held_out_mK": 4.0, "block_sd_mK": 1.0,
                        "n_blocks": 6},
        "valid_T_degC": list(valid),
    }
    e["fit"]["order"] = len(e["coefficients"]) - 1
    for k, v in over.items():
        e[k] = v
    return e


def _write_sheet(path: Path, *entries) -> Path:
    data = {"schema": reg.SCHEMA, "transfer": copy.deepcopy(TRANSFER),
            "methods": {"m": "a test method"}, "calibrations": list(entries)}
    y = YAML(typ="safe", pure=True)
    y.default_flow_style = False
    buf = io.StringIO()
    y.dump(data, buf)
    path.write_text("# a hand comment that must survive\n" + buf.getvalue(), encoding="utf-8")
    reg.load(path)
    return path


# The other bead: T2005's ARCTERX-2022 Interior curve, ~2 K away in t_0.
OTHER = {"t_0": 285.37000905262164, "beta_1": 3044.669964813032,
         "beta_2": 243256.0144392118, "beta_3": 992652.9729320979}


@pytest.fixture
def sheet(tmp_path):
    return _write_sheet(
        tmp_path / reg.YAML_NAME,
        _entry(),
        _entry(bead="T200", deployment="Other bead 2022", coefficients=OTHER, channel="T2"),
    )


def _deployment(T_deep, T_surface, *, seed=7, n_yos=16, serial="T100"):
    cfg = SynthConfig(n_yos=n_yos, yo_seconds=1200.0, fs=4.0, ct_every_n=1,
                      T_surface=T_surface, T_deep=T_deep, files_per_deployment=4, seed=seed)
    probes, ref, truth = make_deployment(cfg, coeffs=POLY, channels=("T1",))
    for p in probes:
        p.instrument_sn = "7"
        p.probe_sn = {"T1": serial}
    return probes, ref, truth


@pytest.fixture(scope="module")
def narrow():
    return _deployment(27.2, 29.2)


@pytest.fixture(scope="module")
def wide():
    return _deployment(14.0, 28.5)


def _cfg(prior=None, *, geometry=False, order="auto"):
    cfg = {"pairs": {"max_gap": 30.0, "min_corr": 0.7}, "lag": {"max_lag": 6.0, "step": 0.5},
           "fit": {"order": order, "geometry": geometry, "robust": True},
           "stability": {"n_blocks": 4}, "channels": ["T1"]}
    if prior is not None:
        cfg["fit"]["prior"] = prior
    return cfg


def _run(probes, ref, out, prior=None, **kw):
    return run_calibration(probes, ref, _cfg(prior, **kw), out, make_figure=False)


def _pairs(dep):
    probes, ref, _truth = dep
    _lr, pairs = temperature_lag(probes, ref, "T1", cfg=PairConfig(max_gap=30.0),
                                 max_lag=6.0, step=0.5)
    return pairs


def _mK_at(coeffs, T):
    """coeffs minus truth [mK] at the L where truth reads T."""
    return reg.poly_difference_mK(np.asarray(coeffs), POLY, T)


# ------------------------------------------------------------- the fits
def test_offset_fit_recovers_t0_and_keeps_the_shape_verbatim(narrow):
    pairs = _pairs(narrow)
    fit = fit_calibration(pairs, fixed_terms=SHAPE)
    assert fit.order == 3
    assert fit.fixed_terms == SHAPE
    ce = fit.config_equivalent
    assert {k: ce[k] for k in SHAPE} == SHAPE  # bit for bit, not 1/(1/beta)
    np.testing.assert_array_equal(fit.coeffs[1:], [1.0 / SHAPE[k] for k in SHAPE])
    assert abs(_mK_at(fit.coeffs, 28.0)) < 0.5
    assert np.isnan(fit.beta1_bracket[0])  # beta_1 is not fitted


def test_offset_fit_needs_no_temperature_range(narrow):
    """An offset needs almost no range: an L that hardly moves is no obstacle."""
    pairs = _pairs(narrow)
    pairs.L = np.full_like(pairs.L, float(np.median(pairs.L)))
    fit_calibration(pairs, fixed_terms=SHAPE)  # does not raise "zero spread"
    with pytest.raises(ValueError, match="zero spread"):
        fit_calibration(pairs, order=1)


def test_joint_offset_fit_recovers_t0_and_dz(wide):
    pairs = _pairs(wide)
    g = local_dTdz(pairs)
    ok = np.isfinite(g)
    pairs.T_ref = pairs.T_ref + np.where(ok, 0.25 * g, 0.0)
    fit, geo = joint_fit(pairs, dTdz=g, fixed_terms=SHAPE)
    assert fit.order == 3 and fit.fixed_terms == SHAPE
    assert geo.dz_m == pytest.approx(0.25, abs=0.05)
    assert abs(_mK_at(fit.coeffs, 20.0)) < 2.0


def test_fixed_shape_must_be_consecutive_betas(narrow):
    pairs = _pairs(narrow)
    with pytest.raises(ValueError, match="consecutive"):
        fit_calibration(pairs, fixed_terms={"beta_2": 3e5})
    with pytest.raises(ValueError, match="consecutive"):
        fit_calibration(pairs, fixed_terms={"t_0": 283.0, "beta_1": 3000.0})


def test_select_order_is_the_shared_held_out_estimator(wide):
    """select_order's scores are held_out_error over shipped_fit, nothing else."""
    pairs = _pairs(wide)
    _best, scores = select_order(pairs)
    for o, s in scores.items():
        h = held_out_error(pairs, lambda sub, o=o: shipped_fit(sub, order=o), min_train=o + 5)
        assert h == s["held_out_K"]


# ------------------------------------------------------- end to end, synthetic
def test_narrow_range_the_prior_shape_wins_and_recovers_t0(narrow, sheet, tmp_path):
    probes, ref, _ = narrow
    res = _run(probes, ref, tmp_path / "o", prior={"sheet": str(sheet)})
    c = res["channels"]["T1"]
    p = c["prior"]
    assert p["status"] == "used" and p["choice"] == "prior shape"
    # 2 K is narrower than the sheet's 5 K shape span: held-out over it cannot
    # test the shape a full fit would extrapolate, so the prior's is kept.
    assert p["margin"]["narrow_range"] and "narrower than the 5 K" in p["why"]
    assert p["n_folds"] == 2
    # exported in config form, the prior's betas verbatim, its order
    assert c["order"] == 3
    assert {k: c["config_equivalent"][k] for k in SHAPE} == SHAPE
    assert abs(_mK_at(c["coefficients"], 28.0)) < 0.5
    # the prior was 8 mK warm at L = 0, i.e. ~9 mK at 28 degC
    assert p["delta_vs_prior_mK"] == pytest.approx(-9.0, abs=1.0)
    assert p["qc"]["verdict"] == "ok"
    assert p["t_0_sigma_mK"] == 15 and not p["electronics_differ"]
    assert p["sheet_sha256"] == hashlib.sha256(sheet.read_bytes()).hexdigest()
    md = (tmp_path / "o" / "T1_report.md").read_text(encoding="utf-8")
    assert "## Prior (`fit.prior`)" in md and "**(chosen)**" in md
    assert "Prefer order 1" not in md  # the shape was not fitted over 2 K
    assert "held fixed" in md


def test_wide_range_a_wrong_shape_loses(wide, tmp_path):
    probes, ref, _ = wide
    bad = {**TRUE, "t_0": _t0_shifted(0.008), "beta_2": TRUE["beta_2"] * 0.8}
    sheet = _write_sheet(tmp_path / "s.yaml", _entry(coefficients=bad))
    p = _run(probes, ref, tmp_path / "o", prior={"sheet": str(sheet)})["channels"]["T1"]["prior"]
    assert p["chosen_kind"] == "full"
    full = p["candidates"][p["choice"]]["held_out_K"]
    assert p["candidates"]["prior shape"]["held_out_K"] > 2 * full
    # The wrong shape clips the ends of the range: over the pairs it kept it
    # looks better than over all of them, and that is warned about.
    off = p["candidates"]["prior shape"]
    assert off["dropped_fraction"] > 0.05 and off["rms_all_K"] > off["in_sample_K"]
    assert any("robust loop dropped" in w for w in p["warnings"])


def test_a_wrong_bead_is_flagged(narrow, sheet, tmp_path, capsys):
    probes, ref, _ = narrow
    res = _run(probes, ref, tmp_path / "o", prior={"sheet": str(sheet), "beads": {"T1": "T200"}})
    p = res["channels"]["T1"]["prior"]
    assert p["bead"] == "T200" and p["bead_from"] == "config"
    assert p["electronics_differ"]  # T200's source is on T2
    assert p["t_0_sigma_mK"] == pytest.approx(np.hypot(15, 10))
    assert p["qc"]["verdict"] == "flag"
    assert abs(p["delta_vs_prior_mK"]) > 1000
    assert "bead swap" in p["qc"]["message"]
    assert "WARNING" in capsys.readouterr().err
    md = (tmp_path / "o" / "T1_report.md").read_text(encoding="utf-8")
    assert "## ⚠ Warnings" in md and "QC: FLAGGED" in md


def test_mode_offset_and_full_force_the_choice(wide, sheet, tmp_path):
    probes, ref, _ = wide
    off = _run(probes, ref, tmp_path / "a", prior={"sheet": str(sheet), "mode": "offset"})
    full = _run(probes, ref, tmp_path / "b", prior={"sheet": str(sheet), "mode": "full"})
    assert off["channels"]["T1"]["prior"]["choice"] == "prior shape"
    assert "mode is offset" in off["channels"]["T1"]["prior"]["why"]
    pf = full["channels"]["T1"]["prior"]
    assert pf["chosen_kind"] == "full" and "mode is full" in pf["why"]
    # QC runs whichever is chosen, and every candidate is scored either way
    assert pf["qc"]["verdict"] == "ok"
    assert pf["candidates"]["prior shape"]["held_out_K"] is not None


def test_a_forced_order_is_scored_against_the_prior(narrow, sheet, tmp_path):
    probes, ref, _ = narrow
    c = _run(probes, ref, tmp_path / "o", prior={"sheet": str(sheet)}, order=1)["channels"]["T1"]
    assert c["order_selection"] == {}  # unchanged meaning: auto selection only
    assert set(c["prior"]["candidates"]) == {"order 1", "prior shape"}


def test_geometry_on_the_offset_fit_is_joint(narrow, sheet, tmp_path):
    probes, ref, _ = narrow
    c = _run(probes, ref, tmp_path / "o", prior={"sheet": str(sheet), "mode": "offset"},
             geometry=True)["channels"]["T1"]
    assert c["prior"]["choice"] == "prior shape"
    assert c["geometry"] is not None and np.isfinite(c["geometry"]["dz_m"])


def test_patch_takes_an_offset_only_record_unchanged(narrow, sheet, tmp_path):
    probes, ref, _ = narrow
    res = _run(probes, ref, tmp_path / "o", prior={"sheet": str(sheet), "mode": "offset"})
    config = {"instrument_info": {"sn": "7"}, "channels": [{
        "name": "T1", "t_0": "289.3", "beta_1": "3143.55", "beta_2": "2.5e5",
        **{k: repr(v) for k, v in BRIDGE.items()}}]}
    plan = build_edits(res, config)
    assert plan.ok, plan.errors
    e = plan.edits["T1"]
    assert float(e["beta_2"]) == SHAPE["beta_2"] and float(e["beta_3"]) == SHAPE["beta_3"]
    assert float(e["t_0"]) == res["channels"]["T1"]["config_equivalent"]["t_0"]


def test_the_deployments_own_entry_is_not_its_prior(narrow, tmp_path):
    """A sheet entry overlapping the fitted span is that deployment's own fit."""
    probes, ref, _ = narrow
    t0 = reg._iso(datetime.fromtimestamp(probes[0].time[0], UTC))
    own = _entry(deployment="This one", first=t0, last=t0, valid=(27.2, 29.2),
                 coefficients={"t_0": 283.2, "beta_1": 3080.0})
    sheet = _write_sheet(tmp_path / "s.yaml", _entry(first="2020-01-01T00:00:00Z",
                                                     last="2020-01-02T00:00:00Z"), own)
    p = _run(probes, ref, tmp_path / "o", prior={"sheet": str(sheet)})["channels"]["T1"]["prior"]
    assert p["source"]["deployment"] == "Prior 2022"
    assert "This one" not in p["considered"]


# ---------------------------------------------------------------- refusals
def test_no_prior_block_writes_no_prior_record(narrow, tmp_path):
    probes, ref, _ = narrow
    c = _run(probes, ref, tmp_path / "o")["channels"]["T1"]
    assert "prior" not in c
    assert "## Prior" not in (tmp_path / "o" / "T1_report.md").read_text(encoding="utf-8")


def test_placeholder_or_inconsistent_serials_are_refused(sheet, tmp_path):
    probes, ref, _ = _deployment(27.2, 29.2, serial="T")
    with pytest.raises(PriorError, match="placeholder"):
        _run(probes, ref, tmp_path / "a", prior={"sheet": str(sheet)})
    probes[0].probe_sn = {"T1": "T100"}
    with pytest.raises(PriorError, match="disagree"):
        _run(probes, ref, tmp_path / "b", prior={"sheet": str(sheet)})
    # an explicit bead is allowed, and `none` means no prior
    c = _run(probes, ref, tmp_path / "c",
             prior={"sheet": str(sheet), "beads": {"T1": "T100"}})["channels"]["T1"]
    assert c["prior"]["bead_from"] == "config"
    c = _run(probes, ref, tmp_path / "d",
             prior={"sheet": str(sheet), "beads": {"T1": "none"}})["channels"]["T1"]
    assert c["prior"]["status"] == "off" and "order_selection" in c


def test_a_bead_with_no_history(sheet, tmp_path):
    probes, ref, _ = _deployment(27.2, 29.2, serial="T999")
    c = _run(probes, ref, tmp_path / "a", prior={"sheet": str(sheet)})["channels"]["T1"]
    assert c["prior"]["status"] == "none" and "T999" in c["prior"]["reason"]
    with pytest.raises(PriorError, match="no prior"):
        _run(probes, ref, tmp_path / "b", prior={"sheet": str(sheet), "mode": "offset"})


@pytest.mark.parametrize("block, match", [
    ("oops", "expected a mapping"),
    ({}, "sheet is required"),
    ({"sheet": "x", "mode": "best"}, "not one of"),
    ({"sheet": "x", "bead": {}}, "unknown key"),
    ({"sheet": "x", "beads": {"T1": "T"}}, "placeholder"),
    ({"sheet": "x", "beads": {"T3": "auto"}}, "not one of the fitted channels"),
    ({"sheet": "x", "beads": ["T1"]}, "expected a mapping"),
])
def test_prior_config_is_strict(block, match):
    with pytest.raises(PriorError, match=match):
        parse_prior_config({"prior": block}, ["T1", "T2"])


def test_prior_config_values():
    assert parse_prior_config({}) is None
    assert parse_prior_config({"prior": None}) is None
    pc = parse_prior_config({"prior": {"sheet": "s.yaml", "beads": {
        "T1": "auto", "T2": None}}}, ["T1", "T2"])
    assert pc is not None and pc.mode == "auto"
    assert pc.beads == {"T1": "auto", "T2": None}


def test_config_dir_resolves_in_the_prior_block(tmp_path):
    cfg_path = tmp_path / "fp07-cal.yaml"
    cfg_path.write_text(
        "fit:\n  prior:\n    sheet: \"<CONFIG_DIR>/sheet.yaml\"\n", encoding="utf-8"
    )
    cfg = _load_config(cfg_path)
    assert cfg["fit"]["prior"]["sheet"] == f"{tmp_path.resolve()}/sheet.yaml"


def test_template_prior_block_parses_when_uncommented():
    y = YAML(typ="safe")
    assert "prior" not in (y.load(TEMPLATE)["fit"] or {})
    out, inside = [], False
    for line in TEMPLATE.splitlines():
        if line.startswith("  # prior:"):
            inside = True
        elif inside and not line.startswith("  #"):
            inside = False
        if not inside:
            out.append(line)
            continue
        text = ("  " + line[4:]).split(" #")[0].rstrip()
        if text.strip():
            out.append(text)
    cfg = y.load("\n".join(out))
    pc = parse_prior_config(cfg["fit"], ["T1", "T2"])
    assert pc is not None and pc.mode == "auto" and pc.beads == {"T1": "auto", "T2": "auto"}
    assert pc.sheet.name == reg.YAML_NAME


def test_fit_command_refuses_a_bad_prior_block_before_loading(tmp_path, capsys):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        f"files: {{p_file_root: '{tmp_path}'}}\nfit:\n  prior: {{mode: auto}}\n", encoding="utf-8"
    )
    assert fp07cal_main(["fit", "-c", str(cfg)]) == 1
    assert "sheet is required" in capsys.readouterr().err
    cfg.write_text(f"files: {{p_file_root: '{tmp_path}'}}\n"
                   f"fit:\n  prior: {{sheet: '{tmp_path}/missing.yaml'}}\n", encoding="utf-8")
    assert fp07cal_main(["fit", "-c", str(cfg)]) == 1
    assert "does not exist" in capsys.readouterr().err


# ------------------------------------------------------- the sheet's schema
def _prior_choice_entry(**over):
    e = _entry(deployment="Narrow 2023", first="2023-01-01T00:00:00Z",
               last="2023-01-05T00:00:00Z", valid=(27.2, 29.2),
               coefficients={**TRUE, "t_0": 283.13})
    e["fit"].update({"order_choice": "prior", "shape_from": "Prior 2022"})
    e.update(over)
    return e


def _sheet_dict(*entries):
    return {"schema": reg.SCHEMA, "transfer": copy.deepcopy(TRANSFER),
            "methods": {"m": "a test method"}, "calibrations": list(entries)}


def test_prior_order_choice_records_its_source():
    s = reg.parse(_sheet_dict(_entry(), _prior_choice_entry()))
    c = next(c for c in s.calibrations if c.deployment == "Narrow 2023")
    assert c.fit.order_choice == "prior" and c.fit.shape_from == "Prior 2022"
    assert c.to_dict()["fit"]["shape_from"] == "Prior 2022"
    row = next(r for r in reg.render_csv(s).splitlines() if "Narrow 2023" in r)
    assert ",prior,Prior 2022," in row
    assert "held fixed from Prior 2022" in reg.render_markdown(s)


def test_prior_order_choice_needs_a_matching_source():
    e = _prior_choice_entry()
    del e["fit"]["shape_from"]
    with pytest.raises(reg.MissingKeyError, match="shape_from"):
        reg.parse(_sheet_dict(_entry(), e))
    with pytest.raises(reg.MissingKeyError, match="not a calibration"):
        reg.parse(_sheet_dict(_entry(), _prior_choice_entry(fit={
            **_entry()["fit"], "order_choice": "prior", "shape_from": "Nowhere"})))
    with pytest.raises(reg.OrderMismatchError, match="not those of"):
        reg.parse(_sheet_dict(_entry(), _prior_choice_entry(
            coefficients={**TRUE, "beta_2": 3.1e5})))
    bad = _entry()
    bad["fit"]["shape_from"] = "Prior 2022"
    with pytest.raises(reg.OrderMismatchError, match="only for order_choice prior"):
        reg.parse(_sheet_dict(bad))


def test_a_borrowed_shape_never_supplies_a_prior():
    """Narrow 2023 is newer and on target, but its shape was not fitted."""
    wider = _prior_choice_entry(valid_T_degC=[10.0, 29.5])  # even if it is wider
    p = reg.parse(_sheet_dict(_entry(), wider)).prior_for("T100", instrument="7",
                                                          channel="T1")
    assert p.source.deployment == "Prior 2022"
    assert p.recent.deployment == "Narrow 2023"  # still the most recent t_0


def test_prior_for_excluding_drops_the_overlapping_entry():
    s = reg.parse(_sheet_dict(_entry()))
    span = (datetime(2022, 3, 5, tzinfo=UTC), datetime(2022, 3, 6, tzinfo=UTC))
    with pytest.raises(reg.RegistryError, match="outside"):
        s.prior_for("T100", excluding=span)
    span = (datetime(2023, 3, 5, tzinfo=UTC), datetime(2023, 3, 6, tzinfo=UTC))
    assert s.prior_for("T100", excluding=span).source.deployment == "Prior 2022"


def test_repository_sheet_has_no_borrowed_shapes_yet():
    assert all(c.fit.order_choice != "prior" for c in reg.load(SHEET).calibrations)


# -------------------------------------------------------------- registry add
@pytest.fixture
def fitted(narrow, sheet, tmp_path):
    probes, ref, _ = narrow
    out = tmp_path / "fit"
    _run(probes, ref, out, prior={"sheet": str(sheet)})
    return out / "coefficients.json"


def _add_args(record, sheet, *extra, deployment="Narrow 2023"):
    return ["registry", "add", str(record), "--channel", "T1", "--deployment", deployment,
            "--instrument-model", "vmp-250", "--method", "m", "--reference-kind", "JAC CT",
            "--reference-serial", "23", "--reference-cal-date", "2016-02-15",
            "--reference-pole-s", "0.7", "-r", str(sheet), *extra]


def test_registry_add_appends_a_valid_entry(fitted, sheet, capsys):
    before = sheet.read_text(encoding="utf-8")
    assert fp07cal_main(_add_args(fitted, sheet, "--notes", "A synthetic: test, #1.")) == 0
    after = sheet.read_text(encoding="utf-8")
    # Every existing line kept, in order. This sheet was dumped with sorted keys,
    # so `calibrations:` is not last and the entry goes inside it, not at the end.
    diff = list(difflib.ndiff(before.splitlines(), after.splitlines()))
    assert not [d for d in diff if d.startswith("- ")]
    assert "# a hand comment that must survive" in after
    s = reg.load(sheet)
    c = next(c for c in s.calibrations if c.deployment == "Narrow 2023")
    rec = json.loads(fitted.read_text(encoding="utf-8"))["channels"]["T1"]
    assert c.coefficients == rec["config_equivalent"]
    assert c.fit.order_choice == "prior" and c.fit.shape_from == "Prior 2022"
    assert c.uncertainty.held_out_mK == pytest.approx(
        rec["prior"]["candidates"]["prior shape"]["held_out_K"] * 1e3, abs=0.005)
    assert c.notes == "A synthetic: test, #1."
    assert not reg.stale_views(sheet)  # views re-rendered
    assert fp07cal_main(["registry", "check", "-r", str(sheet)]) == 0
    # ... and refused the second time
    assert fp07cal_main(_add_args(fitted, sheet)) == 1
    assert "already has T100 / Narrow 2023" in capsys.readouterr().err


def test_registry_add_dry_run_writes_nothing(fitted, sheet, capsys):
    before = sheet.read_text(encoding="utf-8")
    assert fp07cal_main(_add_args(fitted, sheet, "--dry-run")) == 0
    assert sheet.read_text(encoding="utf-8") == before
    out = capsys.readouterr().out
    assert "order_choice: prior" in out and "shape_from: Prior 2022" in out


def test_registry_add_refusals(fitted, sheet, tmp_path, capsys):
    rec = json.loads(fitted.read_text(encoding="utf-8"))
    # QC-flagged fit: refused unless allowed
    flagged = copy.deepcopy(rec)
    flagged["channels"]["T1"]["prior"]["qc"]["verdict"] = "flag"
    p = tmp_path / "flagged" / "coefficients.json"
    p.parent.mkdir()
    p.write_text(json.dumps(flagged), encoding="utf-8")
    report = (fitted.parent / "T1_report.md").read_text(encoding="utf-8")
    (p.parent / "T1_report.md").write_text(report, encoding="utf-8")
    assert fp07cal_main(_add_args(p, sheet)) == 1
    assert "flagged" in capsys.readouterr().err
    # untrusted lag: refused without the flag
    lag = copy.deepcopy(rec)
    lag["channels"]["T1"]["lag_trustworthy"] = False
    p.write_text(json.dumps(lag), encoding="utf-8")
    assert fp07cal_main(_add_args(p, sheet)) == 1
    assert "lag-gate-overridden" in capsys.readouterr().err
    assert fp07cal_main(_add_args(p, sheet, "--flag", "lag-gate-overridden", "--dry-run")) == 0
    capsys.readouterr()
    # a bead that contradicts the one the prior was fitted for
    assert fp07cal_main(_add_args(fitted, sheet, "--bead", "T200")) == 1
    assert "contradicts" in capsys.readouterr().err
    # a missing report
    assert fp07cal_main(_add_args(fitted, sheet, "--report", str(tmp_path / "nope.md"))) == 1
    assert "not found" in capsys.readouterr().err
    # an argument the record cannot supply
    with pytest.raises(SystemExit):
        fp07cal_main(["registry", "add", str(fitted), "--channel", "T1"])


def test_entry_yaml_quotes_what_needs_quoting(sheet):
    rec = {"schema": "fp07-cal/1", "instrument_sn": "7", "n_fit_files": 2,
           "time_start": 1.7e9, "time_end": 1.7e9 + 86400, "software_version": "v: 1 # x",
           "channels": {"T1": {
               "config_equivalent": {"t_0": 283.13, "beta_1": 3050.0},
               "coefficients": [1 / 283.13, 1 / 3050.0], "bridge": BRIDGE, "order": 1,
               "order_selection": {"1": {"held_out_K": 0.004, "in_sample_K": 0.003}},
               "rms_K": 0.003, "n_profiles": 20, "n_pairs": 2000, "lag_s": 0.1,
               "T_range": [27.2, 29.2], "probe_sn": "T100", "probe_sn_trusted": True,
               "geometry": None, "stability": {"n_blocks": 0}}}}
    for label in ("Leg 2: east", "yes", "no", "Null", "- odd", "trailing-", "quote\"d"):
        e = reg.entry_from_record(
            rec, "T1", deployment=label, instrument_model="vmp-250", method="m",
            reference_kind="JAC CT #23", reference_serial="0023", reference_cal_date=None,
            reference_note="date unknown: see #189", reference_pole_s=0.17,
            block_offsets_K=[])
        reg.add_entry(sheet, e, dry_run=True)  # reads back exactly, or raises


def test_registry_add_to_the_repository_sheet_appends_at_the_end(narrow, tmp_path, capsys):
    """The real sheet's layout: comments kept, the entry appended, views current."""
    probes, ref, _ = narrow
    _run(probes, ref, tmp_path / "fit")  # no prior: order_choice auto
    sheet = tmp_path / reg.YAML_NAME
    shutil.copy(SHEET, sheet)
    reg.write_views(sheet)
    before = sheet.read_text(encoding="utf-8")
    args = _add_args(tmp_path / "fit" / "coefficients.json", sheet, deployment="Synthetic 2023")
    args[args.index("--method") + 1] = "vmp-own-jac"
    assert fp07cal_main(args) == 0
    after = sheet.read_text(encoding="utf-8")
    assert after.startswith(before)  # appended: every existing byte kept, comments too
    block = after[len(before):]
    assert block.startswith("\n  - bead: T100\n    deployment: Synthetic 2023\n")
    assert "order_choice: auto" in block and "shape_from" not in block
    s = reg.load(sheet)
    assert len(s.calibrations) == len(reg.load(SHEET).calibrations) + 1
    assert not reg.stale_views(sheet)


def test_isothermal_water_takes_the_prior_shape_not_a_fit_to_noise(sheet, tmp_path):
    """No thermal signal: a flat full fit can win on held-out error, but it is not an FP07.

    Measured on this synthetic (28 degC throughout, 2 mK probe noise): order 1 won
    the held-out comparison with beta_1 = -1.06e5, T rising with L.
    """
    cfg = SynthConfig(n_yos=16, yo_seconds=1200.0, fs=4.0, ct_every_n=1, T_surface=28.0,
                      T_deep=28.0, files_per_deployment=4, seed=7)
    probes, ref, _ = make_deployment(cfg, coeffs=POLY, channels=("T1",))
    for p in probes:
        p.instrument_sn, p.probe_sn = "7", {"T1": "T100"}
    c = _run(probes, ref, tmp_path / "a", prior={"sheet": str(sheet)})["channels"]["T1"]
    p = c["prior"]
    assert p["choice"] == "prior shape" and "not an FP07 curve" in p["why"]
    assert abs(_mK_at(c["coefficients"], 28.0)) < 0.5
    full = _run(probes, ref, tmp_path / "b", prior={"sheet": str(sheet), "mode": "full"})
    pf = full["channels"]["T1"]["prior"]
    assert pf["chosen_kind"] == "full" and pf["qc"]["verdict"] == "flag"
    assert "not an FP07 curve" in pf["qc"]["message"]


# ------------------------------------------------- the reviewed safeguards
def _iso(seed, *, T_deep=28.0, T_surface=28.0):
    cfg = SynthConfig(n_yos=16, yo_seconds=1200.0, fs=4.0, ct_every_n=1, T_surface=T_surface,
                      T_deep=T_deep, files_per_deployment=4, seed=seed)
    probes, ref, _ = make_deployment(cfg, coeffs=POLY, channels=("T1",))
    for p in probes:
        p.instrument_sn, p.probe_sn = "7", {"T1": "T100"}
    return probes, ref


def test_a_full_fit_with_the_right_sign_but_a_wrong_slope_is_not_plausible(sheet, tmp_path):
    """Isothermal seed 2: order 1 has beta_1 = +6.8e4 (an FP07's is ~3048), T falling
    with L as it should, and won held-out 3.434 against 3.477 mK. Before the
    magnitude bound it was exported, 7.6 K wrong at 20 degC, and the QC passed."""
    probes, ref = _iso(2)
    c = _run(probes, ref, tmp_path / "a", prior={"sheet": str(sheet)})["channels"]["T1"]
    p = c["prior"]
    full = p["candidates"][p["full_candidate"]]
    assert full["plausible"] is False and full["slope_ratio"][1] < 0.1
    assert "dT/dL" in full["implausible"]
    assert p["choice"] == "prior shape" and "plausible" in p["why"]
    assert abs(_mK_at(c["coefficients"], 20.0)) < 0.5


def test_plausibility_bounds_the_slope_against_the_prior(wide):
    pairs = _pairs(wide)
    good = fit_calibration(pairs, order=3)
    ok, why, ratio = plausibility(good, POLY)
    assert ok and why == "" and abs(ratio[0] - 1) < 0.01 and abs(ratio[1] - 1) < 0.01
    shallow = copy.copy(good)
    shallow.coeffs = good.coeffs * np.array([1.0, 1.15, 1.15, 1.15])  # dT/dL ~13% smaller
    ok, why, _ = plausibility(shallow, POLY)
    assert not ok and "x the prior's" in why
    rising = copy.copy(good)
    rising.coeffs = good.coeffs * np.array([1.0, -1.0, -1.0, -1.0])
    ok, why, _ = plausibility(rising, POLY)
    assert not ok and "not an FP07 curve" in why


def _prior_matching_at(T_c, betas, valid=(14.0, 28.5)):
    """A sheet entry with these betas whose curve reads the truth at T_c."""
    L_c = reg._L_at(POLY, T_c)
    inv = 1.0 / (T_c + 273.15) - sum(L_c ** i / betas[f"beta_{i}"] for i in range(1, 4))
    return _entry(coefficients={"t_0": 1.0 / inv, **betas}, valid=valid)


def test_qc_runs_over_the_whole_overlap(wide, tmp_path):
    """The center agrees, the ends do not: flagged by the range limit."""
    probes, ref, _ = wide
    betas = {**SHAPE, "beta_2": SHAPE["beta_2"] * 0.7}
    sheet = _write_sheet(tmp_path / "s.yaml", _prior_matching_at(21.25, betas))
    p = _run(probes, ref, tmp_path / "o", prior={"sheet": str(sheet), "mode": "full"}
             )["channels"]["T1"]["prior"]
    qc = p["qc"]
    assert not qc["extrapolated"]
    assert abs(qc["delta_center_mK"]) < qc["limit_center_mK"]
    assert qc["max_abs_delta_mK"] > qc["limit_range_mK"]
    assert qc["limit_range_mK"] == pytest.approx(2 * np.hypot(15, 5))
    assert qc["verdict"] == "flag" and "max |delta|" in qc["message"]


def test_qc_without_overlap_is_an_extrapolation(narrow, tmp_path):
    probes, ref, _ = narrow
    sheet = _write_sheet(tmp_path / "s.yaml", _entry(valid=(18.0, 22.0)))
    p = _run(probes, ref, tmp_path / "o", prior={"sheet": str(sheet)})["channels"]["T1"]["prior"]
    qc = p["qc"]
    assert qc["extrapolated"] and p["overlap_T_degC"] is None
    assert qc["limit_center_mK"] == qc["limit_range_mK"]  # the shape term everywhere
    assert "extrapolated" in qc["message"]


def test_the_tie_band_decides_a_close_call(tmp_path):
    """22-28.5 degC (not narrow), a prior shape 3% off in beta_2: the full fit is
    better by ~9% of held-out error. Beyond a 5% band it wins; inside 20% it does not."""
    probes, ref, _ = _deployment(22.0, 28.5)
    co = {**TRUE, "t_0": _t0_shifted(0.008), "beta_2": TRUE["beta_2"] * 0.97}
    sheet = _write_sheet(tmp_path / "s.yaml", _entry(coefficients=co))
    a = _run(probes, ref, tmp_path / "a", prior={"sheet": str(sheet)})["channels"]["T1"]["prior"]
    b = _run(probes, ref, tmp_path / "b", prior={"sheet": str(sheet), "tie_fraction": 0.2}
             )["channels"]["T1"]["prior"]
    assert not a["margin"]["narrow_range"]
    assert 0.05 < a["margin"]["fraction"] < 0.2
    assert a["chosen_kind"] == "full" and "beyond the 5% tie band" in a["why"]
    assert b["chosen_kind"] == "offset" and "within the 20% tie band" in b["why"]


def test_an_exact_tie_goes_to_the_prior(wide, sheet, tmp_path):
    probes, ref, _ = wide
    p = _run(probes, ref, tmp_path / "o", prior={"sheet": str(sheet), "tie_fraction": 0.0}
             )["channels"]["T1"]["prior"]
    m = p["margin"]
    # with the true shape the prior is no worse; at a zero band, <= still keeps it
    assert m["held_out_prior_K"] <= m["held_out_full_K"] and p["choice"] == "prior shape"


def test_every_candidate_is_scored_on_the_same_folds(wide):
    """A fold that one candidate cannot be fitted on is dropped for all of them."""
    pairs = _pairs(wide)

    def fussy(sub):
        if float(np.min(sub.T_ref)) < 20.0:  # refuses the cold-half training set
            raise ValueError("no")
        return fit_calibration(sub, order=1)

    both, n = held_out_errors(pairs, {"a": lambda s: fit_calibration(s, order=1),
                                      "b": fussy}, min_train=8)
    alone, n1 = held_out_errors(pairs, {"a": lambda s: fit_calibration(s, order=1)},
                                min_train=8)
    assert n == 1 and n1 == 2
    assert both["a"] != alone["a"]  # "a" was scored on the one fold "b" survived


def test_an_implausible_held_out_best_order_gives_way_to_the_next(wide, sheet, tmp_path,
                                                                  monkeypatch):
    import odas_tpw.fp07cal.prior as prior_mod

    real = prior_mod.plausibility
    banned: dict = {}

    def fake(fit, poly, n=201):
        if not banned:
            banned["order"] = fit.order  # the first full candidate checked is order 1
        if fit.order == banned["order"] and fit.fixed_terms is None:
            return False, "banned for the test", [1.0, 1.0]
        return real(fit, poly, n)

    monkeypatch.setattr(prior_mod, "plausibility", fake)
    probes, ref, _ = wide
    p = _run(probes, ref, tmp_path / "o", prior={"sheet": str(sheet), "tie_fraction": 0.0}
             )["channels"]["T1"]["prior"]
    assert p["candidates"][f"order {banned['order']}"]["plausible"] is False
    assert p["full_candidate"] != f"order {banned['order']}"
    assert p["candidates"][p["full_candidate"]]["plausible"]


def test_mode_offset_failure_is_a_channel_refusal_not_a_crash(narrow, sheet, tmp_path,
                                                              monkeypatch):
    import odas_tpw.fp07cal.prior as prior_mod

    real = prior_mod._fit_one

    def failing(pairs, *, order=1, fixed=None, robust, geometry):
        if fixed is not None:
            raise ValueError("simulated")
        return real(pairs, order=order, fixed=fixed, robust=robust, geometry=geometry)

    monkeypatch.setattr(prior_mod, "_fit_one", failing)
    probes, ref, _ = narrow
    res = _run(probes, ref, tmp_path / "o", prior={"sheet": str(sheet), "mode": "offset"})
    c = res["channels"]["T1"]
    assert c["error"] == "offset-only fit failed" and c["prior"]["status"] == "failed"
    assert not (tmp_path / "o" / "T1_report.md").exists()
    assert (tmp_path / "o" / "coefficients.json").exists()


def test_registry_add_records_the_rms_over_all_pairs(fitted, sheet):
    assert fp07cal_main(_add_args(fitted, sheet)) == 0
    c = next(c for c in reg.load(sheet).calibrations if c.deployment == "Narrow 2023")
    rec = json.loads(fitted.read_text(encoding="utf-8"))["channels"]["T1"]["prior"]
    assert c.uncertainty.rms_all_mK == pytest.approx(
        rec["candidates"]["prior shape"]["rms_all_K"] * 1e3, abs=0.005)
    bad = _entry()
    bad["uncertainty"]["rms_all_mK"] = -1.0
    with pytest.raises(reg.RegistryError, match="rms_all_mK"):
        reg.parse(_sheet_dict(bad))


def test_registry_add_keeps_mode_line_endings_and_trailing_comments(fitted, sheet):
    text = sheet.read_text(encoding="utf-8")
    # The fixture's sheet was dumped with sorted keys: calibrations comes first.
    marker = "\nmethods:"
    assert marker in text
    text = text.replace(marker, "\n    # about the last entry\n\n# about methods" + marker, 1)
    sheet.write_bytes(text.replace("\n", "\r\n").encode())
    sheet.chmod(0o640)
    assert fp07cal_main(_add_args(fitted, sheet)) == 0
    raw = sheet.read_bytes().decode()
    assert "\r\n" in raw and "\n" not in raw.replace("\r\n", "")  # CRLF throughout
    lines = raw.split("\r\n")
    i = lines.index("    # about the last entry")
    j = next(k for k, ln in enumerate(lines) if "deployment: Narrow 2023" in ln)
    assert i < j < lines.index("# about methods")  # after the entry's comment, before methods'
    if os.name == "posix":  # Windows has no POSIX mode bits; chmod only sets read-only
        assert (sheet.stat().st_mode & 0o777) == 0o640
    assert reg.load(sheet).calibrations


def _full_fit_record(fitted, tmp_path):
    """The fixture's record turned into an order-1 full fit (no borrowed shape)."""
    rec = json.loads(fitted.read_text(encoding="utf-8"))
    ch = rec["channels"]["T1"]
    ch["prior"]["chosen_kind"] = "full"
    ch["prior"]["choice"] = "order 1"
    ch["order"] = 1
    ch["config_equivalent"] = {"t_0": 283.2, "beta_1": 3080.0}
    ch["order_selection"] = {"1": {"held_out_K": 0.004, "in_sample_K": 0.003}}
    alt = tmp_path / "alt" / "coefficients.json"
    alt.parent.mkdir()
    alt.write_text(json.dumps(rec), encoding="utf-8")
    report = (fitted.parent / "T1_report.md").read_text(encoding="utf-8")
    (alt.parent / "T1_report.md").write_text(report, encoding="utf-8")
    return alt


def test_registry_add_into_an_empty_calibrations_list(fitted, tmp_path):
    sheet = tmp_path / "empty" / reg.YAML_NAME
    sheet.parent.mkdir()
    src = _write_sheet(tmp_path / "src.yaml", _entry()).read_text(encoding="utf-8")
    head = src[:src.index("calibrations:")]
    tail = src[src.index("\nmethods:"):]
    sheet.write_text(head + "calibrations: []" + tail, encoding="utf-8")
    assert fp07cal_main(_add_args(_full_fit_record(fitted, tmp_path), sheet)) == 0
    assert [c.deployment for c in reg.load(sheet).calibrations] == ["Narrow 2023"]


def test_concurrent_adds_do_not_lose_an_entry(fitted, sheet, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    alt = _full_fit_record(fitted, tmp_path)
    rec = json.loads(alt.read_text(encoding="utf-8"))
    del rec["channels"]["T1"]["prior"]  # four other beads, not the one the prior was for
    report = (alt.parent / "T1_report.md").read_text(encoding="utf-8")
    blocks = reg.block_offsets_from_report(report)
    entries = [
        reg.entry_from_record(
            rec, "T1", deployment=f"Narrow 2023 part {k}", instrument_model="vmp-250",
            method="m", reference_kind="JAC CT", reference_serial="23",
            reference_cal_date="2016-02-15", reference_pole_s=0.7, block_offsets_K=blocks,
            bead=f"T{300 + k}", instrument_serial=str(10 + k))  # one channel, one bead
        for k in range(4)
    ]
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda e: reg.add_entry(sheet, e), entries))
    names = {c.deployment for c in reg.load(sheet).calibrations}
    assert {f"Narrow 2023 part {k}" for k in range(4)} <= names
