# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for ``perturb sections`` (auto-generate a sections.yaml by time gap)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from odas_tpw.perturb import mk_sections as M

# --------------------------------------------------------------------------- #
# parse_duration
# --------------------------------------------------------------------------- #


def test_parse_duration_units():
    assert M.parse_duration("3600s") == 3600.0
    assert M.parse_duration("1h") == 3600.0
    assert M.parse_duration("1.5h") == 5400.0
    assert M.parse_duration("90m") == 5400.0
    assert M.parse_duration("2d") == 172800.0
    assert M.parse_duration("120") == 120.0  # bare number = seconds


@pytest.mark.parametrize("bad", ["", "abc", "-1h", "0", "0s", "1x2h", "inf", "nan"])
def test_parse_duration_rejects(bad):
    with pytest.raises(ValueError):
        M.parse_duration(bad)


# --------------------------------------------------------------------------- #
# split_indices (the extension seam)
# --------------------------------------------------------------------------- #


def test_split_indices_groups_on_gap():
    st = np.array([0, 60, 120, 5000, 5060, 20000], dtype=float)
    groups = M.split_indices(st, gap_seconds=1000)
    assert [list(g) for g in groups] == [[0, 1, 2], [3, 4], [5]]


def test_split_indices_edges():
    assert M.split_indices(np.array([]), 100) == []
    one = M.split_indices(np.array([5.0]), 100)
    assert [list(g) for g in one] == [[0]]
    # gap exactly at threshold does NOT split (strictly greater)
    at = M.split_indices(np.array([0.0, 100.0]), 100.0)
    assert [list(g) for g in at] == [[0, 1]]


# --------------------------------------------------------------------------- #
# build_sections
# --------------------------------------------------------------------------- #


def test_build_sections_windows_and_names():
    pt = M.ProfileTimes(np.array([100.0, 200.0, 5000.0]),
                        np.full(3, 13.0), np.full(3, 130.0))
    groups = M.split_indices(pt.stime, 1000.0)  # -> [[0,1],[2]]
    secs = M.build_sections(pt, groups, "time", "km", pad=30.0)
    assert [s["name"] for s in secs] == ["section_00", "section_01"]
    assert [s["n_casts"] for s in secs] == [2, 1]
    assert secs[0]["start"] == M._iso(70.0)   # 100 - 30 pad
    assert secs[0]["stop"] == M._iso(230.0)   # 200 + 30 pad
    assert secs[0]["xaxis"] == {"method": "time"}  # no units for time


def test_build_sections_spatial_carries_units():
    pt = M.ProfileTimes(np.array([0.0, 1.0]), np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    groups = M.split_indices(pt.stime, 100.0)
    secs = M.build_sections(pt, groups, "signed_distance", "nm", pad=0.0)
    assert secs[0]["xaxis"] == {"method": "signed_distance", "units": "nm"}


def test_iso_is_utc_zulu():
    assert M._iso(0.0) == "1970-01-01T00:00:00Z"


def test_build_sections_brackets_with_zero_pad():
    # Fractional cast times at --pad 0 must still be bracketed: floor the start,
    # ceil the stop, so the inclusive plot-side filter never drops a boundary cast.
    pt = M.ProfileTimes(np.array([100.6, 200.4]), np.full(2, 13.0), np.full(2, 130.0))
    groups = M.split_indices(pt.stime, 1000.0)  # one group
    secs = M.build_sections(pt, groups, "time", "km", pad=0.0)
    assert secs[0]["start"] == M._iso(100)  # floor(100.6) <= 100.6
    assert secs[0]["stop"] == M._iso(201)   # ceil(200.4)  >= 200.4


# --------------------------------------------------------------------------- #
# read_profile_times + run (end-to-end via a synthetic combo)
# --------------------------------------------------------------------------- #


def _write_combo(path: Path, stimes, lat=13.0, lon=130.0) -> None:
    import xarray as xr

    n = len(stimes)
    ds = xr.Dataset(
        {"epsilonMean": (("bin", "profile"), np.ones((2, n)))},
        coords={"bin": ("bin", [1.0, 2.0]), "profile": ("profile", np.arange(n))},
    )
    ds["stime"] = (("profile",), np.asarray(stimes, dtype=float))
    ds["lat"] = (("profile",), np.full(n, lat))
    ds["lon"] = (("profile",), np.full(n, lon))
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def test_read_profile_times_sorts_and_drops_nonfinite(tmp_path, monkeypatch):
    from odas_tpw.perturb import resolve

    d = tmp_path / "combo_00"
    _write_combo(d / "combo.nc", [300.0, 100.0, np.nan, 200.0])
    monkeypatch.setattr(resolve, "stage_dir", lambda cfg, product: str(d))
    pt = M.read_profile_times({}, "combo")
    assert list(pt.stime) == [100.0, 200.0, 300.0]  # sorted, NaN dropped


def test_read_profile_times_missing_combo_nc(tmp_path, monkeypatch):
    from odas_tpw.perturb import resolve

    d = tmp_path / "combo_00"
    d.mkdir()  # stage dir resolves but the run never wrote combo.nc
    monkeypatch.setattr(resolve, "stage_dir", lambda cfg, product: str(d))
    with pytest.raises(SystemExit, match=r"combo\.nc"):
        M.read_profile_times({}, "combo")


def test_read_profile_times_unknown_product(monkeypatch):
    from odas_tpw.perturb import resolve

    def boom(cfg, product):
        raise ValueError(f"unknown stage {product!r}; known: [...]")

    monkeypatch.setattr(resolve, "stage_dir", boom)
    # stage_dir raises ValueError for an unknown stage -> clean SystemExit, not a
    # raw traceback (--product is free-form, so a typo is reachable).
    with pytest.raises(SystemExit, match="unknown stage"):
        M.read_profile_times({}, "bogus_product")


def _args(config, output, **kw):
    base = dict(config=str(config), output=str(output), gap="1h", xaxis="time",
                units="km", pad=30.0, product="combo", force=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_run_end_to_end_roundtrips(tmp_path, monkeypatch):
    from odas_tpw.perturb import resolve
    from odas_tpw.perturb.plot.sections import load_sections

    cfg = tmp_path / "perturb.yaml"
    cfg.write_text("files:\n  output_root: results/\n")
    base = 1_700_000_000
    stimes = [base, base + 600, base + 1200, base + 20000, base + 20600]  # 3 + gap + 2
    d = tmp_path / "combo_00"
    _write_combo(d / "combo.nc", stimes)
    monkeypatch.setattr(resolve, "stage_dir", lambda c, product: str(d))

    out = tmp_path / "sections.yaml"
    M.run(_args(cfg, out))
    secs = load_sections(str(out))  # the emitted YAML must load via the plot side
    assert [s.name for s in secs] == ["section_00", "section_01"]
    assert all(s.method == "time" for s in secs)
    # first window brackets the first batch's 3 casts
    assert secs[0].start <= np.datetime64(int(base), "s")
    assert secs[0].stop >= np.datetime64(int(base + 1200), "s")


def test_run_overwrite_guard(tmp_path, monkeypatch):
    from odas_tpw.perturb import resolve

    cfg = tmp_path / "perturb.yaml"
    cfg.write_text("files:\n  output_root: results/\n")
    d = tmp_path / "combo_00"
    _write_combo(d / "combo.nc", [1_700_000_000.0, 1_700_000_600.0])
    monkeypatch.setattr(resolve, "stage_dir", lambda c, product: str(d))
    out = tmp_path / "sections.yaml"
    M.run(_args(cfg, out))
    with pytest.raises(SystemExit, match="exists"):
        M.run(_args(cfg, out))              # refuses to clobber
    M.run(_args(cfg, out, force=True))      # --force overwrites


@pytest.mark.parametrize("bad_pad", [-5.0, float("nan"), float("inf")])
def test_run_rejects_bad_pad(tmp_path, bad_pad):
    # --pad is validated up front, before the config/combo are ever read.
    cfg = tmp_path / "perturb.yaml"
    cfg.write_text("files:\n  output_root: results/\n")
    with pytest.raises(SystemExit, match="pad"):
        M.run(_args(cfg, tmp_path / "s.yaml", pad=bad_pad))


def test_run_rejects_missing_output_parent(tmp_path):
    # A -o path whose parent dir does not exist fails fast (before the combo read).
    cfg = tmp_path / "perturb.yaml"
    cfg.write_text("files:\n  output_root: results/\n")
    out = tmp_path / "nonexistent_dir" / "sections.yaml"
    with pytest.raises(SystemExit, match="does not exist"):
        M.run(_args(cfg, out))


def test_run_rejects_xaxis_needing_params(tmp_path, monkeypatch):
    from odas_tpw.perturb import resolve

    cfg = tmp_path / "perturb.yaml"
    cfg.write_text("files:\n  output_root: results/\n")
    d = tmp_path / "combo_00"
    _write_combo(d / "combo.nc", [1_700_000_000.0, 1_700_000_600.0])
    monkeypatch.setattr(resolve, "stage_dir", lambda c, product: str(d))
    out = tmp_path / "sections.yaml"
    # distance_from_point needs a point the tool can't infer -> validation error
    with pytest.raises((ValueError, SystemExit)):
        M.run(_args(cfg, out, xaxis="distance_from_point"))


def test_cli_registers_sections():
    from odas_tpw.perturb.cli import build_parser

    args = build_parser().parse_args(["sections", "-c", "x.yaml", "--gap", "2h"])
    assert args.command == "sections"
    assert args.config == "x.yaml" and args.gap == "2h"
    assert args.xaxis == "time" and args.product == "combo"


def test_build_parser_does_not_import_numpy():
    # Registering the `sections` subcommand imports mk_sections; that must not
    # pull numpy into the parser path (lazy-import discipline). Run in a fresh
    # subprocess so a numpy already imported by the test session doesn't mask it.
    import subprocess
    import sys

    code = (
        "import sys; from odas_tpw.perturb.cli import build_parser; "
        "build_parser(); "
        "assert 'numpy' not in sys.modules, 'build_parser pulled in numpy'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_cli_rejects_uninferrable_xaxis():
    from odas_tpw.perturb.cli import build_parser

    # distance_from_point/along_line need params the tool can't infer -> argparse
    # rejects them at parse time (SystemExit) rather than emitting bad YAML.
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["sections", "-c", "x.yaml", "--xaxis", "distance_from_point"]
        )
