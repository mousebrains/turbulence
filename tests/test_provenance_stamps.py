# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""Products and logs record the code that wrote them.

Every NetCDF product carries a ``software_version`` attribute and a ``history``
line ``<UTC time>: <step> by microstructure-tpw <version>``, so a bad file names
the code that made it rather than whatever ``--version`` says later. A guard
below walks the source for NetCDF writers that do not stamp.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from odas_tpw import _version_info as vi

_SRC = Path(__file__).resolve().parents[1] / "src" / "odas_tpw"
_SAMPLE = Path(__file__).parent / "data" / "SN479_0006.p"
_LINE = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00: (?P<what>.+) by (?P<by>.+)$")


@pytest.fixture
def sample_p():
    if not _SAMPLE.exists():
        pytest.skip("SN479_0006.p test data not available")
    return _SAMPLE


@pytest.fixture(autouse=True, scope="module")
def _computed_here():
    """Look the version up afresh: a value cached under another test's mock
    would match itself in every assertion below and prove nothing."""
    vi.running_version.cache_clear()
    yield
    vi.running_version.cache_clear()


@pytest.fixture
def fresh_version():
    """Clear the per-process cache around a test that patches the lookup."""
    vi.running_version.cache_clear()
    yield
    vi.running_version.cache_clear()


def _last_step(history: str) -> str:
    """The step named by a history's newest line, which must be ours."""
    m = _LINE.match(history.splitlines()[-1])
    assert m, f"not a provenance line: {history.splitlines()[-1]!r}"
    assert m["by"] == vi.software_version()
    return m["what"]


def _assert_stamped(path: Path) -> str:
    with xr.open_dataset(path, decode_times=False) as ds:
        assert ds.attrs[vi.SOFTWARE_ATTR] == vi.software_version(), path
        return _last_step(ds.attrs["history"])


# ---------------------------------------------------------------------------
# The helpers
# ---------------------------------------------------------------------------


def test_history_line_names_the_step_and_the_code():
    assert _LINE.match(vi.history_line("epsilon"))["what"] == "epsilon"
    assert vi.software_version() == f"microstructure-tpw {vi.running_version()}"


def test_history_is_appended_to_the_inputs():
    attrs = vi.provenance_attrs("chi", "2026-01-01T00:00:00Z: upstream\n")
    lines = attrs["history"].splitlines()
    assert lines[0] == "2026-01-01T00:00:00Z: upstream"
    assert _last_step(attrs["history"]) == "chi" and len(lines) == 2
    assert attrs[vi.SOFTWARE_ATTR] == vi.software_version()


@pytest.mark.parametrize("prior", [None, "", "  \n", 3, b"bytes"])
def test_empty_or_odd_prior_history_is_dropped(prior):
    assert len(vi.provenance_attrs("x", prior)["history"].splitlines()) == 1


def test_stamp_can_start_a_fresh_history():
    attrs = {"history": "the first input's lineage", vi.SOFTWARE_ATTR: "old"}
    vi.stamp_provenance(attrs, "binning", inherit=False)
    assert "first input" not in attrs["history"]
    assert attrs[vi.SOFTWARE_ATTR] == vi.software_version()
    vi.stamp_provenance(attrs, "combo")
    assert [_LINE.match(x)["what"] for x in attrs["history"].splitlines()] == ["binning", "combo"]


def test_lookup_failure_never_fails_a_write(monkeypatch, fresh_version):
    def _boom():
        raise RuntimeError("no metadata")

    monkeypatch.setattr(vi, "version_string", _boom)
    assert vi.running_version() == "unknown (version lookup failed: RuntimeError)"
    assert vi.SOFTWARE_ATTR in vi.provenance_attrs("x")


def test_git_runs_once_per_process(monkeypatch, fresh_version):
    calls = []
    monkeypatch.setattr(vi, "version_string", lambda: calls.append(1) or "9.9.9")
    for _ in range(3):
        vi.history_line("x")
    assert calls == [1]


# ---------------------------------------------------------------------------
# Guard: every NetCDF writer stamps
# ---------------------------------------------------------------------------

_STAMPS = {"stamp_provenance", "provenance_attrs"}
# perturb's atomic writers, which take a finished dataset
_WRAPPERS = {"atomic_to_netcdf", "_atomic_to_netcdf"}
# (file, function) -> (unstamped writes, why). The count pins each entry: a new
# writer added to an allowlisted function fails the guard like any other.
_UNSTAMPED_BY_DESIGN = {
    ("perturb/atomic_io.py", "_write"): (1, "generic atomic writer; its callers stamp"),
    ("perturb/combo.py", "_atomic_to_netcdf"): (1, "generic atomic writer; make_combo stamps"),
    ("perturb/pipeline.py", "_atomic_to_netcdf"): (1, "thin wrapper over atomic_io"),
    ("perturb/pipeline.py", "process_file"): (
        2,
        "diss/chi datasets are stamped where rsi's _compute_epsilon/_compute_chi "
        "build them; test_perturb_products_name_their_code reads the files",
    ),
    ("rsi/helpers.py", "write_profile_results"): (
        1,
        "writes the datasets rsi.dissipation / rsi.chi_io built and stamped",
    ),
    ("chi/fp07_pair_cli.py", "_fit"): (1, "calls ResponseCorrection.to_netcdf, which stamps"),
}


def _name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _is_writer(call: ast.Call) -> bool:
    name = _name(call.func)
    if name == "to_netcdf" or name in _WRAPPERS:
        return True
    if name == "Dataset":  # netCDF4, opened for a new file
        modes = [a.value for a in call.args[1:2] if isinstance(a, ast.Constant)]
        modes += [
            k.value.value
            for k in call.keywords
            if k.arg == "mode" and isinstance(k.value, ast.Constant)
        ]
        return bool({"w", "x"} & set(modes))
    return False


def _own_calls(fn: ast.AST) -> list[ast.Call]:
    """Calls in *fn*'s own body; nested functions are scanned on their own."""
    calls, stack = [], list(ast.iter_child_nodes(fn))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Call):
            calls.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return calls


def test_every_netcdf_writer_stamps():
    unstamped = {}
    for py in sorted(_SRC.rglob("*.py")):
        rel = py.relative_to(_SRC).as_posix()
        if rel.startswith("pyturb/"):  # hosted code, not ours to change
            continue
        for fn in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls = _own_calls(fn)
            writes = [c.lineno for c in calls if _is_writer(c)]
            stamps = sum(_name(c.func) in _STAMPS for c in calls)
            if writes and stamps < len(writes):
                unstamped[(rel, fn.name)] = len(writes) - stamps
    assert unstamped == {k: n for k, (n, _why) in _UNSTAMPED_BY_DESIGN.items()}, (
        "a NetCDF writer that does not call stamp_provenance/provenance_attrs "
        f"(or a stale allowlist entry): {unstamped}"
    )


# ---------------------------------------------------------------------------
# The products themselves
# ---------------------------------------------------------------------------


def test_perturb_products_name_their_code(sample_p, tmp_path):
    from odas_tpw.perturb.pipeline import process_file

    dirs = {k: tmp_path / k for k in ("profiles", "diss", "chi")}
    for d in dirs.values():
        d.mkdir()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        process_file(sample_p, {"chi": {"enable": True}}, None, dirs)
    steps = {k: {_assert_stamped(p) for p in d.glob("*.nc")} for k, d in dirs.items()}
    assert steps["diss"] == {"epsilon"} and steps["chi"] == {"chi"}
    assert steps["profiles"] and all(re.fullmatch(r"profile \d+ extracted", s)
                                     for s in steps["profiles"])
    # Each keeps the lineage of the profile it was computed from.
    for kind, step in (("diss", "epsilon"), ("chi", "chi")):
        for path in dirs[kind].glob("*.nc"):
            with xr.open_dataset(path, decode_times=False) as ds:
                whats = [_LINE.match(x)["what"] for x in ds.attrs["history"].splitlines()]
            assert len(whats) == 2 and whats[1] == step, (path, whats)
            assert re.fullmatch(r"profile \d+ extracted", whats[0])


def test_rsi_pipeline_products_name_their_code(sample_p, tmp_path):
    from odas_tpw.rsi.pipeline import run_pipeline

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_pipeline([sample_p], tmp_path, fft_length=256, chi_fft_length=256)
    steps = {_assert_stamped(p) for p in tmp_path.rglob("*.nc")}
    assert steps >= {"epsilon (L4)", "binning (L5)", "combining profiles (L6)"}


def test_l1_lineage_reaches_the_profiles(sample_p, tmp_path):
    """A product made from one file keeps that file's history, oldest first."""
    from odas_tpw.rsi.convert import p_to_L1
    from odas_tpw.rsi.profile import extract_profiles

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, l1 = p_to_L1(sample_p, tmp_path / "l1.nc")
        assert _assert_stamped(l1) == f"L1 converted from {sample_p.name}"
        profs = extract_profiles(l1, tmp_path / "profiles")
    assert profs
    with xr.open_dataset(profs[0], decode_times=False) as ds:
        steps = [_LINE.match(x)["what"] for x in ds.attrs["history"].splitlines()]
    assert steps == [f"L1 converted from {sample_p.name}", "profile 1 extracted"]


def _binned_input(path: Path) -> None:
    ds = xr.Dataset(
        {"T": (["bin", "profile"], np.ones((3, 2)))},
        coords={"bin": np.arange(3.0), "profile": np.arange(2, dtype=np.int32)},
        attrs={"history": "an input's lineage", vi.SOFTWARE_ATTR: "microstructure-tpw old"},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def test_combo_keeps_a_configured_history_not_its_first_input(tmp_path):
    from odas_tpw.perturb.combo import make_combo
    from odas_tpw.perturb.netcdf_schema import COMBO_SCHEMA

    _binned_input(tmp_path / "binned" / "binned00.nc")
    out = make_combo(
        tmp_path / "binned", tmp_path / "combo", COMBO_SCHEMA,
        netcdf_attrs={"history": "configured by the operator"},
    )
    with xr.open_dataset(out, decode_times=False) as ds:
        lines = ds.attrs["history"].splitlines()
        assert ds.attrs[vi.SOFTWARE_ATTR] == vi.software_version()
        inputs = ds.attrs[vi.INPUTS_ATTR]
    assert lines[0] == "configured by the operator" and len(lines) == 2
    assert _last_step("\n".join(lines)) == "combo"
    assert inputs == "microstructure-tpw old"  # the code that made its input


def _profile_input(path: Path, software: str | None) -> Path:
    ds = xr.Dataset({"T": (["time"], np.arange(5.0))}, coords={"depth": ("time", np.arange(5.0))})
    if software:
        ds.attrs[vi.SOFTWARE_ATTR] = software
    ds.to_netcdf(path)
    return path


def test_binning_records_its_inputs_codes(tmp_path):
    """Binned files are rebuilt from cached per-file products, which another
    code state may have written; an unstamped one says so."""
    from odas_tpw.perturb.binning import bin_by_depth

    files = [
        _profile_input(tmp_path / "a.nc", "microstructure-tpw old"),
        _profile_input(tmp_path / "b.nc", None),
        _profile_input(tmp_path / "c.nc", "microstructure-tpw old"),
    ]
    ds = bin_by_depth(files, bin_width=1.0)
    assert ds.attrs[vi.INPUTS_ATTR] == "(none recorded)\nmicrostructure-tpw old"


def test_input_versions_fold_in_their_inputs():
    attrs = [
        {vi.SOFTWARE_ATTR: "B", vi.INPUTS_ATTR: "A\n(none recorded)"},
        {vi.SOFTWARE_ATTR: "B", vi.INPUTS_ATTR: "C"},
    ]
    assert vi.input_versions(attrs) == "(none recorded)\nA\nB\nC"


def test_combo_does_not_inherit_its_first_inputs_history(tmp_path):
    """With nothing configured (netcdf.history: null, the default)."""
    from odas_tpw.perturb.combo import make_combo
    from odas_tpw.perturb.netcdf_schema import COMBO_SCHEMA

    _binned_input(tmp_path / "binned" / "binned00.nc")
    out = make_combo(tmp_path / "binned", tmp_path / "combo", COMBO_SCHEMA)
    with xr.open_dataset(out, decode_times=False) as ds:
        history = ds.attrs["history"]
    assert len(history.splitlines()) == 1 and _last_step(history) == "combo"


def test_time_binning_records_its_inputs_codes(tmp_path):
    from odas_tpw.perturb.binning import bin_by_time

    files = []
    for name, software in (("a", "microstructure-tpw old"), ("b", None)):
        ds = xr.Dataset({"T": (["t_slow"], np.arange(6.0))},
                        coords={"t_slow": ("t_slow", np.arange(6.0))})
        if software:
            ds.attrs[vi.SOFTWARE_ATTR] = software
        ds.to_netcdf(tmp_path / f"{name}.nc")
        files.append(tmp_path / f"{name}.nc")
    out = bin_by_time(files, bin_width=2.0)
    assert out.attrs[vi.INPUTS_ATTR] == "(none recorded)\nmicrostructure-tpw old"


def test_binned_history_starts_fresh(tmp_path):
    from odas_tpw.perturb.pipeline import _write_binned_or_clear

    _binned_input(tmp_path / "in.nc")
    (tmp_path / "out").mkdir()
    with xr.open_dataset(tmp_path / "in.nc") as ds:
        _write_binned_or_clear(ds.load(), tmp_path / "out")
    assert _assert_stamped(tmp_path / "out" / "binned.nc") == "binning"
    with xr.open_dataset(tmp_path / "out" / "binned.nc") as ds:
        assert "lineage" not in ds.attrs["history"]


# ---------------------------------------------------------------------------
# Records and logs
# ---------------------------------------------------------------------------


def test_patch_banner_names_the_code():
    """patch-config and fp07-cal patch; it used to print stale metadata."""
    from odas_tpw.rsi import config_patch as cp

    banner = cp._build_banner("W", "a", "n", "", "\n")
    assert f"-- {cp.TOOL_NAME} {vi.running_version()}" in banner


def test_fp07cal_record_names_the_code(tmp_path):
    from odas_tpw.fp07cal.cli import main

    out = tmp_path / "demo"
    assert main(["demo", "-o", str(out), "--yos", "12", "--no-figure"]) == 0
    record = json.loads((out / "coefficients.json").read_text())
    assert record["software_version"] == vi.software_version()


@pytest.fixture
def _restore_root_logger():
    root = logging.getLogger()
    saved, level = list(root.handlers), root.level
    yield
    for h in list(root.handlers):
        if h not in saved:
            root.removeHandler(h)
            h.close()
    root.setLevel(level)


@pytest.mark.parametrize("level", ["INFO", "WARNING", "ERROR"])
def test_run_log_starts_with_the_code(tmp_path, _restore_root_logger, level):
    """At every --log-level, in the words the products use."""
    from odas_tpw.perturb.cli import _install_logging

    args = argparse.Namespace(output=str(tmp_path), log_level=level, stdout=False)
    log = _install_logging(args, {})
    logging.getLogger("odas_tpw.perturb").info("an INFO line the level may drop")
    for h in logging.getLogger().handlers:
        h.flush()
    assert log.read_text().splitlines()[0].endswith(f": {vi.software_version()}")


def test_worker_log_names_the_code(tmp_path, _restore_root_logger):
    from odas_tpw.perturb.logging_setup import init_worker_logging

    log = init_worker_logging(tmp_path, "20260101T000000Z")
    for h in logging.getLogger().handlers:
        h.flush()
    assert f"worker {os.getpid()}: {vi.software_version()}" in log.read_text()
