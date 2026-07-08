# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for the ``<CONFIG_DIR>`` YAML-relative-path feature.

A path written ``<CONFIG_DIR>/sub`` in a perturb config resolves against the
config file's own directory at filesystem-access time, but the *token* (not the
resolved absolute path) is what feeds the stage-dir signatures — so the same
config on a different mount point still matches its output directories.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest

from odas_tpw.perturb import config as C
from odas_tpw.perturb.config import config_dir_of, expand_config_dir, load_config


def _write_cfg(directory: Path, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "perturb.yaml"
    path.write_text(body)
    return path


# --------------------------------------------------------------------------- #
# expand_config_dir (the core substitution)
# --------------------------------------------------------------------------- #


def test_expand_joins_relative_remainder():
    assert expand_config_dir("<CONFIG_DIR>/a/b", "/root") == os.path.join("/root", "a", "b")


def test_expand_bare_token_is_the_dir():
    assert expand_config_dir("<CONFIG_DIR>", "/root") == "/root"


def test_expand_strips_leading_separators():
    # both posix and windows separators after the token collapse to a join
    assert expand_config_dir("<CONFIG_DIR>/x", "/root") == os.path.join("/root", "x")
    assert expand_config_dir("<CONFIG_DIR>\\x", "/root") == os.path.join("/root", "x")


def test_expand_passthrough_without_token():
    assert expand_config_dir("/abs/path", "/root") == "/abs/path"
    assert expand_config_dir("rel/path", "/root") == "rel/path"
    assert expand_config_dir("VMP/", "/root") == "VMP/"


def test_expand_passthrough_non_string():
    assert expand_config_dir(None, "/root") is None
    assert expand_config_dir(42, "/root") == 42


def test_expand_token_only_at_start():
    # the token only substitutes as a prefix, per the spec
    assert expand_config_dir("/x/<CONFIG_DIR>/y", "/root") == "/x/<CONFIG_DIR>/y"


def test_expand_token_without_config_dir_raises():
    with pytest.raises(ValueError, match="CONFIG_DIR"):
        expand_config_dir("<CONFIG_DIR>/x", None)
    with pytest.raises(ValueError, match="CONFIG_DIR"):
        expand_config_dir("<CONFIG_DIR>/x", "")


# --------------------------------------------------------------------------- #
# load_config stamps config_dir
# --------------------------------------------------------------------------- #


def test_load_stamps_config_dir(tmp_path):
    p = _write_cfg(tmp_path / "cfg", "files:\n  p_file_root: VMP/\n")
    cfg = load_config(str(p))
    assert config_dir_of(cfg) == str((tmp_path / "cfg").resolve())
    assert cfg["files"]["config_dir"] == str((tmp_path / "cfg").resolve())


def test_load_stamps_config_dir_when_no_files_section(tmp_path):
    p = _write_cfg(tmp_path / "cfg", "epsilon:\n  fit_order: 3\n")
    cfg = load_config(str(p))
    assert config_dir_of(cfg) == str((tmp_path / "cfg").resolve())


def test_config_dir_of_none_for_in_memory_config():
    assert config_dir_of({"files": {"output_root": "results/"}}) is None
    assert config_dir_of({}) is None


# --------------------------------------------------------------------------- #
# Hash portability — the whole point of the feature
# --------------------------------------------------------------------------- #


def _sig(cfg: dict, stage: str) -> str:
    section, params, upstream = C.stage_signature(stage, cfg)
    return C.canonicalize(section, params, upstream)


def test_config_dir_never_affects_the_hash():
    """config_dir is hash-excluded, so adding the feature must not invalidate a
    single existing signature: absent / None / any value all hash identically."""
    base = {"files": {"p_file_root": "VMP/", "output_root": "results/"}}
    absent = _sig(base, "diss")
    for cdir in (None, "/Volumes/A/x", "/mnt/b/y"):
        withdir = {"files": {**base["files"], "config_dir": cdir}}
        assert _sig(withdir, "diss") == absent
    assert "config_dir" not in absent


def test_token_paths_hash_identically_across_mounts(tmp_path):
    """The same YAML at two different absolute locations produces identical
    stage signatures for every stage — a run's cached output dirs survive a
    move/remount of the config + data tree."""
    body = (
        "files:\n"
        "  p_file_root: <CONFIG_DIR>/VMP\n"
        "  p_file_pattern: '**/*.p'\n"
        "  output_root: <CONFIG_DIR>/results\n"
        "gps:\n"
        "  file: <CONFIG_DIR>/gps.nc\n"
        "hotel:\n"
        "  enable: true\n"
        "  file: <CONFIG_DIR>/hotel.csv\n"
    )
    p1 = _write_cfg(tmp_path / "mountA" / "data", body)
    p2 = _write_cfg(tmp_path / "mountB" / "elsewhere" / "data", body)
    c1, c2 = load_config(str(p1)), load_config(str(p2))
    assert config_dir_of(c1) != config_dir_of(c2)  # genuinely different locations
    for stage in sorted(C.STAGES):
        assert _sig(c1, stage) == _sig(c2, stage), stage


def test_token_and_absolute_differ_in_hash(tmp_path):
    """A <CONFIG_DIR> path and the equivalent hand-written absolute path are NOT
    the same string, so they hash differently (the token is what is portable)."""
    p = _write_cfg(tmp_path / "cfg", "files:\n  p_file_root: <CONFIG_DIR>/VMP\n")
    cfg_token = load_config(str(p))
    cfg_abs = {"files": {"p_file_root": f"{config_dir_of(cfg_token)}/VMP"}}
    assert _sig(cfg_token, "diss") != _sig(cfg_abs, "diss")


# --------------------------------------------------------------------------- #
# No token leaks to the filesystem (consumers expand)
# --------------------------------------------------------------------------- #


def test_pipeline_roots_are_expanded(tmp_path):
    from odas_tpw.perturb.pipeline import (
        _configured_input_root,
        _configured_output_root,
    )

    p = _write_cfg(
        tmp_path / "cfg",
        "files:\n  p_file_root: <CONFIG_DIR>/VMP\n  output_root: <CONFIG_DIR>/results\n",
    )
    cfg = load_config(str(p))
    cdir = str((tmp_path / "cfg").resolve())
    assert str(_configured_input_root(cfg)) == os.path.join(cdir, "VMP")
    assert str(_configured_output_root(cfg)) == os.path.join(cdir, "results")
    assert "<CONFIG_DIR>" not in str(_configured_input_root(cfg))
    assert "<CONFIG_DIR>" not in str(_configured_output_root(cfg))


def test_relative_roots_unchanged(tmp_path):
    """A plain relative path (no token) is untouched — the default behavior."""
    from odas_tpw.perturb.pipeline import _configured_input_root

    p = _write_cfg(tmp_path / "cfg", "files:\n  p_file_root: VMP/\n")
    cfg = load_config(str(p))
    assert str(_configured_input_root(cfg)) == "VMP"


# --------------------------------------------------------------------------- #
# Plot-time resolution (resolve.py) expands the physical output root
# --------------------------------------------------------------------------- #


def test_stage_dir_scans_expanded_output_root(tmp_path):
    from odas_tpw.perturb.resolve import stage_dir

    p = _write_cfg(tmp_path / "cfg", "files:\n  output_root: <CONFIG_DIR>/results\n")
    cfg = load_config(str(p))
    with pytest.raises(FileNotFoundError) as excinfo:
        stage_dir(cfg, "profiles")  # nothing on disk -> error names the dir it scanned
    msg = str(excinfo.value)
    assert os.path.join(str((tmp_path / "cfg").resolve()), "results") in msg
    assert "<CONFIG_DIR>" not in msg


def test_require_root_expands_token(tmp_path):
    from odas_tpw.perturb.resolve import require_root

    p = _write_cfg(tmp_path / "cfg", "files:\n  output_root: <CONFIG_DIR>/results\n")
    args = argparse.Namespace(root=None, config=str(p))
    assert require_root(args) == os.path.join(str((tmp_path / "cfg").resolve()), "results")


# --------------------------------------------------------------------------- #
# Adversarial-review additions: consumer expansion, discovery, invariants
# --------------------------------------------------------------------------- #


def test_load_does_not_expand_stored_paths(tmp_path):
    """Portability rests on the token surviving load UNCHANGED — expansion must
    happen only at filesystem access, never in the stored (hashed) config."""
    p = _write_cfg(
        tmp_path / "cfg",
        "files:\n  p_file_root: <CONFIG_DIR>/VMP\n"
        "gps:\n  file: <CONFIG_DIR>/gps.nc\n"
        "hotel:\n  file: <CONFIG_DIR>/hotel.csv\n",
    )
    cfg = load_config(str(p))
    assert cfg["files"]["p_file_root"] == "<CONFIG_DIR>/VMP"
    assert cfg["gps"]["file"] == "<CONFIG_DIR>/gps.nc"
    assert cfg["hotel"]["file"] == "<CONFIG_DIR>/hotel.csv"


def test_config_dir_excluded_from_hash_every_stage():
    """The hash-exclusion invariant must hold for EVERY stage, not just one."""
    base = {"files": {"p_file_root": "VMP/", "output_root": "results/"}}
    for stage in sorted(C.STAGES):
        absent = _sig(base, stage)
        withdir = {"files": {**base["files"], "config_dir": "/some/mount/point"}}
        assert _sig(withdir, stage) == absent, stage
        assert "config_dir" not in absent


def test_gps_and_hotel_files_expanded_at_consumer(tmp_path):
    """gps.file / hotel.file must resolve the token before they are stat'd — the
    hash test alone passes even if these leaked the literal token to disk."""
    from odas_tpw.perturb.pipeline import _external_input_fingerprints

    d = tmp_path / "cfg"
    d.mkdir()
    (d / "gps.nc").write_bytes(b"\x00")
    (d / "hotel.csv").write_text("t\n")
    p = _write_cfg(
        d,
        "gps:\n  source: netcdf\n  file: <CONFIG_DIR>/gps.nc\n"
        "hotel:\n  enable: true\n  file: <CONFIG_DIR>/hotel.csv\n",
    )
    cfg = load_config(str(p))
    hotel_cfg = C.merge_config("hotel", cfg.get("hotel"))
    fp = _external_input_fingerprints(cfg, hotel_cfg)
    # a successful stat (has "size") proves the token resolved to the real file;
    # an unexpanded token would not exist and report {"missing": True}
    assert "size" in fp["gps"], fp
    assert "size" in fp["hotel"], fp


def test_missing_gps_hotel_after_expansion_reports_missing(tmp_path):
    from odas_tpw.perturb.pipeline import _external_input_fingerprints

    d = tmp_path / "cfg"
    d.mkdir()
    p = _write_cfg(
        d,
        "gps:\n  source: netcdf\n  file: <CONFIG_DIR>/nope.nc\n"
        "hotel:\n  enable: true\n  file: <CONFIG_DIR>/nope.csv\n",
    )
    cfg = load_config(str(p))
    hotel_cfg = C.merge_config("hotel", cfg.get("hotel"))
    fp = _external_input_fingerprints(cfg, hotel_cfg)
    assert fp["gps"] == {"missing": True}
    assert fp["hotel"] == {"missing": True}


def test_pipeline_discovers_p_files_under_config_dir(tmp_path):
    """End-to-end: the resolved input root actually finds files on disk."""
    from odas_tpw.perturb.discover import find_p_files
    from odas_tpw.perturb.pipeline import _configured_input_root

    d = tmp_path / "cfg"
    (d / "VMP").mkdir(parents=True)
    (d / "VMP" / "a.p").write_bytes(b"\x00")
    p = _write_cfg(d, "files:\n  p_file_root: <CONFIG_DIR>/VMP\n")
    cfg = load_config(str(p))
    found = find_p_files(_configured_input_root(cfg), "**/*.p")
    assert [f.name for f in found] == ["a.p"]


def test_cli_output_override_expands_token(tmp_path):
    from odas_tpw.perturb.cli import _resolve_output_root

    p = _write_cfg(tmp_path / "cfg", "files:\n  output_root: results/\n")
    cfg = load_config(str(p))
    got = _resolve_output_root(argparse.Namespace(output="<CONFIG_DIR>/out"), cfg)
    assert str(got) == os.path.join(str((tmp_path / "cfg").resolve()), "out")


def test_cli_output_token_without_config_fails_closed():
    from odas_tpw.perturb.cli import _resolve_output_root

    # no config loaded -> no config_dir -> the token cannot resolve; must raise
    # (fail-closed) rather than write to a dir literally named <CONFIG_DIR>
    with pytest.raises(ValueError, match="CONFIG_DIR"):
        _resolve_output_root(argparse.Namespace(output="<CONFIG_DIR>/out"), {})


def test_expand_pathlike_token_does_not_leak():
    from pathlib import Path as _P

    out = expand_config_dir(_P("<CONFIG_DIR>/x"), "/root")
    assert "<CONFIG_DIR>" not in str(out)
    assert str(out) == os.path.join("/root", "x")
    # a plain Path (no token) is returned unchanged, original type preserved
    assert expand_config_dir(_P("plain/x"), "/root") == _P("plain/x")


def test_expand_token_without_separator_is_lenient():
    # documented behavior: "<CONFIG_DIR>foo" joins as config_dir/foo
    assert expand_config_dir("<CONFIG_DIR>foo", "/root") == os.path.join("/root", "foo")
