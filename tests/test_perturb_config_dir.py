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
