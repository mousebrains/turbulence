# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for ``--version``: which release and commit is running.

The git cases build throwaway repositories holding a fake ``src/odas_tpw``
tree and point the module at it, so they do not depend on this checkout's
history (CI clones are shallow and carry no tags).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from odas_tpw import _version_info as vi

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"][
    "scripts"
]

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")


# ---------------------------------------------------------------------------
# Every console script
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", sorted(_SCRIPTS))
def test_every_console_script_has_version(script, monkeypatch, capsys):
    """``<script> --version`` prints one line naming the script, and exits 0.

    Walks ``[project.scripts]``, so a new CLI without ``--version`` fails here.
    """
    module_path, func = _SCRIPTS[script].split(":")
    main = getattr(importlib.import_module(module_path), func)
    monkeypatch.setattr(vi, "version_string", lambda: "9.9.9 (git v9.9.9-0-gabcdef012345)")
    monkeypatch.setattr(sys, "argv", [script, "--version"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code in (0, None)
    assert capsys.readouterr().out == f"{script} 9.9.9 (git v9.9.9-0-gabcdef012345)\n"


def test_version_is_computed_only_when_asked(monkeypatch):
    def _boom():
        raise AssertionError("version_string() ran without --version")

    monkeypatch.setattr(vi, "version_string", _boom)
    parser = argparse.ArgumentParser(prog="x")
    vi.add_version_argument(parser)
    parser.add_argument("n")
    assert parser.parse_args(["1"]).n == "1"


def test_version_line_is_not_wrapped(monkeypatch, capsys):
    """argparse's own version action reflows text at the terminal width."""
    long = "0.4.0 (git v0.4.0-57-g5f6572b4a1c2, uncommitted changes, installed metadata says 0.3.1)"
    monkeypatch.setattr(vi, "version_string", lambda: long)
    monkeypatch.setenv("COLUMNS", "20")
    parser = argparse.ArgumentParser(prog="x")
    vi.add_version_argument(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--version"])
    assert capsys.readouterr().out == f"x {long}\n"


def test_real_checkout_reports_something():
    """Smoke test against whatever is actually installed: never raises."""
    text = vi.version_string()
    assert text and "\n" not in text


# ---------------------------------------------------------------------------
# Throwaway git repositories
# ---------------------------------------------------------------------------


def _run_git(cwd: Path, *args: str) -> str:
    """git as the fixtures run it: the caller's repository pointers dropped."""
    env = {k: v for k, v in os.environ.items() if k not in vi._REPO_LOCAL_ENV}
    proc = subprocess.run(
        ["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


@pytest.fixture(autouse=True)
def _isolated_git(tmp_path, monkeypatch):
    """Shut the developer's own git configuration (hooks, signing, templates,
    fsmonitor, autocrlf) out of the fixtures *and* the code under test."""
    empty = tmp_path / "empty.gitconfig"
    empty.touch()
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "t")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "t@example.com")


def _write_pyproject(root: Path, version: str) -> None:
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "microstructure-tpw"\nversion = "{version}"\n'
    )


def _source_tree(root: Path, version: str = "9.8.7") -> Path:
    """A git repository shaped like this one; returns its ``src/odas_tpw``."""
    pkg = root / "src" / "odas_tpw"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("x = 1\n")
    (root / "docs").mkdir()
    (root / "docs" / "notes.md").write_text("notes\n")
    (root / ".gitignore").write_text("__pycache__/\n*.pyc\n")
    _write_pyproject(root, version)
    _run_git(root, "init", "-q")
    _run_git(root, "add", ".")
    _run_git(root, "commit", "-q", "-m", "init")
    return pkg.resolve()


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A fake source tree, imported as if it were ``odas_tpw``; not installed."""
    repo = tmp_path / "repo"
    repo.mkdir()
    pkg = _source_tree(repo)
    monkeypatch.setattr(vi, "_package_root", lambda: pkg)
    monkeypatch.setattr(vi, "_distribution", lambda: None)
    return repo


@needs_git
def test_at_release_tag(tree):
    _run_git(tree, "tag", "v9.8.7")
    head = _run_git(tree, "rev-parse", "HEAD")
    info = vi.version_info()
    assert (info.source, info.release, info.commit, info.dirty) == ("git", "9.8.7", head, False)
    assert str(info) == f"9.8.7 (git v9.8.7-0-g{head[:12]})"


@needs_git
def test_commits_after_tag(tree):
    _run_git(tree, "tag", "v9.8.7")
    (tree / "docs" / "notes.md").write_text("more\n")
    _run_git(tree, "commit", "-q", "-am", "docs")
    head = _run_git(tree, "rev-parse", "HEAD")
    assert vi.version_string() == f"9.8.7 (git v9.8.7-1-g{head[:12]})"


@needs_git
def test_no_tag_reports_the_bare_commit(tree):
    """A shallow CI clone carries no tags."""
    head = _run_git(tree, "rev-parse", "HEAD")
    info = vi.version_info()
    assert info.describe is None
    assert str(info) == f"9.8.7 (git {head[:12]})"


@needs_git
def test_non_release_tags_are_ignored(tree):
    _run_git(tree, "tag", "scratch")
    assert vi.version_info().describe is None


@needs_git
@pytest.mark.parametrize(
    ("relpath", "content", "stage", "dirty"),
    [
        ("src/odas_tpw/__init__.py", "x = 2\n", False, True),  # modified
        ("src/odas_tpw/__init__.py", "x = 2\n", True, True),  # staged
        ("src/odas_tpw/new.py", "y = 1\n", False, True),  # untracked source
        ("src/odas_tpw/__pycache__/a.pyc", "b", False, False),  # ignored bytecode
        ("src/odas_tpw/results/__init__.py", "w = 1\n", False, True),  # ignored, importable
        ("src/odas_tpw/results/notes.txt", "n\n", False, False),  # ignored, not code
        ("docs/notes.md", "edited\n", False, False),  # outside the package
        ("tests/test_new.py", "z = 1\n", False, False),  # outside the package
    ],
)
def test_uncommitted_changes_are_scoped_to_the_package(tree, relpath, content, stage, dirty):
    """The root .gitignore's unanchored rules (`results/`, `MR/`) also match
    inside the package, where git status cannot see what they hide."""
    with (tree / ".gitignore").open("a") as fh:
        fh.write("results/\n")
    _run_git(tree, "commit", "-q", "-am", "ignore results")
    path = tree / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    if stage:
        _run_git(tree, "add", relpath)
    info = vi.version_info()
    assert info.dirty is dirty
    assert ("uncommitted changes" in str(info)) is dirty


@needs_git
def test_uncommitted_changes_carry_a_source_digest(tree):
    """'uncommitted changes' alone cannot tell two modified trees apart."""
    pkg = tree / "src" / "odas_tpw"
    assert vi.version_info().source_digest is None  # clean: the commit says it all
    (pkg / "__init__.py").write_text("x = 2\n")
    first = vi.version_info()
    assert re.fullmatch(r"[0-9a-f]{12}", first.source_digest or "")
    assert f"uncommitted changes, source {first.source_digest})" in str(first)
    (pkg / "__init__.py").write_text("x = 3\n")
    assert vi.version_info().source_digest != first.source_digest


@pytest.mark.parametrize(
    "info",
    [
        vi.VersionInfo("1.0", "a" * 40, "v1.0-3-gaaaaaaaaaaaa", True, "git",
                       metadata_release="0.9", source_digest="b" * 12),
        vi.VersionInfo("1.0", "a" * 40, None, None, "git"),
        vi.VersionInfo("1.0", "a" * 40, source="pip-vcs", metadata_release="0.9"),
        vi.VersionInfo("1.0", "a" * 40, source="release"),
        vi.VersionInfo("1.0", source="local-build", detail="file:///x", metadata_release="0.9"),
        vi.VersionInfo("1.0", source="unknown", detail="not a release build"),
        vi.VersionInfo("unknown", source="unknown"),
    ],
)
def test_version_text_has_no_semicolon(info):
    """`rsi-tpw v1to6` writes it into an INI value, where ';' is a comment."""
    assert ";" not in str(info) and "\n" not in str(info)


@needs_git
def test_inherited_git_dir_does_not_redirect(tree, tmp_path, monkeypatch):
    """A caller's GIT_DIR (inside a git hook, say) must not name another repo."""
    other = tmp_path / "other"
    other.mkdir()
    _source_tree(other)
    (other / "docs" / "notes.md").write_text("diverge\n")
    _run_git(other, "commit", "-q", "-am", "diverge")
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    assert vi.version_info().commit == _run_git(tree, "rev-parse", "HEAD")


@needs_git
def test_untracked_copy_inside_another_repo_is_not_attributed(tmp_path, monkeypatch):
    """A non-editable install into a .venv inside some project's checkout."""
    project = tmp_path / "project"
    project.mkdir()
    _source_tree(project)
    (project / ".gitignore").write_text(".venv/\n")
    _run_git(project, "commit", "-q", "-am", "ignore venv")
    pkg = project / ".venv" / "lib" / "site-packages" / "odas_tpw"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("x = 1\n")
    monkeypatch.setattr(vi, "_package_root", lambda: pkg.resolve())
    monkeypatch.setattr(vi, "_distribution", lambda: None)
    info = vi.version_info()
    assert info.source == "unknown" and info.commit is None
    assert "commit unknown" in str(info)


@needs_git
def test_git_missing_falls_back(tree, monkeypatch):
    def _no_git(*_a, **_k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(vi.subprocess, "run", _no_git)
    info = vi.version_info()
    assert info.source == "unknown" and info.release == "9.8.7"


@needs_git
def test_git_timeout_falls_back(tree, monkeypatch):
    def _hang(cmd, **_k):
        raise subprocess.TimeoutExpired(cmd, vi._GIT_TIMEOUT_S)

    monkeypatch.setattr(vi.subprocess, "run", _hang)
    assert vi.version_info().source == "unknown"


@needs_git
def test_stale_editable_metadata_is_named(tree, monkeypatch):
    """An editable install's metadata is frozen at install time."""
    record = json.dumps({"url": tree.resolve().as_uri(), "dir_info": {"editable": True}})
    monkeypatch.setattr(vi, "_distribution", lambda: _FakeDist("0.3.1", record, tree))
    text = vi.version_string()
    assert text.startswith("9.8.7 (git ")
    assert text.endswith(", installed metadata says 0.3.1)")


@needs_git
def test_another_checkouts_metadata_is_not_named(tree, tmp_path, monkeypatch):
    """A PYTHONPATH worktree: the venv's metadata is the main checkout's, and
    `pip install -e .` here would re-point the shared venv."""
    main = (tmp_path / "main-checkout").resolve()
    record = json.dumps({"url": main.as_uri(), "dir_info": {"editable": True}})
    monkeypatch.setattr(vi, "_distribution", lambda: _FakeDist("0.3.1", record, main))
    text = vi.version_string()
    assert text.startswith("9.8.7 (git ") and "metadata" not in text


@needs_git
def test_uncommitted_pyproject_bump_is_flagged(tree):
    """The release is read from pyproject.toml, so an edit to it is a change."""
    _run_git(tree, "tag", "v9.8.7")
    _write_pyproject(tree, "9.9.0")
    info = vi.version_info()
    assert (info.release, info.dirty) == ("9.9.0", True)
    assert ", uncommitted changes, source " in str(info)


@needs_git
def test_inherited_index_file_does_not_redirect(tree, tmp_path, monkeypatch):
    """Every git hook runs with GIT_INDEX_FILE set."""
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "some-other-index"))
    info = vi.version_info()
    assert (info.source, info.commit, info.dirty) == (
        "git",
        _run_git(tree, "rev-parse", "HEAD"),
        False,
    )


@needs_git
def test_every_git_call_disables_fsmonitor(tree, monkeypatch):
    """`git status` under core.fsmonitor=true leaves a daemon running."""
    calls = []
    real_run = subprocess.run

    def _spy(cmd, **kwargs):
        calls.append(cmd)
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(vi.subprocess, "run", _spy)
    vi.version_info()
    assert calls and all(cmd[:3] == ["git", "-c", "core.fsmonitor=false"] for cmd in calls)


@needs_git
def test_checkout_owned_by_another_user(tree, monkeypatch):
    """git refuses (safe.directory): say so instead of a bare 'unknown'."""

    def _refuse(cmd, **_k):
        return subprocess.CompletedProcess(
            cmd, 128, "", "fatal: detected dubious ownership in repository at '/x'\n"
        )

    monkeypatch.setattr(vi.subprocess, "run", _refuse)
    info = vi.version_info()
    assert (info.source, info.commit) == ("unknown", None)
    assert "owned by another user" in str(info) and "commit unknown" in str(info)


# ---------------------------------------------------------------------------
# PEP 610 (no git repository tracks the package)
# ---------------------------------------------------------------------------


class _FakeDist:
    def __init__(self, version: str, direct_url: str | None, location: Path):
        self.version = version
        self._direct_url = direct_url
        self._location = location

    def read_text(self, name: str) -> str | None:
        return self._direct_url if name == "direct_url.json" else None

    def locate_file(self, rel: str) -> Path:
        return self._location / rel


@pytest.fixture
def site(tmp_path, monkeypatch):
    """An installed copy of the package, outside any git repository."""
    root = tmp_path / "site-packages"
    pkg = root / "odas_tpw"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("x = 1\n")
    monkeypatch.setattr(vi, "_package_root", lambda: pkg.resolve())
    monkeypatch.setattr(vi, "_git_state", lambda _root, with_pyproject=False: (None, None))
    return root.resolve()


def _install(monkeypatch, record, location):
    text = record if record is None or isinstance(record, str) else json.dumps(record)
    monkeypatch.setattr(vi, "_distribution", lambda: _FakeDist("1.2.3", text, location))


_SHA = "2846bf3666dc008354f8d78a5e26b126bf163a8b"


def _stamp(site: Path, **stamp: str) -> None:
    (site / "odas_tpw" / vi.RELEASE_STAMP).write_text(json.dumps(stamp))


def test_release_wheel_reports_release_and_commit(site, monkeypatch):
    _stamp(site, commit=_SHA, tag="v1.2.3")
    _install(monkeypatch, None, site)
    info = vi.version_info()
    assert (info.source, info.commit) == ("release", _SHA)
    assert str(info) == "1.2.3 (release, git 2846bf3666dc)"


def test_release_wheel_installed_from_its_file(site, monkeypatch):
    """pip records a wheel installed by path as an archive; the stamp still holds."""
    _stamp(site, commit=_SHA, tag="v1.2.3")
    url = "file:///downloads/microstructure_tpw-1.2.3-py3-none-any.whl"
    _install(monkeypatch, {"url": url, "archive_info": {}}, site)
    assert vi.version_string() == "1.2.3 (release, git 2846bf3666dc)"


def test_stamp_is_ignored_for_a_directory_build(site, monkeypatch):
    _stamp(site, commit=_SHA, tag="v1.2.3")
    _install(monkeypatch, {"url": "file:///src/turbulence", "dir_info": {}}, site)
    assert vi.version_info().source == "local-build"


@pytest.mark.parametrize(
    "stamp",
    [
        None,  # a wheel built anywhere but the release workflow
        {"commit": _SHA, "tag": "v1.2.4"},  # names another release
        {"commit": "2846bf3", "tag": "v1.2.3"},  # not a full hash
        "{not json",
    ],
)
def test_unstamped_wheel_is_not_a_release(site, monkeypatch, stamp):
    """Installed by name from a wheelhouse: no direct_url.json, like PyPI."""
    if isinstance(stamp, dict):
        _stamp(site, **stamp)
    elif isinstance(stamp, str):
        (site / "odas_tpw" / vi.RELEASE_STAMP).write_text(stamp)
    _install(monkeypatch, None, site)
    info = vi.version_info()
    assert (info.source, info.commit) == ("unknown", None)
    assert str(info) == "1.2.3 (not a release build, commit unknown)"


def test_vcs_install_reports_the_commit(site, monkeypatch):
    commit = "5f6572b9a1e8dd224e0e5f87082e55648e522a6f"
    record = {
        "url": "https://github.com/mousebrains/turbulence",
        "vcs_info": {"vcs": "git", "commit_id": commit, "requested_revision": "main"},
    }
    _install(monkeypatch, record, site)
    assert vi.version_string() == "1.2.3 (git 5f6572b9a1e8)"


@pytest.mark.parametrize(
    ("record", "where"),
    [
        ({"url": "file:///src/turbulence", "dir_info": {}}, "file:///src/turbulence"),
        (
            {"url": "https://example.com/v1.2.3.zip", "archive_info": {}},
            "https://example.com/v1.2.3.zip",
        ),
        ("{not json", "an unrecorded location"),
        ("[1, 2]", "an unrecorded location"),
    ],
)
def test_local_builds_say_commit_unknown(site, monkeypatch, record, where):
    _install(monkeypatch, record, site)
    assert vi.version_string() == f"1.2.3 (built from {where}, commit unknown)"


def test_editable_install_without_git(tmp_path, monkeypatch):
    """An editable install of a source tree that is not a git checkout."""
    root = tmp_path / "unpacked"
    pkg = root / "src" / "odas_tpw"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("x = 1\n")
    _write_pyproject(root, "2.0")
    monkeypatch.setattr(vi, "_package_root", lambda: pkg.resolve())
    monkeypatch.setattr(vi, "_git_state", lambda _root, with_pyproject=False: (None, None))
    url = root.resolve().as_uri()
    _install(monkeypatch, {"url": url, "dir_info": {"editable": True}}, tmp_path / "elsewhere")
    assert vi.version_string() == (
        f"2.0 (built from {url}, commit unknown, installed metadata says 1.2.3)"
    )


def test_stray_egg_info_is_not_a_release(tmp_path, monkeypatch):
    """PYTHONPATH=<checkout>/src with git unavailable finds src/*.egg-info,
    which has no direct_url.json: that is a source tree, not a PyPI install."""
    root = tmp_path / "checkout"
    pkg = root / "src" / "odas_tpw"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("x = 1\n")
    _write_pyproject(root, "2.0")
    monkeypatch.setattr(vi, "_package_root", lambda: pkg.resolve())
    monkeypatch.setattr(vi, "_git_state", lambda _root, with_pyproject=False: (None, None))
    _install(monkeypatch, None, (root / "src").resolve())
    info = vi.version_info()
    assert info.source == "unknown"
    assert str(info).startswith("2.0 (running from the source tree ")
    assert "commit unknown" in str(info)


def test_shadowed_install_is_not_attributed(site, tmp_path, monkeypatch):
    """PYTHONPATH puts other code ahead of the installed copy."""
    _install(monkeypatch, None, tmp_path / "the-real-site-packages")
    info = vi.version_info()
    assert (info.source, info.release) == ("unknown", "unknown")  # not the other copy's 1.2.3
    assert str(info).endswith("not the installed microstructure-tpw, commit unknown)")


def test_editable_record_for_another_tree_is_not_attributed(site, tmp_path, monkeypatch):
    other = (tmp_path / "main-checkout").resolve()
    _install(monkeypatch, {"url": other.as_uri(), "dir_info": {"editable": True}}, other)
    assert vi.version_info().source == "unknown"


def test_worktree_nested_in_the_editable_checkout(tmp_path, monkeypatch):
    """main/.claude/worktrees/x/src/odas_tpw lies under main/ but is other code."""
    main = (tmp_path / "main").resolve()
    pkg = main / ".claude" / "worktrees" / "x" / "src" / "odas_tpw"
    pkg.mkdir(parents=True)
    monkeypatch.setattr(vi, "_package_root", lambda: pkg)
    monkeypatch.setattr(vi, "_git_state", lambda _root, with_pyproject=False: (None, None))
    _install(monkeypatch, {"url": main.as_uri(), "dir_info": {"editable": True}}, main)
    assert "not the installed" in str(vi.version_info())


def test_percent_in_an_editable_path(tmp_path, monkeypatch):
    """as_uri() escapes a literal % once; decoding it twice names another path."""
    root = (tmp_path / "a%41b").resolve()
    pkg = root / "src" / "odas_tpw"
    pkg.mkdir(parents=True)
    monkeypatch.setattr(vi, "_package_root", lambda: pkg)
    monkeypatch.setattr(vi, "_git_state", lambda _root, with_pyproject=False: (None, None))
    _install(monkeypatch, {"url": root.as_uri(), "dir_info": {"editable": True}}, root)
    assert vi.version_info().source == "local-build"


def test_not_installed_at_all(site, monkeypatch):
    monkeypatch.setattr(vi, "_distribution", lambda: None)
    info = vi.version_info()
    assert info.release == "unknown"
    assert "running from" in str(info) and "commit unknown" in str(info)


# ---------------------------------------------------------------------------
# pyproject.toml
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('[project]\nname = "microstructure-tpw"\nversion = "0.4.0"\n', "0.4.0"),
        ('[project]\nname = "someone-else"\nversion = "0.4.0"\n', None),
        ('[project]\nname = "microstructure-tpw"\ndynamic = ["version"]\n', None),
        ("not = [toml", None),
    ],
)
def test_pyproject_release(tmp_path, text, expected):
    pkg = tmp_path / "src" / "odas_tpw"
    pkg.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(text)
    assert vi._pyproject_release(pkg) == expected


def test_pyproject_missing(tmp_path):
    assert vi._pyproject_release(tmp_path / "src" / "odas_tpw") is None
