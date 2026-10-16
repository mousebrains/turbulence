# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""Which code is running: the release, and the git commit when it is knowable.

Every console script's ``--version`` prints :func:`version_string`, so a bug
report can name the code that produced it. The release number alone cannot:
``pyproject.toml`` carries a static version that stays at the last release for
every commit after it, so ``0.4.0`` on main is any of dozens of commits.

Where the commit comes from, in order:

1. **git**, run in the directory of the *imported* ``odas_tpw`` package, not
   the working directory, and only when that repository tracks the package.
   This covers an editable install and a worktree run with
   ``PYTHONPATH=<worktree>/src``, and reports the code actually imported.
   "uncommitted changes" means a modified, staged or untracked file under the
   package directory (an ignored one too, if Python could import it), or a
   changed ``pyproject.toml``; edits to docs or tests do not count.
2. **PEP 610** ``direct_url.json``, which pip and uv write for a
   ``pip install git+https://...`` install.
3. ``_build.json``, which the release workflow writes into the wheel it
   publishes to PyPI. A wheel without it was built somewhere else, from any
   commit, and is not called a release.

Anything less certain says "commit unknown" rather than guess. The release is
read from the source tree's ``pyproject.toml`` when there is one, because an
editable install's metadata is frozen at install time (one read 0.3.1 against
a ``pyproject.toml`` of 0.4.0).
"""

from __future__ import annotations

import argparse
import importlib.machinery
import json
import os
import re
import subprocess
import tomllib
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

DIST_NAME = "microstructure-tpw"
_GIT_TIMEOUT_S = 5.0
_ABBREV = 12
# Written by .github/workflows/release.yml just before `python -m build`.
RELEASE_STAMP = "_build.json"
# `git rev-parse --local-env-vars` (git 2.54): variables that point git at a
# particular repository, index or object store. A git hook sets some of them
# (GIT_INDEX_FILE always, GIT_DIR in a worktree); inherited, they would make
# git describe that repository instead of the one holding the package.
_REPO_LOCAL_ENV = frozenset(
    {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CONFIG",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_COUNT",
        "GIT_OBJECT_DIRECTORY",
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_IMPLICIT_WORK_TREE",
        "GIT_GRAFT_FILE",
        "GIT_INDEX_FILE",
        "GIT_NO_REPLACE_OBJECTS",
        "GIT_REPLACE_REF_BASE",
        "GIT_PREFIX",
        "GIT_SHALLOW_FILE",
        "GIT_COMMON_DIR",
    }
)


@dataclass(frozen=True)
class VersionInfo:
    """The running code's identity, as far as it can be established.

    Attributes
    ----------
    release : str
        Release number of the running source ("unknown" if none can be found).
    commit : str or None
        Full commit hash, when known.
    describe : str or None
        ``git describe`` of that commit (``v0.4.0-57-g5f6572b4a1c2``); None
        when no release tag is reachable or the commit came from PEP 610.
    dirty : bool or None
        Uncommitted changes under the package directory or in
        ``pyproject.toml``; None if git could not say. Only meaningful when
        ``source == "git"``.
    source : str
        "git", "pip-vcs" (PEP 610 commit), "release" (a wheel stamped by the
        release workflow), "local-build" (PEP 610 directory or archive
        install, commit unknown) or "unknown".
    detail : str or None
        Where a local build came from, or why the commit is unknown.
    metadata_release : str or None
        The installed metadata's version when it disagrees with ``release``.
    """

    release: str
    commit: str | None = None
    describe: str | None = None
    dirty: bool | None = False
    source: str = "unknown"
    detail: str | None = None
    metadata_release: str | None = None

    def __str__(self) -> str:
        notes: list[str] = []
        if self.source == "release" and self.commit:
            notes.append(f"release, git {self.commit[:_ABBREV]}")
        elif self.source in ("git", "pip-vcs") and self.commit:
            notes.append(f"git {self.describe or self.commit[:_ABBREV]}")
            if self.source == "git" and self.dirty is None:
                notes.append("uncommitted changes unknown")
            elif self.source == "git" and self.dirty:
                notes.append("uncommitted changes")
        elif self.source == "local-build":
            notes.append(f"built from {self.detail}, commit unknown")
        elif self.source == "unknown":
            notes.append(f"{self.detail}, commit unknown" if self.detail else "commit unknown")
        text = self.release
        if notes:
            text += f" ({', '.join(notes)})"
        if self.metadata_release:
            text += f"; installed metadata says {self.metadata_release}"
        return text


def _package_root() -> Path:
    import odas_tpw

    return Path(odas_tpw.__file__).resolve().parent


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    """``git <args>`` run in *cwd*; None if git cannot be run at all.

    ``core.fsmonitor=false`` keeps ``status`` from starting a file-system
    monitor daemon that outlives the command (or running a configured hook),
    and ``GIT_OPTIONAL_LOCKS=0`` keeps it from taking the index lock under a
    concurrent commit.
    """
    env = {k: v for k, v in os.environ.items() if k not in _REPO_LOCAL_ENV}
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["LC_ALL"] = env["LANGUAGE"] = "C"  # English stderr, matched below
    try:
        return subprocess.run(
            ["git", "-c", "core.fsmonitor=false", *args],
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _git_out(cwd: Path, *args: str) -> str | None:
    """Stripped stdout of a successful ``git <args>``, else None."""
    proc = _git(cwd, *args)
    if proc is None or proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _git_state(
    pkg_root: Path, with_pyproject: bool = False
) -> tuple[tuple[str, str | None, bool | None] | None, str | None]:
    """``((commit, describe, dirty), None)`` when a git repository tracks
    *pkg_root*; ``(None, reason)`` when git refused it; else ``(None, None)``.

    The ``ls-files`` guard refuses a repository that merely *contains* the
    package (a non-editable install into a ``.venv`` inside some project's
    checkout), whose commit says nothing about the code running.
    *with_pyproject* adds ``../../pyproject.toml`` to the uncommitted-changes
    check, for when the release is read from it.
    """
    probe = _git(pkg_root, "ls-files", "--", "__init__.py")
    if probe is None:
        return None, None
    if probe.returncode != 0:
        if "dubious ownership" in probe.stderr:
            # Never bypass with safe.directory=*: that would run another
            # user's repository configuration (CVE-2022-24765).
            return None, (
                "git refused the checkout: it is owned by another user "
                "(see git's safe.directory)"
            )
        return None, None
    if not probe.stdout.strip():
        return None, None
    commit = _git_out(pkg_root, "rev-parse", "--verify", "HEAD")
    if not commit:
        return None, None
    describe = _git_out(
        pkg_root, "describe", "--tags", "--long", f"--abbrev={_ABBREV}", "--match", "v[0-9]*"
    )
    paths = [".", "../../pyproject.toml"] if with_pyproject else ["."]
    status = _git_out(pkg_root, "status", "--porcelain", "--untracked-files=normal", "--", *paths)
    hidden = _ignored_code(pkg_root)
    dirty = None if status is None or hidden is None else bool(status) or hidden
    return (commit, describe or None, dirty), None


def _ignored_code(pkg_root: Path) -> bool | None:
    """Does an ignored file under *pkg_root* hold code Python could import?

    ``git status`` cannot see an ignored file, and the root ``.gitignore``'s
    unanchored rules (``MR/``, ``results/``, ``bench/``; case-insensitive on
    macOS) would hide a new ``src/odas_tpw/mr/`` package from it.
    """
    proc = _git(
        pkg_root, "ls-files", "-z", "--others", "--ignored", "--exclude-standard", "--", "."
    )
    if proc is None or proc.returncode != 0:
        return None
    suffixes = tuple(importlib.machinery.all_suffixes())
    return any(
        name.endswith(suffixes) and "__pycache__" not in PurePosixPath(name).parts
        for name in proc.stdout.split("\0")
    )


def _pyproject_release(pkg_root: Path) -> str | None:
    """``project.version`` of the source tree holding *pkg_root* (src layout)."""
    path = pkg_root.parent.parent / "pyproject.toml"
    try:
        project = tomllib.loads(path.read_text(encoding="utf-8")).get("project", {})
    except (OSError, ValueError):
        return None
    if project.get("name") != DIST_NAME or not isinstance(project.get("version"), str):
        return None
    return str(project["version"])


def _distribution() -> metadata.Distribution | None:
    try:
        return metadata.distribution(DIST_NAME)
    except metadata.PackageNotFoundError:
        return None


def _direct_url(dist: metadata.Distribution) -> dict[str, Any] | None:
    """The PEP 610 record, ``{}`` if unreadable, None if there is none."""
    text = dist.read_text("direct_url.json")
    if not text:
        return None
    try:
        record = json.loads(text)
    except ValueError:
        return {}
    return record if isinstance(record, dict) else {}


def _file_url_path(url: str) -> Path | None:
    from urllib.request import url2pathname  # ~9 ms; only a fallback needs it

    parsed = urlparse(url)
    if parsed.scheme != "file":
        return None
    path = parsed.path  # url2pathname does the unquoting
    if parsed.netloc not in ("", "localhost"):
        path = f"//{parsed.netloc}{path}"  # a UNC share
    return Path(url2pathname(path))


def _is_installed_copy(
    dist: metadata.Distribution, record: dict[str, Any] | None, pkg_root: Path
) -> bool:
    """Is the imported package the one this distribution installed?

    False when ``PYTHONPATH`` (or a stray checkout on ``sys.path``) shadows the
    installed copy, in which case the installed metadata describes other code.
    """
    dir_info = (record or {}).get("dir_info")
    if isinstance(dir_info, dict) and dir_info.get("editable"):
        src = _file_url_path(str(record.get("url", ""))) if record else None
        if src is None:
            return False
        try:
            # Exactly the project's own package: a worktree nested inside the
            # editable checkout is other code.
            return pkg_root == (src / "src" / "odas_tpw").resolve()
        except OSError:
            return False
    try:
        installed = Path(str(dist.locate_file("odas_tpw/__init__.py"))).resolve().parent
    except OSError:
        return False
    return installed == pkg_root


def _release_stamp(pkg_root: Path, release: str) -> str | None:
    """The commit in the release workflow's stamp, if it names this release."""
    try:
        stamp = json.loads((pkg_root / RELEASE_STAMP).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(stamp, dict) or stamp.get("tag") != f"v{release}":
        return None
    commit = stamp.get("commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        return None
    return commit


def version_info() -> VersionInfo:
    """Identify the running code (runs git; call it only when needed)."""
    pkg_root = _package_root()
    dist = _distribution()
    record = _direct_url(dist) if dist is not None else None
    # Installed metadata describes the imported code only if this is its copy.
    installed = dist is not None and _is_installed_copy(dist, record, pkg_root)
    meta_release = dist.version if dist is not None and installed else None
    src_release = _pyproject_release(pkg_root)
    release = src_release or meta_release or "unknown"
    stale = meta_release if meta_release not in (None, release) else None

    state, refusal = _git_state(pkg_root, with_pyproject=src_release is not None)
    if state is not None:
        commit, describe, dirty = state
        return VersionInfo(
            release, commit, describe, dirty, source="git", metadata_release=stale
        )
    if refusal is not None:
        return VersionInfo(release, source="unknown", detail=refusal, metadata_release=stale)
    if dist is None:
        return VersionInfo(release, source="unknown", detail=f"running from {pkg_root}")
    if not installed:
        return VersionInfo(
            release,
            source="unknown",
            detail=f"running from {pkg_root}, not the installed {DIST_NAME}",
        )

    vcs = (record or {}).get("vcs_info")
    if isinstance(vcs, dict) and isinstance(vcs.get("commit_id"), str):
        return VersionInfo(
            release, vcs["commit_id"], source="pip-vcs", metadata_release=stale
        )
    # The release workflow's stamp marks a PyPI wheel, installed by name or
    # from its downloaded file; it is never trusted beside a pyproject.toml.
    stamped = None if src_release is not None else _release_stamp(pkg_root, release)
    if stamped is not None and (record is None or "archive_info" in record):
        return VersionInfo(release, stamped, source="release")
    if record is not None:
        return VersionInfo(
            release,
            source="local-build",
            detail=str(record.get("url") or "an unrecorded location"),
            metadata_release=stale,
        )
    if src_release is not None:
        # A source tree is never an index install: what was found is a stray
        # src/*.egg-info from an old `pip install -e`, with git unavailable.
        return VersionInfo(
            release,
            source="unknown",
            detail=f"running from the source tree {pkg_root.parent.parent}",
            metadata_release=stale,
        )
    # A wheel built from any commit and installed by name looks just like a
    # PyPI release, except that it carries no stamp.
    return VersionInfo(release, source="unknown", detail="not a release build")


def version_string() -> str:
    """One line for ``--version`` and bug reports, e.g.
    ``0.4.0 (git v0.4.0-57-g5f6572b4a1c2, uncommitted changes)``."""
    return str(version_info())


class _VersionAction(argparse.Action):
    """``--version`` that is computed only when asked for and printed verbatim.

    argparse's own ``action="version"`` formats the string eagerly and reflows
    it through the help formatter, which can wrap a long line at the terminal
    width.
    """

    def __init__(
        self,
        option_strings: list[str],
        dest: str = argparse.SUPPRESS,
        default: str = argparse.SUPPRESS,
        help: str = "print the release and git commit, then exit",
    ) -> None:
        super().__init__(option_strings, dest=dest, default=default, nargs=0, help=help)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        print(f"{parser.prog} {version_string()}", flush=True)
        parser.exit()


def add_version_argument(parser: argparse.ArgumentParser) -> None:
    """Give *parser* a top-level ``--version``."""
    parser.add_argument("--version", action=_VersionAction)
