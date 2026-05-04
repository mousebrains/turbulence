# May-2026, Claude and Pat Welch, pat@mousebrains.com
"""Logging fan-out for perturb runs.

Three layers, each writing to a separate ``FileHandler`` on the root
logger so a single ``logger.info(...)`` call lands in every relevant log:

* **Per-CLI-invocation** — ``<output_root>/logs/run_<timestamp>.log``,
  always written.  ``--stdout`` adds a stderr ``StreamHandler`` so the
  user can also watch progress on the terminal.
* **Per-worker** —
  ``<output_root>/logs/worker_<runtimestamp>_<pid>.log``, installed by
  ``init_worker_logging`` from a ``ProcessPoolExecutor`` initializer so
  each worker writes its own file (the spawn-context default on macOS
  doesn't inherit handlers).
* **Per-stage scope** — ``stage_log(stage_dir, basename)`` is a context
  manager that adds and removes a ``FileHandler`` for the duration of a
  block, so e.g. all log records emitted while processing ``a.p``'s
  diss stage land in ``diss_NN/a.log`` *as well as* the worker and run
  logs.

The handlers stack: a single record fans out to whichever handlers are
attached at emission time.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

_FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s [%(processName)s]: %(message)s"
_STREAM_FORMAT = "%(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

_FILE_FORMATTER = logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT)
_STREAM_FORMATTER = logging.Formatter(_STREAM_FORMAT)

# Marker attribute: handlers we install carry this so repeat calls to
# setup_root_logging can replace their own handlers without touching
# anything the user (e.g. a pytest plugin) attached.
_OWNED_ATTR = "_perturb_logging_owned"


def run_timestamp() -> str:
    """Return a UTC timestamp suitable for log file names (no separators that
    confuse shells)."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


_CURRENT_RUN_STAMP: str | None = None


def current_run_stamp() -> str:
    """Return the run stamp for this process, generating one on first call.

    Used by the CLI for the per-invocation log file name and by
    :mod:`pipeline` to seed worker log file names so all logs from a
    single ``perturb`` invocation share the same prefix.
    """
    global _CURRENT_RUN_STAMP
    if _CURRENT_RUN_STAMP is None:
        _CURRENT_RUN_STAMP = run_timestamp()
    return _CURRENT_RUN_STAMP


def reset_run_stamp() -> None:
    """Test helper: clear the cached run stamp."""
    global _CURRENT_RUN_STAMP
    _CURRENT_RUN_STAMP = None


def _mark_owned(handler: logging.Handler) -> logging.Handler:
    setattr(handler, _OWNED_ATTR, True)
    return handler


def _drop_owned_handlers(logger: logging.Logger) -> None:
    for h in list(logger.handlers):
        if getattr(h, _OWNED_ATTR, False):
            logger.removeHandler(h)
            with contextlib.suppress(Exception):
                h.close()


def setup_root_logging(
    log_file: Path,
    *,
    level: int = logging.INFO,
    stdout: bool = False,
) -> Path:
    """Configure the root logger for a CLI invocation.

    Always installs a ``FileHandler`` at *log_file* (parent dir is
    created).  When *stdout* is True, also installs a stderr
    ``StreamHandler``.

    Calling this twice in the same process is safe — handlers installed
    by previous calls are removed before new ones are added.  Handlers
    attached by other code (e.g. a test fixture's ``caplog``) are left
    alone.
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    _drop_owned_handlers(root)

    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(_FILE_FORMATTER)
    root.addHandler(_mark_owned(fh))

    if stdout:
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(level)
        sh.setFormatter(_STREAM_FORMATTER)
        root.addHandler(_mark_owned(sh))

    return log_file


def init_worker_logging(log_dir: Path, run_stamp: str) -> Path:
    """Install a per-worker ``FileHandler`` on the root logger.

    Designed for ``ProcessPoolExecutor(initializer=init_worker_logging,
    initargs=(log_dir, run_stamp))``.  Each spawn-context worker runs
    this exactly once; ``run_stamp`` is the parent run's timestamp so
    all workers from one invocation share a prefix and old runs aren't
    overwritten.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"worker_{run_stamp}_{os.getpid()}.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    _drop_owned_handlers(root)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(_FILE_FORMATTER)
    root.addHandler(_mark_owned(fh))

    return log_file


@contextmanager
def stage_log(stage_dir: Path | None, basename: str) -> Iterator[Path | None]:
    """Add a ``FileHandler`` at ``<stage_dir>/<basename>.log`` for the
    duration of the block; remove and close on exit.

    A no-op when *stage_dir* is None — lets call sites use the same
    ``with`` construct whether or not a stage is enabled.
    """
    if stage_dir is None:
        yield None
        return

    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    log_file = stage_dir / f"{basename}.log"

    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.setFormatter(_FILE_FORMATTER)

    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield log_file
    finally:
        root.removeHandler(handler)
        with contextlib.suppress(Exception):
            handler.close()
