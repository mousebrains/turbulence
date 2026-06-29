# Jun-2026, Claude and Pat Welch, pat@mousebrains.com
"""Resolve perturb output directories from a config (read-only).

Given a perturb config (the same YAML the pipeline ran), locate the
``{stage}_NN`` output directory whose stored ``.params_sha256_*`` signature
matches that config — so plotting can point at a config instead of guessing
the newest versioned directory (which silently mixes products from different
configs when several share an ``output_root``).

Matching compares the canonical signature JSON section-by-section but **ignores
``files.output_root``**: ``--output`` rewrites it at run time without
normalization, so a run's stored signature can hold an output path the config
YAML never contained, and recomputing the hash would never match — yet the
output location has no bearing on the data. Everything else in ``files`` is
kept, including ``p_file_root``/``p_file_pattern`` (they identify the input
*dataset*, so two runs of different inputs into one ``output_root`` stay
distinguishable) and ``trim``/``merge`` (they change the data). A run done with
``--p-file-root`` therefore won't match by signature and falls back to the
single-dir / drift path below.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

from ruamel.yaml.error import YAMLError

from odas_tpw.config_base import iter_stage_dirs
from odas_tpw.perturb import config as _cfg

# Everything load_config can raise for a bad --config: missing/unreadable
# (OSError, incl. FileNotFoundError), unknown section/key (ValueError), or a
# YAML syntax error (ruamel YAMLError — NOT an OSError/ValueError subclass).
# A bad config is ALWAYS fatal; these are converted to a clean SystemExit
# wherever a config is loaded, so the user never sees a raw traceback.
_CONFIG_LOAD_ERRORS = (OSError, ValueError, YAMLError)

# files key rewritten by --output at run time and irrelevant to the data;
# excluded from the signature comparison. p_file_root/p_file_pattern are kept
# (they identify the input dataset).
_VOLATILE_FILES_KEYS = ("output_root",)


class StageConflict(RuntimeError):
    """The stage directory is ambiguous w.r.t. the config (multiple matches, or
    no match with several candidates / under ``strict``).

    Distinct from ``FileNotFoundError`` (the stage simply does not exist): a
    conflict must surface even for an *optional* stage rather than silently
    degrading, since it means the user's data is present but doesn't agree with
    the config.
    """


def _strip_volatile(canon: dict) -> dict:
    """Drop the run-environment output path from a canonical signature dict."""
    files = canon.get("files")
    if isinstance(files, dict):
        for key in _VOLATILE_FILES_KEYS:
            files.pop(key, None)
    return canon


def _want_signature(config: dict, stage: str) -> dict:
    """The canonical signature *this* config would produce for *stage*."""
    section, params, upstream = _cfg.stage_signature(stage, config)
    return _strip_volatile(json.loads(_cfg.canonicalize(section, params, upstream)))


def _read_signature(directory: Path) -> dict | None:
    """The stored canonical signature of *directory*, or None if unreadable."""
    sigs = sorted(directory.glob(".params_sha256_*"))
    if not sigs:
        return None
    try:
        return _strip_volatile(json.loads(sigs[0].read_text()))
    except (OSError, ValueError):
        return None


def _describe(directory: Path) -> str:
    """One-line note on a candidate dir's config, for error messages."""
    sig = _read_signature(directory)
    if sig is None:
        return f"{directory.name} (no signature)"
    bits = [
        f"{sec}={json.dumps(sig[sec], sort_keys=True)}"
        for sec in ("binning", "chi", "epsilon", "top_trim")
        if sec in sig
    ]
    return f"{directory.name}" + (": " + "; ".join(bits) if bits else "")


def stage_dir(
    config: dict,
    stage: str,
    output_root: str | Path | None = None,
    *,
    strict: bool = False,
    latest: bool = False,
) -> Path:
    """Return the ``{stage}_NN`` directory under *output_root* matching *config*.

    *output_root* defaults to ``config["files"]["output_root"]``. Resolution:

    - exact signature match → that directory;
    - ``latest=True`` → the newest ``{stage}_NN`` regardless of config;
    - no match, exactly one ``{stage}_NN`` exists → that one, with a warning
      (the common "tweak config, re-plot" case — it cannot be a *different*
      config's data, there is only one);
    - no match, two or more exist → ``StageConflict`` listing each directory's
      config (refuse to guess which is the caller's);
    - multiple *exact* matches (a concurrent same-config run that raced for a
      sequence number, or a hand-copied signature) → the newest, with a warning
      (identical signature ⇒ identical config and inputs ⇒ equivalent data);
    - ``strict=True`` → ``StageConflict`` on any non-exact match.

    Raises ``ValueError`` for an unknown *stage*, ``FileNotFoundError`` when no
    ``{stage}_NN`` directory exists at all (stage disabled or never run), and
    ``StageConflict`` when the directory is ambiguous w.r.t. the config. Never
    creates a directory.
    """
    if stage not in _cfg.STAGES:
        raise ValueError(f"unknown stage {stage!r}; known: {sorted(_cfg.STAGES)}")
    root = Path(
        output_root or (config.get("files") or {}).get("output_root") or "."
    ).expanduser()
    candidates = iter_stage_dirs(root, stage)
    if not candidates:
        raise FileNotFoundError(
            f"no {stage}_NN directory under {root} — the {stage} stage was never "
            f"run here (or is disabled in the config)."
        )
    if latest:
        return candidates[-1][1]

    want = _want_signature(config, stage)
    matches = [d for _, d in candidates if _read_signature(d) == want]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Multiple dirs carry the *same* canonical signature. The expected
        # cause is two identical-config `perturb run`s racing to claim a
        # sequence number under one output_root (one creates {stage}_00, the
        # other loses the mkdir, advances, and writes the same signature into
        # {stage}_01); output_root is stripped before comparison and nothing
        # else differs, so those two were produced from the same config and
        # inputs and their data is equivalent — picking the newest is safe and
        # avoids needlessly locking the user out. A signature file is copied
        # *text*, not a checksum of the dir's data, so a hand-copied signature
        # into a dir with different data would also land here and is NOT
        # guaranteed equivalent — hence we warn (listing the dirs) rather than
        # resolve silently.
        listing = ", ".join(d.name for d in matches)
        warnings.warn(
            f"{stage}: {len(matches)} directories under {root} share the given "
            f"config's signature ({listing}) — same config and inputs, so using "
            f"the newest ({matches[-1].name}).",
            stacklevel=2,
        )
        return matches[-1]

    if not strict and len(candidates) == 1:
        only = candidates[0][1]
        warnings.warn(
            f"{stage}: config does not match {only.name}'s signature (config "
            f"drift) but it is the only {stage} directory under {root} — using "
            f"it. Pass strict=True / --strict to require an exact match.",
            stacklevel=2,
        )
        return only

    listing = "\n  ".join(_describe(d) for _, d in candidates)
    reason = "config drift" if strict else "and none matches the given config"
    raise StageConflict(
        f"{len(candidates)} {stage}_NN directories under {root} {reason} — "
        f"refusing to guess which is yours. Candidates:\n  {listing}\n"
        f"Edit the config to match, re-run the pipeline, or force the newest "
        f"with --latest."
    )


# ---------------------------------------------------------------------------
# CLI glue for the existing plot subcommands
# ---------------------------------------------------------------------------

def add_resolve_args(p: argparse.ArgumentParser) -> None:
    """Register ``--config``/``--latest``/``--strict`` on a plot subparser.

    ``--root`` is registered separately (and is now optional); at least one of
    ``--config``/``--root`` must be given (``--root`` may accompany ``--config``
    to override the search location).
    """
    p.add_argument(
        "--config", metavar="PERTURB.YAML",
        help="perturb config; output directories are resolved from its hash "
        "signatures instead of guessing the newest {stage}_NN.",
    )
    p.add_argument(
        "--latest", action="store_true",
        help="with --config, use the newest {stage}_NN even if no signature matches.",
    )
    p.add_argument(
        "--strict", action="store_true",
        help="with --config, error (don't fall back) when no signature matches.",
    )


def require_root(args: argparse.Namespace) -> str:
    """The output root for a plot run: ``--root`` if given, else the
    ``--config``'s ``files.output_root``.

    Call once at the top of a subcommand's ``run`` to backfill ``args.root``
    (used for titles, default output paths, and per-profile attr lookups) when
    only ``--config`` was given. Raises ``SystemExit`` if neither is provided.
    """
    root = getattr(args, "root", None)
    if root:
        return str(root)
    cfg_path = getattr(args, "config", None)
    if cfg_path:
        # require_root runs FIRST in every subcommand's run(), so a bad config
        # must be cleanly fatal here too (not just in resolve_for_args) — else
        # the user gets a raw traceback before resolve_for_args is ever reached.
        try:
            out = _cfg.load_config(cfg_path).get("files", {}).get("output_root")
        except _CONFIG_LOAD_ERRORS as exc:
            raise SystemExit(str(exc)) from exc
        if out:
            return str(out)
    raise SystemExit("one of --config or --root is required")


def resolve_for_args(
    args: argparse.Namespace,
    stage: str,
    *,
    optional: bool = False,
    conflict_ok: bool = False,
) -> str | None:
    """Resolve a stage directory from parsed plot args.

    With ``--config``, resolve by signature; otherwise fall back to the legacy
    newest-``{stage}_NN`` discovery. Returns ``None`` when an *optional* stage
    is absent (so callers can degrade gracefully, e.g. eps-chi without a
    per-profile ``chi`` dir, profiles' diagnostics overlay) — matching the
    legacy ``latest_stage_dir`` behavior. A required stage that cannot be
    resolved raises a clean ``SystemExit``.

    *optional* makes a *missing* stage (no ``{stage}_NN``) degrade to None; by
    default a config *conflict* (ambiguous/drift) still raises, even for an
    optional stage, so the user's data is never silently dropped. *conflict_ok*
    additionally degrades a conflict to None — for **cosmetic** lookups (e.g.
    eps-chi's per-profile ``diss`` dir, used only for the title's processing
    params) that must never fail the whole figure over which dir to caption it.
    """
    cfg_path = getattr(args, "config", None)
    root = getattr(args, "root", None)
    if cfg_path:
        # Loading the config is separate from finding the stage dir: a
        # missing/unreadable/malformed --config (e.g. a typo, or invalid YAML)
        # is ALWAYS fatal and must never be mistaken for "this optional stage
        # was not run". Only a FileNotFoundError from stage_dir (no {stage}_NN
        # dir) degrades to None.
        try:
            config = _cfg.load_config(cfg_path)
        except _CONFIG_LOAD_ERRORS as exc:
            raise SystemExit(str(exc)) from exc
        try:
            return str(
                stage_dir(
                    config, stage, output_root=root,
                    strict=getattr(args, "strict", False),
                    latest=getattr(args, "latest", False),
                )
            )
        except FileNotFoundError as exc:
            if optional:  # stage simply absent -> degrade gracefully
                return None
            raise SystemExit(str(exc)) from exc
        except StageConflict as exc:
            if conflict_ok:  # cosmetic lookup: ambiguity is non-fatal
                return None
            # For a data stage a conflict must surface even when optional,
            # rather than silently dropping the user's data.
            raise SystemExit(str(exc)) from exc
        except (ValueError, OSError) as exc:
            raise SystemExit(str(exc)) from exc
    if not root:
        raise SystemExit("one of --config or --root is required")
    from odas_tpw.perturb.plot import layout

    return layout.latest_stage_dir(root, stage)
