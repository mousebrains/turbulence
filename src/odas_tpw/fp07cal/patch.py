# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""Writing fitted coefficients back into the ``.p`` files.

This is the sink for the whole exercise: ``fp07-cal patch`` drives
``rsi/config_patch.py`` to write a new ``.p`` per input carrying the in-situ
``t_0`` / ``beta_*``, the original configuration retained commented-out, and a
provenance banner.  Perturb then needs no knowledge of any of this --- set
``fp07.calibrate: false`` and every reader, including ODAS MATLAB, sees the
corrected coefficients.

Neutralising a higher-order term
--------------------------------
``convert_therm`` evaluates ``1/T_K = 1/t_0 + (1/beta_1)L + (1/beta_2)L^2 + ...``,
so each config value is the RECIPROCAL of its coefficient.  Emitting an order-1
fit into a config that already carries ``beta_2`` would leave a live quadratic
term fighting the new linear one, and ``config_patch`` can add keys but not
delete them.

Setting ``beta_2 = 0`` does not delete the term --- it is the exact opposite.
``1/0`` raises ``ZeroDivisionError`` and the reader crashes outright (verified
against ``convert_therm``).  What deletes a term is ``1/beta_k -> 0``, i.e.
``beta_k -> infinity``.  ``beta_2 = 1e30`` produces output **bit-identical** to
omitting the key, which is the neutralisation used here.

Refusals
--------
A coefficient set is only valid alongside the bridge constants it was fitted
against and for the instrument it came from, so a mismatch in any bridge
parameter, the instrument serial number, or the live ``beta`` key is an error
rather than a warning.  A file that already carries an **fp07-cal** banner is
refused too: stacking two in-situ calibrations makes the provenance ambiguous.
A file patched by ``rsi-tpw patch-config`` for some other reason (a bridge
parameter fix, say) is fine --- ``config_patch`` keeps a single frozen
original-config block across passes, and fixing the bridge constants first is
exactly the workflow ``logr.py`` recommends.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from odas_tpw.rsi.config_patch import (
    EditSpec,
    patch_files,
    read_config_text,
)
from odas_tpw.rsi.p_file import parse_config

SCHEMA = "fp07-cal/1"
"""The ``schema`` value ``fp07-cal fit`` writes and ``patch`` requires."""

AUTHOR = "fp07-cal"
"""Banner author written by :func:`patch_deployment`; what :func:`already_patched` looks for."""

# config_patch writes "; PATCHED <when> by <author> -- <tool> <version>".
_BANNER_RE = re.compile(r"^;\s*PATCHED\b.*\bby\s+" + re.escape(AUTHOR) + r"\b", re.MULTILINE)

# 1/1e30 is 1e-30; against L^2 ~ 1e-2 and 1/T_K ~ 3.4e-3 the term contributes
# ~1e-32 -- below float64 resolution on the sum, hence bit-identical to absent.
NEUTRAL = "1e30"

_BETA_KEYS = ("beta_1", "beta_2", "beta_3")


@dataclass
class PatchPlan:
    """What would be written, and why it might not be."""

    edits: dict[str, dict[str, str]]
    warnings: list[str]
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors and bool(self.edits)


def _fmt(value: float) -> str:
    """Enough digits to round-trip a float64 exactly."""
    return repr(float(value))


def _validity_warnings(ch: str, entry: dict) -> list[str]:
    """Warn where the coefficients are being used outside what they were fitted on.

    Both are warnings, not errors, and deliberately so: leaving a file on the
    nominal coefficients would re-create exactly the discontinuity this tool
    exists to remove (plan section 3.3).  Consistency beats a false show of
    caution --- as long as it is visible.
    """
    out: list[str] = []
    v = entry.get("validity") or {}
    below = float(v.get("extrapolated_below_K") or 0.0)
    above = float(v.get("extrapolated_above_K") or 0.0)
    if below > 0.05 or above > 0.05:
        lo, hi = v.get("T_fitted", [float("nan")] * 2)
        slo, shi = v.get("T_seen", [float("nan")] * 2)
        out.append(
            f"{ch}: the deployment spans {slo:.2f}..{shi:.2f} degC but the fit only "
            f"covered {lo:.2f}..{hi:.2f} — extrapolating {below:.2f} degC below and "
            f"{above:.2f} degC above on {v.get('n_profiles_outside', '?')} of "
            f"{v.get('n_profiles_total', '?')} profiles. A polynomial is not "
            f"constrained outside its fitted range; check the order."
        )
    if not entry.get("probe_sn_trusted", True):
        out.append(
            f"{ch}: probe serial {entry.get('probe_sn')!r} looks like a placeholder "
            f"(shared across channels), so it is not a usable identity — this "
            f"record is keyed on instrument SN and channel only."
        )
    return out


def build_edits(record: dict, config: dict, *, channels: list[str] | None = None) -> PatchPlan:
    """Turn a coefficient record into per-channel config edits.

    *config* is a parsed ``.p`` configuration (``parse_config`` output) for the
    file about to be patched --- it is what the bridge and key checks run
    against.
    """
    edits: dict[str, dict[str, str]] = {}
    warnings: list[str] = []
    errors: list[str] = []

    by_name = {str(c.get("name", "")).strip(): dict(c) for c in config.get("channels", [])}
    inst = str(config.get("instrument_info", {}).get("sn", "?") or "?").strip()
    rec_inst = str(record.get("instrument_sn", "?") or "?").strip()
    if rec_inst not in ("?", "") and inst not in ("?", "") and rec_inst != inst:
        errors.append(
            f"instrument SN mismatch: record was fitted on {rec_inst!r}, "
            f"file is {inst!r}"
        )

    for ch, entry in record.get("channels", {}).items():
        if channels and ch not in channels:
            continue
        if "config_equivalent" not in entry:
            warnings.append(f"{ch}: no coefficients in the record (skipped)")
            continue
        if ch not in by_name:
            warnings.append(f"{ch}: not present in this file (skipped)")
            continue
        cfg_ch = by_name[ch]

        if not entry.get("lag_trustworthy", True):
            errors.append(
                f"{ch}: the record's lag did not pass the sharpness gate; "
                f"refusing to patch coefficients derived from it"
            )
            continue

        rec_bridge = entry.get("bridge") or {}
        for key, want in rec_bridge.items():
            have = cfg_ch.get(key)
            if have is None:
                errors.append(f"{ch}: file config has no bridge parameter {key!r}")
                continue
            try:
                same = abs(float(have) - float(want)) <= 1e-9 * max(1.0, abs(float(want)))
            except (TypeError, ValueError):
                same = False
            if not same:
                errors.append(
                    f"{ch}: bridge parameter {key} differs (record {want}, file "
                    f"{have}); the fitted coefficients would not reproduce"
                )

        # convert_therm checks the legacy `beta` FIRST, so patching beta_1 on a
        # file carrying `beta` is a silent no-op.
        live = "beta" if cfg_ch.get("beta") not in (None, "") else "beta_1"
        rec_live = entry.get("beta_key", "beta_1")
        if live != rec_live:
            warnings.append(
                f"{ch}: file uses {live!r} for the linear term but the record "
                f"recorded {rec_live!r}; writing to {live!r}, which is what the "
                f"reader will use"
            )

        warnings.extend(_validity_warnings(ch, entry))

        ce = entry["config_equivalent"]
        ch_edits: dict[str, str] = {"t_0": _fmt(ce["t_0"])}
        if "beta_1" in ce:
            ch_edits[live] = _fmt(ce["beta_1"])
        for k in ("beta_2", "beta_3"):
            if k in ce:
                ch_edits[k] = _fmt(ce[k])
            elif cfg_ch.get(k) not in (None, ""):
                # Present in the file, absent from the fit: neutralise it, or it
                # keeps contributing against the new lower-order coefficients.
                ch_edits[k] = NEUTRAL
                warnings.append(
                    f"{ch}: fit is order {len(ce) - 1} but the file carries {k}="
                    f"{cfg_ch[k]}; neutralising it to {NEUTRAL} (equivalent to "
                    f"removing the term — setting it to 0 would divide by zero)"
                )
        if live == "beta" and cfg_ch.get("beta_1") not in (None, ""):
            warnings.append(
                f"{ch}: file also carries beta_1={cfg_ch['beta_1']}, which the "
                f"reader ignores while `beta` is present; left untouched"
            )
        edits[ch] = ch_edits

    return PatchPlan(edits=edits, warnings=warnings, errors=errors)


def _time_range_warnings(record: dict, srcs: list[Path]) -> list[str]:
    """Note when the fit sampled fewer files than are being patched.

    The coefficients are applied to EVERY file by design --- that is the whole
    point of pooling --- but the operator should see how much of the deployment
    actually contributed to them.  The ``.p`` header date is not reliably
    parseable across dialects, so this reports the counts rather than
    re-deriving each file's acquisition time.
    """
    t0, t1 = record.get("time_start"), record.get("time_end")
    n_fit = record.get("n_fit_files")
    if t0 is None or t1 is None or not n_fit or len(srcs) <= n_fit:
        return []
    return [
        f"patching {len(srcs)} files from a fit that sampled {n_fit} of them "
        f"({_iso(t0)}..{_iso(t1)}); coefficients are applied to every file, "
        f"including any acquired outside that window"
    ]


def _iso(t) -> str:
    from datetime import UTC, datetime

    try:
        return datetime.fromtimestamp(float(t), tz=UTC).strftime("%Y-%m-%d")
    except Exception:
        return "?"


def already_patched(path: str | Path) -> bool:
    """True when the file already carries an **fp07-cal** provenance banner.

    Only this tool's banner counts.  A generic ``rsi-tpw patch-config`` pass
    (e.g. fixing a bridge parameter before calibrating) leaves the file
    eligible.
    """
    try:
        return _BANNER_RE.search(read_config_text(path)) is not None
    except Exception:
        return False


_REQUIRED_TOP = ("instrument_sn", "channels")


def validate_record(record: object) -> dict:
    """Check that *record* is a complete ``fp07-cal/1`` coefficient record.

    Every safety gate in :func:`build_edits` is a comparison against a field of
    the record, so a record missing that field makes the gate vacuous.  A
    foreign JSON or a hand-trimmed one must therefore be refused outright
    rather than partially applied.  Raises ``ValueError``.
    """
    if not isinstance(record, dict):
        raise ValueError("coefficient record is not a JSON object")
    schema = record.get("schema")
    if schema != SCHEMA:
        raise ValueError(
            f"coefficient record schema is {schema!r}, expected {SCHEMA!r}; "
            f"this is not an fp07-cal coefficients.json"
        )
    missing = [k for k in _REQUIRED_TOP if k not in record]
    if missing:
        raise ValueError(f"coefficient record is missing {missing}")
    chans = record["channels"]
    if not isinstance(chans, dict) or not chans:
        raise ValueError("coefficient record has no channels")
    problems: list[str] = []
    for ch, entry in chans.items():
        if not isinstance(entry, dict):
            problems.append(f"{ch}: entry is not an object")
            continue
        if "config_equivalent" not in entry:
            # A channel that failed (e.g. zero pairs) is recorded with an
            # "error" key and no coefficients; build_edits skips it with a
            # warning.  Only an entry that claims coefficients is checked.
            if "error" in entry:
                continue
            problems.append(f"{ch}: no config_equivalent")
            continue
        for key in ("bridge", "coefficients"):
            if not entry.get(key):
                problems.append(f"{ch}: missing {key!r}")
        ce = entry["config_equivalent"]
        if not isinstance(ce, dict) or "t_0" not in ce:
            problems.append(f"{ch}: config_equivalent has no t_0")
    if problems:
        raise ValueError("incomplete coefficient record: " + "; ".join(problems))
    return record


# config_size is a uint16 in the .p header; config_patch refuses anything
# larger at write time.  The patch roughly doubles the config (the original is
# retained commented-out), so the projection is checked BEFORE any file is
# written rather than discovered halfway through the batch.
_CONFIG_SIZE_LIMIT = 0xFFFF


def _precheck_destinations(srcs: list[Path], out_dir: str | Path, *, dry_run: bool) -> None:
    """Refuse before writing anything if any output exists or any config would overflow."""
    out_dir = Path(out_dir)
    if not dry_run:
        existing = [s.name for s in srcs if (out_dir / s.name).exists()]
        if existing:
            raise ValueError(
                f"{len(existing)} destination file(s) already exist in {out_dir} "
                f"(e.g. {existing[0]}); remove them or choose another output dir. "
                f"Nothing was written."
            )
    too_big = []
    for s in srcs:
        try:
            n = len(read_config_text(s).encode("latin-1"))
        except Exception:
            continue  # reported by build_edits as an unreadable config
        # Conservative projection: the original is retained commented-out
        # (+"; " per line) plus a banner and a change line per edit.
        projected = 2 * n + 2048
        if projected > _CONFIG_SIZE_LIMIT:
            too_big.append(f"{s.name} ({n} bytes -> ~{projected})")
    if too_big:
        raise ValueError(
            f"patched config would exceed the {_CONFIG_SIZE_LIMIT}-byte config_size "
            f"limit for {len(too_big)} file(s): {', '.join(too_big[:3])}. Nothing was written."
        )


def patch_deployment(
    record_path: str | Path,
    srcs: list[Path],
    out_dir: str | Path,
    *,
    channels: list[str] | None = None,
    note: str = "",
    author: str = AUTHOR,
    dry_run: bool = False,
) -> tuple[PatchPlan, list]:
    """Apply a coefficient record to every ``.p`` file of one deployment.

    Returns ``(plan, results)`` where *results* is ``patch_files``' per-file
    ``(src, dst_or_None, changes)``.
    """
    try:
        record = json.loads(Path(record_path).read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{record_path}: not valid JSON ({exc})") from exc
    validate_record(record)
    srcs = [Path(s) for s in srcs]
    if not srcs:
        raise ValueError("no source .p files")

    done = [s for s in srcs if already_patched(s)]
    if done:
        raise ValueError(
            f"{len(done)} of {len(srcs)} inputs already carry an fp07-cal "
            f"calibration (e.g. {done[0].name}). Patch the files that have not "
            f"been calibrated yet, so the provenance stays unambiguous."
        )
    _precheck_destinations(srcs, out_dir, dry_run=dry_run)

    # EVERY source is validated, not just the first. The instrument-SN and
    # bridge-parameter checks are the whole safety story of this step, and
    # checking one file while patching a list is not a check -- a
    # mixed-instrument directory, or a probe swap that changed a bridge
    # constant partway through a deployment, would sail past it and be written
    # with coefficients that cannot reproduce.
    plan = build_edits(record, parse_config(read_config_text(srcs[0])), channels=channels)
    for src in srcs[1:]:
        try:
            other = build_edits(record, parse_config(read_config_text(src)),
                                channels=channels)
        except Exception as exc:  # unreadable config -- refuse, do not skip
            # All-or-nothing is deliberate: a deployment half-patched is worse
            # than one not patched. But a file that cannot be READ at all
            # (0 bytes, a failed flash write) carries no data to lose, so say
            # how to drop it rather than leaving the user stuck.
            plan.errors.append(
                f"{Path(src).name}: cannot read config: {exc} "
                f"— if this file is junk (check its size and date), list it "
                f"under files.exclude and re-run"
            )
            continue
        plan.errors.extend(f"{Path(src).name}: {e}" for e in other.errors)
        # Warnings are per-file too, but they repeat across a deployment; keep
        # one of each so the operator sees the kind without the volume.
        for w in other.warnings:
            tagged = f"{Path(src).name}: {w}"
            if w not in plan.warnings and tagged not in plan.warnings:
                plan.warnings.append(tagged)
        if other.edits != plan.edits:
            plan.errors.append(
                f"{Path(src).name}: resolves to different edits than "
                f"{Path(srcs[0]).name}; these files do not share one calibration"
            )
    plan.warnings.extend(_time_range_warnings(record, srcs))
    if not plan.ok:
        return plan, []

    spec = EditSpec(
        note=note or (
            f"FP07 in-situ calibration vs {record.get('reference', 'CTD')}; "
            f"{record.get('n_fit_files', '?')} files"
        ),
        author=author,
        channels=plan.edits,
    )
    results = patch_files(
        srcs, out_dir, spec,
        dry_run=dry_run,
        add_keys=True,   # an order-2 fit onto a config with no beta_2
        batch_cal=True,  # a deployment calibration IS a per-instrument edit
    )
    return plan, results
