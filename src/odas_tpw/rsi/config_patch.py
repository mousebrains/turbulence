# Jun-2026, Claude and Pat Welch, pat@mousebrains.com
"""Post-acquisition patching of the embedded configuration string in RSI ``.p`` files.

It is common for an instrument's onboard configuration (calibration coefficients,
vehicle, cruise metadata) to be wrong or incomplete at acquisition time and only
corrected afterwards. This module rewrites selected fields of the INI-style
configuration string embedded in record 0 of a ``.p`` file, writing the result to
a NEW file. The original is never modified, and the full original configuration is
embedded (commented out) in the patched file for provenance and recovery.

Design (verified against RSI ``patch_setupstr.m`` and the ``PFile`` reader):

* The data records are copied byte-for-byte. Only record 0's header word 11
  (``config_size``) is updated to the new configuration length. No reader (this
  package's ``PFile``, RSI ``read_odas.m`` / ``odas_p2mat.m``) consumes the stale
  ``config_size`` carried in the data-record headers, so they are left untouched.
* The header version is preserved (RSI's full rewrite downgrades v6.1 → v6.0).
* Acquisition-defining parameters (the ``[matrix]`` stanza and ``[root]``
  ``rate``/``recsize``/``no-fast``/``no-slow``) are unreachable by design — they
  cannot be addressed through the edit spec, so they can never be corrupted.

The edit spec is a small YAML file (see :func:`scaffold_yaml`). Every value must be
a quoted string so the text written to the file is exactly what the user typed —
this avoids YAML coercing ``0.1130`` → ``0.113`` or ``2.5e5`` → ``250000.0``.

The text editor mirrors :func:`odas_tpw.rsi.p_file.parse_config` tokenization
exactly (comment stripped at the first ``;``, section/assignment regexes on the
stripped line, keys compared case-insensitively, channel ``name`` values verbatim)
so that what the editor targets is exactly what ``PFile`` later reads.
"""

from __future__ import annotations

import difflib
import os
import re
import struct
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from odas_tpw.rsi.p_file import (
    _H,
    HEADER_BYTES,
    HEADER_WORDS,
    _detect_endian,
    parse_config,
)

try:  # package version for the provenance banner
    from importlib.metadata import version as _pkg_version

    _VERSION = _pkg_version("microstructure-tpw")
except Exception:  # pragma: no cover - fallback only when metadata is unavailable
    _VERSION = "unknown"

# Matches RSI patch_setupstr.m so RSI's -revert can still recover the original.
CONFIG_MARKER = "; ### Original Configuration String Below ###"
TOOL_NAME = "rsi-tpw patch-config"

# Top-level keys allowed in the edit spec. Acquisition stanzas (root, matrix) are
# deliberately absent: they cannot be addressed and therefore cannot be corrupted.
ALLOWED_TOP_LEVEL = {"note", "author", "instrument_info", "cruise_info", "channels"}
EDITABLE_SECTIONS = {"instrument_info", "cruise_info"}
# 'name' identifies a channel and (in any stanza) creates an RSI section alias
# (setupstr.m). It must never be set through this tool.
RESERVED_KEY = "name"

# Same tokenization as parse_config (p_file.py).
_SECTION_RE = re.compile(r"^\[(.+)\]$")
_ASSIGN_RE = re.compile(r"^(.+?)\s*=\s*(.*)$")
# Splits a raw assignment line so the value can be replaced while leading
# whitespace, the '=' alignment, trailing whitespace and any inline comment
# are all preserved.
_LINE_RE = re.compile(
    r"^(?P<lead>\s*)(?P<key>[^=\s][^=]*?)(?P<sep>\s*=\s*)(?P<val>.*?)(?P<trail>\s*)$"
)


# ---------------------------------------------------------------------------
# Edit spec
# ---------------------------------------------------------------------------


@dataclass
class EditSpec:
    """A parsed, validated set of configuration edits."""

    note: str
    author: str
    sections: dict[str, dict[str, str]] = field(default_factory=dict)
    channels: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def touches_channels(self) -> bool:
        return bool(self.channels)


@dataclass
class Change:
    """A single applied edit (for provenance comments and CLI reporting)."""

    where: str  # e.g. "instrument_info" or "channel sh1"
    key: str  # original-case key as written to the file
    old: str | None  # None when the key is being added
    new: str


def _check_value(value: object, ctx: str) -> str:
    """Validate one edit-spec value and return its verbatim text.

    Values must be quoted strings in the YAML. A bare ``0.1130`` / ``2.5e5`` /
    ``yes`` would be coerced by the YAML loader to a float/bool and lose its exact
    text, so non-string scalars are rejected with guidance to quote them.
    """
    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError(
            f"{ctx}: value must be quoted text to preserve it exactly "
            f'(e.g. {ctx.rsplit(".", 1)[-1]}: "0.1130"); got {type(value).__name__} {value!r}'
        )
    text = str(value)
    if not text.isascii():
        raise ValueError(f"{ctx}: value must be ASCII (the .p config is ASCII); got {text!r}")
    if ";" in text:
        raise ValueError(
            f"{ctx}: value must not contain ';' (the config parser truncates at it); got {text!r}"
        )
    return text


def _check_keymap(raw: object, ctx: str) -> dict[str, str]:
    """Validate one stanza/channel mapping from the spec and return key→text."""
    if not isinstance(raw, dict):
        raise ValueError(f"{ctx}: expected a mapping of key: value, got {type(raw).__name__}")
    out: dict[str, str] = {}
    for key, value in raw.items():
        key = str(key)
        if key.strip().lower() == RESERVED_KEY:
            raise ValueError(
                f"{ctx}.{key}: '{RESERVED_KEY}' identifies a channel/section and cannot be set"
            )
        out[key] = _check_value(value, f"{ctx}.{key}")
    return out


def load_edit_spec(path: str | Path) -> EditSpec:
    """Load and validate a YAML edit spec.

    Raises ``ValueError`` with an actionable message on any malformed input —
    unknown top-level keys, a missing ``note``, an unquoted/non-ASCII value, or a
    reserved ``name`` key.
    """
    from ruamel.yaml import YAML

    path = Path(path)
    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    with open(path, encoding="utf-8") as fh:
        data = yaml.load(fh)
    if data is None:
        raise ValueError(f"{path.name}: edit spec is empty")
    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: edit spec must be a YAML mapping")

    unknown = {str(k) for k in data} - ALLOWED_TOP_LEVEL
    if unknown:
        raise ValueError(
            f"{path.name}: unknown top-level key(s) {sorted(unknown)}; "
            f"valid keys: {sorted(ALLOWED_TOP_LEVEL)}"
        )

    note = data.get("note")
    if not isinstance(note, str) or not note.strip():
        raise ValueError(f"{path.name}: a non-empty 'note' describing the change is required")
    if not note.isascii():
        raise ValueError(f"{path.name}: 'note' must be ASCII")
    note = note.strip()

    author = data.get("author")
    if author is not None and (not isinstance(author, str) or not author.isascii()):
        raise ValueError(f"{path.name}: 'author' must be ASCII text")
    author = str(author).strip() if author else (os.environ.get("USER") or "unknown")

    sections: dict[str, dict[str, str]] = {}
    for sec in EDITABLE_SECTIONS:
        if sec in data:
            sections[sec] = _check_keymap(data[sec], sec)

    channels: dict[str, dict[str, str]] = {}
    if "channels" in data:
        chans = data["channels"]
        if not isinstance(chans, dict):
            raise ValueError(f"{path.name}: 'channels' must be a mapping of channel-name: {{...}}")
        for name, kv in chans.items():
            channels[str(name)] = _check_keymap(kv, f"channels.{name}")

    if not sections and not channels:
        raise ValueError(f"{path.name}: edit spec contains no edits")

    return EditSpec(note=note, author=author, sections=sections, channels=channels)


# ---------------------------------------------------------------------------
# Config-text parsing and editing (pure string -> string)
# ---------------------------------------------------------------------------


def _code_part(line: str) -> str:
    """Portion of a raw line before the first ';' (matches parse_config)."""
    idx = line.find(";")
    return line if idx < 0 else line[:idx]


def _split_eol(line: str) -> tuple[str, str]:
    for term in ("\r\n", "\r", "\n"):
        if line.endswith(term):
            return line[: -len(term)], term
    return line, ""


def _detect_eol(text: str) -> str:
    if "\r\n" in text:
        return "\r\n"
    if "\r" in text:
        return "\r"
    return "\n"


def _parse_blocks(raw_lines: list[str]) -> list[dict[str, Any]]:
    """Walk lines exactly as parse_config does, recording per-stanza key locations.

    Each block is ``{"section", "name", "keys", "header_idx", "last_key_idx"}``
    where ``keys`` maps a lower-cased key to ``(line_index, original_key, value)``.
    Commented lines (including ones that mention ``[matrix]`` in prose) strip to
    empty and are skipped, so they never create a phantom section or key.
    """
    blocks: list[dict[str, Any]] = []
    cur: dict[str, Any] = {
        "section": "root",
        "name": None,
        "keys": {},
        "header_idx": -1,
        "last_key_idx": -1,
    }
    blocks.append(cur)
    for i, line in enumerate(raw_lines):
        stripped = _code_part(line).strip()
        if not stripped:
            continue
        ms = _SECTION_RE.match(stripped)
        if ms:
            cur = {
                "section": ms.group(1).strip().lower(),
                "name": None,
                "keys": {},
                "header_idx": i,
                "last_key_idx": i,
            }
            blocks.append(cur)
            continue
        ma = _ASSIGN_RE.match(stripped)
        if ma:
            orig = ma.group(1).strip()
            val = ma.group(2).strip()
            kl = orig.lower()
            cur["keys"][kl] = (i, orig, val)
            cur["last_key_idx"] = i
            if cur["section"] == "channel" and kl == RESERVED_KEY:
                cur["name"] = val
    return blocks


def _rewrite_value(raw_line: str, new_value: str) -> str:
    """Replace the value in an assignment line, preserving everything else."""
    body, eol = _split_eol(raw_line)
    idx = body.find(";")
    code, comment = (body, "") if idx < 0 else (body[:idx], body[idx:])
    m = _LINE_RE.match(code)
    if m is None:  # pragma: no cover - only assignment lines are ever passed here
        return raw_line
    return f"{m['lead']}{m['key']}{m['sep']}{new_value}{m['trail']}{comment}{eol}"


def _comment_lines(text: str, eol: str) -> str:
    """Prefix every line of *text* with '; ' (for the embedded original)."""
    out = []
    for line in text.splitlines(keepends=True):
        body, term = _split_eol(line)
        out.append(f"; {body}{term or eol}")
    return "".join(out)


def edit_config_text(
    config_str: str,
    spec: EditSpec,
    *,
    add_keys: bool = False,
    when: str = "",
    source_label: str = "",
) -> tuple[str, list[Change]]:
    """Apply *spec* to a configuration string.

    Returns ``(new_config_text, changes)``. When ``changes`` is empty the input is
    returned unchanged (the caller then skips writing the file). Raises
    ``ValueError`` for unknown stanzas/keys (unless ``add_keys``), unknown or
    ambiguous channel names, or a missing target section.
    """
    eol = _detect_eol(config_str)

    # Separate any previously-embedded original block; we only edit above it and
    # re-attach it verbatim (idempotent re-patch, mirroring patch_setupstr.m).
    marker_idx = config_str.find(CONFIG_MARKER)
    if marker_idx >= 0:
        active = config_str[:marker_idx]
        original_block: str | None = config_str[marker_idx:]
    else:
        active = config_str
        original_block = None

    raw_lines = active.splitlines(keepends=True)
    blocks = _parse_blocks(raw_lines)

    sect_block: dict[str, dict[str, Any]] = {}
    sect_keys: dict[str, dict[str, Any]] = {}
    chan_by_name: dict[str, list[dict[str, Any]]] = {}
    for b in blocks:
        if b["section"] == "channel":
            if b["name"] is not None:
                chan_by_name.setdefault(b["name"], []).append(b)
        else:
            sect_block[b["section"]] = b
            sect_keys.setdefault(b["section"], {}).update(b["keys"])

    changes: list[Change] = []
    replace: dict[int, tuple[str, str]] = {}
    inserts: dict[int, list[tuple[str, str]]] = {}

    def provenance(ch: Change) -> str:
        head = f"; [PATCH {when} {spec.author}]".rstrip()
        # Include the stanza/channel context and quote both values so the
        # original and new value are unambiguous (e.g. which channel's 'SN',
        # and that an empty old value really was empty).
        if ch.old is None:
            return f'{head} {ch.where} {ch.key}: (added) = "{ch.new}"  ({spec.note}){eol}'
        return f'{head} {ch.where} {ch.key}: "{ch.old}" -> "{ch.new}"  ({spec.note}){eol}'

    def schedule(
        keys: dict, block: dict, where: str, key: str, new_value: str, *, is_inst_sn: bool
    ) -> None:
        kl = key.lower()
        if kl in keys:
            idx, orig, cur = keys[kl]
            if cur == new_value:
                return  # already at the target value — not a change
            ch = Change(where, orig, cur, new_value)
            changes.append(ch)
            replace[idx] = (provenance(ch), _rewrite_value(raw_lines[idx], new_value))
            if is_inst_sn:
                warnings.warn(
                    f"{where}: changing the instrument serial number 'sn' "
                    f"({cur!r} -> {new_value!r}); this is rarely correct",
                    stacklevel=2,
                )
        else:
            if not add_keys:
                avail = sorted({orig for _i, orig, _v in keys.values()})
                raise ValueError(
                    f"{where}: unknown key '{key}'; existing keys: {avail} "
                    f"(use --add-keys to add a new one)"
                )
            ch = Change(where, key, None, new_value)
            changes.append(ch)
            ins = block["last_key_idx"] + 1
            inserts.setdefault(ins, []).append((provenance(ch), f"{key} = {new_value}{eol}"))

    for section, kv in spec.sections.items():
        if section not in sect_block:
            raise ValueError(f"[{section}] section is not present in this file")
        for key, val in kv.items():
            schedule(
                sect_keys[section],
                sect_block[section],
                section,
                key,
                val,
                is_inst_sn=(section == "instrument_info" and key.lower() == "sn"),
            )

    for name, kv in spec.channels.items():
        blist = chan_by_name.get(name)
        if not blist:
            raise ValueError(
                f"channel '{name}' not found; available channels: {sorted(chan_by_name)}"
            )
        if len(blist) > 1:
            raise ValueError(
                f"channel name '{name}' appears {len(blist)} times; cannot target it unambiguously"
            )
        for key, val in kv.items():
            schedule(blist[0]["keys"], blist[0], f"channel {name}", key, val, is_inst_sn=False)

    if not changes:
        return config_str, []

    # Reassemble the active region with provenance comments interleaved.
    out: list[str] = []
    for i, line in enumerate(raw_lines):
        for prov, newl in inserts.get(i, []):
            out.append(prov)
            out.append(newl)
        if i in replace:
            prov, newl = replace[i]
            out.append(prov)
            out.append(newl)
        else:
            out.append(line)
    for prov, newl in inserts.get(len(raw_lines), []):
        out.append(prov)
        out.append(newl)
    edited_active = "".join(out)

    banner = _build_banner(when, spec.author, spec.note, source_label, eol)
    new_config = banner + edited_active
    if not new_config.endswith(("\n", "\r")):
        new_config += eol
    if original_block is not None:
        new_config += original_block
    else:
        new_config += eol + CONFIG_MARKER + eol + _comment_lines(active, eol)
    return new_config, changes


def _build_banner(when: str, author: str, note: str, source_label: str, eol: str) -> str:
    bar = "; " + "=" * 70 + eol
    lines = [
        bar,
        f"; PATCHED {when} by {author} -- {TOOL_NAME} {_VERSION}".rstrip() + eol,
    ]
    if source_label:
        lines.append(f"; source: {source_label}{eol}")
    lines.append(f"; note: {note}{eol}")
    lines.append(bar)
    return "".join(lines)


# ---------------------------------------------------------------------------
# Binary I/O
# ---------------------------------------------------------------------------


def read_config_text(path: str | Path) -> str:
    """Read just the embedded configuration string (latin-1, lossless)."""
    path = Path(path)
    with open(path, "rb") as f:
        head = f.read(HEADER_BYTES)
        if len(head) < HEADER_BYTES:
            raise ValueError(f"{path.name}: file too small for a header")
        endian = _detect_endian(head, path)
        words = struct.unpack(f"{endian}{HEADER_WORDS}H", head)
        header_size = words[_H["header_size"]]
        config_size = words[_H["config_size"]]
        f.seek(header_size)
        return f.read(config_size).decode("latin-1")


def write_patched_pfile(src: str | Path, dst: str | Path, new_config_text: str) -> Path:
    """Write *dst* = *src* with its embedded config replaced by *new_config_text*.

    Copies the original header (patching only ``config_size``) and every data
    record byte-for-byte. Refuses legacy (header major version < 6) files and any
    configuration that would overflow the uint16 ``config_size`` field.
    """
    src = Path(src)
    dst = Path(dst)
    new_bytes = new_config_text.encode("latin-1")
    with open(src, "rb") as f:
        head = f.read(HEADER_BYTES)
        if len(head) < HEADER_BYTES:
            raise ValueError(f"{src.name}: file too small for a header")
        endian = _detect_endian(head, src)
        words = struct.unpack(f"{endian}{HEADER_WORDS}H", head)
        header_size = words[_H["header_size"]]
        config_size = words[_H["config_size"]]
        major = words[_H["header_version"]] >> 8
        if major < 6:
            raise ValueError(
                f"{src.name}: header version {major}.x (< 6) is not supported by patch-config"
            )
        if len(new_bytes) > 0xFFFF:
            raise ValueError(
                f"{src.name}: patched config is {len(new_bytes)} bytes, exceeds the "
                f"65535-byte limit of the config_size header field"
            )
        f.seek(0)
        header_region = bytearray(f.read(header_size))
        struct.pack_into(f"{endian}H", header_region, _H["config_size"] * 2, len(new_bytes))

        dst.parent.mkdir(parents=True, exist_ok=True)
        f.seek(header_size + config_size)  # start of the first data record
        with open(dst, "wb") as out:
            out.write(header_region)
            out.write(new_bytes)
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    return dst


# ---------------------------------------------------------------------------
# Template scaffolding
# ---------------------------------------------------------------------------


def scaffold_yaml(pfile_path: str | Path) -> str:
    """Build a commented edit-spec template pre-filled from *pfile_path*."""
    p = Path(pfile_path)
    cfg = parse_config(read_config_text(p))
    ii = cfg.get("instrument_info", {})
    ci = cfg.get("cruise_info", {})
    chans = cfg.get("channels", [])

    lines = [
        f"# rsi-tpw patch-config edit spec  (generated from {p.name})",
        "#",
        "# HOW TO USE:",
        '#   1. Change the values you want.  QUOTE every value, e.g.  sens: "0.0812"',
        "#   2. Delete every line you are NOT changing (only real edits are applied).",
        f"#   3. rsi-tpw patch-config {p.name} --edits THIS.yaml --out patched/",
        "#",
        "# Your original file is never modified.  If nothing actually changes, no",
        "# file is written.",
        "",
        'note: "describe what you changed and why"',
        '# author: "your name"',
        "",
        "instrument_info:",
        f'  vehicle: "{ii.get("vehicle", "")}"   # the field that actually affects processing',
    ]
    if ii.get("model") is not None:
        lines.append(f'  # model: "{ii.get("model", "")}"')
    lines.append("")

    if ci:
        lines.append("# cruise_info:")
        for key, val in ci.items():
            lines.append(f'#   {key}: "{val}"')
        lines.append("")

    lines.append("# channels:   # per-channel calibration (usually one instrument at a time)")
    for ch in chans:
        name = ch.get("name")
        if not name:
            continue
        shown = [k for k in ("sens", "sn", "cal_date", "coef0", "coef1") if k in ch]
        if not shown:
            continue
        lines.append(f"#   {name}:")
        for key in shown:
            lines.append(f'#     {key}: "{ch[key]}"')
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

# Editing any channel key is per-instrument calibration; applying it to many files
# at once is usually a mistake, so it requires an explicit opt-in.


def patch_files(
    srcs: list[Path],
    out_dir: str | Path,
    spec: EditSpec,
    *,
    dry_run: bool = False,
    add_keys: bool = False,
    batch_cal: bool = False,
    when: str = "",
) -> list[tuple[Path, Path | None, list[Change]]]:
    """Apply *spec* to each source file, writing patched copies into *out_dir*.

    Returns ``(src, dst_or_None, changes)`` per file. ``dst`` is ``None`` when the
    file was a no-op (no values changed) or for a dry run. Progress is printed.
    """
    out_dir = Path(out_dir)
    if spec.touches_channels and len(srcs) > 1 and not batch_cal:
        raise ValueError(
            "refusing to apply per-channel calibration edits to multiple files "
            "(calibrations are per-instrument); pass --batch-cal if this is intended"
        )

    results: list[tuple[Path, Path | None, list[Change]]] = []
    for src in srcs:
        src = Path(src)
        dst = out_dir / src.name
        if dst.resolve() == src.resolve():
            raise ValueError(
                f"--out would overwrite the original {src}; choose a different directory"
            )
        cfg = read_config_text(src)
        new_text, changes = edit_config_text(
            cfg, spec, add_keys=add_keys, when=when, source_label=str(src.resolve())
        )
        if not changes:
            print(f"{src.name}: no changes; not written")
            results.append((src, None, []))
            continue
        if dry_run:
            print(f"--- {src.name} (dry run, {len(changes)} change(s)) ---")
            diff = difflib.unified_diff(
                cfg.splitlines(), new_text.splitlines(), src.name, dst.name, lineterm=""
            )
            for line in diff:
                print(line)
            results.append((src, None, changes))
            continue
        if dst.exists():
            raise FileExistsError(f"{dst} already exists; remove it or choose another --out")
        write_patched_pfile(src, dst, new_text)
        print(f"{src.name}: {len(changes)} change(s) -> {dst}")
        for ch in changes:
            arrow = (
                f'(added) = "{ch.new}"' if ch.old is None else f'"{ch.old}" -> "{ch.new}"'
            )
            print(f"    {ch.where} {ch.key}: {arrow}")
        results.append((src, dst, changes))
    return results


def _print_error_and_exit(message: str) -> None:  # pragma: no cover - CLI glue
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)
