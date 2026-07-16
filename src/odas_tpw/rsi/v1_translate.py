# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""
Translate ODAS header-version-1 (legacy, pre-2015) ``.p`` files to v6.

A v1 file differs from a v6 file **only in record 0**: v6 embeds an INI
configuration string there (``first_record_size = header_size +
config_size``); v1 stores the binary address matrix (row-major uint16, zero
padded) in a full-size record and the configuration lives in an external
setup file (old ``key: values`` dialect — see :mod:`odas_tpw.rsi.setup_v1`).
Data records (128-byte header + int16 samples in matrix scan order) are
identical between the versions.

The translation therefore: (1) parses the external setup file, (2)
synthesizes an equivalent v6 INI configuration, (3) writes a new record 0 =
converted header + INI, and (4) copies every data record verbatim. The
translated file is a normal v6 file — every existing tool (``PFile``, trim,
merge, patch-config, sensors, perturb) works on it unchanged.

Vendor precedent: ``odas/patch_setupstr.m`` (its ``convert_header`` sets
header words 11-14 to ``0x0600``/config-size/0/0 and copies every other word
unchanged — vendor-authoritative proof the 64-word header layout is shared).
Deliberate deviations from the vendor tool, all documented in
``docs/rsi-tpw/legacy_v1.md``:

- the vendor requires a hand-written v6 ``.cfg``; we **synthesize** the INI
  from the old-dialect setup file (with provenance keys in ``[root]``);
- the vendor overwrites record 1's header with a copy of record 0's
  converted header (losing its timestamp, record number, and bad-buffer
  flags); we copy **all** data records verbatim — lossless, and the
  bad-buffer (header word 16) warning parity survives translation;
- the vendor replaces the file in place; we write a new file.

See GitHub issue #141 for the format evidence and corpus.
"""

import hashlib
import struct
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from odas_tpw.rsi.setup_v1 import discover_setup_candidates, load_setup_file

__all__ = ["synthesize_ini", "translate_v1_bytes", "translate_v1_to_v6"]

# Keys written first in each [channel] section, in this order.
_CHANNEL_KEY_ORDER = ("id", "name", "type")


def synthesize_ini(cfg: dict[str, Any], provenance: dict[str, str]) -> str:
    """Render a parse_config-shaped dict as a v6 INI configuration string.

    Round-trip guarantee (tested):
    ``parse_config(synthesize_ini(cfg, ...))`` reproduces ``cfg``'s
    ``matrix``, ``channels``, ``instrument_info``, and ``root`` content (the
    provenance keys are added to ``[root]``).
    """
    lines: list[str] = []
    lines.append("; v6 configuration synthesized from a header-v1 setup file")
    lines.append("; by microstructure-tpw 'rsi-tpw v1to6' (GitHub issue #141).")
    lines.append("; Machine-readable provenance is in the [root] keys below.")
    lines.append("")
    lines.append("[root]")
    for k, v in cfg.get("root", {}).items():
        lines.append(f"{k} = {v}")
    for k, v in provenance.items():
        lines.append(f"{k} = {v}")
    lines.append("")

    inst = cfg.get("instrument_info", {})
    lines.append("[instrument_info]")
    if not (inst.get("model") or inst.get("sn")):
        lines.append("; v1 setup files record no instrument identity; add the")
        lines.append("; model:/sn: extension keys to the setup file to set one.")
    for k, v in inst.items():
        lines.append(f"{k} = {v}")
    lines.append("")

    matrix = cfg.get("matrix", [])
    lines.append("[matrix]")
    lines.append(f"num_rows = {len(matrix)}")
    for i, row in enumerate(matrix, start=1):
        lines.append(f"row{i:02d} = " + " ".join(str(int(a)) for a in row))
    lines.append("")

    for ch in cfg.get("channels", []):
        lines.append("[channel]")
        for k in _CHANNEL_KEY_ORDER:
            if k in ch:
                lines.append(f"{k} = {ch[k]}")
        for k, v in ch.items():
            if k not in _CHANNEL_KEY_ORDER:
                lines.append(f"{k} = {v}")
        lines.append("")

    return "\n".join(lines)


def _check_setup_against_header(
    cfg: dict[str, Any], header: dict[str, int], f_clock: float, source: str, setup_name: str
) -> None:
    """Vendor-parity sanity checks (patch_setupstr.m v1 branch).

    The acquisition-critical geometry in the setup file must match the data
    file's header: requested rate vs the header clock (tolerance 0.5 Hz,
    vendor value), no-fast, and no-slow. Errors are fatal — a geometry
    mismatch means this setup file did not drive this acquisition.
    """
    root = cfg.get("root", {})
    n_cols = header["fast_cols"] + header["slow_cols"]
    o_rate = f_clock / n_cols if n_cols else 0.0

    def _root_int(key: str) -> int | None:
        try:
            return int(float(root[key]))
        except (KeyError, ValueError):
            return None

    n_rate = _root_int("rate")
    if n_rate is None:
        warnings.warn(f"{source}: setup file {setup_name} has no 'rate:'; rate check skipped")
    elif abs(n_rate - o_rate) > 0.5:
        raise ValueError(
            f"{source}: rate mismatch: setup file {setup_name} says {n_rate} Hz "
            f"but the header clock gives {o_rate:.4f} Hz"
        )
    for key, hdr_key in (("no-fast", "fast_cols"), ("no-slow", "slow_cols")):
        n_val = _root_int(key)
        if n_val is not None and n_val != header[hdr_key]:
            raise ValueError(
                f"{source}: {key} mismatch: setup file {setup_name} says {n_val} "
                f"but the header says {header[hdr_key]}"
            )


def _cross_check_candidates(
    chosen: Path,
    chosen_cfg: dict[str, Any],
    others: list[Path],
    source: str,
) -> None:
    """Warn when other parseable setup candidates disagree with the chosen one.

    Compared: the address matrix and the pressure polynomial (the two fields
    known to be WRONG in the 2013 corpus's sibling ``.cfg`` files). A
    disagreement is a warning, not an error — the precedence order (and the
    record-0 matrix assertion) decides which file is authoritative.
    """

    def _pressure_coeffs(cfg: dict[str, Any]) -> list[str] | None:
        for ch in cfg.get("channels", []):
            ids = str(ch.get("id", "")).replace(",", " ").split()
            if ch.get("name") == "P" or ids == ["10"]:
                return [str(ch.get(f"coef{i}", "0")) for i in range(3)]
        return None

    chosen_p = _pressure_coeffs(chosen_cfg)
    for path in others:
        try:
            cfg, _ = load_setup_file(path)
        except (OSError, ValueError):
            continue
        disagreements = []
        if cfg.get("matrix") != chosen_cfg.get("matrix"):
            disagreements.append("address matrix")
        other_p = _pressure_coeffs(cfg)
        if chosen_p is not None and other_p is not None:
            try:
                if [float(c) for c in chosen_p] != [float(c) for c in other_p]:
                    disagreements.append("pressure polynomial")
            except ValueError:
                disagreements.append("pressure polynomial")
        if disagreements:
            warnings.warn(
                f"{source}: setup candidate {path.name} disagrees with "
                f"{chosen.name} on: {', '.join(disagreements)}; using "
                f"{chosen.name} (precedence order; see issue #141)"
            )


def _resolve_setup(
    src: Path,
    setup_file: str | Path | None,
    sens: dict[str, float] | None,
) -> tuple[Path, dict[str, Any], str]:
    """Choose + parse the setup file; returns (path, config dict, sens source)."""
    candidates: list[Path]
    if setup_file is not None:
        chosen = Path(setup_file)
        if not chosen.is_file():
            raise FileNotFoundError(f"setup file not found: {chosen}")
        cfg, _dialect = load_setup_file(chosen, sens_overrides=sens)
        candidates = [c for c in discover_setup_candidates(src) if c.resolve() != chosen.resolve()]
        _cross_check_candidates(chosen, cfg, candidates, src.name)
    else:
        candidates = discover_setup_candidates(src)
        if not candidates:
            raise ValueError(
                f"{src.name}: header-v1 file needs an external setup file, but "
                f"none was found next to it (searched {src.resolve().parent} and "
                "one level up for setup.txt, setup*.txt, setup*.cfg — "
                "case-insensitive); pass one explicitly with "
                "--setup-file / setup_file="
            )
        found: Path | None = None
        cfg = {}
        errors: list[str] = []
        for cand in candidates:
            try:
                cfg, _dialect = load_setup_file(cand, sens_overrides=sens)
                found = cand
                break
            except (OSError, ValueError) as exc:
                errors.append(f"{cand.name}: {exc}")
        if found is None:
            raise ValueError(
                f"{src.name}: no candidate setup file parsed cleanly: " + "; ".join(errors)
            )
        chosen = found
        _cross_check_candidates(chosen, cfg, [c for c in candidates if c != chosen], src.name)

    if sens:
        sens_source = "--sens override"
    elif any("sens" in ch for ch in cfg["channels"] if ch.get("type") == "shear"):
        sens_source = f"{chosen.name} <name>_sens keys"
    elif any(ch.get("type") == "shear" for ch in cfg["channels"]):
        sens_source = "none (inject with 'rsi-tpw patch-config' before processing)"
    else:
        sens_source = "n/a (no shear channels)"
    return chosen, cfg, sens_source


def translate_v1_bytes(
    src: str | Path,
    *,
    setup_file: str | Path | None = None,
    sens: dict[str, float] | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Translate a header-v1 ``.p`` file to v6, in memory.

    Parameters
    ----------
    src : path
        The v1 ``.p`` file.
    setup_file : path, optional
        Explicit setup file (either dialect). Default: auto-detect siblings
        (see :func:`odas_tpw.rsi.setup_v1.discover_setup_candidates`).
    sens : dict, optional
        Shear sensitivities by channel name (e.g. ``{"sh1": 0.0893}``);
        overrides ``<name>_sens:`` keys in the setup file.

    Returns
    -------
    (translated_bytes, meta)
        ``meta`` keys: ``setup_file``, ``setup_md5``, ``sens_source``,
        ``config_str`` (the synthesized INI), ``n_records`` (data records).
    """
    from odas_tpw.rsi.p_file import HEADER_BYTES, HEADER_WORDS, _detect_endian, _parse_header

    src = Path(src)
    blob = src.read_bytes()
    if len(blob) < HEADER_BYTES:
        raise ValueError(f"{src.name}: file too small for header")
    endian = _detect_endian(blob[:HEADER_BYTES], src)
    header = _parse_header(blob[:HEADER_BYTES], endian)

    version_raw = header["header_version"]
    if version_raw != 1:
        raise ValueError(
            f"{src.name}: not a header-v1 file (header word 11 = {version_raw} "
            f"= v{version_raw >> 8}.{version_raw & 0xFF}); v1to6 only translates "
            "legacy v1 files (issue #141)"
        )
    if endian == ">":
        warnings.warn(
            f"{src.name}: big-endian header-v1 file — no big-endian v1 corpus "
            "has been available for verification (issue #141); check the "
            "translated output carefully"
        )

    header_size = header["header_size"]
    record_size = header["record_size"]
    if header_size != HEADER_BYTES:
        raise ValueError(f"{src.name}: invalid v1 header_size={header_size} (expected 128)")
    if record_size <= header_size:
        raise ValueError(f"{src.name}: invalid record_size={record_size}")

    n_total = len(blob) // record_size
    if len(blob) % record_size != 0:
        warnings.warn(
            f"{src.name}: file size is not an integer number of records; "
            "trailing partial record ignored"
        )
    if n_total < 2:
        raise ValueError(f"{src.name}: contains no data records")

    rows = header["n_rows"]
    n_cols = header["fast_cols"] + header["slow_cols"]
    f_clock = header["clock_hz"] + header["clock_frac"] / 1000
    if rows < 1 or n_cols < 1 or f_clock <= 0:
        raise ValueError(
            f"{src.name}: invalid matrix geometry rows={rows} n_cols={n_cols} "
            f"f_clock={f_clock}"
        )
    if header_size + 2 * rows * n_cols > record_size:
        raise ValueError(
            f"{src.name}: record 0 too small for a {rows}x{n_cols} address matrix"
        )

    # Record-0 data block = the binary address matrix (row-major uint16) plus
    # zero padding. Non-zero padding words mean this is probably NOT a v1
    # matrix record — warn loudly rather than silently translating garbage.
    udtype = ">u2" if endian == ">" else "<u2"
    matrix = (
        np.frombuffer(blob, dtype=udtype, count=rows * n_cols, offset=header_size)
        .reshape(rows, n_cols)
        .tolist()
    )
    pad = np.frombuffer(
        blob,
        dtype=udtype,
        count=(record_size - header_size) // 2 - rows * n_cols,
        offset=header_size + 2 * rows * n_cols,
    )
    if np.any(pad != 0):
        warnings.warn(
            f"{src.name}: record 0 has {int(np.count_nonzero(pad))} non-zero "
            "words beyond the address matrix; the file may not be a clean v1 "
            "matrix record (issue #141)"
        )

    chosen, cfg, sens_source = _resolve_setup(src, setup_file, sens)

    # The record-0 binary matrix is ground truth; the setup file must agree
    # (vendor patch_setupstr.m errors on any element mismatch).
    if [[int(a) for a in row] for row in cfg["matrix"]] != matrix:
        raise ValueError(
            f"{src.name}: address matrix in setup file {chosen.name} does not "
            f"match the binary matrix in record 0; wrong setup file for this "
            "acquisition (issue #141)"
        )
    _check_setup_against_header(cfg, header, f_clock, src.name, chosen.name)

    setup_md5 = hashlib.md5(chosen.read_bytes()).hexdigest()
    from odas_tpw.rsi import __version__ as _pkg_version

    provenance = {
        "translated_from": "odas_v1",
        "v1_source_file": src.name,
        "setup_file_source": str(chosen),
        "setup_file_md5": setup_md5,
        "sens_source": sens_source,
        "translator": f"microstructure-tpw {_pkg_version} rsi-tpw v1to6",
        "translated_on": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    ini = synthesize_ini(cfg, provenance)
    ini_bytes = ini.encode("ascii", errors="replace")
    if len(ini_bytes) > 0xFFFF:
        raise ValueError(
            f"{src.name}: synthesized configuration is {len(ini_bytes)} bytes, "
            "exceeding the 65535-byte config_size header field"
        )

    # Vendor convert_header parity (patch_setupstr.m:392-411): copy every
    # header word unchanged except 11 (version -> 6.0), 12 (config size),
    # 13 (product id -> 0 = legacy, so a converted file is always
    # recognizable), 14 (build -> 0).
    words = list(struct.unpack(f"{endian}{HEADER_WORDS}H", blob[:HEADER_BYTES]))
    words[10] = 6 << 8  # header_version
    words[11] = len(ini_bytes)  # config_size
    words[12] = 0  # product_id
    words[13] = 0  # build_number
    new_header = struct.pack(f"{endian}{HEADER_WORDS}H", *words)

    out = b"".join([new_header, ini_bytes, blob[record_size : n_total * record_size]])
    meta = {
        "setup_file": str(chosen),
        "setup_md5": setup_md5,
        "sens_source": sens_source,
        "config_str": ini,
        "n_records": n_total - 1,
    }
    return out, meta


def translate_v1_to_v6(
    src: str | Path,
    dst: str | Path,
    *,
    setup_file: str | Path | None = None,
    sens: dict[str, float] | None = None,
    overwrite: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Translate a header-v1 ``.p`` file to a v6 ``.p`` file on disk.

    Refuses to overwrite an existing ``dst`` unless ``overwrite=True``, and
    refuses ``dst == src`` outright (the source is never modified).
    Returns ``(dst, meta)`` — see :func:`translate_v1_bytes` for ``meta``.
    """
    src = Path(src)
    dst = Path(dst)
    if dst.resolve() == src.resolve():
        raise ValueError(f"{src.name}: refusing to overwrite the v1 source in place")
    if dst.exists() and not overwrite:
        raise FileExistsError(f"{dst} exists; pass overwrite/--force to replace it")
    blob, meta = translate_v1_bytes(src, setup_file=setup_file, sens=sens)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(blob)
    return dst, meta
