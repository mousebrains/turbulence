# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""
Read Rockland Scientific .p binary data files.

Implements the format described in RSI Technical Note 051 (Rockland Data File
Anatomy) and mirrors the conversion logic in the ODAS MATLAB Library
(read_odas.m, convert_odas.m).
"""

import re
import struct
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from odas_tpw.rsi.channels import CONVERTERS
from odas_tpw.rsi.deconvolve import deconvolve

# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------

HEADER_WORDS = 64
HEADER_BYTES = 128

# 0-indexed word positions (TN-051 Table 1 uses 1-indexed)
_H = {
    "file_number": 0,
    "record_number": 1,
    "year": 3,
    "month": 4,
    "day": 5,
    "hour": 6,
    "minute": 7,
    "second": 8,
    "millisecond": 9,
    "header_version": 10,
    "config_size": 11,
    "product_id": 12,
    "build_number": 13,
    "timezone_min": 14,
    "buffer_status": 15,
    "restarted": 16,
    "header_size": 17,
    "record_size": 18,
    "n_records_written": 19,
    "clock_hz": 20,
    "clock_frac": 21,
    "fast_cols": 28,
    "slow_cols": 29,
    "n_rows": 30,
    "profile": 62,
    "endian": 63,
}


def _detect_endian(raw_header: bytes, source: str | Path | None = None) -> str:
    """Return '>' (big) or '<' (little) endian prefix for struct.

    *source* is an optional file path/name folded into the warning text so an
    ambiguous-endian file can be identified during batch processing. Because
    the filename is part of the message, Python's default warning de-dup keys
    on it — so each distinct offending file is reported once (not just the
    very first across the whole run).
    """
    where = f" ({source})" if source is not None else ""
    be = struct.unpack_from(">H", raw_header, 63 * 2)[0]
    le = struct.unpack_from("<H", raw_header, 63 * 2)[0]
    if be == 2:
        return ">"
    if le == 1:
        return "<"
    if be == 0 or le == 0:
        warnings.warn(f"Endian flag is 0; assuming little-endian{where}")
        return "<"
    be_hs = struct.unpack_from(">H", raw_header, 17 * 2)[0]
    le_hs = struct.unpack_from("<H", raw_header, 17 * 2)[0]
    if be_hs == 128:
        return ">"
    if le_hs == 128:
        return "<"
    warnings.warn(f"Cannot determine endian; defaulting to big-endian{where}")
    return ">"


def _parse_header(raw: bytes, endian: str) -> dict:
    fmt = f"{endian}{HEADER_WORDS}H"
    words = struct.unpack(fmt, raw)
    return {name: words[idx] for name, idx in _H.items()}


def extract_pfile_segment(
    source: str | Path,
    dest: str | Path,
    *,
    start_record: int = 0,
    n_records: int = 60,
    overwrite: bool = False,
) -> Path:
    """Copy a contiguous data-record range from a Rockland ``.p`` file.

    This is a byte-level debugging utility. It copies the first record
    containing the binary header and configuration string unchanged, then
    appends ``n_records`` complete data records starting at the 0-based
    ``start_record`` index. The output is a parseable P-file segment with the
    original calibration metadata preserved, but header fields are copied
    verbatim. Absolute time is correct only when ``start_record`` is 0; for
    later starts, data is shifted relative to the copied timestamp. The header
    record count is not authoritative because local readers derive the count
    from file size.
    """
    source = Path(source)
    dest = Path(dest)

    if start_record < 0:
        raise ValueError("start_record must be >= 0")
    if n_records < 1:
        raise ValueError("n_records must be >= 1")
    if not source.exists():
        raise FileNotFoundError(source)
    if dest.exists() and not overwrite:
        raise FileExistsError(f"{dest} exists; pass overwrite=True to replace it")

    with open(source, "rb") as src:
        raw_hdr = src.read(HEADER_BYTES)
        if len(raw_hdr) < HEADER_BYTES:
            raise ValueError(f"{source.name}: file too small for header")

        endian = _detect_endian(raw_hdr, source)
        header = _parse_header(raw_hdr, endian)
        header_size = int(header["header_size"])
        config_size = int(header["config_size"])
        record_size = int(header["record_size"])

        if header_size < HEADER_BYTES:
            raise ValueError(f"{source.name}: invalid header_size={header_size}")
        if config_size < 0:
            raise ValueError(f"{source.name}: invalid config_size={config_size}")
        if record_size <= 0:
            raise ValueError(f"{source.name}: invalid record_size={record_size}")
        if record_size < header_size:
            raise ValueError(
                f"{source.name}: invalid record_size={record_size}; "
                f"expected >= header_size={header_size}"
            )

        first_record_size = header_size + config_size
        src.seek(0, 2)
        file_size = src.tell()
        data_bytes = file_size - first_record_size
        if data_bytes < 0:
            raise ValueError(f"{source.name}: file smaller than first record")

        total_records = data_bytes // record_size
        if start_record >= total_records:
            raise ValueError(
                f"start_record={start_record} out of range "
                f"(file has {total_records} complete data records)"
            )

        available = total_records - start_record
        if n_records > available:
            record_word = "record" if available == 1 else "records"
            verb = "is" if available == 1 else "are"
            raise ValueError(
                f"requested {n_records} records starting at {start_record}, "
                f"but only {available} complete {record_word} {verb} available"
            )

        src.seek(0)
        first_record = src.read(first_record_size)
        src.seek(first_record_size + start_record * record_size)
        data_records = src.read(n_records * record_size)

    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as out:
        out.write(first_record)
        out.write(data_records)

    return dest


# ---------------------------------------------------------------------------
# Configuration string parsing
# ---------------------------------------------------------------------------


def parse_config(config_str: str) -> dict[str, Any]:
    """Parse the INI-style configuration string embedded in the P file.

    Parameters
    ----------
    config_str : str
        The raw INI-style configuration text from record 0 of the P file.

    Returns
    -------
    dict with keys:
      'matrix': list of lists (the address matrix rows)
      'channels': list of dicts, one per [channel] section
      'instrument_info': dict
      'cruise_info': dict
      'root': dict
      plus one dict per other section found in the file (setupstr.m keeps
      every section). Section names are normalized: lower-cased with internal
      whitespace folded to '_', so the CASPER-era ``[cruise info]`` lands in
      'cruise_info'. (Deviation from setupstr.m, which keeps names verbatim;
      the fold gives old and modern dialects one canonical key.)

    Channel dicts of type ``accel`` with ``coef0 = 0`` and ``coef1 = 1`` are
    rewritten to type ``piezo`` (setupstr.m:478-495: configs made before type
    'piezo' existed), so every consumer sees the corrected type.
    """
    result: dict[str, Any] = {
        "matrix": [],
        "channels": [],
        "instrument_info": {},
        "cruise_info": {},
        "root": {},
    }

    lines = config_str.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cleaned = []
    for line in lines:
        idx = line.find(";")
        if idx >= 0:
            line = line[:idx]
        cleaned.append(line.strip())

    current_section = "root"
    current_channel: dict[str, str] | None = None

    for line in cleaned:
        if not line:
            continue

        m = re.match(r"^\[(.+)\]$", line)
        if m:
            current_section = "_".join(m.group(1).strip().lower().split())
            if current_section == "channel":
                current_channel = {}
                result["channels"].append(current_channel)
            else:
                current_channel = None
                # Keep unknown sections (setupstr.m keeps every section)
                # instead of dropping their keys. 'matrix' and 'channels'
                # already hold lists — a section by either name must not
                # clobber them, so only absent names get a fresh dict (the
                # assignment branch below drops keys for non-dict targets).
                if current_section not in result:
                    result[current_section] = {}
            continue

        m = re.match(r"^(.+?)\s*=\s*(.*)$", line)
        if m:
            key = m.group(1).strip().lower()
            val = m.group(2).strip()
            if current_section == "channel" and current_channel is not None:
                current_channel[key] = val
            elif current_section == "matrix":
                if key.startswith("row"):
                    # A corrupt/garbled config (e.g. partially overwritten
                    # record 0) otherwise raises a bare "invalid literal for
                    # int()" with no section/token context.
                    try:
                        result["matrix"].append([int(x) for x in val.split()])
                    except ValueError as exc:
                        raise ValueError(f"malformed matrix row {key!r}: {val!r}") from exc
            elif current_section in result and isinstance(result[current_section], dict):
                result[current_section][key] = val

    # setupstr.m:478-495 parity: configuration files made before type 'piezo'
    # existed declare piezo sensors as type=accel with coef0=0 and coef1=1
    # (all 2013-2016 CASPER MicroRider configs). EXACT-string comparison like
    # the vendor's strcmp — "0.0"/"1.0" must NOT flip, and both keys must be
    # present. Rewriting here means every consumer (PFile conversion, adapter,
    # convert, sensor inventory) agrees on the corrected type.
    for ch in result["channels"]:
        if (
            ch.get("type", "").strip().lower() == "accel"
            and ch.get("coef0", "").strip() == "0"
            and ch.get("coef1", "").strip() == "1"
        ):
            ch["type"] = "piezo"

    return result


def instrument_sn(info: dict[str, str]) -> str:
    """Instrument serial number from a parsed ``instrument_info`` mapping.

    Modern configs write ``sn``; CASPER-era (pre-2017) MicroRider configs
    write ``serial_num`` (vendor parity: quick_look.m reads serial_num).
    Returns "" when neither key carries a non-empty value.
    """
    return info.get("sn") or info.get("serial_num") or ""


def read_config_string(filepath: str | Path) -> str:
    """Return the raw embedded configuration (INI) string from a .p file.

    This reads only the 128-byte header and the record-0 config block; it does
    not demultiplex any data records. Unlike constructing a :class:`PFile`, it
    therefore succeeds on startup files that carry a config record but no data
    (e.g. an instrument power-up file), which are exactly the files whose
    configuration you most often want to inspect. It still validates the
    header-geometry (``header_size``) and that the advertised config block is
    fully present in the file, so a genuinely truncated/corrupt config raises
    rather than silently returning a partial string (matching
    :func:`odas_tpw.rsi.config_patch.read_config_text`).
    """
    filepath = Path(filepath)
    with open(filepath, "rb") as f:
        raw_hdr = f.read(HEADER_BYTES)
        if len(raw_hdr) < HEADER_BYTES:
            raise ValueError(f"{filepath.name}: file too small for header")
        endian = _detect_endian(raw_hdr, filepath)
        header = _parse_header(raw_hdr, endian)

        header_size = header["header_size"]
        config_size = header["config_size"]
        if header_size < HEADER_BYTES:
            raise ValueError(f"{filepath.name}: invalid header_size={header_size}")

        file_size = f.seek(0, 2)
        if header_size + config_size > file_size:
            raise ValueError(
                f"{filepath.name}: header_size+config_size ({header_size + config_size}) "
                f"exceeds file size ({file_size}); truncated or corrupt"
            )

        f.seek(header_size)
        cfg = f.read(config_size)
        if len(cfg) != config_size:
            raise ValueError(f"{filepath.name}: config string truncated")
        return cfg.decode("ascii", errors="replace")


# ---------------------------------------------------------------------------
# Main reader
# ---------------------------------------------------------------------------


class PFile:
    """Represents a parsed Rockland Scientific .p binary data file.

    Attributes
    ----------
    filepath : Path
        Path to the source .p file.
    channels : dict[str, ndarray]
        Channel data arrays keyed by name (e.g. 'P', 'T1', 'sh1', 'Ax').
        Values are in physical units after conversion, EXCEPT the shear
        channels 'sh1'/'sh2', which hold the ODAS intermediate still missing
        the /speed² fall-rate normalization (applied downstream in the
        epsilon/chi path); they become physical shear (s⁻¹) only after that.
        Note:
        pre-emphasized channels keep their raw names after deconvolution —
        'T1_dT1' holds the *deconvolved* fast temperature in deg C (not
        the pre-emphasized signal), and 'P_dP' holds the deconvolved
        slow pressure.  (ODAS renames these to 'T1_fast'/'P_slow'.)
    channel_info : dict[str, dict]
        Per-channel metadata: 'type', 'units', and config parameters.
    config : dict
        Parsed INI configuration (see :func:`parse_config`).
    config_str : str
        Raw configuration string from the file header.
    t_fast : ndarray
        Time vector for fast-sampled channels [s since start].
    t_slow : ndarray
        Time vector for slow-sampled channels [s since start].
    fs_fast : float
        Fast sampling rate [Hz] (typically ~512 Hz).
    fs_slow : float
        Slow sampling rate [Hz] (typically ~64 Hz).
    start_time : datetime
        Recording start time (UTC).
    header : dict
        Parsed 128-byte binary header fields.
    endian : str
        Byte order ('little' or 'big').
    """

    def __init__(self, filepath: str | Path, *, deconvolve: bool = True) -> None:
        self.filepath = Path(filepath)
        # deconvolve=False leaves the pre-emphasized channels (T1_dT1, P_dP, …)
        # and their bases (T1, P, …) as the raw counts read from the file,
        # skipping the Mudge & Lueck high-resolution reconstruction. The bench
        # test (bench.py) needs the raw pre-emphasized counts, because the
        # Rockland bench-test checklist thresholds are defined on that signal
        # (and compared against the RSI calibration report). Every production
        # caller uses the default and gets the deconvolved data as before.
        self._deconvolve = deconvolve
        self._read()

    def _read(self):
        with open(self.filepath, "rb") as f:
            raw_hdr = f.read(HEADER_BYTES)
            # Short-header guard: a truncated file otherwise raises a raw
            # struct.error from _detect_endian/_parse_header, which is not an
            # OSError/ValueError and so escapes every batch handler. Mirror
            # extract_pfile_segment's ValueError that all callers catch.
            if len(raw_hdr) < HEADER_BYTES:
                raise ValueError(f"{self.filepath.name}: file too small for header")
            self.endian = _detect_endian(raw_hdr, self.filepath)
            self.header = _parse_header(raw_hdr, self.endian)

            header_size = self.header["header_size"]
            config_size = self.header["config_size"]
            record_size = self.header["record_size"]

            # Header-geometry guard (mirrors extract_pfile_segment): without it
            # a corrupt record_size/header_size surfaces only later as an opaque
            # numpy "cannot reshape array" error instead of a clear diagnostic.
            if header_size < HEADER_BYTES:
                raise ValueError(f"{self.filepath.name}: invalid header_size={header_size}")
            if config_size < 0:
                raise ValueError(f"{self.filepath.name}: invalid config_size={config_size}")
            if record_size <= header_size:
                raise ValueError(
                    f"{self.filepath.name}: invalid record_size={record_size}; "
                    f"expected > header_size={header_size}"
                )

            f.seek(header_size)
            self.config_str = f.read(config_size).decode("ascii", errors="replace")
            self.config = parse_config(self.config_str)

            self.fast_cols = self.header["fast_cols"]
            self.slow_cols = self.header["slow_cols"]
            self.n_cols = self.fast_cols + self.slow_cols
            self.n_rows = self.header["n_rows"]
            self.matrix = np.array(self.config["matrix"])

            f_clock = self.header["clock_hz"] + self.header["clock_frac"] / 1000
            # Geometry guard: a corrupt header with n_cols/n_rows == 0 (or a
            # zero clock) otherwise produces a bare ZeroDivisionError with no
            # file context in the sampling-rate divisions below.
            if self.n_cols < 1 or self.n_rows < 1 or f_clock <= 0:
                raise ValueError(
                    f"{self.filepath.name}: invalid matrix geometry "
                    f"n_cols={self.n_cols} n_rows={self.n_rows} f_clock={f_clock}"
                )
            self.fs_fast = f_clock / self.n_cols
            self.fs_slow = self.fs_fast / self.n_rows

            h = self.header
            self.start_time = datetime(
                h["year"],
                h["month"],
                h["day"],
                h["hour"],
                h["minute"],
                h["second"],
                h["millisecond"] * 1000,
                # Header words are unpacked unsigned; a negative timezone
                # (west of UTC) is stored as two's complement and must be
                # reinterpreted as int16 or timezone() raises ValueError.
                tzinfo=timezone(
                    timedelta(
                        minutes=h["timezone_min"] - 2**16
                        if h["timezone_min"] >= 2**15
                        else h["timezone_min"]
                    )
                ),
            )

            # ODAS defines the DATA start time as the header timestamp MINUS the
            # record duration (odas_p2mat.m:413-417: t_start = filetime - recsize),
            # because the header time marks the END of record 0. setupstr defaults
            # recsize to 1.0 s when neither 'recsize' nor 'recordDuration' is present
            # in the [root] config. Without this subtraction every absolute time
            # coordinate is one record (~1 s) later than the Rockland/ODAS reference
            # (verified +1.0000076 s vs the MATLAB _allch.nc on SN479_0013).
            # parse_config lower-cases all keys, so match 'recsize' /
            # 'recordduration' (ODAS's two accepted spellings).
            root_cfg = self.config.get("root", {}) if isinstance(self.config, dict) else {}
            _recsize_raw = root_cfg.get("recsize", root_cfg.get("recordduration"))
            try:
                self.recsize = float(_recsize_raw) if _recsize_raw is not None else 1.0
            except (TypeError, ValueError):
                self.recsize = 1.0
            self.start_time -= timedelta(seconds=self.recsize)

            first_record_size = header_size + config_size
            f.seek(0, 2)
            file_size = f.tell()
            n_records = (file_size - first_record_size) // record_size
            if n_records < 1:
                raise ValueError(f"{self.filepath.name} contains no data records")
            if (file_size - first_record_size) % record_size != 0:
                # read_odas.m:184-187 warns in this case; a partial record
                # usually means a truncated download or interrupted DAQ.
                warnings.warn(
                    f"{self.filepath.name}: file size is not an integer number "
                    f"of records; trailing partial record ignored",
                    stacklevel=2,
                )

            data_words = (record_size - header_size) // 2
            # Scan-geometry guard: data_words must be a positive multiple of
            # n_cols, otherwise the records['data'].reshape below raises an
            # opaque "cannot reshape array" ValueError on a corrupt header.
            if data_words < self.n_cols or data_words % self.n_cols != 0:
                raise ValueError(
                    f"{self.filepath.name}: corrupt record geometry; "
                    f"data words per record ({data_words}) is not a positive "
                    f"multiple of n_cols ({self.n_cols})"
                )
            dtype = ">i2" if self.endian == ">" else "<i2"
            udtype = dtype.replace("i", "u")

            # Single np.fromfile read with a structured dtype: avoids the
            # per-record Python loop and the n_records-way list+vstack that
            # used to dominate the loader for large files (15-record VMP →
            # 5,000-record SN465 jumps from <1 s to ~7 s with the old loop).
            record_dtype = np.dtype(
                [
                    ("hdr", udtype, (header_size // 2,)),
                    ("data", dtype, (data_words,)),
                ]
            )
            f.seek(first_record_size)
            records = np.fromfile(f, dtype=record_dtype, count=n_records)
            scans_per_record = data_words // self.n_cols
            total_scans = n_records * scans_per_record
            raw_flat = records["data"].reshape(total_scans, self.n_cols)

            self.channels_raw = {}
            self.channels = {}
            self.channel_info = {}
            self._record_headers = records["hdr"]
            # ODAS odas_p2mat runs check_bad_buffers before conversion: header
            # word 16 (0-indexed 15), buffer_status, is a per-record bad-buffer /
            # DAQ-dropout flag. We do not repair flagged records, but warn so
            # silent DAQ corruption is not mistaken for clean data feeding
            # epsilon/chi. Clean acquisitions (e.g. all SN479 VMP files) carry 0.
            if self._record_headers.shape[1] > 15:
                n_bad_buf = int(np.count_nonzero(self._record_headers[:, 15]))
                if n_bad_buf:
                    warnings.warn(
                        f"{self.filepath.name}: {n_bad_buf}/{n_records} records "
                        "flagged bad-buffer (header word 16); DAQ dropouts may "
                        "corrupt samples (ODAS check_bad_buffers would patch "
                        "these before conversion)",
                        stacklevel=2,
                    )
            # Names of joined 2-id (32-bit) channels — needed below to apply
            # ODAS's default-signed correction only to true 32-bit values.
            joined_channels: set[str] = set()

            ch_config = {}
            for ch in self.config["channels"]:
                if "name" not in ch or "type" not in ch:
                    continue
                if "id" in ch:
                    id_str = ch["id"]
                elif "id_even" in ch and "id_odd" in ch:
                    # setupstr.m:441-455 dialect: old configs may declare a
                    # 32-bit channel's word pair as id_even/id_odd with no
                    # 'id'. Synthesize even-then-odd (even = low word), so the
                    # 2-id join below behaves identically to an explicit
                    # "id = even,odd" (#131 m7).
                    id_str = f"{ch['id_even']} {ch['id_odd']}"
                else:
                    continue
                # Guard the channel-id parse the same way as the matrix rows:
                # a non-numeric id token from a corrupt config otherwise raises
                # a bare "invalid literal for int()" with no file/channel context.
                try:
                    ids = [int(x) for x in id_str.replace(",", " ").split()]
                except ValueError as exc:
                    raise ValueError(
                        f"{self.filepath.name}: malformed channel id "
                        f"{id_str!r} for channel {ch.get('name', '?')!r}"
                    ) from exc
                ch_config[ch["name"].strip()] = {"ids": ids, **ch}

            matrix = self.matrix
            unique_ids = set(matrix.flatten())
            matrix_count = total_scans // self.n_rows

            # read_odas.m:285-302 parity: a matrix address with no usable
            # [channel] section is silently unprocessable — warn so the data
            # loss is visible. Address 255 is the RSI special character, for
            # which read_odas.m:236-243 inserts a synthetic 'ch255' section;
            # old CASPER matrices carry it with no section, so it is exempt.
            claimed_ids = {i for info in ch_config.values() for i in info["ids"]}
            orphan_ids = sorted(int(i) for i in unique_ids - claimed_ids if i != 255)
            if orphan_ids:
                warnings.warn(
                    f"{self.filepath.name}: matrix address(es) "
                    f"{', '.join(map(str, orphan_ids))} have no usable "
                    f"[channel] section; their samples will not be processed",
                    stacklevel=2,
                )

            for ch_name, info in ch_config.items():
                ids = info["ids"]

                if len(ids) == 1:
                    ch_id = ids[0]
                    if ch_id not in unique_ids:
                        continue
                    col_positions = np.where(matrix == ch_id)
                    if len(col_positions[1]) == 0:
                        continue
                    all_rows_same = np.all(matrix[:, col_positions[1][0]] == ch_id)
                    n_occ = len(col_positions[0])

                    # Only two layouts are supported: a full matrix column
                    # (fast channel) or a single occurrence (slow channel).
                    # Intermediate rates (read_odas.m:328-333 gathers every
                    # occurrence in scan order) would be silently decimated
                    # here — warn so the data loss is visible.  Ground
                    # reference channels are exempt: sampling 'gnd' several
                    # times per scan is normal instrument design and its
                    # data content is a zero reference, not a measurement.
                    expected_occ = self.n_rows if all_rows_same else 1
                    is_gnd = (
                        info.get("type", "").strip().lower() == "gnd"
                        or ch_name.strip().lower() == "gnd"
                    )
                    if n_occ != expected_occ and not is_gnd:
                        warnings.warn(
                            f"{self.filepath.name}: channel '{ch_name}' (id {ch_id}) "
                            f"appears {n_occ}x per scan matrix; only "
                            f"{'one column' if all_rows_same else 'the first occurrence'} "
                            f"is used, so its sample rate and data are incomplete",
                            stacklevel=2,
                        )

                    # Keep raw int16 — converters auto-promote via numpy
                    # broadcasting when they multiply by float scale factors.
                    # Skipping the .astype(np.float64) here saves a permanent
                    # 4x-per-channel buffer (~40-60 MB on SN465-class files).
                    # .copy() so subsequent in-place ops (unsigned wrap,
                    # deconvolution overwrites) don't alias raw_flat.
                    if all_rows_same:
                        col_idx = col_positions[1][0]
                        raw_ch = raw_flat[: matrix_count * self.n_rows, col_idx].copy()
                    else:
                        row_idx, col_idx = col_positions[0][0], col_positions[1][0]
                        raw_ch = raw_flat[
                            row_idx : matrix_count * self.n_rows : self.n_rows,
                            col_idx,
                        ].copy()

                    self.channels_raw[ch_name] = raw_ch

                elif len(ids) == 2:
                    # Config-declared order, not sorted(): ODAS read_odas.m
                    # tags the FIRST-listed id even/low word (`_E`) and the
                    # SECOND odd/high word (`_O`), then joins chO*2^16 + chE.
                    # sorted() only coincides when the low word is listed first
                    # (#0).
                    id_even, id_odd = ids[0], ids[1]
                    if id_even not in unique_ids or id_odd not in unique_ids:
                        continue
                    col_e = np.where(matrix == id_even)
                    col_o = np.where(matrix == id_odd)
                    row_e, ce = col_e[0][0], col_e[1][0]
                    row_o, co = col_o[0][0], col_o[1][0]
                    even_data = raw_flat[
                        row_e : matrix_count * self.n_rows : self.n_rows,
                        ce,
                    ].astype(np.float64)
                    odd_data = raw_flat[
                        row_o : matrix_count * self.n_rows : self.n_rows,
                        co,
                    ].astype(np.float64)
                    even_data[even_data < 0] += 2**16
                    odd_data[odd_data < 0] += 2**16
                    self.channels_raw[ch_name] = odd_data * 2**16 + even_data
                    joined_channels.add(ch_name)

            self.t_fast = np.arange(matrix_count * self.n_rows) / self.fs_fast
            self.t_slow = np.arange(matrix_count) / self.fs_slow

            self._fast_channels = set()
            self._slow_channels = set()
            for ch_name, info in ch_config.items():
                if ch_name not in self.channels_raw:
                    continue
                ids = info["ids"]
                ch_id = ids[0]
                col_positions = np.where(matrix == ch_id)
                if len(col_positions[1]) == 0:
                    continue
                if np.all(matrix[:, col_positions[1][0]] == ch_id) and len(ids) == 1:
                    self._fast_channels.add(ch_name)
                else:
                    self._slow_channels.add(ch_name)

            # --- Deconvolution (Mudge & Lueck 1994) ---
            # Channels with diff_gain (except shear probes) are deconvolved
            # by combining the slow-rate channel X with its pre-emphasized
            # fast-rate counterpart X_dX to produce a high-resolution signal.
            # This matches ODAS odas_p2mat.m lines 516-570.
            if self._deconvolve:
                self._apply_deconvolution(ch_config)

            # Unsigned wrapping: channels with sign=unsigned (or jac_t type)
            # need negative int16 values converted to unsigned before conversion.
            # Matches ODAS read_odas.m lines 370-398.
            _ALWAYS_UNSIGNED = {"jac_t"}
            # Types ODAS skips entirely in its sign loop (already converted).
            _SIGN_SKIP = {"sbt", "sbc", "jac_c", "o2_43f"}
            for ch_name in list(self.channels_raw.keys()):
                info = ch_config.get(ch_name, {})
                ch_type = info.get("type", "raw").strip().lower()
                sign = info.get("sign", "").strip().lower()
                if sign == "unsigned" or ch_type in _ALWAYS_UNSIGNED:
                    raw = self.channels_raw[ch_name]
                    # Zero-copy bit reinterpretation — int16 -1 → uint16 65535,
                    # which is what ``raw[raw<0] += 2**16`` produced when raw
                    # was already float64.  Falls through to the in-place add
                    # for any non-int16 (deconvolved float64) channels, where
                    # the wrap can't overflow.  ``raw.dtype == np.int16``
                    # would fail on the endian-marked ``>i2`` / ``<i2`` dtypes
                    # that frombuffer/fromfile actually return — compare on
                    # kind+itemsize instead.
                    if raw.dtype.kind == "i" and raw.dtype.itemsize == 2:
                        self.channels_raw[ch_name] = raw.view(
                            np.dtype(raw.dtype.str.replace("i", "u"))
                        )
                    else:
                        raw[raw < 0] += 2**16
                elif ch_name in joined_channels and ch_type not in _SIGN_SKIP:
                    # ODAS default-signed branch (read_odas.m:393-397): every
                    # joined 32-bit channel not in the skip set and not marked
                    # unsigned is signed by default, so values with the high bit
                    # set must wrap down by 2**32.  Real ARCTERX/SN479 2-id data
                    # is jac_c (skipped, unsigned), so this only fires for other
                    # signed 32-bit configs, but without it the port diverged
                    # from ODAS by 2**32 on such channels.
                    raw = self.channels_raw[ch_name]
                    raw[raw >= 2**31] -= 2**32

            for ch_name in list(self.channels_raw.keys()):
                info = ch_config.get(ch_name, {})
                ch_type = info.get("type", "raw").strip().lower()
                convert_info = dict(info)

                # When deconvolution is skipped, a pre-emphasized channel
                # (X_dX, e.g. T1_dT1 / P_dP) keeps its own sparse config, which
                # lacks the base channel's calibration coefficients. Converting
                # it would emit misleading "physical units are suspect" warnings
                # and produce a physical value that is undefined anyway (the
                # differentiator output only becomes temperature/pressure after
                # deconvolution). Keep raw counts instead — callers that ask for
                # deconvolve=False want the raw pre-emphasized counts.
                if not self._deconvolve and re.match(r"^(\w+)_d\1$", ch_name):
                    self.channels[ch_name] = self.channels_raw[ch_name]
                    self.channel_info[ch_name] = {"units": "counts", "type": ch_type}
                    continue

                converter = CONVERTERS.get(ch_type)
                if converter is None:
                    warnings.warn(f"No converter for type '{ch_type}' (channel {ch_name})")
                    self.channels[ch_name] = self.channels_raw[ch_name]
                    self.channel_info[ch_name] = {"units": "counts", "type": ch_type}
                    continue

                phys, units = converter(self.channels_raw[ch_name], convert_info)
                self.channels[ch_name] = phys
                self.channel_info[ch_name] = {"units": units, "type": ch_type}
                # Shear (sh1/sh2) carries the ODAS intermediate that still needs
                # the /speed^2 fall-rate normalization to become physical shear.
                # ``units`` stays UDUNITS-valid "s-1"; flag the pre-normalization
                # state CF-legally in a free-text ``comment`` so a reader of the
                # per-profile NetCDF does not treat it as final shear. (#104 U1-1.)
                if ch_type == "shear":
                    self.channel_info[ch_name]["comment"] = (
                        "un-normalized ODAS shear intermediate; still missing the "
                        "/speed^2 fall-rate normalization applied downstream in the "
                        "epsilon/chi path (physical velocity shear, s-1, only after)"
                    )

    def _apply_deconvolution(self, ch_config: dict) -> None:
        """Deconvolve pre-emphasized channels to produce high-resolution data.

        For each channel with ``diff_gain`` that is not a shear probe,
        look for a matching X / X_dX pair (e.g. T1 / T1_dT1, P / P_dP).
        The deconvolved high-resolution signal replaces the original
        slow-rate channel and is also stored at the fast rate.

        Mirrors the deconvolution block in odas_p2mat.m (lines 516-610).
        """
        shear_types = {"shear", "xmp_shear"}
        n_slow = len(self.t_slow)
        n_fast = len(self.t_fast)
        if n_slow == 0:
            return

        for ch_name, info in list(ch_config.items()):
            ch_type = info.get("type", "").strip().lower()
            if ch_type in shear_types:
                continue
            if "diff_gain" not in info:
                continue

            # This channel has diff_gain and is not shear → candidate.
            # Check if it matches the X_dX naming pattern.
            m = re.match(r"^(\w+)_d\1$", ch_name)
            if not m:
                continue

            base_name = m.group(1)
            dX_name = ch_name  # e.g. T1_dT1

            if dX_name not in self.channels_raw:
                continue

            diff_gain_val = float(info["diff_gain"])
            X_dX_raw = self.channels_raw[dX_name]

            # Sampling rate of the pre-emphasized channel, derived from the
            # ARRAY actually extracted rather than the matrix occurrence
            # count (odas_p2mat.m l.564 counts occurrences, but ODAS also
            # KEEPS every occurrence; extraction above keeps either a full
            # fast column or only the FIRST occurrence per scan). For a
            # duplicate-sampled slow channel (CASPER 4x10 P/P_dP, 2x per
            # scan) the matrix count would drive deconvolve() at 2x the
            # decimated array's true rate, applying the Butterworth
            # pre-emphasis crossover at twice its design frequency and
            # mis-blending X and X_dX (#131 M12).
            rate_mult = len(X_dX_raw) // n_slow
            # Extraction guarantees exact multiples: a full column yields
            # n_rows*n_slow samples, anything else exactly n_slow.
            assert rate_mult * n_slow == len(X_dX_raw)
            fs_dX = self.fs_slow * rate_mult
            is_dX_fast = rate_mult == self.n_rows

            # Get the non-pre-emphasized channel if available
            X_raw = self.channels_raw.get(base_name)

            # Deconvolve on raw data (before physical-unit conversion)
            hres = deconvolve(X_raw, X_dX_raw, fs_dX, diff_gain_val)

            if is_dX_fast:
                # T1_dT1 case: X_dX is fast-rate → hres is fast-rate.
                # Replace slow-rate base with hres subsampled to slow rate.
                self.channels_raw[base_name] = hres[:: self.n_rows][:n_slow]
                # Replace _dX raw data with full fast-rate hres.
                self.channels_raw[dX_name] = hres[:n_fast]
            else:
                # P_dP case: both X and X_dX are slow-rate → hres is
                # slow-rate.  Replace the base channel with hres.
                self.channels_raw[base_name] = hres[:n_slow]
                # The _dX raw data also becomes hres (same slow rate).
                self.channels_raw[dX_name] = hres[:n_slow]

            # Both branches leave the base holding a slow-length view, no
            # matter how it was sampled. Old MR configs (RSI 2013 small-form
            # template; CASPER 2015_East CAS_001-006) sample the base as a
            # full fast column too — reclassify it or is_fast() lies and the
            # rate/length invariant breaks downstream (nc gradT crash,
            # #131 M11). The full fast-rate signal lives on in the dX channel.
            if base_name in self._fast_channels:
                self._fast_channels.discard(base_name)
                self._slow_channels.add(base_name)

            # The _dX channel should now be converted using the base
            # channel's calibration parameters (not its own sparse ones).
            if base_name in ch_config:
                base_info = dict(ch_config[base_name])
                base_info.pop("diff_gain", None)
                ch_config[dX_name] = {**base_info, "name": dX_name}
                if is_dX_fast:
                    self._fast_channels.add(dX_name)
                    self._slow_channels.discard(dX_name)

    def is_fast(self, ch_name: str) -> bool:
        return ch_name in self._fast_channels

    def summary(self) -> None:
        print(f"File: {self.filepath.name}")
        print(
            f"Instrument: {self.config['instrument_info'].get('model', '?')} "
            f"SN {instrument_sn(self.config['instrument_info']) or '?'}"
        )
        print(f"Start: {self.start_time.isoformat()}")
        print(f"Endian: {'big' if self.endian == '>' else 'little'}")
        print(
            f"Matrix: {self.n_rows} rows x {self.n_cols} cols "
            f"({self.fast_cols} fast + {self.slow_cols} slow)"
        )
        print(f"fs_fast = {self.fs_fast:.3f} Hz, fs_slow = {self.fs_slow:.3f} Hz")
        print(f"Duration: {self.t_fast[-1]:.1f} s")
        print(f"\nChannels ({len(self.channels)}):")
        for name in sorted(self.channels.keys()):
            info = self.channel_info[name]
            rate = "fast" if self.is_fast(name) else "slow"
            n = len(self.channels[name])
            print(f"  {name:>15s}  [{info['units']:>10s}]  {rate:4s}  n={n}")
