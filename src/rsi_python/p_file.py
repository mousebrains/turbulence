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

from rsi_python.channels import CONVERTERS
from rsi_python.deconvolve import deconvolve

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


def _detect_endian(raw_header: bytes) -> str:
    """Return '>' (big) or '<' (little) endian prefix for struct."""
    be = struct.unpack_from(">H", raw_header, 63 * 2)[0]
    le = struct.unpack_from("<H", raw_header, 63 * 2)[0]
    if be == 2:
        return ">"
    if le == 1:
        return "<"
    if be == 0 or le == 0:
        warnings.warn("Endian flag is 0; assuming little-endian")
        return "<"
    be_hs = struct.unpack_from(">H", raw_header, 17 * 2)[0]
    le_hs = struct.unpack_from("<H", raw_header, 17 * 2)[0]
    if be_hs == 128:
        return ">"
    if le_hs == 128:
        return "<"
    warnings.warn("Cannot determine endian; defaulting to big-endian")
    return ">"


def _parse_header(raw: bytes, endian: str) -> dict:
    fmt = f"{endian}{HEADER_WORDS}H"
    words = struct.unpack(fmt, raw)
    return {name: words[idx] for name, idx in _H.items()}


# ---------------------------------------------------------------------------
# Configuration string parsing
# ---------------------------------------------------------------------------


def parse_config(config_str: str) -> dict[str, Any]:
    """Parse the INI-style configuration string embedded in the P file.

    Returns dict with keys:
      'matrix': list of lists (the address matrix rows)
      'channels': list of dicts, one per [channel] section
      'instrument_info': dict
      'cruise_info': dict
      'root': dict
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
            current_section = m.group(1).strip().lower()
            if current_section == "channel":
                current_channel = {}
                result["channels"].append(current_channel)
            else:
                current_channel = None
            continue

        m = re.match(r"^(.+?)\s*=\s*(.*)$", line)
        if m:
            key = m.group(1).strip().lower()
            val = m.group(2).strip()
            if current_section == "channel" and current_channel is not None:
                current_channel[key] = val
            elif current_section == "matrix":
                if key.startswith("row"):
                    result["matrix"].append([int(x) for x in val.split()])
            elif current_section in result and isinstance(result[current_section], dict):
                result[current_section][key] = val

    return result


# ---------------------------------------------------------------------------
# Main reader
# ---------------------------------------------------------------------------


class PFile:
    """Represents a parsed Rockland .p data file."""

    def __init__(self, filepath: str | Path) -> None:
        self.filepath = Path(filepath)
        self._read()

    def _read(self):
        with open(self.filepath, "rb") as f:
            raw_hdr = f.read(HEADER_BYTES)
            self.endian = _detect_endian(raw_hdr)
            self.header = _parse_header(raw_hdr, self.endian)

            header_size = self.header["header_size"]
            config_size = self.header["config_size"]
            record_size = self.header["record_size"]

            f.seek(header_size)
            self.config_str = f.read(config_size).decode("ascii", errors="replace")
            self.config = parse_config(self.config_str)

            self.fast_cols = self.header["fast_cols"]
            self.slow_cols = self.header["slow_cols"]
            self.n_cols = self.fast_cols + self.slow_cols
            self.n_rows = self.header["n_rows"]
            self.matrix = np.array(self.config["matrix"])

            f_clock = self.header["clock_hz"] + self.header["clock_frac"] / 1000
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
                tzinfo=timezone(timedelta(minutes=h["timezone_min"])),
            )

            first_record_size = header_size + config_size
            f.seek(0, 2)
            file_size = f.tell()
            n_records = (file_size - first_record_size) // record_size
            if n_records < 1:
                raise ValueError(f"{self.filepath.name} contains no data records")

            data_words = (record_size - header_size) // 2
            dtype = ">i2" if self.endian == ">" else "<i2"

            f.seek(first_record_size)
            all_data = []
            record_headers = []
            for _ in range(n_records):
                rec_hdr = np.frombuffer(f.read(header_size), dtype=dtype.replace("i", "u"))
                record_headers.append(rec_hdr)
                rec_data = np.frombuffer(f.read(record_size - header_size), dtype=dtype)
                all_data.append(rec_data)

            raw_block = np.vstack(all_data)
            scans_per_record = data_words // self.n_cols
            total_scans = n_records * scans_per_record
            raw_flat = raw_block.reshape(total_scans, self.n_cols)

            self.channels_raw = {}
            self.channels = {}
            self.channel_info = {}
            self._record_headers = np.vstack(record_headers)

            ch_config = {}
            for ch in self.config["channels"]:
                if "id" not in ch or "name" not in ch or "type" not in ch:
                    continue
                ids = [int(x) for x in ch["id"].replace(",", " ").split()]
                ch_config[ch["name"].strip()] = {"ids": ids, **ch}

            matrix = self.matrix
            unique_ids = set(matrix.flatten())
            matrix_count = total_scans // self.n_rows

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

                    if all_rows_same:
                        col_idx = col_positions[1][0]
                        raw_ch = raw_flat[: matrix_count * self.n_rows, col_idx].astype(np.float64)
                    else:
                        row_idx, col_idx = col_positions[0][0], col_positions[1][0]
                        raw_ch = raw_flat[
                            row_idx : matrix_count * self.n_rows : self.n_rows,
                            col_idx,
                        ].astype(np.float64)

                    self.channels_raw[ch_name] = raw_ch

                elif len(ids) == 2:
                    id_even, id_odd = sorted(ids)
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
            self._apply_deconvolution(ch_config, matrix)

            for ch_name in list(self.channels_raw.keys()):
                info = ch_config.get(ch_name, {})
                ch_type = info.get("type", "raw").strip().lower()
                convert_info = dict(info)

                converter = CONVERTERS.get(ch_type)
                if converter is None:
                    warnings.warn(f"No converter for type '{ch_type}' (channel {ch_name})")
                    self.channels[ch_name] = self.channels_raw[ch_name]
                    self.channel_info[ch_name] = {"units": "counts", "type": ch_type}
                    continue

                phys, units = converter(self.channels_raw[ch_name], convert_info)
                self.channels[ch_name] = phys
                self.channel_info[ch_name] = {"units": units, "type": ch_type}

    def _apply_deconvolution(self, ch_config: dict, matrix: np.ndarray) -> None:
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

            # Determine sampling rate of the pre-emphasized channel from
            # its occurrence count in the address matrix (odas_p2mat.m l.564).
            dX_id = int(info["ids"][0])
            occurrences = np.sum(matrix == dX_id)
            fs_dX = self.fs_fast * occurrences / self.n_rows
            is_dX_fast = occurrences == self.n_rows

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
            f"SN {self.config['instrument_info'].get('sn', '?')}"
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
