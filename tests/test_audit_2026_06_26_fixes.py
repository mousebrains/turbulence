# Jun-2026, Claude and Pat Welch, pat@mousebrains.com
"""Regression tests for MAJOR findings of the 2026-06-26 multi-round review.

Each test is constructed to FAIL on the pre-fix code and pass on the fix:
  M14  extract_profiles crashes (AttributeError) on a channel with _FillValue
  M20  config_patch edit-spec KEY injection ([root] stanza via a newline/'=')
  M41  chi_io Method-1 time base wrong for a non-UTC start_time offset
  M95  merge chains files with sequential numbers but no time continuity
  M101 chi_io Method-1 probe filter uses the wrong FOM statistic (fom vs FM)
"""

from __future__ import annotations

import struct
from datetime import datetime

import numpy as np
import pytest

from odas_tpw.rsi.p_file import _H, HEADER_BYTES, HEADER_WORDS

# --------------------------------------------------------------------------- #
# M14: extract_profiles must not crash on a channel carrying _FillValue
# --------------------------------------------------------------------------- #


class TestM14FillValueAttr:
    def test_extract_profiles_handles_fillvalue_channel(self, tmp_path):
        nc = pytest.importorskip("netCDF4")
        from odas_tpw.rsi.profile import extract_profiles

        n_fast, n_slow = 64, 8
        path = tmp_path / "fv.nc"
        ds = nc.Dataset(str(path), "w")
        ds.fs_fast = 512.0
        ds.fs_slow = 64.0
        ds.createDimension("time_fast", n_fast)
        ds.createDimension("time_slow", n_slow)
        ds.createVariable("t_fast", "f8", ("time_fast",))[:] = np.arange(n_fast) / 512.0
        ds.createVariable("t_slow", "f8", ("time_slow",))[:] = np.arange(n_slow) / 64.0
        ds.createVariable("P", "f8", ("time_slow",))[:] = np.linspace(1.0, 50.0, n_slow)
        # A channel that declares a _FillValue (external/CF/ATOMIX style) — the
        # pre-fix per-profile write loop did setattr(var, "_FillValue", ...),
        # which netCDF4 forbids post-creation -> AttributeError.
        t1 = ds.createVariable("T1", "f8", ("time_fast",), fill_value=-999.0)
        t1[:] = np.linspace(10.0, 11.0, n_fast)
        t1.units = "degC"
        ds.close()

        out = tmp_path / "profiles"
        paths = extract_profiles(path, out, profiles=[(0, n_slow - 1)])
        assert len(paths) >= 1
        assert paths[0].exists()


# --------------------------------------------------------------------------- #
# M20: edit-spec KEYS must be validated like values (no stanza injection)
# --------------------------------------------------------------------------- #


class TestM20KeyInjection:
    @staticmethod
    def _spec(tmp_path, text):
        p = tmp_path / "e.yaml"
        p.write_text(text, encoding="utf-8")
        return p

    def test_newline_in_key_rejected(self, tmp_path):
        from odas_tpw.rsi.config_patch import load_edit_spec

        # A double-quoted key with \n injects a [root] stanza when written.
        yaml = 'note: "x"\ninstrument_info:\n  "a = 1\\n[root]\\nrate": "9"\n'
        with pytest.raises(ValueError, match="key"):
            load_edit_spec(self._spec(tmp_path, yaml))

    def test_equals_in_key_rejected(self, tmp_path):
        from odas_tpw.rsi.config_patch import load_edit_spec

        with pytest.raises(ValueError, match="key"):
            load_edit_spec(
                self._spec(tmp_path, 'note: "x"\ninstrument_info:\n  "foo = bar": "9"\n')
            )


# --------------------------------------------------------------------------- #
# M41: Method-1 time base must honor a non-UTC start_time offset
# --------------------------------------------------------------------------- #


class TestM41Timezone:
    def test_non_utc_start_time_offset_honored(self):
        import xarray as xr

        from odas_tpw.rsi.chi_io import _epsilon_ds_to_l4data

        n = 4
        start_utc = np.datetime64("2026-03-25T10:00:00.000")
        offsets_s = np.array([10.0, 11.0, 12.0, 13.0])
        t = start_utc + (offsets_s * 1e9).astype("timedelta64[ns]")
        ds = xr.Dataset(
            {"epsilon": (["probe", "time"], np.full((2, n), 1e-7))},
            coords={"probe": ["sh1", "sh2"], "t": ("time", t)},
            # Local wall-clock 20:00+10:00 == 10:00 UTC (the true reference).
            attrs={"start_time": "2026-03-25T20:00:00+10:00"},
        )
        l4 = _epsilon_ds_to_l4data(ds)
        # Pre-fix stripped the offset -> reference 36000 s off -> times shifted.
        np.testing.assert_allclose(l4.time, offsets_s, atol=1e-6)


# --------------------------------------------------------------------------- #
# M101: Method-1 probe filter must use the Lueck FM statistic, not fom
# --------------------------------------------------------------------------- #


class TestM101FomVsFM:
    def test_filters_on_fm_not_variance_ratio_fom(self):
        import xarray as xr

        from odas_tpw.rsi.chi_io import _epsilon_ds_to_l4data

        n = 4
        eps = np.array([[1e-7] * n, [3e-7] * n], dtype=np.float64)
        # Probe 2: variance-ratio fom = 1.3 (>1.15, "bad" under the old wrong
        # filter) but FM = 0.5 (a genuinely GOOD fit). It must be KEPT.
        fom = np.array([[1.0] * n, [1.3] * n], dtype=np.float64)
        fm = np.array([[0.5] * n, [0.5] * n], dtype=np.float64)
        ds = xr.Dataset(
            {
                "epsilon": (["probe", "time"], eps),
                "fom": (["probe", "time"], fom),
                "FM": (["probe", "time"], fm),
            },
            coords={"probe": ["sh1", "sh2"], "t": ("time", np.arange(n, dtype=float))},
        )
        l4 = _epsilon_ds_to_l4data(ds)
        # Both probes kept (filtered on FM) -> 2-probe mean, not just probe 1.
        np.testing.assert_allclose(l4.epsi_final, 2e-7)


# --------------------------------------------------------------------------- #
# M95: merge requires recording time-continuity, not just sequential numbers
# --------------------------------------------------------------------------- #


def _make_timed_p_file(path, *, file_number, start: datetime, n_records=10):
    """Synthetic .p with valid date words + clock geometry so _read_merge_info
    can compute start_time and duration (each record = 0.1 s here)."""
    record_size, header_size, config_size = 512, HEADER_BYTES, 128
    words = [0] * HEADER_WORDS
    words[_H["header_size"]] = header_size
    words[_H["config_size"]] = config_size
    words[_H["record_size"]] = record_size
    words[_H["file_number"]] = file_number
    words[_H["endian"]] = 1
    words[_H["year"]] = start.year
    words[_H["month"]] = start.month
    words[_H["day"]] = start.day
    words[_H["hour"]] = start.hour
    words[_H["minute"]] = start.minute
    words[_H["second"]] = start.second
    words[_H["millisecond"]] = start.microsecond // 1000
    # n_cols = 10, fs_fast = clock/n_cols = 1900/10 = 190 Hz;
    # scans_per_record = ((512-128)//2)//10 = 19; duration = n_rec*19/190 = 0.1*n_rec s
    words[_H["fast_cols"]] = 8
    words[_H["slow_cols"]] = 2
    words[_H["clock_hz"]] = 1900
    words[_H["clock_frac"]] = 0
    hdr = struct.pack(f"<{HEADER_WORDS}H", *words).ljust(header_size, b"\x00")
    cfg = b"[root]\nversion=1\n"[:config_size].ljust(config_size, b"\x00")
    path.write_bytes(hdr + cfg + bytes([1]) * record_size * n_records)
    return path


class TestM95MergeContinuity:
    def test_continuous_files_merge(self, tmp_path):
        from odas_tpw.perturb.merge import find_mergeable_files

        # Each file is 1.0 s (10 records). File 2 starts where file 1 ended.
        f1 = _make_timed_p_file(
            tmp_path / "SN_0001.p", file_number=1, start=datetime(2025, 1, 14, 15, 30, 0)
        )
        f2 = _make_timed_p_file(
            tmp_path / "SN_0002.p", file_number=2, start=datetime(2025, 1, 14, 15, 30, 1)
        )
        chains = find_mergeable_files([f1, f2])
        assert len(chains) == 1 and len(chains[0]) == 2

    def test_independent_casts_not_merged(self, tmp_path):
        from odas_tpw.perturb.merge import find_mergeable_files

        # Sequential numbers + identical config/geometry, but file 2 starts 30
        # minutes after file 1 ended -> two independent casts, must NOT chain.
        f1 = _make_timed_p_file(
            tmp_path / "SN_0001.p", file_number=1, start=datetime(2025, 1, 14, 15, 30, 0)
        )
        f2 = _make_timed_p_file(
            tmp_path / "SN_0002.p", file_number=2, start=datetime(2025, 1, 14, 16, 0, 0)
        )
        chains = find_mergeable_files([f1, f2])
        assert len(chains) == 0
