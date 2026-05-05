# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for rsi.profile — _load_source error paths, NC reading edge cases."""

from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np
import pytest

from odas_tpw.rsi.profile import _extract_one, _load_from_nc, _load_source, extract_profiles


def _write_nc(path: Path, *, with_l1_group=False, with_root_vars=True, with_pressure=True,
              with_pres_slow=False, no_time=False, with_extra_attr=None,
              t_units_days=False) -> Path:
    """Build minimal NC test files in different layouts."""
    n_fast = 256
    n_slow = 32
    fs_fast = 512.0
    fs_slow = 64.0

    ds = netCDF4.Dataset(str(path), "w", format="NETCDF4")
    try:
        ds.fs_fast = fs_fast
        ds.fs_slow = fs_slow

        if with_extra_attr:
            for k, v in with_extra_attr.items():
                setattr(ds, k, v)

        if with_root_vars:
            ds.createDimension("time_fast", n_fast)
            ds.createDimension("time_slow", n_slow)

            if not no_time:
                tf = ds.createVariable("t_fast", "f8", ("time_fast",))
                if t_units_days:
                    tf[:] = np.arange(n_fast) / fs_fast / 86400
                    tf.units = "days since 2025-01-01"
                else:
                    tf[:] = np.arange(n_fast) / fs_fast
                    tf.units = "seconds"
                ts = ds.createVariable("t_slow", "f8", ("time_slow",))
                if t_units_days:
                    ts[:] = np.arange(n_slow) / fs_slow / 86400
                    ts.units = "days since 2025-01-01"
                else:
                    ts[:] = np.arange(n_slow) / fs_slow
                    ts.units = "seconds"

            if with_pressure:
                p = ds.createVariable("P", "f8", ("time_slow",))
                p[:] = np.linspace(5.0, 50.0, n_slow)
                p.units = "dbar"

            sh1 = ds.createVariable("sh1", "f8", ("time_fast",))
            sh1[:] = np.random.default_rng(0).standard_normal(n_fast) * 0.05
            sh1.units = "s-1"

        if with_l1_group:
            g = ds.createGroup("L1_converted")
            g.createDimension("TIME", n_fast)
            g.createDimension("TIME_SLOW", n_slow)
            tg = g.createVariable("TIME", "f8", ("TIME",))
            tg[:] = np.arange(n_fast) / fs_fast
            tsg = g.createVariable("TIME_SLOW", "f8", ("TIME_SLOW",))
            tsg[:] = np.arange(n_slow) / fs_slow

            if with_pres_slow:
                p = g.createVariable("PRES_SLOW", "f8", ("TIME_SLOW",))
                p[:] = np.linspace(5.0, 50.0, n_slow)
            elif with_pressure:
                p = g.createVariable("PRES", "f8", ("TIME_SLOW",))
                p[:] = np.linspace(5.0, 50.0, n_slow)

            shg = g.createVariable("SHEAR", "f8", ("TIME",))
            shg[:] = np.random.default_rng(0).standard_normal(n_fast) * 0.05
    finally:
        ds.close()
    return path


# ---------------------------------------------------------------------------
# _load_source — unsupported file
# ---------------------------------------------------------------------------


class TestLoadSource:
    def test_unsupported_suffix_raises(self, tmp_path):
        bad = tmp_path / "data.txt"
        bad.write_text("oops")
        with pytest.raises(ValueError, match="Unsupported file type"):
            _load_source(bad)

    def test_pfile_object_dispatch(self, tmp_path):
        """Passing a PFile object goes through _load_from_pfile branch."""
        # Build a PFile-like stub
        from datetime import datetime

        class StubPF:
            def __init__(self):
                self.fs_fast = 512.0
                self.fs_slow = 64.0
                self.t_fast = np.arange(256) / 512.0
                self.t_slow = np.arange(32) / 64.0
                self.start_time = datetime(2025, 1, 1)
                self.channels = {"P": np.linspace(5, 50, 32)}
                self.channel_info = {"P": {"units": "dbar", "type": "pres"}}
                self.config = {
                    "instrument_info": {"model": "VMP-250", "sn": "479"},
                    "cruise_info": {"operator": "Pat", "project": "Test"},
                }
                self.config_str = "[instrument_info]\nmodel=VMP-250"
                self.filepath = Path("fake.p")
                self._fast_channels = set()

            def is_fast(self, name):
                return name in self._fast_channels

        from odas_tpw.rsi.profile import _load_from_pfile

        data = _load_from_pfile(StubPF())
        assert data["fs_fast"] == 512.0
        assert "P" in data
        assert data["stem"] == "fake"
        assert data["global_attrs"]["instrument_model"] == "VMP-250"
        assert data["global_attrs"]["configuration_string"].startswith("[instrument_info]")


# ---------------------------------------------------------------------------
# _load_from_nc — error paths and fallbacks
# ---------------------------------------------------------------------------


class TestLoadFromNc:
    def test_root_format_with_pressure(self, tmp_path):
        nc = _write_nc(tmp_path / "root.nc")
        data = _load_from_nc(nc)
        assert data["fs_fast"] == 512.0
        assert "P" in data
        assert data["channels"]

    def test_l1_group_with_pres_slow(self, tmp_path):
        """L1_converted group with PRES_SLOW takes precedence over PRES."""
        nc = _write_nc(
            tmp_path / "l1.nc",
            with_root_vars=False,
            with_l1_group=True,
            with_pres_slow=True,
        )
        data = _load_from_nc(nc)
        assert data["P"].shape == (32,)

    def test_l1_group_with_pres(self, tmp_path):
        """Falls back to PRES when PRES_SLOW absent."""
        nc = _write_nc(
            tmp_path / "l1pres.nc",
            with_root_vars=False,
            with_l1_group=True,
        )
        data = _load_from_nc(nc)
        assert "P" in data

    def test_no_pressure_in_l1_group_raises(self, tmp_path):
        """L1_converted with no PRES or PRES_SLOW raises."""
        nc = _write_nc(
            tmp_path / "no_pres.nc",
            with_root_vars=False,
            with_l1_group=True,
            with_pressure=False,
        )
        with pytest.raises(ValueError, match="No pressure variable"):
            _load_from_nc(nc)

    def test_no_pressure_no_l1_raises(self, tmp_path):
        """Root format without P raises."""
        nc = _write_nc(tmp_path / "no_p.nc", with_pressure=False)
        with pytest.raises(ValueError, match="No pressure variable"):
            _load_from_nc(nc)

    def test_no_time_variables_raises(self, tmp_path):
        """No t_fast and no L1_converted group → no time vars → raise."""
        nc = _write_nc(tmp_path / "no_t.nc", no_time=True)
        with pytest.raises(ValueError, match="No time variables"):
            _load_from_nc(nc)

    def test_get_nc_attr_default_path(self, tmp_path):
        """When attr missing on both root and L1, default is returned."""
        # Build NC with no fs_slow, fs_fast on either location
        path = tmp_path / "no_attrs.nc"
        ds = netCDF4.Dataset(str(path), "w", format="NETCDF4")
        ds.createDimension("time_fast", 256)
        ds.createDimension("time_slow", 32)
        tf = ds.createVariable("t_fast", "f8", ("time_fast",))
        tf[:] = np.arange(256) / 512.0
        ts = ds.createVariable("t_slow", "f8", ("time_slow",))
        ts[:] = np.arange(32) / 64.0
        p = ds.createVariable("P", "f8", ("time_slow",))
        p[:] = np.linspace(5.0, 50.0, 32)
        ds.close()

        # No fs_fast attribute → should raise AttributeError
        with pytest.raises(AttributeError, match="fs_fast"):
            _load_from_nc(path)

    def test_l1_group_attrs_used_when_root_missing(self, tmp_path):
        """fs_fast missing from root but on L1_converted → use group attr."""
        path = tmp_path / "l1attrs.nc"
        ds = netCDF4.Dataset(str(path), "w", format="NETCDF4")
        # Root has no attrs
        g = ds.createGroup("L1_converted")
        g.fs_fast = 512.0
        g.fs_slow = 64.0
        g.createDimension("TIME", 256)
        g.createDimension("TIME_SLOW", 32)
        tg = g.createVariable("TIME", "f8", ("TIME",))
        tg[:] = np.arange(256) / 512.0
        tsg = g.createVariable("TIME_SLOW", "f8", ("TIME_SLOW",))
        tsg[:] = np.arange(32) / 64.0
        pg = g.createVariable("PRES_SLOW", "f8", ("TIME_SLOW",))
        pg[:] = np.linspace(5.0, 50.0, 32)
        ds.close()

        data = _load_from_nc(path)
        assert data["fs_fast"] == 512.0
        assert data["fs_slow"] == 64.0

    def test_days_units_converted_to_seconds(self, tmp_path):
        """Time in days → converted to seconds for consistency."""
        nc = _write_nc(tmp_path / "days.nc", t_units_days=True)
        data = _load_from_nc(nc)
        assert data["t_fast_units"] == "seconds"
        # Time should be small (under one day in seconds = 86400)
        assert data["t_fast"][-1] < 1.0  # 256/512Hz is small

    def test_excludes_configuration_string(self, tmp_path):
        """configuration_string handled separately to avoid double copy."""
        nc = _write_nc(
            tmp_path / "with_config.nc",
            with_extra_attr={"configuration_string": "[root]\nfoo=bar"},
        )
        data = _load_from_nc(nc)
        assert data["global_attrs"]["configuration_string"] == "[root]\nfoo=bar"


# ---------------------------------------------------------------------------
# extract_profiles — log warning when no profiles found
# ---------------------------------------------------------------------------


class TestExtractProfilesNoProfile:
    def test_logs_warning_when_no_profiles(self, tmp_path, caplog):
        """A flat-pressure file produces no profiles → empty output."""
        # Build NC with flat pressure → get_profiles returns []
        nc = _write_nc(tmp_path / "flat.nc")
        # Overwrite pressure to be flat
        ds = netCDF4.Dataset(str(nc), "a")
        p = ds.variables["P"]
        p[:] = np.full(len(p), 5.0)
        ds.close()

        out_dir = tmp_path / "out"
        with caplog.at_level("WARNING"):
            paths = extract_profiles(nc, out_dir)
        assert paths == []


# ---------------------------------------------------------------------------
# _extract_one worker (called from parallel path)
# ---------------------------------------------------------------------------


class TestExtractOneWorker:
    def test_extract_one_returns_path_count(self, tmp_path):
        """Worker returns (path_str, count) tuple."""
        nc = _write_nc(tmp_path / "src.nc")
        # Flatten so no profiles found → count=0
        ds = netCDF4.Dataset(str(nc), "a")
        p = ds.variables["P"]
        p[:] = np.full(len(p), 5.0)
        ds.close()

        out_dir = tmp_path / "out"
        result = _extract_one((nc, out_dir, {}))
        path_str, count = result
        assert path_str == str(nc)
        assert count == 0
