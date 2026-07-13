# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for rsi.profile — _load_source error paths, NC reading edge cases."""

from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np
import pytest

from odas_tpw.rsi.profile import _extract_one, _load_from_nc, _load_source, extract_profiles


def _write_nc(
    path: Path,
    *,
    with_l1_group=False,
    with_root_vars=True,
    with_pressure=True,
    with_pres_slow=False,
    no_time=False,
    with_extra_attr=None,
    t_units_days=False,
) -> Path:
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


# ---------------------------------------------------------------------------
# extract_profiles — e_fast must be clamped to the fast-axis length (#6)
# ---------------------------------------------------------------------------


class TestExtractProfilesFastClamp:
    def _write_short_fast_nc(
        self, path: Path, n_slow: int, n_fast: int, fs_fast: float, fs_slow: float
    ) -> Path:
        """NC whose fast axis is short of n_slow*ratio, forcing the final
        profile's (e_slow+1)*ratio to over-run len(t_fast)."""
        ds = netCDF4.Dataset(str(path), "w", format="NETCDF4")
        try:
            ds.fs_fast = fs_fast
            ds.fs_slow = fs_slow
            ds.createDimension("time_fast", n_fast)
            ds.createDimension("time_slow", n_slow)
            tf = ds.createVariable("t_fast", "f8", ("time_fast",))
            tf[:] = np.arange(n_fast) / fs_fast
            tf.units = "seconds"
            ts = ds.createVariable("t_slow", "f8", ("time_slow",))
            ts[:] = np.arange(n_slow) / fs_slow
            ts.units = "seconds"
            p = ds.createVariable("P", "f8", ("time_slow",))
            p[:] = np.linspace(5.0, 50.0, n_slow)
            p.units = "dbar"
            sh1 = ds.createVariable("sh1", "f8", ("time_fast",))
            sh1[:] = np.random.default_rng(0).standard_normal(n_fast) * 0.05
            sh1.units = "s-1"
        finally:
            ds.close()
        return path

    def test_final_profile_clamps_to_fast_length(self, tmp_path):
        """Passing an explicit full-record profile whose (e_slow+1)*ratio
        exceeds len(t_fast): without the clamp the declared time_fast dim
        (256) mismatches the clamped slice (250) and netCDF raises. The clamp
        sizes the dim to the available fast samples."""
        n_slow, fs_fast, fs_slow = 32, 512.0, 64.0  # ratio = 8
        n_fast = 250  # 6 short of n_slow*ratio = 256
        nc = self._write_short_fast_nc(tmp_path / "short.nc", n_slow, n_fast, fs_fast, fs_slow)
        out_dir = tmp_path / "out"
        # Explicit final profile reaching the last slow index (e_slow = 31):
        # (31+1)*8 = 256 > 250.
        paths = extract_profiles(nc, out_dir, profiles=[(0, n_slow - 1)])
        assert len(paths) == 1
        prof = netCDF4.Dataset(str(paths[0]))
        try:
            # Dim clamped to the fast samples actually available from s_fast=0.
            assert prof.dimensions["time_fast"].size == n_fast
            assert prof.variables["t_fast"].shape[0] == n_fast
            assert prof.variables["sh1"].shape[0] == n_fast
        finally:
            prof.close()


class TestExtractProfilesAtomicWrite:
    """Per-profile writes go temp + os.replace, so a mid-write drop never leaves
    a readable partial *_profNNN.nc at the live path (#104 U5-2)."""

    def test_success_leaves_no_tmp(self, tmp_path):
        nc = _write_nc(tmp_path / "src.nc")
        out_dir = tmp_path / "out"
        paths = extract_profiles(nc, out_dir, profiles=[(0, 31)])
        assert len(paths) == 1 and Path(paths[0]).exists()
        assert not list(out_dir.glob(".*.tmp"))  # temp consumed by os.replace

    def test_publish_failure_leaves_no_partial_at_live_path(self, tmp_path, monkeypatch):
        import odas_tpw.rsi.profile as prof_mod

        nc = _write_nc(tmp_path / "src.nc")
        out_dir = tmp_path / "out"

        def boom(src, dst):
            raise OSError("network drop during publish")

        # os.replace is the ONLY step that writes the live product name, so a
        # failed publish must leave NO *_profNNN.nc (only the hidden temp).
        monkeypatch.setattr(prof_mod.os, "replace", boom)
        with pytest.raises(OSError):
            extract_profiles(nc, out_dir, profiles=[(0, 31)])
        assert not list(out_dir.glob("*_prof*.nc"))


# ---------------------------------------------------------------------------
# Audit M2 downstream: a single mid-cast _FillValue in slow pressure must not
# silently drop every profile in the file (NaN smearing through fall rate).
# ---------------------------------------------------------------------------


class TestMidCastFillDoesNotDropProfiles:
    @staticmethod
    def _write_cast_nc(path: Path, *, fill_idx=None, fill_slice=None) -> Path:
        """A detectable down-cast (~1 dbar/s for 16 s), optionally with one
        unwritten slow-pressure element or a contiguous unwritten span
        (-> masked/_FillValue -> NaN)."""
        fs_fast, fs_slow = 512.0, 64.0
        n_slow = 1024
        n_fast = n_slow * 8
        lo, hi = fill_slice if fill_slice is not None else (None, None)
        ds = netCDF4.Dataset(str(path), "w")
        try:
            ds.fs_fast = fs_fast
            ds.fs_slow = fs_slow
            ds.createDimension("time_fast", n_fast)
            ds.createDimension("time_slow", n_slow)
            tf = ds.createVariable("t_fast", "f8", ("time_fast",))
            tf[:] = np.arange(n_fast) / fs_fast
            tf.units = "seconds"
            ts = ds.createVariable("t_slow", "f8", ("time_slow",))
            ts[:] = np.arange(n_slow) / fs_slow
            ts.units = "seconds"
            pv = 1.0 + np.arange(n_slow) / fs_slow * 1.0  # descend 1 -> 17 dbar
            p = ds.createVariable("P", "f8", ("time_slow",), fill_value=9.969209968386869e36)
            p.units = "dbar"
            for i in range(n_slow):
                in_slice = lo is not None and lo <= i < hi
                if i != fill_idx and not in_slice:
                    p[i] = pv[i]  # leave fill_idx / fill_slice unwritten -> NaN
            t1 = ds.createVariable("T1", "f8", ("time_fast",))
            t1[:] = np.linspace(10.0, 11.0, n_fast)
            t1.units = "degC"
        finally:
            ds.close()
        return path

    def test_clean_cast_finds_one_profile(self, tmp_path):
        nc = self._write_cast_nc(tmp_path / "clean.nc")
        paths = extract_profiles(nc, tmp_path / "out_clean")
        assert len(paths) == 1

    def test_single_midcast_fill_still_finds_profile(self, tmp_path):
        """Regression: one mid-cast _FillValue used to NaN the whole fall rate
        (gradient + zero-phase filtfilt smear), yielding zero profiles for the
        entire file. The NaN repair keeps the cast usable."""
        nc = self._write_cast_nc(tmp_path / "fill.nc", fill_idx=500)
        paths = extract_profiles(nc, tmp_path / "out_fill")
        assert len(paths) == 1, "mid-cast fill silently dropped all profiles"

    def test_long_gap_profile_dropped(self, tmp_path, caplog):
        """PR #79 review: a long contiguous fill (> max_repair_gap_s) fabricates
        pressure; the profile overlapping it must be dropped, not detected over
        synthetic data. fs_slow=64, default 1 s -> max_gap=64 samples; a 200-
        sample gap is fabricated."""
        nc = self._write_cast_nc(tmp_path / "biggap.nc", fill_slice=(400, 600))
        with caplog.at_level("WARNING"):
            paths = extract_profiles(nc, tmp_path / "out_biggap")
        assert len(paths) == 0
        assert any("fabricated pressure gap" in r.message for r in caplog.records)

    def test_subthreshold_gap_still_repaired(self, tmp_path):
        """A contiguous fill shorter than max_gap (64) is an isolated fill: it is
        repaired and the profile is kept."""
        nc = self._write_cast_nc(tmp_path / "smallgap.nc", fill_slice=(480, 520))
        paths = extract_profiles(nc, tmp_path / "out_smallgap")
        assert len(paths) == 1

    def test_long_gap_can_be_repaired_with_larger_threshold(self, tmp_path):
        """The cap is configurable: raising max_repair_gap_s readmits the cast."""
        nc = self._write_cast_nc(tmp_path / "cfggap.nc", fill_slice=(400, 600))
        paths = extract_profiles(nc, tmp_path / "out_cfggap", max_repair_gap_s=10.0)
        assert len(paths) == 1


# ---------------------------------------------------------------------------
# _repair_nans — short gaps repaired, long gaps flagged (PR #79 review)
# ---------------------------------------------------------------------------


class TestRepairNans:
    def test_short_repaired_long_flagged(self):
        from odas_tpw.rsi.profile import _repair_nans

        x = np.arange(100, dtype=np.float64)
        x[10:13] = np.nan  # 3-sample run (short)
        x[40:60] = np.nan  # 20-sample run (long, > max_gap=5)
        repaired, long_gap = _repair_nans(x, "P_slow", "stem", max_gap=5)
        # Every NaN is interpolated so the fall-rate filter has a clean array.
        assert np.isfinite(repaired).all()
        # Short run interpolated back to the true ramp; not flagged.
        np.testing.assert_allclose(repaired[10:13], [10.0, 11.0, 12.0])
        assert not long_gap[10:13].any()
        # Long run flagged exactly over its span, nothing else.
        assert long_gap[40:60].all()
        assert long_gap.sum() == 20

    def test_no_nan_returns_clean_mask(self):
        from odas_tpw.rsi.profile import _repair_nans

        x = np.linspace(0.0, 1.0, 50)
        repaired, long_gap = _repair_nans(x, "P_slow", "stem", max_gap=5)
        assert repaired is x
        assert not long_gap.any()

    def test_gap_exactly_at_threshold_not_flagged(self):
        from odas_tpw.rsi.profile import _repair_nans

        x = np.arange(50, dtype=np.float64)
        x[20:25] = np.nan  # exactly 5 samples; not > max_gap=5
        _, long_gap = _repair_nans(x, "P_slow", "stem", max_gap=5)
        assert not long_gap.any()


# ---------------------------------------------------------------------------
# Audit io-hazard: _load_from_nc must close the Dataset on every exit path.
# ---------------------------------------------------------------------------


class TestLoadFromNcClosesOnError:
    def test_handle_closed_on_error_path(self, tmp_path, monkeypatch):
        """A NetCDF with no time variables raises ValueError; the open Dataset
        must still be closed (handle not leaked)."""
        # NC with attrs but no time/pressure variables -> raises in _load_from_nc.
        path = tmp_path / "notime.nc"
        ds0 = netCDF4.Dataset(str(path), "w")
        ds0.fs_fast = 512.0
        ds0.fs_slow = 64.0
        ds0.close()

        opened = []
        real_dataset = netCDF4.Dataset

        def _tracking_dataset(*args, **kwargs):
            ds = real_dataset(*args, **kwargs)
            opened.append(ds)
            return ds

        # _load_from_nc does `import netCDF4 as nc`, so patch the module symbol.
        monkeypatch.setattr("netCDF4.Dataset", _tracking_dataset)

        with pytest.raises(ValueError, match="No time variables"):
            _load_from_nc(path)

        assert opened, "expected the Dataset to have been opened"
        assert not opened[-1].isopen(), "Dataset handle leaked on error path"
