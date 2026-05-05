# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Branch-coverage tests for rsi.dissipation — error paths."""

from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np
import pytest

from odas_tpw.rsi.dissipation import _compute_diss_one, get_diss


def _write_nc(path: Path, *, with_shear=True, with_accel=True) -> Path:
    """Build a minimal NC for get_diss."""
    n_fast = 1024
    n_slow = 128
    fs_fast = 512.0
    fs_slow = 64.0
    rng = np.random.default_rng(0)

    ds = netCDF4.Dataset(str(path), "w", format="NETCDF4")
    try:
        ds.createDimension("time_fast", n_fast)
        ds.createDimension("time_slow", n_slow)
        ds.fs_fast = fs_fast
        ds.fs_slow = fs_slow

        tf = ds.createVariable("t_fast", "f8", ("time_fast",))
        tf[:] = np.arange(n_fast) / fs_fast
        ts = ds.createVariable("t_slow", "f8", ("time_slow",))
        ts[:] = np.arange(n_slow) / fs_slow

        p = ds.createVariable("P", "f8", ("time_slow",))
        p[:] = np.linspace(5.0, 50.0, n_slow)
        t1 = ds.createVariable("T1", "f8", ("time_slow",))
        t1[:] = np.linspace(20.0, 5.0, n_slow)

        if with_shear:
            sh1 = ds.createVariable("sh1", "f8", ("time_fast",))
            sh1[:] = rng.standard_normal(n_fast) * 0.05
        if with_accel:
            ax = ds.createVariable("Ax", "f8", ("time_fast",))
            ax[:] = rng.standard_normal(n_fast) * 0.005
    finally:
        ds.close()
    return path


# ---------------------------------------------------------------------------
# Error paths in get_diss
# ---------------------------------------------------------------------------


class TestGetDissErrorPaths:
    def test_no_shear_raises(self, tmp_path):
        """No shear channels → raise ValueError."""
        nc = _write_nc(tmp_path / "no_shear.nc", with_shear=False)
        with pytest.raises(ValueError, match="No shear channels"):
            get_diss(nc)

    def test_goodman_no_accel_raises(self, tmp_path):
        """goodman=True with no accelerometer → raise ValueError."""
        nc = _write_nc(tmp_path / "no_accel.nc", with_accel=False)
        with pytest.raises(ValueError, match="No accelerometer"):
            get_diss(nc, goodman=True)


# ---------------------------------------------------------------------------
# _compute_diss_one worker
# ---------------------------------------------------------------------------


class TestComputeDissOneWorker:
    def test_worker_returns_path_count(self, tmp_path):
        """Worker returns (source_path_str, len(paths))."""
        nc = _write_nc(tmp_path / "src.nc")
        # Make pressure flat so no profiles → returns []
        ds = netCDF4.Dataset(str(nc), "a")
        p = ds.variables["P"]
        p[:] = np.full(len(p), 5.0)
        ds.close()

        out_dir = tmp_path / "out"
        result = _compute_diss_one((nc, out_dir, {"goodman": False}))
        path_str, count = result
        assert path_str == str(nc)
        assert count == 0
