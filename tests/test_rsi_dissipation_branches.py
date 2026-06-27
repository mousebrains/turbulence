# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Branch-coverage tests for rsi.dissipation — error paths."""

from __future__ import annotations

import warnings
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from odas_tpw.rsi.dissipation import _compute_diss_one, _compute_epsilon, get_diss


def _write_nc(
    path: Path,
    *,
    with_shear=True,
    with_accel=True,
    n_fast=1024,
    n_slow=128,
    temp_nan=False,
) -> Path:
    """Build a minimal NC for get_diss.

    ``n_fast``/``n_slow`` can be enlarged so at least one dissipation window
    is produced; ``temp_nan`` fills the T1 channel with NaN to exercise the
    non-finite window-temperature guard.
    """
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
        t1[:] = np.nan if temp_nan else np.linspace(20.0, 5.0, n_slow)

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


# ---------------------------------------------------------------------------
# Non-finite window-temperature guard in _compute_epsilon
# ---------------------------------------------------------------------------


class TestNanTemperatureGuard:
    def test_nan_temperature_warns_and_recovers(self, tmp_path):
        """All-NaN window temperature: _compute_epsilon must substitute the
        10 degC default (warning) and yield finite epsilon, mirroring
        process_l4.  On the old code visc35(NaN)->NaN produced all-NaN
        epsilon with no warning.
        """
        # Large enough to yield at least one dissipation window.
        nc = _write_nc(
            tmp_path / "nan_temp.nc",
            n_fast=8192,
            n_slow=1024,
            temp_nan=True,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = _compute_epsilon(nc, goodman=False, fft_length=512)

        assert results, "expected at least one dissipation dataset"
        for ds in results:
            eps = ds["epsilon"].values
            # Old code: every epsilon NaN. New code: all finite.
            assert np.all(np.isfinite(eps))
        assert any(
            "Non-finite window temperature" in str(w.message) for w in caught
        )
