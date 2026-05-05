# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for rsi.adapter.nc_to_l1data — extended branch coverage.

The existing test_adapter.py tests the happy path via the atomix_nc_file
fixture. This file targets the branches missed there:
- PRES with mismatched length → np.interp
- SHEAR absent → empty shear
- VIB instead of ACC, neither → vib_type fallback
- PSPD_REL with mismatched length → np.interp
- TEMP shape variations
- TIME_SLOW / PRES_SLOW present
- root-level (no L1_converted group) layout
"""

from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np

from odas_tpw.rsi.adapter import nc_to_l1data


def _build_nc(
    path: Path,
    *,
    use_l1_group: bool = True,
    n_time: int = 256,
    n_pres: int | None = None,
    include_shear: bool = True,
    vib_var: str | None = "ACC",  # "ACC", "VIB", or None
    n_vib: int = 2,
    include_pspd: bool = True,
    n_pspd: int | None = None,
    include_temp: bool = False,
    temp_shape: str = "fast2d",  # "fast2d", "wrong2d", "1d"
    include_time_slow: bool = False,
    include_pres_slow: bool = False,
    n_slow: int = 32,
    fs_fast: float | None = 512.0,
    extra_attrs: dict | None = None,
) -> Path:
    """Build a configurable NetCDF for nc_to_l1data tests."""
    rng = np.random.default_rng(0)
    n_pres = n_pres if n_pres is not None else n_time
    n_pspd = n_pspd if n_pspd is not None else n_time

    ds = netCDF4.Dataset(str(path), "w", format="NETCDF4")
    try:
        target = ds.createGroup("L1_converted") if use_l1_group else ds

        target.createDimension("TIME", n_time)
        if n_pres != n_time:
            target.createDimension("PRES_DIM", n_pres)
            pres_dim = "PRES_DIM"
        else:
            pres_dim = "TIME"
        if include_pspd and n_pspd != n_time:
            target.createDimension("PSPD_DIM", n_pspd)
            pspd_dim = "PSPD_DIM"
        else:
            pspd_dim = "TIME"

        t = target.createVariable("TIME", "f8", ("TIME",))
        t[:] = np.arange(n_time) / 512.0

        p = target.createVariable("PRES", "f8", (pres_dim,))
        p[:] = np.linspace(5.0, 50.0, n_pres)

        if include_shear:
            target.createDimension("N_SHEAR", 2)
            sh = target.createVariable("SHEAR", "f8", ("N_SHEAR", "TIME"))
            sh[:] = rng.standard_normal((2, n_time)) * 0.05

        if vib_var is not None:
            target.createDimension("N_VIB", n_vib)
            v = target.createVariable(vib_var, "f8", ("N_VIB", "TIME"))
            v[:] = rng.standard_normal((n_vib, n_time)) * 0.005

        if include_pspd:
            spd = target.createVariable("PSPD_REL", "f8", (pspd_dim,))
            spd[:] = np.full(n_pspd, 0.7)

        if include_temp:
            if temp_shape == "fast2d":
                target.createDimension("N_TEMP", 1)
                temp = target.createVariable("TEMP", "f8", ("N_TEMP", "TIME"))
                temp[:] = np.full((1, n_time), 12.0)
            elif temp_shape == "wrong2d":
                target.createDimension("OTHER", 5)
                temp = target.createVariable("TEMP", "f8", ("OTHER",))
                temp[:] = np.linspace(10, 15, 5)
            elif temp_shape == "1d":
                temp = target.createVariable("TEMP", "f8", ("TIME",))
                temp[:] = np.full(n_time, 12.0)

        if include_time_slow:
            target.createDimension("TIME_SLOW", n_slow)
            tslow = target.createVariable("TIME_SLOW", "f8", ("TIME_SLOW",))
            tslow[:] = np.arange(n_slow) / 64.0
            if include_pres_slow:
                pslow = target.createVariable("PRES_SLOW", "f8", ("TIME_SLOW",))
                pslow[:] = np.linspace(5.0, 50.0, n_slow)

        if fs_fast is not None:
            target.fs_fast = fs_fast
        target.fs_slow = 64.0
        target.f_AA = 98.0
        target.vehicle = "vmp"
        target.profile_dir = "down"
        target.time_reference_year = 2025

        if extra_attrs:
            for k, v in extra_attrs.items():
                setattr(target, k, v)
    finally:
        ds.close()
    return path


class TestNcToL1DataPressureInterp:
    def test_pres_mismatched_length_interpolated(self, tmp_path):
        """PRES on a different grid → linear-interpolate to TIME length."""
        nc = _build_nc(tmp_path / "p.nc", n_pres=64)
        l1 = nc_to_l1data(nc)
        assert l1.pres.shape[0] == 256
        # Should still span the original range after interp
        assert 5.0 <= float(l1.pres[0]) <= 50.0


class TestNcToL1DataShearAbsent:
    def test_no_shear_returns_empty(self, tmp_path):
        """SHEAR variable missing → empty (0, N) array."""
        nc = _build_nc(tmp_path / "ns.nc", include_shear=False)
        l1 = nc_to_l1data(nc)
        assert l1.shear.shape == (0, 256)
        assert l1.n_shear == 0


class TestNcToL1DataVibrationFallback:
    def test_vib_var_used_when_no_acc(self, tmp_path):
        """ACC absent, VIB present → vib_type=VIB."""
        nc = _build_nc(tmp_path / "vibvar.nc", vib_var="VIB")
        l1 = nc_to_l1data(nc)
        assert l1.vib_type == "VIB"
        assert l1.vib.shape == (2, 256)

    def test_no_vib_at_all(self, tmp_path):
        """Neither ACC nor VIB → vib_type=NONE."""
        nc = _build_nc(tmp_path / "novib.nc", vib_var=None)
        l1 = nc_to_l1data(nc)
        assert l1.vib_type == "NONE"
        assert l1.vib.shape == (0, 256)


class TestNcToL1DataSpeedInterp:
    def test_pspd_mismatched_length_interpolated(self, tmp_path):
        """PSPD_REL on different grid → linear-interpolate to TIME length."""
        nc = _build_nc(tmp_path / "pspd.nc", n_pspd=64)
        l1 = nc_to_l1data(nc)
        assert l1.pspd_rel.shape[0] == 256

    def test_no_pspd_empty_array(self, tmp_path):
        """No PSPD_REL → empty pspd_rel array."""
        nc = _build_nc(tmp_path / "nopspd.nc", include_pspd=False)
        l1 = nc_to_l1data(nc)
        assert l1.pspd_rel.size == 0


class TestNcToL1DataTemp:
    def test_temp_2d_correct_shape(self, tmp_path):
        """TEMP with shape (n_temp, n_time) → use as temp_fast."""
        nc = _build_nc(tmp_path / "temp.nc", include_temp=True, temp_shape="fast2d")
        l1 = nc_to_l1data(nc)
        assert l1.temp_fast.shape == (1, 256)

    def test_temp_wrong_shape_ignored(self, tmp_path):
        """TEMP with mismatched shape → temp_fast stays empty."""
        nc = _build_nc(tmp_path / "tempwrong.nc", include_temp=True, temp_shape="wrong2d")
        l1 = nc_to_l1data(nc)
        assert l1.temp_fast.size == 0

    def test_no_temp_zero_array(self, tmp_path):
        """No TEMP variable → temp_fast and temp empty."""
        nc = _build_nc(tmp_path / "notemp.nc", include_temp=False)
        l1 = nc_to_l1data(nc)
        assert l1.temp_fast.size == 0


class TestNcToL1DataSlowGrid:
    def test_time_slow_present(self, tmp_path):
        """TIME_SLOW present → time_slow is populated."""
        nc = _build_nc(tmp_path / "slow.nc", include_time_slow=True)
        l1 = nc_to_l1data(nc)
        assert l1.time_slow.shape[0] == 32

    def test_pres_slow_present(self, tmp_path):
        """PRES_SLOW present → pres_slow is populated."""
        nc = _build_nc(
            tmp_path / "presslow.nc",
            include_time_slow=True,
            include_pres_slow=True,
        )
        l1 = nc_to_l1data(nc)
        assert l1.pres_slow.shape[0] == 32


class TestNcToL1DataAttrFallback:
    def test_attr_default_when_missing(self, tmp_path):
        """Missing attrs use defaults from _get_attr fallback."""
        # Build NC without fs_fast → should fall back to default 512.0
        nc = _build_nc(tmp_path / "nofs.nc", fs_fast=None)
        l1 = nc_to_l1data(nc)
        assert l1.fs_fast == 512.0

    def test_attr_on_root_when_missing_on_group(self, tmp_path):
        """Root-level attr is used when group attr missing."""
        # Build a NC with no L1_converted group → attrs are on root
        nc = _build_nc(tmp_path / "rootattrs.nc", use_l1_group=False)
        l1 = nc_to_l1data(nc)
        assert l1.fs_fast == 512.0
