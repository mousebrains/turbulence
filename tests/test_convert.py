# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for .p to ATOMIX L1_converted NetCDF conversion."""

from pathlib import Path

import numpy as np
import pytest

TEST_DATA_DIR = Path(__file__).parent / "data"
SAMPLE_FILE = TEST_DATA_DIR / "SN479_0006.p"


@pytest.fixture
def skip_no_data():
    if not SAMPLE_FILE.exists():
        pytest.skip("Test data not available")


def test_p_to_L1_roundtrip(skip_no_data, tmp_path):
    """Convert .p -> L1 .nc, read back, verify ATOMIX group structure."""
    import netCDF4 as nc

    from odas_tpw.rsi.convert import p_to_L1

    out_path = tmp_path / "test_output.nc"
    pf, nc_path = p_to_L1(SAMPLE_FILE, out_path)

    assert nc_path.exists()
    assert nc_path.stat().st_size > 0

    ds = nc.Dataset(str(nc_path), "r")

    # Check root dimensions
    assert "TIME" in ds.dimensions
    assert "TIME_SLOW" in ds.dimensions
    assert len(ds.dimensions["TIME"]) == len(pf.t_fast)
    assert len(ds.dimensions["TIME_SLOW"]) == len(pf.t_slow)

    # Check L1_converted group exists
    assert "L1_converted" in ds.groups
    L1 = ds.groups["L1_converted"]

    # Check group attributes
    assert float(L1.fs_fast) == pytest.approx(pf.fs_fast, rel=1e-3)
    assert float(L1.fs_slow) == pytest.approx(pf.fs_slow, rel=1e-3)
    assert hasattr(L1, "vehicle")
    assert hasattr(L1, "f_AA")
    assert hasattr(L1, "time_reference_year")

    # Check core ATOMIX variables exist in L1 group
    assert "TIME" in L1.variables
    assert "TIME_SLOW" in L1.variables
    assert "PRES" in L1.variables
    assert "PRES_SLOW" in L1.variables

    # Check pressure values roundtrip
    P_slow_nc = L1.variables["PRES_SLOW"][:].data
    P_src = pf.channels.get("P_dP", pf.channels.get("P"))
    np.testing.assert_allclose(P_slow_nc, P_src, rtol=1e-10)

    ds.close()


def test_p_to_L1_default_path(skip_no_data, tmp_path, monkeypatch):
    """p_to_L1 with nc_filepath=None should write next to the .p file."""
    import shutil

    local_p = tmp_path / SAMPLE_FILE.name
    shutil.copy2(SAMPLE_FILE, local_p)

    from odas_tpw.rsi.convert import p_to_L1

    _pf, nc_path = p_to_L1(local_p)
    assert nc_path.exists()
    assert nc_path.suffix == ".nc"
    assert nc_path.parent == tmp_path


def test_p_to_L1_shear_sensors(skip_no_data, tmp_path):
    """SHEAR variable should have sensor-indexed dimension."""
    import netCDF4 as nc

    from odas_tpw.rsi.convert import p_to_L1

    out_path = tmp_path / "test_shear.nc"
    pf, nc_path = p_to_L1(SAMPLE_FILE, out_path)

    ds = nc.Dataset(str(nc_path), "r")
    L1 = ds.groups["L1_converted"]

    # Count shear channels in source
    n_shear = sum(1 for n in pf.channels if pf.channel_info[n]["type"] == "shear")
    if n_shear > 0:
        assert "N_SHEAR_SENSORS" in ds.dimensions
        assert len(ds.dimensions["N_SHEAR_SENSORS"]) == n_shear
        assert "SHEAR" in L1.variables
        shear = L1.variables["SHEAR"]
        assert shear.shape == (n_shear, len(pf.t_fast))
        assert shear.units == "s-1"
        assert shear.standard_name == "sea_water_velocity_shear"

    ds.close()


def test_p_to_L1_gradt(skip_no_data, tmp_path):
    """GRADT should contain temperature gradient in K/m."""
    import netCDF4 as nc

    from odas_tpw.rsi.convert import p_to_L1

    out_path = tmp_path / "test_gradt.nc"
    pf, nc_path = p_to_L1(SAMPLE_FILE, out_path)

    ds = nc.Dataset(str(nc_path), "r")
    L1 = ds.groups["L1_converted"]

    n_gradt = sum(1 for n in pf.channels if pf.channel_info[n]["type"] == "therm" and pf.is_fast(n))
    if n_gradt > 0:
        assert "GRADT" in L1.variables
        gradt = L1.variables["GRADT"]
        assert gradt.units == "degrees_Celsius m-1"
        # Should be finite (no NaN/Inf in the bulk)
        data = gradt[:].data
        assert np.isfinite(data).sum() > 0.9 * data.size

    ds.close()


def test_convert_all_serial(skip_no_data, tmp_path):
    """convert_all with jobs=1 (serial) should convert files."""
    from odas_tpw.rsi.convert import convert_all

    convert_all([SAMPLE_FILE], tmp_path, jobs=1)
    nc_files = list(tmp_path.glob("*.nc"))
    assert len(nc_files) == 1
    assert nc_files[0].stat().st_size > 0


def test_convert_all_parallel(skip_no_data, tmp_path):
    """convert_all with jobs=2 (parallel) should convert files."""
    from odas_tpw.rsi.convert import convert_all

    convert_all([SAMPLE_FILE], tmp_path, jobs=2)
    nc_files = list(tmp_path.glob("*.nc"))
    assert len(nc_files) == 1
    assert nc_files[0].stat().st_size > 0


def test_cf_compliance(skip_no_data, tmp_path):
    """Output NetCDF should be CF-1.13 compliant with ATOMIX structure."""
    import netCDF4 as nc

    from odas_tpw.rsi.convert import p_to_L1

    out_path = tmp_path / "test_cf.nc"
    _pf, nc_path = p_to_L1(SAMPLE_FILE, out_path)
    ds = nc.Dataset(str(nc_path), "r")

    # Required global attributes
    assert ds.Conventions == "CF-1.13, ACDD-1.3"
    assert hasattr(ds, "title")
    assert hasattr(ds, "history")

    L1 = ds.groups["L1_converted"]

    # Time coordinate variables in L1 group
    for tvar_name in ("TIME", "TIME_SLOW"):
        tvar = L1.variables[tvar_name]
        assert tvar.standard_name == "time"
        assert tvar.axis == "T"
        assert "Days since" in tvar.units

    # Pressure variable CF attributes
    P_var = L1.variables["PRES"]
    assert P_var.standard_name == "sea_water_pressure"

    ds.close()


def test_backward_compat_alias(skip_no_data, tmp_path):
    """p_to_netcdf should still work as an alias for p_to_L1."""
    from odas_tpw.rsi.convert import p_to_L1, p_to_netcdf

    assert p_to_netcdf is p_to_L1

    out_path = tmp_path / "test_alias.nc"
    _pf, nc_path = p_to_netcdf(SAMPLE_FILE, out_path)
    assert nc_path.exists()
