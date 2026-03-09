# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for .p to NetCDF conversion."""

from pathlib import Path

import numpy as np
import pytest

TEST_DATA_DIR = Path(__file__).parent / "data"
SAMPLE_FILE = TEST_DATA_DIR / "SN479_0006.p"


@pytest.fixture
def skip_no_data():
    if not SAMPLE_FILE.exists():
        pytest.skip("Test data not available")


def test_p_to_netcdf_roundtrip(skip_no_data, tmp_path):
    """Convert .p → .nc, read back, verify channels/shapes/attrs."""
    import netCDF4 as nc

    from rsi_python.convert import p_to_netcdf

    out_path = tmp_path / "test_output.nc"
    pf, nc_path = p_to_netcdf(SAMPLE_FILE, out_path)

    assert nc_path.exists()
    assert nc_path.stat().st_size > 0

    ds = nc.Dataset(str(nc_path), "r")

    # Check global attributes
    assert hasattr(ds, "fs_fast")
    assert hasattr(ds, "fs_slow")
    assert float(ds.fs_fast) == pytest.approx(pf.fs_fast, rel=1e-3)

    # Check time dimensions
    assert "time_fast" in ds.dimensions
    assert "time_slow" in ds.dimensions
    assert len(ds.dimensions["time_fast"]) == len(pf.t_fast)
    assert len(ds.dimensions["time_slow"]) == len(pf.t_slow)

    # Check a channel roundtrips
    assert "P" in ds.variables
    P_nc = ds.variables["P"][:].data
    np.testing.assert_allclose(P_nc, pf.channels["P"].astype(np.float32), rtol=1e-5)

    ds.close()


def test_cf_compliance(skip_no_data, tmp_path):
    """Output NetCDF should be CF-1.13 compliant."""
    import netCDF4 as nc

    from rsi_python.convert import p_to_netcdf

    out_path = tmp_path / "test_cf.nc"
    pf, nc_path = p_to_netcdf(SAMPLE_FILE, out_path)
    ds = nc.Dataset(str(nc_path), "r")

    # Required global attributes
    assert ds.Conventions == "CF-1.13"
    assert hasattr(ds, "title")
    assert hasattr(ds, "history")

    # Time coordinate variables
    for tvar_name in ("t_fast", "t_slow"):
        tvar = ds.variables[tvar_name]
        assert tvar.standard_name == "time"
        assert tvar.calendar == "standard"
        assert tvar.axis == "T"
        assert "seconds since" in tvar.units

    # Pressure variable CF attributes
    P_var = ds.variables["P"]
    assert P_var.standard_name == "sea_water_pressure"
    assert P_var.positive == "down"

    # All data variables should have units and long_name
    for vname in ds.variables:
        if vname in ("t_fast", "t_slow"):
            continue
        var = ds.variables[vname]
        assert hasattr(var, "units"), f"{vname} missing units"
        assert hasattr(var, "long_name"), f"{vname} missing long_name"

    ds.close()
