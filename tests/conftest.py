# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Shared test fixtures."""

from pathlib import Path

import pytest

TEST_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def sample_p_file():
    """Path to the trimmed SN479_0006.p test data file."""
    path = TEST_DATA_DIR / "SN479_0006.p"
    if not path.exists():
        pytest.skip("Test data not available")
    return path


@pytest.fixture
def sample_nc_file(tmp_path, sample_p_file):
    """Convert the sample .p file to NetCDF and return the path."""
    from rsi_python.convert import p_to_netcdf

    nc_path = tmp_path / "SN479_0006.nc"
    p_to_netcdf(sample_p_file, nc_path)
    return nc_path
