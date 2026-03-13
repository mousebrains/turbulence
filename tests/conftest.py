# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Shared test fixtures."""

from pathlib import Path

import pytest

TEST_DATA_DIR = Path(__file__).parent / "data"

# ---------------------------------------------------------------------------
# Session-scoped caches for expensive computations
# ---------------------------------------------------------------------------

_diss_cache = {}
_chi_cache = {}


def _cached_get_diss(p_path, **kwargs):
    key = (Path(p_path).resolve(), frozenset(kwargs.items()))
    if key not in _diss_cache:
        from rsi_python.dissipation import get_diss

        _diss_cache[key] = get_diss(p_path, **kwargs)
    return _diss_cache[key]


def _cached_get_chi(p_path, **kwargs):
    key = (Path(p_path).resolve(), frozenset(kwargs.items()))
    if key not in _chi_cache:
        from rsi_python.chi import get_chi

        _chi_cache[key] = get_chi(p_path, **kwargs)
    return _chi_cache[key]


@pytest.fixture(scope="session")
def cached_get_diss():
    return _cached_get_diss


@pytest.fixture(scope="session")
def cached_get_chi():
    return _cached_get_chi


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
    from rsi_python.convert import p_to_L1

    nc_path = tmp_path / "SN479_0006.nc"
    p_to_L1(sample_p_file, nc_path)
    return nc_path
