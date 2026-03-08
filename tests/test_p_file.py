# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Basic tests for PFile reader using trimmed test data."""

from pathlib import Path

import numpy as np
import pytest

from rsi_python.p_file import PFile

TEST_DATA_DIR = Path(__file__).parent / "data"
SAMPLE_FILE = TEST_DATA_DIR / "SN479_0006.p"


@pytest.fixture
def pf():
    if not SAMPLE_FILE.exists():
        pytest.skip("Test data not available")
    return PFile(SAMPLE_FILE)


def test_header(pf):
    assert pf.header["header_size"] == 128
    assert pf.header["year"] == 2025
    assert pf.endian == ">"


def test_matrix_shape(pf):
    assert pf.n_rows == 8
    assert pf.n_cols == 10
    assert pf.fast_cols == 8
    assert pf.slow_cols == 2
    assert pf.matrix.shape == (8, 10)


def test_sampling_rates(pf):
    assert abs(pf.fs_fast - 512.0) < 1.0
    assert abs(pf.fs_slow - 64.0) < 1.0


def test_channels_present(pf):
    expected_fast = {"Ax", "Ay", "sh1", "sh2", "T1_dT1", "T2_dT2", "Chlorophyll", "Turbidity"}
    expected_slow = {"P", "T1", "T2", "V_Bat", "Incl_X", "Incl_Y", "Incl_T", "JAC_C", "JAC_T"}
    assert expected_fast.issubset(pf._fast_channels)
    assert expected_slow.issubset(pf._slow_channels)


def test_pressure_reasonable(pf):
    P = pf.channels["P"]
    assert np.nanmax(P) < 500.0


def test_temperature_reasonable(pf):
    T = pf.channels["T1"]
    assert 10.0 < np.nanmean(T) < 35.0


def test_shear_centered(pf):
    sh = pf.channels["sh1"]
    assert abs(np.nanmean(sh)) < 1.0
