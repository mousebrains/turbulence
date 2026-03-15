# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Shared test fixtures."""

from pathlib import Path

import numpy as np
import pytest

TEST_DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# ATOMIX NetCDF fixture
# ---------------------------------------------------------------------------


def _create_atomix_nc(path, **overrides):
    """Create a minimal ATOMIX-format NetCDF4 file with 4 groups.

    Parameters
    ----------
    path : Path
        Output file path.
    **overrides : dict
        Override defaults. Supported keys:
        - n_time, n_shear, n_vib, n_spectra, n_freq, fs_fast
        - no_l1_speed : bool — omit PSPD_REL from L1_converted
        - vib_var_name : str — use "VIB" instead of "ACC" in L1/L2
        - no_diss_length : bool — omit diss_length attr from L3
    """
    import netCDF4

    N_TIME = overrides.get("n_time", 2048)
    N_SHEAR = overrides.get("n_shear", 2)
    N_VIB = overrides.get("n_vib", 2)
    N_SPECTRA = overrides.get("n_spectra", 5)
    N_FREQ = overrides.get("n_freq", 129)
    fs_fast = overrides.get("fs_fast", 512.0)

    vib_var = overrides.get("vib_var_name", "ACC")
    no_l1_speed = overrides.get("no_l1_speed", False)
    no_diss_length = overrides.get("no_diss_length", False)

    ds = netCDF4.Dataset(str(path), "w", format="NETCDF4")
    try:
        # ---- L1_converted ----
        g1 = ds.createGroup("L1_converted")
        g1.createDimension("N_TIME", N_TIME)
        g1.createDimension("N_SHEAR", N_SHEAR)
        g1.createDimension("N_VIB", N_VIB)

        t = g1.createVariable("TIME", "f8", ("N_TIME",))
        t[:] = np.linspace(0, N_TIME / fs_fast / 86400, N_TIME)

        p = g1.createVariable("PRES", "f8", ("N_TIME",))
        p[:] = np.linspace(5, 100, N_TIME)

        sh = g1.createVariable("SHEAR", "f8", ("N_SHEAR", "N_TIME"))
        sh[:] = np.random.default_rng(42).standard_normal((N_SHEAR, N_TIME)) * 0.01

        vib = g1.createVariable(vib_var, "f8", ("N_VIB", "N_TIME"))
        vib[:] = np.random.default_rng(43).standard_normal((N_VIB, N_TIME)) * 0.001

        if not no_l1_speed:
            spd = g1.createVariable("PSPD_REL", "f8", ("N_TIME",))
            spd[:] = np.full(N_TIME, 0.7)

        g1.fs_fast = fs_fast
        g1.f_AA = 98.0
        g1.vehicle = "vmp"
        g1.profile_dir = "down"
        g1.time_reference_year = 2025

        # ---- L2_cleaned ----
        g2 = ds.createGroup("L2_cleaned")
        g2.createDimension("N_TIME", N_TIME)
        g2.createDimension("N_SHEAR", N_SHEAR)
        g2.createDimension("N_VIB", N_VIB)

        t2 = g2.createVariable("TIME", "f8", ("N_TIME",))
        t2[:] = np.linspace(0, N_TIME / fs_fast / 86400, N_TIME)

        sh2 = g2.createVariable("SHEAR", "f8", ("N_SHEAR", "N_TIME"))
        sh2[:] = np.random.default_rng(44).standard_normal((N_SHEAR, N_TIME)) * 0.01

        vib2 = g2.createVariable(vib_var, "f8", ("N_VIB", "N_TIME"))
        vib2[:] = np.random.default_rng(45).standard_normal((N_VIB, N_TIME)) * 0.001

        spd2 = g2.createVariable("PSPD_REL", "f8", ("N_TIME",))
        spd2[:] = np.full(N_TIME, 0.7)

        sec2 = g2.createVariable("SECTION_NUMBER", "f8", ("N_TIME",))
        sec2[:] = np.ones(N_TIME)

        g2.HP_cut = 0.25
        g2.despike_sh = np.array([8.0, 0.5, 0.04])
        g2.speed_tau = 1.5

        # ---- L3_spectra ----
        g3 = ds.createGroup("L3_spectra")
        g3.createDimension("N_SPECTRA", N_SPECTRA)
        g3.createDimension("N_FREQ", N_FREQ)
        g3.createDimension("N_SHEAR", N_SHEAR)

        t3 = g3.createVariable("TIME", "f8", ("N_SPECTRA",))
        t3[:] = np.linspace(0, 0.01, N_SPECTRA)

        p3 = g3.createVariable("PRES", "f8", ("N_SPECTRA",))
        p3[:] = np.linspace(10, 90, N_SPECTRA)

        spd3 = g3.createVariable("PSPD_REL", "f8", ("N_SPECTRA",))
        spd3[:] = np.full(N_SPECTRA, 0.7)

        sec3 = g3.createVariable("SECTION_NUMBER", "f8", ("N_SPECTRA",))
        sec3[:] = np.ones(N_SPECTRA)

        kc = g3.createVariable("KCYC", "f8", ("N_FREQ", "N_SPECTRA"))
        kc[:] = np.tile(np.linspace(0, 200, N_FREQ)[:, np.newaxis], (1, N_SPECTRA))

        sp3 = g3.createVariable("SH_SPEC", "f8", ("N_SHEAR", "N_FREQ", "N_SPECTRA"))
        rng = np.random.default_rng(46)
        sp3[:] = np.abs(rng.standard_normal((N_SHEAR, N_FREQ, N_SPECTRA))) * 1e-6

        sp3c = g3.createVariable("SH_SPEC_CLEAN", "f8", ("N_SHEAR", "N_FREQ", "N_SPECTRA"))
        sp3c[:] = sp3[:] * 0.9

        g3.fft_length = 256
        if not no_diss_length:
            g3.diss_length = 512
        g3.overlap = 256
        g3.fs_fast = fs_fast
        g3.goodman = 1

        # ---- L4_dissipation ----
        g4 = ds.createGroup("L4_dissipation")
        g4.createDimension("N_SPECTRA", N_SPECTRA)
        g4.createDimension("N_SHEAR", N_SHEAR)

        t4 = g4.createVariable("TIME", "f8", ("N_SPECTRA",))
        t4[:] = np.linspace(0, 0.01, N_SPECTRA)

        p4 = g4.createVariable("PRES", "f8", ("N_SPECTRA",))
        p4[:] = np.linspace(10, 90, N_SPECTRA)

        spd4 = g4.createVariable("PSPD_REL", "f8", ("N_SPECTRA",))
        spd4[:] = np.full(N_SPECTRA, 0.7)

        sec4 = g4.createVariable("SECTION_NUMBER", "f8", ("N_SPECTRA",))
        sec4[:] = np.ones(N_SPECTRA)

        epsi = g4.createVariable("EPSI", "f8", ("N_SHEAR", "N_SPECTRA"))
        epsi[:] = np.full((N_SHEAR, N_SPECTRA), 1e-8)

        ef = g4.createVariable("EPSI_FINAL", "f8", ("N_SPECTRA",))
        ef[:] = np.full(N_SPECTRA, 1e-8)

        efl = g4.createVariable("EPSI_FLAGS", "f8", ("N_SHEAR", "N_SPECTRA"))
        efl[:] = np.zeros((N_SHEAR, N_SPECTRA))

        fom = g4.createVariable("FOM", "f8", ("N_SHEAR", "N_SPECTRA"))
        fom[:] = np.ones((N_SHEAR, N_SPECTRA))

        mad = g4.createVariable("MAD", "f8", ("N_SHEAR", "N_SPECTRA"))
        mad[:] = np.full((N_SHEAR, N_SPECTRA), 0.1)

        km = g4.createVariable("KMAX", "f8", ("N_SHEAR", "N_SPECTRA"))
        km[:] = np.full((N_SHEAR, N_SPECTRA), 50.0)

        meth = g4.createVariable("METHOD", "f8", ("N_SHEAR", "N_SPECTRA"))
        meth[:] = np.zeros((N_SHEAR, N_SPECTRA))

        vr = g4.createVariable("VAR_RESOLVED", "f8", ("N_SHEAR", "N_SPECTRA"))
        vr[:] = np.ones((N_SHEAR, N_SPECTRA))
    finally:
        ds.close()

    return path


@pytest.fixture(scope="session")
def atomix_nc_file(tmp_path_factory):
    """Session-scoped ATOMIX-format NetCDF4 file for scor160 tests."""
    path = tmp_path_factory.mktemp("atomix") / "benchmark.nc"
    return _create_atomix_nc(path)

# ---------------------------------------------------------------------------
# Session-scoped caches for expensive computations
# ---------------------------------------------------------------------------

_diss_cache = {}
_chi_cache = {}


def _cached_get_diss(p_path, **kwargs):
    key = (Path(p_path).resolve(), frozenset(kwargs.items()))
    if key not in _diss_cache:
        from odas_tpw.rsi.dissipation import get_diss

        _diss_cache[key] = get_diss(p_path, **kwargs)
    return _diss_cache[key]


def _cached_get_chi(p_path, **kwargs):
    key = (Path(p_path).resolve(), frozenset(kwargs.items()))
    if key not in _chi_cache:
        from odas_tpw.rsi.chi_io import get_chi

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
    from odas_tpw.rsi.convert import p_to_L1

    nc_path = tmp_path / "SN479_0006.nc"
    p_to_L1(sample_p_file, nc_path)
    return nc_path
