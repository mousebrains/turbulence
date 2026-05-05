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
        - no_vib : bool — omit both ACC and VIB (covers vib_type=NONE)
        - no_diss_length : bool — omit diss_length attr from L3
        - pres_on_slow_grid : bool — write PRES on a slower TIME_CTD grid
        - pres_dim_no_time_var : bool — slow PRES with no matching time var
        - pspd_on_slow_grid : bool — write PSPD_REL on a slower TIME_CTD grid
        - pspd_dim_no_time_var : bool — slow PSPD_REL with no matching time var
        - l2_pspd_on_slow_grid : bool — same on the L2 group
        - l2_pspd_dim_no_time_var : bool — slow PSPD_REL on L2 with no time var
        - slow_optional_vars : bool — add TIME_SLOW/PRES_SLOW/PITCH/ROLL/TEMP
        - l3_spectral_attrs : bool — use spectral_* attrs (Epsifish-style)
        - l3_overlap_percent : bool — overlap as percent (with spectral_*)
        - l3_no_attrs : bool — omit all fft/diss/overlap attrs (KCYC fallback)
        - l3_no_fft_kcyc : bool — also omit KCYC (default fallback fft=512)
        - l3_temp : bool — include TEMP in L3_spectra
        - l3_no_sh_spec_clean : bool — omit SH_SPEC_CLEAN from L3_spectra
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
    no_vib = overrides.get("no_vib", False)
    no_diss_length = overrides.get("no_diss_length", False)
    pres_on_slow_grid = overrides.get("pres_on_slow_grid", False)
    pres_dim_no_time_var = overrides.get("pres_dim_no_time_var", False)
    pspd_on_slow_grid = overrides.get("pspd_on_slow_grid", False)
    pspd_dim_no_time_var = overrides.get("pspd_dim_no_time_var", False)
    l2_pspd_on_slow_grid = overrides.get("l2_pspd_on_slow_grid", False)
    l2_pspd_dim_no_time_var = overrides.get("l2_pspd_dim_no_time_var", False)
    slow_optional_vars = overrides.get("slow_optional_vars", False)
    l3_spectral_attrs = overrides.get("l3_spectral_attrs", False)
    l3_overlap_percent = overrides.get("l3_overlap_percent", False)
    l3_no_attrs = overrides.get("l3_no_attrs", False)
    l3_no_fft_kcyc = overrides.get("l3_no_fft_kcyc", False)
    l3_temp = overrides.get("l3_temp", False)
    l3_no_sh_spec_clean = overrides.get("l3_no_sh_spec_clean", False)

    N_SLOW = N_TIME // 4  # used by *_on_slow_grid options

    ds = netCDF4.Dataset(str(path), "w", format="NETCDF4")
    try:
        # ---- L1_converted ----
        g1 = ds.createGroup("L1_converted")
        g1.createDimension("N_TIME", N_TIME)
        g1.createDimension("N_SHEAR", N_SHEAR)
        g1.createDimension("N_VIB", N_VIB)
        if pres_on_slow_grid or pspd_on_slow_grid:
            g1.createDimension("N_TIME_CTD", N_SLOW)

        t = g1.createVariable("TIME", "f8", ("N_TIME",))
        t[:] = np.linspace(0, N_TIME / fs_fast / 86400, N_TIME)

        if pres_on_slow_grid:
            p = g1.createVariable("PRES", "f8", ("N_TIME_CTD",))
            p[:] = np.linspace(5, 100, N_SLOW)
            if not pres_dim_no_time_var:
                tp = g1.createVariable("N_TIME_CTD", "f8", ("N_TIME_CTD",))
                tp[:] = np.linspace(0, N_TIME / fs_fast / 86400, N_SLOW)
        else:
            p = g1.createVariable("PRES", "f8", ("N_TIME",))
            p[:] = np.linspace(5, 100, N_TIME)

        sh = g1.createVariable("SHEAR", "f8", ("N_SHEAR", "N_TIME"))
        sh[:] = np.random.default_rng(42).standard_normal((N_SHEAR, N_TIME)) * 0.01

        if not no_vib:
            vib = g1.createVariable(vib_var, "f8", ("N_VIB", "N_TIME"))
            vib[:] = np.random.default_rng(43).standard_normal((N_VIB, N_TIME)) * 0.001

        if not no_l1_speed:
            if pspd_on_slow_grid:
                spd = g1.createVariable("PSPD_REL", "f8", ("N_TIME_CTD",))
                spd[:] = np.full(N_SLOW, 0.7)
                # Only write the time-coord variable if not testing the
                # "no matching time variable" fallback
                if not pres_on_slow_grid and not pspd_dim_no_time_var:
                    tp = g1.createVariable("N_TIME_CTD", "f8", ("N_TIME_CTD",))
                    tp[:] = np.linspace(0, N_TIME / fs_fast / 86400, N_SLOW)
                elif pspd_dim_no_time_var and "N_TIME_CTD" in g1.variables:
                    # The time var exists from the pres path; can't both
                    # exclude and include.  Skip gracefully.
                    pass
            else:
                spd = g1.createVariable("PSPD_REL", "f8", ("N_TIME",))
                spd[:] = np.full(N_TIME, 0.7)

        if slow_optional_vars:
            g1.createDimension("N_TIME_SLOW", N_SLOW)
            tslow = g1.createVariable("TIME_SLOW", "f8", ("N_TIME_SLOW",))
            tslow[:] = np.linspace(0, N_TIME / fs_fast / 86400, N_SLOW)
            pslow = g1.createVariable("PRES_SLOW", "f8", ("N_TIME_SLOW",))
            pslow[:] = np.linspace(5, 100, N_SLOW)
            pitch = g1.createVariable("PITCH", "f8", ("N_TIME_SLOW",))
            pitch[:] = np.zeros(N_SLOW)
            roll = g1.createVariable("ROLL", "f8", ("N_TIME_SLOW",))
            roll[:] = np.zeros(N_SLOW)
            temp = g1.createVariable("TEMP", "f8", ("N_TIME_SLOW",))
            temp[:] = np.full(N_SLOW, 10.0)

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
        if l2_pspd_on_slow_grid:
            g2.createDimension("N_TIME_CTD", N_SLOW)

        t2 = g2.createVariable("TIME", "f8", ("N_TIME",))
        t2[:] = np.linspace(0, N_TIME / fs_fast / 86400, N_TIME)

        sh2 = g2.createVariable("SHEAR", "f8", ("N_SHEAR", "N_TIME"))
        sh2[:] = np.random.default_rng(44).standard_normal((N_SHEAR, N_TIME)) * 0.01

        if not no_vib:
            vib2 = g2.createVariable(vib_var, "f8", ("N_VIB", "N_TIME"))
            vib2[:] = np.random.default_rng(45).standard_normal((N_VIB, N_TIME)) * 0.001

        if l2_pspd_on_slow_grid:
            spd2 = g2.createVariable("PSPD_REL", "f8", ("N_TIME_CTD",))
            spd2[:] = np.full(N_SLOW, 0.7)
            if not l2_pspd_dim_no_time_var:
                tp2 = g2.createVariable("N_TIME_CTD", "f8", ("N_TIME_CTD",))
                tp2[:] = np.linspace(0, N_TIME / fs_fast / 86400, N_SLOW)
        else:
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

        if l3_temp:
            t3v = g3.createVariable("TEMP", "f8", ("N_SPECTRA",))
            t3v[:] = np.full(N_SPECTRA, 10.0)

        if not l3_no_fft_kcyc:
            kc = g3.createVariable("KCYC", "f8", ("N_FREQ", "N_SPECTRA"))
            kc[:] = np.tile(np.linspace(0, 200, N_FREQ)[:, np.newaxis], (1, N_SPECTRA))

        sp3 = g3.createVariable("SH_SPEC", "f8", ("N_SHEAR", "N_FREQ", "N_SPECTRA"))
        rng = np.random.default_rng(46)
        sp3[:] = np.abs(rng.standard_normal((N_SHEAR, N_FREQ, N_SPECTRA))) * 1e-6

        if not l3_no_sh_spec_clean:
            sp3c = g3.createVariable("SH_SPEC_CLEAN", "f8", ("N_SHEAR", "N_FREQ", "N_SPECTRA"))
            sp3c[:] = sp3[:] * 0.9

        if l3_spectral_attrs:
            g3.spectral_fft_length = 256
            if not no_diss_length:
                g3.spectral_disslength = 512
            if l3_overlap_percent:
                g3.spectral_disslength_overlap = 50.0
        elif l3_no_attrs:
            # No fft_length, diss_length, overlap — relies on inference fallbacks.
            pass
        else:
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
