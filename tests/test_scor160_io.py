# Tests for odas_tpw.scor160.io
"""Unit tests for dataclasses and I/O utilities."""

import numpy as np

from odas_tpw.scor160.io import (
    AtomixData,
    L1Data,
    L2Data,
    L2Params,
    L3Data,
    L3Params,
    L4Data,
    L4Params,
    read_atomix,
)


def _make_l1(n_time=10000, n_shear=2, n_vib=2, fs=512.0):
    """Create a minimal L1Data for testing."""
    return L1Data(
        time=np.linspace(0, n_time / fs / 86400, n_time),
        pres=np.linspace(5, 100, n_time),
        shear=np.zeros((n_shear, n_time)),
        vib=np.zeros((n_vib, n_time)),
        vib_type="ACC",
        fs_fast=fs,
        f_AA=98.0,
        vehicle="vmp",
        profile_dir="down",
        time_reference_year=2024,
    )


class TestL1Data:
    def test_properties(self):
        l1 = _make_l1()
        assert l1.n_shear == 2
        assert l1.n_vib == 2
        assert l1.n_time == 10000

    def test_has_speed_empty(self):
        l1 = _make_l1()
        assert not l1.has_speed

    def test_has_speed_nonempty(self):
        l1 = _make_l1()
        l1.pspd_rel = np.ones(l1.n_time)
        assert l1.has_speed

    def test_default_optional_arrays(self):
        l1 = _make_l1()
        assert l1.time_slow.size == 0
        assert l1.pres_slow.size == 0
        assert l1.pitch.size == 0
        assert l1.roll.size == 0
        assert l1.temp.size == 0
        assert l1.fs_slow == 0.0


class TestL2Params:
    def test_construction(self):
        params = L2Params(
            HP_cut=0.25,
            despike_sh=np.array([8.0, 0.5, 0.04]),
            despike_A=np.array([np.inf, 0.5, 0.04]),
            profile_min_W=0.1,
            profile_min_P=1.0,
            profile_min_duration=10.0,
            speed_tau=1.5,
        )
        assert params.HP_cut == 0.25
        assert params.speed_tau == 1.5


class TestL2Data:
    def test_construction(self):
        n = 1000
        l2 = L2Data(
            time=np.zeros(n),
            shear=np.zeros((2, n)),
            vib=np.zeros((2, n)),
            vib_type="ACC",
            pspd_rel=np.ones(n),
            section_number=np.ones(n),
        )
        assert l2.shear.shape == (2, n)


class TestL3Params:
    def test_construction(self):
        params = L3Params(
            fft_length=256,
            diss_length=2048,
            overlap=1024,
            HP_cut=0.25,
            fs_fast=512.0,
            goodman=True,
        )
        assert params.fft_length == 256
        assert params.goodman is True


class TestL3Data:
    def test_properties(self):
        n_wn, n_spec, n_sh = 129, 20, 2
        l3 = L3Data(
            time=np.zeros(n_spec),
            pres=np.zeros(n_spec),
            temp=np.zeros(n_spec),
            pspd_rel=np.ones(n_spec),
            section_number=np.ones(n_spec),
            kcyc=np.zeros((n_wn, n_spec)),
            sh_spec=np.zeros((n_sh, n_wn, n_spec)),
            sh_spec_clean=np.zeros((n_sh, n_wn, n_spec)),
        )
        assert l3.n_spectra == n_spec
        assert l3.n_wavenumber == n_wn
        assert l3.n_shear == n_sh


class TestL4Params:
    def test_construction(self):
        params = L4Params(
            fft_length=256,
            diss_length=2048,
            overlap=1024,
            fs_fast=512.0,
            fit_order=3,
            f_AA=98.0,
            FOM_limit=1.15,
            variance_resolved_limit=0.5,
        )
        assert params.fit_order == 3


class TestL4Data:
    def test_properties(self):
        n_sh, n_spec = 2, 30
        l4 = L4Data(
            time=np.zeros(n_spec),
            pres=np.zeros(n_spec),
            pspd_rel=np.ones(n_spec),
            section_number=np.ones(n_spec),
            epsi=np.full((n_sh, n_spec), 1e-8),
            epsi_final=np.full(n_spec, 1e-8),
            epsi_flags=np.zeros((n_sh, n_spec)),
            fom=np.ones((n_sh, n_spec)),
            mad=np.full((n_sh, n_spec), 0.1),
            kmax=np.full((n_sh, n_spec), 50.0),
            method=np.zeros((n_sh, n_spec)),
            var_resolved=np.ones((n_sh, n_spec)),
        )
        assert l4.n_spectra == n_spec
        assert l4.n_shear == n_sh


# ---------------------------------------------------------------------------
# read_atomix() tests using synthetic ATOMIX NetCDF fixture
# ---------------------------------------------------------------------------


class TestReadAtomix:
    """Tests for read_atomix() using the atomix_nc_file session fixture."""

    def test_read_atomix_returns_all_levels(self, atomix_nc_file):
        result = read_atomix(atomix_nc_file)
        assert isinstance(result, AtomixData)
        assert result.l1.n_shear == 2
        assert result.l1.n_time == 2048

    def test_l1_fields(self, atomix_nc_file):
        l1 = read_atomix(atomix_nc_file).l1
        assert l1.time.shape == (2048,)
        assert l1.pres.shape == (2048,)
        assert l1.shear.shape == (2, 2048)
        assert l1.vib.shape == (2, 2048)
        assert l1.vib_type == "ACC"
        assert l1.fs_fast == 512.0
        assert l1.f_AA == 98.0
        assert l1.vehicle == "vmp"

    def test_l2_params(self, atomix_nc_file):
        l2p = read_atomix(atomix_nc_file).l2_params
        assert l2p.HP_cut == 0.25
        np.testing.assert_array_equal(l2p.despike_sh, [8.0, 0.5, 0.04])
        assert l2p.speed_tau == 1.5

    def test_l2_ref_fields(self, atomix_nc_file):
        l2 = read_atomix(atomix_nc_file).l2_ref
        assert l2.shear.shape == (2, 2048)
        assert l2.vib.shape == (2, 2048)
        assert l2.pspd_rel.shape == (2048,)
        assert l2.section_number.shape == (2048,)

    def test_l3_params(self, atomix_nc_file):
        l3p = read_atomix(atomix_nc_file).l3_params
        assert l3p.fft_length == 256
        assert l3p.diss_length == 512
        assert l3p.overlap == 256
        assert l3p.goodman is True

    def test_l3_ref_fields(self, atomix_nc_file):
        l3 = read_atomix(atomix_nc_file).l3_ref
        assert l3.n_spectra == 5
        assert l3.n_wavenumber == 129
        assert l3.n_shear == 2
        assert l3.sh_spec_clean is not None
        assert l3.sh_spec_clean.shape == (2, 129, 5)

    def test_l4_ref_fields(self, atomix_nc_file):
        l4 = read_atomix(atomix_nc_file).l4_ref
        assert l4.epsi.shape == (2, 5)
        assert l4.epsi_final.shape == (5,)
        assert l4.fom.shape == (2, 5)

    def test_l1_no_speed_uses_l2(self, tmp_path):
        """When L1 has no PSPD_REL, speed should be copied from L2."""
        from tests.conftest import _create_atomix_nc

        path = tmp_path / "no_speed.nc"
        _create_atomix_nc(path, no_l1_speed=True)
        result = read_atomix(path)
        assert result.l1.has_speed
        np.testing.assert_allclose(result.l1.pspd_rel, 0.7)

    def test_l1_vib_fallback(self, tmp_path):
        """VIB variable used when ACC is absent."""
        from tests.conftest import _create_atomix_nc

        path = tmp_path / "vib.nc"
        _create_atomix_nc(path, vib_var_name="VIB")
        result = read_atomix(path)
        assert result.l1.vib_type == "VIB"
        assert result.l1.n_vib == 2

    def test_infer_diss_length(self, tmp_path):
        """When diss_length attr is missing from L3, it is inferred."""
        from tests.conftest import _create_atomix_nc

        path = tmp_path / "no_diss_len.nc"
        _create_atomix_nc(path, no_diss_length=True)
        result = read_atomix(path)
        # diss_length should be inferred (>= 2 * fft_length = 512)
        assert result.l3_params.diss_length >= 512


# ---------------------------------------------------------------------------
# read_atomix() edge-case branches
# ---------------------------------------------------------------------------


class TestReadAtomixEdgeCases:
    """Cover the alternate-grid, no-vib, and Epsifish-attr paths in io.py."""

    def test_pres_on_slow_grid(self, tmp_path):
        """PRES on a slower TIME_CTD grid is interpolated to the fast grid."""
        from tests.conftest import _create_atomix_nc

        path = tmp_path / "pres_slow.nc"
        _create_atomix_nc(path, pres_on_slow_grid=True)
        result = read_atomix(path)
        assert result.l1.pres.shape == (2048,)
        # Interpolated values should span roughly the same physical range
        assert 5.0 <= result.l1.pres.min() <= 6.0
        assert 99.0 <= result.l1.pres.max() <= 101.0

    def test_pres_on_slow_grid_no_time_var(self, tmp_path):
        """Slow PRES with no matching time var falls back to uniform interp."""
        from tests.conftest import _create_atomix_nc

        path = tmp_path / "pres_slow_no_time.nc"
        _create_atomix_nc(path, pres_on_slow_grid=True, pres_dim_no_time_var=True)
        result = read_atomix(path)
        assert result.l1.pres.shape == (2048,)

    def test_l1_no_vib_returns_none(self, tmp_path):
        """When neither ACC nor VIB is present, vib_type='NONE'."""
        from tests.conftest import _create_atomix_nc

        path = tmp_path / "no_vib.nc"
        _create_atomix_nc(path, no_vib=True)
        result = read_atomix(path)
        assert result.l1.vib_type == "NONE"
        assert result.l1.vib.shape == (0, result.l1.n_time)
        assert result.l2_ref.vib_type == "NONE"
        assert result.l2_ref.vib.shape == (0, 2048)

    def test_pspd_on_slow_grid_with_time_var(self, tmp_path):
        """L1 PSPD_REL on slow grid with matching time var → time-based interp."""
        from tests.conftest import _create_atomix_nc

        path = tmp_path / "pspd_slow.nc"
        _create_atomix_nc(path, pspd_on_slow_grid=True)
        result = read_atomix(path)
        assert result.l1.pspd_rel.shape == (2048,)

    def test_pspd_on_slow_grid_no_time_var(self, tmp_path):
        """L1 PSPD_REL on slow grid with no time var → uniform interp fallback."""
        from tests.conftest import _create_atomix_nc

        path = tmp_path / "pspd_slow_no_time.nc"
        _create_atomix_nc(path, pspd_on_slow_grid=True, pspd_dim_no_time_var=True)
        result = read_atomix(path)
        assert result.l1.pspd_rel.shape == (2048,)

    def test_l2_pspd_on_slow_grid(self, tmp_path):
        """L2 PSPD_REL on slow grid with matching time var → interp."""
        from tests.conftest import _create_atomix_nc

        path = tmp_path / "l2_pspd_slow.nc"
        _create_atomix_nc(path, l2_pspd_on_slow_grid=True)
        result = read_atomix(path)
        assert result.l2_ref.pspd_rel.shape == (2048,)

    def test_l2_pspd_on_slow_grid_no_time_var(self, tmp_path):
        """L2 PSPD_REL on slow grid with no time var → uniform interp fallback."""
        from tests.conftest import _create_atomix_nc

        path = tmp_path / "l2_pspd_slow_no_time.nc"
        _create_atomix_nc(
            path, l2_pspd_on_slow_grid=True, l2_pspd_dim_no_time_var=True
        )
        result = read_atomix(path)
        assert result.l2_ref.pspd_rel.shape == (2048,)

    def test_slow_optional_vars_loaded(self, tmp_path):
        """TIME_SLOW/PRES_SLOW/PITCH/ROLL/TEMP populate the L1 optional fields."""
        from tests.conftest import _create_atomix_nc

        path = tmp_path / "slow_opt.nc"
        _create_atomix_nc(path, slow_optional_vars=True)
        result = read_atomix(path)
        l1 = result.l1
        assert l1.time_slow.size > 0
        assert l1.pres_slow.size > 0
        assert l1.pitch.size > 0
        assert l1.roll.size > 0
        assert l1.temp.size > 0

    def test_l3_spectral_attrs(self, tmp_path):
        """Epsifish-style spectral_fft_length / spectral_disslength attrs."""
        from tests.conftest import _create_atomix_nc

        path = tmp_path / "l3_spectral.nc"
        _create_atomix_nc(path, l3_spectral_attrs=True)
        result = read_atomix(path)
        assert result.l3_params.fft_length == 256
        assert result.l3_params.diss_length == 512

    def test_l3_overlap_percent(self, tmp_path):
        """Epsifish overlap-as-percent attr produces an integer sample overlap."""
        from tests.conftest import _create_atomix_nc

        path = tmp_path / "l3_overlap_pct.nc"
        _create_atomix_nc(
            path, l3_spectral_attrs=True, l3_overlap_percent=True
        )
        result = read_atomix(path)
        # 50% of 512 == 256
        assert result.l3_params.overlap == 256

    def test_l3_no_attrs_uses_kcyc(self, tmp_path):
        """With no fft_length/spectral_fft_length, fft is inferred from KCYC."""
        from tests.conftest import _create_atomix_nc

        path = tmp_path / "l3_no_attrs.nc"
        _create_atomix_nc(path, l3_no_attrs=True)
        result = read_atomix(path)
        # n_freq = 129 → fft_length = (129-1)*2 = 256
        assert result.l3_params.fft_length == 256

    def test_l3_no_kcyc_default_fft(self, tmp_path):
        """With no KCYC and no fft attrs, fft_length defaults to 512.

        Calls _read_l3_params directly because _read_l3 requires KCYC.
        """
        import netCDF4

        from odas_tpw.scor160.io import _read_l3_params
        from tests.conftest import _create_atomix_nc

        path = tmp_path / "l3_no_kcyc.nc"
        _create_atomix_nc(path, l3_no_attrs=True, l3_no_fft_kcyc=True)
        ds = netCDF4.Dataset(str(path), "r")
        try:
            l3p = _read_l3_params(ds)
        finally:
            ds.close()
        assert l3p.fft_length == 512

    def test_l3_temp_variable_loaded(self, tmp_path):
        """If TEMP is present in L3_spectra, _read_l3 picks it up."""
        from tests.conftest import _create_atomix_nc

        path = tmp_path / "l3_temp.nc"
        _create_atomix_nc(path, l3_temp=True)
        result = read_atomix(path)
        np.testing.assert_allclose(result.l3_ref.temp, 10.0)

    def test_l3_no_sh_spec_clean_falls_back_to_sh_spec(self, tmp_path):
        """With no SH_SPEC_CLEAN, sh_spec_clean defaults to a copy of SH_SPEC."""
        from tests.conftest import _create_atomix_nc

        path = tmp_path / "no_clean.nc"
        _create_atomix_nc(path, l3_no_sh_spec_clean=True)
        result = read_atomix(path)
        np.testing.assert_array_equal(
            result.l3_ref.sh_spec_clean, result.l3_ref.sh_spec
        )


# ---------------------------------------------------------------------------
# _infer_diss_length helper
# ---------------------------------------------------------------------------


class TestInferDissLength:
    """Direct unit tests for the _infer_diss_length helper fallbacks."""

    def _make_l3_only(self, path, time=None, sec=None):
        """Build a tiny NC with just an L3_spectra group and the requested vars."""
        import netCDF4

        ds = netCDF4.Dataset(str(path), "w", format="NETCDF4")
        try:
            g = ds.createGroup("L3_spectra")
            if time is not None:
                g.createDimension("N_SPECTRA", len(time))
                t = g.createVariable("TIME", "f8", ("N_SPECTRA",))
                t[:] = time
                if sec is not None:
                    s = g.createVariable("SECTION_NUMBER", "f8", ("N_SPECTRA",))
                    s[:] = sec
        finally:
            ds.close()

    def test_no_time_or_section(self, tmp_path):
        """Missing TIME or SECTION_NUMBER → fallback to fft_length * 4."""
        import netCDF4

        from odas_tpw.scor160.io import _infer_diss_length

        path = tmp_path / "no_vars.nc"
        self._make_l3_only(path)  # no variables at all
        ds = netCDF4.Dataset(str(path), "r")
        try:
            result = _infer_diss_length(ds.groups["L3_spectra"], 256, 512.0)
        finally:
            ds.close()
        assert result == 1024  # 256 * 4

    def test_too_few_samples(self, tmp_path):
        """len(time) < 2 → fallback to fft_length * 4."""
        import netCDF4

        from odas_tpw.scor160.io import _infer_diss_length

        path = tmp_path / "tiny.nc"
        self._make_l3_only(path, time=[0.0], sec=[1.0])
        ds = netCDF4.Dataset(str(path), "r")
        try:
            result = _infer_diss_length(ds.groups["L3_spectra"], 128, 512.0)
        finally:
            ds.close()
        assert result == 512  # 128 * 4

    def test_no_positive_sections(self, tmp_path):
        """All SECTION_NUMBER==0 → no positive sections, fallback to fft*4."""
        import netCDF4

        from odas_tpw.scor160.io import _infer_diss_length

        path = tmp_path / "zero_sec.nc"
        self._make_l3_only(
            path, time=np.linspace(0, 1, 8), sec=np.zeros(8)
        )
        ds = netCDF4.Dataset(str(path), "r")
        try:
            result = _infer_diss_length(ds.groups["L3_spectra"], 64, 512.0)
        finally:
            ds.close()
        assert result == 256  # 64 * 4
