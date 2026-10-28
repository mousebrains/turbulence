# Tests for odas_tpw.scor160.l3
"""Unit tests for L2→L3 spectral processing."""

import numpy as np
import pytest

from odas_tpw.scor160.io import L1Data, L2Data, L3Data, L3Params
from odas_tpw.scor160.l3 import _section_groups, process_l3


def _make_l1_l2(
    n_time=20000,
    n_shear=2,
    n_vib=2,
    fs=512.0,
):
    """Create paired L1 and L2 data for L3 processing tests."""
    rng = np.random.default_rng(42)
    time = np.arange(n_time) / fs / 86400

    l1 = L1Data(
        time=time,
        pres=np.linspace(10, 50, n_time),
        shear=rng.standard_normal((n_shear, n_time)) * 0.05,
        vib=rng.standard_normal((n_vib, n_time)) * 0.01,
        vib_type="ACC",
        fs_fast=fs,
        f_AA=98.0,
        vehicle="vmp",
        profile_dir="down",
        time_reference_year=2024,
        temp=np.full(n_time, 10.0),
    )

    l2 = L2Data(
        time=time,
        shear=rng.standard_normal((n_shear, n_time)) * 0.05,
        vib=rng.standard_normal((n_vib, n_time)) * 0.01,
        vib_type="ACC",
        pspd_rel=np.full(n_time, 0.6),
        section_number=np.ones(n_time),  # all data in section 1
    )

    return l1, l2


def _make_params(fs=512.0):
    return L3Params(
        fft_length=256,
        diss_length=2048,
        overlap=1024,
        HP_cut=0.25,
        fs_fast=fs,
        goodman=True,
    )


class TestProcessL3:
    """Tests for the main process_l3 function."""

    def test_output_type(self):
        l1, l2 = _make_l1_l2()
        params = _make_params()
        l3 = process_l3(l2, l1, params)
        assert isinstance(l3, L3Data)

    def test_output_shapes(self):
        l1, l2 = _make_l1_l2()
        params = _make_params()
        l3 = process_l3(l2, l1, params)

        n_freq = params.fft_length // 2 + 1
        n_spec = l3.n_spectra
        assert n_spec > 0
        assert l3.kcyc.shape == (n_freq, n_spec)
        assert l3.sh_spec.shape == (2, n_freq, n_spec)
        assert l3.sh_spec_clean.shape == (2, n_freq, n_spec)
        assert l3.time.shape == (n_spec,)
        assert l3.pres.shape == (n_spec,)
        assert l3.temp.shape == (n_spec,)
        assert l3.pspd_rel.shape == (n_spec,)
        assert l3.section_number.shape == (n_spec,)

    def test_n_spectra_expected(self):
        """Number of spectra should match expected from windowing."""
        n_time = 20000
        l1, l2 = _make_l1_l2(n_time=n_time)
        params = _make_params()
        l3 = process_l3(l2, l1, params)

        sec_len = n_time
        diss_step = params.diss_length - params.overlap
        expected_n = (sec_len - params.diss_length) // diss_step + 1
        assert l3.n_spectra == expected_n

    def test_wavenumber_grid(self):
        """Wavenumber grid should be f/W where W is the mean speed."""
        l1, l2 = _make_l1_l2()
        params = _make_params()
        l3 = process_l3(l2, l1, params)

        fs = params.fs_fast
        nfft = params.fft_length
        F = np.arange(nfft // 2 + 1) * fs / nfft
        W = l3.pspd_rel[0]
        expected_k = F / W
        np.testing.assert_allclose(l3.kcyc[:, 0], expected_k)

    def test_spectra_positive(self):
        """Shear spectra should be non-negative (auto-spectra)."""
        l1, l2 = _make_l1_l2()
        params = _make_params()
        l3 = process_l3(l2, l1, params)
        assert np.all(l3.sh_spec >= 0)

    def test_section_numbers_propagated(self):
        l1, l2 = _make_l1_l2()
        params = _make_params()
        l3 = process_l3(l2, l1, params)
        assert np.all(l3.section_number == 1)

    def test_pressure_in_range(self):
        l1, l2 = _make_l1_l2()
        params = _make_params()
        l3 = process_l3(l2, l1, params)
        assert np.all(l3.pres >= 10)
        assert np.all(l3.pres <= 50)

    def test_speed_preserved(self):
        l1, l2 = _make_l1_l2()
        params = _make_params()
        l3 = process_l3(l2, l1, params)
        np.testing.assert_allclose(l3.pspd_rel, 0.6, atol=0.01)

    def test_all_nan_speed_window_floors_wavenumber_axis(self):
        """An all-NaN-speed window records NaN speed but its wavenumber axis is
        floored (W=0.05) so its epsilon is not silently NaN'd. max(NaN, 0.05)
        previously left the whole window's kcyc NaN (#59)."""
        l1, l2 = _make_l1_l2()
        params = _make_params()  # diss_length=2048 -> window 0 is samples [0, 2048)
        l2.pspd_rel[:2048] = np.nan
        l3 = process_l3(l2, l1, params)
        assert np.isnan(l3.pspd_rel[0])  # recorded window speed is NaN
        assert np.all(np.isfinite(l3.kcyc[:, 0]))  # but W floored -> finite kcyc
        assert l3.pspd_rel[1] > 0.5  # a later finite window is unaffected

    def test_partial_nan_speed_window_uses_finite_mean(self):
        """A window with some NaN speed samples averages the finite ones rather
        than collapsing to NaN/floor (#59)."""
        l1, l2 = _make_l1_l2()
        params = _make_params()
        l2.pspd_rel[:1000] = np.nan  # ~half of window 0's samples
        l3 = process_l3(l2, l1, params)
        assert l3.pspd_rel[0] > 0.5  # finite-sample mean ~0.6, not NaN/floor

    def test_properties(self):
        l1, l2 = _make_l1_l2()
        params = _make_params()
        l3 = process_l3(l2, l1, params)
        assert l3.n_shear == 2
        assert l3.n_wavenumber == 129


class TestProcessL3NoGoodman:
    """Test L3 processing without Goodman cleaning."""

    def test_no_goodman(self):
        l1, l2 = _make_l1_l2()
        params = _make_params()
        params.goodman = False
        l3 = process_l3(l2, l1, params)
        # Without Goodman, clean == raw
        np.testing.assert_array_equal(l3.sh_spec, l3.sh_spec_clean)


class TestProcessL3NoSections:
    """Test with no valid sections."""

    def test_empty_output(self):
        l1, l2 = _make_l1_l2()
        l2.section_number = np.zeros(l2.section_number.shape)  # no sections
        params = _make_params()
        l3 = process_l3(l2, l1, params)
        assert l3.n_spectra == 0
        assert l3.time.shape == (0,)

    def test_short_section(self):
        """Section shorter than diss_length should produce no spectra."""
        l1, l2 = _make_l1_l2(n_time=1000)
        params = _make_params()
        params.diss_length = 2048
        l3 = process_l3(l2, l1, params)
        assert l3.n_spectra == 0


class TestProcessL3NoVib:
    """Test L3 with no vibration channels."""

    def test_no_vib(self):
        l1, l2 = _make_l1_l2(n_vib=0)
        l1.vib = np.zeros((0, l1.n_time))
        l2.vib = np.zeros((0, l2.time.shape[0]))
        params = _make_params()
        l3 = process_l3(l2, l1, params)
        # Without vibration, clean == raw
        np.testing.assert_array_equal(l3.sh_spec, l3.sh_spec_clean)


class TestSectionGroups:
    """Window layout: top-anchored (historical) vs bottom-anchored (BBL)."""


    def test_top_anchor_is_the_historical_layout(self):
        g = _section_groups(1.0, 0, 5000, 1024, 512, anchor="top")
        assert len(g) == 1
        _, starts, dl = g[0]
        assert dl == 1024
        assert starts[0] == 0
        expected = (5000 - 1024) // 512 + 1
        assert len(starts) == expected
        assert np.array_equal(starts, np.arange(expected) * 512)

    def test_top_anchor_discards_the_bottom_remainder(self):
        _, starts, dl = _section_groups(1.0, 0, 5000, 1024, 512, anchor="top")[0]
        assert starts[-1] + dl < 5000  # the whole point

    def test_bottom_anchor_pins_the_deepest_edge_to_sec_end(self):
        _, starts, dl = _section_groups(1.0, 0, 5000, 1024, 512, anchor="bottom")[0]
        assert starts[-1] + dl == 5000

    def test_bottom_anchor_moves_the_remainder_to_the_top(self):
        _, starts, _ = _section_groups(1.0, 0, 5000, 1024, 512, anchor="bottom")[0]
        assert starts[0] > 0

    def test_both_anchors_give_the_same_window_count(self):
        top = _section_groups(1.0, 0, 5000, 1024, 512, anchor="top")[0][1]
        bot = _section_groups(1.0, 0, 5000, 1024, 512, anchor="bottom")[0][1]
        assert len(top) == len(bot)

    def test_bbl_group_is_shorter_and_deepest(self):
        g = _section_groups(
            1.0, 0, 5000, 1024, 512,
            anchor="bottom", bbl_diss_length=512, bbl_step=256, bbl_extent=1536,
        )
        assert len(g) == 2
        (_, bbl_starts, bbl_len), (_, nrm_starts, nrm_len) = g
        assert bbl_len == 512 and nrm_len == 1024
        assert bbl_starts[-1] + bbl_len == 5000       # pinned to the seabed
        assert nrm_starts[-1] + nrm_len <= bbl_starts[0]  # no straddling

    def test_bbl_windows_put_the_deepest_centre_closer_to_the_seabed(self):
        """The reason for the whole feature."""
        plain = _section_groups(1.0, 0, 5000, 1024, 512, anchor="bottom")[0]
        bbl = _section_groups(
            1.0, 0, 5000, 1024, 512,
            anchor="bottom", bbl_diss_length=512, bbl_step=256, bbl_extent=1536,
        )[0]
        centre_plain = plain[1][-1] + plain[2] // 2
        centre_bbl = bbl[1][-1] + bbl[2] // 2
        assert centre_bbl > centre_plain
        assert 5000 - centre_bbl == pytest.approx(256, abs=1)

    def test_bbl_extent_shorter_than_one_window_yields_no_bbl_group(self):
        g = _section_groups(
            1.0, 0, 5000, 1024, 512,
            anchor="bottom", bbl_diss_length=512, bbl_step=256, bbl_extent=100,
        )
        assert all(dl == 1024 for _, _, dl in g)

    def test_section_shorter_than_window_yields_nothing(self):
        assert _section_groups(1.0, 0, 500, 1024, 512, anchor="top") == []
        assert _section_groups(1.0, 0, 500, 1024, 512, anchor="bottom") == []

    def test_short_section_still_gets_a_bbl_window(self):
        """A cast too short for a full window is not necessarily too short for
        a BBL window -- that is the case bottom-up work most cares about."""
        g = _section_groups(
            1.0, 0, 700, 1024, 512,
            anchor="bottom", bbl_diss_length=512, bbl_step=256, bbl_extent=700,
        )
        assert len(g) == 1 and g[0][2] == 512
        assert g[0][1][-1] + 512 == 700

    def test_unknown_anchor_rejected(self):
        with pytest.raises(ValueError, match="anchor"):
            _section_groups(1.0, 0, 5000, 1024, 512, anchor="sideways")

    def test_windows_stay_inside_the_section(self):
        for kw in ({"anchor": "top"}, {"anchor": "bottom"},
                   {"anchor": "bottom", "bbl_diss_length": 512,
                    "bbl_step": 256, "bbl_extent": 1536}):
            for _, starts, dl in _section_groups(1.0, 100, 5100, 1024, 512, **kw):
                assert starts.min() >= 100
                assert starts.max() + dl <= 5100
