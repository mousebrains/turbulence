"""Smoke tests for interactive viewers (quick_look, diss_look)."""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, MUST be before pyplot import

from pathlib import Path

import numpy as np
import pytest

VMP_DIR = Path(__file__).parent.parent / "VMP"
P_FILES = sorted(VMP_DIR.glob("*.p"))


@pytest.mark.skipif(not P_FILES, reason="No VMP .p files available")
class TestQuickLook:
    def test_smoke(self):
        """quick_look() creates a viewer without crashing."""
        from odas_tpw.rsi.p_file import PFile
        from odas_tpw.rsi.quick_look import QuickLookViewer

        pf = PFile(P_FILES[0])
        viewer = QuickLookViewer(pf, fft_length=256, f_AA=98.0)
        assert viewer.profiles  # at least one profile detected
        assert viewer.shear  # shear channels exist


@pytest.mark.skipif(not P_FILES, reason="No VMP .p files available")
class TestDissLook:
    def test_smoke(self):
        """DissLookViewer creates a viewer without crashing."""
        from odas_tpw.rsi.diss_look import DissLookViewer
        from odas_tpw.rsi.p_file import PFile

        pf = PFile(P_FILES[0])
        viewer = DissLookViewer(pf, fft_length=256, f_AA=98.0)
        assert viewer.profiles
        assert viewer.shear


# ---------------------------------------------------------------------------
# Mixing viewer (ml) — CI-safe, uses the committed test .p file
# ---------------------------------------------------------------------------

TEST_P = Path(__file__).parent / "data" / "SN479_0006.p"


@pytest.mark.skipif(not TEST_P.exists(), reason="SN479_0006.p test data not available")
class TestMixingLook:
    def _viewer(self):
        from odas_tpw.rsi.mixing_look import MixingLookViewer
        from odas_tpw.rsi.p_file import PFile

        pf = PFile(TEST_P)
        return MixingLookViewer(pf, fft_length=256, f_AA=98.0)

    def test_smoke(self):
        """MixingLookViewer constructs without crashing."""
        viewer = self._viewer()
        assert viewer.profiles
        assert viewer.shear

    def test_measured_salinity_from_jac(self):
        """VMP carries JAC C/T → measured salinity on the slow time base."""
        viewer = self._viewer()
        assert isinstance(viewer.salinity, np.ndarray)
        assert len(viewer.salinity) == len(viewer.t_slow)
        finite = viewer.salinity[np.isfinite(viewer.salinity)]
        assert finite.size > 0
        assert np.all((finite > 20.0) & (finite < 40.0))

    def test_salinity_override(self):
        """An explicit salinity overrides the measured JAC value."""
        from odas_tpw.rsi.mixing_look import MixingLookViewer
        from odas_tpw.rsi.p_file import PFile

        viewer = MixingLookViewer(PFile(TEST_P), fft_length=256, salinity=34.5)
        assert viewer.salinity == 34.5

    def test_window_times_align_with_diss_windows(self):
        """Window-center times line up 1:1 with compute_windowed_diss output."""
        from odas_tpw.rsi.viewer_base import compute_windowed_diss

        viewer = self._viewer()
        s_slow, e_slow = viewer.profiles[0]
        sel_fast = viewer._slow_to_fast_slice(s_slow, e_slow)
        d = compute_windowed_diss(
            viewer.shear, viewer.accel, viewer.therm_fast, viewer.diff_gains,
            viewer.P_fast, viewer.T, viewer.speed_fast, viewer.fs_fast, sel_fast,
            viewer.fft_length, viewer.f_AA, viewer.goodman, diss_length=viewer.diss_length,
        )
        t_win = viewer._window_center_times(sel_fast)
        assert len(t_win) == len(d["P_windows"])

    def test_draw_runs(self):
        """Full _draw() exercises stratification + mixing + spectra without crashing."""
        import matplotlib.pyplot as plt

        viewer = self._viewer()
        viewer.fig, viewer.axes = plt.subplots(viewer._nrows, viewer._ncols, squeeze=False)
        try:
            viewer._setup_axes()
            viewer._draw()  # N2/dTdz/K_T/Gamma/K_rho + eps/chi spectra + FM/FOM
            assert viewer._mix is not None
            assert len(viewer._mix.Gamma) == len(viewer._P_win)
            assert viewer._cached_spec is not None  # row-2 spectra computed
        finally:
            plt.close(viewer.fig)

    def test_grid_is_three_rows(self):
        """ml requests the extra spectra row."""
        assert self._viewer()._nrows == 3

    def test_nav_diamond_buttons(self):
        """The nav diamond builds four textless arrow buttons wired to handlers."""
        import matplotlib.pyplot as plt

        viewer = self._viewer()
        viewer.fig = plt.figure()
        try:
            viewer._add_nav_diamond(0.04, 0.98, 0.08, 0.92, 0.35)
            assert viewer.btn_prev.label.get_text() == "◀"
            assert viewer.btn_next.label.get_text() == "▶"
            assert viewer.btn_spec_up.label.get_text() == "▲"
            assert viewer.btn_spec_dn.label.get_text() == "▼"
        finally:
            plt.close(viewer.fig)


@pytest.mark.skipif(not TEST_P.exists(), reason="SN479_0006.p test data not available")
class TestDissLookDraw:
    """Covers the shared FM / chi-spectra / chi-FOM panels moved to the base."""

    def test_draw_runs(self):
        import matplotlib.pyplot as plt

        from odas_tpw.rsi.diss_look import DissLookViewer
        from odas_tpw.rsi.p_file import PFile

        viewer = DissLookViewer(PFile(TEST_P), fft_length=256, f_AA=98.0)
        viewer.fig, viewer.axes = plt.subplots(viewer._nrows, viewer._ncols, squeeze=False)
        try:
            viewer._setup_axes()
            viewer._draw()
            assert viewer._cached_diss is not None
            assert viewer._cached_spec is not None
        finally:
            plt.close(viewer.fig)


class TestInterpSlowToFast:
    """Per-window temperature for viewer viscosity/noise (M-7)."""

    def test_linear_interp_to_fast_length(self):
        from odas_tpw.rsi.viewer_base import _interp_slow_to_fast

        T_slow = np.array([10.0, 20.0])      # warm surface -> cool deep
        out = _interp_slow_to_fast(T_slow, 5)
        assert out.shape == (5,)
        np.testing.assert_allclose(out, [10.0, 12.5, 15.0, 17.5, 20.0])

    def test_window_mean_differs_from_full_cast_mean(self):
        from odas_tpw.rsi.viewer_base import _interp_slow_to_fast

        # 26 C surface -> 6 C deep; a deep window must NOT see the ~16 C cast mean.
        T_slow = np.linspace(26.0, 6.0, 50)
        T_fast = _interp_slow_to_fast(T_slow, 1000)
        deep = slice(900, 1000)
        assert np.mean(T_fast[deep]) < 8.0          # near the deep value
        assert abs(np.mean(T_fast[deep]) - np.mean(T_slow)) > 5.0  # not the cast mean

    def test_edge_cases(self):
        from odas_tpw.rsi.viewer_base import _interp_slow_to_fast

        # Already fast-length -> unchanged; single sample -> constant fill.
        np.testing.assert_array_equal(
            _interp_slow_to_fast(np.array([1.0, 2.0, 3.0]), 3), [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(
            _interp_slow_to_fast(np.array([7.0]), 4), np.full(4, 7.0))
