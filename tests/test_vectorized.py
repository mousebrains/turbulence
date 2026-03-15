# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for vectorized spectral, Goodman, Nasmyth grid, and dissipation code.

Validates that batched/gridded implementations produce results matching
the original per-window/per-call implementations.
"""

from pathlib import Path

import numpy as np
import pytest

VMP_DIR = Path(__file__).resolve().parent.parent / "VMP"
P_FILES = sorted(VMP_DIR.glob("*.p"))


# ---------------------------------------------------------------------------
# spectral.py — _detrend_batch
# ---------------------------------------------------------------------------


class TestDetrendBatch:
    """Validate _detrend_batch matches per-segment _detrend_segment."""

    def _make_segments(self, rng, shape=(5, 3, 256, 2)):
        """Random segments: (n_win, n_seg, nfft, n_ch)."""
        return rng.standard_normal(shape)

    def test_none(self):
        from odas_tpw.scor160.spectral import _detrend_batch

        rng = np.random.default_rng(42)
        segs = self._make_segments(rng)
        result = _detrend_batch(segs, "none", axis=2)
        np.testing.assert_array_equal(result, segs)

    def test_constant(self):
        from odas_tpw.scor160.spectral import _detrend_batch

        rng = np.random.default_rng(42)
        segs = self._make_segments(rng)
        result = _detrend_batch(segs, "constant", axis=2)

        # Verify mean along nfft axis is ~zero
        means = np.mean(result, axis=2)
        np.testing.assert_allclose(means, 0, atol=1e-12)

    def test_linear_matches_per_segment(self):
        from odas_tpw.scor160.spectral import _detrend_batch, _detrend_segment

        rng = np.random.default_rng(42)
        segs = self._make_segments(rng, shape=(3, 2, 256, 2))
        result = _detrend_batch(segs, "linear", axis=2)
        ramp = np.arange(256, dtype=np.float64)

        for w in range(3):
            for s in range(2):
                for c in range(2):
                    ref = _detrend_segment(segs[w, s, :, c], "linear", ramp)
                    np.testing.assert_allclose(result[w, s, :, c], ref, atol=1e-10)

    def test_parabolic_matches_per_segment(self):
        from odas_tpw.scor160.spectral import _detrend_batch, _detrend_segment

        rng = np.random.default_rng(42)
        segs = self._make_segments(rng, shape=(2, 2, 128, 1))
        result = _detrend_batch(segs, "parabolic", axis=2)
        ramp = np.arange(128, dtype=np.float64)

        for w in range(2):
            for s in range(2):
                ref = _detrend_segment(segs[w, s, :, 0], "parabolic", ramp)
                np.testing.assert_allclose(result[w, s, :, 0], ref, atol=1e-8)

    def test_cubic_matches_per_segment(self):
        from odas_tpw.scor160.spectral import _detrend_batch, _detrend_segment

        rng = np.random.default_rng(42)
        segs = self._make_segments(rng, shape=(2, 2, 128, 1))
        result = _detrend_batch(segs, "cubic", axis=2)
        ramp = np.arange(128, dtype=np.float64)

        for w in range(2):
            for s in range(2):
                ref = _detrend_segment(segs[w, s, :, 0], "cubic", ramp)
                np.testing.assert_allclose(result[w, s, :, 0], ref, atol=1e-8)


# ---------------------------------------------------------------------------
# spectral.py — csd_matrix_batch
# ---------------------------------------------------------------------------


class TestCsdMatrixBatch:
    """Validate csd_matrix_batch matches per-window csd_matrix."""

    @pytest.fixture()
    def random_windows(self):
        rng = np.random.default_rng(42)
        n_windows = 7
        diss_length = 512
        return {
            "x": rng.standard_normal((n_windows, diss_length, 2)),
            "y": rng.standard_normal((n_windows, diss_length, 3)),
            "nfft": 256,
            "rate": 512.0,
            "overlap": 128,
        }

    def test_auto_matches_per_window(self, random_windows):
        from odas_tpw.scor160.spectral import csd_matrix, csd_matrix_batch

        d = random_windows
        Cxy_b, F_b, _, _ = csd_matrix_batch(
            d["x"],
            None,
            d["nfft"],
            d["rate"],
            d["overlap"],
        )
        assert Cxy_b.shape == (7, 129, 2, 2)

        for w in range(7):
            Cxy_ref, F_ref, _, _ = csd_matrix(
                d["x"][w],
                None,
                d["nfft"],
                d["rate"],
                overlap=d["overlap"],
            )
            np.testing.assert_allclose(Cxy_b[w], Cxy_ref, atol=1e-12)
        np.testing.assert_array_equal(F_b, F_ref)

    def test_cross_matches_per_window(self, random_windows):
        from odas_tpw.scor160.spectral import csd_matrix, csd_matrix_batch

        d = random_windows
        Cxy_b, _F_b, Cxx_b, Cyy_b = csd_matrix_batch(
            d["x"],
            d["y"],
            d["nfft"],
            d["rate"],
            d["overlap"],
        )
        assert Cxy_b.shape == (7, 129, 2, 3)
        assert Cxx_b.shape == (7, 129, 2, 2)
        assert Cyy_b.shape == (7, 129, 3, 3)

        for w in range(7):
            Cxy_ref, _, Cxx_ref, Cyy_ref = csd_matrix(
                d["x"][w],
                d["y"][w],
                d["nfft"],
                d["rate"],
                overlap=d["overlap"],
            )
            np.testing.assert_allclose(Cxy_b[w], Cxy_ref, atol=1e-12)
            np.testing.assert_allclose(Cxx_b[w], Cxx_ref, atol=1e-12)
            np.testing.assert_allclose(Cyy_b[w], Cyy_ref, atol=1e-12)

    def test_single_window(self):
        """Batch of 1 should match scalar csd_matrix."""
        from odas_tpw.scor160.spectral import csd_matrix, csd_matrix_batch

        rng = np.random.default_rng(99)
        x = rng.standard_normal((1, 512, 2))
        Cxy_b, _, _, _ = csd_matrix_batch(x, None, 256, 512.0, 128)
        Cxy_ref, _, _, _ = csd_matrix(x[0], None, 256, 512.0, overlap=128)
        np.testing.assert_allclose(Cxy_b[0], Cxy_ref, atol=1e-12)

    def test_auto_diagonal_positive_real(self, random_windows):
        """Auto-spectral diagonal should be real and positive."""
        from odas_tpw.scor160.spectral import csd_matrix_batch

        d = random_windows
        Cxy, _, _, _ = csd_matrix_batch(
            d["x"],
            None,
            d["nfft"],
            d["rate"],
            d["overlap"],
        )
        for ch in range(2):
            diag = Cxy[:, :, ch, ch]
            assert np.all(np.imag(diag) == 0) or np.allclose(np.imag(diag), 0, atol=1e-15)
            assert np.all(np.real(diag) > 0)

    def test_invalid_ndim_raises(self):
        from odas_tpw.scor160.spectral import csd_matrix_batch

        with pytest.raises(ValueError, match="3-D"):
            csd_matrix_batch(np.zeros((512, 2)), None, 256, 512.0)

    def test_diss_length_too_short_raises(self):
        from odas_tpw.scor160.spectral import csd_matrix_batch

        with pytest.raises(ValueError, match="too short"):
            csd_matrix_batch(np.zeros((5, 100, 2)), None, 256, 512.0)

    def test_shape_mismatch_raises(self):
        from odas_tpw.scor160.spectral import csd_matrix_batch

        x = np.zeros((5, 512, 2))
        y = np.zeros((3, 512, 2))  # different n_windows
        with pytest.raises(ValueError, match="incompatible"):
            csd_matrix_batch(x, y, 256, 512.0)

    def test_constant_detrend(self):
        """Constant detrend should still match per-window."""
        from odas_tpw.scor160.spectral import csd_matrix, csd_matrix_batch

        rng = np.random.default_rng(77)
        x = rng.standard_normal((3, 512, 2))
        Cxy_b, _, _, _ = csd_matrix_batch(x, None, 256, 512.0, detrend="constant")
        for w in range(3):
            Cxy_ref, _, _, _ = csd_matrix(x[w], None, 256, 512.0, detrend="constant")
            np.testing.assert_allclose(Cxy_b[w], Cxy_ref, atol=1e-12)


# ---------------------------------------------------------------------------
# nasmyth.py — NasmythGrid and nasmyth_grid
# ---------------------------------------------------------------------------


class TestNasmythGrid:
    """Validate nasmyth_grid matches nasmyth within interpolation tolerance."""

    def test_matches_nasmyth(self):
        from odas_tpw.scor160.nasmyth import nasmyth, nasmyth_grid

        k = np.logspace(-1, 2, 200)
        nu = 1.2e-6
        for eps in [1e-10, 1e-8, 1e-6, 1e-4]:
            ref = nasmyth(eps, nu, k)
            grid = nasmyth_grid(eps, nu, k)
            np.testing.assert_allclose(grid, ref, rtol=1e-4)

    def test_shape_preserved(self):
        from odas_tpw.scor160.nasmyth import nasmyth_grid

        k = np.logspace(-1, 2, 50)
        phi = nasmyth_grid(1e-7, 1.2e-6, k)
        assert phi.shape == (50,)

    def test_positive(self):
        from odas_tpw.scor160.nasmyth import nasmyth_grid

        k = np.logspace(-1, 2, 50)
        phi = nasmyth_grid(1e-7, 1.2e-6, k)
        assert np.all(phi > 0)

    def test_negative_epsilon_warns(self):
        from odas_tpw.scor160.nasmyth import nasmyth_grid

        with pytest.warns(UserWarning):
            result = nasmyth_grid(-1e-8, 1e-6, np.array([1, 10]))
        assert np.all(np.isnan(result))

    def test_zero_epsilon_warns(self):
        from odas_tpw.scor160.nasmyth import nasmyth_grid

        with pytest.warns(UserWarning):
            result = nasmyth_grid(0.0, 1e-6, np.array([1, 10]))
        assert np.all(np.isnan(result))

    def test_zero_nu_raises(self):
        from odas_tpw.scor160.nasmyth import nasmyth_grid

        with pytest.raises(ValueError):
            nasmyth_grid(1e-8, 0.0, np.array([1, 10]))

    def test_higher_epsilon_higher_spectrum(self):
        from odas_tpw.scor160.nasmyth import nasmyth_grid

        k = np.logspace(0, 2, 50)
        nu = 1e-6
        phi_low = nasmyth_grid(1e-9, nu, k)
        phi_high = nasmyth_grid(1e-5, nu, k)
        assert np.mean(phi_high) > np.mean(phi_low)

    def test_singleton_reused(self):
        from odas_tpw.scor160.nasmyth import _get_grid

        g1 = _get_grid()
        g2 = _get_grid()
        assert g1 is g2

    def test_interp_g2_scalar(self):
        from odas_tpw.scor160.nasmyth import NasmythGrid, _nasmyth_g2

        grid = NasmythGrid(n=2000)
        x = 0.05
        ref = float(_nasmyth_g2(np.array([x]))[0])
        result = grid.interp_g2(np.array(x))
        assert isinstance(result, (float, np.floating))
        np.testing.assert_allclose(result, ref, rtol=1e-4)

    def test_interp_g2_out_of_range_fallback(self):
        """Values outside grid range should fall back to direct computation."""
        from odas_tpw.scor160.nasmyth import NasmythGrid, _nasmyth_g2

        grid = NasmythGrid(n=100, x_min=1e-4, x_max=0.5)
        x = np.array([1e-7, 0.8])  # both outside range
        ref = _nasmyth_g2(x)
        result = grid.interp_g2(x)
        np.testing.assert_allclose(result, ref, rtol=1e-12)


# ---------------------------------------------------------------------------
# goodman.py — clean_shear_spec_batch
# ---------------------------------------------------------------------------


class TestCleanShearSpecBatch:
    """Validate clean_shear_spec_batch matches per-window clean_shear_spec."""

    def _make_windows(self, rng, n_windows=5, diss_length=512, n_shear=2, n_accel=2):
        return {
            "shear": rng.standard_normal((n_windows, diss_length, n_shear)),
            "accel": rng.standard_normal((n_windows, diss_length, n_accel)),
        }

    def test_matches_per_window(self):
        from odas_tpw.scor160.goodman import clean_shear_spec, clean_shear_spec_batch

        rng = np.random.default_rng(42)
        d = self._make_windows(rng, n_windows=5)
        nfft = 256
        rate = 512.0

        clean_batch, F_batch = clean_shear_spec_batch(
            d["accel"],
            d["shear"],
            nfft,
            rate,
        )
        assert clean_batch.shape == (5, 129, 2, 2)

        for w in range(5):
            clean_ref, _, _, _, F_ref = clean_shear_spec(
                d["accel"][w],
                d["shear"][w],
                nfft,
                rate,
            )
            np.testing.assert_allclose(clean_batch[w], clean_ref, atol=1e-10)
        np.testing.assert_array_equal(F_batch, F_ref)

    def test_single_window(self):
        from odas_tpw.scor160.goodman import clean_shear_spec, clean_shear_spec_batch

        rng = np.random.default_rng(99)
        d = self._make_windows(rng, n_windows=1)
        nfft = 256
        rate = 512.0

        clean_batch, _ = clean_shear_spec_batch(d["accel"], d["shear"], nfft, rate)
        clean_ref, _, _, _, _ = clean_shear_spec(
            d["accel"][0],
            d["shear"][0],
            nfft,
            rate,
        )
        np.testing.assert_allclose(clean_batch[0], clean_ref, atol=1e-10)

    def test_cleaned_spectra_real(self):
        from odas_tpw.scor160.goodman import clean_shear_spec_batch

        rng = np.random.default_rng(42)
        d = self._make_windows(rng)
        clean, _ = clean_shear_spec_batch(d["accel"], d["shear"], 256, 512.0)
        assert clean.dtype in (np.float64, np.float32)

    def test_diagonal_positive(self):
        """Cleaned auto-spectra (diagonal) should be positive for well-conditioned data."""
        from odas_tpw.scor160.goodman import clean_shear_spec_batch

        rng = np.random.default_rng(42)
        d = self._make_windows(rng, n_windows=3)
        clean, _ = clean_shear_spec_batch(d["accel"], d["shear"], 256, 512.0)
        # Diagonal elements should be mostly positive
        for ch in range(2):
            diag = clean[:, 1:, ch, ch]  # skip DC
            frac_positive = np.mean(diag > 0)
            assert frac_positive > 0.9

    def test_noise_reduction(self):
        """Cleaning should reduce vibration-coherent energy."""
        from odas_tpw.scor160.goodman import clean_shear_spec, clean_shear_spec_batch

        rng = np.random.default_rng(42)
        n_win = 3
        diss_length = 512
        nfft = 256
        rate = 512.0

        # Create shear with vibration-coherent component
        t = np.arange(diss_length) / rate
        vib_freq = 50.0
        vib = np.sin(2 * np.pi * vib_freq * t)

        shear = rng.standard_normal((n_win, diss_length, 2)) * 0.1
        accel = rng.standard_normal((n_win, diss_length, 2)) * 0.1
        for w in range(n_win):
            shear[w, :, 0] += vib * 2  # add vibration to shear
            accel[w, :, 0] += vib  # same vibration in accel

        clean_batch, _F = clean_shear_spec_batch(accel, shear, nfft, rate)

        # Verify per-window equivalence
        for w in range(n_win):
            clean_ref, _, UU_ref, _, _ = clean_shear_spec(
                accel[w],
                shear[w],
                nfft,
                rate,
            )
            np.testing.assert_allclose(clean_batch[w], clean_ref, atol=1e-10)

            # Cleaned spectrum should have less energy at vibration frequency
            vib_bin = int(vib_freq / (rate / nfft))
            cleaned_power = np.real(clean_ref[vib_bin, 0, 0])
            uncleaned_power = np.real(UU_ref[vib_bin, 0, 0])
            assert cleaned_power < uncleaned_power


# ---------------------------------------------------------------------------
# Integration: get_diss end-to-end on real data
# ---------------------------------------------------------------------------


@pytest.mark.skipif(len(P_FILES) < 6, reason="Need VMP .p files")
class TestVectorizedIntegration:
    """Integration test: vectorized get_diss on a real .p file."""

    def test_get_diss_produces_valid_results(self):
        from odas_tpw.rsi.dissipation import get_diss
        from odas_tpw.rsi.p_file import PFile

        pf = PFile(P_FILES[5])  # file 0006
        results = get_diss(pf)
        assert len(results) > 0

        for ds in results:
            eps = ds["epsilon"].values
            assert np.all(np.isfinite(eps))
            assert np.all(eps > 1e-14)
            assert np.all(eps < 1.0)

    def test_qc_variables_present(self):
        from odas_tpw.rsi.dissipation import get_diss
        from odas_tpw.rsi.p_file import PFile

        pf = PFile(P_FILES[5])
        results = get_diss(pf)
        for ds in results:
            for var in ("fom", "K_max_ratio", "mad", "FM", "method"):
                assert var in ds, f"Missing QC variable: {var}"
            assert np.any(np.isfinite(ds["fom"].values))

    def test_cf_attributes(self):
        from odas_tpw.rsi.dissipation import get_diss
        from odas_tpw.rsi.p_file import PFile

        pf = PFile(P_FILES[5])
        results = get_diss(pf)
        ds = results[0]
        assert ds.attrs.get("Conventions") == "CF-1.13"
        assert "history" in ds.attrs
        assert ds.coords["t"].attrs.get("standard_name") == "time"
