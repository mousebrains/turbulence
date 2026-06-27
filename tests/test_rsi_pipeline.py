# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for rsi.pipeline.run_pipeline using real .p test data."""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

TEST_DATA_DIR = Path(__file__).parent / "data"
SAMPLE_FILE = TEST_DATA_DIR / "SN479_0006.p"


@pytest.fixture
def sample_p():
    if not SAMPLE_FILE.exists():
        pytest.skip("Test data not available")
    return SAMPLE_FILE


# ---------------------------------------------------------------------------
# run_pipeline — defaulting of dissipation parameters
# ---------------------------------------------------------------------------


class TestRunPipelineDefaults:
    def test_diss_length_default_4x_fft(self, sample_p, tmp_path):
        """When diss_length is None, defaults to 4 * fft_length."""
        from odas_tpw.rsi.pipeline import run_pipeline

        # Use small fft_length so the pipeline runs quickly
        out = run_pipeline(
            [sample_p],
            tmp_path,
            fft_length=256,
            diss_length=None,  # → 1024 default
            overlap=None,  # → 512 default
            chi_fft_length=256,
            chi_diss_length=None,  # → 1024 default
            chi_overlap=None,
            compute_chi_epsilon=False,
            compute_chi_fit=False,
        )
        assert out == tmp_path
        assert (tmp_path / sample_p.stem).exists()

    def test_mismatched_diss_lengths_warn(self, sample_p, tmp_path):
        """When epsilon and chi diss_lengths differ, emit UserWarning."""
        from odas_tpw.rsi.pipeline import run_pipeline

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            run_pipeline(
                [sample_p],
                tmp_path,
                fft_length=256,
                diss_length=1024,
                overlap=512,
                chi_fft_length=256,
                chi_diss_length=2048,  # Differs from diss_length
                chi_overlap=1024,
                compute_chi_epsilon=False,
                compute_chi_fit=False,
            )
        msgs = [str(rec.message) for rec in w]
        assert any("different dissipation lengths" in m for m in msgs)


# ---------------------------------------------------------------------------
# run_pipeline — chi method coverage
# ---------------------------------------------------------------------------


class TestRunPipelineChiMethods:
    def test_chi_fit_method(self, sample_p, tmp_path):
        """compute_chi_fit=True produces L4_chi_fit.nc per profile."""
        from odas_tpw.rsi.pipeline import run_pipeline

        run_pipeline(
            [sample_p],
            tmp_path,
            fft_length=256,
            diss_length=1024,
            overlap=512,
            chi_fft_length=256,
            chi_diss_length=1024,
            chi_overlap=512,
            compute_chi_epsilon=False,
            compute_chi_fit=True,
        )
        # Should produce L4_chi_fit.nc somewhere under the output
        chi_fit_files = list((tmp_path / sample_p.stem).rglob("L4_chi_fit.nc"))
        assert len(chi_fit_files) >= 1

    def test_chi_fit_method_writes_mixing_quantities(self, sample_p, tmp_path):
        """A Method-2-only run (compute_chi_epsilon=False) must still write the
        stratification/mixing block to L4_chi_fit.nc; previously the whole block
        was gated on the Method-1 result and silently dropped (bug_003)."""
        from odas_tpw.rsi.pipeline import run_pipeline

        run_pipeline(
            [sample_p],
            tmp_path,
            fft_length=256,
            diss_length=1024,
            overlap=512,
            chi_fft_length=256,
            chi_diss_length=1024,
            chi_overlap=512,
            compute_chi_epsilon=False,
            compute_chi_fit=True,
        )
        chi_fit_files = list((tmp_path / sample_p.stem).rglob("L4_chi_fit.nc"))
        assert len(chi_fit_files) >= 1
        with xr.open_dataset(chi_fit_files[0]) as ds:
            # N2/dTdz need only the CTD profile, so they must be present even
            # without Method-1 chi-from-epsilon.
            assert "N2" in ds
            assert "dTdz" in ds

    def test_chi_epsilon_method(self, sample_p, tmp_path):
        """compute_chi_epsilon=True produces L4_chi_epsilon.nc per profile."""
        from odas_tpw.rsi.pipeline import run_pipeline

        run_pipeline(
            [sample_p],
            tmp_path,
            fft_length=256,
            diss_length=1024,
            overlap=512,
            chi_fft_length=256,
            chi_diss_length=1024,
            chi_overlap=512,
            compute_chi_epsilon=True,
            compute_chi_fit=False,
        )
        chi_eps_files = list((tmp_path / sample_p.stem).rglob("L4_chi_epsilon.nc"))
        assert len(chi_eps_files) >= 1


# ---------------------------------------------------------------------------
# run_pipeline — no profiles detected
# ---------------------------------------------------------------------------


class TestRunPipelineNoProfiles:
    def test_unrealistic_thresholds_no_profiles(self, sample_p, tmp_path, caplog):
        """Set P_min/W_min so high that no profiles are detected → log warning."""
        from odas_tpw.rsi.pipeline import run_pipeline

        with caplog.at_level("WARNING"):
            run_pipeline(
                [sample_p],
                tmp_path,
                P_min=1e6,  # Absurdly high → no profiles will satisfy
                W_min=1e6,
                fft_length=256,
                diss_length=1024,
                overlap=512,
                chi_fft_length=256,
                chi_diss_length=1024,
                chi_overlap=512,
                compute_chi_epsilon=False,
                compute_chi_fit=False,
            )
        # The pipeline should log a warning about no profiles
        msgs = [r.message for r in caplog.records]
        assert any("No profiles detected" in m for m in msgs)


class TestRunPipelineMetadata:
    @patch("odas_tpw.rsi.pipeline.combine_profiles")
    @patch("odas_tpw.rsi.pipeline._process_profile")
    @patch("odas_tpw.rsi.profile.get_profiles")
    @patch("odas_tpw.rsi.profile._smooth_fall_rate", return_value=np.ones(40))
    @patch("odas_tpw.rsi.p_file.PFile")
    def test_skipped_profiles_do_not_shift_metadata(
        self,
        mock_pfile_cls,
        mock_smooth,
        mock_get_profiles,
        mock_process_profile,
        mock_combine,
        tmp_path,
    ):
        from odas_tpw.rsi.pipeline import run_pipeline

        pf = MagicMock()
        pf.channels = {"P": np.linspace(0, 20, 40)}
        pf.fs_slow = 2.0
        pf.config = {"instrument_info": {}}
        mock_pfile_cls.return_value = pf
        mock_get_profiles.return_value = [(0, 9), (10, 19), (20, 29)]

        binned = xr.Dataset({"T": (["depth_bin"], [10.0])}, coords={"depth_bin": [1.0]})
        mock_process_profile.side_effect = [binned, None, binned]
        mock_combine.return_value = xr.Dataset(
            {"T": (["profile", "depth_bin"], [[10.0], [11.0]])},
            coords={"profile": [0, 1], "depth_bin": [1.0]},
        )

        run_pipeline([tmp_path / "cast.p"], tmp_path)

        metadata = mock_combine.call_args.args[1]
        assert [m["profile_number"] for m in metadata] == [1, 3]


# ---------------------------------------------------------------------------
# run_pipeline — output files written
# ---------------------------------------------------------------------------


class TestRunPipelineOutputs:
    def test_l4_epsilon_written(self, sample_p, tmp_path):
        from odas_tpw.rsi.pipeline import run_pipeline

        run_pipeline(
            [sample_p],
            tmp_path,
            fft_length=256,
            diss_length=1024,
            overlap=512,
            chi_fft_length=256,
            chi_diss_length=1024,
            chi_overlap=512,
            compute_chi_epsilon=False,
            compute_chi_fit=False,
        )
        eps_files = list((tmp_path / sample_p.stem).rglob("L4_epsilon.nc"))
        assert len(eps_files) >= 1

    def test_l5_binned_written(self, sample_p, tmp_path):
        from odas_tpw.rsi.pipeline import run_pipeline

        run_pipeline(
            [sample_p],
            tmp_path,
            fft_length=256,
            diss_length=1024,
            overlap=512,
            chi_fft_length=256,
            chi_diss_length=1024,
            chi_overlap=512,
            compute_chi_epsilon=False,
            compute_chi_fit=False,
        )
        binned_files = list((tmp_path / sample_p.stem).rglob("L5_binned.nc"))
        assert len(binned_files) >= 1

    def test_l6_combined_written(self, sample_p, tmp_path):
        """When at least one profile produces output → L6_combined.nc is written."""
        from odas_tpw.rsi.pipeline import run_pipeline

        run_pipeline(
            [sample_p],
            tmp_path,
            fft_length=256,
            diss_length=1024,
            overlap=512,
            chi_fft_length=256,
            chi_diss_length=1024,
            chi_overlap=512,
            compute_chi_epsilon=False,
            compute_chi_fit=False,
        )
        combined_files = list((tmp_path / sample_p.stem).rglob("L6_combined.nc"))
        assert len(combined_files) >= 1


# ---------------------------------------------------------------------------
# _resolve_salinity — chi viscosity / stratification share one salinity source
# ---------------------------------------------------------------------------


def _make_l1(n_time, salinity):
    """Minimal L1Data carrying only the fields _resolve_salinity reads."""
    from odas_tpw.scor160.io import L1Data

    return L1Data(
        time=np.arange(n_time, dtype=np.float64),
        pres=np.zeros(n_time),
        shear=np.zeros((0, n_time)),
        vib=np.zeros((0, n_time)),
        vib_type="NONE",
        fs_fast=512.0,
        f_AA=98.0,
        vehicle="vmp",
        profile_dir="down",
        time_reference_year=2025,
        salinity=np.asarray(salinity, dtype=np.float64),
    )


class TestResolveSalinity:
    def test_measured_preferred_over_user_value(self):
        """A finite per-sample measured salinity wins over a scalar default."""
        from odas_tpw.rsi.pipeline import _resolve_salinity

        meas = np.full(10, 34.6)
        val, measured = _resolve_salinity(_make_l1(10, meas), 35.0)
        assert measured is True
        np.testing.assert_array_equal(val, meas)

    def test_partial_nan_measured_still_used(self):
        """`any` finite is enough; the chi core nan-handles per window."""
        from odas_tpw.rsi.pipeline import _resolve_salinity

        meas = np.full(10, 34.6)
        meas[:3] = np.nan
        val, measured = _resolve_salinity(_make_l1(10, meas), None)
        assert measured is True
        assert isinstance(val, np.ndarray)

    def test_all_nan_measured_falls_back(self):
        """An all-NaN measured array is not 'measured'; user value is used."""
        from odas_tpw.rsi.pipeline import _resolve_salinity

        meas = np.full(10, np.nan)
        val, measured = _resolve_salinity(_make_l1(10, meas), 34.0)
        assert measured is False
        assert val == 34.0

    def test_empty_measured_scalar_user(self):
        """No measured salinity → scalar user value passes through."""
        from odas_tpw.rsi.pipeline import _resolve_salinity

        val, measured = _resolve_salinity(_make_l1(10, []), 34.2)
        assert measured is False
        assert val == 34.2

    def test_empty_measured_none(self):
        """No measured, no user value → None (→ 35 PSU downstream)."""
        from odas_tpw.rsi.pipeline import _resolve_salinity

        val, measured = _resolve_salinity(_make_l1(10, []), None)
        assert measured is False
        assert val is None

    def test_user_array_wrong_length_collapsed(self):
        """A user array that doesn't match the fast base collapses to nanmean."""
        from odas_tpw.rsi.pipeline import _resolve_salinity

        val, measured = _resolve_salinity(_make_l1(10, []), np.array([34.0, 36.0, np.nan]))
        assert measured is False
        assert val == pytest.approx(35.0)

    def test_user_array_matching_length_passthrough(self):
        """A user array already on the fast base passes through unchanged."""
        from odas_tpw.rsi.pipeline import _resolve_salinity

        user = np.linspace(34.0, 35.0, 10)
        val, measured = _resolve_salinity(_make_l1(10, []), user)
        assert measured is False
        np.testing.assert_array_equal(val, user)


# ---------------------------------------------------------------------------
# _epsilon_window_salinity — measured per-sample salinity -> per-window for L4
# ---------------------------------------------------------------------------


class TestEpsilonWindowSalinity:
    def test_measured_series_interpolated_onto_windows(self):
        """A measured per-sample series becomes per-window (size n_spectra)."""
        from odas_tpw.rsi.pipeline import _epsilon_window_salinity

        sample_time = np.linspace(0.0, 100.0, 1000)
        sal = np.linspace(34.0, 35.0, 1000)  # rises linearly with time
        win_time = np.array([25.0, 50.0, 75.0])
        out = _epsilon_window_salinity(sal, sample_time, 1000, win_time, 3)
        assert isinstance(out, np.ndarray) and out.size == 3
        # Linear series -> exact linear interpolation at the window centers,
        # and NOT collapsed to the profile mean (~34.5 everywhere).
        np.testing.assert_allclose(out, [34.25, 34.5, 34.75], rtol=1e-6)
        assert not np.allclose(out, np.nanmean(sal))

    def test_nan_samples_excluded_from_interp(self):
        from odas_tpw.rsi.pipeline import _epsilon_window_salinity

        sample_time = np.linspace(0.0, 100.0, 1000)
        sal = np.linspace(34.0, 35.0, 1000)
        sal[400:600] = np.nan  # a gap straddling the middle window
        win_time = np.array([50.0])
        out = _epsilon_window_salinity(sal, sample_time, 1000, win_time, 1)
        assert np.isfinite(out).all()
        np.testing.assert_allclose(out, [34.5], atol=1e-3)

    def test_scalar_none_and_mismatched_passthrough(self):
        """Non per-sample inputs are returned unchanged (process_l4 handles them)."""
        from odas_tpw.rsi.pipeline import _epsilon_window_salinity

        st = np.linspace(0.0, 1.0, 100)
        wt = np.array([0.5])
        assert _epsilon_window_salinity(34.5, st, 100, wt, 1) == 34.5
        assert _epsilon_window_salinity(None, st, 100, wt, 1) is None
        # Already per-window (size != n_time) passes through to process_l4.
        per_win = np.array([34.0, 35.0])
        np.testing.assert_array_equal(
            _epsilon_window_salinity(per_win, st, 100, wt, 1), per_win
        )

    def test_all_nan_series_passthrough(self):
        from odas_tpw.rsi.pipeline import _epsilon_window_salinity

        st = np.linspace(0.0, 1.0, 100)
        sal = np.full(100, np.nan)
        wt = np.array([0.5])
        out = _epsilon_window_salinity(sal, st, 100, wt, 1)
        assert out is sal  # unchanged; process_l4 applies its own 35-PSU fallback


# ---------------------------------------------------------------------------
# _epsilon_hp_cut — shear high-pass scales with the FFT length (match perturb)
# ---------------------------------------------------------------------------


class TestEpsilonHpCut:
    def test_default_1024_is_quarter_hz(self):
        """The default 1024-sample FFT at 512 Hz gives the historical 0.25 Hz."""
        from odas_tpw.rsi.pipeline import _epsilon_hp_cut

        assert _epsilon_hp_cut(512.0, 1024, None) == pytest.approx(0.25)

    def test_scales_inversely_with_fft_length(self):
        """A 256-sample FFT gives 1.0 Hz — 4x the 1024-sample cutoff."""
        from odas_tpw.rsi.pipeline import _epsilon_hp_cut

        assert _epsilon_hp_cut(512.0, 256, None) == pytest.approx(1.0)
        assert _epsilon_hp_cut(512.0, 512, None) == pytest.approx(0.5)

    def test_matches_modular_compute_epsilon_formula(self):
        """Identical to dissipation.py `_compute_epsilon` (0.5*fs/fft_length)."""
        from odas_tpw.rsi.pipeline import _epsilon_hp_cut

        for fs in (256.0, 512.0, 1024.0):
            for fft in (256, 512, 1024, 2048):
                assert _epsilon_hp_cut(fs, fft, None) == pytest.approx(0.5 * fs / fft)

    def test_explicit_override_wins(self):
        """A pinned cutoff overrides the scaling default."""
        from odas_tpw.rsi.pipeline import _epsilon_hp_cut

        assert _epsilon_hp_cut(512.0, 256, 0.3) == pytest.approx(0.3)

    def test_chi_hp_cut_constant_is_canonical(self):
        """Chi vib high-pass stays pinned at the canonical 0.25 Hz."""
        from odas_tpw.rsi.pipeline import _CHI_HP_CUT

        assert abs(_CHI_HP_CUT - 0.25) < 1e-12


# ---------------------------------------------------------------------------
# _qc_chi_final — per-probe spectral QC before the mixing products (K_T/Gamma)
# ---------------------------------------------------------------------------


class TestQcChiFinal:
    """The mixing-quantity chi drops poorly-fit probes, unlike the raw
    geometric-mean chi_final that the L4_chi output carries."""

    def test_high_fom_probe_excluded(self):
        """A probe with fom above the limit is dropped from the geometric mean.

        On the OLD code the mixing call used the unfiltered all-probe mean, so
        the bad probe pulled the result toward sqrt(1e-8 * 1e-6); the QC mean
        keeps only the good probe.
        """
        from odas_tpw.rsi.pipeline import _CHI_FOM_LIMIT, _qc_chi_final

        chi = np.array([[1e-8], [1e-6]])  # 2 probes, 1 window
        fom = np.array([[1.0], [5.0]])  # probe 1 good, probe 2 bad fom
        kmr = np.array([[0.8], [0.8]])
        out = _qc_chi_final(chi, fom, kmr)
        assert out[0] == pytest.approx(1e-8)  # only the good probe
        # differs from the unfiltered all-probe geometric mean
        raw = np.exp(np.mean(np.log(chi[:, 0])))
        assert abs(out[0] - raw) > 1e-9
        assert pytest.approx(1.15) == _CHI_FOM_LIMIT

    def test_low_k_max_ratio_probe_excluded(self):
        """A probe with K_max_ratio below 0.5 (mostly extrapolated) is dropped."""
        from odas_tpw.rsi.pipeline import _qc_chi_final

        chi = np.array([[1e-8], [1e-6]])
        fom = np.array([[1.0], [1.0]])
        kmr = np.array([[0.8], [0.3]])  # probe 2 under-resolved
        out = _qc_chi_final(chi, fom, kmr)
        assert out[0] == pytest.approx(1e-8)

    def test_both_probes_pass_is_geometric_mean(self):
        """When every probe passes QC the result is the plain geometric mean."""
        from odas_tpw.rsi.pipeline import _qc_chi_final

        chi = np.array([[1e-8], [1e-6]])
        fom = np.array([[1.0], [1.0]])
        kmr = np.array([[0.8], [0.8]])
        out = _qc_chi_final(chi, fom, kmr)
        assert out[0] == pytest.approx(np.sqrt(1e-8 * 1e-6))

    def test_fallback_when_no_probe_passes(self):
        """No window is silently lost: fall back to the all-finite-probe mean."""
        from odas_tpw.rsi.pipeline import _qc_chi_final

        chi = np.array([[1e-8], [1e-6]])
        fom = np.array([[5.0], [5.0]])  # both fail fom
        kmr = np.array([[0.8], [0.8]])
        out = _qc_chi_final(chi, fom, kmr)
        assert out[0] == pytest.approx(np.sqrt(1e-8 * 1e-6))

    def test_nan_chi_window_stays_nan(self):
        """A window with no finite chi>0 stays NaN (no spurious estimate)."""
        from odas_tpw.rsi.pipeline import _qc_chi_final

        chi = np.array([[np.nan], [-1.0]])
        fom = np.array([[1.0], [1.0]])
        kmr = np.array([[0.8], [0.8]])
        out = _qc_chi_final(chi, fom, kmr)
        assert np.isnan(out[0])
