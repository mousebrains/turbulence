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
