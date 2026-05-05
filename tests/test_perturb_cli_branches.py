# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Branch-coverage tests for perturb.cli — flag passthrough and dispatch edges."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from odas_tpw.perturb.cli import _glob_p_files, main

# ---------------------------------------------------------------------------
# _glob_p_files — filtering branches (lines 73-75)
# ---------------------------------------------------------------------------


class TestGlobPFiles:
    def test_returns_none_for_empty_patterns(self):
        assert _glob_p_files(None) is None
        assert _glob_p_files([]) is None

    def test_skips_non_p_suffix(self, tmp_path):
        """Files matching glob but not ending in .p are filtered out."""
        (tmp_path / "good.p").write_bytes(b"x")
        (tmp_path / "bad.txt").write_text("noise")
        result = _glob_p_files([str(tmp_path / "*")])
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "good.p"

    def test_skips_directories(self, tmp_path):
        """A directory matching glob is not included."""
        (tmp_path / "real.p").write_bytes(b"x")
        (tmp_path / "dir.p").mkdir()  # directory, not file
        result = _glob_p_files([str(tmp_path / "*")])
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "real.p"

    def test_no_matches_returns_none(self, tmp_path):
        """When pattern matches nothing → returns None (not empty list)."""
        result = _glob_p_files([str(tmp_path / "nonexistent_*.p")])
        assert result is None


# ---------------------------------------------------------------------------
# _run_analysis flag passthrough (covers _cmd_profiles/_cmd_diss/_cmd_chi/_cmd_ctd)
# ---------------------------------------------------------------------------


class TestRunAnalysisFlags:
    def test_profiles_with_output_and_jobs(self, tmp_path):
        """Lines 87, 89 — --output and --jobs propagate to config."""
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("files:\n  p_file_root: VMP/\n")
        with patch("odas_tpw.perturb.pipeline.run_pipeline") as mock_rp:
            main(["profiles", "-c", str(cfg), "-o", str(tmp_path / "out"), "-j", "4"])
        called_config = mock_rp.call_args[0][0]
        assert called_config["files"]["output_root"] == str(tmp_path / "out")
        assert called_config["parallel"]["jobs"] == 4

    def test_diss_with_output(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("files:\n  p_file_root: VMP/\n")
        with patch("odas_tpw.perturb.pipeline.run_pipeline") as mock_rp:
            main(["diss", "-c", str(cfg), "-o", str(tmp_path / "diss_out")])
        called_config = mock_rp.call_args[0][0]
        assert called_config["files"]["output_root"] == str(tmp_path / "diss_out")

    def test_chi_with_jobs(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("files:\n  p_file_root: VMP/\n")
        with patch("odas_tpw.perturb.pipeline.run_pipeline") as mock_rp:
            main(["chi", "-c", str(cfg), "-j", "2"])
        called_config = mock_rp.call_args[0][0]
        assert called_config["parallel"]["jobs"] == 2


# ---------------------------------------------------------------------------
# _cmd_run flag passthrough (lines 136, 144-145)
# ---------------------------------------------------------------------------


class TestCmdRunFlags:
    def test_run_with_jobs(self, tmp_path):
        """Line 136 — `run --jobs` propagates to parallel config."""
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("files:\n  p_file_root: VMP/\n")
        with patch("odas_tpw.perturb.pipeline.run_pipeline") as mock_rp:
            main(["run", "-c", str(cfg), "-j", "8"])
        called_config = mock_rp.call_args[0][0]
        assert called_config["parallel"]["jobs"] == 8

    def test_run_with_hotel_file(self, tmp_path):
        """Lines 144-145 — `run --hotel-file` enables hotel and sets file."""
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("files:\n  p_file_root: VMP/\n")
        hotel = tmp_path / "hotel.csv"
        hotel.write_text("time,speed\n0,0.5\n")
        with patch("odas_tpw.perturb.pipeline.run_pipeline") as mock_rp:
            main(["run", "-c", str(cfg), "--hotel-file", str(hotel)])
        called_config = mock_rp.call_args[0][0]
        assert called_config["hotel"]["enable"] is True
        assert called_config["hotel"]["file"] == str(hotel)


# ---------------------------------------------------------------------------
# _cmd_trim / _cmd_merge flag passthrough (lines 159, 161, 176, 178)
# ---------------------------------------------------------------------------


class TestCmdTrimMergeFlags:
    def test_trim_with_p_file_root_and_output(self, tmp_path):
        """Lines 159, 161 — trim accepts --p-file-root and --output."""
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("files:\n  p_file_root: ignored/\n")
        with patch("odas_tpw.perturb.pipeline.run_trim", return_value=[]) as mock_rt:
            main([
                "trim",
                "-c",
                str(cfg),
                "--p-file-root",
                str(tmp_path / "input"),
                "-o",
                str(tmp_path / "output"),
            ])
        config = mock_rt.call_args[0][0]
        assert config["files"]["p_file_root"] == str(tmp_path / "input")
        assert config["files"]["output_root"] == str(tmp_path / "output")

    def test_merge_with_p_file_root_and_output(self, tmp_path):
        """Lines 176, 178 — merge accepts --p-file-root and --output."""
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("files:\n  p_file_root: ignored/\n")
        with patch("odas_tpw.perturb.pipeline.run_merge", return_value=[]) as mock_rm:
            main([
                "merge",
                "-c",
                str(cfg),
                "--p-file-root",
                str(tmp_path / "input"),
                "-o",
                str(tmp_path / "output"),
            ])
        config = mock_rm.call_args[0][0]
        assert config["files"]["p_file_root"] == str(tmp_path / "input")
        assert config["files"]["output_root"] == str(tmp_path / "output")


# ---------------------------------------------------------------------------
# main() dispatch error path (lines 557-558)
# ---------------------------------------------------------------------------


class TestMainDispatchUnknown:
    def test_unknown_command_via_argparse_exits(self):
        """argparse rejects unknown commands and exits with code 2."""
        with pytest.raises(SystemExit) as exc_info:
            main(["bogus_command"])
        # argparse uses code 2 for unknown choices
        assert exc_info.value.code in (1, 2)


# ---------------------------------------------------------------------------
# _cmd_run with no matching files exits cleanly
# ---------------------------------------------------------------------------


class TestCmdRunNoFilesPattern:
    def test_explicit_glob_with_no_matches_exits(self, tmp_path):
        """When --files glob matches nothing → exit 1 with error msg."""
        with pytest.raises(SystemExit) as exc_info:
            main(["run", str(tmp_path / "nonexistent_*.p")])
        assert exc_info.value.code == 1
