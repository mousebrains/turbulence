# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb.cli — subcommand argument parsing and init."""

from unittest.mock import patch

import pytest

from perturb.cli import _load_and_merge, build_parser, main


class TestBuildParser:
    def test_all_subcommands_exist(self):
        parser = build_parser()
        # Parser should have all expected subcommands
        # Parse each one to verify it's registered
        expected = [
            "init", "run", "trim", "merge", "profiles",
            "diss", "chi", "ctd", "bin", "combo",
        ]
        for cmd in expected:
            args = parser.parse_args([cmd])
            assert args.command == cmd

    def test_init_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["init"])
        assert args.path == "config.yaml"
        assert args.force is False

    def test_init_custom_path(self):
        parser = build_parser()
        args = parser.parse_args(["init", "my_config.yaml"])
        assert args.path == "my_config.yaml"

    def test_init_force(self):
        parser = build_parser()
        args = parser.parse_args(["init", "--force"])
        assert args.force is True

    def test_run_common_args(self):
        parser = build_parser()
        args = parser.parse_args(["run", "-c", "config.yaml", "-o", "out/", "-j", "4"])
        assert args.config == "config.yaml"
        assert args.output == "out/"
        assert args.jobs == 4

    def test_run_file_args(self):
        parser = build_parser()
        args = parser.parse_args(["run", "VMP/*.p"])
        assert args.files == ["VMP/*.p"]

    def test_bin_has_no_jobs(self):
        parser = build_parser()
        args = parser.parse_args(["bin", "-c", "config.yaml"])
        assert args.config == "config.yaml"
        assert not hasattr(args, "jobs")

    def test_combo_has_no_jobs(self):
        parser = build_parser()
        args = parser.parse_args(["combo", "-o", "out/"])
        assert args.output == "out/"
        assert not hasattr(args, "jobs")

    def test_no_command_exits_zero(self):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 0


class TestCmdInit:
    def test_creates_template(self, tmp_path):
        out = tmp_path / "test_config.yaml"
        main(["init", str(out)])
        assert out.exists()
        content = out.read_text()
        # Should contain all 13 sections
        for section in ["files:", "gps:", "profiles:", "fp07:", "ct:", "bottom:",
                        "top_trim:", "epsilon:", "chi:", "ctd:", "binning:",
                        "netcdf:", "parallel:"]:
            assert section in content

    def test_refuses_overwrite(self, tmp_path):
        out = tmp_path / "exists.yaml"
        out.write_text("existing")
        with pytest.raises(SystemExit):
            main(["init", str(out)])

    def test_force_overwrites(self, tmp_path):
        out = tmp_path / "exists.yaml"
        out.write_text("existing")
        main(["init", str(out), "--force"])
        assert "files:" in out.read_text()


class TestLoadAndMerge:
    def test_none_returns_empty(self):
        assert _load_and_merge(None) == {}

    def test_loads_config(self, tmp_path):
        cfg = tmp_path / "test.yaml"
        cfg.write_text("files:\n  p_file_root: myroot/\n")
        result = _load_and_merge(str(cfg))
        assert result["files"]["p_file_root"] == "myroot/"


class TestCmdRun:
    def test_no_files_exits(self):
        """Explicit glob that matches nothing should exit with error."""
        with pytest.raises(SystemExit) as exc_info:
            main(["run", "/nonexistent_dir_xyz/*.p"])
        assert exc_info.value.code == 1

    def test_config_overrides(self, tmp_path):
        """CLI --output and --p-file-root override config values."""
        cfg = tmp_path / "test.yaml"
        cfg.write_text("files:\n  p_file_root: original/\n")
        with patch("perturb.pipeline.run_pipeline") as mock_rp:
            main(["run", "-c", str(cfg), "-o", "myout/", "--p-file-root", "myroot/"])
        called_config = mock_rp.call_args[0][0]
        assert called_config["files"]["output_root"] == "myout/"
        assert called_config["files"]["p_file_root"] == "myroot/"


class TestCmdTrim:
    def test_dispatches(self, tmp_path):
        """Verify trim subcommand dispatches to run_trim."""
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("files:\n  p_file_root: VMP/\n")
        with patch("perturb.pipeline.run_trim", return_value=[]) as mock_rt:
            main(["trim", "-c", str(cfg)])
        mock_rt.assert_called_once()


class TestCmdMerge:
    def test_dispatches(self, tmp_path):
        """Verify merge subcommand dispatches to run_merge."""
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("files:\n  p_file_root: VMP/\n")
        with patch("perturb.pipeline.run_merge", return_value=[]) as mock_rm:
            main(["merge", "-c", str(cfg)])
        mock_rm.assert_called_once()


class TestCmdBin:
    def test_empty_dirs(self, tmp_path, capsys):
        """With no matching dirs, bin should print completion message and not crash."""
        main(["bin", "-o", str(tmp_path)])
        captured = capsys.readouterr()
        assert "Binning complete." in captured.out


class TestCmdCombo:
    def test_empty_dirs(self, tmp_path, capsys):
        """With no matching dirs, combo should print completion message and not crash."""
        main(["combo", "-o", str(tmp_path)])
        captured = capsys.readouterr()
        assert "Combo assembly complete." in captured.out
