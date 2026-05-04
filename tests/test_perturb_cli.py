# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb.cli — subcommand argument parsing and init."""

from unittest.mock import patch

import pytest

from odas_tpw.perturb.cli import _load_and_merge, build_parser, main


class TestBuildParser:
    def test_all_subcommands_exist(self):
        parser = build_parser()
        # Parser should have all expected subcommands
        # Parse each one to verify it's registered
        expected = [
            "init",
            "run",
            "trim",
            "merge",
            "profiles",
            "diss",
            "chi",
            "ctd",
            "bin",
            "combo",
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
        for section in [
            "files:",
            "gps:",
            "profiles:",
            "fp07:",
            "ct:",
            "bottom:",
            "top_trim:",
            "epsilon:",
            "chi:",
            "ctd:",
            "binning:",
            "netcdf:",
            "parallel:",
        ]:
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
        with patch("odas_tpw.perturb.pipeline.run_pipeline") as mock_rp:
            main(["run", "-c", str(cfg), "-o", "myout/", "--p-file-root", "myroot/"])
        called_config = mock_rp.call_args[0][0]
        assert called_config["files"]["output_root"] == "myout/"
        assert called_config["files"]["p_file_root"] == "myroot/"


class TestCmdTrim:
    def test_dispatches(self, tmp_path):
        """Verify trim subcommand dispatches to run_trim."""
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("files:\n  p_file_root: VMP/\n")
        with patch("odas_tpw.perturb.pipeline.run_trim", return_value=[]) as mock_rt:
            main(["trim", "-c", str(cfg)])
        mock_rt.assert_called_once()


class TestCmdMerge:
    def test_dispatches(self, tmp_path):
        """Verify merge subcommand dispatches to run_merge."""
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("files:\n  p_file_root: VMP/\n")
        with patch("odas_tpw.perturb.pipeline.run_merge", return_value=[]) as mock_rm:
            main(["merge", "-c", str(cfg)])
        mock_rm.assert_called_once()


class TestCmdBin:
    def test_empty_dirs(self, tmp_path, capsys):
        """With no matching dirs, bin should print completion message and not crash."""
        main(["bin", "-o", str(tmp_path)])
        captured = capsys.readouterr()
        assert "Binning complete." in captured.out

    def test_bin_writes_signatures_with_full_upstream_chain(self, tmp_path):
        """`perturb bin` with populated upstream dirs writes binned dirs whose
        ``.params_sha256_*`` includes the full upstream chain — covers the
        three ``_upstream_for(...)`` call sites in ``_cmd_bin`` (profiles,
        diss, chi)."""
        import json

        import numpy as np
        import xarray as xr

        # Build a minimal profile NetCDF the binner will accept.
        prof = tmp_path / "profiles_00"
        prof.mkdir()
        depth = np.linspace(1.0, 5.0, 5)
        prof_ds = xr.Dataset(
            {
                "depth": (["time"], depth),
                "T1": (["time"], np.ones(5)),
                "lat": ((), 10.0),
                "lon": ((), 100.0),
                "stime": ((), 0.0),
                "etime": ((), 5.0),
            },
        )
        prof_ds.to_netcdf(prof / "p001.nc")

        # Diss profile: bin_diss expects 1-D epsilonMean(depth) per profile NC.
        diss = tmp_path / "diss_00"
        diss.mkdir()
        diss_ds = xr.Dataset(
            {
                "depth": (["time"], depth),
                "epsilonMean": (["time"], np.full(5, 1e-9)),
                "lat": ((), 10.0),
                "lon": ((), 100.0),
                "stime": ((), 0.0),
                "etime": ((), 5.0),
            },
        )
        diss_ds.to_netcdf(diss / "p001.nc")

        # Chi profile: bin_chi expects 1-D chi/kB per depth.
        chi = tmp_path / "chi_00"
        chi.mkdir()
        chi_ds = xr.Dataset(
            {
                "depth": (["time"], depth),
                "chi_1": (["time"], np.full(5, 1e-9)),
                "kB_1": (["time"], np.full(5, 100.0)),
                "lat": ((), 10.0),
                "lon": ((), 100.0),
                "stime": ((), 0.0),
                "etime": ((), 5.0),
            },
        )
        chi_ds.to_netcdf(chi / "p001.nc")

        main(["bin", "-o", str(tmp_path)])

        for sub, must_contain in (
            ("profiles_binned_00", {"binning", "profiles"}),
            ("diss_binned_00", {"binning", "epsilon", "profiles"}),
            ("chi_binned_00", {"binning", "chi", "epsilon", "profiles"}),
        ):
            sigs = list((tmp_path / sub).glob(".params_sha256_*"))
            assert len(sigs) == 1, f"{sub} should have a signature"
            body = json.loads(sigs[0].read_text())
            assert must_contain.issubset(body.keys()), (
                f"{sub} missing sections: {must_contain - body.keys()}"
            )


class TestCmdCombo:
    def test_empty_dirs(self, tmp_path, capsys):
        """With no matching dirs, combo should print completion message and not crash."""
        main(["combo", "-o", str(tmp_path)])
        captured = capsys.readouterr()
        assert "Combo assembly complete." in captured.out

    def test_combo_writes_signature_for_each_combo_dir(self, tmp_path):
        """`perturb combo` writes a `.params_sha256_*` file in each combo dir
        it produces — covers all four write_signature branches in
        ``_cmd_combo`` (combo / diss_combo / chi_combo / ctd_combo)."""
        import numpy as np
        import xarray as xr

        # Create one populated upstream binned dir for each of the four
        # combo flavours; the schema is loose enough that combo will accept
        # any 2-D var with bin/profile dims.
        for sub in ("profiles_binned_00", "diss_binned_00", "chi_binned_00"):
            d = tmp_path / sub
            d.mkdir()
            ds = xr.Dataset(
                {"depth": (["bin", "profile"], np.ones((3, 1)))},
                coords={"bin": np.arange(3.0), "profile": np.arange(1)},
            )
            ds.to_netcdf(d / "binned.nc")

        ctd = tmp_path / "ctd_00"
        ctd.mkdir()
        ds = xr.Dataset(
            {"JAC_T": (["time"], np.arange(5.0))},
            coords={"time": np.arange(5.0)},
        )
        ds.to_netcdf(ctd / "file.nc")

        main(["combo", "-o", str(tmp_path)])

        for combo_name in ("combo", "diss_combo", "chi_combo", "ctd_combo"):
            sigs = list((tmp_path / combo_name).glob(".params_sha256_*"))
            assert len(sigs) == 1, f"{combo_name} should have exactly one signature file"

    def test_combo_follows_binning_method_time(self, tmp_path):
        """When ``binning.method=time`` is set in the config, ``perturb
        combo`` must glue the diss/chi/profile combos lengthwise (CF
        ``featureType=trajectory``) — the layout for moored/AUV runs.

        Sanity-check against the depth default which produces ``profile``."""
        import numpy as np
        import xarray as xr
        import yaml

        prof_dir = tmp_path / "profiles_binned_00"
        prof_dir.mkdir()
        for i, t0 in enumerate([0.0, 10.0]):
            ds = xr.Dataset(
                {"T": (["time"], np.arange(5.0))},
                coords={"time": t0 + np.arange(5.0)},
            )
            ds.to_netcdf(prof_dir / f"f{i}.nc")

        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(yaml.safe_dump({"binning": {"method": "time"}}))

        main(["combo", "-c", str(cfg), "-o", str(tmp_path)])

        combo = xr.open_dataset(tmp_path / "combo" / "combo.nc")
        assert combo.attrs["featureType"] == "trajectory"
        assert combo.sizes["time"] == 10
        combo.close()


class TestCmdProfiles:
    def test_dispatches(self):
        """profiles subcommand dispatches to run_pipeline with trim/merge disabled."""
        with patch("odas_tpw.perturb.pipeline.run_pipeline") as mock_rp:
            main(["profiles", "/nonexistent_xyz/*.p"])
        called_config = mock_rp.call_args[0][0]
        assert called_config["files"]["trim"] is False
        assert called_config["files"]["merge"] is False


class TestCmdDiss:
    def test_dispatches(self):
        """diss subcommand dispatches to run_pipeline."""
        with patch("odas_tpw.perturb.pipeline.run_pipeline") as mock_rp:
            main(["diss", "/nonexistent_xyz/*.p"])
        mock_rp.assert_called_once()


class TestCmdChi:
    def test_dispatches(self):
        """chi subcommand enables chi.enable=True in config."""
        with patch("odas_tpw.perturb.pipeline.run_pipeline") as mock_rp:
            main(["chi", "/nonexistent_xyz/*.p"])
        called_config = mock_rp.call_args[0][0]
        assert called_config["chi"]["enable"] is True


class TestCmdCtd:
    def test_dispatches(self):
        """ctd subcommand enables ctd.enable=True in config."""
        with patch("odas_tpw.perturb.pipeline.run_pipeline") as mock_rp:
            main(["ctd", "/nonexistent_xyz/*.p"])
        called_config = mock_rp.call_args[0][0]
        assert called_config["ctd"]["enable"] is True
