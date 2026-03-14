# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for the rsi-tpw CLI."""

import sys
from types import SimpleNamespace

import pytest


def test_help_exits_cleanly(monkeypatch):
    """rsi-tpw --help should exit with code 0."""
    monkeypatch.setattr(sys, "argv", ["rsi-tpw", "--help"])
    from microstructure_tpw.rsi.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0


def test_init_creates_template(monkeypatch, tmp_path):
    """rsi-tpw init should produce a valid YAML file."""
    out_file = tmp_path / "config.yaml"
    monkeypatch.setattr(sys, "argv", ["rsi-tpw", "init", str(out_file)])
    from microstructure_tpw.rsi.cli import main

    main()
    assert out_file.exists()

    from ruamel.yaml import YAML

    yaml = YAML()
    with open(out_file) as fh:
        data = yaml.load(fh)
    assert "epsilon" in data
    assert "chi" in data
    assert "profiles" in data


def test_init_refuses_overwrite(monkeypatch, tmp_path):
    """rsi-tpw init should refuse to overwrite without --force."""
    out_file = tmp_path / "config.yaml"
    out_file.write_text("existing")
    monkeypatch.setattr(sys, "argv", ["rsi-tpw", "init", str(out_file)])
    from microstructure_tpw.rsi.cli import main

    with pytest.raises(SystemExit):
        main()
    assert out_file.read_text() == "existing"


def test_init_force_overwrites(monkeypatch, tmp_path):
    """rsi-tpw init --force should overwrite existing file."""
    out_file = tmp_path / "config.yaml"
    out_file.write_text("existing")
    monkeypatch.setattr(sys, "argv", ["rsi-tpw", "init", "--force", str(out_file)])
    from microstructure_tpw.rsi.cli import main

    main()
    assert "rsi-tpw configuration" in out_file.read_text()


def test_config_flag_accepted(monkeypatch, tmp_path):
    """-c config.yaml should be parsed without error (even if subcommand needs files)."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("epsilon:\n  fft_length: 512\n")
    # Just verify that parsing works — the eps command will fail for lack of files,
    # but should get past the argparse stage.
    monkeypatch.setattr(
        sys, "argv", ["rsi-tpw", "-c", str(cfg), "eps", "-o", str(tmp_path), "nonexistent.p"]
    )
    from microstructure_tpw.rsi.cli import main

    with pytest.raises(SystemExit):
        # Will exit because nonexistent.p doesn't exist
        main()


def test_output_required_eps(monkeypatch):
    """rsi-tpw eps without -o should error."""
    monkeypatch.setattr(sys, "argv", ["rsi-tpw", "eps", "VMP/*.p"])
    from microstructure_tpw.rsi.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2  # argparse error


def test_output_required_chi(monkeypatch):
    """rsi-tpw chi without -o should error."""
    monkeypatch.setattr(sys, "argv", ["rsi-tpw", "chi", "VMP/*.p"])
    from microstructure_tpw.rsi.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2


def test_output_required_prof(monkeypatch):
    """rsi-tpw prof without -o should error."""
    monkeypatch.setattr(sys, "argv", ["rsi-tpw", "prof", "VMP/*.p"])
    from microstructure_tpw.rsi.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2


def test_output_required_pipeline(monkeypatch):
    """rsi-tpw pipeline without -o should error."""
    monkeypatch.setattr(sys, "argv", ["rsi-tpw", "pipeline", "VMP/*.p"])
    from microstructure_tpw.rsi.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2


def test_no_config_same_behavior(monkeypatch, tmp_path):
    """Omitting -c should give the same merged params as defaults."""
    from microstructure_tpw.rsi.cli import _merge_for_section
    from microstructure_tpw.rsi.config import DEFAULTS

    # Create a minimal args namespace with all None (no CLI overrides, no config)
    args = type("Args", (), {"config": None})()
    for key in DEFAULTS["epsilon"]:
        setattr(args, key, None)
    args.no_goodman = None

    merged = _merge_for_section(args, "epsilon")
    # Should match defaults with None values stripped
    expected = {k: v for k, v in DEFAULTS["epsilon"].items() if v is not None}
    assert merged == expected


# ---------------------------------------------------------------------------
# _extract_cli_overrides tests
# ---------------------------------------------------------------------------


class TestExtractCliOverrides:
    def test_profiles_section(self):
        from microstructure_tpw.rsi.cli import _extract_cli_overrides

        args = SimpleNamespace(P_min=1.0, W_min=None, direction="up", min_duration=None)
        overrides = _extract_cli_overrides(args, "profiles")
        assert overrides == {"P_min": 1.0, "direction": "up"}

    def test_epsilon_section(self):
        from microstructure_tpw.rsi.cli import _extract_cli_overrides

        args = SimpleNamespace(
            fft_length=512,
            diss_length=None,
            overlap=None,
            speed=None,
            direction="up",
            f_AA=None,
            salinity=34.5,
            no_goodman=None,
        )
        overrides = _extract_cli_overrides(args, "epsilon")
        assert overrides == {"fft_length": 512, "direction": "up", "salinity": 34.5}
        assert "goodman" not in overrides  # no_goodman is None

    def test_epsilon_no_goodman_inverts(self):
        from microstructure_tpw.rsi.cli import _extract_cli_overrides

        args = SimpleNamespace(
            fft_length=None,
            diss_length=None,
            overlap=None,
            speed=None,
            direction=None,
            f_AA=None,
            salinity=None,
            no_goodman=True,
        )
        overrides = _extract_cli_overrides(args, "epsilon")
        assert overrides["goodman"] is False

    def test_chi_section(self):
        from microstructure_tpw.rsi.cli import _extract_cli_overrides

        args = SimpleNamespace(
            fft_length=1024,
            diss_length=None,
            overlap=None,
            speed=None,
            direction=None,
            fp07_model="double_pole",
            f_AA=None,
            fit_method="iterative",
            spectrum_model=None,
            salinity=None,
        )
        overrides = _extract_cli_overrides(args, "chi")
        assert overrides == {
            "fft_length": 1024,
            "fp07_model": "double_pole",
            "fit_method": "iterative",
        }

    def test_epsilon_pipeline_section(self):
        from microstructure_tpw.rsi.cli import _extract_cli_overrides

        args = SimpleNamespace(
            eps_fft_length=512,
            direction=None,
            f_AA=50.0,
            speed=None,
            salinity=None,
            no_goodman=True,
        )
        overrides = _extract_cli_overrides(args, "epsilon_pipeline")
        assert overrides == {"fft_length": 512, "f_AA": 50.0, "goodman": False}

    def test_chi_pipeline_section(self):
        from microstructure_tpw.rsi.cli import _extract_cli_overrides

        args = SimpleNamespace(
            chi_fft_length=256,
            direction="up",
            f_AA=None,
            fp07_model=None,
            spectrum_model="kraichnan",
            speed=None,
            salinity=None,
        )
        overrides = _extract_cli_overrides(args, "chi_pipeline")
        assert overrides == {
            "fft_length": 256,
            "direction": "up",
            "spectrum_model": "kraichnan",
        }

    def test_unknown_section_returns_empty(self):
        from microstructure_tpw.rsi.cli import _extract_cli_overrides

        args = SimpleNamespace()
        assert _extract_cli_overrides(args, "nonexistent") == {}

    def test_all_none_returns_empty(self):
        from microstructure_tpw.rsi.cli import _extract_cli_overrides

        args = SimpleNamespace(
            fft_length=None,
            diss_length=None,
            overlap=None,
            speed=None,
            direction=None,
            f_AA=None,
            salinity=None,
            no_goodman=None,
        )
        overrides = _extract_cli_overrides(args, "epsilon")
        assert overrides == {}


# ---------------------------------------------------------------------------
# _merge_for_section tests
# ---------------------------------------------------------------------------


class TestMergeForSection:
    def test_merge_with_config_file(self, tmp_path):
        from microstructure_tpw.rsi.cli import _merge_for_section

        cfg = tmp_path / "config.yaml"
        cfg.write_text("epsilon:\n  fft_length: 512\n  salinity: 34.5\n")

        args = SimpleNamespace(config=str(cfg))
        for key in [
            "fft_length",
            "diss_length",
            "overlap",
            "speed",
            "direction",
            "f_AA",
            "salinity",
        ]:
            setattr(args, key, None)
        args.no_goodman = None

        merged = _merge_for_section(args, "epsilon")
        assert merged["fft_length"] == 512
        assert merged["salinity"] == 34.5

    def test_cli_overrides_config_file(self, tmp_path):
        from microstructure_tpw.rsi.cli import _merge_for_section

        cfg = tmp_path / "config.yaml"
        cfg.write_text("epsilon:\n  fft_length: 512\n")

        args = SimpleNamespace(config=str(cfg))
        for key in ["diss_length", "overlap", "speed", "direction", "f_AA", "salinity"]:
            setattr(args, key, None)
        args.fft_length = 1024  # CLI override
        args.no_goodman = None

        merged = _merge_for_section(args, "epsilon")
        assert merged["fft_length"] == 1024

    def test_pipeline_pseudo_section(self, tmp_path):
        from microstructure_tpw.rsi.cli import _merge_for_section

        cfg = tmp_path / "config.yaml"
        cfg.write_text("epsilon:\n  fft_length: 512\n")

        args = SimpleNamespace(config=str(cfg))
        args.eps_fft_length = None
        args.direction = None
        args.f_AA = None
        args.speed = None
        args.salinity = None
        args.no_goodman = None

        merged = _merge_for_section(args, "epsilon_pipeline")
        assert merged["fft_length"] == 512  # from config file epsilon section

    def test_no_config_uses_defaults(self):
        from microstructure_tpw.rsi.cli import _merge_for_section
        from microstructure_tpw.rsi.config import DEFAULTS

        args = SimpleNamespace(config=None)
        for key in DEFAULTS["chi"]:
            setattr(args, key, None)

        merged = _merge_for_section(args, "chi")
        assert merged["fft_length"] == 512
        assert merged["fp07_model"] == "single_pole"

    def test_profiles_section(self, tmp_path):
        from microstructure_tpw.rsi.cli import _merge_for_section

        cfg = tmp_path / "config.yaml"
        cfg.write_text("profiles:\n  P_min: 1.0\n  min_duration: 10.0\n")

        args = SimpleNamespace(config=str(cfg))
        args.P_min = None
        args.W_min = None
        args.direction = None
        args.min_duration = None

        merged = _merge_for_section(args, "profiles")
        assert merged["P_min"] == 1.0
        assert merged["min_duration"] == 10.0
        assert merged["direction"] == "down"  # default


# ---------------------------------------------------------------------------
# _setup_output_dir tests
# ---------------------------------------------------------------------------


class TestSetupOutputDir:
    def test_creates_dir_with_signature_file_and_config(self, tmp_path):
        from microstructure_tpw.rsi.cli import _setup_output_dir

        args = SimpleNamespace(output=str(tmp_path))
        d = _setup_output_dir(args, "eps", "epsilon", {"fft_length": 256})
        assert d.is_dir()
        assert d.name == "eps_00"
        assert (d / "config.yaml").exists()
        assert list(d.glob(".params_sha256_*"))

    def test_with_upstream(self, tmp_path):
        from microstructure_tpw.rsi.cli import _setup_output_dir

        args = SimpleNamespace(output=str(tmp_path))
        upstream = [("epsilon", {"fft_length": 256})]
        d = _setup_output_dir(args, "chi", "chi", {"fft_length": 512}, upstream=upstream)
        assert d.name == "chi_00"
        # config.yaml should have both sections
        import json

        signature_files = list(d.glob(".params_sha256_*"))
        content = json.loads(signature_files[0].read_text())
        assert "epsilon" in content
        assert "chi" in content

    def test_pipeline_section_stripped(self, tmp_path):
        from microstructure_tpw.rsi.cli import _setup_output_dir

        args = SimpleNamespace(output=str(tmp_path))
        d = _setup_output_dir(args, "eps", "epsilon_pipeline", {"fft_length": 256})
        assert d.name == "eps_00"
        # config.yaml should use "epsilon" not "epsilon_pipeline"
        from ruamel.yaml import YAML

        yaml = YAML()
        with open(d / "config.yaml") as fh:
            data = yaml.load(fh)
        assert "epsilon" in data
        assert "epsilon_pipeline" not in data


# ---------------------------------------------------------------------------
# _load_file_config tests
# ---------------------------------------------------------------------------


class TestLoadFileConfig:
    def test_no_config_returns_empty(self):
        from microstructure_tpw.rsi.cli import _load_file_config

        args = SimpleNamespace(config=None)
        assert _load_file_config(args) == {}

    def test_missing_attr_returns_empty(self):
        from microstructure_tpw.rsi.cli import _load_file_config

        args = SimpleNamespace()
        assert _load_file_config(args) == {}

    def test_loads_config(self, tmp_path):
        from microstructure_tpw.rsi.cli import _load_file_config

        cfg = tmp_path / "config.yaml"
        cfg.write_text("epsilon:\n  fft_length: 512\n")
        args = SimpleNamespace(config=str(cfg))
        config = _load_file_config(args)
        assert config["epsilon"]["fft_length"] == 512


# ---------------------------------------------------------------------------
# init default path test
# ---------------------------------------------------------------------------


def test_init_default_path(monkeypatch, tmp_path):
    """rsi-tpw init with no args should default to config.yaml."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["rsi-tpw", "init"])
    from microstructure_tpw.rsi.cli import main

    main()
    assert (tmp_path / "config.yaml").exists()


# ---------------------------------------------------------------------------
# _resolve_p_files / _resolve_files tests
# ---------------------------------------------------------------------------


def test_resolve_p_files_single(sample_p_file):
    """_resolve_p_files with a real .p file path should return it."""
    from microstructure_tpw.rsi.cli import _resolve_p_files

    result = _resolve_p_files([str(sample_p_file)])
    assert len(result) == 1
    assert result[0].name == sample_p_file.name


def test_resolve_p_files_no_match():
    """_resolve_p_files with no matching glob should exit(1)."""
    from microstructure_tpw.rsi.cli import _resolve_p_files

    with pytest.raises(SystemExit) as exc_info:
        _resolve_p_files(["nonexistent_dir_xyz/*.p"])
    assert exc_info.value.code == 1


def test_resolve_files_with_extensions(sample_p_file):
    """_resolve_files should filter by extension set."""
    from microstructure_tpw.rsi.cli import _resolve_files

    result = _resolve_files([str(sample_p_file)], extensions={".p"})
    assert len(result) == 1
    assert result[0].suffix == ".p"


def test_resolve_files_no_match():
    """_resolve_files with no matching files should exit(1)."""
    from microstructure_tpw.rsi.cli import _resolve_files

    with pytest.raises(SystemExit) as exc_info:
        _resolve_files(["nonexistent_dir_xyz/*"], extensions={".p", ".nc"})
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# CLI command execution tests (real data)
# ---------------------------------------------------------------------------


def test_cmd_info(monkeypatch, sample_p_file, capsys):
    """rsi-tpw info should print file summary."""
    monkeypatch.setattr(sys, "argv", ["rsi-tpw", "info", str(sample_p_file)])
    from microstructure_tpw.rsi.cli import main

    main()
    captured = capsys.readouterr()
    assert "SN479" in captured.out or "479" in captured.out


def test_cmd_nc_to_dir(monkeypatch, sample_p_file, tmp_path):
    """rsi-tpw nc <file> -o <dir> should create a .nc file."""
    out_dir = tmp_path / "nc_out"
    out_dir.mkdir()
    monkeypatch.setattr(sys, "argv", ["rsi-tpw", "nc", str(sample_p_file), "-o", str(out_dir)])
    from microstructure_tpw.rsi.cli import main

    main()
    nc_files = list(out_dir.glob("*.nc"))
    assert len(nc_files) == 1


def test_cmd_nc_to_file(monkeypatch, sample_p_file, tmp_path):
    """rsi-tpw nc with single file and explicit output path."""
    out_file = tmp_path / "output.nc"
    monkeypatch.setattr(sys, "argv", ["rsi-tpw", "nc", str(sample_p_file), "-o", str(out_file)])
    from microstructure_tpw.rsi.cli import main

    main()
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_cmd_prof(monkeypatch, sample_p_file, tmp_path):
    """rsi-tpw prof should extract profiles to sequential output dir."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["rsi-tpw", "prof", str(sample_p_file), "-o", str(tmp_path)],
    )
    from microstructure_tpw.rsi.cli import main

    main()
    # Should create a sequential directory like prof_00/
    prof_dirs = [d for d in tmp_path.iterdir() if d.is_dir() and d.name.startswith("prof_")]
    assert len(prof_dirs) >= 1
    # Should have at least one profile .nc file
    nc_files = list(prof_dirs[0].glob("*.nc"))
    assert len(nc_files) >= 1


def test_cmd_eps_serial(monkeypatch, sample_p_file, tmp_path):
    """rsi-tpw eps with jobs=1 (serial) should produce epsilon output."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rsi-tpw",
            "eps",
            str(sample_p_file),
            "-o",
            str(tmp_path),
            "-j",
            "1",
        ],
    )
    from microstructure_tpw.rsi.cli import main

    main()
    eps_dirs = [d for d in tmp_path.iterdir() if d.is_dir() and d.name.startswith("eps_")]
    assert len(eps_dirs) >= 1
    nc_files = list(eps_dirs[0].glob("*_eps.nc"))
    assert len(nc_files) >= 1


def test_cmd_eps_parallel(monkeypatch, sample_p_file, tmp_path):
    """rsi-tpw eps with -j 2 (parallel) should produce epsilon output."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rsi-tpw",
            "eps",
            str(sample_p_file),
            "-o",
            str(tmp_path),
            "-j",
            "2",
        ],
    )
    from microstructure_tpw.rsi.cli import main

    main()
    eps_dirs = [d for d in tmp_path.iterdir() if d.is_dir() and d.name.startswith("eps_")]
    assert len(eps_dirs) >= 1
    nc_files = list(eps_dirs[0].glob("*_eps.nc"))
    assert len(nc_files) >= 1


def test_cmd_chi_method2(monkeypatch, sample_p_file, tmp_path):
    """rsi-tpw chi without --epsilon-dir should use Method 2."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rsi-tpw",
            "chi",
            str(sample_p_file),
            "-o",
            str(tmp_path),
            "-j",
            "1",
        ],
    )
    from microstructure_tpw.rsi.cli import main

    main()
    chi_dirs = [d for d in tmp_path.iterdir() if d.is_dir() and d.name.startswith("chi_")]
    assert len(chi_dirs) >= 1
    nc_files = list(chi_dirs[0].glob("*_chi.nc"))
    assert len(nc_files) >= 1


def test_cmd_chi_method1(monkeypatch, sample_p_file, tmp_path):
    """rsi-tpw chi with --epsilon-dir should use Method 1."""
    from microstructure_tpw.rsi.dissipation import compute_diss_file

    eps_dir = tmp_path / "eps_input"
    eps_dir.mkdir()
    compute_diss_file(sample_p_file, eps_dir, fft_length=256, goodman=True)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rsi-tpw",
            "chi",
            str(sample_p_file),
            "--epsilon-dir",
            str(eps_dir),
            "-o",
            str(tmp_path),
            "-j",
            "1",
        ],
    )
    from microstructure_tpw.rsi.cli import main

    main()
    chi_dirs = [d for d in tmp_path.iterdir() if d.is_dir() and d.name.startswith("chi_")]
    assert len(chi_dirs) >= 1


def test_cmd_chi_missing_epsilon_warns(monkeypatch, sample_p_file, tmp_path, capsys):
    """--epsilon-dir pointing to empty dir should warn and use Method 2."""
    empty_eps_dir = tmp_path / "empty_eps"
    empty_eps_dir.mkdir()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rsi-tpw",
            "chi",
            str(sample_p_file),
            "--epsilon-dir",
            str(empty_eps_dir),
            "-o",
            str(tmp_path),
            "-j",
            "1",
        ],
    )
    from microstructure_tpw.rsi.cli import main

    main()
    captured = capsys.readouterr()
    assert "Warning" in captured.out or "Method 2" in captured.out


def test_cmd_pipeline(monkeypatch, sample_p_file, tmp_path):
    """rsi-tpw pipeline should produce both eps and chi output."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rsi-tpw",
            "pipeline",
            str(sample_p_file),
            "-o",
            str(tmp_path),
        ],
    )
    from microstructure_tpw.rsi.cli import main

    main()
    eps_dirs = [d for d in tmp_path.iterdir() if d.is_dir() and d.name.startswith("eps_")]
    chi_dirs = [d for d in tmp_path.iterdir() if d.is_dir() and d.name.startswith("chi_")]
    assert len(eps_dirs) >= 1
    assert len(chi_dirs) >= 1
