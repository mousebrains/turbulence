# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb.config — YAML config loading, merging, hashing, and directory management."""

import json

import pytest

from odas_tpw.perturb.config import (
    DEFAULTS,
    canonicalize,
    compute_hash,
    generate_template,
    load_config,
    merge_config,
    resolve_output_dir,
    validate_config,
    write_resolved_config,
    write_signature,
)

# ---------------------------------------------------------------------------
# DEFAULTS structure
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_has_14_sections(self):
        assert len(DEFAULTS) == 14

    def test_expected_sections(self):
        expected = {
            "files", "gps", "hotel", "profiles", "fp07", "ct", "bottom",
            "top_trim", "epsilon", "chi", "ctd", "binning", "netcdf",
            "parallel",
        }
        assert set(DEFAULTS.keys()) == expected

    def test_diagnostics_in_expected_sections(self):
        diag_sections = {"profiles", "epsilon", "chi", "ctd", "binning"}
        for section in diag_sections:
            assert "diagnostics" in DEFAULTS[section], f"{section} missing diagnostics"

    def test_diagnostics_default_false(self):
        for section, params in DEFAULTS.items():
            if "diagnostics" in params:
                assert params["diagnostics"] is False


# ---------------------------------------------------------------------------
# Canonicalize and hash
# ---------------------------------------------------------------------------


class TestCanonicalizeAndHash:
    def test_stable_hash(self):
        h1 = compute_hash("epsilon", {"fft_length": 256})
        h2 = compute_hash("epsilon", {"fft_length": 256})
        assert h1 == h2
        assert len(h1) == 64

    def test_explicit_defaults_match_implicit(self):
        h_implicit = compute_hash("epsilon", {})
        h_explicit = compute_hash("epsilon", dict(DEFAULTS["epsilon"]))
        assert h_implicit == h_explicit

    def test_different_params_different_hash(self):
        h1 = compute_hash("epsilon", {"fft_length": 256})
        h2 = compute_hash("epsilon", {"fft_length": 512})
        assert h1 != h2

    def test_float_int_normalization(self):
        h_int = compute_hash("epsilon", {"fft_length": 256})
        h_float = compute_hash("epsilon", {"fft_length": 256.0})
        assert h_int == h_float

    def test_diagnostics_excluded_from_hash(self):
        """Toggling diagnostics must not change the hash."""
        h_off = compute_hash("profiles", {"diagnostics": False})
        h_on = compute_hash("profiles", {"diagnostics": True})
        assert h_off == h_on

    def test_diagnostics_excluded_from_canonical(self):
        c = canonicalize("epsilon", {})
        parsed = json.loads(c)
        assert "diagnostics" not in parsed["epsilon"]

    def test_canonical_json_is_compact_sorted(self):
        c = canonicalize("epsilon", {})
        parsed = json.loads(c)
        assert list(parsed.keys()) == sorted(parsed.keys())
        assert list(parsed["epsilon"].keys()) == sorted(parsed["epsilon"].keys())
        assert " " not in c

    def test_upstream_changes_hash(self):
        h_no = compute_hash("chi", {})
        h_with = compute_hash("chi", {}, upstream=[("epsilon", {})])
        assert h_no != h_with

    def test_upstream_param_change_changes_hash(self):
        ups1 = [("epsilon", {"fft_length": 256})]
        ups2 = [("epsilon", {"fft_length": 512})]
        h1 = compute_hash("chi", {}, upstream=ups1)
        h2 = compute_hash("chi", {}, upstream=ups2)
        assert h1 != h2


# ---------------------------------------------------------------------------
# Validate config
# ---------------------------------------------------------------------------


class TestValidateConfig:
    def test_valid_config_passes(self):
        validate_config({"epsilon": {"fft_length": 256}})

    def test_unknown_section_raises(self):
        with pytest.raises(ValueError, match="Unknown config section"):
            validate_config({"bogus": {"key": 1}})

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown key"):
            validate_config({"epsilon": {"nonexistent_key": 42}})

    def test_empty_config_passes(self):
        validate_config({})

    def test_all_sections_valid(self):
        config = {section: {} for section in DEFAULTS}
        validate_config(config)

    def test_perturb_specific_sections(self):
        validate_config({
            "files": {"p_file_root": "VMP/"},
            "gps": {"source": "fixed"},
            "fp07": {"calibrate": False},
            "ct": {"align": True},
            "bottom": {"enable": True},
            "top_trim": {"dz": 1.0},
        })


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


class TestMergeConfig:
    def test_empty_inputs(self):
        m = merge_config("epsilon")
        assert m["fft_length"] == 256
        assert "diss_length" not in m  # None stripped

    def test_config_overrides_default(self):
        m = merge_config("epsilon", file_values={"fft_length": 512})
        assert m["fft_length"] == 512

    def test_cli_overrides_config(self):
        m = merge_config(
            "epsilon",
            file_values={"fft_length": 512},
            cli_overrides={"fft_length": 1024},
        )
        assert m["fft_length"] == 1024

    def test_none_cli_does_not_mask_config(self):
        m = merge_config(
            "epsilon",
            file_values={"fft_length": 512},
            cli_overrides={"fft_length": None},
        )
        assert m["fft_length"] == 512

    def test_unknown_section_raises(self):
        with pytest.raises(ValueError, match="Unknown section"):
            merge_config("nonexistent")

    def test_files_section(self):
        m = merge_config("files", file_values={"p_file_root": "/data/"})
        assert m["p_file_root"] == "/data/"
        assert m["trim"] is True

    def test_diagnostics_preserved_in_merge(self):
        m = merge_config("profiles", file_values={"diagnostics": True})
        assert m["diagnostics"] is True


# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_valid_yaml(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("epsilon:\n  fft_length: 512\n")
        config = load_config(cfg)
        assert config["epsilon"]["fft_length"] == 512

    def test_empty_file(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("")
        assert load_config(cfg) == {}

    def test_unknown_section_raises(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("bogus:\n  key: 1\n")
        with pytest.raises(ValueError, match="Unknown config section"):
            load_config(cfg)

    def test_perturb_sections(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("files:\n  trim: false\ngps:\n  source: fixed\n")
        config = load_config(cfg)
        assert config["files"]["trim"] is False
        assert config["gps"]["source"] == "fixed"


# ---------------------------------------------------------------------------
# Directory management
# ---------------------------------------------------------------------------


class TestDirectoryManagement:
    def test_creates_prefix_00(self, tmp_path):
        d = resolve_output_dir(tmp_path, "prof", "profiles", {"P_min": 0.5})
        assert d.name == "prof_00"
        assert d.exists()

    def test_sequential_numbering(self, tmp_path):
        d0 = resolve_output_dir(tmp_path, "prof", "profiles", {"P_min": 0.5})
        d1 = resolve_output_dir(tmp_path, "prof", "profiles", {"P_min": 1.0})
        assert d0.name == "prof_00"
        assert d1.name == "prof_01"

    def test_reuse_on_hash_match(self, tmp_path):
        d0 = resolve_output_dir(tmp_path, "prof", "profiles", {"P_min": 0.5})
        d_again = resolve_output_dir(tmp_path, "prof", "profiles", {"P_min": 0.5})
        assert d0 == d_again

    def test_diagnostics_does_not_create_new_dir(self, tmp_path):
        """Toggling diagnostics must reuse the same directory."""
        d_off = resolve_output_dir(tmp_path, "prof", "profiles", {"diagnostics": False})
        d_on = resolve_output_dir(tmp_path, "prof", "profiles", {"diagnostics": True})
        assert d_off == d_on

    def test_signature_file_contains_valid_json(self, tmp_path):
        d = resolve_output_dir(tmp_path, "eps", "epsilon", {"fft_length": 256})
        sig_files = list(d.glob(".params_sha256_*"))
        assert len(sig_files) == 1
        content = json.loads(sig_files[0].read_text())
        assert content["epsilon"]["fft_length"] == 256

    def test_write_signature_standalone(self, tmp_path):
        tf = write_signature(tmp_path, "epsilon", {"fft_length": 256})
        assert tf.exists()

    def test_upstream_changes_directory(self, tmp_path):
        ups1 = [("profiles", {"P_min": 0.5})]
        ups2 = [("profiles", {"P_min": 1.0})]
        d0 = resolve_output_dir(tmp_path, "eps", "epsilon", {}, upstream=ups1)
        d1 = resolve_output_dir(tmp_path, "eps", "epsilon", {}, upstream=ups2)
        assert d0.name == "eps_00"
        assert d1.name == "eps_01"


# ---------------------------------------------------------------------------
# Template generation
# ---------------------------------------------------------------------------


class TestGenerateTemplate:
    def test_valid_yaml(self, tmp_path):
        p = generate_template(tmp_path / "config.yaml")
        from ruamel.yaml import YAML
        yaml = YAML()
        with open(p) as fh:
            data = yaml.load(fh)
        assert isinstance(data, dict)

    def test_all_14_sections_present(self, tmp_path):
        p = generate_template(tmp_path / "config.yaml")
        from ruamel.yaml import YAML
        yaml = YAML()
        with open(p) as fh:
            data = yaml.load(fh)
        for section in DEFAULTS:
            assert section in data, f"Missing section: {section}"

    def test_diagnostics_flags_present(self, tmp_path):
        p = generate_template(tmp_path / "config.yaml")
        text = p.read_text()
        assert "diagnostics:" in text

    def test_has_comments(self, tmp_path):
        p = generate_template(tmp_path / "config.yaml")
        text = p.read_text()
        assert "#" in text


# ---------------------------------------------------------------------------
# Write resolved config
# ---------------------------------------------------------------------------


class TestWriteResolvedConfig:
    def test_file_written(self, tmp_path):
        p = write_resolved_config(tmp_path, "epsilon", {"fft_length": 512})
        assert p.exists()
        assert p.name == "config.yaml"

    def test_round_trips(self, tmp_path):
        write_resolved_config(tmp_path, "epsilon", {"fft_length": 512})
        config = load_config(tmp_path / "config.yaml")
        assert config["epsilon"]["fft_length"] == 512

    def test_upstream_included(self, tmp_path):
        ups = [("profiles", {"P_min": 1.0})]
        write_resolved_config(tmp_path, "epsilon", {}, upstream=ups)
        config = load_config(tmp_path / "config.yaml")
        assert "profiles" in config
        assert "epsilon" in config
