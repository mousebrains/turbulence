# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Shared config tests parametrized over rsi.config and perturb.config."""

import json

import pytest

from odas_tpw.perturb import config as perturb_config
from odas_tpw.rsi import config as rsi_config


@pytest.fixture(params=[rsi_config, perturb_config], ids=["rsi", "perturb"])
def config_mod(request):
    """Yield the config module under test."""
    return request.param


# ---------------------------------------------------------------------------
# Canonicalize and hash
# ---------------------------------------------------------------------------


class TestCanonicalizeAndHash:
    def test_stable_hash(self, config_mod):
        h1 = config_mod.compute_hash("epsilon", {"fft_length": 256})
        h2 = config_mod.compute_hash("epsilon", {"fft_length": 256})
        assert h1 == h2
        assert len(h1) == 64

    def test_explicit_defaults_match_implicit(self, config_mod):
        h_implicit = config_mod.compute_hash("epsilon", {})
        h_explicit = config_mod.compute_hash(
            "epsilon", dict(config_mod.DEFAULTS["epsilon"])
        )
        assert h_implicit == h_explicit

    def test_different_params_different_hash(self, config_mod):
        h1 = config_mod.compute_hash("epsilon", {"fft_length": 256})
        h2 = config_mod.compute_hash("epsilon", {"fft_length": 512})
        assert h1 != h2

    def test_float_int_normalization(self, config_mod):
        h_int = config_mod.compute_hash("epsilon", {"fft_length": 256})
        h_float = config_mod.compute_hash("epsilon", {"fft_length": 256.0})
        assert h_int == h_float

    def test_canonical_json_is_compact_sorted(self, config_mod):
        c = config_mod.canonicalize("epsilon", {})
        parsed = json.loads(c)
        assert list(parsed.keys()) == sorted(parsed.keys())
        assert list(parsed["epsilon"].keys()) == sorted(parsed["epsilon"].keys())
        assert " " not in c

    def test_upstream_changes_hash(self, config_mod):
        h_no = config_mod.compute_hash("chi", {})
        h_with = config_mod.compute_hash("chi", {}, upstream=[("epsilon", {})])
        assert h_no != h_with

    def test_upstream_param_change_changes_hash(self, config_mod):
        ups1 = [("epsilon", {"fft_length": 256})]
        ups2 = [("epsilon", {"fft_length": 512})]
        h1 = config_mod.compute_hash("chi", {}, upstream=ups1)
        h2 = config_mod.compute_hash("chi", {}, upstream=ups2)
        assert h1 != h2


# ---------------------------------------------------------------------------
# Validate config
# ---------------------------------------------------------------------------


class TestValidateConfig:
    def test_valid_config_passes(self, config_mod):
        config_mod.validate_config({"epsilon": {"fft_length": 256}})

    def test_unknown_section_raises(self, config_mod):
        with pytest.raises(ValueError, match="Unknown config section"):
            config_mod.validate_config({"bogus": {"key": 1}})

    def test_unknown_key_raises(self, config_mod):
        with pytest.raises(ValueError, match="Unknown key"):
            config_mod.validate_config({"epsilon": {"nonexistent_key": 42}})

    def test_empty_config_passes(self, config_mod):
        config_mod.validate_config({})


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


class TestMergeConfig:
    def test_empty_inputs(self, config_mod):
        m = config_mod.merge_config("epsilon")
        assert m["fft_length"] == 256
        assert "diss_length" not in m

    def test_config_overrides_default(self, config_mod):
        m = config_mod.merge_config("epsilon", file_values={"fft_length": 512})
        assert m["fft_length"] == 512

    def test_cli_overrides_config(self, config_mod):
        m = config_mod.merge_config(
            "epsilon",
            file_values={"fft_length": 512},
            cli_overrides={"fft_length": 1024},
        )
        assert m["fft_length"] == 1024

    def test_none_cli_does_not_mask_config(self, config_mod):
        m = config_mod.merge_config(
            "epsilon",
            file_values={"fft_length": 512},
            cli_overrides={"fft_length": None},
        )
        assert m["fft_length"] == 512

    def test_unknown_section_raises(self, config_mod):
        with pytest.raises(ValueError, match="Unknown section"):
            config_mod.merge_config("nonexistent")


# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_valid_yaml(self, config_mod, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("epsilon:\n  fft_length: 512\n")
        config = config_mod.load_config(cfg)
        assert config["epsilon"]["fft_length"] == 512

    def test_empty_file(self, config_mod, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("")
        assert config_mod.load_config(cfg) == {}

    def test_unknown_section_raises(self, config_mod, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("bogus:\n  key: 1\n")
        with pytest.raises(ValueError, match="Unknown config section"):
            config_mod.load_config(cfg)


# ---------------------------------------------------------------------------
# Directory management
# ---------------------------------------------------------------------------


class TestDirectoryManagement:
    def test_creates_prefix_00(self, config_mod, tmp_path):
        d = config_mod.resolve_output_dir(
            tmp_path, "eps", "epsilon", {"fft_length": 256}
        )
        assert d.name == "eps_00"
        assert d.exists()

    def test_sequential_numbering(self, config_mod, tmp_path):
        d0 = config_mod.resolve_output_dir(
            tmp_path, "eps", "epsilon", {"fft_length": 256}
        )
        d1 = config_mod.resolve_output_dir(
            tmp_path, "eps", "epsilon", {"fft_length": 512}
        )
        assert d0.name == "eps_00"
        assert d1.name == "eps_01"

    def test_reuse_on_hash_match(self, config_mod, tmp_path):
        d0 = config_mod.resolve_output_dir(
            tmp_path, "eps", "epsilon", {"fft_length": 256}
        )
        d_again = config_mod.resolve_output_dir(
            tmp_path, "eps", "epsilon", {"fft_length": 256}
        )
        assert d0 == d_again

    def test_signature_file_contains_valid_json(self, config_mod, tmp_path):
        d = config_mod.resolve_output_dir(
            tmp_path, "eps", "epsilon", {"fft_length": 256}
        )
        sig_files = list(d.glob(".params_sha256_*"))
        assert len(sig_files) == 1
        content = json.loads(sig_files[0].read_text())
        assert content["epsilon"]["fft_length"] == 256

    def test_write_signature_standalone(self, config_mod, tmp_path):
        tf = config_mod.write_signature(tmp_path, "epsilon", {"fft_length": 256})
        assert tf.exists()


# ---------------------------------------------------------------------------
# Template generation
# ---------------------------------------------------------------------------


class TestGenerateTemplate:
    def test_valid_yaml(self, config_mod, tmp_path):
        p = config_mod.generate_template(tmp_path / "config.yaml")
        from ruamel.yaml import YAML

        yaml = YAML()
        with open(p) as fh:
            data = yaml.load(fh)
        assert isinstance(data, dict)

    def test_all_sections_present(self, config_mod, tmp_path):
        p = config_mod.generate_template(tmp_path / "config.yaml")
        from ruamel.yaml import YAML

        yaml = YAML()
        with open(p) as fh:
            data = yaml.load(fh)
        for section in config_mod.DEFAULTS:
            assert section in data

    def test_has_comments(self, config_mod, tmp_path):
        p = config_mod.generate_template(tmp_path / "config.yaml")
        text = p.read_text()
        assert "#" in text


# ---------------------------------------------------------------------------
# Write resolved config
# ---------------------------------------------------------------------------


class TestWriteResolvedConfig:
    def test_file_written(self, config_mod, tmp_path):
        p = config_mod.write_resolved_config(
            tmp_path, "epsilon", {"fft_length": 512}
        )
        assert p.exists()
        assert p.name == "config.yaml"

    def test_round_trips(self, config_mod, tmp_path):
        config_mod.write_resolved_config(
            tmp_path, "epsilon", {"fft_length": 512}
        )
        config = config_mod.load_config(tmp_path / "config.yaml")
        assert config["epsilon"]["fft_length"] == 512

    def test_upstream_included(self, config_mod, tmp_path):
        ups = [("epsilon", {"fft_length": 512})]
        config_mod.write_resolved_config(tmp_path, "chi", {}, upstream=ups)
        config = config_mod.load_config(tmp_path / "config.yaml")
        assert "epsilon" in config
        assert "chi" in config
