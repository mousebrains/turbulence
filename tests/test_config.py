# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for rsi.config — YAML config loading, merging, hashing, and directory management."""

import json

import pytest

from microstructure_tpw.rsi.config import (
    DEFAULTS,
    _normalize_value,
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
# Canonicalize and hash
# ---------------------------------------------------------------------------


class TestCanonicalizeAndHash:
    def test_stable_hash(self):
        """Same params → same hash every time."""
        h1 = compute_hash("epsilon", {"fft_length": 256})
        h2 = compute_hash("epsilon", {"fft_length": 256})
        assert h1 == h2
        assert len(h1) == 64  # full SHA-256 hex

    def test_explicit_defaults_match_implicit(self):
        """Passing default values explicitly must produce the same hash as omitting them."""
        h_implicit = compute_hash("epsilon", {})
        h_explicit = compute_hash("epsilon", dict(DEFAULTS["epsilon"]))
        assert h_implicit == h_explicit

    def test_different_params_different_hash(self):
        h1 = compute_hash("epsilon", {"fft_length": 256})
        h2 = compute_hash("epsilon", {"fft_length": 512})
        assert h1 != h2

    def test_float_int_normalization(self):
        """Integer-valued float (e.g. 256.0) should match integer (256)."""
        h_int = compute_hash("epsilon", {"fft_length": 256})
        h_float = compute_hash("epsilon", {"fft_length": 256.0})
        assert h_int == h_float

    def test_bool_not_treated_as_int(self):
        """True (bool) must not be normalized to 1 (int)."""
        c_bool = canonicalize("epsilon", {"goodman": True})
        c_int = canonicalize("epsilon", {"goodman": 1})
        # In JSON, true != 1 — they serialize differently
        assert "true" in c_bool
        # goodman=1 would become int 1 in JSON
        assert c_bool != c_int

    def test_canonical_json_is_compact_sorted(self):
        c = canonicalize("epsilon", {})
        parsed = json.loads(c)
        assert list(parsed.keys()) == sorted(parsed.keys())
        # Inner section keys should also be sorted
        assert list(parsed["epsilon"].keys()) == sorted(parsed["epsilon"].keys())
        assert " " not in c  # compact separators

    def test_upstream_changes_hash(self):
        """Including upstream params must change the hash."""
        h_no_upstream = compute_hash("chi", {})
        h_with_upstream = compute_hash("chi", {}, upstream=[("epsilon", {})])
        assert h_no_upstream != h_with_upstream

    def test_upstream_param_change_changes_hash(self):
        """Changing an upstream parameter must change the downstream hash."""
        ups1 = [("epsilon", {"fft_length": 256})]
        ups2 = [("epsilon", {"fft_length": 512})]
        h1 = compute_hash("chi", {}, upstream=ups1)
        h2 = compute_hash("chi", {}, upstream=ups2)
        assert h1 != h2

    def test_upstream_same_params_same_hash(self):
        """Same upstream params must produce the same hash."""
        ups = [("epsilon", {"fft_length": 256})]
        h1 = compute_hash("chi", {}, upstream=ups)
        h2 = compute_hash("chi", {}, upstream=ups)
        assert h1 == h2

    def test_canonical_includes_upstream_sections(self):
        """Canonical JSON should include upstream section keys."""
        c = canonicalize("chi", {}, upstream=[("epsilon", {"fft_length": 256})])
        parsed = json.loads(c)
        assert "epsilon" in parsed
        assert "chi" in parsed
        assert parsed["epsilon"]["fft_length"] == 256


# ---------------------------------------------------------------------------
# Normalize value
# ---------------------------------------------------------------------------


class TestNormalizeValue:
    def test_none_passthrough(self):
        assert _normalize_value(None) is None

    def test_bool_stays_bool(self):
        assert _normalize_value(True) is True
        assert _normalize_value(False) is False

    def test_int_stays_int(self):
        assert _normalize_value(42) == 42

    def test_integer_float_becomes_int(self):
        assert _normalize_value(256.0) == 256
        assert isinstance(_normalize_value(256.0), int)

    def test_fractional_float_rounded(self):
        val = _normalize_value(0.123456789012345)
        assert isinstance(val, float)
        assert val == round(0.123456789012345, 10)

    def test_string_passthrough(self):
        assert _normalize_value("hello") == "hello"

    def test_unknown_type_passthrough(self):
        """Types not explicitly handled should pass through."""
        val = [1, 2, 3]
        assert _normalize_value(val) is val


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

    def test_multiple_valid_sections(self):
        validate_config(
            {
                "epsilon": {"fft_length": 256},
                "chi": {"fft_length": 512},
                "profiles": {"P_min": 0.5},
            }
        )


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


class TestMergeConfig:
    def test_empty_inputs(self):
        """No file or CLI values → defaults (with Nones stripped)."""
        m = merge_config("epsilon")
        assert m["fft_length"] == 256
        assert "diss_length" not in m  # None default is stripped

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
        """CLI value of None means 'not specified', should not overwrite config."""
        m = merge_config(
            "epsilon",
            file_values={"fft_length": 512},
            cli_overrides={"fft_length": None},
        )
        assert m["fft_length"] == 512

    def test_null_in_config_treated_as_absent(self):
        """null in config file (None) should not mask the default."""
        m = merge_config("epsilon", file_values={"fft_length": None})
        assert m["fft_length"] == 256  # default

    def test_unknown_section_raises(self):
        with pytest.raises(ValueError, match="Unknown section"):
            merge_config("nonexistent")

    def test_profiles_merge(self):
        m = merge_config("profiles", file_values={"P_min": 1.0})
        assert m["P_min"] == 1.0
        assert m["direction"] == "down"


# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_valid_yaml(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("epsilon:\n  fft_length: 512\n")
        config = load_config(cfg_file)
        assert config["epsilon"]["fft_length"] == 512

    def test_empty_file(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("")
        config = load_config(cfg_file)
        assert config == {}

    def test_unknown_section_raises(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("bogus:\n  key: 1\n")
        with pytest.raises(ValueError, match="Unknown config section"):
            load_config(cfg_file)

    def test_unknown_key_raises(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("epsilon:\n  bad_key: 42\n")
        with pytest.raises(ValueError, match="Unknown key"):
            load_config(cfg_file)

    def test_section_with_null_value(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("epsilon:\n  diss_length: null\n")
        config = load_config(cfg_file)
        assert config["epsilon"]["diss_length"] is None

    def test_empty_section(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("epsilon:\n")
        config = load_config(cfg_file)
        assert config["epsilon"] == {}


# ---------------------------------------------------------------------------
# Directory management
# ---------------------------------------------------------------------------


class TestDirectoryManagement:
    def test_creates_prefix_00(self, tmp_path):
        d = resolve_output_dir(tmp_path, "eps", "epsilon", {"fft_length": 256})
        assert d.name == "eps_00"
        assert d.exists()

    def test_sequential_numbering(self, tmp_path):
        d0 = resolve_output_dir(tmp_path, "eps", "epsilon", {"fft_length": 256})
        d1 = resolve_output_dir(tmp_path, "eps", "epsilon", {"fft_length": 512})
        assert d0.name == "eps_00"
        assert d1.name == "eps_01"

    def test_reuse_on_hash_match(self, tmp_path):
        d0 = resolve_output_dir(tmp_path, "eps", "epsilon", {"fft_length": 256})
        d_again = resolve_output_dir(tmp_path, "eps", "epsilon", {"fft_length": 256})
        assert d0 == d_again

    def test_signature_file_contains_valid_json(self, tmp_path):
        d = resolve_output_dir(tmp_path, "eps", "epsilon", {"fft_length": 256})
        signature_files = list(d.glob(".params_sha256_*"))
        assert len(signature_files) == 1
        content = signature_files[0].read_text()
        parsed = json.loads(content)
        assert parsed["epsilon"]["fft_length"] == 256

    def test_signature_file_name_has_hash(self, tmp_path):
        params = {"fft_length": 256}
        d = resolve_output_dir(tmp_path, "eps", "epsilon", params)
        expected_hash = compute_hash("epsilon", params)
        signature_files = list(d.glob(".params_sha256_*"))
        assert len(signature_files) == 1
        assert signature_files[0].name == f".params_sha256_{expected_hash}"

    def test_write_signature_standalone(self, tmp_path):
        tf = write_signature(tmp_path, "epsilon", {"fft_length": 256})
        assert tf.exists()
        content = json.loads(tf.read_text())
        assert "fft_length" in content["epsilon"]

    def test_upstream_changes_directory(self, tmp_path):
        """Different upstream params should produce different chi directories."""
        ups1 = [("epsilon", {"fft_length": 256})]
        ups2 = [("epsilon", {"fft_length": 512})]
        d0 = resolve_output_dir(tmp_path, "chi", "chi", {}, upstream=ups1)
        d1 = resolve_output_dir(tmp_path, "chi", "chi", {}, upstream=ups2)
        assert d0.name == "chi_00"
        assert d1.name == "chi_01"

    def test_upstream_reuse_on_match(self, tmp_path):
        """Same upstream + same params should reuse directory."""
        ups = [("epsilon", {"fft_length": 256})]
        d0 = resolve_output_dir(tmp_path, "chi", "chi", {}, upstream=ups)
        d1 = resolve_output_dir(tmp_path, "chi", "chi", {}, upstream=ups)
        assert d0 == d1

    def test_signature_file_with_upstream_contains_all_sections(self, tmp_path):
        """Touchfile should contain both upstream and primary section."""
        ups = [("epsilon", {"fft_length": 256})]
        d = resolve_output_dir(tmp_path, "chi", "chi", {}, upstream=ups)
        signature_files = list(d.glob(".params_sha256_*"))
        assert len(signature_files) == 1
        content = json.loads(signature_files[0].read_text())
        assert "epsilon" in content
        assert "chi" in content

    def test_non_dir_match_skipped(self, tmp_path):
        """A file matching the prefix_NN pattern should be skipped for hash lookup."""
        # Create eps_00 as a real dir, then eps_01 as a file
        d0 = resolve_output_dir(tmp_path, "eps", "epsilon", {"fft_length": 256})
        assert d0.name == "eps_00"
        fake = tmp_path / "eps_01"
        fake.write_text("I'm a file, not a dir")
        # New params should skip eps_01 (file) and create eps_02
        d = resolve_output_dir(tmp_path, "eps", "epsilon", {"fft_length": 512})
        assert d.name == "eps_02"
        assert d.is_dir()

    def test_invalid_seq_number_skipped(self, tmp_path):
        """A directory with a non-numeric suffix should be skipped gracefully."""
        weird = tmp_path / "eps_xx"
        weird.mkdir()
        d = resolve_output_dir(tmp_path, "eps", "epsilon", {"fft_length": 256})
        # eps_xx doesn't match [0-9][0-9] glob, so first real dir is eps_00
        assert d.name == "eps_00"


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

    def test_all_sections_present(self, tmp_path):
        p = generate_template(tmp_path / "config.yaml")
        from ruamel.yaml import YAML

        yaml = YAML()
        with open(p) as fh:
            data = yaml.load(fh)
        for section in DEFAULTS:
            assert section in data

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

    def test_upstream_included_in_resolved(self, tmp_path):
        """Resolved config with upstream should include all sections."""
        ups = [("epsilon", {"fft_length": 512})]
        write_resolved_config(tmp_path, "chi", {}, upstream=ups)
        config = load_config(tmp_path / "config.yaml")
        assert "epsilon" in config
        assert "chi" in config
        assert config["epsilon"]["fft_length"] == 512
