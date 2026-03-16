# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for rsi.config — rsi-specific behavior not covered by test_config_shared."""

import json

import pytest

from odas_tpw.rsi.config import (
    _normalize_value,
    canonicalize,
    compute_hash,
    load_config,
    merge_config,
    resolve_output_dir,
    write_resolved_config,
)

# ---------------------------------------------------------------------------
# Canonicalize and hash (rsi-specific)
# ---------------------------------------------------------------------------


class TestCanonicalizeAndHashRsi:
    def test_bool_not_treated_as_int(self):
        """True (bool) must not be normalized to 1 (int)."""
        c_bool = canonicalize("epsilon", {"goodman": True})
        c_int = canonicalize("epsilon", {"goodman": 1})
        assert "true" in c_bool
        assert c_bool != c_int

    def test_upstream_same_params_same_hash(self):
        ups = [("epsilon", {"fft_length": 256})]
        h1 = compute_hash("chi", {}, upstream=ups)
        h2 = compute_hash("chi", {}, upstream=ups)
        assert h1 == h2

    def test_canonical_includes_upstream_sections(self):
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
        val = [1, 2, 3]
        assert _normalize_value(val) is val


# ---------------------------------------------------------------------------
# Validate config (rsi-specific)
# ---------------------------------------------------------------------------


class TestValidateConfigRsi:
    def test_multiple_valid_sections(self):
        from odas_tpw.rsi.config import validate_config

        validate_config(
            {
                "epsilon": {"fft_length": 256},
                "chi": {"fft_length": 512},
                "profiles": {"P_min": 0.5},
            }
        )


# ---------------------------------------------------------------------------
# Merge (rsi-specific)
# ---------------------------------------------------------------------------


class TestMergeConfigRsi:
    def test_null_in_config_treated_as_absent(self):
        m = merge_config("epsilon", file_values={"fft_length": None})
        assert m["fft_length"] == 1024

    def test_profiles_merge(self):
        m = merge_config("profiles", file_values={"P_min": 1.0})
        assert m["P_min"] == 1.0
        assert m["direction"] == "auto"


# ---------------------------------------------------------------------------
# Load config (rsi-specific)
# ---------------------------------------------------------------------------


class TestLoadConfigRsi:
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
# Directory management (rsi-specific)
# ---------------------------------------------------------------------------


class TestDirectoryManagementRsi:
    def test_signature_file_name_has_hash(self, tmp_path):
        params = {"fft_length": 256}
        d = resolve_output_dir(tmp_path, "eps", "epsilon", params)
        expected_hash = compute_hash("epsilon", params)
        signature_files = list(d.glob(".params_sha256_*"))
        assert len(signature_files) == 1
        assert signature_files[0].name == f".params_sha256_{expected_hash}"

    def test_upstream_reuse_on_match(self, tmp_path):
        ups = [("epsilon", {"fft_length": 256})]
        d0 = resolve_output_dir(tmp_path, "chi", "chi", {}, upstream=ups)
        d1 = resolve_output_dir(tmp_path, "chi", "chi", {}, upstream=ups)
        assert d0 == d1

    def test_signature_file_with_upstream_contains_all_sections(self, tmp_path):
        ups = [("epsilon", {"fft_length": 256})]
        d = resolve_output_dir(tmp_path, "chi", "chi", {}, upstream=ups)
        signature_files = list(d.glob(".params_sha256_*"))
        assert len(signature_files) == 1
        content = json.loads(signature_files[0].read_text())
        assert "epsilon" in content
        assert "chi" in content

    def test_non_dir_match_skipped(self, tmp_path):
        d0 = resolve_output_dir(tmp_path, "eps", "epsilon", {"fft_length": 256})
        assert d0.name == "eps_00"
        fake = tmp_path / "eps_01"
        fake.write_text("I'm a file, not a dir")
        d = resolve_output_dir(tmp_path, "eps", "epsilon", {"fft_length": 512})
        assert d.name == "eps_02"
        assert d.is_dir()

    def test_invalid_seq_number_skipped(self, tmp_path):
        weird = tmp_path / "eps_xx"
        weird.mkdir()
        d = resolve_output_dir(tmp_path, "eps", "epsilon", {"fft_length": 256})
        assert d.name == "eps_00"


# ---------------------------------------------------------------------------
# Write resolved config (rsi-specific)
# ---------------------------------------------------------------------------


class TestWriteResolvedConfigRsi:
    def test_upstream_included_in_resolved(self, tmp_path):
        ups = [("epsilon", {"fft_length": 512})]
        write_resolved_config(tmp_path, "chi", {}, upstream=ups)
        config = load_config(tmp_path / "config.yaml")
        assert "epsilon" in config
        assert "chi" in config
        assert config["epsilon"]["fft_length"] == 512
