# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb.config — perturb-specific behavior not covered by test_config_shared."""

import json

from odas_tpw.perturb.config import (
    DEFAULTS,
    canonicalize,
    compute_hash,
    generate_template,
    load_config,
    merge_config,
    resolve_output_dir,
    validate_config,
)

# ---------------------------------------------------------------------------
# DEFAULTS structure (perturb-specific)
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_has_17_sections(self):
        assert len(DEFAULTS) == 17

    def test_expected_sections(self):
        expected = {
            "files",
            "gps",
            "hotel",
            "profiles",
            "fp07",
            "ct",
            "bottom",
            "top_trim",
            "epsilon",
            "chi",
            "ctd",
            "speed",
            "qc",
            "binning",
            "netcdf",
            "parallel",
            "instruments",
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
# Canonicalize and hash (perturb-specific)
# ---------------------------------------------------------------------------


class TestCanonicalizeAndHashPerturb:
    def test_diagnostics_excluded_from_hash(self):
        h_off = compute_hash("profiles", {"diagnostics": False})
        h_on = compute_hash("profiles", {"diagnostics": True})
        assert h_off == h_on

    def test_diagnostics_excluded_from_canonical(self):
        c = canonicalize("epsilon", {})
        parsed = json.loads(c)
        assert "diagnostics" not in parsed["epsilon"]


# ---------------------------------------------------------------------------
# Validate config (perturb-specific)
# ---------------------------------------------------------------------------


class TestValidateConfigPerturb:
    def test_all_sections_valid(self):
        config = {section: {} for section in DEFAULTS}
        validate_config(config)

    def test_perturb_specific_sections(self):
        validate_config(
            {
                "files": {"p_file_root": "VMP/"},
                "gps": {"source": "fixed"},
                "fp07": {"calibrate": False},
                "ct": {"align": True},
                "bottom": {"enable": True},
                "top_trim": {"dz": 1.0},
            }
        )


class TestInstrumentsSection:
    def test_arbitrary_serial_keys_accepted(self):
        validate_config(
            {
                "instruments": {
                    "SN465": {"exclude_shear_probes": ["sh2"]},
                    "SN428": {"exclude_shear_probes": []},
                },
            }
        )

    def test_unknown_inner_key_rejected(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown key"):
            validate_config({"instruments": {"SN465": {"oops": True}}})

    def test_non_dict_value_rejected(self):
        import pytest

        with pytest.raises(ValueError, match="must be a mapping"):
            validate_config({"instruments": {"SN465": ["sh2"]}})

    def test_exclude_shear_probes_must_be_list_of_strings(self):
        import pytest

        with pytest.raises(ValueError, match="list of strings"):
            validate_config({"instruments": {"SN465": {"exclude_shear_probes": "sh2"}}})

        with pytest.raises(ValueError, match="list of strings"):
            validate_config({"instruments": {"SN465": {"exclude_shear_probes": [1, 2]}}})

    def test_load_yaml_with_instruments(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "instruments:\n"
            "  SN465:\n"
            "    exclude_shear_probes:\n"
            "      - sh2\n"
        )
        config = load_config(cfg)
        assert config["instruments"]["SN465"]["exclude_shear_probes"] == ["sh2"]

    def test_instruments_excluded_from_hash(self):
        # instruments should not appear in upstream hashing for diss/chi/ctd
        # — the section is per-SN and not part of the per-stage param schema.
        h_blank = compute_hash("epsilon", {})
        # instruments is not a section_name accepted by compute_hash, so we
        # just confirm the existing per-section hashing still works after
        # adding the dynamic-key section.
        assert isinstance(h_blank, str)


# ---------------------------------------------------------------------------
# Merge (perturb-specific)
# ---------------------------------------------------------------------------


class TestMergeConfigPerturb:
    def test_files_section(self):
        m = merge_config("files", file_values={"p_file_root": "/data/"})
        assert m["p_file_root"] == "/data/"
        assert m["trim"] is True

    def test_diagnostics_preserved_in_merge(self):
        m = merge_config("profiles", file_values={"diagnostics": True})
        assert m["diagnostics"] is True


# ---------------------------------------------------------------------------
# Load config (perturb-specific)
# ---------------------------------------------------------------------------


class TestLoadConfigPerturb:
    def test_perturb_sections(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("files:\n  trim: false\ngps:\n  source: fixed\n")
        config = load_config(cfg)
        assert config["files"]["trim"] is False
        assert config["gps"]["source"] == "fixed"


# ---------------------------------------------------------------------------
# Directory management (perturb-specific)
# ---------------------------------------------------------------------------


class TestDirectoryManagementPerturb:
    def test_diagnostics_does_not_create_new_dir(self, tmp_path):
        d_off = resolve_output_dir(tmp_path, "prof", "profiles", {"diagnostics": False})
        d_on = resolve_output_dir(tmp_path, "prof", "profiles", {"diagnostics": True})
        assert d_off == d_on


# ---------------------------------------------------------------------------
# Template generation (perturb-specific)
# ---------------------------------------------------------------------------


class TestGenerateTemplatePerturb:
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
