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
    write_resolved_config,
)

# ---------------------------------------------------------------------------
# DEFAULTS structure (perturb-specific)
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_has_18_sections(self):
        assert len(DEFAULTS) == 18

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
            "stratification",
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
        cfg.write_text("instruments:\n  SN465:\n    exclude_shear_probes:\n      - sh2\n")
        config = load_config(cfg)
        assert config["instruments"]["SN465"]["exclude_shear_probes"] == ["sh2"]

    def test_instruments_do_not_affect_direct_section_hash(self):
        # instruments are only hashed when supplied as an explicit upstream
        # dependency by the pipeline.
        h_blank = compute_hash("epsilon", {})
        assert isinstance(h_blank, str)

    def test_write_resolved_config_preserves_instruments(self, tmp_path):
        write_resolved_config(
            tmp_path,
            "epsilon",
            {},
            upstream=[
                ("instruments", {"SN465": {"exclude_shear_probes": ["sh2"]}}),
            ],
        )
        config = load_config(tmp_path / "config.yaml")
        assert config["instruments"]["SN465"]["exclude_shear_probes"] == ["sh2"]

    def test_instrument_named_like_hash_exclude_key_is_not_dropped(self):
        # Regression: hash_exclude_keys ({'diagnostics'}) is meant for per-section
        # toggle keys in STATIC sections; it must NOT be applied to dynamic-key
        # (instrument-name) sections. An instrument literally named 'diagnostics'
        # was silently dropped from the hash/canonical config.
        with_override = canonicalize(
            "instruments", {"diagnostics": {"exclude_shear_probes": ["sh1"]}}
        )
        empty = canonicalize("instruments", {})
        assert with_override != empty
        assert "diagnostics" in json.loads(with_override)["instruments"]


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

    def test_dynamic_section_overrides_not_discarded(self):
        # Regression: merge_config seeded `merged = defaults[section]` ({} for
        # instruments) and kept only keys already present, silently dropping
        # ALL user-supplied overrides for dynamic-key sections.
        m = merge_config(
            "instruments",
            file_values={"SN465": {"exclude_shear_probes": ["sh1"]}},
        )
        assert m == {"SN465": {"exclude_shear_probes": ["sh1"]}}

    def test_dynamic_section_cli_overrides_file_and_filters_none(self):
        m = merge_config(
            "instruments",
            file_values={"SN465": {"exclude_shear_probes": ["sh1"]}},
            cli_overrides={
                "SN465": {"exclude_shear_probes": ["sh2"]},
                "SN479": None,  # None layer values must not be retained
            },
        )
        assert m == {"SN465": {"exclude_shear_probes": ["sh2"]}}


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

    def test_merge_annotated_with_cast_fusion_warning(self, tmp_path):
        # Regression: files.merge was undocumented, giving operators no in-config
        # signal that enabling it fuses sequential casts into one file.
        p = generate_template(tmp_path / "config.yaml")
        text = p.read_text()
        merge_line = next(line for line in text.splitlines() if line.strip().startswith("merge:"))
        assert "#" in merge_line, "merge line must carry a warning comment"
        assert "fuses" in merge_line.lower()


class TestWindowDurationResolution:
    """Duration-first window keys (fft_sec/diss_sec/overlap_sec)."""

    def test_default_is_one_second_across_sampling_rates(self):
        from odas_tpw.perturb.config import merge_config, resolve_window_config

        merged = merge_config("epsilon", None)
        assert merged["fft_sec"] == 1.0
        assert "fft_length" not in merged  # duration governs by default
        for fs, want in ((512.03275, 512), (1024.0, 1024), (2048.0, 2048)):
            r = resolve_window_config(merged, fs)
            assert r["fft_length"] == want
            assert r["diss_length"] == 4 * want  # default diss = 4 x fft
            assert "fft_sec" not in r  # stripped: safe to splat downstream

    def test_explicit_samples_win_and_keep_signature_stable(self):
        # A legacy config pinning sample counts must merge to a dict WITHOUT
        # the inert duration keys (otherwise every legacy stage-directory
        # signature would drift) and resolve to exactly those counts.
        from odas_tpw.perturb.config import merge_config, resolve_window_config

        merged = merge_config("epsilon", {"fft_length": 256})
        assert "fft_sec" not in merged
        r = resolve_window_config(merged, 512.0)
        assert (r["fft_length"], r["diss_length"]) == (256, 1024)

    def test_legacy_signature_is_byte_identical_pre_change(self):
        # THE invariant: signatures hash the CANONICAL section (defaults
        # overlaid, then the perturb view postprocess), so a legacy config
        # that pins fft_length must produce a canonical form byte-identical
        # to a pre-change manager built from the OLD defaults (no *_sec
        # keys, fft_length 256). Merged-dict equality is NOT sufficient —
        # canonicalization re-overlays onto DEFAULTS.
        from odas_tpw.config_base import ConfigManager
        from odas_tpw.perturb import config as C

        old_eps = {
            k: v
            for k, v in C.DEFAULTS["epsilon"].items()
            if k not in ("fft_sec", "diss_sec", "overlap_sec")
        }
        old_eps["fft_length"] = 256
        old_mgr = ConfigManager(
            {**C.DEFAULTS, "epsilon": old_eps},
            hash_exclude_keys=C._HASH_EXCLUDE_KEYS,
            dynamic_key_sections=C._DYNAMIC_KEY_SECTIONS,
            engine_fingerprint=C.engine_fingerprint,
        )
        user = {"fft_length": 256}
        assert old_mgr._canonicalize_section("epsilon", user) == C._mgr._canonicalize_section(
            "epsilon", user
        )
        assert old_mgr.compute_hash("epsilon", user) == C._mgr.compute_hash("epsilon", user)
        # ...while a GOVERNING duration key changes the hash (it must).
        assert C._mgr.compute_hash("epsilon", {}) != C._mgr.compute_hash("epsilon", user)

    def test_resolved_dict_strips_all_duration_keys(self):
        # Splat-safety: the resolver output is passed as **kwargs into
        # _compute_epsilon/_compute_chi; ANY surviving *_sec key would
        # TypeError there.
        from odas_tpw.perturb.config import merge_config, resolve_window_config

        merged = merge_config("epsilon", {"fft_sec": 2.0, "diss_sec": 10.0, "overlap_sec": 5.0})
        r = resolve_window_config(merged, 512.0)
        assert not {"fft_sec", "diss_sec", "overlap_sec"} & set(r)

    def test_nonpositive_durations_rejected(self):
        import pytest

        from odas_tpw.perturb.config import merge_config, resolve_window_config

        for bad in (0.0, -3.0, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="positive finite"):
                resolve_window_config(merge_config("epsilon", {"fft_sec": bad}), 512.0)

    def test_overlap_must_be_smaller_than_window(self):
        import pytest

        from odas_tpw.perturb.config import merge_config, resolve_window_config

        for sec in (4.0, 10.0):  # == window and > window
            with pytest.raises(ValueError, match="overlap"):
                resolve_window_config(merge_config("epsilon", {"overlap_sec": sec}), 512.0)

    def test_load_time_validation_of_durations(self, tmp_path):
        # A sign typo must fail at config LOAD, before any processing.
        import pytest

        from odas_tpw.perturb.config import load_config

        p = tmp_path / "bad.yaml"
        p.write_text("epsilon:\n  fft_sec: -1.0\n")
        with pytest.raises(ValueError, match="positive finite"):
            load_config(p)
        p.write_text("chi:\n  fft_sec: 2.0\n  diss_sec: 1.0\n")
        with pytest.raises(ValueError, match="shorter than fft_sec"):
            load_config(p)
        p.write_text('epsilon:\n  fft_sec: "abc"\n')
        with pytest.raises(ValueError, match="expected a number"):
            load_config(p)

    def test_duration_overrides_and_even_rounding(self):
        from odas_tpw.perturb.config import merge_config, resolve_window_config

        merged = merge_config("epsilon", {"fft_sec": 2.0, "diss_sec": 10.0, "overlap_sec": 5.0})
        r = resolve_window_config(merged, 512.0)
        assert (r["fft_length"], r["diss_length"], r["overlap"]) == (1024, 5120, 2560)
        # odd sample products round to the nearest even count
        r2 = resolve_window_config(merge_config("epsilon", {"fft_sec": 0.999}), 512.0)
        assert r2["fft_length"] % 2 == 0

    def test_diss_shorter_than_fft_rejected(self):
        import pytest

        from odas_tpw.perturb.config import merge_config, resolve_window_config

        merged = merge_config("epsilon", {"fft_sec": 2.0, "diss_sec": 1.0})
        with pytest.raises(ValueError, match="shorter than"):
            resolve_window_config(merged, 512.0)

    def test_bad_sampling_rate_rejected(self):
        import pytest

        from odas_tpw.perturb.config import merge_config, resolve_window_config

        with pytest.raises(ValueError, match="sampling rate"):
            resolve_window_config(merge_config("epsilon", None), 0.0)

    def test_chi_section_mirrors_epsilon(self):
        from odas_tpw.perturb.config import merge_config, resolve_window_config

        r = resolve_window_config(merge_config("chi", None), 1024.0, section="chi")
        assert (r["fft_length"], r["diss_length"]) == (1024, 4096)
