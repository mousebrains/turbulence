# Jun-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb-plot's figure-spec driver (plot/figure.py)."""

import os
from types import SimpleNamespace

import pytest
import yaml

from odas_tpw.perturb.plot import figure as fig
from odas_tpw.perturb.plot import profiles, scalar


def _cli(**kw):
    base = dict(spec=None, list_presets=False, dump_preset=None,
                select=None, strict=False, latest=False)
    base.update(kw)
    return SimpleNamespace(**base)


class TestValidateSource:
    def test_needs_exactly_one(self):
        with pytest.raises(SystemExit):
            fig._validate_source({"source": {}})
        with pytest.raises(SystemExit):
            fig._validate_source({"source": {"config": "c", "root": "r"}})
        with pytest.raises(SystemExit):
            fig._validate_source({})
        assert fig._validate_source({"source": {"config": "c"}}) == {"config": "c"}


class TestSectionsFile:
    def test_file_ref(self):
        assert fig._sections_file({"sections": {"file": "s.yaml"}}, []) == "s.yaml"

    def test_none(self):
        assert fig._sections_file({}, []) is None

    def test_inline_written_to_temp(self):
        tmp = []
        f = fig._sections_file(
            {"sections": [{"name": "a", "xaxis": {"method": "time"}}]}, tmp
        )
        assert f and os.path.exists(f) and tmp == [f]
        with open(f) as fh:
            loaded = yaml.safe_load(fh)
        assert loaded["sections"][0]["name"] == "a"
        os.unlink(f)

    def test_bad_type(self):
        with pytest.raises(SystemExit):
            fig._sections_file({"sections": "nope"}, [])


class TestBuildArgs:
    def test_scalar_vars_clim_output(self, tmp_path):
        figure = {"name": "x", "preset": "scalar", "vars": ["JAC_T", "SP"],
                  "clim": {"JAC_T": [18, 28]}, "depth_max": 150}
        run_fn, args = fig._build_args(figure, {"config": "c.yaml"}, None, tmp_path, False, False)
        assert run_fn is scalar.run
        assert args.var == ["JAC_T", "SP"]
        assert args.clim == [["JAC_T", "18", "28"]]
        assert args.depth_max == 150
        assert args.config == "c.yaml"
        assert args.out_dir == str(tmp_path / "x")

    def test_var_singular_to_list(self, tmp_path):
        _, args = fig._build_args(
            {"name": "x", "preset": "profiles", "var": "chiMean"},
            {"config": "c"}, None, tmp_path, False, False,
        )
        assert args.var == ["chiMean"]

    def test_root_source(self, tmp_path):
        _, args = fig._build_args(
            {"name": "x", "preset": "scalar"},
            {"root": "/data"}, None, tmp_path, False, False,
        )
        assert args.config is None and args.root == "/data"

    def test_eps_chi_output_is_file(self, tmp_path):
        _run_fn, args = fig._build_args(
            {"name": "ov", "preset": "eps-chi"}, {"config": "c"}, None, tmp_path, False, False
        )
        assert args.out == str(tmp_path / "ov" / "eps_chi.png")

    def test_unknown_preset(self, tmp_path):
        with pytest.raises(SystemExit, match="unknown preset"):
            fig._build_args({"name": "x", "preset": "bogus"}, {"config": "c"},
                            None, tmp_path, False, False)

    def test_option_invalid_for_preset(self, tmp_path):
        # eps-chi has no --var
        with pytest.raises(SystemExit, match="not valid for preset"):
            fig._build_args({"name": "x", "preset": "eps-chi", "vars": ["a"]},
                            {"config": "c"}, None, tmp_path, False, False)

    def test_section_rejected_on_eps_chi(self, tmp_path):
        with pytest.raises(SystemExit, match="no x-axis"):
            fig._build_args({"name": "x", "preset": "eps-chi", "section": "full"},
                            {"config": "c"}, None, tmp_path, False, False)

    def test_section_name_needs_sections_block(self, tmp_path):
        with pytest.raises(SystemExit, match="no 'sections' block"):
            fig._build_args({"name": "x", "preset": "profiles", "section": "full"},
                            {"config": "c"}, None, tmp_path, False, False)

    def test_section_star_is_all(self, tmp_path):
        _, args = fig._build_args(
            {"name": "x", "preset": "profiles", "section": "*"},
            {"config": "c"}, "s.yaml", tmp_path, False, False,
        )
        assert args.select is None and args.sections == "s.yaml"

    def test_section_list_select(self, tmp_path):
        _, args = fig._build_args(
            {"name": "x", "preset": "profiles", "section": ["a", "b"]},
            {"config": "c"}, "s.yaml", tmp_path, False, False,
        )
        assert args.select == ["a", "b"]

    def test_clim_must_be_mapping(self, tmp_path):
        with pytest.raises(SystemExit, match="clim"):
            fig._build_args({"name": "x", "preset": "scalar", "clim": [1, 2]},
                            {"config": "c"}, None, tmp_path, False, False)

    def test_clim_bad_pair(self, tmp_path):
        with pytest.raises(SystemExit, match="min, max"):
            fig._build_args({"name": "x", "preset": "scalar", "clim": {"JAC_T": [18]}},
                            {"config": "c"}, None, tmp_path, False, False)

    def test_numeric_coerced_from_string(self, tmp_path):
        # PyYAML parses `1e-7` (no dot) as a string; _coerce must float() it.
        _, args = fig._build_args(
            {"name": "x", "preset": "profiles", "p_max": "150", "gap_factor": "4"},
            {"config": "c"}, None, tmp_path, False, False,
        )
        assert args.p_max == 150.0 and isinstance(args.p_max, float)
        assert args.gap_factor == 4.0

    def test_bad_numeric_value(self, tmp_path):
        with pytest.raises(SystemExit, match="not a valid"):
            fig._build_args({"name": "x", "preset": "profiles", "p_max": "abc"},
                            {"config": "c"}, None, tmp_path, False, False)

    def test_invalid_choice_rejected(self, tmp_path):
        with pytest.raises(SystemExit, match="must be one of"):
            fig._build_args({"name": "x", "preset": "profiles", "product": "bogus"},
                            {"config": "c"}, None, tmp_path, False, False)

    def test_reserved_key_rejected(self, tmp_path):
        with pytest.raises(SystemExit, match="set by source"):
            fig._build_args({"name": "x", "preset": "scalar", "root": "/other"},
                            {"config": "c"}, None, tmp_path, False, False)

    def test_reserved_hyphen_key_rejected(self, tmp_path):
        with pytest.raises(SystemExit, match="set by source"):
            fig._build_args({"name": "x", "preset": "scalar", "out-dir": "/x"},
                            {"config": "c"}, None, tmp_path, False, False)

    def test_list_for_scalar_option_rejected(self, tmp_path):
        with pytest.raises(SystemExit, match="single value"):
            fig._build_args({"name": "x", "preset": "profiles", "p_max": [100, 200]},
                            {"config": "c"}, None, tmp_path, False, False)

    def test_boolean_string_rejected(self, tmp_path):
        with pytest.raises(SystemExit, match="true/false"):
            fig._build_args({"name": "x", "preset": "profiles", "apply_qc": "false"},
                            {"config": "c"}, None, tmp_path, False, False)

    def test_boolean_bool_accepted(self, tmp_path):
        _, args = fig._build_args(
            {"name": "x", "preset": "profiles", "apply_qc": False},
            {"config": "c"}, None, tmp_path, False, False,
        )
        assert args.apply_qc is False

    def test_hyphenated_key_accepted(self, tmp_path):
        _, args = fig._build_args(
            {"name": "x", "preset": "profiles", "p-max": 150},
            {"config": "c"}, None, tmp_path, False, False,
        )
        assert args.p_max == 150.0

    def test_float_not_truncated(self, tmp_path):
        _, args = fig._build_args(
            {"name": "x", "preset": "profiles", "p_max": 1.5},
            {"config": "c"}, None, tmp_path, False, False,
        )
        assert args.p_max == 1.5  # str()-coercion keeps the float (no int trunc)

    def test_unknown_option_lists_valid(self, tmp_path):
        with pytest.raises(SystemExit, match="Valid options"):
            fig._build_args({"name": "x", "preset": "scalar", "bogus_opt": 1},
                            {"config": "c"}, None, tmp_path, False, False)


class TestRun:
    def test_missing_spec(self):
        with pytest.raises(SystemExit, match="--spec is required"):
            fig.run(_cli())

    def test_list_presets(self, capsys):
        fig.run(_cli(list_presets=True))
        out = capsys.readouterr().out
        assert "scalar" in out and "profiles" in out and "eps-chi" in out

    def test_dump_preset(self, capsys):
        fig.run(_cli(dump_preset="profiles"))
        assert "preset: profiles" in capsys.readouterr().out

    def test_dump_unknown_preset(self):
        with pytest.raises(SystemExit, match="no example preset"):
            fig.run(_cli(dump_preset="bogus"))

    def test_dispatch_and_select(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(scalar, "run", lambda a: calls.append(("scalar", a.var)))
        monkeypatch.setattr(profiles, "run", lambda a: calls.append(("profiles", a.product)))
        spec = tmp_path / "spec.yaml"
        spec.write_text(
            "source: {config: c.yaml}\n"
            f"output_dir: {tmp_path}/out\n"
            "figures:\n"
            "  - {name: a, preset: scalar, vars: [JAC_T]}\n"
            "  - {name: b, preset: profiles, product: chi}\n"
        )
        fig.run(_cli(spec=str(spec)))
        assert {c[0] for c in calls} == {"scalar", "profiles"}
        calls.clear()
        fig.run(_cli(spec=str(spec), select=["a"]))  # only figure 'a'
        assert [c[0] for c in calls] == ["scalar"]

    def test_select_unknown_name(self, tmp_path):
        spec = tmp_path / "spec.yaml"
        spec.write_text(
            "source: {config: c}\n"
            f"output_dir: {tmp_path}/out\n"
            "figures:\n  - {name: a, preset: scalar}\n"
        )
        with pytest.raises(SystemExit, match="unknown figure name"):
            fig.run(_cli(spec=str(spec), select=["nope"]))

    def test_empty_figures(self, tmp_path):
        spec = tmp_path / "spec.yaml"
        spec.write_text("source: {config: c}\nfigures: []\n")
        with pytest.raises(SystemExit, match="non-empty 'figures'"):
            fig.run(_cli(spec=str(spec)))

    def test_duplicate_figure_names(self, tmp_path):
        spec = tmp_path / "spec.yaml"
        spec.write_text(
            "source: {config: c}\noutput_dir: o\nfigures:\n"
            "  - {name: a, preset: scalar}\n  - {name: a, preset: profiles}\n"
        )
        with pytest.raises(SystemExit, match="unique 'name:'"):
            fig.run(_cli(spec=str(spec)))

    def test_select_by_preset_when_unnamed(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(scalar, "run", lambda a: calls.append("scalar"))
        spec = tmp_path / "spec.yaml"
        spec.write_text(
            "source: {config: c}\n" f"output_dir: {tmp_path}/o\n"
            "figures:\n  - {preset: scalar}\n"
        )
        fig.run(_cli(spec=str(spec), select=["scalar"]))  # selectable by preset
        assert calls == ["scalar"]
