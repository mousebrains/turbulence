# Jun-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb-plot's figure-spec driver (plot/figure.py)."""

import argparse
import os
from types import SimpleNamespace

import pytest
from ruamel.yaml import YAML

from odas_tpw.perturb.plot import figure as fig
from odas_tpw.perturb.plot import profiles, scalar


def _cli(**kw):
    base = dict(spec=None, list_presets=False, dump_preset=None,
                select=None, strict=False, latest=False)
    base.update(kw)
    return SimpleNamespace(**base)


class TestYamlBackend:
    def test_uses_ruamel_not_pyyaml(self):
        """figure.py must read/write YAML via ruamel.yaml (a declared runtime
        dependency), NOT PyYAML — which is not a runtime dep, so `import yaml`
        breaks `perturb-plot` in a non-dev install. Guard against a regression
        to a bare PyYAML import (this whole module imports unconditionally from
        cli.py, so a missing dep fails every subcommand before argparse)."""
        import ruamel.yaml

        assert fig.YAML is ruamel.yaml.YAML
        assert not hasattr(fig, "yaml")  # no top-level PyYAML reference


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
            loaded = YAML(typ="safe").load(fh)
        assert loaded["sections"][0]["name"] == "a"
        os.unlink(f)

    def test_bad_type(self):
        with pytest.raises(SystemExit):
            fig._sections_file({"sections": "nope"}, [])


class TestBuildArgs:
    def test_scalar_vars_clim_output(self, tmp_path):
        figure = {"name": "x", "preset": "scalar", "vars": ["JAC_T", "SP"],
                  "clim": {"JAC_T": [18, 28]}, "depth_max": 150}
        mod, args = fig._build_args(figure, {"config": "c.yaml"}, None, tmp_path, False, False)
        assert mod is scalar  # the preset module (driver calls mod.run / mod.build_figures)
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

    def test_nargs_exact_list_accepted(self, tmp_path):
        """--point is nargs=2: a 2-element list is accepted and float-coerced."""
        _, args = fig._build_args(
            {"name": "x", "preset": "scalar", "point": [12, 130]},
            {"config": "c"}, None, tmp_path, False, False,
        )
        assert args.point == [12.0, 130.0]

    def test_nargs_too_few_rejected(self, tmp_path):
        with pytest.raises(SystemExit, match="exactly 2 values"):
            fig._build_args({"name": "x", "preset": "scalar", "point": [12]},
                            {"config": "c"}, None, tmp_path, False, False)

    def test_nargs_too_many_rejected(self, tmp_path):
        with pytest.raises(SystemExit, match="exactly 2 values"):
            fig._build_args({"name": "x", "preset": "scalar", "point": [12, 130, 7]},
                            {"config": "c"}, None, tmp_path, False, False)

    def test_nargs_scalar_rejected(self, tmp_path):
        """A bare scalar for an nargs list option is rejected (the CLI would
        never produce a non-list there — downstream indexing would crash)."""
        with pytest.raises(SystemExit, match="expects a list"):
            fig._build_args({"name": "x", "preset": "scalar", "point": 12.0},
                            {"config": "c"}, None, tmp_path, False, False)

    def test_append_nargs_not_length_checked(self, tmp_path):
        """clim is append+nargs=3 (a list of 3-tuples = several --clim); the
        nargs length check must NOT apply to it (one entry here, not three)."""
        _, args = fig._build_args(
            {"name": "x", "preset": "scalar", "clim": {"T": [0, 10]}},
            {"config": "c"}, None, tmp_path, False, False,
        )
        assert args.clim == [["T", "0", "10"]]

    def test_nargs_plus_empty_rejected(self):
        """nargs='+' requires >=1 value, exactly like the CLI (no preset uses
        '+' today, so exercise _coerce directly with a synthetic action)."""
        act = argparse.ArgumentParser(add_help=False).add_argument(
            "--xs", nargs="+", type=int)
        with pytest.raises(SystemExit, match="at least one value"):
            fig._coerce(act, "xs", [], "fig")

    def test_nargs_star_empty_ok(self):
        """nargs='*' permits an empty list, like the CLI."""
        act = argparse.ArgumentParser(add_help=False).add_argument(
            "--xs", nargs="*", type=int)
        assert fig._coerce(act, "xs", [], "fig") == []

    @pytest.mark.parametrize("dpi", [0, -10, 1.5])
    def test_dpi_nonpositive_rejected(self, tmp_path, dpi):
        """A per-figure dpi must be a positive int (same rule as top-level dpi
        and the CLI), failing fast as a SpecError — not at matplotlib draw."""
        with pytest.raises(SystemExit, match="positive_int"):
            fig._build_args({"name": "x", "preset": "scalar", "dpi": dpi},
                            {"config": "c"}, None, tmp_path, False, False)

    @pytest.mark.parametrize("figure", [
        {"name": "x", "preset": "scalar", "dpi": True},
        {"name": "x", "preset": "scalar", "figsize": [True, 9]},
        {"name": "x", "preset": "scalar", "depth_max": True},
    ])
    def test_boolean_rejected_for_numeric_options(self, tmp_path, figure):
        """A YAML boolean for a typed numeric option must be rejected up front,
        not pass through as Python True and crash matplotlib (e.g. dpi=True ->
        FT_Set_Char_Size invalid ppem)."""
        with pytest.raises(SystemExit, match="is a boolean"):
            fig._build_args(figure, {"config": "c"}, None, tmp_path, False, False)


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


class TestOutputConfig:
    def test_default_is_cwd_png_tree(self):
        assert fig._output_config({}) == (".", None, None)

    def test_pdf_selected(self):
        assert fig._output_config({"output_pdf": "r.pdf"}) == (None, "r.pdf", None)

    def test_dir_and_pdf_conflict(self):
        with pytest.raises(SystemExit, match="only one of"):
            fig._output_config({"output_dir": "x", "output_pdf": "y.pdf"})

    def test_default_dpi_passthrough(self):
        assert fig._output_config({"dpi": 200}) == (".", None, 200)

    def test_dpi_non_integer_rejected(self):
        with pytest.raises(SystemExit, match="must be an integer"):
            fig._output_config({"dpi": 1.5})
        with pytest.raises(SystemExit, match="must be an integer"):
            fig._output_config({"dpi": True})  # bool is not a real dpi

    def test_dpi_nonpositive_rejected(self):
        with pytest.raises(SystemExit, match="positive"):
            fig._output_config({"dpi": 0})


def _write_min_ctd_combo(root):
    """A minimal ctd_combo_00 trajectory for end-to-end figure-driver tests."""
    import numpy as np
    import xarray as xr

    per = 30
    t0 = np.datetime64("2025-01-20T00:00:00")
    depth = np.concatenate([np.linspace(0, 60, per // 2),
                            np.linspace(60, 0, per // 2)])
    time = np.array([t0 + np.timedelta64(i, "s") for i in range(per)],
                    dtype="datetime64[ns]")
    ds = xr.Dataset(
        {"JAC_T": (("time",), 28 - 0.1 * depth, {"units": "degree_Celsius"}),
         "SP": (("time",), 34.5 + 0.01 * depth, {"units": "PSU"}),
         "sigma0": (("time",), -1 + 0.25 * depth, {"units": "kg/m^3"}),
         "depth": (("time",), depth, {"units": "m", "positive": "down"}),
         "lat": (("time",), np.full(per, 18.0)),
         "lon": (("time",), np.full(per, 130.0))},
        coords={"time": ("time", time)},
    )
    ds.attrs["id"] = "drv"
    out = root / "ctd_combo_00"
    out.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out / "combo.nc")


class TestPdfOutput:
    def test_multipage_pdf(self, tmp_path):
        pytest.importorskip("matplotlib").use("Agg")
        _write_min_ctd_combo(tmp_path)
        spec = tmp_path / "spec.yaml"
        spec.write_text(
            f"source: {{root: {tmp_path}}}\n"
            f"output_pdf: {tmp_path}/report.pdf\n"
            "dpi: 120\n"
            "figures:\n"
            "  - {name: ts, preset: scalar, vars: [JAC_T, SP], figsize: [7, 6], title: TS}\n"
            "  - {name: dens, preset: scalar, vars: [sigma0]}\n"
        )
        res = fig.run(_cli(spec=str(spec)))
        pdf = tmp_path / "report.pdf"
        assert res == str(pdf)
        assert pdf.exists() and pdf.read_bytes().startswith(b"%PDF")
        # one /Pages tree node + two /Page leaves -> two rendered pages
        body = pdf.read_bytes()
        assert body.count(b"/Type /Page") + body.count(b"/Type/Page") == 3

    def test_empty_pdf_raises_and_writes_nothing(self, tmp_path, monkeypatch):
        pytest.importorskip("matplotlib").use("Agg")
        # A preset that yields no figures (e.g. every section empty) must not
        # leave behind an invalid zero-page PDF.
        monkeypatch.setattr(scalar, "build_figures", lambda a: [])
        spec = tmp_path / "spec.yaml"
        spec.write_text(
            f"source: {{root: {tmp_path}}}\n"
            f"output_pdf: {tmp_path}/empty.pdf\n"
            "figures:\n  - {name: a, preset: scalar, vars: [JAC_T]}\n"
        )
        with pytest.raises(SystemExit, match="nothing to write"):
            fig.run(_cli(spec=str(spec)))
        assert not (tmp_path / "empty.pdf").exists()

    def test_figsize_wrong_length_rejected(self, tmp_path):
        with pytest.raises(SystemExit, match="exactly 2 values"):
            fig._build_args({"name": "x", "preset": "scalar", "figsize": [7]},
                            {"config": "c"}, None, tmp_path, False, False)

    def test_later_failure_leaves_no_partial_pdf(self, tmp_path):
        """A first valid figure renders, but a second figure with an invalid
        option must abort with NO report.pdf and no leftover temp file — not a
        plausible-looking PDF silently missing pages."""
        pytest.importorskip("matplotlib").use("Agg")
        _write_min_ctd_combo(tmp_path)
        spec = tmp_path / "spec.yaml"
        spec.write_text(
            f"source: {{root: {tmp_path}}}\n"
            f"output_pdf: {tmp_path}/report.pdf\n"
            "figures:\n"
            "  - {name: ok, preset: scalar, vars: [JAC_T]}\n"
            "  - {name: bad, preset: profiles, product: epsilon}\n"  # invalid choice
        )
        with pytest.raises(SystemExit, match="must be one of"):
            fig.run(_cli(spec=str(spec)))
        assert not (tmp_path / "report.pdf").exists()
        assert not list(tmp_path.glob(".figspec_*.pdf"))  # temp cleaned up
