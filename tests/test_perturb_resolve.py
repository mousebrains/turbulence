# Jun-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb.resolve — config -> output directory (read-only)."""

from types import SimpleNamespace

import pytest

from odas_tpw.perturb import config as cfg
from odas_tpw.perturb import resolve


def _write_stage(root, stage, config, seq):
    """Create ``{stage}_{seq:02d}`` with the exact signature the pipeline writes."""
    d = root / f"{stage}_{seq:02d}"
    d.mkdir()
    section, params, upstream = cfg.stage_signature(stage, config)
    cfg.write_signature(d, section, params, upstream=upstream)
    return d


def _cfg(output_root, **overrides):
    """A minimal perturb config dict (sections default via merge_config)."""
    base = {"files": {"output_root": str(output_root)}}
    base.update(overrides)
    return base


class TestStageDir:
    def test_exact_match(self, tmp_path):
        c = _cfg(tmp_path)
        d = _write_stage(tmp_path, "chi_combo", c, 0)
        assert resolve.stage_dir(c, "chi_combo") == d

    def test_ignores_output_root(self, tmp_path):
        """The F-1 case: a run done with --output stores an output_root the
        config YAML never had; matching must ignore output_root."""
        run_cfg = {"files": {"output_root": "/elsewhere", "p_file_root": "/in"}}
        d = _write_stage(tmp_path, "chi_combo", run_cfg, 0)
        plot_cfg = {"files": {"output_root": str(tmp_path), "p_file_root": "/in"}}
        assert resolve.stage_dir(plot_cfg, "chi_combo", output_root=tmp_path) == d

    def test_p_file_root_discriminates(self, tmp_path):
        """Different input datasets in one output_root stay distinguishable
        (p_file_root is kept in the comparison, unlike output_root)."""
        cA = {"files": {"output_root": str(tmp_path), "p_file_root": "/legA"}}
        cB = {"files": {"output_root": str(tmp_path), "p_file_root": "/legB"}}
        dA = _write_stage(tmp_path, "chi_combo", cA, 0)
        dB = _write_stage(tmp_path, "chi_combo", cB, 1)
        assert resolve.stage_dir(cA, "chi_combo") == dA
        assert resolve.stage_dir(cB, "chi_combo") == dB

    def test_duplicate_signature_multi_match_uses_newest(self, tmp_path):
        """Two dirs with the *same* signature (a concurrent same-config run that
        raced for a sequence number, or a hand-copied signature) are config- and
        input-identical, so their data is equivalent: resolve to the newest with
        a warning rather than locking the user out with a StageConflict."""
        c = _cfg(tmp_path)
        d0 = _write_stage(tmp_path, "chi_combo", c, 0)
        d1 = tmp_path / "chi_combo_01"
        d1.mkdir()
        sig = next(d0.glob(".params_sha256_*"))
        (d1 / sig.name).write_text(sig.read_text())
        with pytest.warns(UserWarning, match="share the given config's signature"):
            got = resolve.stage_dir(c, "chi_combo")
        assert got == d1  # newest of the two equivalent matches

    def test_duplicate_signature_multi_match_strict_uses_newest(self, tmp_path):
        """Even under strict the duplicates are *exact* matches (not drift), so
        the same warn-and-pick-newest applies rather than a hard error."""
        c = _cfg(tmp_path)
        d0 = _write_stage(tmp_path, "chi_combo", c, 0)
        d1 = tmp_path / "chi_combo_01"
        d1.mkdir()
        sig = next(d0.glob(".params_sha256_*"))
        (d1 / sig.name).write_text(sig.read_text())
        with pytest.warns(UserWarning, match="share the given config's signature"):
            assert resolve.stage_dir(c, "chi_combo", strict=True) == d1

    def test_keeps_non_path_files_flags(self, tmp_path):
        """trim/merge are data-affecting and must still discriminate."""
        c_trim = _cfg(tmp_path, files={"output_root": str(tmp_path), "trim": True})
        c_notrim = _cfg(tmp_path, files={"output_root": str(tmp_path), "trim": False})
        _write_stage(tmp_path, "chi_combo", c_trim, 0)
        with pytest.raises(resolve.StageConflict):
            resolve.stage_dir(c_notrim, "chi_combo", strict=True)

    def test_discriminates_configs(self, tmp_path):
        c0 = _cfg(tmp_path, top_trim={"enable": False})
        c1 = _cfg(tmp_path, top_trim={"enable": True})
        d0 = _write_stage(tmp_path, "chi_combo", c0, 0)
        d1 = _write_stage(tmp_path, "chi_combo", c1, 1)
        assert resolve.stage_dir(c1, "chi_combo") == d1
        assert resolve.stage_dir(c0, "chi_combo") == d0

    def test_per_stage_independence(self, tmp_path):
        """A chi change must not perturb the ctd_combo signature."""
        c0 = _cfg(tmp_path, chi={"enable": True, "fft_length": 256})
        c1 = _cfg(tmp_path, chi={"enable": True, "fft_length": 512})
        _write_stage(tmp_path, "ctd_combo", c0, 0)
        assert resolve.stage_dir(c1, "ctd_combo") == tmp_path / "ctd_combo_00"

    def test_drift_single_dir_warns_and_returns(self, tmp_path):
        _write_stage(tmp_path, "chi_combo", _cfg(tmp_path, top_trim={"enable": False}), 0)
        drifted = _cfg(tmp_path, top_trim={"enable": True})
        with pytest.warns(UserWarning, match="config drift"):
            assert resolve.stage_dir(drifted, "chi_combo").name == "chi_combo_00"

    def test_drift_multiple_dirs_raises(self, tmp_path):
        _write_stage(tmp_path, "chi_combo", _cfg(tmp_path, top_trim={"enable": False}), 0)
        _write_stage(tmp_path, "chi_combo", _cfg(tmp_path, top_trim={"enable": True}), 1)
        other = _cfg(tmp_path, binning={"width": 7.0})  # matches neither
        with pytest.raises(resolve.StageConflict, match="refusing to guess"):
            resolve.stage_dir(other, "chi_combo")

    def test_strict_raises_on_single_drift(self, tmp_path):
        _write_stage(tmp_path, "chi_combo", _cfg(tmp_path, top_trim={"enable": False}), 0)
        with pytest.raises(resolve.StageConflict):
            resolve.stage_dir(_cfg(tmp_path, top_trim={"enable": True}), "chi_combo", strict=True)

    def test_latest_ignores_signature(self, tmp_path):
        _write_stage(tmp_path, "chi_combo", _cfg(tmp_path, top_trim={"enable": False}), 0)
        _write_stage(tmp_path, "chi_combo", _cfg(tmp_path, top_trim={"enable": True}), 1)
        other = _cfg(tmp_path, binning={"width": 7.0})
        assert resolve.stage_dir(other, "chi_combo", latest=True).name == "chi_combo_01"

    def test_no_dirs_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="never run"):
            resolve.stage_dir(_cfg(tmp_path), "chi_combo")

    def test_glob_active_chars_in_output_root(self, tmp_path):
        """An output_root containing glob chars (e.g. '[') must still resolve."""
        root = tmp_path / "leg_[A]"
        root.mkdir()
        c = _cfg(root)
        d = _write_stage(root, "chi_combo", c, 0)
        assert resolve.stage_dir(c, "chi_combo") == d

    def test_none_files_section(self, tmp_path):
        """A config with files: None must not AttributeError."""
        c = {"files": None}
        d = _write_stage(tmp_path, "chi_combo", c, 0)
        assert resolve.stage_dir(c, "chi_combo", output_root=tmp_path) == d

    def test_unknown_stage_raises(self, tmp_path):
        with pytest.raises(ValueError, match="unknown stage"):
            resolve.stage_dir(_cfg(tmp_path), "bogus")

    def test_output_root_override(self, tmp_path):
        """Data lives under a dir different from the config's output_root."""
        c = {"files": {"output_root": "/not/here"}}
        d = _write_stage(tmp_path, "chi_combo", c, 0)
        assert resolve.stage_dir(c, "chi_combo", output_root=tmp_path) == d


class TestResolveForArgs:
    def test_config_path(self, tmp_path):
        _write_stage(tmp_path, "chi_combo", _cfg(tmp_path), 0)
        ypath = tmp_path / "p.yaml"
        ypath.write_text(f"files:\n  output_root: {tmp_path}\n")
        args = SimpleNamespace(config=str(ypath), root=None, strict=False, latest=False)
        assert resolve.resolve_for_args(args, "chi_combo") == str(tmp_path / "chi_combo_00")

    def test_config_failure_is_systemexit(self, tmp_path):
        ypath = tmp_path / "p.yaml"
        ypath.write_text(f"files:\n  output_root: {tmp_path}\n")  # no chi_combo dir
        args = SimpleNamespace(config=str(ypath), root=None, strict=False, latest=False)
        with pytest.raises(SystemExit):
            resolve.resolve_for_args(args, "chi_combo")

    def test_optional_absent_returns_none(self, tmp_path):
        """An optional stage with no dir degrades to None, not SystemExit."""
        ypath = tmp_path / "p.yaml"
        ypath.write_text(f"files:\n  output_root: {tmp_path}\n")
        args = SimpleNamespace(config=str(ypath), root=None, strict=False, latest=False)
        assert resolve.resolve_for_args(args, "chi_combo", optional=True) is None
        with pytest.raises(SystemExit):  # required (default) still aborts
            resolve.resolve_for_args(args, "chi_combo")

    def test_optional_does_not_swallow_conflict(self, tmp_path):
        """A config conflict (ambiguous/drift) must surface even for an optional
        stage — not be masked as a silent degrade-to-None."""
        _write_stage(tmp_path, "chi_combo", _cfg(tmp_path, top_trim={"enable": False}), 0)
        _write_stage(tmp_path, "chi_combo", _cfg(tmp_path, top_trim={"enable": True}), 1)
        ypath = tmp_path / "p.yaml"
        ypath.write_text(
            f"files: {{output_root: {tmp_path}}}\n"
            "top_trim: {enable: true, max_depth: 99}\n"  # matches neither dir
        )
        args = SimpleNamespace(config=str(ypath), root=None, strict=False, latest=False)
        with pytest.raises(SystemExit, match="refusing to guess"):
            resolve.resolve_for_args(args, "chi_combo", optional=True)

    def test_missing_config_not_swallowed_by_optional(self, tmp_path):
        """A typo'd/absent --config raises FileNotFoundError inside load_config;
        the optional-stage degrade-to-None must NOT swallow it, or the user gets
        a plot silently missing data with no hint their config never loaded."""
        args = SimpleNamespace(
            config=str(tmp_path / "does_not_exist.yaml"),
            root=None, strict=False, latest=False,
        )
        with pytest.raises(SystemExit):  # required
            resolve.resolve_for_args(args, "chi_combo")
        with pytest.raises(SystemExit):  # optional must ALSO abort (B1)
            resolve.resolve_for_args(args, "chi_combo", optional=True)

    def test_malformed_config_not_swallowed_by_optional(self, tmp_path):
        """A config with an unknown section raises ValueError in load_config;
        that is always fatal too, never a silent optional degrade."""
        ypath = tmp_path / "p.yaml"
        ypath.write_text("not_a_real_section:\n  foo: 1\n")
        args = SimpleNamespace(config=str(ypath), root=None, strict=False, latest=False)
        with pytest.raises(SystemExit):
            resolve.resolve_for_args(args, "chi_combo", optional=True)

    def test_latest_fallback_without_config(self, tmp_path):
        (tmp_path / "chi_combo_00").mkdir()
        args = SimpleNamespace(config=None, root=str(tmp_path), strict=False, latest=False)
        assert resolve.resolve_for_args(args, "chi_combo") is not None

    def test_requires_root_or_config(self):
        args = SimpleNamespace(config=None, root=None, strict=False, latest=False)
        with pytest.raises(SystemExit, match="one of --config or --root"):
            resolve.resolve_for_args(args, "chi_combo")
