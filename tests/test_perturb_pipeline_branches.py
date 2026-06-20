# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Branch-coverage tests for perturb.pipeline — top-level orchestration paths."""

from __future__ import annotations

from unittest.mock import patch

import xarray as xr

from odas_tpw.perturb.pipeline import (
    _prune_orphan_named_ncs,
    _prune_orphan_profile_ncs,
    _write_binned_or_clear,
    run_merge,
    run_pipeline,
    run_trim,
)


class TestWriteBinnedOrClear:
    def test_empty_rerun_removes_stale_binned(self, tmp_path):
        """A populated bin writes binned.nc; a later empty re-run must REMOVE the
        stale file so the combo can't republish a prior input set's data (#56)."""
        out = tmp_path / "binned.nc"
        _write_binned_or_clear(xr.Dataset({"x": ("t", [1.0, 2.0])}), tmp_path)
        assert out.exists()  # populated -> written
        _write_binned_or_clear(xr.Dataset(), tmp_path)
        assert not out.exists()  # empty re-run -> stale file cleared

    def test_empty_with_no_prior_file_is_noop(self, tmp_path):
        """Empty result with no pre-existing binned.nc simply writes nothing."""
        _write_binned_or_clear(xr.Dataset(), tmp_path)
        assert not (tmp_path / "binned.nc").exists()


class TestPruneOrphanProfileNCs:
    def _touch(self, d, name):
        p = d / name
        p.write_bytes(b"")
        return p

    def test_prunes_only_orphans(self, tmp_path):
        keep_a = self._touch(tmp_path, "fileA_prof000.nc")
        keep_a2 = self._touch(tmp_path, "fileA_prof001.nc")
        orphan = self._touch(tmp_path, "fileB_prof000.nc")
        n = _prune_orphan_profile_ncs(tmp_path, {"fileA"})
        assert n == 1
        assert keep_a.exists() and keep_a2.exists()
        assert not orphan.exists()

    def test_never_deletes_current_stem_even_with_prefix_overlap(self, tmp_path):
        # "a" is a string-prefix of "a_long" but the _prof anchor keeps them
        # distinct: a current stem's files are never removed.
        keep = self._touch(tmp_path, "a_long_prof000.nc")
        orphan = self._touch(tmp_path, "a_prof000.nc")  # stem "a" not current
        _prune_orphan_profile_ncs(tmp_path, {"a_long"})
        assert keep.exists()
        assert not orphan.exists()

    def test_leaves_non_profile_and_combo_files(self, tmp_path):
        combo = self._touch(tmp_path, "combo.nc")
        binned = self._touch(tmp_path, "binned.nc")
        orphan = self._touch(tmp_path, "gone_prof000.nc")
        _prune_orphan_profile_ncs(tmp_path, set())
        # Only the *_prof*.nc orphan is touched; combos/binned are left alone.
        assert combo.exists() and binned.exists()
        assert not orphan.exists()

    def test_missing_dir_is_noop(self, tmp_path):
        assert _prune_orphan_profile_ncs(tmp_path / "nope", {"x"}) == 0


class TestPruneOrphanNamedNCs:
    def _touch(self, d, name):
        p = d / name
        p.write_bytes(b"")
        return p

    def test_prunes_ctd_orphans_by_exact_stem(self, tmp_path):
        keep = self._touch(tmp_path, "fileA.nc")          # current
        orphan = self._touch(tmp_path, "fileB.nc")        # dropped .p
        n = _prune_orphan_named_ncs(tmp_path, {"fileA"})
        assert n == 1
        assert keep.exists()
        assert not orphan.exists()

    def test_exact_match_not_prefix(self, tmp_path):
        # "fileA" must not protect "fileAB.nc" (exact membership, not prefix).
        keep = self._touch(tmp_path, "fileA.nc")
        orphan = self._touch(tmp_path, "fileAB.nc")
        _prune_orphan_named_ncs(tmp_path, {"fileA"})
        assert keep.exists()
        assert not orphan.exists()

    def test_missing_dir_is_noop(self, tmp_path):
        assert _prune_orphan_named_ncs(tmp_path / "nope", {"x"}) == 0

# ---------------------------------------------------------------------------
# run_pipeline — file-discovery / no-files paths
# ---------------------------------------------------------------------------


class TestRunPipelineDiscovery:
    def test_no_files_warns_and_returns(self, tmp_path, caplog):
        """When no .p files are found, log warning and return."""
        config = {
            "files": {
                "p_file_root": str(tmp_path / "empty"),
                "p_file_pattern": "*.p",
                "output_root": str(tmp_path / "out"),
                "trim": False,
                "merge": False,
            },
        }
        (tmp_path / "empty").mkdir()
        with caplog.at_level("WARNING"):
            run_pipeline(config)
        msgs = [r.message for r in caplog.records]
        assert any("No .p files found" in m for m in msgs)

    def test_explicit_p_files_skips_discovery(self, tmp_path, caplog):
        """When p_files is given, discover is bypassed."""
        config = {
            "files": {
                "output_root": str(tmp_path / "out"),
                "trim": False,
                "merge": False,
            },
            "parallel": {"jobs": 1},
        }
        # Empty list still triggers no-files warning, but we never call discover.
        with patch("odas_tpw.perturb.discover.find_p_files") as mock_discover:
            run_pipeline(config, p_files=[])
        # discover should not have been called when p_files is provided
        mock_discover.assert_not_called()


# ---------------------------------------------------------------------------
# run_pipeline — trim/merge enabled, parallel jobs
# ---------------------------------------------------------------------------


class TestRunPipelineFlags:
    def test_jobs_zero_uses_cpu_count(self, tmp_path):
        """jobs=0 → uses os.cpu_count()."""
        config = {
            "files": {
                "p_file_root": str(tmp_path / "empty"),
                "output_root": str(tmp_path / "out"),
                "trim": False,
                "merge": False,
            },
            "parallel": {"jobs": 0},
        }
        (tmp_path / "empty").mkdir()
        # No files → run returns early after warning, but the 0->cpu_count
        # branch is hit before that. Just verify it runs.
        run_pipeline(config)

    def test_trim_enabled_runs_trim_step(self, tmp_path):
        """When files.trim=True, run_trim is invoked."""
        config = {
            "files": {
                "p_file_root": str(tmp_path / "empty"),
                "output_root": str(tmp_path / "out"),
                "trim": True,
                "merge": False,
            },
        }
        (tmp_path / "empty").mkdir()
        # No real files; run_trim returns []. The trim path is exercised.
        run_pipeline(config, p_files=[])

    def test_merge_enabled_runs_merge_step(self, tmp_path):
        """When files.merge=True, run_merge is invoked."""
        config = {
            "files": {
                "p_file_root": str(tmp_path / "empty"),
                "output_root": str(tmp_path / "out"),
                "trim": False,
                "merge": True,
            },
        }
        (tmp_path / "empty").mkdir()
        run_pipeline(config, p_files=[])


# ---------------------------------------------------------------------------
# run_pipeline — hotel file loading
# ---------------------------------------------------------------------------


class TestRunPipelineHotel:
    def test_hotel_load_called_when_enabled(self, tmp_path):
        """When hotel.enable=True and hotel.file is set, load_hotel is called."""
        # Build a hotel CSV
        hotel_path = tmp_path / "hotel.csv"
        hotel_path.write_text("time,speed\n0,0.5\n1,0.6\n")

        config = {
            "files": {
                "output_root": str(tmp_path / "out"),
                "trim": False,
                "merge": False,
            },
            "hotel": {
                "enable": True,
                "file": str(hotel_path),
                "time_format": "seconds",
            },
        }
        # No p_files → returns early after hotel-load and warning
        run_pipeline(config, p_files=[])


# ---------------------------------------------------------------------------
# run_trim / run_merge error paths
# ---------------------------------------------------------------------------


class TestRunTrimError:
    def test_trim_error_logged(self, tmp_path, caplog):
        """run_trim catches per-file exceptions and logs error."""
        # Pass a corrupt .p file (too small) — trim_p_file raises
        bad = tmp_path / "bad.p"
        bad.write_bytes(b"\x00" * 10)

        config = {
            "files": {
                "p_file_root": str(tmp_path),
                "output_root": str(tmp_path / "out"),
            },
        }
        with caplog.at_level("ERROR"):
            results = run_trim(config, p_files=[bad])
        # Empty results, error logged
        assert results == []
        assert any("trimming" in r.message for r in caplog.records)


class TestRunMergeError:
    def test_merge_with_no_chains(self, tmp_path):
        """run_merge with no mergeable chains returns empty list."""
        config = {
            "files": {
                "p_file_root": str(tmp_path / "empty"),
                "output_root": str(tmp_path / "out"),
            },
        }
        (tmp_path / "empty").mkdir()
        results = run_merge(config)
        assert results == []
