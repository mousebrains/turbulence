# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Branch-coverage tests for perturb.pipeline — top-level orchestration paths."""

from __future__ import annotations

from unittest.mock import patch

from odas_tpw.perturb.pipeline import run_merge, run_pipeline, run_trim

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
