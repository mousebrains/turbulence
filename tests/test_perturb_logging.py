# May-2026, Claude and Pat Welch, pat@mousebrains.com
"""Unit tests for perturb.logging_setup."""

from __future__ import annotations

import contextlib
import logging

import pytest

from odas_tpw.perturb.logging_setup import (
    init_worker_logging,
    run_timestamp,
    setup_root_logging,
    stage_log,
)


@pytest.fixture(autouse=True)
def _reset_root_logger():
    """Snapshot/restore root logger state so tests don't bleed."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    yield
    for h in list(root.handlers):
        if h not in saved_handlers:
            root.removeHandler(h)
            with contextlib.suppress(Exception):
                h.close()
    root.setLevel(saved_level)


class TestRunTimestamp:
    def test_format(self):
        ts = run_timestamp()
        # YYYYMMDDTHHMMSSZ — 16 chars
        assert len(ts) == 16
        assert ts[8] == "T"
        assert ts.endswith("Z")


class TestSetupRootLogging:
    def test_writes_file_only_by_default(self, tmp_path):
        log_file = tmp_path / "logs" / "run.log"
        setup_root_logging(log_file)

        logging.getLogger("test").info("hello world")

        # Force flush by removing the handler
        logging.shutdown()

        text = log_file.read_text()
        assert "hello world" in text

    def test_creates_parent_directory(self, tmp_path):
        log_file = tmp_path / "deeper" / "logs" / "run.log"
        assert not log_file.parent.exists()
        setup_root_logging(log_file)
        assert log_file.parent.is_dir()

    def test_stdout_flag_adds_stream_handler(self, tmp_path, capsys):
        log_file = tmp_path / "run.log"
        setup_root_logging(log_file, stdout=True)

        logging.getLogger("test").info("on console")

        captured = capsys.readouterr()
        # stderr because we use sys.stderr
        assert "on console" in captured.err

    def test_no_stdout_means_no_stream_handler(self, tmp_path, capsys):
        log_file = tmp_path / "run.log"
        setup_root_logging(log_file, stdout=False)

        logging.getLogger("test").info("file only")

        captured = capsys.readouterr()
        assert "file only" not in captured.err
        assert "file only" not in captured.out

    def test_idempotent_replaces_owned_handlers(self, tmp_path):
        # First call creates one handler set
        setup_root_logging(tmp_path / "first.log", stdout=True)
        first_count = len(logging.getLogger().handlers)

        # Second call should drop the first set and install the second
        setup_root_logging(tmp_path / "second.log", stdout=True)
        second_count = len(logging.getLogger().handlers)

        assert first_count == second_count

    def test_does_not_disturb_external_handlers(self, tmp_path):
        root = logging.getLogger()
        external = logging.NullHandler()
        root.addHandler(external)
        try:
            setup_root_logging(tmp_path / "run.log")
            assert external in root.handlers
            setup_root_logging(tmp_path / "run2.log")
            assert external in root.handlers
        finally:
            root.removeHandler(external)


class TestStageLog:
    def test_writes_to_stage_dir(self, tmp_path):
        setup_root_logging(tmp_path / "run.log")

        with stage_log(tmp_path / "diss_00", "a") as log_path:
            assert log_path == tmp_path / "diss_00" / "a.log"
            logging.getLogger("test").info("inside diss for a")

        logging.shutdown()
        text = (tmp_path / "diss_00" / "a.log").read_text()
        assert "inside diss for a" in text

    def test_creates_stage_dir_if_missing(self, tmp_path):
        setup_root_logging(tmp_path / "run.log")
        new_dir = tmp_path / "new_stage"
        with stage_log(new_dir, "b"):
            pass
        assert new_dir.is_dir()

    def test_handler_removed_on_normal_exit(self, tmp_path):
        setup_root_logging(tmp_path / "run.log")
        before = len(logging.getLogger().handlers)
        with stage_log(tmp_path / "stage", "c"):
            during = len(logging.getLogger().handlers)
            assert during == before + 1
        after = len(logging.getLogger().handlers)
        assert after == before

    def test_handler_removed_on_exception(self, tmp_path):
        setup_root_logging(tmp_path / "run.log")
        before = len(logging.getLogger().handlers)
        with pytest.raises(RuntimeError), stage_log(tmp_path / "stage", "d"):
            raise RuntimeError("boom")
        after = len(logging.getLogger().handlers)
        assert after == before

    def test_no_op_when_stage_dir_is_none(self, tmp_path):
        setup_root_logging(tmp_path / "run.log")
        before = len(logging.getLogger().handlers)
        with stage_log(None, "e") as log_path:
            assert log_path is None
            assert len(logging.getLogger().handlers) == before

    def test_records_route_to_run_log_too(self, tmp_path):
        run_log = tmp_path / "run.log"
        setup_root_logging(run_log)
        with stage_log(tmp_path / "stage_00", "f"):
            logging.getLogger("test").info("fan-out check")
        logging.shutdown()
        assert "fan-out check" in run_log.read_text()
        assert "fan-out check" in (tmp_path / "stage_00" / "f.log").read_text()


class TestInitWorkerLogging:
    def test_creates_pid_log(self, tmp_path):
        ts = "20260504T120000Z"
        log_file = init_worker_logging(tmp_path, ts)

        # File name uses parent run stamp + this process's pid
        import os

        assert log_file.name == f"worker_{ts}_{os.getpid()}.log"
        assert log_file.parent == tmp_path

        logging.getLogger("test").info("worker line")
        logging.shutdown()

        assert "worker line" in log_file.read_text()

    def test_replaces_owned_handlers_on_repeat(self, tmp_path):
        ts = "20260504T120000Z"
        init_worker_logging(tmp_path, ts)
        first = len(logging.getLogger().handlers)
        init_worker_logging(tmp_path, ts)
        second = len(logging.getLogger().handlers)
        assert first == second


# ---------------------------------------------------------------------------
# Integration: each new wiring point produces the expected log file
# ---------------------------------------------------------------------------


class TestCliInstallLogging:
    """``_install_logging`` builds the log path from the resolved output_root
    and installs handlers so the file appears immediately."""

    def test_writes_run_log_under_output_root(self, tmp_path):
        from argparse import Namespace

        from odas_tpw.perturb.cli import _install_logging
        from odas_tpw.perturb.logging_setup import reset_run_stamp

        reset_run_stamp()
        args = Namespace(
            output=str(tmp_path / "results"), stdout=False, log_level="INFO"
        )
        log_path = _install_logging(args, config={})

        assert log_path.parent == tmp_path / "results" / "logs"
        assert log_path.name.startswith("run_") and log_path.suffix == ".log"

        logging.getLogger("test").info("cli install check")
        logging.shutdown()
        assert "cli install check" in log_path.read_text()


class TestBinningPerInputLogs:
    """A.p, B.p binned together should produce A.log, B.log inside the binned dir."""

    def test_bin_by_depth_per_input_logs(self, tmp_path):
        import numpy as np
        import xarray as xr

        from odas_tpw.perturb.binning import bin_by_depth

        # Two synthetic per-profile NetCDFs with depth + one variable
        files = []
        for name in ("A", "B"):
            ds = xr.Dataset(
                {"epsilon": (("depth",), np.linspace(1e-8, 1e-9, 10))},
                coords={"depth": np.linspace(0, 9, 10)},
            )
            f = tmp_path / f"{name}.nc"
            ds.to_netcdf(f)
            files.append(f)

        binned_dir = tmp_path / "profiles_binned_00"
        binned_dir.mkdir()

        # Drive a dummy log handler so the FileHandler records emit something
        setup_root_logging(tmp_path / "run.log")
        out = bin_by_depth(files, bin_width=1.0, log_dir=binned_dir)
        # Touch a log line attributable to each file path so the FileHandlers
        # have something to flush; the binning code itself is silent.
        # (We only verify the log files were *opened* by stage_log.)
        logging.shutdown()

        assert out.data_vars  # binning produced something
        assert (binned_dir / "A.log").exists()
        assert (binned_dir / "B.log").exists()


class TestProcessFilePerStageLogs:
    """Each stage block in process_file should drop ``<stem>.log`` into its
    output dir, even when downstream stages early-exit."""

    def test_profiles_stage_log_created_when_no_pressure(self, tmp_path):
        """Even on the early ``no pressure channel`` exit, the profiles stage
        log captures the warning."""
        from unittest.mock import MagicMock, patch

        import numpy as np

        from odas_tpw.perturb.pipeline import process_file

        with patch("odas_tpw.rsi.p_file.PFile") as mock_pfile_cls:
            mock_pf = MagicMock()
            # No 'P' channel — process_file warns and returns early.
            mock_pf.channels = {"T1": np.zeros(50)}
            mock_pfile_cls.return_value = mock_pf

            prof_dir = tmp_path / "profiles_00"
            prof_dir.mkdir()
            output_dirs = {"profiles": prof_dir}

            config = {
                "files": {"output_root": str(tmp_path)},
                "profiles": {"P_min": 0.5},
                "epsilon": {},
                "fp07": {"calibrate": False},
                "ct": {"align": False},
                "ctd": {"enable": False},
                "chi": {"enable": False},
            }

            setup_root_logging(tmp_path / "run.log")
            process_file(tmp_path / "demo.p", config, gps=None, output_dirs=output_dirs)
            logging.shutdown()

        assert (prof_dir / "demo.log").exists()
        assert "No pressure channel" in (prof_dir / "demo.log").read_text()

    def test_diss_stage_log_created(self, tmp_path):
        """When diss runs, ``diss_NN/<stem>.log`` should exist."""
        from unittest.mock import MagicMock, patch

        import numpy as np
        import xarray as xr

        from odas_tpw.perturb.pipeline import process_file

        prof_dir = tmp_path / "profiles_00"
        prof_dir.mkdir()
        diss_dir = tmp_path / "diss_00"
        diss_dir.mkdir()
        prof_nc = prof_dir / "demo.nc"
        prof_nc.touch()

        with (
            patch("odas_tpw.rsi.p_file.PFile") as mock_pfile_cls,
            patch("odas_tpw.rsi.profile._smooth_fall_rate", return_value=np.zeros(100)),
            patch(
                "odas_tpw.rsi.profile.get_profiles",
                return_value=[{"start": 0, "end": 50}],
            ),
            patch(
                "odas_tpw.rsi.profile.extract_profiles", return_value=[prof_nc]
            ),
            patch(
                "odas_tpw.rsi.dissipation._compute_epsilon",
                return_value=[xr.Dataset({"epsilon": (("time",), [1e-8])})],
            ),
        ):
            mock_pf = MagicMock()
            mock_pf.channels = {"P": np.linspace(0, 50, 100), "T1": np.zeros(100)}
            mock_pf.fs_slow = 64.0
            mock_pfile_cls.return_value = mock_pf

            config = {
                "files": {"output_root": str(tmp_path)},
                "profiles": {"P_min": 0.5},
                "epsilon": {},
                "fp07": {"calibrate": False},
                "ct": {"align": False},
                "ctd": {"enable": False},
                "chi": {"enable": False},
            }
            output_dirs = {"profiles": prof_dir, "diss": diss_dir}

            setup_root_logging(tmp_path / "run.log")
            process_file(tmp_path / "demo.p", config, gps=None, output_dirs=output_dirs)
            logging.shutdown()

        assert (prof_dir / "demo.log").exists()
        assert (diss_dir / "demo.log").exists()


class TestComboStageLog:
    """Wrapping a combo step in stage_log should produce ``combo.log``."""

    def test_combo_log_created(self, tmp_path):
        combo_dir = tmp_path / "combo_00"
        combo_dir.mkdir()

        setup_root_logging(tmp_path / "run.log")
        with stage_log(combo_dir, "combo"):
            logging.getLogger("odas_tpw.perturb.combo").info("assembled X profiles")
        logging.shutdown()

        log_file = combo_dir / "combo.log"
        assert log_file.exists()
        assert "assembled X profiles" in log_file.read_text()
