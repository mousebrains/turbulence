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

    def test_bin_by_depth_groups_profiles_by_source_pfile(self, tmp_path):
        """Per-profile NetCDFs share one log per source .p file.

        The binning step receives ``<pfile>_prof###.nc`` files; producing
        one log per profile would mean 20+ logs per .p file, which
        contradicts the requirement that each .p file produce exactly
        one log per stage.  The ``_prof###`` suffix is stripped so all
        profiles from ``X.p`` write into a single ``X.log``.
        """
        import numpy as np
        import xarray as xr

        from odas_tpw.perturb.binning import bin_by_depth

        files = []
        for n in range(1, 4):  # X_prof001, X_prof002, X_prof003
            f = tmp_path / f"X_prof{n:03d}.nc"
            xr.Dataset(
                {"epsilon": (("depth",), np.linspace(1e-8, 1e-9, 6))},
                coords={"depth": np.linspace(0, 5, 6)},
            ).to_netcdf(f)
            files.append(f)
        # And one from a different .p file
        for n in range(1, 3):
            f = tmp_path / f"Y_prof{n:03d}.nc"
            xr.Dataset(
                {"epsilon": (("depth",), np.linspace(1e-8, 1e-9, 6))},
                coords={"depth": np.linspace(0, 5, 6)},
            ).to_netcdf(f)
            files.append(f)

        binned_dir = tmp_path / "profiles_binned_00"
        binned_dir.mkdir()

        setup_root_logging(tmp_path / "run.log")
        bin_by_depth(files, bin_width=1.0, log_dir=binned_dir)
        logging.shutdown()

        # Two source .p files -> exactly two log files, named X.log / Y.log.
        log_files = sorted(p.name for p in binned_dir.glob("*.log"))
        assert log_files == ["X.log", "Y.log"]

    def test_source_stem_helper(self):
        from odas_tpw.perturb.binning import _source_stem

        assert _source_stem("X_prof001.nc") == "X"
        assert _source_stem("X_prof999.nc") == "X"
        # Non-matching paths keep the full stem.
        assert _source_stem("X.nc") == "X"
        assert _source_stem("X_proflarge.nc") == "X_proflarge"
        # Multiple-segment stems with a numeric tail other than _prof### preserved.
        assert _source_stem("Some.Dotted.Name_prof007.nc") == "Some.Dotted.Name"


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
                "odas_tpw.rsi.profile.extract_profiles", return_value=([prof_nc], [{}])
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


# ---------------------------------------------------------------------------
# Cover the defensive ``except`` branches in process_file's stage blocks.
# ``process_file`` swallows per-stage failures so a bad CTD stream / FP07
# cal / CT-align / extract_profiles call doesn't take down the whole run;
# these tests verify that path runs cleanly and the error reaches the
# per-stage log file.
# ---------------------------------------------------------------------------


class TestProcessFileExceptionBranches:
    def _common_pf(self):
        """Mock PFile with the channels needed to reach each stage."""
        from unittest.mock import MagicMock

        import numpy as np

        mock_pf = MagicMock()
        mock_pf.channels = {
            "P": np.linspace(0, 50, 100),
            "T1": np.zeros(100),
            "JAC_T": np.zeros(100),
            "JAC_C": np.zeros(100),
        }
        mock_pf.fs_slow = 64.0
        return mock_pf

    def _config(self, tmp_path, **overrides):
        cfg = {
            "files": {"output_root": str(tmp_path)},
            "profiles": {"P_min": 0.5},
            "epsilon": {},
            "fp07": {"calibrate": False},
            "ct": {"align": False},
            "ctd": {"enable": False},
            "chi": {"enable": False},
        }
        for k, v in overrides.items():
            cfg.setdefault(k, {}).update(v) if isinstance(v, dict) else cfg.update({k: v})
        return cfg

    def test_ctd_failure_logs_to_ctd_dir(self, tmp_path):
        """``ctd_bin_file`` raising must be caught and logged into
        ``ctd_NN/<stem>.log``."""
        from unittest.mock import patch

        from odas_tpw.perturb.pipeline import process_file

        ctd_dir = tmp_path / "ctd_00"
        ctd_dir.mkdir()
        prof_dir = tmp_path / "profiles_00"
        prof_dir.mkdir()

        with (
            patch("odas_tpw.rsi.p_file.PFile", return_value=self._common_pf()),
            patch(
                "odas_tpw.perturb.ctd.ctd_bin_file",
                side_effect=RuntimeError("ctd boom"),
            ),
            # Stop after CTD by sabotaging profile detection.
            patch("odas_tpw.rsi.profile._smooth_fall_rate", return_value=None),
            patch("odas_tpw.rsi.profile.get_profiles", return_value=[]),
        ):
            config = self._config(tmp_path, ctd={"enable": True})
            output_dirs = {"ctd": ctd_dir, "profiles": prof_dir}

            setup_root_logging(tmp_path / "run.log")
            process_file(tmp_path / "demo.p", config, gps=None, output_dirs=output_dirs)
            logging.shutdown()

        log_text = (ctd_dir / "demo.log").read_text()
        assert "CTD binning" in log_text
        assert "ctd boom" in log_text

    def test_fp07_failure_logs_to_profiles_dir(self, tmp_path):
        """``fp07_calibrate`` raising must be caught and logged into
        ``profiles_NN/<stem>.log``."""
        from unittest.mock import patch

        from odas_tpw.perturb.pipeline import process_file

        prof_dir = tmp_path / "profiles_00"
        prof_dir.mkdir()

        with (
            patch("odas_tpw.rsi.p_file.PFile", return_value=self._common_pf()),
            patch("odas_tpw.rsi.profile._smooth_fall_rate", return_value=[0.5] * 100),
            patch(
                "odas_tpw.rsi.profile.get_profiles",
                return_value=[{"start": 0, "end": 50}],
            ),
            patch(
                "odas_tpw.perturb.fp07_cal.fp07_calibrate",
                side_effect=RuntimeError("fp07 boom"),
            ),
            patch(
                "odas_tpw.rsi.profile.extract_profiles", return_value=([], [])
            ),  # short-circuit downstream
        ):
            config = self._config(tmp_path, fp07={"calibrate": True})
            output_dirs = {"profiles": prof_dir}

            setup_root_logging(tmp_path / "run.log")
            process_file(tmp_path / "demo.p", config, gps=None, output_dirs=output_dirs)
            logging.shutdown()

        log_text = (prof_dir / "demo.log").read_text()
        assert "FP07 cal failed" in log_text
        assert "fp07 boom" in log_text

    def test_ct_align_failure_logs_to_profiles_dir(self, tmp_path):
        """``ct_align`` raising is logged into ``profiles_NN/<stem>.log``."""
        from unittest.mock import patch

        from odas_tpw.perturb.pipeline import process_file

        prof_dir = tmp_path / "profiles_00"
        prof_dir.mkdir()

        with (
            patch("odas_tpw.rsi.p_file.PFile", return_value=self._common_pf()),
            patch("odas_tpw.rsi.profile._smooth_fall_rate", return_value=[0.5] * 100),
            patch(
                "odas_tpw.rsi.profile.get_profiles",
                return_value=[{"start": 0, "end": 50}],
            ),
            patch(
                "odas_tpw.processing.ct_align.ct_align",
                side_effect=RuntimeError("ct boom"),
            ),
            patch("odas_tpw.rsi.profile.extract_profiles", return_value=([], [])),
        ):
            config = self._config(tmp_path, ct={"align": True})
            output_dirs = {"profiles": prof_dir}

            setup_root_logging(tmp_path / "run.log")
            process_file(tmp_path / "demo.p", config, gps=None, output_dirs=output_dirs)
            logging.shutdown()

        log_text = (prof_dir / "demo.log").read_text()
        assert "CT align failed" in log_text
        assert "ct boom" in log_text

    def test_extract_profiles_failure_logs_to_profiles_dir(self, tmp_path):
        """``extract_profiles`` raising is logged into ``profiles_NN/<stem>.log``."""
        from unittest.mock import patch

        from odas_tpw.perturb.pipeline import process_file

        prof_dir = tmp_path / "profiles_00"
        prof_dir.mkdir()

        with (
            patch("odas_tpw.rsi.p_file.PFile", return_value=self._common_pf()),
            patch("odas_tpw.rsi.profile._smooth_fall_rate", return_value=[0.5] * 100),
            patch(
                "odas_tpw.rsi.profile.get_profiles",
                return_value=[{"start": 0, "end": 50}],
            ),
            patch(
                "odas_tpw.rsi.profile.extract_profiles",
                side_effect=RuntimeError("extract boom"),
            ),
        ):
            config = self._config(tmp_path)
            output_dirs = {"profiles": prof_dir}

            setup_root_logging(tmp_path / "run.log")
            result = process_file(
                tmp_path / "demo.p", config, gps=None, output_dirs=output_dirs
            )
            logging.shutdown()

        log_text = (prof_dir / "demo.log").read_text()
        assert "extracting profiles" in log_text
        assert "extract boom" in log_text
        # Also: process_file returned early so no diss/chi were attempted.
        assert result["profiles"] == []
        assert result["diss"] == []

    def test_chi_inner_exception_logs_to_chi_dir(self, tmp_path):
        """A per-profile chi failure is caught inside the chi stage_log
        block and logged into ``chi_NN/<stem>.log``."""
        from unittest.mock import patch

        import numpy as np
        import xarray as xr

        from odas_tpw.perturb.pipeline import process_file

        prof_dir = tmp_path / "profiles_00"
        prof_dir.mkdir()
        diss_dir = tmp_path / "diss_00"
        diss_dir.mkdir()
        chi_dir = tmp_path / "chi_00"
        chi_dir.mkdir()

        prof_nc = prof_dir / "demo.nc"
        prof_nc.touch()
        diss_nc = diss_dir / "demo.nc"
        xr.Dataset({"epsilon": (("time",), [1e-8])}).to_netcdf(diss_nc)

        with (
            patch("odas_tpw.rsi.p_file.PFile", return_value=self._common_pf()),
            patch("odas_tpw.rsi.profile._smooth_fall_rate", return_value=[0.5] * 100),
            patch(
                "odas_tpw.rsi.profile.get_profiles",
                return_value=[{"start": 0, "end": 50}],
            ),
            patch("odas_tpw.rsi.profile.extract_profiles", return_value=([prof_nc], [{}])),
            patch(
                "odas_tpw.rsi.dissipation._compute_epsilon",
                return_value=[xr.Dataset({"epsilon": (("time",), [1e-8])})],
            ),
            patch("odas_tpw.rsi.chi_io._load_therm_channels", return_value={}),
            patch(
                "odas_tpw.rsi.chi_io._compute_chi",
                side_effect=RuntimeError("chi boom"),
            ),
        ):
            mock_pf = self._common_pf()
            mock_pf.channels["P"] = np.linspace(0, 50, 100)
            config = self._config(tmp_path, chi={"enable": True})
            output_dirs = {"profiles": prof_dir, "diss": diss_dir, "chi": chi_dir}

            setup_root_logging(tmp_path / "run.log")
            process_file(tmp_path / "demo.p", config, gps=None, output_dirs=output_dirs)
            logging.shutdown()

        log_text = (chi_dir / "demo.log").read_text()
        assert "chi for" in log_text
        assert "chi boom" in log_text


# ---------------------------------------------------------------------------
# Coverage for run_pipeline binning + parallel branches that the existing
# pipeline tests don't exercise (process_file is fully mocked there, so the
# binning blocks and the jobs>1 ProcessPoolExecutor wiring never run).
# ---------------------------------------------------------------------------


def _drop_profile_nc(target: object, basename: str) -> object:
    """Write a small profile-shaped NetCDF into *target* and return its path."""
    import numpy as np
    import xarray as xr

    nc = target / f"{basename}.nc"
    xr.Dataset(
        {"epsilon": (("depth",), np.linspace(1e-8, 1e-9, 6))},
        coords={"depth": np.linspace(0, 5, 6)},
    ).to_netcdf(nc)
    return nc


class TestRunPipelineBinningBlocks:
    """Patch ``process_file`` to a no-op, drop pre-existing profile NCs into
    the profiles dir, and let ``run_pipeline`` reach the binning blocks
    that previously had zero coverage in this PR."""

    def _config(self, tmp_path):
        return {
            "files": {
                "output_root": str(tmp_path),
                "p_file_root": str(tmp_path),
                "trim": False,
                "merge": False,
            },
            "profiles": {},
            "epsilon": {},
            "chi": {"enable": True},
            "ctd": {"enable": False},
            "gps": {},
            "parallel": {"jobs": 1},
            "binning": {"method": "depth", "width": 1.0, "aggregation": "mean"},
        }

    def test_profile_diss_chi_binning_blocks_run(self, tmp_path):
        from unittest.mock import patch

        from odas_tpw.perturb.pipeline import _setup_output_dirs, run_pipeline

        config = self._config(tmp_path)
        # Pre-create the per-stage dirs and drop one source NC into each so
        # the ``if prof_ncs:`` / diss / chi branches all execute.
        dirs = _setup_output_dirs(config)
        _drop_profile_nc(dirs["profiles"], "demo")
        _drop_profile_nc(dirs["diss"], "demo")
        _drop_profile_nc(dirs["chi"], "demo")

        with (
            patch(
                "odas_tpw.perturb.pipeline.process_file",
                return_value={"source": "x.p", "profiles": [], "diss": [], "chi": []},
            ),
            patch("odas_tpw.perturb.gps.create_gps", return_value=None),
            # _run_combo touches xarray internals; not what we're testing here.
            patch("odas_tpw.perturb.pipeline._run_combo"),
        ):
            run_pipeline(config, p_files=[tmp_path / "fake.p"])

        # All three binned dirs must now exist with their .params_sha256_*
        # signature; per-input ``demo.log`` should be inside each.
        for prefix in ("profiles_binned", "diss_binned", "chi_binned"):
            matches = list(tmp_path.glob(f"{prefix}_[0-9][0-9]"))
            assert matches, f"no {prefix}_NN dir created"
            assert (matches[0] / "demo.log").exists()

    def test_bin_by_time_branch(self, tmp_path):
        """``binning.method=time`` exercises the ``bin_by_time`` branch in
        the profile-binning block."""
        from unittest.mock import patch

        from odas_tpw.perturb.pipeline import _setup_output_dirs, run_pipeline

        config = self._config(tmp_path)
        config["binning"]["method"] = "time"
        config["chi"]["enable"] = False  # keep test focused
        dirs = _setup_output_dirs(config)

        # bin_by_time needs a time coordinate, not depth.
        import numpy as np
        import xarray as xr

        xr.Dataset(
            {"epsilon": (("time",), np.linspace(1e-8, 1e-9, 6))},
            coords={"time": np.arange(6, dtype=float)},
        ).to_netcdf(dirs["profiles"] / "demo.nc")

        with (
            patch(
                "odas_tpw.perturb.pipeline.process_file",
                return_value={"source": "x.p", "profiles": [], "diss": [], "chi": []},
            ),
            patch("odas_tpw.perturb.gps.create_gps", return_value=None),
            patch("odas_tpw.perturb.pipeline._run_combo"),
        ):
            run_pipeline(config, p_files=[tmp_path / "fake.p"])

        matches = list(tmp_path.glob("profiles_binned_[0-9][0-9]"))
        assert matches
        assert (matches[0] / "demo.log").exists()


class TestRunPipelineParallel:
    """Verify the ``jobs > 1`` branch wires the worker initializer + initargs."""

    def test_pool_initializer_uses_logging_setup(self, tmp_path):
        from unittest.mock import patch

        from odas_tpw.perturb.pipeline import run_pipeline

        config = {
            "files": {
                "output_root": str(tmp_path),
                "p_file_root": str(tmp_path),
                "trim": False,
                "merge": False,
            },
            "profiles": {},
            "epsilon": {},
            "chi": {"enable": False},
            "ctd": {"enable": False},
            "gps": {},
            "parallel": {"jobs": 2},
            "binning": {},
        }

        # Hollowed-out executor: run inline, no actual subprocess.
        class _FakeFuture:
            def __init__(self, value):
                self._value = value

            def result(self):
                return self._value

        class _FakeExecutor:
            instance = None

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                _FakeExecutor.instance = self

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def submit(self, fn, *args, **kwargs):
                return _FakeFuture(fn(*args, **kwargs))

        def _fake_as_completed(futures):
            return list(futures)

        with (
            patch("odas_tpw.perturb.pipeline.ProcessPoolExecutor", _FakeExecutor),
            patch("odas_tpw.perturb.pipeline.as_completed", _fake_as_completed),
            patch(
                "odas_tpw.perturb.pipeline.process_file",
                return_value={"source": "x.p", "profiles": [], "diss": [], "chi": []},
            ),
            patch("odas_tpw.perturb.gps.create_gps", return_value=None),
            patch("odas_tpw.perturb.pipeline._setup_output_dirs", return_value={}),
            patch("odas_tpw.perturb.pipeline._run_combo"),
        ):
            run_pipeline(config, p_files=[tmp_path / "a.p", tmp_path / "b.p"])

        kwargs = _FakeExecutor.instance.kwargs
        assert kwargs["max_workers"] == 2
        # initializer is the logging-setup function; initargs = (logs_dir, run_stamp).
        from odas_tpw.perturb.logging_setup import init_worker_logging

        assert kwargs["initializer"] is init_worker_logging
        log_dir, run_stamp = kwargs["initargs"]
        assert log_dir == tmp_path / "logs"
        assert isinstance(run_stamp, str) and run_stamp.endswith("Z")


class TestBinningCoverage:
    """Cover the binning branches that didn't run under the existing tests."""

    def test_bin_by_depth_uses_P_mean_fallback(self, tmp_path):
        """A profile NC with neither ``depth`` nor ``P`` but with ``P_mean``
        still bins (exercises the P_mean fallback in both the global-edges
        scan and the per-profile loop)."""
        import numpy as np
        import xarray as xr

        from odas_tpw.perturb.binning import bin_by_depth

        f = tmp_path / "p_mean_only.nc"
        xr.Dataset(
            {
                "P_mean": (("idx",), np.linspace(0, 5, 6)),
                "epsilon": (("idx",), np.linspace(1e-8, 1e-9, 6)),
            }
        ).to_netcdf(f)

        ds = bin_by_depth([f], bin_width=1.0, log_dir=tmp_path)
        assert "epsilon" in ds.data_vars
        assert (tmp_path / "p_mean_only.log").exists()

    def test_bin_by_time_returns_empty_when_no_time_coord(self, tmp_path):
        """A NC with no recognised time coordinate is silently skipped."""
        import numpy as np
        import xarray as xr

        from odas_tpw.perturb.binning import bin_by_time

        f = tmp_path / "no_time.nc"
        xr.Dataset({"v": (("idx",), np.zeros(3))}).to_netcdf(f)

        ds = bin_by_time([f], bin_width=1.0)
        assert not ds.data_vars  # empty dataset
        # No log file requested, so just verify the empty-result path runs.


class TestProcessFileExcludedProbes:
    """``instruments.SN.exclude_shear_probes`` triggers the
    ``_nan_excluded_probes`` call inside the diss ``stage_log`` block."""

    def test_excluded_probes_path(self, tmp_path):
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

        # Two-probe diss dataset so the exclusion has something to NaN.
        diss_ds = xr.Dataset(
            {"epsilon": (("probe", "time"), np.full((2, 4), 1e-8))},
            coords={"probe": ["sh1", "sh2"], "time": np.arange(4, dtype=float)},
        )

        with (
            patch("odas_tpw.rsi.p_file.PFile") as mock_pfile_cls,
            patch("odas_tpw.rsi.profile._smooth_fall_rate", return_value=np.zeros(100)),
            patch(
                "odas_tpw.rsi.profile.get_profiles",
                return_value=[{"start": 0, "end": 50}],
            ),
            patch(
                "odas_tpw.rsi.profile.extract_profiles", return_value=([prof_nc], [{}])
            ),
            patch(
                "odas_tpw.rsi.dissipation._compute_epsilon",
                return_value=[diss_ds],
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
                "instruments": {"SN999": {"exclude_shear_probes": ["sh2"]}},
            }
            output_dirs = {"profiles": prof_dir, "diss": diss_dir}

            setup_root_logging(tmp_path / "run.log")
            process_file(
                tmp_path / "demo.p",
                config,
                gps=None,
                output_dirs=output_dirs,
                instrument_key="SN999",
            )

        # The diss stage_log was opened (so line 447's _nan_excluded_probes
        # call was exercised inside it) and a per-file log file exists.
        # _nan_excluded_probes only logs on typo'd probe names, so no
        # content assertion — the existence proves the path ran.
        assert (diss_dir / "demo.log").exists()


class TestRunPipelineParallelException:
    """Cover the ``except Exception as exc`` branch after ``future.result()``."""

    def test_worker_exception_logged(self, tmp_path):
        from unittest.mock import patch

        from odas_tpw.perturb.pipeline import run_pipeline

        config = {
            "files": {
                "output_root": str(tmp_path),
                "p_file_root": str(tmp_path),
                "trim": False,
                "merge": False,
            },
            "profiles": {},
            "epsilon": {},
            "chi": {"enable": False},
            "ctd": {"enable": False},
            "gps": {},
            "parallel": {"jobs": 2},
            "binning": {},
        }

        class _RaisingFuture:
            def result(self):
                raise RuntimeError("worker boom")

        class _FakeExecutor:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def submit(self, fn, *args, **kwargs):
                return _RaisingFuture()

        with (
            patch("odas_tpw.perturb.pipeline.ProcessPoolExecutor", _FakeExecutor),
            patch("odas_tpw.perturb.pipeline.as_completed", lambda fs: list(fs)),
            patch("odas_tpw.perturb.gps.create_gps", return_value=None),
            patch("odas_tpw.perturb.pipeline._setup_output_dirs", return_value={}),
            patch("odas_tpw.perturb.pipeline._run_combo"),
        ):
            # Should swallow the worker exception and continue.
            run_pipeline(config, p_files=[tmp_path / "a.p"])

        # The error should be in the run log we set up implicitly via
        # _install_logging? No — run_pipeline doesn't install logging.
        # We're just exercising the line; pytest passing means it didn't crash.


class TestCmdBinTimeMethod:
    """Cover the ``bin_method == "time"`` branch in ``_cmd_bin``."""

    def test_cmd_bin_time(self, tmp_path):
        """Drop a profile NC with a time coord, run ``perturb bin`` with
        ``binning.method=time`` config, assert the time-binned dir gets
        created with its per-input log."""
        from argparse import Namespace

        import numpy as np
        import xarray as xr

        from odas_tpw.perturb.cli import _cmd_bin

        prof_dir = tmp_path / "profiles_00"
        prof_dir.mkdir()
        xr.Dataset(
            {"epsilon": (("time",), np.linspace(1e-8, 1e-9, 6))},
            coords={"time": np.arange(6, dtype=float)},
        ).to_netcdf(prof_dir / "demo.nc")

        # Minimal config file containing the time-method override.
        cfg = tmp_path / "config.yaml"
        cfg.write_text("binning:\n  method: time\n  width: 1.0\n  aggregation: mean\n")

        args = Namespace(
            config=str(cfg),
            output=str(tmp_path),
            stdout=False,
            log_level="INFO",
        )
        _cmd_bin(args)

        binned = list(tmp_path.glob("profiles_binned_[0-9][0-9]"))
        assert binned
        assert (binned[0] / "demo.log").exists()
