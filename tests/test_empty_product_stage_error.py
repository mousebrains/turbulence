"""A per-profile stage that produced NOTHING must fail the run.

`stage_errors` has always meant "a product the config asked for was not
written", but only a stage that raised as a whole recorded one.  A fault
hitting every profile identically arrives as N file errors -- absorbed one at
a time -- so the run wrote an empty product directory, reported "N file
error(s)", and exited 0.

That is issue #184: a `_compute_chi()` TypeError cost 1330 profiles of chi on
ARCTERX-2022 and the exit status said the run succeeded.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import xarray as xr

from odas_tpw.perturb.pipeline import PipelineResult, _empty_product_stage_errors


def _dirs(tmp_path: Path, *stages: str) -> dict[str, Path]:
    out = {}
    for s in stages:
        d = tmp_path / f"{s}_00"
        d.mkdir()
        out[s] = d
    return out


def _ncs(n: int) -> list[Path]:
    return [Path(f"prof{i:03d}.nc") for i in range(n)]


class TestEmptyProductIsAStageError:
    def test_the_issue_184_shape_fails_the_run(self, tmp_path):
        """1330 profiles, chi enabled, chi/ empty -> stage error, not exit 0."""
        errs = _empty_product_stage_errors(
            _dirs(tmp_path, "profiles", "diss", "chi"),
            _ncs(1330),
            {"diss": _ncs(1330), "chi": []},
            n_file_errors=1330,
        )
        assert len(errs) == 1
        assert errs[0].startswith("chi:")
        assert "1330 profile(s) exist" in errs[0]
        assert "chi_00/ holds no .nc files" in errs[0]
        # and it must actually flip the run's exit status
        outcome = PipelineResult()
        outcome.stage_errors += errs
        assert outcome.ok is False
        assert "STAGE FAILED" in outcome.summary()

    def test_healthy_run_is_silent(self, tmp_path):
        assert (
            _empty_product_stage_errors(
                _dirs(tmp_path, "profiles", "diss", "chi"),
                _ncs(1330),
                {"diss": _ncs(1330), "chi": _ncs(1330)},
                n_file_errors=0,
            )
            == []
        )

    def test_no_profiles_is_not_a_failure(self, tmp_path):
        """An all-deck-file corpus legitimately yields nothing downstream."""
        assert (
            _empty_product_stage_errors(
                _dirs(tmp_path, "profiles", "diss", "chi"),
                [],
                {"diss": [], "chi": []},
                n_file_errors=0,
            )
            == []
        )

    def test_disabled_stage_is_not_a_failure(self, tmp_path):
        """chi.enable=false leaves no chi dir; that is not a missing product."""
        assert (
            _empty_product_stage_errors(
                _dirs(tmp_path, "profiles", "diss"),
                _ncs(10),
                {"diss": _ncs(10), "chi": []},
                n_file_errors=0,
            )
            == []
        )

    def test_both_stages_empty_reports_both(self, tmp_path):
        errs = _empty_product_stage_errors(
            _dirs(tmp_path, "profiles", "diss", "chi"),
            _ncs(5),
            {"diss": [], "chi": []},
            n_file_errors=5,
        )
        assert sorted(e.split(":")[0] for e in errs) == ["chi", "diss"]

    @pytest.mark.parametrize(
        ("n_file_errors", "expected"),
        [
            (7, "7 file error(s) were reported"),
            (0, "no file errors were reported"),
        ],
    )
    def test_message_distinguishes_failure_from_silence(
        self, tmp_path, n_file_errors, expected
    ):
        """Empty-with-errors and empty-without-errors are different bugs: one
        failed loudly per profile, the other produced nothing quietly."""
        errs = _empty_product_stage_errors(
            _dirs(tmp_path, "profiles", "chi"),
            _ncs(3),
            {"chi": []},
            n_file_errors=n_file_errors,
        )
        assert expected in errs[0]


class TestCacheInteraction:
    def test_a_fully_cached_rerun_does_not_false_positive(self, tmp_path):
        """THE false positive to avoid: on a re-run where every file is served
        from cache, the per-file results carry no paths while the products sit
        complete on disk. The check counts the DIRECTORY, so it stays silent."""
        errs = _empty_product_stage_errors(
            _dirs(tmp_path, "profiles", "diss", "chi"),
            _ncs(1330),  # globbed from disk, not from this run's results
            {"diss": _ncs(1330), "chi": _ncs(1330)},
            n_file_errors=0,
        )
        assert errs == []


class TestWiredIntoRunPipeline:
    """The helper is only useful if `run_pipeline` actually consults it and the
    result reaches the exit status."""

    @staticmethod
    def _config(tmp_path: Path) -> dict:
        return {
            "files": {
                "p_file_root": str(tmp_path),
                "output_root": str(tmp_path / "out"),
                "trim": False,
                "merge": False,
            },
            "profiles": {},
            "epsilon": {},
            "chi": {"enable": True},
            "ctd": {"enable": False},
            "gps": {},
            "parallel": {"jobs": 1},
            "binning": {},
        }

    def test_chi_failing_on_every_profile_flips_the_exit_status(self, tmp_path):
        """The #184 shape end to end: profiles and diss written, chi raises on
        every profile, so chi_00/ ends up empty.  Before this check the run
        reported '0 stage failure(s)' and exited 0."""
        from unittest.mock import patch

        from odas_tpw.perturb import pipeline as pl

        out = tmp_path / "out"
        dirs = {}
        for stage in ("profiles", "diss", "chi"):
            d = out / f"{stage}_00"
            d.mkdir(parents=True)
            dirs[stage] = d

        p_file = tmp_path / "cast.p"
        p_file.touch()

        def _fake_process_file(*args, **kwargs):
            """Write profiles and diss; fail chi on every profile.

            Writing them from INSIDE process_file matters: the pipeline unlinks
            a stem's previous outputs before reprocessing it, so files staged
            beforehand are (correctly) swept away.
            """
            stem = kwargs["output_stem"]
            written = {"profiles": [], "diss": []}
            for i in range(1, 4):
                for stage in ("profiles", "diss"):
                    f = dirs[stage] / f"{stem}_prof{i:03d}.nc"
                    f.touch()
                    written[stage].append(str(f))
            return {
                "source": str(p_file),
                "profiles": written["profiles"],
                "diss": written["diss"],
                "chi": [],
                "errors": [f"chi {stem}_prof{i:03d}.nc: boom" for i in range(1, 4)],
            }

        with (
            patch.object(pl, "_setup_output_dirs", return_value=dirs),
            patch.object(pl, "_run_combo", return_value=[]),
            patch("odas_tpw.perturb.gps.create_gps", return_value=None),
            # An empty Dataset, not None: the writer checks `ds.data_vars`
            # and skips an empty one, which is what we want -- binning is not
            # what this test is about.
            patch("odas_tpw.perturb.binning.bin_by_depth", return_value=xr.Dataset()),
            patch("odas_tpw.perturb.binning.bin_by_time", return_value=xr.Dataset()),
            patch("odas_tpw.perturb.binning.bin_diss", return_value=xr.Dataset()),
            patch("odas_tpw.perturb.binning.bin_chi", return_value=xr.Dataset()),
            patch.object(pl, "process_file", side_effect=_fake_process_file),
        ):
            result = pl.run_pipeline(self._config(tmp_path), p_files=[p_file])

        assert not list(dirs["chi"].glob("*.nc")), "test setup: chi must be empty"
        assert list(dirs["profiles"].glob("*.nc")), "test setup: profiles must exist"

        assert result.ok is False, "an entirely empty chi product must fail the run"
        assert any(e.startswith("chi:") for e in result.stage_errors), (
            f"expected a chi stage error, got {result.stage_errors}"
        )
        assert "STAGE FAILED" in result.summary()

    def test_a_healthy_run_still_exits_clean(self, tmp_path):
        """The guard must not make every ordinary run fail."""
        from unittest.mock import patch

        from odas_tpw.perturb import pipeline as pl

        out = tmp_path / "out"
        dirs = {}
        for stage in ("profiles", "diss", "chi"):
            d = out / f"{stage}_00"
            d.mkdir(parents=True)
            dirs[stage] = d

        p_file = tmp_path / "cast.p"
        p_file.touch()

        def _fake_process_file(*args, **kwargs):
            stem = kwargs["output_stem"]
            written = {"profiles": [], "diss": [], "chi": []}
            for i in range(1, 4):
                for stage in ("profiles", "diss", "chi"):
                    f = dirs[stage] / f"{stem}_prof{i:03d}.nc"
                    f.touch()
                    written[stage].append(str(f))
            return {"source": str(p_file), **written}

        with (
            patch.object(pl, "_setup_output_dirs", return_value=dirs),
            patch.object(pl, "_run_combo", return_value=[]),
            patch("odas_tpw.perturb.gps.create_gps", return_value=None),
            # An empty Dataset, not None: the writer checks `ds.data_vars`
            # and skips an empty one, which is what we want -- binning is not
            # what this test is about.
            patch("odas_tpw.perturb.binning.bin_by_depth", return_value=xr.Dataset()),
            patch("odas_tpw.perturb.binning.bin_by_time", return_value=xr.Dataset()),
            patch("odas_tpw.perturb.binning.bin_diss", return_value=xr.Dataset()),
            patch("odas_tpw.perturb.binning.bin_chi", return_value=xr.Dataset()),
            patch.object(pl, "process_file", side_effect=_fake_process_file),
        ):
            result = pl.run_pipeline(self._config(tmp_path), p_files=[p_file])

        assert result.ok is True, f"unexpected stage errors: {result.stage_errors}"
