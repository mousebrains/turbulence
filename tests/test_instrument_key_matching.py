# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""An ``instruments`` key that matches no input file must not fail silently.

process_file looks a file's per-instrument settings up by the raw directory
name of the ``.p`` file, so a key naming anything else is ignored without a
word. Two guards, each pinned here:

* a key YAML loads as something other than a string (``465:`` is an int,
  ``2017-10-24:`` a date) can never match, so the config is refused at load;
* a string key that matches no input file of the run is reported at run start
  and in the run summary, whichever settings it holds.
"""

import datetime
import logging
from unittest.mock import patch

import pytest
import xarray as xr

from odas_tpw.perturb import pipeline as pl
from odas_tpw.perturb.config import _validate_instruments, load_config
from odas_tpw.perturb.pipeline import _instrument_key_notices, _probe_exclusion_notices


class TestNonStringKeyRefusedAtLoad:
    @pytest.mark.parametrize("key", [465, 194.5, datetime.date(2017, 10, 24), True])
    def test_validator_refuses(self, key):
        with pytest.raises(ValueError, match=r"quote it \("):
            _validate_instruments({key: {"exclude_shear_probes": ["sh2"]}})

    def test_string_key_still_accepted(self):
        _validate_instruments({"465": {"exclude_shear_probes": ["sh2"]}})

    def test_unquoted_numeric_key_in_yaml(self, tmp_path):
        cfg = tmp_path / "perturb.yaml"
        cfg.write_text("instruments:\n  465:\n    exclude_shear_probes: [sh2]\n")
        with pytest.raises(ValueError, match=r"key 465 loaded as int.*quote it \('465'\)"):
            load_config(cfg)

    def test_quoted_numeric_key_in_yaml(self, tmp_path):
        cfg = tmp_path / "perturb.yaml"
        cfg.write_text("instruments:\n  '465':\n    exclude_shear_probes: [sh2]\n")
        assert "465" in load_config(cfg)["instruments"]


class TestUnmatchedKeyNotices:
    def test_unmatched_key_is_named_with_its_settings(self):
        (line,) = _instrument_key_notices(
            {"SN194": {"fp07_tau_scale": {"T1": 1.1}, "exclude_shear_probes": ["sh2"]}},
            {"VMP"},
        )
        assert line.startswith(
            "Instrument key: instruments.SN194 (exclude_shear_probes, fp07_tau_scale) "
            "matches no input file of this run (instrument keys: VMP)"
        )
        assert "NOT applied" in line and "raw directory name" in line

    def test_matched_key_is_silent(self):
        assert _instrument_key_notices({"VMP": {"exclude_shear_probes": ["sh2"]}}, {"VMP"}) == []

    def test_no_instruments_is_silent(self):
        assert _instrument_key_notices(None, {"VMP"}) == []
        assert _instrument_key_notices({}, {"VMP"}) == []

    def test_one_line_per_unmatched_key_whatever_it_holds(self):
        lines = _instrument_key_notices(
            {"a": {"exclude": []}, "b": {"fp07_response": "x.nc"}, "VMP": {}}, {"VMP"}
        )
        assert [ln.split(" (")[0] for ln in lines] == [
            "Instrument key: instruments.a",
            "Instrument key: instruments.b",
        ]

    def test_non_string_key_built_in_code_gets_the_hint(self):
        """load_config refuses it, but run_pipeline also takes a config built in code."""
        (line,) = _instrument_key_notices({465: {"exclude_shear_probes": ["sh2"]}}, {"465"})
        assert "not a string (int); quote it ('465')" in line

    def test_probe_exclusion_notices_leave_it_to_the_general_notice(self):
        """One line per unmatched key, not one per setting that noticed it."""
        exc = {"sn194": {"exclude": [{"probe": "sh2", "reason": "bad"}]}}
        assert _probe_exclusion_notices(exc, {"SN194"}, {}) == []
        assert len(_instrument_key_notices(exc, {"SN194"})) == 1


def test_run_pipeline_reports_an_unmatched_key_once(tmp_path, caplog):
    """At run start, at WARNING, and first in the summary -- logged exactly once."""
    out = tmp_path / "out"
    dirs = {}
    for stage in ("profiles", "diss"):
        dirs[stage] = out / f"{stage}_00"
        dirs[stage].mkdir(parents=True)
    src = tmp_path / "VMP"
    src.mkdir()
    p_file = src / "cast.p"
    p_file.touch()

    def _fake_process_file(*args, **kwargs):
        stem = kwargs["output_stem"]
        written = {}
        for stage in ("profiles", "diss"):
            f = dirs[stage] / f"{stem}_prof001.nc"
            f.touch()
            written[stage] = [str(f)]
        return {"source": str(p_file), **written}

    config = {
        "files": {"p_file_root": str(tmp_path), "output_root": str(out),
                  "trim": False, "merge": False},
        "profiles": {}, "epsilon": {}, "chi": {"enable": False},
        "ctd": {"enable": False}, "gps": {}, "parallel": {"jobs": 1}, "binning": {},
        "instruments": {
            "SN194": {"exclude_shear_probes": ["sh2"]},  # files live in VMP/
            "VMP": {"exclude_shear_probes": ["sh1"]},
        },
    }
    with (
        patch.object(pl, "_setup_output_dirs", return_value=dirs),
        patch.object(pl, "_run_combo", return_value=[]),
        patch("odas_tpw.perturb.gps.create_gps", return_value=None),
        patch("odas_tpw.perturb.binning.bin_by_depth", return_value=xr.Dataset()),
        patch("odas_tpw.perturb.binning.bin_by_time", return_value=xr.Dataset()),
        patch("odas_tpw.perturb.binning.bin_diss", return_value=xr.Dataset()),
        patch.object(pl, "process_file", side_effect=_fake_process_file),
        caplog.at_level(logging.WARNING, logger=pl.logger.name),
    ):
        result = pl.run_pipeline(config, p_files=[p_file])
    assert result.ok
    assert result.notices[0].startswith("Instrument key: instruments.SN194 ")
    assert not any("instruments.VMP " in n for n in result.notices)
    logged = [r.getMessage() for r in caplog.records if "instruments.SN194" in r.getMessage()]
    assert len(logged) == 1
