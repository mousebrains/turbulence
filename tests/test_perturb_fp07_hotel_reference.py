# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""The in-pipeline FP07 fit against a sparse or gappy reference.

Three changes, each pinned here:

* a NaN in a profile's reference splits the profile instead of silencing it;
* a reference from the HOTEL file is refused unless explicitly allowed, before
  any stage runs;
* the run summary says which files kept the factory coefficients.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

from odas_tpw.perturb.config import _mgr
from odas_tpw.perturb.fp07_cal import _lag_segments, _lowpass_filter, fp07_calibrate
from odas_tpw.perturb.hotel import HotelData
from odas_tpw.perturb.pipeline import (
    PipelineResult,
    _fp07_notices,
    _hotel_reference_refusal,
    run_pipeline,
)


class TestLagSegments:
    def test_clean_profile_is_returned_whole(self):
        """No non-finite sample: scored exactly as before, however short."""
        x = np.arange(50.0)
        assert _lag_segments(x, x, 10, 20, min_len=1000) == [(10, 20)]

    def test_split_at_gaps_in_either_series(self):
        T = np.arange(100.0)
        f = np.arange(100.0)
        T[40:45] = np.nan  # hotel gap blanked by the merge
        f[80] = np.nan  # a non-finite probe sample
        assert _lag_segments(T, f, 0, 99, min_len=10) == [(0, 39), (45, 79), (81, 99)]

    def test_pieces_shorter_than_min_len_are_dropped(self):
        T = np.arange(100.0)
        T[40:45] = np.nan
        assert _lag_segments(T, np.arange(100.0), 0, 99, min_len=50) == [(45, 99)]
        assert _lag_segments(T, np.arange(100.0), 0, 99, min_len=60) == []

    def test_indices_are_absolute(self):
        T = np.arange(200.0)
        T[150] = np.nan
        assert _lag_segments(T, np.arange(200.0), 100, 199, min_len=5) == [
            (100, 149),
            (151, 199),
        ]


class _PF:
    """The PFile attributes fp07_calibrate reads."""

    def __init__(self, channels, config, channels_raw, fs_slow=64.0):
        self.channels = channels
        self.channel_info = {"T1": {"type": "therm"}, "JAC_T": {"type": "jac_t"}}
        self.config = config
        self.channels_raw = channels_raw
        self.fs_slow = fs_slow
        self.fs_fast = fs_slow
        n = len(channels["JAC_T"])
        self.t_slow = np.arange(n) / fs_slow
        self.t_fast = self.t_slow
        self.filepath = Path("synthetic.p")

    def is_fast(self, ch):
        return False


_T1 = {"name": "T1", "type": "therm", "e_b": "2.5", "a": "0", "b": "1", "g": "1",
       "adc_fs": "5", "adc_bits": "16"}


def _synthetic(n=6400, fs=64.0, gap=None):
    """A thermistor whose counts track a structured reference exactly.

    Fine structure (two sinusoids on a ramp) gives the derivative
    cross-correlation something to lock onto; the counts are the exact inverse
    of a Steinhart-Hart law, so the fit has a right answer.
    """
    t = np.arange(n) / fs
    T = 10.0 + 4.0 * t / t[-1] + 0.3 * np.sin(2 * np.pi * t / 7.0) + 0.1 * np.sin(
        2 * np.pi * t / 2.3
    )
    a0, a1, a2 = 3.35e-3, 2.6e-4, 3.0e-6
    inv_T = 1.0 / (T + 273.15)
    # Solve a2 L^2 + a1 L + (a0 - inv_T) = 0 for L, then invert log_r.
    L = (-a1 + np.sqrt(a1**2 - 4 * a2 * (a0 - inv_T))) / (2 * a2)
    Z = (1 - np.exp(L)) / (1 + np.exp(L))
    counts = Z / ((5.0 / 2**16) * 2.0 / (1.0 * 2.5))
    T_ref = T.copy()
    if gap is not None:
        T_ref[gap[0] : gap[1]] = np.nan
    pf = _PF(
        channels={"JAC_T": T_ref, "T1": T.copy(), "P": np.linspace(5, 300, n)},
        config={"channels": [dict(_T1)]},
        channels_raw={"T1": counts},
        fs_slow=fs,
    )
    return pf, [(0, n - 1)]


class TestNaNSplitsTheProfile:
    def _fit(self, gap):
        pf, profiles = _synthetic(gap=gap)
        truth = pf.channels["T1"].copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = fp07_calibrate(pf, profiles, reference="JAC_T", must_be_negative=False)  # type: ignore[arg-type]
        return res, truth

    def test_clean_profile_from_the_top_of_the_file_fits(self):
        """The profile starts at sample 0, as a glider file begun mid-dive does.
        Before the low-pass started at rest this scored |r| = 0.09 and fell
        back to factory coefficients."""
        res, truth = self._fit(gap=None)
        assert res["info"]["T1"]["n_profiles"] == 1
        assert res["factory"] == {}
        assert np.max(np.abs(res["channels"]["T1"] - truth)) < 2e-3

    def test_one_gap_no_longer_silences_the_profile(self):
        """Before: one NaN made the whole profile abstain and T1 kept the
        factory coefficients, with a warning that counted it nowhere."""
        res, truth = self._fit(gap=(3000, 3200))
        assert res["info"]["T1"]["n_profiles"] == 2  # the two sides of the gap
        assert res["info"]["T1"]["n_profiles_unscored"] == 0
        assert np.max(np.abs(res["channels"]["T1"] - truth)) < 2e-3

    def test_lowpass_starts_at_rest(self):
        counts = np.full(500, -5460.0)
        out = _lowpass_filter(counts, "sci_water_temp", 64.0, np.ones(500), [(0, 499)],
                              reference_interval=4.5)
        np.testing.assert_allclose(out, counts)  # no ramp up from zero

    def test_unscorable_pieces_are_counted_and_reported(self):
        """Gaps that leave no stretch of twice the lag window: factory
        coefficients, and the reason says why instead of '0 rejected'."""
        pf, _ = _synthetic()
        pf.channels["JAC_T"][::600] = np.nan  # pieces of ~9 s < 2 x 10 s
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = fp07_calibrate(pf, [(0, 6399)], reference="JAC_T", must_be_negative=False)  # type: ignore[arg-type]
        assert "T1" not in res["coefficients"]
        assert "1 unscored" in res["factory"]["T1"]
        assert any("could not be scored" in str(m.message) for m in w)

    def test_missing_reference_names_every_fp07_channel(self):
        pf, profiles = _synthetic()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = fp07_calibrate(pf, profiles, reference="sci_water_temp")  # type: ignore[arg-type]
        assert res["factory"] == {"T1": "reference channel 'sci_water_temp' not found"}


def _hotel(*names):
    return HotelData(time=np.arange(3.0), channels={n: np.zeros(3) for n in names})


class TestHotelReferenceRefusal:
    def test_hotel_reference_is_refused_by_default(self):
        msg = _hotel_reference_refusal(
            {"fp07": {"reference": "sci_water_temp"}}, {}, _hotel("sci_water_temp", "speed")
        )
        assert msg is not None
        assert "fp07-cal" in msg and "allow_hotel_reference" in msg

    @pytest.mark.parametrize(
        "fp07",
        [
            {"reference": "sci_water_temp", "allow_hotel_reference": True},
            {"reference": "sci_water_temp", "calibrate": False},
            # process_file reads the raw section, where null is falsy and skips
            # the fit; merge_config would have dropped it and refused.
            {"reference": "sci_water_temp", "calibrate": None},
            {"reference": "JAC_T"},  # inside the .p file
        ],
    )
    def test_allowed(self, fp07):
        assert _hotel_reference_refusal({"fp07": fp07}, {}, _hotel("sci_water_temp")) is None

    def test_no_hotel_no_refusal(self):
        assert _hotel_reference_refusal({"fp07": {"reference": "T"}}, {}, None) is None

    def test_names_are_the_merge_names(self):
        """A renamed hotel channel is matched by its merged name, not its source."""
        hotel_cfg = {"channels": {"sci_water_temp": {"name": "T_ctd"}}}
        data = _hotel("sci_water_temp")
        assert _hotel_reference_refusal({"fp07": {"reference": "T_ctd"}}, hotel_cfg, data)
        assert (
            _hotel_reference_refusal({"fp07": {"reference": "sci_water_temp"}}, hotel_cfg, data)
            is None
        )

    def test_run_refuses_before_any_stage(self, tmp_path):
        """Raised before trimming: the .p files named here do not even exist."""
        csv = tmp_path / "hotel.csv"
        csv.write_text("time,sci_water_temp\n0,10\n1,10.1\n2,10.2\n")
        config = {
            "files": {"output_root": str(tmp_path / "out")},
            "hotel": {"enable": True, "file": str(csv), "time_format": "seconds",
                      "max_gap": 30.0},
            "fp07": {"calibrate": True, "reference": "sci_water_temp"},
        }
        with pytest.raises(ValueError, match="allow_hotel_reference"):
            run_pipeline(config, p_files=[tmp_path / "does_not_exist.p"])
        assert not (tmp_path / "out").exists()

    def test_the_hash_exclusion_hits_only_this_key(self):
        """_HASH_EXCLUDE_KEYS matches a key NAME in every section, so a key of
        the same name added elsewhere would silently leave its hash."""
        from odas_tpw.perturb.config import DEFAULTS

        owners = [sec for sec, d in DEFAULTS.items()
                  if isinstance(d, dict) and "allow_hotel_reference" in d]
        assert owners == ["fp07"]

    def test_the_switch_does_not_re_version_products(self):
        """It decides whether a run starts, never what it writes."""
        base = {"reference": "sci_water_temp"}
        h_off = _mgr.canonicalize("fp07", {**base, "allow_hotel_reference": False})
        h_on = _mgr.canonicalize("fp07", {**base, "allow_hotel_reference": True})
        assert h_on == h_off
        assert "allow_hotel_reference" not in h_on


class TestRunSummaryReportsFactoryFallback:
    def test_all_fitted_says_nothing(self):
        out = {"a.p": {"fitted": ["T1", "T2"], "factory": {}}}
        assert _fp07_notices(out) == []

    def test_mixed_deployment_is_reported_per_channel(self):
        out = {
            "a.p": {"fitted": ["T1", "T2"], "factory": {}},
            "b.p": {"fitted": ["T2"], "factory": {"T1": "no profile segment reached the "
                                                      "lag gate (0 rejected, 2 unscored)"}},
            "c.p": {"fitted": ["T2"], "factory": {"T1": "no profile segment reached the "
                                                      "lag gate (1 rejected, 0 unscored)"}},
        }
        head, files = _fp07_notices(out)
        assert head.startswith("FP07 T1: FACTORY coefficients kept in 2 file(s), in-situ fit in 1")
        # one cause, however the per-file counts differ
        assert "reason (2 of 2): no profile segment reached the lag gate." in head
        assert files == "FP07 T1 files: b.p, c.p"

    def test_a_reason_that_leads_with_its_number_is_still_one_cause(self):
        """Eight railed files at eight percentages are one cause, and the most
        common: grouping on raw text named the two lag-gate files instead."""
        out = {f"r{i}.p": {"fitted": [], "factory": {
            "T1": f"{30 + i}.{i}% of fit samples on the bridge rail"}} for i in range(8)}
        out |= {f"g{i}.p": {"fitted": [], "factory": {
            "T1": f"no profile segment reached the lag gate ({i} rejected, 0 unscored)"}}
            for i in range(2)}
        head, _ = _fp07_notices(out)
        assert "reason (8 of 10): N% of fit samples on the bridge rail." in head

    def test_the_file_list_is_whole(self):
        """The summary clips a line; the notice itself must not lose files."""
        names = [f"ARCTERX_Thompson_2025_SN479_{i:04d}.p" for i in range(30)]
        out = {n: {"fitted": [], "factory": {"T1": "reference channel 'X' not found"}}
               for n in names}
        _, files = _fp07_notices(out)
        assert all(n in files for n in names)

    def test_a_raised_calibration_is_not_called_a_channel(self):
        out = {"a.p": {"fitted": [], "factory": {"all": "calibration raised: boom"}}}
        head, files = _fp07_notices(out)
        assert head.startswith("FP07: in-situ calibration RAISED in 1 file(s)")
        assert "FP07 all" not in head and files == "FP07 files: a.p"

    def test_summary_prints_notices_without_failing_the_run(self):
        r = PipelineResult(notices=["FP07 T1: FACTORY coefficients kept in 2 file(s)"])
        assert r.ok
        lines = r.summary().splitlines()
        assert lines[0] == "Pipeline complete: no errors."
        assert lines[1].startswith("  notice: FP07 T1: FACTORY")


class TestNoticeReachesTheRun:
    """The fit's outcome survives the trip from a worker to the run summary."""

    def test_process_file_returns_the_outcome(self, tmp_path):
        from odas_tpw.perturb.pipeline import process_file

        p = Path(__file__).parent / "data" / "SN479_0006.p"
        if not p.exists():
            pytest.skip("SN479_0006.p test data not available")
        cfg = {"fp07": {"calibrate": True, "reference": "NOPE"}}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = process_file(p, cfg, None, {"profiles": tmp_path / "profiles"})
        assert result["fp07"] == {
            "fitted": [],
            "factory": {"T1": "reference channel 'NOPE' not found",
                        "T2": "reference channel 'NOPE' not found"},
        }

    def test_run_pipeline_aggregates_worker_outcomes(self, tmp_path, caplog):
        import logging
        from unittest.mock import patch

        import xarray as xr

        from odas_tpw.perturb import pipeline as pl

        out = tmp_path / "out"
        dirs = {}
        for stage in ("profiles", "diss", "chi"):
            dirs[stage] = out / f"{stage}_00"
            dirs[stage].mkdir(parents=True)
        p_file = tmp_path / "cast.p"
        p_file.touch()

        def _fake_process_file(*args, **kwargs):
            stem = kwargs["output_stem"]
            written = {}
            for stage in ("profiles", "diss", "chi"):
                f = dirs[stage] / f"{stem}_prof001.nc"
                f.touch()
                written[stage] = [str(f)]
            return {"source": str(p_file), **written,
                    "fp07": {"fitted": ["T2"], "factory": {"T1": "reference channel 'X' "
                                                                 "not found"}}}

        config = {
            "files": {"p_file_root": str(tmp_path), "output_root": str(out),
                      "trim": False, "merge": False},
            "profiles": {}, "epsilon": {}, "chi": {"enable": True},
            "ctd": {"enable": False}, "gps": {}, "parallel": {"jobs": 1}, "binning": {},
        }
        with (
            patch.object(pl, "_setup_output_dirs", return_value=dirs),
            patch.object(pl, "_run_combo", return_value=[]),
            patch("odas_tpw.perturb.gps.create_gps", return_value=None),
            patch("odas_tpw.perturb.binning.bin_by_depth", return_value=xr.Dataset()),
            patch("odas_tpw.perturb.binning.bin_by_time", return_value=xr.Dataset()),
            patch("odas_tpw.perturb.binning.bin_diss", return_value=xr.Dataset()),
            patch("odas_tpw.perturb.binning.bin_chi", return_value=xr.Dataset()),
            patch.object(pl, "process_file", side_effect=_fake_process_file),
            caplog.at_level(logging.WARNING, logger=pl.logger.name),
        ):
            result = pl.run_pipeline(config, p_files=[p_file])
        assert result.ok
        assert result.notices[0].startswith("FP07 T1: FACTORY coefficients kept in 1 file(s), "
                                            "in-situ fit in 0")
        assert result.notices[1] == "FP07 T1 files: cast.p"
        # whole, at WARNING, where `--log-level warning` still shows it
        assert any(r.getMessage() == "FP07 T1 files: cast.p" for r in caplog.records)
