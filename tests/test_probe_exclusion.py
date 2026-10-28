# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""``instruments.<key>.exclude``: one probe masked over a pressure/time window (#190).

The masking must be exactly what ``exclude_shear_probes`` does to its probe,
restricted to the window, and it must happen before ``mk_epsilon_mean`` /
``mk_chi_mean`` so ``epsilonMean`` -- and through it chi Method 1 -- loses the
probe there. The worked case is CASPER-West 2017 SN 194, whose sh2 was 15-18x
wrong in a 30-70 dbar box that no QC statistic caught.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from odas_tpw.perturb.config import (
    _INSTRUMENT_KEYS_BY_STAGE,
    _INSTRUMENT_VALID_KEYS,
    _TEMPLATE,
    canonical_instruments_for_hash,
    compute_hash,
    load_config,
    stage_signature,
    validate_config,
)
from odas_tpw.perturb.pipeline import (
    _apply_probe_exclusions,
    _nan_excluded_probes,
    _probe_exclusion_notices,
    _record_inherited_exclusions,
    process_file,
)
from odas_tpw.perturb.probe_exclusion import parse_exclusions
from odas_tpw.processing.chi_combine import mk_chi_mean
from odas_tpw.processing.epsilon_combine import mk_epsilon_mean

ORIGIN = "2017-10-01T00:00:00+00:00"
ORIGIN_S = dt.datetime.fromisoformat(ORIGIN).timestamp()
DAY = 86400.0

# The issue's CASPER-West example, verbatim.
CASPER = [
    {
        "probe": "sh2",
        "P_min": 30,
        "P_max": 70,
        "start": "2017-10-09T00:00:00Z",
        "stop": "2017-10-24T00:00:00Z",
        "reason": "pressure-related fault, window opens ~10-09",
    },
    {
        "probe": "sh2",
        "start": "2017-10-24T00:00:00Z",
        "reason": "probe failed; e_2/e_1 = 0.37-0.82 at all depths",
    },
]

# Probes are removable: sh2 failed, was taken off the instrument, and the entry
# that excluded its bad fortnight stays in the config.
REMOVED_SH2 = {
    "probe": "sh2",
    "start": "2017-10-09",
    "stop": "2017-10-24",
    "reason": "bad fortnight, then removed",
}


def _epoch(iso: str) -> float:
    return dt.datetime.fromisoformat(iso).timestamp()


def _diss_ds(P, t_s, *, e1=1e-8, e2=4e-8, origin=ORIGIN):
    """A two-shear-probe diss dataset shaped like ``_compute_epsilon``'s.

    *t_s* is seconds since *origin* (the product's own encoding).
    """
    P = np.asarray(P, dtype=np.float64)
    n = P.size
    eps = np.vstack([np.full(n, e1), np.full(n, e2)])
    ds = xr.Dataset(
        {
            "epsilon": (("probe", "time"), eps),
            "fom": (("probe", "time"), np.full((2, n), 1.0)),
            "FM": (("probe", "time"), np.full((2, n), 0.8)),
            "P_mean": (("time",), P),
            "speed": (("time",), np.full(n, 0.7)),
            "nu": (("time",), np.full(n, 1.2e-6)),
        },
        coords={"probe": ["sh1", "sh2"], "t": (("time",), np.asarray(t_s, dtype=np.float64))},
        attrs={"diss_length": 2048, "fs_fast": 512.0, "start_time": origin},
    )
    ds["t"].attrs["units"] = f"seconds since {origin}" if origin else "seconds"
    return ds


def _chi_ds(P, t_s, *, c1=1e-8, c2=4e-8):
    """A two-thermistor chi dataset shaped like ``_compute_chi``'s."""
    P = np.asarray(P, dtype=np.float64)
    n = P.size
    ds = xr.Dataset(
        {
            "chi": (("probe", "time"), np.vstack([np.full(n, c1), np.full(n, c2)])),
            "epsilon_T": (("probe", "time"), np.full((2, n), 1e-8)),
            "fom": (("probe", "time"), np.full((2, n), 1.0)),
            "K_max_ratio": (("probe", "time"), np.full((2, n), 2.0)),
            "P_mean": (("time",), P),
            "speed": (("time",), np.full(n, 0.7)),
            "nu": (("time",), np.full(n, 1.2e-6)),
        },
        coords={
            "probe": ["T1_dT1", "T2_dT2"],
            "t": (("time",), np.asarray(t_s, dtype=np.float64)),
        },
        attrs={"diss_length": 2048, "fs_fast": 512.0, "start_time": ORIGIN},
    )
    ds["t"].attrs["units"] = f"seconds since {ORIGIN}"
    return ds


def _ex(*entries, sn="SN194"):
    return parse_exclusions(sn, list(entries))


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_exclude_is_a_valid_instrument_key(self):
        assert "exclude" in _INSTRUMENT_VALID_KEYS

    def test_the_casper_example_validates(self):
        validate_config({"instruments": {"SN194": {"exclude": CASPER}}})
        a, b = parse_exclusions("SN194", CASPER)
        assert (a.probe, a.P_min, a.P_max) == ("sh2", 30.0, 70.0)
        assert a.start == _epoch("2017-10-09T00:00:00+00:00")
        assert a.stop == _epoch("2017-10-24T00:00:00+00:00")
        assert (b.P_min, b.P_max, b.stop) == (None, None, None)
        assert (a.entry, b.entry) == (1, 2)
        assert a.product == b.product == "diss"

    def test_thermistor_names_validate(self):
        out = _ex(
            {"probe": "T1", "reason": "bead"},
            {"probe": "T2_dT2", "P_max": 10, "reason": "bead"},
        )
        assert [e.product for e in out] == ["chi", "chi"]
        # One bead, one spelling: the gradient-channel name is stored as the bead.
        assert [e.probe for e in out] == ["T1", "T2"]

    def test_combines_with_exclude_shear_probes(self):
        validate_config(
            {"instruments": {"SN194": {"exclude_shear_probes": ["sh1"], "exclude": CASPER}}}
        )

    def test_empty_and_null_are_no_entries(self):
        validate_config({"instruments": {"SN194": {"exclude": []}}})
        validate_config({"instruments": {"SN194": {"exclude": None}}})
        assert parse_exclusions("SN194", None) == []

    def test_yaml_timestamps_quoted_or_not_mean_the_same(self, tmp_path):
        """An UNQUOTED YAML timestamp reaches us as a datetime, a bare date as a date."""
        cfg = tmp_path / "perturb.yaml"
        cfg.write_text(
            "instruments:\n"
            "  SN194:\n"
            "    exclude:\n"
            "      - {probe: sh2, start: 2017-10-09T00:00:00Z, stop: 2017-10-24,"
            " reason: unquoted}\n"
            "      - {probe: sh2, start: '2017-10-09T00:00:00Z', stop: '2017-10-24',"
            " reason: unquoted}\n"
        )
        a, b = parse_exclusions("SN194", load_config(cfg)["instruments"]["SN194"]["exclude"])
        assert (a.start, a.stop) == (b.start, b.stop)
        assert a.stop == _epoch("2017-10-24T00:00:00+00:00")

    def test_a_time_without_a_zone_is_utc(self):
        (e,) = _ex({"probe": "sh1", "start": "2017-10-09T00:00:00", "reason": "r"})
        assert e.start == _epoch("2017-10-09T00:00:00+00:00")

    def test_a_zone_offset_is_honored(self):
        (e,) = _ex({"probe": "sh1", "start": "2017-10-09T10:00:00+10:00", "reason": "r"})
        assert e.start == _epoch("2017-10-09T00:00:00+00:00")

    @pytest.mark.parametrize(
        ("entry", "match"),
        [
            ({"probe": "sh2", "reason": "r", "depth_min": 3}, r"unknown key\(s\) \['depth_min'\]"),
            ({"probe": "sh2", "reason": "r", "P_min": 70, "P_max": 30}, "P_min .* less than"),
            ({"probe": "sh2", "reason": "r", "P_min": 30, "P_max": 30}, "P_min .* less than"),
            (
                {"probe": "sh2", "reason": "r", "start": "2017-10-24", "stop": "2017-10-09"},
                "start .* before stop",
            ),
            (
                {"probe": "sh2", "reason": "r", "start": "2017-10-09", "stop": "2017-10-09"},
                "start .* before stop",
            ),
            ({"probe": "sh2", "reason": "r", "start": "10/09/2017"}, "not an ISO-8601 time"),
            ({"probe": "sh2", "reason": "r", "stop": "yesterday"}, "not an ISO-8601 time"),
            ({"probe": "sh2", "reason": "r", "start": 1507507200}, "ISO-8601 time string"),
            ({"probe": "shear2", "reason": "r"}, "probe must name"),
            ({"probe": "t1", "reason": "r"}, "probe must name"),
            ({"probe": "T1_dT", "reason": "r"}, "probe must name"),
            ({"probe": "T1_dT2", "reason": "r"}, "probe must name"),  # which bead?
            ({"probe": 2, "reason": "r"}, "probe must name"),
            ({"reason": "no probe"}, "probe must name"),
            ({"probe": "sh2"}, "reason is required"),
            ({"probe": "sh2", "reason": "  "}, "reason is required"),
            ({"probe": "sh2", "reason": "r", "P_min": "30"}, "P_min must be a number"),
            ({"probe": "sh2", "reason": "r", "P_max": True}, "P_max must be a number"),
            ({"probe": "sh2", "reason": "r", "P_min": float("nan")}, "P_min must be finite"),
            ({"probe": "sh2", "reason": "r", "P_max": float("inf")}, "P_max must be finite"),
        ],
    )
    def test_malformed_entry_names_instrument_and_entry(self, entry, match):
        good = {"probe": "sh1", "reason": "fine"}
        with pytest.raises(ValueError, match=match) as err:
            validate_config({"instruments": {"SN194": {"exclude": [good, entry]}}})
        assert "instruments.SN194.exclude entry 2" in str(err.value)

    def test_entry_must_be_a_mapping(self):
        with pytest.raises(ValueError, match=r"instruments\.SN194\.exclude entry 1: must be a"):
            validate_config({"instruments": {"SN194": {"exclude": ["sh2"]}}})

    def test_exclude_must_be_a_list(self):
        with pytest.raises(ValueError, match=r"instruments\.SN194\.exclude: must be a list"):
            validate_config({"instruments": {"SN194": {"exclude": CASPER[0]}}})

    def test_load_config_rejects_a_malformed_entry(self, tmp_path):
        cfg = tmp_path / "perturb.yaml"
        cfg.write_text(
            "instruments:\n  SN194:\n    exclude:\n"
            "      - {probe: sh2, P_min: 70, P_max: 30, reason: inverted}\n"
        )
        with pytest.raises(ValueError, match=r"SN194\.exclude entry 1"):
            load_config(cfg)

    def test_template_example_validates(self, tmp_path):
        """The commented example in `perturb init`'s template must be a valid config."""
        tail = _TEMPLATE.split("# Example:\n", 1)[1]
        body = "\n".join(line[2:] for line in tail.splitlines() if line.startswith("# "))
        cfg = tmp_path / "example.yaml"
        cfg.write_text(body.replace("<CONFIG_DIR>/fp07pair/", "fp07pair/"))
        from ruamel.yaml import YAML

        instruments = YAML().load(cfg.read_text())["instruments"]
        instruments["SN428"].pop("fp07_response")  # a path to a file that does not exist
        validate_config({"instruments": instruments})
        assert len(parse_exclusions("SN194", instruments["SN194"]["exclude"])) == 2


# ---------------------------------------------------------------------------
# Masking semantics
# ---------------------------------------------------------------------------


class TestPressureBounds:
    P = (10.0, 29.999, 30.0, 50.0, 69.999, 70.0, 90.0)

    def test_P_min_inclusive_P_max_exclusive(self):
        ds = _diss_ds(self.P, np.zeros(len(self.P)))
        _apply_probe_exclusions(
            ds, _ex({"probe": "sh2", "P_min": 30, "P_max": 70, "reason": "r"}), "diss", "f"
        )
        masked = np.isnan(ds["epsilon"].values[1])
        assert masked.tolist() == [False, False, True, True, True, False, False]
        assert np.all(np.isfinite(ds["epsilon"].values[0]))  # sh1 untouched

    def test_one_sided_bounds(self):
        ds = _diss_ds(self.P, np.zeros(len(self.P)))
        _apply_probe_exclusions(ds, _ex({"probe": "sh2", "P_min": 70, "reason": "r"}), "diss", "f")
        assert np.isnan(ds["epsilon"].values[1]).tolist() == [False] * 5 + [True, True]
        ds = _diss_ds(self.P, np.zeros(len(self.P)))
        _apply_probe_exclusions(ds, _ex({"probe": "sh2", "P_max": 30, "reason": "r"}), "diss", "f")
        assert np.isnan(ds["epsilon"].values[1]).tolist() == [True, True] + [False] * 5

    def test_a_nan_P_mean_is_not_placed_inside(self):
        ds = _diss_ds([np.nan, 50.0], [0.0, 0.0])
        _apply_probe_exclusions(
            ds, _ex({"probe": "sh2", "P_min": 30, "P_max": 70, "reason": "r"}), "diss", "f"
        )
        assert np.isnan(ds["epsilon"].values[1]).tolist() == [False, True]


class TestTimeBounds:
    start = _epoch("2017-10-09T00:00:00+00:00")
    stop = _epoch("2017-10-24T00:00:00+00:00")

    def _t(self):
        # Seconds since ORIGIN: just before start, AT start, mid, just before
        # stop, AT stop, after.
        abs_t = [self.start - 1, self.start, self.start + DAY, self.stop - 1, self.stop,
                 self.stop + DAY]
        return np.asarray(abs_t) - ORIGIN_S

    def test_start_inclusive_stop_exclusive(self):
        t = self._t()
        ds = _diss_ds(np.full(t.size, 50.0), t)
        entry = {"probe": "sh2", "start": "2017-10-09T00:00:00Z",
                 "stop": "2017-10-24T00:00:00Z", "reason": "r"}
        _apply_probe_exclusions(ds, _ex(entry), "diss", "f")
        assert np.isnan(ds["epsilon"].values[1]).tolist() == [False, True, True, True, False,
                                                              False]

    def test_tested_against_absolute_time_not_seconds_since_start(self):
        """Two files with different start times but the same absolute window times
        must be masked identically: the origin is part of the time."""
        t_abs = self.start + 3600.0
        for origin in (ORIGIN, "2017-10-08T23:00:00+00:00"):
            o = dt.datetime.fromisoformat(origin).timestamp()
            ds = _diss_ds([50.0], [t_abs - o], origin=origin)
            _apply_probe_exclusions(
                ds, _ex({"probe": "sh2", "start": "2017-10-09T00:00:00Z", "reason": "r"}),
                "diss", "f",
            )
            assert np.isnan(ds["epsilon"].values[1, 0])

    def test_a_decoded_datetime_t_works_too(self):
        """A product read back from NetCDF carries datetime64 ``t``."""
        t = self._t()
        ds = _diss_ds(np.full(t.size, 50.0), t)
        ds = ds.assign_coords(
            t=("time", np.array([np.datetime64(int((ORIGIN_S + s) * 1e6), "us") for s in t]))
        )
        _apply_probe_exclusions(
            ds, _ex({"probe": "sh2", "start": "2017-10-09T00:00:00Z", "reason": "r"}), "diss", "f"
        )
        assert np.isnan(ds["epsilon"].values[1]).tolist() == [False] + [True] * 5

    def test_a_time_bound_without_an_absolute_time_raises(self):
        ds = _diss_ds([50.0], [10.0], origin="")
        with pytest.raises(ValueError, match="no absolute reference"):
            _apply_probe_exclusions(
                ds, _ex({"probe": "sh2", "start": "2017-10-09", "reason": "r"}), "diss", "f"
            )
        assert np.all(np.isfinite(ds["epsilon"].values))


class TestCombinedBoundsAndScope:
    def test_every_bound_must_hold(self):
        """The CASPER box: inside the depth range but before the window opens is kept."""
        start = _epoch("2017-10-09T00:00:00+00:00")
        P = [50.0, 50.0, 80.0, 50.0]
        t_abs = [start - DAY, start + DAY, start + DAY, _epoch("2017-10-25T00:00:00+00:00")]
        ds = _diss_ds(P, np.asarray(t_abs) - ORIGIN_S)
        _apply_probe_exclusions(ds, parse_exclusions("SN194", CASPER), "diss", "f")
        # before window | in box | below box | after 10-24: entry 2, every depth
        assert np.isnan(ds["epsilon"].values[1]).tolist() == [False, True, False, True]
        assert ds.attrs["probe_exclusion_1_n_windows"] == 1
        assert ds.attrs["probe_exclusion_2_n_windows"] == 1

    def test_no_bounds_is_the_whole_deployment(self):
        ds = _diss_ds([5.0, 50.0, 500.0], [0.0, 1.0, 2.0])
        _apply_probe_exclusions(ds, _ex({"probe": "sh1", "reason": "r"}), "diss", "f")
        assert np.all(np.isnan(ds["epsilon"].values[0]))
        assert np.all(np.isfinite(ds["epsilon"].values[1]))

    def test_metadata_is_kept(self):
        """As every per-probe cut: fom/FM are what showed the fault."""
        ds = _diss_ds([50.0, 50.0], [0.0, 1.0])
        _apply_probe_exclusions(ds, _ex({"probe": "sh2", "reason": "r"}), "diss", "f")
        assert np.all(np.isfinite(ds["fom"].values)) and np.all(np.isfinite(ds["FM"].values))

    def test_masks_the_e_N_companions(self):
        ds = _diss_ds([20.0, 50.0], [0.0, 1.0])
        ds["e_1"] = ("time", ds["epsilon"].values[0].copy())
        ds["e_2"] = ("time", ds["epsilon"].values[1].copy())
        _apply_probe_exclusions(ds, _ex({"probe": "sh2", "P_min": 30, "reason": "r"}), "diss", "f")
        assert np.isnan(ds["e_2"].values).tolist() == [False, True]
        assert np.all(np.isfinite(ds["e_1"].values))

    def test_each_product_takes_only_its_own_probes(self):
        exc = _ex({"probe": "sh2", "reason": "shear"}, {"probe": "T1", "reason": "bead"})
        diss, chi = _diss_ds([50.0], [0.0]), _chi_ds([50.0], [0.0])
        _apply_probe_exclusions(diss, exc, "diss", "f")
        _apply_probe_exclusions(chi, exc, "chi", "f")
        assert np.isnan(diss["epsilon"].values[1, 0]) and np.isfinite(diss["epsilon"].values[0, 0])
        assert np.isnan(chi["chi"].values[0, 0]) and np.isfinite(chi["chi"].values[1, 0])
        assert "probe_exclusion_1_probe" in diss.attrs
        assert "probe_exclusion_2_probe" not in diss.attrs
        assert "probe_exclusion_2_probe" in chi.attrs
        assert "probe_exclusion_1_probe" not in chi.attrs

    def test_thermistor_masks_chi_chi_N_and_epsilon_T(self):
        """Every chi per-probe cut masks epsilon_T too, or chiLnSigma would keep the probe."""
        chi = _chi_ds([20.0, 50.0], [0.0, 1.0])
        chi["chi_1"] = ("time", chi["chi"].values[0].copy())
        _apply_probe_exclusions(chi, _ex({"probe": "T1", "P_min": 30, "reason": "r"}), "chi", "f")
        for v in (chi["chi"].values[0], chi["epsilon_T"].values[0], chi["chi_1"].values):
            assert np.isnan(v).tolist() == [False, True]
        assert np.all(np.isfinite(chi["epsilon_T"].values[1]))

    def test_gradient_channel_spelling_matches(self):
        chi = _chi_ds([50.0], [0.0])
        _apply_probe_exclusions(chi, _ex({"probe": "T2_dT2", "reason": "r"}), "chi", "f")
        assert np.isnan(chi["chi"].values[1, 0])


class TestUnknownProbe:
    def test_an_absent_shear_probe_raises_naming_it(self):
        ds = _diss_ds([50.0], [0.0])
        exc = _ex({"probe": "sh2", "reason": "ok"}, {"probe": "sh3", "reason": "typo"})
        with pytest.raises(ValueError, match=r"SN194\.exclude entry 2 \(probe 'sh3'\).*sh1"):
            _apply_probe_exclusions(ds, exc, "diss", "f")
        # Refused before masking anything -- entry 1 was valid, and is not applied.
        assert np.all(np.isfinite(ds["epsilon"].values))
        assert "probe_exclusion_1_probe" not in ds.attrs

    def test_an_absent_thermistor_raises(self):
        with pytest.raises(ValueError, match=r"probe 'T3'.*T1_dT1"):
            _apply_probe_exclusions(
                _chi_ds([50.0], [0.0]), _ex({"probe": "T3", "reason": "r"}), "chi", "f"
            )

    def test_a_removed_probe_does_not_fail_profiles_outside_its_window(self):
        later = _epoch("2017-10-30T00:00:00+00:00") - ORIGIN_S
        ds = _diss_ds([50.0, 90.0], [later, later + 60]).isel(probe=[0])  # sh1 only
        counts = _apply_probe_exclusions(ds, _ex(REMOVED_SH2), "diss", "f")
        assert counts["instruments.SN194.exclude entry 1 (probe 'sh2')"]["n_windows"] == 0
        assert ds.attrs["probe_exclusion_1_n_windows"] == 0
        assert np.all(np.isfinite(ds["epsilon"].values))

    def test_a_missing_probe_inside_its_window_still_raises(self):
        """A typo, or a probe really missing where the entry reaches, is an error."""
        inside = _epoch("2017-10-12T00:00:00+00:00") - ORIGIN_S
        ds = _diss_ds([50.0], [inside]).isel(probe=[0])
        with pytest.raises(ValueError, match=r"probe 'sh2'.*\['sh1'\]"):
            _apply_probe_exclusions(ds, _ex(REMOVED_SH2), "diss", "f")

    def test_a_pressure_bound_without_P_mean_raises(self):
        ds = _diss_ds([50.0], [0.0]).drop_vars("P_mean")
        with pytest.raises(ValueError, match="no per-window P_mean"):
            _apply_probe_exclusions(
                ds, _ex({"probe": "sh2", "P_min": 1, "reason": "r"}), "diss", "f"
            )

    def test_exclude_shear_probes_still_only_warns(self, caplog):
        """Backwards compatibility: the old key's typo behavior is unchanged."""
        ds = _diss_ds([50.0], [0.0])
        with caplog.at_level("WARNING", logger="odas_tpw.perturb.pipeline"):
            _nan_excluded_probes(ds, ["sh3"], "f")
        assert "not present" in caplog.text
        assert np.all(np.isfinite(ds["epsilon"].values))


class TestProvenance:
    def test_attrs_record_entry_bounds_reason_and_count(self):
        start = _epoch("2017-10-09T00:00:00+00:00")
        ds = _diss_ds([50.0, 50.0, 90.0], np.asarray([start, start + 1, start]) - ORIGIN_S)
        _apply_probe_exclusions(ds, parse_exclusions("SN194", CASPER), "diss", "f")
        a = ds.attrs
        assert a["probe_exclusion_1_probe"] == "sh2"
        assert a["probe_exclusion_1_P_min"] == 30.0 and a["probe_exclusion_1_P_max"] == 70.0
        assert a["probe_exclusion_1_start"] == "2017-10-09T00:00:00Z"
        assert a["probe_exclusion_1_stop"] == "2017-10-24T00:00:00Z"
        assert a["probe_exclusion_1_reason"] == CASPER[0]["reason"]
        assert a["probe_exclusion_1_masked_in"] == "diss"
        assert a["probe_exclusion_1_n_windows"] == 2
        assert "probe_exclusion_1_comment" not in a  # masked here, not inherited
        # An omitted bound is absent (unbounded), and a zero count is recorded.
        assert "probe_exclusion_2_P_min" not in a and "probe_exclusion_2_stop" not in a
        assert a["probe_exclusion_2_n_windows"] == 0

    def test_thermistor_entry_records_its_bead_name(self):
        chi = _chi_ds([50.0], [0.0])
        _apply_probe_exclusions(chi, _ex({"probe": "T2_dT2", "reason": "r"}), "chi", "f")
        assert chi.attrs["probe_exclusion_1_probe"] == "T2"
        assert chi.attrs["probe_exclusion_1_masked_in"] == "chi"

    def test_chi_records_the_shear_entries_it_inherits(self):
        """A shear mask changes chi through epsilonMean; the chi file must say so."""
        exc = _ex(
            {"probe": "T1", "P_min": 1000, "reason": "bead"},
            {"probe": "sh2", "reason": "shear"},
        )
        chi = _chi_ds([50.0], [0.0])
        _apply_probe_exclusions(chi, exc, "chi", "f")
        diss_counts = {"instruments.SN194.exclude entry 2 (probe 'sh2')": {"n_windows": 7}}
        _record_inherited_exclusions(chi, exc, diss_counts)
        a = chi.attrs
        assert a["probe_exclusion_1_masked_in"] == "chi" and a["probe_exclusion_1_n_windows"] == 0
        assert a["probe_exclusion_2_probe"] == "sh2"
        assert a["probe_exclusion_2_masked_in"] == "diss"
        assert a["probe_exclusion_2_n_windows"] == 7
        assert "epsilonMean" in a["probe_exclusion_2_comment"]
        assert "probe_exclusion_1_comment" not in a
        # Its own probes are untouched by the shear entry.
        assert np.all(np.isfinite(chi["chi"].values))

    def test_attrs_survive_a_netcdf_round_trip(self, tmp_path):
        ds = _diss_ds([50.0], [0.0])
        _apply_probe_exclusions(ds, parse_exclusions("SN194", CASPER), "diss", "f")
        ds.to_netcdf(tmp_path / "d.nc")
        with xr.open_dataset(tmp_path / "d.nc") as back:
            assert back.attrs["probe_exclusion_1_start"] == "2017-10-09T00:00:00Z"
            assert int(back.attrs["probe_exclusion_2_n_windows"]) == 0

    def test_no_exclusion_adds_no_attrs(self):
        ds = _diss_ds([50.0], [0.0])
        before = dict(ds.attrs)
        assert _apply_probe_exclusions(ds, [], "diss", "f") == {}
        _nan_excluded_probes(ds, ["sh2"], "f")
        assert ds.attrs == before


# ---------------------------------------------------------------------------
# Propagation into the combined products
# ---------------------------------------------------------------------------


class TestCombinedMeans:
    def test_shear_exclusion_reaches_epsilonMean_inside_only(self):
        ds = _diss_ds([20.0, 50.0], [0.0, 1.0], e1=1e-8, e2=4e-8)
        _apply_probe_exclusions(ds, _ex({"probe": "sh2", "P_min": 30, "reason": "r"}), "diss", "f")
        out = mk_epsilon_mean(ds)
        np.testing.assert_allclose(out["epsilonMean"].values, [2e-8, 1e-8], rtol=1e-12)

    def test_thermistor_exclusion_reaches_chiMean_inside_only(self):
        chi = _chi_ds([20.0, 50.0], [0.0, 1.0], c1=1e-8, c2=4e-8)
        _apply_probe_exclusions(chi, _ex({"probe": "T2", "P_min": 30, "reason": "r"}), "chi", "f")
        out = mk_chi_mean(chi, fom_limit=2.0, k_max_ratio_min=0.5)
        np.testing.assert_allclose(out["chiMean"].values, [2e-8, 1e-8], rtol=1e-12)

    def test_unbounded_exclude_equals_exclude_shear_probes(self):
        """One semantics: the old key is the always-everywhere form of the new one."""
        old, new = _diss_ds([20.0, 50.0], [0.0, 1.0]), _diss_ds([20.0, 50.0], [0.0, 1.0])
        _nan_excluded_probes(old, ["sh2"], "f")
        _apply_probe_exclusions(new, _ex({"probe": "sh2", "reason": "r"}), "diss", "f")
        old, new = mk_epsilon_mean(old), mk_epsilon_mean(new)
        for v in ("epsilon", "e_1", "e_2", "epsilonMean", "epsilonLnSigma"):
            np.testing.assert_array_equal(old[v].values, new[v].values)


# ---------------------------------------------------------------------------
# process_file wiring: diss -> epsilonMean -> chi Method 1, and chi
# ---------------------------------------------------------------------------


def _run_process_file(tmp_path, instruments, *, diss_probes=(0, 1), shift_days=0.0):
    """Run process_file over one mocked profile; return (result, diss ds, chi ds, seen).

    ``seen`` holds what chi Method 1 was handed: the epsilonMean of the diss file
    process_file wrote, and the epsi_final that ``_epsilon_ds_to_l4data`` (the
    Method-1 entry) derives from it. *diss_probes* picks the shear probes the
    instrument carries (``(0,)`` = sh1 only); *shift_days* moves the profile.
    """
    from odas_tpw.rsi.chi_io import _epsilon_ds_to_l4data

    start = _epoch("2017-10-09T00:00:00+00:00")
    P = [20.0, 50.0, 50.0, 90.0]
    t_s = np.asarray([start + 60, start + 60, start - DAY, start + 60]) - ORIGIN_S
    t_s = t_s + shift_days * DAY
    prof_dir, diss_dir, chi_dir = (tmp_path / d for d in ("profiles", "diss", "chi"))
    for d in (prof_dir, diss_dir, chi_dir):
        d.mkdir(parents=True, exist_ok=True)
    prof_nc = prof_dir / "SN194__cast_prof001.nc"
    prof_nc.touch()
    seen: dict = {}

    def fake_chi(prof_path, **kwargs):
        eps = kwargs["epsilon_ds"]
        seen["epsilonMean"] = eps["epsilonMean"].values.copy()
        seen["epsi_final"] = _epsilon_ds_to_l4data(eps).epsi_final.copy()
        return [_chi_ds(P, t_s)]

    mock_pf = MagicMock()
    mock_pf.channels = {"P": np.linspace(0, 100, 100), "T1": np.zeros(100)}
    mock_pf.t_slow = np.linspace(0, 100, 100)
    mock_pf.fs_slow = 64.0
    mock_pf.fs_fast = 512.0
    with (
        patch("odas_tpw.rsi.p_file.PFile", return_value=mock_pf),
        patch("odas_tpw.rsi.profile._smooth_fall_rate", return_value=np.zeros(100)),
        patch("odas_tpw.rsi.profile.get_profiles", return_value=[{"start": 0, "end": 50}]),
        patch("odas_tpw.rsi.profile.extract_profiles", return_value=([prof_nc], [{}])),
        patch(
            "odas_tpw.rsi.dissipation._compute_epsilon",
            return_value=[_diss_ds(P, t_s).isel(probe=list(diss_probes))],
        ),
        patch("odas_tpw.rsi.chi_io._load_therm_channels", return_value={}),
        patch("odas_tpw.rsi.chi_io._compute_chi", side_effect=fake_chi),
    ):
        config = {
            "files": {"output_root": str(tmp_path)},
            "profiles": {"P_min": 0.5},
            "fp07": {"calibrate": False},
            "ct": {"align": False},
            "ctd": {"enable": False},
            "stratification": {"enable": False},
            "chi": {"enable": True, "mixing": False},
            "instruments": instruments,
        }
        result = process_file(
            tmp_path / "cast.p",
            config,
            None,
            {"profiles": prof_dir, "diss": diss_dir, "chi": chi_dir},
            instrument_key="SN194",
        )
    diss = xr.load_dataset(diss_dir / prof_nc.name) if result["diss"] else None
    chi = xr.load_dataset(chi_dir / prof_nc.name) if result["chi"] else None
    return result, diss, chi, seen


SH2_BOX = [{"probe": "sh2", "P_min": 30, "P_max": 70, "start": "2017-10-09T00:00:00Z",
            "reason": "box"}]
T2_BOX = [{"probe": "T2", "P_min": 30, "P_max": 70, "start": "2017-10-09T00:00:00Z",
           "reason": "bead"}]


class TestProcessFile:
    # Windows: [20 dbar in window-time, 50 in, 50 BEFORE the window opens, 90 in].
    INSIDE = (False, True, False, False)

    def test_shear_exclusion_reaches_epsilonMean_and_chi_method_1(self, tmp_path):
        result, diss, chi, seen = _run_process_file(tmp_path, {"SN194": {"exclude": SH2_BOX}})
        assert not result.get("errors")
        assert np.isnan(diss["e_2"].values).tolist() == list(self.INSIDE)
        want = np.where(self.INSIDE, 1e-8, 2e-8)  # e_1 alone inside; geo-mean of 1e-8, 4e-8 out
        np.testing.assert_allclose(diss["epsilonMean"].values, want, rtol=1e-12)
        # chi Method 1 seeds kB from exactly that epsilonMean.
        np.testing.assert_allclose(seen["epsilonMean"], want, rtol=1e-12)
        np.testing.assert_allclose(seen["epsi_final"], want, rtol=1e-12)
        assert diss.attrs["probe_exclusion_1_n_windows"] == 1
        # A shear entry leaves the chi product's own thermistors alone, but the
        # chi file says it inherited the mask through epsilonMean.
        assert np.all(np.isfinite(chi["chi"].values))
        assert chi.attrs["probe_exclusion_1_probe"] == "sh2"
        assert chi.attrs["probe_exclusion_1_masked_in"] == "diss"
        assert chi.attrs["probe_exclusion_1_n_windows"] == 1
        assert "epsilonMean" in chi.attrs["probe_exclusion_1_comment"]
        assert result["probe_exclusions"] == {
            "instruments.SN194.exclude entry 1 (probe 'sh2')": {
                "instrument": "SN194", "entry": 1, "product": "diss",
                "n_windows": 1, "n_total": 4,
            }
        }

    def test_thermistor_exclusion_affects_chi_only(self, tmp_path):
        _, diss0, chi0, _ = _run_process_file(tmp_path / "a", {})
        result, diss, chi, _ = _run_process_file(tmp_path / "b", {"SN194": {"exclude": T2_BOX}})
        assert not result.get("errors")
        # epsilon untouched: same values as the run with no exclusion.
        for v in ("epsilon", "e_1", "e_2", "epsilonMean"):
            np.testing.assert_array_equal(diss[v].values, diss0[v].values)
        assert not any(k.startswith("probe_exclusion") for k in diss.attrs)
        # chi: T2 masked in the box, chiMean = chi_1 there.
        assert np.isnan(chi["chi_2"].values).tolist() == list(self.INSIDE)
        assert np.all(np.isfinite(chi["chi_1"].values))
        np.testing.assert_allclose(
            chi["chiMean"].values, np.where(self.INSIDE, 1e-8, 2e-8), rtol=1e-12
        )
        assert chi.attrs["probe_exclusion_1_probe"] == "T2"
        assert chi.attrs["probe_exclusion_1_masked_in"] == "chi"
        assert chi.attrs["probe_exclusion_1_n_windows"] == 1
        assert "probe_exclusion_1_comment" not in chi.attrs
        assert not any(k.startswith("probe_exclusion") for k in chi0.attrs)

    def test_an_absent_probe_is_a_profile_error_naming_it(self, tmp_path):
        result, diss, _, _ = _run_process_file(
            tmp_path, {"SN194": {"exclude": [{"probe": "sh4", "reason": "typo"}]}}
        )
        assert diss is None
        assert any("probe 'sh4'" in e and "SN194" in e for e in result["errors"])

    def test_a_probe_removed_after_its_window_does_not_fail_later_profiles(self, tmp_path):
        """sh2 failed, was taken off, and the entry for its bad fortnight stays.

        A profile after the window, from the instrument without sh2, must still
        get its diss and chi; the entry records that it masked nothing there.
        """
        result, diss, chi, _ = _run_process_file(
            tmp_path, {"SN194": {"exclude": [REMOVED_SH2]}}, diss_probes=(0,),
            shift_days=20,
        )
        assert not result.get("errors")
        assert diss is not None and chi is not None
        assert list(diss["probe"].values) == ["sh1"]
        assert diss.attrs["probe_exclusion_1_n_windows"] == 0
        assert chi.attrs["probe_exclusion_1_n_windows"] == 0
        label = "instruments.SN194.exclude entry 1 (probe 'sh2')"
        assert result["probe_exclusions"][label]["n_windows"] == 0

    def test_the_same_entry_still_fails_a_profile_inside_its_window(self, tmp_path):
        result, diss, _, _ = _run_process_file(
            tmp_path, {"SN194": {"exclude": [REMOVED_SH2]}}, diss_probes=(0,)
        )
        assert diss is None
        assert any("probe 'sh2'" in e for e in result["errors"])

    def test_exclude_shear_probes_alone_is_unchanged(self, tmp_path):
        """Backwards compatibility: no attrs, no run-summary record, same numbers."""
        result, diss, _, _ = _run_process_file(
            tmp_path, {"SN194": {"exclude_shear_probes": ["sh2"]}}
        )
        assert "probe_exclusions" not in result
        assert np.all(np.isnan(diss["e_2"].values))
        np.testing.assert_allclose(diss["epsilonMean"].values, 1e-8, rtol=1e-12)
        assert not any(k.startswith("probe_exclusion") for k in diss.attrs)

    def test_no_instruments_block_leaves_no_trace(self, tmp_path):
        result, diss, chi, _ = _run_process_file(tmp_path, {})
        assert "probe_exclusions" not in result
        assert not any(k.startswith("probe_exclusion") for k in {**diss.attrs, **chi.attrs})


# ---------------------------------------------------------------------------
# Stage fingerprint
# ---------------------------------------------------------------------------


def _sig(instruments, stage):
    """The stage signature exactly as the pipeline computes it."""
    section, params, upstream = stage_signature(stage, {"instruments": instruments})
    return compute_hash(section, params, upstream=upstream)


DISS_STAGES = ("diss", "diss_binned", "diss_combo")
CHI_STAGES = ("chi", "chi_binned", "chi_combo")
T1 = {"probe": "T1", "P_min": 100, "reason": "bead"}
SH2 = {"probe": "sh2", "P_min": 30, "reason": "fault"}


class TestFingerprint:
    @pytest.mark.parametrize("stage", DISS_STAGES + CHI_STAGES)
    def test_a_shear_entry_reversions_diss_and_chi(self, stage):
        assert _sig({}, stage) != _sig({"SN194": {"exclude": [SH2]}}, stage)

    @pytest.mark.parametrize("stage", DISS_STAGES)
    def test_a_thermistor_entry_leaves_diss_alone(self, stage):
        """It cannot change a diss byte, so the epsilon output is reused."""
        assert _sig({}, stage) == _sig({"SN194": {"exclude": [T1]}}, stage)
        assert _sig({"SN194": {"exclude": [SH2]}}, stage) == _sig(
            {"SN194": {"exclude": [SH2, T1]}}, stage
        )

    @pytest.mark.parametrize("stage", CHI_STAGES)
    def test_a_thermistor_entry_reversions_chi(self, stage):
        assert _sig({}, stage) != _sig({"SN194": {"exclude": [T1]}}, stage)

    def test_routed_to_both_stages(self):
        assert "exclude" in _INSTRUMENT_KEYS_BY_STAGE["diss"]
        assert "exclude" in _INSTRUMENT_KEYS_BY_STAGE["chi"]

    def test_entry_numbers_survive_the_projection(self):
        """diss sees only sh2, but as entry 2: its attributes are probe_exclusion_2_*."""
        assert _sig({"SN194": {"exclude": [T1, SH2]}}, "diss") != _sig(
            {"SN194": {"exclude": [SH2]}}, "diss"
        )
        other_t = {**T1, "probe": "T2", "reason": "other bead"}
        assert _sig({"SN194": {"exclude": [T1, SH2]}}, "diss") == _sig(
            {"SN194": {"exclude": [other_t, SH2]}}, "diss"
        )
        proj = canonical_instruments_for_hash(
            {"SN194": {"exclude": [T1, SH2]}}, exclude_products=frozenset({"diss"})
        )
        assert proj == {
            "SN194": {"exclude": [{"probe": "sh2", "reason": "fault", "P_min": 30.0, "entry": 2}]}
        }

    @pytest.mark.parametrize(
        "change",
        [{"P_min": 31}, {"P_max": 71}, {"start": "2017-10-10T00:00:00Z"},
         {"stop": "2017-10-23T00:00:00Z"}, {"probe": "sh1"}, {"reason": "reworded"}],
    )
    def test_changing_a_shear_entry_reversions(self, change):
        """reason included: it is written into the products."""
        changed = [{**CASPER[0], **change}, CASPER[1]]
        for stage in ("diss", "chi"):
            assert _sig({"SN194": {"exclude": CASPER}}, stage) != _sig(
                {"SN194": {"exclude": changed}}, stage
            )

    def test_changing_a_thermistor_reason_reversions_chi(self):
        assert _sig({"SN194": {"exclude": [T1]}}, "chi") != _sig(
            {"SN194": {"exclude": [{**T1, "reason": "reworded"}]}}, "chi"
        )

    def test_entry_order_reversions(self):
        """Each entry's position names its attributes in the product."""
        assert _sig({"SN194": {"exclude": CASPER}}, "diss") != _sig(
            {"SN194": {"exclude": CASPER[::-1]}}, "diss"
        )

    def test_equal_meanings_hash_equal(self):
        """Including an unquoted-YAML datetime, which plain json.dumps would refuse."""
        same = [
            {**CASPER[0], "P_min": 30.0, "start": dt.datetime(2017, 10, 9, tzinfo=dt.UTC),
             "stop": dt.date(2017, 10, 24)},
            {**CASPER[1], "start": "2017-10-24T00:00:00+00:00"},
        ]
        for stage in ("diss", "chi"):
            assert _sig({"SN194": {"exclude": CASPER}}, stage) == _sig(
                {"SN194": {"exclude": same}}, stage
            )

    def test_bead_and_gradient_channel_spellings_hash_equal(self):
        for stage in CHI_STAGES:
            assert _sig({"SN194": {"exclude": [T1]}}, stage) == _sig(
                {"SN194": {"exclude": [{**T1, "probe": "T1_dT1"}]}}, stage
            )

    def test_empty_exclude_hashes_as_absent(self):
        for stage in ("diss", "chi"):
            assert _sig({}, stage) == _sig({"SN194": {"exclude": []}}, stage)
        assert _sig({"SN194": {"exclude_shear_probes": ["sh2"]}}, "diss") == _sig(
            {"SN194": {"exclude_shear_probes": ["sh2"], "exclude": None}}, "diss"
        )


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------


class TestNotices:
    LABEL = "instruments.SN194.exclude entry 1 (probe 'sh2')"

    def _counts(self, n, total, entry=1, label=None):
        return {label or self.LABEL: {"instrument": "SN194", "entry": entry, "product": "diss",
                                      "n_windows": n, "n_total": total}}

    def test_reports_windows_masked_across_files(self):
        lines = _probe_exclusion_notices(
            {"SN194": {"exclude": CASPER[:1]}},
            {"SN194"},
            {"a.p": self._counts(3, 10), "b.p": self._counts(4, 20)},
        )
        assert lines == [f"Probe exclusion: {self.LABEL} masked 7 of 30 diss windows in 2 "
                         "file(s) this run."]

    def test_an_entry_that_masked_nothing_says_so(self):
        (line,) = _probe_exclusion_notices(
            {"SN194": {"exclude": CASPER[:1]}}, {"SN194"}, {"a.p": self._counts(0, 10)}
        )
        assert "masked NO windows" in line and "dbar" in line

    def test_an_unmatched_instrument_key_is_named(self):
        """The known trap: an instruments key matching no directory is otherwise silent."""
        (line,) = _probe_exclusion_notices(
            {"sn194": {"exclude": CASPER}, "SN479": {"fp07_tau_scale": {"T1": 1.1}}},
            {"SN194"},
            {},
        )
        assert "instruments.sn194 has 2 exclude entries" in line and "SN194" in line

    def test_an_unquoted_numeric_key_is_named(self):
        """`465:` loads as an int, which process_file's string lookup never matches."""
        (line,) = _probe_exclusion_notices({465: {"exclude": CASPER[:1]}}, {"465"}, {})
        assert "loaded as the number 465; quote it" in line

    def test_entries_sort_by_number_not_text(self):
        l10 = "instruments.SN194.exclude entry 10 (probe 'sh2')"
        counts = {**self._counts(1, 5, entry=10, label=l10), **self._counts(1, 5)}
        lines = _probe_exclusion_notices({}, {"SN194"}, {"a.p": counts})
        assert [ln.split(" (probe")[0][-2:] for ln in lines] == [" 1", "10"]

    def test_a_thermistor_entry_when_chi_did_not_run_says_so(self):
        exc = {"SN194": {"exclude": [CASPER[0], {"probe": "T3", "reason": "typo?"}]}}
        lines = _probe_exclusion_notices(
            exc, {"SN194"}, {"a.p": self._counts(1, 10)}, chi_ran=False
        )
        assert lines[0].startswith(
            "Probe exclusion: instruments.SN194.exclude entry 2 (probe 'T3') was NOT applied"
        )
        assert "chi did not run" in lines[0]
        assert len(lines) == 2  # plus entry 1's masked count; no NOT-applied line for it
        assert _probe_exclusion_notices(exc, {"SN194"}, {}, chi_ran=True) == []

    def test_silent_without_exclude(self):
        instruments = {"SN194": {"exclude_shear_probes": ["sh2"]}}
        assert _probe_exclusion_notices(instruments, set(), {}) == []


def test_module_stays_numpy_free_at_import():
    """perturb.config imports it, and building the CLI parser must not pull numpy."""
    import subprocess
    import sys

    code = (
        "import sys; import odas_tpw.perturb.probe_exclusion; "
        "assert 'numpy' not in sys.modules, 'probe_exclusion pulled in numpy'"
    )
    env_path = str(Path(__file__).resolve().parents[1] / "src")
    subprocess.run(
        [sys.executable, "-c", code], check=True, env={"PYTHONPATH": env_path, "PATH": ""}
    )
