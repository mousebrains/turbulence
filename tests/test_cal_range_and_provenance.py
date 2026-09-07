# Sep-2026, Claude and Pat Welch, pat@mousebrains.com
"""Plausible-range checks and calibration provenance (Jesse Cusack's pyturb concepts).

Two ideas taken from upstream pyturb and adapted:

* a **range check** on well-formed calibration coefficients, which catches the
  un-filled ``sens = 1.0`` placeholder that ``_parse_finite_float`` (issue #180
  F06) cannot — that guard only fails closed on values that are not numbers;
* **per-variable provenance**, so a converted file records the coefficients
  that produced it.

The range bounds are pinned against our own shear inventory rather than taken
on faith; see ``channels._SHEAR_SENS_MIN`` for the corpus statistics.
"""

import warnings

import numpy as np
import pytest

from odas_tpw.rsi.channels import (
    _SHEAR_DIFF_GAIN_MAX,
    _SHEAR_DIFF_GAIN_MIN,
    _SHEAR_SENS_MAX,
    _SHEAR_SENS_MIN,
    CalRecorder,
    convert_poly,
    convert_shear,
)

DATA = np.array([-1000.0, 0.0, 1000.0])


def _warnings_from(params: dict) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        convert_shear(DATA, dict(params))
    return [str(w.message) for w in caught]


class TestShearRangeCheck:
    """The gap #180 F06's fix leaves open: a coefficient that parses but lies."""

    def test_placeholder_sens_is_flagged(self):
        """``sens = 1.0`` converts cleanly and scales epsilon by ~200x."""
        msgs = _warnings_from({"name": "sh1", "diff_gain": "0.98", "sens": "1.0"})
        assert any("'sens'=1" in m and "plausible range" in m for m in msgs)

    def test_flagged_value_is_still_used(self):
        """A warning, not a substitution — we never invent a coefficient.

        The whole point of the #180 F06 work is that a fabricated default is
        worse than a wrong-but-honest number, because it is invisible.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            odd, _ = convert_shear(DATA, {"name": "sh1", "diff_gain": "0.98", "sens": "1.0"})
            ref, _ = convert_shear(DATA, {"name": "sh1", "diff_gain": "0.98", "sens": "0.1"})
        # Output scales exactly as sens^-1: the flagged value was honoured.
        np.testing.assert_allclose(odd, ref * (0.1 / 1.0), rtol=1e-12)

    def test_decimal_slip_in_sens_is_flagged(self):
        """0.0678 -> 0.00678 keeps a plausible shape but is 10x out."""
        msgs = _warnings_from({"name": "sh2", "diff_gain": "0.98", "sens": "0.00678"})
        assert any("'sens'" in m and "plausible range" in m for m in msgs)

    @pytest.mark.parametrize("sens", ["0.041", "0.123"])
    def test_corpus_extremes_are_silent(self, sens):
        """The min and max sens over 15998 real channel-configs must not warn.

        A check that fires on real data is a check that gets muted.  These are
        the observed extremes of the shear inventory across ARCTERX, SUNRISE,
        RIOT, CASPER, Taiwan, ASTRAL, Keck and goflow.
        """
        msgs = _warnings_from({"name": "sh1", "diff_gain": "0.98", "sens": sens})
        assert not [m for m in msgs if "plausible range" in m]

    def test_low_band_diff_gain_is_silent(self):
        """A sub-0.1 differentiator gain is real hardware — do not warn.

        2236 of 15998 rows sit at 0.090-0.099, and the SAMPLING RATE separates
        them: every 1024 Hz instrument is low, every 512 Hz one is high.  The
        same-model comparison isolates it — MR1000RDL-EM reads 0.927/0.941 on
        the 512 Hz gliders SN 433/435 and 0.099/0.094 on the 1024 Hz SN 429.
        This converter sees only the channel config, never the rate, so it
        cannot bound on it.

        Emphatically NOT a verdict on SN 132 (issue #178) — the one instrument
        the rate does not explain, at 511.95 Hz against siblings at 0.937-0.99.
        That case needs an instrument-keyed comparison, not a static range.
        """
        msgs = _warnings_from({"name": "sh1", "diff_gain": "0.09", "sens": "0.0678"})
        assert not [m for m in msgs if "plausible range" in m]

    def test_unseen_2048hz_rung_does_not_warn(self):
        """Rockland's 512/1024/2048 Hz builds step the gain by 10x each.

        We own none of the 2048 Hz units, so ~0.0095 appears nowhere in the
        corpus — which is exactly why a floor fitted to the observed data
        (0.01) would have flagged every channel of the first one we read.
        """
        msgs = _warnings_from({"name": "sh1", "diff_gain": "0.0095", "sens": "0.0678"})
        assert not [m for m in msgs if "plausible range" in m]

    @pytest.mark.parametrize("diff_gain", ["0.00009", "50"])
    def test_impossible_diff_gain_is_flagged(self, diff_gain):
        """The wide bound still catches a value that is no gain at all."""
        msgs = _warnings_from({"name": "sh1", "diff_gain": diff_gain, "sens": "0.0678"})
        assert any("'diff_gain'" in m and "plausible range" in m for m in msgs)

    def test_bounds_bracket_the_corpus(self):
        """Guard the constants themselves against a careless edit."""
        assert _SHEAR_SENS_MIN < 0.041 and _SHEAR_SENS_MAX > 0.123
        assert _SHEAR_DIFF_GAIN_MIN < 0.09 and _SHEAR_DIFF_GAIN_MAX > 1.01
        # The floor must also clear Rockland's 2048 Hz rung (~0.0095), which we
        # do not own yet — otherwise the check fires on correct hardware the
        # day one arrives.
        assert _SHEAR_DIFF_GAIN_MIN < 0.0095


class TestCalRecorder:
    """Provenance by observation: what the converter read, not what we listed."""

    def test_records_only_what_was_read(self):
        params = CalRecorder({"sens": "0.0678", "diff_gain": "0.98", "unused_key": "3.0"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            convert_shear(DATA, params)
        cal = params.calibration()
        assert cal["sens"] == pytest.approx(0.0678)
        assert cal["diff_gain"] == pytest.approx(0.98)
        assert "unused_key" not in cal

    def test_absent_probe_is_not_provenance(self):
        """``convert_poly`` walks coef0..coef9; a missing coef2 is its exit test."""
        params = CalRecorder({"coef0": "1.0", "coef1": "2.0"})
        convert_poly(np.array([1.0]), params)
        assert set(params.calibration()) == {"coef0", "coef1"}

    def test_structural_keys_excluded(self):
        """``name`` and ``units`` are read by converters but are not coefficients."""
        params = CalRecorder({"name": "sh1", "sens": "0.07", "diff_gain": "0.95"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            convert_shear(DATA, params)
        assert set(params.calibration()) == {"sens", "diff_gain"}

    def test_numeric_structural_key_still_excluded(self):
        """The case ``float()`` alone would not catch: a channel named "2".

        A numeric ``name`` is legal in an RSI config, and without the explicit
        exclusion it would be published as a coefficient called ``cal_name``.
        """
        params = CalRecorder({"name": "2", "units": "3", "sens": "0.07", "diff_gain": "0.95"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            convert_shear(DATA, params)
        assert params["units"] == "3"  # convert_poly reads this; record the read
        assert set(params.calibration()) == {"sens", "diff_gain"}

    def test_non_numeric_value_dropped(self):
        params = CalRecorder({"coef0": "1.0", "coef1": "2.0"})
        assert params["coef0"] == "1.0"  # record the read
        assert params.calibration()["coef0"] == pytest.approx(1.0)

    def test_defaulted_coefficient_is_not_recorded(self):
        """A key the config omits is absent, even though the default was used.

        ``cal_*`` records what the INSTRUMENT declared, not every number that
        entered the arithmetic — so a missing ``cal_adc_fs`` reads as "the
        config was silent", not "no ADC scaling was applied".  Pinned because
        it is a design boundary, not an accident.
        """
        params = CalRecorder({"name": "sh1", "sens": "0.0678", "diff_gain": "0.98"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            convert_shear(DATA, params)  # adc_fs/adc_bits fall back to defaults
        assert set(params.calibration()) == {"sens", "diff_gain"}

    def test_behaves_as_a_plain_dict(self):
        """Converters must not need to know they were handed a recorder."""
        params = CalRecorder({"a": "1", "b": "2"})
        assert params["a"] == "1"
        assert params.get("b") == "2"
        assert params.get("missing", "fallback") == "fallback"
        assert "a" in params and "missing" not in params
        assert dict(params) == {"a": "1", "b": "2"}


class TestProvenanceOnRealFile:
    """End-to-end: coefficients reach the L1 variables, in probe order."""

    @pytest.fixture(scope="class")
    @staticmethod
    def l1(tmp_path_factory):
        import netCDF4

        from odas_tpw.rsi.convert import p_to_L1

        src = "tests/data/SN479_0006.p"
        out = tmp_path_factory.mktemp("l1") / "SN479_0006.nc"
        _, path = p_to_L1(src, out)
        with netCDF4.Dataset(str(path)) as ds:
            group = next(iter(ds.groups.values()))
            yield {
                name: {a: var.getncattr(a) for a in var.ncattrs()}
                for name, var in group.variables.items()
            }

    def test_shear_carries_its_coefficients(self, l1):
        attrs = l1["SHEAR"]
        np.testing.assert_allclose(attrs["cal_sens"], [0.1075, 0.113])
        np.testing.assert_allclose(attrs["cal_diff_gain"], [0.954, 0.933])

    def test_arrays_are_parallel_to_sensor_names(self, l1):
        """Probe 0's coefficient must stay attached to probe 0."""
        for var in ("SHEAR", "GRADT", "TEMP"):
            names = [n.strip() for n in l1[var]["sensor_names"].split(",")]
            for key, value in l1[var].items():
                if key.startswith("cal_"):
                    assert len(np.atleast_1d(value)) == len(names), f"{var}.{key}"

    def test_single_channel_variable_is_scalar(self, l1):
        """A one-channel variable gets a scalar, not a length-1 array."""
        assert np.isscalar(l1["TEMP_CTD"]["cal_a"]) or np.ndim(l1["TEMP_CTD"]["cal_a"]) == 0

    def test_prefix_avoids_cf_collision(self, l1):
        """TEMP_CTD's coefficients are named a..f; bare names would collide.

        This is why the L1 mechanism is prefixed while the older per-profile
        therm attrs (``diff_gain``, ``beta_1``, ... — written by
        ``profile.extract_profiles`` for ``chi_io``) are not: that one is
        confined to pre-emphasized gradient channels, where the key names
        happen not to clash.
        """
        assert "cal_a" in l1["TEMP_CTD"]
        assert "a" not in l1["TEMP_CTD"]
        assert l1["TEMP_CTD"]["units"] == "degree_Celsius"

    def test_supplementary_channels_carry_provenance(self, l1):
        np.testing.assert_allclose(l1["V_Bat"]["cal_g"], 0.1)

    def test_ragged_stack_is_nan_filled_and_survives_netcdf(self, tmp_path):
        """A coefficient only one probe carries must not shorten the array.

        Dropping it, or emitting a length-1 array against two sensor_names,
        would silently re-attribute probe 1's value to probe 0.  NaN fill is
        only useful if it round-trips, so write and read it back.
        """
        import types

        import netCDF4

        from odas_tpw.rsi.convert import _cal_attrs

        pf = types.SimpleNamespace(
            channel_info={
                "sh1": {"cal": {"sens": 0.1075, "diff_gain": 0.954}},
                "sh2": {"cal": {"sens": 0.113}},  # no diff_gain
            }
        )

        attrs = _cal_attrs(pf, ["sh1", "sh2"])
        np.testing.assert_allclose(attrs["cal_sens"], [0.1075, 0.113])
        assert len(attrs["cal_diff_gain"]) == 2
        assert attrs["cal_diff_gain"][0] == pytest.approx(0.954)
        assert np.isnan(attrs["cal_diff_gain"][1])

        path = tmp_path / "ragged.nc"
        with netCDF4.Dataset(str(path), "w", format="NETCDF4") as ds:
            ds.createDimension("N", 2)
            var = ds.createVariable("SHEAR", "f8", ("N",))
            for key, val in attrs.items():
                setattr(var, key, val)
        with netCDF4.Dataset(str(path)) as ds:
            got = ds.variables["SHEAR"].getncattr("cal_diff_gain")
        assert got[0] == pytest.approx(0.954)
        assert np.isnan(got[1])

    def test_pass_through_channel_claims_nothing(self, l1):
        """VIB is a counts pass-through here — no coefficients, so no attrs.

        Empty provenance is the honest answer, not a gap: ``convert_piezo``
        reads only ``a_0``, which this instrument's config does not set.
        """
        assert not [k for k in l1["VIB"] if k.startswith("cal_")]
