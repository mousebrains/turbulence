# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for the Dinkum -> hotel converter (odas_tpw.dinkum)."""

from __future__ import annotations

import datetime as dt

import netCDF4 as nc
import numpy as np
import pytest
import xarray as xr

from odas_tpw.dinkum.build import (
    DEFAULT_MIN_TIME,
    build_hotel,
    project_sensor,
    resolve_time_bounds,
    sanitize_time,
    time_validity,
)
from odas_tpw.dinkum.config import normalize_sensors, required_sensor_names
from odas_tpw.dinkum.reader import load_dinkum, resolve_backend, sensor_inventory

T0 = 1738627532.0


# ---------------------------------------------------------------- time bounds


def test_resolve_time_bounds_numeric():
    assert resolve_time_bounds(100, 2e9) == (100.0, 2e9)


def test_resolve_time_bounds_defaults_are_100_and_a_year_out():
    lo, hi = resolve_time_bounds(None, None, now=T0)
    assert lo == DEFAULT_MIN_TIME
    assert hi == pytest.approx(T0 + 365 * 86400.0)


def test_resolve_time_bounds_accepts_iso_dates():
    lo, hi = resolve_time_bounds("2025-01-15T00:00:00Z", "2025-04-01")
    assert lo == dt.datetime(2025, 1, 15, tzinfo=dt.UTC).timestamp()
    assert hi == dt.datetime(2025, 4, 1, tzinfo=dt.UTC).timestamp()


def test_resolve_time_bounds_naive_iso_is_utc():
    # A bare date must not be read as local time; that would shift the window
    # by the operator's offset and silently reject good data.
    lo, _ = resolve_time_bounds("2025-01-15T00:00:00", 2e9)
    assert lo == dt.datetime(2025, 1, 15, tzinfo=dt.UTC).timestamp()


@pytest.mark.parametrize(
    "lo,hi",
    [("2025-13-99", None), (None, "not-a-date"), (5, 5), (10, 5), (True, None)],
)
def test_resolve_time_bounds_rejects_bad_input(lo, hi):
    with pytest.raises(ValueError):
        resolve_time_bounds(lo, hi)


def test_time_validity_masks_nan_and_out_of_range():
    t = np.array([np.nan, 0.0, 50.0, T0, 9.9e9])
    assert time_validity(t, 100.0, 2e9).tolist() == [False, False, False, True, False]


# ------------------------------------------------------------- sanitize_time


def test_sanitize_time_removes_every_pathology():
    t = np.array([np.nan, 0.0, 50.0, T0, T0, T0 + 1, 9.9e9, T0 + 0.5])
    times, stats = sanitize_time(t, 100.0, 2e9)
    assert np.all(np.diff(times) > 0), "output must be strictly increasing"
    assert times.tolist() == [T0, T0 + 0.5, T0 + 1]
    assert stats == {
        "n_total": 8,
        "n_nan": 1,
        "n_out_of_range": 3,
        "n_duplicate": 1,
        "n_kept": 3,
    }


def test_sanitize_time_rejects_unknown_dedupe():
    with pytest.raises(ValueError, match="dedupe"):
        sanitize_time(np.array([T0]), 100.0, 2e9, dedupe="median")


# ----------------------------------------------------------- project_sensor


def _interleaved():
    """Slocum-shaped input: a value and its clock present on the same rows."""
    src_t = np.array([T0, np.nan, T0 + 2, 0.0, T0 + 4, T0 + 5, T0 + 5, T0 + 6, T0 + 7])
    src_v = np.array([10.0, np.nan, 12.0, 99.0, 14.0, 15.0, 17.0, 999.0, 18.0])
    dst = T0 + np.arange(8.0)
    return src_t, src_v, dst


def test_project_sensor_full_pipeline():
    src_t, src_v, dst = _interleaved()
    out, stats = project_sensor(
        src_t, src_v, dst, valid_min=-5.0, valid_max=100.0, time_lo=100.0, time_hi=2e9
    )
    # 99.0 dropped (bad timestamp), 999.0 dropped (out of range), the pair at
    # T0+5 averaged to 16.
    assert stats["n_bad_time"] == 1
    assert stats["n_out_of_range"] == 1
    assert stats["n_duplicate"] == 1
    assert stats["n_source"] == 5
    np.testing.assert_allclose(out, [10, 11, 12, 13, 14, 16, 17, 18])


@pytest.mark.parametrize("method,expected", [("first", 15.0), ("last", 17.0), ("mean", 16.0)])
def test_project_sensor_dedupe_methods(method, expected):
    src_t, src_v, dst = _interleaved()
    out, _ = project_sensor(
        src_t,
        src_v,
        dst,
        dedupe=method,
        valid_min=-5,
        valid_max=100,
        time_lo=100,
        time_hi=2e9,
    )
    assert out[5] == pytest.approx(expected)


def test_project_sensor_range_check_precedes_interpolation():
    # An out-of-range spike must be removed, not smeared into its neighbours.
    t = T0 + np.arange(5.0)
    v = np.array([10.0, 11.0, -999.0, 13.0, 14.0])
    out, _ = project_sensor(t, v, t, valid_min=-5, valid_max=100, time_lo=100, time_hi=2e9)
    np.testing.assert_allclose(out, [10, 11, 12, 13, 14])


def test_project_sensor_no_extrapolation_by_default():
    t = T0 + np.arange(3.0)
    v = np.array([1.0, 2.0, 3.0])
    dst = np.array([T0 - 10, T0 + 1, T0 + 100])
    out, _ = project_sensor(t, v, dst, time_lo=100, time_hi=2e9)
    assert np.isnan(out[0]) and np.isnan(out[2])
    assert out[1] == pytest.approx(2.0)


def test_project_sensor_all_nan_degrades_to_nan():
    t = T0 + np.arange(5.0)
    out, stats = project_sensor(t, np.full(5, np.nan), t, time_lo=100, time_hi=2e9)
    assert stats["n_source"] == 0
    assert np.all(np.isnan(out))


def test_project_sensor_previous_holds_state():
    t = T0 + np.array([0.0, 2.0, 4.0])
    v = np.array([1.0, 2.0, 3.0])
    dst = T0 + np.arange(5.0)
    out, _ = project_sensor(t, v, dst, method="previous", time_lo=100, time_hi=2e9)
    np.testing.assert_allclose(out, [1, 1, 2, 2, 3])


def test_project_sensor_length_mismatch_raises():
    with pytest.raises(ValueError, match="differ in length"):
        project_sensor(np.zeros(3), np.zeros(4), np.zeros(3))


# ------------------------------------------------------------------ max_gap


def test_max_gap_blanks_interpolated_but_keeps_exact_samples():
    # The regression that matters: a sensor riding the base clock lands on
    # every output time exactly. Those are measurements, and a wide
    # neighbouring gap must not NaN them.
    t = T0 + np.array([0.0, 1.0, 2.0, 50.0, 51.0])
    v = np.arange(5.0)
    out, stats = project_sensor(t, v, t, max_gap=10.0, time_lo=100, time_hi=2e9)
    np.testing.assert_allclose(out, v)
    assert stats["n_gap_blanked"] == 0


def test_max_gap_blanks_across_a_dropout():
    t = T0 + np.array([0.0, 1.0, 50.0, 51.0])
    v = np.array([0.0, 1.0, 2.0, 3.0])
    dst = T0 + np.array([0.0, 1.0, 25.0, 50.0, 51.0])
    out, stats = project_sensor(t, v, dst, max_gap=10.0, time_lo=100, time_hi=2e9)
    assert np.isnan(out[2]), "the point inside the 49 s gap must be blanked"
    np.testing.assert_allclose(out[[0, 1, 3, 4]], [0, 1, 2, 3])
    assert stats["n_gap_blanked"] == 1


# ------------------------------------------------------------------- config


def test_normalize_sensors_forms():
    out = normalize_sensors(
        {"a": None, "b": "bee", "c": {"name": "cee", "time_sensor": "m_present_time"}},
        "sci_ctd41cp_timestamp",
    )
    assert out["a"] == {"name": "a", "time_sensor": "sci_ctd41cp_timestamp"}
    assert out["b"]["name"] == "bee"
    assert out["c"]["time_sensor"] == "m_present_time"


def test_normalize_sensors_empty_is_an_error():
    with pytest.raises(ValueError, match="list at least one"):
        normalize_sensors({}, "t")


@pytest.mark.parametrize(
    "opts,match",
    [
        ({"bogus": 1}, "unknown option"),
        ({"method": "spline"}, "method"),
        ({"max_gap": 0}, "max_gap"),
        ({"valid_min": 10, "valid_max": 1}, "valid_min"),
    ],
)
def test_normalize_sensors_validation(opts, match):
    with pytest.raises(ValueError, match=match):
        normalize_sensors({"s": opts}, "t")


def test_required_sensor_names_includes_every_time_sensor():
    sensors = normalize_sensors(
        {"sci_water_temp": None, "m_pitch": {"time_sensor": "m_present_time"}},
        "sci_ctd41cp_timestamp",
    )
    assert required_sensor_names(sensors, "sci_ctd41cp_timestamp") == [
        "m_pitch",
        "m_present_time",
        "sci_ctd41cp_timestamp",
        "sci_water_temp",
    ]


# ------------------------------------------------------------------- reader


def test_resolve_backend_prefers_netcdf_for_netcdf_input(tmp_path):
    assert resolve_backend("auto", [tmp_path / "a.nc"]) == "netcdf"


def test_resolve_backend_rejects_unknown():
    with pytest.raises(ValueError, match="not one of"):
        resolve_backend("magic", [])


# ------------------------------------------------------------- end-to-end


@pytest.fixture
def glider_nc(tmp_path):
    """A Slocum-shaped NetCDF carrying every pathology we defend against."""
    n = 400
    path = tmp_path / "glider.nc"
    sci = np.zeros(n, bool)
    sci[::2] = True
    sci_t = np.where(sci, T0 + np.arange(n) * 0.5, np.nan)
    ctd_rows = np.zeros(n, bool)
    ctd_rows[::4] = True
    ctd_t = sci_t.copy()
    last = np.nan
    for i in range(n):
        if ctd_rows[i]:
            last = sci_t[i]
        elif sci[i]:
            ctd_t[i] = last  # Slocum repeats the last CTD stamp
    ctd_t[~sci] = np.nan
    ctd_t[10] = 0.0  # never-set fill
    ctd_t[12] = 4.0e9  # runaway clock
    temp = np.where(ctd_rows, 20.0 - 0.001 * np.arange(n), np.nan)
    temp[40] = -999.0  # out-of-range spike
    cond = np.where(ctd_rows, 5.4 + 1e-5 * np.arange(n), np.nan)
    flight_t = np.where(~sci, T0 + np.arange(n) * 0.5, np.nan)
    pitch = np.where(~sci, 0.4, np.nan)

    with nc.Dataset(path, "w") as ds:
        ds.createDimension("i", n)
        for name, data, units in (
            ("sci_m_present_time", sci_t, "timestamp"),
            ("sci_ctd41cp_timestamp", ctd_t, "timestamp"),
            ("m_present_time", flight_t, "timestamp"),
            ("sci_water_temp", temp, "degC"),
            ("sci_water_cond", cond, "S/m"),
            ("m_pitch", pitch, "rad"),
        ):
            var = ds.createVariable(name, "f8", ("i",), fill_value=np.nan)
            var[:] = data
            var.units = units
    return path


def _config(path, **over):
    cfg = {
        "files": {
            "root": str(path.parent),
            "patterns": [path.name],
            "output": str(path.parent / "hotel.nc"),
            "reader": "netcdf",
        },
        "time": {"base": "sci_ctd41cp_timestamp", "min_value": 100, "dedupe": "mean"},
        "projection": {"method": "linear"},
        "sensors": {
            "sci_water_temp": {"units": "degree_Celsius", "valid_min": -5, "valid_max": 45},
            "sci_water_cond": {"scale": 10.0, "units": "mS/cm"},
            "m_pitch": {"time_sensor": "m_present_time", "method": "previous"},
        },
    }
    for section, vals in over.items():
        cfg[section].update(vals)
    return cfg


def test_build_hotel_end_to_end(glider_nc):
    out = build_hotel(_config(glider_nc), now=T0)
    ds = xr.open_dataset(out, decode_cf=False)

    # The time variable keeps the base sensor's name, so the perturb side's
    # `time_column: "sci_ctd41cp_timestamp"` means exactly what it says.
    t = ds["sci_ctd41cp_timestamp"].values
    assert np.all(np.diff(t) > 0), "hotel time must be strictly increasing"
    assert t.min() > 100.0 and t.max() < 4.0e9

    assert set(ds.data_vars) == {"sci_water_temp", "sci_water_cond", "m_pitch"}
    assert ds["sci_water_cond"].attrs["units"] == "mS/cm"
    # S/m -> mS/cm applied exactly once.
    assert float(ds["sci_water_cond"].values[0]) == pytest.approx(54.0, abs=0.01)
    # The -999 spike was removed before interpolating.
    assert float(np.nanmin(ds["sci_water_temp"].values)) > 0.0
    # Per-sensor time attribution is recorded.
    assert ds["m_pitch"].attrs["dinkum_time_sensor"] == "m_present_time"
    assert ds["sci_water_temp"].attrs["dinkum_time_sensor"] == "sci_ctd41cp_timestamp"
    # Provenance for the rejections.
    assert ds.attrs["dinkum_time_base_rejected_range"] >= 2
    assert ds.attrs["dinkum_time_base_duplicates"] > 0


def test_build_hotel_output_override(glider_nc, tmp_path):
    dest = tmp_path / "sub" / "custom.nc"
    out = build_hotel(_config(glider_nc), output=dest, now=T0)
    assert out == dest and dest.exists()


def test_build_hotel_missing_sensor_names_it(glider_nc):
    cfg = _config(glider_nc)
    cfg["sensors"]["sci_not_a_sensor"] = None
    with pytest.raises(KeyError, match="sci_not_a_sensor"):
        build_hotel(cfg, now=T0)


def test_build_hotel_rejects_degenerate_time_base(glider_nc):
    # The window admits only the single runaway 4e9 stamp. One time is not a
    # usable hotel file -- perturb's loader skips any channel with < 2 times --
    # so this must fail here rather than produce a file that merges as nothing.
    cfg = _config(glider_nc, time={"min_value": 3.9e9, "max_value": 4.1e9})
    with pytest.raises(ValueError, match="at least 2 are needed"):
        build_hotel(cfg, now=T0)


def test_build_hotel_rejects_time_base_with_no_valid_samples(glider_nc):
    cfg = _config(glider_nc, time={"min_value": 5.0e9, "max_value": 6.0e9})
    with pytest.raises(ValueError, match="at least 2 are needed"):
        build_hotel(cfg, now=T0)


def test_build_hotel_no_files_matched(tmp_path):
    cfg = {
        "files": {"root": str(tmp_path), "patterns": ["*.nope"], "reader": "netcdf"},
        "time": {"base": "t"},
        "sensors": {"a": None},
    }
    with pytest.raises(FileNotFoundError, match="No Dinkum files matched"):
        build_hotel(cfg, now=T0)


def test_load_dinkum_and_inventory(glider_nc):
    ds = load_dinkum([glider_nc], backend="netcdf")
    assert ds.sizes["record"] == 400
    rows = {r["name"]: r for r in sensor_inventory(ds)}
    assert rows["sci_water_temp"]["n_finite"] == 100
    assert rows["sci_water_cond"]["units"] == "S/m"


def test_load_dinkum_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dinkum([tmp_path / "nope.nc"], backend="netcdf")


def test_load_dinkum_empty_list():
    with pytest.raises(ValueError, match="No input files"):
        load_dinkum([], backend="netcdf")
