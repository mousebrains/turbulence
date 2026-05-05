# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for rsi.helpers — channel loading, profile prep, dataset builder."""

from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np
import pytest
import xarray as xr

from odas_tpw.rsi.helpers import (
    _build_l1data_from_channels,
    _build_result_dataset,
    _channels_from_nc,
    load_channels,
    prepare_profiles,
    write_profile_results,
)


def _make_channels_dict(
    *,
    n_slow=200,
    fs_fast=512.0,
    fs_slow=64.0,
    is_profile=False,
    n_shear=2,
    n_accel=2,
    pressure_ramp=(5.0, 50.0),
    vehicle="vmp",
):
    """Build a synthetic channels dict shaped like load_channels output."""
    ratio = round(fs_fast / fs_slow)
    n_fast = n_slow * ratio
    rng = np.random.default_rng(42)

    return {
        "shear": [
            (f"sh{i + 1}", rng.standard_normal(n_fast) * 0.05) for i in range(n_shear)
        ],
        "accel": [
            (f"A{c}", rng.standard_normal(n_fast) * 0.005)
            for c in ["x", "y", "z"][:n_accel]
        ],
        "P": np.linspace(pressure_ramp[0], pressure_ramp[1], n_slow),
        "T": np.linspace(20.0, 5.0, n_slow),
        "t_fast": np.arange(n_fast) / fs_fast,
        "t_slow": np.arange(n_slow) / fs_slow,
        "fs_fast": fs_fast,
        "fs_slow": fs_slow,
        "is_profile": is_profile,
        "vehicle": vehicle,
        "metadata": {"source": "synthetic"},
    }


# ---------------------------------------------------------------------------
# load_channels — error path
# ---------------------------------------------------------------------------


class TestLoadChannelsUnsupported:
    def test_unsupported_suffix_raises(self, tmp_path):
        bad_path = tmp_path / "data.txt"
        bad_path.write_text("not a real file")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_channels(bad_path)


# ---------------------------------------------------------------------------
# _channels_from_nc — vehicle inference branches
# ---------------------------------------------------------------------------


def _write_minimal_nc(
    path: Path,
    *,
    instrument_model: str | None = None,
    vehicle: str | None = None,
) -> Path:
    """Write a minimal NetCDF that load_channels can parse."""
    n_fast = 256
    n_slow = 32
    fs_fast = 512.0
    fs_slow = 64.0

    ds = netCDF4.Dataset(str(path), "w", format="NETCDF4")
    try:
        ds.createDimension("time_fast", n_fast)
        ds.createDimension("time_slow", n_slow)

        tf = ds.createVariable("t_fast", "f8", ("time_fast",))
        tf[:] = np.arange(n_fast) / fs_fast
        ts = ds.createVariable("t_slow", "f8", ("time_slow",))
        ts[:] = np.arange(n_slow) / fs_slow

        p = ds.createVariable("P", "f8", ("time_slow",))
        p[:] = np.linspace(5.0, 50.0, n_slow)
        t1 = ds.createVariable("T1", "f8", ("time_slow",))
        t1[:] = np.linspace(20.0, 5.0, n_slow)

        sh1 = ds.createVariable("sh1", "f8", ("time_fast",))
        sh1[:] = np.random.default_rng(0).standard_normal(n_fast) * 0.05
        ax = ds.createVariable("Ax", "f8", ("time_fast",))
        ax[:] = np.random.default_rng(1).standard_normal(n_fast) * 0.005

        ds.fs_fast = fs_fast
        ds.fs_slow = fs_slow
        if instrument_model is not None:
            ds.instrument_model = instrument_model
        if vehicle is not None:
            ds.vehicle = vehicle
    finally:
        ds.close()
    return path


class TestChannelsFromNcVehicleInference:
    """Vehicle inference falls through model-name heuristics when no vehicle attr."""

    def test_vehicle_attr_wins(self, tmp_path):
        path = _write_minimal_nc(tmp_path / "v.nc", vehicle="VMP")
        data = load_channels(path)
        assert data["vehicle"] == "vmp"

    def test_vmp_from_model(self, tmp_path):
        path = _write_minimal_nc(tmp_path / "vmp.nc", instrument_model="VMP-250")
        data = load_channels(path)
        assert data["vehicle"] == "vmp"

    def test_xmp_from_model(self, tmp_path):
        path = _write_minimal_nc(tmp_path / "xmp.nc", instrument_model="XMP-PROFILER")
        data = load_channels(path)
        assert data["vehicle"] == "xmp"

    def test_microrider_from_model(self, tmp_path):
        path = _write_minimal_nc(tmp_path / "mr.nc", instrument_model="MicroRider")
        data = load_channels(path)
        assert data["vehicle"] == "slocum_glider"

    def test_mr_from_model(self, tmp_path):
        path = _write_minimal_nc(tmp_path / "mr2.nc", instrument_model="MR-1000")
        data = load_channels(path)
        assert data["vehicle"] == "slocum_glider"

    def test_no_vehicle_no_model(self, tmp_path):
        path = _write_minimal_nc(tmp_path / "none.nc")
        data = load_channels(path)
        assert data["vehicle"] == ""

    def test_unrecognized_model(self, tmp_path):
        path = _write_minimal_nc(tmp_path / "weird.nc", instrument_model="QuantumScope3000")
        data = load_channels(path)
        assert data["vehicle"] == ""

    def test_load_via_function_with_kwarg_pattern(self, tmp_path):
        """Custom shear/accel patterns trigger the re.compile branch."""
        path = _write_minimal_nc(tmp_path / "custom.nc")
        data = _channels_from_nc(
            path,
            sh_pat=r"^sh[12]$",  # different from default → re.compile path
            ac_pat=r"^A[xy]$",  # different from default
            p_name="P",
            t_name="T1",
        )
        assert len(data["shear"]) == 1
        assert data["shear"][0][0] == "sh1"


# ---------------------------------------------------------------------------
# prepare_profiles — vehicle/profile/salinity branches
# ---------------------------------------------------------------------------


class TestPrepareProfiles:
    def test_vehicle_none_uses_data_vehicle(self):
        data = _make_channels_dict(is_profile=True, vehicle="vmp")
        result = prepare_profiles(data, speed=0.5, direction="auto", salinity=None, vehicle=None)
        assert result is not None

    def test_vehicle_none_data_missing_vehicle_key(self):
        """When data has no vehicle key, falls through to empty string."""
        data = _make_channels_dict(is_profile=True)
        del data["vehicle"]
        # Should not raise — defaults to ""
        result = prepare_profiles(data, speed=0.5, direction="down", salinity=None, vehicle=None)
        assert result is not None

    def test_no_profiles_detected_returns_none(self):
        """When get_profiles finds nothing, return None."""
        # is_profile=False forces detection. A flat pressure profile won't satisfy
        # get_profiles default thresholds.
        data = _make_channels_dict(
            is_profile=False, pressure_ramp=(0.1, 0.1), n_slow=100
        )
        # Use direction='down' with flat pressure → no descent → no profiles
        result = prepare_profiles(data, speed=0.5, direction="down", salinity=None, vehicle="vmp")
        assert result is None

    def test_salinity_scalar(self):
        data = _make_channels_dict(is_profile=True)
        result = prepare_profiles(data, speed=0.5, direction="down", salinity=34.5, vehicle="vmp")
        _profs, _spd, _P, _T, sal_fast, *_ = result
        assert sal_fast == 34.5

    def test_salinity_array_slow_length(self):
        data = _make_channels_dict(is_profile=True)
        sal = np.full(len(data["t_slow"]), 33.0)
        result = prepare_profiles(data, speed=0.5, direction="down", salinity=sal, vehicle="vmp")
        _profs, _spd, _P, _T, sal_fast, *_ = result
        assert len(sal_fast) == len(data["t_fast"])
        np.testing.assert_allclose(sal_fast, 33.0)

    def test_salinity_array_fast_length(self):
        data = _make_channels_dict(is_profile=True)
        sal = np.full(len(data["t_fast"]), 32.0)
        result = prepare_profiles(data, speed=0.5, direction="down", salinity=sal, vehicle="vmp")
        _profs, _spd, _P, _T, sal_fast, *_ = result
        assert len(sal_fast) == len(data["t_fast"])
        np.testing.assert_allclose(sal_fast, 32.0)

    def test_salinity_array_wrong_length_raises(self):
        data = _make_channels_dict(is_profile=True)
        sal = np.full(7, 35.0)  # neither slow nor fast length
        with pytest.raises(ValueError, match="salinity array length"):
            prepare_profiles(data, speed=0.5, direction="down", salinity=sal, vehicle="vmp")

    def test_salinity_none(self):
        data = _make_channels_dict(is_profile=True)
        result = prepare_profiles(data, speed=0.5, direction="down", salinity=None, vehicle="vmp")
        _profs, _spd, _P, _T, sal_fast, *_ = result
        assert sal_fast is None

    def test_speed_none_computes_from_pressure(self):
        """When speed is None, use compute_speed_fast."""
        data = _make_channels_dict(is_profile=True, pressure_ramp=(5.0, 80.0), n_slow=500)
        result = prepare_profiles(data, speed=None, direction="down", salinity=None, vehicle="vmp")
        _profs, speed_fast, *_ = result
        # Speed should be positive everywhere (>= speed_cutout)
        assert np.all(speed_fast >= 0.05)


# ---------------------------------------------------------------------------
# _build_l1data_from_channels — empty arrays edge cases
# ---------------------------------------------------------------------------


class TestBuildL1DataFromChannelsEdges:
    def test_no_shear_zero_array(self):
        """When shear list is empty, an empty (0, n) array is built."""
        data = _make_channels_dict(n_shear=0)
        n_slow = len(data["t_slow"])
        ratio = round(data["fs_fast"] / data["fs_slow"])

        s_fast, e_fast = 0, n_slow * ratio
        speed_fast = np.full(e_fast, 0.5)
        P_fast = np.full(e_fast, 10.0)
        T_fast = np.full(e_fast, 12.0)

        l1 = _build_l1data_from_channels(
            data, s_fast, e_fast, speed_fast, P_fast, T_fast, direction="down"
        )
        assert l1.shear.shape[0] == 0

    def test_no_accel_vib_type_none(self):
        """When accel list is empty, vib_type = NONE."""
        data = _make_channels_dict(n_accel=0)
        n_slow = len(data["t_slow"])
        ratio = round(data["fs_fast"] / data["fs_slow"])

        s_fast, e_fast = 0, n_slow * ratio
        speed_fast = np.full(e_fast, 0.5)
        P_fast = np.full(e_fast, 10.0)
        T_fast = np.full(e_fast, 12.0)

        l1 = _build_l1data_from_channels(
            data, s_fast, e_fast, speed_fast, P_fast, T_fast, direction="down"
        )
        assert l1.vib_type == "NONE"
        assert l1.vib.shape[0] == 0

    def test_no_therm_list(self):
        """When no therm list is given, temp_fast is empty."""
        data = _make_channels_dict()
        n_slow = len(data["t_slow"])
        ratio = round(data["fs_fast"] / data["fs_slow"])

        s_fast, e_fast = 0, n_slow * ratio
        speed_fast = np.full(e_fast, 0.5)
        P_fast = np.full(e_fast, 10.0)
        T_fast = np.full(e_fast, 12.0)

        l1 = _build_l1data_from_channels(
            data,
            s_fast,
            e_fast,
            speed_fast,
            P_fast,
            T_fast,
            direction="down",
            therm_list=None,
        )
        assert l1.temp_fast.size == 0


# ---------------------------------------------------------------------------
# write_profile_results — single vs multi naming
# ---------------------------------------------------------------------------


def _make_tiny_ds(n=4, P_offset=10.0):
    return xr.Dataset(
        {
            "P_mean": (["time"], np.linspace(P_offset, P_offset + 5, n)),
            "epsilon": (["time"], np.full(n, 1e-9)),
        },
        coords={"time": np.arange(n)},
    )


class TestWriteProfileResults:
    def test_single_result_no_prof_suffix(self, tmp_path):
        """One result → name pattern is {stem}_{suffix}.nc (no prof index)."""
        ds = _make_tiny_ds()
        source = tmp_path / "src.p"
        out = tmp_path / "out"
        paths = write_profile_results([ds], source, out, suffix="eps")
        assert len(paths) == 1
        assert paths[0].name == "src_eps.nc"
        assert paths[0].exists()

    def test_multiple_results_get_prof_index(self, tmp_path):
        """Multiple results → name pattern includes prof### index."""
        ds_list = [_make_tiny_ds(P_offset=10 * i) for i in range(1, 4)]
        source = tmp_path / "src.p"
        out = tmp_path / "out"
        paths = write_profile_results(ds_list, source, out, suffix="chi")
        assert len(paths) == 3
        names = [p.name for p in paths]
        assert names == ["src_prof001_chi.nc", "src_prof002_chi.nc", "src_prof003_chi.nc"]

    def test_creates_nested_output_dir(self, tmp_path):
        ds = _make_tiny_ds()
        source = tmp_path / "src.p"
        deep = tmp_path / "a" / "b" / "c"
        paths = write_profile_results([ds], source, deep, suffix="x")
        assert paths[0].parent == deep


# ---------------------------------------------------------------------------
# _build_result_dataset
# ---------------------------------------------------------------------------


class TestBuildResultDataset:
    def test_assembles_dataset(self):
        n_time = 5
        n_probe = 2
        variables = [
            (
                "epsilon",
                ["probe", "time"],
                np.full((n_probe, n_time), 1e-9),
                {"units": "W/kg"},
            ),
        ]
        probe_names = ["sh1", "sh2"]
        t_out = np.linspace(0, 1, n_time)
        ds = _build_result_dataset(
            variables,
            probe_names,
            t_out,
            probe_long_name="shear probe",
            global_attrs={"foo": "bar"},
        )
        assert ds.attrs["foo"] == "bar"
        assert list(ds["probe"].values) == probe_names
        assert ds["epsilon"].shape == (n_probe, n_time)
        assert ds.coords["probe"].attrs["long_name"] == "shear probe"
