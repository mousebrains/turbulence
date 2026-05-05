"""Tests for pyturb-cli package."""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

# ---------------------------------------------------------------------------
# _compat.seconds_to_samples
# ---------------------------------------------------------------------------


class TestSecondsToSamples:
    """Test _compat.seconds_to_samples (Jesse's convention — even, no pow2)."""

    def test_exact_even(self):
        from odas_tpw.pyturb._compat import seconds_to_samples

        # 1.0 s * 512 Hz = 512 (already even)
        assert seconds_to_samples(1.0, 512.0) == 512

    def test_typical_diss_len(self):
        from odas_tpw.pyturb._compat import seconds_to_samples

        # 4 s * 512 Hz = 2048
        assert seconds_to_samples(4.0, 512.0) == 2048

    def test_non_power_stays_non_power(self):
        from odas_tpw.pyturb._compat import seconds_to_samples

        # 2.0 s * 100 Hz = 200 (even, NOT rounded to 256)
        assert seconds_to_samples(2.0, 100.0) == 200

    def test_odd_made_even(self):
        from odas_tpw.pyturb._compat import seconds_to_samples

        # 3.0 s * 101 Hz = 303 -> 304 (made even)
        assert seconds_to_samples(3.0, 101.0) == 304

    def test_small_value_minimum_2(self):
        from odas_tpw.pyturb._compat import seconds_to_samples

        result = seconds_to_samples(0.001, 512.0)
        assert result >= 2

    def test_typical_fft_len(self):
        from odas_tpw.pyturb._compat import seconds_to_samples

        # 1 s * 512 Hz = 512
        assert seconds_to_samples(1.0, 512.0) == 512

    def test_half_second(self):
        from odas_tpw.pyturb._compat import seconds_to_samples

        # 0.5 s * 512 Hz = 256
        assert seconds_to_samples(0.5, 512.0) == 256

    @pytest.mark.parametrize("secs,fs", [(0.25, 512), (2.0, 512), (8.0, 512), (1.0, 1024)])
    def test_always_even(self, secs, fs):
        from odas_tpw.pyturb._compat import seconds_to_samples

        result = seconds_to_samples(secs, fs)
        assert result >= 2
        assert result % 2 == 0


class TestComputeWindowParameters:
    """Test _compat.compute_window_parameters (Jesse's convention)."""

    def test_standard_512hz(self):
        from odas_tpw.pyturb._compat import compute_window_parameters

        n_fft, n_diss, fft_ovlp, diss_ovlp = compute_window_parameters(1.0, 4.0, 512.0)
        assert n_fft == 512
        assert n_diss == 2048
        assert n_diss % n_fft == 0
        assert fft_ovlp == 256
        assert diss_ovlp == 256

    def test_diss_rounded_to_multiple_of_fft(self):
        from odas_tpw.pyturb._compat import compute_window_parameters

        # 1.5 s * 512 Hz = 768 -> round to multiple of 512 -> 1024
        n_fft, n_diss, _, _ = compute_window_parameters(1.0, 1.5, 512.0)
        assert n_fft == 512
        assert n_diss == 1024
        assert n_diss % n_fft == 0

    def test_non_standard_fs(self):
        from odas_tpw.pyturb._compat import compute_window_parameters

        n_fft, n_diss, fft_ovlp, diss_ovlp = compute_window_parameters(1.0, 4.0, 100.0)
        assert n_fft == 100
        assert n_diss == 400
        assert n_diss % n_fft == 0
        assert fft_ovlp == 50
        assert diss_ovlp == 50

    def test_diss_equals_fft(self):
        from odas_tpw.pyturb._compat import compute_window_parameters

        n_fft, n_diss, _, _ = compute_window_parameters(1.0, 1.0, 512.0)
        assert n_fft == 512
        assert n_diss == 512


# ---------------------------------------------------------------------------
# _compat.check_overwrite
# ---------------------------------------------------------------------------


class TestCheckOverwrite:
    def test_nonexistent_file(self, tmp_path):
        from odas_tpw.pyturb._compat import check_overwrite

        assert check_overwrite(tmp_path / "nope.nc", False) is True

    def test_existing_no_overwrite(self, tmp_path):
        from odas_tpw.pyturb._compat import check_overwrite

        f = tmp_path / "exists.nc"
        f.touch()
        assert check_overwrite(f, False) is False

    def test_existing_with_overwrite(self, tmp_path):
        from odas_tpw.pyturb._compat import check_overwrite

        f = tmp_path / "exists.nc"
        f.touch()
        assert check_overwrite(f, True) is True


# ---------------------------------------------------------------------------
# _compat.load_auxiliary
# ---------------------------------------------------------------------------


class TestLoadAuxiliary:
    def test_valid_auxiliary(self, tmp_path):
        from odas_tpw.pyturb._compat import load_auxiliary

        ds = xr.Dataset(
            {
                "lat": (["time"], [45.0, 45.1]),
                "lon": (["time"], [-130.0, -130.1]),
                "temperature": (["time"], [10.0, 10.5]),
                "salinity": (["time"], [34.0, 34.1]),
                "density": (["time"], [1025.0, 1025.1]),
            }
        )
        path = tmp_path / "aux.nc"
        ds.to_netcdf(path)

        loaded = load_auxiliary(path)
        assert "lat" in loaded
        assert "salinity" in loaded

    def test_missing_variable(self, tmp_path):
        from odas_tpw.pyturb._compat import load_auxiliary

        ds = xr.Dataset({"lat": (["time"], [45.0])})
        path = tmp_path / "aux.nc"
        ds.to_netcdf(path)

        with pytest.raises(KeyError, match="lon"):
            load_auxiliary(path)

    def test_custom_var_names(self, tmp_path):
        from odas_tpw.pyturb._compat import load_auxiliary

        ds = xr.Dataset(
            {
                "LAT": (["time"], [45.0]),
                "LON": (["time"], [-130.0]),
                "T": (["time"], [10.0]),
                "S": (["time"], [34.0]),
                "D": (["time"], [1025.0]),
            }
        )
        path = tmp_path / "aux.nc"
        ds.to_netcdf(path)

        loaded = load_auxiliary(
            path, lat_var="LAT", lon_var="LON", temp_var="T", sal_var="S", dens_var="D"
        )
        assert "LAT" in loaded


# ---------------------------------------------------------------------------
# _profind — Peak-finding profile detection
# ---------------------------------------------------------------------------


class TestProfilePeaks:
    """Test _profind.find_profiles_peaks."""

    def test_single_downcast(self):
        from odas_tpw.pyturb._profind import find_profiles_peaks

        fs = 64.0
        n = int(60 * fs)
        t = np.arange(n) / fs
        pressure = 50.0 * np.sin(np.pi * t / 60.0)
        pressure = np.maximum(pressure, 0)

        profiles = find_profiles_peaks(
            pressure, fs, direction="down",
            peaks_height=10.0, peaks_distance=100, peaks_prominence=10.0,
        )
        assert len(profiles) >= 1
        s, e = profiles[0]
        assert s < e

    def test_no_profiles_flat(self):
        from odas_tpw.pyturb._profind import find_profiles_peaks

        pressure = np.ones(1000) * 2.0
        profiles = find_profiles_peaks(pressure, 64.0, peaks_height=25.0)
        assert len(profiles) == 0

    def test_multiple_casts(self):
        from odas_tpw.pyturb._profind import find_profiles_peaks

        fs = 64.0
        n = int(120 * fs)
        t = np.arange(n) / fs
        pressure = 50.0 * np.abs(np.sin(2 * np.pi * t / 60.0))

        profiles = find_profiles_peaks(
            pressure, fs, direction="down",
            peaks_height=10.0, peaks_distance=100, peaks_prominence=10.0,
        )
        assert len(profiles) >= 2

    def test_short_pressure(self):
        from odas_tpw.pyturb._profind import find_profiles_peaks

        pressure = np.array([1.0, 2.0, 3.0])
        profiles = find_profiles_peaks(pressure, 64.0)
        assert profiles == []

    def test_upcast_direction(self):
        from odas_tpw.pyturb._profind import find_profiles_peaks

        fs = 64.0
        n = int(60 * fs)
        t = np.arange(n) / fs
        # Down then up
        pressure = 50.0 * np.sin(np.pi * t / 60.0)
        pressure = np.maximum(pressure, 0)

        profiles = find_profiles_peaks(
            pressure, fs, direction="up",
            peaks_height=10.0, peaks_distance=100, peaks_prominence=10.0,
        )
        # Should detect the upcast segment
        assert isinstance(profiles, list)

    def test_both_direction(self):
        from odas_tpw.pyturb._profind import find_profiles_peaks

        fs = 64.0
        n = int(60 * fs)
        t = np.arange(n) / fs
        pressure = 50.0 * np.sin(np.pi * t / 60.0)
        pressure = np.maximum(pressure, 0)

        profiles = find_profiles_peaks(
            pressure, fs, direction="both",
            peaks_height=10.0, peaks_distance=100, peaks_prominence=10.0,
        )
        assert isinstance(profiles, list)

    def test_profiles_sorted_by_start(self):
        from odas_tpw.pyturb._profind import find_profiles_peaks

        fs = 64.0
        n = int(180 * fs)
        t = np.arange(n) / fs
        pressure = 50.0 * np.abs(np.sin(2 * np.pi * t / 60.0))

        profiles = find_profiles_peaks(
            pressure, fs, direction="down",
            peaks_height=10.0, peaks_distance=100, peaks_prominence=10.0,
        )
        if len(profiles) > 1:
            starts = [p[0] for p in profiles]
            assert starts == sorted(starts)

    def test_custom_peaks_params(self):
        from odas_tpw.pyturb._profind import find_profiles_peaks

        fs = 64.0
        n = int(60 * fs)
        t = np.arange(n) / fs
        pressure = 50.0 * np.sin(np.pi * t / 60.0)
        pressure = np.maximum(pressure, 0)

        # Very strict parameters — may find fewer profiles
        profiles_strict = find_profiles_peaks(
            pressure, fs,
            peaks_height=40.0, peaks_distance=500, peaks_prominence=40.0,
        )
        # Relaxed parameters — may find more
        profiles_relaxed = find_profiles_peaks(
            pressure, fs,
            peaks_height=5.0, peaks_distance=50, peaks_prominence=5.0,
        )
        assert len(profiles_relaxed) >= len(profiles_strict)


# ---------------------------------------------------------------------------
# CLI --help
# ---------------------------------------------------------------------------


class TestCliHelp:
    """Test that CLI --help works for each subcommand."""

    @pytest.mark.parametrize("cmd", ["p2nc", "merge", "eps", "bin"])
    def test_help(self, cmd):
        result = subprocess.run(
            [sys.executable, "-m", "odas_tpw.pyturb.cli", cmd, "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert cmd in result.stdout.lower() or "--help" in result.stdout

    def test_main_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "odas_tpw.pyturb.cli", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_no_command_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, "-m", "odas_tpw.pyturb.cli"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


class TestCliParsing:
    """Test argparse parses flags correctly."""

    def test_p2nc_defaults(self):

        from odas_tpw.pyturb.cli import main

        # Capture namespace by monkey-patching
        captured = {}

        def _capture(args):
            captured.update(vars(args))

        with patch("odas_tpw.pyturb.cli._run_p2nc", _capture):
            main(["p2nc", "test.p"])

        assert captured["output"] == "."
        assert captured["compress"] is True
        assert captured["compression_level"] == 4
        assert captured["n_workers"] == 1
        assert captured["min_file_size"] == 100000
        assert captured["overwrite"] is False

    def test_eps_defaults(self):
        from odas_tpw.pyturb.cli import main

        captured = {}

        def _capture(args):
            captured.update(vars(args))

        with patch("odas_tpw.pyturb.cli._run_eps", _capture):
            main(["eps", "test.nc", "-o", "/tmp/out"])

        assert captured["diss_len"] == 4.0
        assert captured["fft_len"] == 1.0
        assert captured["min_speed"] == 0.2
        assert captured["direction"] == "down"
        assert captured["peaks_height"] == 25.0
        assert captured["peaks_distance"] == 200
        assert captured["peaks_prominence"] == 25.0
        assert captured["salinity"] == 35.0
        assert captured["temperature"] == "JAC_T"
        assert captured["goodman"] is False  # off by default (matches Jesse)

    def test_bin_defaults(self):
        from odas_tpw.pyturb.cli import main

        captured = {}

        def _capture(args):
            captured.update(vars(args))

        with patch("odas_tpw.pyturb.cli._run_bin", _capture):
            main(["bin", "test.nc"])

        assert captured["output"] == "binned_profiles.nc"
        assert captured["bin_width"] == 2.0
        assert captured["dmin"] == 0.0
        assert captured["dmax"] == 1000.0
        assert captured["lat"] == 45.0
        assert captured["pressure"] is False

    def test_merge_requires_output(self):
        from odas_tpw.pyturb.cli import main

        with pytest.raises(SystemExit):
            main(["merge", "test.nc"])

    def test_eps_verbose(self):
        from odas_tpw.pyturb.cli import main

        captured = {}

        def _capture(args):
            captured.update(vars(args))

        with patch("odas_tpw.pyturb.cli._run_eps", _capture):
            main(["-vv", "eps", "test.nc", "-o", "/tmp/out"])

    def test_eps_goodman_flag(self):
        from odas_tpw.pyturb.cli import main

        captured = {}

        def _capture(args):
            captured.update(vars(args))

        with patch("odas_tpw.pyturb.cli._run_eps", _capture):
            main(["eps", "test.nc", "-o", "/tmp/out", "--goodman"])

        assert captured["goodman"] is True

    def test_p2nc_no_compress(self):
        from odas_tpw.pyturb.cli import main

        captured = {}

        def _capture(args):
            captured.update(vars(args))

        with patch("odas_tpw.pyturb.cli._run_p2nc", _capture):
            main(["p2nc", "test.p", "--no-compress"])

        assert captured["compress"] is False


# ---------------------------------------------------------------------------
# _compat.rename_eps_dataset
# ---------------------------------------------------------------------------


class TestRenameEpsDataset:
    """Test _compat.rename_eps_dataset with mock data."""

    def _make_l4(self, n_spec=5, n_shear=2):
        from odas_tpw.scor160.io import L4Data

        return L4Data(
            time=np.arange(n_spec, dtype=float),
            pres=np.linspace(10, 50, n_spec),
            pspd_rel=np.full(n_spec, 0.7),
            section_number=np.ones(n_spec),
            epsi=np.full((n_shear, n_spec), 1e-8),
            epsi_final=np.full(n_spec, 1e-8),
            epsi_flags=np.zeros((n_shear, n_spec)),
            fom=np.ones((n_shear, n_spec)),
            mad=np.full((n_shear, n_spec), 0.1),
            kmax=np.full((n_shear, n_spec), 50.0),
            method=np.zeros((n_shear, n_spec)),
            var_resolved=np.full((n_shear, n_spec), 0.9),
        )

    def _make_l3(self, n_spec=5, n_freq=257, n_shear=2):
        from odas_tpw.scor160.io import L3Data

        return L3Data(
            time=np.arange(n_spec, dtype=float),
            pres=np.linspace(10, 50, n_spec),
            temp=np.full(n_spec, 15.0),
            pspd_rel=np.full(n_spec, 0.7),
            section_number=np.ones(n_spec),
            kcyc=np.tile(np.arange(n_freq, dtype=float)[:, None], (1, n_spec)),
            sh_spec=np.ones((n_shear, n_freq, n_spec)) * 1e-6,
            sh_spec_clean=np.ones((n_shear, n_freq, n_spec)) * 1e-6,
        )

    def test_empty_l4(self):
        from odas_tpw.pyturb._compat import rename_eps_dataset
        from odas_tpw.scor160.io import L3Data, L4Data

        l4 = L4Data(
            time=np.array([]), pres=np.array([]), pspd_rel=np.array([]),
            section_number=np.array([]), epsi=np.zeros((2, 0)),
            epsi_final=np.array([]), epsi_flags=np.zeros((2, 0)),
            fom=np.zeros((2, 0)), mad=np.zeros((2, 0)), kmax=np.zeros((2, 0)),
            method=np.zeros((2, 0)), var_resolved=np.zeros((2, 0)),
        )
        l3 = L3Data(
            time=np.array([]), pres=np.array([]), temp=np.array([]),
            pspd_rel=np.array([]), section_number=np.array([]),
            kcyc=np.zeros((257, 0)), sh_spec=np.zeros((2, 257, 0)),
            sh_spec_clean=np.zeros((2, 257, 0)),
        )
        ds = rename_eps_dataset(l4, l3, None)
        assert ds.sizes == {}

    def test_basic_output_variables(self):
        from odas_tpw.pyturb._compat import rename_eps_dataset

        l4 = self._make_l4()
        l3 = self._make_l3()
        ds = rename_eps_dataset(l4, l3, None)

        # Jesse's variable names
        for v in ["eps_1", "eps_2", "pressure", "W", "temperature",
                   "nu", "salinity", "density", "S_sh1", "S_sh2",
                   "eps_final", "k_max_1", "k_max_2", "fom_1", "fom_2",
                   "mad_1", "mad_2", "method_1", "method_2"]:
            assert v in ds, f"Missing variable: {v}"
        assert ds.sizes["time"] == 5

    def test_frequency_coordinate(self):
        from odas_tpw.pyturb._compat import rename_eps_dataset

        l4 = self._make_l4()
        l3 = self._make_l3()
        ds = rename_eps_dataset(l4, l3, None)

        assert "frequency" in ds.coords
        assert "k" in ds.coords
        assert ds.sizes["frequency"] == 257

    def test_with_chi_data(self):
        from odas_tpw.chi.l3_chi import L3ChiData
        from odas_tpw.pyturb._compat import rename_eps_dataset

        n_spec = 5
        n_freq = 257
        l4 = self._make_l4(n_spec)
        l3 = self._make_l3(n_spec, n_freq)

        l3_chi = L3ChiData(
            time=np.arange(n_spec, dtype=float),
            pres=np.linspace(10, 50, n_spec),
            temp=np.full(n_spec, 15.0),
            pspd_rel=np.full(n_spec, 0.7),
            section_number=np.ones(n_spec),
            nu=np.full(n_spec, 1.1e-6),
            kcyc=np.tile(np.arange(n_freq, dtype=float)[:, None], (1, n_spec)),
            freq=np.arange(n_freq, dtype=float),
            gradt_spec=np.ones((2, n_freq, n_spec)) * 1e-4,
            noise_spec=np.ones((2, n_freq, n_spec)) * 1e-6,
            H2=np.ones((n_spec, n_freq)),
            tau0=np.full(n_spec, 0.01),
        )

        ds = rename_eps_dataset(l4, l3, l3_chi)
        assert "S_gradT1" in ds
        assert "S_gradT2" in ds
        assert ds["S_gradT1"].dims == ("frequency", "time")

    def test_single_shear_probe(self):
        from odas_tpw.pyturb._compat import rename_eps_dataset

        l4 = self._make_l4(n_shear=1)
        l3 = self._make_l3(n_shear=1)
        ds = rename_eps_dataset(l4, l3, None)

        assert "eps_1" in ds
        assert "eps_2" not in ds
        assert "S_sh1" in ds
        assert "S_sh2" not in ds

    def test_output_writeable(self, tmp_path):
        from odas_tpw.pyturb._compat import rename_eps_dataset

        l4 = self._make_l4()
        l3 = self._make_l3()
        ds = rename_eps_dataset(l4, l3, None)

        out = tmp_path / "test_eps.nc"
        ds.to_netcdf(out)
        assert out.exists()

        reopened = xr.open_dataset(out)
        assert "eps_1" in reopened
        reopened.close()


# ---------------------------------------------------------------------------
# p2nc module
# ---------------------------------------------------------------------------


class TestP2nc:
    """Test p2nc subcommand logic."""

    def test_skip_small_files(self, tmp_path):
        from odas_tpw.pyturb.p2nc import run_p2nc

        # Create a tiny file
        small = tmp_path / "tiny.p"
        small.write_bytes(b"\x00" * 100)

        args = MagicMock()
        args.files = [str(small)]
        args.output = str(tmp_path / "out")
        args.compress = True
        args.compression_level = 4
        args.n_workers = 1
        args.min_file_size = 1000
        args.overwrite = False

        # Should skip without error
        run_p2nc(args)
        out_dir = tmp_path / "out"
        assert not list(out_dir.glob("*.nc")) if out_dir.exists() else True

    def test_skip_existing(self, tmp_path):
        from odas_tpw.pyturb.p2nc import run_p2nc

        # Create input and pre-existing output
        big = tmp_path / "data.p"
        big.write_bytes(b"\x00" * 200000)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        existing = out_dir / "data.nc"
        existing.touch()

        args = MagicMock()
        args.files = [str(big)]
        args.output = str(out_dir)
        args.compress = True
        args.compression_level = 4
        args.n_workers = 1
        args.min_file_size = 1000
        args.overwrite = False

        # Should not overwrite — existing file stays empty
        run_p2nc(args)
        assert existing.stat().st_size == 0


# ---------------------------------------------------------------------------
# merge module
# ---------------------------------------------------------------------------


class TestMerge:
    """Test merge subcommand logic."""

    def test_dry_run(self, tmp_path, capsys):
        from odas_tpw.pyturb.merge import run_merge

        # Create two small NetCDF files
        for i in range(2):
            ds = xr.Dataset({"T": (["TIME"], np.random.randn(10))})
            ds.to_netcdf(tmp_path / f"file{i}.nc")

        args = MagicMock()
        args.files = [str(tmp_path / "file0.nc"), str(tmp_path / "file1.nc")]
        args.output = str(tmp_path / "merged.nc")
        args.dry_run = True
        args.overwrite = False

        run_merge(args)
        captured = capsys.readouterr()
        assert "Would merge 2 files" in captured.out

    def test_actual_merge(self, tmp_path):
        from odas_tpw.pyturb.merge import run_merge

        for i in range(3):
            ds = xr.Dataset(
                {"T": (["TIME"], np.arange(10, dtype=float) + i * 10)},
                coords={"TIME": np.arange(10, dtype=float) + i * 10},
            )
            ds.to_netcdf(tmp_path / f"file{i}.nc")

        args = MagicMock()
        args.files = [str(tmp_path / f"file{i}.nc") for i in range(3)]
        args.output = str(tmp_path / "merged.nc")
        args.dry_run = False
        args.overwrite = True

        run_merge(args)

        merged = xr.open_dataset(tmp_path / "merged.nc")
        assert merged.sizes["TIME"] == 30
        merged.close()


# ---------------------------------------------------------------------------
# bin module
# ---------------------------------------------------------------------------


class TestBin:
    """Test bin subcommand logic."""

    def test_basic_binning(self, tmp_path):
        from odas_tpw.pyturb.bin import run_bin

        # Create fake profile output
        ds = xr.Dataset({
            "pressure": (["time"], np.linspace(5, 100, 50)),
            "eps_1": (["time"], np.full(50, 1e-8)),
            "eps_2": (["time"], np.full(50, 2e-8)),
            "W": (["time"], np.full(50, 0.7)),
            "temperature": (["time"], np.full(50, 15.0)),
        })
        ds.to_netcdf(tmp_path / "profile_p0000.nc")

        args = MagicMock()
        args.files = [str(tmp_path / "profile_p0000.nc")]
        args.output = str(tmp_path / "binned.nc")
        args.bin_width = 10.0
        args.dmin = 0.0
        args.dmax = 200.0
        args.lat = 45.0
        args.pressure = True  # Bin by pressure to avoid gsw dep in test
        args.vars = "eps_1,eps_2,W,temperature"
        args.n_workers = 1

        run_bin(args)

        result = xr.open_dataset(tmp_path / "binned.nc")
        assert "eps_1" in result
        assert "depth_bin" in result.dims
        result.close()


# ---------------------------------------------------------------------------
# Module imports and version
# ---------------------------------------------------------------------------


class TestModuleStructure:
    """Test that all modules import cleanly."""

    def test_version(self):
        from odas_tpw.pyturb import __version__

        assert __version__

    def test_import_compat(self):
        from odas_tpw.pyturb._compat import (
            check_overwrite,
            compute_window_parameters,
            load_auxiliary,
            rename_eps_dataset,
            seconds_to_samples,
        )
        assert callable(seconds_to_samples)
        assert callable(compute_window_parameters)
        assert callable(check_overwrite)
        assert callable(load_auxiliary)
        assert callable(rename_eps_dataset)

    def test_import_profind(self):
        from odas_tpw.pyturb._profind import find_profiles_peaks

        assert callable(find_profiles_peaks)

    def test_import_cli(self):
        from odas_tpw.pyturb.cli import main

        assert callable(main)

    def test_import_p2nc(self):
        from odas_tpw.pyturb.p2nc import run_p2nc

        assert callable(run_p2nc)

    def test_import_merge(self):
        from odas_tpw.pyturb.merge import run_merge

        assert callable(run_merge)

    def test_import_eps(self):
        from odas_tpw.pyturb.eps import run_eps

        assert callable(run_eps)

    def test_import_bin(self):
        from odas_tpw.pyturb.bin import run_bin

        assert callable(run_bin)


# ---------------------------------------------------------------------------
# Dispatch wrappers and main() entry paths
# ---------------------------------------------------------------------------


class TestDispatchWrappers:
    """Cover the thin wrapper functions that import + call the run_* helpers."""

    def test_run_p2nc_calls_module(self):
        from odas_tpw.pyturb.cli import _run_p2nc

        sentinel = MagicMock()
        with patch("odas_tpw.pyturb.p2nc.run_p2nc") as run:
            _run_p2nc(sentinel)
        run.assert_called_once_with(sentinel)

    def test_run_merge_calls_module(self):
        from odas_tpw.pyturb.cli import _run_merge

        sentinel = MagicMock()
        with patch("odas_tpw.pyturb.merge.run_merge") as run:
            _run_merge(sentinel)
        run.assert_called_once_with(sentinel)

    def test_run_bin_calls_module(self):
        from odas_tpw.pyturb.cli import _run_bin

        sentinel = MagicMock()
        with patch("odas_tpw.pyturb.bin.run_bin") as run:
            _run_bin(sentinel)
        run.assert_called_once_with(sentinel)

    def test_run_eps_calls_module(self):
        from odas_tpw.pyturb.cli import _run_eps

        args = MagicMock()
        args.aoa = None
        args.pitch_correction = False
        with patch("odas_tpw.pyturb.eps.run_eps") as run:
            _run_eps(args)
        run.assert_called_once_with(args)

    def test_run_eps_warns_on_aoa(self, caplog):
        import logging

        from odas_tpw.pyturb.cli import _run_eps

        args = MagicMock()
        args.aoa = 5.0
        args.pitch_correction = False
        with caplog.at_level(logging.WARNING, logger="pyturb"), \
             patch("odas_tpw.pyturb.eps.run_eps"):
            _run_eps(args)
        assert any("aoa" in r.message for r in caplog.records)

    def test_run_eps_warns_on_pitch_correction(self, caplog):
        import logging

        from odas_tpw.pyturb.cli import _run_eps

        args = MagicMock()
        args.aoa = None
        args.pitch_correction = True
        with caplog.at_level(logging.WARNING, logger="pyturb"), \
             patch("odas_tpw.pyturb.eps.run_eps"):
            _run_eps(args)
        assert any("pitch-correction" in r.message for r in caplog.records)


class TestMainEntryPaths:
    """Cover main() paths: verbose, no-command, dispatch."""

    def test_verbose_single_sets_info(self):
        """-v should set logging level to INFO via main()."""
        import logging

        from odas_tpw.pyturb.cli import main

        with patch("odas_tpw.pyturb.cli._run_eps") as run, \
             patch("logging.basicConfig") as basic:
            main(["-v", "eps", "test.nc", "-o", "/tmp/out"])
        run.assert_called_once()
        # basicConfig is invoked with level=INFO
        kwargs = basic.call_args.kwargs
        assert kwargs.get("level") == logging.INFO

    def test_verbose_double_sets_debug(self):
        import logging

        from odas_tpw.pyturb.cli import main

        with patch("odas_tpw.pyturb.cli._run_eps") as run, \
             patch("logging.basicConfig") as basic:
            main(["-vv", "eps", "test.nc", "-o", "/tmp/out"])
        run.assert_called_once()
        kwargs = basic.call_args.kwargs
        assert kwargs.get("level") == logging.DEBUG

    def test_no_command_prints_help_and_exits(self, capsys):
        """main() with no subcommand should print help and exit 1."""
        from odas_tpw.pyturb.cli import main

        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 1
        captured = capsys.readouterr()
        # argparse prints help to stdout
        assert "usage:" in captured.out.lower() or "pyturb-cli" in captured.out
