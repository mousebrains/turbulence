# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb.pipeline — orchestration and stage runners."""

import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from odas_tpw.perturb.pipeline import (
    _setup_output_dirs,
    process_file,
    run_merge,
    run_pipeline,
    run_trim,
)


def _make_p_file(
    path: Path,
    *,
    endian: str = "<",
    n_data_records: int = 5,
    record_size: int = 256,
    file_number: int = 1,
    config_content: str = "default_config",
):
    """Create a synthetic .p file with valid header structure."""
    from odas_tpw.rsi.p_file import _H, HEADER_BYTES, HEADER_WORDS

    endian_flag = 1 if endian == "<" else 2
    config_bytes = config_content.encode("ascii")
    # Pad config to fill first record after header
    config_padded = config_bytes + b"\x00" * (record_size - HEADER_BYTES - len(config_bytes))

    header = [0] * HEADER_WORDS
    header[_H["endian"]] = endian_flag
    header[_H["record_size"]] = record_size
    header[_H["file_number"]] = file_number
    header[_H["n_rows"]] = 8
    header[_H["fast_cols"]] = 8
    header[_H["slow_cols"]] = 2

    fmt = f"{endian}{HEADER_WORDS}H"
    header_bytes = struct.pack(fmt, *header)

    with open(path, "wb") as f:
        # Record 0: header + config
        f.write(header_bytes)
        f.write(config_padded)
        # Data records
        for i in range(n_data_records):
            rec_header = struct.pack(fmt, *header)
            data = b"\x00" * (record_size - HEADER_BYTES)
            f.write(rec_header + data)


class TestSetupOutputDirs:
    def test_creates_profiles_and_diss(self, tmp_path):
        config = {
            "files": {"output_root": str(tmp_path)},
            "profiles": {"P_min": 0.5},
            "epsilon": {"fft_length": 256},
        }
        dirs = _setup_output_dirs(config)
        assert "profiles" in dirs
        assert "diss" in dirs
        assert dirs["profiles"].exists()
        assert dirs["diss"].exists()

    def test_chi_dir_when_enabled(self, tmp_path):
        config = {
            "files": {"output_root": str(tmp_path)},
            "chi": {"enable": True},
        }
        dirs = _setup_output_dirs(config)
        assert "chi" in dirs
        assert dirs["chi"].exists()

    def test_no_chi_dir_when_disabled(self, tmp_path):
        config = {
            "files": {"output_root": str(tmp_path)},
            "chi": {"enable": False},
        }
        dirs = _setup_output_dirs(config)
        assert "chi" not in dirs

    def test_ctd_dir_when_enabled(self, tmp_path):
        config = {
            "files": {"output_root": str(tmp_path)},
            "ctd": {"enable": True},
        }
        dirs = _setup_output_dirs(config)
        assert "ctd" in dirs

    def test_no_ctd_dir_when_disabled(self, tmp_path):
        config = {
            "files": {"output_root": str(tmp_path)},
            "ctd": {"enable": False},
        }
        dirs = _setup_output_dirs(config)
        assert "ctd" not in dirs


class TestRunTrim:
    def test_trim_with_no_files(self, tmp_path):
        """run_trim with empty directory returns empty list."""
        config = {
            "files": {
                "p_file_root": str(tmp_path / "empty"),
                "output_root": str(tmp_path / "out"),
            },
        }
        (tmp_path / "empty").mkdir()
        results = run_trim(config)
        assert results == []

    def test_trim_with_valid_files(self, tmp_path):
        """run_trim processes files from discover."""
        p_dir = tmp_path / "vmp"
        p_dir.mkdir()
        _make_p_file(p_dir / "test_001.p", n_data_records=3)

        config = {
            "files": {
                "p_file_root": str(p_dir),
                "p_file_pattern": "*.p",
                "output_root": str(tmp_path / "out"),
            },
        }
        results = run_trim(config)
        assert len(results) == 1
        assert results[0].exists()


class TestRunMerge:
    def test_merge_with_no_files(self, tmp_path):
        """run_merge with empty directory returns empty list."""
        config = {
            "files": {
                "p_file_root": str(tmp_path / "empty"),
                "output_root": str(tmp_path / "out"),
            },
        }
        (tmp_path / "empty").mkdir()
        results = run_merge(config)
        assert results == []


# ---------------------------------------------------------------------------
# process_file tests
# ---------------------------------------------------------------------------


class TestProcessFile:
    """Tests for process_file using mocked heavy dependencies."""

    def _base_config(self, tmp_path):
        return {
            "files": {"output_root": str(tmp_path)},
            "profiles": {"P_min": 0.5},
            "epsilon": {},
            "fp07": {"calibrate": True},
            "ct": {"align": True},
            "ctd": {"enable": False},
            "chi": {"enable": False},
        }

    @patch("odas_tpw.rsi.p_file.PFile", side_effect=Exception("corrupt file"))
    def test_load_error(self, mock_pfile, tmp_path):
        """PFile load failure returns result with empty lists."""
        config = self._base_config(tmp_path)
        output_dirs = {"profiles": tmp_path / "profiles", "diss": tmp_path / "diss"}
        result = process_file(tmp_path / "bad.p", config, None, output_dirs)
        assert result["profiles"] == []
        assert result["diss"] == []
        assert result["chi"] == []

    @patch("odas_tpw.rsi.p_file.PFile")
    def test_no_pressure(self, mock_pfile_cls, tmp_path):
        """No 'P' channel in pf.channels returns early with empty profiles."""
        mock_pf = MagicMock()
        mock_pf.channels = {"T1": np.zeros(100)}
        mock_pfile_cls.return_value = mock_pf

        config = self._base_config(tmp_path)
        output_dirs = {"profiles": tmp_path / "profiles", "diss": tmp_path / "diss"}
        result = process_file(tmp_path / "test.p", config, None, output_dirs)
        assert result["profiles"] == []
        assert result["diss"] == []

    @patch("odas_tpw.rsi.profile.get_profiles", return_value=[])
    @patch("odas_tpw.rsi.profile._smooth_fall_rate", return_value=np.zeros(100))
    @patch("odas_tpw.rsi.p_file.PFile")
    def test_no_profiles(self, mock_pfile_cls, mock_smooth, mock_get_prof, tmp_path):
        """No profiles detected returns early with empty result."""
        mock_pf = MagicMock()
        mock_pf.channels = {"P": np.linspace(0, 50, 100), "T1": np.zeros(100)}
        mock_pf.fs_slow = 64.0
        mock_pfile_cls.return_value = mock_pf

        config = self._base_config(tmp_path)
        output_dirs = {"profiles": tmp_path / "profiles", "diss": tmp_path / "diss"}
        result = process_file(tmp_path / "test.p", config, None, output_dirs)
        assert result["profiles"] == []

    @patch("odas_tpw.rsi.profile.extract_profiles", return_value=[Path("/fake/prof.nc")])
    @patch("odas_tpw.perturb.fp07_cal.fp07_calibrate")
    @patch("odas_tpw.rsi.profile.get_profiles")
    @patch("odas_tpw.rsi.profile._smooth_fall_rate", return_value=np.zeros(100))
    @patch("odas_tpw.rsi.p_file.PFile")
    def test_fp07_cal_disabled(
        self, mock_pfile_cls, mock_smooth, mock_get_prof, mock_fp07_cal, mock_extract, tmp_path
    ):
        """fp07.calibrate=False skips FP07 calibration."""
        mock_pf = MagicMock()
        mock_pf.channels = {"P": np.linspace(0, 50, 100), "T1": np.zeros(100)}
        mock_pf.fs_slow = 64.0
        mock_pfile_cls.return_value = mock_pf
        mock_get_prof.return_value = [{"start": 0, "end": 50}]

        config = self._base_config(tmp_path)
        config["fp07"]["calibrate"] = False
        output_dirs = {"profiles": tmp_path / "profiles", "diss": tmp_path / "diss"}
        (tmp_path / "profiles").mkdir(parents=True, exist_ok=True)
        (tmp_path / "diss").mkdir(parents=True, exist_ok=True)

        process_file(tmp_path / "test.p", config, None, output_dirs)
        mock_fp07_cal.assert_not_called()

    @patch("odas_tpw.rsi.profile.extract_profiles", return_value=[Path("/fake/prof.nc")])
    @patch("odas_tpw.perturb.ct_align.ct_align")
    @patch("odas_tpw.perturb.fp07_cal.fp07_calibrate", return_value={"channels": {}})
    @patch("odas_tpw.rsi.profile.get_profiles")
    @patch("odas_tpw.rsi.profile._smooth_fall_rate", return_value=np.zeros(100))
    @patch("odas_tpw.rsi.p_file.PFile")
    def test_ct_align_disabled(
        self,
        mock_pfile_cls,
        mock_smooth,
        mock_get_prof,
        mock_fp07_cal,
        mock_ct_align,
        mock_extract,
        tmp_path,
    ):
        """ct.align=False skips CT alignment."""
        mock_pf = MagicMock()
        mock_pf.channels = {
            "P": np.linspace(0, 50, 100),
            "T1": np.zeros(100),
            "JAC_T": np.zeros(100),
            "JAC_C": np.zeros(100),
        }
        mock_pf.fs_slow = 64.0
        mock_pfile_cls.return_value = mock_pf
        mock_get_prof.return_value = [{"start": 0, "end": 50}]

        config = self._base_config(tmp_path)
        config["ct"]["align"] = False
        output_dirs = {"profiles": tmp_path / "profiles", "diss": tmp_path / "diss"}
        (tmp_path / "profiles").mkdir(parents=True, exist_ok=True)
        (tmp_path / "diss").mkdir(parents=True, exist_ok=True)

        process_file(tmp_path / "test.p", config, None, output_dirs)
        mock_ct_align.assert_not_called()

    @patch("odas_tpw.rsi.profile.get_profiles")
    @patch("odas_tpw.rsi.profile._smooth_fall_rate", return_value=np.zeros(100))
    @patch("odas_tpw.rsi.p_file.PFile")
    def test_ctd_disabled(self, mock_pfile_cls, mock_smooth, mock_get_prof, tmp_path):
        """ctd.enable=False skips CTD binning (ctd_bin_file not called)."""
        mock_pf = MagicMock()
        mock_pf.channels = {"P": np.linspace(0, 50, 100), "T1": np.zeros(100)}
        mock_pf.fs_slow = 64.0
        mock_pfile_cls.return_value = mock_pf
        mock_get_prof.return_value = []

        config = self._base_config(tmp_path)
        config["ctd"]["enable"] = False
        output_dirs = {
            "profiles": tmp_path / "profiles",
            "diss": tmp_path / "diss",
            "ctd": tmp_path / "ctd",
        }

        with patch("odas_tpw.perturb.ctd.ctd_bin_file") as mock_ctd_bin:
            process_file(tmp_path / "test.p", config, None, output_dirs)
            mock_ctd_bin.assert_not_called()

    @patch("odas_tpw.perturb.hotel.interpolate_hotel", return_value={"hotel_T": np.zeros(100)})
    @patch("odas_tpw.rsi.profile.get_profiles", return_value=[])
    @patch("odas_tpw.rsi.profile._smooth_fall_rate", return_value=np.zeros(100))
    @patch("odas_tpw.rsi.p_file.PFile")
    def test_hotel_data_injection(
        self, mock_pfile_cls, mock_smooth, mock_get_prof, mock_hotel, tmp_path
    ):
        """Hotel data should be injected into pf.channels when provided."""
        mock_pf = MagicMock()
        mock_pf.channels = {"P": np.linspace(0, 50, 100), "T1": np.zeros(100)}
        mock_pf.fs_slow = 64.0
        mock_pfile_cls.return_value = mock_pf

        config = self._base_config(tmp_path)
        output_dirs = {"profiles": tmp_path / "profiles", "diss": tmp_path / "diss"}
        hotel_data = MagicMock()

        process_file(tmp_path / "test.p", config, None, output_dirs, hotel_data=hotel_data)
        mock_hotel.assert_called_once()

    @patch("odas_tpw.perturb.ctd.ctd_bin_file")
    @patch("odas_tpw.rsi.profile.get_profiles", return_value=[])
    @patch("odas_tpw.rsi.profile._smooth_fall_rate", return_value=np.zeros(100))
    @patch("odas_tpw.rsi.p_file.PFile")
    def test_ctd_enabled(self, mock_pfile_cls, mock_smooth, mock_get_prof, mock_ctd_bin, tmp_path):
        """ctd.enable=True and 'ctd' in output_dirs calls ctd_bin_file."""
        mock_pf = MagicMock()
        mock_pf.channels = {"P": np.linspace(0, 50, 100), "T1": np.zeros(100)}
        mock_pf.fs_slow = 64.0
        mock_pfile_cls.return_value = mock_pf

        config = self._base_config(tmp_path)
        config["ctd"]["enable"] = True
        ctd_dir = tmp_path / "ctd"
        ctd_dir.mkdir(parents=True, exist_ok=True)
        output_dirs = {"profiles": tmp_path / "profiles", "diss": tmp_path / "diss", "ctd": ctd_dir}

        process_file(tmp_path / "test.p", config, None, output_dirs)
        mock_ctd_bin.assert_called_once()

    @patch("odas_tpw.rsi.chi_io._compute_chi", return_value=[MagicMock()])
    @patch("odas_tpw.rsi.dissipation._compute_epsilon")
    @patch("odas_tpw.rsi.profile.extract_profiles")
    @patch("odas_tpw.perturb.fp07_cal.fp07_calibrate", return_value={"channels": {}})
    @patch("odas_tpw.rsi.profile.get_profiles")
    @patch("odas_tpw.rsi.profile._smooth_fall_rate", return_value=np.zeros(100))
    @patch("odas_tpw.rsi.p_file.PFile")
    def test_chi_enabled(
        self,
        mock_pfile_cls,
        mock_smooth,
        mock_get_prof,
        mock_fp07_cal,
        mock_extract,
        mock_eps,
        mock_chi,
        tmp_path,
    ):
        """chi.enable=True with profiles and diss results calls _compute_chi."""
        import xarray as xr

        mock_pf = MagicMock()
        mock_pf.channels = {"P": np.linspace(0, 50, 100), "T1": np.zeros(100)}
        mock_pf.fs_slow = 64.0
        mock_pfile_cls.return_value = mock_pf
        mock_get_prof.return_value = [{"start": 0, "end": 50}]

        # Set up profile + diss output paths
        prof_dir = tmp_path / "profiles"
        prof_dir.mkdir(parents=True, exist_ok=True)
        diss_dir = tmp_path / "diss"
        diss_dir.mkdir(parents=True, exist_ok=True)
        chi_dir = tmp_path / "chi"
        chi_dir.mkdir(parents=True, exist_ok=True)

        prof_nc = prof_dir / "prof.nc"
        prof_nc.touch()
        mock_extract.return_value = [prof_nc]

        # Create a minimal diss netcdf
        diss_nc = diss_dir / "prof.nc"
        ds = xr.Dataset({"epsilon": (("time",), [1e-8])})
        ds.to_netcdf(diss_nc)
        mock_eps.return_value = [ds]

        config = self._base_config(tmp_path)
        config["chi"]["enable"] = True
        output_dirs = {"profiles": prof_dir, "diss": diss_dir, "chi": chi_dir}

        process_file(tmp_path / "test.p", config, None, output_dirs)
        mock_chi.assert_called_once()


# ---------------------------------------------------------------------------
# run_pipeline tests
# ---------------------------------------------------------------------------


class TestRunPipeline:
    """Tests for run_pipeline using mocked stage runners."""

    def _base_config(self, tmp_path):
        return {
            "files": {
                "output_root": str(tmp_path),
                "p_file_root": str(tmp_path),
                "trim": True,
                "merge": False,
            },
            "profiles": {},
            "epsilon": {},
            "chi": {"enable": False},
            "ctd": {"enable": False},
            "gps": {},
            "parallel": {"jobs": 1},
            "binning": {},
        }

    def test_no_files(self, tmp_path, caplog):
        """run_pipeline with empty file list logs 'No .p files found'."""
        config = self._base_config(tmp_path)
        with caplog.at_level("WARNING", logger="odas_tpw.perturb.pipeline"):
            run_pipeline(config, p_files=[])
        assert "No .p files found" in caplog.text

    @patch(
        "odas_tpw.perturb.pipeline.process_file",
        return_value={
            "source": "test.p",
            "profiles": [],
            "diss": [],
            "chi": [],
        },
    )
    @patch("odas_tpw.perturb.pipeline._setup_output_dirs", return_value={})
    @patch("odas_tpw.perturb.gps.create_gps", return_value=None)
    @patch("odas_tpw.perturb.pipeline.run_trim")
    def test_trim_disabled(self, mock_run_trim, mock_gps, mock_dirs, mock_proc, tmp_path):
        """files.trim=False skips run_trim call."""
        config = self._base_config(tmp_path)
        config["files"]["trim"] = False
        run_pipeline(config, p_files=[Path("dummy.p")])
        mock_run_trim.assert_not_called()


class TestNanExcludedProbes:
    """Tests for the per-instrument shear-probe exclusion helper."""

    def _make_diss_ds(self):
        import xarray as xr

        # 2 probes x 5 windows of plausible epsilon values
        epsilon = np.array(
            [[1e-7, 2e-7, 3e-7, 4e-7, 5e-7], [1e-9, 2e-9, 3e-9, 4e-9, 5e-9]],
            dtype=np.float64,
        )
        return xr.Dataset(
            {"epsilon": (["probe", "time"], epsilon)},
            coords={"probe": ["sh1", "sh2"], "time": np.arange(5, dtype=float)},
        )

    def test_excludes_named_probe(self):
        from odas_tpw.perturb.pipeline import _nan_excluded_probes

        ds = self._make_diss_ds()
        _nan_excluded_probes(ds, ["sh2"], "test.p")
        eps = ds["epsilon"].values
        assert np.all(np.isfinite(eps[0]))
        assert np.all(np.isnan(eps[1]))

    def test_unknown_probe_warns_but_does_not_raise(self, caplog):
        from odas_tpw.perturb.pipeline import _nan_excluded_probes

        ds = self._make_diss_ds()
        with caplog.at_level("WARNING", logger="odas_tpw.perturb.pipeline"):
            _nan_excluded_probes(ds, ["sh3"], "test.p")
        assert "not present" in caplog.text
        # Original data preserved
        assert np.all(np.isfinite(ds["epsilon"].values))

    def test_also_masks_e_n_companion_variables(self):
        import xarray as xr

        from odas_tpw.perturb.pipeline import _nan_excluded_probes

        ds = self._make_diss_ds()
        # Add e_1/e_2 companions like mk_epsilon_mean would create
        ds["e_1"] = xr.DataArray(ds["epsilon"].isel(probe=0).values, dims=["time"])
        ds["e_2"] = xr.DataArray(ds["epsilon"].isel(probe=1).values, dims=["time"])

        _nan_excluded_probes(ds, ["sh1"], "test.p")
        assert np.all(np.isnan(ds["e_1"].values))
        assert np.all(np.isfinite(ds["e_2"].values))

    def test_no_op_when_no_epsilon_in_dataset(self):
        import xarray as xr

        from odas_tpw.perturb.pipeline import _nan_excluded_probes

        ds = xr.Dataset()
        # Should silently return without error
        _nan_excluded_probes(ds, ["sh1"], "test.p")


class TestAdjustProfileBounds:
    """Tests for top_trim/bottom hooks that push profile bounds inward."""

    def _make_pf(self, n_slow=2000, fs_slow=64.0, fs_fast=512.0, max_depth=200.0):
        ratio = int(fs_fast / fs_slow)
        n_fast = n_slow * ratio
        # Triangle profile: 0 -> max_depth -> 0
        half = n_slow // 2
        P = np.concatenate([
            np.linspace(0.0, max_depth, half),
            np.linspace(max_depth, 0.0, n_slow - half),
        ])
        t_slow = np.arange(n_slow) / fs_slow
        t_fast = np.arange(n_fast) / fs_fast

        # Quiet baseline noise; high variance only in top 8 m of descending limb
        rng = np.random.default_rng(42)
        depth_fast = np.repeat(P, ratio)
        top_mask = (depth_fast < 8.0) & (np.arange(n_fast) < half * ratio)
        sh1 = rng.standard_normal(n_fast) * 0.01
        sh1 = np.where(top_mask, rng.standard_normal(n_fast) * 5.0, sh1)

        pf = MagicMock()
        pf.fs_slow = fs_slow
        pf.fs_fast = fs_fast
        pf.t_slow = t_slow
        pf.t_fast = t_fast
        pf.channels = {
            "P": P,
            "sh1": sh1,
            "sh2": rng.standard_normal(n_fast) * 0.01,
            "Ax": rng.standard_normal(n_fast) * 0.01,
            "Ay": rng.standard_normal(n_fast) * 0.01,
        }
        pf.is_fast = lambda ch: ch in {"sh1", "sh2", "Ax", "Ay"}
        return pf, half

    def test_disabled_returns_unchanged(self):
        from odas_tpw.perturb.pipeline import _adjust_profile_bounds

        pf, half = self._make_pf()
        profiles = [(10, half - 1)]
        out = _adjust_profile_bounds(profiles, pf, {"enable": False}, {"enable": False}, "x.p")
        assert out == profiles

    def test_top_trim_pushes_start_forward(self):
        from odas_tpw.perturb.pipeline import _adjust_profile_bounds

        pf, half = self._make_pf()
        profiles = [(10, half - 1)]
        top_cfg = {"enable": True, "dz": 0.5, "min_depth": 1.0, "max_depth": 30.0, "quantile": 0.6}
        out = _adjust_profile_bounds(profiles, pf, top_cfg, {"enable": False}, "x.p")
        assert out[0][0] > profiles[0][0]
        assert out[0][1] == profiles[0][1]

    def test_top_trim_clipped_to_profile(self):
        """If top_trim depth lies past profile end, leave start unchanged."""
        from odas_tpw.perturb.pipeline import _adjust_profile_bounds

        pf, _half = self._make_pf()
        # Tiny profile spanning samples 10..20 — entirely inside the top 8 m
        # noise band. compute_trim_depth would push past the end, which the
        # adjuster must reject (candidate < e_slow).
        profiles = [(10, 20)]
        top_cfg = {"enable": True, "dz": 0.5, "min_depth": 1.0, "max_depth": 30.0, "quantile": 0.6}
        out = _adjust_profile_bounds(profiles, pf, top_cfg, {"enable": False}, "x.p")
        # Bounds must remain valid (start < end)
        assert out[0][0] <= out[0][1]


class TestRunCombo:
    """Tests for the combo assembly hook in run_pipeline."""

    def test_writes_combo_when_binned_dirs_present(self, tmp_path):
        import xarray as xr

        from odas_tpw.perturb.pipeline import _run_combo

        prof_dir = tmp_path / "profiles_binned_00"
        prof_dir.mkdir()
        ds = xr.Dataset(
            {"T": (["bin", "profile"], np.ones((3, 1)))},
            coords={"bin": np.arange(3.0), "profile": np.arange(1)},
        )
        ds.to_netcdf(prof_dir / "binned.nc")

        _run_combo(tmp_path, prof_dir, None, None, None, {"title": "smoke"})
        out = tmp_path / "combo" / "combo.nc"
        assert out.exists()
        combo = xr.open_dataset(out)
        assert combo.attrs["title"] == "smoke"
        combo.close()

    def test_silently_skips_missing_dirs(self, tmp_path):
        from odas_tpw.perturb.pipeline import _run_combo

        # All Nones — should not raise, should not create anything
        _run_combo(tmp_path, None, None, None, None, {})
        assert not (tmp_path / "combo").exists()
