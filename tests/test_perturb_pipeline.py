# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb.pipeline — orchestration and stage runners."""

import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from odas_tpw.perturb.pipeline import (
    _setup_output_dirs,
    _upstream_for,
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


class TestUpstreamFor:
    """The .params_sha256_<hash> signature on each output dir must hash
    the stage's own params plus every ancestor's params, so a change to a
    deep upstream knob (e.g. profiles.P_min) propagates to a new chi/binned
    hash and a fresh output dir.  This locks in the chains."""

    def _config(self):
        return {
            "profiles": {"P_min": 0.5},
            "epsilon": {"fft_length_sec": 0.5},
            "chi": {"enable": True},
            "ctd": {"enable": True},
            "binning": {"width": 1.0},
        }

    def _sections(self, stage):
        return [name for name, _ in _upstream_for(stage, self._config())]

    def test_profiles_has_no_upstream(self):
        assert self._sections("profiles") == []

    def test_diss_includes_profiles(self):
        assert self._sections("diss") == ["profiles"]

    def test_chi_includes_epsilon_and_profiles(self):
        # chi -> diss -> profiles
        assert set(self._sections("chi")) == {"epsilon", "profiles"}

    def test_ctd_has_no_upstream(self):
        assert self._sections("ctd") == []

    def test_profiles_binned_includes_profiles(self):
        assert self._sections("profiles_binned") == ["profiles"]

    def test_diss_binned_includes_epsilon_and_profiles(self):
        # was missing profiles before _upstream_for refactor
        assert set(self._sections("diss_binned")) == {"epsilon", "profiles"}

    def test_chi_binned_includes_full_chain(self):
        # was missing epsilon and profiles before _upstream_for refactor
        assert set(self._sections("chi_binned")) == {"chi", "epsilon", "profiles"}

    def test_combo_includes_profiles(self):
        assert self._sections("combo") == ["profiles"]

    def test_diss_combo_includes_epsilon_and_profiles(self):
        assert set(self._sections("diss_combo")) == {"epsilon", "profiles"}

    def test_chi_combo_includes_full_chain(self):
        assert set(self._sections("chi_combo")) == {"chi", "epsilon", "profiles"}

    def test_ctd_combo_includes_ctd(self):
        assert self._sections("ctd_combo") == ["ctd"]


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

    @patch("odas_tpw.rsi.profile.extract_profiles", return_value=([Path("/fake/prof.nc")], [{}]))
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

    @patch("odas_tpw.rsi.profile.extract_profiles", return_value=([Path("/fake/prof.nc")], [{}]))
    @patch("odas_tpw.processing.ct_align.ct_align")
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
        mock_extract.return_value = ([prof_nc], [{}])

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


class TestCopyProfileScalars:
    """Tests for the per-profile scalar copy helper."""

    def test_copies_present_scalars(self, tmp_path):
        import xarray as xr

        from odas_tpw.perturb.pipeline import _copy_profile_scalars

        prof_path = tmp_path / "src.nc"
        src = xr.Dataset(
            {
                "lat": ((), 7.0),
                "lon": ((), 134.0),
                "stime": ((), 1767225600.0),
                "etime": ((), 1767225700.0),
                "ignored_array": (["t"], np.zeros(3)),
            }
        )
        src.to_netcdf(prof_path)

        target = xr.Dataset({"epsilon": (["t"], np.zeros(3))})
        _copy_profile_scalars(prof_path, target)
        for sname in ("lat", "lon", "stime", "etime"):
            assert sname in target.data_vars
            assert target[sname].ndim == 0
        # 1-D arrays must not be picked up as scalars.
        assert "ignored_array" not in target.data_vars

    def test_silently_swallows_missing_file(self, tmp_path):
        """Older runs / external NCs without scalars must not raise."""
        import xarray as xr

        from odas_tpw.perturb.pipeline import _copy_profile_scalars

        target = xr.Dataset()
        _copy_profile_scalars(tmp_path / "does_not_exist.nc", target)
        assert len(target.data_vars) == 0

    def test_skips_non_scalar_named_vars(self, tmp_path):
        """A 1-D ``lat`` (e.g. fast-rate channel snapshot) must not be
        treated as the per-profile scalar."""
        import xarray as xr

        from odas_tpw.perturb.pipeline import _copy_profile_scalars

        prof_path = tmp_path / "src.nc"
        xr.Dataset({"lat": (["t"], np.array([1.0, 2.0, 3.0]))}).to_netcdf(prof_path)

        target = xr.Dataset()
        _copy_profile_scalars(prof_path, target)
        assert "lat" not in target.data_vars


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

    def test_no_pressure_returns_unchanged(self):
        """Missing 'P' channel short-circuits the adjuster."""
        from odas_tpw.perturb.pipeline import _adjust_profile_bounds

        pf = MagicMock()
        pf.fs_slow = 64.0
        pf.fs_fast = 512.0
        pf.channels = {}  # no P
        profiles = [(0, 100)]
        out = _adjust_profile_bounds(profiles, pf, {"enable": True}, {"enable": True}, "x.p")
        assert out == profiles

    def test_p_fast_length_mismatch_repeated(self):
        """P_fast longer than t_fast is truncated; shorter is padded."""
        from odas_tpw.perturb.pipeline import _adjust_profile_bounds

        # Build a pf where P_fast (via repeat) overshoots t_fast
        rng = np.random.default_rng(7)
        n_slow = 200
        ratio = 8
        # t_fast intentionally shorter than n_slow * ratio so the repeat path
        # has to truncate to len(t_fast).
        t_slow = np.arange(n_slow) / 64.0
        t_fast = np.arange(n_slow * ratio - 5) / 512.0
        n_fast = len(t_fast)
        pf = MagicMock()
        pf.fs_slow = 64.0
        pf.fs_fast = 512.0
        pf.t_slow = t_slow
        pf.t_fast = t_fast
        pf.channels = {
            "P": np.linspace(0.0, 50.0, n_slow),
            "sh1": rng.standard_normal(n_fast) * 0.01,
        }
        pf.is_fast = lambda ch: ch == "sh1"

        # Don't care about top_trim succeeding — just that the length-mismatch
        # branch executes without raising.
        out = _adjust_profile_bounds(
            [(10, 100)],
            pf,
            {"enable": True, "dz": 0.5, "min_depth": 1.0, "max_depth": 30.0, "quantile": 0.6},
            {"enable": False},
            "x.p",
        )
        assert len(out) == 1

    def test_p_fast_shorter_than_t_fast_padded(self):
        """P_fast shorter than t_fast hits the concatenate-pad branch."""
        from odas_tpw.perturb.pipeline import _adjust_profile_bounds

        n_slow = 100
        ratio = 8
        t_fast = np.arange(n_slow * ratio + 7) / 512.0
        n_fast = len(t_fast)
        pf = MagicMock()
        pf.fs_slow = 64.0
        pf.fs_fast = 512.0
        pf.t_slow = np.arange(n_slow) / 64.0
        pf.t_fast = t_fast
        pf.channels = {
            "P": np.linspace(0.0, 50.0, n_slow),
            "sh1": np.zeros(n_fast),
        }
        pf.is_fast = lambda ch: ch == "sh1"
        out = _adjust_profile_bounds(
            [(0, n_slow - 1)],
            pf,
            {"enable": True},
            {"enable": False},
            "x.p",
        )
        assert len(out) == 1

    def test_top_trim_exception_logged_and_swallowed(self, caplog):
        from odas_tpw.perturb.pipeline import _adjust_profile_bounds

        pf, _half = self._make_pf()
        # Pass an invalid kwarg into top_trim_cfg; compute_trim_depth raises
        # TypeError, which the adjuster catches and logs.
        bad_cfg = {"enable": True, "this_is_not_a_real_param": True}
        with caplog.at_level("WARNING", logger="odas_tpw.perturb.pipeline"):
            out = _adjust_profile_bounds([(10, 100)], pf, bad_cfg, {"enable": False}, "x.p")
        assert out[0] == (10, 100)
        assert "top_trim failed" in caplog.text

    def test_bottom_pushes_end_backward(self):
        """Synthetic crash spike near the bottom pulls e_slow back."""
        from odas_tpw.perturb.pipeline import _adjust_profile_bounds

        rng = np.random.default_rng(0)
        n_slow = 2000
        ratio = 8
        n_fast = n_slow * ratio
        # Monotonic-down profile so the slow-index→pressure mapping is simple
        P = np.linspace(0.0, 200.0, n_slow)
        t_slow = np.arange(n_slow) / 64.0
        t_fast = np.arange(n_fast) / 512.0

        # Quiet accel everywhere except a 50-sample spike at ~95% depth
        Ax = rng.standard_normal(n_fast) * 0.01
        Ay = rng.standard_normal(n_fast) * 0.01
        crash_idx = int(0.95 * n_fast)
        Ax[crash_idx - 50 : crash_idx + 50] = 10.0
        Ay[crash_idx - 50 : crash_idx + 50] = 10.0

        pf = MagicMock()
        pf.fs_slow = 64.0
        pf.fs_fast = 512.0
        pf.t_slow = t_slow
        pf.t_fast = t_fast
        pf.channels = {"P": P, "Ax": Ax, "Ay": Ay}
        pf.is_fast = lambda ch: ch in {"Ax", "Ay"}

        bottom_cfg = {"enable": True, "vibration_factor": 3.0, "depth_minimum": 10.0}
        profiles = [(0, n_slow - 1)]
        out = _adjust_profile_bounds(profiles, pf, {"enable": False}, bottom_cfg, "x.p")
        assert out[0][0] == 0
        assert out[0][1] < n_slow - 1

    def test_bottom_no_op_when_accel_not_fast(self):
        """If Ax/Ay aren't fast channels, bottom path is skipped silently."""
        from odas_tpw.perturb.pipeline import _adjust_profile_bounds

        n_slow = 200
        ratio = 8
        pf = MagicMock()
        pf.fs_slow = 64.0
        pf.fs_fast = 512.0
        pf.t_slow = np.arange(n_slow) / 64.0
        pf.t_fast = np.arange(n_slow * ratio) / 512.0
        pf.channels = {
            "P": np.linspace(0.0, 50.0, n_slow),
            # Ax/Ay present but flagged as slow
            "Ax": np.zeros(n_slow),
            "Ay": np.zeros(n_slow),
        }
        pf.is_fast = lambda ch: False
        out = _adjust_profile_bounds(
            [(0, n_slow - 1)], pf, {"enable": False}, {"enable": True}, "x.p"
        )
        assert out == [(0, n_slow - 1)]

    def test_bottom_exception_logged_and_swallowed(self, caplog):
        from odas_tpw.perturb.pipeline import _adjust_profile_bounds

        n_slow = 200
        ratio = 8
        n_fast = n_slow * ratio
        pf = MagicMock()
        pf.fs_slow = 64.0
        pf.fs_fast = 512.0
        pf.t_slow = np.arange(n_slow) / 64.0
        pf.t_fast = np.arange(n_fast) / 512.0
        pf.channels = {
            "P": np.linspace(0.0, 50.0, n_slow),
            "Ax": np.zeros(n_fast),
            "Ay": np.zeros(n_fast),
        }
        pf.is_fast = lambda ch: ch in {"Ax", "Ay"}
        bad_cfg = {"enable": True, "this_is_not_a_real_param": True}
        with caplog.at_level("WARNING", logger="odas_tpw.perturb.pipeline"):
            out = _adjust_profile_bounds([(0, n_slow - 1)], pf, {"enable": False}, bad_cfg, "x.p")
        assert out == [(0, n_slow - 1)]
        assert "bottom failed" in caplog.text


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

    def test_ctd_combo_written(self, tmp_path):
        """ctd_dir populated => ctd_combo/combo.nc is produced."""
        import xarray as xr

        from odas_tpw.perturb.pipeline import _run_combo

        ctd_dir = tmp_path / "ctd_00"
        ctd_dir.mkdir()
        ds = xr.Dataset(
            {"T": (["time"], np.arange(5.0))},
            coords={"time": np.arange(5.0)},
        )
        ds.to_netcdf(ctd_dir / "file.nc")

        _run_combo(tmp_path, None, None, None, ctd_dir, {})
        assert (tmp_path / "ctd_combo" / "combo.nc").exists()

    def test_combo_exception_logged_and_swallowed(self, tmp_path, caplog):
        """A make_combo failure is logged but does not propagate."""
        from odas_tpw.perturb.pipeline import _run_combo

        prof_dir = tmp_path / "profiles_binned_00"
        prof_dir.mkdir()
        # Put a non-NetCDF file in there to make xr.open_dataset raise.
        (prof_dir / "broken.nc").write_bytes(b"this is not a netcdf file")
        with caplog.at_level("ERROR", logger="odas_tpw.perturb.pipeline"):
            _run_combo(tmp_path, prof_dir, None, None, None, {})
        assert "combo combo" in caplog.text or "combo" in caplog.text

    def test_ctd_combo_exception_logged_and_swallowed(self, tmp_path, caplog):
        """A make_ctd_combo failure is logged but does not propagate."""
        from odas_tpw.perturb.pipeline import _run_combo

        ctd_dir = tmp_path / "ctd_00"
        ctd_dir.mkdir()
        (ctd_dir / "broken.nc").write_bytes(b"this is not a netcdf file")
        with caplog.at_level("ERROR", logger="odas_tpw.perturb.pipeline"):
            _run_combo(tmp_path, None, None, None, ctd_dir, {})
        assert "ctd combo" in caplog.text

    def test_combo_writes_signature_when_config_passed(self, tmp_path):
        """When _run_combo is given *config*, each produced combo dir gets a
        ``.params_sha256_<hash>`` file capturing its full upstream chain."""
        import xarray as xr

        from odas_tpw.perturb.pipeline import _run_combo

        prof_dir = tmp_path / "profiles_binned_00"
        prof_dir.mkdir()
        ds = xr.Dataset(
            {"T": (["bin", "profile"], np.ones((3, 1)))},
            coords={"bin": np.arange(3.0), "profile": np.arange(1)},
        )
        ds.to_netcdf(prof_dir / "binned.nc")

        _run_combo(tmp_path, prof_dir, None, None, None, {}, config={"profiles": {"P_min": 0.5}})
        sigs = list((tmp_path / "combo").glob(".params_sha256_*"))
        assert len(sigs) == 1, "combo dir should carry one signature file"

    def test_combo_signature_changes_with_upstream(self, tmp_path):
        """Two configs that differ only in *profiles* must produce different
        signatures on the combo dir — proves the upstream chain is hashed."""
        import xarray as xr

        from odas_tpw.perturb.pipeline import _run_combo

        def _build(out_root):
            prof_dir = out_root / "profiles_binned_00"
            prof_dir.mkdir()
            ds = xr.Dataset(
                {"T": (["bin", "profile"], np.ones((3, 1)))},
                coords={"bin": np.arange(3.0), "profile": np.arange(1)},
            )
            ds.to_netcdf(prof_dir / "binned.nc")
            return prof_dir

        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        prof_a = _build(a)
        prof_b = _build(b)

        _run_combo(a, prof_a, None, None, None, {}, config={"profiles": {"P_min": 0.5}})
        _run_combo(b, prof_b, None, None, None, {}, config={"profiles": {"P_min": 1.5}})

        hash_a = next((a / "combo").glob(".params_sha256_*")).name
        hash_b = next((b / "combo").glob(".params_sha256_*")).name
        assert hash_a != hash_b

    def test_ctd_combo_writes_signature_when_config_passed(self, tmp_path):
        """The ctd_combo branch of _run_combo writes a signature when
        *config* is supplied — covers the second write_signature call."""
        import xarray as xr

        from odas_tpw.perturb.pipeline import _run_combo

        ctd_dir = tmp_path / "ctd_00"
        ctd_dir.mkdir()
        ds = xr.Dataset(
            {"T": (["time"], np.arange(5.0))},
            coords={"time": np.arange(5.0)},
        )
        ds.to_netcdf(ctd_dir / "file.nc")

        _run_combo(tmp_path, None, None, None, ctd_dir, {}, config={"ctd": {"bin_width": 0.5}})
        sigs = list((tmp_path / "ctd_combo").glob(".params_sha256_*"))
        assert len(sigs) == 1, "ctd_combo dir should carry one signature file"

    def test_time_binned_combo_uses_lengthwise_glue(self, tmp_path):
        """When *bin_method* is ``time``, the diss/chi/profile combos must
        concatenate lengthwise (along the time dimension) — the layout used
        for moored or horizontally-flying instruments where every per-file
        binned NC is a time-series, not a (bin x profile) grid.
        """
        import xarray as xr

        from odas_tpw.perturb.pipeline import _run_combo

        prof_dir = tmp_path / "profiles_binned_00"
        prof_dir.mkdir()
        # Two per-file time-binned NetCDFs whose time vectors must concat.
        for i, t0 in enumerate([0.0, 10.0]):
            ds = xr.Dataset(
                {"T": (["time"], np.arange(5.0) + i)},
                coords={"time": t0 + np.arange(5.0)},
            )
            ds.to_netcdf(prof_dir / f"f{i}.nc")

        _run_combo(tmp_path, prof_dir, None, None, None, {}, bin_method="time")
        out = tmp_path / "combo" / "combo.nc"
        assert out.exists()
        combo = xr.open_dataset(out)
        # 10 time bins (5 per file, two files), monotonic, with the
        # CF trajectory featureType selected by make_combo.
        assert combo.sizes["time"] == 10
        assert combo.attrs["featureType"] == "trajectory"
        combo.close()

    def test_time_binned_combo_unions_sensor_sets(self, tmp_path):
        """For the moored / scalar case, two files with different sensor
        sets must merge into a wider combo — DO from file 1, Chl from
        file 2, both NaN-filled where they didn't exist."""
        import xarray as xr

        from odas_tpw.perturb.pipeline import _run_combo

        prof_dir = tmp_path / "profiles_binned_00"
        prof_dir.mkdir()
        ds1 = xr.Dataset(
            {"T": (["time"], np.array([1.0, 2.0])), "DO": (["time"], np.array([5.0, 6.0]))},
            coords={"time": np.array([0.0, 1.0])},
        )
        ds2 = xr.Dataset(
            {"T": (["time"], np.array([3.0, 4.0])), "Chl": (["time"], np.array([7.0, 8.0]))},
            coords={"time": np.array([2.0, 3.0])},
        )
        ds1.to_netcdf(prof_dir / "a.nc")
        ds2.to_netcdf(prof_dir / "b.nc")

        _run_combo(tmp_path, prof_dir, None, None, None, {}, bin_method="time")
        combo = xr.open_dataset(tmp_path / "combo" / "combo.nc")
        # Outer join: union of T, DO, Chl across files; missing-by-file
        # samples come back NaN.
        for v in ("T", "DO", "Chl"):
            assert v in combo.data_vars
        assert np.isnan(combo["DO"].values[2:]).all()
        assert np.isnan(combo["Chl"].values[:2]).all()
        combo.close()
