# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb.pipeline — orchestration and stage runners."""

import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from odas_tpw.perturb.pipeline import (
    _inject_seawater_properties,
    _profile_practical_salinity,
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


class TestProfilePracticalSalinity:
    """chi.salinity='measured' resolves to TEOS-10 salinity from C/T/P."""

    def test_derives_salinity_from_ctp(self, tmp_path):
        import gsw
        import xarray as xr

        P = np.array([1.0, 5.0, 10.0])
        T = np.array([20.0, 19.5, 19.0])
        C = np.array([45.0, 44.8, 44.5])
        path = tmp_path / "prof.nc"
        xr.Dataset(
            {
                "P": ("time_slow", P),
                "JAC_T": ("time_slow", T),
                "JAC_C": ("time_slow", C),
            }
        ).to_netcdf(path)

        sal = _profile_practical_salinity(path, "JAC_T", "JAC_C")
        np.testing.assert_allclose(sal, gsw.SP_from_C(C, T, P))

    def test_returns_none_when_conductivity_missing(self, tmp_path):
        import xarray as xr

        path = tmp_path / "prof_noC.nc"
        xr.Dataset(
            {
                "P": ("time_slow", np.array([1.0, 2.0])),
                "JAC_T": ("time_slow", np.array([20.0, 19.0])),
            }
        ).to_netcdf(path)

        assert _profile_practical_salinity(path, "JAC_T", "JAC_C") is None


class TestComputeSlowStratification:
    """Per-cast sorted N2/dTdz on the slow grid, fed to profile + CTD products."""

    def test_aligns_with_slow_grid_and_stable(self):
        from odas_tpw.perturb.pipeline import _compute_slow_stratification

        n = 400
        P = np.linspace(1.0, 50.0, n)
        T = 20.0 - 0.1 * P  # stable
        C = np.full(n, 45.0)
        pf = MagicMock()
        pf.channels = {"P": P, "JAC_T": T, "JAC_C": C}
        pf.is_fast = lambda name: False  # JAC_C is a slow channel

        N2, dTdz = _compute_slow_stratification(pf, [(0, n - 1)], "JAC_T", "JAC_C", 2.0)
        assert N2.shape == (n,) and dTdz.shape == (n,)
        assert np.isfinite(N2).any() and np.isfinite(dTdz).any()
        assert np.all(N2[np.isfinite(N2)] > 0)  # stable column
        # Outside the (single, full-span) cast nothing is masked here, but a
        # short cast yields all-NaN for that cast rather than raising.
        N2b, _ = _compute_slow_stratification(pf, [(0, 2)], "JAC_T", "JAC_C", 2.0)
        assert np.all(np.isnan(N2b[0:3]))

    def test_returns_none_without_pressure(self):
        from odas_tpw.perturb.pipeline import _compute_slow_stratification

        pf = MagicMock()
        pf.channels = {"JAC_T": np.zeros(10)}
        pf.is_fast = lambda name: False
        N2, dTdz = _compute_slow_stratification(pf, [(0, 9)], "JAC_T", "JAC_C", 2.0)
        assert N2 is None and dTdz is None


class TestAttachWindowStratification:
    """N2/dTdz attached to a window-grid (diss) dataset from a profile's CTD."""

    def _make_profile_nc(self, path):
        import xarray as xr

        n = 200
        t_slow = np.linspace(0.0, 100.0, n)
        P = np.linspace(1.0, 50.0, n)
        T = 20.0 - 0.1 * P  # stable: cooling downward
        C = np.full(n, 45.0)
        xr.Dataset(
            {
                "t_slow": ("time_slow", t_slow),
                "P": ("time_slow", P),
                "JAC_T": ("time_slow", T),
                "JAC_C": ("time_slow", C),
                "lat": ((), 15.0),
                "lon": ((), 145.0),
            }
        ).to_netcdf(path)

    def test_attaches_n2_and_dtdz(self, tmp_path):
        import xarray as xr

        from odas_tpw.perturb.pipeline import _attach_window_stratification

        prof = tmp_path / "prof.nc"
        self._make_profile_nc(prof)
        ds = xr.Dataset(
            {"epsilonMean": ("time", [1e-9, 1e-9, 1e-9])},
            coords={"t": ("time", np.array([25.0, 50.0, 75.0]))},
        )
        _attach_window_stratification(ds, prof, 5.0, "JAC_T", "JAC_C", "prof.nc", "dissipation")
        assert "N2" in ds and "dTdz" in ds
        assert ds["N2"].dims == ("time",)
        assert np.isfinite(ds["N2"].values).all()
        assert np.all(ds["N2"].values > 0)  # stable column

    def test_no_op_when_profile_unreadable(self, tmp_path):
        import xarray as xr

        from odas_tpw.perturb.pipeline import _attach_window_stratification

        bad = tmp_path / "empty.nc"
        bad.touch()
        ds = xr.Dataset(coords={"t": ("time", np.array([1.0]))})
        _attach_window_stratification(ds, bad, 5.0, "JAC_T", "JAC_C", "empty.nc", "dissipation")
        assert "N2" not in ds  # unreadable profile -> silently skipped


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


class TestEpsilonConfigDefaultsResolved:
    """process_file resolves epsilon/chi via merge_config, so an omitted or
    null fft_length uses the documented default (256) -- not _compute_epsilon's
    own 1024 -- and the diss-dir provenance hash (also merge_config-based)
    therefore matches the data, and diss_length_seconds is sized correctly
    (M-8/M-9, and the fft_length:null TypeError)."""

    def test_omitted_and_null_fft_length_resolve_to_default(self):
        from odas_tpw.perturb.config import merge_config

        assert merge_config("epsilon", {}).get("fft_length") == 256
        assert merge_config("epsilon", {"fft_length": None}).get("fft_length") == 256

    def test_diss_length_seconds_derives_from_merged_fft_length(self):
        from odas_tpw.perturb.config import merge_config

        # Mirrors the pipeline: diss_length or 4*fft_length, on the MERGED cfg
        # (so omitted -> 4*256, never 4*1024 and never 4*None -> TypeError).
        for raw in ({}, {"fft_length": None}):
            eps = merge_config("epsilon", raw)
            diss_len = eps.get("diss_length") or 4 * eps.get("fft_length")
            assert diss_len == 4 * 256


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

    def test_profiles_includes_preprocessing_sections(self):
        assert set(self._sections("profiles")) == {
            "files",
            "gps",
            "hotel",
            "speed",
            "qc",
            "fp07",
            "ct",
            "bottom",
            "top_trim",
            "stratification",
        }

    def test_diss_includes_profiles_and_instruments(self):
        assert {"profiles", "instruments"}.issubset(set(self._sections("diss")))

    def test_chi_includes_epsilon_and_profiles(self):
        # chi -> diss -> profiles
        assert {"epsilon", "profiles", "instruments"}.issubset(set(self._sections("chi")))

    def test_ctd_includes_file_flow_and_gps(self):
        # No 'stratification': N2/dTdz are profile-only and not on the CTD
        # product, so stratification.* must not version the CTD output.
        assert set(self._sections("ctd")) == {
            "files",
            "gps",
            "hotel",
            "speed",
            "qc",
            "ct",
            "profiles",
        }

    def test_profiles_binned_includes_profiles(self):
        assert "profiles" in self._sections("profiles_binned")

    def test_diss_binned_includes_epsilon_and_profiles(self):
        # was missing profiles before _upstream_for refactor
        assert {"epsilon", "profiles", "instruments"}.issubset(set(self._sections("diss_binned")))

    def test_chi_binned_includes_full_chain(self):
        # was missing epsilon and profiles before _upstream_for refactor
        assert {"chi", "epsilon", "profiles", "instruments"}.issubset(
            set(self._sections("chi_binned"))
        )

    def test_combo_includes_profiles(self):
        assert {"profiles", "netcdf"}.issubset(set(self._sections("combo")))

    def test_diss_combo_includes_epsilon_and_profiles(self):
        assert {"epsilon", "profiles", "instruments", "netcdf"}.issubset(
            set(self._sections("diss_combo"))
        )

    def test_chi_combo_includes_full_chain(self):
        assert {"chi", "epsilon", "profiles", "instruments", "netcdf"}.issubset(
            set(self._sections("chi_combo"))
        )

    def test_ctd_combo_includes_ctd(self):
        assert {"ctd", "gps", "hotel", "speed", "qc", "netcdf"}.issubset(
            set(self._sections("ctd_combo"))
        )

    def test_profiles_hash_changes_when_fp07_changes(self):
        from odas_tpw.perturb.config import compute_hash, merge_config

        cfg_a = self._config()
        cfg_b = self._config()
        cfg_b["fp07"] = {"order": 3}
        h_a = compute_hash(
            "profiles",
            merge_config("profiles", cfg_a.get("profiles")),
            upstream=_upstream_for("profiles", cfg_a),
        )
        h_b = compute_hash(
            "profiles",
            merge_config("profiles", cfg_b.get("profiles")),
            upstream=_upstream_for("profiles", cfg_b),
        )
        assert h_a != h_b

    def test_instrument_probe_exclusion_order_does_not_change_diss_hash(self):
        from odas_tpw.perturb.config import compute_hash, merge_config

        cfg_a = self._config()
        cfg_b = self._config()
        cfg_a["instruments"] = {"SN001": {"exclude_shear_probes": ["sh1", "sh2"]}}
        cfg_b["instruments"] = {"SN001": {"exclude_shear_probes": ["sh2", "sh1"]}}
        h_a = compute_hash(
            "epsilon",
            merge_config("epsilon", cfg_a.get("epsilon")),
            upstream=_upstream_for("diss", cfg_a),
        )
        h_b = compute_hash(
            "epsilon",
            merge_config("epsilon", cfg_b.get("epsilon")),
            upstream=_upstream_for("diss", cfg_b),
        )
        assert h_a == h_b


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

    def test_trim_preserves_relative_paths_for_duplicate_basenames(self, tmp_path):
        """Same basename in two instrument folders must not collide.

        Uses incomplete files so they are actually written to the trim tree
        (complete files are referenced in place and never land in trimmed/).
        """
        root = tmp_path / "vmp"
        (root / "SN001").mkdir(parents=True)
        (root / "SN002").mkdir(parents=True)
        for sn in ("SN001", "SN002"):
            p = root / sn / "cast.p"
            _make_p_file(p)
            # Append a partial final record so trimming materializes the file.
            p.write_bytes(p.read_bytes() + b"\xff" * 10)

        config = {
            "files": {
                "p_file_root": str(root),
                "p_file_pattern": "**/*.p",
                "output_root": str(tmp_path / "out"),
            },
        }
        results = sorted(run_trim(config))
        assert results == [
            tmp_path / "out" / "trimmed" / "SN001" / "cast.p",
            tmp_path / "out" / "trimmed" / "SN002" / "cast.p",
        ]

    def test_trim_references_complete_files_in_place(self, tmp_path):
        """Complete files are returned as their originals; trimmed/ is untouched."""
        p_dir = tmp_path / "vmp"
        p_dir.mkdir()
        src = p_dir / "complete.p"
        _make_p_file(src, n_data_records=3)  # multiple of record_size → complete

        config = {
            "files": {
                "p_file_root": str(p_dir),
                "p_file_pattern": "*.p",
                "output_root": str(tmp_path / "out"),
            },
        }
        results = run_trim(config)
        # The original path is returned, not a copy under trimmed/.
        assert results == [src]
        # Nothing was materialized in the trim directory.
        assert not (tmp_path / "out" / "trimmed").exists()

    def test_trim_rejects_duplicate_outputs_without_root(self, tmp_path):
        """Explicit files outside p_file_root fall back to basename and must collide loudly."""
        a = tmp_path / "a" / "cast.p"
        b = tmp_path / "b" / "cast.p"
        a.parent.mkdir()
        b.parent.mkdir()
        _make_p_file(a)
        _make_p_file(b)

        config = {
            "files": {
                "p_file_root": str(tmp_path / "not-the-root"),
                "output_root": str(tmp_path / "out"),
            },
        }
        with pytest.raises(ValueError, match="same output path"):
            run_trim(config, p_files=[a, b])


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

    def test_merge_uses_current_inputs_and_keeps_singletons(self, tmp_path):
        """After trim, merge should use trimmed files and keep non-merged files.

        Files are made incomplete so trimming materializes them under
        out/trimmed/ (complete files would be referenced from their original
        location instead).
        """
        root = tmp_path / "vmp"
        (root / "SN001").mkdir(parents=True)
        (root / "SN002").mkdir(parents=True)
        _make_p_file(root / "SN001" / "cast_0001.p", file_number=1)
        _make_p_file(root / "SN001" / "cast_0002.p", file_number=2)
        _make_p_file(root / "SN002" / "solo.p", file_number=10)
        for rel in ("SN001/cast_0001.p", "SN001/cast_0002.p", "SN002/solo.p"):
            p = root / rel
            p.write_bytes(p.read_bytes() + b"\xff" * 10)

        config = {
            "files": {
                "p_file_root": str(root),
                "p_file_pattern": "**/*.p",
                "output_root": str(tmp_path / "out"),
            },
        }
        trimmed = run_trim(config)
        trim_root = tmp_path / "out" / "trimmed"
        merged = sorted(
            run_merge(
                config,
                trimmed,
                include_singletons=True,
                input_root=trim_root,
            )
        )

        assert tmp_path / "out" / "merged" / "SN001" / "cast_0001.p" in merged
        assert tmp_path / "out" / "trimmed" / "SN002" / "solo.p" in merged
        assert root / "SN002" / "solo.p" not in merged


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
    def test_direction_auto_resolves_via_vehicle(
        self, mock_pfile_cls, mock_smooth, mock_get_prof, mock_fp07_cal, mock_extract, tmp_path
    ):
        """profiles.direction='auto' must hit resolve_direction so a Slocum
        glider gets ``glide`` (up + down). Without the resolver, ``get_profiles``
        falls through to its default ``down`` branch and silently drops every
        ascending profile -- which on a MR-on-during-climb deployment is *all*
        the real flight data.
        """
        mock_pf = MagicMock()
        mock_pf.channels = {"P": np.linspace(0, 50, 100), "T1": np.zeros(100)}
        mock_pf.fs_slow = 64.0
        mock_pf.config = {"instrument_info": {"vehicle": "slocum_glider"}}
        mock_pfile_cls.return_value = mock_pf
        mock_get_prof.return_value = []  # we just want the kwargs

        config = self._base_config(tmp_path)
        config["profiles"]["direction"] = "auto"
        output_dirs = {"profiles": tmp_path / "profiles", "diss": tmp_path / "diss"}
        (tmp_path / "profiles").mkdir(parents=True, exist_ok=True)
        (tmp_path / "diss").mkdir(parents=True, exist_ok=True)

        process_file(tmp_path / "test.p", config, None, output_dirs)
        mock_get_prof.assert_called()
        kwargs = mock_get_prof.call_args.kwargs
        assert kwargs["direction"] == "glide", (
            f"slocum_glider 'auto' should resolve to 'glide', got {kwargs['direction']!r}"
        )

    @patch("odas_tpw.rsi.profile.extract_profiles", return_value=([Path("/fake/prof.nc")], [{}]))
    @patch("odas_tpw.perturb.fp07_cal.fp07_calibrate")
    @patch("odas_tpw.rsi.profile.get_profiles")
    @patch("odas_tpw.rsi.profile._smooth_fall_rate", return_value=np.zeros(100))
    @patch("odas_tpw.rsi.p_file.PFile")
    def test_direction_explicit_passes_through(
        self, mock_pfile_cls, mock_smooth, mock_get_prof, mock_fp07_cal, mock_extract, tmp_path
    ):
        """Explicit direction values bypass the resolver."""
        mock_pf = MagicMock()
        mock_pf.channels = {"P": np.linspace(0, 50, 100), "T1": np.zeros(100)}
        mock_pf.fs_slow = 64.0
        mock_pf.config = {"instrument_info": {"vehicle": "slocum_glider"}}
        mock_pfile_cls.return_value = mock_pf
        mock_get_prof.return_value = []

        config = self._base_config(tmp_path)
        config["profiles"]["direction"] = "down"
        output_dirs = {"profiles": tmp_path / "profiles", "diss": tmp_path / "diss"}
        (tmp_path / "profiles").mkdir(parents=True, exist_ok=True)
        (tmp_path / "diss").mkdir(parents=True, exist_ok=True)

        process_file(tmp_path / "test.p", config, None, output_dirs)
        kwargs = mock_get_prof.call_args.kwargs
        assert kwargs["direction"] == "down"

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

    @patch("odas_tpw.perturb.ctd.ctd_bin_file")
    @patch("odas_tpw.rsi.profile.get_profiles", return_value=[])
    @patch("odas_tpw.rsi.profile._smooth_fall_rate", return_value=np.zeros(100))
    @patch("odas_tpw.rsi.p_file.PFile")
    def test_ctd_receives_output_stem(
        self, mock_pfile_cls, mock_smooth, mock_get_prof, mock_ctd_bin, tmp_path
    ):
        """CTD outputs use the pipeline's unique per-file stem."""
        mock_pf = MagicMock()
        mock_pf.channels = {"P": np.linspace(0, 50, 100), "T1": np.zeros(100)}
        mock_pf.fs_slow = 64.0
        mock_pfile_cls.return_value = mock_pf

        config = self._base_config(tmp_path)
        config["ctd"]["enable"] = True
        output_dirs = {
            "profiles": tmp_path / "profiles",
            "diss": tmp_path / "diss",
            "ctd": tmp_path / "ctd",
        }
        output_dirs["ctd"].mkdir(parents=True, exist_ok=True)

        process_file(
            tmp_path / "test.p",
            config,
            None,
            output_dirs,
            output_stem="SN001__cast",
        )
        assert mock_ctd_bin.call_args.kwargs["output_stem"] == "SN001__cast"

    @patch("odas_tpw.rsi.profile.extract_profiles", return_value=([Path("/fake/prof.nc")], [{}]))
    @patch("odas_tpw.perturb.fp07_cal.fp07_calibrate", return_value={"channels": {}})
    @patch("odas_tpw.rsi.profile.get_profiles")
    @patch("odas_tpw.rsi.profile._smooth_fall_rate", return_value=np.zeros(100))
    @patch("odas_tpw.rsi.p_file.PFile")
    def test_profile_extract_receives_output_stem(
        self, mock_pfile_cls, mock_smooth, mock_get_prof, mock_fp07_cal, mock_extract, tmp_path
    ):
        """Profile outputs use the pipeline's unique per-file stem."""
        mock_pf = MagicMock()
        mock_pf.channels = {"P": np.linspace(0, 50, 100), "T1": np.zeros(100)}
        mock_pf.fs_slow = 64.0
        mock_pfile_cls.return_value = mock_pf
        mock_get_prof.return_value = [{"start": 0, "end": 50}]

        config = self._base_config(tmp_path)
        output_dirs = {"profiles": tmp_path / "profiles", "diss": tmp_path / "diss"}
        for d in output_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        process_file(
            tmp_path / "test.p",
            config,
            None,
            output_dirs,
            output_stem="SN001__cast",
        )
        assert mock_extract.call_args.kwargs["output_stem"] == "SN001__cast"

    @patch("odas_tpw.rsi.chi_io._load_therm_channels", return_value={})
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
        mock_load_therm,
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
        # Default use_epsilon=True → diss_ds is passed in (Method 1).
        kwargs = mock_chi.call_args.kwargs
        assert kwargs["epsilon_ds"] is not None, (
            "use_epsilon=True (default) should pass the diss dataset"
        )

    @patch("odas_tpw.rsi.chi_io._load_therm_channels", return_value={})
    @patch("odas_tpw.rsi.chi_io._compute_chi", return_value=[MagicMock()])
    @patch("odas_tpw.rsi.dissipation._compute_epsilon")
    @patch("odas_tpw.rsi.profile.extract_profiles")
    @patch("odas_tpw.perturb.fp07_cal.fp07_calibrate", return_value={"channels": {}})
    @patch("odas_tpw.rsi.profile.get_profiles")
    @patch("odas_tpw.rsi.profile._smooth_fall_rate", return_value=np.zeros(100))
    @patch("odas_tpw.rsi.p_file.PFile")
    def test_chi_use_epsilon_false_passes_none(
        self,
        mock_pfile_cls,
        mock_smooth,
        mock_get_prof,
        mock_fp07_cal,
        mock_extract,
        mock_eps,
        mock_chi,
        mock_load_therm,
        tmp_path,
    ):
        """chi.use_epsilon=False routes to Method 2 (epsilon_ds=None)."""
        import xarray as xr

        mock_pf = MagicMock()
        mock_pf.channels = {"P": np.linspace(0, 50, 100), "T1": np.zeros(100)}
        mock_pf.fs_slow = 64.0
        mock_pfile_cls.return_value = mock_pf
        mock_get_prof.return_value = [{"start": 0, "end": 50}]

        prof_dir = tmp_path / "profiles"
        prof_dir.mkdir(parents=True, exist_ok=True)
        diss_dir = tmp_path / "diss"
        diss_dir.mkdir(parents=True, exist_ok=True)
        chi_dir = tmp_path / "chi"
        chi_dir.mkdir(parents=True, exist_ok=True)

        prof_nc = prof_dir / "prof.nc"
        prof_nc.touch()
        mock_extract.return_value = ([prof_nc], [{}])

        diss_nc = diss_dir / "prof.nc"
        xr.Dataset({"epsilon": (("time",), [1e-8])}).to_netcdf(diss_nc)
        mock_eps.return_value = [xr.Dataset({"epsilon": (("time",), [1e-8])})]

        config = self._base_config(tmp_path)
        config["chi"]["enable"] = True
        config["chi"]["use_epsilon"] = False
        output_dirs = {"profiles": prof_dir, "diss": diss_dir, "chi": chi_dir}

        process_file(tmp_path / "test.p", config, None, output_dirs)
        mock_chi.assert_called_once()
        kwargs = mock_chi.call_args.kwargs
        assert kwargs["epsilon_ds"] is None, (
            "use_epsilon=False should call _compute_chi with epsilon_ds=None"
        )
        assert "use_epsilon" not in kwargs, (
            "use_epsilon must be popped, not forwarded to _compute_chi"
        )

    @patch("odas_tpw.rsi.chi_io._load_therm_channels", return_value={})
    @patch("odas_tpw.rsi.chi_io._compute_chi")
    @patch("odas_tpw.rsi.dissipation._compute_epsilon")
    @patch("odas_tpw.rsi.profile.extract_profiles")
    @patch("odas_tpw.perturb.fp07_cal.fp07_calibrate", return_value={"channels": {}})
    @patch("odas_tpw.rsi.profile.get_profiles")
    @patch("odas_tpw.rsi.profile._smooth_fall_rate", return_value=np.zeros(100))
    @patch("odas_tpw.rsi.p_file.PFile")
    def test_chi_skips_profile_when_matching_diss_failed(
        self,
        mock_pfile_cls,
        mock_smooth,
        mock_get_prof,
        mock_fp07_cal,
        mock_extract,
        mock_eps,
        mock_chi,
        mock_load_therm,
        tmp_path,
    ):
        """A diss failure in the middle must not shift later epsilon files left."""
        import xarray as xr

        mock_pf = MagicMock()
        mock_pf.channels = {"P": np.linspace(0, 50, 100), "T1": np.zeros(100)}
        mock_pf.fs_slow = 64.0
        mock_pfile_cls.return_value = mock_pf
        mock_get_prof.return_value = [{"start": 0, "end": 50}]

        prof_dir = tmp_path / "profiles"
        diss_dir = tmp_path / "diss"
        chi_dir = tmp_path / "chi"
        for d in (prof_dir, diss_dir, chi_dir):
            d.mkdir(parents=True, exist_ok=True)
        profs = [prof_dir / f"prof{i}.nc" for i in range(1, 4)]
        for prof in profs:
            prof.touch()
        mock_extract.return_value = (profs, [{}, {}, {}])

        def eps_side_effect(prof_path, **kwargs):
            if Path(prof_path).name == "prof2.nc":
                raise RuntimeError("middle profile failed")
            value = 1e-8 if Path(prof_path).name == "prof1.nc" else 3e-8
            return [xr.Dataset({"epsilon": (("time",), [value])})]

        mock_eps.side_effect = eps_side_effect

        def chi_side_effect(prof_path, **kwargs):
            return [xr.Dataset({"chi": (("time",), [1e-9])})]

        mock_chi.side_effect = chi_side_effect

        config = self._base_config(tmp_path)
        config["chi"]["enable"] = True
        output_dirs = {"profiles": prof_dir, "diss": diss_dir, "chi": chi_dir}

        result = process_file(tmp_path / "test.p", config, None, output_dirs)

        called_profiles = [Path(call.args[0]).name for call in mock_chi.call_args_list]
        assert called_profiles == ["prof1.nc", "prof3.nc"]
        assert result["diss"] == [str(diss_dir / "prof1.nc"), str(diss_dir / "prof3.nc")]
        assert result["chi"] == [str(chi_dir / "prof1.nc"), str(chi_dir / "prof3.nc")]


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

    @patch(
        "odas_tpw.perturb.pipeline.process_file",
        return_value={"source": "test.p", "profiles": [], "diss": [], "chi": []},
    )
    @patch("odas_tpw.perturb.pipeline._setup_output_dirs", return_value={})
    @patch("odas_tpw.perturb.gps.create_gps", return_value=None)
    def test_duplicate_basenames_get_distinct_output_stems(
        self, mock_gps, mock_dirs, mock_proc, tmp_path
    ):
        """Same basename in different folders must not collide after trim/merge."""
        root = tmp_path / "vmp"
        p1 = root / "SN001" / "cast.p"
        p2 = root / "SN002" / "cast.p"
        p1.parent.mkdir(parents=True)
        p2.parent.mkdir(parents=True)
        p1.touch()
        p2.touch()

        config = self._base_config(tmp_path)
        config["files"]["p_file_root"] = str(root)
        config["files"]["trim"] = False

        run_pipeline(config, p_files=[p1, p2])

        stems = [call.kwargs["output_stem"] for call in mock_proc.call_args_list]
        assert stems == ["SN001__cast", "SN002__cast"]

    @patch(
        "odas_tpw.perturb.pipeline.process_file",
        return_value={"source": "test.p", "profiles": [], "diss": [], "chi": []},
    )
    @patch("odas_tpw.perturb.pipeline._setup_output_dirs", return_value={})
    @patch("odas_tpw.perturb.gps.create_gps", return_value=None)
    def test_trim_merge_preserves_distinct_output_stems_and_instruments(
        self, mock_gps, mock_dirs, mock_proc, tmp_path
    ):
        """Merged chains and singleton files keep distinct stems and instrument keys."""
        root = tmp_path / "vmp"
        (root / "SN001").mkdir(parents=True)
        (root / "SN002").mkdir(parents=True)
        _make_p_file(root / "SN001" / "cast_0001.p", file_number=1)
        _make_p_file(root / "SN001" / "cast_0002.p", file_number=2)
        _make_p_file(root / "SN002" / "cast.p", file_number=10)

        config = self._base_config(tmp_path)
        config["files"]["p_file_root"] = str(root)
        config["files"]["merge"] = True

        run_pipeline(config, p_files=sorted(root.glob("**/*.p")))

        calls = mock_proc.call_args_list
        stems = [call.kwargs["output_stem"] for call in calls]
        instrument_keys = [call.kwargs["instrument_key"] for call in calls]
        assert stems == ["SN001__cast_0001", "SN002__cast"]
        assert instrument_keys == ["SN001", "SN002"]


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


class TestApplyFomCut:
    """Tests for ``_apply_fom_cut`` per-probe per-segment FOM mask."""

    def _make_diss_ds(self):
        import xarray as xr

        # 2 probes x 5 segments. probe 0 fom=[0.5, 1, 5, 10, 0.8],
        # probe 1 fom=[0.5, 0.5, 0.5, 50, 0.5]. With fom_max=2.0:
        #   probe 0 bad at segs 2, 3.
        #   probe 1 bad at seg 3.
        fom = np.array(
            [[0.5, 1.0, 5.0, 10.0, 0.8], [0.5, 0.5, 0.5, 50.0, 0.5]],
            dtype=np.float64,
        )
        epsilon = np.full_like(fom, 1e-9)
        return xr.Dataset(
            {
                "epsilon": (["probe", "time"], epsilon),
                "fom": (["probe", "time"], fom),
                "e_1": (["time"], epsilon[0].copy()),
                "e_2": (["time"], epsilon[1].copy()),
            },
            coords={"probe": ["sh1", "sh2"], "time": np.arange(5, dtype=float)},
        )

    def test_per_probe_per_segment_mask(self):
        from odas_tpw.perturb.pipeline import _apply_fom_cut

        ds = self._make_diss_ds()
        n = _apply_fom_cut(ds, fom_max=2.0, file_label="test.p")
        # 2 cells from probe 0 + 1 cell from probe 1 = 3
        assert n == 3
        eps = ds["epsilon"].values
        assert np.isnan(eps[0, 2]) and np.isnan(eps[0, 3])
        assert np.isfinite(eps[0, 0]) and np.isfinite(eps[0, 1]) and np.isfinite(eps[0, 4])
        assert np.isnan(eps[1, 3])
        assert np.isfinite(eps[1, [0, 1, 2, 4]]).all()

    def test_companions_mirror_per_probe_mask(self):
        from odas_tpw.perturb.pipeline import _apply_fom_cut

        ds = self._make_diss_ds()
        _apply_fom_cut(ds, fom_max=2.0, file_label="test.p")
        e1 = ds["e_1"].values
        e2 = ds["e_2"].values
        # e_1 follows probe 0's mask (bad at 2, 3); e_2 follows probe 1's (bad at 3)
        assert np.isnan(e1[2]) and np.isnan(e1[3])
        assert np.isfinite(e1[[0, 1, 4]]).all()
        assert np.isnan(e2[3]) and np.isfinite(e2[[0, 1, 2, 4]]).all()

    def test_fom_var_left_untouched(self):
        from odas_tpw.perturb.pipeline import _apply_fom_cut

        ds = self._make_diss_ds()
        original = ds["fom"].values.copy()
        _apply_fom_cut(ds, fom_max=2.0, file_label="test.p")
        # fom is left intact -- it's diagnostic, used by other tools
        np.testing.assert_array_equal(ds["fom"].values, original)

    def test_no_op_when_fom_absent(self):
        import xarray as xr

        from odas_tpw.perturb.pipeline import _apply_fom_cut

        ds = xr.Dataset({"epsilon": (("time",), np.ones(3))})
        assert _apply_fom_cut(ds, fom_max=2.0, file_label="test.p") == 0

    def test_chi_dataset_path(self):
        """Same helper applies to chi datasets (chi + chi_N companions)."""
        import xarray as xr

        from odas_tpw.perturb.pipeline import _apply_fom_cut

        fom = np.array(
            [[0.5, 50.0], [0.5, 0.5]],
            dtype=np.float64,
        )
        chi = np.full_like(fom, 1e-9)
        ds = xr.Dataset(
            {
                "chi": (["probe", "time"], chi),
                "fom": (["probe", "time"], fom),
                "chi_1": (["time"], chi[0].copy()),
                "chi_2": (["time"], chi[1].copy()),
            },
            coords={"probe": ["t1", "t2"], "time": np.arange(2, dtype=float)},
        )
        assert _apply_fom_cut(ds, fom_max=2.0, file_label="test.p") == 1
        assert np.isnan(ds["chi"].values[0, 1])
        assert np.isnan(ds["chi_1"].values[1])
        assert np.isfinite(ds["chi"].values[1, :]).all()


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
        P = np.concatenate(
            [
                np.linspace(0.0, max_depth, half),
                np.linspace(max_depth, 0.0, n_slow - half),
            ]
        )
        t_slow = np.arange(n_slow) / fs_slow
        t_fast = np.arange(n_fast) / fs_fast

        # Quiet baseline noise; high variance only in top 8 m of descending
        # limb. top_trim is driven by the accelerometers, so the prop-wash
        # signal lives on Ax/Ay (instrument motion), not the shear probes.
        rng = np.random.default_rng(42)
        depth_fast = np.repeat(P, ratio)
        top_mask = (depth_fast < 8.0) & (np.arange(n_fast) < half * ratio)
        ax = rng.standard_normal(n_fast) * 0.01
        ax = np.where(top_mask, rng.standard_normal(n_fast) * 5.0, ax)
        ay = rng.standard_normal(n_fast) * 0.01
        ay = np.where(top_mask, rng.standard_normal(n_fast) * 5.0, ay)

        pf = MagicMock()
        pf.fs_slow = fs_slow
        pf.fs_fast = fs_fast
        pf.t_slow = t_slow
        pf.t_fast = t_fast
        pf.channels = {
            "P": P,
            "sh1": rng.standard_normal(n_fast) * 0.01,
            "sh2": rng.standard_normal(n_fast) * 0.01,
            "Ax": ax,
            "Ay": ay,
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
        combo_dirs = sorted(tmp_path.glob("combo_[0-9][0-9]"))
        assert len(combo_dirs) == 1, "exactly one versioned combo dir expected"
        sigs = list(combo_dirs[0].glob(".params_sha256_*"))
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

        hash_a = next(next(a.glob("combo_[0-9][0-9]")).glob(".params_sha256_*")).name
        hash_b = next(next(b.glob("combo_[0-9][0-9]")).glob(".params_sha256_*")).name
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
        ctd_combo_dirs = sorted(tmp_path.glob("ctd_combo_[0-9][0-9]"))
        assert len(ctd_combo_dirs) == 1, "exactly one versioned ctd_combo dir expected"
        sigs = list(ctd_combo_dirs[0].glob(".params_sha256_*"))
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


class TestAddMixingQuantities:
    """Mixing-quantity integration on per-profile chi datasets."""

    def _make_inputs(self, tmp_path):
        import gsw
        import xarray as xr

        fs_slow = 64.0
        n = 4000
        t = np.arange(n) / fs_slow  # file-relative seconds
        P = 10.0 + 0.7 * t
        depth = -gsw.z_from_p(P, 0.0)
        T = 20.0 - 0.05 * depth
        C = gsw.C_from_SP(np.full(n, 35.0), T, P)

        units = "seconds since 2025-01-16T01:57:58"
        prof = xr.Dataset(
            {
                "P": (["time_slow"], P),
                "JAC_T": (["time_slow"], T),
                "JAC_C": (["time_slow"], C),
            },
        )
        prof["t_slow"] = xr.DataArray(t, dims=["time_slow"], attrs={"units": units})
        prof_path = tmp_path / "prof.nc"
        prof.to_netcdf(prof_path)

        win_t = np.arange(8.0, 50.0, 8.0)
        chi_ds = xr.Dataset(
            {"chiMean": (["time"], np.full(len(win_t), 1e-8))},
            coords={"t": ("time", win_t, {"units": units})},
            attrs={"diss_length": 2048, "fs_fast": 512.0},
        )
        # Distinct per-window epsilon on a slightly offset grid, so a wrong
        # pairing (off-by-one, reversed, verbatim copy) is detectable.
        eps_vals = 1e-9 * (1.0 + np.arange(len(win_t)))
        diss_ds = xr.Dataset(
            {"epsilonMean": (["time"], eps_vals)},
            coords={"t": ("time", win_t + 0.25, {"units": units})},
        )
        return chi_ds, diss_ds, prof_path

    def test_variables_added_with_real_salinity(self, tmp_path):
        from odas_tpw.perturb.pipeline import _add_mixing_quantities

        chi_ds, diss_ds, prof_path = self._make_inputs(tmp_path)
        out = _add_mixing_quantities(chi_ds, diss_ds, prof_path, file_label="t")
        for v in ("N2", "dTdz", "K_T", "Gamma", "K_rho", "epsilon_paired"):
            assert v in out
            assert np.all(np.isfinite(out[v].values))
        # epsilon_paired is the exact epsilon that entered Gamma: each chi
        # window pairs with its nearest (0.25-s offset) diss window, whose
        # values are distinct per window — a wrong pairing cannot pass.
        np.testing.assert_array_equal(out["epsilon_paired"].values, diss_ds["epsilonMean"].values)
        assert out["epsilon_paired"].attrs["units"] == "W/kg"
        np.testing.assert_allclose(out["dTdz"].values, -0.05, rtol=1e-2)
        assert "practical salinity from JAC_C" in out["N2"].attrs["comment"]
        assert np.all(out["N2"].values > 0)

    def test_no_conductivity_falls_back(self, tmp_path):
        import xarray as xr

        from odas_tpw.perturb.pipeline import _add_mixing_quantities

        chi_ds, diss_ds, prof_path = self._make_inputs(tmp_path)
        prof = xr.open_dataset(prof_path, decode_times=False)
        prof = prof.drop_vars("JAC_C")
        prof_path2 = prof_path.parent / "prof_noc.nc"
        prof.to_netcdf(prof_path2)
        prof.close()
        out = _add_mixing_quantities(chi_ds, diss_ds, prof_path2, file_label="t")
        assert "assumed 35 PSU" in out["N2"].attrs["comment"]
        assert np.all(np.isfinite(out["N2"].values))

    def test_missing_epsilon_is_noop(self, tmp_path):
        import xarray as xr

        from odas_tpw.perturb.pipeline import _add_mixing_quantities

        chi_ds, _, prof_path = self._make_inputs(tmp_path)
        empty = xr.Dataset()
        out = _add_mixing_quantities(chi_ds, empty, prof_path, file_label="t")
        assert "Gamma" not in out
        assert "epsilon_paired" not in out

    def test_chi_qc_drop_nans_chi_derived_mixing(self, tmp_path):
        """Audit r1-4: a QC-dropped chi segment must NaN the chi-derived mixing
        quantities (K_T, Gamma) — the pipeline applies chi QC BEFORE deriving
        them. K_rho is epsilon-derived and stays finite (gated by epsilon QC)."""
        from odas_tpw.perturb.pipeline import _add_mixing_quantities
        from odas_tpw.perturb.qc_gate import apply_qc_to_dataset

        class _StubPF:
            def __init__(self, t_slow, channels):
                self.t_slow = np.asarray(t_slow, dtype=np.float64)
                self.channels = channels
                self.channel_info = {}

        chi_ds, diss_ds, prof_path = self._make_inputs(tmp_path)
        win_t = chi_ds["t"].values  # [8, 16, 24, 32, 40, 48]
        flagged = 2  # segment centered at t=24 s
        t_slow = np.arange(4000) / 64.0
        q = (np.abs(t_slow - win_t[flagged]) <= 1.0).astype(np.uint8)
        pf = _StubPF(t_slow, {"q_drop_chi": q})

        # Pipeline order: QC the chi vars, THEN derive mixing quantities.
        apply_qc_to_dataset(
            chi_ds,
            pf,
            ["q_drop_chi"],
            2.0,
            flag_var_name="qc_drop_chi",
            value_vars=["chiMean"],
        )
        assert np.isnan(chi_ds["chiMean"].values[flagged])
        out = _add_mixing_quantities(chi_ds, diss_ds, prof_path, file_label="t")

        # chi-derived quantities dropped at the flagged segment...
        assert np.isnan(out["K_T"].values[flagged])
        assert np.isnan(out["Gamma"].values[flagged])
        # ...but epsilon-derived K_rho is not gated by the chi QC (eps is good).
        assert np.isfinite(out["K_rho"].values[flagged])
        # Unflagged segments keep finite chi-derived mixing quantities.
        keep = [i for i in range(len(win_t)) if i != flagged]
        assert np.all(np.isfinite(out["K_T"].values[keep]))
        assert np.all(np.isfinite(out["Gamma"].values[keep]))


class TestTimeEpochSeconds:
    def test_cf_units_converted(self):
        import xarray as xr

        from odas_tpw.perturb.pipeline import _time_epoch_seconds

        da = xr.DataArray(
            np.array([0.0, 10.0]),
            attrs={"units": "seconds since 1970-01-01T00:01:00"},
        )
        np.testing.assert_allclose(_time_epoch_seconds(da), [60.0, 70.0])

    def test_datetime64_converted(self):
        import xarray as xr

        from odas_tpw.perturb.pipeline import _time_epoch_seconds

        da = xr.DataArray(np.array(["1970-01-01T00:02:00"], dtype="datetime64[ns]"))
        np.testing.assert_allclose(_time_epoch_seconds(da), [120.0])

    def test_tz_suffix_stripped(self):
        import xarray as xr

        from odas_tpw.perturb.pipeline import _time_epoch_seconds

        da = xr.DataArray(
            np.array([5.0]),
            attrs={"units": "seconds since 1970-01-01T00:00:00+00:00"},
        )
        np.testing.assert_allclose(_time_epoch_seconds(da), [5.0])


class _FakePF:
    """Minimal PFile stand-in for the seawater-injection helper."""

    def __init__(self, n: int = 120, drop: tuple[str, ...] = ()):
        import datetime

        self.t_slow = np.arange(n, dtype=float)
        self.start_time = datetime.datetime(2025, 1, 1, tzinfo=datetime.UTC)
        self.channels = {
            "JAC_T": np.full(n, 15.0),  # in-situ temperature [°C]
            "JAC_C": np.full(n, 42.0),  # conductivity [mS/cm] ~ oceanic
            "P": np.linspace(0.0, 50.0, n),  # pressure [dbar]
        }
        for name in drop:
            del self.channels[name]
        self.channel_info: dict[str, dict] = {}


class _FakeGPS:
    def lat(self, t):
        return np.full(np.asarray(t, dtype=float).shape, 15.0)

    def lon(self, t):
        return np.full(np.asarray(t, dtype=float).shape, 145.0)


class TestInjectSeawaterProperties:
    def test_injects_full_suite_on_slow_grid(self):
        pf, gps = _FakePF(), _FakeGPS()
        names = _inject_seawater_properties(pf, gps, "JAC_T", "JAC_C")
        assert names == ("SP", "SA", "CT", "sigma0", "rho")
        for name in names:
            assert name in pf.channels
            assert len(pf.channels[name]) == len(pf.t_slow)  # slow-length
            assert np.all(np.isfinite(pf.channels[name]))
            assert pf.channel_info[name]["type"] == "derived"
        # C~42 mS/cm at 15 °C, near-surface -> plausibly oceanic salinity.
        assert 20.0 < float(np.nanmean(pf.channels["SP"])) < 40.0
        assert pf.channel_info["SP"]["units"] == "1"  # CF dimensionless
        assert pf.channel_info["CT"]["units"] == "degree_Celsius"

    def test_noop_when_conductivity_missing(self):
        pf, gps = _FakePF(drop=("JAC_C",)), _FakeGPS()
        assert _inject_seawater_properties(pf, gps, "JAC_T", "JAC_C") == ()
        assert "SP" not in pf.channels

    def test_noop_on_length_mismatch(self):
        pf, gps = _FakePF(), _FakeGPS()
        pf.channels["JAC_C"] = np.full(len(pf.t_slow) // 2, 42.0)  # e.g. a fast C
        assert _inject_seawater_properties(pf, gps, "JAC_T", "JAC_C") == ()
        assert "SP" not in pf.channels
