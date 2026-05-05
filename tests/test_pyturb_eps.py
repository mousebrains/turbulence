# May-2026, Claude and Pat Welch, pat@mousebrains.com
"""Coverage tests for ``odas_tpw.pyturb.eps``.

The eps module is the heart of ``pyturb-cli eps``: it dispatches per-file
processing in serial or parallel, opens .p / .nc inputs, detects profiles,
and walks each profile through the SCOR-160 + chi processing chain.

These tests mock the heavy chain (PFile, L1Data construction, process_l2/3/4)
so each branch is reachable in milliseconds, without depending on real
.p data being present in CI.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# ---------------------------------------------------------------------------
# Shared fixtures: mock SCOR-160 results so _process_profile can return a
# valid xr.Dataset without running the real numerics.
# ---------------------------------------------------------------------------


def _make_l1(*, has_temp_fast: bool = False, n_time: int = 100):
    """Return a mock L1Data-like object with the attributes _process_profile reads."""
    l1 = MagicMock()
    l1.fs_fast = 512.0
    l1.fs_slow = 64.0
    l1.f_AA = 98.0
    l1.has_temp_fast = has_temp_fast
    l1.n_time = n_time
    return l1


def _make_l4(n_spectra: int = 5, n_shear: int = 2):
    """Build an L4Data-shaped object via the real dataclass."""
    from odas_tpw.scor160.io import L4Data

    return L4Data(
        time=np.arange(n_spectra, dtype=float),
        pres=np.linspace(10, 50, n_spectra),
        pspd_rel=np.full(n_spectra, 0.7),
        section_number=np.ones(n_spectra),
        epsi=np.full((n_shear, n_spectra), 1e-8),
        epsi_final=np.full(n_spectra, 1e-8),
        epsi_flags=np.zeros((n_shear, n_spectra)),
        fom=np.ones((n_shear, n_spectra)),
        mad=np.full((n_shear, n_spectra), 0.1),
        kmax=np.full((n_shear, n_spectra), 50.0),
        method=np.zeros((n_shear, n_spectra)),
        var_resolved=np.full((n_shear, n_spectra), 0.9),
    )


def _make_l3(n_spectra: int = 5, n_freq: int = 257, n_shear: int = 2):
    from odas_tpw.scor160.io import L3Data

    return L3Data(
        time=np.arange(n_spectra, dtype=float),
        pres=np.linspace(10, 50, n_spectra),
        temp=np.full(n_spectra, 15.0),
        pspd_rel=np.full(n_spectra, 0.7),
        section_number=np.ones(n_spectra),
        kcyc=np.tile(np.arange(n_freq, dtype=float)[:, None], (1, n_spectra)),
        sh_spec=np.ones((n_shear, n_freq, n_spectra)) * 1e-6,
        sh_spec_clean=np.ones((n_shear, n_freq, n_spectra)) * 1e-6,
    )


def _good_chain_patches():
    """Return patches that replace process_l2/3/4 with successful mocks."""
    l2 = MagicMock()
    l3 = _make_l3()
    l4 = _make_l4()
    return [
        patch("odas_tpw.scor160.l2.process_l2", return_value=l2),
        patch("odas_tpw.scor160.l3.process_l3", return_value=l3),
        patch("odas_tpw.scor160.l4.process_l4", return_value=l4),
    ]


# ---------------------------------------------------------------------------
# _process_profile
# ---------------------------------------------------------------------------


class TestProcessProfile:
    def test_shear_chain_failure_returns_none(self):
        from odas_tpw.pyturb.eps import _process_profile

        l1 = _make_l1()
        with patch("odas_tpw.scor160.l2.process_l2", side_effect=RuntimeError("l2 boom")):
            assert _process_profile(l1, MagicMock(), MagicMock(), 35.0, None) is None

    def test_no_spectra_returns_none(self):
        from odas_tpw.pyturb.eps import _process_profile

        empty_l4 = _make_l4(n_spectra=0)
        with (
            patch("odas_tpw.scor160.l2.process_l2", return_value=MagicMock()),
            patch("odas_tpw.scor160.l3.process_l3", return_value=_make_l3(n_spectra=0)),
            patch("odas_tpw.scor160.l4.process_l4", return_value=empty_l4),
        ):
            assert _process_profile(_make_l1(), MagicMock(), MagicMock(), 35.0, None) is None

    def test_chi_failure_continues_without_chi(self):
        """``has_temp_fast`` true + chi raises ⇒ shear-only dataset returned."""
        from odas_tpw.pyturb.eps import _process_profile

        l1 = _make_l1(has_temp_fast=True)
        with (
            patch("odas_tpw.scor160.l2.process_l2", return_value=MagicMock()),
            patch("odas_tpw.scor160.l3.process_l3", return_value=_make_l3()),
            patch("odas_tpw.scor160.l4.process_l4", return_value=_make_l4()),
            patch(
                "odas_tpw.chi.l2_chi.process_l2_chi", side_effect=RuntimeError("chi boom")
            ),
        ):
            ds = _process_profile(l1, MagicMock(), MagicMock(), 35.0, None)

        assert ds is not None
        assert "S_gradT1" not in ds  # chi fall-through

    def test_chi_success_includes_gradt(self):
        from odas_tpw.chi.l3_chi import L3ChiData
        from odas_tpw.pyturb.eps import _process_profile

        l1 = _make_l1(has_temp_fast=True)
        n_spec, n_freq = 5, 257
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
        with (
            patch("odas_tpw.scor160.l2.process_l2", return_value=MagicMock()),
            patch("odas_tpw.scor160.l3.process_l3", return_value=_make_l3()),
            patch("odas_tpw.scor160.l4.process_l4", return_value=_make_l4()),
            patch("odas_tpw.chi.l2_chi.process_l2_chi", return_value=MagicMock()),
            patch("odas_tpw.chi.l3_chi.process_l3_chi", return_value=l3_chi),
        ):
            ds = _process_profile(l1, MagicMock(), MagicMock(), 35.0, None)

        assert ds is not None
        assert "S_gradT1" in ds
        assert "S_gradT2" in ds


# ---------------------------------------------------------------------------
# _process_one_file — .p path
# ---------------------------------------------------------------------------


def _common_kwargs(**overrides):
    base = dict(
        fft_length=512,
        diss_length=2048,
        overlap=256,
        direction="down",
        min_speed=0.2,
        min_profile_pressure=1.0,
        peaks_height=10.0,
        peaks_distance=200,
        peaks_prominence=10.0,
        despike_passes=2,
        salinity=35.0,
        overwrite=True,
        goodman=False,
    )
    base.update(overrides)
    return base


class TestProcessOneFilePFile:
    def test_no_pressure_channel(self, tmp_path):
        from odas_tpw.pyturb.eps import _process_one_file

        mock_pf = MagicMock()
        mock_pf.channels = {"T1": np.zeros(100)}  # no P / P_dP
        mock_pf.fs_fast = 512.0
        mock_pf.fs_slow = 64.0

        fake_path = tmp_path / "demo.p"
        fake_path.touch()

        with patch("odas_tpw.rsi.p_file.PFile", return_value=mock_pf):
            out = _process_one_file(fake_path, tmp_path / "out", **_common_kwargs())

        assert out == []

    def test_no_profiles_detected(self, tmp_path):
        from odas_tpw.pyturb.eps import _process_one_file

        mock_pf = MagicMock()
        mock_pf.channels = {"P": np.zeros(100)}
        mock_pf.fs_fast = 512.0
        mock_pf.fs_slow = 64.0

        fake_path = tmp_path / "demo.p"
        fake_path.touch()

        with (
            patch("odas_tpw.rsi.p_file.PFile", return_value=mock_pf),
            patch("odas_tpw.pyturb._profind.find_profiles_peaks", return_value=[]),
        ):
            out = _process_one_file(fake_path, tmp_path / "out", **_common_kwargs())

        assert out == []

    def test_normal_path_writes_per_profile(self, tmp_path):
        """One .p with 2 profiles, ``_process_profile`` returns a real dataset
        ⇒ two NetCDFs written."""
        import xarray as xr

        from odas_tpw.pyturb.eps import _process_one_file

        mock_pf = MagicMock()
        mock_pf.channels = {"P": np.linspace(0, 100, 1000)}
        mock_pf.fs_fast = 512.0
        mock_pf.fs_slow = 64.0

        fake_path = tmp_path / "demo.p"
        fake_path.touch()

        ds_out = xr.Dataset({"eps_1": (("time",), [1e-8])})
        with (
            patch("odas_tpw.rsi.p_file.PFile", return_value=mock_pf),
            patch(
                "odas_tpw.pyturb._profind.find_profiles_peaks",
                return_value=[(0, 50), (60, 100)],
            ),
            patch("odas_tpw.rsi.adapter.pfile_to_l1data", return_value=_make_l1()),
            patch("odas_tpw.pyturb.eps._process_profile", return_value=ds_out),
        ):
            out = _process_one_file(fake_path, tmp_path / "out", **_common_kwargs())

        assert len(out) == 2
        for p in out:
            assert p.exists()

    def test_skip_existing_when_no_overwrite(self, tmp_path):
        """When ``overwrite=False`` and the output already exists, the
        per-profile call is skipped — _process_profile is never invoked."""
        from odas_tpw.pyturb.eps import _process_one_file

        mock_pf = MagicMock()
        mock_pf.channels = {"P": np.linspace(0, 100, 1000)}
        mock_pf.fs_fast = 512.0
        mock_pf.fs_slow = 64.0

        fake_path = tmp_path / "demo.p"
        fake_path.touch()
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        # Pre-existing output; with overwrite=False this profile is skipped.
        (out_dir / "demo_p0000.nc").touch()

        proc = MagicMock()
        with (
            patch("odas_tpw.rsi.p_file.PFile", return_value=mock_pf),
            patch(
                "odas_tpw.pyturb._profind.find_profiles_peaks",
                return_value=[(0, 50)],
            ),
            patch("odas_tpw.rsi.adapter.pfile_to_l1data", return_value=_make_l1()),
            patch("odas_tpw.pyturb.eps._process_profile", proc),
        ):
            out = _process_one_file(
                fake_path, out_dir, **_common_kwargs(overwrite=False)
            )

        assert out == []
        proc.assert_not_called()

    def test_uses_P_dP_when_present(self, tmp_path):
        """``P_dP`` (de-spiked pressure) takes precedence over plain ``P``."""
        from odas_tpw.pyturb.eps import _process_one_file

        # Distinct pressure arrays so the find_profiles call records which one.
        P_plain = np.linspace(0, 50, 100)
        P_dp = np.linspace(0, 100, 100)

        mock_pf = MagicMock()
        mock_pf.channels = {"P": P_plain, "P_dP": P_dp}
        mock_pf.fs_fast = 512.0
        mock_pf.fs_slow = 64.0

        fake_path = tmp_path / "demo.p"
        fake_path.touch()

        find_mock = MagicMock(return_value=[])
        with (
            patch("odas_tpw.rsi.p_file.PFile", return_value=mock_pf),
            patch("odas_tpw.pyturb._profind.find_profiles_peaks", find_mock),
        ):
            _process_one_file(fake_path, tmp_path / "out", **_common_kwargs())

        # First positional arg of find_profiles_peaks should be P_dP, not P.
        called_pressure = find_mock.call_args.args[0]
        assert called_pressure is P_dp


# ---------------------------------------------------------------------------
# _process_one_file — .nc path
# ---------------------------------------------------------------------------


class TestProcessOneFileNC:
    def _make_l1_for_nc(self, *, n_time: int = 100, multi_profile: bool = False):
        """Build a real-ish L1Data."""
        from odas_tpw.scor160.io import L1Data

        # Pressure: small range for single-profile branch, large for multi.
        if multi_profile:
            t = np.arange(n_time)
            pres = 50.0 * np.abs(np.sin(2 * np.pi * t / 50))
        else:
            pres = np.linspace(2.0, 4.0, n_time)  # range = 2 dbar; below 2 * peaks_height

        return L1Data(
            time=np.arange(n_time, dtype=float),
            pres=pres,
            shear=np.zeros((2, n_time)),
            vib=np.zeros((2, n_time)),
            vib_type="ACC",
            fs_fast=512.0,
            f_AA=98.0,
            vehicle="vmp",
            profile_dir="down",
            time_reference_year=2025,
            pres_slow=pres[::8].copy() if multi_profile else np.array([]),
            fs_slow=64.0,
        )

    def test_no_pressure_data(self, tmp_path):
        from odas_tpw.pyturb.eps import _process_one_file
        from odas_tpw.scor160.io import L1Data

        empty_l1 = L1Data(
            time=np.array([]), pres=np.array([]),
            shear=np.zeros((0, 0)), vib=np.zeros((0, 0)), vib_type="ACC",
            fs_fast=512.0, f_AA=98.0, vehicle="vmp", profile_dir="down",
            time_reference_year=2025,
        )
        fake_path = tmp_path / "demo.nc"
        fake_path.touch()

        with patch("odas_tpw.rsi.adapter.nc_to_l1data", return_value=empty_l1):
            out = _process_one_file(fake_path, tmp_path / "out", **_common_kwargs())

        assert out == []

    def test_single_profile_path(self, tmp_path):
        """Pressure range below ``2 * peaks_height`` ⇒ single-profile branch."""
        import xarray as xr

        from odas_tpw.pyturb.eps import _process_one_file

        l1 = self._make_l1_for_nc(multi_profile=False)
        fake_path = tmp_path / "demo.nc"
        fake_path.touch()

        ds_out = xr.Dataset({"eps_1": (("time",), [1e-8])})
        with (
            patch("odas_tpw.rsi.adapter.nc_to_l1data", return_value=l1),
            patch("odas_tpw.pyturb.eps._process_profile", return_value=ds_out),
        ):
            out = _process_one_file(fake_path, tmp_path / "out", **_common_kwargs())

        assert len(out) == 1
        assert out[0].name == "demo_p0000.nc"

    def test_single_profile_skip_existing(self, tmp_path):
        from odas_tpw.pyturb.eps import _process_one_file

        l1 = self._make_l1_for_nc(multi_profile=False)
        fake_path = tmp_path / "demo.nc"
        fake_path.touch()
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "demo_p0000.nc").touch()

        proc = MagicMock()
        with (
            patch("odas_tpw.rsi.adapter.nc_to_l1data", return_value=l1),
            patch("odas_tpw.pyturb.eps._process_profile", proc),
        ):
            out = _process_one_file(fake_path, out_dir, **_common_kwargs(overwrite=False))

        assert out == []
        proc.assert_not_called()

    def test_multi_profile_path(self, tmp_path):
        import xarray as xr

        from odas_tpw.pyturb.eps import _process_one_file

        l1 = self._make_l1_for_nc(multi_profile=True)
        fake_path = tmp_path / "demo.nc"
        fake_path.touch()

        ds_out = xr.Dataset({"eps_1": (("time",), [1e-8])})
        with (
            patch("odas_tpw.rsi.adapter.nc_to_l1data", return_value=l1),
            patch(
                "odas_tpw.pyturb._profind.find_profiles_peaks",
                return_value=[(0, 5), (10, 15)],
            ),
            patch("odas_tpw.pyturb.eps._process_profile", return_value=ds_out),
        ):
            out = _process_one_file(fake_path, tmp_path / "out", **_common_kwargs())

        assert len(out) == 2

    def test_multi_profile_no_profiles_detected(self, tmp_path):
        from odas_tpw.pyturb.eps import _process_one_file

        l1 = self._make_l1_for_nc(multi_profile=True)
        fake_path = tmp_path / "demo.nc"
        fake_path.touch()

        with (
            patch("odas_tpw.rsi.adapter.nc_to_l1data", return_value=l1),
            patch("odas_tpw.pyturb._profind.find_profiles_peaks", return_value=[]),
        ):
            out = _process_one_file(fake_path, tmp_path / "out", **_common_kwargs())

        assert out == []

    def test_multi_profile_no_pres_slow(self, tmp_path):
        """When ``pres_slow`` is empty, fast pressure is used for peak finding."""
        from odas_tpw.pyturb.eps import _process_one_file
        from odas_tpw.scor160.io import L1Data

        n_time = 200
        t = np.arange(n_time)
        pres = 50.0 * np.abs(np.sin(2 * np.pi * t / 50))
        l1 = L1Data(
            time=np.arange(n_time, dtype=float),
            pres=pres,
            shear=np.zeros((2, n_time)),
            vib=np.zeros((2, n_time)),
            vib_type="ACC",
            fs_fast=512.0,
            f_AA=98.0,
            vehicle="vmp",
            profile_dir="down",
            time_reference_year=2025,
            pres_slow=np.array([]),  # empty triggers fast-pressure fallback
            fs_slow=0.0,
        )

        fake_path = tmp_path / "demo.nc"
        fake_path.touch()

        find_mock = MagicMock(return_value=[])
        with (
            patch("odas_tpw.rsi.adapter.nc_to_l1data", return_value=l1),
            patch("odas_tpw.pyturb._profind.find_profiles_peaks", find_mock),
        ):
            _process_one_file(fake_path, tmp_path / "out", **_common_kwargs())

        # Called with fast-rate pressure, not pres_slow.
        called_pressure = find_mock.call_args.args[0]
        assert called_pressure is l1.pres


# ---------------------------------------------------------------------------
# _process_one_file — unsupported format
# ---------------------------------------------------------------------------


class TestProcessOneFileUnsupported:
    def test_unsupported_extension(self, tmp_path):
        from odas_tpw.pyturb.eps import _process_one_file

        f = tmp_path / "demo.txt"
        f.touch()

        out = _process_one_file(f, tmp_path / "out", **_common_kwargs())
        assert out == []


# ---------------------------------------------------------------------------
# run_eps — top-level CLI entry
# ---------------------------------------------------------------------------


def _eps_args(tmp_path: Path, files: list[str], n_workers: int = 1, **overrides):
    base = dict(
        files=files,
        output=str(tmp_path / "out"),
        fft_len=1.0,
        diss_len=4.0,
        goodman=False,
        direction="down",
        min_speed=0.2,
        min_profile_pressure=1.0,
        peaks_height=25.0,
        peaks_distance=200,
        peaks_prominence=25.0,
        despike_passes=2,
        salinity=35.0,
        overwrite=True,
        n_workers=n_workers,
        aux=None,
        aux_lat="lat",
        aux_lon="lon",
        aux_temp="temperature",
        aux_sal="salinity",
        aux_dens="density",
        pressure_smoothing=0.25,
    )
    base.update(overrides)
    return Namespace(**base)


class TestRunEps:
    def test_no_files_logs_error(self, tmp_path, caplog):
        from odas_tpw.pyturb.eps import run_eps

        args = _eps_args(tmp_path, files=["nonexistent_pattern_*.p"])
        with caplog.at_level("ERROR", logger="odas_tpw.pyturb.eps"):
            run_eps(args)
        assert "No input files found" in caplog.text

    def test_serial_path_calls_process_one_file(self, tmp_path):
        from odas_tpw.pyturb.eps import run_eps

        f = tmp_path / "a.p"
        f.touch()
        args = _eps_args(tmp_path, files=[str(f)], n_workers=1)

        proc = MagicMock(return_value=[tmp_path / "out" / "a_p0000.nc"])
        with patch("odas_tpw.pyturb.eps._process_one_file", proc):
            run_eps(args)

        proc.assert_called_once()
        # First positional arg is the file path.
        assert proc.call_args.args[0] == f

    def test_serial_path_logs_exception(self, tmp_path, caplog):
        from odas_tpw.pyturb.eps import run_eps

        f = tmp_path / "a.p"
        f.touch()
        args = _eps_args(tmp_path, files=[str(f)], n_workers=1)

        with (
            patch(
                "odas_tpw.pyturb.eps._process_one_file",
                side_effect=RuntimeError("worker boom"),
            ),
            caplog.at_level("ERROR", logger="odas_tpw.pyturb.eps"),
        ):
            run_eps(args)

        assert "worker boom" in caplog.text

    def test_parallel_path_calls_process_one_file(self, tmp_path):
        from odas_tpw.pyturb.eps import run_eps

        files = [tmp_path / f"f{i}.p" for i in range(3)]
        for f in files:
            f.touch()
        args = _eps_args(tmp_path, files=[str(f) for f in files], n_workers=2)

        # Mock the executor to run sequentially in-process.
        class _Executor:
            def __init__(self, *a, **kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def submit(self, fn, *a, **kw):
                fut = MagicMock()
                fut.result.return_value = fn(*a, **kw)
                return fut

        proc = MagicMock(return_value=[])
        with (
            patch("odas_tpw.pyturb.eps.ProcessPoolExecutor", _Executor),
            patch("odas_tpw.pyturb.eps.as_completed", lambda fs: list(fs)),
            patch("odas_tpw.pyturb.eps._process_one_file", proc),
        ):
            run_eps(args)

        assert proc.call_count == 3

    def test_parallel_worker_exception_logged(self, tmp_path, caplog):
        from odas_tpw.pyturb.eps import run_eps

        f = tmp_path / "a.p"
        f.touch()
        args = _eps_args(tmp_path, files=[str(f)], n_workers=2)

        class _Future:
            def result(self):
                raise RuntimeError("parallel boom")

        class _Executor:
            def __init__(self, *a, **kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def submit(self, fn, *a, **kw):
                return _Future()

        with (
            patch("odas_tpw.pyturb.eps.ProcessPoolExecutor", _Executor),
            patch("odas_tpw.pyturb.eps.as_completed", lambda fs: list(fs)),
            caplog.at_level("ERROR", logger="odas_tpw.pyturb.eps"),
        ):
            run_eps(args)

        assert "parallel boom" in caplog.text

    def test_aux_path_loads_auxiliary_data(self, tmp_path):
        """When ``--aux`` is set, ``aux_path`` and ``aux_kwargs`` are forwarded
        through to ``_process_one_file`` for use by ``load_auxiliary``."""
        import xarray as xr

        from odas_tpw.pyturb.eps import run_eps

        # Real auxiliary NetCDF so load_auxiliary inside _process_one_file
        # works without needing to patch it specifically.
        aux_path = tmp_path / "aux.nc"
        xr.Dataset(
            {
                "lat": (("time",), [0.0]),
                "lon": (("time",), [0.0]),
                "temperature": (("time",), [15.0]),
                "salinity": (("time",), [35.0]),
                "density": (("time",), [1025.0]),
            }
        ).to_netcdf(aux_path)

        f = tmp_path / "a.p"
        f.touch()
        args = _eps_args(tmp_path, files=[str(f)], n_workers=1, aux=str(aux_path))

        proc = MagicMock(return_value=[])
        with patch("odas_tpw.pyturb.eps._process_one_file", proc):
            run_eps(args)

        kwargs = proc.call_args.kwargs
        assert kwargs["aux_path"] == str(aux_path)
        assert kwargs["aux_kwargs"]["lat_var"] == "lat"

    def test_glob_pattern_expansion(self, tmp_path, monkeypatch):
        """A pattern (no exact file) is glob-expanded relative to cwd."""
        from odas_tpw.pyturb.eps import run_eps

        # Two files in a subdir; cwd onto tmp_path so glob "sub/*.p" matches.
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "a.p").touch()
        (sub / "b.p").touch()
        monkeypatch.chdir(tmp_path)

        args = _eps_args(tmp_path, files=["sub/*.p"], n_workers=1)
        proc = MagicMock(return_value=[])
        with patch("odas_tpw.pyturb.eps._process_one_file", proc):
            run_eps(args)

        assert proc.call_count == 2


# ---------------------------------------------------------------------------
# Sanity: the glue helpers used by _process_profile work end-to-end with
# successful chain mocks (covers the chi-disabled code path).
# ---------------------------------------------------------------------------


class TestProcessProfileChainSuccess:
    def test_no_temp_fast_returns_dataset(self):
        """Without fast temperature, chi is skipped and a shear-only Dataset
        comes back."""
        from odas_tpw.pyturb.eps import _process_profile

        l1 = _make_l1(has_temp_fast=False)
        with (
            patch("odas_tpw.scor160.l2.process_l2", return_value=MagicMock()),
            patch("odas_tpw.scor160.l3.process_l3", return_value=_make_l3()),
            patch("odas_tpw.scor160.l4.process_l4", return_value=_make_l4()),
        ):
            ds = _process_profile(l1, MagicMock(), MagicMock(), 35.0, None)

        assert ds is not None
        assert "eps_1" in ds
        assert "S_gradT1" not in ds  # chi was skipped
