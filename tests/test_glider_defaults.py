# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""Glider-correct defaults and CLI plumbing (issue #131 W1b, M1-M5/M8-M9/M13).

Covers: vehicle-resolved W_min (prof/eps/chi/run_pipeline incl. L2Params),
`rsi-tpw prof` vehicle resolution (PFile and NetCDF sources), the
--speed-method/--aoa plumbing on eps/chi with speed_source provenance,
honest empty results (explain_no_profiles warning + eps/chi exit 1), and
startup-file batch robustness for info/prof.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

_DATA_DIR = Path(__file__).parent / "data"
_MR_FILE = _DATA_DIR / "MR_SL435.p"
_STARTUP_FILE = _DATA_DIR / "VMP142_startup_noclock.p"


def _require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"test fixture {path.name} not available")
    return path


# ---------------------------------------------------------------------------
# Synthetic NetCDF builders
# ---------------------------------------------------------------------------


def _write_profile_nc(
    path: Path,
    *,
    u_em: float | None = 0.43,
    incl: bool = False,
    speed_fast: float | None = None,
    speed_attrs: dict | None = None,
    therm: bool = False,
) -> Path:
    """Per-profile NC (is_profile=True): 8 s at 512/64 Hz, P descending
    0.5 dbar/s, shear+accel noise, optional U_EM / inclinometers /
    precomputed speed_fast / perturb-style speed provenance attrs."""
    import netCDF4 as nc

    fs_fast, fs_slow = 512.0, 64.0
    dur = 8.0
    n_fast, n_slow = int(dur * fs_fast), int(dur * fs_slow)
    rng = np.random.default_rng(7)

    ds = nc.Dataset(str(path), "w", format="NETCDF4")
    ds.createDimension("time_fast", n_fast)
    ds.createDimension("time_slow", n_slow)
    ds.createVariable("t_fast", "f8", ("time_fast",))[:] = np.arange(n_fast) / fs_fast
    ds.createVariable("t_slow", "f8", ("time_slow",))[:] = np.arange(n_slow) / fs_slow
    ds.createVariable("P", "f8", ("time_slow",))[:] = (
        10.0 + 0.5 * np.arange(n_slow) / fs_slow
    )
    ds.createVariable("T1", "f8", ("time_slow",))[:] = np.linspace(10.0, 11.0, n_slow)
    for name in ("sh1", "sh2"):
        ds.createVariable(name, "f8", ("time_fast",))[:] = (
            rng.standard_normal(n_fast) * 0.01
        )
    for name in ("Ax", "Ay"):
        ds.createVariable(name, "f8", ("time_fast",))[:] = (
            rng.standard_normal(n_fast) * 0.001
        )
    if u_em is not None:
        ds.createVariable("U_EM", "f8", ("time_slow",))[:] = np.full(n_slow, u_em)
    if incl:
        # MR convention (CLAUDE.md): Incl_Y ~ pitch, Incl_X mostly roll. A
        # small pitch wobble gives Incl_Y the larger percentile spread so
        # the flight model's pitch-axis auto-pick genuinely selects it.
        ds.createVariable("Incl_X", "f8", ("time_slow",))[:] = np.full(n_slow, 2.0)
        ds.createVariable("Incl_Y", "f8", ("time_slow",))[:] = (
            -30.0 + 0.5 * np.sin(2 * np.pi * np.arange(n_slow) / n_slow)
        )
    if speed_fast is not None:
        ds.createVariable("speed_fast", "f8", ("time_fast",))[:] = np.full(
            n_fast, speed_fast
        )
    if therm:
        ds.createVariable("T1_dT1", "f8", ("time_fast",))[:] = (
            np.linspace(10.0, 11.0, n_fast) + rng.standard_normal(n_fast) * 1e-3
        )
    ds.fs_fast = fs_fast
    ds.fs_slow = fs_slow
    ds.profile_number = 1
    ds.instrument_model = "MR1000RDL-EM"
    ds.instrument_sn = "435"
    ds.start_time = "2026-01-01T00:00:00"
    for k, v in (speed_attrs or {}).items():
        setattr(ds, k, v)
    ds.close()
    return path


def _write_glider_full_nc(path: Path, platform_type: str | None = "slocum_glider") -> Path:
    """Full-record NC: V-shaped glider cast (30 s down + 30 s up at
    0.12 dbar/s — above the glide floor 0.05, below the VMP floor 0.3)."""
    import netCDF4 as nc

    fs_slow, fs_fast = 8.0, 64.0
    leg = 30.0
    n_slow, n_fast = int(2 * leg * fs_slow), int(2 * leg * fs_fast)
    t_slow = np.arange(n_slow) / fs_slow
    rng = np.random.default_rng(11)

    ds = nc.Dataset(str(path), "w", format="NETCDF4")
    ds.createDimension("time_fast", n_fast)
    ds.createDimension("time_slow", n_slow)
    ds.createVariable("t_fast", "f8", ("time_fast",))[:] = np.arange(n_fast) / fs_fast
    ds.createVariable("t_slow", "f8", ("time_slow",))[:] = t_slow
    P = np.where(t_slow < leg, 1.0 + 0.12 * t_slow, 1.0 + 0.12 * leg - 0.12 * (t_slow - leg))
    ds.createVariable("P", "f8", ("time_slow",))[:] = P
    ds.createVariable("T1", "f8", ("time_slow",))[:] = np.linspace(10.0, 11.0, n_slow)
    ds.createVariable("sh1", "f8", ("time_fast",))[:] = rng.standard_normal(n_fast) * 0.01
    ds.createVariable("Ax", "f8", ("time_fast",))[:] = rng.standard_normal(n_fast) * 0.001
    ds.fs_fast = fs_fast
    ds.fs_slow = fs_slow
    ds.instrument_model = "MR1000"
    if platform_type is not None:
        ds.platform_type = platform_type
    ds.close()
    return path


# ---------------------------------------------------------------------------
# rsi-tpw prof: vehicle resolution (M13, F2/F3)
# ---------------------------------------------------------------------------


class TestProfVehicleResolution:
    def test_prof_cli_vehicle_flag_no_crash(self, monkeypatch, tmp_path, capsys):
        """`rsi-tpw prof --vehicle slocum_glider` used to crash with a
        TypeError (get_profiles got an unexpected 'vehicle' kwarg); it now
        resolves direction=glide and the 0.05 dbar/s floor."""
        _require(_MR_FILE)
        captured: dict = {}

        def fake_get_profiles(P, W, fs, **kwargs):
            captured.update(kwargs)
            return []

        monkeypatch.setattr("odas_tpw.rsi.profile.get_profiles", fake_get_profiles)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "rsi-tpw", "prof", str(_MR_FILE),
                "--vehicle", "slocum_glider",
                "-o", str(tmp_path),
            ],
        )
        from odas_tpw.rsi.cli import main

        main()  # must not raise
        assert captured["direction"] == "glide"
        assert captured["W_min"] == 0.05

    def test_prof_nc_source_resolves_vehicle_and_w_min(self, tmp_path):
        """Full-record NC with platform_type=slocum_glider: direction auto
        resolves to glide (down + up detected) and W_min auto to 0.05 —
        with the old fixed 0.3 default this cast produced nothing (F2)."""
        from odas_tpw.rsi.profile import extract_profiles

        nc_path = _write_glider_full_nc(tmp_path / "glider_full.nc")
        paths = extract_profiles(nc_path, tmp_path / "out")
        assert len(paths) == 2  # one down + one up profile

    def test_prof_nc_explicit_w_min_passes_through(self, tmp_path):
        """An explicit W_min bypasses the auto floor exactly (here it
        rejects the slow glider cast, proving 0.3 was really applied)."""
        from odas_tpw.rsi.profile import extract_profiles

        nc_path = _write_glider_full_nc(tmp_path / "glider_full.nc")
        assert extract_profiles(nc_path, tmp_path / "out", W_min=0.3) == []

    def test_extract_profiles_extra_attrs(self, tmp_path):
        """extra_attrs (perturb speed provenance) land as global attrs on
        every per-profile NetCDF (F12)."""
        import netCDF4 as nc

        from odas_tpw.rsi.profile import extract_profiles

        nc_path = _write_glider_full_nc(tmp_path / "glider_full.nc")
        paths = extract_profiles(
            nc_path,
            tmp_path / "out",
            extra_attrs={"speed_method": "em", "speed_source": "em"},
        )
        assert paths
        ds = nc.Dataset(str(paths[0]), "r")
        try:
            assert ds.speed_method == "em"
            assert ds.speed_source == "em"
        finally:
            ds.close()


# ---------------------------------------------------------------------------
# info/prof batch robustness (M5, F16)
# ---------------------------------------------------------------------------


class TestBatchRobustness:
    def test_info_continues_past_startup_file(self, monkeypatch, capsys):
        _require(_STARTUP_FILE)
        _require(_MR_FILE)
        monkeypatch.setattr(
            sys, "argv", ["rsi-tpw", "info", str(_STARTUP_FILE), str(_MR_FILE)]
        )
        from odas_tpw.rsi.cli import main

        main()  # one good file -> exit 0
        out = capsys.readouterr().out
        assert "ERROR:" in out
        assert "1 of 2 file(s) failed" in out

    def test_info_all_failed_exits_1(self, monkeypatch, capsys):
        _require(_STARTUP_FILE)
        monkeypatch.setattr(sys, "argv", ["rsi-tpw", "info", str(_STARTUP_FILE)])
        from odas_tpw.rsi.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        assert "ERROR:" in capsys.readouterr().out

    def test_prof_continues_past_startup_file(self, monkeypatch, tmp_path, capsys):
        _require(_STARTUP_FILE)
        _require(_MR_FILE)
        monkeypatch.setattr(
            sys,
            "argv",
            ["rsi-tpw", "prof", str(_STARTUP_FILE), str(_MR_FILE), "-o", str(tmp_path)],
        )
        from odas_tpw.rsi.cli import main

        main()  # one good file -> exit 0
        out = capsys.readouterr().out
        assert "ERROR:" in out
        assert "1 of 2 file(s) failed" in out


# ---------------------------------------------------------------------------
# eps/chi speed methods + speed_source provenance (M1/M2/M8, F8/F9/F12)
# ---------------------------------------------------------------------------


class TestSpeedMethods:
    def test_em_speed_and_provenance(self, tmp_path):
        from odas_tpw.rsi.dissipation import _compute_epsilon

        nc_path = _write_profile_nc(tmp_path / "prof.nc")
        results = _compute_epsilon(nc_path, speed_method="em")
        assert len(results) == 1
        ds = results[0]
        assert ds.attrs["speed_source"] == "em (U_EM)"
        assert ds.attrs["speed_method"] == "em"
        np.testing.assert_allclose(float(ds["speed"].median()), 0.43, atol=1e-3)

    def test_chi_em_speed_and_provenance(self, tmp_path):
        from odas_tpw.rsi.chi_io import _compute_chi

        nc_path = _write_profile_nc(tmp_path / "prof.nc", therm=True)
        results = _compute_chi(nc_path, speed_method="em")
        assert len(results) == 1
        ds = results[0]
        assert ds.attrs["speed_source"] == "em (U_EM)"
        np.testing.assert_allclose(float(ds["speed"].median()), 0.43, atol=1e-3)

    def test_flight_speed_and_provenance(self, tmp_path):
        from odas_tpw.rsi.dissipation import _compute_epsilon

        nc_path = _write_profile_nc(tmp_path / "prof.nc", u_em=None, incl=True)
        ds = _compute_epsilon(nc_path, speed_method="flight")[0]
        assert ds.attrs["speed_source"] == "flight (aoa=3)"
        assert ds.attrs["speed_method"] == "flight"
        # pitch ~ -30 deg (on Incl_Y), aoa 3 deg, W = 0.5 dbar/s
        # -> U ~ 0.5 / sin(27 deg); the +-0.5 deg pitch wobble allows ~2%.
        np.testing.assert_allclose(
            float(ds["speed"].median()), 0.5 / np.sin(np.deg2rad(27.0)), rtol=0.02
        )

    def test_fixed_speed_provenance(self, tmp_path):
        from odas_tpw.rsi.dissipation import _compute_epsilon

        nc_path = _write_profile_nc(tmp_path / "prof.nc")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = _compute_epsilon(nc_path, speed=0.3)[0]
        assert ds.attrs["speed_source"] == "fixed --speed 0.3"
        np.testing.assert_allclose(float(ds["speed"].median()), 0.3, atol=1e-6)

    def test_pressure_default_provenance_and_em_hint(self, tmp_path):
        """The historical pressure path stamps its provenance, and on a
        glide-direction source with U_EM present the bias warning names the
        concrete fix (M2)."""
        from odas_tpw.rsi.dissipation import _compute_epsilon

        nc_path = _write_profile_nc(tmp_path / "prof.nc")
        with pytest.warns(UserWarning, match=r"pass --speed-method em \(U_EM present\)"):
            ds = _compute_epsilon(nc_path)[0]
        assert ds.attrs["speed_source"] == "pressure |dP/dt|"
        np.testing.assert_allclose(float(ds["speed"].median()), 0.5, atol=1e-2)

    def test_precomputed_provenance_enriched(self, tmp_path):
        """A perturb per-profile NC (speed_fast + speed_method/speed_source
        attrs) yields the enriched provenance string (F12)."""
        from odas_tpw.rsi.dissipation import _compute_epsilon

        nc_path = _write_profile_nc(
            tmp_path / "prof.nc",
            speed_fast=0.37,
            speed_attrs={"speed_method": "em", "speed_source": "em"},
        )
        ds = _compute_epsilon(nc_path)[0]
        assert ds.attrs["speed_source"] == "precomputed speed_fast (perturb speed.method=em)"
        np.testing.assert_allclose(float(ds["speed"].median()), 0.37, atol=1e-6)

    def test_precomputed_provenance_plain(self, tmp_path):
        from odas_tpw.rsi.dissipation import _compute_epsilon

        nc_path = _write_profile_nc(tmp_path / "prof.nc", speed_fast=0.37)
        ds = _compute_epsilon(nc_path)[0]
        assert ds.attrs["speed_source"] == "precomputed speed_fast channel"

    def test_precomputed_hotel_provenance_lands_in_diss_attrs(self, tmp_path):
        """End-to-end for the perturb hotel speed method (#131 M10): the
        'hotel:<var>' source string stamped on a perturb per-profile NC rides
        the W1b precomputed-speed mechanism into the diss product attrs —
        the upstream source is retained because it says more than the
        method name (it names the hotel channel)."""
        from odas_tpw.rsi.dissipation import _compute_epsilon

        nc_path = _write_profile_nc(
            tmp_path / "prof.nc",
            speed_fast=0.37,
            speed_attrs={"speed_method": "hotel", "speed_source": "hotel:speed"},
        )
        ds = _compute_epsilon(nc_path)[0]
        assert ds.attrs["speed_method"] == "hotel"
        assert "hotel:speed" in ds.attrs["speed_source"]
        assert ds.attrs["speed_source"] == (
            "precomputed speed_fast (perturb speed.method=hotel, source=hotel:speed)"
        )
        np.testing.assert_allclose(float(ds["speed"].median()), 0.37, atol=1e-6)

    def test_hotel_method_rejected_with_perturb_hint(self, tmp_path):
        """The rsi layer never sees hotel channels (they are merged in
        perturb), so speed_method='hotel' is rejected with a hint pointing
        at the perturb pipeline instead of a bare vocabulary error."""
        from odas_tpw.rsi.dissipation import _compute_epsilon

        nc_path = _write_profile_nc(tmp_path / "prof.nc")
        with pytest.raises(ValueError, match=r"hotel speed method is perturb-only"):
            _compute_epsilon(nc_path, speed_method="hotel")

    def test_explicit_pressure_overrides_precomputed(self, tmp_path):
        """An EXPLICIT --speed-method pressure forces the |dP/dt| path even
        when the source carries a precomputed speed_fast channel; only the
        default (None) prefers the precomputed channel."""
        from odas_tpw.rsi.dissipation import _compute_epsilon

        nc_path = _write_profile_nc(
            tmp_path / "prof.nc",
            speed_fast=0.37,
            speed_attrs={"speed_method": "em", "speed_source": "em"},
        )
        ds = _compute_epsilon(nc_path, speed_method="pressure")[0]
        assert ds.attrs["speed_source"] == "pressure |dP/dt|"
        assert "speed_method" not in ds.attrs  # stale upstream attr cleared
        np.testing.assert_allclose(float(ds["speed"].median()), 0.5, atol=1e-2)

    def test_fixed_speed_clears_stale_speed_method(self, tmp_path):
        """A fixed --speed must not ship next to a stale upstream
        speed_method attr (contradictory provenance)."""
        from odas_tpw.rsi.dissipation import _compute_epsilon

        nc_path = _write_profile_nc(
            tmp_path / "prof.nc",
            speed_fast=0.37,
            speed_attrs={"speed_method": "em", "speed_source": "em"},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = _compute_epsilon(nc_path, speed=0.6)[0]
        assert ds.attrs["speed_source"] == "fixed --speed 0.6"
        assert "speed_method" not in ds.attrs

    def test_constant_method_rejected_with_hint(self, tmp_path):
        """'constant' is compute_speed_for_pfile vocabulary (perturb has the
        value plumbing); here the fixed --speed is the equivalent, so the
        error must say so instead of 'speed.value is null'."""
        from odas_tpw.rsi.dissipation import _compute_epsilon

        nc_path = _write_profile_nc(tmp_path / "prof.nc")
        with pytest.raises(ValueError, match=r"Use --speed for a fixed"):
            _compute_epsilon(nc_path, speed_method="constant")

    def test_em_missing_u_em_errors(self, tmp_path):
        from odas_tpw.rsi.dissipation import _compute_epsilon

        nc_path = _write_profile_nc(tmp_path / "prof.nc", u_em=None, incl=True)
        with pytest.raises(ValueError, match="U_EM is missing from the source"):
            _compute_epsilon(nc_path, speed_method="em")

    def test_speed_and_method_mutually_exclusive(self, tmp_path):
        from odas_tpw.rsi.chi_io import _compute_chi
        from odas_tpw.rsi.dissipation import _compute_epsilon

        nc_path = _write_profile_nc(tmp_path / "prof.nc", therm=True)
        with pytest.raises(ValueError, match="mutually exclusive"):
            _compute_epsilon(nc_path, speed=0.3, speed_method="em")
        with pytest.raises(ValueError, match="mutually exclusive"):
            _compute_chi(nc_path, speed=0.3, speed_method="em")


# ---------------------------------------------------------------------------
# Honest empty results (M4, F11)
# ---------------------------------------------------------------------------


class TestEmptyResults:
    def test_prepare_profiles_warns_with_observed_rate(self, tmp_path):
        """No profiles -> explain_no_profiles text (observed peak rate vs
        W_min) reaches the caller as a warning, and the result is empty."""
        from odas_tpw.rsi.dissipation import _compute_epsilon

        nc_path = _write_glider_full_nc(tmp_path / "glider_full.nc")
        with pytest.warns(UserWarning, match=r"No profiles detected .*peak"):
            results = _compute_epsilon(nc_path, W_min=99.0)
        assert results == []

    def test_eps_cli_exit_1_when_nothing_produced(self, monkeypatch, tmp_path, capsys):
        nc_path = _write_glider_full_nc(tmp_path / "glider_full.nc")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "rsi-tpw", "eps", str(nc_path),
                "--W-min", "99", "-o", str(tmp_path / "eps"),
            ],
        )
        from odas_tpw.rsi.cli import main

        with pytest.raises(SystemExit) as exc_info, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            main()
        assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "no profiles detected" in out
        assert "0 of 1 file(s) produced output" in out


# ---------------------------------------------------------------------------
# run_pipeline: resolved W_min reaches detection AND L2Params (M3/M7, F1)
# ---------------------------------------------------------------------------


class TestRunPipelineWmin:
    def _run(self, monkeypatch, tmp_path, **kwargs):
        _require(_MR_FILE)
        detect: dict = {}
        l2_params_seen: list = []

        def fake_get_profiles(P, W, fs, **kw):
            detect.update(kw)
            return [(0, len(np.asarray(P)) - 1)]

        def fake_process_profile(*args, **kw):
            l2_params_seen.append(kw["l2_params"])
            return None

        monkeypatch.setattr("odas_tpw.rsi.profile.get_profiles", fake_get_profiles)
        monkeypatch.setattr(
            "odas_tpw.rsi.pipeline._process_profile", fake_process_profile
        )
        from odas_tpw.rsi.pipeline import run_pipeline

        run_pipeline([_MR_FILE], tmp_path, **kwargs)
        assert l2_params_seen, "pipeline never reached _process_profile"
        return detect, l2_params_seen[0]

    def test_auto_w_min_resolves_glide_floor(self, monkeypatch, tmp_path):
        detect, l2_params = self._run(monkeypatch, tmp_path)
        assert detect["direction"] == "glide"  # MR_SL435 declares slocum_glider
        assert detect["W_min"] == 0.05
        assert l2_params.profile_min_W == 0.05

    def test_explicit_w_min_passes_through(self, monkeypatch, tmp_path):
        detect, l2_params = self._run(monkeypatch, tmp_path, W_min=0.2)
        assert detect["W_min"] == 0.2
        assert l2_params.profile_min_W == 0.2


# ---------------------------------------------------------------------------
# pipeline CLI routes YAML epsilon.speed_method/aoa_deg/W_min (F10)
# ---------------------------------------------------------------------------


def test_pipeline_cli_routes_yaml_speed_method(monkeypatch, tmp_path, capsys):
    _require(_MR_FILE)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "epsilon:\n  speed_method: em\n  aoa_deg: 4.5\n  W_min: 0.07\n"
        "chi:\n  speed_method: em\n  aoa_deg: 4.5\n  W_min: 0.07\n"
    )
    captured: dict = {}

    def fake_run_pipeline(p_files, output_dir, **kwargs):
        captured.update(kwargs)
        return output_dir

    monkeypatch.setattr("odas_tpw.rsi.pipeline.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rsi-tpw", "-c", str(cfg), "pipeline",
            str(_MR_FILE), "-o", str(tmp_path / "out"),
        ],
    )
    from odas_tpw.rsi.cli import main

    main()
    assert captured["speed_method"] == "em"
    assert captured["aoa_deg"] == 4.5
    assert captured["W_min"] == 0.07
    # Matching [chi] values -> no divergence warning.
    assert "is ignored in favor of" not in capsys.readouterr().err


def test_pipeline_cli_warns_on_chi_speed_method_divergence(monkeypatch, tmp_path, capsys):
    """The pipeline uses the [epsilon] speed_method for both stages; a
    differing [chi] value must warn instead of being silently dropped."""
    _require(_MR_FILE)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("chi:\n  speed_method: em\n")
    monkeypatch.setattr("odas_tpw.rsi.pipeline.run_pipeline", lambda *a, **k: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rsi-tpw", "-c", str(cfg), "pipeline",
            str(_MR_FILE), "-o", str(tmp_path / "out"),
        ],
    )
    from odas_tpw.rsi.cli import main

    main()
    err = capsys.readouterr().err
    assert "speed_method" in err
    assert "is ignored in favor of" in err


def test_run_pipeline_rejects_speed_and_method(tmp_path):
    """run_pipeline enforces the same speed/speed_method exclusivity as the
    eps/chi commands (the adapter would otherwise silently prefer speed)."""
    from odas_tpw.rsi.pipeline import run_pipeline

    with pytest.raises(ValueError, match="mutually exclusive"):
        run_pipeline([Path("does_not_exist.p")], tmp_path, speed=0.6, speed_method="em")
