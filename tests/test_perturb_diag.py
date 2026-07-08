# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb.diag — the interactive dissipation inspector.

Interaction is exercised headlessly (Agg backend): the overview loaders and the
cell -> file/window mapping are unit-tested, and the single-figure inspector is
driven by calling ``select`` / ``snapshot`` directly rather than through GUI
events.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg")


def _write_combo(path, *, n_bin=5, stimes=(1000.0, 1000.5), qc_cell=None):
    """A minimal diss_combo: epsilonMean/e_1/e_2/qc over (bin, profile)."""
    bins = np.linspace(10.0, 100.0, n_bin)
    nprof = len(stimes)
    rng = np.random.default_rng(0)
    eps = 10.0 ** rng.uniform(-9.0, -5.0, size=(n_bin, nprof))
    qc = np.zeros((n_bin, nprof))
    if qc_cell is not None:
        qc[qc_cell] = 1.0
    ds = xr.Dataset(
        {
            "epsilonMean": (("bin", "profile"), eps),
            "e_1": (("bin", "profile"), eps * 1.1),
            "e_2": (("bin", "profile"), eps * 0.9),
            "qc_drop_epsilon": (("bin", "profile"), qc),
            "stime": (("profile",), np.asarray(stimes, dtype=float)),
        },
        coords={"bin": bins, "profile": np.arange(nprof)},
    )
    ds.to_netcdf(path)
    return bins, np.asarray(stimes, dtype=float), eps


def _write_diss(path, *, stime, pressures=(10.0, 50.0, 100.0), n_freq=8,
                n_probe=2, lat=0.0, diss_length=1024, fs_fast=512.0):
    """A minimal per-profile diss file matching the real schema."""
    P = np.asarray(pressures, dtype=float)
    n_win = P.size
    K = np.tile(np.linspace(1.0, 100.0, n_freq)[:, None], (1, n_win))
    spec = np.ones((n_probe, n_freq, n_win))
    eps = np.full((n_probe, n_win), 1e-7)
    ds = xr.Dataset(
        {
            "epsilon": (("probe", "time"), eps),
            "K_max": (("probe", "time"), np.full((n_probe, n_win), 30.0)),
            "method": (("probe", "time"), np.ones((n_probe, n_win))),
            "FM": (("probe", "time"), np.full((n_probe, n_win), 1.0)),
            "fom": (("probe", "time"), np.full((n_probe, n_win), 1.0)),
            "speed": (("time",), np.full(n_win, 0.7)),
            "nu": (("time",), np.full(n_win, 1e-6)),
            "P_mean": (("time",), P),
            "T_mean": (("time",), np.full(n_win, 15.0)),
            "spec_shear": (("probe", "freq", "time"), spec),
            "spec_nasmyth": (("probe", "freq", "time"), spec * 0.5),
            "K": (("freq", "time"), K),
            "F": (("freq", "time"), K),
            "e_1": (("time",), eps[0]),
            "e_2": (("time",), eps[1]),
            "epsilonMean": (("time",), eps.mean(0)),
            "epsilonLnSigma": (("time",), np.full(n_win, 0.2)),
            "lat": ((), float(lat)),
            "stime": ((), float(stime)),
        },
        coords={"probe": np.arange(n_probe)},
    )
    ds.attrs["diss_length"] = diss_length
    ds.attrs["fs_fast"] = fs_fast
    ds.to_netcdf(path)
    return P


def _write_chi(path, *, stime, pressures=(10.0, 50.0, 100.0), n_freq=8,
               n_probe=2, lat=0.0, diss_length=1024, fs_fast=512.0,
               with_mixing=False, spectrum_model="batchelor",
               fit_method="iterative", fp07_model="single_pole"):
    """A minimal per-profile chi file matching the real chi_NN schema.

    ``with_mixing`` appends the (window,) mixing quantities a mixing run writes
    (N2/dTdz/K_T/K_rho/Gamma), so the mixing-strip panels render real lines.
    """
    P = np.asarray(pressures, dtype=float)
    n = P.size
    K = np.tile(np.linspace(1.0, 100.0, n_freq)[:, None], (1, n))
    spec = np.ones((n_probe, n_freq, n))
    data = {
        "chi": (("probe", "time"), np.full((n_probe, n), 1e-8)),
        "epsilon_T": (("probe", "time"), np.full((n_probe, n), 1e-8)),
        "fom": (("probe", "time"), np.full((n_probe, n), 1.0)),
        "kB": (("probe", "time"), np.full((n_probe, n), 150.0)),
        "K_max_T": (("probe", "time"), np.full((n_probe, n), 80.0)),
        "K_max_ratio": (("probe", "time"), np.full((n_probe, n), 0.5)),
        "speed": (("time",), np.full(n, 0.7)),
        "nu": (("time",), np.full(n, 1e-6)),
        "P_mean": (("time",), P),
        "T_mean": (("time",), np.full(n, 15.0)),
        "spec_gradT": (("probe", "freq", "time"), spec),
        "spec_batch": (("probe", "freq", "time"), spec * 0.8),
        "spec_noise": (("probe", "freq", "time"), spec * 0.01),
        "K": (("freq", "time"), K),
        "F": (("freq", "time"), K),
        "lat": ((), float(lat)),
        "stime": ((), float(stime)),
    }
    if with_mixing:
        data.update({
            "N2": (("time",), np.full(n, 1e-4)),
            "dTdz": (("time",), np.full(n, 0.05)),
            "K_T": (("time",), np.full(n, 2e-6)),
            "K_rho": (("time",), np.full(n, 1e-5)),
            "Gamma": (("time",), np.full(n, 0.2)),
        })
    ds = xr.Dataset(data, coords={"probe": np.arange(n_probe)})
    ds.attrs["diss_length"] = diss_length
    ds.attrs["fs_fast"] = fs_fast
    ds.attrs["spectrum_model"] = spectrum_model
    ds.attrs["fit_method"] = fit_method
    ds.attrs["fp07_model"] = fp07_model
    ds.to_netcdf(path)
    return P


def _write_chi_combo(path, *, stimes=(1000.0, 1000.5), n_bin=5):
    """A minimal chi_combo: chiMean/chi_1/chi_2/qc over (bin, profile)."""
    bins = np.linspace(10.0, 100.0, n_bin)
    nprof = len(stimes)
    chi = 10.0 ** np.random.default_rng(1).uniform(-10.0, -7.0, size=(n_bin, nprof))
    xr.Dataset(
        {
            "chiMean": (("bin", "profile"), chi),
            "chi_1": (("bin", "profile"), chi * 1.1),
            "chi_2": (("bin", "profile"), chi * 0.9),
            "qc_drop_chi": (("bin", "profile"), np.zeros((n_bin, nprof))),
            "stime": (("profile",), np.asarray(stimes, dtype=float)),
        },
        coords={"bin": bins, "profile": np.arange(nprof)},
    ).to_netcdf(path)
    return bins, np.asarray(stimes, dtype=float)


def _write_mixing_combo(path, *, stimes=(1000.0, 1000.5), n_bin=5):
    """A minimal mixing overview: K_rho/K_T/Gamma/qc over (bin, profile)."""
    bins = np.linspace(10.0, 100.0, n_bin)
    nprof = len(stimes)
    rng = np.random.default_rng(2)
    k_rho = 10.0 ** rng.uniform(-6.0, -3.0, size=(n_bin, nprof))
    k_t = 10.0 ** rng.uniform(-7.0, -4.0, size=(n_bin, nprof))
    gamma = 0.2 * np.exp(rng.normal(0.0, 0.3, size=(n_bin, nprof)))
    xr.Dataset(
        {
            "K_rho": (("bin", "profile"), k_rho),
            "K_T": (("bin", "profile"), k_t),
            "Gamma": (("bin", "profile"), gamma),
            "qc_drop_chi": (("bin", "profile"), np.zeros((n_bin, nprof))),
            "stime": (("profile",), np.asarray(stimes, dtype=float)),
        },
        coords={"bin": bins, "profile": np.arange(nprof)},
    ).to_netcdf(path)
    return bins, np.asarray(stimes, dtype=float)


def _write_profile(path, *, stime, n_slow=200, fast_ratio=8, lat=0.0,
                   depth_max=100.0):
    """A minimal ``profiles_NN`` cast with the raw diagnostic channels.

    Slow inclinometer (offset), fast accel/shear (zero-mean) and fast
    temperature-gradient (large DC + small fluctuation, to exercise the HP
    overlay).  Fast and slow clocks span the same time so pressure interpolates.
    """
    n_fast = n_slow * fast_ratio
    t_slow = np.arange(n_slow) / 64.0
    t_fast = np.arange(n_fast) / (64.0 * fast_ratio)
    P = np.linspace(0.5, depth_max, n_slow)  # dbar, descending
    rng = np.random.default_rng(1)
    ds = xr.Dataset(
        {
            "P": (("time_slow",), P),
            "Incl_X": (("time_slow",), -1.0 + 0.2 * rng.standard_normal(n_slow)),
            "Incl_Y": (("time_slow",), 85.0 + 0.3 * rng.standard_normal(n_slow)),
            "Ax": (("time_fast",), 50.0 * rng.standard_normal(n_fast)),
            "Ay": (("time_fast",), 50.0 * rng.standard_normal(n_fast)),
            "sh1": (("time_fast",), 0.05 * rng.standard_normal(n_fast)),
            "sh2": (("time_fast",), 0.05 * rng.standard_normal(n_fast)),
            "T1_dT1": (("time_fast",), 6.0 + 0.01 * rng.standard_normal(n_fast)),
            "T2_dT2": (("time_fast",), 8.0 + 0.01 * rng.standard_normal(n_fast)),
            "t_fast": (("time_fast",), t_fast),
            "t_slow": (("time_slow",), t_slow),
            "lat": ((), float(lat)),
            "stime": ((), float(stime)),
        },
    )
    ds.to_netcdf(path)
    return P


# --------------------------------------------------------------------- overview
class TestLoadOverview:
    def test_reads_fields_and_derives_time(self, tmp_path):
        from odas_tpw.perturb.diag.data import load_overview

        combo = tmp_path / "combo.nc"
        bins, stimes, _eps = _write_combo(combo)
        ov = load_overview(str(combo), ("epsilonMean", "e_1", "e_2"))

        assert ov.bin.shape == bins.shape
        np.testing.assert_allclose(ov.bin, bins)
        assert set(ov.fields) == {"epsilonMean", "e_1", "e_2"}
        assert ov.fields["epsilonMean"].shape == (bins.size, stimes.size)
        # stime_epoch is the raw float; t is the same instant as datetime64.
        np.testing.assert_allclose(ov.stime_epoch, stimes)
        assert ov.t.dtype == np.dtype("datetime64[ns]")
        assert ov.t[0].astype("datetime64[s]").astype(np.int64) == int(stimes[0])

    def test_qc_masks_flagged_cell(self, tmp_path):
        from odas_tpw.perturb.diag.data import load_overview

        combo = tmp_path / "combo.nc"
        _write_combo(combo, qc_cell=(2, 0))

        masked = load_overview(str(combo), ("epsilonMean",), apply_qc=True)
        assert np.isnan(masked.fields["epsilonMean"][2, 0])
        # A neighbor is untouched.
        assert np.isfinite(masked.fields["epsilonMean"][1, 0])

        raw = load_overview(str(combo), ("epsilonMean",), apply_qc=False)
        assert np.isfinite(raw.fields["epsilonMean"][2, 0])

    def test_missing_variable_errors(self, tmp_path):
        from odas_tpw.perturb.diag.data import load_overview

        combo = tmp_path / "combo.nc"
        _write_combo(combo)
        with pytest.raises(SystemExit, match="chiMean"):
            load_overview(str(combo), ("chiMean",))


# ----------------------------------------------------------------- cell mapping
class TestEpsilonCellSource:
    def test_maps_stime_to_file_and_nearest_window(self, tmp_path):
        from odas_tpw.perturb.diag.data import EpsilonCellSource

        diss = tmp_path / "diss_00"
        diss.mkdir()
        _write_diss(diss / "vmp_prof001.nc", stime=1000.0,
                    pressures=(10.0, 50.0, 100.0))

        src = EpsilonCellSource(str(diss))
        # A click near 50 m selects the middle window (pressure 50 -> depth ~50).
        cell = src.cell(1000.0, 52.0)
        assert cell is not None
        assert cell.profile.path.endswith("vmp_prof001.nc")
        assert cell.window == 1
        # A click near the surface selects the shallow window.
        assert src.cell(1000.0, 8.0).window == 0

    def test_no_time_match_returns_none(self, tmp_path):
        from odas_tpw.perturb.diag.data import EpsilonCellSource

        diss = tmp_path / "diss_00"
        diss.mkdir()
        _write_diss(diss / "vmp_prof001.nc", stime=1000.0)

        src = EpsilonCellSource(str(diss), tol=1.0)
        # 100 s away from the only file -> outside tolerance.
        assert src.cell(1100.0, 50.0) is None

    def test_file_is_cached(self, tmp_path):
        from odas_tpw.perturb.diag.data import EpsilonCellSource

        diss = tmp_path / "diss_00"
        diss.mkdir()
        _write_diss(diss / "vmp_prof001.nc", stime=1000.0)
        src = EpsilonCellSource(str(diss))
        a = src.cell(1000.0, 50.0)
        b = src.cell(1000.0, 10.0)
        # Same ProfileFile object reused across two clicks in one profile.
        assert a.profile is b.profile


# --------------------------------------------------------------------- renderer
def test_band_extent_uses_window_length(tmp_path):
    from odas_tpw.perturb.diag.data import load_profile_file
    from odas_tpw.perturb.diag.render import band_extent

    diss = tmp_path / "diss_00"
    diss.mkdir()
    _write_diss(diss / "vmp_prof001.nc", stime=1000.0,
                pressures=(10.0, 50.0, 100.0), diss_length=1024, fs_fast=512.0)
    pf = load_profile_file(str(diss / "vmp_prof001.nc"))
    lo, hi = band_extent(pf, 1)
    # half = 0.5 * (diss_length/fs_fast) * speed = 0.5 * 2 s * 0.7 m/s = 0.7 m
    assert hi - lo == pytest.approx(1.4, abs=1e-6)
    assert 0.5 * (lo + hi) == pytest.approx(pf.depth[1], abs=1e-6)


# --------------------------------------------------------------------- inspector
def _make_inspector(tmp_path):
    from odas_tpw.perturb.diag import render
    from odas_tpw.perturb.diag.data import EpsilonCellSource, load_overview
    from odas_tpw.perturb.diag.inspector import DiagInspector

    combo = tmp_path / "combo.nc"
    _write_combo(combo, stimes=(1000.0, 1000.5))
    diss = tmp_path / "diss_00"
    diss.mkdir()
    _write_diss(diss / "a_prof001.nc", stime=1000.0)
    _write_diss(diss / "b_prof001.nc", stime=1000.5)

    ov = load_overview(str(combo), ("epsilonMean", "e_1", "e_2"))
    src = EpsilonCellSource(str(diss))
    return DiagInspector(
        ov, src,
        field_specs=[("epsilonMean", "m"), ("e_1", "1"), ("e_2", "2")],
        spectra_fn=render.draw_eps_spectra,
        strip_fn=render.draw_diss_strip,
        n_strip=5,
        title="test",
    )


class TestDiagInspector:
    def test_select_shows_crosshair_and_redraws(self, tmp_path):
        insp = _make_inspector(tmp_path)
        insp.select(1, 2)
        assert insp.iProfile == 1
        assert insp.iDepth == 2
        # Crosshair became visible and tracks the selected cast/depth.
        for xh in insp._xhair:
            assert xh.get_visible()
            assert xh.get_xdata()[0] == insp.cast_x[1]
        for yh in insp._yhair:
            assert yh.get_ydata()[0] == insp.data.bin[2]

    def test_select_clamps_out_of_range(self, tmp_path):
        insp = _make_inspector(tmp_path)
        insp.select(999, -5)
        assert insp.iProfile == len(insp.data.stime_epoch) - 1
        assert insp.iDepth == 0

    def test_snapshot_writes_png(self, tmp_path):
        insp = _make_inspector(tmp_path)
        out = tmp_path / "snap.png"
        insp.snapshot(str(out))
        assert out.exists() and out.stat().st_size > 5000

    def test_click_event_selects_cell(self, tmp_path):
        insp = _make_inspector(tmp_path)

        class _Evt:
            inaxes = insp.ax_ov[0]
            xdata = float(insp.cast_x[1])
            ydata = float(insp.data.bin[3])

        insp._on_click(_Evt())
        assert insp.iProfile == 1
        assert insp.iDepth == 3


# --------------------------------------------------------------------- sections
def _write_sections(path, body):
    path.write_text(body, encoding="utf-8")
    return str(path)


class TestApplySections:
    def _overview(self, tmp_path):
        # Three casts a day apart so section windows can pick subsets.
        from odas_tpw.perturb.diag.data import load_overview

        stimes = (
            np.datetime64("2025-01-20T00:00:00").astype("datetime64[s]").astype(float),
            np.datetime64("2025-01-21T00:00:00").astype("datetime64[s]").astype(float),
            np.datetime64("2025-01-22T00:00:00").astype("datetime64[s]").astype(float),
        )
        combo = tmp_path / "combo.nc"
        _write_combo(combo, stimes=stimes)
        return load_overview(str(combo), ("epsilonMean", "e_1", "e_2"))

    def test_none_returns_all(self, tmp_path):
        from odas_tpw.perturb.diag.data import apply_sections

        ov = self._overview(tmp_path)
        out = apply_sections(ov, None, None)
        assert out.stime_epoch.size == 3

    def test_time_window_narrows_casts(self, tmp_path):
        from odas_tpw.perturb.diag.data import apply_sections

        ov = self._overview(tmp_path)
        secs = _write_sections(tmp_path / "s.yaml", """
sections:
  - name: mid
    start: "2025-01-20T12:00:00Z"
    stop:  "2025-01-21T12:00:00Z"
""")
        out = apply_sections(ov, secs, None)
        # Only the 2025-01-21 cast falls inside the window.
        assert out.stime_epoch.size == 1
        assert out.fields["epsilonMean"].shape[1] == 1

    def test_select_picks_named_section(self, tmp_path):
        from odas_tpw.perturb.diag.data import apply_sections

        ov = self._overview(tmp_path)
        secs = _write_sections(tmp_path / "s.yaml", """
sections:
  - name: early
    stop: "2025-01-20T12:00:00Z"
  - name: late
    start: "2025-01-21T12:00:00Z"
""")
        out = apply_sections(ov, secs, ["late"])
        assert out.stime_epoch.size == 1
        # The kept cast is 2025-01-22.
        assert str(out.t[0].astype("datetime64[D]")) == "2025-01-22"

    def test_select_without_sections_errors(self, tmp_path):
        from odas_tpw.perturb.diag.data import apply_sections

        ov = self._overview(tmp_path)
        with pytest.raises(SystemExit, match="only applies together with"):
            apply_sections(ov, None, ["late"])

    def test_empty_selection_errors(self, tmp_path):
        from odas_tpw.perturb.diag.data import apply_sections

        ov = self._overview(tmp_path)
        secs = _write_sections(tmp_path / "s.yaml", """
sections:
  - name: nope
    start: "2030-01-01T00:00:00Z"
""")
        with pytest.raises(SystemExit, match="no casts fall within"):
            apply_sections(ov, secs, None)


# --------------------------------------------------------------------------- CLI
def test_cli_parser_registers_epsilon():
    from odas_tpw.perturb.diag.cli import build_parser

    args = build_parser().parse_args(
        ["epsilon", "--root", "results", "--out", "x.png", "--clim", "-10", "-4",
         "--sections", "s.yaml", "--select", "a,b"]
    )
    assert args.command == "epsilon"
    assert args.root == "results"
    assert args.clim == [-10.0, -4.0]
    assert args.apply_qc is True
    assert args.sections == "s.yaml"
    assert args.select == ["a,b"]
    assert args.diag is True  # raw-diagnostics row on by default


def test_cli_no_diag_flag():
    from odas_tpw.perturb.diag.cli import build_parser

    args = build_parser().parse_args(["epsilon", "--root", "r", "--no-diag"])
    assert args.diag is False


# --------------------------------------------------------------- raw diagnostics
class TestLoadRawProfile:
    def test_loads_channels_hp_and_proc(self, tmp_path):
        from odas_tpw.perturb.diag.data import load_raw_profile

        p = tmp_path / "raw_prof001.nc"
        _write_profile(p, stime=1000.0)
        rp = load_raw_profile(str(p))

        # Fast channels carry hp + proc; the slow inclinometer carries neither.
        for ch in ("Ax", "Ay", "sh1", "sh2", "T1_dT1", "T2_dT2"):
            assert rp.is_fast[ch]
            assert ch in rp.hp and ch in rp.proc
        for ch in ("Incl_X", "Incl_Y"):
            assert not rp.is_fast[ch]
            assert ch not in rp.hp and ch not in rp.proc
        # Depths line up with each clock.
        assert rp.depth_fast.size == rp.raw["Ax"].size
        assert rp.depth_slow.size == rp.raw["Incl_X"].size
        # The HP filter strips the temp-gradient channel's large DC offset:
        # raw sits near 6, hp near 0 (this is the fix for the squashed panel).
        assert np.nanmedian(rp.raw["T1_dT1"]) > 3.0
        assert abs(np.nanmedian(rp.hp["T1_dT1"])) < 0.5

    def test_missing_coords_returns_empty(self, tmp_path):
        from odas_tpw.perturb.diag.data import load_raw_profile

        p = tmp_path / "bad_prof001.nc"
        xr.Dataset({"stime": ((), 1.0)}).to_netcdf(p)
        rp = load_raw_profile(str(p))
        assert rp.raw == {} and rp.hp == {} and rp.proc == {}
        assert rp.depth_fast.size == 0


class TestRawAttach:
    def _dirs(self, tmp_path):
        diss = tmp_path / "diss_00"
        prof = tmp_path / "profiles_00"
        diss.mkdir()
        prof.mkdir()
        _write_diss(diss / "a_prof001.nc", stime=1000.0,
                    pressures=(10.0, 50.0, 100.0))
        _write_profile(prof / "a_prof001.nc", stime=1000.0)
        return str(diss), str(prof)

    def test_cell_attaches_raw_with_profiles_dir(self, tmp_path):
        from odas_tpw.perturb.diag.data import EpsilonCellSource

        diss, prof = self._dirs(tmp_path)
        src = EpsilonCellSource(diss, profiles_dir=prof)
        cell = src.cell(1000.0, 50.0)
        assert cell is not None and cell.raw is not None
        assert "sh1" in cell.raw.proc

    def test_no_profiles_dir_leaves_raw_none(self, tmp_path):
        from odas_tpw.perturb.diag.data import EpsilonCellSource

        diss, _ = self._dirs(tmp_path)
        src = EpsilonCellSource(diss)
        cell = src.cell(1000.0, 50.0)
        assert cell is not None and cell.raw is None

    def test_raw_is_cached(self, tmp_path):
        from odas_tpw.perturb.diag.data import EpsilonCellSource

        diss, prof = self._dirs(tmp_path)
        src = EpsilonCellSource(diss, profiles_dir=prof)
        a = src.cell(1000.0, 50.0)
        b = src.cell(1000.0, 10.0)
        assert a.raw is b.raw  # same RawProfile reused across clicks in one cast


def test_stime_index_is_lazy(tmp_path):
    """The index is not scanned at construction — only on first lookup."""
    from odas_tpw.perturb.diag.data import _StimeIndex

    d = tmp_path / "diss_00"
    d.mkdir()
    _write_diss(d / "a_prof001.nc", stime=1000.0)
    idx = _StimeIndex(str(d), tol=1.0)
    assert idx._keys is None  # deferred: opening the inspector must not scan
    assert idx.path_for(1000.0) is not None
    assert idx._keys is not None  # built on first use


# --------------------------------------------------------------- decimation
def test_envelope_preserves_extremes():
    from odas_tpw.perturb.diag.render import _envelope

    n = 10000
    y = np.zeros(n)
    depth = np.arange(n, dtype=float)
    y[5000], y[7000] = 100.0, -100.0  # narrow spikes a plain stride would miss
    yy, dd = _envelope(y, depth, maxn=1000)
    assert yy.size <= 1200
    assert yy.max() == 100.0 and yy.min() == -100.0
    assert np.all(np.diff(dd) >= 0)  # original order preserved


def test_stride_reduces_length():
    from odas_tpw.perturb.diag.render import _stride

    y = np.arange(10000, dtype=float)
    yy, dd = _stride(y, y.copy(), maxn=1000)
    assert yy.size <= 1100 and dd.size == yy.size


# --------------------------------------------------------------- diag renderer
def _diag_cell(tmp_path):
    from odas_tpw.perturb.diag.data import EpsilonCellSource

    diss = tmp_path / "diss_00"
    prof = tmp_path / "profiles_00"
    diss.mkdir()
    prof.mkdir()
    _write_diss(diss / "a_prof001.nc", stime=1000.0,
                pressures=(10.0, 50.0, 100.0))
    _write_profile(prof / "a_prof001.nc", stime=1000.0)
    src = EpsilonCellSource(str(diss), profiles_dir=str(prof))
    return src.cell(1000.0, 50.0)


def test_draw_diag_strip_renders(tmp_path):
    import matplotlib.pyplot as plt

    from odas_tpw.perturb.diag import render

    cell = _diag_cell(tmp_path)
    fig, axes = plt.subplots(1, render.DIAG_PANEL_COUNT)
    render.draw_diag_strip(list(axes), cell)
    # Every panel drew at least one trace and labeled its x-axis.
    assert all(len(ax.lines) >= 1 for ax in axes)
    assert axes[0].get_xlabel().startswith("Incl")
    plt.close(fig)


def test_draw_diag_strip_without_raw(tmp_path):
    import matplotlib.pyplot as plt

    from odas_tpw.perturb.diag import render
    from odas_tpw.perturb.diag.data import Cell, load_profile_file

    diss = tmp_path / "diss_00"
    diss.mkdir()
    _write_diss(diss / "a_prof001.nc", stime=1000.0,
                pressures=(10.0, 50.0, 100.0))
    pf = load_profile_file(str(diss / "a_prof001.nc"))
    cell = Cell(profile=pf, window=1, raw=None)  # no profiles product

    fig, axes = plt.subplots(1, render.DIAG_PANEL_COUNT)
    render.draw_diag_strip(list(axes), cell)  # must not raise
    assert all(len(ax.lines) == 0 for ax in axes)
    plt.close(fig)


# --------------------------------------------------------------- diag inspector
def _make_diag_inspector(tmp_path):
    from odas_tpw.perturb.diag import render
    from odas_tpw.perturb.diag.data import EpsilonCellSource, load_overview
    from odas_tpw.perturb.diag.inspector import DiagInspector

    combo = tmp_path / "combo.nc"
    _write_combo(combo, stimes=(1000.0, 1000.5))
    diss = tmp_path / "diss_00"
    prof = tmp_path / "profiles_00"
    diss.mkdir()
    prof.mkdir()
    for name, st in (("a", 1000.0), ("b", 1000.5)):
        _write_diss(diss / f"{name}_prof001.nc", stime=st)
        _write_profile(prof / f"{name}_prof001.nc", stime=st)

    ov = load_overview(str(combo), ("epsilonMean", "e_1", "e_2"))
    src = EpsilonCellSource(str(diss), profiles_dir=str(prof))
    return DiagInspector(
        ov, src,
        field_specs=[("epsilonMean", "m"), ("e_1", "1"), ("e_2", "2")],
        spectra_fn=render.draw_eps_spectra,
        strip_fn=render.draw_diss_strip,
        n_strip=5,
        diag_fn=render.draw_diag_strip,
        n_diag=render.DIAG_PANEL_COUNT,
        title="test",
    )


class TestDiagInspectorRow:
    def test_builds_diag_axes(self, tmp_path):
        from odas_tpw.perturb.diag import render

        insp = _make_diag_inspector(tmp_path)
        assert len(insp.ax_diag) == render.DIAG_PANEL_COUNT

    def test_diag_shares_strip_depth_axis(self, tmp_path):
        insp = _make_diag_inspector(tmp_path)
        insp.select(0, 1)
        # Shared y: the diag row tracks the strip's inverted depth limits.
        assert insp.ax_diag[0].get_ylim() == insp.ax_strip[0].get_ylim()

    def test_snapshot_writes_png(self, tmp_path):
        insp = _make_diag_inspector(tmp_path)
        out = tmp_path / "snap.png"
        insp.snapshot(str(out))
        assert out.exists() and out.stat().st_size > 5000


# --------------------------------------------------------------- prewarm
class TestPrewarm:
    def _src(self, tmp_path):
        from odas_tpw.perturb.diag.data import EpsilonCellSource

        diss = tmp_path / "diss_00"
        prof = tmp_path / "profiles_00"
        diss.mkdir()
        prof.mkdir()
        _write_diss(diss / "a_prof001.nc", stime=1000.0)
        _write_profile(prof / "a_prof001.nc", stime=1000.0)
        return EpsilonCellSource(str(diss), profiles_dir=str(prof))

    def test_prewarm_builds_both_indexes(self, tmp_path):
        src = self._src(tmp_path)
        assert src._diss._keys is None and src._raw._keys is None  # lazy
        src.prewarm()
        assert src._diss._keys is not None and src._raw._keys is not None
        assert src.cell(1000.0, 50.0) is not None

    def test_prewarm_from_thread_then_cell(self, tmp_path):
        import threading

        src = self._src(tmp_path)
        t = threading.Thread(target=src.prewarm)
        t.start()
        cell = src.cell(1000.0, 50.0)  # may race the build; the lock serializes
        t.join()
        assert cell is not None


# ------------------------------------------------------- overview depth extent
def test_overview_depth_limits_span_finite_data(tmp_path):
    """The overview depth axis excludes deep bins with no finite data."""
    from odas_tpw.perturb.diag import render
    from odas_tpw.perturb.diag.data import EpsilonCellSource, load_overview
    from odas_tpw.perturb.diag.inspector import DiagInspector

    bins = np.linspace(10.0, 110.0, 6)  # [10,30,50,70,90,110]
    eps = 10.0 ** np.random.default_rng(0).uniform(-9, -6, size=(6, 2))
    eps[4:, :] = np.nan  # deepest two bins (90, 110 m) empty
    combo = tmp_path / "combo.nc"
    xr.Dataset(
        {
            "epsilonMean": (("bin", "profile"), eps),
            "e_1": (("bin", "profile"), eps),
            "e_2": (("bin", "profile"), eps),
            "qc_drop_epsilon": (("bin", "profile"), np.zeros((6, 2))),
            "stime": (("profile",), np.array([1000.0, 1000.5])),
        },
        coords={"bin": bins, "profile": [0, 1]},
    ).to_netcdf(combo)
    diss = tmp_path / "diss_00"
    diss.mkdir()
    _write_diss(diss / "a_prof001.nc", stime=1000.0)
    _write_diss(diss / "b_prof001.nc", stime=1000.5)

    ov = load_overview(str(combo), ("epsilonMean", "e_1", "e_2"))
    insp = DiagInspector(
        ov, EpsilonCellSource(str(diss)),
        field_specs=[("epsilonMean", "m"), ("e_1", "1"), ("e_2", "2")],
        spectra_fn=render.draw_eps_spectra, strip_fn=render.draw_diss_strip,
        n_strip=5, title="t",
    )
    deep = max(insp.ax_ov[0].get_ylim())  # inverted axis: deepest limit
    assert 70.0 <= deep < 90.0  # spans to the deepest finite bin (70), not 110


# --------------------------------------------------------- depth-axis linking
class TestDepthLinking:
    def test_overview_strip_diag_share_depth(self, tmp_path):
        insp = _make_diag_inspector(tmp_path)
        insp.ax_ov[0].set_ylim(120.0, 0.0)  # a "zoom" on the overview
        assert insp.ax_strip[0].get_ylim() == (120.0, 0.0)
        assert insp.ax_diag[0].get_ylim() == (120.0, 0.0)

    def test_selection_preserves_depth_zoom(self, tmp_path):
        insp = _make_diag_inspector(tmp_path)
        insp.ax_ov[0].set_ylim(120.0, 0.0)
        insp.select(0, 1)  # redraw must not autoscale the linked group away
        assert insp.ax_ov[0].get_ylim() == (120.0, 0.0)


def test_overview_x_ticks_are_integers(tmp_path):
    insp = _make_diag_inspector(tmp_path)
    insp.fig.canvas.draw()
    lo, hi = insp.ax_ov[0].get_xlim()
    visible = [t for t in insp.ax_ov[0].get_xticks() if lo <= t <= hi]
    assert visible and all(float(t).is_integer() for t in visible)


# --------------------------------------------------------------------- chi
class TestChi:
    def test_load_chi_profile_file(self, tmp_path):
        from odas_tpw.perturb.diag.data import load_chi_profile_file

        p = tmp_path / "c_prof001.nc"
        _write_chi(p, stime=1000.0, pressures=(10.0, 50.0, 100.0))
        cf = load_chi_profile_file(str(p))
        assert cf.n_probe == 2
        assert cf.spec_gradT.shape == (2, 8, 3)
        assert cf.spec_batch.shape == (2, 8, 3)
        assert cf.spec_noise.shape == (2, 8, 3)
        assert cf.chi.shape == (2, 3)
        assert cf.depth.size == 3

    def test_draw_chi_spectra_and_strip_render(self, tmp_path):
        import matplotlib.pyplot as plt

        from odas_tpw.perturb.diag import render
        from odas_tpw.perturb.diag.data import Cell, load_chi_profile_file

        p = tmp_path / "c_prof001.nc"
        _write_chi(p, stime=1000.0)
        cell = Cell(profile=load_chi_profile_file(str(p)), window=1)

        fig, ax = plt.subplots()
        render.draw_chi_spectra(ax, cell)
        assert len(ax.lines) >= 1
        assert "cpm" in ax.get_xlabel()
        plt.close(fig)

        fig, axes = plt.subplots(1, 5)
        render.draw_chi_strip(list(axes), cell)
        assert len(axes[0].lines) >= 1  # per-probe chi + geomean
        plt.close(fig)

    def test_chi_source_uses_chi_loader(self, tmp_path):
        from odas_tpw.perturb.diag.data import (
            ChiProfileFile,
            EpsilonCellSource,
            load_chi_profile_file,
        )

        chidir = tmp_path / "chi_00"
        chidir.mkdir()
        _write_chi(chidir / "a_prof001.nc", stime=1000.0)
        src = EpsilonCellSource(str(chidir), loader=load_chi_profile_file)
        cell = src.cell(1000.0, 50.0)
        assert cell is not None
        assert isinstance(cell.profile, ChiProfileFile)

    def test_chi_inspector_snapshot(self, tmp_path):
        from odas_tpw.perturb.diag import render
        from odas_tpw.perturb.diag.data import (
            EpsilonCellSource,
            load_chi_profile_file,
            load_overview,
        )
        from odas_tpw.perturb.diag.inspector import DiagInspector

        combo = tmp_path / "combo.nc"
        _write_chi_combo(combo, stimes=(1000.0, 1000.5))
        chidir = tmp_path / "chi_00"
        chidir.mkdir()
        _write_chi(chidir / "a_prof001.nc", stime=1000.0)
        _write_chi(chidir / "b_prof001.nc", stime=1000.5)

        ov = load_overview(str(combo), ("chiMean", "chi_1", "chi_2"),
                           qc_var="qc_drop_chi")
        src = EpsilonCellSource(str(chidir), loader=load_chi_profile_file)
        insp = DiagInspector(
            ov, src,
            field_specs=[("chiMean", "m"), ("chi_1", "1"), ("chi_2", "2")],
            spectra_fn=render.draw_chi_spectra, strip_fn=render.draw_chi_strip,
            n_strip=5, cbar_label=r"$\chi$",
        )
        out = tmp_path / "chi.png"
        insp.snapshot(str(out))
        assert out.exists() and out.stat().st_size > 5000

    def test_cli_registers_chi(self):
        from odas_tpw.perturb.diag.cli import build_parser

        args = build_parser().parse_args(["chi", "--root", "r", "--no-diag"])
        assert args.command == "chi"
        assert args.diag is False

    def test_chi_spectra_model_matches_fom_within_fit_band(self, tmp_path):
        # The overlaid model must be spec_batch*|H|^2 + spec_noise (what the fom
        # integrates), not bare spec_batch — regression for the "fit looks off".
        import matplotlib.pyplot as plt

        from odas_tpw.perturb.diag import render
        from odas_tpw.perturb.diag.data import Cell, load_chi_profile_file

        p = tmp_path / "c_prof001.nc"
        _write_chi(p, stime=1000.0, spectrum_model="kraichnan")
        cf = load_chi_profile_file(str(p))
        fig, ax = plt.subplots()
        render.draw_chi_spectra(ax, Cell(profile=cf, window=1))
        # Title carries the spectrum model + fit method (user request).
        assert "Kraichnan" in ax.get_title()
        assert "iterative" in ax.get_title()
        # The model line (dashed) sits at spec_batch*H2 + noise; with H2<=1 and a
        # positive floor it exceeds bare spec_batch (0.8) somewhere in-band.
        dashed = [ln for ln in ax.lines if ln.get_linestyle() == "--"]
        assert dashed, "model curve should be drawn"
        plt.close(fig)


# ------------------------------------------------------------------- mixing
class TestMixing:
    def test_load_chi_reads_mixing_quantities(self, tmp_path):
        from odas_tpw.perturb.diag.data import load_chi_profile_file

        plain = tmp_path / "plain_prof001.nc"
        _write_chi(plain, stime=1000.0, with_mixing=False)
        cf = load_chi_profile_file(str(plain))
        assert cf.K_rho is None and cf.N2 is None  # absent -> None

        mixed = tmp_path / "mixed_prof001.nc"
        _write_chi(mixed, stime=1000.0, with_mixing=True)
        cm = load_chi_profile_file(str(mixed))
        assert cm.K_rho is not None and cm.K_rho.shape == (3,)
        assert cm.N2 is not None and cm.Gamma is not None

    def test_mixing_source_resolves_both_products(self, tmp_path):
        from odas_tpw.perturb.diag.data import (
            ChiProfileFile,
            MixingCellSource,
            ProfileFile,
        )

        diss = tmp_path / "diss_00"
        chi = tmp_path / "chi_00"
        diss.mkdir()
        chi.mkdir()
        _write_diss(diss / "a_prof001.nc", stime=1000.0)
        _write_chi(chi / "a_prof001.nc", stime=1000.0, with_mixing=True)
        src = MixingCellSource(str(diss), str(chi))
        cell = src.cell(1000.0, 50.0)
        assert cell is not None
        assert isinstance(cell.profile, ChiProfileFile)  # chi is primary
        assert isinstance(cell.profiles["eps"], ProfileFile)
        assert cell.profiles["chi"] is cell.profile

    def test_mixing_spectra_dispatches_on_field(self, tmp_path):
        import matplotlib.pyplot as plt

        from odas_tpw.perturb.diag import render
        from odas_tpw.perturb.diag.data import MixingCellSource

        diss = tmp_path / "diss_00"
        chi = tmp_path / "chi_00"
        diss.mkdir()
        chi.mkdir()
        _write_diss(diss / "a_prof001.nc", stime=1000.0)
        _write_chi(chi / "a_prof001.nc", stime=1000.0, with_mixing=True)
        src = MixingCellSource(str(diss), str(chi))
        cell = src.cell(1000.0, 50.0)

        # K_rho -> shear spectrum: x label is shear wavenumber (cpm), y is shear PSD
        cell.field = "K_rho"
        fig, ax = plt.subplots()
        render.draw_mixing_spectra(ax, cell)
        assert "cpm" in ax.get_xlabel()
        assert "s$^{-2}$" in ax.get_ylabel()  # shear-spectrum units
        plt.close(fig)

        # K_T -> chi (temperature-gradient) spectrum: y is K^2 m^-1
        cell.field = "K_T"
        fig, ax = plt.subplots()
        render.draw_mixing_spectra(ax, cell)
        assert "K$^2$ m$^{-1}$" in ax.get_ylabel()
        plt.close(fig)

        # Gamma -> a note, no spectrum line
        cell.field = "Gamma"
        fig, ax = plt.subplots()
        render.draw_mixing_spectra(ax, cell)
        assert not ax.lines
        assert any("Gamma" in t.get_text() or "\\Gamma" in t.get_text()
                   for t in ax.texts)
        plt.close(fig)

    def test_mixing_strip_renders_quantities(self, tmp_path):
        import matplotlib.pyplot as plt

        from odas_tpw.perturb.diag import render
        from odas_tpw.perturb.diag.data import MixingCellSource

        diss = tmp_path / "diss_00"
        chi = tmp_path / "chi_00"
        diss.mkdir()
        chi.mkdir()
        _write_diss(diss / "a_prof001.nc", stime=1000.0)
        _write_chi(chi / "a_prof001.nc", stime=1000.0, with_mixing=True)
        cell = MixingCellSource(str(diss), str(chi)).cell(1000.0, 50.0)
        fig, axes = plt.subplots(1, 7)
        render.draw_mixing_strip(list(axes), cell)
        # every panel got a line (eps, chi, N2, dTdz, K_T, K_rho, Gamma present)
        assert all(len(ax.lines) >= 1 for ax in axes)
        plt.close(fig)

    def test_mixing_strip_na_without_mixing_vars(self, tmp_path):
        import matplotlib.pyplot as plt

        from odas_tpw.perturb.diag import render
        from odas_tpw.perturb.diag.data import MixingCellSource

        diss = tmp_path / "diss_00"
        chi = tmp_path / "chi_00"
        diss.mkdir()
        chi.mkdir()
        _write_diss(diss / "a_prof001.nc", stime=1000.0)
        _write_chi(chi / "a_prof001.nc", stime=1000.0, with_mixing=False)
        cell = MixingCellSource(str(diss), str(chi)).cell(1000.0, 50.0)
        fig, axes = plt.subplots(1, 7)
        render.draw_mixing_strip(list(axes), cell)
        # N2 panel (index 2) has no line but an 'n/a' note
        assert not axes[2].lines
        assert any("n/a" in t.get_text() for t in axes[2].texts)
        plt.close(fig)

    def test_mixing_inspector_snapshot_and_ifield(self, tmp_path):
        from odas_tpw.perturb.diag import render
        from odas_tpw.perturb.diag.data import MixingCellSource, load_overview
        from odas_tpw.perturb.diag.inspector import DiagInspector

        combo = tmp_path / "combo.nc"
        _write_mixing_combo(combo, stimes=(1000.0, 1000.5))
        diss = tmp_path / "diss_00"
        chi = tmp_path / "chi_00"
        diss.mkdir()
        chi.mkdir()
        for s, tag in ((1000.0, "a"), (1000.5, "b")):
            _write_diss(diss / f"{tag}_prof001.nc", stime=s)
            _write_chi(chi / f"{tag}_prof001.nc", stime=s, with_mixing=True)

        data = load_overview(str(combo), ("K_rho", "K_T", "Gamma"),
                             qc_var="qc_drop_chi")
        src = MixingCellSource(str(diss), str(chi))
        insp = DiagInspector(
            data, src,
            field_specs=[("K_rho", r"$K_\rho$"), ("K_T", r"$K_T$"),
                         ("Gamma", r"$\Gamma$")],
            spectra_fn=render.draw_mixing_spectra,
            strip_fn=render.draw_mixing_strip, n_strip=7,
            per_field_clim=True,
        )
        # selecting field 1 (K_T) makes the drill-down draw the chi spectrum
        insp.iField = 1
        i_depth, i_profile = insp.default_cell()
        insp.select(i_profile, i_depth)
        assert "K$^2$ m$^{-1}$" in insp.ax_spec.get_ylabel()
        out = tmp_path / "mix.png"
        insp.snapshot(str(out))
        assert out.exists() and out.stat().st_size > 5000

    def test_cli_registers_mixing(self):
        from odas_tpw.perturb.diag.cli import build_parser

        args = build_parser().parse_args(["mixing", "--root", "r", "--no-diag"])
        assert args.command == "mixing"
        assert args.diag is False
