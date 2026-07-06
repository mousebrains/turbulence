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


# --------------------------------------------------------------------------- CLI
def test_cli_parser_registers_epsilon():
    from odas_tpw.perturb.diag.cli import build_parser

    args = build_parser().parse_args(
        ["epsilon", "--root", "results", "--out", "x.png", "--clim", "-10", "-4"]
    )
    assert args.command == "epsilon"
    assert args.root == "results"
    assert args.clim == [-10.0, -4.0]
    assert args.apply_qc is True
