# Jun-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for the plot-time diagnostic pseudo-variables (plot/diagnostics.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from odas_tpw.perturb.plot import diagnostics  # noqa: E402


def _write_raw_profile(out_dir: Path, stem: str, stime: float,
                       n_slow: int = 256, fs_slow: float = 64.0, fs_fast: float = 512.0,
                       seed: int = 0) -> None:
    """A minimal raw per-profile file with the fast channels the diagnostics read."""
    import xarray as xr

    ratio = int(fs_fast / fs_slow)
    n_fast = n_slow * ratio
    t_slow = np.arange(n_slow) / fs_slow
    t_fast = np.arange(n_fast) / fs_fast
    pres = np.linspace(0.0, 100.0, n_slow)  # a descent 0..100 dbar
    rng = np.random.RandomState(seed)
    sh1 = 1.0e-2 * np.sin(2 * np.pi * 5.0 * t_fast) + 1.0e-3 * rng.standard_normal(n_fast)
    ax = 5000.0 + 100.0 * rng.standard_normal(n_fast)
    t1_dt1 = 28.0 - 0.1 * np.interp(t_fast, t_slow, pres) + 1e-3 * rng.standard_normal(n_fast)
    ds = xr.Dataset({
        "sh1": (("time_fast",), sh1, {"units": "s-1"}),
        "Ax": (("time_fast",), ax, {"units": "counts"}),
        "T1_dT1": (("time_fast",), t1_dt1, {"units": "degree_Celsius"}),
        "P": (("time_slow",), pres, {"units": "dbar"}),
        "t_fast": (("time_fast",), t_fast, {}),
        "t_slow": (("time_slow",), t_slow, {}),
        "stime": ((), float(stime), {"units": "seconds since 1970-01-01"}),
        "lat": ((), 20.0, {}),
        "lon": ((), 130.0, {}),
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_dir / f"{stem}.nc")


def test_pseudo_metadata():
    assert diagnostics.is_pseudo_var("sh1_var")
    assert diagnostics.is_pseudo_var("Ax_var")
    assert not diagnostics.is_pseudo_var("epsilonMean")
    assert "sh1" in diagnostics.pseudo_label("sh1_var")
    assert diagnostics.pseudo_cmap("Ax_var")  # a cmap name


def test_variance_by_bin():
    # >2 samples required per bin; bin0 var=1, bin1 var=0.
    sig = np.array([0.0, 2.0, 0.0, 2.0, 10.0, 10.0, 10.0])
    pres = np.array([0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5])
    edges = np.array([0.0, 1.0, 2.0])
    out = diagnostics._variance_by_bin(sig, pres, edges)
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(0.0)


def test_variance_by_bin_too_few_samples_nan():
    out = diagnostics._variance_by_bin(
        np.array([1.0, 2.0]), np.array([0.5, 0.5]), np.array([0.0, 1.0])
    )
    assert np.isnan(out[0])  # <= 2 samples


def test_binned_channel_variance_real_filter(tmp_path: Path):
    diagnostics._STIME_CACHE.clear()
    _write_raw_profile(tmp_path, "a_prof001", stime=1000.0)
    edges = np.arange(0.0, 51.0)  # 50 bins, 0..50 dbar
    out = diagnostics._binned_channel_variance(
        str(tmp_path / "a_prof001.nc"), "sh1", edges,
        hp_cut=1.0, despike_thresh=8.0, despike_smooth=0.5,
    )
    assert out is not None and out.shape == (50,)
    assert np.any(np.isfinite(out)) and np.all(out[np.isfinite(out)] >= 0)


def test_compute_pseudo_grid_matches_by_stime(tmp_path: Path):
    diagnostics._STIME_CACHE.clear()
    pdir = tmp_path / "profiles_00"
    _write_raw_profile(pdir, "z_prof001", stime=1000.0, seed=1)  # alphabetical first
    _write_raw_profile(pdir, "a_prof001", stime=2000.0, seed=2)  # but later stime
    bin_centers = np.arange(50) + 0.5
    # combo order (stime-sorted) is [1000, 2000]; a third cast has no raw match.
    combo_stime = np.array([1000.0, 2000.0, 9.9e8])
    grid = diagnostics.compute_pseudo_grid(
        "sh1_var", combo_stime, bin_centers, str(pdir), hp_cut=1.0, tol=1.0,
    )
    assert grid.shape == (50, 3)
    assert np.any(np.isfinite(grid[:, 0]))      # matched 1000 -> z_prof001
    assert np.any(np.isfinite(grid[:, 1]))      # matched 2000 -> a_prof001
    assert np.all(np.isnan(grid[:, 2]))         # unmatched -> NaN column


def test_compute_pseudo_grid_tolerance(tmp_path: Path):
    diagnostics._STIME_CACHE.clear()
    pdir = tmp_path / "profiles_00"
    _write_raw_profile(pdir, "a_prof001", stime=1000.0)
    grid = diagnostics.compute_pseudo_grid(
        "sh1_var", np.array([1000.5]), np.arange(50) + 0.5, str(pdir), tol=1.0,
    )
    assert np.any(np.isfinite(grid[:, 0]))      # 0.5 s within 1 s tolerance
    grid2 = diagnostics.compute_pseudo_grid(
        "sh1_var", np.array([1003.0]), np.arange(50) + 0.5, str(pdir), tol=1.0,
    )
    assert np.all(np.isnan(grid2[:, 0]))        # 3 s > 1 s -> no match


def _write_diss_combo(root: Path, stime: np.ndarray) -> None:
    import xarray as xr

    n = len(stime)
    ds = xr.Dataset(
        {"epsilonMean": (("bin", "profile"), 1e-7 * np.ones((50, n)),
                         {"units": "W/kg", "long_name": "eps"})},
        coords={"bin": ("bin", np.arange(50, dtype=float) + 0.5),
                "profile": ("profile", np.arange(n, dtype=np.int32))},
    )
    ds["lat"] = (("profile",), np.full(n, 20.0), {"units": "degrees_north"})
    ds["lon"] = (("profile",), np.linspace(130.0, 131.0, n), {"units": "degrees_east"})
    ds["stime"] = (("profile",), stime, {"units": "seconds since 1970-01-01"})
    ds["etime"] = (("profile",), stime + 300.0, {"units": "seconds since 1970-01-01"})
    out = root / "diss_combo_00"
    out.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out / "combo.nc")


def test_cli_diagnostic_pseudo_var(tmp_path: Path):
    diagnostics._STIME_CACHE.clear()
    from odas_tpw.perturb.plot.cli import main

    stime = np.array([1000.0, 2000.0])
    _write_diss_combo(tmp_path, stime)
    _write_raw_profile(tmp_path / "profiles_00", "a_prof001", stime=1000.0, seed=1)
    _write_raw_profile(tmp_path / "profiles_00", "b_prof001", stime=2000.0, seed=2)
    rc = main(["epsilon", "--root", str(tmp_path),
               "--out-dir", str(tmp_path), "--name", "d",
               "--var", "epsilonMean", "--var", "sh1_var", "--var", "Ax_var"])
    assert rc == 0
    assert (tmp_path / "epsilon_d.png").exists()
