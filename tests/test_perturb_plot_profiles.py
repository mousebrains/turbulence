# Jun-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for ``perturb-plot profiles`` (binned (bin, profile) products)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from odas_tpw.perturb.plot import profiles  # noqa: E402

# Profiles in two time clusters separated by a multi-day gap (tests gap-split).
_STIME = np.array([0.0, 300.0, 600.0, 1.0e6, 1.0e6 + 300, 1.0e6 + 600])
_LAT = np.linspace(10.0, 12.0, 6)
_LON = np.linspace(130.0, 132.0, 6)


def _write_product(root: Path, prefix: str, data_vars: dict, n_bin: int = 10) -> None:
    import xarray as xr

    nprof = len(_STIME)
    ds = xr.Dataset(
        {name: (("bin", "profile"), arr, attrs) for name, (arr, attrs) in data_vars.items()},
        coords={
            "bin": ("bin", np.arange(n_bin, dtype=float) + 0.5),
            "profile": ("profile", np.arange(nprof, dtype=np.int32)),
        },
    )
    ds["bin"].attrs.update(units="m", standard_name="depth", positive="down")
    ds["lat"] = (("profile",), _LAT, {"units": "degrees_north"})
    ds["lon"] = (("profile",), _LON, {"units": "degrees_east"})
    ds["stime"] = (("profile",), _STIME, {"units": "seconds since 1970-01-01"})
    ds["etime"] = (("profile",), _STIME + 300.0, {"units": "seconds since 1970-01-01"})
    ds.attrs["id"] = "test"
    out = root / f"{prefix}_00"
    out.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out / "combo.nc")


def _bp(n_bin: int = 10):
    """A (bin, profile) array that decays with depth and varies across casts."""
    bins = (np.arange(n_bin)[:, None] + 1.0)
    cast = (np.arange(6)[None, :] + 1.0)
    return bins, cast


def _build_all_products(root: Path) -> None:
    bins, cast = _bp()
    eps = 1.0e-6 * np.exp(-bins / 5.0) * cast  # positive, log-spread
    chi = 1.0e-7 * np.exp(-bins / 5.0) * cast
    qc = np.zeros((10, 6))
    qc[0, 0] = 1.0  # one flagged cell
    _write_product(root, "combo", {
        "T1": (28.0 - bins * np.ones((1, 6)), {"units": "degree_Celsius", "long_name": "T1"}),
        "T2": (28.0 - bins * np.ones((1, 6)), {"units": "degree_Celsius", "long_name": "T2"}),
        "N2": (1.0e-4 * np.ones((10, 6)), {"units": "s-2", "long_name": "N2"}),
        "dTdz": (-0.1 * np.ones((10, 6)), {"units": "K m-1", "long_name": "dTdz"}),
    })
    _write_product(root, "diss_combo", {
        "epsilonMean": (eps, {"units": "W/kg", "long_name": "epsilon"}),
        "qc_drop_epsilon": (qc, {}),
    })
    _write_product(root, "chi_combo", {
        "chiMean": (chi, {"units": "K^2/s", "long_name": "chi"}),
        "K_T": (1.0e-4 * cast * np.ones((10, 1)), {"units": "m2 s-1", "long_name": "K_T"}),
        "Gamma": (0.2 * np.ones((10, 6)), {"units": "1", "long_name": "Gamma"}),
        "K_rho": (1.0e-4 * cast * np.ones((10, 1)), {"units": "m2 s-1", "long_name": "K_rho"}),
        "qc_drop_chi": (np.zeros((10, 6)), {}),
    })


def _run(argv) -> int:
    from odas_tpw.perturb.plot.cli import main

    return main(argv)


# ---------------------------------------------------------------------------
# Unit: profile-window filter + norm selection
# ---------------------------------------------------------------------------


def test_profile_window_filters_by_stime():
    sec = profiles.Section(name="w", method="time",
                           start=np.datetime64("1970-01-01T00:00:00"),
                           stop=np.datetime64("1970-01-01T00:11:00"))  # 0..660 s
    mask = profiles._profile_window(_STIME, sec)
    assert mask.tolist() == [True, True, True, False, False, False]


def test_make_norm_picks_scale():
    from matplotlib.colors import LogNorm, Normalize

    args = argparse.Namespace(vmin=None, vmax=None)
    log = profiles._make_norm("epsilonMean", np.array([[1e-8, 1e-6], [1e-7, 1e-5]]), args, {})
    assert isinstance(log, LogNorm)
    div = profiles._make_norm("dTdz", np.array([[-1.0, 1.0], [0.0, 0.5]]), args, {})
    assert isinstance(div, Normalize) and not isinstance(div, LogNorm)
    assert div.vmin == pytest.approx(-div.vmax)            # symmetric about 0
    lin = profiles._make_norm("T1", np.array([[5.0, 6.0], [7.0, 8.0]]), args, {})
    assert isinstance(lin, Normalize) and not isinstance(lin, LogNorm)
    none = profiles._make_norm("epsilonMean", np.full((2, 2), np.nan), args, {})
    assert none is None                                    # log field, no positive data


def test_make_norm_n2_uses_symlog_for_negatives():
    """N2 can be negative (overturning); it must use SymLogNorm so negatives
    are visible, not LogNorm which would mask them as no-data (#20)."""
    from matplotlib.colors import LogNorm, SymLogNorm

    args = argparse.Namespace(vmin=None, vmax=None)
    n = profiles._make_norm("N2", np.array([[1e-5, -1e-6], [1e-4, 1e-3]]), args, {})
    assert isinstance(n, SymLogNorm)
    assert not isinstance(n, LogNorm)
    assert n.vmin < 0  # the negative N2 is within range, not clipped away


def test_make_norm_reversed_clim_errors_on_linear_var():
    """A reversed --clim (MIN >= MAX) on a linear/diverging var must raise, not
    silently invert the colorbar (#21)."""
    args = argparse.Namespace(vmin=None, vmax=None)
    with pytest.raises(SystemExit):
        profiles._make_norm("T1", np.array([[5.0, 6.0]]), args, {"T1": (6.0, 2.0)})
    with pytest.raises(SystemExit):
        profiles._make_norm("dTdz", np.array([[-1.0, 1.0]]), args, {"dTdz": (1.0, -1.0)})


# ---------------------------------------------------------------------------
# End-to-end across products
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("product,var", [
    ("profiles", "T1"), ("diss", "epsilonMean"), ("chi", "chiMean"), ("mixing", "K_T"),
])
def test_render_each_product(tmp_path: Path, product, var):
    _build_all_products(tmp_path)
    rc = _run(["profiles", "--root", str(tmp_path), "--product", product,
               "--out-dir", str(tmp_path), "--name", "t", "--xaxis", "time", "--var", var])
    assert rc == 0
    assert (tmp_path / f"{product}_t.png").exists()


def test_default_vars_and_latitude_axis(tmp_path: Path):
    _build_all_products(tmp_path)
    # mixing defaults to K_T/Gamma/K_rho; latitude x-axis.
    rc = _run(["profiles", "--root", str(tmp_path), "--product", "mixing",
               "--out-dir", str(tmp_path), "--name", "m", "--xaxis", "latitude"])
    assert rc == 0
    assert (tmp_path / "mixing_m.png").exists()


def test_profile_window_skips_when_empty(tmp_path: Path):
    _build_all_products(tmp_path)
    cfg = tmp_path / "s.yaml"
    cfg.write_text(
        "sections:\n  - name: future\n    start: '2099-01-01T00:00:00Z'\n"
        "    xaxis: {method: time}\n"
    )
    rc = _run(["profiles", "--root", str(tmp_path), "--product", "diss",
               "--sections", str(cfg), "--out-dir", str(tmp_path)])
    assert rc == 0
    assert not (tmp_path / "diss_future.png").exists()


def test_missing_default_vars_errors(tmp_path: Path):
    # A diss combo with none of the default vars -> clear error.
    _write_product(tmp_path, "diss_combo", {"speed": (np.ones((10, 6)), {"units": "m/s"})})
    with pytest.raises(SystemExit):
        _run(["profiles", "--root", str(tmp_path), "--product", "diss",
              "--out-dir", str(tmp_path)])


def test_log_clim_nonpositive_errors(tmp_path: Path):
    _build_all_products(tmp_path)
    with pytest.raises(SystemExit):
        _run(["profiles", "--root", str(tmp_path), "--product", "diss",
              "--out-dir", str(tmp_path), "--var", "epsilonMean",
              "--clim", "epsilonMean", "-1", "1"])  # MIN <= 0 on a log var


def test_plot_columns_tied_x_renders():
    """Two casts at the same x (re-occupied station) must not crash pcolormesh."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    from odas_tpw.perturb.plot import layout

    fig, ax = plt.subplots()
    try:
        x = np.array([10.0, 10.0, 11.0])  # first two tied
        depth = np.arange(5.0)
        z = np.linspace(0.0, 1.0, 15).reshape(5, 3)
        pcm = layout.plot_columns(ax, fig, x, depth, z, plt.cm.viridis, Normalize(0, 1), "v")
        assert pcm is not None
    finally:
        plt.close(fig)


def test_plot_columns_clusters_on_original_x():
    """Many tied casts must not deflate the cluster median and spuriously split
    the spaced casts (regression: strictly_increasing must run AFTER clustering)."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import QuadMesh
    from matplotlib.colors import Normalize

    from odas_tpw.perturb.plot import layout

    fig, ax = plt.subplots()
    try:
        x = np.concatenate([np.zeros(8), [100.0, 200.0]])  # 8 tied + 2 evenly spaced
        depth = np.arange(4.0)
        z = np.ones((4, 10))
        layout.plot_columns(ax, fig, x, depth, z, plt.cm.viridis, Normalize(0, 1), "v")
        meshes = [c for c in ax.collections if isinstance(c, QuadMesh)]
        assert len(meshes) == 1  # one cluster; the buggy nudge-first split it into 3
    finally:
        plt.close(fig)


def test_clim_and_no_qc(tmp_path: Path):
    _build_all_products(tmp_path)
    rc = _run(["profiles", "--root", str(tmp_path), "--product", "diss", "--no-qc",
               "--out-dir", str(tmp_path), "--name", "c", "--var", "epsilonMean",
               "--clim", "epsilonMean", "1e-9", "1e-5"])
    assert rc == 0
    assert (tmp_path / "diss_c.png").exists()
