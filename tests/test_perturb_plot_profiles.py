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


def test_make_norm_diverging_one_signed_not_symmetric():
    """A diverging field whose robust (1/99%) range is one-signed must NOT be
    mirrored about zero -- an all-stable dTdz (~ -0.2..0) or an offset Incl_Y
    (~76..90 deg) would otherwise waste half the colorbar.  Zero-straddling
    data stays symmetric (covered by test_make_norm_picks_scale)."""
    from matplotlib.colors import Normalize

    args = argparse.Namespace(vmin=None, vmax=None)
    z = np.array([[-0.20, -0.02], [-0.15, -0.05]])  # all-negative dTdz
    div = profiles._make_norm("dTdz", z, args, {})
    assert isinstance(div, Normalize)
    assert div.vmax < 0.0                             # positive half not padded on
    assert div.vmin == pytest.approx(-0.20, abs=0.02)
    iy = profiles._make_norm(
        "Incl_Y", np.array([[76.0, 90.0], [80.0, 88.0]]), args, {})
    assert iy.vmin > 0.0                              # not mirrored to negative
    assert iy.vmax == pytest.approx(90.0, abs=0.5)


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


@pytest.mark.parametrize("cmd,var", [
    ("profiles", "T1"), ("epsilon", "epsilonMean"), ("chi", "chiMean"), ("mixing", "K_T"),
])
def test_render_each_product(tmp_path: Path, cmd, var):
    _build_all_products(tmp_path)
    rc = _run([cmd, "--root", str(tmp_path),
               "--out-dir", str(tmp_path), "--name", "t", "--xaxis", "time", "--var", var])
    assert rc == 0
    assert (tmp_path / f"{cmd}_t.png").exists()  # file stem follows the subcommand


def test_default_vars_and_latitude_axis(tmp_path: Path):
    _build_all_products(tmp_path)
    # mixing defaults to K_T/Gamma/K_rho; latitude x-axis.
    rc = _run(["mixing", "--root", str(tmp_path),
               "--out-dir", str(tmp_path), "--name", "m", "--xaxis", "latitude"])
    assert rc == 0
    assert (tmp_path / "mixing_m.png").exists()


def test_profiles_var_on_1d_var_is_graceful(tmp_path: Path):
    """`--var lat` (a 1-D profile-only var) must be treated as missing with a
    clear message, not crash on transpose('bin','profile') (M-15)."""
    _build_all_products(tmp_path)
    rc = _run(["profiles", "--root", str(tmp_path),
               "--out-dir", str(tmp_path), "--name", "x", "--var", "lat"])
    assert rc == 0                                   # graceful skip, no traceback
    assert not (tmp_path / "profiles_x.png").exists()  # no panel built


def test_profile_window_skips_when_empty(tmp_path: Path):
    _build_all_products(tmp_path)
    cfg = tmp_path / "s.yaml"
    cfg.write_text(
        "sections:\n  - name: future\n    start: '2099-01-01T00:00:00Z'\n"
        "    xaxis: {method: time}\n"
    )
    rc = _run(["epsilon", "--root", str(tmp_path),
               "--sections", str(cfg), "--out-dir", str(tmp_path)])
    assert rc == 0
    assert not (tmp_path / "epsilon_future.png").exists()


def test_missing_default_vars_errors(tmp_path: Path):
    # A diss combo with none of the default vars -> clear error.
    _write_product(tmp_path, "diss_combo", {"speed": (np.ones((10, 6)), {"units": "m/s"})})
    with pytest.raises(SystemExit):
        _run(["epsilon", "--root", str(tmp_path), "--out-dir", str(tmp_path)])


def test_log_clim_nonpositive_errors(tmp_path: Path):
    _build_all_products(tmp_path)
    with pytest.raises(SystemExit):
        _run(["epsilon", "--root", str(tmp_path),
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
    rc = _run(["epsilon", "--root", str(tmp_path), "--no-qc",
               "--out-dir", str(tmp_path), "--name", "c", "--var", "epsilonMean",
               "--clim", "epsilonMean", "1e-9", "1e-5"])
    assert rc == 0
    assert (tmp_path / "epsilon_c.png").exists()


def test_ncols_grid_layout(tmp_path: Path):
    """The profiles family honors --ncols: 4 default vars at ncols=2 give a
    2-row figure (height 3*2+1) instead of the 4-row stack, with 4 colorbars."""
    import matplotlib.pyplot as plt
    import xarray as xr

    bins, _cast = _bp()
    col = np.ones((1, 6))
    _write_product(tmp_path, "combo", {
        "T1": (28.0 - bins * col, {"units": "degree_Celsius", "long_name": "T1"}),
        "T2": (28.0 - bins * col, {"units": "degree_Celsius", "long_name": "T2"}),
        "N2": (1.0e-4 * np.ones((10, 6)), {"units": "s-2", "long_name": "N2"}),
        "dTdz": (-0.1 * np.ones((10, 6)), {"units": "K m-1", "long_name": "dTdz"}),
    })
    ds = xr.open_dataset(tmp_path / "combo_00" / "combo.nc", decode_times=False)
    sec = profiles.Section(name="all", method="time")
    args = argparse.Namespace(
        root=str(tmp_path), product="profiles", p_max=None, gap_factor=4.0,
        apply_qc=True, hp_cut=1.0, despike_thresh=8.0, despike_smooth=0.5,
        stime_tol=1.0, vmin=None, vmax=None, var=None, clim=[], ncols=2,
    )
    try:
        fig = profiles._build_profiles_figure(
            ds, sec, list(profiles.PRODUCTS["profiles"].default_vars),
            args, {}, profiles.PRODUCTS["profiles"],
        )
        assert fig is not None
        assert list(fig.get_size_inches()) == [11.0, 7.0]  # 2x2 -> 3*2 + 1
        cbars = [ax for ax in fig.axes if getattr(ax, "_colorbar", None) is not None]
        assert len(cbars) == 4  # one colorbar per default var, grid or not
    finally:
        ds.close()
        plt.close("all")


def test_profiles_default_preset_is_3x3(tmp_path: Path):
    """The profiles preset defaults to the 9-variable 3-column overview grid
    when --ncols is not given."""
    import matplotlib.pyplot as plt
    import xarray as xr

    prod = profiles.PRODUCTS["profiles"]
    assert prod.default_vars == (
        "JAC_T", "T1", "T2", "SP", "rho", "sigma0", "W_slow", "dTdz", "N2")
    assert prod.default_ncols == 3

    bins, _cast = _bp()
    col = np.ones((1, 6))
    data = {v: (20.0 + 0.1 * bins * col, {"units": "1", "long_name": v})
            for v in prod.default_vars}
    _write_product(tmp_path, "combo", data)
    ds = xr.open_dataset(tmp_path / "combo_00" / "combo.nc", decode_times=False)
    sec = profiles.Section(name="all", method="time")
    args = argparse.Namespace(  # deliberately NO ncols -> product default (3)
        root=str(tmp_path), product="profiles", p_max=None, gap_factor=4.0,
        apply_qc=True, hp_cut=1.0, despike_thresh=8.0, despike_smooth=0.5,
        stime_tol=1.0, vmin=None, vmax=None, var=None, clim=[],
    )
    try:
        fig = profiles._build_profiles_figure(
            ds, sec, list(prod.default_vars), args, {}, prod)
        assert fig is not None
        cbars = [ax for ax in fig.axes if getattr(ax, "_colorbar", None) is not None]
        assert len(cbars) == 9
        # 9 vars / 3 cols = 3 rows -> width max(11, 5.5*3)=16.5, height 3*3+1=10.
        assert list(fig.get_size_inches()) == [16.5, 10.0]
    finally:
        ds.close()
        plt.close("all")


def test_cbar_label_overrides(tmp_path: Path):
    """The profile scalars carry the custom T_1/T_2/N^2/dT-dz colorbar labels,
    overriding the CF long_name/units default."""
    import matplotlib.pyplot as plt
    import xarray as xr

    bins, _cast = _bp()
    col = np.ones((1, 6))
    _write_product(tmp_path, "combo", {
        "T1": (28.0 - bins * col, {"units": "degree_Celsius", "long_name": "ignored"}),
        "T2": (28.0 - bins * col, {"units": "degree_Celsius", "long_name": "ignored"}),
        "N2": (1.0e-4 * np.ones((10, 6)), {"units": "s-2", "long_name": "ignored"}),
        "dTdz": (-0.1 * np.ones((10, 6)), {"units": "K m-1", "long_name": "ignored"}),
    })
    ds = xr.open_dataset(tmp_path / "combo_00" / "combo.nc", decode_times=False)
    sec = profiles.Section(name="all", method="time")
    args = argparse.Namespace(
        root=str(tmp_path), product="profiles", p_max=None, gap_factor=4.0,
        apply_qc=True, hp_cut=1.0, despike_thresh=8.0, despike_smooth=0.5,
        stime_tol=1.0, vmin=None, vmax=None, var=None, clim=[],
    )
    try:
        fig = profiles._build_profiles_figure(
            ds, sec, ["T1", "T2", "N2", "dTdz"], args, {},
            profiles.PRODUCTS["profiles"],
        )
        labels = {ax.yaxis.label.get_text()
                  for ax in fig.axes if getattr(ax, "_colorbar", None) is not None}
        assert labels == {
            r"$T_1$ (°C)", r"$T_2$ (°C)",
            r"$N^2$ (s$^{-2}$) (Thorpe-sorted)",
            r"$\frac{dT}{dz}$ (°C m$^{-1}$) (+ downwards)",
        }
    finally:
        ds.close()
        plt.close("all")


def test_pdp_incl_cbar_label_overrides(tmp_path: Path):
    """P_dP and the inclinometer channels carry the custom short colorbar
    labels; Incl_T shows °C."""
    import matplotlib.pyplot as plt
    import xarray as xr

    bins, _cast = _bp()
    col = np.ones((1, 6))
    _write_product(tmp_path, "combo", {
        "P_dP": (5.0 + bins * col, {"units": "dbar", "long_name": "ignored"}),
        "Incl_X": (bins * col - 5.0, {"units": "degree", "long_name": "ignored"}),
        "Incl_Y": (76.0 + bins * col, {"units": "degree", "long_name": "ignored"}),
        "Incl_T": (18.0 + bins * col, {"units": "degree_Celsius", "long_name": "x"}),
        "W_slow": (0.8 + 0.01 * bins * col, {"units": "dbar s-1", "long_name": "x"}),
    })
    ds = xr.open_dataset(tmp_path / "combo_00" / "combo.nc", decode_times=False)
    sec = profiles.Section(name="all", method="time")
    args = argparse.Namespace(
        root=str(tmp_path), product="profiles", p_max=None, gap_factor=4.0,
        apply_qc=True, hp_cut=1.0, despike_thresh=8.0, despike_smooth=0.5,
        stime_tol=1.0, vmin=None, vmax=None, var=None, clim=[],
    )
    try:
        fig = profiles._build_profiles_figure(
            ds, sec, ["P_dP", "Incl_X", "Incl_Y", "Incl_T", "W_slow"], args, {},
            profiles.PRODUCTS["profiles"],
        )
        labels = {ax.yaxis.label.get_text()
                  for ax in fig.axes if getattr(ax, "_colorbar", None) is not None}
        assert labels == {
            "P_dP (dbar)", "incl_X (°)", "incl_Y (°)", "incl_T (°C)",
            r"W_slow (dbar s$^{-1}$)",
        }
    finally:
        ds.close()
        plt.close("all")


def test_depth_axis_fits_valid_data(tmp_path: Path):
    """Without --p-max, the depth axis fits where there is valid data, not the
    full combo bin grid (which would pad empty gray below the deepest sample)."""
    import matplotlib.pyplot as plt
    import xarray as xr

    z = 20.0 * np.ones((10, 6))          # bins at 0.5, 1.5, ..., 9.5 m
    z[5:, :] = np.nan                    # valid only in bins 0..4 (0.5..4.5 m)
    _write_product(tmp_path, "combo", {
        "T1": (z, {"units": "degree_Celsius", "long_name": "x"}),
    })
    ds = xr.open_dataset(tmp_path / "combo_00" / "combo.nc", decode_times=False)
    sec = profiles.Section(name="all", method="time")
    args = argparse.Namespace(
        root=str(tmp_path), product="profiles", p_max=None, gap_factor=4.0,
        apply_qc=True, hp_cut=1.0, despike_thresh=8.0, despike_smooth=0.5,
        stime_tol=1.0, vmin=None, vmax=None, var=None, clim=[],
    )
    try:
        fig = profiles._build_profiles_figure(
            ds, sec, ["T1"], args, {}, profiles.PRODUCTS["profiles"])
        deep, shallow = fig.axes[0].get_ylim()   # inverted: deep bound first
        assert deep == pytest.approx(5.0)        # 4.5 deepest valid + 0.5 half-bin
        assert shallow == pytest.approx(0.0)
    finally:
        ds.close()
        plt.close("all")


def test_depth_axis_respects_explicit_p_max(tmp_path: Path):
    """--p-max overrides the data-driven fit: surface to p_max."""
    import matplotlib.pyplot as plt
    import xarray as xr

    z = 20.0 * np.ones((10, 6))
    z[5:, :] = np.nan
    _write_product(tmp_path, "combo", {
        "T1": (z, {"units": "degree_Celsius", "long_name": "x"}),
    })
    ds = xr.open_dataset(tmp_path / "combo_00" / "combo.nc", decode_times=False)
    sec = profiles.Section(name="all", method="time")
    args = argparse.Namespace(
        root=str(tmp_path), product="profiles", p_max=8.0, gap_factor=4.0,
        apply_qc=True, hp_cut=1.0, despike_thresh=8.0, despike_smooth=0.5,
        stime_tol=1.0, vmin=None, vmax=None, var=None, clim=[],
    )
    try:
        fig = profiles._build_profiles_figure(
            ds, sec, ["T1"], args, {}, profiles.PRODUCTS["profiles"])
        deep, shallow = fig.axes[0].get_ylim()
        assert deep == pytest.approx(8.0)        # explicit p_max, not the ~5 fit
        assert shallow == pytest.approx(0.0)
    finally:
        ds.close()
        plt.close("all")


def test_jac_cbar_label_overrides(tmp_path: Path):
    """JAC_T/JAC_C read as the familiar JAC_T/JAC_C names (JFE dropped),
    overriding their CF long_name/units."""
    import matplotlib.pyplot as plt
    import xarray as xr

    bins, _cast = _bp()
    col = np.ones((1, 6))
    _write_product(tmp_path, "combo", {
        "JAC_T": (28.0 - bins * col,
                  {"units": "degree_Celsius", "long_name": "in-situ temperature (JFE)"}),
        "JAC_C": (50.0 + bins * col,
                  {"units": "mS/cm", "long_name": "conductivity (JFE)"}),
    })
    ds = xr.open_dataset(tmp_path / "combo_00" / "combo.nc", decode_times=False)
    sec = profiles.Section(name="all", method="time")
    args = argparse.Namespace(
        root=str(tmp_path), product="profiles", p_max=None, gap_factor=4.0,
        apply_qc=True, hp_cut=1.0, despike_thresh=8.0, despike_smooth=0.5,
        stime_tol=1.0, vmin=None, vmax=None, var=None, clim=[],
    )
    try:
        fig = profiles._build_profiles_figure(
            ds, sec, ["JAC_T", "JAC_C"], args, {}, profiles.PRODUCTS["profiles"])
        labels = {ax.yaxis.label.get_text()
                  for ax in fig.axes if getattr(ax, "_colorbar", None) is not None}
        assert labels == {"JAC_T (°C)", "JAC_C (mS/cm)"}
    finally:
        ds.close()
        plt.close("all")


def test_reverse_cbar_inverts_pdp_colorbar(tmp_path: Path):
    """P_dP's colorbar reads with the smallest value at the top (axis inverted);
    a normal channel's colorbar keeps the default orientation."""
    import matplotlib.pyplot as plt
    import xarray as xr

    bins, _cast = _bp()
    col = np.ones((1, 6))
    _write_product(tmp_path, "combo", {
        "P_dP": (5.0 + bins * col, {"units": "dbar", "long_name": "x"}),
        "T1": (28.0 - bins * col, {"units": "degree_Celsius", "long_name": "x"}),
    })
    ds = xr.open_dataset(tmp_path / "combo_00" / "combo.nc", decode_times=False)
    sec = profiles.Section(name="all", method="time")
    args = argparse.Namespace(
        root=str(tmp_path), product="profiles", p_max=None, gap_factor=4.0,
        apply_qc=True, hp_cut=1.0, despike_thresh=8.0, despike_smooth=0.5,
        stime_tol=1.0, vmin=None, vmax=None, var=None, clim=[],
    )
    try:
        fig = profiles._build_profiles_figure(
            ds, sec, ["P_dP", "T1"], args, {}, profiles.PRODUCTS["profiles"],
        )
        cbars = {ax.yaxis.label.get_text(): ax
                 for ax in fig.axes if getattr(ax, "_colorbar", None) is not None}
        assert cbars["P_dP (dbar)"].yaxis_inverted()       # smallest at top
        assert not cbars[r"$T_1$ (°C)"].yaxis_inverted()   # default orientation
    finally:
        ds.close()
        plt.close("all")


def test_seawater_vars_render_on_profiles(tmp_path: Path):
    """SP/SA/CT/sigma0/rho render on the profiles product with the curated
    labels (Theta, sigma_0, rho-1000, Salinity); SP and sigma0 read
    smallest-at-top."""
    import matplotlib.pyplot as plt
    import xarray as xr

    bins, _cast = _bp()
    col = np.ones((1, 6))
    _write_product(tmp_path, "combo", {
        "SP": (34.0 + 0.02 * bins * col,
               {"units": "1", "long_name": "practical salinity (PSU)"}),
        "SA": (34.2 + 0.02 * bins * col,
               {"units": "g/kg", "long_name": "absolute salinity"}),
        "CT": (28.0 - 0.1 * bins * col,
               {"units": "degree_Celsius", "long_name": "conservative temperature"}),
        "sigma0": (22.0 + 0.05 * bins * col,
                   {"units": "kg m-3", "long_name": "potential density anomaly"}),
        "rho": (22.5 + 0.05 * bins * col,
                {"units": "kg m-3", "long_name": "in-situ density - 1000"}),
    })
    ds = xr.open_dataset(tmp_path / "combo_00" / "combo.nc", decode_times=False)
    sec = profiles.Section(name="all", method="time")
    args = argparse.Namespace(
        root=str(tmp_path), product="profiles", p_max=None, gap_factor=4.0,
        apply_qc=True, hp_cut=1.0, despike_thresh=8.0, despike_smooth=0.5,
        stime_tol=1.0, vmin=None, vmax=None, var=None, clim=[],
    )
    try:
        fig = profiles._build_profiles_figure(
            ds, sec, ["SP", "SA", "CT", "sigma0", "rho"], args, {},
            profiles.PRODUCTS["profiles"])
        assert fig is not None
        cbars = {ax.yaxis.label.get_text(): ax
                 for ax in fig.axes if getattr(ax, "_colorbar", None) is not None}
        assert cbars.keys() == {
            r"$\Theta$ (°C)", "Salinity (PSU)", "Salinity (g/kg)",
            r"$\sigma_0$ (kg m$^{-3}$)", r"$\rho$-1000 (kg m$^{-3}$) (in-situ)",
        }
        assert cbars["Salinity (PSU)"].yaxis_inverted()             # min at top
        assert cbars[r"$\sigma_0$ (kg m$^{-3}$)"].yaxis_inverted()
        assert not cbars["Salinity (g/kg)"].yaxis_inverted()        # default
    finally:
        ds.close()
        plt.close("all")


def test_in_situ_calib_label_from_metadata(tmp_path: Path):
    """A channel whose ``calibration`` attr marks it in-situ calibrated gets
    ``(in-situ calib)`` appended to its colorbar label; an untagged channel does
    not.  The tag is driven by the variable metadata, not the config."""
    import matplotlib.pyplot as plt
    import xarray as xr

    bins, _cast = _bp()
    col = np.ones((1, 6))
    _write_product(tmp_path, "combo", {
        "T1": (28.0 - bins * col,
               {"units": "degree_Celsius", "long_name": "ignored",
                "calibration": "in-situ (Steinhart-Hart order 1 vs JAC_T)"}),
        "T2": (28.0 - bins * col, {"units": "degree_Celsius", "long_name": "ignored"}),
    })
    ds = xr.open_dataset(tmp_path / "combo_00" / "combo.nc", decode_times=False)
    sec = profiles.Section(name="all", method="time")
    args = argparse.Namespace(
        root=str(tmp_path), product="profiles", p_max=None, gap_factor=4.0,
        apply_qc=True, hp_cut=1.0, despike_thresh=8.0, despike_smooth=0.5,
        stime_tol=1.0, vmin=None, vmax=None, var=None, clim=[],
    )
    try:
        fig = profiles._build_profiles_figure(
            ds, sec, ["T1", "T2"], args, {}, profiles.PRODUCTS["profiles"],
        )
        labels = {ax.yaxis.label.get_text()
                  for ax in fig.axes if getattr(ax, "_colorbar", None) is not None}
        assert labels == {r"$T_1$ (°C) (in-situ calib)", r"$T_2$ (°C)"}
    finally:
        ds.close()
        plt.close("all")


def test_long_colorbar_labels_fit_within_bars(tmp_path: Path):
    """Regression: verbose ``long_name [units]`` colorbar labels on stacked
    panels must be shrunk to fit their bars, not overflow into the neighboring
    panel's label.  Four verbose labels are tall enough to collide on a
    3-in-per-panel figure, so the builder must call ``layout.fit_colorbar_labels``
    — which shrinks the font (never grows it) so each label roughly fits its bar.

    Uses variables whose labels come from ``var_label`` (not the short
    ``_CBAR_LABEL`` overrides for T1/T2/N2/dTdz), so the labels are genuinely
    long. We assert the *behavior* (font shrunk below the default; label brought
    to within its bar), not an exact pixel fit: rendered text height for a given
    font size varies ~5-8% across matplotlib backends/platforms, so a strict
    ``label_h <= bar_h`` is environment-fragile (passed locally, failed CI).
    The precise geometry of ``fit_colorbar_labels`` is pinned by its unit tests
    in ``test_perturb_plot_layout.py``.
    """
    import matplotlib.pyplot as plt
    import xarray as xr
    from matplotlib.font_manager import FontProperties

    bins, _cast = _bp()
    col = np.ones((1, 6))
    # Non-overridden, non-Celsius vars so the labels come from var_label and stay
    # genuinely long (overrides would shorten them; degree_Celsius -> "(°C)"
    # would too).
    _write_product(tmp_path, "combo", {
        "Gnd": (bins * col - 5.0,
                {"units": "V",
                 "long_name": "instrument ground reference voltage on analog board"}),
        "V_Bat": (bins * col - 5.0,
                  {"units": "V",
                   "long_name": "vehicle battery pack terminal voltage during cast"}),
        "PV": (bins * col - 5.0,
               {"units": "V",
                "long_name": "pressure transducer excitation voltage on sensor"}),
        "speed": (bins * col + 1.0,
                  {"units": "m s-1",
                   "long_name": "vehicle profiling speed through the water column"}),
    })
    ds = xr.open_dataset(tmp_path / "combo_00" / "combo.nc", decode_times=False)
    sec = profiles.Section(name="all", method="time")
    args = argparse.Namespace(
        root=str(tmp_path), product="profiles", p_max=None, gap_factor=4.0,
        apply_qc=True, hp_cut=1.0, despike_thresh=8.0, despike_smooth=0.5,
        stime_tol=1.0, vmin=None, vmax=None, var=None, clim=[], ncols=1,
    )
    try:
        fig = profiles._build_profiles_figure(
            ds, sec, ["Gnd", "V_Bat", "PV", "speed"],
            args, {}, profiles.PRODUCTS["profiles"],
        )
        assert fig is not None
        fig.draw_without_rendering()
        cbars = [ax for ax in fig.axes if getattr(ax, "_colorbar", None) is not None]
        assert len(cbars) == 4  # one colorbar per default var
        # Default colorbar-label size the fix starts from (it only ever shrinks).
        default_fs = FontProperties(
            size=plt.rcParams["axes.labelsize"]).get_size_in_points()
        for ax in cbars:
            bar_h = ax.get_window_extent().height
            label = ax.yaxis.label
            label_h = label.get_window_extent().height
            # The fix ran and shrank this over-tall label below the default size.
            assert label.get_fontsize() < default_fs, (
                label.get_text(), label.get_fontsize(), default_fs)
            # ...and brought it within its bar, allowing a font-metric tolerance
            # (without the fix these labels are ~1.5x the bar; see docstring).
            assert label_h <= bar_h * 1.10, (label.get_text(), label_h, bar_h)
    finally:
        ds.close()
        plt.close("all")
