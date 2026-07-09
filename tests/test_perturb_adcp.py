# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for the CODAS gridded ADCP reader and finescale shear (perturb/adcp.py)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from odas_tpw.perturb.adcp import (
    AdcpData,
    read_codas,
    shear_squared,
    window_shear,
)

EPOCH_2025 = np.datetime64("2025-01-01T00:00:00", "s").astype(np.int64) * 1.0


def _codas_dataset(
    n_t=6,
    n_cell=10,
    dz=2.0,
    z0=9.0,
    du_dz=0.01,
    dv_dz=-0.005,
    dt_s=120.0,
):
    """Synthetic CODAS-like gridded dataset: linear shear, clean flags."""
    days = 30.0 + np.arange(n_t) * dt_s / 86400.0
    depth = np.tile(z0 + dz * np.arange(n_cell), (n_t, 1)).astype(np.float32)
    u = (0.1 + du_dz * depth).astype(np.float32)
    v = (0.02 + dv_dz * depth).astype(np.float32)
    ds = xr.Dataset(
        {
            "u": (("time", "depth_cell"), u),
            "v": (("time", "depth_cell"), v),
            "depth": (("time", "depth_cell"), depth),
            "pflag": (("time", "depth_cell"), np.zeros((n_t, n_cell), np.int8)),
            "pg": (("time", "depth_cell"), np.full((n_t, n_cell), 90.0, np.float32)),
        },
        coords={"time": ("time", days)},
    )
    ds["time"].attrs["units"] = "days since 2025-01-01 00:00:00"
    ds["time"].attrs["calendar"] = "proleptic_gregorian"
    return ds


def _write(ds, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return path


# --------------------------------------------------------------------------- #
# read_codas
# --------------------------------------------------------------------------- #


def test_read_codas_decodes_time_and_masks(tmp_path):
    ds = _codas_dataset()
    ds["pflag"].values[1, 3] = 2  # flagged cell with a (bogus) finite velocity
    ds["u"].values[2, 4] = np.nan  # pre-masked by CODAS editing
    p = _write(ds, tmp_path / "wh300.nc")
    a = read_codas(p)
    assert a.name == "wh300"
    # 'days since 2025-01-01' decodes to epoch seconds
    np.testing.assert_allclose(a.time[0], EPOCH_2025 + 30.0 * 86400.0, rtol=0, atol=1e-3)
    np.testing.assert_allclose(np.diff(a.time), 120.0)
    assert np.isnan(a.u[1, 3]) and np.isnan(a.v[1, 3])  # pflag != 0 masked
    assert np.isnan(a.u[2, 4]) and np.isnan(a.v[2, 4])  # NaN respected
    assert np.isfinite(a.u).sum() == a.u.size - 2


def test_read_codas_min_pg(tmp_path):
    ds = _codas_dataset()
    ds["pg"].values[0, 0] = 40.0
    p = _write(ds, tmp_path / "wh300.nc")
    assert np.isfinite(read_codas(p).u[0, 0])  # no pg cut by default
    assert np.isnan(read_codas(p, min_pg=50.0).u[0, 0])
    # a requested pg floor with no pg variable must fail loudly, not no-op
    p2 = _write(ds.drop_vars("pg"), tmp_path / "nopg.nc")
    with pytest.raises(ValueError, match="no 'pg'"):
        read_codas(p2, min_pg=50.0)
    read_codas(p2)  # without the floor the pg-less file is fine


def test_read_codas_custom_name_and_monotonic_guard(tmp_path):
    ds = _codas_dataset()
    p = _write(ds, tmp_path / "os75nb.nc")
    assert read_codas(p, name="os75").name == "os75"
    # non-monotonic time is a corrupt file, not a soft condition
    tvals = ds["time"].values.copy()
    tvals[3] = tvals[1]
    ds2 = ds.assign_coords(time=("time", tvals))
    ds2["time"].attrs.update(ds["time"].attrs)
    p2 = _write(ds2, tmp_path / "bad.nc")
    with pytest.raises(ValueError, match="strictly increasing"):
        read_codas(p2)


# --------------------------------------------------------------------------- #
# shear_squared
# --------------------------------------------------------------------------- #


def test_shear_squared_linear_shear_exact():
    z = np.arange(10.0, 30.0, 2.0)
    a, b = 0.01, -0.005
    u = 0.3 + a * z
    v = -0.1 + b * z
    s2, z_mid = shear_squared(u, v, z)
    np.testing.assert_allclose(s2, a**2 + b**2, rtol=1e-12)
    np.testing.assert_allclose(z_mid, z[:-1] + 1.0)
    assert s2.shape == (z.size - 1,)


def test_shear_squared_gap_does_not_bridge():
    z = np.arange(10.0, 30.0, 2.0)
    u = 0.01 * z
    v = np.zeros_like(z)
    u[4] = np.nan  # one masked cell
    s2, _ = shear_squared(u, v, z)
    assert np.isnan(s2[3]) and np.isnan(s2[4])  # both touching pairs NaN
    assert np.isfinite(s2[2]) and np.isfinite(s2[5])


def test_shear_squared_2d_and_bad_dz():
    z = np.tile(np.arange(0.0, 10.0, 2.0), (3, 1))
    u = 0.02 * z
    v = np.zeros_like(z)
    s2, _z_mid = shear_squared(u, v, z)
    assert s2.shape == (3, 4)
    np.testing.assert_allclose(s2, 4e-4, rtol=1e-12)
    # zero or negative spacing -> NaN, no divide warning
    z_flat = np.array([0.0, 2.0, 2.0, 4.0])
    s2f, _ = shear_squared(0.01 * z_flat, np.zeros(4), z_flat)
    assert np.isnan(s2f[1])


# --------------------------------------------------------------------------- #
# window_shear
# --------------------------------------------------------------------------- #


def _adcp(n_t=11, du_dz=0.01, dv_dz=0.0, dt_s=120.0, **kw):
    ds = _codas_dataset(n_t=n_t, du_dz=du_dz, dv_dz=dv_dz, dt_s=dt_s, **kw)
    t = EPOCH_2025 + np.asarray(ds["time"].values) * 86400.0
    return AdcpData(
        name="syn",
        time=t,
        depth=np.asarray(ds["depth"].values, np.float64),
        u=np.asarray(ds["u"].values, np.float64),
        v=np.asarray(ds["v"].values, np.float64),
        path=__import__("pathlib").Path("synthetic"),
    )


def test_window_shear_linear_shear_exact():
    a = _adcp(du_dz=0.01)
    t0 = float(a.time[5])
    res = window_shear(a, [t0, t0], [15.0, 20.5])
    # rtol 1e-4: u/v/depth are stored float32 (as in real CODAS files),
    # so the linear-shear identity holds only to single precision.
    np.testing.assert_allclose(res.S2, 1e-4, rtol=1e-4)
    assert res.n_ens[0] == 5  # +-300 s at 120-s ensembles
    # outside the column: shallower than the first mid-depth (10 m)
    out = window_shear(a, [t0], [5.0])
    assert np.isnan(out.S2[0])


def test_window_shear_averages_before_differencing():
    # Two ensembles with equal-and-opposite noise: averaging first gives the
    # exact linear shear; differencing first would rectify the noise into S2.
    a = _adcp(n_t=2, du_dz=0.01, dt_s=60.0)
    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, 0.05, a.u.shape[1])
    a.u[0] += noise
    a.u[1] -= noise
    t_mid = float(a.time.mean())
    res = window_shear(a, [t_mid], [15.0], time_tolerance=300.0)
    np.testing.assert_allclose(res.S2[0], 1e-4, rtol=1e-4)  # float32 storage
    assert res.n_ens[0] == 2


def test_window_shear_time_tolerance_excludes_far_ensembles():
    a = _adcp(n_t=11)
    res = window_shear(a, [float(a.time[0]) - 1e6], [15.0])
    assert np.isnan(res.S2[0]) and res.n_ens[0] == 0
    tight = window_shear(a, [float(a.time[5])], [15.0], time_tolerance=60.0)
    assert tight.n_ens[0] == 1


def test_window_shear_gap_guard():
    # Masking cells 4:6 (17/19 m) NaNs mids 16/18/20 m, so the bracketing
    # finite mids are 14 and 22 m — 8 m apart on a 2-m grid, beyond the
    # default 2.5x (5 m) gap -> NaN rather than bridging.
    a = _adcp(n_t=3)
    a.u[:, 4:6] = np.nan
    t0 = float(a.time[1])
    z_gap = float(a.depth[0, 4]) + 1.0  # inside the masked region
    res = window_shear(a, [t0], [z_gap])
    assert np.isnan(res.S2[0])
    # but an explicit larger max_gap allows it
    res2 = window_shear(a, [t0], [z_gap], max_gap=10.0)
    assert np.isfinite(res2.S2[0])


def test_window_shear_edge_depths_and_nan_queries():
    a = _adcp(n_t=3)
    t0 = float(a.time[1])
    z_mids = 0.5 * (a.depth[0, 1:] + a.depth[0, :-1])
    res = window_shear(
        a,
        [t0, t0, t0, np.nan],
        [float(z_mids[0]), float(z_mids[-1]), 1e4, 15.0],
    )
    assert np.isfinite(res.S2[0])  # exactly on the shallowest estimate
    assert np.isfinite(res.S2[1])  # exactly on the deepest estimate
    assert np.isnan(res.S2[2])  # far below the column
    assert np.isnan(res.S2[3])  # NaN query
    assert res.n_ens[3] == 0


def test_window_shear_shape_mismatch():
    a = _adcp(n_t=3)
    with pytest.raises(ValueError, match="same shape"):
        window_shear(a, [0.0, 1.0], [10.0])


def test_window_shear_all_masked_returns_nan():
    a = _adcp(n_t=3)
    a.u[:] = np.nan
    res = window_shear(a, [float(a.time[1])], [15.0])
    assert np.isnan(res.S2[0])
    assert res.n_ens[0] == 3  # ensembles were in range, just unusable


def test_read_codas_roundtrip_through_window_shear(tmp_path):
    # End-to-end: write a synthetic file, read it, query a linear shear.
    p = _write(_codas_dataset(n_t=11, du_dz=0.02), tmp_path / "wh300.nc")
    a = read_codas(p)
    res = window_shear(a, [float(a.time[5])], [20.0])
    # dataset default dv_dz=-0.005 contributes too: S2 = 0.02^2 + 0.005^2
    np.testing.assert_allclose(res.S2[0], 4.25e-4, rtol=1e-4)  # float32 storage


def test_read_codas_rejects_descending_depths(tmp_path):
    ds = _codas_dataset()
    ds["depth"].values[:] = ds["depth"].values[:, ::-1]
    p = _write(ds, tmp_path / "flipped.nc")
    with pytest.raises(ValueError, match="shallow-to-deep"):
        read_codas(p)


def test_window_shear_rejects_bad_queries_and_tolerance():
    a = _adcp(n_t=3)
    t0 = float(a.time[1])
    with pytest.raises(ValueError, match="1-D"):
        window_shear(a, [[t0], [t0]], [[10.0], [12.0]])  # 2-D columns
    with pytest.raises(ValueError, match="positive"):
        window_shear(a, [t0], [15.0], time_tolerance=-5.0)
    with pytest.raises(ValueError, match="positive"):
        window_shear(a, [t0], [15.0], time_tolerance=np.nan)


def test_window_shear_single_cell_fails_loudly():
    a = _adcp(n_t=3, n_cell=1)
    with pytest.raises(ValueError, match="no usable cell spacing"):
        window_shear(a, [float(a.time[1])], [9.0])
