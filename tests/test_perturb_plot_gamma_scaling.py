# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb-plot gamma-scaling (Lewin et al. 2025 Fig. 5 analog)."""

from __future__ import annotations

import argparse

import numpy as np
import pytest
import xarray as xr

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from odas_tpw.perturb.plot import gamma_scaling as G  # noqa: E402

# Synthetic-run constants. Chosen so the paper QC passes by construction:
# Gamma = N2*chi/(2*eps*dTdz^2) = 2.0, Re_b = 250, Cox ~ 3.6e3.
N2 = 1.0e-4
CHI = 1.0e-7
DTDZ = 0.01
EPS = 2.5e-8
NU = 1.0e-6
FS_SLOW = 32.0
FALL = 0.9  # dbar/s
T0 = np.datetime64("2025-02-11T00:00:00", "ns")

# The inserted overturn: sigma0/T1 flipped over this time span (~1.1 m at
# FALL) — small enough not to trip the 0.4*span truncation flag in a 4-s
# sort window, big enough to clear the density-route L_T floor and the
# run-length test at FS_SLOW.
OT_LO, OT_HI = 27.4, 28.6


def _times(seconds):
    return T0 + (np.asarray(seconds) * 1e9).astype("timedelta64[ns]")


def _write_run(
    root, n_prof=2, with_overturn=True, with_profiles=True, diss_offset=0.0, ot_span=(OT_LO, OT_HI)
):
    """Write a synthetic chi_00/diss_00/profiles_00 run under *root*."""
    for k in range(n_prof):
        name = f"synth_prof{k:03d}.nc"
        wt = np.arange(10.0, 40.0, 1.0)  # window centers [s]
        n = wt.size
        chi = xr.Dataset(
            {
                "chiMean": ("time", np.full(n, CHI)),
                "N2": ("time", np.full(n, N2)),
                "dTdz": ("time", np.full(n, -DTDZ)),
                "nu": ("time", np.full(n, NU)),
                "P_mean": ("time", 5.0 + FALL * wt),
                "T_mean": ("time", np.full(n, 20.0)),
                "speed": ("time", np.full(n, FALL)),
                "qc_drop_chi": ("time", np.zeros(n)),
                "stime": ((), _times(0.0 + 86400 * k)),
                "lat": ((), 20.5),
                "lon": ((), 130.5),
            },
            coords={"t": ("time", _times(wt + 86400 * k))},
        )
        diss = xr.Dataset(
            {
                "epsilonMean": ("time", np.full(n, EPS)),
                "qc_drop_epsilon": ("time", np.zeros(n)),
            },
            coords={"t": ("time", _times(wt + diss_offset + 86400 * k))},
        )
        (root / "chi_00").mkdir(exist_ok=True, parents=True)
        (root / "diss_00").mkdir(exist_ok=True, parents=True)
        chi.to_netcdf(root / "chi_00" / name)
        diss.to_netcdf(root / "diss_00" / name)
        if not with_profiles:
            continue
        ts = np.arange(0.0, 60.0, 1.0 / FS_SLOW)
        P = 5.0 + FALL * ts
        # Stable stratification whose density gradient reproduces N2.
        dsig_dz = N2 * 1025.0 / 9.81
        sigma = 25.0 + dsig_dz * P
        T1 = 25.0 - DTDZ * P
        if with_overturn:
            m = (ts >= ot_span[0]) & (ts <= ot_span[1])
            sigma[m] = sigma[m][::-1]
            T1[m] = T1[m][::-1]
        prof = xr.Dataset(
            {
                "P": ("time_slow", P),
                "sigma0": ("time_slow", sigma),
                "T1": ("time_slow", T1),
                "JAC_T": ("time_slow", T1),
                "SP": ("time_slow", np.full(ts.size, 35.0)),
                "SA": ("time_slow", np.full(ts.size, 35.16)),
                "CT": ("time_slow", T1),
                "stime": ((), _times(0.0 + 86400 * k)),
                "lat": ((), 20.5),
                "lon": ((), 130.5),
            },
            coords={"t_slow": ("time_slow", _times(ts + 86400 * k))},
        )
        (root / "profiles_00").mkdir(exist_ok=True, parents=True)
        prof.to_netcdf(root / "profiles_00" / name)


def _write_adcp(path, du_dz=0.01, n_t=40, dt_s=120.0):
    """Synthetic CODAS file spanning the synthetic run's times and depths."""
    days = 41.0 + np.arange(n_t) * dt_s / 86400.0  # Feb 11 = day 41 of 2025
    depth = np.tile(3.0 + 2.0 * np.arange(30), (n_t, 1))
    u = 0.1 + du_dz * depth
    v = np.zeros_like(u)
    ds = xr.Dataset(
        {
            "u": (("time", "depth_cell"), u),
            "v": (("time", "depth_cell"), v),
            "depth": (("time", "depth_cell"), depth),
            "pflag": (("time", "depth_cell"), np.zeros(u.shape, np.int8)),
        },
        coords={"time": ("time", days)},
    )
    ds["time"].attrs["units"] = "days since 2025-01-01 00:00:00"
    ds["time"].attrs["calendar"] = "proleptic_gregorian"
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return path


def _args(root, **kw):
    base = dict(
        root=str(root),
        config=None,
        latest=False,
        sections=None,
        select=None,
        out_dir=None,
        adcp=None,
        min_pg=None,
        time_tolerance=1e6,
        sort_window=4.0,
        route="density",
        lt_floor_density=None,
        lt_floor_temperature=None,
        min_reb=G.DEFAULT_MIN_REB,
        min_cox=G.DEFAULT_MIN_COX,
        min_depth=G.DEFAULT_MIN_DEPTH,
        max_dt=G.DEFAULT_MAX_DT,
        ri_n2_span=G.DEFAULT_RI_N2_SPAN,
        min_run=10,
        keep_truncated=False,
        max_profiles=0,
        qc_flags=True,
        figsize=None,
        ncols=None,
        dpi=None,
        title=None,
    )
    base.update(kw)
    return argparse.Namespace(**base)


# --------------------------------------------------------------------------- #
# sweep
# --------------------------------------------------------------------------- #


def test_sweep_recomputes_gamma_from_components(tmp_path):
    _write_run(tmp_path)
    table = G.sweep(_args(tmp_path))
    assert table.n_profiles == 2
    ok = np.isfinite(table["Gamma"])
    assert ok.sum() == table.n
    np.testing.assert_allclose(table["Gamma"][ok], 2.0, rtol=1e-6)
    unresolved = ~np.isfinite(table["R_OT"])
    assert unresolved.any()  # guard: the check below must not be vacuous
    np.testing.assert_allclose(
        table["Re_b"][unresolved][:1],
        EPS / (NU * N2),
        rtol=0.05,
    )
    # Cox pinned exactly against the same kappa_T the sweep uses (SP=35
    # window means, chi-file T_mean/P_mean).
    from odas_tpw.scor160.ocean import kappa_T

    expect = CHI / (2.0 * kappa_T(20.0, 35.0, table["P"]) * DTDZ**2)
    np.testing.assert_allclose(table["Cox"], expect, rtol=1e-9)


def test_sweep_thorpe_resolves_only_the_overturn(tmp_path):
    _write_run(tmp_path)
    table = G.sweep(_args(tmp_path))
    resolved = table["resolved"] > 0
    assert resolved.any(), "the inserted overturn must be resolved"
    # Resolved windows cluster on the overturn's window times.
    ot_center = 5.0 + FALL * 0.5 * (OT_LO + OT_HI)  # overturn center [dbar]
    assert np.all(np.abs(table["P"][resolved] - ot_center) < 4.0)
    # R_OT is finite exactly where the overturn is resolved.
    assert np.isfinite(table["R_OT"][resolved]).all()
    assert not np.isfinite(table["R_OT"][~resolved]).any()
    # patch-N2 for a fully-sampled linear-gradient overturn ~ background N2.
    n2p = table["N2p_sigma"][resolved]
    np.testing.assert_allclose(n2p, N2, rtol=0.35)


def test_sweep_without_overturn_falls_back_to_background(tmp_path):
    _write_run(tmp_path, with_overturn=False)
    table = G.sweep(_args(tmp_path))
    assert not (table["resolved"] > 0).any()
    np.testing.assert_allclose(table["N2_scale"][np.isfinite(table["N2_scale"])], N2, rtol=1e-6)
    assert not np.isfinite(table["R_OT"]).any()


def test_sweep_without_profiles_dir_degrades(tmp_path):
    _write_run(tmp_path, with_profiles=False)
    table = G.sweep(_args(tmp_path))
    assert not np.isfinite(table["LT_sigma"]).any()
    # Re_b still available through the chi-window N2 fallback.
    assert np.isfinite(table["Re_b"]).all()


def test_sweep_pairs_offset_diss_grid(tmp_path):
    _write_run(tmp_path, diss_offset=0.2)  # within max_dt=0.5
    table = G.sweep(_args(tmp_path))
    assert np.isfinite(table["eps"]).all()
    # Beyond tolerance nothing pairs. (On a uniform 1-s window grid an
    # offset > spacing merely aliases onto a neighboring window, so the
    # no-pair case is exercised with a tighter --max-dt instead.)
    _write_run(tmp_path / "far", diss_offset=0.3)
    table2 = G.sweep(_args(tmp_path / "far", max_dt=0.1))
    assert not np.isfinite(table2["eps"]).any()


def test_sweep_requires_chi_and_diss(tmp_path):
    (tmp_path / "profiles_00").mkdir(parents=True)
    with pytest.raises(SystemExit, match="chi"):
        G.sweep(_args(tmp_path))


def test_sweep_adcp_rig_per_sonar_never_blended(tmp_path):
    _write_run(tmp_path)
    a1 = _write_adcp(tmp_path / "wh300.nc", du_dz=0.01)
    a2 = _write_adcp(tmp_path / "os75nb.nc", du_dz=0.005)
    table = G.sweep(_args(tmp_path, adcp=[str(a1), str(a2)]))
    assert set(table.rig) == {"wh300", "os75nb"}
    r1 = table.rig["wh300"]
    r2 = table.rig["os75nb"]
    fin = np.isfinite(r1) & np.isfinite(r2)
    assert fin.any()
    # Ri_g = N2_ri / S2 with the known linear shears (S2 = du_dz^2). N2_ri
    # is the matched-span TEOS-10 stratification from the synthetic T/S
    # profile (NOT the nominal chi-file N2), so compare against the table's
    # own numerator; the sonars differ by exactly the shear ratio (2^2).
    np.testing.assert_allclose(r1[fin], table["N2_ri"][fin] / 1e-4, rtol=0.05)
    np.testing.assert_allclose(r2[fin] / r1[fin], 4.0, rtol=1e-6)


# --------------------------------------------------------------------------- #
# QC
# --------------------------------------------------------------------------- #


def test_qc_mask_depth_reb_and_flags(tmp_path):
    _write_run(tmp_path)
    args = _args(tmp_path)
    table = G.sweep(args)
    ok = G.qc_mask(table, args)
    assert ok.any()
    assert np.all(table["P"][ok] >= args.min_depth)
    # an impossible Re_b floor kills everything
    assert not G.qc_mask(table, _args(tmp_path, min_reb=1e9)).any()
    # qc_drop flag masks windows unless --no-qc-flags
    table.cols["qc"] = np.ones(table.n)
    assert not G.qc_mask(table, args).any()
    assert G.qc_mask(table, _args(tmp_path, qc_flags=False)).any()


# --------------------------------------------------------------------------- #
# figures / driver
# --------------------------------------------------------------------------- #


def test_run_writes_both_figures(tmp_path):
    _write_run(tmp_path)
    adcp = _write_adcp(tmp_path / "wh300.nc")
    out = tmp_path / "figs"
    got = G.run(_args(tmp_path, adcp=[str(adcp)], out_dir=str(out)))
    assert got == str(out)
    assert (out / "gamma_scaling.png").exists()
    assert (out / "gamma_thorpe_compare.png").exists()


def test_run_without_adcp_two_panels(tmp_path):
    _write_run(tmp_path)
    out = tmp_path / "figs"
    G.run(_args(tmp_path, out_dir=str(out)))
    assert (out / "gamma_scaling.png").exists()


def test_run_without_profiles_skips_compare_figure(tmp_path):
    _write_run(tmp_path, with_profiles=False)
    out = tmp_path / "figs"
    G.run(_args(tmp_path, out_dir=str(out)))
    assert (out / "gamma_scaling.png").exists()
    assert not (out / "gamma_thorpe_compare.png").exists()


def test_sections_coloring(tmp_path):
    _write_run(tmp_path)
    sec = tmp_path / "sections.yaml"
    sec.write_text(
        "sections:\n"
        "  - name: first\n"
        '    start: "2025-02-11T00:00:00Z"\n'
        '    stop:  "2025-02-11T12:00:00Z"\n'
        "    xaxis: {method: time}\n"
    )
    args = _args(tmp_path, sections=str(sec))
    table = G.sweep(args)
    idx, names = G._section_colors(table, args)
    assert names == ["first"]
    # profile 0 is on Feb 11, profile 1 a day later -> half the windows.
    assert (idx == 0).sum() == table.n // 2
    out = tmp_path / "figs"
    G.run(_args(tmp_path, sections=str(sec), out_dir=str(out)))
    assert (out / "gamma_scaling.png").exists()


def test_cli_registers_gamma_scaling():
    from odas_tpw.perturb.plot.cli import build_parser

    args = build_parser().parse_args(
        ["gamma-scaling", "--root", "x", "--adcp", "a.nc", "--adcp", "b.nc"]
    )
    assert args.command == "gamma-scaling"
    assert args.adcp == ["a.nc", "b.nc"]
    assert args.route == "density" and args.min_reb == 20.0


def test_route_temperature_uses_its_own_floor_and_patch(tmp_path):
    _write_run(tmp_path)
    # An impossible density floor kills the density route entirely...
    t_dens = G.sweep(_args(tmp_path, lt_floor_density=1e9))
    assert not (t_dens["resolved"] > 0).any()
    # ...but the temperature route is untouched by it (route plumbing).
    t_temp = G.sweep(_args(tmp_path, route="temperature", lt_floor_density=1e9))
    res = t_temp["resolved"] > 0
    assert res.any()
    # and the temperature route's patch N2 (alpha*g) feeds N2_scale there.
    np.testing.assert_allclose(t_temp["N2_scale"][res], t_temp["N2p_temp"][res], rtol=1e-12)


def test_min_run_and_keep_truncated_gate_resolution(tmp_path):
    _write_run(tmp_path)
    # A run-length demand no real overturn meets -> nothing resolves.
    t = G.sweep(_args(tmp_path, min_run=10_000))
    assert not (t["resolved"] > 0).any()
    # An overturn spanning most of the sort window is edge-truncated:
    # excluded by default, admitted with --keep-truncated.
    big = tmp_path / "big"
    _write_run(big, ot_span=(26.2, 29.8))  # ~3.2 m of a ~3.6 m window
    t_def = G.sweep(_args(big))
    t_keep = G.sweep(_args(big, keep_truncated=True))
    assert not (t_def["resolved"] > 0).any()
    assert (t_keep["resolved"] > 0).any()


def test_duplicate_adcp_stems_not_blended(tmp_path):
    _write_run(tmp_path)
    a1 = _write_adcp(tmp_path / "leg1" / "wh300.nc", du_dz=0.01)
    a2 = _write_adcp(tmp_path / "leg2" / "wh300.nc", du_dz=0.005)
    table = G.sweep(_args(tmp_path, adcp=[str(a1), str(a2)]))
    assert len(table.rig) == 2  # second file gets a disambiguated label
    assert "wh300" in table.rig
    (other,) = [k for k in table.rig if k != "wh300"]
    r1, r2 = table.rig["wh300"], table.rig[other]
    fin = np.isfinite(r1) & np.isfinite(r2)
    assert fin.any()
    np.testing.assert_allclose(r2[fin] / r1[fin], 4.0, rtol=1e-6)


def test_select_requires_sections(tmp_path):
    _write_run(tmp_path)
    with pytest.raises(SystemExit, match="--select requires --sections"):
        list(G.build_figures(_args(tmp_path, select=["s1"])))


def test_section_umbrella_does_not_swallow_narrow_sections(tmp_path):
    _write_run(tmp_path)
    sec = tmp_path / "sections.yaml"
    sec.write_text(
        "sections:\n"
        "  - name: everything\n"  # unbounded umbrella FIRST
        "    xaxis: {method: time}\n"
        "  - name: narrow\n"
        '    start: "2025-02-11T00:00:00Z"\n'
        '    stop:  "2025-02-11T12:00:00Z"\n'
        "    xaxis: {method: time}\n"
    )
    args = _args(tmp_path, sections=str(sec))
    table = G.sweep(args)
    idx, names = G._section_colors(table, args)
    assert names == ["everything", "narrow"]
    # The narrow section wins its own window; the umbrella takes the rest.
    assert (idx == 1).sum() == table.n // 2
    assert (idx == 0).sum() == table.n - table.n // 2
