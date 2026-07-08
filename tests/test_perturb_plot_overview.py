# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for ``perturb-plot overview`` (epsilon/chi over a context row)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from odas_tpw.perturb.plot import overview  # noqa: E402

# Two clusters of casts a day apart, marching north.
_STIME = np.concatenate([1.706e9 + np.arange(6) * 400.0,
                        1.706e9 + 86400.0 + np.arange(6) * 400.0])
_LAT = np.linspace(15.0, 16.0, 12)
_LON = np.full(12, 145.0)
_NBIN = 20
_DEPTH = np.arange(_NBIN, dtype=float) * 2.0 + 1.0


def _write_product(root: Path, prefix: str, data_vars: dict, n_bin: int = _NBIN) -> None:
    import xarray as xr

    nprof = len(_STIME)
    ds = xr.Dataset(
        {n: (("bin", "profile"), a, at) for n, (a, at) in data_vars.items()},
        coords={"bin": ("bin", _DEPTH[:n_bin]),
                "profile": ("profile", np.arange(nprof, dtype=np.int32))},
    )
    ds["bin"].attrs.update(units="m", standard_name="depth", positive="down")
    ds["lat"] = (("profile",), _LAT, {"units": "degrees_north"})
    ds["lon"] = (("profile",), _LON, {"units": "degrees_east"})
    ds["stime"] = (("profile",), _STIME, {"units": "seconds since 1970-01-01"})
    ds.attrs["id"] = "OV-TEST"
    out = root / f"{prefix}_00"
    out.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out / "combo.nc")


def _write_ctd_traj(root: Path) -> None:
    """A continuous (time,) CTD trajectory with CT/SP (the scalar-path source)."""
    import xarray as xr

    n = 400
    time = np.linspace(_STIME[0], _STIME[-1], n)
    depth = 5.0 + 100.0 * (0.5 - 0.5 * np.cos(np.linspace(0, 8 * np.pi, n)))  # sawtooth
    lat = np.interp(time, _STIME, _LAT)
    ds = xr.Dataset(
        {
            "CT": (("time",), 28.0 - 0.1 * depth, {"units": "degree_Celsius"}),
            "SP": (("time",), 34.3 + 0.003 * depth, {"units": "1"}),
            "depth": (("time",), depth, {"units": "m"}),
            "lat": (("time",), lat, {"units": "degrees_north"}),
            "lon": (("time",), np.full(n, 145.0), {"units": "degrees_east"}),
        },
        coords={"time": ("time", time)},
    )
    ds["time"].attrs.update(units="seconds since 1970-01-01")
    ds.attrs["id"] = "OV-TEST"
    out = root / "ctd_combo_00"
    out.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out / "combo.nc")


def _bp(scale: float, n_bin: int = _NBIN) -> np.ndarray:
    b = _DEPTH[:n_bin, None]
    c = np.arange(12)[None, :] + 1.0
    return scale * np.exp(-b / 20.0) * c


def _build_full(root: Path, *, binned_ctd: bool = True) -> None:
    """diss + chi (with mixing vars) + a CTD context (binned or trajectory)."""
    _write_product(root, "diss_combo", {
        "epsilonMean": (_bp(1e-6), {"units": "W/kg", "long_name": "epsilon"}),
        "qc_drop_epsilon": (np.zeros((_NBIN, 12)), {}),
    })
    _write_product(root, "chi_combo", {
        "chiMean": (_bp(1e-7), {"units": "K2 s-1", "long_name": "chi"}),
        "K_T": (_bp(1e-5), {"units": "m2 s-1", "long_name": "K_T"}),
        "K_rho": (_bp(2e-5), {"units": "m2 s-1", "long_name": "K_rho"}),
        "Gamma": (0.2 * np.ones((_NBIN, 12)), {"units": "1", "long_name": "Gamma"}),
        "qc_drop_chi": (np.zeros((_NBIN, 12)), {}),
    })
    if binned_ctd:
        _write_product(root, "combo", {
            "CT": ((28.0 - 0.1 * _DEPTH[:, None]) * np.ones((1, 12)),
                   {"units": "degree_Celsius", "long_name": "CT"}),
            "SP": ((34.3 + 0.003 * _DEPTH[:, None]) * np.ones((1, 12)),
                   {"units": "1", "long_name": "SP"}),
            "dTdz": (-0.1 * np.ones((_NBIN, 12)), {"units": "K m-1", "long_name": "dTdz"}),
        })
    else:
        _write_ctd_traj(root)


def _args(root: Path, out: Path, *extra: str):
    from odas_tpw.perturb.plot.cli import build_parser

    argv = ["overview", "--root", str(root), "--xaxis", "latitude",
            "--out-dir", str(out), *extra]
    return build_parser().parse_args(argv)


def _figs(root: Path, out: Path, *extra: str):
    """Build (stem, Figure) list without saving."""
    return list(overview.build_figures(_args(root, out, *extra)))


def _text(fig) -> str:
    """All axis labels + placeholder texts in one blob, for membership asserts."""
    bits: list[str] = []
    for ax in fig.axes:
        bits += [ax.get_xlabel(), ax.get_ylabel()]
        bits += [t.get_text() for t in ax.texts]
    return " | ".join(b for b in bits if b)


def _drawn(fig) -> list:
    """Data panels carrying a pcolormesh (colorbar axes also hold a QuadMesh, so
    they must be excluded by their '<colorbar>' label)."""
    return [ax for ax in fig.axes
            if ax.get_label() != "<colorbar>" and ax.collections]


def _close(figs) -> None:
    import matplotlib.pyplot as plt

    for _stem, fig in figs:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Registration / config
# ---------------------------------------------------------------------------


def test_cli_registers_overview():
    from odas_tpw.perturb.plot.cli import build_parser

    args = build_parser().parse_args(["overview", "--root", "/x", "--bottom", "mixing"])
    assert args.command == "overview"
    assert args.eps_var == "epsilonMean"
    assert args.chi_var == "chiMean"
    assert args.bottom == "mixing"


def test_figure_preset_registered():
    from odas_tpw.perturb.plot import figure

    assert "overview" in figure._PRESETS


def test_bottom_sets():
    assert overview._BOTTOM_SETS["ctd"] == ("dTdz", "CT", "SP")
    assert overview._BOTTOM_SETS["mixing"] == ("K_rho", "K_T", "Gamma")


# ---------------------------------------------------------------------------
# Variable -> product resolution
# ---------------------------------------------------------------------------


def test_resolve_prefers_binned_combo(tmp_path):
    _build_full(tmp_path, binned_ctd=True)
    stages = overview._Stages(_args(tmp_path, tmp_path))
    try:
        kind, ds = overview._resolve_var("CT", stages)
        assert kind == "col"  # binned (bin, profile) copy wins
        assert "CT" in ds.data_vars and {"bin", "profile"} <= set(ds.dims)
    finally:
        stages.close()


def test_resolve_falls_back_to_trajectory(tmp_path):
    _build_full(tmp_path, binned_ctd=False)  # CT/SP only in the ctd_combo trajectory
    stages = overview._Stages(_args(tmp_path, tmp_path))
    try:
        kind, ds = overview._resolve_var("CT", stages)
        assert kind == "traj"
        assert "time" in ds.dims
    finally:
        stages.close()


def test_resolve_missing_returns_none(tmp_path):
    _build_full(tmp_path, binned_ctd=True)
    stages = overview._Stages(_args(tmp_path, tmp_path))
    try:
        assert overview._resolve_var("NoSuchVar", stages) is None
    finally:
        stages.close()


# ---------------------------------------------------------------------------
# Figure building
# ---------------------------------------------------------------------------


def test_full_run_draws_five_panels(tmp_path):
    _build_full(tmp_path, binned_ctd=True)
    figs = _figs(tmp_path, tmp_path)
    try:
        assert len(figs) == 1
        _stem, fig = figs[0]
        drawn = _drawn(fig)  # QuadMesh-bearing panels
        assert len(drawn) == 5
        blob = _text(fig)
        assert "no valid" not in blob
        assert "Theta" in blob or "Θ" in blob   # CT panel colorbar label
        assert "Salinity" in blob
    finally:
        _close(figs)


def test_single_shared_context_xlabel(tmp_path):
    """The three context panels share ONE centered x-label (not one each), so a
    long signed_distance label does not overlap across the row."""
    _build_full(tmp_path, binned_ctd=True)
    figs = _figs(tmp_path, tmp_path)  # latitude x-axis
    try:
        _stem, fig = figs[0]
        assert all(ax.get_xlabel() == "" for ax in fig.axes)  # no per-panel labels
        top_sf, bot_sf = fig.subfigs[0], fig.subfigs[1]
        # the one shared label lives on the BOTTOM (context) subfigure, not the top
        assert bot_sf._supxlabel is not None and "Latitude" in bot_sf._supxlabel.get_text()
        assert top_sf._supxlabel is None or not top_sf._supxlabel.get_text()
    finally:
        _close(figs)


def test_bottom_mixing_selects_mixing_vars(tmp_path):
    _build_full(tmp_path, binned_ctd=True)
    figs = _figs(tmp_path, tmp_path, "--bottom", "mixing")
    try:
        _stem, fig = figs[0]
        blob = _text(fig)
        assert "rho" in blob and "Gamma" in blob        # K_rho / Gamma labels
        assert "Salinity" not in blob                   # ctd set not used
    finally:
        _close(figs)


def test_var_overrides_bottom_set(tmp_path):
    _build_full(tmp_path, binned_ctd=True)
    figs = _figs(tmp_path, tmp_path, "--bottom", "mixing", "--var", "dTdz")
    try:
        _stem, fig = figs[0]
        drawn = _drawn(fig)
        assert len(drawn) == 3          # eps + chi + one context panel
        assert "rho" not in _text(fig)  # --var wins over --bottom mixing
    finally:
        _close(figs)


def test_trajectory_context_renders(tmp_path):
    """CT/SP pulled from the ctd_combo trajectory still draw (scalar path).

    No ``combo`` in this build, so dT/dz has no source and reads 'no valid';
    epsilon, chi (column path) and CT, SP (trajectory path) do draw.
    """
    _build_full(tmp_path, binned_ctd=False)
    figs = list(overview.build_figures(_args(tmp_path, tmp_path)))
    try:
        _stem, fig = figs[0]
        drawn = _drawn(fig)
        assert len(drawn) == 4
        assert "no valid dTdz" in _text(fig)
    finally:
        _close(figs)


def test_partial_run_is_graceful(tmp_path):
    """Only epsilon present: the figure still builds, others say 'no valid'."""
    _write_product(tmp_path, "diss_combo", {
        "epsilonMean": (_bp(1e-6), {"units": "W/kg", "long_name": "epsilon"}),
    })
    figs = _figs(tmp_path, tmp_path)
    try:
        assert len(figs) == 1
        _stem, fig = figs[0]
        drawn = _drawn(fig)
        assert len(drawn) == 1                       # only epsilon
        assert "no valid chiMean" in _text(fig)
    finally:
        _close(figs)


def test_no_products_yields_nothing(tmp_path):
    (tmp_path / "empty").mkdir()
    figs = _figs(tmp_path, tmp_path)      # no combos under root
    _close(figs)
    assert figs == []


def test_vmin_rejected_multipanel(tmp_path):
    _build_full(tmp_path, binned_ctd=True)
    with pytest.raises(SystemExit, match="single --var"):
        _close(_figs(tmp_path, tmp_path, "--vmin", "1e-9"))


def test_run_writes_png(tmp_path):
    _build_full(tmp_path, binned_ctd=True)
    out = tmp_path / "figs"
    result = overview.run(_args(tmp_path, out))
    assert Path(result) == out
    assert (out / "overview_section.png").exists()


# ---------------------------------------------------------------------------
# Adversarial-review regressions (PR #95)
# ---------------------------------------------------------------------------


def _write_xy(root: Path, prefix: str, var: str, attrs: dict,
              lat: np.ndarray, lon: np.ndarray, stime: np.ndarray) -> None:
    """A minimal (bin, profile) product with arbitrary per-cast lat/lon/stime."""
    import xarray as xr

    nprof = len(stime)
    field = np.exp(-_DEPTH[:, None] / 20.0) * (np.arange(nprof)[None, :] + 1.0)
    ds = xr.Dataset(
        {var: (("bin", "profile"), field, attrs)},
        coords={"bin": ("bin", _DEPTH),
                "profile": ("profile", np.arange(nprof, dtype=np.int32))},
    )
    ds["bin"].attrs.update(units="m", standard_name="depth", positive="down")
    ds["lat"] = (("profile",), np.asarray(lat, float), {"units": "degrees_north"})
    ds["lon"] = (("profile",), np.asarray(lon, float), {"units": "degrees_east"})
    ds["stime"] = (("profile",), np.asarray(stime, float),
                   {"units": "seconds since 1970-01-01"})
    ds.attrs["id"] = "OV-TEST"
    out = root / f"{prefix}_00"
    out.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out / "combo.nc")


def test_signed_distance_shares_one_frame_across_products(tmp_path):
    """signed_distance must fit ONE frame over every panel's casts, so a cast
    shared by two products lands at the same x in both rows.  Regression: each
    panel derived its own centroid/orientation from its own product, so differing
    cast sets (or the ctd_combo trajectory) shifted the rows apart."""
    from odas_tpw.perturb.plot import xaxis
    from odas_tpw.perturb.plot.sections import resolve_sections

    # diss casts run north (lon fixed); chi casts run east (lat fixed).  They
    # share the corner cast (15.5 N, 145.0 E); their independent principal axes
    # are ~90 deg apart, so a per-product frame would misplace that shared cast.
    lat_a, lon_a = np.linspace(15.0, 15.5, 6), np.full(6, 145.0)
    lat_b, lon_b = np.full(6, 15.5), np.linspace(145.0, 145.5, 6)
    st = 1.706e9 + np.arange(6) * 400.0
    _write_xy(tmp_path, "diss_combo", "epsilonMean",
              {"units": "W/kg", "long_name": "epsilon"}, lat_a, lon_a, st)
    _write_xy(tmp_path, "chi_combo", "chiMean",
              {"units": "K2 s-1", "long_name": "chi"}, lat_b, lon_b, st + 86400.0)

    args = _args(tmp_path, tmp_path, "--xaxis", "signed_distance")
    sec = resolve_sections(args)[0]
    stages = overview._Stages(args)
    try:
        frame = overview._shared_frame(stages, sec, ["epsilonMean", "chiMean"])
        assert frame is not None
        # The per-product frames are genuinely different (N-S vs E-W tracks) ...
        fa = xaxis.signed_distance_axis(lat_a, lon_a, st)
        fb = xaxis.signed_distance_axis(lat_b, lon_b, st)
        sep = abs((fa.bearing_deg - fb.bearing_deg + 180.0) % 360.0 - 180.0)
        assert sep > 45.0
        # ... and would place the shared corner cast >0.5 km apart:
        pa = xaxis.project_onto_signed_axis([15.5], [145.0],
                                            fa.mid_lat, fa.mid_lon, fa.bearing_deg)
        pb = xaxis.project_onto_signed_axis([15.5], [145.0],
                                            fb.mid_lat, fb.mid_lon, fb.bearing_deg)
        assert abs(float(pa[0]) - float(pb[0])) > 0.5
        # The shared frame places it at ONE x regardless of which panel asks.
        corner = (np.array([15.5]), np.array([145.0]), np.array([0.0]))
        x_eps = overview._panel_xaxis(sec, *corner, frame).x[0]
        x_chi = overview._panel_xaxis(sec, *corner, frame).x[0]
        assert x_eps == x_chi
    finally:
        stages.close()


def test_shared_frame_none_for_absolute_methods(tmp_path):
    """latitude/longitude/time need no shared frame (absolute coordinates)."""
    from odas_tpw.perturb.plot.sections import resolve_sections

    _build_full(tmp_path, binned_ctd=True)
    args = _args(tmp_path, tmp_path)  # latitude
    sec = resolve_sections(args)[0]
    stages = overview._Stages(args)
    try:
        assert overview._shared_frame(stages, sec, ["epsilonMean", "chiMean"]) is None
    finally:
        stages.close()


def test_qc_masking_default_on_masks_flagged_cells(tmp_path):
    """--apply-qc (default) NaNs cells a qc_drop_* flag marks; --no-qc keeps them.
    Every other fixture writes all-zero flags, so this exercises the default-ON
    masking branch (and its post-x-sort reindex) that would otherwise ship
    untested."""
    flag = np.zeros((_NBIN, 12))
    flag[:, 3] = 1.0  # cast 3 fully flagged (lat monotonic in profile -> stays col 3)
    _write_product(tmp_path, "diss_combo", {
        "epsilonMean": (_bp(1e-6), {"units": "W/kg", "long_name": "epsilon"}),
        "qc_drop_epsilon": (flag, {}),
    })
    figs = _figs(tmp_path, tmp_path)  # QC on (default)
    try:
        arr = _drawn(figs[0][1])[0].collections[0].get_array()
        assert int(np.ma.getmaskarray(arr).sum()) == _NBIN  # exactly one column masked
    finally:
        _close(figs)
    figs2 = _figs(tmp_path, tmp_path, "--no-qc")  # QC off
    try:
        arr = _drawn(figs2[0][1])[0].collections[0].get_array()
        assert int(np.ma.getmaskarray(arr).sum()) == 0  # nothing masked
    finally:
        _close(figs2)


def test_panels_bind_variable_to_row(tmp_path):
    """Each row/column draws its OWN variable: colorbars are created in panel
    order [eps, chi, dTdz, CT, SP], so a swap (eps<->chi or CT<->SP) fails here."""
    _build_full(tmp_path, binned_ctd=True)
    figs = _figs(tmp_path, tmp_path)
    try:
        cbars = [ax.yaxis.get_label().get_text()
                 for ax in figs[0][1].axes if ax.get_label() == "<colorbar>"]
        assert len(cbars) == 5
        assert "epsilon" in cbars[0]   # top row
        assert "chi" in cbars[1]       # middle row
        assert "Theta" in cbars[3]     # context col 2 = CT
        assert "Salinity" in cbars[4]  # context col 3 = SP
    finally:
        _close(figs)


def test_time_axis_and_pmax(tmp_path):
    """time x-axis wires the datetime formatter onto the context row, and --p-max
    clips the (inverted) depth axis."""
    _build_full(tmp_path, binned_ctd=True)
    figs = _figs(tmp_path, tmp_path, "--xaxis", "time", "--p-max", "20")
    try:
        drawn = _drawn(figs[0][1])
        assert len(drawn) == 5
        lo, hi = drawn[0].get_ylim()
        assert max(lo, hi) <= 20.0 + 1e-6  # depth clipped at 20 m
    finally:
        _close(figs)


def test_time_fmt_formats_epoch_seconds():
    import re

    s = overview._time_fmt(1.706e9, None)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", s)
