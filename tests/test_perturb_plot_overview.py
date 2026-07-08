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
        supx = [getattr(sf, "_supxlabel", None) for sf in fig.subfigs]
        texts = [t.get_text() for t in supx if t is not None and t.get_text()]
        assert len(texts) == 1 and "Latitude" in texts[0]  # one shared label
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
