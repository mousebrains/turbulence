# Jun-2026, Claude and Pat Welch, pat@mousebrains.com
"""Config-loader and end-to-end tests for ``perturb-plot section``.

The synthetic CTD combo deliberately moves lat/lon, sawtooths depth (so the
grid's cell-averaging is exercised by revisited depth bins), and carries a
negative sigma0 near the surface (so signed colour limits are exercised) —
the clean fixed-field fixtures elsewhere would not catch those.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from odas_tpw.perturb.plot import section  # noqa: E402


def _write_ctd_combo(root: Path, n_cast: int = 4, per: int = 60) -> None:
    """Write a tiny sawtooth CTD trajectory to ``root/ctd_combo_00/combo.nc``."""
    import xarray as xr

    depth_list, t_list, lat_list, lon_list = [], [], [], []
    t0 = np.datetime64("2025-01-20T00:00:00")
    clock = 0
    for k in range(n_cast):
        # Down then up: a sawtooth that revisits each depth bin twice.
        down = np.linspace(0.0, 100.0, per // 2)
        prof = np.concatenate([down, down[::-1]])
        depth_list.append(prof)
        for _ in prof:
            t_list.append(t0 + np.timedelta64(clock, "s"))
            clock += 1
        # Track moves NE across casts.
        lat_list.append(np.full(prof.size, 18.0 + 0.5 * k))
        lon_list.append(np.full(prof.size, 130.0 + 0.4 * k))

    depth = np.concatenate(depth_list)
    time = np.array(t_list, dtype="datetime64[ns]")
    lat = np.concatenate(lat_list)
    lon = np.concatenate(lon_list)
    # Physically-flavoured scalars; sigma0 dips negative in the warm surface.
    jac_t = 28.0 - 0.1 * depth
    sp = 34.5 + 0.01 * depth
    sigma0 = -1.0 + 0.25 * depth  # negative for depth < 4 m
    dtdz = 0.02 + 0.0001 * depth  # one-signed (stable) -> diverging-cmap path

    ds = xr.Dataset(
        {
            "JAC_T": (("time",), jac_t, {"long_name": "in-situ temperature",
                                          "units": "degree_Celsius"}),
            "SP": (("time",), sp, {"long_name": "practical salinity", "units": "PSU"}),
            "sigma0": (("time",), sigma0, {"long_name": "potential density anomaly",
                                            "units": "kg/m^3"}),
            "dTdz": (("time",), dtdz, {"long_name": "temperature gradient",
                                        "units": "K m-1"}),
            "depth": (("time",), depth, {"units": "m", "positive": "down"}),
            "lat": (("time",), lat, {"units": "degrees_north"}),
            "lon": (("time",), lon, {"units": "degrees_east"}),
        },
        coords={"time": ("time", time)},
    )
    ds.attrs["id"] = "test_ctd_combo"
    out = root / "ctd_combo_00"
    out.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out / "combo.nc")


# ---------------------------------------------------------------------------
# Config loader / validation
# ---------------------------------------------------------------------------


def test_parse_time_utc_and_offset_rejection():
    assert section._parse_time(None) is None
    assert section._parse_time("2025-01-20T00:00:00Z") == np.datetime64("2025-01-20T00:00:00")
    assert section._parse_time("2025-01-20T06:30:00") == np.datetime64("2025-01-20T06:30:00")
    with pytest.raises(ValueError):
        section._parse_time("2025-01-20T00:00:00+09:00")


def test_load_sections_valid(tmp_path: Path):
    cfg = tmp_path / "sections.yaml"
    cfg.write_text(
        "sections:\n"
        "  - name: t\n"
        "    xaxis: {method: time}\n"
        "  - name: line\n"
        "    start: '2025-01-20T00:00:00Z'\n"
        "    xaxis:\n"
        "      method: along_line\n"
        "      units: nm\n"
        "      waypoints: [[18.0, 130.0], [20.0, 132.0]]\n"
    )
    secs = section.load_sections(str(cfg))
    assert [s.name for s in secs] == ["t", "line"]
    assert secs[0].method == "time"
    assert secs[1].method == "along_line"
    assert secs[1].params["units"] == "nm"
    assert secs[1].params["waypoints"] == [[18.0, 130.0], [20.0, 132.0]]
    assert secs[1].start == np.datetime64("2025-01-20T00:00:00")


def test_load_sections_rejects_bad_specs(tmp_path: Path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("sections: []\n")
    with pytest.raises(ValueError):
        section.load_sections(str(empty))

    bad_method = tmp_path / "bad.yaml"
    bad_method.write_text("sections:\n  - {name: x, xaxis: {method: spiral}}\n")
    with pytest.raises(ValueError):
        section.load_sections(str(bad_method))

    no_wp = tmp_path / "nowp.yaml"
    no_wp.write_text("sections:\n  - {name: x, xaxis: {method: along_line}}\n")
    with pytest.raises(ValueError):
        section.load_sections(str(no_wp))

    no_pt = tmp_path / "nopt.yaml"
    no_pt.write_text("sections:\n  - {name: x, xaxis: {method: distance_from_point}}\n")
    with pytest.raises(ValueError):
        section.load_sections(str(no_pt))


def test_parse_waypoints():
    assert section._parse_waypoints("18.0,130.0; 20.0,132.5") == [[18.0, 130.0], [20.0, 132.5]]
    with pytest.raises(SystemExit):
        section._parse_waypoints("18.0,130.0")  # only one point


# ---------------------------------------------------------------------------
# End-to-end rendering
# ---------------------------------------------------------------------------


def test_adhoc_time_section_writes_png(tmp_path: Path):
    _write_ctd_combo(tmp_path)
    out_dir = tmp_path / "figs"
    rc = section.run(
        argparse.Namespace(
            root=str(tmp_path), ctd_combo=None, sections=None, out_dir=str(out_dir),
            var=None, z_bin=2.0, x_bin=None, depth_max=None, vmin=None, vmax=None,
            name="adhoc_time", xaxis="time", start=None, stop=None, point=None,
            waypoints=None, units="km",
        )
    )
    png = out_dir / "section_adhoc_time.png"
    assert png.exists() and png.stat().st_size > 0
    assert rc == str(out_dir)


def _run_cli(argv):
    from odas_tpw.perturb.plot.cli import main

    return main(argv)


def test_cli_sections_yaml_multipanel(tmp_path: Path):
    _write_ctd_combo(tmp_path)
    cfg = tmp_path / "sections.yaml"
    cfg.write_text(
        "sections:\n"
        "  - name: by_lat\n"
        "    xaxis: {method: latitude}\n"
        "  - name: dist\n"
        "    xaxis: {method: distance_from_point, units: km, point: [18.0, 130.0]}\n"
        "  - name: line\n"
        "    xaxis:\n"
        "      method: along_line\n"
        "      waypoints: [[18.0, 130.0], [20.0, 132.0]]\n"
    )
    rc = _run_cli([
        "section", "--root", str(tmp_path), "--sections", str(cfg),
        "--out-dir", str(tmp_path), "--var", "JAC_T", "--var", "sigma0",
    ])
    assert rc == 0
    for name in ("by_lat", "dist", "line"):
        png = tmp_path / f"section_{name}.png"
        assert png.exists() and png.stat().st_size > 0


def test_empty_time_window_skipped(tmp_path: Path):
    _write_ctd_combo(tmp_path)
    cfg = tmp_path / "sections.yaml"
    cfg.write_text(
        "sections:\n"
        "  - name: future\n"
        "    start: '2099-01-01T00:00:00Z'\n"
        "    xaxis: {method: time}\n"
    )
    rc = _run_cli(["section", "--root", str(tmp_path), "--sections", str(cfg),
                   "--out-dir", str(tmp_path)])
    assert rc == 0
    assert not (tmp_path / "section_future.png").exists()  # no data -> no figure


def test_default_variables(tmp_path: Path):
    import xarray as xr

    _write_ctd_combo(tmp_path)
    with xr.open_dataset(tmp_path / "ctd_combo_00" / "combo.nc") as ds:
        # dTdz is present but not a default panel.
        assert section._default_variables(ds) == ["JAC_T", "SP", "sigma0"]


def test_diverging_variable_renders(tmp_path: Path):
    """A one-signed diverging field (dTdz) exercises the symmetric-norm path."""
    _write_ctd_combo(tmp_path)
    rc = _run_cli(["section", "--root", str(tmp_path), "--out-dir", str(tmp_path),
                   "--name", "grad", "--xaxis", "time", "--var", "dTdz"])
    assert rc == 0
    png = tmp_path / "section_grad.png"
    assert png.exists() and png.stat().st_size > 0
