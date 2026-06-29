# Jun-2026, Claude and Pat Welch, pat@mousebrains.com
"""Config-loader and end-to-end tests for ``perturb-plot scalar``.

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

from odas_tpw.perturb.plot import scalar  # noqa: E402


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


def test_var_label_omits_dimensionless_units():
    """A dimensionless unit ("1", e.g. practical salinity after PSU->"1") must
    not render as "[1]"; meaningful units still show (#38)."""
    import xarray as xr

    from odas_tpw.perturb.plot.sections import var_label

    ds = xr.Dataset(
        {
            "SP": (("t",), np.zeros(2),
                   {"long_name": "practical salinity (PSU)", "units": "1"}),
            "depth": (("t",), np.zeros(2), {"long_name": "depth", "units": "m"}),
        }
    )
    assert var_label(ds, "SP") == "practical salinity (PSU)"  # no "[1]"
    assert var_label(ds, "depth") == "depth [m]"  # real unit still shown


def test_time_subset_handles_numeric_time_axis():
    """A combo whose time is numeric epoch seconds (no CF units decoded) must
    subset cleanly, not raise a cryptic UFuncTypeError (#66)."""
    import xarray as xr

    # epoch seconds for 2025-01-20T00:00:00 .. +9 s
    t0 = int(np.datetime64("2025-01-20T00:00:00", "s").astype("int64"))
    t = np.arange(t0, t0 + 10, dtype=np.float64)
    ds = xr.Dataset({"depth": (("time",), np.arange(10.0))},
                    coords={"time": ("time", t)})
    sec = scalar.Section(
        name="w", method="time",
        start=np.datetime64("2025-01-20T00:00:03"),
        stop=np.datetime64("2025-01-20T00:00:06"),
    )
    sub = scalar._time_subset(ds, sec)
    assert sub.sizes["time"] == 4  # seconds 3,4,5,6 inclusive


def test_parse_time_utc_and_offset_rejection():
    assert scalar._parse_time(None) is None
    assert scalar._parse_time("2025-01-20T00:00:00Z") == np.datetime64("2025-01-20T00:00:00")
    assert scalar._parse_time("2025-01-20T06:30:00") == np.datetime64("2025-01-20T06:30:00")
    with pytest.raises(ValueError):
        scalar._parse_time("2025-01-20T00:00:00+09:00")


def test_parse_time_rejects_space_separated_offset():
    """An unquoted offset timestamp in YAML resolves to a tz-aware datetime
    whose str() uses a SPACE separator (e.g. '2025-01-20 00:00:00+09:00').
    parse_time must reject the offset, not silently shift it 9h — the offset
    check has to look past a space as well as a 'T'."""
    import datetime

    aware = datetime.datetime(
        2025, 1, 20, 0, 0, 0,
        tzinfo=datetime.timezone(datetime.timedelta(hours=9)),
    )
    with pytest.raises(ValueError, match="must be UTC"):
        scalar._parse_time(aware)
    with pytest.raises(ValueError, match="must be UTC"):
        scalar._parse_time("2025-01-20 00:00:00+09:00")
    with pytest.raises(ValueError, match="must be UTC"):
        scalar._parse_time("2025-01-20 00:00:00-05:00")
    # a space-separated UTC time (naive datetime str) must still parse fine
    assert scalar._parse_time("2025-01-20 06:30:00") == np.datetime64("2025-01-20T06:30:00")
    assert scalar._parse_time(datetime.datetime(2025, 1, 20, 6, 30, 0)) == \
        np.datetime64("2025-01-20T06:30:00")


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
    secs = scalar.load_sections(str(cfg))
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
        scalar.load_sections(str(empty))

    bad_method = tmp_path / "bad.yaml"
    bad_method.write_text("sections:\n  - {name: x, xaxis: {method: spiral}}\n")
    with pytest.raises(ValueError):
        scalar.load_sections(str(bad_method))

    no_wp = tmp_path / "nowp.yaml"
    no_wp.write_text("sections:\n  - {name: x, xaxis: {method: along_line}}\n")
    with pytest.raises(ValueError):
        scalar.load_sections(str(no_wp))

    no_pt = tmp_path / "nopt.yaml"
    no_pt.write_text("sections:\n  - {name: x, xaxis: {method: distance_from_point}}\n")
    with pytest.raises(ValueError):
        scalar.load_sections(str(no_pt))


def test_parse_waypoints():
    assert scalar._parse_waypoints("18.0,130.0; 20.0,132.5") == [[18.0, 130.0], [20.0, 132.5]]
    with pytest.raises(SystemExit):
        scalar._parse_waypoints("18.0,130.0")  # only one point


# ---------------------------------------------------------------------------
# End-to-end rendering
# ---------------------------------------------------------------------------


def test_adhoc_time_section_writes_png(tmp_path: Path):
    _write_ctd_combo(tmp_path)
    out_dir = tmp_path / "figs"
    rc = scalar.run(
        argparse.Namespace(
            root=str(tmp_path), ctd_combo=None, sections=None, select=None,
            out_dir=str(out_dir),
            var=None, z_bin=2.0, x_bin=None, depth_max=None, vmin=None, vmax=None,
            clim=None, name="adhoc_time", xaxis="time", start=None, stop=None,
            point=None, waypoints=None, units="km",
        )
    )
    png = out_dir / "scalar_adhoc_time.png"
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
        "scalar", "--root", str(tmp_path), "--sections", str(cfg),
        "--out-dir", str(tmp_path), "--var", "JAC_T", "--var", "sigma0",
    ])
    assert rc == 0
    for name in ("by_lat", "dist", "line"):
        png = tmp_path / f"scalar_{name}.png"
        assert png.exists() and png.stat().st_size > 0


def _three_section_yaml(tmp_path: Path) -> Path:
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
    return cfg


def test_select_sections_filters_and_validates():
    secs = [scalar.Section(name=n, method="time") for n in ("a", "b", "c")]
    # single name
    assert [s.name for s in scalar._select_sections(secs, ["b"])] == ["b"]
    # comma-separated + repeated, file order preserved (not request order)
    assert [s.name for s in scalar._select_sections(secs, ["c,a"])] == ["a", "c"]
    assert [s.name for s in scalar._select_sections(secs, ["c", "a"])] == ["a", "c"]
    # unknown name is a hard error
    with pytest.raises(SystemExit):
        scalar._select_sections(secs, ["nope"])


def test_cli_select_one_section(tmp_path: Path):
    _write_ctd_combo(tmp_path)
    cfg = _three_section_yaml(tmp_path)
    rc = _run_cli(["scalar", "--root", str(tmp_path), "--sections", str(cfg),
                   "--out-dir", str(tmp_path), "--select", "dist"])
    assert rc == 0
    assert (tmp_path / "scalar_dist.png").exists()
    assert not (tmp_path / "scalar_by_lat.png").exists()
    assert not (tmp_path / "scalar_line.png").exists()


def test_cli_select_multiple_sections(tmp_path: Path):
    _write_ctd_combo(tmp_path)
    cfg = _three_section_yaml(tmp_path)
    rc = _run_cli(["scalar", "--root", str(tmp_path), "--sections", str(cfg),
                   "--out-dir", str(tmp_path), "--select", "line,by_lat"])
    assert rc == 0
    assert (tmp_path / "scalar_by_lat.png").exists()
    assert (tmp_path / "scalar_line.png").exists()
    assert not (tmp_path / "scalar_dist.png").exists()


def test_cli_select_unknown_name_errors(tmp_path: Path):
    _write_ctd_combo(tmp_path)
    cfg = _three_section_yaml(tmp_path)
    with pytest.raises(SystemExit):
        _run_cli(["scalar", "--root", str(tmp_path), "--sections", str(cfg),
                  "--out-dir", str(tmp_path), "--select", "missing"])


def test_cli_select_without_sections_errors(tmp_path: Path):
    _write_ctd_combo(tmp_path)
    with pytest.raises(SystemExit):
        _run_cli(["scalar", "--root", str(tmp_path), "--out-dir", str(tmp_path),
                  "--select", "anything"])


def test_parse_clim():
    assert scalar._parse_clim([["JAC_T", "18", "28"], ["SP", "34.5", "34.9"]]) == {
        "JAC_T": (18.0, 28.0), "SP": (34.5, 34.9),
    }
    assert scalar._parse_clim(None) == {}
    with pytest.raises(SystemExit):
        scalar._parse_clim([["JAC_T", "x", "28"]])


def test_override_xaxis_replaces_method_keeps_window():
    sec = scalar.Section(
        name="leg", method="along_line",
        start=np.datetime64("2025-01-20"), stop=np.datetime64("2025-01-21"),
        params={"units": "km", "waypoints": [[0.0, 0.0], [1.0, 1.0]]},
    )
    args = argparse.Namespace(xaxis="latitude", units="km", point=None, waypoints=None)
    new = scalar._override_xaxis(sec, args)
    assert new.method == "latitude" and new.name == "leg"
    assert new.start == sec.start and new.stop == sec.stop
    assert new.params == {"units": "km"}  # original waypoints dropped


def test_override_xaxis_spatial_method_needs_its_params():
    sec = scalar.Section(name="x", method="time")
    args = argparse.Namespace(
        xaxis="distance_from_point", units="km", point=None, waypoints=None
    )
    with pytest.raises(SystemExit):
        scalar._override_xaxis(sec, args)


def test_cli_vmin_with_multiple_vars_errors(tmp_path: Path):
    _write_ctd_combo(tmp_path)
    with pytest.raises(SystemExit):
        _run_cli(["scalar", "--root", str(tmp_path), "--out-dir", str(tmp_path),
                  "--var", "JAC_T", "--var", "SP", "--vmin", "0"])


def test_cli_clim_per_variable_runs(tmp_path: Path):
    _write_ctd_combo(tmp_path)
    rc = _run_cli(["scalar", "--root", str(tmp_path), "--out-dir", str(tmp_path),
                   "--name", "cl", "--var", "JAC_T", "--var", "SP",
                   "--clim", "JAC_T", "20", "30", "--clim", "SP", "34", "35"])
    assert rc == 0
    assert (tmp_path / "scalar_cl.png").exists()


def test_cli_xaxis_overrides_sections(tmp_path: Path):
    _write_ctd_combo(tmp_path)
    cfg = _three_section_yaml(tmp_path)  # by_lat / dist / line (spatial methods)
    # Override every section to a param-free axis; the dist/line sections that
    # need point/waypoints in the YAML must still render under the override.
    rc = _run_cli(["scalar", "--root", str(tmp_path), "--sections", str(cfg),
                   "--out-dir", str(tmp_path), "--xaxis", "time", "--var", "JAC_T"])
    assert rc == 0
    for name in ("by_lat", "dist", "line"):
        assert (tmp_path / f"scalar_{name}.png").exists()


def test_empty_time_window_skipped(tmp_path: Path):
    _write_ctd_combo(tmp_path)
    cfg = tmp_path / "sections.yaml"
    cfg.write_text(
        "sections:\n"
        "  - name: future\n"
        "    start: '2099-01-01T00:00:00Z'\n"
        "    xaxis: {method: time}\n"
    )
    rc = _run_cli(["scalar", "--root", str(tmp_path), "--sections", str(cfg),
                   "--out-dir", str(tmp_path)])
    assert rc == 0
    assert not (tmp_path / "scalar_future.png").exists()  # no data -> no figure


def test_default_variables(tmp_path: Path):
    import xarray as xr

    _write_ctd_combo(tmp_path)
    with xr.open_dataset(tmp_path / "ctd_combo_00" / "combo.nc") as ds:
        # dTdz is present but not a default panel.
        assert scalar._default_variables(ds) == ["JAC_T", "SP", "sigma0"]


def test_diverging_variable_renders(tmp_path: Path):
    """A one-signed diverging field (dTdz) exercises the symmetric-norm path."""
    _write_ctd_combo(tmp_path)
    rc = _run_cli(["scalar", "--root", str(tmp_path), "--out-dir", str(tmp_path),
                   "--name", "grad", "--xaxis", "time", "--var", "dTdz"])
    assert rc == 0
    png = tmp_path / "scalar_grad.png"
    assert png.exists() and png.stat().st_size > 0
