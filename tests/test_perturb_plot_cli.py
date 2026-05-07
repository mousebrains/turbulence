# May-2026, Claude and Pat Welch, pat@mousebrains.com
"""End-to-end tests for the ``perturb-plot`` CLI dispatcher.

Builds a tiny synthetic perturb-output tree (diss_combo + chi_combo +
chi per-profile NetCDFs) and runs the eps-chi subcommand, exercising
both the chi_combo path and the legacy per-profile fallback.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Force a non-interactive backend for headless test runs.
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from odas_tpw.perturb.plot import cli  # noqa: E402


def _write_diss_combo(root: Path, n_bin: int = 5, n_prof: int = 3) -> None:
    import xarray as xr

    bin_centers = np.arange(n_bin, dtype=float) + 1.0
    profile = np.arange(n_prof)
    eps = np.full((n_bin, n_prof), 1e-8, dtype=float)
    eps[:, 1] = 1e-7
    stime = np.array(["2026-03-25T00:00:00",
                      "2026-03-25T00:10:00",
                      "2026-03-25T00:20:00"], dtype="datetime64[ns]")[:n_prof]
    ds = xr.Dataset(
        {
            "epsilonMean": (("bin", "profile"), eps),
            "stime": (("profile",), stime),
        },
        coords={
            "bin": ("bin", bin_centers),
            "profile": ("profile", profile),
        },
    )
    out = root / "diss_combo_00"
    out.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out / "combo.nc")


def _write_chi_combo(root: Path, with_chi_mean: bool, n_bin: int = 5,
                     n_prof: int = 3) -> None:
    import xarray as xr

    bin_centers = np.arange(n_bin, dtype=float) + 1.0
    profile = np.arange(n_prof)
    stime = np.array(["2026-03-25T00:00:00",
                      "2026-03-25T00:10:00",
                      "2026-03-25T00:20:00"], dtype="datetime64[ns]")[:n_prof]
    data_vars = {
        "stime": (("profile",), stime),
    }
    if with_chi_mean:
        chi = np.full((n_bin, n_prof), 1e-9, dtype=float)
        chi[:, 1] = 1e-8
        data_vars["chiMean"] = (("bin", "profile"), chi)
    ds = xr.Dataset(
        data_vars,
        coords={
            "bin": ("bin", bin_centers),
            "profile": ("profile", profile),
        },
    )
    out = root / "chi_combo_00"
    out.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out / "combo.nc")


def _write_chi_per_profile(root: Path, n_bin: int = 5, n_prof: int = 3) -> None:
    """Per-profile chi NetCDFs used both for attrs and the legacy fallback."""
    import xarray as xr

    out = root / "chi_00"
    out.mkdir(parents=True, exist_ok=True)
    bin_centers = np.arange(n_bin, dtype=float) + 1.0
    stimes = np.array(["2026-03-25T00:00:00",
                       "2026-03-25T00:10:00",
                       "2026-03-25T00:20:00"], dtype="datetime64[ns]")
    for j in range(n_prof):
        chi = np.full((2, n_bin), 1e-9, dtype=float)
        ds = xr.Dataset(
            {
                "chi": (("probe", "time"), chi),
                "P_mean": (("time",), bin_centers),
                "stime": ((), stimes[j]),
            },
            coords={"probe": ("probe", ["t1", "t2"]),
                    "time": ("time", np.arange(n_bin))},
            attrs={
                "fft_length": 512,
                "diss_length": 2048,
                "fs_fast": 512.0,
                "spectrum_model": "kraichnan",
                "fp07_model": "single_pole",
            },
        )
        ds.to_netcdf(out / f"test_prof{j:03d}.nc")


def _write_diss_per_profile(root: Path, n_bin: int = 5, n_prof: int = 1) -> None:
    """Diss per-profile NetCDFs supply attrs for the title."""
    import xarray as xr

    out = root / "diss_00"
    out.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        attrs={
            "fft_length": 256,
            "diss_length": 1024,
            "fs_fast": 512.0,
        },
    )
    ds.to_netcdf(out / "prof_000.nc")


class TestEpsChiCLI:
    def test_chi_combo_path_writes_png(self, tmp_path):
        _write_diss_combo(tmp_path)
        _write_diss_per_profile(tmp_path)
        _write_chi_combo(tmp_path, with_chi_mean=True)
        _write_chi_per_profile(tmp_path)

        out_png = tmp_path / "fig.png"
        rc = cli.main(["eps-chi", "--root", str(tmp_path), "--out", str(out_png)])
        assert rc == 0
        assert out_png.exists() and out_png.stat().st_size > 0

    def test_legacy_fallback_when_chi_combo_lacks_mean(self, tmp_path):
        """chi_combo without chiMean falls back to per-profile binning."""
        _write_diss_combo(tmp_path)
        _write_diss_per_profile(tmp_path)
        _write_chi_combo(tmp_path, with_chi_mean=False)
        _write_chi_per_profile(tmp_path)

        out_png = tmp_path / "fig.png"
        rc = cli.main(["eps-chi", "--root", str(tmp_path), "--out", str(out_png)])
        assert rc == 0
        assert out_png.exists() and out_png.stat().st_size > 0

    def test_default_out_path_is_under_root(self, tmp_path):
        _write_diss_combo(tmp_path)
        _write_diss_per_profile(tmp_path)
        _write_chi_combo(tmp_path, with_chi_mean=True)
        _write_chi_per_profile(tmp_path)

        rc = cli.main(["eps-chi", "--root", str(tmp_path)])
        assert rc == 0
        assert (tmp_path / "eps_chi_pcolor.png").exists()

    def test_missing_diss_combo_errors(self, tmp_path):
        with pytest.raises(SystemExit):
            cli.main(["eps-chi", "--root", str(tmp_path)])

    def test_help_lists_eps_chi(self, capsys):
        with pytest.raises(SystemExit):
            cli.main(["--help"])
        out = capsys.readouterr().out
        assert "eps-chi" in out


class TestBuildParser:
    def test_subcommand_required(self):
        parser = cli.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])
