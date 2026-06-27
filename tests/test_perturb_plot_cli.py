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
                     n_prof: int = 3, bin_centers: np.ndarray | None = None) -> None:
    import xarray as xr

    if bin_centers is None:
        bin_centers = np.arange(n_bin, dtype=float) + 1.0
    else:
        n_bin = len(bin_centers)
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


def _write_nan_combo(root: Path, prefix: str, var: str) -> None:
    """An all-NaN (bin, profile) combo to exercise the no-finite-data path."""
    import xarray as xr

    bin_c = np.arange(5, dtype=float) + 1.0
    prof = np.arange(3)
    stime = np.array(["2026-03-25T00:00:00", "2026-03-25T00:10:00",
                      "2026-03-25T00:20:00"], dtype="datetime64[ns]")
    ds = xr.Dataset(
        {var: (("bin", "profile"), np.full((5, 3), np.nan)),
         "stime": (("profile",), stime)},
        coords={"bin": ("bin", bin_c), "profile": ("profile", prof)},
    )
    out = root / f"{prefix}_00"
    out.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out / "combo.nc")


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

    def test_chi_mean_without_chi_dir_does_not_crash(self, tmp_path):
        """chiMean on the combo but NO chi_NN per-profile dir -> load from the
        combo (attrs={} for title), not raise 'No chi_NN dir' (M-22)."""
        _write_diss_combo(tmp_path)
        _write_diss_per_profile(tmp_path)
        _write_chi_combo(tmp_path, with_chi_mean=True)  # no _write_chi_per_profile
        out_png = tmp_path / "fig.png"
        rc = cli.main(["eps-chi", "--root", str(tmp_path), "--out", str(out_png)])
        assert rc == 0
        assert out_png.exists() and out_png.stat().st_size > 0

    def test_all_nan_eps_chi_does_not_crash(self, tmp_path):
        """All-NaN eps/chi -> quantile limits (None, None); panels must render a
        placeholder, not crash on LogNorm(None, None) (M-14)."""
        _write_nan_combo(tmp_path, "diss_combo", "epsilonMean")
        _write_nan_combo(tmp_path, "chi_combo", "chiMean")
        _write_diss_per_profile(tmp_path)
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

    def test_chi_combo_different_bin_grid_reindexes(self, tmp_path):
        """chi binned on a different depth grid than eps must be re-binned onto
        the eps grid, not broadcast element-wise (was a ValueError crash when
        the bin counts differed, silent depth-shift when they matched)."""
        _write_diss_combo(tmp_path, n_bin=5)  # eps centers 1..5
        _write_diss_per_profile(tmp_path)
        # chi on a finer grid with offset centers AND a different bin count.
        _write_chi_combo(tmp_path, with_chi_mean=True,
                         bin_centers=np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5]))
        _write_chi_per_profile(tmp_path)

        out_png = tmp_path / "fig.png"
        rc = cli.main(["eps-chi", "--root", str(tmp_path), "--out", str(out_png)])
        assert rc == 0
        assert out_png.exists() and out_png.stat().st_size > 0

    def test_reindex_rows_to_depth_aligns_grids(self):
        from odas_tpw.perturb.plot import eps_chi

        # src centers offset by +1 from dst; reindex should shift rows down one
        # eps bin (value at src 1.5 lands in dst bin centered at 2.0).
        src_depth = np.array([1.5, 2.5, 3.5])
        dst_depth = np.array([1.0, 2.0, 3.0])
        arr = np.array([[10.0], [20.0], [30.0]])
        out = eps_chi._reindex_rows_to_depth(arr, src_depth, dst_depth)
        # dst edges are [0.5,1.5,2.5,3.5] -> src 1.5->bin1, 2.5->bin2, 3.5->out
        assert np.isnan(out[0, 0])
        assert out[1, 0] == 10.0
        assert out[2, 0] == 20.0
        # identical grids are a no-op (common matched case)
        same = eps_chi._reindex_rows_to_depth(arr, dst_depth, dst_depth)
        assert same is arr

    def test_reindex_rows_to_depth_max_ors_drop_bitfield(self):
        """reduce='max' must bitwise-OR drop bitfields, not take the max: two
        source bins with flags 1 and 2 collapsing into one dst bin -> 3, not 2
        (np.maximum would lose the bit) (#52)."""
        from odas_tpw.perturb.plot import eps_chi

        # Both src centers fall in dst bin 0 (edges [0.5, 1.5, 2.5]).
        src_depth = np.array([0.9, 1.1])
        dst_depth = np.array([1.0, 2.0])
        arr = np.array([[1.0], [2.0]])  # flag bitfields: bit0 and bit1
        out = eps_chi._reindex_rows_to_depth(arr, src_depth, dst_depth, reduce="max")
        assert out[0, 0] == 3.0  # 1 | 2, not max(1, 2) = 2
        assert np.isnan(out[1, 0])

    def test_align_chi_to_eps_matched_but_all_nan_not_counted_unmatched(self):
        """A chi column that matched in time (within 5 s) but is entirely NaN
        over depth (e.g. QC dropped every bin) must NOT be counted as an
        unmatched eps slot. Inferring n_unmatched from finiteness over-reports
        the 'no chi' count; an explicit match boolean reports it correctly."""
        from odas_tpw.perturb.plot import eps_chi

        t0 = np.datetime64("2025-01-01T00:00:00")
        # 3 eps slots; chi has a column at each slot's time but the middle chi
        # column is all-NaN over depth (matched in time, no valid chi values).
        t_eps = t0 + np.array([0, 60, 120], "timedelta64[s]")
        t_chi = t0 + np.array([0, 60, 120], "timedelta64[s]")
        chi = np.array(
            [[1.0, np.nan, 3.0], [2.0, np.nan, 4.0]]  # 2 depth bins, 3 profiles
        )
        aligned, aligned_qc, n_unmatched = eps_chi._align_chi_to_eps(
            t_eps, t_chi, chi, None
        )
        # All three chi columns matched in time -> zero genuinely unmatched.
        # The old finiteness-based count would report 1 (the all-NaN column).
        assert n_unmatched == 0
        assert aligned_qc is None
        # The matched-but-all-NaN column stays NaN; the others are placed.
        assert aligned[0, 0] == 1.0 and aligned[1, 2] == 4.0
        assert np.isnan(aligned[0, 1]) and np.isnan(aligned[1, 1])

    def test_align_chi_to_eps_counts_time_unmatched_slots(self):
        """An eps slot with no chi column within 5 s is correctly unmatched."""
        from odas_tpw.perturb.plot import eps_chi

        t0 = np.datetime64("2025-01-01T00:00:00")
        t_eps = t0 + np.array([0, 60, 120], "timedelta64[s]")
        # Only two chi columns, near eps slots 0 and 2; slot 1 has no chi.
        t_chi = t0 + np.array([1, 121], "timedelta64[s]")
        chi = np.array([[1.0, 3.0], [2.0, 4.0]])
        chi_qc = np.array([[0.0, 1.0], [0.0, 0.0]])
        aligned, aligned_qc, n_unmatched = eps_chi._align_chi_to_eps(
            t_eps, t_chi, chi, chi_qc
        )
        assert n_unmatched == 1  # the middle eps slot
        assert aligned_qc is not None
        assert aligned[0, 0] == 1.0 and aligned[0, 2] == 3.0
        assert np.isnan(aligned[0, 1])
        assert aligned_qc[0, 2] == 1.0

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
