"""Tests for scor160.cli — ATOMIX benchmark processing CLI."""

from unittest.mock import patch

import matplotlib
import pytest

matplotlib.use("Agg")

from odas_tpw.scor160.cli import main


class TestScor160CLI:
    """Exercise CLI dispatch paths using the synthetic ATOMIX fixture."""

    def test_main_l1_l4(self, atomix_nc_file):
        """Full pipeline L1→L4 runs without error."""
        main(["l1-l4", str(atomix_nc_file)])

    def test_main_l1_l2(self, atomix_nc_file):
        """L1→L2 covers section selection, despike, HP filter."""
        main(["l1-l2", str(atomix_nc_file)])

    def test_main_l2_l3(self, atomix_nc_file):
        """L2→L3 covers spectral computation from reference L2."""
        main(["l2-l3", str(atomix_nc_file)])

    def test_main_l3_l4(self, atomix_nc_file):
        """L3→L4 covers dissipation from reference spectra."""
        main(["l3-l4", str(atomix_nc_file)])

    def test_main_l1_l3(self, atomix_nc_file):
        """L1→L3 covers L2+L3 pipeline."""
        main(["l1-l3", str(atomix_nc_file)])

    def test_main_l2_l4(self, atomix_nc_file):
        """L2→L4 covers L3+L4 pipeline."""
        main(["l2-l4", str(atomix_nc_file)])

    def test_main_no_command_errors(self):
        """Missing required subcommand should raise SystemExit."""
        with pytest.raises(SystemExit):
            main([])


class TestPlotL2:
    """Exercise the --plot path through _plot_l2."""

    def test_plot_acc_path(self, atomix_nc_file):
        """--plot on l1-l2 with default (ACC) vib_type runs through _plot_l2."""
        with patch("matplotlib.pyplot.show"):
            main(["l1-l2", str(atomix_nc_file), "--plot"])

    def test_plot_vib_path(self, tmp_path_factory):
        """vib_type='VIB' covers the alternate Vib label branch in _plot_l2."""
        from tests.conftest import _create_atomix_nc

        path = tmp_path_factory.mktemp("atomix_vib") / "benchmark.nc"
        _create_atomix_nc(path, vib_var_name="VIB")
        with patch("matplotlib.pyplot.show"):
            main(["l1-l2", str(path), "--plot"])

    def test_plot_no_sections(self, tmp_path, capsys):
        """When section_number is all zero, _plot_l2 reports and returns early."""
        import netCDF4

        from tests.conftest import _create_atomix_nc

        path = tmp_path / "no_sections.nc"
        _create_atomix_nc(path)
        with netCDF4.Dataset(str(path), "r+") as ds:
            ds["L2_cleaned/SECTION_NUMBER"][:] = 0.0

        with patch("matplotlib.pyplot.show"):
            main(["l1-l2", str(path), "--plot"])

        captured = capsys.readouterr()
        assert "No sections to plot" in captured.out

    def test_plot_single_shear_collapses_axes(self, tmp_path):
        """n_rows == 1 path: 0 shear and 0 vib leaves only the speed panel."""
        from types import SimpleNamespace

        import numpy as np

        from odas_tpw.scor160.cli import _plot_l2

        n = 64
        l1 = SimpleNamespace(
            fs_fast=64.0, n_time=n, n_shear=0, n_vib=0, vib_type="ACC",
        )
        l2_ref = SimpleNamespace(
            section_number=np.ones(n),
            pspd_rel=np.full(n, 0.7),
            shear=np.zeros((0, n)),
            vib=np.zeros((0, n)),
            vib_type="ACC",
        )
        l2_comp = SimpleNamespace(
            section_number=np.ones(n),
            pspd_rel=np.full(n, 0.7),
            shear=np.zeros((0, n)),
            vib=np.zeros((0, n)),
            vib_type="ACC",
        )
        with patch("matplotlib.pyplot.show"):
            _plot_l2(l1, l2_comp, l2_ref, "test")
