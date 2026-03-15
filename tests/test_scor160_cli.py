"""Tests for scor160.cli — ATOMIX benchmark processing CLI."""

import pytest

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
