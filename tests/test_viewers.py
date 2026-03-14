"""Smoke tests for interactive viewers (quick_look, diss_look)."""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, MUST be before pyplot import

from pathlib import Path

import pytest

VMP_DIR = Path(__file__).parent.parent / "VMP"
P_FILES = sorted(VMP_DIR.glob("*.p"))


@pytest.mark.skipif(not P_FILES, reason="No VMP .p files available")
class TestQuickLook:
    def test_smoke(self):
        """quick_look() creates a viewer without crashing."""
        from microstructure_tpw.rsi.quick_look import QuickLookViewer

        from microstructure_tpw.rsi.p_file import PFile

        pf = PFile(P_FILES[0])
        viewer = QuickLookViewer(pf, fft_length=256, f_AA=98.0)
        assert viewer.profiles  # at least one profile detected
        assert viewer.shear  # shear channels exist


@pytest.mark.skipif(not P_FILES, reason="No VMP .p files available")
class TestDissLook:
    def test_smoke(self):
        """DissLookViewer creates a viewer without crashing."""
        from microstructure_tpw.rsi.diss_look import DissLookViewer

        from microstructure_tpw.rsi.p_file import PFile

        pf = PFile(P_FILES[0])
        viewer = DissLookViewer(pf, fft_length=256, f_AA=98.0)
        assert viewer.profiles
        assert viewer.shear
