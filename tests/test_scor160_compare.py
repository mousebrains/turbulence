# Tests for odas_tpw.scor160.compare
"""Unit tests for comparison and reporting utilities."""

import numpy as np
import pytest

from odas_tpw.scor160.compare import (
    compare_l2,
    compare_l3,
    compare_l4,
    format_l2_report,
    format_l3_report,
    format_l4_report,
)
from odas_tpw.scor160.io import L2Data, L3Data, L4Data


def _make_l2(n=10000, n_sh=2, n_vib=2, section_val=1.0, speed=0.5):
    """Create a minimal L2Data."""
    return L2Data(
        time=np.linspace(0, 1, n),
        shear=np.random.default_rng(42).standard_normal((n_sh, n)) * 0.1,
        vib=np.random.default_rng(43).standard_normal((n_vib, n)) * 0.01,
        vib_type="ACC",
        pspd_rel=np.full(n, speed),
        section_number=np.full(n, section_val),
    )


def _make_l3(n_spec=20, n_wn=129, n_sh=2, speed=0.5):
    """Create a minimal L3Data."""
    rng = np.random.default_rng(44)
    return L3Data(
        time=np.linspace(0, 1, n_spec),
        pres=np.linspace(10, 50, n_spec),
        temp=np.full(n_spec, 10.0),
        pspd_rel=np.full(n_spec, speed),
        section_number=np.ones(n_spec),
        kcyc=np.tile(np.linspace(0, 200, n_wn)[:, None], (1, n_spec)),
        sh_spec=np.abs(rng.standard_normal((n_sh, n_wn, n_spec))) * 1e-5 + 1e-10,
        sh_spec_clean=np.abs(rng.standard_normal((n_sh, n_wn, n_spec))) * 1e-5 + 1e-10,
    )


def _make_l4(n_spec=20, n_sh=2):
    """Create a minimal L4Data."""
    rng = np.random.default_rng(45)
    epsi = 10 ** (rng.uniform(-9, -5, (n_sh, n_spec)))
    return L4Data(
        time=np.linspace(0, 1, n_spec),
        pres=np.linspace(10, 50, n_spec),
        pspd_rel=np.full(n_spec, 0.5),
        section_number=np.ones(n_spec),
        epsi=epsi,
        epsi_final=np.exp(np.mean(np.log(epsi), axis=0)),
        epsi_flags=np.zeros((n_sh, n_spec)),
        fom=rng.uniform(0.8, 1.2, (n_sh, n_spec)),
        mad=rng.uniform(0.05, 0.3, (n_sh, n_spec)),
        kmax=np.full((n_sh, n_spec), 50.0),
        method=np.zeros((n_sh, n_spec)),
        var_resolved=rng.uniform(0.5, 1.0, (n_sh, n_spec)),
    )


class TestCompareL2:
    def test_identical_data(self):
        l2 = _make_l2()
        metrics = compare_l2(l2, l2)
        assert metrics["section"]["overlap_frac"] == pytest.approx(1.0)
        for sh in metrics["shear"]:
            assert sh["rms_diff"] == pytest.approx(0.0)
            assert sh["corr"] == pytest.approx(1.0)

    def test_different_speed(self):
        ref = _make_l2(speed=0.5)
        comp = _make_l2(speed=0.6)
        metrics = compare_l2(comp, ref)
        assert metrics["speed"]["rms_diff"] > 0

    def test_no_overlap(self):
        ref = _make_l2(section_val=1.0)
        comp = _make_l2(section_val=0.0)
        metrics = compare_l2(comp, ref)
        assert metrics["section"]["comp_n_selected"] == 0

    def test_keys_present(self):
        l2 = _make_l2()
        metrics = compare_l2(l2, l2)
        assert "section" in metrics
        assert "speed" in metrics
        assert "shear" in metrics
        assert "vibration" in metrics


class TestCompareL3:
    def test_identical_data(self):
        l3 = _make_l3()
        metrics = compare_l3(l3, l3)
        assert metrics["n_spectra"]["ref"] == metrics["n_spectra"]["comp"]
        for ss in metrics["spectra"]:
            assert ss["rms_log_diff"] == pytest.approx(0.0)

    def test_different_spectra(self):
        ref = _make_l3()
        comp = _make_l3()
        comp.sh_spec = comp.sh_spec * 1.5  # scale up
        metrics = compare_l3(comp, ref)
        for ss in metrics["spectra"]:
            if ss["type"] == "raw":
                assert ss["rms_log_diff"] > 0

    def test_empty_data(self):
        ref = _make_l3(n_spec=0)
        comp = _make_l3(n_spec=0)
        metrics = compare_l3(comp, ref)
        assert metrics["n_spectra"]["ref"] == 0

    def test_keys_present(self):
        l3 = _make_l3()
        metrics = compare_l3(l3, l3)
        assert "n_spectra" in metrics
        assert "spectra" in metrics


class TestCompareL4:
    def test_identical_data(self):
        l4 = _make_l4()
        metrics = compare_l4(l4, l4)
        for ps in metrics["probes"]:
            assert ps["rms_log_diff"] == pytest.approx(0.0)
        assert metrics["epsi_final"]["rms_log_diff"] == pytest.approx(0.0)

    def test_method_agreement(self):
        l4 = _make_l4()
        metrics = compare_l4(l4, l4)
        for ms in metrics["method"]:
            assert ms["agreement"] == pytest.approx(1.0)

    def test_different_epsilon(self):
        ref = _make_l4()
        comp = _make_l4()
        comp.epsi = comp.epsi * 10  # 10x difference
        comp.epsi_final = comp.epsi_final * 10
        metrics = compare_l4(comp, ref)
        for ps in metrics["probes"]:
            assert ps["rms_log_diff"] > 0

    def test_keys_present(self):
        l4 = _make_l4()
        metrics = compare_l4(l4, l4)
        assert "probes" in metrics
        assert "epsi_final" in metrics
        assert "fom" in metrics
        assert "method" in metrics


class TestFormatReports:
    def test_l2_report_string(self):
        l2 = _make_l2()
        metrics = compare_l2(l2, l2)
        report = format_l2_report(metrics, "test_file.nc")
        assert "test_file.nc" in report
        assert "Section selection" in report

    def test_l2_report_no_filename(self):
        l2 = _make_l2()
        metrics = compare_l2(l2, l2)
        report = format_l2_report(metrics)
        assert "L2 comparison" in report

    def test_l3_report_string(self):
        l3 = _make_l3()
        metrics = compare_l3(l3, l3)
        report = format_l3_report(metrics, "test.nc")
        assert "test.nc" in report
        assert "spectra" in report.lower()

    def test_l4_report_string(self):
        l4 = _make_l4()
        metrics = compare_l4(l4, l4)
        report = format_l4_report(metrics, "test.nc")
        assert "test.nc" in report
        assert "EPSI_FINAL" in report

    def test_l4_report_no_filename(self):
        l4 = _make_l4()
        metrics = compare_l4(l4, l4)
        report = format_l4_report(metrics)
        assert "L4 comparison" in report
