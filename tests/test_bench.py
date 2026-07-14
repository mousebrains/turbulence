# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for the bench-test diagnostic (rsi-tpw bench / odas_tpw.rsi.bench).

Split into two tiers:
  * pure-logic tests (checklist evaluation, report rendering, PSD scaling) that
    build synthetic stats and run in CI without any data file;
  * fixture tests that use the committed tests/data/SN479_0006.p (guarded with
    skipif) to exercise the reader flag, stats, figures and CLI end-to-end.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, MUST be before pyplot import

import numpy as np
import pytest

from odas_tpw.rsi.bench import (
    FAIL,
    NA,
    PASS,
    REVIEW,
    BenchStats,
    ChannelStat,
    CheckItem,
    _psd,
    build_ctclu_figure,
    build_spectra_figure,
    build_timeseries_figure,
    compute_bench_stats,
    evaluate_checklist,
    render_checklist_text,
    run_bench,
)

TEST_P = Path(__file__).parent / "data" / "SN479_0006.p"
# A real dummy-probe bench recording (VMP-250 SN142, ~90 s, trimmed from a Lake
# Cowichan 2025 bench test) — the core channels are expected to pass.
BENCH_P = Path(__file__).parent / "data" / "VMP142_bench.p"


# ---------------------------------------------------------------------------
# Synthetic stat builders (no data file needed)
# ---------------------------------------------------------------------------


def _cs(
    name,
    ctype,
    *,
    is_fast=True,
    unit="counts",
    mean=0.0,
    std=1.0,
    max_abs_dev=0.0,
    typ_dev=None,
    psd_max=float("nan"),
    psd_at_probe=float("nan"),
    peak_freq=float("nan"),
    with_psd=False,
):
    # Range checks use typ_dev (99.9th-percentile deviation); default it to
    # max_abs_dev so a test can drive the range verdict with either field.
    cs = ChannelStat(
        name=name,
        ctype=ctype,
        is_fast=is_fast,
        unit=unit,
        n=1000,
        mean=mean,
        std=std,
        typ_dev=max_abs_dev if typ_dev is None else typ_dev,
        max_abs_dev=max_abs_dev,
        psd_max=psd_max,
        psd_at_probe=psd_at_probe,
        peak_freq=peak_freq,
    )
    if with_psd:
        cs.freq = np.linspace(0.5, 256.0, 8)
        cs.psd = np.full(8, 1.0)
    return cs


def _stats(*channels):
    bs = BenchStats(
        filename="fake.p",
        sn="42",
        fs_fast=512.0,
        fs_slow=64.0,
        fft_sec=2.0,
        start_time="2025-01-01T00:00:00+00:00",
    )
    for cs in channels:
        bs.channels[cs.name] = cs
    return bs


def _find(items, label):
    return next(it for it in items if it.label == label)


# ---------------------------------------------------------------------------
# Checklist evaluation
# ---------------------------------------------------------------------------


class TestEvaluateChecklist:
    def test_time_series_pass_fail(self):
        stats = _stats(
            _cs("Ax", "piezo", max_abs_dev=200.0, std=100.0),
            _cs("Ay", "piezo", max_abs_dev=600.0, std=120.0),  # over +/-500
            _cs("sh1", "shear", mean=3.0, max_abs_dev=25.0),
            _cs("sh2", "shear", mean=50.0, max_abs_dev=25.0),  # mean over 10
            _cs("T1_dT1", "therm", mean=60.0, max_abs_dev=30.0),
            _cs("T2_dT2", "therm", mean=150.0, max_abs_dev=30.0),  # offset over 100
            _cs("P", "poly", is_fast=False, max_abs_dev=1.0),
            _cs("P_dP", "poly", is_fast=False, max_abs_dev=20.0),  # over +/-10
        )
        items = evaluate_checklist(stats)
        assert _find(items, "Ax range").status == PASS
        assert _find(items, "Ay range").status == FAIL
        assert _find(items, "sh1 mean").status == PASS
        assert _find(items, "sh1 range").status == PASS
        assert _find(items, "sh2 mean").status == FAIL
        assert _find(items, "T1_dT1 range").status == PASS
        assert _find(items, "T1_dT1 offset").status == PASS
        assert _find(items, "T2_dT2 offset").status == FAIL
        assert _find(items, "P range").status == PASS
        assert _find(items, "P_dP range").status == FAIL

    def test_range_uses_robust_typ_dev_not_max(self):
        # A lone spike (huge max) with a small typical envelope must PASS:
        # "typically within +/-500" is the 99.9th percentile, not the max.
        stats = _stats(_cs("Ax", "piezo", typ_dev=300.0, max_abs_dev=5000.0))
        items = evaluate_checklist(stats)
        assert _find(items, "Ax range").status == PASS

    def test_range_fail_just_above_each_limit(self):
        # Guards the constants: each range check FAILs when typ_dev exceeds it.
        cases = {
            "Ax range": _cs("Ax", "piezo", typ_dev=501.0),
            "T1_dT1 range": _cs("T1_dT1", "therm", typ_dev=41.0),
            "sh1 range": _cs("sh1", "shear", typ_dev=31.0),
            "P range": _cs("P", "poly", is_fast=False, typ_dev=2.1),
            "P_dP range": _cs("P_dP", "poly", is_fast=False, typ_dev=10.1),
            "JAC_T range": _cs("JAC_T", "jac_t", is_fast=False, typ_dev=51.0),
            "Turbidity range": _cs("Turbidity", "poly", typ_dev=51.0),
            "Chlorophyll range": _cs("Chlorophyll", "poly", typ_dev=401.0),
        }
        for label, cs in cases.items():
            items = evaluate_checklist(_stats(cs))
            assert _find(items, label).status == FAIL, label

    def test_range_surfaces_max_when_over_limit(self):
        # typ_dev within limit (PASS) but the true max beyond it -> the max is
        # surfaced so the robust metric never silently masks an excursion.
        cs = _cs("Ax", "piezo", typ_dev=300.0, max_abs_dev=5000.0)
        item = _find(evaluate_checklist(_stats(cs)), "Ax range")
        assert item.status == PASS
        assert "max" in item.measured

    def test_pdp_spectra_middle_band_density_pass_peak_fail(self):
        # psd_max in [3, 10): density-max check PASSes, peak check FAILs — the
        # band that distinguishes the two P_dP spectral checks.
        cs = _cs("P_dP", "poly", is_fast=False, psd_max=5.0, peak_freq=1.0, with_psd=True)
        items = evaluate_checklist(_stats(cs))
        assert _find(items, "P_dP density max").status == PASS
        assert _find(items, "P_dP peak").status == FAIL

    def test_missing_ucond_is_na(self):
        items = evaluate_checklist(_stats(_cs("Ax", "piezo", max_abs_dev=10.0)))
        assert _find(items, "C1_dC1 range").status == NA
        assert _find(items, "C1_dC1 offset").status == NA

    def test_present_ucond_evaluated(self):
        stats = _stats(_cs("C1_dC1", "ucond", mean=100.0, max_abs_dev=40.0))
        items = evaluate_checklist(stats)
        assert _find(items, "C1_dC1 range").status == PASS  # 40 <= 50
        assert _find(items, "C1_dC1 offset").status == PASS  # 100 < 6000

    def test_spectra_pass_fail(self):
        stats = _stats(
            _cs("P_dP", "poly", is_fast=False, psd_max=1.0, peak_freq=1.5, with_psd=True),
            _cs("Ax", "piezo", psd_max=200.0, with_psd=True),  # over 100
            _cs("Ay", "piezo", psd_max=5.0, with_psd=True),
            _cs("T1_dT1", "therm", psd_at_probe=0.1, with_psd=True),
            _cs("sh1", "shear", psd_at_probe=0.01, with_psd=True),
        )
        items = evaluate_checklist(stats)
        assert _find(items, "P_dP density max").status == PASS
        assert _find(items, "P_dP peak").status == PASS
        assert _find(items, "Ax peak").status == FAIL
        assert _find(items, "Ay peak").status == PASS
        # subjective spectral shape checks -> REVIEW
        assert _find(items, "T1_dT1 rising ~1e-1 near 100 Hz").status == REVIEW
        assert _find(items, "sh1 rising ~1e-2 near 100 Hz").status == REVIEW

    def test_pdp_density_over_limit_fails(self):
        stats = _stats(
            _cs("P_dP", "poly", is_fast=False, psd_max=50.0, peak_freq=0.5, with_psd=True)
        )
        items = evaluate_checklist(stats)
        assert _find(items, "P_dP density max").status == FAIL
        assert _find(items, "P_dP peak").status == FAIL

    def test_subjective_items_are_review(self):
        stats = _stats(
            _cs("Ax", "piezo", max_abs_dev=10.0, std=5.0),
            _cs("Ay", "piezo", max_abs_dev=10.0, std=5.0),
            _cs("Incl_T", "inclt", is_fast=False, unit="deg_C", mean=22.0),
        )
        items = evaluate_checklist(stats)
        assert _find(items, "Ax~Ay similar, Ax>Ay").status == REVIEW
        assert _find(items, "Incl_T reasonable & constant").status == REVIEW

    def test_ctclu_evaluation(self):
        stats = _stats(
            _cs("JAC_T", "jac_t", is_fast=False, max_abs_dev=30.0),
            _cs("Turbidity", "poly", max_abs_dev=100.0),  # over 50
            _cs("Chlorophyll", "poly", max_abs_dev=200.0),
            _cs("JAC_C", "jac_c", is_fast=False),
        )
        items = evaluate_checklist(stats)
        assert _find(items, "JAC_T range").status == PASS
        assert _find(items, "Turbidity range").status == FAIL
        assert _find(items, "Chlorophyll range").status == PASS
        assert _find(items, "JAC_C I/V").status == REVIEW


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def test_render_checklist_text():
    items = [
        CheckItem("Time Series", "Ax range", "+/-10 counts", "<= +/-500", PASS),
        CheckItem("Time Series", "sh2 mean", "50 counts", "< 10", FAIL),
        CheckItem("Spectra", "T1_dT1 rising", "0.1 at 100 Hz", "rising", REVIEW),
    ]
    text = render_checklist_text(items, "Bench test  fake.p\nSN 42")
    assert "-- Time Series --" in text
    assert "-- Spectra --" in text
    assert "1 pass, 1 fail, 1 review, 0 n/a" in text
    assert "Ax range" in text and "sh2 mean" in text
    # each status symbol appears
    assert "[ ok ]" in text and "[FAIL]" in text and "[rvw ]" in text


def test_render_checklist_empty_header_no_crash():
    # An empty header must not raise (max() over no lines); public API guard.
    assert isinstance(render_checklist_text([], ""), str)


def test_checklist_text_figure_renders_report():
    import matplotlib.pyplot as plt

    from odas_tpw.rsi.bench import _checklist_text_figure

    fig = _checklist_text_figure("Bench test X\nSummary: 1 pass\n  [ ok ] P range")
    assert any("Summary" in t.get_text() for t in fig.texts)
    plt.close(fig)


# ---------------------------------------------------------------------------
# PSD scaling
# ---------------------------------------------------------------------------


def test_psd_parseval():
    """One-sided PSD integrates to the signal variance (counts^2/Hz scaling)."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(40000) * 3.0
    out = _psd(x, 1024, 512.0)
    assert out is not None
    freq, psd = out
    df = freq[1] - freq[0]
    assert np.sum(psd) * df == pytest.approx(np.var(x), rel=0.05)
    assert freq[-1] == pytest.approx(256.0)  # Nyquist for fs=512


def test_psd_short_returns_none():
    assert _psd(np.zeros(100), 1024, 512.0) is None


# ---------------------------------------------------------------------------
# Fixture-backed tests (committed SN479_0006.p)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TEST_P.exists(), reason="SN479_0006.p test data not available")
class TestWithFixture:
    def test_deconvolve_false_keeps_raw_counts(self):
        from odas_tpw.rsi.p_file import PFile

        raw = PFile(TEST_P, deconvolve=False)
        deconv = PFile(TEST_P)  # default deconvolve=True
        for name in ("T1_dT1", "P_dP", "P", "T1"):
            # deconvolve=False leaves int counts; default is deconvolved float64
            assert raw.channels_raw[name].dtype.kind == "i"
            assert deconv.channels_raw[name].dtype.kind == "f"
        # untouched channels are identical either way
        assert np.array_equal(raw.channels_raw["Ax"], deconv.channels_raw["Ax"])

    def test_compute_bench_stats(self):
        from odas_tpw.rsi.p_file import PFile

        pf = PFile(TEST_P, deconvolve=False)
        stats = compute_bench_stats(pf, fft_sec=2.0)
        assert stats.sn == "479"
        for name in ("Ax", "Ay", "sh1", "sh2", "T1_dT1", "P_dP"):
            cs = stats.get(name)
            assert cs is not None
            assert np.isfinite(cs.mean) and np.isfinite(cs.max_abs_dev)
        # spectrum channels carry a PSD; P (base) does not
        assert stats.get("Ax").psd is not None
        assert np.isfinite(stats.get("P_dP").peak_freq)
        assert stats.get("P").psd is None

    def test_build_figures_headless(self):
        import matplotlib.pyplot as plt

        from odas_tpw.rsi.p_file import PFile

        pf = PFile(TEST_P, deconvolve=False)
        stats = compute_bench_stats(pf, fft_sec=2.0)
        for fig in (
            build_timeseries_figure(pf, stats, "t"),
            build_spectra_figure(pf, stats, "s"),
            build_ctclu_figure(pf, stats, "c"),
        ):
            assert fig is not None
            assert len(fig.axes) >= 1
            plt.close(fig)

    def test_figures_draw_valid_range_bands(self):
        import matplotlib.pyplot as plt

        from odas_tpw.rsi.p_file import PFile

        pf = PFile(TEST_P, deconvolve=False)
        stats = compute_bench_stats(pf)
        ts = build_timeseries_figure(pf, stats, "t")
        # at least one time-series panel drew a shaded valid-range band (axhspan)
        assert any(len(ax.patches) > 0 for ax in ts.axes)
        plt.close(ts)
        sp = build_spectra_figure(pf, stats, "s")
        # spectra drew target boxes / ceilings (Rectangle patches + axhlines)
        assert len(sp.axes[0].patches) > 0
        plt.close(sp)

    def test_no_figure_leak_on_save_error(self, monkeypatch):
        import matplotlib.pyplot as plt

        before = len(plt.get_fignums())

        def boom(self, *a, **k):
            raise OSError("disk full")

        monkeypatch.setattr(plt.Figure, "savefig", boom)
        with pytest.raises(OSError):
            run_bench(TEST_P, out_dir="/tmp/rsi_bench_should_not_write", fmt="png")
        assert len(plt.get_fignums()) == before  # figures closed despite the error

    def test_spectra_no_spectra_branch(self):
        import matplotlib.pyplot as plt

        from odas_tpw.rsi.bench import BenchStats, build_spectra_figure
        from odas_tpw.rsi.p_file import PFile

        pf = PFile(TEST_P, deconvolve=False)
        empty = BenchStats(
            filename="x", sn="1", fs_fast=512.0, fs_slow=64.0, fft_sec=2.0, start_time="t"
        )
        fig = build_spectra_figure(pf, empty, "s")
        assert any("No spectra" in t.get_text() for t in fig.axes[0].texts)
        plt.close(fig)

    def test_run_bench_writes_files(self, tmp_path):
        stats, items = run_bench(TEST_P, out_dir=tmp_path, fft_sec=2.0)
        assert stats.sn == "479"
        assert items
        base = f"QB_479_{TEST_P.stem}"
        for suffix in ("timeseries.png", "spectra.png", "ctclu.png", "checklist.txt"):
            assert (tmp_path / f"{base}_{suffix}").exists()

    def test_run_bench_format_both(self, tmp_path):
        run_bench(TEST_P, out_dir=tmp_path, fft_sec=2.0, fmt="both")
        base = f"QB_479_{TEST_P.stem}"
        assert (tmp_path / f"{base}_timeseries.pdf").exists()
        assert (tmp_path / f"{base}_timeseries.png").exists()

    def test_run_bench_pdf_bundle(self, tmp_path):
        run_bench(TEST_P, out_dir=tmp_path, fft_sec=2.0, fmt="pdf-bundle")
        base = f"QB_479_{TEST_P.stem}"
        bundle = tmp_path / f"{base}.pdf"
        assert bundle.exists() and bundle.stat().st_size > 1000
        # a single bundle, not per-figure files
        assert not (tmp_path / f"{base}_timeseries.png").exists()
        assert not (tmp_path / f"{base}_spectra.pdf").exists()
        assert not (tmp_path / f"{base}.pdf.tmp").exists()  # temp cleaned up
        assert (tmp_path / f"{base}_checklist.txt").exists()
        # bundle carries a page per figure PLUS the checklist text page
        import re

        n_pages = len(re.findall(rb"/Type\s*/Page\b(?!s)", bundle.read_bytes()))
        assert n_pages == 4  # timeseries, spectra, ctclu, checklist

    def test_run_bench_no_out_dir_writes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _, items = run_bench(TEST_P, out_dir=None)
        assert items
        assert not list(tmp_path.iterdir())  # nothing written

    def test_run_bench_sn_override(self, tmp_path):
        run_bench(TEST_P, out_dir=tmp_path, sn="XYZ")
        assert (tmp_path / f"QB_XYZ_{TEST_P.stem}_timeseries.png").exists()

    def test_cli_bench_end_to_end(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["rsi-tpw", "bench", str(TEST_P), "-o", str(tmp_path), "--fft-sec", "2.0"],
        )
        from odas_tpw.rsi.cli import main

        main()
        assert (tmp_path / f"QB_479_{TEST_P.stem}_checklist.txt").exists()

    def test_cli_bench_default_output_dir(self, tmp_path, monkeypatch):
        # No -o: writes into ./bench/ relative to the working directory.
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["rsi-tpw", "bench", str(TEST_P)])
        from odas_tpw.rsi.cli import main

        main()
        assert (tmp_path / "bench" / f"QB_479_{TEST_P.stem}_checklist.txt").exists()

    def test_cli_bench_bad_file_is_caught(self, tmp_path, monkeypatch, capsys):
        # A malformed .p is caught per-file (stderr) without aborting the batch.
        bad = tmp_path / "bad.p"
        bad.write_bytes(b"not a real p file")
        monkeypatch.setattr(
            sys, "argv", ["rsi-tpw", "bench", str(bad), "-o", str(tmp_path / "out")]
        )
        from odas_tpw.rsi.cli import main

        main()  # must not raise
        assert "ERROR" in capsys.readouterr().err


@pytest.mark.skipif(not BENCH_P.exists(), reason="VMP142_bench.p test data not available")
class TestRealBenchFile:
    """A genuine dummy-probe bench recording — the core FP07 / shear / pressure
    checks must PASS. This locks in the checklist thresholds and the robust
    (99.9th-percentile) "typically within" range metric against regressions.
    """

    CORE_PASS = (
        "T1_dT1 range",
        "T1_dT1 offset",
        "T2_dT2 range",
        "T2_dT2 offset",
        "sh1 mean",
        "sh1 range",
        "sh2 mean",
        "sh2 range",
        "P range",
        "P_dP range",
        "P_dP density max",
        "P_dP peak",
    )

    def test_core_checks_pass(self):
        _, items = run_bench(BENCH_P, out_dir=None)
        status = {it.label: it.status for it in items}
        for label in self.CORE_PASS:
            assert status.get(label) == PASS, f"{label} -> {status.get(label)}"

    def test_robust_metric_beats_max_on_accelerometer(self):
        # The bench bumps make max|Ax-mean| ~10x the typical (p99.9) envelope;
        # the range check must use the robust one.
        from odas_tpw.rsi.p_file import PFile

        pf = PFile(BENCH_P, deconvolve=False)
        ax = compute_bench_stats(pf).get("Ax")
        assert ax.max_abs_dev > 2 * ax.typ_dev

    def test_spectra_bands_near_checklist(self):
        # Thermistor gradient rises to ~1e-1 and shear to ~1e-2 near 100 Hz;
        # P_dP rolls off by ~2 Hz. (Rockland checklist reference magnitudes.)
        from odas_tpw.rsi.p_file import PFile

        stats = compute_bench_stats(PFile(BENCH_P, deconvolve=False))
        assert 3e-2 < stats.get("T1_dT1").psd_at_probe < 3e-1
        assert 2e-3 < stats.get("sh1").psd_at_probe < 3e-2
        assert stats.get("P_dP").peak_freq < 3.0

    def test_ctclu_figure_present(self):
        import matplotlib.pyplot as plt

        from odas_tpw.rsi.p_file import PFile

        pf = PFile(BENCH_P, deconvolve=False)
        fig = build_ctclu_figure(pf, compute_bench_stats(pf), "c")
        assert fig is not None  # JAC channels present
        plt.close(fig)

    def test_deconvolve_false_no_suspect_warnings(self):
        # The pre-emphasized channels keep raw counts (units "counts") rather than
        # emitting misleading "physical units are suspect" warnings from their
        # sparse config.
        import warnings

        from odas_tpw.rsi.p_file import PFile

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pf = PFile(BENCH_P, deconvolve=False)
        assert not any("suspect" in str(w.message) for w in caught)
        assert pf.channel_info["T1_dT1"]["units"] == "counts"
        assert pf.channel_info["P_dP"]["units"] == "counts"
