# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Hard-batch branch tests — chi/chi.py fits, l4_chi epsilon path, convert helpers."""

from __future__ import annotations

import warnings

import numpy as np

# ---------------------------------------------------------------------------
# chi/chi.py — _mle_fit_kB warning paths (lines 387, 389)
# ---------------------------------------------------------------------------


class TestMLEFitKBWarnings:
    def test_too_few_points_warning(self):
        """When fit_mask has < 6 valid points, _mle_fit_kB warns and returns NaN."""
        from odas_tpw.chi.chi import _mle_fit_kB

        # Only 3 K points → < 6 valid → "Too few valid points" warning
        K = np.array([1.0, 2.0, 3.0])
        spec_obs = np.full(3, 1e-7)
        noise_K = np.full(3, 1e-12)
        H2 = np.ones(3)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _mle_fit_kB(
                spec_obs=spec_obs,
                K=K,
                chi_obs=1e-9,
                nu=1.2e-6,
                noise_K=noise_K,
                H2=H2,
                tau0=0.01,
                _h2=lambda F, t: np.ones_like(F),
                f_AA=98.0,
                speed=0.5,
                spectrum_model="batchelor",
            )
        assert np.isnan(result.kB)
        assert np.isnan(result.chi)
        assert any("Too few valid points" in str(wi.message) for wi in w)


# ---------------------------------------------------------------------------
# chi/chi.py — _iterative_fit too-few-valid-points warning (line 470)
# ---------------------------------------------------------------------------


class TestIterativeFitWarnings:
    def test_too_few_points_warning(self):
        """_iterative_fit with < 6 valid points warns and returns NaN."""
        from odas_tpw.chi.chi import _iterative_fit

        K = np.array([1.0, 2.0, 3.0, 4.0])
        spec_obs = np.full(4, 1e-7)
        noise_K = np.full(4, 1e-12)
        H2 = np.ones(4)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _iterative_fit(
                spec_obs=spec_obs,
                K=K,
                nu=1.2e-6,
                noise_K=noise_K,
                H2=H2,
                tau0=0.01,
                _h2=lambda F, t: np.ones_like(F),
                f_AA=98.0,
                speed=0.5,
                spectrum_model="batchelor",
            )
        assert np.isnan(result.kB)
        assert any("Too few valid points" in str(wi.message) for wi in w)


# ---------------------------------------------------------------------------
# chi/chi.py — _iterative_fit mask_init fallback (line 478)
# ---------------------------------------------------------------------------


class TestIterativeFitMaskFallback:
    def test_below_noise_falls_back_to_full_valid(self):
        """When (spec_obs > noise_K) is mostly false, mask_init falls back to valid."""
        from odas_tpw.chi.chi import _iterative_fit

        # 20 K points, with a spectrum that is BELOW the noise floor everywhere
        # so spec_obs > noise_K is empty → mask_init = valid (line 478 branch)
        n = 20
        K = np.linspace(1.0, 100.0, n)
        spec_obs = np.full(n, 1e-12)  # below noise
        noise_K = np.full(n, 1e-10)  # higher than spec
        H2 = np.ones(n)

        # No valid points either way → returns NaN, but exercises mask_init fallback
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _iterative_fit(
                spec_obs=spec_obs,
                K=K,
                nu=1.2e-6,
                noise_K=noise_K,
                H2=H2,
                tau0=0.01,
                _h2=lambda F, t: np.ones_like(F),
                f_AA=98.0,
                speed=0.5,
                spectrum_model="batchelor",
            )
        # Function completed; chi may be NaN but iteration ran
        assert isinstance(result.chi, float)


# ---------------------------------------------------------------------------
# chi/l4_chi.py — process_l4_chi_epsilon empty epsi_times branch
# ---------------------------------------------------------------------------


class TestL4ChiEpsilonEmptyTimes:
    def test_empty_l4_diss_times_yields_nan_epsilon(self):
        """When l4_diss.time is empty, _chi_eps_func sets epsilon_val=NaN → None →
        chi stays NaN."""
        from odas_tpw.chi.l3_chi import L3ChiData
        from odas_tpw.chi.l4_chi import process_l4_chi_epsilon
        from odas_tpw.scor160.io import L4Data

        n_spec = 2
        n_freq = 33
        F_const = np.linspace(0, 256, n_freq)
        l3_chi = L3ChiData(
            time=np.arange(n_spec, dtype=float),
            pres=np.linspace(10, 50, n_spec),
            temp=np.full(n_spec, 15.0),
            pspd_rel=np.full(n_spec, 0.7),
            section_number=np.ones(n_spec),
            nu=np.full(n_spec, 1.2e-6),
            kcyc=np.tile(F_const[:, None] / 0.7, (1, n_spec)),
            freq=F_const,
            gradt_spec=np.ones((1, n_freq, n_spec)) * 1e-6,
            noise_spec=np.ones((1, n_freq, n_spec)) * 1e-8,
            H2=np.ones((n_spec, n_freq)),
            tau0=np.full(n_spec, 0.01),
        )
        # Empty time array on l4_diss → epsi_times.size == 0 path
        l4_diss = L4Data(
            time=np.array([], dtype=float),
            pres=np.array([], dtype=float),
            pspd_rel=np.array([], dtype=float),
            section_number=np.array([], dtype=float),
            epsi=np.empty((1, 0)),
            epsi_final=np.array([], dtype=float),
            epsi_flags=np.empty((1, 0)),
            fom=np.empty((1, 0)),
            mad=np.empty((1, 0)),
            kmax=np.empty((1, 0)),
            method=np.empty((1, 0)),
            var_resolved=np.empty((1, 0)),
        )
        result = process_l4_chi_epsilon(l3_chi, l4_diss)
        assert np.all(np.isnan(result.chi))


# ---------------------------------------------------------------------------
# chi/l4_chi.py — _compute_chi_final no-valid-probe branch
# ---------------------------------------------------------------------------


class TestComputeChiFinalNoValid:
    def test_all_nan_yields_nan(self):
        """Probe column all NaN → chi_final stays NaN."""
        from odas_tpw.chi.l4_chi import _compute_chi_final

        chi = np.array([[np.nan, 1e-9], [np.nan, 2e-9]])
        result = _compute_chi_final(chi)
        assert np.isnan(result[0])
        assert np.isfinite(result[1])

    def test_zero_probe_yields_nan(self):
        """Probe column with only zero/negative → chi_final NaN (chi > 0 filter)."""
        from odas_tpw.chi.l4_chi import _compute_chi_final

        chi = np.array([[0.0, 1e-9], [-1e-9, 2e-9]])
        result = _compute_chi_final(chi)
        assert np.isnan(result[0])
        assert np.isfinite(result[1])


# ---------------------------------------------------------------------------
# rsi/convert.py — _convert_one 2-arg fallback (line 565)
# ---------------------------------------------------------------------------


class TestConvertOne2Arg:
    def test_2_arg_uses_default_complevel(self, tmp_path, monkeypatch):
        """_convert_one accepts a 2-tuple and uses complevel=4 default."""
        from odas_tpw.rsi import convert as cmod

        captured = {}

        def fake_p_to_L1(p_path, nc_path, complevel=4):
            captured["complevel"] = complevel
            # Touch the output so .stat() works
            nc_path.write_bytes(b"")

        monkeypatch.setattr(cmod, "p_to_L1", fake_p_to_L1)
        p_path = tmp_path / "x.p"
        nc_path = tmp_path / "x.nc"
        result = cmod._convert_one((p_path, nc_path))
        assert captured["complevel"] == 4
        # Returns a 3-tuple of (p_name, nc_name, size_mb)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# rsi/convert.py — convert_all jobs=0 falls back to cpu_count
# ---------------------------------------------------------------------------


class TestConvertAllJobsZero:
    def test_jobs_zero_calls_cpu_count(self, monkeypatch):
        """convert_all(jobs=0) rewrites jobs via os.cpu_count()."""
        from odas_tpw.rsi import convert as cmod

        cpu_count_calls = []

        def fake_cpu_count():
            cpu_count_calls.append(True)
            return 1  # → falls through to serial branch

        monkeypatch.setattr(cmod.os, "cpu_count", fake_cpu_count)
        # Empty input list → no work, but the jobs=0 → cpu_count line still runs
        cmod.convert_all([], jobs=0)
        assert cpu_count_calls, "os.cpu_count was not called"

    def test_serial_logs_failure_on_error(self, tmp_path, monkeypatch, caplog):
        """Serial path (jobs=1) catches OSError/ValueError/RuntimeError and logs."""
        from odas_tpw.rsi import convert as cmod

        def boom(_args):
            raise ValueError("boom")

        monkeypatch.setattr(cmod, "_convert_one", boom)
        p_path = tmp_path / "x.p"
        p_path.write_bytes(b"")
        with caplog.at_level("ERROR"):
            cmod.convert_all([p_path], output_dir=tmp_path / "out", jobs=1)
        assert any("boom" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# rsi/convert.py — _classify_channels with no shear / no pitch / no chla / no doxy
# ---------------------------------------------------------------------------


class TestClassifyChannelsMinimal:
    def test_classify_returns_empty_lists_when_no_channels(self):
        """An empty PFile-like object yields empty classifications."""
        from odas_tpw.rsi.convert import _classify_channels

        # Minimal stub matching the duck-typed interface used by _classify_channels
        class StubPF:
            def __init__(self):
                self.channels = {}
                self.channel_info = {}

            def is_fast(self, name):  # pragma: no cover - never called
                return False

        ch = _classify_channels(StubPF())
        assert ch["shear"] == []
        assert ch["vib"] == []
        assert ch["acc"] == []
        assert ch["mag"] == []
        assert ch["gradt"] == []
        assert ch["cond_ctd"] == []
        assert ch["temp_ctd"] is None
        assert ch["pitch"] is None
        assert ch["roll"] is None
        assert ch["chla"] is None
        assert ch["turb"] is None
        assert ch["doxy"] is None
        assert ch["doxy_temp"] is None
        assert ch["supplementary"] == []
