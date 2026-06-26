# Jun-2026, Claude and Pat Welch, pat@mousebrains.com
"""Regression tests for the four MAJOR findings of the 2026-06-25 deep audit.

Each test is constructed to FAIL on the pre-fix code and pass on the fix:
  M1  chi variance-correction grid under-resolved when kB << K_max
  M2  NetCDF _FillValue leaking into pressure/channels via masked .data
  M3  a single NaN T/C sample poisoning the CT lag estimate
  M4  bottom-crash reporting a bin center below the deepest real sample
"""

from __future__ import annotations

import numpy as np
import pytest

# --------------------------------------------------------------------------- #
# M1: chi variance correction is accurate even when kB << K_max
# --------------------------------------------------------------------------- #


class TestM1VarianceCorrection:
    @staticmethod
    def _reference(kB, K_max, K_min, speed, tau0, grad_func):
        from odas_tpw.chi.fp07 import fp07_transfer

        K_upper = max(K_max * 5, kB * 60)
        K = np.linspace(K_upper / 2_000_000, K_upper, 2_000_000)
        spec = grad_func(K, kB, 1.0)
        V_total = np.trapezoid(spec, K)
        h2 = np.abs(fp07_transfer(K * speed, tau0)) ** 2
        m = (K_min <= K) & (K_max >= K)
        return V_total / np.trapezoid(spec[m] * h2[m], K[m])

    @pytest.mark.parametrize("kB", [3.0, 5.0, 8.0, 15.0])
    @pytest.mark.parametrize("model", ["kraichnan", "batchelor"])
    def test_correction_within_tolerance_of_high_n_reference(self, kB, model):
        # Low-epsilon window: kB is small while K_max is pushed out near K_AA.
        # The old fixed-2000-point grid biases the correction by ~7-15% here;
        # the kB-scaled grid must stay well under 0.5%.
        from odas_tpw.chi.batchelor import batchelor_grad, kraichnan_grad
        from odas_tpw.chi.chi import _variance_correction
        from odas_tpw.chi.fp07 import fp07_transfer

        grad_func = kraichnan_grad if model == "kraichnan" else batchelor_grad
        K_max, K_min, speed, tau0 = 700.0, 1.0, 0.7, 7e-3

        def h2(F, t0):
            return np.abs(fp07_transfer(F, t0)) ** 2

        got = _variance_correction(kB, K_max, speed, tau0, h2, grad_func, K_min=K_min)
        ref = self._reference(kB, K_max, K_min, speed, tau0, grad_func)
        err = abs(got - ref) / ref
        assert err < 0.005, (
            f"{model} kB={kB}: correction err {err:.4%} (ref {ref:.5f}, got {got:.5f})"
        )


# --------------------------------------------------------------------------- #
# M2: _load_from_nc must turn _FillValue into NaN, not leak ~9.97e36
# --------------------------------------------------------------------------- #


class TestM2FillValueLeak:
    @staticmethod
    def _write_min_nc(path, *, p_fill_idx):
        """A minimal root-format full-record NetCDF _load_from_nc can read,
        with one slow-pressure element left unwritten (-> _FillValue)."""
        nc = pytest.importorskip("netCDF4")
        n_fast, n_slow = 16, 8
        ds = nc.Dataset(str(path), "w")
        ds.fs_fast = 512.0
        ds.fs_slow = 64.0
        ds.createDimension("time_fast", n_fast)
        ds.createDimension("time_slow", n_slow)
        tf = ds.createVariable("t_fast", "f8", ("time_fast",))
        tf[:] = np.arange(n_fast) / 512.0
        ts = ds.createVariable("t_slow", "f8", ("time_slow",))
        ts[:] = np.arange(n_slow) / 64.0
        P = ds.createVariable("P", "f8", ("time_slow",), fill_value=9.969209968386869e36)
        pv = np.arange(n_slow, dtype="f8")
        for i in range(n_slow):
            if i != p_fill_idx:
                P[i] = pv[i]  # leave p_fill_idx unwritten -> masked/fill
        T1 = ds.createVariable("T1", "f8", ("time_fast",))
        T1[:] = np.linspace(10.0, 11.0, n_fast)
        ds.close()

    def test_pressure_fill_becomes_nan(self, tmp_path):
        from odas_tpw.rsi.profile import _load_from_nc

        path = tmp_path / "fill.nc"
        self._write_min_nc(path, p_fill_idx=5)
        data = _load_from_nc(path)
        P = np.asarray(data["P"], dtype=float)
        # The pre-fix `[:].data` exposed the raw ~9.97e36 fill buffer here.
        assert not np.any(P > 1e30), f"fill value leaked into P: {P}"
        assert np.isnan(P[5]), "unwritten pressure element should be NaN"
        assert P[0] == 0.0 and P[4] == 4.0  # written values intact


# --------------------------------------------------------------------------- #
# M3: a single NaN sample must not poison the CT lag estimate
# --------------------------------------------------------------------------- #


class TestM3CtAlignNaN:
    @staticmethod
    def _make_profile(n, lag_samples, rng):
        base = np.cumsum(rng.standard_normal(n + abs(lag_samples) + 4))
        T = base[: n].copy()
        C = base[lag_samples : lag_samples + n].copy()  # C is base shifted
        return T, C

    def test_nan_profile_does_not_drive_minus_max_lag(self):
        from odas_tpw.processing.ct_align import ct_align

        fs, max_lag_s = 64.0, 5.0
        rng = np.random.default_rng(0)
        n = 900
        T_clean, C_clean = self._make_profile(n, 3, rng)

        # Lag from the clean profile alone.
        _, clean_lag = ct_align(T_clean, C_clean, fs, [(0, n - 1)], max_lag_s)

        # Now append a second profile that carries a single NaN in C.
        T2, C2 = self._make_profile(n, 3, rng)
        C2[n // 2] = np.nan
        T_all = np.concatenate([T_clean, T2])
        C_all = np.concatenate([C_clean, C2])
        _, mixed_lag = ct_align(
            T_all, C_all, fs, [(0, n - 1), (n, 2 * n - 1)], max_lag_s
        )

        # Pre-fix: the NaN profile slips the norm<=0/total<=0 guards and drags
        # the weighted median to the most-negative lag (-max_lag_seconds).
        assert mixed_lag != pytest.approx(-max_lag_s), "NaN profile drove -max_lag"
        assert mixed_lag == pytest.approx(clean_lag), (
            f"NaN profile changed the consensus lag {clean_lag} -> {mixed_lag}"
        )


# --------------------------------------------------------------------------- #
# M4: bottom-crash depth must never exceed the deepest real sample
# --------------------------------------------------------------------------- #


class TestM4BottomCrashDepth:
    def test_reported_depth_within_observed_max(self):
        from odas_tpw.processing.bottom import detect_bottom_crash

        # depth_minimum=10, bin_size=4 -> deepest bin [98, 102); with max depth
        # 99.0 its CENTER is 100.0 > 99.0, so the pre-fix code returned a depth
        # below the deepest sample and the caller's `P >= depth` matched nothing.
        n = 6000
        depth = np.linspace(10.0, 99.0, n)
        rng = np.random.default_rng(1)
        vib = 0.01 + np.abs(rng.standard_normal(n)) * 1e-3
        vib[depth >= 98.0] += np.abs(rng.standard_normal(int((depth >= 98.0).sum()))) * 5.0

        out = detect_bottom_crash(
            depth, {"v": vib}, fs=512.0, depth_window=4.0, depth_minimum=10.0
        )
        assert out is not None, "crash spike in the deepest bin should be detected"
        assert out <= np.nanmax(depth) + 1e-9, (
            f"reported crash depth {out} exceeds deepest sample {np.nanmax(depth)}"
        )
        # And it must fall within the flagged (deepest) bin's real samples.
        assert out >= 98.0 - 1e-9
