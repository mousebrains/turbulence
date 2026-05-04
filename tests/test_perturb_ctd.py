# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb.ctd — CTD time-binning."""

from pathlib import Path

import numpy as np
import xarray as xr

from odas_tpw.perturb.ctd import _time_bin, ctd_bin_file
from odas_tpw.perturb.gps import GPSFixed


class _PFileStub:
    def __init__(
        self,
        channels,
        t_slow,
        t_fast=None,
        fs_slow=64.0,
        fs_fast=512.0,
        filepath=None,
        fast_channels=None,
    ):
        self.channels = channels
        self.t_slow = t_slow
        self.t_fast = t_fast if t_fast is not None else t_slow
        self.fs_slow = fs_slow
        self.fs_fast = fs_fast
        self.filepath = filepath or Path("test_001.p")
        self._fast = fast_channels or set()

    def is_fast(self, ch):
        return ch in self._fast


class TestTimeBin:
    def test_basic_mean(self):
        t = np.array([0.0, 0.1, 0.2, 0.5, 0.6, 0.7, 1.0, 1.1])
        data = {"T": np.array([10.0, 10.2, 10.4, 11.0, 11.2, 11.4, 12.0, 12.2])}
        result = _time_bin(t, data, bin_width=0.5, method="mean")
        assert "bin_centers" in result
        assert "T" in result
        assert len(result["bin_centers"]) == len(result["T"])

    def test_median_method(self):
        t = np.arange(10.0)
        data = {"T": np.array([1, 1, 1, 1, 1, 100, 1, 1, 1, 1], dtype=float)}
        result_mean = _time_bin(t, data, bin_width=20.0, method="mean")
        result_median = _time_bin(t, data, bin_width=20.0, method="median")
        # Median should be more robust to the outlier
        assert result_median["T"][0] < result_mean["T"][0]

    def test_diagnostics_output(self):
        t = np.arange(20.0)
        data = {"T": np.random.randn(20)}
        result = _time_bin(t, data, bin_width=5.0, diagnostics=True)
        assert "n_samples" in result
        assert "T_std" in result

    def test_empty_bins(self):
        t = np.array([0.0, 0.1, 10.0, 10.1])
        data = {"T": np.array([1.0, 2.0, 3.0, 4.0])}
        result = _time_bin(t, data, bin_width=1.0)
        # Some bins should be NaN (gap between 0.1 and 10.0)
        assert np.any(np.isnan(result["T"]))

    def test_single_point(self):
        t = np.array([5.0])
        data = {"T": np.array([10.0])}
        result = _time_bin(t, data, bin_width=2.0)
        assert len(result["bin_centers"]) >= 1
        assert np.isfinite(result["T"][0])


class TestCtdBinFile:
    def _make_pf(self, channels, t_slow=None, fast_channels=None, filepath=None):
        if t_slow is None:
            t_slow = np.arange(0.0, 10.0, 1.0 / 64)
        return _PFileStub(
            channels=channels,
            t_slow=t_slow,
            fast_channels=fast_channels,
            filepath=filepath,
        )

    def test_basic(self, tmp_path):
        n = 640
        t_slow = np.linspace(0, 10, n)
        channels = {
            "JAC_T": np.linspace(5.0, 15.0, n),
            "JAC_C": np.linspace(30.0, 35.0, n),
            "P": np.linspace(0.0, 100.0, n),
        }
        pf = self._make_pf(channels, t_slow=t_slow)
        gps = GPSFixed(15.0, 145.0)

        out = ctd_bin_file(pf, gps, tmp_path, bin_width=0.5)
        assert out is not None
        assert out.exists()

        ds = xr.open_dataset(out)
        for var in ("JAC_T", "JAC_C", "P", "lat", "lon"):
            assert var in ds, f"{var} missing from output"
        assert ds.attrs["bin_width"] == 0.5
        assert ds.attrs["method"] == "mean"
        assert ds.attrs["source_file"] == "test_001.p"
        assert ds.attrs["Conventions"] == "CF-1.13, ACDD-1.3"
        # Seawater properties should be present (T, C, P all available)
        for var in ("SP", "SA", "CT", "sigma0", "rho", "depth"):
            assert var in ds, f"seawater property {var} missing"
        ds.close()

    def test_gps_interpolation(self, tmp_path):
        n = 640
        t_slow = np.linspace(0, 10, n)
        channels = {
            "JAC_T": np.full(n, 10.0),
            "JAC_C": np.full(n, 33.0),
            "P": np.full(n, 50.0),
        }
        pf = self._make_pf(channels, t_slow=t_slow)
        gps = GPSFixed(15.0, 145.0)

        out = ctd_bin_file(pf, gps, tmp_path)
        assert out is not None
        ds = xr.open_dataset(out)
        np.testing.assert_allclose(ds["lat"].values, 15.0)
        np.testing.assert_allclose(ds["lon"].values, 145.0)
        ds.close()

    def test_no_channels_returns_none(self, tmp_path):
        n = 64
        t_slow = np.linspace(0, 1, n)
        channels = {"Ax": np.zeros(n), "Ay": np.zeros(n)}
        pf = self._make_pf(channels, t_slow=t_slow)
        gps = GPSFixed(0.0, 0.0)

        result = ctd_bin_file(pf, gps, tmp_path)
        assert result is None

    def test_slow_and_fast_channels_share_time_dim(self, tmp_path):
        """Regression: when slow and fast channels are both binned, their
        last samples differ by ``1/fs_slow - 1/fs_fast`` and an independent
        ``arange(t_min, t_max + dt, dt)`` per call can flip one extra bin
        into existence. Both calls must share a single bin_edges so the
        resulting xarray Dataset has one consistent ``time`` dimension.

        Mirrors the SN465 OH3465_0030 case where Turbidity (fast) ended up
        with 281 bins vs JAC_C (slow) with 280, breaking dataset assembly.
        """
        fs_slow, fs_fast = 64.0, 512.0
        n_slow = 8960
        n_fast = n_slow * int(fs_fast / fs_slow)
        t_slow = np.arange(n_slow) / fs_slow
        t_fast = np.arange(n_fast) / fs_fast
        # t_fast ends 1/fs_slow - 1/fs_fast = ~13.7 ms past t_slow — the
        # fencepost that triggered the bug at 0.5 s bin width.
        assert t_fast[-1] > t_slow[-1]

        channels = {
            "JAC_T": np.linspace(20.0, 25.0, n_slow),
            "JAC_C": np.linspace(33.0, 35.0, n_slow),
            "P": np.linspace(0.0, 130.0, n_slow),
            "Turbidity": np.linspace(0.1, 0.5, n_fast),
        }
        pf = _PFileStub(
            channels=channels,
            t_slow=t_slow,
            t_fast=t_fast,
            fs_slow=fs_slow,
            fs_fast=fs_fast,
            fast_channels={"Turbidity"},
        )
        gps = GPSFixed(7.0, 134.0)

        out = ctd_bin_file(pf, gps, tmp_path, bin_width=0.5)
        assert out is not None

        ds = xr.open_dataset(out)
        # All channels must share the same time dim length.
        for var in ("JAC_T", "JAC_C", "P", "Turbidity"):
            assert ds[var].shape == ds["JAC_T"].shape, (
                f"{var} shape {ds[var].shape} != JAC_T {ds['JAC_T'].shape}"
            )
        ds.close()

    def test_diagnostics(self, tmp_path):
        n = 640
        t_slow = np.linspace(0, 10, n)
        channels = {
            "JAC_T": np.linspace(5.0, 15.0, n),
            "JAC_C": np.linspace(30.0, 35.0, n),
            "P": np.linspace(0.0, 100.0, n),
        }
        pf = self._make_pf(channels, t_slow=t_slow)
        gps = GPSFixed(15.0, 145.0)

        out = ctd_bin_file(pf, gps, tmp_path, diagnostics=True)
        assert out is not None
        ds = xr.open_dataset(out)
        assert "n_samples" in ds
        for var in ("JAC_T_std", "JAC_C_std", "P_std"):
            assert var in ds, f"diagnostic variable {var} missing"
        ds.close()

    def test_custom_variables(self, tmp_path):
        n = 640
        t_slow = np.linspace(0, 10, n)
        channels = {
            "JAC_T": np.linspace(5.0, 15.0, n),
            "JAC_C": np.linspace(30.0, 35.0, n),
            "P": np.linspace(0.0, 100.0, n),
            "DO": np.linspace(200.0, 250.0, n),
            "Chlorophyll": np.linspace(0.1, 1.0, n),
        }
        pf = self._make_pf(channels, t_slow=t_slow)
        gps = GPSFixed(15.0, 145.0)

        out = ctd_bin_file(pf, gps, tmp_path, variables=["DO"])
        assert out is not None
        ds = xr.open_dataset(out)
        # DO was explicitly requested
        assert "DO" in ds
        # Chlorophyll should NOT be present (auto-detect is skipped when variables is set)
        assert "Chlorophyll" not in ds
        # T, C, P are always collected when present
        for var in ("JAC_T", "JAC_C", "P"):
            assert var in ds
        ds.close()
