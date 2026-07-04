# Mar-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for perturb.ctd — CTD time-binning."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from odas_tpw.perturb.ctd import _assign_gps_with_casts, _time_bin, ctd_bin_file
from odas_tpw.perturb.gps import GPSFixed


class _MovingGPS:
    """Straight-line ship track: lat = t, lon = 2t (degrees vs epoch seconds)."""

    def lat(self, t):
        return np.asarray(t, dtype=float) * 1.0

    def lon(self, t):
        return np.asarray(t, dtype=float) * 2.0


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

    def test_diagnostics_n_excludes_nan_samples(self):
        """{name}_n counts only the finite values actually averaged; n_samples
        counts all in-range samples. A NaN data point makes them differ (#42)."""
        t = np.array([0.0, 1.0, 2.0, 3.0])
        T = np.array([1.0, np.nan, 3.0, 4.0])
        for method in ("mean", "median"):
            r = _time_bin(t, {"T": T}, bin_width=10.0, method=method, diagnostics=True)
            assert "T_n" in r, method
            # One bin holds all four times; only three T values are finite.
            assert int(r["n_samples"][0]) == 4, method
            assert int(r["T_n"][0]) == 3, method

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

    def test_nan_and_out_of_range_times_excluded_not_folded(self):
        """NaN and below/above-edge times must NOT pollute the first/last bin
        (the np.clip fold this replaced did, audit #69/#70)."""
        edges = np.array([0.0, 1.0, 2.0, 3.0])  # 3 bins centered 0.5/1.5/2.5
        # One clean sample per bin, plus three pollutants:
        #   t=-5 (below), t=99 (above), t=NaN — each carrying value 1000.
        t = np.array([0.5, 1.5, 2.5, -5.0, 99.0, np.nan])
        vals = np.array([10.0, 20.0, 30.0, 1000.0, 1000.0, 1000.0])
        for method in ("mean", "median"):
            r = _time_bin(t, {"T": vals}, bin_width=1.0, method=method,
                          diagnostics=True, bin_edges=edges)
            np.testing.assert_allclose(r["T"], [10.0, 20.0, 30.0])
            # The pollutants never entered a bin: counts are exactly one each.
            np.testing.assert_array_equal(r["n_samples"], [1, 1, 1])


class TestAssignGpsWithCasts:
    """The VMP-aware position model (ports Matlab ctd2binned.addGPS)."""

    def test_pinned_descent_and_linear_gap_decay(self):
        gps = _MovingGPS()
        bin_epoch = np.array([5.0, 15.0, 30.0, 45.0, 52.0, 60.0])
        starts = np.array([10.0, 40.0])
        ends = np.array([20.0, 50.0])
        lat, lon = _assign_gps_with_casts(bin_epoch, starts, ends, gps)
        # before first cast: ship fix at the bin time
        assert (lon[0], lat[0]) == pytest.approx((10.0, 5.0))
        # during cast 0: pinned to the ship fix at the cast START (t=10)
        assert (lon[1], lat[1]) == pytest.approx((20.0, 10.0))
        # gap 0 interior (t=30): linear decay from drop point onto ship track
        assert (lon[2], lat[2]) == pytest.approx((50.0, 25.0))
        # during cast 1: pinned to the ship fix at t=40
        assert (lon[3], lat[3]) == pytest.approx((80.0, 40.0))
        # gap 1 interior (t=52)
        assert (lon[4], lat[4]) == pytest.approx((88.0, 44.0))
        # final bin (end of the trailing gap) is NaN, matching the reference
        assert np.isnan(lon[5]) and np.isnan(lat[5])

    def test_gap_runs_from_drop_point_to_ship(self):
        gps = _MovingGPS()
        # cast [10, 20]; next cast starts at 40. Probe just inside each gap end.
        bin_epoch = np.array([15.0, 20.0001, 39.9999, 45.0])
        _lat, lon = _assign_gps_with_casts(
            bin_epoch, np.array([10.0, 40.0]), np.array([20.0, 50.0]), gps
        )
        drop = lon[0]  # during cast 0 == gps.lon(10) == 20 (the drop point)
        assert lon[1] == pytest.approx(drop, abs=1e-2)         # gap start ~ drop
        assert lon[2] == pytest.approx(gps.lon(40.0), abs=1e-2)  # gap end ~ ship

    def test_no_casts_is_plain_ship_track(self):
        gps = _MovingGPS()
        bin_epoch = np.array([1.0, 2.0, 3.0])
        lat, lon = _assign_gps_with_casts(
            bin_epoch, np.array([]), np.array([]), gps
        )
        assert np.allclose(lon, [2.0, 4.0, 6.0])
        assert np.allclose(lat, [1.0, 2.0, 3.0])


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

    def test_stratification_excluded_from_ctd(self, tmp_path):
        # N2/dTdz are profile-only; even when the pipeline has injected them as
        # slow channels, the CTD product (whole up/down trajectory) must not
        # carry them. The base scalars/seawater props are still present.
        n = 640
        t_slow = np.linspace(0, 10, n)
        channels = {
            "JAC_T": np.linspace(5.0, 15.0, n),
            "JAC_C": np.linspace(30.0, 35.0, n),
            "P": np.linspace(0.0, 100.0, n),
            "N2": np.linspace(1e-5, 1e-4, n),
            "dTdz": np.linspace(0.01, 0.1, n),
        }
        pf = self._make_pf(channels, t_slow=t_slow)
        gps = GPSFixed(15.0, 145.0)

        out = ctd_bin_file(pf, gps, tmp_path, bin_width=0.5)
        assert out is not None
        ds = xr.open_dataset(out)
        try:
            assert "N2" not in ds and "dTdz" not in ds
            for var in ("JAC_T", "JAC_C", "P", "SP", "sigma0"):
                assert var in ds, f"{var} missing from output"
        finally:
            ds.close()

    def test_time_coord_has_no_fillvalue(self, tmp_path):
        # Regression: CF-1.13 §2.5.1 forbids _FillValue on coordinate
        # variables. xarray auto-emits one for the float "time" coord unless
        # encoding={'time': {'_FillValue': None}} is passed to to_netcdf.
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

        # Inspect the raw on-disk attributes (decode_cf=False keeps _FillValue
        # in .attrs rather than folding it into .encoding).
        raw = xr.open_dataset(out, decode_cf=False)
        try:
            assert "_FillValue" not in raw["time"].attrs
            for cname in raw.coords:
                assert "_FillValue" not in raw[cname].attrs, (
                    f"_FillValue present on coordinate {cname}"
                )
        finally:
            raw.close()

    def test_vmp_aware_pins_descent_and_differs_from_flat(self, tmp_path):
        n = 640
        t_slow = np.linspace(0.0, 10.0, n)
        half = n // 2
        # Sawtooth pressure: descend in the first half, reel in over the second.
        P = np.concatenate([np.linspace(0, 100, half), np.linspace(100, 0, n - half)])
        channels = {
            "JAC_T": np.linspace(5.0, 15.0, n),
            "JAC_C": np.linspace(30.0, 35.0, n),
            "P": P,
        }
        gps = _MovingGPS()

        flat_path = ctd_bin_file(self._make_pf(dict(channels), t_slow=t_slow), gps,
                                 tmp_path / "flat", bin_width=0.5)
        vmp_path = ctd_bin_file(self._make_pf(dict(channels), t_slow=t_slow), gps,
                                tmp_path / "vmp", bin_width=0.5,
                                profiles=[(0, half - 1)], direction="down")
        # decode_times=False keeps `time` numeric (epoch seconds) for the window test.
        flat = xr.open_dataset(flat_path, decode_times=False)
        vmp = xr.open_dataset(vmp_path, decode_times=False)

        # The cast-aware path must change the positions vs the plain ship track.
        assert not np.allclose(
            flat["lat"].values, vmp["lat"].values, equal_nan=True
        )
        # During the down-cast window the VMP-aware latitude is pinned (constant),
        # whereas the flat ship track ramps across it.
        cast_end_t = t_slow[half - 1]
        bt = np.asarray(vmp["time"].values, dtype=float)  # epoch s; offset 0 here
        in_cast = (bt >= 0.0) & (bt <= cast_end_t)
        assert np.nanstd(vmp["lat"].values[in_cast]) < 1e-9
        assert np.nanstd(flat["lat"].values[in_cast]) > 1e-3
        flat.close()
        vmp.close()

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

    def test_output_stem_override(self, tmp_path):
        n = 64
        t_slow = np.linspace(0, 1, n)
        channels = {
            "JAC_T": np.linspace(5.0, 15.0, n),
            "JAC_C": np.linspace(30.0, 35.0, n),
            "P": np.linspace(0.0, 100.0, n),
        }
        pf = self._make_pf(channels, t_slow=t_slow, filepath=Path("cast.p"))

        out = ctd_bin_file(
            pf, GPSFixed(15.0, 145.0), tmp_path, output_stem="SN001__cast"
        )
        assert out == tmp_path / "SN001__cast.nc"
        assert out.exists()

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
