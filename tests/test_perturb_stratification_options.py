"""perturb wiring of stratification.sort / dTdz_source and the kappa notices."""

import json
import logging
from typing import ClassVar

import gsw
import numpy as np
import pytest
import xarray as xr

from odas_tpw.perturb import config as cfg
from odas_tpw.perturb.pipeline import (
    _stratification_notices,
    _validate_stratification_options,
    _window_stratification_for_profile,
)

N = 1280  # 20 s at 64 Hz


def _profile_nc(path, *, fp07=("T1", "T2"), t2_flat=False, salinity_noise=0.0, fast=False):
    """A synthetic profile NetCDF: JAC_T/JAC_C plus FP07 channels.

    The FP07s read 3.1 K warm (nominal coefficients), which must not matter.
    """
    rng = np.random.default_rng(3)
    t_slow = np.arange(N) / 64.0
    P = 100.0 + 0.8 * t_slow
    z = -gsw.z_from_p(P, 0.0)
    T = 18.0 - 0.07 * (z - z[0])
    S = 36.0 - 0.0037 * (z - z[0])
    if salinity_noise:
        k = np.exp(-np.arange(-128, 129) ** 2 / (2 * 32.0**2))
        noise = np.convolve(rng.normal(size=N + 256), k / k.sum(), mode="valid")[:N]
        S = S + salinity_noise * noise / noise.std()
    C = gsw.C_from_SP(S, T, P)
    data = {
        "t_slow": ("time_slow", t_slow),
        "P": ("time_slow", P),
        "JAC_T": ("time_slow", T),
        "JAC_C": ("time_slow", C),
    }
    attrs = {"fs_fast": 512.0, "fs_slow": 64.0}
    if fast:
        t_fast = np.arange(8 * N) / 512.0
        data["t_fast"] = ("time_fast", t_fast)
    for i, name in enumerate(fp07):
        x = T + 3.1 + 0.6 * i
        if t2_flat and name == "T2":
            x = np.full(N, 16.145)  # a test plug reads the config's T_0
        dim = "time_slow"
        if fast:
            x = np.interp(np.arange(8 * N) / 512.0, t_slow, x)
            dim = "time_fast"
        data[name] = xr.Variable(
            dim, x, attrs={"sensor_type": "therm", "calibration": f"in-situ {name}"}
        )
    xr.Dataset(data, attrs=attrs).to_netcdf(path)
    return t_slow, P, z, T


class TestConfig:
    # main's (cc5107d) canonical stratification section, engine fingerprint
    # excluded. Its products are density-sorted.
    MAIN: ClassVar[dict] = {"enable": True, "salinity": None, "window": 2}

    def _canon(self, user):
        merged = cfg.merge_config("stratification", user)
        return json.loads(cfg.canonicalize("stratification", merged))["stratification"]

    def test_density_keeps_mains_canonical_form(self):
        # sort: density is what main computed, so it keeps main's stage signature
        # and its products stay valid.
        assert self._canon({"sort": "density"}) == self.MAIN
        assert self._canon({"sort": "density", "dTdz_source": "ct"}) == self.MAIN

    def test_default_hashes_apart_from_main(self):
        # The default is now the temperature sort: a config that leaves sort
        # unset must not reuse a density-sorted product from before the change.
        assert self._canon({}) == {**self.MAIN, "sort": "temperature"}
        assert self._canon({"sort": None}) == self._canon({})
        assert self._canon({"sort": "temperature"}) == self._canon({})

    def test_setting_a_key_changes_the_canonical_form(self):
        a = self._canon({})
        for extra in ({"sort": "density"}, {"dTdz_source": "fp07"}, {"sort": "none"}):
            assert self._canon(extra) != a, extra

    @pytest.mark.parametrize(
        "strat",
        [{"sort": "Density"}, {"sort": "salinity"}, {"dTdz_source": "FP07"}, {"dTdz_source": "T1"}],
    )
    def test_bad_values_rejected_at_load_and_per_file(self, strat):
        with pytest.raises(ValueError, match="stratification"):
            cfg.validate_config({"stratification": strat})
        with pytest.raises(ValueError, match="stratification"):
            _validate_stratification_options({"stratification": strat})

    def test_good_values_accepted(self):
        for strat in ({}, {"sort": "temperature", "dTdz_source": "fp07"}, {"sort": "none"}):
            cfg.validate_config({"stratification": strat})
            _validate_stratification_options({"stratification": strat})


class TestWindowFp07:
    WIN = np.array([4.0, 8.0, 12.0, 16.0])

    def test_fp07_gradient_used_and_named(self, tmp_path):
        prof = tmp_path / "p.nc"
        _profile_nc(prof)
        diag: dict = {}
        N2, dTdz, _ = _window_stratification_for_profile(
            prof, self.WIN, 2.0, "JAC_T", "JAC_C", "p.nc", dTdz_source="fp07", diag=diag
        )
        np.testing.assert_allclose(dTdz, -0.07, rtol=0.01)  # the 3 K offsets don't matter
        assert "T1 (in-situ T1)" in diag["dTdz_comment"]
        assert "T2 (in-situ T2)" in diag["dTdz_comment"]
        assert diag["stratification"] == "sort=temperature; dTdz_source=fp07 (T1, T2)"
        assert np.all(N2 > 0)

    def test_fast_fp07_channels(self, tmp_path):
        prof = tmp_path / "p.nc"
        _profile_nc(prof, fast=True)
        diag: dict = {}
        _, dTdz, _ = _window_stratification_for_profile(
            prof, self.WIN, 2.0, "JAC_T", "JAC_C", "p.nc", dTdz_source="fp07", diag=diag
        )
        np.testing.assert_allclose(dTdz, -0.07, rtol=0.01)
        assert "fp07" in diag["stratification"]

    def test_flat_probe_is_dropped_not_averaged(self, tmp_path, caplog):
        prof = tmp_path / "p.nc"
        _profile_nc(prof, t2_flat=True)
        diag: dict = {}
        with caplog.at_level(logging.WARNING):
            _, dTdz, _ = _window_stratification_for_profile(
                prof, self.WIN, 2.0, "JAC_T", "JAC_C", "p.nc", dTdz_source="fp07", diag=diag
            )
        # Averaging the flat bead in would halve dT/dz and put K_T 4x high.
        np.testing.assert_allclose(dTdz, -0.07, rtol=0.01)
        assert "T2" in diag["fp07_dropped"]
        assert "does not track the reference" in diag["fp07_dropped"]["T2"]
        assert "FP07 dT/dz dropped T2" in caplog.text

    def test_no_fp07_falls_back_to_ct(self, tmp_path, caplog):
        prof = tmp_path / "p.nc"
        _profile_nc(prof, fp07=())
        diag: dict = {}
        with caplog.at_level(logging.WARNING):
            out = _window_stratification_for_profile(
                prof, self.WIN, 2.0, "JAC_T", "JAC_C", "p.nc", dTdz_source="fp07", diag=diag
            )
        assert out is not None
        np.testing.assert_allclose(out[1], -0.07, rtol=0.01)
        assert "fell back to JAC_T" in diag["dTdz_comment"]
        assert diag["fp07_fallback"]
        assert "falls back to JAC_T" in caplog.text

    def test_fp07_path_raising_falls_back(self, tmp_path, monkeypatch):
        # A stratification problem must not cost the diss/chi product.
        import odas_tpw.perturb.pipeline as pl

        def boom(*a, **k):
            raise RuntimeError("synthetic")

        monkeypatch.setattr(pl, "_window_fp07_gradient", boom)
        prof = tmp_path / "p.nc"
        _profile_nc(prof)
        diag: dict = {}
        out = _window_stratification_for_profile(
            prof, self.WIN, 2.0, "JAC_T", "JAC_C", "p.nc", dTdz_source="fp07", diag=diag
        )
        assert out is not None and np.isfinite(out[1]).all()
        assert "synthetic" in diag["fp07_fallback"]

    def test_reader_raising_falls_back(self, tmp_path, monkeypatch):
        import odas_tpw.perturb.pipeline as pl

        monkeypatch.setattr(pl, "_profile_fp07_series", lambda *a: 1 / 0)
        prof = tmp_path / "p.nc"
        _profile_nc(prof)
        diag: dict = {}
        out = _window_stratification_for_profile(
            prof, self.WIN, 2.0, "JAC_T", "JAC_C", "p.nc", dTdz_source="fp07", diag=diag
        )
        assert out is not None
        assert "ZeroDivisionError" in diag["fp07_fallback"]


class TestWindowSortAndKappa:
    WIN = np.arange(1.5, 19.0, 0.75)  # 24 windows

    def test_ct_comment_names_the_sort(self, tmp_path):
        prof = tmp_path / "p.nc"
        _profile_nc(prof)
        for sort, phrase in (
            ("density", "density-sorted (Thorpe)"),
            ("temperature", "temperature-sorted"),
            ("none", "unsorted"),
        ):
            diag: dict = {}
            _window_stratification_for_profile(
                prof, self.WIN, 0.75, "JAC_T", "JAC_C", "p.nc", sort=sort, diag=diag
            )
            assert phrase in diag["dTdz_comment"], sort
            assert "in-situ" not in diag["dTdz_comment"]

    @pytest.mark.parametrize(
        ("sort", "phrase"),
        [
            (None, "a density-sorted dT/dz would be"),  # the default, temperature
            ("density", "the density-sorted dT/dz is"),
        ],
    )
    def test_noisy_salinity_sets_kappa_and_notice(self, tmp_path, caplog, sort, phrase):
        from odas_tpw.perturb.pipeline import _attach_window_stratification

        prof = tmp_path / "p.nc"
        _profile_nc(prof, salinity_noise=0.2)
        ds = xr.Dataset(coords={"t": ("time", self.WIN)})
        notices: list = []
        kw = {} if sort is None else {"sort": sort}
        with caplog.at_level(logging.WARNING):
            _attach_window_stratification(
                ds, prof, 0.75, "JAC_T", "JAC_C", "p.nc", "dissipation", notices=notices, **kw
            )
        assert ds["N2"].attrs["sort_kappa"] < 0.8
        assert ds["N2"].attrs["stratification"] == (
            f"sort={sort or 'temperature'}; dTdz_source=ct (JAC_T)"
        )
        assert phrase in caplog.text
        assert notices and notices[0]["kappa"] < 0.8

    def test_fp07_gradient_is_not_called_density_biased(self, tmp_path, caplog):
        # sort: density with dTdz_source: fp07 writes a temperature-sorted FP07
        # gradient, so the warning must not call that dT/dz biased low (nor
        # advise the fp07 source it already uses).
        from odas_tpw.perturb.pipeline import _attach_window_stratification

        prof = tmp_path / "p.nc"
        _profile_nc(prof, salinity_noise=0.2)
        ds = xr.Dataset(coords={"t": ("time", self.WIN)})
        with caplog.at_level(logging.WARNING):
            _attach_window_stratification(
                ds, prof, 0.75, "JAC_T", "JAC_C", "p.nc", "dissipation",
                sort="density", dTdz_source="fp07",
            )
        assert ds["N2"].attrs["sort_kappa"] < 0.8
        assert "a density-sorted dT/dz would be" in caplog.text
        assert "biased low" not in caplog.text

    def test_fixed_salinity_has_no_kappa(self, tmp_path):
        prof = tmp_path / "p.nc"
        _profile_nc(prof, salinity_noise=0.2)
        diag: dict = {}
        _window_stratification_for_profile(
            prof, self.WIN, 0.75, "JAC_T", "JAC_C", "p.nc", 35.0, diag=diag
        )
        assert diag["kappa"] is None


class TestNotices:
    def test_quiet_when_nothing_to_say(self):
        assert _stratification_notices({"a.p": {"windows": []}}) == []

    def test_kappa_fallback_and_dropped_lines(self):
        out = _stratification_notices(
            {
                "a.p": {
                    "profile": {"kappa": {"n_low": 2, "n": 3, "worst": 0.4, "worst_P": 175.0}},
                    "windows": [{"kappa": 0.6, "where": "x"}],
                },
                "b.p": {"windows": [{"fp07_fallback": "no FP07 channel", "where": "y"}]},
                "c.p": {"windows": [{"fp07_dropped": {"T2": "flat"}, "where": "z"}]},
            }
        )
        text = "\n".join(out)
        assert "scrambles the density order in 1 file(s)" in text and "lowest 0.40" in text
        assert "(temperature-sorted) is spared" in text  # the default sort
        assert "fell back to the CT temperature in 1 file(s)" in text
        assert "FP07 T2 dropped from dT/dz in 1 file(s): c.p" in text

    def test_kappa_line_follows_the_sort(self):
        rec = {"a.p": {"windows": [{"kappa": 0.6, "where": "x"}]}}
        dens = "\n".join(_stratification_notices(rec, "density"))
        assert "dT/dz is biased low" in dens
        assert "biased low" not in "\n".join(_stratification_notices(rec, "temperature"))
        assert "(in measured order) is spared" in "\n".join(_stratification_notices(rec, "none"))


class TestProfileProduct:
    def _pf(self, salinity_noise=0.0, t2_flat=False):
        from unittest.mock import MagicMock

        rng = np.random.default_rng(5)
        n = 64 * 150
        t = np.arange(n) / 64.0
        P = 150.0 + 0.8 * t
        z = -gsw.z_from_p(P, 0.0)
        T = 18.0 - 0.07 * (z - z[0])
        S = 36.0 - 0.0037 * (z - z[0])
        if salinity_noise:
            k = np.exp(-np.arange(-256, 257) ** 2 / (2 * 64.0**2))
            noise = np.convolve(rng.normal(size=n + 512), k / k.sum(), mode="valid")[:n]
            S = S + salinity_noise * noise / noise.std()
        pf = MagicMock()
        pf.channels = {
            "P": P,
            "JAC_T": T,
            "JAC_C": gsw.C_from_SP(S, T, P),
            "T1": T + 3.1,
            "T2": np.full(n, 16.145) if t2_flat else T + 3.7,
        }
        pf.channel_info = {
            "T1": {"type": "therm", "calibration": "in-situ (Steinhart-Hart order 2 vs JAC_T)"},
            "T2": {"type": "therm"},
        }
        pf.is_fast = lambda name: False
        pf.t_slow, pf.t_fast = t, np.arange(8 * n) / 512.0
        pf.fs_slow, pf.fs_fast = 64.0, 512.0
        return pf, n

    def test_fp07_source_in_profile_product(self):
        from odas_tpw.perturb.pipeline import _inject_slow_stratification

        pf, n = self._pf(t2_flat=True)
        out = _inject_slow_stratification(
            pf, [(0, n - 1)], {"window": 2.0, "dTdz_source": "fp07"}, {}, "f.p"
        )
        d = pf.channels["dTdz"]
        np.testing.assert_allclose(np.nanmedian(d), -0.07, rtol=0.01)
        info = pf.channel_info["dTdz"]
        assert info["stratification"] == "sort=temperature; dTdz_source=fp07 (T1)"
        assert "T1 (in-situ (Steinhart-Hart order 2 vs JAC_T))" in info["comment"]
        assert "Dropped: T2" in info["comment"]
        assert out["fp07_dropped"].keys() == {"T2"}
        assert np.isfinite(pf.channels["N2"]).any()

    def test_kappa_recorded_for_noisy_salinity(self, caplog):
        from odas_tpw.perturb.pipeline import _inject_slow_stratification

        pf, n = self._pf(salinity_noise=0.2)
        with caplog.at_level(logging.WARNING):
            out = _inject_slow_stratification(
                pf, [(0, n - 1)], {"window": 2.0, "sort": "density"}, {}, "f.p"
            )
        assert out["kappa"]["worst"] < 0.8
        assert "sort_kappa" in pf.channel_info["N2"]["comment"]
        assert "the density-sorted dT/dz is" in caplog.text
        dens = np.nanmedian(pf.channels["dTdz"])
        # ...and the default (temperature) sort brings dT/dz back, with the same
        # kappa and a warning worded for it.
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            out_t = _inject_slow_stratification(pf, [(0, n - 1)], {"window": 2.0}, {}, "f.p")
        np.testing.assert_allclose(np.nanmedian(pf.channels["dTdz"]), -0.07, rtol=0.03)
        assert abs(dens) < 0.9 * abs(np.nanmedian(pf.channels["dTdz"]))
        assert out_t["kappa"] == out["kappa"]
        assert "a density-sorted dT/dz would be" in caplog.text


DATA = __import__("pathlib").Path(__file__).parent / "data"


class TestDensityPathIdentity:
    """N2/dTdz under sort="density" must equal origin/main's (cc5107d) on real data.

    main's default was the density sort; the fixture was made with main's code.
    """

    def test_matches_stored_main_result(self):
        from odas_tpw.processing.mixing import profile_stratification, sorted_stratification
        from odas_tpw.rsi import PFile

        ref = np.load(DATA / "strat_main_cc5107d_SN479_0006.npz")
        pf = PFile(DATA / "SN479_0006.p")
        P = np.asarray(pf.channels["P"], float)
        T = np.asarray(pf.channels["JAC_T"], float)
        SP = gsw.SP_from_C(np.asarray(pf.channels["JAC_C"], float), T, P)
        t = np.asarray(pf.t_slow, float)
        sel = P > 3.0
        tp, N2p, dTp = profile_stratification(P[sel], T[sel], SP[sel], window=2.0, sort="density")
        w = sorted_stratification(ref["win_t"], 1.0, t, P, T, S=SP, sort="density")
        w35 = sorted_stratification(ref["win_t"], 1.0, t, P, T, S=None, sort="density")
        # rtol, not equality: LAPACK may differ in the last bits across platforms.
        for got, key in (
            (tp, "profile_P"), (N2p, "profile_N2"), (dTp, "profile_dTdz"),
            (w.N2, "window_N2"), (w.dTdz, "window_dTdz"),
            (w35.N2, "window35_N2"), (w35.dTdz, "window35_dTdz"),
        ):
            np.testing.assert_allclose(got, ref[key], rtol=1e-10, atol=0, equal_nan=True,
                                       err_msg=key)
        # The default (temperature) sort really is a different computation here:
        # SN479's JAC salinity reorders a density sort (sort_kappa ~0.75).
        wt = sorted_stratification(ref["win_t"], 1.0, t, P, T, S=SP)
        ok = np.isfinite(wt.dTdz) & np.isfinite(ref["window_dTdz"])
        assert np.median(np.abs(wt.dTdz[ok])) > np.median(np.abs(ref["window_dTdz"][ok]))


class TestProcessFileEndToEnd:
    def _run(self, tmp_path, strat):
        import warnings

        from odas_tpw.perturb.gps import GPSFixed
        from odas_tpw.perturb.pipeline import process_file

        dirs = {k: tmp_path / k for k in ("profiles", "diss", "chi")}
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = process_file(
                DATA / "SN479_0006.p",
                {"chi": {"enable": True}, "stratification": strat},
                GPSFixed(15.0, 145.0),
                dirs,
            )
        return res, dirs

    def test_options_reach_every_product(self, tmp_path):
        res, dirs = self._run(tmp_path, {"sort": "temperature", "dTdz_source": "fp07"})
        assert not res.get("errors")
        for stage in ("profiles", "diss", "chi"):
            files = sorted(dirs[stage].glob("*.nc"))
            assert files, stage
            with xr.open_dataset(files[0]) as ds:
                assert ds["dTdz"].attrs["stratification"] == (
                    "sort=temperature; dTdz_source=fp07 (T1, T2)"
                ), stage
                # The in-situ tag exists only after fp07_calibrate: the profile
                # product's stratification now runs after the calibration.
                assert "in-situ (Steinhart-Hart" in ds["dTdz"].attrs["comment"], stage

    def test_defaults_are_labeled(self, tmp_path):
        _, dirs = self._run(tmp_path, {})
        for stage in ("profiles", "diss", "chi"):
            with xr.open_dataset(sorted(dirs[stage].glob("*.nc"))[0]) as ds:
                assert ds["dTdz"].attrs["stratification"] == (
                    "sort=temperature; dTdz_source=ct (JAC_T)"
                ), stage
                assert "temperature-sorted" in ds["dTdz"].attrs["comment"], stage
                if stage != "profiles":
                    # sort_kappa is computed under every sort (JAC salinity is
                    # per-sample), so the default products keep the diagnostic.
                    assert "sort_kappa" in ds["N2"].attrs, stage


class TestFastChannelRails:
    def test_rail_is_gated_before_the_boxcar(self, tmp_path):
        # A 58 C rail sample in a fast FP07 channel, box-car averaged first, would
        # become an in-range spike and steepen the gradient several-fold.
        prof = tmp_path / "p.nc"
        _profile_nc(prof, fast=True)
        with xr.open_dataset(prof) as ds:
            ds = ds.load()
        x = ds["T1"].values.copy()
        x[1000::4000] = 58.46
        ds["T1"] = ds["T1"].copy(data=x)
        prof2 = tmp_path / "p2.nc"
        ds.to_netcdf(prof2)
        diag: dict = {}
        _, dTdz, _ = _window_stratification_for_profile(
            prof2, TestWindowFp07.WIN, 2.0, "JAC_T", "JAC_C", "p2.nc", dTdz_source="fp07",
            diag=diag,
        )
        np.testing.assert_allclose(dTdz, -0.07, rtol=0.01)
