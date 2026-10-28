"""Record-locked shear interference removal (`epsilon.record_locked_shear`).

Unit tests of :func:`odas_tpw.rsi.record_locked.remove_record_locked` on
synthetic series, then the end-to-end contract on the real VMP-250 test file
(SN479_0006): a synthetic two-pulse waveform injected at FIXED record samples
into its clean shear, at 2x, 5x and 20x the in-profile record rms, with an
amplitude that drifts 6x over the cast as a real VMP-5500's did.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np
import pytest

from odas_tpw.rsi.dissipation import _compute_epsilon
from odas_tpw.rsi.helpers import load_channels, prepare_profiles
from odas_tpw.rsi.record_locked import (
    LOCAL_HALF_WINDOW,
    MIN_RECORDS,
    remove_record_locked,
)
from odas_tpw.scor160.l2 import _hp_filter

SAMPLE = Path(__file__).parent / "data" / "SN479_0006.p"
SPR = 512
FS = 512.03275  # a VMP-5500's: NOT an integer number of samples per second
HP = 0.25


def pulse(spr: int = SPR, at: tuple[float, ...] = (6.0, 81.5)) -> np.ndarray:
    """~16 ms pulses (sigma 3 samples) with a small undershoot."""
    p = np.arange(spr, dtype=float)
    out = np.zeros(spr)
    for c in at:
        out += np.exp(-0.5 * ((p - c) / 3.0) ** 2)
        out -= 0.15 * np.exp(-0.5 * ((p - c - 7.0) / 2.5) ** 2)
    return out


def noise(n: int, seed: int = 0) -> np.ndarray:
    """Band-limited stand-in for shear: white noise through a 3-point smoother."""
    rng = np.random.default_rng(seed)
    return np.convolve(rng.standard_normal(n + 2), np.ones(3) / 3, mode="valid")


def record_mean_z(y: np.ndarray, index0: int, spr: int = SPR) -> np.ndarray:
    """|z| of the record-locked mean of the high-passed *y*, per record sample."""
    k0 = (-index0) % spr
    nr = (len(y) - k0) // spr
    stack = _hp_filter(y, FS, HP)[k0 : k0 + nr * spr].reshape(nr, spr)
    m = stack.mean(axis=0)
    m -= np.median(m)
    return np.abs(m / (stack.std(axis=0, ddof=1) / np.sqrt(nr)))


class TestRemoveRecordLocked:
    def test_clean_noise_is_left_bit_identical(self):
        x = noise(300 * SPR)
        res = remove_record_locked(x, index0=0, samples_per_record=SPR, fs=FS, hp_cut=HP)
        assert res.samples == 0
        assert res.records >= MIN_RECORDS
        assert 0 <= res.snr < 5
        np.testing.assert_array_equal(res.signal, x)
        assert res.signal is not x

    def test_too_few_records_does_nothing(self):
        x = noise((MIN_RECORDS - 1) * SPR) + 10 * np.tile(pulse(), MIN_RECORDS - 1)
        res = remove_record_locked(x, index0=0, samples_per_record=SPR, fs=FS, hp_cut=HP)
        assert res.samples == 0
        assert np.isnan(res.snr) and np.isnan(res.peak)
        np.testing.assert_array_equal(res.signal, x)

    def test_removes_a_sub_threshold_pulse_and_touches_nothing_else(self):
        """2x the noise rms: invisible to the despiker, plain in the record mean."""
        n_rec, index0 = 300, 1234  # starts mid-record: partial records at both ends
        x0 = noise(n_rec * SPR + 300)
        phase = (index0 + np.arange(len(x0))) % SPR
        x = x0 + 2 * x0.std() * pulse()[phase]
        res = remove_record_locked(x, index0=index0, samples_per_record=SPR, fs=FS, hp_cut=HP)
        assert 10 <= res.samples <= 60
        assert res.snr > 20
        changed = res.signal != x
        # Only the significant record samples move -- in partial records too.
        corrected = np.unique(phase[changed])
        assert len(corrected) == res.samples
        assert {6, 81, 82} <= set(corrected.tolist())
        # Near the pulses only; the mask is circular, so the one at sample 6
        # may reach back across the record boundary to 510-511.
        assert np.all(
            (corrected < 30) | ((corrected > 60) & (corrected < 110)) | (corrected > 500)
        )
        tail = len(x) - (index0 + len(x)) % SPR  # first sample of the partial last record
        assert changed[: (-index0) % SPR].any() and changed[tail + 6]
        # And what is left is not record-locked.
        assert record_mean_z(res.signal, index0).max() < 5
        assert record_mean_z(x, index0).max() > 20

    def test_framing_is_by_sample_index_not_time(self):
        """fs is 512.03 Hz: over 1200 records a time-modulo phase walks ~40
        samples off the pulse. The index-modulo phase stays on it."""
        n_rec = 1200
        x0 = noise(n_rec * SPR, seed=1)
        x = x0 + 3 * x0.std() * np.tile(pulse(), n_rec)
        res = remove_record_locked(x, index0=0, samples_per_record=SPR, fs=FS, hp_cut=HP)
        t = np.arange(len(x)) / FS
        time_phase = np.floor((t % (SPR / 512.0)) * 512.0).astype(int)
        assert np.count_nonzero(time_phase != np.arange(len(x)) % SPR) > len(x) // 2
        late = slice((n_rec - 100) * SPR, n_rec * SPR)  # the last 100 records
        assert record_mean_z(res.signal[late], 0).max() < 5
        assert record_mean_z(x[late], 0).max() > 20

    def test_tracks_a_drifting_amplitude(self):
        """A single profile mean over- and under-subtracts a pulse that decays
        6x through the cast; the local waveform follows it."""
        n_rec = 600
        x0 = noise(n_rec * SPR, seed=2)
        amp = 5 * x0.std() * 6 ** (-np.arange(n_rec) / (n_rec - 1))
        x = x0 + (amp[:, None] * pulse()[None, :]).ravel()
        res = remove_record_locked(x, index0=0, samples_per_record=SPR, fs=FS, hp_cut=HP)
        lo, hi = res.peak_range
        assert hi / lo > 3
        for block in np.array_split(np.arange(n_rec), 6):
            seg = slice(block[0] * SPR, (block[-1] + 1) * SPR)
            assert record_mean_z(res.signal[seg], 0).max() < 5

    def test_a_second_pass_catches_what_loud_records_hid(self, monkeypatch):
        """Records whose pulse is strong enough to make them loud are left out
        of the first detection; a broad shoulder that is significant only in
        them is found once they are corrected and quiet."""
        import odas_tpw.rsi.record_locked as rl

        n_loud, n_rec = 150, 450
        x0 = noise(n_rec * SPR, seed=6)
        p = np.arange(SPR, dtype=float)
        shape = pulse() + 0.05 * np.exp(-0.5 * ((p - 250) / 40) ** 2)
        amp = np.where(np.arange(n_rec) < n_loud, 30.0, 2.0) * x0.std()
        x = x0 + (amp[:, None] * shape[None, :]).ravel()
        # Away from the step, which no +-16-record window can follow exactly.
        loud = slice(0, (n_loud - 2 * rl.LOCAL_HALF_WINDOW) * SPR)

        def residual(passes: int) -> float:
            monkeypatch.setattr(rl, "MAX_PASSES", passes)
            res = remove_record_locked(x, index0=0, samples_per_record=SPR, fs=FS, hp_cut=HP)
            return float(record_mean_z(res.signal[loud], 0).max())

        shipped = rl.MAX_PASSES
        assert residual(1) > 10  # the shoulder, left in the loud records
        assert residual(shipped) < 5

    def test_leave_one_out(self):
        """A record's own content never enters the waveform subtracted from it."""
        n_rec = 200
        x = noise(n_rec * SPR, seed=3) + 3 * np.tile(pulse(), n_rec)
        base = remove_record_locked(x, index0=0, samples_per_record=SPR, fs=FS, hp_cut=HP)
        bumped = x.copy()
        bumped[100 * SPR + 6] += 0.5  # one record, one significant sample
        res = remove_record_locked(bumped, index0=0, samples_per_record=SPR, fs=FS, hp_cut=HP)
        d = (res.signal - bumped) - (base.signal - x)
        # Its neighbors' corrections take ~1/32 of the bump; record 100's own
        # moves only through the high-pass's leak into adjacent records.
        assert abs(d[99 * SPR + 6]) > 0.5 / (2 * LOCAL_HALF_WINDOW) / 2
        assert abs(d[100 * SPR + 6]) < abs(d[99 * SPR + 6]) / 100

    def test_bad_and_nonfinite_records_are_left_out_but_corrected(self):
        n_rec = 200
        x = noise(n_rec * SPR, seed=4) + 3 * np.tile(pulse(), n_rec)
        bad = np.zeros(len(x), dtype=bool)
        bad[10 * SPR + 100] = True
        x[20 * SPR + 300] = np.nan
        ref = remove_record_locked(x, index0=0, samples_per_record=SPR, fs=FS, hp_cut=HP)
        res = remove_record_locked(
            x, index0=0, samples_per_record=SPR, fs=FS, hp_cut=HP, bad=bad
        )
        assert res.records < ref.records
        assert res.signal[10 * SPR + 6] != x[10 * SPR + 6]
        assert np.isnan(res.signal[20 * SPR + 300])
        assert res.signal[20 * SPR + 6] != x[20 * SPR + 6]

    def test_dead_channel(self):
        x = np.full(100 * SPR, 3.0)
        res = remove_record_locked(x, index0=0, samples_per_record=SPR, fs=FS, hp_cut=HP)
        assert res.samples == 0
        np.testing.assert_array_equal(res.signal, x)

    def test_does_not_modify_input(self):
        x = noise(100 * SPR, seed=5) + 3 * np.tile(pulse(), 100)
        before = x.copy()
        remove_record_locked(x, index0=0, samples_per_record=SPR, fs=FS, hp_cut=HP)
        np.testing.assert_array_equal(x, before)

    def test_phase_power_ratio(self):
        """~1 once a repeated pulse is gone; above 1 where its amplitude varies
        from record to record, which no mean removes; NaN when nothing was
        corrected."""
        n_rec = 300
        x0 = noise(n_rec * SPR, seed=8)
        kw = {"index0": 0, "samples_per_record": SPR, "fs": FS, "hp_cut": HP}
        assert np.isnan(remove_record_locked(x0, **kw).phase_power_ratio)
        fixed = remove_record_locked(x0 + 3 * x0.std() * np.tile(pulse(), n_rec), **kw)
        assert fixed.samples > 0
        assert 0.9 < fixed.phase_power_ratio < 1.1
        amp = 3 * x0.std() * (1 + np.random.default_rng(9).standard_normal(n_rec))
        varying = remove_record_locked(x0 + (amp[:, None] * pulse()[None, :]).ravel(), **kw)
        assert varying.samples > 0
        assert varying.phase_power_ratio > 1.5

    def test_window_constant_is_sane(self):
        assert 2 * LOCAL_HALF_WINDOW + 1 < MIN_RECORDS


# ---------------------------------------------------------------------------
# End to end on real VMP-250 shear
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def clean():
    if not SAMPLE.exists():
        pytest.skip("Test data not available")
    return load_channels(SAMPLE)


def _epsilon(data, **kwargs):
    res = _compute_epsilon(SAMPLE, _pre_loaded=copy.deepcopy(data), **kwargs)
    return res, np.concatenate([d["epsilon"].values for d in res], axis=1)


def _inject(data, k: float, seed: int = 7, at: tuple[float, ...] = (6.0, 81.5)):
    """k x the in-profile record rms, drifting 6x over each profile, 5% jitter,
    at the absolute file samples whose index modulo 512 is *at*."""
    offset0, spr = data["record_framing"]
    fs = data["fs_fast"]
    n = len(data["t_fast"])
    prep = prepare_profiles(copy.deepcopy(data), None, "auto", None)
    profiles, ratio = prep[0], prep[-2]
    rng = np.random.default_rng(seed)
    out = copy.deepcopy(data)
    shear = []
    for name, arr in data["shear"]:
        a = np.asarray(arr, dtype=float).copy()
        for s_slow, e_slow in profiles:
            s, e = s_slow * ratio, min((e_slow + 1) * ratio, n)
            k0 = (-(offset0 + s)) % spr
            nr = (e - s - k0) // spr
            stack = _hp_filter(a[s:e], fs, 0.5 * fs / 1024)[k0 : k0 + nr * spr]
            rr = float(np.median(np.sqrt(np.mean(stack.reshape(nr, spr) ** 2, axis=1))))
            idx = np.arange(s, e)
            rec = (offset0 + idx) // spr
            r = rec - rec[0]
            drift = 6.0 ** (-r / r.max())
            jitter = rng.lognormal(0.0, 0.05, r.max() + 1)[r]
            a[s:e] += k * rr * drift / drift.mean() * jitter * pulse(spr, at)[(offset0 + idx) % spr]
        shear.append((name, a))
    out["shear"] = shear
    return out


class TestEndToEnd:
    def test_framing_comes_from_the_header(self, clean):
        from odas_tpw.rsi.p_file import PFile

        pf = PFile(SAMPLE)
        assert pf.samples_per_record == 512
        assert len(pf.t_fast) == len(pf._record_headers) * pf.samples_per_record
        assert pf.fs_fast != 512.0  # so time is no stand-in for the index
        assert clean["record_framing"] == (0, 512)

    def test_clean_shear_is_bit_identical_to_off(self, clean, caplog):
        with caplog.at_level(logging.INFO, logger="odas_tpw.rsi.record_locked"):
            on, e_on = _epsilon(clean, record_locked_shear=True)
        off, e_off = _epsilon(clean)
        np.testing.assert_array_equal(e_on, e_off)
        for a, b in zip(on, off):
            for name in a.data_vars:
                np.testing.assert_array_equal(a[name].values, b[name].values)
            extra = set(a.attrs) - set(b.attrs)
            assert extra and all(k.startswith("record_locked_") for k in extra)
            assert a.attrs["record_locked_sh1_samples"] == 0
            assert a.attrs["record_locked_sh2_samples"] == 0
            assert a.attrs["record_locked_sh1_snr"] < 5
            assert not any(k.startswith("record_locked_") for k in b.attrs)
        assert "no significant record-locked interference" in caplog.text

    @pytest.mark.parametrize("k", [2.0, 5.0, 20.0])
    def test_epsilon_recovers_the_clean_value(self, clean, k, caplog):
        _, e_clean = _epsilon(clean)
        dirty = _inject(clean, k)
        _, e_off = _epsilon(dirty)
        with caplog.at_level(logging.INFO, logger="odas_tpw.rsi.record_locked"):
            res, e_on = _epsilon(dirty, record_locked_shear=True)
        ok = np.isfinite(e_clean) & np.isfinite(e_off) & np.isfinite(e_on)
        assert ok.sum() > 50
        bias_off = np.log10(e_off[ok] / e_clean[ok])
        bias_on = np.log10(e_on[ok] / e_clean[ok])
        # Uncorrected, the pulse biases epsilon (high below the despike
        # threshold, low above it, where the despiker cuts out ~60 ms).
        assert abs(np.median(bias_off)) > 0.03
        # Corrected: median within 0.02 dex, 90% of windows within 0.12.
        assert abs(np.median(bias_on)) < 0.02
        assert np.percentile(np.abs(bias_on), 90) < 0.12
        attrs = res[0].attrs
        for ch in ("sh1", "sh2"):
            assert 20 <= attrs[f"record_locked_{ch}_samples"] <= 80
            assert attrs[f"record_locked_{ch}_snr"] > 10
            lo, hi = attrs[f"record_locked_{ch}_peak_range"]
            assert hi / lo > 2  # the drift is seen
            # What the 5% record-to-record jitter leaves at the corrected
            # samples: nothing to speak of at 2x and 5x; at 20x it is ~1x the
            # record rms, and the ratio shows it.
            ratio = attrs[f"record_locked_{ch}_phase_power_ratio"]
            assert 0.8 < ratio < 1.3 if k < 10 else ratio > 1.2
        assert "record-locked interference removed" in caplog.text

    def test_profile_netcdf_route_matches_the_p_route(self, clean, tmp_path):
        """extract_profiles carries the framing, so epsilon from a profile
        NetCDF -- perturb's route -- is corrected like the direct .p route.
        (The two routes differ at the profile's end windows anyway, feature
        off; so each is compared with its own clean result.)"""
        import netCDF4

        from odas_tpw.rsi.p_file import PFile
        from odas_tpw.rsi.profile import extract_profiles

        dirty = _inject(clean, 5.0)
        pf = PFile(SAMPLE)
        for name, arr in dirty["shear"]:
            pf.channels[name] = arr
        paths = extract_profiles(pf, tmp_path / "dirty")
        clean_paths = extract_profiles(SAMPLE, tmp_path / "clean")
        assert len(paths) == len(clean_paths) == 1
        with netCDF4.Dataset(paths[0]) as ds:
            assert ds.samples_per_record_fast == 512
            offset = int(ds.record_offset_fast)
        assert offset != 0  # so an offset taken as 0 would show
        assert load_channels(paths[0])["record_framing"] == (offset, 512)
        nc_on = _compute_epsilon(paths[0], record_locked_shear=True)
        p_on, _ = _epsilon(dirty, record_locked_shear=True)
        for ch in ("sh1", "sh2"):
            assert nc_on[0].attrs[f"record_locked_{ch}_samples"] > 0
            assert nc_on[0].attrs[f"record_locked_{ch}_samples"] == (
                p_on[0].attrs[f"record_locked_{ch}_samples"]
            )
        e_nc = nc_on[0]["epsilon"].values
        e_nc_off = _compute_epsilon(paths[0])[0]["epsilon"].values
        e_nc_clean = _compute_epsilon(clean_paths[0])[0]["epsilon"].values
        ok = np.isfinite(e_nc) & np.isfinite(e_nc_off) & np.isfinite(e_nc_clean)
        assert abs(np.median(np.log10(e_nc_off[ok] / e_nc_clean[ok]))) > 0.03
        assert abs(np.median(np.log10(e_nc[ok] / e_nc_clean[ok]))) < 0.02

    @pytest.mark.parametrize("route", ["p", "rsi-tpw nc"])
    def test_the_correction_lands_on_the_injected_samples(self, clean, tmp_path, route):
        """Pulses written at known ABSOLUTE file samples (index modulo 512 =
        300, far from where an offset of 0 would put them) are found, through a
        profile NetCDF's framing attrs, at those samples. Each corrected
        sample's absolute index is recovered from t_fast * fs_fast, which does
        not use the framing at all."""
        from odas_tpw.rsi.convert import p_to_L1
        from odas_tpw.rsi.p_file import PFile
        from odas_tpw.rsi.profile import extract_profiles
        from odas_tpw.rsi.record_locked import remove_from_profile

        if route == "p":
            dirty = _inject(clean, 5.0, at=(300.0,))
            pf = PFile(SAMPLE)
            for name, arr in dirty["shear"]:
                pf.channels[name] = arr
            paths = extract_profiles(pf, tmp_path)
        else:
            # rsi-tpw nc writes the whole file with offset 0; rsi-tpw prof
            # carries it to each profile. Only the framing is checked on this
            # route: its profiles hold no 1-D P or shear for epsilon to read.
            nc = tmp_path / "full.nc"
            p_to_L1(SAMPLE, nc)
            paths = extract_profiles(nc, tmp_path / "prof")
        assert paths
        prof = load_channels(paths[0]) if route == "p" else None
        import netCDF4

        with netCDF4.Dataset(paths[0]) as ds:
            offset, spr = int(ds.record_offset_fast), int(ds.samples_per_record_fast)
            t0 = float(ds.variables["t_fast"][0])
            fs = float(ds.fs_fast)
        first = round(t0 * fs)  # absolute index of the profile's first sample
        assert spr == 512 and offset == first % 512 and offset != 0
        if prof is None:
            return
        target = [(n, np.array(a, dtype=float)) for n, a in prof["shear"]]
        attrs = remove_from_profile(
            prof,
            target,
            0,
            len(prof["t_fast"]),
            framing=prof["record_framing"],
            fs=prof["fs_fast"],
            hp_cut=0.5 * prof["fs_fast"] / 1024,
            context="test",
        )
        absolute = np.rint(np.asarray(prof["t_fast"]) * prof["fs_fast"]).astype(np.int64)
        for (name, before), (_, after) in zip(prof["shear"], target):
            changed = after != np.asarray(before, dtype=float)
            assert attrs[f"record_locked_{name}_samples"] > 0
            phases = np.unique(absolute[changed] % 512)
            assert 300 in phases
            assert np.all(np.abs(phases - 300) <= 25), phases

    def test_source_without_framing_is_refused(self, clean):
        data = copy.deepcopy(clean)
        del data["record_framing"]
        with pytest.raises(ValueError, match="record framing"):
            _compute_epsilon(SAMPLE, _pre_loaded=data, record_locked_shear=True)
        # ...but only when asked for.
        assert _compute_epsilon(SAMPLE, _pre_loaded=data)

    @pytest.mark.parametrize("bad", ["yes", "false", 1, 0, None])
    def test_only_a_bool_switches_it(self, bad):
        with pytest.raises(ValueError, match="record_locked_shear"):
            _compute_epsilon(SAMPLE, record_locked_shear=bad)


@pytest.mark.parametrize("front_end", ["rsi", "perturb"])
class TestConfig:
    def _cfg(self, front_end):
        if front_end == "rsi":
            from odas_tpw.rsi import config
        else:
            from odas_tpw.perturb import config  # type: ignore[no-redef]
        return config

    def test_off_by_default_and_in_the_template(self, front_end, tmp_path):
        cfg = self._cfg(front_end)
        assert cfg.DEFAULTS["epsilon"]["record_locked_shear"] is False
        template = cfg.generate_template(tmp_path / "c.yaml").read_text()
        assert "record_locked_shear: false" in template

    def test_off_leaves_the_signature_unchanged(self, front_end):
        cfg = self._cfg(front_end)
        assert "record_locked_shear" not in cfg.canonicalize("epsilon", {})
        assert cfg.compute_hash("epsilon", {"record_locked_shear": False}) == cfg.compute_hash(
            "epsilon", {}
        )
        assert cfg.compute_hash("epsilon", {"record_locked_shear": True}) != cfg.compute_hash(
            "epsilon", {}
        )

    @pytest.mark.parametrize("bad", ["yes", "'true'", "1", "0"])
    def test_non_bool_fails_at_load(self, front_end, tmp_path, bad):
        cfg = self._cfg(front_end)
        path = tmp_path / "c.yaml"
        path.write_text(f"epsilon:\n  record_locked_shear: {bad}\n")
        with pytest.raises(ValueError, match="record_locked_shear"):
            cfg.load_config(path)


class TestRsiCli:
    @pytest.fixture
    def run(self, tmp_path, monkeypatch):
        import sys

        from odas_tpw.rsi.cli import main

        if not SAMPLE.exists():
            pytest.skip("Test data not available")

        def _run(epsilon: dict, command: str) -> Path:
            import yaml

            cfg = tmp_path / f"{command}.yaml"
            cfg.write_text(yaml.safe_dump({"epsilon": epsilon}))
            out = tmp_path / f"{command}_out"
            argv = ["rsi-tpw", "-c", str(cfg), command, str(SAMPLE), "-o", str(out)]
            monkeypatch.setattr(sys, "argv", argv)
            main()
            return out

        return _run

    def test_eps_on_a_p_file(self, run):
        import xarray as xr

        out = run({"record_locked_shear": True}, "eps")
        files = sorted(out.rglob("*.nc"))
        assert files
        with xr.open_dataset(files[0]) as ds:
            assert ds.attrs["record_locked_samples_per_record"] == 512
            assert ds.attrs["record_locked_sh1_samples"] == 0  # clean shear

    @pytest.mark.parametrize(
        "epsilon", [{"record_locked_shear": True}, {"despike_half_width": 0.01}]
    )
    def test_pipeline_refuses_rather_than_ignores(self, run, epsilon, capsys):
        with pytest.raises(SystemExit) as exc:
            run(epsilon, "pipeline")
        assert exc.value.code == 2
        assert "rsi-tpw pipeline does not apply" in capsys.readouterr().err


class TestPerturbProcessFile:
    """perturb computes epsilon from its per-profile NetCDFs, so the framing
    must survive extract_profiles; chi on (the channels dict is shared with the
    chi stage, which must not see corrected shear written through)."""

    @staticmethod
    def _run(tmp_path, epsilon: dict):
        import warnings

        from odas_tpw.perturb.gps import GPSFixed
        from odas_tpw.perturb.pipeline import process_file

        if not SAMPLE.exists():
            pytest.skip("Test data not available")
        dirs = {k: tmp_path / k for k in ("profiles", "diss", "chi")}
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = process_file(
                SAMPLE, {"chi": {"enable": True}, "epsilon": epsilon}, GPSFixed(15.0, 145.0), dirs
            )
        assert not res.get("errors")
        return dirs

    def test_on_and_off(self, tmp_path):
        import xarray as xr

        on = self._run(tmp_path / "on", {"record_locked_shear": True})
        off = self._run(tmp_path / "off", {})
        for stage in ("profiles", "diss", "chi"):
            a_files = sorted(on[stage].glob("*.nc"))
            b_files = sorted(off[stage].glob("*.nc"))
            assert a_files and [f.name for f in a_files] == [f.name for f in b_files]
            for fa, fb in zip(a_files, b_files):
                with xr.open_dataset(fa) as a, xr.open_dataset(fb) as b:
                    for name in a.data_vars:  # clean shear: nothing corrected
                        np.testing.assert_array_equal(a[name].values, b[name].values)
                    if stage == "profiles":
                        assert a.attrs["samples_per_record_fast"] == 512
                    if stage == "diss":
                        assert a.attrs["record_locked_sh1_samples"] == 0
                        assert "record_locked_comment" not in b.attrs


class TestPerturbStageSignatures:
    """On re-versions what it changes -- diss, and chi through Method 1's
    epsilon -- and leaves the profiles (whose data it never touches) alone."""

    @staticmethod
    def _sig(stage: str, on: bool) -> str:
        from odas_tpw.perturb.config import compute_hash, stage_signature

        cfg = {"epsilon": {"record_locked_shear": True}} if on else {}
        section, params, upstream = stage_signature(stage, cfg)
        return compute_hash(section, params, upstream=upstream)

    @pytest.mark.parametrize(
        "stage", ["diss", "chi", "diss_binned", "chi_binned", "diss_combo", "chi_combo"]
    )
    def test_reversions(self, stage):
        assert self._sig(stage, True) != self._sig(stage, False)

    @pytest.mark.parametrize("stage", ["profiles", "profiles_binned", "combo", "ctd"])
    def test_does_not_reversion(self, stage):
        assert self._sig(stage, True) == self._sig(stage, False)
