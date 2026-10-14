"""Per-bead FP07 response equalization (chi/fp07_pair.py)."""

import copy
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from odas_tpw.chi import fp07_pair as P
from odas_tpw.chi.fp07 import fp07_tau
from odas_tpw.chi.l2_chi import L2ChiData
from odas_tpw.chi.l3_chi import process_l3_chi
from odas_tpw.chi.l4_chi import process_l4_chi_epsilon
from odas_tpw.scor160.io import L3Params, L4Data

FS = 512.0
N = 8192
T0 = datetime(2022, 3, 1, tzinfo=UTC).timestamp()


def _corr(ln_e1=0.0, ln_e2=0.0, f=None, **kw):
    f = np.geomspace(1.0, 60.0, 30) if f is None else f
    ln1 = np.broadcast_to(np.asarray(ln_e1, dtype=float), f.shape).copy()
    ln2 = np.broadcast_to(np.asarray(ln_e2, dtype=float), f.shape).copy()
    base = dict(instrument="SN194", channels=("T1_dT1", "T2_dT2"), serials=("T1592", "T2005"),
                t_start=T0, t_end=T0 + 30 * 86400, f_ref=f, ln_e=np.vstack([ln1, ln2]),
                f_lo=float(f[0]), f_hi=float(f[-1]), W_ref=1.0, tau_model="lueck")
    base.update(kw)
    return P.ResponseCorrection(**base)


# --- the correction object ---------------------------------------------------


class TestFactor:
    def test_holds_outside_the_band(self):
        f = np.geomspace(2.0, 40.0, 10)
        c = _corr(ln_e1=np.linspace(0.0, -1.0, 10), ln_e2=np.linspace(0.0, 1.0, 10), f=f)
        e = c.factor(0, np.array([0.5, 1.0, 2.0, 40.0, 80.0, 200.0]), 1.0)
        assert e[0] == e[1] == pytest.approx(1.0)  # held at the f_lo value
        assert e[3] == e[4] == e[5] == pytest.approx(np.exp(-1.0))  # held at f_hi, not tapered

    def test_speed_scaling_follows_the_tau_model(self):
        # tau ~ W^-1/2 (lueck): the response at f and 4 W_ref equals the
        # reference response at f * tau(4 W_ref)/tau(W_ref) = f/2.
        f = np.geomspace(1.0, 60.0, 40)
        c = _corr(ln_e1=-np.log1p(f / 10.0), f=f)
        F = np.array([4.0, 10.0, 30.0])
        assert np.allclose(c.factor(0, F, 4.0), c.factor(0, F / 2.0, 1.0))
        ratio = fp07_tau(4.0, model="lueck") / fp07_tau(1.0, model="lueck")
        assert ratio == pytest.approx(0.5)

    def test_speed_independent_model_ignores_speed(self):
        f = np.geomspace(1.0, 60.0, 40)
        c = _corr(ln_e1=-np.log1p(f / 10.0), f=f, tau_model="peterson")
        # peterson: tau ~ W^-0.32
        F = np.array([10.0])
        expect = c.factor(0, F * 2.0 ** -0.32, 1.0)
        assert np.allclose(c.factor(0, F, 2.0), expect)

    def test_window_vector_gives_one_row_per_window(self):
        c = _corr(ln_e1=-0.3)
        e = c.factor(0, np.linspace(1, 50, 7), np.array([0.5, 1.0, 2.0]))
        assert e.shape == (3, 7)


class TestPersistence:
    def test_round_trip_and_digest(self, tmp_path):
        c = _corr(ln_e1=-0.2, ln_e2=0.2)
        p = c.to_netcdf(tmp_path / "a.nc")
        d = P.ResponseCorrection.from_netcdf(p)
        assert d.digest() == c.digest()
        assert d.channels == c.channels and d.serials == c.serials
        np.testing.assert_allclose(d.ln_e, c.ln_e)

    def test_digest_ignores_file_name_and_creation_time(self, tmp_path):
        c = _corr(ln_e1=-0.2)
        d = copy.deepcopy(c)
        d.provenance = {"created": "another day"}
        assert c.digest() == d.digest()
        e = copy.deepcopy(c)
        e.ln_e = e.ln_e * 1.01
        assert e.digest() != c.digest()

    def test_hand_edited_file_is_refused(self, tmp_path):
        import xarray as xr

        p = _corr(ln_e1=-0.2).to_netcdf(tmp_path / "a.nc")
        with xr.open_dataset(p) as ds:
            ds.load()
        ds["ln_e"].values[0, 3] += 0.5
        ds.to_netcdf(tmp_path / "b.nc")
        with pytest.raises(ValueError, match="digest"):
            P.ResponseCorrection.from_netcdf(tmp_path / "b.nc")


class TestMatching:
    CH = ("T1_dT1", "T2_dT2")

    def test_by_instrument_both_serials_and_time(self):
        c = _corr()
        assert c.matches("SN194", self.CH, ["T1592", "T2005"], T0 + 86400)[0]
        ok, why = c.matches("SN194", self.CH, ["T1592", "T2007"], T0 + 86400)
        assert not ok and "T2007" in why
        assert not c.matches("SN194", self.CH, ["T1592", "T2005"], T0 - 86400)[0]
        ok, why = c.matches("SN428", self.CH, ["T1592", "T2005"], T0 + 86400)
        assert not ok and "SN428" in why

    def test_swapped_channels_do_not_match(self):
        # The same two beads on swapped channels: applying it would invert the
        # correction and double the disagreement.
        c = _corr()
        assert not c.matches("SN194", self.CH, ["T2005", "T1592"], T0 + 86400)[0]

    @pytest.mark.parametrize("placeholder", ["T", "?", ""])
    def test_placeholder_serials_fall_back_to_instrument_and_time(self, placeholder):
        c = _corr(serials=("T", "T"))
        assert c.matches("SN194", self.CH, [placeholder, "T9"], T0 + 1)[0]
        assert not c.matches("SN194", self.CH, [placeholder, "T9"], T0 - 1)[0]
        # a placeholder is never a wildcard for the INSTRUMENT (review: an
        # SN194 correction applied itself to SN479, whose beads read 'T')
        assert not c.matches("SN479", self.CH, [placeholder, "T9"], T0 + 1)[0]
        real = _corr()
        assert not real.matches("SN479", self.CH, ["T", "T"], T0 + 1)[0]

    def test_overlapping_corrections_are_an_error(self):
        a, b = _corr(), _corr(ln_e1=-0.1)
        with pytest.raises(ValueError, match="overlap"):
            P.select([a, b], "SN194", self.CH, ["T1592", "T2005"], T0 + 1)

    def test_exact_serials_beat_a_placeholder_wildcard(self):
        # The pair's serials changed real -> placeholder mid-deployment: the
        # groups' one-hour outer margins overlap, and a boundary file matches
        # both records. The record whose serials EQUAL the file's wins.
        real = _corr(t_start=T0, t_end=T0 + 10 * 3600)
        ph = _corr(serials=("T", "T"), t_start=T0 + 9 * 3600, t_end=T0 + 20 * 3600)
        t = T0 + 9.5 * 3600
        assert P.select([real, ph], "SN194", self.CH, ["T1592", "T2005"], t)[0] is real
        assert P.select([real, ph], "SN194", self.CH, ["T", "T"], t)[0] is ph
        # two records of the same rank are still an error
        with pytest.raises(ValueError, match="overlap"):
            P.select([real, ph], "SN194", self.CH, ["?", "?"], t)


# --- the chi integration -----------------------------------------------------


def _l2(speed=1.0):
    rng = np.random.default_rng(20261013)
    return L2ChiData(
        time=np.arange(N) / FS,
        pres=np.linspace(10.0, 80.0, N),
        temp=np.full(N, 15.0),
        temp_fast=np.full((2, N), 15.0) + rng.standard_normal((2, N)) * 1e-3,
        # S/N >= ~80 in every bin: the factor is then applied in full
        # (fp07_pair.signal_weighted), so these tests can demand exactness.
        gradt=rng.standard_normal((2, N)) * 1e-1,
        vib=rng.standard_normal((2, N)) * 1e-5,
        pspd_rel=np.full(N, speed),
        section_number=np.ones(N, dtype=int),
        diff_gains=[0.94, 0.94],
        fs_fast=FS,
    )


def _l3():
    params = L3Params(fft_length=256, diss_length=1024, overlap=512, fs_fast=FS, HP_cut=1.0,
                      goodman=False)
    return process_l3_chi(_l2(), params)


def _l4_eps(l3, eps=1e-7):
    n = l3.n_spectra
    return L4Data(
        time=l3.time.copy(), pres=l3.pres.copy(), pspd_rel=l3.pspd_rel.copy(),
        section_number=np.ones(n), epsi=np.full((2, n), eps), epsi_final=np.full(n, eps),
        epsi_flags=np.zeros((2, n)), fom=np.ones((2, n)), mad=np.zeros((2, n)),
        kmax=np.full((2, n), 50.0), method=np.zeros((2, n)), var_resolved=np.ones((2, n)),
    )


class TestChiEqualizesTheData:
    def test_constant_factor_scales_method1_chi_by_its_inverse(self):
        # e = c on probe 1 only (anchor T2): its spectrum AND noise are divided
        # by c, so the band is unchanged and chi scales by exactly 1/c.
        c = 0.8
        base = _l3()
        corr = _corr(ln_e1=np.log(c), ln_e2=0.0, anchor="T2")
        fixed = copy.deepcopy(base)
        P.apply_to_l3(fixed, corr, ["T1_dT1", "T2_dT2"])
        a = process_l4_chi_epsilon(base, _l4_eps(base))
        b = process_l4_chi_epsilon(fixed, _l4_eps(fixed))
        good = np.isfinite(a.chi[0]) & np.isfinite(b.chi[0])
        assert good.sum() >= 3
        np.testing.assert_allclose(b.chi[0][good] / a.chi[0][good], 1 / c, rtol=1e-6)
        np.testing.assert_array_equal(b.chi[1], a.chi[1])

    def test_pair_agrees_whatever_the_spectral_shape(self):
        # Probe 2 = probe 1 seen through a slower bead (spectrum AND noise times
        # e2(f)); the spectra are white, nothing like the Batchelor model.
        # Equalizing the data makes the two probes identical, so their chi must
        # agree EXACTLY. Multiplying the model by e2 instead would not: Method 1
        # applies the response through a model-weighted variance correction,
        # which mis-weights a relative correction when the data are not
        # model-shaped (on ARCTERX-2022 it left most of the K_max_ratio trend).
        base = _l3()
        f = np.geomspace(0.5, FS / 2, 60)
        corr = _corr(ln_e1=0.0, ln_e2=-np.log1p((f / 20.0) ** 2), f=f, anchor="T1")
        e2 = corr.factor(1, base.freq, 1.0)[:, np.newaxis]
        pair = copy.deepcopy(base)
        pair.gradt_spec[1] = base.gradt_spec[0] * e2
        pair.noise_spec[1] = base.noise_spec[0] * e2
        raw = process_l4_chi_epsilon(pair, _l4_eps(pair))
        fixed = copy.deepcopy(pair)
        P.apply_to_l3(fixed, corr, ["T1_dT1", "T2_dT2"])
        out = process_l4_chi_epsilon(fixed, _l4_eps(fixed))
        good = np.isfinite(out.chi).all(axis=0) & np.isfinite(raw.chi).all(axis=0)
        assert good.sum() >= 3
        # not vacuous: uncorrected, the slow bead reads well low
        assert np.all(raw.chi[1][good] / raw.chi[0][good] < 0.8)
        np.testing.assert_allclose(out.chi[1][good], out.chi[0][good], rtol=1e-9)
        np.testing.assert_array_equal(out.chi[0], raw.chi[0])

    def test_channels_mapped_by_name_not_index(self):
        base = _l3()
        corr = _corr(ln_e1=np.log(0.8), ln_e2=0.0, anchor="T2",
                     channels=("T2_dT2", "T1_dT1"), serials=("T2005", "T1592"))
        fixed = copy.deepcopy(base)
        P.apply_to_l3(fixed, corr, ["T1_dT1", "T2_dT2"])
        # corr's first channel is T2_dT2 -> probe index 1 gets the 0.8
        assert np.allclose(fixed.response_factor[1](base.freq, 1.0), 0.8)
        assert np.allclose(fixed.response_factor[0](base.freq, 1.0), 1.0)

    def test_applied_in_full_only_where_the_signal_dominates(self):
        e = np.full(5, 0.5)
        N = np.ones(5)
        for snr, expect in ((1.0, 1.0), (2.0, 1.0), (6.0, 0.5 ** 0.5), (10.0, 0.5), (500.0, 0.5)):
            got = P.signal_weighted(e, snr * N, N)
            np.testing.assert_allclose(got, expect, rtol=1e-12)

    def test_running_snr_is_not_diluted_at_the_edges(self):
        # a zero-padded mean read 4/7 of the true S/N in the first bin
        np.testing.assert_allclose(P._smoothed_snr(np.full(20, 7.0), np.ones(20)), 7.0)
        assert P._smoothed_snr(np.full(3, 7.0), np.ones(3)).shape == (3,)

    def test_noise_dominated_bins_are_not_rescaled(self):
        # A spectrum at the noise floor above 40 Hz: the factor must leave
        # those bins alone however large it is there.
        F = np.linspace(0, 256, 129)
        N = np.ones_like(F)
        S = np.where(F < 40, 100.0, 1.2)
        e = np.exp(-np.log1p((F / 20.0) ** 2))
        out = P.signal_weighted(e, S, N)
        assert np.all(out[F > 60] == 1.0)
        np.testing.assert_allclose(out[F < 20], e[F < 20])

    def test_l3_is_left_untouched(self):
        # The stored spectra must stay raw (fp07-pair refits and the guard read
        # them) and H2 stays the model: the factor is applied inside l4_chi.
        base = _l3()
        fixed = copy.deepcopy(base)
        P.apply_to_l3(fixed, _corr(ln_e1=np.log(1.5), ln_e2=np.log(1 / 1.5)), ["T1_dT1", "T2_dT2"])
        for name in ("gradt_spec", "noise_spec", "H2"):
            np.testing.assert_array_equal(getattr(fixed, name), getattr(base, name))


# --- estimator and guard -----------------------------------------------------


def _pair_windows(n, h1, h2, F, rng, snr=500.0, gain2=1.1):
    """(window, freq) spectra of two beads seeing one field."""
    S1, S2, N1, N2 = [], [], [], []
    for _ in range(n):
        g = F * np.exp(-F / 25.0) * np.exp(rng.normal(0, 1))
        nz = np.full(F.size, g.max() / snr)
        S1.append((g * h1 + nz) * rng.chisquare(9.5, F.size) / 9.5)
        S2.append((gain2 * g * h2 + nz) * rng.chisquare(9.5, F.size) / 9.5)
        N1.append(nz)
        N2.append(nz)
    return map(np.array, (S1, S2, N1, N2))


class TestGuardAndRatios:
    def test_log_ratio_recovers_relative_response(self):
        rng = np.random.default_rng(1)
        F = np.arange(1, 200) * 0.5
        h1 = 1 / (1 + (2 * np.pi * F * 0.015) ** 2)
        h2 = 1 / (1 + (2 * np.pi * F * 0.010) ** 2)
        S1, S2, N1, N2 = _pair_windows(2000, h1, h2, F, rng)
        lr = P.window_log_ratios(S1, S2, N1, N2)
        med = np.nanmedian(lr, axis=0)
        # Smoothed over +/-3 bins, as the estimator smooths: a single bin's
        # median of 2,000 log ratios at 9.5 DOF has a standard error ~0.02.
        sm = np.convolve(med, np.ones(7) / 7, mode="same")
        truth = np.convolve(np.log(h2 / h1), np.ones(7) / 7, mode="same")
        k = (F >= 5) & (F <= 40)
        np.testing.assert_allclose(sm[k] - np.log(1.1), truth[k], atol=0.03)

    def test_guard_flags_a_swapped_pair(self):
        rng = np.random.default_rng(2)
        F = np.arange(1, 200) * 0.5
        h1 = 1 / (1 + (2 * np.pi * F * 0.015) ** 2)
        h2 = 1 / (1 + (2 * np.pi * F * 0.008) ** 2)
        S1, S2, N1, N2 = _pair_windows(400, h1, h2, F, rng)
        lr = P.window_log_ratios(S1, S2, N1, N2)
        x = np.tile(F, (lr.shape[0], 1))
        mid = float(np.nanmedian(lr[:, (F >= 15) & (F <= 30)]) - np.nanmedian(lr[:, F <= 4]))
        right = _corr(mid_lnR=mid)
        wrong = _corr(mid_lnR=-mid)  # what a channel swap looks like
        assert P.guard(right, lr, x)[0]
        ok, delta = P.guard(wrong, lr, x)
        assert not ok and abs(delta) > P.GUARD_MAX


# --- the estimator, end to end on synthetic chi products ----------------------


def _write_source(path, sn1="T1592", sn2="T2005"):
    import netCDF4

    cfg = ("[instrument_info]\nsn = 194\n"
           f"[channel]\nname = T1\ntype = therm\nid = 4\nsn = {sn1}\n"
           f"[channel]\nname = T2\ntype = therm\nid = 5\nsn = {sn2}\n")
    with netCDF4.Dataset(path, "w") as ds:
        ds.configuration_string = cfg
    return str(path)


def _write_chi(path, source, t0, h1f, h2f, rng, n_win=40, W=1.0, gain2=1.1):
    import netCDF4

    F = np.arange(0, 129) * 0.5  # 0-64 Hz
    S = np.zeros((2, F.size, n_win))
    Nn = np.zeros_like(S)
    for j in range(n_win):
        g = (F + 0.1) * np.exp(-F / 25.0) * np.exp(rng.normal(0, 1))
        nz = np.full(F.size, g.max() / 500.0)
        S[0, :, j] = (g * h1f(F) + nz) * rng.chisquare(9.5, F.size) / 9.5
        S[1, :, j] = (gain2 * g * h2f(F) + nz) * rng.chisquare(9.5, F.size) / 9.5
        Nn[:, :, j] = nz
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("probe", 2)
        ds.createDimension("freq", F.size)
        ds.createDimension("time", n_win)
        v = ds.createVariable("spec_gradT", "f8", ("probe", "freq", "time"))
        v[:] = S
        v = ds.createVariable("spec_noise", "f8", ("probe", "freq", "time"))
        v[:] = Nn
        v = ds.createVariable("F", "f8", ("freq", "time"))
        v[:] = np.tile(F[:, None], (1, n_win))
        v = ds.createVariable("speed", "f8", ("time",))
        v[:] = np.full(n_win, W)
        pv = ds.createVariable("probe", str, ("probe",))
        pv[0], pv[1] = "T1_dT1", "T2_dT2"
        ds.instrument_sn = "194"
        ds.source = source
        ds.start_time = datetime.fromtimestamp(t0, UTC).isoformat()
        ds.fp07_model = "single_pole"
    return str(path)


def _sp(tau):
    return lambda F: 1 / (1 + (2 * np.pi * F * tau) ** 2)


class TestEstimate:
    def _deploy(self, tmp_path, segs, rng, gain2=1.1, t0=T0, step=3600.0):
        src = _write_source(tmp_path / "src.nc")
        paths, k = [], 0
        for n_files, h1f, h2f in segs:
            for _ in range(n_files):
                paths.append(_write_chi(tmp_path / f"c{k:03d}.nc", src, t0 + k * step, h1f, h2f,
                                        rng, gain2=gain2))
                k += 1
        return paths

    def test_recovers_the_relative_response(self, tmp_path):
        rng = np.random.default_rng(5)
        h1, h2 = _sp(0.015), _sp(0.010)
        res = P.estimate(self._deploy(tmp_path, [(16, h1, h2)], rng), n_boot=30)
        assert len(res.corrections) == 1, res.report + res.refused
        c = res.corrections[0]
        assert c.serials == ("T1592", "T2005") and c.anchor == "geomean"
        F = np.array([5.0, 10.0, 20.0, 30.0])
        lnR = np.log(c.factor(1, F, 1.0)) - np.log(c.factor(0, F, 1.0))
        ref = np.log(h2(np.array([1.5])) / h1(np.array([1.5])))
        truth = np.log(h2(F) / h1(F)) - ref
        np.testing.assert_allclose(lnR, truth, atol=0.06)
        # geometric-mean anchor: the factors multiply to one
        np.testing.assert_allclose(c.factor(0, F, 1.0) * c.factor(1, F, 1.0), 1.0, atol=1e-12)
        assert c.ln_G == pytest.approx(np.log(1.1) + ref[0], abs=0.05)

    def test_a_change_mid_deployment_splits_the_correction(self, tmp_path):
        rng = np.random.default_rng(6)
        segs = [(14, _sp(0.015), _sp(0.010)), (14, _sp(0.010), _sp(0.020))]
        paths = self._deploy(tmp_path, segs, rng)
        res = P.estimate(paths, n_boot=20)
        assert len(res.corrections) == 2, res.report + res.refused
        a, b = sorted(res.corrections, key=lambda c: c.t_start)
        assert a.t_end < b.t_start  # spans never overlap
        assert np.sign(a.mid_lnR) != np.sign(b.mid_lnR)
        # every file falls in exactly one span
        t = [T0 + k * 3600 for k in range(28)]
        assert all((a.t_start <= x <= a.t_end) != (b.t_start <= x <= b.t_end) for x in t)

    def test_a_small_clean_segment_keeps_its_band(self, tmp_path):
        # One lumpy bootstrap bin used to end the band at a few Hz and refuse
        # a clean 10-file segment (review: 7 of 30 seeds at 12 files).
        src = _write_source(tmp_path / "s.nc")
        for seed in range(1000, 1006):
            rng = np.random.default_rng(seed)
            paths = [_write_chi(tmp_path / f"s{seed}_{k:03d}.nc", src, T0 + k * 1800,
                                _sp(0.015), _sp(0.010), rng) for k in range(10)]
            (c,) = P.estimate(paths, n_boot=50).corrections
            assert c.applies and c.f_hi > 30, (seed, c.reason, c.f_hi)

    def test_guard_is_centered_on_the_files_it_was_fitted_from(self, tmp_path):
        # mid_lnR must be the per-file statistic the guard recomputes, so the
        # fit's own files sit at delta ~ 0 (review finding 5: a curve-based
        # mid_lnR put them +0.04..+0.08 off center).
        rng = np.random.default_rng(13)
        paths = self._deploy(tmp_path, [(16, _sp(0.020), _sp(0.010))], rng)
        (c,) = P.estimate(paths, n_boot=10).corrections
        deltas = []
        for path in paths:
            prod = P._read_product(path)
            lr = P.window_log_ratios(prod["S"][0].T, prod["S"][1].T, prod["N"][0].T,
                                     prod["N"][1].T)
            x = np.vstack([P.reduced_frequency(prod["F"][:, j], prod["W"][j], c.W_ref,
                                               c.tau_model) for j in range(prod["W"].size)])
            ok, d = P.guard(c, lr, x)
            assert ok
            deltas.append(d)
        assert abs(np.median(deltas)) < 1e-9

    def test_never_splits_profiles_that_share_a_start_time(self, tmp_path):
        # 30 profiles cut from ONE .p file share its start_time, and the pair
        # changes for the middle 20 of them. Splitting inside the group made a
        # segment whose span was empty (seen on ARCTERX-2022 SN194, 11 profiles
        # at one start); splits must fall between distinct times, so the whole
        # group lands in one span.
        rng = np.random.default_rng(12)
        src = _write_source(tmp_path / "src.nc")
        shared = T0 + 6 * 3600
        times = [T0 + k * 3600 for k in range(6)] + [shared] * 30 + \
                [T0 + (7 + k) * 3600 for k in range(6)]
        paths = []
        for k, t in enumerate(times):
            h2 = _sp(0.020) if 11 <= k < 31 else _sp(0.010)
            paths.append(_write_chi(tmp_path / f"c{k:03d}.nc", src, t, _sp(0.015), h2, rng))
        res = P.estimate(paths, n_boot=10, min_segment_files=5, min_files=5)
        assert res.corrections
        assert all(c.t_end > c.t_start for c in res.corrections)
        hits = [c for c in res.corrections if c.t_start <= shared <= c.t_end]
        assert len(hits) == 1 and hits[0].n_files >= 30

    def test_a_large_flat_gain_is_refused(self, tmp_path):
        rng = np.random.default_rng(7)
        paths = self._deploy(tmp_path, [(12, _sp(0.015), _sp(0.010))], rng, gain2=3.0)
        res = P.estimate(paths, n_boot=10)
        assert any("amplitude" in r for r in res.refused)
        # the span stays covered by a record that is never applied
        assert [c.applies for c in res.corrections] == [False]
        assert "amplitude" in res.corrections[0].reason
        assert P.output_name(res.corrections[0]).endswith("_notapplied.nc")
        res = P.estimate(paths, n_boot=10, allow_amplitude=True)
        assert res.corrections and "amplitude" in res.corrections[0].flags

    def test_unreadable_serials_are_reported_not_pooled(self, tmp_path):
        rng = np.random.default_rng(8)
        paths = self._deploy(tmp_path, [(10, _sp(0.015), _sp(0.010))], rng)
        bad = _write_chi(tmp_path / "bad.nc", str(tmp_path / "missing.nc"), T0, _sp(0.01),
                         _sp(0.01), rng)
        res = P.estimate([*paths, bad], n_boot=10)
        assert any(r.startswith("SKIPPED bad.nc") for r in res.report)
        assert res.report[0].startswith("11 chi files: 10 read, 1 skipped")

    def test_cli_writes_files_and_a_report(self, tmp_path):
        from odas_tpw.chi.fp07_pair_cli import main

        rng = np.random.default_rng(9)
        self._deploy(tmp_path, [(12, _sp(0.015), _sp(0.010))], rng)
        out = tmp_path / "out"
        assert main(["fit", str(tmp_path / "c*.nc"), "-o", str(out)]) == 0
        files = sorted(out.glob("fp07pair_*.nc"))
        assert len(files) == 1 and (out / "fp07pair_report.txt").exists()
        assert "fp07_response:" in (out / "fp07pair_report.txt").read_text()
        assert main(["show", str(files[0])]) == 0

    def test_cli_keeps_two_segments_from_one_day(self, tmp_path):
        # Both spans start on 2022-03-01 (files every 20 min from 06:00): a
        # date-only name wrote the second over the first (found on osu684).
        from odas_tpw.chi.fp07_pair_cli import main

        rng = np.random.default_rng(6)
        self._deploy(tmp_path, [(14, _sp(0.015), _sp(0.010)), (14, _sp(0.010), _sp(0.020))], rng,
                     t0=T0 + 6 * 3600, step=1200.0)
        out = tmp_path / "out"
        assert main(["fit", str(tmp_path / "c*.nc"), "-o", str(out)]) == 0
        files = sorted(out.glob("fp07pair_*.nc"))
        assert len(files) == 2
        spans = sorted(P.ResponseCorrection.from_netcdf(f).t_start for f in files)
        assert spans[0] < spans[1]


# --- perturb configuration ---------------------------------------------------


class TestPerturbConfig:
    def test_path_or_list_accepted(self):
        from odas_tpw.perturb.config import _validate_instruments

        _validate_instruments({"SN194": {"fp07_response": "a.nc"}})
        _validate_instruments({"SN194": {"fp07_response": ["a.nc", "b.nc"]}})

    @pytest.mark.parametrize("bad", [[], 3, [3], {"T1": "a.nc"}])
    def test_bad_values_raise(self, bad):
        from odas_tpw.perturb.config import _validate_instruments

        with pytest.raises(ValueError, match="fp07_response"):
            _validate_instruments({"SN194": {"fp07_response": bad}})

    def test_never_stacked_on_a_tau_scale(self):
        from odas_tpw.perturb.config import _validate_instruments

        with pytest.raises(ValueError, match="mutually exclusive"):
            _validate_instruments({"SN194": {"fp07_response": "a.nc",
                                             "fp07_tau_scale": {"T1": 0.3}}})

    def _config(self, tmp_path, rel="corr/a.nc", ln=-0.2):
        import yaml

        (tmp_path / "corr").mkdir(exist_ok=True)
        _corr(ln_e1=ln, ln_e2=-ln).to_netcdf(tmp_path / rel)
        cfg = tmp_path / "perturb.yaml"
        cfg.write_text(yaml.safe_dump(
            {"instruments": {"SN194": {"fp07_response": f"<CONFIG_DIR>/{rel}"}}}))
        return cfg

    def test_config_dir_token_resolves_against_the_config(self, tmp_path):
        from odas_tpw.perturb.config import load_config

        cfg = load_config(self._config(tmp_path))
        (p,) = cfg["instruments"]["SN194"]["fp07_response"]
        assert Path(p).is_absolute() and Path(p) == (tmp_path / "corr" / "a.nc").resolve()

    def test_plain_relative_paths_follow_the_working_directory(self, tmp_path, monkeypatch):
        # perturb's convention for every path: no token -> relative to the cwd
        import yaml

        from odas_tpw.perturb.config import load_config

        self._config(tmp_path)
        cfg = tmp_path / "sub" / "perturb.yaml"
        cfg.parent.mkdir()
        cfg.write_text(yaml.safe_dump({"instruments": {"SN194": {"fp07_response": "corr/a.nc"}}}))
        monkeypatch.chdir(tmp_path)
        (p,) = load_config(cfg)["instruments"]["SN194"]["fp07_response"]
        assert Path(p) == (tmp_path / "corr" / "a.nc").resolve()
        monkeypatch.chdir(cfg.parent)
        with pytest.raises(ValueError, match="does not exist"):
            load_config(cfg)

    def test_a_missing_correction_fails_at_load(self, tmp_path):
        import yaml

        from odas_tpw.perturb.config import load_config

        cfg = tmp_path / "perturb.yaml"
        cfg.write_text(yaml.safe_dump({"instruments": {"SN194": {"fp07_response": "nope.nc"}}}))
        with pytest.raises(ValueError, match="does not exist"):
            load_config(cfg)

    def test_hash_follows_content_not_location(self, tmp_path):
        import shutil

        from odas_tpw.perturb.config import canonical_instruments_for_hash

        a = tmp_path / "a.nc"
        _corr(ln_e1=-0.2, ln_e2=0.2).to_netcdf(a)
        moved = tmp_path / "elsewhere.nc"
        shutil.copy(a, moved)
        b = tmp_path / "b.nc"
        _corr(ln_e1=-0.25, ln_e2=0.25).to_netcdf(b)
        h = lambda p: canonical_instruments_for_hash({"S": {"fp07_response": [str(p)]}})  # noqa: E731
        assert h(a) == h(moved)
        assert h(a) != h(b)

    def test_epsilon_does_not_reversion(self, tmp_path):
        import json

        from odas_tpw.perturb.config import load_config, upstream_for

        cfg = load_config(self._config(tmp_path))
        plain = {k: v for k, v in cfg.items() if k != "instruments"}
        dump = lambda c, st: json.dumps(upstream_for(st, c), sort_keys=True, default=str)  # noqa: E731
        assert dump(cfg, "diss") == dump(plain, "diss")
        assert dump(cfg, "chi") != dump(plain, "chi")
        assert dump(cfg, "chi_combo") != dump(plain, "chi_combo")


class TestDiagReconstruction:
    """perturb-diag must plot the response the chi fit actually used."""

    def test_correction_matches_the_fit(self):
        from odas_tpw.perturb.diag.render import _fp07_h2

        base = _l3()
        corr = _corr(ln_e1=-np.log1p(np.geomspace(1, 60, 30) / 20), ln_e2=0.0, anchor="T2")
        fixed = copy.deepcopy(base)
        P.apply_to_l3(fixed, corr, ["T1_dT1", "T2_dT2"])
        j = 0
        W = float(fixed.pspd_rel[j])
        K = fixed.freq / W
        attrs = {"fp07_model": "single_pole", "fp07_response_applied": 1,
                 "fp07_response_W_ref": corr.W_ref, "fp07_response_tau_model": corr.tau_model}
        S, Nz = fixed.gradt_spec[0, :, j], fixed.noise_spec[0, :, j]
        h2 = _fp07_h2(K, W, attrs, probe="T1_dT1", response=(corr.f_ref, corr.ln_e[0]),
                      obs=S, noise=Nz)
        # chi fitted the model to raw/e (e weighted by S/N); drawn against the
        # raw data that is the model response times that same e
        e = P.signal_weighted(fixed.response_factor[0](fixed.freq, W), S, Nz)
        np.testing.assert_allclose(h2, fixed.H2_for(0, j) * e, rtol=1e-10)

    def test_tau_scale_attribute_is_honored(self):
        from odas_tpw.chi.fp07 import fp07_transfer
        from odas_tpw.perturb.diag.render import _fp07_h2

        K = np.linspace(1, 100, 50)
        attrs = {"fp07_model": "single_pole", "fp07_tau_scale_T2_dT2": 0.5}
        h2 = _fp07_h2(K, 0.8, attrs, probe="T2_dT2")
        np.testing.assert_allclose(h2, fp07_transfer(K * 0.8, 0.5 * fp07_tau(0.8, model="lueck")))
        # a probe without a recorded scale stays on the model
        np.testing.assert_allclose(_fp07_h2(K, 0.8, attrs, probe="T1_dT1"),
                                   fp07_transfer(K * 0.8, fp07_tau(0.8, model="lueck")))


def test_dc_bin_is_excluded_from_the_grid():
    """The DC bin of a detrended spectrum is meaningless; it must not land in
    the lowest grid bin (where it became the reference level on real data)."""
    rng = np.random.default_rng(11)
    F = np.arange(0, 65, dtype=float)  # 1 Hz grid: nothing real at 0.5 Hz
    n = 30
    S = np.ones((2, F.size, n)) + rng.random((2, F.size, n))
    prod = {"S": S, "N": np.full_like(S, 1e-6), "F": np.tile(F[:, None], (1, n)),
            "W": np.full(n, 1.0), "start": datetime.fromtimestamp(T0, UTC).isoformat()}
    grid = P._grid(64.0)
    fr = P._file_ratio(prod, 1.0, "lueck", grid, "x")
    assert fr.cnt[0] == 0  # grid[0] = 0.5 Hz: only the DC bin could have landed here
    assert fr.cnt[np.argmin(abs(grid - 1.0))] > 0


# --- rsi chi_io integration (real fixture) -------------------------------------

PROFILE_FILE = Path(__file__).parent / "data" / "SN479_0006.p"
P_T0 = datetime(2025, 1, 16, 1, 57, 57, tzinfo=UTC).timestamp()  # the fixture's start


def _p_corr(**kw):
    # SN479's config names its beads 'T' (placeholders): matching falls back to
    # the time span, which must contain the file's start.
    base = dict(instrument="SN479", serials=("T", "T"), t_start=P_T0 - 3600,
                t_end=P_T0 + 3600, W_ref=0.6)
    base.update(kw)
    return _corr(**base)


def _run(**kw):
    from odas_tpw.rsi.chi_io import _compute_chi

    out = _compute_chi(PROFILE_FILE, fft_length=1024, goodman=False, **kw)
    assert out
    return out[0]


@pytest.fixture(scope="module")
def p_baseline():
    if not PROFILE_FILE.exists():
        pytest.skip("SN479 fixture missing")
    plain = _run()
    # The file's own mid-band ratio, read back through the guard of a
    # do-nothing correction: guard_delta = observed - mid_lnR(=0).
    probe = _run(fp07_response=[_p_corr()])
    return plain, float(probe.attrs["fp07_response_guard_delta"])


@pytest.mark.skipif(not PROFILE_FILE.exists(), reason="SN479 fixture missing")
class TestChiIo:
    def test_applied_scales_chi_and_records_itself(self, p_baseline):
        plain, mid = p_baseline
        c = 0.8
        ds = _run(fp07_response=[_p_corr(ln_e1=np.log(c), ln_e2=0.0, anchor="T2", mid_lnR=mid)])
        assert ds.attrs["fp07_response_applied"] == 1
        assert "fp07_response_reason" not in ds.attrs
        a, b = plain["chi"].values, ds["chi"].values
        good = np.isfinite(a[0]) & np.isfinite(b[0])
        assert good.sum() >= 3
        # e = c < 1 on T1 raises its chi by up to 1/c: in full where the
        # window is signal-dominated, less where noise-floor bins carry weight
        r = b[0][good] / a[0][good]
        assert np.all(r >= 1 - 1e-12) and np.all(r <= 1 / c + 1e-9)
        assert np.median(r) > 1.1
        np.testing.assert_array_equal(b[1], a[1])
        np.testing.assert_allclose(ds["fp07_response_ln_e"].values[0], np.log(c))
        # the stored spectra stay raw
        np.testing.assert_array_equal(ds["spec_gradT"].values, plain["spec_gradT"].values)

    def test_guard_turns_a_disagreeing_correction_off(self, p_baseline):
        plain, mid = p_baseline
        ds = _run(fp07_response=[_p_corr(ln_e1=np.log(0.8), anchor="T2", mid_lnR=mid + 1.0)])
        assert ds.attrs["fp07_response_applied"] == 0
        assert ds.attrs["fp07_response_reason"].startswith("guard:")
        np.testing.assert_array_equal(ds["chi"].values, plain["chi"].values)

    def test_not_applied_record_runs_uncorrected_and_says_why(self, p_baseline):
        plain, _mid = p_baseline
        rec = _p_corr(ln_e1=np.log(0.8), flags=(P.NOT_APPLIED,),
                      provenance={"reason": "usable band ends at 22.3 Hz"})
        ds = _run(fp07_response=[rec])
        assert ds.attrs["fp07_response_applied"] == 0
        assert ds.attrs["fp07_response_reason"] == "refused at fit: usable band ends at 22.3 Hz"
        np.testing.assert_array_equal(ds["chi"].values, plain["chi"].values)

    def test_no_matching_correction_is_an_error(self):
        with pytest.raises(ValueError, match="no correction matches"):
            _run(fp07_response=[_p_corr(t_start=P_T0 + 86400, t_end=P_T0 + 2 * 86400)])

    def test_never_with_a_tau_scale(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            _run(fp07_response=[_p_corr()], fp07_tau_scale={"T1": 0.5})

    def test_another_instruments_correction_never_applies(self):
        # SN479's beads read 'T'; an SN194 correction covering the same time
        # used to apply itself through the placeholder (review finding 2).
        other = _p_corr(instrument="SN194", serials=("T1592", "T2005"))
        with pytest.raises(ValueError, match="no correction matches SN479"):
            _run(fp07_response=[other])


def test_an_unidentifiable_file_is_refused_not_wildcarded(tmp_path):
    # No configuration_string: the beads cannot be read. With fp07_response
    # configured that is an error, never a '?' that matches by time alone.
    import netCDF4

    from odas_tpw.rsi.chi_io import _select_pair_correction

    src = tmp_path / "noconfig.nc"
    with netCDF4.Dataset(src, "w") as ds:
        ds.title = "no configuration_string"
    data = {"metadata": {"start_time": "2022-03-01T01:00:00+00:00"}}
    with pytest.raises(ValueError, match="cannot be identified"):
        _select_pair_correction([_corr(serials=("T", "T"))], None, data, src,
                                ["T1_dT1", "T2_dT2"])


class TestRsiCli:
    """`rsi-tpw chi --fp07-response` must version its output directory."""

    def _dir(self, base, merged, responses):
        from odas_tpw.rsi.cli import _chi_response_params
        from odas_tpw.rsi.config import resolve_output_dir

        _kw, hp = _chi_response_params(merged, responses)
        return resolve_output_dir(base, "chi", "chi", hp).name

    def test_corrected_and_uncorrected_runs_never_share_a_directory(self, tmp_path):
        a = _corr(ln_e1=-0.2).to_netcdf(tmp_path / "a.nc")
        b = _corr(ln_e1=-0.3).to_netcdf(tmp_path / "b.nc")
        out = tmp_path / "out"
        plain = self._dir(out, {}, [])
        with_a = self._dir(out, {}, [str(a)])
        with_b = self._dir(out, {}, [str(b)])
        assert len({plain, with_a, with_b}) == 3
        # moving the file does not re-version; listing it again finds the dir
        moved = tmp_path / "elsewhere"
        moved.mkdir()
        (moved / "a.nc").write_bytes(a.read_bytes())
        assert self._dir(out, {}, [str(moved / "a.nc")]) == with_a

    def test_unset_leaves_every_existing_hash_alone(self):
        from odas_tpw.rsi.config import canonicalize

        assert "fp07_response" not in canonicalize("chi", {})

    @pytest.mark.skipif(not PROFILE_FILE.exists(), reason="SN479 fixture missing")
    def test_pipeline_refuses_rather_than_ignores_it(self, tmp_path, monkeypatch, capsys):
        # `rsi-tpw pipeline` computes chi on its own path; a [chi] fp07_response
        # in the config must stop it, not be silently dropped.
        import sys

        import yaml

        from odas_tpw.rsi.cli import main

        a = _corr(ln_e1=-0.2).to_netcdf(tmp_path / "a.nc")
        cfg = tmp_path / "rsi.yaml"
        cfg.write_text(yaml.safe_dump({"chi": {"fp07_response": [str(a)]}}))
        monkeypatch.setattr(sys, "argv", ["rsi-tpw", "-c", str(cfg), "pipeline",
                                          str(PROFILE_FILE), "-o", str(tmp_path / "out")])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 2
        assert "does not apply fp07_response" in capsys.readouterr().err  # not argparse's 2

    def test_config_file_value_is_hashed_too(self, tmp_path):
        from odas_tpw.rsi.cli import _chi_response_params

        a = _corr(ln_e1=-0.2).to_netcdf(tmp_path / "a.nc")
        kw, hp = _chi_response_params({"fp07_response": [str(a)]}, [])
        assert kw["fp07_response"] == [str(a.resolve())]
        assert hp["fp07_response"] == [_corr(ln_e1=-0.2).digest()]
