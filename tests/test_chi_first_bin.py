# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""chi never uses FFT bin 1 (issue #191 step 2).

Every FFT segment is linearly detrended before its transform, which leaves the
bin-1 estimate off the spectrum at its own wavenumber by an amount that depends
on the spectral slope around it. These tests pin:

* the bias itself, through the package's own spectral path (csd_matrix_batch as
  l3_chi calls it): bin 1 reads low on a flat and on a red spectrum, while bins
  2-8 are unbiased;
* that chi computed from such estimated spectra matches chi computed from the
  exact spectrum when the band starts at bin 2, and does not when it starts at
  bin 1, on Method 1 and both Method-2 fits;
* that no chi path reads bin 1 at all: poisoning it leaves every output alone;
* that ``first_bin = 1`` reproduces the code before the change exactly (values
  pinned from main bb459b7) while the default does not;
* the ``chi.first_bin`` config key in both front ends: validated at load,
  hashed when not 1, recorded on the product.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from odas_tpw.chi.batchelor import KAPPA_T, batchelor_kB, kraichnan_grad
from odas_tpw.chi.chi import (
    FIRST_BIN,
    _chi_from_epsilon,
    _iterative_fit,
    _mle_fit_kB,
    _usable_bins,
    validate_first_bin,
)
from odas_tpw.chi.fp07 import fp07_tau, fp07_transfer
from odas_tpw.chi.l3_chi import L3ChiData
from odas_tpw.chi.l4_chi import process_l4_chi_epsilon, process_l4_chi_fit
from odas_tpw.scor160.io import L4Data
from odas_tpw.scor160.spectral import csd_matrix_batch

FS, W, NU, F_AA_EFF = 512.0, 0.7, 1.0e-6, 0.9 * 98.0


def _estimate(psd_per_hz, nfft: int, nlong: int = 2**20, seed: int = 191) -> np.ndarray:
    """Expected csd_matrix_batch spectrum [per Hz] of a Gaussian record with *psd_per_hz*.

    One long record is synthesized in the frequency domain, cut into
    ``4 * nfft`` dissipation windows and passed through csd_matrix_batch the way
    l3_chi calls it (nfft, half overlap, linear detrend); the window average is
    the estimator's expectation to within ~1% per bin.
    """
    rng = np.random.default_rng(seed)
    fj = np.arange(nlong // 2 + 1) * FS / nlong
    S = np.asarray(psd_per_hz(fj), dtype=np.float64)
    noise = rng.standard_normal(fj.size) + 1j * rng.standard_normal(fj.size)
    X = np.sqrt(S * FS * nlong / 4) * noise
    X[0] = X[-1] = 0.0
    x = np.fft.irfft(X, n=nlong)
    diss = 4 * nfft
    nw = nlong // diss
    C = csd_matrix_batch(
        x[: nw * diss].reshape(nw, diss, 1), None, nfft, FS, overlap=nfft // 2, detrend="linear"
    ).Cxy[:, :, 0, 0].real
    return C.mean(axis=0)


class TestBin1IsBiased:
    @pytest.mark.parametrize(
        ("shape", "bin1_below"),
        [
            # flat: the detrend takes part of bin 1's own variance (ln ~ -0.33)
            ("flat", -0.2),
            # falling as 1/f below bin 1 too: nothing leaks back in (ln ~ -0.6)
            ("red", -0.4),
        ],
    )
    def test_bin1_reads_low_and_bins_2_to_8_do_not(self, shape, bin1_below):
        nfft = 512
        f1 = FS / nfft
        if shape == "flat":
            psd = np.ones_like
        else:
            def psd(f):
                return f1 / np.maximum(f, f1 / 10)

        F = np.arange(nfft // 2 + 1) * FS / nfft
        lr = np.log(_estimate(psd, nfft)[1:9] / psd(F[1:9]))
        assert lr[0] < bin1_below
        assert np.all(np.abs(lr[1:]) < 0.1)


# --- chi from estimated vs exact spectra ---------------------------------------

_NFFT = 256
_EPS = _CHI = 1e-8
_KB = float(batchelor_kB(_EPS, NU))
_TAU = float(fp07_tau(W))
_F = np.arange(_NFFT // 2 + 1) * FS / _NFFT
_K = _F / W
_H2 = fp07_transfer(_F, _TAU)
# A low-wavenumber term 4x the Kraichnan level at bin 1, falling as 1/k (the
# issue #191 excess), and a white noise floor far below the signal.
_A = 4.0 * float(kraichnan_grad(_K[1:2], _KB, _CHI)[0]) * _K[1]
_KC = _K[1] / 10
_NOISE = 1e-4 * float(np.max(kraichnan_grad(_K[1:], _KB, _CHI) * _H2[1:]))


def _s_cpm(k, f):
    with np.errstate(divide="ignore"):
        model = kraichnan_grad(k, _KB, _CHI) + _A / np.maximum(k, _KC)
    return fp07_transfer(f, _TAU) * model + _NOISE


_FITS = {
    "method1": lambda S, fb: _chi_from_epsilon(
        S, _K, _EPS, NU, np.full_like(_K, _NOISE), _H2, _TAU, fp07_transfer, F_AA_EFF, W,
        "kraichnan", KAPPA_T, first_bin=fb,
    ),
    "iterative": lambda S, fb: _iterative_fit(
        S, _K, NU, np.full_like(_K, _NOISE), _H2, _TAU, fp07_transfer, F_AA_EFF, W,
        "kraichnan", KAPPA_T, first_bin=fb,
    ),
    "mle": lambda S, fb: _mle_fit_kB(
        S, _K, _CHI, NU, np.full_like(_K, _NOISE), _H2, _TAU, fp07_transfer, F_AA_EFF, W,
        "kraichnan", KAPPA_T, first_bin=fb,
    ),
}


@pytest.fixture(scope="module")
def spectra():
    def per_hz(f):
        s = _s_cpm(f / W, f) / W
        s[f == 0] = 0.0
        return s

    est = _estimate(per_hz, _NFFT) * W
    true = _s_cpm(_K, _F)
    est[0] = true[0] = 0.0
    return est, true


@pytest.mark.parametrize("name", sorted(_FITS))
class TestChiFromBin2FollowsTheSpectrum:
    """chi(estimated spectrum) vs chi(exact spectrum), by the SAME estimator.

    Measured at this geometry (fft_length 256, VMP speed, kB 135 cpm): with
    bin 1 in the band the estimate moves chi by ln -0.032 / +0.034 / -0.017
    (Method 1 / iterative / MLE); from bin 2 by at most 0.0011.
    """

    def test_bin1_is_the_biased_bin(self, spectra, name):
        est, true = spectra
        lr = np.log(est[1:6] / true[1:6])
        assert lr[0] < -0.3
        assert np.all(np.abs(lr[1:]) < 0.05)

    def test_band_from_bin2_tracks_the_exact_spectrum(self, spectra, name):
        est, true = spectra
        fit = _FITS[name]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d1 = np.log(fit(est, 1).chi / fit(true, 1).chi)
            d2 = np.log(fit(est, 2).chi / fit(true, 2).chi)
        assert abs(d2) < 0.01
        assert abs(d1) > 0.015
        assert abs(d2) < abs(d1) / 3


# --- no chi path reads bin 1 ------------------------------------------------------

FIELDS = ("chi", "kB", "K_max", "fom", "K_max_ratio", "var_resolved", "mad_shape")
L4_FIELDS = (*FIELDS, "epsilon_T", "chi_final", "chi_qc_fallback")


def _synth_l3(n: int = 16, seed: int = 191, nfft: int = 256):
    """Two probes, n windows of Kraichnan + a 1/k low-wavenumber excess + noise,
    chi-squared scatter. The fixture pinned against main bb459b7 below."""
    dof = 13.3
    kappa = 1.4e-7
    rng = np.random.default_rng(seed)
    F = np.arange(nfft // 2 + 1) * FS / nfft
    Ws = rng.uniform(0.55, 0.85, n)
    eps = 10 ** rng.uniform(-9.5, -7.5, n)
    chi = 10 ** rng.uniform(-9.5, -7.0, n)
    K = F[:, None] / Ws[None, :]
    tau0 = np.array([fp07_tau(w) for w in Ws])
    H2 = np.stack([fp07_transfer(F, t) for t in tau0])
    N = np.full(F.size, 3.4e-6)
    spec = np.empty((2, F.size, n))
    for j in range(n):
        kB = float(batchelor_kB(eps[j], 1.2e-6, kappa))
        B = kraichnan_grad(K[:, j], kB, chi[j], kappa_T=kappa)
        k1 = K[1, j]
        low = 4.0 * B[1] * k1 / np.maximum(K[:, j], k1 / 10)
        for p in range(2):
            spec[p, :, j] = ((B + low) * H2[j] + N) * rng.chisquare(dof, F.size) / dof
    spec[:, 0, :] = 0.0
    l3 = L3ChiData(
        time=np.arange(n, dtype=float) * 4.0, pres=np.linspace(10, 60, n), temp=np.full(n, 20.0),
        pspd_rel=Ws, section_number=np.ones(n), nu=np.full(n, 1.2e-6),
        kappa_T=np.full(n, kappa), kcyc=K, freq=F,
        gradt_spec=spec, noise_spec=np.tile(N[None, :, None], (2, 1, n)),
        H2=np.stack([H2, H2]), tau0=np.stack([tau0, tau0]),
        diff_gains=[0.94, 0.94], dof_spec=np.full(n, dof),
    )
    l4 = L4Data(
        time=l3.time.copy(), pres=l3.pres.copy(), pspd_rel=Ws.copy(),
        section_number=np.ones(n), epsi=np.stack([eps, eps]), epsi_final=eps.copy(),
        epsi_flags=np.zeros((2, n)), fom=np.ones((2, n)), mad=np.zeros((2, n)),
        kmax=np.full((2, n), 50.0), method=np.zeros((2, n)), var_resolved=np.ones((2, n)),
    )
    return l3, l4


_L4_RUNS = {
    "epsilon": lambda l3, l4, **kw: process_l4_chi_epsilon(l3, l4, **kw),
    "iterative": lambda l3, l4, **kw: process_l4_chi_fit(l3, fit_method="iterative", **kw),
    "mle": lambda l3, l4, **kw: process_l4_chi_fit(l3, fit_method="mle", **kw),
}


def _run(name, l3, l4, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _L4_RUNS[name](l3, l4, **kw)


def _poison(l3, b: int, value: float):
    """A copy of *l3* with bin *b* of every observed and noise spectrum set to *value*."""
    import copy

    out = copy.deepcopy(l3)
    out.gradt_spec[:, b, :] = value
    out.noise_spec[:, b, :] = value
    return out


@pytest.mark.parametrize("name", sorted(_L4_RUNS))
class TestNoChiPathReadsBin1:
    @pytest.mark.parametrize("value", [np.nan, 1e6, 0.0])
    def test_poisoned_bin1_changes_nothing(self, name, value):
        l3, l4 = _synth_l3()
        ref = _run(name, l3, l4)
        got = _run(name, _poison(l3, 1, value), l4)
        for f in L4_FIELDS:
            np.testing.assert_array_equal(getattr(got, f), getattr(ref, f), err_msg=f)

    def test_poisoned_bin2_does_change_it(self, name):
        """Not vacuous: bin 2 is in the band, so the same poison moves chi."""
        l3, l4 = _synth_l3()
        ref = _run(name, l3, l4)
        got = _run(name, _poison(l3, 2, 1e6), l4)
        assert not np.array_equal(got.chi, ref.chi, equal_nan=True)

    def test_bin1_is_read_with_first_bin_1(self, name):
        l3, l4 = _synth_l3()
        ref = _run(name, l3, l4, first_bin=1)
        got = _run(name, _poison(l3, 1, 1e6), l4, first_bin=1)
        assert not np.array_equal(got.chi, ref.chi, equal_nan=True)


@pytest.mark.parametrize("value", [np.nan, 1e6])
class TestWindowFunctionsNeverReadBin1:
    """The per-window estimators, on one window, poisoned in the data only."""

    def _window(self):
        l3, l4 = _synth_l3(n=1, seed=7)
        return (l3.gradt_spec[0, :, 0], l3.kcyc[:, 0], l3.noise_spec[0, :, 0], l3.H2[0, 0],
                float(l3.tau0[0, 0]), float(l3.pspd_rel[0]), float(l4.epsi_final[0]))

    def _calls(self, S, K, N, H2, tau, w, eps):
        kap = 1.4e-7
        return {
            "method1": _chi_from_epsilon(S, K, eps, 1.2e-6, N, H2, tau, fp07_transfer, F_AA_EFF, w,
                                         "kraichnan", kap),
            "iterative": _iterative_fit(S, K, 1.2e-6, N, H2, tau, fp07_transfer, F_AA_EFF, w,
                                        "kraichnan", kap),
            "mle": _mle_fit_kB(S, K, 1e-8, 1.2e-6, N, H2, tau, fp07_transfer, F_AA_EFF, w,
                               "kraichnan", kap),
        }

    def test_outputs_identical(self, value):
        S, K, N, H2, tau, w, eps = self._window()
        S2, N2 = S.copy(), N.copy()
        S2[1] = value
        N2[1] = value
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = self._calls(S, K, N, H2, tau, w, eps)
            got = self._calls(S2, K, N2, H2, tau, w, eps)
        for name in ref:
            assert np.isfinite(ref[name].chi), name
            for f in FIELDS:
                if hasattr(ref[name], f):
                    assert np.array_equal(getattr(got[name], f), getattr(ref[name], f),
                                          equal_nan=True), (name, f)


# --- first_bin = 1 is the code before the change -----------------------------------

# Per path and field: (count of finite positive values, sum of their ln), from
# main bb459b7 on _synth_l3() -- the code before chi.first_bin existed. The
# branch's first_bin = 1 reproduced all 27 arrays bit for bit on macOS/arm64;
# the relative tolerance only absorbs last-ulp libm differences across
# platforms (a band change moves these sums by 0.1-5).
_MAIN_BB459B7 = {
    "epsilon": {
        "chi": (32, -587.203715281443), "kB": (32, 142.4536778273769),
        "K_max": (32, 140.3352532374147), "fom": (32, 0.07348055072931292),
        "K_max_ratio": (32, -2.1184245899622223), "var_resolved": (32, -3.2096650744754713),
        "mad_shape": (32, -59.21008691967836), "epsilon_T": (32, -641.2229707451212),
        "chi_final": (16, -293.6328637724763),
    },
    "iterative": {
        "chi": (32, -593.9310936630748), "kB": (32, 137.31427934843558),
        "K_max": (32, 140.3352532374147), "fom": (32, 2.2377492586523005),
        "K_max_ratio": (32, 3.020973888979097), "var_resolved": (32, -1.8011798202199252),
        "mad_shape": (32, -59.772028427523566), "epsilon_T": (32, -661.7805646608865),
        "chi_final": (16, -296.884213725005),
    },
    "mle": {
        "chi": (32, -591.2132400542725), "kB": (32, 137.78560692106655),
        "K_max": (32, 140.3352532374147), "fom": (32, 0.08382580172069824),
        "K_max_ratio": (32, 2.549646316348139), "var_resolved": (32, -1.7854746915222885),
        "mad_shape": (32, -59.459969460517165), "epsilon_T": (32, -659.8952543703626),
        "chi_final": (16, -295.60662002713616),
    },
}


def _aggregates(res) -> dict[str, tuple[int, float]]:
    out = {}
    for f in _MAIN_BB459B7["epsilon"]:
        a = np.asarray(getattr(res, f), dtype=float)
        ok = np.isfinite(a) & (a > 0)
        out[f] = (int(ok.sum()), float(np.sum(np.log(a[ok]))))
    return out


@pytest.mark.parametrize("name", sorted(_L4_RUNS))
class TestFirstBinOneReproducesMain:
    def test_first_bin_1_is_main(self, name):
        l3, l4 = _synth_l3()
        got = _aggregates(_run(name, l3, l4, first_bin=1))
        for f, (n, s) in _MAIN_BB459B7[name].items():
            assert got[f][0] == n, f
            assert got[f][1] == pytest.approx(s, rel=1e-12, abs=1e-12), f

    def test_default_is_not_main(self, name):
        l3, l4 = _synth_l3()
        got = _aggregates(_run(name, l3, l4))
        assert got["chi"][1] != pytest.approx(_MAIN_BB459B7[name]["chi"][1], rel=1e-6)


# --- the helper and validation -------------------------------------------------------


class TestUsableBins:
    def test_production_grid_is_index_ge_first_bin(self):
        K = np.arange(129) * 2.0
        for fb in (1, 2, 3, 5):
            np.testing.assert_array_equal(_usable_bins(K, fb), np.arange(129) >= fb)

    def test_grid_without_dc_drops_the_same_bins(self):
        K = np.arange(1, 129) * 2.0  # starts at bin 1
        np.testing.assert_array_equal(_usable_bins(K, 2), np.arange(1, 129) >= 2)

    def test_first_bin_1_is_k_positive(self):
        K = np.linspace(0.0, 100.0, 50)
        np.testing.assert_array_equal(_usable_bins(K, 1), K > 0)

    def test_default_is_first_bin(self):
        assert FIRST_BIN == 2
        K = np.arange(65) * 1.0
        np.testing.assert_array_equal(_usable_bins(K), _usable_bins(K, FIRST_BIN))


class TestValidateFirstBin:
    @pytest.mark.parametrize("good", [1, 2, 7, np.int64(3)])
    def test_good(self, good):
        assert validate_first_bin(good) == int(good)
        assert type(validate_first_bin(good)) is int

    @pytest.mark.parametrize("bad", [0, -1, 2.0, True, False, "2", None])
    def test_bad(self, bad):
        with pytest.raises(ValueError, match="first_bin"):
            validate_first_bin(bad)

    @pytest.mark.parametrize("name", sorted(_L4_RUNS))
    def test_l4_entry_points_refuse_bin_0(self, name):
        l3, l4 = _synth_l3(n=2)
        with pytest.raises(ValueError, match="first_bin"):
            _L4_RUNS[name](l3, l4, first_bin=0)

    def test_rsi_pipeline_refuses_bin_0(self, tmp_path):
        from odas_tpw.rsi.pipeline import run_pipeline

        with pytest.raises(ValueError, match="chi_first_bin"):
            run_pipeline([], tmp_path, chi_first_bin=0)


# --- config: both front ends ----------------------------------------------------------


@pytest.mark.parametrize("front_end", ["rsi", "perturb"])
class TestConfig:
    def _cfg(self, front_end):
        if front_end == "rsi":
            from odas_tpw.rsi import config
        else:
            from odas_tpw.perturb import config  # type: ignore[no-redef]
        return config

    def test_in_defaults_and_template(self, front_end, tmp_path):
        cfg = self._cfg(front_end)
        assert cfg.DEFAULTS["chi"]["first_bin"] == FIRST_BIN
        path = cfg.generate_template(tmp_path / "config.yaml")
        assert "first_bin: 2" in path.read_text(encoding="utf-8")

    def test_hash(self, front_end):
        """Bin 1 (the code before the key) leaves the key out of the canonical
        view; the default and any other value are hashed."""
        cfg = self._cfg(front_end)
        assert '"first_bin":2' in cfg.canonicalize("chi", {})
        assert "first_bin" not in cfg.canonicalize("chi", {"first_bin": 1})
        h = {fb: cfg.compute_hash("chi", {"first_bin": fb}) for fb in (1, 2, 3)}
        assert len(set(h.values())) == 3
        assert cfg.compute_hash("chi", {}) == h[2]

    @pytest.mark.parametrize("bad", ["0", "-1", "1.5", "2.0", "true", "'2'"])
    def test_bad_value_fails_at_load(self, front_end, tmp_path, bad):
        cfg = self._cfg(front_end)
        path = tmp_path / "config.yaml"
        path.write_text(f"chi:\n  first_bin: {bad}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="first_bin"):
            cfg.load_config(path)

    @pytest.mark.parametrize("good", [1, 3])
    def test_good_value_loads(self, front_end, tmp_path, good):
        cfg = self._cfg(front_end)
        path = tmp_path / "config.yaml"
        path.write_text(f"chi:\n  first_bin: {good}\n", encoding="utf-8")
        assert cfg.load_config(path)["chi"]["first_bin"] == good


def test_perturb_chi_stages_are_versioned_by_first_bin():
    """chi and everything built on it get a new stage directory per value."""
    from odas_tpw.perturb.config import compute_hash, stage_signature

    for stage in ("chi", "chi_binned", "chi_combo"):
        hashes = {
            fb: compute_hash(*stage_signature(stage, {"chi": {"first_bin": fb}}))
            for fb in (1, 2, 3)
        }
        assert len(set(hashes.values())) == 3, stage
    # ...and diss, which never reads it, does not.
    assert len({compute_hash(*stage_signature("diss", {"chi": {"first_bin": fb}}))
                for fb in (1, 2)}) == 1


# --- the product records it -----------------------------------------------------------

P_FILE = Path(__file__).parent / "data" / "SN479_0006.p"


@pytest.mark.skipif(not P_FILE.exists(), reason="SN479 fixture missing")
def test_product_attr_written_only_when_not_bin_1():
    from odas_tpw.rsi.chi_io import _compute_chi

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new = _compute_chi(P_FILE, fft_length=512)[0]
        old = _compute_chi(P_FILE, fft_length=512, first_bin=1)[0]
    assert new.attrs["first_bin"] == 2
    assert "first_bin" not in old.attrs
    np.testing.assert_array_equal(new["spec_gradT"].values, old["spec_gradT"].values)
    assert not np.array_equal(new["chi"].values, old["chi"].values, equal_nan=True)
