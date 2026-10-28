# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""chi ``mad_shape``: the per-bin SHAPE statistic beside the integrated ``fom``.

``fom`` integrates observed and model variance over the fit band, so a
mid-band deficit cancels against excesses at the ends: on CASPER-West a model
wrong by ~2x bin by bin scored fom = 1.001 (issue #191). ``mad_shape`` is the
mean absolute deviation of log10(observed/model) about its own median, which
cannot cancel. These tests pin:

* a spectrum equal to the model gives ~0, on all three chi paths;
* a model off by a structured factor of 2, built so that fom stays at 1, gives
  ~0.3 dex -- the case fom cannot see;
* a constant level error is invisible to it (that is fom's job);
* with chi-squared scatter and a perfect model it reads the documented floor
  (mad_shape_floor), not 0;
* it uses fom's band and fom's model exactly: the contiguous band (not the
  valid-bin subset), the noise term included, and on the iterative fit the
  fom band rather than the chi integration band;
* it is NaN exactly where fom is NaN, and kept like fom where a guard drops chi;
* it reaches L4ChiData and both chi NetCDF writers.
"""

from __future__ import annotations

import types
import warnings

import numpy as np
import pytest

from odas_tpw.chi.batchelor import KAPPA_T, batchelor_kB, kraichnan_grad
from odas_tpw.chi.chi import (
    _chi_from_epsilon,
    _iterative_fit,
    _mle_fit_kB,
    _shape_mad,
    mad_shape_floor,
)
from odas_tpw.chi.fp07 import fp07_tau, fp07_transfer

FS, NFFT, SPEED, NU, EPS, CHI = 512.0, 1024, 0.7, 1e-6, 1e-8, 1e-8
F = np.fft.rfftfreq(NFFT, 1 / FS)
K = F / SPEED
TAU = fp07_tau(SPEED)
H2 = fp07_transfer(F, TAU)
KB = float(batchelor_kB(EPS, NU))
# End the band at 1.2 kB, where the signal still dwarfs the (tiny) noise, so
# every bin in (0, K_AA] is in the fit band and the band is known exactly.
K_AA = 1.2 * KB
F_AA = K_AA * SPEED
SIGNAL = kraichnan_grad(K, KB, CHI) * H2
NOISE = np.full_like(K, SIGNAL[1:].max() * 1e-6)
BAND = (K > 0) & (K <= K_AA)


def _structured_by_two(signal: np.ndarray) -> np.ndarray:
    """The signal x2 in the outer quarters of the band and /2 in the middle half,
    rescaled so its integral over the band is unchanged -- the issue #191 shape
    (excess at both ends, deficit mid-band) with fom held at 1."""
    idx = np.flatnonzero(BAND)
    q = idx.size // 4
    g = np.ones_like(signal)
    g[idx[:q]] = 2.0
    g[idx[-q:]] = 2.0
    g[idx[q:-q]] = 0.5
    c = np.trapezoid(signal[BAND], K[BAND]) / np.trapezoid((signal * g)[BAND], K[BAND])
    return signal * g * c


def _m1(obs, dof=0.0):
    return _chi_from_epsilon(
        obs, K, EPS, NU, NOISE, H2, TAU, fp07_transfer, F_AA, SPEED, "kraichnan", dof=dof
    )


def _iter(obs):
    return _iterative_fit(obs, K, NU, NOISE, H2, TAU, fp07_transfer, F_AA, SPEED, "kraichnan")


def _mle(obs):
    return _mle_fit_kB(obs, K, CHI, NU, NOISE, H2, TAU, fp07_transfer, F_AA, SPEED, "kraichnan")


FITTERS = {"method1": _m1, "iterative": _iter, "mle": _mle}


class TestShapeMadHelper:
    def test_identical_spectra_give_zero(self):
        s = np.geomspace(1e-3, 1e2, 50)
        assert _shape_mad(s, s) == 0.0

    def test_constant_level_error_is_invisible(self):
        """A model off by a constant factor is a LEVEL error, which fom reports;
        the shape statistic must not."""
        rng = np.random.default_rng(0)
        obs = np.geomspace(1e-3, 1e2, 50) * rng.chisquare(10, 50) / 10
        model = np.geomspace(1e-3, 1e2, 50)
        base = _shape_mad(obs, model)
        for a in (0.1, 0.5, 2.0, 37.0):
            assert _shape_mad(a * obs, model) == pytest.approx(base, rel=1e-12, abs=1e-15)

    def test_two_level_ratio_reads_half_its_spread(self):
        """Half the bins x2, half /2: |r - median| = log10(2) in every bin."""
        model = np.ones(40)
        obs = np.r_[np.full(20, 2.0), np.full(20, 0.5)]
        assert _shape_mad(obs, model) == pytest.approx(np.log10(2.0))

    def test_median_center_is_the_best_level(self):
        """It is the epsilon-style mad (about 0) minimized over constant level
        shifts, so it never exceeds the uncentered mad."""
        rng = np.random.default_rng(1)
        r = rng.normal(0.1, 0.3, 61)
        obs, model = 10.0**r, np.ones_like(r)
        uncentered = float(np.mean(np.abs(r)))
        best = min(np.mean(np.abs(r - c)) for c in np.linspace(-1, 1, 4001))
        assert _shape_mad(obs, model) <= uncentered
        assert _shape_mad(obs, model) == pytest.approx(best, abs=1e-3)

    def test_unusable_bins_are_dropped(self):
        model = np.ones(6)
        obs = np.array([1.0, 1.0, 1.0, 0.0, -1.0, np.nan])
        assert _shape_mad(obs, model) == 0.0  # three usable bins, all on the model
        # A zero model bin is an infinite ratio: dropped, leaving two -> NaN.
        assert np.isnan(_shape_mad(np.ones(3), np.array([1.0, 0.0, 1.0])))

    def test_fewer_than_three_usable_bins_is_nan(self):
        assert np.isnan(_shape_mad(np.ones(2), np.ones(2)))
        assert np.isnan(_shape_mad(np.array([1.0, 1.0, 0.0]), np.ones(3)))
        assert np.isnan(_shape_mad(np.array([]), np.array([])))


@pytest.mark.parametrize("name", sorted(FITTERS))
class TestThroughTheFitters:
    def test_model_equal_spectrum_gives_about_zero(self, name):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = FITTERS[name](SIGNAL + NOISE)
        assert res.fom == pytest.approx(1.0, abs=0.02)
        # Method 1 fixes kB, so the model is the spectrum up to a constant the
        # centering removes; Method 2 fits kB on a grid and lands within ~2%.
        assert 0.0 <= res.mad_shape < (1e-6 if name == "method1" else 0.02)

    def test_structured_factor_of_two_is_visible_at_fom_one(self, name):
        """The issue #191 case: fom says perfect, the shape says 0.3 dex."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = FITTERS[name](_structured_by_two(SIGNAL) + NOISE)
        assert res.fom == pytest.approx(1.0, abs=0.01)
        assert res.mad_shape == pytest.approx(np.log10(2.0), abs=0.02)

    def test_non_detection_is_nan_like_fom(self, name):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = FITTERS[name](NOISE.copy())
        assert np.isnan(res.fom)
        assert np.isnan(res.mad_shape)


class TestChiSquaredFloor:
    def test_perfect_model_reads_the_documented_floor_not_zero(self):
        """chi-squared scatter alone gives E|log10(chi2_d/d) - median| = 0.138 dex
        at d = 13.3 (documented in _shape_mad and the product attrs)."""
        d = 13.3
        rng = np.random.default_rng(191)
        vals = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(40):
                obs = (SIGNAL + NOISE) * rng.chisquare(d, size=K.size) / d
                vals.append(_m1(obs, dof=d).mad_shape)
        assert np.mean(vals) == pytest.approx(0.1384, abs=0.006)


class TestFloorHelper:
    def test_matches_the_documented_table(self):
        for d, v in ((6.0, 0.214), (9.5, 0.166), (13.3, 0.138), (19.0, 0.115), (26.6, 0.096)):
            assert mad_shape_floor(d) == pytest.approx(v, abs=6e-4)

    def test_fewer_dof_means_a_higher_floor(self):
        """A BBL window with half the segments must not be judged by the long
        window's floor."""
        assert mad_shape_floor(1.9 * 2) > mad_shape_floor(1.9 * 5) > mad_shape_floor(1.9 * 7)

    def test_matches_monte_carlo(self):
        rng = np.random.default_rng(5)
        d = 7.6
        r = np.log10(rng.chisquare(d, 400_000) / d)
        assert mad_shape_floor(d) == pytest.approx(np.mean(np.abs(r - np.median(r))), abs=2e-3)

    def test_invalid_dof_is_nan(self):
        assert np.isnan(mad_shape_floor(0.0))
        assert np.isnan(mad_shape_floor(float("nan")))


# A band built to tell the right bins and model from three plausible wrong ones:
# noise crosses the signal at 0.6 kB, so the upper band is noise-dominated (a
# model without the noise term is far off there); one bin at 1.1 kB pokes above
# 2x noise, so the contiguous band spans a long hole in the valid mask (the
# valid-bin subset is a different set); and the lowest bins carry a 10x excess
# below the iterative fit's chi integration limit k_l (its chi band differs
# from its fom band exactly there).
_IC = int(np.argmin(np.abs(K - 0.6 * KB)))
NOISE_HI = np.full_like(K, SIGNAL[_IC])
_K_L = 3 * 0.04 * KB * np.sqrt(KAPPA_T / NU)
OBS_HOLE = np.where((K > 0) & (K < _K_L), 10.0 * SIGNAL, SIGNAL) + NOISE_HI
_ISP = int(np.argmin(np.abs(K - 1.1 * KB)))
OBS_HOLE[_ISP] = 3.0 * NOISE_HI[_ISP]

HOLE_FITTERS = {
    "method1": lambda: _chi_from_epsilon(
        OBS_HOLE, K, EPS, NU, NOISE_HI, H2, TAU, fp07_transfer, F_AA, SPEED, "kraichnan"
    ),
    "iterative": lambda: _iterative_fit(
        OBS_HOLE, K, NU, NOISE_HI, H2, TAU, fp07_transfer, F_AA, SPEED, "kraichnan"
    ),
    "mle": lambda: _mle_fit_kB(
        OBS_HOLE, K, CHI, NU, NOISE_HI, H2, TAU, fp07_transfer, F_AA, SPEED, "kraichnan"
    ),
}


@pytest.mark.parametrize("name", sorted(HOLE_FITTERS))
class TestUsesFomsBandAndModel:
    def test_fixture_has_a_hole_and_a_noise_dominated_upper_band(self, name):
        valid = (OBS_HOLE > 2 * NOISE_HI) & (K > 0) & (K <= K_AA)
        idx = np.flatnonzero(valid)
        assert idx[-1] == _ISP
        assert (idx[-1] - idx[0] + 1) - idx.size > 50  # interior hole
        assert np.all(SIGNAL[_IC + 1 : _ISP] < NOISE_HI[_IC + 1 : _ISP])

    def test_value_is_the_contiguous_fom_band_against_the_noisy_model(self, name):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = HOLE_FITTERS[name]()
        assert np.isfinite(res.fom)
        # fom's band on every path: first to last bin above 2x noise, up to K_max
        # (the iterative fit's k_u), on the ORIGINAL grid.
        valid = (OBS_HOLE > 2 * NOISE_HI) & (K > 0) & (res.K_max >= K)
        idx = np.flatnonzero(valid)
        band = slice(idx[0], idx[-1] + 1)
        model = res.spec_batch * H2 + NOISE_HI
        right = _shape_mad(OBS_HOLE[band], model[band])
        assert res.mad_shape == pytest.approx(right, rel=1e-12)

        # The fixture separates each wrong variant by far more than rounding.
        subset = _shape_mad(OBS_HOLE[valid], model[valid])
        no_noise = _shape_mad(OBS_HOLE[band], (res.spec_batch * H2)[band])
        assert abs(right - subset) > 0.015
        assert no_noise > right + 0.2
        if name == "iterative":
            k_l = max(K[1], 3 * 0.04 * res.kB * np.sqrt(KAPPA_T / NU))
            chi_band = (k_l <= K) & (res.K_max >= K)
            assert abs(right - _shape_mad(OBS_HOLE[chi_band], model[chi_band])) > 0.005


class TestNaNMatchesFom:
    def test_nan_pattern_is_fom_nan_pattern(self):
        """Across signal levels from pure noise to strong, including windows the
        detection gate rejects, mad_shape is finite exactly where fom is."""
        rng = np.random.default_rng(7)
        d = 13.3
        n_finite = n_nan = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for amp in np.geomspace(1e-9, 1.0, 25):
                obs = (amp * SIGNAL + NOISE) * rng.chisquare(d, size=K.size) / d
                for fit in FITTERS.values():
                    res = fit(obs)
                    assert np.isnan(res.fom) == np.isnan(res.mad_shape), (amp, res)
                    n_finite += bool(np.isfinite(res.fom))
                    n_nan += bool(np.isnan(res.fom))
        assert n_finite > 0 and n_nan > 0  # both branches exercised

    def test_too_few_points_is_nan(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _chi_from_epsilon(
                SIGNAL[:3] + NOISE[:3], K[:3], EPS, NU, NOISE[:3], H2[:3], TAU,
                fp07_transfer, F_AA, SPEED, "kraichnan",
            )
        assert np.isnan(res.fom) and np.isnan(res.mad_shape)


def _l3(n_spec=2, n_gradt=2):
    from odas_tpw.chi.l3_chi import L3ChiData

    obs = _structured_by_two(SIGNAL) + NOISE
    return L3ChiData(
        time=np.arange(n_spec, dtype=float),
        pres=np.full(n_spec, 10.0),
        temp=np.full(n_spec, 10.0),
        pspd_rel=np.full(n_spec, SPEED),
        section_number=np.ones(n_spec),
        nu=np.full(n_spec, NU),
        kappa_T=np.full(n_spec, 1.4e-7),
        kcyc=np.tile(K[:, None], (1, n_spec)),
        freq=F,
        gradt_spec=np.tile(obs[None, :, None], (n_gradt, 1, n_spec)),
        noise_spec=np.tile(NOISE[None, :, None], (n_gradt, 1, n_spec)),
        H2=np.tile(H2[None, :], (n_spec, 1)),
        tau0=np.full(n_spec, TAU),
    )


class TestCarriedIntoProducts:
    def test_l4_fit_and_epsilon_paths_carry_it(self):
        from odas_tpw.chi.l4_chi import process_l4_chi_epsilon, process_l4_chi_fit

        l3 = _l3()
        l4_diss = types.SimpleNamespace(epsi_final=np.full(2, EPS), time=l3.time.copy())
        # process_l4_chi_* multiply f_AA by 0.9 internally.
        f_AA = F_AA / 0.9
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outs = [
                process_l4_chi_epsilon(l3, l4_diss, f_AA=f_AA),
                process_l4_chi_fit(l3, f_AA=f_AA),
                process_l4_chi_fit(l3, fit_method="mle", f_AA=f_AA),
            ]
        for out in outs:
            assert out.mad_shape is not None and out.mad_shape.shape == out.fom.shape
            assert np.isfinite(out.mad_shape).all()
            assert out.mad_shape == pytest.approx(np.log10(2.0), abs=0.02)
            assert out.fom == pytest.approx(1.0, abs=0.01)

    def test_kept_like_fom_where_a_guard_drops_chi(self):
        """The bad-buffer guard NaNs chi but keeps fom for audit; mad_shape
        follows fom, not chi."""
        from odas_tpw.chi.l4_chi import _process_l4_chi

        def chi_func(j, ci, *_a):
            return (1e-8, 1e-9, 50.0, 40.0, 1.0, 0.9, 0.95, 0.123)

        l3 = _l3()
        l3.bad_fraction = np.zeros((2, 2))
        l3.interp_fraction = np.zeros((2, 2))
        l3.bad_fraction[:, 1] = 0.01
        out = _process_l4_chi(l3, chi_func, "epsilon", 98.0)
        assert np.isnan(out.chi[:, 1]).all() and np.isfinite(out.chi[:, 0]).all()
        assert np.all(out.fom == 1.0)
        assert np.all(out.mad_shape == 0.123)

    def test_rsi_chi_dataset_has_the_variable(self):
        from odas_tpw.chi.batchelor import batchelor_grad
        from odas_tpw.rsi.chi_io import _build_chi_ds_from_pipeline

        l3 = _l3(n_spec=1, n_gradt=1)
        l4 = types.SimpleNamespace(
            chi=np.full((1, 1), 1e-8),
            epsilon_T=np.full((1, 1), EPS),
            kB=np.full((1, 1), KB),
            K_max=np.full((1, 1), K_AA),
            fom=np.ones((1, 1)),
            mad_shape=np.full((1, 1), 0.25),
            K_max_ratio=np.full((1, 1), 1.2),
            var_resolved=np.full((1, 1), 0.9),
            method="epsilon",
        )
        kw = dict(
            fft_length=NFFT, diss_length=4 * NFFT, overlap=2 * NFFT, fs_fast=FS,
            fp07_model="single_pole", spectrum_model="kraichnan", fit_method="iterative",
            f_AA=98.0, grad_func=batchelor_grad,
        )
        ds = _build_chi_ds_from_pipeline(l3, l4, ["T1_dT1"], **kw)
        assert ds["mad_shape"].dims == ds["fom"].dims
        assert float(ds["mad_shape"].values[0, 0]) == 0.25
        assert ds["mad_shape"].attrs["units"] == "1"
        assert "SHAPE" in ds["mad_shape"].attrs["comment"]
        # An L4-like object from before the field existed still builds, as NaN.
        del l4.mad_shape
        ds_old = _build_chi_ds_from_pipeline(l3, l4, ["T1_dT1"], **kw)
        assert np.isnan(ds_old["mad_shape"].values).all()

    def test_rsi_pipeline_l4_chi_file_has_the_variable(self, tmp_path):
        import xarray as xr

        from odas_tpw.chi.l4_chi import process_l4_chi_fit
        from odas_tpw.rsi.pipeline import _write_l4_chi

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            l4 = process_l4_chi_fit(_l3(), f_AA=F_AA / 0.9)
        path = tmp_path / "L4_chi_fit.nc"
        _write_l4_chi(l4, path, "2025-01-01T00:00:00")
        with xr.open_dataset(path) as ds:
            assert ds["mad_shape"].dims == ("probe", "time")
            np.testing.assert_array_equal(ds["mad_shape"].values, l4.mad_shape)
