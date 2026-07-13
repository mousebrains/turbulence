# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""_reconstruct_spec_batch must bind the per-window kappa_T (#104 I2a-4).

The diagnostic overlay reconstructs the fitted gradient spectrum from the stored
per-window kB and chi. It must use the same per-window molecular thermal
diffusivity kappa_T the fit used; the bare grad_func would default to
KAPPA_T=1.4e-7 and, since amplitude ∝ 1/kappa_T, run ~6% high in warm water.
"""

import types

import numpy as np

from odas_tpw.chi.batchelor import KAPPA_T, batchelor_grad
from odas_tpw.chi.l3_chi import L3ChiData
from odas_tpw.rsi.chi_io import _reconstruct_spec_batch


def _l3(kappa_window, n_wave=8, n_spec=1) -> L3ChiData:
    kcyc = np.tile(np.linspace(1.0, 100.0, n_wave)[:, None], (1, n_spec))
    return L3ChiData(
        time=np.arange(n_spec, dtype=float),
        pres=np.zeros(n_spec),
        temp=np.full(n_spec, 20.0),
        pspd_rel=np.full(n_spec, 1.0),
        section_number=np.ones(n_spec, dtype=int),
        nu=np.full(n_spec, 1.3e-6),
        kappa_T=np.full(n_spec, kappa_window),
        kcyc=kcyc,
        freq=np.linspace(1.0, 98.0, n_wave),
        gradt_spec=np.zeros((1, n_wave, n_spec)),
        noise_spec=np.zeros((1, n_wave, n_spec)),
        H2=np.ones((n_spec, n_wave)),
        tau0=np.full(n_spec, 0.007),
    )


def test_binds_per_window_kappa():
    kappa_window = 1.49e-7  # warm/tropical, distinctly above the 1.4e-7 default
    assert kappa_window != KAPPA_T
    l3 = _l3(kappa_window)
    kB, chi = 50.0, 1e-7
    l4 = types.SimpleNamespace(kB=np.array([[kB]]), chi=np.array([[chi]]))

    out = _reconstruct_spec_batch(l3, l4, batchelor_grad)

    expected = batchelor_grad(l3.kcyc[:, 0], kB, chi, kappa_T=kappa_window)
    np.testing.assert_allclose(out[0, :, 0], expected, rtol=1e-12)
    # And it must DIFFER from the pre-fix default-kappa reconstruction.
    default = batchelor_grad(l3.kcyc[:, 0], kB, chi)  # kappa_T=KAPPA_T
    assert not np.allclose(out[0, :, 0], default)
    # Amplitude ∝ 1/kappa_T: warmer kappa => LOWER overlay than the default.
    assert np.nanmax(out[0, :, 0]) < np.nanmax(default)


def test_nonfinite_kB_left_nan():
    l3 = _l3(1.49e-7)
    l4 = types.SimpleNamespace(kB=np.array([[np.nan]]), chi=np.array([[1e-7]]))
    out = _reconstruct_spec_batch(l3, l4, batchelor_grad)
    assert np.all(np.isnan(out[0, :, 0]))


def test_fallback_to_default_when_kappa_not_per_window():
    """A degenerate kappa_T (size != n_spec) falls back to the bare grad_func."""
    l3 = _l3(1.49e-7, n_spec=2)
    l3.kappa_T = np.array([])  # not per-window sized
    kB, chi = 50.0, 1e-7
    l4 = types.SimpleNamespace(kB=np.full((1, 2), kB), chi=np.full((1, 2), chi))
    out = _reconstruct_spec_batch(l3, l4, batchelor_grad)
    expected = batchelor_grad(l3.kcyc[:, 0], kB, chi)  # default kappa
    np.testing.assert_allclose(out[0, :, 0], expected, rtol=1e-12)
