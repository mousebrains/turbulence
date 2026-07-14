# Jul-2026, Claude and Pat Welch, pat@mousebrains.com
"""Tests for the Method-2 applicability guard in ``chi.l4_chi`` (#104 U3-C3).

The guard drops spectral-fit ("fit") chi/epsilon_T windows whose fitted
Batchelor wavenumber ``kB`` reaches past ``_CHI_KB_KAA_MAX * K_AA`` — proof the
rolloff is unresolved and both estimates are biased low. It must never fire for
Method 1 ("epsilon"), where kB is fixed from the shear epsilon rather than fit.

The guard is exercised in isolation from the fitters by injecting a ``chi_func``
that returns controlled ``kB`` values per window, so the boundary and the
method gating are tested deterministically.
"""

import logging

import numpy as np

from odas_tpw.chi.l3_chi import L3ChiData
from odas_tpw.chi.l4_chi import _CHI_KB_KAA_MAX, _process_l4_chi

F_AA = 98.0
F_AA_EFF = 0.9 * F_AA  # matches _process_l4_chi internally
SPEED = 1.0
K_AA = F_AA_EFF / SPEED  # anti-alias wavenumber for SPEED (cpm)

# Sentinel chi/eps the injected fitter reports for every window; the guard NaNs
# them only where kB is under-resolved.
CHI_VAL = 1e-6
EPS_VAL = 1e-8


def _make_l3(n_spec: int, speed: float = SPEED, n_wave: int = 4, n_freq: int = 4) -> L3ChiData:
    """Minimal L3ChiData sized for _process_l4_chi (1 gradient channel)."""
    return L3ChiData(
        time=np.arange(n_spec, dtype=float),
        pres=np.zeros(n_spec),
        temp=np.full(n_spec, 10.0),
        pspd_rel=np.full(n_spec, speed),
        section_number=np.ones(n_spec, dtype=int),
        nu=np.full(n_spec, 1.3e-6),
        kappa_T=np.full(n_spec, 1.4e-7),
        kcyc=np.tile(np.linspace(1.0, 100.0, n_wave)[:, None], (1, n_spec)),
        freq=np.linspace(1.0, F_AA, n_freq),
        gradt_spec=np.zeros((1, n_wave, n_spec)),
        noise_spec=np.zeros((1, n_wave, n_spec)),
        H2=np.ones((n_spec, n_freq)),
        tau0=np.full(n_spec, 0.007),
    )


def _chi_func_for(kB_by_window):
    """A chi_func returning fixed chi/eps and a per-window kB (keyed by j)."""

    def _chi_func(j, _ci, _spec, _noise, _K, _W, _nu, _kappa, _tau0, _H2, _h2, _f_AA_eff):
        # (chi, eps, kB, K_max, fom, K_max_ratio, var_resolved) — fom/Kmr chosen
        # to pass QC; var_resolved=1.0 (fully resolved) is inert for these guards.
        return CHI_VAL, EPS_VAL, float(kB_by_window[j]), 50.0, 1.0, 1.0, 1.0

    return _chi_func


class TestU3C3Guard:
    def test_fit_drops_under_resolved_keeps_resolved(self):
        """kB above 1.5*K_AA -> chi & epsilon_T NaN; a resolved window kept."""
        l3 = _make_l3(2)
        kB = {0: 2.0 * K_AA, 1: 0.5 * K_AA}  # window 0 under-resolved, 1 fine
        l4 = _process_l4_chi(l3, _chi_func_for(kB), "fit", F_AA)

        assert np.isnan(l4.chi[0, 0])
        assert np.isnan(l4.epsilon_T[0, 0])
        assert l4.chi[0, 1] == CHI_VAL
        assert l4.epsilon_T[0, 1] == EPS_VAL

    def test_method1_epsilon_never_dropped(self):
        """The guard is fit-only: an identical kB under Method 1 is untouched."""
        l3 = _make_l3(1)
        kB = {0: 2.0 * K_AA}  # would be dropped under "fit"
        l4 = _process_l4_chi(l3, _chi_func_for(kB), "epsilon", F_AA)

        assert l4.chi[0, 0] == CHI_VAL
        assert l4.epsilon_T[0, 0] == EPS_VAL

    def test_threshold_boundary(self):
        """Just below 1.5*K_AA is kept; just above is dropped."""
        l3 = _make_l3(2)
        kB = {0: (_CHI_KB_KAA_MAX - 0.1) * K_AA, 1: (_CHI_KB_KAA_MAX + 0.1) * K_AA}
        l4 = _process_l4_chi(l3, _chi_func_for(kB), "fit", F_AA)

        assert l4.chi[0, 0] == CHI_VAL  # below threshold -> kept
        assert np.isnan(l4.chi[0, 1])  # above threshold -> dropped

    def test_nonfinite_kB_not_dropped(self):
        """A NaN kB (fit failed) is left as-is by the guard, not spuriously cut
        (it is already NaN chi/eps or handled elsewhere)."""
        l3 = _make_l3(1)
        kB = {0: np.nan}
        l4 = _process_l4_chi(l3, _chi_func_for(kB), "fit", F_AA)
        # kB NaN -> under_resolved False -> chi/eps keep the fitter's values.
        assert l4.chi[0, 0] == CHI_VAL

    def test_drop_is_logged(self, caplog):
        l3 = _make_l3(1)
        kB = {0: 2.0 * K_AA}
        with caplog.at_level(logging.WARNING, logger="odas_tpw.chi.l4_chi"):
            _process_l4_chi(l3, _chi_func_for(kB), "fit", F_AA)
        assert "under-resolved" in caplog.text

    def test_zero_speed_window_not_dropped(self):
        """A degenerate window (pspd_rel=0 -> K_AA=inf under the guard's
        errstate) must not be spuriously dropped: kB > 1.5*inf is False, so the
        window is kept and defers to the existing fom/K_max_ratio QC. Pins the
        safe direction of the divide-by-zero path."""
        l3 = _make_l3(1, speed=0.0)
        kB = {0: 500.0}  # large kB, but K_AA is inf so the guard cannot fire
        l4 = _process_l4_chi(l3, _chi_func_for(kB), "fit", F_AA)
        assert l4.chi[0, 0] == CHI_VAL
