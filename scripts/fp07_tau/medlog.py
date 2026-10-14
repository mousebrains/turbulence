# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""Robust T1/T2 ratio from per-window spectra: median over windows of ln(S2/S1).

For two chi-squared spectral estimates with equal DOF, the median of their log
ratio is unbiased for the log of the expected ratio, and the median ignores
the few very energetic windows that dominate a plain sum (the source of a
spurious step at 30-35 Hz on ARCTERX-22 SN194). Each window's bins are gated
on a running mean of S/N over neighboring bins (both probes), not on each
bin's own noisy value, which limits the selection bias of gating on the
quantity being estimated.
"""
from __future__ import annotations

import numpy as np


def smoothed_snr(S, N, k=3):
    """Running mean of S/N over +/-k bins along the last axis."""
    r = np.where(N > 0, S / N, 0.0)
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    return np.apply_along_axis(lambda v: np.convolve(v, ker, mode="same"), -1, r)


def ratio_curve(S1, S2, N1, N2, snr_min=10.0, min_windows=30, k=3):
    """Median over windows of ln((S2-N2)/(S1-N1)) per frequency bin.

    S*, N*: (window, freq). Returns (lnR, n_used) per bin; NaN where fewer
    than *min_windows* windows pass the gate.
    """
    D1, D2 = S1 - N1, S2 - N2
    ok = (smoothed_snr(S1, N1, k) >= snr_min) & (smoothed_snr(S2, N2, k) >= snr_min) & (D1 > 0) & (D2 > 0)
    lr = np.full(S1.shape[1], np.nan)
    n = ok.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.log(np.where(ok, D2 / D1, np.nan))
    for i in range(S1.shape[1]):
        if n[i] >= min_windows:
            lr[i] = np.nanmedian(L[:, i])
    return lr, n


if __name__ == "__main__":
    # Synthetic check: per-window spectra with a known pair of single poles.
    import sys
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import ratio as R
    import ratio_synth as RS

    F = RS.F
    rng = np.random.default_rng(9)
    for taus, snr_peak in [((0.015, 0.010), 300), ((0.015, 0.010), 60), ((0.020, 0.013), 300), ((0.025, 0.010), 100)]:
        S1l, S2l, N1l, N2l = [], [], [], []
        for _ in range(3000):
            W = np.exp(rng.normal(0, 0.05)); kB = np.exp(rng.normal(np.log(120), 0.6))
            x = np.maximum(F / W / kB, 1e-6); g = RS.shape(x, rng); lvl = np.exp(rng.normal(0, 1.2))
            nz = np.full(F.size, lvl * np.max(g) / snr_peak) * np.exp(rng.normal(0, 1.0))  # windows differ in SNR
            t1, t2 = taus[0] * W ** -0.5, taus[1] * W ** -0.5
            a = (lvl * g * R.h2(F, t1, 1) + nz) * rng.chisquare(RS.DOF, F.size) / RS.DOF
            b = (1.1 * lvl * g * R.h2(F, t2, 1) + nz) * rng.chisquare(RS.DOF, F.size) / RS.DOF
            S1l.append(a); S2l.append(b); N1l.append(nz); N2l.append(nz)
        S1, S2, N1, N2 = map(np.array, (S1l, S2l, N1l, N2l))
        lr, n = ratio_curve(S1, S2, N1, N2)
        for fhi in (25.0, 60.0):
            m = np.isfinite(lr) & (F >= 1) & (fhi >= F)
            G, a1, a2, rms = R.fit_ratio(F[m], np.exp(lr[m]), 1)
            print(f"truth {taus[0]*1e3:.0f}/{taus[1]*1e3:.0f} ms, peak S/N {snr_peak}: fit 1-{fhi:.0f} Hz -> {a1*1e3:5.1f}/{a2*1e3:5.1f} ms")
