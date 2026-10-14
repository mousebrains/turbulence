# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""Synthetic check of the spectral-ratio tau estimators (ratio.py).

Two probes see one turbulent temperature field. Each window draws its own
kB, chi and W; the gradient spectrum's SHAPE is deliberately non-universal
(a low-k excess whose size varies window to window, and a mid-band dip), to
show the ratio does not care. Each probe gets its own response, gain, noise,
and optionally narrowband vibration lines. Spectra carry chi-squared
estimation noise at the real DOF.

Two estimators:
  T1/T2  : <S2>/<S1> = G |H2|^2/|H1|^2            -> tau1, tau2
  T/uC   : <S_T>/<S_C> = G |H_T|^2 / |H_C|^2      -> tau_T  (uC response flat
           or a known fast pole), with an optional SALINITY leak: conductivity
           variance that continues past the temperature Batchelor rolloff.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import ratio as R

FS, NFFT, DOF = 512.0, 1024, 9.5
F = np.arange(NFFT // 2 + 1) * FS / NFFT


def shape(x, rng, excess=True):
    """Kraichnan gradient shape with a per-window low-k excess and mid dip."""
    g = x * np.exp(-np.sqrt(6 * 5.26) * x)
    if excess:
        amp = np.exp(rng.normal(np.log(3.0), 0.7))  # 3x at x=0.03, varies
        g *= 1 + amp * (0.03 / np.maximum(x, 1e-4)) ** 1.5
        g *= 1 - 0.4 * np.exp(-0.5 * (np.log(x / 0.3) / 0.4) ** 2)
    return g


def make_windows(n, W_med, rng, tau=(0.010, 0.010), poles=1, gain=(1.0, 1.15),
                 noise_snr=200.0, vib=False, salinity=0.0, uC=False):
    """Average spectra of probe A and probe B over n windows.

    Probe B is either a second FP07 (tau[1]) or, with uC=True, a fast
    reference (flat response) whose spectrum includes a salinity leak:
    conductivity variance extending to x ~ 10 (salinity Batchelor scale) with
    relative size `salinity` at x = 1.
    """
    SA = np.zeros_like(F)
    SB = np.zeros_like(F)
    for _ in range(n):
        W = W_med * np.exp(rng.normal(0, 0.05))
        kB = np.exp(rng.normal(np.log(150), 0.5))  # cpm
        K = F / W
        x = np.maximum(K / kB, 1e-6)
        g = shape(x, rng)
        lvl = np.exp(rng.normal(0, 1.0))
        tA, tB = tau[0] * (W / W_med) ** -0.5, tau[1] * (W / W_med) ** -0.5
        nA = lvl * np.max(g) / noise_snr
        trueA = gain[0] * lvl * g * R.h2(F, tA, poles) + nA
        if uC:
            gs = g + salinity * lvl * np.max(g) * np.exp(-np.sqrt(6 * 5.26) * x / 10) * x / (1 + x)
            trueB = gain[1] * lvl * gs + nA
        else:
            trueB = gain[1] * lvl * g * R.h2(F, tB, poles) + nA
        if vib:  # probe-specific narrowband lines (a cable/strumming peak)
            for f0, a in ((17.0, 3.0), (33.0, 2.0)):
                trueA += a * nA * np.exp(-0.5 * ((F - f0) / 0.6) ** 2)
            trueB += 2.5 * nA * np.exp(-0.5 * ((F - 23.0) / 0.6) ** 2)
        SA += trueA * rng.chisquare(DOF, F.size) / DOF - nA  # noise model subtracted
        SB += trueB * rng.chisquare(DOF, F.size) / DOF - nA
    return SA / n, SB / n


def fit(SA, SB, f_lo=1.0, f_hi=60.0, poles=1, ref_flat=False, robust_median=0):
    m = (f_lo <= F) & (f_hi >= F) & (SA > 0) & (SB > 0)
    f, rat = F[m], SB[m] / SA[m]
    if robust_median:  # running median across frequency rejects narrow lines
        k = robust_median
        rat = np.array([np.median(rat[max(0, i - k): i + k + 1]) for i in range(rat.size)])
    if ref_flat:
        # <S_T>/<S_C>: invert so the model is G * |H_T|^2 with H_C = 1
        from scipy.optimize import least_squares
        lr = np.log(SA[m] / SB[m]) if not robust_median else np.log(1 / rat)

        def res(p):
            return lr - (p[0] + np.log(R.h2(f, np.exp(p[1]), poles)))

        best = min((least_squares(res, [0.0, np.log(t)]) for t in (0.003, 0.01, 0.03)),
                   key=lambda r: r.cost)
        return np.exp(best.x[1]), None
    G, t1, t2, _ = R.fit_ratio(f, rat, poles)
    return t1, t2


if __name__ == "__main__" and sys.argv[1:] != ["analyse"]:
    rng = np.random.default_rng(7)
    n = 3000
    print("T1/T2 ratio: true (tau1, tau2) at W -> recovered  [single pole, 1-60 Hz]")
    for W, taus in [(1.0, (0.010, 0.010)), (1.0, (0.012, 0.010)), (1.0, (0.015, 0.010)),
                    (1.0, (0.025, 0.010)), (1.0, (0.006, 0.004)), (0.3, (0.020, 0.015)),
                    (0.3, (0.030, 0.012))]:
        SA, SB = make_windows(n, W, rng, tau=taus)
        t1, t2 = fit(SA, SB)
        print(f"  W {W:.1f}  true {taus[0]*1e3:5.1f}/{taus[1]*1e3:5.1f} ms -> {t1*1e3:6.2f}/{t2*1e3:6.2f} ms")
    print("T1/T2 with narrowband vibration lines (raw, then 7-bin running median):")
    SA, SB = make_windows(n, 1.0, rng, tau=(0.015, 0.010), vib=True)
    for k in (0, 3):
        t1, t2 = fit(SA, SB, robust_median=k)
        print(f"  median half-width {k}: 15.0/10.0 -> {t1*1e3:6.2f}/{t2*1e3:6.2f} ms")
    print("T1/T2 low SNR (peak S/N 20):")
    SA, SB = make_windows(n, 1.0, rng, tau=(0.015, 0.010), noise_snr=20)
    t1, t2 = fit(SA, SB); print(f"  15.0/10.0 -> {t1*1e3:6.2f}/{t2*1e3:6.2f} ms")
    print("T1/T2 when the truth is a DOUBLE pole but a single pole is fitted, and vice versa:")
    SA, SB = make_windows(n, 1.0, rng, tau=(0.008, 0.005), poles=2)
    for p in (1, 2):
        t1, t2 = fit(SA, SB, poles=p)
        print(f"  truth DP 8/5 ms, fit {p}-pole -> {t1*1e3:6.2f}/{t2*1e3:6.2f} ms")
    print("T/uC ratio (glider W 0.3, FP07 single pole 18 ms, uC flat):")
    for sal, fhi in [(0.0, 30.0), (0.05, 30.0), (0.2, 30.0), (0.2, 12.0)]:
        SA, SB = make_windows(n, 0.3, rng, tau=(0.018, 0.0), uC=True, salinity=sal)
        tT, _ = fit(SA, SB, f_hi=fhi, ref_flat=True)
        print(f"  salinity leak {sal:.2f}, f <= {fhi:.0f} Hz -> tau_T {tT*1e3:6.2f} ms")



def group_from_synth(n_files, per_file, W, rng, **kw):
    """A ratio.Group built from synthetic 'files', to exercise analyse()."""
    grp = R.Group(F=F.copy())
    for i in range(n_files):
        SA, SB = make_windows(per_file, W, rng, **kw)
        name = f"file{i:03d}"
        grp.d1[name], grp.d2[name] = SA * per_file, SB * per_file
        grp.ok[name] = np.full(F.size, per_file)  # high-SNR synthetic: band set by f limits
        grp.n[name] = per_file
        grp.W.extend([W] * per_file)
    return grp


def check_analyse():
    rng = np.random.default_rng(3)
    print("analyse() with bootstrap over 60 files x 50 windows:")
    for taus in [(0.010, 0.010), (0.012, 0.010), (0.015, 0.010), (0.025, 0.010)]:
        g = group_from_synth(60, 50, 1.0, rng, tau=taus)
        r = R.analyse(f"truth {taus[0]*1e3:.0f}/{taus[1]*1e3:.0f}", g, n_boot=100, rng=rng)
        print("  " + R.fmt(r))


if __name__ == "__main__" and sys.argv[1:] == ["analyse"]:
    check_analyse()
