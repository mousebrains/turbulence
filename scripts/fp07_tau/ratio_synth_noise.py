"""Ratio-method bias when the noise MODEL is wrong for one probe, ARCTERX-22-like band."""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import ratio as R
import ratio_synth as RS

F = RS.F

def windows(n, W_med, tau, rng, snr_peak, noise_true=(1.0, 1.0)):
    SA = np.zeros_like(F); SB = np.zeros_like(F)
    for _ in range(n):
        W = W_med*np.exp(rng.normal(0, 0.05)); kB = np.exp(rng.normal(np.log(70), 0.5))  # ARCTERX-22: kB ~70 cpm
        x = np.maximum(F/W/kB, 1e-6); g = RS.shape(x, rng); lvl = np.exp(rng.normal(0, 1.0))
        nmod = lvl*np.max(g)/snr_peak
        tA, tB = tau[0]*(W/W_med)**-0.5, tau[1]*(W/W_med)**-0.5
        a = lvl*g*R.h2(F, tA, 1) + nmod*noise_true[0]; b = 1.1*lvl*g*R.h2(F, tB, 1) + nmod*noise_true[1]
        SA += a*rng.chisquare(RS.DOF, F.size)/RS.DOF - nmod   # pipeline subtracts the MODEL
        SB += b*rng.chisquare(RS.DOF, F.size)/RS.DOF - nmod
    return SA/n, SB/n

rng = np.random.default_rng(5)
print("truth SP 13.0/19.5 ms, W 0.95, kB ~70 cpm, band 1-26 Hz, 4000 windows")
for snr in (300, 60):
    for nt in ((1, 1), (1, 1.5), (1, 0.67), (1.5, 1.5)):
        SA, SB = windows(4000, 0.95, (0.013, 0.0195), rng, snr, nt)
        m = (F >= 1) & (F <= 26) & (SA > 0) & (SB > 0)
        G, t1, t2, rms = R.fit_ratio(F[m], RS._running_median(SB[m]/SA[m], 3) if hasattr(RS, "_running_median") else SB[m]/SA[m], 1)
        print(f"  peak S/N {snr:3d}, true noise / model = {nt}: tau {t1*1e3:5.1f}/{t2*1e3:5.1f} ms")
