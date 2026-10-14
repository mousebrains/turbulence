"""T/uC with a physical salinity leak (S has its own kB_S = 10 kB_T) and per-window x gating."""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import ratio as R
import ratio_synth as RS
from scipy.optimize import least_squares

F = RS.F
def run(n, W_med, tauT, r_sal, x_max, rng, snr=200.0):
    num = np.zeros_like(F); den = np.zeros_like(F)
    for _ in range(n):
        W = W_med*np.exp(rng.normal(0,0.05)); kB = np.exp(rng.normal(np.log(150),0.5)); K = F/W
        xT = np.maximum(K/kB, 1e-6); xS = xT/10
        q6 = np.sqrt(6*5.26)
        amp = np.exp(rng.normal(np.log(3.0), 0.7))
        fac = 1 + amp*(0.03/np.maximum(xT,1e-4))**1.5          # finestructure: stirs T AND S alike
        dip = 1 - 0.4*np.exp(-0.5*(np.log(xT/0.3)/0.4)**2)
        gT = xT*np.exp(-q6*xT)*fac*dip                          # T gradient spectrum (non-universal)
        gS = r_sal*xT*np.exp(-q6*xS)*fac                        # S part: same low-k form, rolls off at kB_S = 10 kB_T
        lvl = np.exp(rng.normal(0,1)); nz = lvl*np.max(gT)/snr
        t = tauT*(W/W_med)**-0.5
        ST = (lvl*gT*R.h2(F,t,1)+nz)*rng.chisquare(RS.DOF,F.size)/RS.DOF - nz
        SC = (1.3*lvl*(gT+gS)+nz)*rng.chisquare(RS.DOF,F.size)/RS.DOF - nz
        m = (xT <= x_max) if x_max else np.ones_like(F, bool)
        num += np.where(m, ST, 0); den += np.where(m, SC, 0)
    ok = (F>=1)&(F<=30)&(num>0)&(den>0)
    f, lr = F[ok], np.log(num[ok]/den[ok])
    res = lambda p: lr-(p[0]+np.log(R.h2(f,np.exp(p[1]),1)))
    b = min((least_squares(res,[0.0,np.log(t0)]) for t0 in (0.005,0.015,0.04)), key=lambda r: r.cost)
    return np.exp(b.x[1])
rng = np.random.default_rng(11)
print("FP07 18 ms at W 0.3; salinity share r of conductivity variance at low k (absorbed into gain)")
for r in (0.0, 0.1, 0.3):
    row = [f"r={r:.1f}:"]
    for xm in (None, 0.3, 0.15, 0.08):
        row.append(f"x<={xm if xm else 'all'}: {run(2500, 0.3, 0.018, r, xm, rng)*1e3:5.1f} ms")
    print("  " + "   ".join(row))
