"""Is the JAC_T-vs-SBE gain real, or aliased from a depth-varying vertical offset?"""
import copy
import numpy as np
from crosscheck import read_cnv, bin_to, CNV, PFILE, P_MIN
from odas_tpw.rsi import PFile
from odas_tpw.perturb.fp07_cal import fp07_calibrate

rng = np.random.default_rng(1)
sbe = read_cnv(CNV); pb = sbe["prDM"]; T0 = sbe["t090C"]; gT = np.gradient(T0, pb)
pf = PFile(str(PFILE)); c = pf.channels; P = c["P"]; t = pf.t_slow
imax = int(np.nanargmax(P)); i0 = int(np.argmax(P > 2.0)); down = np.zeros(P.size, bool); down[i0:imax+1] = True
W = np.gradient(np.convolve(P, np.ones(64)/64, "same"), t)
Wb, _ = bin_to(pb, P, W, down)
jt, _ = bin_to(pb, P, c["JAC_T"], down)
cal = fp07_calibrate(copy.deepcopy(pf), [(i0, imax)], reference="JAC_T", order=2, max_lag_seconds=10.0, must_be_negative=True)
t1, _ = bin_to(pb, P, cal["channels"]["T1"], down)
base = (pb >= P_MIN) & np.isfinite(jt) & np.isfinite(Wb)

def run(label, y, cols, sel, gi):
    s = np.where(sel & np.isfinite(y))[0]
    X = lambda ix: np.column_stack([f(ix) for f in cols])
    est = np.linalg.lstsq(X(s), y[s], rcond=None)[0]
    blocks = [s[i:i+10] for i in range(0, s.size, 10)]
    bs = []
    for _ in range(3000):
        ix = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
        bs.append(np.linalg.lstsq(X(ix), y[ix], rcond=None)[0][gi])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    r = y[s] - X(s) @ est
    print(f"  {label:52s} n={s.size:3d} gain {est[gi]*1e3:+5.2f} mK/K [{lo*1e3:+5.2f},{hi*1e3:+5.2f}]  resid MAD {1.4826*np.median(np.abs(r-np.median(r)))*1e3:5.2f} mK")

one = lambda ix: np.ones(ix.size)
Tc = lambda ix: T0[ix] - 19.5
g = lambda ix: gT[ix]
up = lambda ix: gT[ix] * (pb[ix] < 100)
lo_ = lambda ix: gT[ix] * (pb[ix] >= 100)
gW = lambda ix: gT[ix] * Wb[ix]
for name, y in (("JAC_T - SBE", jt - T0), ("T1 (fit over whole cast) - SBE", t1 - T0)):
    print(name)
    run("1. single b (as before)", y, [one, g, Tc], base, 2)
    run("2. separate b above/below 100 dbar", y, [one, up, lo_, Tc], base, 3)
    run("3. b = b0 + tau*W (lag x descent rate)", y, [one, g, gW, Tc], base, 3)
    run("4. piecewise b AND lag term", y, [one, up, lo_, gW, Tc], base, 4)
    for thr in (0.005, 0.01):
        w = base & (np.abs(gT) < thr)
        print(f"     weak gradient |dT/dp|<{thr}: T span {T0[w].min():.2f}-{T0[w].max():.2f} C")
        run(f"5. no b, weak-gradient bins only (<{thr} K/dbar)", y, [one, Tc], w, 1)
