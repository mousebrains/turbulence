"""Offset, vertical offset and GAIN with 10-dbar block-bootstrap 95% intervals.

    dX = a + b * dRef/dp + s * (Tref - Tbar)

a: offset at the cast-mean temperature; b: effective vertical offset [dbar];
s: fractional gain error of X against the SBE [K/K]. chi ~ (dT/dz)^2, so chi
scales by ~(1+s)^2.
"""
import copy
import numpy as np, xarray as xr
from crosscheck import read_cnv, bin_to, CNV, PFILE, PROF_B, P_MIN
from odas_tpw.rsi import PFile
from odas_tpw.perturb.fp07_cal import fp07_calibrate

rng = np.random.default_rng(20170226)
sbe = read_cnv(CNV); pb = sbe["prDM"]; T0 = sbe["t090C"]; C0 = sbe["c0mS/cm"]
gT = np.gradient(T0, pb)

def fit3(y, sel, block=10, nboot=4000):
    s = np.where(sel & np.isfinite(y))[0]
    Tbar = T0[s].mean()
    def solve(ix):
        A = np.column_stack([np.ones(ix.size), gT[ix], T0[ix] - Tbar])
        return np.linalg.lstsq(A, y[ix], rcond=None)[0]
    est = solve(s)
    blocks = [s[i:i + block] for i in range(0, s.size, block)]
    boots = np.array([solve(np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])) for _ in range(nboot)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return est, lo, hi, Tbar, s.size

def show(label, y, sel):
    (a, b, g), lo, hi, Tbar, n = fit3(y, sel)
    print(f"  {label:30s} n={n:3d} Tbar={Tbar:5.2f}C  a={a*1e3:+6.1f} mK [{lo[0]*1e3:+6.1f},{hi[0]*1e3:+6.1f}]"
          f"  b={b:+5.2f} dbar [{lo[1]:+5.2f},{hi[1]:+5.2f}]"
          f"  gain s={g*1e3:+6.2f} mK/K [{lo[2]*1e3:+6.2f},{hi[2]*1e3:+6.2f}]"
          f"  -> chi x{(1+g)**2:.4f} [{(1+lo[2])**2:.4f},{(1+hi[2])**2:.4f}]")

pf = PFile(str(PFILE)); c = pf.channels; P = c["P"]
imax = int(np.nanargmax(P)); i0 = int(np.argmax(P > 2.0)); down = np.zeros(P.size, bool); down[i0:imax + 1] = True
prof = [(i0, imax)]
deep = pb >= P_MIN

print("reference: SBE secondary vs primary")
show("t190C - t090C", sbe["t190C"] - T0, deep)

print("A  JAC_T vs SBE, whole downcast")
jt, _ = bin_to(pb, P, c["JAC_T"], down)
show("JAC_T - t090C", jt - T0, deep)

print("C  fp07_calibrate over the whole downcast (order 2), vs SBE")
cal = fp07_calibrate(copy.deepcopy(pf), prof, reference="JAC_T", order=2, max_lag_seconds=10.0, must_be_negative=True)
for ch in ("T1", "T2"):
    x, _ = bin_to(pb, P, cal["channels"][ch], down)
    show(f"{ch} (C) - t090C", x - T0, deep)
    show(f"{ch} (C) - JAC_T", x - jt, deep)

print("B  pipeline output, its 102.7-248.7 dbar profile, vs SBE")
d = xr.open_dataset(PROF_B, decode_times=False); pP = d["P"].values
inb = deep & (pb >= np.ceil(np.nanmin(pP))) & (pb <= np.floor(np.nanmax(pP)))
for ch in ("T1", "T2", "JAC_T"):
    x, _ = bin_to(pb, pP, d[ch].values, np.ones(pP.size, bool))
    show(f"{ch} (B) - t090C", x - T0, inb)
