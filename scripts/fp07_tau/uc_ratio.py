# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""FP07 response from the FP07 / micro-conductivity spectral ratio.

On a MicroRider carrying an SBE7-type micro-conductivity probe (`C1_dC1`,
type `ucond`), both sensors see the same microstructure. Where temperature
drives the conductivity fluctuations,

    S_T(f) / (S_C(f) / alpha^2) = G * |H_T(f)|^2 / |H_C(f)|^2,

alpha = dC/dT at the window's T, S, P, and the uC response H_C is taken as
flat (it is far faster than an FP07). No turbulence spectral model enters.

Salinity microstructure reaches ~10x higher wavenumbers than temperature, so
above the temperature Batchelor rolloff conductivity carries extra variance
that would mimic FP07 attenuation. Each window is therefore gated on
x = k/kB <= x_max with kB from the SHEAR epsilon (validated in
uc_gate_synth.py: x <= 0.15 recovers tau within ~4% at a 30% salinity share).

Both spectra are formed here, identically, from the raw channels:
  T: deconvolved temperature (PFile T1_dT1), first-differenced, times the
     bilinear correction the chi pipeline applies;
  C: raw pre-emphasized counts, first-differenced, divided by the exact
     analog pre-emphasis response 1 + (2 pi f G)^2.
The first difference and the anti-alias filters are common and cancel.
"""

from __future__ import annotations

import glob
import os
import sys
import warnings
from dataclasses import dataclass, field

import netCDF4
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import welch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from odas_tpw.chi.chi import _bilinear_correction
from odas_tpw.rsi.p_file import PFile
from odas_tpw.scor160.ocean import kappa_T


def dC_dT(T, S=35.0, P=0.0):
    import gsw

    dT = 0.01
    return (gsw.C_from_SP(S, T + dT, P) - gsw.C_from_SP(S, T - dT, P)) / (2 * dT)  # mS/cm/K


@dataclass
class Acc:
    F: np.ndarray | None = None
    sT: dict = field(default_factory=dict)  # file -> gated sum of S_T
    sC: dict = field(default_factory=dict)  # file -> gated sum of S_C/alpha^2
    cnt: dict = field(default_factory=dict)  # file -> windows contributing per bin
    W: list = field(default_factory=list)
    kB: list = field(default_factory=list)


def accumulate(eps_glob, x_max=0.15, f_lo=0.5, f_hi=60.0, t_name="T1_dT1", c_name="C1_dC1",
               nfft=1024, salinity=35.0, w_edges=None):
    """Gated per-file sums, grouped by W bin (w_edges) or one group."""
    groups: dict = {}
    for ef in sorted(glob.glob(eps_glob)):
        with netCDF4.Dataset(ef) as ds:
            src = ds.getncattr("source")
            n_diss = int(ds.getncattr("diss_length"))
            t = np.asarray(ds["t"][:], float)
            W = np.asarray(ds["speed"][:], float)
            eps = np.asarray(ds["epsilon"][:], float)
            nu = np.asarray(ds["nu"][:], float)
            Tm = np.asarray(ds["T_mean"][:], float)
            Pm = np.asarray(ds["P_mean"][:], float) if "P_mean" in ds.variables else np.zeros_like(t)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pf = PFile(src)
        if t_name not in pf.channels or c_name not in pf.channels_raw:
            continue
        fs = pf.fs_fast
        T = np.asarray(pf.channels[t_name], float)
        C = np.asarray(pf.channels_raw[c_name], float)
        cfg = {ch["name"]: ch for ch in pf.config["channels"]}
        gT = float(cfg[t_name]["diff_gain"])
        gC = float(cfg[c_name]["diff_gain"])
        with np.errstate(invalid="ignore", divide="ignore"):
            eps_g = np.exp(np.nanmean(np.log(np.where(eps > 0, eps, np.nan)), axis=0))
        kap = kappa_T(Tm, salinity, Pm)
        kB = (1 / (2 * np.pi)) * (eps_g / (nu * kap**2)) ** 0.25
        alpha = dC_dT(Tm, salinity, Pm)
        for k in range(t.size):
            if not (np.isfinite(kB[k]) and kB[k] > 0 and np.isfinite(W[k]) and W[k] > 0.05):
                continue
            i0 = int(round(t[k] * fs - n_diss / 2))
            if i0 < 0 or i0 + n_diss > T.size:
                continue
            xt, xc = np.diff(T[i0:i0 + n_diss]), np.diff(C[i0:i0 + n_diss])
            if not (np.all(np.isfinite(xt)) and np.all(np.isfinite(xc))):
                continue
            F, sT = welch(xt, fs=fs, nperseg=nfft, noverlap=nfft // 2, window="hann", detrend="linear")
            _, sC = welch(xc, fs=fs, nperseg=nfft, noverlap=nfft // 2, window="hann", detrend="linear")
            sT = sT * _bilinear_correction(F, gT, fs)
            sC = sC / (1 + (2 * np.pi * F * gC) ** 2) / alpha[k] ** 2
            x = (F / W[k]) / kB[k]
            gate = (x <= x_max) & (f_lo <= F) & (f_hi >= F)
            b = 0 if w_edges is None else int(np.searchsorted(w_edges, W[k]) - 1)
            if w_edges is not None and not (0 <= b < len(w_edges) - 1):
                continue
            a = groups.setdefault(b, Acc(F=F))
            a.sT[ef] = a.sT.get(ef, 0) + np.where(gate, sT, 0)
            a.sC[ef] = a.sC.get(ef, 0) + np.where(gate, sC, 0)
            a.cnt[ef] = a.cnt.get(ef, 0) + gate.astype(int)
            a.W.append(W[k])
            a.kB.append(kB[k])
    return groups


def h2(f, tau, poles):
    return (1.0 + (2 * np.pi * f * tau) ** 2) ** (-poles)


def fit(F, sT, sC, cnt, poles, min_cnt=30):
    m = (cnt >= min_cnt) & (sT > 0) & (sC > 0)
    f, lr = F[m], np.log(sT[m] / sC[m])

    def res(p):
        return lr - (p[0] + np.log(h2(f, np.exp(p[1]), poles)))

    best = min((least_squares(res, [0.0, np.log(t0)], bounds=([-50, np.log(5e-4)], [50, np.log(0.3)]))
                for t0 in (0.003, 0.01, 0.03, 0.08)), key=lambda r: r.cost)
    return float(np.exp(best.x[1])), float(np.sqrt(2 * best.cost / max(f.size, 1))), f, lr, best.x[0]


def analyse(a: Acc, n_boot=200, rng=None, min_cnt=30):
    rng = rng or np.random.default_rng(0)
    files = list(a.sT)
    tot = lambda fl, d: sum(d[f] for f in fl)  # noqa: E731
    out = {}
    for poles in (1, 2):
        tau, rms, f, lr, c = fit(a.F, tot(files, a.sT), tot(files, a.sC), tot(files, a.cnt), poles, min_cnt)
        boots = []
        for _ in range(n_boot):
            pick = list(rng.choice(files, size=len(files), replace=True))
            boots.append(fit(a.F, tot(pick, a.sT), tot(pick, a.sC), tot(pick, a.cnt), poles, min_cnt)[0])
        out[poles] = (tau, rms, tuple(np.percentile(boots, [16, 84])), f, lr, c)
    return out


if __name__ == "__main__":
    eps_glob = sys.argv[1]
    x_max = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
    edges = np.array([float(v) for v in sys.argv[3].split(",")]) if len(sys.argv) > 3 else None
    groups = accumulate(eps_glob, x_max=x_max, w_edges=edges)
    for b, a in sorted(groups.items()):
        W = np.median(a.W)
        cnt = sum(a.cnt.values())
        print(f"group {b}: files {len(a.sT)} windows {len(a.W)} W med {W:.3f} kB p10/50/90 "
              f"{np.percentile(a.kB, [10, 50, 90]).round(0)}  bins with >=30 windows: "
              f"{int((cnt >= 30).sum())} ({a.F[cnt >= 30].min() if (cnt >= 30).any() else np.nan:.1f}-"
              f"{a.F[cnt >= 30].max() if (cnt >= 30).any() else np.nan:.1f} Hz)")
        res = analyse(a)
        for p, (tau, rms, ci, f, lr, c) in res.items():
            f3 = np.sqrt(2 ** (1 / p) - 1) / (2 * np.pi * tau)
            print(f"   {p}-pole: tau {tau*1e3:.1f} ms [{ci[0]*1e3:.1f}-{ci[1]*1e3:.1f}]  rms {rms:.3f}  "
                  f"f3dB {f3:.1f} Hz   (lueck {1/(2*np.pi*0.01*W**-0.5):.1f} Hz)")
        tau, rms, ci, f, lr, c = res[1]
        sel = [int(np.argmin(abs(f - q))) for q in (0.5, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 40) if f.size and f.min() <= q <= f.max()]
        print("   10log10(S_T/S_C) - offset [dB]: " + " ".join(f"{f[i]:.1f}:{10*(lr[i]-c)/np.log(10):+.2f}" for i in sel))
