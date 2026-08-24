"""Extract the GPCTD thermistor response time from osu685.

Two independent estimates, because neither is clean on its own.

A. Residual sweep.  Sweep the pole applied to the FP07, refit the calibration
   at each value (re-estimating the lag, since the pole changes group delay),
   and take the minimum residual.  A pole MISMATCH is not a delay -- the lag
   search cannot absorb it -- so a mismatched pole leaves a gradient-
   proportional residual and the curve should have a minimum near truth.

B. Transfer function.  Block-average the FP07 onto the CTD's own 1 Hz sample
   times (a boxcar whose sinc is known and divided out), then fit

       |H(f)| = 1 / sqrt(1 + (2*pi*f*tau)^2)

   over the band where the two sensors are still coherent.

The confound both share, stated up front: the sensors are ~1 m apart, so they
sample different water below that scale.  At w ~ 0.3 m/s, 1 m is ~3 s ~ 0.3 Hz
-- right where a 0.5-0.75 s pole rolls off.  Spatial decorrelation and the pole
are therefore hard to separate, and both methods will be biased toward LARGER
tau.  Coherence is reported so the reader can see where the estimate stops
meaning anything.

The CTD is sampled at 1 Hz, so tau ~ 0.5-0.75 s is sub-sample; its corner
f = 1/(2*pi*tau) = 0.21-0.32 Hz does sit below the 0.5 Hz Nyquist, so the
roll-off is partially in band -- but only just.
"""
import glob
import sys

import numpy as np

sys.path.insert(0, "/Users/pat/Desktop/turbulence/.claude/worktrees/fp07-insitu-cal/src")
from odas_tpw.fp07cal import PairConfig, load_hotel_reference, load_probe_series, log_r
from odas_tpw.fp07cal.fit import fit_calibration
from odas_tpw.fp07cal.lag import temperature_lag
from odas_tpw.fp07cal.logr import temperature

D = "/Volumes/SeaChest/ARCTERX/2025/Interior/Gliders/osu685"
CH = "T1"


def load(n_files=12):
    ref = load_hotel_reference(
        f"{D}/PASS0/ebd.nc", value_var="sci_water_temp",
        pressure_var="sci_water_pressure", pressure_scale=10.0, valid_min=1.0,
    )
    cands = [f for f in sorted(glob.glob(f"{D}/MR/*.p")) if int(f[-6:-2]) >= 3]
    probes = []
    for f in cands[:: max(1, len(cands) // n_files)][:n_files]:
        try:
            p = load_probe_series(f)
        except Exception:
            continue
        if p.profiles:
            probes.append(p)
    return probes, ref


def sweep(probes, ref, taus):
    print("A. residual sweep (pole applied to the FP07, lag re-fitted at each)")
    print(f"   {'tau [s]':>8}  {'lag [s]':>8}  {'rms [mK]':>9}  {'beta_1':>9}  {'n pairs':>8}")
    rows = []
    for tau in taus:
        cfg = PairConfig(max_gap=30.0, min_speed=0.0, kernel_tau=float(tau))
        lr, ps = temperature_lag(probes, ref, CH, cfg=cfg, max_lag=20.0, step=0.5)
        if len(ps) < 500:
            continue
        fit = fit_calibration(ps, order=2)
        rows.append((tau, lr.lag, fit.rms_K, fit.config_equivalent["beta_1"], len(ps)))
        print(f"   {tau:8.2f}  {lr.lag:8.2f}  {fit.rms_K*1e3:9.4f}  "
              f"{fit.config_equivalent['beta_1']:9.2f}  {len(ps):8d}")
    if rows:
        a = np.array([(r[0], r[2]) for r in rows])
        i = int(np.argmin(a[:, 1]))
        print(f"   -> minimum residual at tau = {a[i,0]:.2f} s "
              f"({a[i,1]*1e3:.4f} mK)")
        # Parabolic refine on the three points around the minimum.
        if 0 < i < len(a) - 1:
            x, y = a[i-1:i+2, 0], a[i-1:i+2, 1]
            d = y[0] - 2 * y[1] + y[2]
            if d > 0:
                xr = x[1] + 0.5 * (x[2] - x[0]) * (y[0] - y[2]) / (2 * d) * 0 + \
                     0.5 * (x[1] - x[0]) * (y[0] - y[2]) / d
                print(f"   -> parabolic refinement: tau = {xr:.2f} s")
    return rows


def transfer(probes, ref, lag):
    """B. |CTD(f)/FP07(f)| on the CTD's own 1 Hz grid, sinc-corrected."""
    from scipy.signal import csd, welch

    print("\nB. transfer function on the CTD's own timestamps")
    X, Y = [], []
    for p in probes:
        if CH not in p.counts:
            continue
        L, clip = log_r(p.counts[CH], p.bridge[CH])
        T_fp = np.where(clip, np.nan, temperature(L, p.factory[CH]))
        for s, e in p.profiles:
            if e - s < 20000:
                continue
            t0, t1 = p.time[s], p.time[e]
            m = (ref.time >= t0 + 2) & (ref.time <= t1 - 2)
            tk = ref.time[m]
            if tk.size < 512:
                continue
            dt = float(np.median(np.diff(tk)))
            if not (0.5 < dt < 2.0):
                continue
            # Block-average the FP07 over each CTD sample interval, centred at
            # t_k - lag so the two are time-aligned. This is the anti-alias;
            # its sinc is divided out below.
            edges_lo = np.searchsorted(p.time, tk - lag - dt / 2)
            edges_hi = np.searchsorted(p.time, tk - lag + dt / 2)
            good = np.isfinite(T_fp)
            cs = np.concatenate(([0.0], np.cumsum(np.where(good, T_fp, 0.0))))
            cn = np.concatenate(([0.0], np.cumsum(good.astype(float))))
            num = cs[edges_hi] - cs[edges_lo]
            den = cn[edges_hi] - cn[edges_lo]
            x = np.where(den > 0, num / np.maximum(den, 1), np.nan)
            y = ref.value[m]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 512:
                continue
            X.append(x[ok] - np.mean(x[ok]))
            Y.append(y[ok] - np.mean(y[ok]))
    if not X:
        print("   no usable segments")
        return None
    fs = 1.0
    nper = 256
    Hs, Cs, fr = [], [], None
    for x, y in zip(X, Y):
        if x.size < nper * 2:
            continue
        f, pxx = welch(x, fs, nperseg=nper)
        _, pyy = welch(y, fs, nperseg=nper)
        _, pxy = csd(x, y, fs, nperseg=nper)
        with np.errstate(invalid="ignore", divide="ignore"):
            Hs.append(np.abs(pxy) / pxx)
            Cs.append(np.abs(pxy) ** 2 / (pxx * pyy))
        fr = f
    if not Hs:
        print("   segments too short")
        return None
    H = np.nanmedian(np.vstack(Hs), axis=0)
    C = np.nanmedian(np.vstack(Cs), axis=0)
    sinc = np.sinc(fr * 1.0)          # np.sinc(x) = sin(pi x)/(pi x)
    Hc = H / np.where(np.abs(sinc) > 1e-3, sinc, np.nan)

    print(f"   {len(Hs)} segments, {nper}-point Welch")
    print(f"   {'f [Hz]':>7}  {'|H| raw':>8}  {'sinc-corr':>10}  {'coherence':>10}")
    for i in range(1, min(len(fr), 40)):
        if fr[i] > 0.5:
            break
        print(f"   {fr[i]:7.3f}  {H[i]:8.3f}  {Hc[i]:10.3f}  {C[i]:10.3f}")

    band = (fr > 0.02) & (fr < 0.45) & (C > 0.5) & np.isfinite(Hc) & (Hc > 0.05)
    if band.sum() >= 4:
        def model(tau):
            return 1.0 / np.sqrt(1.0 + (2 * np.pi * fr[band] * tau) ** 2)
        taus = np.arange(0.05, 4.0, 0.01)
        err = [np.sum((model(t) - Hc[band]) ** 2) for t in taus]
        best = taus[int(np.argmin(err))]
        print(f"   -> fit over {band.sum()} bins with coherence > 0.5: "
              f"tau = {best:.2f} s")
        print(f"      (band {fr[band].min():.3f}-{fr[band].max():.3f} Hz)")
        return best
    print("   -> too few coherent bins to fit")
    return None


if __name__ == "__main__":
    probes, ref = load()
    print(f"{len(probes)} files\n")
    rows = sweep(probes, ref, np.concatenate([np.arange(0.1, 1.6, 0.1),
                                              np.arange(1.8, 4.1, 0.4)]))
    lag = rows[0][1] if rows else 7.4
    transfer(probes, ref, lag)
