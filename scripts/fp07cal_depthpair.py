"""Pair on PRESSURE instead of time, and see whether the geometry term vanishes.

The argument
------------
The MR's pressure sensor sits 1-2 cm from the FP07; the CTD's pressure sensor is
colocated with its thermistor.  So each instrument measures the depth of its own
temperature sample.  In a steady, horizontally homogeneous column T = T(z), and
matching an FP07 sample to the CTD sample at the SAME MEASURED PRESSURE makes
the sensor separation cancel identically -- longitudinal and perpendicular alike
-- without modelling it at all.  Clock skew cancels too, because time never
enters the pairing.

What should survive is only the CTD's thermal response (~2.7 s), which in depth
space is a shift of w*tau and reverses sign between dive and climb.

Prediction, if that is right
----------------------------
Time-pairing needed a dz*dT/dz term worth ~0.8 m to absorb the geometry, and
left a -3.2 mK residual feature concentrated in the thermocline.  Depth-pairing
should shrink that feature substantially, because the thing it was absorbing was
geometry rather than calibration.

The trade
---------
Depth-pairing is only as good as the CROSS-CALIBRATION of the two pressure
sensors: a zero offset between them injects a depth error directly, where
time-pairing was immune to it.  That offset is measured here too.
"""
import glob
import sys

import numpy as np

sys.path.insert(0, "/Users/pat/Desktop/turbulence/.claude/worktrees/fp07-insitu-cal/src")
from odas_tpw.fp07cal import PairConfig, load_hotel_reference, load_probe_series, log_r
from odas_tpw.fp07cal.fit import fit_calibration
from odas_tpw.fp07cal.geometry import joint_fit, local_dTdz
from odas_tpw.fp07cal.lag import temperature_lag
from odas_tpw.fp07cal.pairs import PairSet, _single_pole

D = "/Volumes/SeaChest/ARCTERX/2025/Interior/Gliders/osu685"
LAG_HINT = 7.4      # only used to find which CTD samples belong to a profile
BIN_DBAR = 0.30     # ~ the depth the CTD integrates over at 0.3 dbar/s, 1 Hz


def depth_pairs(probe, ref, channel, cfg, p_offset=0.0):
    """Match FP07 to CTD at equal MEASURED pressure, within each profile.

    ``p_offset`` is added to the CTD pressure -- the cross-calibration knob.
    """
    dt = 1.0 / probe.fs
    L_raw, clipped = log_r(probe.counts[channel], probe.bridge[channel])
    L_slow = _single_pole(L_raw, dt, cfg.kernel_tau)
    L_slow = np.where(clipped, np.nan, L_slow)

    T, P, Lp, W, uid = [], [], [], [], []
    for i, (s, e) in enumerate(probe.profiles):
        if e - s < 5000:
            continue
        t0, t1 = probe.time[s], probe.time[e]
        m = (ref.time >= t0) & (ref.time <= t1)
        if int(m.sum()) < 100 or ref.pressure is None:
            continue
        pm = probe.pressure[s : e + 1]
        lm = L_slow[s : e + 1]
        good = np.isfinite(pm) & np.isfinite(lm)
        if good.sum() < 1000:
            continue
        pm, lm = pm[good], lm[good]
        w_prof = np.gradient(probe.pressure[s : e + 1], probe.time[s : e + 1])[good]

        # Sort by pressure so the profile becomes a monotone function of depth.
        o = np.argsort(pm)
        pm, lm, w_prof = pm[o], lm[o], w_prof[o]

        pc = ref.pressure[m] + p_offset
        tc = ref.value[m]
        ok = np.isfinite(pc) & np.isfinite(tc) & (pc >= pm.min()) & (pc <= pm.max())
        if int(ok.sum()) < 50:
            continue
        pc, tc = pc[ok], tc[ok]

        # Average L over the pressure interval the CTD integrated across.
        lo = np.searchsorted(pm, pc - BIN_DBAR / 2)
        hi = np.searchsorted(pm, pc + BIN_DBAR / 2)
        csum = np.concatenate(([0.0], np.cumsum(lm)))
        n = (hi - lo).astype(float)
        with np.errstate(invalid="ignore"):
            lbar = np.where(n > 0, (csum[hi] - csum[lo]) / np.maximum(n, 1), np.nan)
        keep = np.isfinite(lbar) & (n >= 2)
        if keep.sum() < 50:
            continue
        T.append(tc[keep])
        P.append(pc[keep])
        Lp.append(lbar[keep])
        W.append(np.interp(pc[keep], pm, w_prof))
        uid.append(np.full(int(keep.sum()), f"{probe.label}#{i}", dtype=object))

    if not T:
        return PairSet(channel=channel)
    return PairSet(
        time=np.concatenate(P), T_ref=np.concatenate(T), L=np.concatenate(Lp),
        pressure=np.concatenate(P), w=np.concatenate(W),
        direction=np.sign(np.concatenate(W)).astype(np.int8),
        profile_uid=np.concatenate(uid),
        file_label=np.concatenate(uid), channel=channel,
    )


def main() -> None:
    ref = load_hotel_reference(
        f"{D}/PASS0/ebd.nc", value_var="sci_water_temp",
        pressure_var="sci_water_pressure", pressure_scale=10.0, valid_min=1.0,
    )
    cands = [f for f in sorted(glob.glob(f"{D}/MR/*.p")) if int(f[-6:-2]) >= 3]
    probes = []
    for f in cands[:: max(1, len(cands) // 24)][:24]:
        try:
            p = load_probe_series(f)
        except Exception:
            continue
        if p.profiles:
            probes.append(p)
    cfg = PairConfig(max_gap=30.0, min_speed=0.0)
    print(f"{len(probes)} files\n")

    for ch in ("T1", "T2"):
        # ---- time-paired (what the tool does today) ------------------------
        _lr, tp = temperature_lag(probes, ref, ch, cfg=cfg, max_lag=20.0, step=0.5)
        g_t = local_dTdz(tp)
        f_t, geo_t = joint_fit(tp, order=2, dTdz=g_t)
        print(f"=== {ch}")
        print(f"  TIME-paired : {len(tp)} pairs  rms {f_t.rms_K*1e3:6.3f} mK  "
              f"dz {geo_t.dz_m*100:+.1f} cm  tau {geo_t.tau_s:+.2f} s  "
              f"collin {geo_t.collinearity:.3f}")

        # ---- depth-paired ---------------------------------------------------
        dp = PairSet(channel=ch)
        for p in probes:
            if ch in p.counts:
                dp = dp.concat(depth_pairs(p, ref, ch, cfg))
        if not len(dp):
            print("  DEPTH-paired: no pairs")
            continue
        f_d = fit_calibration(dp, order=2)
        g_d = local_dTdz(dp)
        f_dj, geo_d = joint_fit(dp, order=2, dTdz=g_d)
        print(f"  DEPTH-paired: {len(dp)} pairs  rms {f_d.rms_K*1e3:6.3f} mK "
              f"(no geometry term at all)")
        print(f"     with a geometry term anyway: rms {f_dj.rms_K*1e3:6.3f} mK  "
              f"dz {geo_d.dz_m*100:+.1f} cm  tau {geo_d.tau_s:+.2f} s")
        print(f"     -> if depth-pairing removed the geometry, that dz should "
              f"collapse toward 0 (was {geo_t.dz_m*100:+.1f} cm)")

        for tag, f_, ps in (("time ", f_t, tp), ("depth", f_d, dp)):
            g = local_dTdz(ps)
            ok = f_.kept & np.isfinite(g) & (np.abs(g) > 0.005)
            if ok.sum() > 100:
                thermo = ok & (np.abs(g) > 0.05)
                deep = ok & (np.abs(g) < 0.01)
                print(f"     {tag}: residual in strong gradient "
                      f"{np.nanmean(f_.residual_K[thermo])*1e3:+.2f} mK, "
                      f"weak gradient {np.nanmean(f_.residual_K[deep])*1e3:+.2f} mK")
        print(f"     coefficients: time  t_0 {f_t.config_equivalent['t_0']:.4f}  "
              f"beta_1 {f_t.config_equivalent['beta_1']:.2f}")
        print(f"                   depth t_0 {f_d.config_equivalent['t_0']:.4f}  "
              f"beta_1 {f_d.config_equivalent['beta_1']:.2f}")
        print()

    # ---- how well do the two pressure sensors agree? ---------------------
    print("pressure cross-calibration (the thing depth-pairing now depends on):")
    diffs = []
    for p in probes[:8]:
        m = (ref.time >= p.time[0] + LAG_HINT) & (ref.time <= p.time[-1] + LAG_HINT)
        if m.sum() < 200:
            continue
        pm = np.interp(ref.time[m] - LAG_HINT, p.time, p.pressure)
        d = ref.pressure[m] - pm
        shallow = pm < 20
        diffs.append((float(np.median(d)), float(np.median(d[shallow]))
                      if shallow.sum() > 20 else np.nan))
    if diffs:
        a = np.array(diffs)
        print(f"  P_ctd - P_mr: whole profile median {np.nanmedian(a[:,0]):+.3f} dbar, "
              f"shallow (<20 dbar) {np.nanmedian(a[:,1]):+.3f} dbar")
        print("  A constant part is a sensor zero offset and biases depth-pairing;")
        print("  a pitch-dependent part is the real geometry and is what cancels.")


if __name__ == "__main__":
    main()
