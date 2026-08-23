"""Deployment-wide temporal and depth stability — osu685.

Two passes, because the pairs do not fit in memory: ~3M CTD samples land inside
MR climbs, and holding every pair for both channels would be gigabytes.

Pass 1 (subsample)  fit the calibration once, jointly with the sensor geometry,
                    over files spread across the whole deployment.  beta_1 is a
                    slope and needs the full temperature range, so it must come
                    from a pooled fit.
Pass 2 (every file) apply those fixed coefficients and keep only per-profile
                    summaries.  The profile is the independent unit for every
                    statistic downstream, so nothing is lost by collapsing to
                    it, and the memory goes from gigabytes to kilobytes.

Depth and elapsed-time are confounded on this deployment (climbs only, so the
glider always runs deep->shallow).  The one lever that separates them is that
the glider alternated ~500 m and ~1000 m climbs: at a given depth the deep
climbs have been running about twice as long, so comparing the two populations
at matched depth isolates a genuine depth dependence from anything that grows
with elapsed time.
"""
import glob
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/pat/Desktop/turbulence/.claude/worktrees/fp07-insitu-cal/src")
from odas_tpw.fp07cal import PairConfig, load_hotel_reference, load_probe_series, log_r
from odas_tpw.fp07cal.geometry import joint_fit, local_dTdz
from odas_tpw.fp07cal.lag import pressure_offset, temperature_lag
from odas_tpw.fp07cal.logr import temperature
from odas_tpw.fp07cal.pairs import build_pairs
from odas_tpw.fp07cal.stability import SECONDS_PER_DAY, Block, drift_fit

D = "/Volumes/SeaChest/ARCTERX/2025/Interior/Gliders/osu685"
CH = ("T1", "T2")
DEPTH_EDGES = np.arange(0.0, 1050.0, 50.0)
FIT_STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 8
RUN_STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 1
OUT = sys.argv[3] if len(sys.argv) > 3 else "scratch/deployment"


def files(stride):
    return [f for f in sorted(glob.glob(f"{D}/MR/*.p")) if int(f[-6:-2]) >= 3][::stride]


def main() -> None:
    import os

    os.makedirs(OUT, exist_ok=True)
    ref = load_hotel_reference(
        f"{D}/PASS0/ebd.nc", value_var="sci_water_temp",
        pressure_var="sci_water_pressure", pressure_scale=10.0, valid_min=1.0,
    )
    cfg = PairConfig(max_gap=30.0, min_speed=0.0)
    print(f"reference: {ref.time.size} samples, "
          f"{ref.coverage_report(30.0)['duty_cycle']:.3f} duty cycle", flush=True)

    # ---------------- pass 1: coefficients + geometry, on a subsample --------
    t0 = time.time()
    sub = []
    for f in files(FIT_STRIDE):
        try:
            p = load_probe_series(f)
        except Exception:
            continue
        if p.profiles:
            sub.append(p)
    print(f"pass 1: {len(sub)} files ({time.time()-t0:.0f}s)", flush=True)

    po = pressure_offset(sub, ref, max_lag=25.0, step=0.25)
    print(f"  {po.summary()}", flush=True)

    model = {}
    for ch in CH:
        lr, pairs = temperature_lag(sub, ref, ch, cfg=cfg, max_lag=20.0, step=0.25)
        g = local_dTdz(pairs)
        fit, geo = joint_fit(pairs, order=2, dTdz=g)
        model[ch] = {"lag": lr.lag, "coeffs": fit.coeffs, "geo": geo}
        print(f"  {ch}: {lr.summary()}", flush=True)
        print(f"     response {lr.lag - po.lag:+.2f} s | {len(pairs)} pairs, "
              f"{pairs.n_profiles()} profiles | T {pairs.T_range[0]:.2f}..{pairs.T_range[1]:.2f}",
              flush=True)
        print(f"     t_0 {fit.config_equivalent['t_0']:.4f}  "
              f"beta_1 {fit.config_equivalent['beta_1']:.2f}  "
              f"beta_2 {fit.config_equivalent.get('beta_2', float('nan')):.4g}  "
              f"rms {fit.rms_K:.5f} K", flush=True)
        print(f"     {geo.summary()}  collinearity {geo.collinearity:.3f}", flush=True)
    del sub

    # ---------------- pass 2: every file, per-profile summaries only ---------
    recs = {ch: [] for ch in CH}
    t1t2 = []
    t0 = time.time()
    n_ok = n_skip = 0
    allf = files(RUN_STRIDE)
    for i, f in enumerate(allf):
        try:
            p = load_probe_series(f)
        except Exception:
            n_skip += 1
            continue
        if not p.profiles:
            n_skip += 1
            continue
        n_ok += 1

        # T1 - T2 needs no reference: available on every profile.
        if all(c in p.counts for c in CH):
            La, ca = log_r(p.counts["T1"], p.bridge["T1"])
            Lb, cb = log_r(p.counts["T2"], p.bridge["T2"])
            d = np.where(ca | cb,
                         np.nan,
                         temperature(La, model["T1"]["coeffs"])
                         - temperature(Lb, model["T2"]["coeffs"]))
            for k, (s, e) in enumerate(p.profiles):
                seg = d[s : e + 1]
                ok = np.isfinite(seg)
                if ok.sum() > 100:
                    t1t2.append((float(np.mean(p.time[s : e + 1])),
                                 float(np.mean(seg[ok])), int(ok.sum())))

        for ch in CH:
            if ch not in p.counts:
                continue
            m = model[ch]
            ps = build_pairs(p, ref, ch, lag=m["lag"], cfg=cfg)
            if len(ps) < 100:
                continue
            g = local_dTdz(ps)
            geo_corr = np.where(np.isfinite(g),
                                m["geo"].dz_m * g + m["geo"].tau_s * ps.w * g, 0.0)
            T_pred = temperature(ps.L, m["coeffs"]) + geo_corr
            resid = ps.T_ref - T_pred
            y = 1.0 / (ps.T_ref + 273.15)
            hi = np.zeros_like(ps.L)
            for j, a in enumerate(m["coeffs"]):
                if j:
                    hi = hi + a * ps.L**j
            a0 = y - hi

            uid = np.asarray(ps.profile_uid, dtype=object)
            for u in np.unique(uid):
                sel = uid == u
                r = resid[sel]
                good = np.isfinite(r)
                if good.sum() < 100:
                    continue
                P = ps.pressure[sel]
                bidx = np.clip(np.digitize(P, DEPTH_EDGES) - 1, 0, DEPTH_EDGES.size - 2)
                bs = np.zeros(DEPTH_EDGES.size - 1)
                bc = np.zeros(DEPTH_EDGES.size - 1)
                np.add.at(bs, bidx[good], r[good])
                np.add.at(bc, bidx[good], 1.0)
                recs[ch].append({
                    "t": float(np.mean(ps.time[sel])),
                    "n": int(good.sum()),
                    "a0": float(np.mean(a0[sel][good])),
                    "resid": float(np.mean(r[good])),
                    "T": float(np.median(ps.T_ref[sel])),
                    "Pmax": float(np.nanmax(P)),
                    "w": float(np.nanmedian(ps.w[sel])),
                    "bin_sum": bs, "bin_cnt": bc,
                })
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(allf)} files, {len(recs['T1'])} profiles, "
                  f"{time.time()-t0:.0f}s", flush=True)
    print(f"pass 2: {n_ok} usable, {n_skip} skipped, {time.time()-t0:.0f}s", flush=True)

    out = {"pressure_offset_s": po.lag, "n_files": n_ok, "channels": {}}
    for ch in CH:
        R = recs[ch]
        if len(R) < 10:
            print(f"{ch}: only {len(R)} profiles")
            continue
        t = np.array([r["t"] for r in R])
        a0 = np.array([r["a0"] for r in R])
        res = np.array([r["resid"] for r in R])
        Pmax = np.array([r["Pmax"] for r in R])
        o = np.argsort(t)
        t, a0, res, Pmax = t[o], a0[o], res[o], Pmax[o]
        days = (t - t.min()) / SECONDS_PER_DAY
        m = model[ch]
        print(f"\n=== {ch}: {len(R)} profiles over {days.max():.1f} days")
        print(f"  per-profile residual: median {np.median(res):+.5f} K, "
              f"sd {res.std():.5f} K, 5-95% {np.percentile(res, 5):+.5f}.."
              f"{np.percentile(res, 95):+.5f}")

        entry = {"n_profiles": len(R), "span_days": float(days.max()),
                 "lag_s": m["lag"], "dz_m": m["geo"].dz_m, "tau_s": m["geo"].tau_s,
                 "coeffs": m["coeffs"].tolist(),
                 "resid_median_K": float(np.median(res)), "resid_sd_K": float(res.std())}

        # ---- temporal: blocked t_0, profiles as the independent unit --------
        for nb in (6, 12, 24):
            edges = np.linspace(t.min(), t.max(), nb + 1)
            blocks = []
            for b in range(nb):
                sel = (t >= edges[b]) & (t <= edges[b + 1] if b == nb - 1 else t < edges[b + 1])
                if sel.sum() < 3:
                    continue
                v = a0[sel]
                mean = float(np.mean(v))
                se = float(np.std(v, ddof=1) / np.sqrt(v.size))
                TK = float(np.median([r["T"] for r in R])) + 273.15
                blocks.append(Block(
                    t_mid=float(np.mean(t[sel])), t_start=float(edges[b]),
                    t_end=float(edges[b + 1]), n_pairs=int(sum(
                        r["n"] for r, s in zip([R[i] for i in o], sel) if s)),
                    n_profiles=int(v.size), a0=mean, a0_se=se,
                    t_0=1.0 / mean if mean else float("nan"),
                    dT_K=float(-(mean - np.mean(a0)) * TK**2),
                    dT_se_K=float(se * TK**2),
                ))
            st = drift_fit(blocks, n_permutations=4000)
            st.channel = ch
            print(f"  TIME ({nb:2d} blocks): {st.summary()}")
            entry[f"stability_{nb}"] = {
                "probe_drift_K_per_day": st.probe_drift_K_per_day,
                "se": st.drift_se_K_per_day, "p": st.permutation_p,
                "significant": st.significant, "n_blocks": len(st.blocks),
                "dT": [b.dT_K for b in st.blocks],
                "t_mid": [b.t_mid for b in st.blocks],
            }

        # ---- depth, split by climb depth to break the depth/elapsed tie -----
        cen = 0.5 * (DEPTH_EDGES[:-1] + DEPTH_EDGES[1:])
        prof = {}
        for name, sel in (("all", np.ones(len(R), bool)),
                          ("deep", Pmax > 700), ("shallow", Pmax <= 700)):
            # sel is indexed by SORTED position; o maps sorted -> original.
            idx = [o[i] for i in range(len(R)) if sel[i]]
            s = np.zeros(cen.size)
            c = np.zeros(cen.size)
            for i in idx:
                s += R[i]["bin_sum"]
                c += R[i]["bin_cnt"]
            with np.errstate(invalid="ignore"):
                prof[name] = np.where(c > 50, s / np.maximum(c, 1), np.nan)
            print(f"  DEPTH [{name:8s}] n={len(idx):4d}  residual by 50-dbar bin "
                  f"[{np.nanmin(prof[name]):+.4f}..{np.nanmax(prof[name]):+.4f}] K")
        entry["depth_centers"] = cen.tolist()
        entry["depth_profile"] = {k: np.where(np.isfinite(v), v, None).tolist()
                                  for k, v in prof.items()}
        both = np.isfinite(prof["deep"]) & np.isfinite(prof["shallow"])
        if both.sum() > 3:
            d = prof["deep"][both] - prof["shallow"][both]
            print(f"  DEPTH deep-minus-shallow at matched depth: "
                  f"median {np.median(d):+.5f} K, max |{np.max(np.abs(d)):.5f}| K")
            print("    (a genuine DEPTH dependence cancels here; what survives "
                  "tracks elapsed time)")
            entry["deep_minus_shallow_K"] = d.tolist()
        out["channels"][ch] = entry

    if t1t2:
        a = np.array(t1t2)
        o = np.argsort(a[:, 0])
        tt, vv = a[o, 0], a[o, 1]
        dd = (tt - tt.min()) / SECONDS_PER_DAY
        sl = float(np.polyfit(dd, vv, 1)[0])
        print(f"\nT1-T2: {vv.size} profiles, mean {np.mean(vv):+.4f} K, "
              f"slope {sl:+.3e} K/day (reference-free)")
        out["t1t2"] = {"n": int(vv.size), "mean_K": float(np.mean(vv)),
                       "slope_K_per_day": sl,
                       "t": tt.tolist(), "v": vv.tolist()}
        np.savez_compressed(f"{OUT}/t1t2.npz", t=tt, v=vv)

    with open(f"{OUT}/summary.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    for ch in CH:
        if recs[ch]:
            np.savez_compressed(
                f"{OUT}/{ch}_profiles.npz",
                t=np.array([r["t"] for r in recs[ch]]),
                a0=np.array([r["a0"] for r in recs[ch]]),
                resid=np.array([r["resid"] for r in recs[ch]]),
                Pmax=np.array([r["Pmax"] for r in recs[ch]]),
                T=np.array([r["T"] for r in recs[ch]]),
                bin_sum=np.array([r["bin_sum"] for r in recs[ch]]),
                bin_cnt=np.array([r["bin_cnt"] for r in recs[ch]]),
                edges=DEPTH_EDGES,
            )
    print(f"\nwrote {OUT}/")


if __name__ == "__main__":
    main()
