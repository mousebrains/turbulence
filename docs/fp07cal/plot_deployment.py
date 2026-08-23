"""Figures for the osu685 temporal / depth stability result."""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "scratch/deployment"
S = json.load(open(f"{OUT}/summary.json"))
DAY = 86400.0


def main() -> None:
    chans = [c for c in ("T1", "T2") if c in S["channels"]]
    fig, ax = plt.subplots(3, len(chans), figsize=(7 * len(chans), 12), squeeze=False)

    for j, ch in enumerate(chans):
        e = S["channels"][ch]
        d = np.load(f"{OUT}/{ch}_profiles.npz", allow_pickle=False)
        t, resid, Pmax = d["t"], d["resid"], d["Pmax"]
        days = (t - t.min()) / DAY
        deep = Pmax > 700

        # --- per-profile residual vs time, with the blocked trend ------------
        a = ax[0][j]
        a.plot(days[~deep], resid[~deep] * 1e3, ".", ms=3, alpha=0.5,
               label=f"~500 m climbs (n={int((~deep).sum())})", color="tab:blue")
        a.plot(days[deep], resid[deep] * 1e3, ".", ms=3, alpha=0.5,
               label=f"~1000 m climbs (n={int(deep.sum())})", color="tab:red")
        st = e.get("stability_12", {})
        if st.get("t_mid"):
            bt = (np.array(st["t_mid"]) - t.min()) / DAY
            a.plot(bt, np.array(st["dT"]) * 1e3, "o-", color="k", lw=1.5, ms=5,
                   label="blocked offset")
        a.axhline(0, color="0.6", lw=0.8)
        a.set_xlabel("days since deployment start")
        a.set_ylabel("reference − probe [mK]")
        drift = st.get("probe_drift_K_per_day", float("nan"))
        sig = "SIGNIFICANT" if st.get("significant") else "not significant"
        a.set_title(f"{ch}: probe drift {drift*1e3:+.3f} mK/day  "
                    f"(p={st.get('p', float('nan')):.3f}, {sig})", fontsize=10)
        a.legend(fontsize=8, markerscale=3)

        # --- residual vs depth, split by climb depth -------------------------
        a = ax[1][j]
        cen = np.array(e["depth_centers"])
        for name, col in (("all", "k"), ("deep", "tab:red"), ("shallow", "tab:blue")):
            v = np.array([np.nan if x is None else x for x in e["depth_profile"][name]],
                         dtype=float)
            a.plot(v * 1e3, cen, "-o", ms=3, lw=1.2, color=col, label=name)
        a.axvline(0, color="0.6", lw=0.8)
        a.invert_yaxis()
        a.set_xlabel("mean residual [mK]")
        a.set_ylabel("pressure [dbar]")
        a.set_title(f"{ch}: residual vs depth (geometry removed)", fontsize=10)
        a.legend(fontsize=8)

        # --- deep minus shallow: depth cancels, elapsed time does not --------
        a = ax[2][j]
        dm = e.get("deep_minus_shallow_K")
        if dm:
            vd = np.array([np.nan if x is None else x
                           for x in e["depth_profile"]["deep"]], dtype=float)
            vs = np.array([np.nan if x is None else x
                           for x in e["depth_profile"]["shallow"]], dtype=float)
            both = np.isfinite(vd) & np.isfinite(vs)
            a.plot((vd[both] - vs[both]) * 1e3, cen[both], "-o", ms=4, color="tab:purple")
            a.axvline(0, color="0.6", lw=0.8)
            a.invert_yaxis()
            a.set_xlabel("deep − shallow at matched depth [mK]")
            a.set_ylabel("pressure [dbar]")
            a.set_title(f"{ch}: a genuine DEPTH dependence cancels here;\n"
                        f"what survives tracks elapsed time", fontsize=9)
        else:
            a.text(0.5, 0.5, "no matched depths", ha="center", transform=a.transAxes)

    fig.tight_layout()
    fig.savefig(f"{OUT}/stability.png", dpi=130)
    print(f"wrote {OUT}/stability.png")

    if "t1t2" in S:
        d = np.load(f"{OUT}/t1t2.npz")
        t, v = d["t"], d["v"]
        days = (t - t.min()) / DAY
        f2, a = plt.subplots(figsize=(10, 4))
        a.plot(days, (v - np.mean(v)) * 1e3, ".", ms=3, alpha=0.6, color="tab:green")
        sl = S["t1t2"]["slope_K_per_day"]
        a.plot(days, (sl * (days - days.mean())) * 1e3, "-", color="k", lw=1.5,
               label=f"{sl*1e3:+.3f} mK/day")
        a.axhline(0, color="0.6", lw=0.8)
        a.set_xlabel("days since deployment start")
        a.set_ylabel("T1 − T2, demeaned [mK]")
        a.set_title(f"T1 − T2 over {len(v)} profiles — reference-free, so it covers "
                    f"every profile", fontsize=10)
        a.legend(fontsize=9)
        f2.tight_layout()
        f2.savefig(f"{OUT}/t1t2.png", dpi=130)
        print(f"wrote {OUT}/t1t2.png")


if __name__ == "__main__":
    main()
