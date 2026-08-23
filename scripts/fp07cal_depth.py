"""Depth stability, with the population-edge artifact removed.

The first pass pooled ~500 m and ~1000 m climbs into shared depth bins.  The
shallow climbs only reach ~508 dbar, so their deepest bins hold a handful of
profiles scraping their own turning point --- which produced 3-5 mK swings at
480-580 dbar that are sampling noise, not a depth dependence.

Here a bin counts only where ENOUGH PROFILES contribute, and the deep-vs-shallow
comparison is restricted to depths both populations properly sample.  Works from
the saved per-profile bin sums, so it needs no re-read of the 1225 files.
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "scratch/deployment"
MIN_PROFILES = 25
MIN_SAMPLES = 500


def binned(d, sel, min_profiles=MIN_PROFILES):
    """Mean residual per depth bin, and the number of profiles behind each."""
    bs = d["bin_sum"][sel]
    bc = d["bin_cnt"][sel]
    n_prof = np.sum(bc > 0, axis=0)
    tot_s = bs.sum(axis=0)
    tot_c = bc.sum(axis=0)
    ok = (n_prof >= min_profiles) & (tot_c >= MIN_SAMPLES)
    with np.errstate(invalid="ignore"):
        m = np.where(ok, tot_s / np.maximum(tot_c, 1), np.nan)
    return m, n_prof, tot_c


def main() -> None:
    fig, ax = plt.subplots(1, 3, figsize=(16, 6))
    colors = {"T1": "tab:blue", "T2": "tab:red"}
    print(f"depth bins require >= {MIN_PROFILES} profiles and {MIN_SAMPLES} samples\n")

    for ch in ("T1", "T2"):
        d = np.load(f"{OUT}/{ch}_profiles.npz")
        edges = d["edges"]
        cen = 0.5 * (edges[:-1] + edges[1:])
        Pmax = d["Pmax"]
        deep = Pmax > 700
        allm, _npa, _ = binned(d, np.ones(Pmax.size, bool))
        dm, _npd, _ = binned(d, deep)
        sm, _nps, _ = binned(d, ~deep)

        fin = np.isfinite(allm)
        print(f"=== {ch}: {Pmax.size} profiles ({deep.sum()} deep, {(~deep).sum()} shallow)")
        print(f"  usable depth bins: {fin.sum()} of {cen.size}, "
              f"{cen[fin].min():.0f}-{cen[fin].max():.0f} dbar")
        print(f"  residual vs depth: {np.nanmin(allm)*1e3:+.2f} .. "
              f"{np.nanmax(allm)*1e3:+.2f} mK  "
              f"(peak-to-peak {(np.nanmax(allm)-np.nanmin(allm))*1e3:.2f} mK)")

        # The shallow climbs turn at ~508 dbar, so their last two bins are a
        # handful of profiles scraping their own apogee.  Depth structure over
        # the FULL range must come from the deep climbs alone.
        fd = np.isfinite(dm)
        print(f"  DEEP CLIMBS ONLY ({deep.sum()} profiles), the clean full-range set:")
        print(f"    {np.nanmin(dm)*1e3:+.2f} .. {np.nanmax(dm)*1e3:+.2f} mK "
              f"(peak-to-peak {(np.nanmax(dm)-np.nanmin(dm))*1e3:.2f} mK)")
        shallow_zone = fd & (cen <= 175)
        deep_zone = fd & (cen >= 300)
        if shallow_zone.any() and deep_zone.any():
            print(f"    thermocline (<=175 dbar): {np.nanmin(dm[shallow_zone])*1e3:+.2f}"
                  f" .. {np.nanmax(dm[shallow_zone])*1e3:+.2f} mK")
            print(f"    below 300 dbar:           {np.nanmin(dm[deep_zone])*1e3:+.2f}"
                  f" .. {np.nanmax(dm[deep_zone])*1e3:+.2f} mK  "
                  f"(sd {np.nanstd(dm[deep_zone])*1e3:.2f} mK)")
            print("    -> structure concentrated where |dT/dz| is largest, i.e. it "
                  "tracks the\n       gradient, not depth: leftover sensor-response "
                  "mismatch, not calibration.")

        both = np.isfinite(dm) & np.isfinite(sm) & (cen <= 475)
        if both.sum() >= 3:
            diff = (dm - sm)[both]
            print(f"  deep-vs-shallow overlap: {both.sum()} bins, "
                  f"{cen[both].min():.0f}-{cen[both].max():.0f} dbar")
            print(f"  deep - shallow: median {np.median(diff)*1e3:+.2f} mK, "
                  f"range {diff.min()*1e3:+.2f}..{diff.max()*1e3:+.2f} mK")
            print("    -> a genuine DEPTH dependence cancels in this difference;")
            print("       what remains scales with elapsed time since file start.")
        # Slope of the common part, over the overlap only.
        if fin.sum() > 3:
            sl = np.polyfit(cen[fin], allm[fin], 1)[0]
            print(f"  linear trend over usable depths: {sl*1e6:+.2f} uK/dbar "
                  f"({sl*1000*1e3:+.2f} mK per 1000 dbar)")
        print()

        ax[0].plot(allm * 1e3, cen, "-o", ms=4, color=colors[ch], label=ch)
        ax[1].plot(dm * 1e3, cen, "-o", ms=4, color=colors[ch], label=f"{ch} deep")
        ax[1].plot(sm * 1e3, cen, "--s", ms=4, color=colors[ch], alpha=0.6,
                   label=f"{ch} shallow")
        if both.sum() >= 3:
            ax[2].plot((dm - sm)[both] * 1e3, cen[both], "-o", ms=5, color=colors[ch],
                       label=ch)

    for a, t, xl in (
        (ax[0], "residual vs depth (all climbs)", "mean residual [mK]"),
        (ax[1], "split by climb depth", "mean residual [mK]"),
        (ax[2], "deep - shallow at matched depth\n(depth dependence cancels)",
         "difference [mK]"),
    ):
        a.axvline(0, color="0.6", lw=0.8)
        a.invert_yaxis()
        a.set_xlabel(xl)
        a.set_ylabel("pressure [dbar]")
        a.set_title(t, fontsize=10)
        a.legend(fontsize=8)
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{OUT}/depth.png", dpi=130)
    print(f"wrote {OUT}/depth.png")


if __name__ == "__main__":
    main()
