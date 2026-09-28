#!/usr/bin/env python3
"""Close-approach epsilon comparison: VMP SN 194 vs MicroRider SN 134, CASPER-West 2017.

THE TWO PLATFORMS WERE NOT CO-LOCATED. The MicroRider held a ~5 x 8 km virtual
mooring offshore for the whole deployment; the VMP worked a station grid from
the Sally Ride, at a median ~26 km away. This therefore compares only the
subset of VMP casts that came close in BOTH space and time, and even then it
bounds gross error rather than calibrating anything -- epsilon varies by
decades over a few km, and the two platforms sample different settings.

VMP probe caveat applied here (CASPER-West VMP README): sh2 has a
depth-windowed fault from 2017-10-09 (worst 35-65 m) and fails outright on
10-24, so the VMP side uses e_1 alone from 10-09 above 65 m, and e_1 alone at
all depths from 10-24.

    colocate.py [max_km] [max_hours]
"""
import glob, sys
import numpy as np
import pandas as pd
import xarray as xr

MAX_KM = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
MAX_H  = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
VMP = "/Volumes/SeaChest/CASPER/2017_West/vmp/analysis/VMP_results/diss_binned_00/*.nc"
MR  = ("/Volumes/SeaChest/CASPER/2017_West/gliders/20170929_doug_CASPER/"
       "analysis/Processed/diss_binned_00/*.nc")
SH2_WINDOW_START = pd.Timestamp("2017-10-09")
SH2_DEAD_START   = pd.Timestamp("2017-10-24")


def load(pattern, label):
    """Return (bins, list of (time, lat, lon, eps_profile, e1_profile))."""
    out = []
    bins = None
    for f in sorted(glob.glob(pattern)):
        try:
            d = xr.open_dataset(f)
        except Exception:
            continue
        if "epsilonMean" not in d or "bin" not in d.coords:
            d.close(); continue
        b = np.asarray(d["bin"].values, float)
        bins = b if bins is None else bins
        def arr(name):
            if name not in d:
                return None
            x = np.asarray(d[name].values, float)
            return x.T if x.shape[0] == b.size else x
        em, e1 = arr("epsilonMean"), arr("e_1")
        la, lo = arr("lat"), arr("lon")
        # stime is a float epoch in binned.nc but datetime64 in combo.nc --
        # pd.to_datetime on the float would read it as NANOseconds and collapse
        # every profile to 1970, silently disabling the time filter.
        st = np.asarray(d["stime"].values).ravel()
        t = (pd.to_datetime(st) if np.issubdtype(st.dtype, np.datetime64)
             else pd.to_datetime(st.astype(float), unit="s"))
        n = em.shape[0]
        for i in range(n):
            lai = np.nanmedian(la[i]) if la is not None and la.ndim == 2 else (
                float(la[i]) if la is not None else np.nan)
            loi = np.nanmedian(lo[i]) if lo is not None and lo.ndim == 2 else (
                float(lo[i]) if lo is not None else np.nan)
            if not (np.isfinite(lai) and np.isfinite(loi)):
                continue
            out.append((t[i] if i < len(t) else t[0], lai, loi, em[i],
                        e1[i] if e1 is not None else em[i]))
        d.close()
    print("  %-4s %d profiles on %d bins" % (label, len(out), 0 if bins is None else bins.size))
    return bins, out


def main():
    print(__doc__.split("\n")[0])
    print("  thresholds: <= %.1f km and <= %.1f h\n" % (MAX_KM, MAX_H))
    vb, V = load(VMP, "VMP")
    mb, M = load(MR, "MR")
    if not V or not M:
        print("  missing products"); return
    # common depth bins
    common = np.intersect1d(np.round(vb, 3), np.round(mb, 3))
    vi = {round(float(x), 3): k for k, x in enumerate(vb)}
    mi = {round(float(x), 3): k for k, x in enumerate(mb)}
    print("  common depth bins: %d (%.1f .. %.1f m)\n" % (common.size, common.min(), common.max()))

    mt = pd.to_datetime([m[0] for m in M])
    mla = np.array([m[1] for m in M]); mlo = np.array([m[2] for m in M])

    pairs = []
    for vt, vla, vlo, vem, ve1 in V:
        dt = np.abs((mt - vt).total_seconds()) / 3600.0
        km = np.hypot((mla - vla) * 111.0, (mlo - vlo) * 111.0 * np.cos(np.radians(33.7)))
        sel = np.flatnonzero((dt <= MAX_H) & (km <= MAX_KM))
        if sel.size == 0:
            continue
        # VMP side: which probe(s) may be used at this time
        use_e1_only_shallow = vt >= SH2_WINDOW_START
        use_e1_only_all = vt >= SH2_DEAD_START
        for k in sel:
            pairs.append((vt, km[k], dt[k], vem, ve1, M[k][3],
                          use_e1_only_shallow, use_e1_only_all))
    print("  close-approach pairs: %d  (from %d VMP casts)\n" % (len(pairs), len(V)))
    if not pairs:
        return

    bands = [(2.5, 35), (35, 65), (65, 130), (130, 195)]
    print("  %-14s %8s %10s %10s %9s %8s" % ("depth band", "n pairs", "VMP eps", "MR eps", "VMP/MR", "MAD dex"))
    for lo_d, hi_d in bands:
        rs = []
        for vt, km, dt, vem, ve1, mem, e1sh, e1all in pairs:
            r = []
            for b in common:
                if not (lo_d <= b < hi_d):
                    continue
                # VMP probe rule
                vv = (ve1 if (e1all or (e1sh and b < 65)) else vem)[vi[round(float(b), 3)]]
                mm = mem[mi[round(float(b), 3)]]
                if np.isfinite(vv) and np.isfinite(mm) and vv > 0 and mm > 0:
                    r.append((np.log10(vv), np.log10(mm)))
            if len(r) >= 5:
                a = np.array(r)
                # PAIRED: median over depth bins of log10(VMP/MR) within this
                # pair. median(log VMP) - median(log MR) is NOT the same thing
                # and gave 1.38 where the paired statistic gives 1.75.
                rs.append((np.median(a[:, 0]), np.median(a[:, 1]),
                           np.median(a[:, 0] - a[:, 1])))
        if not rs:
            print("  %-14s %8s" % ("%g-%g m" % (lo_d, hi_d), "none")); continue
        a = np.array(rs); d = a[:, 2]
        print("  %-14s %8d %10.3g %10.3g %9.2f %8.2f" % (
            "%g-%g m" % (lo_d, hi_d), len(rs), 10 ** np.median(a[:, 0]),
            10 ** np.median(a[:, 1]), 10 ** np.median(d),
            float(np.median(np.abs(d - np.median(d))))))


if __name__ == "__main__":
    main()
