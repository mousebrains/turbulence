#!/usr/bin/env python3
"""MR SN 134 clock offset against glider doug's flight-computer clock (Taiwan 2017).

Copied from Taiwan13/gliders/20130526_husker/flight/mr_clock.py; only the paths
and the clock-block split differ. Its docstring below applies unchanged.


    python mr_clock.py extract   # MR .P -> mr1hz/*.npz (1 Hz P + accels), resumable
    python mr_clock.py solve     # -> mr_clock_offsets.csv + mr_clock_model.json

Run with microstructure-tpw that can read MR046's config vintage (one named
stanza per channel: [pitch], [shear1], ...). That is main at or after 29f3ea9
(PR #196, merged 2026-09-16); v0.4.0 and earlier build ZERO channels from
these files. (doug's own MR uses modern [channel] stanzas
and reads with any version; this note was copied from husker.)

DEFINITION.  glider_time = MR_time + L.  perturb's hotel.time_offset is ADDED to
hotel timestamps to put them on the instrument clock, so time_offset = -L. This
deployment needs two different L's plus a drift, which one scalar cannot hold,
so make_flight_speed.py writes the hotel file already on the MR clock.

METHOD.  For a trial L, glider m_pressure (dbar) interpolated at MR_time + L is
regressed on MR pressure (fitting the two sensors' offset and gain) and the
residual RMS is the cost. A glider yo is a sawtooth with a turn every ~20-45
min, so unlike a monotonic ramp a shifted copy does NOT fit, and the minimum is
sharp. Three gates, all required:
  * sharpness  = median cost over the +-3600 s search / best cost  > 5
  * the best lag is not on the search boundary
  * ambiguity  = best cost outside +-300 s of the minimum / best cost > 3 --
    the yo is periodic, so a short file can fit one cycle away. FSA_011
    (1174 s of overlap) did exactly that: -2715.5 s, one ~45-min cycle from -2.
Files failing a gate get no per-file value; they take the block model.

MODEL.  Per clock block (the MR clock was evidently reset during the
2013-05-03..05-17 gap), L(t) = a + b (t - t_ref), fitted to the gated per-file
values by least squares.
"""
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "microrider"
CACHE = HERE / "mr1hz"
FLIGHT = HERE / "flight_inputs.nc"
# One continuous deployment: a single clock block unless the per-file lags say otherwise.
BLOCK_SPLIT = np.datetime64("2100-01-01T00:00:00", "s").astype(float)


def extract():
    import odas_tpw
    from odas_tpw.rsi.p_file import PFile

    CACHE.mkdir(exist_ok=True)
    files = sorted(SRC.glob("*.P"))
    for i, f in enumerate(files, 1):
        out = CACHE / (f.stem + ".npz")
        if out.exists():
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pf = PFile(f)
            if "P" not in pf.channels:
                raise RuntimeError(f"no P channel -- is this odas_tpw ({odas_tpw.__file__}) able to read the named-stanza config?")
            t0 = pf.start_time.timestamp()
            ts, tf = np.asarray(pf.t_slow, float), np.asarray(pf.t_fast, float)
            sec = np.floor(ts).astype(int)
            n = int(sec.max()) + 1
            cnt = np.bincount(sec, minlength=n)[:n]
            P1 = np.bincount(sec, weights=np.asarray(pf.channels["P"], float), minlength=n)[:n] / np.maximum(cnt, 1)
            secf = np.floor(tf).astype(int)
            cntf = np.bincount(secf, minlength=n)[:n]
            acc = {k: np.bincount(secf, weights=np.asarray(pf.channels[k], float), minlength=n)[:n] / np.maximum(cntf, 1)
                   for k in ("Ax", "Ay", "Az") if k in pf.channels}  # doug's MR has no Az
            np.savez(out, t=t0 + np.arange(n) + 0.5, P=P1, valid=cnt > 0, start=t0, **acc)
            print(f"[{i}/{len(files)}] {f.name} {n} s", flush=True)
        except Exception as exc:  # record and continue
            print(f"[{i}/{len(files)}] {f.name} FAILED: {exc}", flush=True)


def _glider():
    import xarray as xr

    g = xr.open_dataset(FLIGHT)
    tg = g["m_present_time"].values
    tg = tg.astype("datetime64[ns]").astype("int64") / 1e9 if np.issubdtype(tg.dtype, np.datetime64) else tg.astype(float)
    Pg = g["m_pressure"].values * 10.0
    ok = np.isfinite(tg) & np.isfinite(Pg)
    return tg[ok], Pg[ok]


def solve():
    tg, Pg = _glider()
    gap_ok = np.r_[np.diff(tg) < 30, False]

    def cost(tm, Pm, L):
        tq = tm + L
        i = np.searchsorted(tg, tq) - 1
        inside = (i >= 0) & (i < tg.size - 1)
        inside[inside] &= gap_ok[i[inside]]
        if inside.sum() < 600:
            return np.nan, 0
        ii = i[inside]
        w = (tq[inside] - tg[ii]) / (tg[ii + 1] - tg[ii])
        pg = Pg[ii] * (1 - w) + Pg[ii + 1] * w
        A = np.c_[np.ones(inside.sum()), Pm[inside]]
        coef, *_ = np.linalg.lstsq(A, pg, rcond=None)
        return float(np.sqrt(np.mean((pg - A @ coef) ** 2))), int(inside.sum())

    rows = []
    for f in sorted(CACHE.glob("*.npz")):
        d = np.load(f)
        sel = d["valid"] & np.isfinite(d["P"]) & (d["P"] > 3)
        tm, Pm = d["t"][sel], d["P"][sel]
        row = dict(file=f.stem, t_mid=float(np.median(tm)) if tm.size else np.nan, status="", lag_s=np.nan,
                   rms_dbar=np.nan, sharpness=np.nan, ambiguity=np.nan, n_overlap=0)
        if tm.size < 1800 or np.ptp(Pm) < 40:
            row["status"] = "skip: <1800 s in water or no yo"
            rows.append(row)
            continue
        coarse = np.arange(-3600.0, 3600.1, 10.0)
        c = np.array([cost(tm, Pm, L)[0] for L in coarse])
        if np.all(np.isnan(c)):
            row["status"] = "skip: no glider overlap"
            rows.append(row)
            continue
        k = int(np.nanargmin(c))
        if k in (0, coarse.size - 1):
            row.update(status="REFUSED: boundary", lag_s=coarse[k])
            rows.append(row)
            continue
        fine = np.arange(coarse[k] - 15, coarse[k] + 15.01, 0.25)
        cf = np.array([cost(tm, Pm, L)[0] for L in fine])
        j = int(np.nanargmin(cf))
        best, rms = float(fine[j]), float(cf[j])
        far = np.abs(coarse - best) > 300
        sharp = float(np.nanmedian(c) / rms)
        amb = float(np.nanmin(c[far]) / rms)
        status = "ok" if (sharp > 5 and amb > 3) else f"REFUSED: sharpness {sharp:.1f} ambiguity {amb:.1f}"
        row.update(status=status, lag_s=best, rms_dbar=rms, sharpness=sharp, ambiguity=amb, n_overlap=cost(tm, Pm, best)[1])
        rows.append(row)

    ok = [r for r in rows if r["status"] == "ok"]
    model = {"definition": "glider_time = MR_time + lag_s; perturb hotel.time_offset would be -lag_s", "blocks": {}}
    for name, in_block in (("A_2017-02-16..28", lambda t: t < BLOCK_SPLIT), ("B_unused", lambda t: t >= BLOCK_SPLIT)):
        r = [x for x in ok if in_block(x["t_mid"])]
        if not r:
            continue
        t = np.array([x["t_mid"] for x in r]); L = np.array([x["lag_s"] for x in r])
        t_ref = float(np.median(t))
        if len(r) >= 3 and np.ptp(t) > 86400:
            b, a = np.polyfit((t - t_ref) / 86400.0, L, 1)
        else:
            b, a = 0.0, float(np.median(L))
        res = L - (a + b * (t - t_ref) / 86400.0)
        model["blocks"][name] = dict(n_files=len(r), t_ref_epoch=t_ref, lag_at_ref_s=float(a), drift_s_per_day=float(b),
                                     residual_rms_s=float(np.sqrt(np.mean(res**2))), t_min_epoch=float(t.min()), t_max_epoch=float(t.max()),
                                     split_epoch=float(BLOCK_SPLIT))
    with open(HERE / "mr_clock_offsets.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (HERE / "mr_clock_model.json").write_text(json.dumps(model, indent=2) + "\n")
    for r in rows:
        print(f"{r['file']:9s} {r['status']:38s} {r['lag_s']:9.2f} {r['rms_dbar']:6.3f} {r['sharpness']:7.1f} {r['ambiguity']:6.1f} {r['n_overlap']:6d}")
    print(json.dumps(model, indent=2))


if __name__ == "__main__":
    {"extract": extract, "solve": solve}[sys.argv[1]]()
