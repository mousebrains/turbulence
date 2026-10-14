# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""T1/T2 spectral-ratio fits for every BEAD PAIR in a set of deployments.

usage: run_fleet2.py OUTDIR LABEL=GLOB [LABEL=GLOB ...]

Groups windows by (deployment, instrument, T1 serial, T2 serial) — probes move
between instruments and are swapped mid-cruise, so the instrument alone is
not a bead identity — and splits each pair's files into time halves (a tau
that changes between halves is drift: fouling or damage). Serials come from
the chi file's source: a perturb per-profile NetCDF (`configuration_string`)
or a .p file. Each deployment appends to OUTDIR/fleet.csv and fleet.txt when
it finishes and writes a DONE marker, so a rerun skips finished deployments.
"""
import csv
import glob
import os
import re
import sys
import warnings
from datetime import UTC, datetime

import netCDF4

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import ratio as R  # noqa: E402

from odas_tpw.rsi.p_file import parse_config, read_config_string  # noqa: E402

_cfg_cache: dict = {}


def therm_serials(src: str) -> dict:
    """{channel name: sn} for therm channels in the source's config."""
    if src in _cfg_cache:
        return _cfg_cache[src]
    out = {}
    try:
        if src.lower().endswith(".nc"):
            with netCDF4.Dataset(src) as s:
                text = getattr(s, "configuration_string", "")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                text = read_config_string(src)
        cfg = parse_config(text)
        out = {ch.get("name"): str(ch.get("sn", "?")).strip() or "?"
               for ch in cfg.get("channels", []) if str(ch.get("type", "")).lower() == "therm"}
    except Exception:
        out = {}
    _cfg_cache[src] = out
    return out


def file_meta(path):
    with netCDF4.Dataset(path) as ds:
        if "spec_gradT" not in ds.variables or ds["spec_gradT"].shape[0] < 2:
            return None
        sn = getattr(ds, "instrument_sn", None) or getattr(ds, "sn", None)
        if not sn:
            m = re.search(r"SN(\d+)", os.path.basename(path))
            sn = m.group(1) if m else "?"
        probes = None
        if "probe" in ds.variables:
            pv = ds["probe"][:]
            try:
                probes = [str(x) for x in (netCDF4.chartostring(pv) if pv.dtype.kind == "S" else pv)]
            except Exception:
                probes = None
        start = getattr(ds, "start_time", None)
        src = getattr(ds, "source", "")
    names = [re.sub(r"_d\w+$", "", p) for p in (probes or ["T1_dT1", "T2_dT2"])[:2]]
    sers = therm_serials(src) if src else {}
    pair = tuple(sers.get(n, "?") for n in names)
    t0 = None
    if start:
        try:
            t0 = datetime.fromisoformat(str(start).replace("Z", "+00:00")).timestamp()
        except ValueError:
            t0 = None
    return {"inst": f"SN{sn}", "pair": pair, "names": tuple(names), "t0": t0}


def merge(groups):
    out = R.Group(F=groups[0].F.copy())
    for g in groups:
        for d in ("d1", "d2", "ok", "n"):
            getattr(out, d).update(getattr(g, d))
        out.W.extend(g.W)
    return out


def main():
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    txt, csvp = os.path.join(outdir, "fleet.txt"), os.path.join(outdir, "fleet.csv")
    done = set()
    if os.path.exists(txt):
        done = {ln.split()[1] for ln in open(txt) if ln.startswith("DONE ")}
    new_csv = not os.path.exists(csvp)
    for spec in sys.argv[2:]:
        label, pattern = spec.split("=", 1)
        if label in done:
            print("skip", label, flush=True)
            continue
        files = sorted(glob.glob(pattern))
        meta = {}
        for f in files:
            try:
                m = file_meta(f)
            except OSError:
                m = None
            if m:
                meta[f] = m
        # halves per (instrument, pair), by start time (file order if absent)
        by = {}
        for f, m in meta.items():
            by.setdefault((m["inst"], m["pair"]), []).append(f)
        half = {}
        for k, fl in by.items():
            fl = sorted(fl, key=lambda f: (meta[f]["t0"] or 0, f))
            for i, f in enumerate(fl):
                half[f] = "H1" if i < len(fl) / 2 else "H2"

        def key(path, ds, j):
            m = meta.get(path)
            if m is None:
                return None
            return (label, m["inst"], "/".join(m["pair"]), half[path])

        groups = R.collect(pattern, key)
        # Cache every group's per-file sums so later analyses (ratio curves at
        # fixed f, band studies) run locally instead of re-reading SMB.
        import pickle
        with open(os.path.join(outdir, f"{label}_groups.pkl"), "wb") as fh:
            pickle.dump({"groups": groups, "meta": meta}, fh)
        pairs = {}
        for g in groups:
            pairs.setdefault(g[:3], []).append(g)
        results = []
        for base, gl in sorted(pairs.items()):
            for g in sorted(gl):
                results.append((g, groups[g]))
            if len(gl) > 1:
                results.append((base + ("all",), merge([groups[g] for g in gl])))
        lines, rows = [], []
        for g, grp in results:
            r = R.analyse(g, grp, n_boot=200)
            if r is None:
                continue
            lines.append(R.fmt(r))
            fl = list(grp.d1)
            t0s = [meta[f]["t0"] for f in fl if meta.get(f) and meta[f]["t0"]]
            row = {"deployment": g[0], "instrument": g[1], "pair": g[2], "half": g[3],
                   "W": round(r.W, 3), "windows": r.n_windows, "files": r.n_files,
                   "f_lo": round(r.f_band[0], 2), "f_hi": round(r.f_band[1], 2),
                   "pole_pref": r.preferred or "",
                   "t_start": datetime.fromtimestamp(min(t0s), UTC).date().isoformat() if t0s else "",
                   "t_end": datetime.fromtimestamp(max(t0s), UTC).date().isoformat() if t0s else ""}
            for p, pf in r.fits.items():
                row.update({f"p{p}_tau1_ms": round(pf.tau1 * 1e3, 2), f"p{p}_tau1_lo": round(pf.ci1[0] * 1e3, 2),
                            f"p{p}_tau1_hi": round(pf.ci1[1] * 1e3, 2), f"p{p}_tau2_ms": round(pf.tau2 * 1e3, 2),
                            f"p{p}_tau2_lo": round(pf.ci2[0] * 1e3, 2), f"p{p}_tau2_hi": round(pf.ci2[1] * 1e3, 2),
                            f"p{p}_rms": round(pf.rms, 4), f"p{p}_ident": int(pf.identifiable)})
            rows.append(row)
        with open(txt, "a") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))
            fh.write(f"DONE {label} {len(files)} files, {len(meta)} with two probes\n")
        if rows:
            with open(csvp, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                if new_csv:
                    w.writeheader()
                    new_csv = False
                w.writerows(rows)
        print(f"DONE {label}: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
