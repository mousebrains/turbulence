# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""``mr-clocksync`` --- per-file clock offsets against a reference pressure record.

    mr-clocksync init    -o clocksync.yaml     commented template
    mr-clocksync probe   -c clocksync.yaml     what is in the band? run this first
    mr-clocksync extract -c clocksync.yaml     .p trees -> cache (resumable)
    mr-clocksync solve   -c clocksync.yaml     per-file offset + rate -> CSV
    mr-clocksync report  -c clocksync.yaml     read the CSV back, summarize
"""

from __future__ import annotations

import argparse
import glob as globmod
import sys
from pathlib import Path

import numpy as np

from odas_tpw.clocksync.config import load_config, resolve, write_template
from odas_tpw.clocksync.extract import extract_tree, load_cached
from odas_tpw.clocksync.fit import fit_file
from odas_tpw.clocksync.lag import band_power_fraction, bandpass
from odas_tpw.clocksync.qc import check_continuity, summarize
from odas_tpw.clocksync.reference import load_reference
from odas_tpw.clocksync.report import coverage_line, text_report, write_csv
from odas_tpw.clocksync.series import epoch_to_iso


def _sources(cfg: dict, tgt: dict) -> list[Path]:
    pat = str(resolve(cfg, tgt["glob"]))
    return sorted(Path(p) for p in globmod.glob(pat))


def _reference(cfg: dict):
    r = dict(cfg["reference"])
    return load_reference(
        resolve(cfg, r.pop("file")),
        kind=r.get("kind", "auto"),
        time_var=r.get("time_var", "T"),
        pressure_var=r.get("pressure_var", "P"),
        time_units=r.get("time_units", "auto"),
        name=r.get("name", "reference"),
        scale=float(r.get("scale", 1.0)),
        offset=float(r.get("offset", 0.0)),
    )


def cmd_init(args) -> int:
    p = write_template(args.output)
    print(f"wrote {p}")
    print("Edit reference.file and targets:, then run 'mr-clocksync probe'.")
    return 0


def cmd_probe(args) -> int:
    """Is there a usable wave signal, and where is the band?

    Run before anything else.  It costs one file per target and it answers the
    only question that decides whether this method applies at all.
    """
    cfg = load_config(args.config)
    an = cfg["analysis"]
    band = tuple(an["band"])
    ref = _reference(cfg)
    print(f"reference: {ref.name}  {ref.source}")
    rt0, rt1 = ref.span
    print(coverage_line(ref.name, rt0, rt1, 1) + f"  nominal {ref.fs:.4f} Hz")
    u = ref.uniform(an["rate"], max_gap=an["max_gap"])
    good = np.isfinite(u.p)
    if good.sum() > 1024:
        seg = u.p[good][: int(an["rate"] * 3600)]
        print(f"  band {band[0]}-{band[1]} Hz holds "
              f"{100 * band_power_fraction(seg, an['rate'], band):.1f}% of the variance; "
              f"band-passed rms {1000 * np.std(bandpass(seg, an['rate'], *band)):.1f} mm")
    for tgt in cfg["targets"]:
        src = _sources(cfg, tgt)
        if not src:
            print(f"\ntarget {tgt['name']}: NO FILES match {tgt['glob']}")
            continue
        from odas_tpw.clocksync.extract import read_pfile_pressure

        s = read_pfile_pressure(src[len(src) // 2], rate=an["rate"],
                                channel=tgt.get("channel", "P"), name=tgt["name"])
        print(f"\ntarget {tgt['name']}: {len(src)} files, sampled {Path(s.source).name}")
        st0, st1 = s.span
        print(coverage_line(tgt["name"], st0, st1, len(src)))
        p = s.p[np.isfinite(s.p)]
        if p.size > 1024:
            print(f"  band {band[0]}-{band[1]} Hz holds "
                  f"{100 * band_power_fraction(p, an['rate'], band):.1f}% of the variance; "
                  f"band-passed rms {1000 * np.std(bandpass(p, an['rate'], *band)):.1f} mm"
                  f"   (gate is {1000 * cfg['gates']['min_amplitude']:.0f} mm)")
    return 0


def cmd_extract(args) -> int:
    cfg = load_config(args.config)
    cache = resolve(cfg, cfg["cache"])
    rate = float(cfg["analysis"]["rate"])
    for tgt in cfg["targets"]:
        src = _sources(cfg, tgt)
        if not src:
            print(f"{tgt['name']}: NO FILES match {tgt['glob']}", file=sys.stderr)
            continue
        print(f"{tgt['name']}: {len(src)} files -> {cache / tgt['name']}")
        extract_tree(src, cache / tgt["name"], rate,
                     channel=tgt.get("channel", "P"), force=args.force)
    return 0


def cmd_solve(args) -> int:
    cfg = load_config(args.config)
    an, gates, cont = cfg["analysis"], cfg["gates"], cfg["continuity"]
    cache = resolve(cfg, cfg["cache"])
    outdir = resolve(cfg, cfg["output"])
    ref = _reference(cfg)
    print(f"reference {ref.name}: {epoch_to_iso(ref.span[0])} -> {epoch_to_iso(ref.span[1])}")

    fits, summaries = {}, {}
    for tgt in cfg["targets"]:
        name = tgt["name"]
        entries = sorted((cache / name).glob("*.npz"))
        if not entries:
            print(f"{name}: cache empty -- run 'mr-clocksync extract' first", file=sys.stderr)
            continue
        print(f"\n{name}: {len(entries)} cached files")
        lst = []
        for i, e in enumerate(entries, 1):
            s = load_cached(e)
            # Clip the reference to this file plus the search width, so a 26-day
            # reference does not get gridded 431 times.
            r = ref.clip(s.span[0] - an["max_lag"] - 60, s.span[1] + an["max_lag"] + 60)
            if r.t.size < 1024:
                lst.append(_empty(s))
                continue
            f = fit_file(
                r, s, float(an["rate"]),
                window=float(an["window"]),
                step=an["step"],
                band=tuple(an["band"]),
                max_lag=float(an["max_lag"]),
                max_gap=float(an["max_gap"]),
                fit_rate=bool(an["fit_rate"]),
                min_windows=int(an["min_windows"]),
                min_amplitude=float(gates["min_amplitude"]),
                min_dynamic_range=float(gates["min_dynamic_range"]),
                min_coherence=float(gates["min_coherence"]),
                max_chi2=float(gates["max_chi2"]),
                coh_min=float(gates["coh_min"]),
            )
            f.source = s.source or str(e)
            lst.append(f)
            if i % 20 == 0 or i == len(entries):
                state = f"{f.offset:+.3f} s" if f.ok else "unsolved"
                print(f"  [{i}/{len(entries)}] {e.stem}: {state}")
        check_continuity(lst, n_sigma=float(cont["n_sigma"]), window=int(cont["window"]),
                         wave_period=cont.get("wave_period"))
        fits[name] = lst
        summaries[name] = summarize(lst)

    if not fits:
        print("nothing solved", file=sys.stderr)
        return 1
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv(outdir / "clock_offsets.csv", fits)
    report = text_report(fits, summaries)
    (outdir / "clock_report.txt").write_text(report)
    print(report)
    print(f"\nwrote {csv_path}")
    print(f"wrote {outdir / 'clock_report.txt'}")
    return 0


def _empty(s):
    from odas_tpw.clocksync.fit import FileFit

    return FileFit(name=s.name, source=s.source, t0=s.span[0], t1=s.span[1],
                   why="no overlapping reference data")


def cmd_report(args) -> int:
    cfg = load_config(args.config)
    path = resolve(cfg, cfg["output"]) / "clock_report.txt"
    if not path.exists():
        print(f"{path} does not exist -- run 'mr-clocksync solve'", file=sys.stderr)
        return 1
    print(path.read_text())
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mr-clocksync", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="write a commented config template")
    p.add_argument("-o", "--output", default="clocksync.yaml")
    p.set_defaults(func=cmd_init)

    for nm, fn, helptext in (
        ("probe", cmd_probe, "is there a usable wave signal? run this first"),
        ("extract", cmd_extract, "cache pressure from the .p trees (resumable)"),
        ("solve", cmd_solve, "per-file offset and rate -> CSV + report"),
        ("report", cmd_report, "print the last report"),
    ):
        p = sub.add_parser(nm, help=helptext)
        p.add_argument("-c", "--config", required=True)
        if nm == "extract":
            p.add_argument("--force", action="store_true",
                           help="rebuild cache entries even if they look current")
        p.set_defaults(func=fn)

    args = ap.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
