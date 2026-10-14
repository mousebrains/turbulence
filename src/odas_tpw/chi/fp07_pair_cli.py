# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""``fp07-pair``: measure per-bead FP07 response corrections from chi products.

    fp07-pair fit 'results/chi_00/*.nc' -o fp07pair/   # one .nc per bead pair and segment
    fp07-pair show fp07pair/*.nc                       # what a correction contains

Then list the files under the instrument in perturb's config::

    instruments:
      SN194:
        fp07_response: [<CONFIG_DIR>/fp07pair/fp07pair_SN194_T1592-T2005_20220321T0359Z.nc]

and rerun (chi, chi_binned and chi_combo re-version; epsilon does not).
See docs/fp07pair/runbook.md.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

from odas_tpw.chi import fp07_pair as P


def _fit(args: argparse.Namespace) -> int:
    paths: list[str] = []
    for pat in args.chi:
        hits = sorted(glob.glob(pat))
        paths.extend(hits if hits else [pat])
    if not paths:
        print("no chi files", file=sys.stderr)
        return 2
    res = P.estimate(paths, anchor=args.anchor, min_files=args.min_files,
                     segment=not args.no_segment, allow_amplitude=args.allow_amplitude)
    out = Path(args.output).resolve()  # absolute in the printed config snippet
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, list[str]] = {}
    for corr in res.corrections:
        name = P.output_name(corr)
        if any(name in names for names in written.values()):
            raise RuntimeError(f"two corrections map to one file name {name}; refusing to "
                               "overwrite one with the other")
        corr.to_netcdf(out / name)
        written.setdefault(corr.instrument, []).append(name)
    lines = list(res.report)
    if res.refused:
        lines.append("REFUSED (a *_notapplied.nc record covers each refused span: its files run")
        lines.append("UNCORRECTED and say why; a refusal before a span exists writes nothing):")
        lines += [f"  {r}" for r in res.refused]
    lines.append("")
    lines.append("perturb config (key each block by the perturb `instruments:` key for that")
    lines.append("instrument -- the RAW DIRECTORY NAME, which may differ from the SN below):")
    lines.append("instruments:")
    for inst, names in sorted(written.items()):
        lines.append(f"  {inst}:")
        lines.append("    fp07_response: [" + ", ".join(str(out / n) for n in names) + "]")
    text = "\n".join(lines)
    (out / "fp07pair_report.txt").write_text(text + "\n")
    print(text)
    return 0 if any(c.applies for c in res.corrections) else 1


def _show(args: argparse.Namespace) -> int:
    for p in args.files:
        c = P.ResponseCorrection.from_netcdf(p)
        print(f"{p}")
        print(f"  {c.instrument}: {c.channels[0]}={c.serials[0]}, {c.channels[1]}={c.serials[1]}")
        print(f"  span {P._iso(c.t_start)} .. {P._iso(c.t_end)}; {c.n_files} files, "
              f"{c.n_windows} windows")
        if not c.applies:
            print(f"  NOT APPLIED (files in this span run uncorrected): {c.reason}")
            print(f"  digest {c.digest()}")
            continue
        print(f"  band {c.f_lo:.2f}-{c.f_hi:.1f} Hz at W_ref {c.W_ref:.3f} m/s ({c.tau_model}); "
              f"anchor {c.anchor}; ln G {c.ln_G:+.3f}; flags {','.join(c.flags) or '-'}")
        for f0 in (2, 5, 10, 20, 40, 60):
            if c.f_lo <= f0 <= c.f_hi * 1.5:
                e = [float(c.factor(k, np.array([f0]), c.W_ref)[0]) for k in (0, 1)]
                print(f"    {f0:3d} Hz: e = {e[0]:.3f} / {e[1]:.3f}")
        print(f"  digest {c.digest()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fp07-pair", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="estimate corrections from chi products")
    f.add_argument("chi", nargs="+", help="chi product NetCDFs or glob patterns")
    f.add_argument("-o", "--output", required=True, help="output directory")
    f.add_argument("--anchor", choices=P.ANCHORS, default="geomean",
                   help="which response stays the default model (geomean: neither, the pair's "
                        "average does)")
    f.add_argument("--min-files", type=int, default=8)
    f.add_argument("--no-segment", action="store_true",
                   help="one correction per pair, even if the pair changed in time")
    f.add_argument("--allow-amplitude", action="store_true",
                   help="write corrections even for pairs whose flat gain exceeds ln 2")
    f.set_defaults(func=_fit)
    s = sub.add_parser("show", help="print what a correction contains")
    s.add_argument("files", nargs="+")
    s.set_defaults(func=_show)
    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
