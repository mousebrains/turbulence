# Oct-2026, Claude and Pat Welch, pat@mousebrains.com
"""T1/T2 spectral-ratio fits per deployment, instrument and half-deployment.

usage: run_fleet.py OUT.txt LABEL=GLOB [LABEL=GLOB ...]

Groups windows by (label, instrument, half) where the half splits each
instrument's files in filename (time) order — a tau that changes between
halves is fouling or damage, not a bead constant. Results append to OUT.txt
as each deployment finishes (checkpoint), so a rerun can skip done labels.
"""
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ratio as R


def instrument(path, ds):
    sn = getattr(ds, "instrument_sn", None) or getattr(ds, "sn", None)
    if sn:
        return f"SN{sn}"
    m = re.search(r"SN(\d+)", os.path.basename(path))
    return f"SN{m.group(1)}" if m else "SN?"


def main():
    out = sys.argv[1]
    done = set()
    if os.path.exists(out):
        done = {ln.split("|")[0].strip() for ln in open(out) if ln.startswith("DONE ")}
    for spec in sys.argv[2:]:
        label, pattern = spec.split("=", 1)
        if f"DONE {label}" in done:
            print("skip", label)
            continue
        files = sorted(glob.glob(pattern))
        # half per instrument, by file order
        by_inst: dict = {}
        import netCDF4
        inst_of = {}
        for f in files:
            try:
                with netCDF4.Dataset(f) as ds:
                    inst_of[f] = instrument(f, ds)
            except OSError:
                continue
            by_inst.setdefault(inst_of[f], []).append(f)
        half_of = {}
        for inst, fl in by_inst.items():
            for i, f in enumerate(fl):
                half_of[f] = "H1" if i < len(fl) / 2 else "H2"

        def key(path, ds, j, _inst=inst_of, _half=half_of):
            if path not in _inst:
                return None
            return (label, _inst[path], _half[path])

        def key_all(path, ds, j, _inst=inst_of):
            return (label, _inst[path], "all") if path in _inst else None

        groups = R.collect(pattern, key)
        groups.update(R.collect(pattern, key_all))
        lines = []
        for g in sorted(groups):
            r = R.analyse(g, groups[g], n_boot=200)
            if r is not None:
                lines.append(R.fmt(r))
        with open(out, "a") as fh:
            fh.write("\n".join(lines) + "\n")
            fh.write(f"DONE {label} | {len(files)} files\n")
        print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
