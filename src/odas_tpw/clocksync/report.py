# Aug-2026, Claude and Pat Welch, pat@mousebrains.com
"""Write the solution out, and say plainly what it is worth.

The solution table is the product; the report exists so that a number is never
used without its uncertainty and its provenance in view.  Rejected windows are
counted and their reasons listed rather than quietly dropped --- a file solved
from 2 of 12 windows and one solved from 12 of 12 carry the same sigma column
and very different weight.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from odas_tpw.clocksync.fit import FileFit
from odas_tpw.clocksync.series import epoch_to_iso

COLUMNS = [
    "target", "file", "t_start", "t_end", "offset_s", "offset_sigma_s",
    "rate_s_per_s", "rate_sigma_s_per_s", "drift_s_per_day", "cov_offset_rate",
    "chi2", "coherence", "n_windows", "n_used", "ok", "flags", "why",
]


def write_csv(path: Path, fits: dict[str, list[FileFit]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for target, lst in fits.items():
            for f in sorted(lst, key=lambda x: x.t0):
                w.writerow([
                    target, Path(f.source).name if f.source else f.name,
                    f"{f.t0:.6f}", f"{f.t1:.6f}",
                    f"{f.offset:.6f}", f"{f.offset_sigma:.6f}",
                    f"{f.rate:.6e}", f"{f.rate_sigma:.6e}",
                    f"{f.drift_per_day:.6f}", f"{f.covariance:.6e}",
                    f"{f.chi2:.3f}", f"{f.coherence:.4f}",
                    f.n_windows, f.n_used, int(f.ok),
                    " | ".join(f.flags), f.why,
                ])
    return path


def text_report(fits: dict[str, list[FileFit]], summaries: dict[str, dict]) -> str:
    out: list[str] = []
    for target, lst in fits.items():
        s = summaries.get(target, {})
        rule = "=" * 100
        head = f"{target}  --  {s.get('n_solved', 0)}/{s.get('n_files', 0)} files solved"
        out.append(f"\n{rule}\n{head}\n{rule}")
        if not s.get("n_solved"):
            bad = {f.why for f in lst if f.why}
            out.append("  nothing solved. reasons: " + ("; ".join(sorted(bad)) or "unknown"))
            continue
        out.append(
            f"  offset   median {s['offset_median']:+9.3f} s   MAD {s['offset_mad']:7.3f} s"
            f"   range [{s['offset_min']:+.3f}, {s['offset_max']:+.3f}] s"
        )
        out.append(
            f"  sigma    median {s['sigma_median'] * 1000:9.1f} ms  worst "
            f"{s['sigma_max'] * 1000:.1f} ms"
        )
        if "rate_median" in s:
            out.append(
                f"  drift    median {s['drift_median_s_per_day']:+9.3f} s/day"
                f"   MAD {s['rate_mad'] * 86400:.3f} s/day"
            )
        out.append(
            f"  step     |offset[i]-offset[i-1]| median {s['step_median']:.3f} s"
            f"   max {s['step_max']:.3f} s"
        )
        out.append(
            f"  windows  {s['windows_used']}/{s['windows_total']} used"
            f"   median coherence {s['coherence_median']:.3f}"
        )
        flagged = [f for f in lst if f.flags]
        if flagged:
            out.append(f"\n  {len(flagged)} file(s) flagged:")
            for f in sorted(flagged, key=lambda x: x.t0)[:20]:
                nm = Path(f.source).name if f.source else f.name
                for msg in f.flags:
                    out.append(f"    {nm}  {msg}")
            if len(flagged) > 20:
                out.append(f"    ... and {len(flagged) - 20} more (see the CSV)")
        rejected = [f for f in lst if not f.ok]
        if rejected:
            out.append(f"\n  {len(rejected)} file(s) unsolved:")
            for f in sorted(rejected, key=lambda x: x.t0)[:10]:
                nm = Path(f.source).name if f.source else f.name
                out.append(f"    {nm}  {f.why}")
            if len(rejected) > 10:
                out.append(f"    ... and {len(rejected) - 10} more")
    return "\n".join(out)


def coverage_line(name: str, t0: float, t1: float, n: int) -> str:
    return (
        f"  {name:<12s} {epoch_to_iso(t0)} -> {epoch_to_iso(t1)}"
        f"  ({(t1 - t0) / 86400:.2f} d, {n} files)"
    )


def apply_corrections(fits: list[FileFit], t: np.ndarray) -> np.ndarray:
    """Corrected epoch times for samples *t*, using each sample's own file fit.

    Samples outside every solved file come back NaN rather than being pushed
    through the nearest neighbour's model: an unsolved file is unsolved, and
    extrapolating a clock across a jump it was never fitted over would invent
    exactly the error this package removes.
    """
    t = np.asarray(t, dtype=np.float64)
    out = np.full(t.shape, np.nan)
    for f in fits:
        if not f.ok:
            continue
        m = (t >= f.t0) & (t <= f.t1)
        if m.any():
            out[m] = t[m] + f.correction(t[m])
    return out
