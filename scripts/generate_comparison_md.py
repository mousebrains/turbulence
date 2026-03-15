#!/usr/bin/env python3
"""Generate docs/pyturb/comparison.md from pre-computed pyturb vs pyturb-cli results.

Reads the NetCDF output from both Jesse's pyturb and pyturb-cli (produced by
compare_pyturb.py) and generates a structured markdown comparison report.

Usage:
    python scripts/generate_comparison_md.py
    python scripts/generate_comparison_md.py --work-dir /tmp/pyturb_compare_all
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import numpy as np
import xarray as xr

JESSE_DIR_DEFAULT = "/tmp/pyturb_compare_all/jesse_eps"
TPW_DIR_DEFAULT = "/tmp/pyturb_compare_all/tpw_eps"
VMP_DIR_DEFAULT = "VMP"
OUTPUT_DEFAULT = "docs/pyturb/comparison.md"


def _index(path: Path) -> int:
    name = path.stem
    return int(name.rsplit("_p", 1)[-1])


def _file_number(stem: str) -> str:
    """Extract file number like '0026' from stem."""
    parts = stem.rsplit("_", 1)
    return parts[-1] if len(parts) > 1 else stem


def compare_one_profile(jesse_path: Path, tpw_path: Path) -> dict:
    """Compare a single profile pair, return stats dict."""
    row: dict = {}
    try:
        ds_j = xr.open_dataset(jesse_path)
        ds_t = xr.open_dataset(tpw_path)
    except Exception as e:
        row["status"] = "error"
        row["note"] = str(e)
        return row

    row["n_jesse"] = ds_j.sizes.get("time", 0)
    row["n_tpw"] = ds_t.sizes.get("time", 0)

    if "pressure" not in ds_j or "pressure" not in ds_t:
        row["status"] = "no_pressure"
        ds_j.close()
        ds_t.close()
        return row

    p_j = ds_j["pressure"].values
    p_t = ds_t["pressure"].values
    row["pres_range_j"] = (float(np.nanmin(p_j)), float(np.nanmax(p_j)))
    row["pres_range_t"] = (float(np.nanmin(p_t)), float(np.nanmax(p_t)))

    for var in ["eps_1", "eps_2"]:
        if var not in ds_j or var not in ds_t:
            row[f"{var}_status"] = "missing"
            continue

        j_vals = ds_j[var].values
        t_vals = ds_t[var].values

        if len(j_vals) == 0 or len(t_vals) == 0:
            row[f"{var}_status"] = "empty"
            continue

        valid_t = np.isfinite(p_t) & np.isfinite(t_vals) & (t_vals > 0)
        valid_j = np.isfinite(p_j) & np.isfinite(j_vals) & (j_vals > 0)

        if valid_t.sum() < 2 or valid_j.sum() < 2:
            row[f"{var}_status"] = "no_valid"
            continue

        t_interp = np.interp(
            p_j[valid_j], p_t[valid_t], np.log10(t_vals[valid_t]),
            left=np.nan, right=np.nan,
        )
        j_log = np.log10(j_vals[valid_j])
        both_ok = np.isfinite(t_interp) & np.isfinite(j_log)

        if both_ok.sum() == 0:
            row[f"{var}_status"] = "no_overlap"
            continue

        log_ratio = t_interp[both_ok] - j_log[both_ok]
        row[f"{var}_mean_log_ratio"] = float(np.mean(log_ratio))
        row[f"{var}_std_log_ratio"] = float(np.std(log_ratio))
        row[f"{var}_max_abs_log_ratio"] = float(np.max(np.abs(log_ratio)))
        row[f"{var}_n_valid"] = int(both_ok.sum())
        row[f"{var}_median_ratio"] = float(10 ** np.median(log_ratio))

        med_abs = float(np.median(np.abs(log_ratio)))
        if med_abs < 0.15:
            row[f"{var}_status"] = "good"
        elif med_abs < 0.5:
            row[f"{var}_status"] = "fair"
        else:
            row[f"{var}_status"] = "poor"

    row["status"] = "compared"
    ds_j.close()
    ds_t.close()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jesse-dir", default=JESSE_DIR_DEFAULT)
    parser.add_argument("--tpw-dir", default=TPW_DIR_DEFAULT)
    parser.add_argument("--vmp-dir", default=VMP_DIR_DEFAULT)
    parser.add_argument("-o", "--output", default=OUTPUT_DEFAULT)
    args = parser.parse_args()

    jesse_dir = Path(args.jesse_dir)
    tpw_dir = Path(args.tpw_dir)
    vmp_dir = Path(args.vmp_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Discover all .p file stems
    p_files = sorted(vmp_dir.glob("*.p"))
    stems = [f.stem for f in p_files]

    # Collect all results grouped by file
    file_results: dict[str, list[dict]] = {}
    for stem in stems:
        jesse_files = {_index(f): f for f in sorted(jesse_dir.glob(f"{stem}_p*.nc"))}
        tpw_files = {_index(f): f for f in sorted(tpw_dir.glob(f"{stem}_p*.nc"))}
        all_indices = sorted(set(jesse_files) | set(tpw_files))

        profiles: list[dict] = []
        for idx in all_indices:
            row = {"profile": idx, "stem": stem}
            if idx not in jesse_files:
                row["status"] = "tpw-only"
                profiles.append(row)
                continue
            if idx not in tpw_files:
                row["status"] = "jesse-only"
                profiles.append(row)
                continue
            row.update(compare_one_profile(jesse_files[idx], tpw_files[idx]))
            profiles.append(row)

        if profiles:
            file_results[stem] = profiles

    # Flatten for aggregate stats
    all_rows = [r for rows in file_results.values() for r in rows]
    compared = [r for r in all_rows if r.get("status") == "compared"]

    # Aggregate stats
    n_files = len(file_results)
    n_profiles_total = len(all_rows)
    n_compared = len(compared)
    n_tpw_only = sum(1 for r in all_rows if r.get("status") == "tpw-only")
    n_jesse_only = sum(1 for r in all_rows if r.get("status") == "jesse-only")

    def _var_stats(var: str) -> dict:
        log_ratios = [r[f"{var}_mean_log_ratio"] for r in compared if f"{var}_mean_log_ratio" in r]
        median_ratios = [r[f"{var}_median_ratio"] for r in compared if f"{var}_median_ratio" in r]
        statuses = [r.get(f"{var}_status", "?") for r in compared]
        if not log_ratios:
            return {}
        return {
            "mean_log": np.mean(log_ratios),
            "std_log": np.std(log_ratios),
            "mean_median_ratio": np.mean(median_ratios),
            "min_median_ratio": np.min(median_ratios),
            "max_median_ratio": np.max(median_ratios),
            "n_good": statuses.count("good"),
            "n_fair": statuses.count("fair"),
            "n_poor": statuses.count("poor"),
            "n_total": len(log_ratios),
        }

    eps1_stats = _var_stats("eps_1")
    eps2_stats = _var_stats("eps_2")

    # Per-file summary
    file_summaries: list[dict] = []
    for stem, rows in file_results.items():
        comp = [r for r in rows if r.get("status") == "compared"]
        fnum = _file_number(stem)
        s: dict = {"file": fnum, "n_profiles": len(rows), "n_compared": len(comp)}
        for var in ["eps_1", "eps_2"]:
            meds = [r[f"{var}_median_ratio"] for r in comp if f"{var}_median_ratio" in r]
            sts = [r.get(f"{var}_status", "?") for r in comp]
            if meds:
                s[f"{var}_mean_ratio"] = float(np.mean(meds))
                s[f"{var}_min_ratio"] = float(np.min(meds))
                s[f"{var}_max_ratio"] = float(np.max(meds))
            s[f"{var}_good"] = sts.count("good")
            s[f"{var}_fair"] = sts.count("fair")
            s[f"{var}_poor"] = sts.count("poor")
        file_summaries.append(s)

    # Build markdown
    lines: list[str] = []

    lines.append("# pyturb vs pyturb-cli: Quantitative Comparison")
    lines.append("")
    lines.append("Comparison of epsilon (TKE dissipation rate) estimates from")
    lines.append("[Jesse's pyturb](https://github.com/oceancascades/pyturb) and")
    lines.append("`pyturb-cli` (this repository) on ARCTERX VMP-250 data")
    lines.append("(SN 479, R/V Thompson, January 2025).")
    lines.append("")
    lines.append("Both tools were run with matching parameters:")
    lines.append("- `--fft-len 1.0` (512 samples at 512 Hz)")
    lines.append("- `--diss-len 4.0` (2048 samples at 512 Hz)")
    lines.append("- Goodman cleaning: **off** (`pyturb-cli` default matches pyturb)")
    lines.append("- Overlap: n_fft // 2 = 256 samples")
    lines.append("")
    lines.append("Epsilon values are compared by interpolating `log10(epsilon)` from")
    lines.append("pyturb-cli onto pyturb's pressure grid. The median ratio")
    lines.append("`pyturb-cli / pyturb` is reported per profile.")
    lines.append("")
    today = datetime.date.today().isoformat()
    lines.append(f"*Generated {today} from {len(p_files)} VMP .p files.*")
    lines.append("")

    # Overall summary
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| .p files processed | {n_files} |")
    lines.append(f"| Total profiles | {n_profiles_total} |")
    lines.append(f"| Profiles compared | {n_compared} |")
    lines.append(f"| pyturb-cli only | {n_tpw_only} |")
    lines.append(f"| pyturb only | {n_jesse_only} |")
    lines.append("")

    # Epsilon agreement table
    lines.append("## Epsilon Agreement")
    lines.append("")
    lines.append("Agreement categories based on median |log10(ratio)| per profile:")
    lines.append("- **good**: < 0.15 (within ~40%)")
    lines.append("- **fair**: 0.15 -- 0.5 (within ~3x)")
    lines.append("- **poor**: > 0.5")
    lines.append("")

    lines.append("| Variable | mean log10(ratio) | std | median ratio range | good | fair | poor |")
    lines.append("|----------|-------------------|-----|--------------------|------|------|------|")
    for var, st in [("eps_1", eps1_stats), ("eps_2", eps2_stats)]:
        if st:
            lines.append(
                f"| `{var}` | {st['mean_log']:+.4f} | {st['std_log']:.4f} "
                f"| {st['min_median_ratio']:.2f} -- {st['max_median_ratio']:.2f} "
                f"| {st['n_good']} | {st['n_fair']} | {st['n_poor']} |"
            )
    lines.append("")

    # Interpretation
    if eps1_stats and eps2_stats:
        pct1 = (10**eps1_stats["mean_log"] - 1) * 100
        pct2 = (10**eps2_stats["mean_log"] - 1) * 100
        lines.append(f"Overall, pyturb-cli epsilon is ~{pct1:.0f}% higher than pyturb for eps_1 "
                     f"and ~{pct2:.0f}% higher for eps_2. This small systematic offset is "
                     f"attributable to differences in the spectral estimation method "
                     f"(SCOR-160 vs custom) and Macoun & Lueck spatial response correction "
                     f"(applied in pyturb-cli, not in pyturb).")
        lines.append("")

    # Per-file summary table
    lines.append("## Per-File Summary")
    lines.append("")
    lines.append("| File | Profiles | eps_1 ratio | eps_2 ratio | eps_1 | eps_2 |")
    lines.append("|------|----------|-------------|-------------|-------|-------|")
    for s in file_summaries:
        if "eps_1_mean_ratio" in s:
            e1r = (
                f"{s['eps_1_mean_ratio']:.2f} "
                f"({s['eps_1_min_ratio']:.2f}--{s['eps_1_max_ratio']:.2f})"
            )
        else:
            e1r = "--"
        if "eps_2_mean_ratio" in s:
            e2r = (
                f"{s['eps_2_mean_ratio']:.2f} "
                f"({s['eps_2_min_ratio']:.2f}--{s['eps_2_max_ratio']:.2f})"
            )
        else:
            e2r = "--"
        e1s = f"{s['eps_1_good']}g/{s['eps_1_fair']}f/{s['eps_1_poor']}p"
        e2s = f"{s['eps_2_good']}g/{s['eps_2_fair']}f/{s['eps_2_poor']}p"
        nc = s["n_compared"]
        np_ = s["n_profiles"]
        lines.append(
            f"| {s['file']} | {nc}/{np_} "
            f"| {e1r} | {e2r} | {e1s} | {e2s} |"
        )
    lines.append("")

    # Per-profile detail
    lines.append("## Per-Profile Detail")
    lines.append("")
    lines.append("Median ratio = `median(pyturb-cli / pyturb)` for pressure-matched estimates.")
    lines.append("")

    for stem, rows in file_results.items():
        fnum = _file_number(stem)
        comp = [r for r in rows if r.get("status") == "compared"]
        tpw_only = [r for r in rows if r.get("status") == "tpw-only"]
        jesse_only = [r for r in rows if r.get("status") == "jesse-only"]

        lines.append(f"### File {fnum}")
        if tpw_only:
            lines.append("")
            lines.append(f"*{len(tpw_only)} profile(s) in pyturb-cli only (not in pyturb).*")
        if jesse_only:
            lines.append("")
            lines.append(f"*{len(jesse_only)} profile(s) in pyturb only (not in pyturb-cli).*")
        lines.append("")
        if not comp:
            lines.append("No comparable profiles.")
            lines.append("")
            continue

        lines.append("| Profile | n_pyturb | n_cli | eps_1 | ratio | eps_2 | ratio |")
        lines.append("|---------|----------|-------|-------|-------|-------|-------|")
        for r in comp:
            pi = r["profile"]
            nj = r.get("n_jesse", "--")
            nt = r.get("n_tpw", "--")
            e1s = r.get("eps_1_status", "--")
            e1m = f"{r['eps_1_median_ratio']:.2f}" if "eps_1_median_ratio" in r else "--"
            e2s = r.get("eps_2_status", "--")
            e2m = f"{r['eps_2_median_ratio']:.2f}" if "eps_2_median_ratio" in r else "--"
            lines.append(f"| {pi} | {nj} | {nt} | {e1s} | {e1m} | {e2s} | {e2m} |")
        lines.append("")

    # Processing differences section
    lines.append("## Processing Differences")
    lines.append("")
    lines.append("| Feature | pyturb | pyturb-cli |")
    lines.append("|---------|--------|------------|")
    lines.append("| Shear spectrum | Custom | SCOR-160 (Lueck 2024) |")
    lines.append("| Noise removal | None | Goodman (opt-in via `--goodman`) |")
    lines.append("| Spatial correction | None | Macoun & Lueck (2004) |")
    lines.append("| Window conversion | `int(s*fs)`, even | Same (matching pyturb) |")
    lines.append("| Overlap | n_fft // 2 | Same (matching pyturb) |")
    lines.append("| Profile detection | profinder | scipy.signal.find_peaks |")
    lines.append("| CLI framework | Typer | argparse |")
    lines.append("")

    md = "\n".join(lines)
    out_path.write_text(md)
    print(f"Wrote {out_path} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
