#!/usr/bin/env python3
"""Generate docs/pyturb/comparison.md from pre-computed pyturb vs pyturb-cli results.

Reads the NetCDF output from both Jesse's pyturb and pyturb-cli (produced by
compare_pyturb.py) and generates a structured markdown comparison report with
histograms of the per-profile epsilon ratios.

Usage:
    python scripts/generate_comparison_md.py
    python scripts/generate_comparison_md.py --work-dir /tmp/pyturb_compare_all
"""

from __future__ import annotations

import argparse
import datetime
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

matplotlib.use("Agg")

JESSE_DIR_DEFAULT = "/tmp/pyturb_compare_all/jesse_eps"
TPW_DIR_DEFAULT = "/tmp/pyturb_compare_all/tpw_eps"
VMP_DIR_DEFAULT = "VMP"
OUTPUT_DEFAULT = "docs/pyturb/comparison.md"

# Agreement thresholds (median |log10(ratio)| per profile)
GOOD_THRESHOLD = 0.15  # ~41% — within a factor of 1.41
FAIR_THRESHOLD = math.log10(2)  # 0.301 — within a factor of 2


def _index(path: Path) -> int:
    name = path.stem
    return int(name.rsplit("_p", 1)[-1])


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
            p_j[valid_j],
            p_t[valid_t],
            np.log10(t_vals[valid_t]),
            left=np.nan,
            right=np.nan,
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
        if med_abs < GOOD_THRESHOLD:
            row[f"{var}_status"] = "good"
        elif med_abs < FAIR_THRESHOLD:
            row[f"{var}_status"] = "fair"
        else:
            row[f"{var}_status"] = "poor"

    row["status"] = "compared"
    ds_j.close()
    ds_t.close()
    return row


def make_histogram(
    ratios_1: list[float],
    ratios_2: list[float],
    out_path: Path,
) -> None:
    """Generate side-by-side histograms of per-profile median epsilon ratios."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, ratios, label in zip(
        axes, [ratios_1, ratios_2], ["eps_1", "eps_2"]
    ):
        arr = np.array(ratios)
        bins = np.arange(0.80, 1.30, 0.02)
        ax.hist(arr, bins=bins, edgecolor="black", linewidth=0.5, alpha=0.8)
        ax.axvline(1.0, color="k", linestyle="--", linewidth=0.8, label="1:1")
        med = float(np.median(arr))
        ax.axvline(
            med, color="red", linestyle="-", linewidth=1.2,
            label=f"median = {med:.2f}",
        )
        ax.set_xlabel("Median ratio (pyturb-cli / pyturb)")
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.set_xlim(0.80, 1.30)

    axes[0].set_ylabel("Number of profiles")
    fig.suptitle(
        "Per-profile median epsilon ratio: pyturb-cli / pyturb",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    all_rows: list[dict] = []
    n_files_with_profiles = 0
    for stem in stems:
        jesse_files = {
            _index(f): f for f in sorted(jesse_dir.glob(f"{stem}_p*.nc"))
        }
        tpw_files = {
            _index(f): f for f in sorted(tpw_dir.glob(f"{stem}_p*.nc"))
        }
        all_indices = sorted(set(jesse_files) | set(tpw_files))
        if not all_indices:
            continue
        n_files_with_profiles += 1

        for idx in all_indices:
            row = {"profile": idx, "stem": stem}
            if idx not in jesse_files:
                row["status"] = "tpw-only"
            elif idx not in tpw_files:
                row["status"] = "jesse-only"
            else:
                row.update(
                    compare_one_profile(jesse_files[idx], tpw_files[idx])
                )
            all_rows.append(row)

    compared = [r for r in all_rows if r.get("status") == "compared"]
    n_tpw_only = sum(1 for r in all_rows if r.get("status") == "tpw-only")
    n_jesse_only = sum(1 for r in all_rows if r.get("status") == "jesse-only")

    # Collect per-profile median ratios for histograms
    eps1_ratios = [
        r["eps_1_median_ratio"] for r in compared
        if "eps_1_median_ratio" in r
    ]
    eps2_ratios = [
        r["eps_2_median_ratio"] for r in compared
        if "eps_2_median_ratio" in r
    ]

    # Aggregate stats
    def _var_stats(var: str) -> dict:
        log_ratios = [
            r[f"{var}_mean_log_ratio"] for r in compared
            if f"{var}_mean_log_ratio" in r
        ]
        median_ratios = [
            r[f"{var}_median_ratio"] for r in compared
            if f"{var}_median_ratio" in r
        ]
        statuses = [r.get(f"{var}_status", "?") for r in compared]
        if not log_ratios:
            return {}
        return {
            "mean_log": float(np.mean(log_ratios)),
            "std_log": float(np.std(log_ratios)),
            "median_of_medians": float(np.median(median_ratios)),
            "min_median_ratio": float(np.min(median_ratios)),
            "max_median_ratio": float(np.max(median_ratios)),
            "p05": float(np.percentile(median_ratios, 5)),
            "p95": float(np.percentile(median_ratios, 95)),
            "n_good": statuses.count("good"),
            "n_fair": statuses.count("fair"),
            "n_poor": statuses.count("poor"),
            "n_total": len(log_ratios),
        }

    eps1_stats = _var_stats("eps_1")
    eps2_stats = _var_stats("eps_2")

    # Generate histogram
    hist_filename = "epsilon_ratio_histograms.png"
    hist_path = out_path.parent / hist_filename
    if eps1_ratios and eps2_ratios:
        make_histogram(eps1_ratios, eps2_ratios, hist_path)

    # Build markdown
    lines: list[str] = []

    lines.append("# pyturb vs pyturb-cli: Quantitative Comparison")
    lines.append("")
    lines.append("Comparison of epsilon (TKE dissipation rate) estimates from")
    lines.append(
        "[Jesse's pyturb](https://github.com/oceancascades/pyturb) and"
    )
    lines.append("`pyturb-cli` (this repository) on ARCTERX VMP-250 data")
    lines.append("(SN 479, R/V Thompson, January 2025).")
    lines.append("")
    lines.append("Both tools were run with matching parameters:")
    lines.append("- `--fft-len 1.0` (512 samples at 512 Hz)")
    lines.append("- `--diss-len 4.0` (2048 samples at 512 Hz)")
    lines.append(
        "- Goodman cleaning: **off** (`pyturb-cli` default matches pyturb)"
    )
    lines.append("- Overlap: n_fft // 2 = 256 samples")
    lines.append("")
    lines.append(
        "Epsilon values are compared by interpolating `log10(epsilon)` from"
    )
    lines.append("pyturb-cli onto pyturb's pressure grid. The median ratio")
    lines.append("`pyturb-cli / pyturb` is reported per profile.")
    lines.append("")
    today = datetime.date.today().isoformat()
    lines.append(
        f"*Generated {today} from {len(p_files)} VMP .p files.*"
    )
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| .p files processed | {n_files_with_profiles} |")
    lines.append(f"| Total profiles | {len(all_rows)} |")
    lines.append(f"| Profiles compared | {len(compared)} |")
    lines.append(f"| pyturb-cli only | {n_tpw_only} |")
    lines.append(f"| pyturb only | {n_jesse_only} |")
    lines.append("")

    # Epsilon agreement
    lines.append("## Epsilon Agreement")
    lines.append("")
    lines.append(
        "Agreement categories based on median |log10(ratio)| per profile:"
    )
    lines.append(
        f"- **good**: < {GOOD_THRESHOLD} "
        f"(within a factor of {10**GOOD_THRESHOLD:.2f})"
    )
    lines.append(
        f"- **fair**: {GOOD_THRESHOLD} -- {FAIR_THRESHOLD:.3f} "
        f"(within a factor of {10**FAIR_THRESHOLD:.1f})"
    )
    lines.append(
        f"- **poor**: > {FAIR_THRESHOLD:.3f} "
        f"(outside a factor of {10**FAIR_THRESHOLD:.1f})"
    )
    lines.append("")

    hdr = (
        "| Variable | median ratio | 5th--95th pctl | range "
        "| good | fair | poor |"
    )
    sep = (
        "|----------|--------------|----------------|-------"
        "|------|------|------|"
    )
    lines.append(hdr)
    lines.append(sep)
    for var, st in [("eps_1", eps1_stats), ("eps_2", eps2_stats)]:
        if st:
            lines.append(
                f"| `{var}` "
                f"| {st['median_of_medians']:.3f} "
                f"| {st['p05']:.2f} -- {st['p95']:.2f} "
                f"| {st['min_median_ratio']:.2f} -- "
                f"{st['max_median_ratio']:.2f} "
                f"| {st['n_good']} | {st['n_fair']} | {st['n_poor']} |"
            )
    lines.append("")

    # Interpretation
    if eps1_stats and eps2_stats:
        pct1 = (10 ** eps1_stats["mean_log"] - 1) * 100
        pct2 = (10 ** eps2_stats["mean_log"] - 1) * 100
        lines.append(
            f"Overall, pyturb-cli epsilon is ~{pct1:.0f}% higher than "
            f"pyturb for eps_1 and ~{pct2:.0f}% higher for eps_2. "
            f"This small systematic offset is attributable to differences "
            f"in the spectral estimation method (SCOR-160 vs custom) and "
            f"Macoun & Lueck spatial response correction (applied in "
            f"pyturb-cli, not in pyturb)."
        )
        lines.append("")

    # Histogram
    lines.append("## Distribution of Per-Profile Epsilon Ratios")
    lines.append("")
    lines.append(f"![Epsilon ratio histograms]({hist_filename})")
    lines.append("")

    # Processing differences
    lines.append("## Processing Differences")
    lines.append("")
    lines.append("| Feature | pyturb | pyturb-cli |")
    lines.append("|---------|--------|------------|")
    lines.append("| Shear spectrum | Custom | SCOR-160 (Lueck 2024) |")
    lines.append(
        "| Noise removal | None "
        "| Goodman (opt-in via `--goodman`) |"
    )
    lines.append("| Spatial correction | None | Macoun & Lueck (2004) |")
    lines.append(
        "| Window conversion | `int(s*fs)`, even "
        "| Same (matching pyturb) |"
    )
    lines.append("| Overlap | n_fft // 2 | Same (matching pyturb) |")
    lines.append(
        "| Profile detection | profinder | scipy.signal.find_peaks |"
    )
    lines.append("| CLI framework | Typer | argparse |")
    lines.append("")

    md = "\n".join(lines)
    out_path.write_text(md)
    print(f"Wrote {out_path} ({len(lines)} lines)")
    if hist_path.exists():
        print(f"Wrote {hist_path}")


if __name__ == "__main__":
    main()
