#!/usr/bin/env python3
"""Generate docs/pyturb/comparison.md from pre-computed pyturb vs pyturb-cli results.

Reads the NetCDF output from both Jesse's pyturb and pyturb-cli (produced by
compare_pyturb.py) and generates a structured markdown comparison report with
quantile statistics and histograms of log10(ratio) over all pressure-matched
bins.

Usage:
    python scripts/generate_comparison_md.py
    python scripts/generate_comparison_md.py --work-dir /tmp/pyturb_compare_all
"""

from __future__ import annotations

import argparse
import datetime
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


def _index(path: Path) -> int:
    name = path.stem
    return int(name.rsplit("_p", 1)[-1])


def compare_one_profile(jesse_path: Path, tpw_path: Path) -> dict[str, np.ndarray]:
    """Compare a single profile pair.

    Returns a dict mapping variable name to the array of per-bin
    log10(pyturb-cli / pyturb) values.  Empty array if comparison
    is not possible for that variable.
    """
    result: dict[str, np.ndarray] = {}
    try:
        ds_j = xr.open_dataset(jesse_path)
        ds_t = xr.open_dataset(tpw_path)
    except Exception:
        return result

    if "pressure" not in ds_j or "pressure" not in ds_t:
        ds_j.close()
        ds_t.close()
        return result

    p_j = ds_j["pressure"].values
    p_t = ds_t["pressure"].values

    for var in ["eps_1", "eps_2"]:
        if var not in ds_j or var not in ds_t:
            continue

        j_vals = ds_j[var].values
        t_vals = ds_t[var].values

        if len(j_vals) == 0 or len(t_vals) == 0:
            continue

        valid_t = np.isfinite(p_t) & np.isfinite(t_vals) & (t_vals > 0)
        valid_j = np.isfinite(p_j) & np.isfinite(j_vals) & (j_vals > 0)

        if valid_t.sum() < 2 or valid_j.sum() < 2:
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
            continue

        result[var] = t_interp[both_ok] - j_log[both_ok]

    ds_j.close()
    ds_t.close()
    return result


def make_histogram(
    log_ratios_1: np.ndarray,
    log_ratios_2: np.ndarray,
    out_path: Path,
) -> None:
    """Histograms of per-bin log10(pyturb-cli / pyturb)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, lr, label in zip(axes, [log_ratios_1, log_ratios_2], ["eps_1", "eps_2"]):
        bins = np.arange(-0.6, 0.65, 0.025)
        ax.hist(lr, bins=bins, edgecolor="black", linewidth=0.4, alpha=0.8)
        ax.axvline(0.0, color="k", linestyle="--", linewidth=0.8, label="0")
        med = float(np.median(lr))
        ax.axvline(
            med,
            color="red",
            linestyle="-",
            linewidth=1.2,
            label=f"Q50 = {med:+.3f}",
        )
        log2 = np.log10(2)
        ax.axvline(-log2, color="orange", linestyle=":", linewidth=1.0, label="2x")
        ax.axvline(log2, color="orange", linestyle=":", linewidth=1.0)
        ax.set_xlabel("log10(pyturb-cli / pyturb)")
        ax.set_title(label)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Number of bins")
    fig.suptitle(
        "Per-bin log10(epsilon ratio) across all profiles",
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

    # Collect all per-bin log10(ratio) values across all profiles
    all_log_ratios: dict[str, list[np.ndarray]] = {
        "eps_1": [],
        "eps_2": [],
    }
    n_files_with_profiles = 0
    n_profiles_total = 0
    n_profiles_compared = 0
    n_tpw_only = 0
    n_jesse_only = 0

    for stem in stems:
        jesse_files = {_index(f): f for f in sorted(jesse_dir.glob(f"{stem}_p*.nc"))}
        tpw_files = {_index(f): f for f in sorted(tpw_dir.glob(f"{stem}_p*.nc"))}
        all_indices = sorted(set(jesse_files) | set(tpw_files))
        if not all_indices:
            continue
        n_files_with_profiles += 1

        for idx in all_indices:
            n_profiles_total += 1
            if idx not in jesse_files:
                n_tpw_only += 1
                continue
            if idx not in tpw_files:
                n_jesse_only += 1
                continue

            result = compare_one_profile(jesse_files[idx], tpw_files[idx])
            if result:
                n_profiles_compared += 1
            for var in ["eps_1", "eps_2"]:
                if var in result and len(result[var]) > 0:
                    all_log_ratios[var].append(result[var])

    # Pool all per-bin values
    eps1_lr = np.concatenate(all_log_ratios["eps_1"]) if all_log_ratios["eps_1"] else np.array([])
    eps2_lr = np.concatenate(all_log_ratios["eps_2"]) if all_log_ratios["eps_2"] else np.array([])

    def _quantile_stats(lr: np.ndarray) -> dict:
        if len(lr) == 0:
            return {}
        return {
            "n_bins": len(lr),
            "Q0": float(np.min(lr)),
            "Q5": float(np.percentile(lr, 5)),
            "Q50": float(np.median(lr)),
            "Q95": float(np.percentile(lr, 95)),
            "Q100": float(np.max(lr)),
        }

    eps1_stats = _quantile_stats(eps1_lr)
    eps2_stats = _quantile_stats(eps2_lr)

    # Generate histogram
    hist_filename = "epsilon_ratio_histograms.png"
    hist_path = out_path.parent / hist_filename
    if len(eps1_lr) > 0 and len(eps2_lr) > 0:
        make_histogram(eps1_lr, eps2_lr, hist_path)

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
    lines.append("pyturb-cli onto pyturb's pressure grid.  Statistics are computed")
    lines.append("over all pressure-matched bins across all profiles.")
    lines.append("")
    today = datetime.date.today().isoformat()
    lines.append(f"*Generated {today} from {len(p_files)} VMP .p files.*")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| .p files processed | {n_files_with_profiles} |")
    lines.append(f"| Total profiles | {n_profiles_total} |")
    lines.append(f"| Profiles compared | {n_profiles_compared} |")
    lines.append(f"| pyturb-cli only | {n_tpw_only} |")
    lines.append(f"| pyturb only | {n_jesse_only} |")
    if eps1_stats:
        lines.append(f"| Pressure-matched bins (eps_1) | {eps1_stats['n_bins']} |")
    if eps2_stats:
        lines.append(f"| Pressure-matched bins (eps_2) | {eps2_stats['n_bins']} |")
    lines.append("")

    # Quantile table
    lines.append("## log10(pyturb-cli / pyturb) Quantiles")
    lines.append("")
    lines.append(
        "Quantiles of `log10(eps_cli / eps_pyturb)` over all "
        "pressure-matched bins.  A value of 0 means perfect agreement; "
        "+0.04 means pyturb-cli is ~10% higher."
    )
    lines.append("")
    lines.append("| Variable | N | Q0 (min) | Q5 | Q50 (median) | Q95 | Q100 (max) |")
    lines.append("|----------|---|----------|----|--------------|-----|------------|")
    for var, st in [("eps_1", eps1_stats), ("eps_2", eps2_stats)]:
        if st:
            lines.append(
                f"| `{var}` | {st['n_bins']} "
                f"| {st['Q0']:+.3f} "
                f"| {st['Q5']:+.3f} "
                f"| {st['Q50']:+.3f} "
                f"| {st['Q95']:+.3f} "
                f"| {st['Q100']:+.3f} |"
            )
    lines.append("")

    # Interpretation
    if eps1_stats and eps2_stats:
        pct1 = (10 ** eps1_stats["Q50"] - 1) * 100
        pct2 = (10 ** eps2_stats["Q50"] - 1) * 100
        max_factor = max(
            10 ** abs(eps1_stats["Q100"]),
            10 ** abs(eps1_stats["Q0"]),
            10 ** abs(eps2_stats["Q100"]),
            10 ** abs(eps2_stats["Q0"]),
        )
        lines.append(
            f"At the median, pyturb-cli epsilon is {pct1:+.0f}% relative "
            f"to pyturb for eps_1 and {pct2:+.0f}% for eps_2. "
            f"The worst-case bin differs by a factor of {max_factor:.1f}x. "
            f"The systematic offset is attributable to differences "
            f"in the spectral estimation method (SCOR-160 vs custom) and "
            f"Macoun & Lueck spatial response correction (applied in "
            f"pyturb-cli, not in pyturb)."
        )
        lines.append("")

    # Histogram
    lines.append("## Distribution of Per-Bin log10(epsilon ratio)")
    lines.append("")
    lines.append(f"![Epsilon ratio histograms]({hist_filename})")
    lines.append("")

    # Processing differences
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
    if hist_path.exists():
        print(f"Wrote {hist_path}")


if __name__ == "__main__":
    main()
