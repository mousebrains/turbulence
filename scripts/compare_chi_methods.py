#!/usr/bin/env python3
"""Compare chi estimation methods across all VMP .p files.

For each dissipation window in every profile:
  - Compute epsilon from shear probes
  - Compute chi with Method 1 (from epsilon), M2-MLE, M2-Iter
  - Repeat for both Batchelor and Kraichnan spectrum models

Produces:
  - Summary statistics table (printed + CSV)
  - Ratio histograms (PNG) saved to docs/chi/

Usage:
    python scripts/compare_chi_methods.py
    python scripts/compare_chi_methods.py --vmp-dir VMP/ --files '*0026*'
"""

from __future__ import annotations

import argparse
import logging
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from odas_tpw.rsi.p_file import PFile
from odas_tpw.rsi.profile import get_profiles
from odas_tpw.rsi.viewer_base import _smooth_fall_rate
from odas_tpw.rsi.window import compute_chi_window, compute_eps_window

logger = logging.getLogger(__name__)

SH_PAT = re.compile(r"^sh\d+$", re.IGNORECASE)
AC_PAT = re.compile(r"^A[xyz]\d*$", re.IGNORECASE)
DT_PAT = re.compile(r"^T\d+_dT\d+$", re.IGNORECASE)
T_PAT = re.compile(r"^T\d+$", re.IGNORECASE)


def _extract_channels(pf):
    """Extract shear, accel, therm_fast, diff_gains, T from a PFile."""
    shear = sorted(
        [(n, pf.channels[n]) for n in pf._fast_channels if SH_PAT.match(n)],
        key=lambda x: x[0],
    )
    accel = sorted(
        [(n, pf.channels[n]) for n in pf._fast_channels if AC_PAT.match(n)],
        key=lambda x: x[0],
    )
    therm_fast = sorted(
        [(n, pf.channels[n]) for n in pf._fast_channels if DT_PAT.match(n)],
        key=lambda x: x[0],
    )
    diff_gains = []
    for name, _ in therm_fast:
        ch_cfg = next((ch for ch in pf.config["channels"] if ch.get("name") == name), {})
        diff_gains.append(float(ch_cfg.get("diff_gain", "0.94")))

    P = pf.channels["P"]
    T = pf.channels.get("T1", pf.channels.get("T", np.zeros_like(P)))

    # Interpolate slow -> fast
    P_fast = np.interp(pf.t_fast, pf.t_slow, P)
    speed_fast = np.abs(np.interp(pf.t_fast, pf.t_slow, _smooth_fall_rate(P, pf.fs_slow)))
    ratio = round(pf.fs_fast / pf.fs_slow)

    # Convert shear from piezo to du/dz
    shear = [(n, d / speed_fast**2) for n, d in shear]

    return shear, accel, therm_fast, diff_gains, P, T, P_fast, speed_fast, ratio


def process_file(
    p_path: Path,
    fft_length: int = 1024,
    f_AA: float = 98.0,
    do_goodman: bool = True,
) -> list[dict]:
    """Process one .p file, return list of per-window, per-probe result dicts."""
    pf = PFile(p_path)
    shear, accel, therm_fast, diff_gains, P, T, P_fast, speed_fast, ratio = _extract_channels(pf)

    if not shear or not therm_fast:
        logger.warning(f"{p_path.name}: no shear or thermistor channels, skipping")
        return []

    W_slow = _smooth_fall_rate(P, pf.fs_slow)
    profiles = get_profiles(P, W_slow, pf.fs_slow, P_min=0.5, W_min=0.3, direction="down", min_duration=7.0)
    if not profiles:
        logger.warning(f"{p_path.name}: no profiles detected")
        return []

    diss_length = 4 * fft_length
    step = diss_length // 2
    f_AA_chi = 0.9 * f_AA
    mean_T = float(np.mean(T))
    n_therm = len(therm_fast)

    results = []

    for prof_idx, (s_slow, e_slow) in enumerate(profiles):
        s_fast = s_slow * ratio
        e_fast = min((e_slow + 1) * ratio, len(pf.t_fast))
        N = e_fast - s_fast

        n_windows = max(0, 1 + (N - diss_length) // step)
        for win_idx in range(n_windows):
            s = s_fast + win_idx * step
            e = s + diss_length
            if e > e_fast:
                break
            w_sel = slice(s, e)

            # --- Epsilon ---
            sh_seg = np.column_stack([d[w_sel] for _, d in shear])
            ac_seg = np.column_stack([d[w_sel] for _, d in accel]) if accel else None
            er = compute_eps_window(
                sh_seg, ac_seg, speed_fast[w_sel], P_fast[w_sel],
                mean_T, pf.fs_fast, fft_length, f_AA, do_goodman,
            )
            if not np.any(np.isfinite(er.epsilon)):
                continue

            therm_segs = [therm_fast[ci][1][w_sel] for ci in range(n_therm)]

            # --- Chi: all methods × spectrum models ---
            for spectrum_model in ("kraichnan", "batchelor"):
                cr_m1 = compute_chi_window(
                    therm_segs, diff_gains, er.W, mean_T, er.nu,
                    pf.fs_fast, fft_length, f_AA_chi,
                    spectrum_model=spectrum_model, epsilon=er.epsilon,
                    fom=er.fom, method=1,
                )
                cr_m2 = compute_chi_window(
                    therm_segs, diff_gains, er.W, mean_T, er.nu,
                    pf.fs_fast, fft_length, f_AA_chi,
                    spectrum_model=spectrum_model, method=2,
                )

                for ci in range(n_therm):
                    results.append({
                        "file": p_path.name,
                        "profile": prof_idx,
                        "window": win_idx,
                        "probe": ci,
                        "pressure": float(np.mean(P_fast[w_sel])),
                        "speed": er.W,
                        "epsilon": float(er.epsilon[min(ci, len(er.epsilon) - 1)]),
                        "spectrum_model": spectrum_model,
                        "M1_chi": float(cr_m1.chi[ci]),
                        "M1_fom": float(cr_m1.fom[ci]),
                        "M2_Iter_chi": float(cr_m2.chi[ci]),
                        "M2_Iter_fom": float(cr_m2.fom[ci]),
                    })

    return results


def compute_ratios(rows: list[dict]) -> dict:
    """Compute ratio arrays for valid pairs."""
    out = {}
    for model in ("kraichnan", "batchelor"):
        sub = [r for r in rows if r["spectrum_model"] == model]
        m1 = np.array([r["M1_chi"] for r in sub])
        m2i = np.array([r["M2_Iter_chi"] for r in sub])

        # M1 / M2-Iter
        valid = np.isfinite(m1) & np.isfinite(m2i) & (m1 > 0) & (m2i > 0)
        out[f"M1_M2Iter_{model}"] = m1[valid] / m2i[valid]

    # Cross-model: kraichnan / batchelor for each method
    kr = {(r["file"], r["profile"], r["window"], r["probe"]): r
          for r in rows if r["spectrum_model"] == "kraichnan"}
    ba = {(r["file"], r["profile"], r["window"], r["probe"]): r
          for r in rows if r["spectrum_model"] == "batchelor"}
    common = set(kr) & set(ba)
    for method in ("M1_chi", "M2_Iter_chi"):
        k_vals, b_vals = [], []
        for key in common:
            kv, bv = kr[key][method], ba[key][method]
            if np.isfinite(kv) and np.isfinite(bv) and kv > 0 and bv > 0:
                k_vals.append(kv)
                b_vals.append(bv)
        k_arr = np.array(k_vals)
        b_arr = np.array(b_vals)
        label = method.replace("_chi", "")
        out[f"{label}_kr_ba"] = k_arr / b_arr if len(k_arr) > 0 else np.array([])

    return out


def summary_table(ratios: dict) -> str:
    """Format a markdown table of ratio statistics."""
    lines = [
        "| Ratio | N | Median | Mean | Std | Q5 | Q95 |",
        "|:------|--:|-------:|-----:|----:|---:|----:|",
    ]
    for label, arr in sorted(ratios.items()):
        if len(arr) == 0:
            lines.append(f"| {label} | 0 | — | — | — | — | — |")
            continue
        log_r = np.log10(arr)
        lines.append(
            f"| {label} | {len(arr)} "
            f"| {np.median(arr):.3f} "
            f"| {np.mean(arr):.3f} "
            f"| {np.std(log_r):.3f} (log10) "
            f"| {np.percentile(arr, 5):.3f} "
            f"| {np.percentile(arr, 95):.3f} |"
        )
    return "\n".join(lines)


def plot_histograms(ratios: dict, out_dir: Path):
    """Save ratio histograms as PNGs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, arr in sorted(ratios.items()):
        if len(arr) < 5:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        log_r = np.log10(arr)
        ax.hist(log_r, bins=60, edgecolor="black", linewidth=0.3, alpha=0.8)
        ax.axvline(0, color="red", linestyle="--", linewidth=1, label="ratio = 1")
        ax.axvline(np.log10(2), color="orange", linestyle=":", linewidth=0.8, label="×2")
        ax.axvline(-np.log10(2), color="orange", linestyle=":", linewidth=0.8, label="×0.5")
        med = np.median(log_r)
        ax.axvline(med, color="blue", linestyle="-", linewidth=1, label=f"median={10**med:.2f}")
        ax.set_xlabel("log₁₀(ratio)")
        ax.set_ylabel("Count")
        ax.set_title(label.replace("_", " "))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fname = out_dir / f"{label}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        logger.info(f"  Saved {fname}")


def write_markdown(ratios: dict, out_dir: Path, n_files: int, n_rows: int):
    """Write docs/chi/chi_comparison.md."""
    table = summary_table(ratios)
    hist_links = []
    for label in sorted(ratios.keys()):
        if len(ratios[label]) >= 5:
            hist_links.append(f"### {label.replace('_', ' ')}\n\n![{label}]({label}.png)\n")

    md = f"""# Chi Method Comparison

Comparison of chi (thermal variance dissipation rate) estimates from three
methods across {n_files} VMP .p files ({n_rows} window-probe observations).

## Methods

| Method | Description | kB source | Chi estimation |
|:-------|:------------|:----------|:---------------|
| **M1** | Dillon & Caldwell 1980 | Fixed from shear-probe epsilon | Log-space LS fit: min Σ[log(model)-log(obs)]² with kB fixed |
| **M2-Iter** | Peterson & Fer 2014 | MLE grid search (iterative) | Noise-subtracted integration + unresolved variance from model |
| **M2-MLE** | Ruddick et al. 2000 | MLE grid search | Variance correction with fitted kB |

## Spectrum Models

| Model | Reference | Rolloff | Constant |
|:------|:----------|:--------|:---------|
| **Batchelor** | Dillon & Caldwell 1980 | Gaussian (erfc) | q = 3.7 |
| **Kraichnan** | Bogucki et al. 1997 | Exponential | q = 5.26 |

## Ratio Statistics

Ratios are computed per-window, per-probe where both values are finite and positive.
"Std (log10)" is the standard deviation of log₁₀(ratio), measuring spread on
a multiplicative scale.

{table}

## Histograms

{"".join(hist_links)}
## Notes

- **M1 vs M2-Iter**: M1 uses epsilon from shear probes to fix the Batchelor
  wavenumber kB, then fits chi amplitude via log-space least squares. M2-Iter
  fits both kB and chi from the temperature gradient spectrum alone. When
  shear-derived epsilon is consistent with the thermal spectrum, these should
  agree; large ratios indicate epsilon/chi inconsistency (variable dissipation
  ratio).

- **Kraichnan vs Batchelor**: The Kraichnan model has a slower (exponential)
  rolloff than Batchelor (Gaussian erfc), which better matches DNS results.
  Both are normalised to integrate to χ/(6κ_T), so chi values should be
  similar; differences arise from the spectral shape affecting the fit and
  variance correction.

- FOM (figure of merit) values near 1.0 indicate the fitted model matches the
  observed spectrum well. Values far from 1 suggest shape mismatch (wrong kB)
  or noise contamination.
"""
    md_path = out_dir / "chi_comparison.md"
    md_path.write_text(md)
    logger.info(f"Wrote {md_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vmp-dir", type=Path, default=Path("VMP"))
    parser.add_argument("--files", default="*.p", help="glob pattern for .p files")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/chi"))
    parser.add_argument("--fft-length", type=int, default=1024)
    parser.add_argument("--f-AA", type=float, default=98.0)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    p_files = sorted(args.vmp_dir.glob(args.files))
    if not p_files:
        logger.error(f"No .p files found matching {args.vmp_dir / args.files}")
        return

    logger.info(f"Processing {len(p_files)} files...")
    all_rows: list[dict] = []

    for i, pf in enumerate(p_files):
        logger.info(f"[{i+1}/{len(p_files)}] {pf.name}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows = process_file(pf, fft_length=args.fft_length, f_AA=args.f_AA)
        all_rows.extend(rows)
        logger.info(f"  → {len(rows)} observations")

    logger.info(f"\nTotal: {len(all_rows)} observations from {len(p_files)} files")

    if not all_rows:
        logger.error("No valid observations collected")
        return

    ratios = compute_ratios(all_rows)

    print("\n" + summary_table(ratios) + "\n")

    plot_histograms(ratios, args.out_dir)
    write_markdown(ratios, args.out_dir, len(p_files), len(all_rows))

    logger.info("Done.")


if __name__ == "__main__":
    main()
