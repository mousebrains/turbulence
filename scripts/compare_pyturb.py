#!/usr/bin/env python3
"""Compare Jesse's pyturb and pyturb-cli (odas_tpw) epsilon output.

For each .p file in VMP/:
  1. Run ``pyturb eps`` (Jesse's)
  2. Run ``pyturb-cli eps`` (odas_tpw)
  3. Compare per-profile eps_1, eps_2 values

Usage:
    python scripts/compare_pyturb.py
    python scripts/compare_pyturb.py --vmp-dir VMP/ --files '*0002*'
    python scripts/compare_pyturb.py --skip-pyturb   # only run pyturb-cli
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# Defaults — adjust to your layout
PYTURB_CMD = "pyturb"              # Jesse's CLI (pip install pyturb)
PYTURB_CLI_CMD = "pyturb-cli"      # This repo's CLI


def _run(cmd: list[str], label: str) -> bool:
    """Run a command, log stdout/stderr, return True on success."""
    logger.info(f"[{label}] {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        logger.error(f"[{label}] command not found: {cmd[0]}")
        return False
    if result.stdout.strip():
        for line in result.stdout.strip().splitlines():
            logger.info(f"  {line}")
    if result.returncode != 0:
        logger.error(f"[{label}] exit {result.returncode}")
        if result.stderr.strip():
            for line in result.stderr.strip().splitlines():
                logger.error(f"  {line}")
        return False
    return True


def _compare_profiles(
    jesse_dir: Path, tpw_dir: Path, stem: str
) -> list[dict]:
    """Compare per-profile output from both tools for one source file."""
    jesse_files = sorted(jesse_dir.glob(f"{stem}_p*.nc"))
    tpw_files = sorted(tpw_dir.glob(f"{stem}_p*.nc"))

    results = []

    # Build lookup by profile index
    def _index(path: Path) -> int:
        # Extract NNNN from stem_pNNNN.nc
        name = path.stem
        idx_str = name.rsplit("_p", 1)[-1]
        return int(idx_str)

    jesse_by_idx = {_index(f): f for f in jesse_files}
    tpw_by_idx = {_index(f): f for f in tpw_files}

    all_indices = sorted(set(jesse_by_idx) | set(tpw_by_idx))

    for idx in all_indices:
        row: dict = {"stem": stem, "profile": idx}

        if idx not in jesse_by_idx:
            row["status"] = "pyturb-only"
            row["note"] = "Profile only in pyturb-cli"
            results.append(row)
            continue
        if idx not in tpw_by_idx:
            row["status"] = "tpw-only"
            row["note"] = "Profile only in pyturb"
            results.append(row)
            continue

        try:
            ds_j = xr.open_dataset(jesse_by_idx[idx])
            ds_t = xr.open_dataset(tpw_by_idx[idx])
        except Exception as e:
            row["status"] = "error"
            row["note"] = str(e)
            results.append(row)
            continue

        row["n_jesse"] = ds_j.sizes.get("time", 0)
        row["n_tpw"] = ds_t.sizes.get("time", 0)

        # Compare epsilon values by pressure-matching
        # (Different window parameters produce different numbers of
        # estimates, so index-based comparison is meaningless.)
        if "pressure" not in ds_j or "pressure" not in ds_t:
            row["status"] = "no_pressure"
            results.append(row)
            ds_j.close()
            ds_t.close()
            continue

        p_j = ds_j["pressure"].values
        p_t = ds_t["pressure"].values
        row["pres_range_j"] = f"{np.nanmin(p_j):.0f}-{np.nanmax(p_j):.0f}"
        row["pres_range_t"] = f"{np.nanmin(p_t):.0f}-{np.nanmax(p_t):.0f}"

        for var in ["eps_1", "eps_2"]:
            if var not in ds_j or var not in ds_t:
                row[f"{var}_status"] = "missing"
                continue

            j_vals = ds_j[var].values
            t_vals = ds_t[var].values

            if len(j_vals) == 0 or len(t_vals) == 0:
                row[f"{var}_status"] = "empty"
                continue

            # Interpolate tpw onto Jesse's pressure grid for comparison
            valid_t = np.isfinite(p_t) & np.isfinite(t_vals) & (t_vals > 0)
            valid_j = np.isfinite(p_j) & np.isfinite(j_vals) & (j_vals > 0)

            if valid_t.sum() < 2 or valid_j.sum() < 2:
                row[f"{var}_status"] = "no_valid"
                continue

            # Interpolate log10(epsilon) from tpw onto Jesse's pressures
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

            # Status based on agreement (using median absolute log ratio)
            med_abs = float(np.median(np.abs(log_ratio)))
            if med_abs < 0.15:  # median within ~40%
                row[f"{var}_status"] = "good"
            elif med_abs < 0.5:  # median within ~3x
                row[f"{var}_status"] = "fair"
            else:
                row[f"{var}_status"] = "poor"

        row["status"] = "compared"
        results.append(row)

        ds_j.close()
        ds_t.close()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Jesse's pyturb and pyturb-cli epsilon output"
    )
    parser.add_argument(
        "--vmp-dir", default="VMP", help="Directory with .p files (default: VMP)"
    )
    parser.add_argument(
        "--files", default="*.p", help="Glob pattern for .p files (default: *.p)"
    )
    parser.add_argument(
        "--work-dir", default="/tmp/pyturb_compare",
        help="Working directory for output (default: /tmp/pyturb_compare)",
    )
    parser.add_argument(
        "--skip-pyturb", action="store_true",
        help="Skip running Jesse's pyturb (use existing results)",
    )
    parser.add_argument(
        "--skip-tpw", action="store_true",
        help="Skip running pyturb-cli (use existing results)",
    )
    parser.add_argument(
        "--jesse-eps-dir", default=None,
        help="Path to pre-existing Jesse pyturb eps results directory",
    )
    parser.add_argument(
        "--diss-len", type=float, default=4.0, help="Dissipation length [s]"
    )
    parser.add_argument(
        "--fft-len", type=float, default=1.0, help="FFT length [s]"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    vmp_dir = Path(args.vmp_dir)
    work_dir = Path(args.work_dir)
    p_files = sorted(vmp_dir.glob(args.files))

    if not p_files:
        logger.error(f"No .p files matching {args.files} in {vmp_dir}")
        sys.exit(1)

    logger.info(f"Found {len(p_files)} .p files")

    # Prepare output directories
    jesse_nc = work_dir / "jesse_nc"
    jesse_eps = work_dir / "jesse_eps"
    tpw_nc = work_dir / "tpw_nc"
    tpw_eps = work_dir / "tpw_eps"

    for d in [jesse_nc, jesse_eps, tpw_nc, tpw_eps]:
        d.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert with each tool (p2nc)
    if not args.skip_pyturb:
        logger.info("=== Running Jesse's pyturb p2nc ===")
        _run(
            [PYTURB_CMD, "p2nc"] + [str(f) for f in p_files]
            + ["-o", str(jesse_nc), "--overwrite"],
            "pyturb p2nc",
        )

    if not args.skip_tpw:
        logger.info("=== Running pyturb-cli p2nc ===")
        _run(
            [PYTURB_CLI_CMD, "p2nc"] + [str(f) for f in p_files]
            + ["-o", str(tpw_nc), "-w"],
            "pyturb-cli p2nc",
        )

    # Step 2: Compute epsilon with each tool
    if not args.skip_pyturb:
        logger.info("=== Running Jesse's pyturb eps ===")
        jesse_nc_files = sorted(jesse_nc.glob("*.nc"))
        if jesse_nc_files:
            _run(
                [PYTURB_CMD, "eps"] + [str(f) for f in jesse_nc_files]
                + ["-o", str(jesse_eps), "--overwrite",
                   "-d", str(args.diss_len), "-f", str(args.fft_len)],
                "pyturb eps",
            )

    if not args.skip_tpw:
        logger.info("=== Running pyturb-cli eps ===")
        # pyturb-cli can take .p files directly
        _run(
            [PYTURB_CLI_CMD, "eps"] + [str(f) for f in p_files]
            + ["-o", str(tpw_eps), "-w",
               "-d", str(args.diss_len), "-f", str(args.fft_len)],
            "pyturb-cli eps",
        )

    # Step 3: Compare results
    logger.info("=== Comparing results ===")
    stems = sorted({f.stem for f in p_files})

    # Allow pointing at pre-existing Jesse results
    jesse_eps_compare = Path(args.jesse_eps_dir) if args.jesse_eps_dir else jesse_eps

    all_results: list[dict] = []
    for stem in stems:
        results = _compare_profiles(jesse_eps_compare, tpw_eps, stem)
        all_results.extend(results)

    # Print summary
    n_compared = sum(1 for r in all_results if r.get("status") == "compared")
    n_jesse_only = sum(1 for r in all_results if r.get("status") == "pyturb-only")
    n_tpw_only = sum(1 for r in all_results if r.get("status") == "tpw-only")

    print(f"\n{'=' * 70}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"Files compared:       {len(stems)}")
    print(f"Profiles compared:    {n_compared}")
    print(f"pyturb-only profiles: {n_jesse_only}")
    print(f"tpw-only profiles:    {n_tpw_only}")

    # Aggregate epsilon statistics
    for var in ["eps_1", "eps_2"]:
        log_ratios = [
            r[f"{var}_mean_log_ratio"]
            for r in all_results
            if f"{var}_mean_log_ratio" in r
        ]
        if log_ratios:
            print(f"\n{var}:")
            print(f"  mean log10(tpw/jesse): {np.mean(log_ratios):+.4f}")
            print(f"  std  log10(tpw/jesse): {np.std(log_ratios):.4f}")

            statuses = [
                r.get(f"{var}_status", "?")
                for r in all_results if r.get("status") == "compared"
            ]
            n_good = statuses.count("good")
            n_fair = statuses.count("fair")
            n_poor = statuses.count("poor")
            print(f"  good (<25%): {n_good}  fair (<3x): {n_fair}  poor: {n_poor}")

    # Per-file detail
    hdr = f"{'Prof':>4} {'n_j':>5} {'n_t':>5} {'eps1':>6} {'e1_med':>7} {'eps2':>6} {'e2_med':>7}"
    print(f"\n{'=' * 70}")
    print("PER-PROFILE DETAIL (pressure-matched comparison)")
    print(f"{'=' * 70}")
    print(hdr)
    print("-" * len(hdr))

    for r in all_results:
        pi = r.get("profile", "?")
        n_j = r.get("n_jesse", "-")
        n_t = r.get("n_tpw", "-")
        e1 = r.get("eps_1_status", "-")
        e2 = r.get("eps_2_status", "-")
        e1m = f"{r['eps_1_median_ratio']:.2f}" if "eps_1_median_ratio" in r else "-"
        e2m = f"{r['eps_2_median_ratio']:.2f}" if "eps_2_median_ratio" in r else "-"
        print(f"{pi:>4} {n_j!s:>5} {n_t!s:>5} {e1:>6} {e1m:>7} {e2:>6} {e2m:>7}")

    print()


if __name__ == "__main__":
    main()
