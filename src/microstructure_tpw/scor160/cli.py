"""CLI for SCOR/ATOMIX benchmark processing.

Usage:
    scor160-tpw l1-l2 data/*.nc       # L1->L2: section selection, despike, HP filter
    scor160-tpw l2-l3 data/*.nc       # L2->L3: wavenumber spectra (uses reference L2)
    scor160-tpw l1-l3 data/*.nc       # L1->L3: full pipeline through spectra
    scor160-tpw l3-l4 data/*.nc       # L3->L4: dissipation from spectra
    scor160-tpw l2-l4 data/*.nc       # L2->L4: spectra + dissipation
    scor160-tpw l1-l4 data/*.nc       # L1->L4: full pipeline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="scor160-tpw",
        description="SCOR/ATOMIX benchmark dataset processing",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- l1-l2 ---
    p = sub.add_parser("l1-l2", help="L1->L2: section selection, despike, HP filter")
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--plot", action="store_true")

    # --- l2-l3 ---
    p = sub.add_parser("l2-l3", help="L2->L3: wavenumber spectra (reference L2 input)")
    p.add_argument("files", nargs="+", type=Path)

    # --- l1-l3 ---
    p = sub.add_parser("l1-l3", help="L1->L3: full pipeline through spectra")
    p.add_argument("files", nargs="+", type=Path)

    # --- l3-l4 ---
    p = sub.add_parser("l3-l4", help="L3->L4: dissipation from spectra")
    p.add_argument("files", nargs="+", type=Path)

    # --- l2-l4 ---
    p = sub.add_parser("l2-l4", help="L2->L4: spectra + dissipation")
    p.add_argument("files", nargs="+", type=Path)

    # --- l1-l4 ---
    p = sub.add_parser("l1-l4", help="L1->L4: full pipeline")
    p.add_argument("files", nargs="+", type=Path)

    args = parser.parse_args(argv)

    dispatch = {
        "l1-l2": _cmd_l1_l2,
        "l2-l3": _cmd_l2_l3,
        "l1-l3": _cmd_l1_l3,
        "l3-l4": _cmd_l3_l4,
        "l2-l4": _cmd_l2_l4,
        "l1-l4": _cmd_l1_l4,
    }
    dispatch[args.command](args)


# ---------------------------------------------------------------------------
# l1-l2
# ---------------------------------------------------------------------------

def _cmd_l1_l2(args: argparse.Namespace) -> None:
    from microstructure_tpw.scor160.compare import compare_l2, format_l2_report
    from microstructure_tpw.scor160.io import read_atomix
    from microstructure_tpw.scor160.l2 import process_l2

    for path in args.files:
        print(f"\nProcessing {path.name} ...")
        l1, l2_params, l2_ref, _l3_params, _l3_ref, _l4_ref = read_atomix(path)
        print(f"  L1: {l1.n_time} samples, {l1.n_shear} shear, {l1.n_vib} {l1.vib_type}")
        print(f"  Params: HP_cut={l2_params.HP_cut} Hz, despike_sh={l2_params.despike_sh}"
              f", speed_tau={l2_params.speed_tau} s"
              f", min_W={l2_params.profile_min_W} m/s")

        l2_comp = process_l2(l1, l2_params)
        metrics = compare_l2(l2_comp, l2_ref)
        print()
        print(format_l2_report(metrics, filename=path.name))

        if args.plot:
            _plot_l2(l1, l2_comp, l2_ref, path.stem)


# ---------------------------------------------------------------------------
# l2-l3 (reference L2 -> computed L3)
# ---------------------------------------------------------------------------

def _cmd_l2_l3(args: argparse.Namespace) -> None:
    from microstructure_tpw.scor160.compare import compare_l3, format_l3_report
    from microstructure_tpw.scor160.io import read_atomix
    from microstructure_tpw.scor160.l3 import process_l3

    for path in args.files:
        print(f"\nProcessing {path.name} ...")
        l1, _l2_params, l2_ref, l3_params, l3_ref, _l4_ref = read_atomix(path)
        print(f"  L2 ref: {l2_ref.shear.shape[1]} samples, "
              f"{l2_ref.shear.shape[0]} shear, {l2_ref.vib.shape[0]} {l2_ref.vib_type}")
        print(f"  L3 params: fft={l3_params.fft_length}, diss={l3_params.diss_length}, "
              f"overlap={l3_params.overlap}, goodman={l3_params.goodman}")

        l3_comp = process_l3(l2_ref, l1, l3_params)
        metrics = compare_l3(l3_comp, l3_ref)
        print()
        print(format_l3_report(metrics, filename=path.name))


# ---------------------------------------------------------------------------
# l1-l3 (L1 -> computed L2 -> computed L3)
# ---------------------------------------------------------------------------

def _cmd_l1_l3(args: argparse.Namespace) -> None:
    from microstructure_tpw.scor160.compare import (
        compare_l2,
        compare_l3,
        format_l2_report,
        format_l3_report,
    )
    from microstructure_tpw.scor160.io import read_atomix
    from microstructure_tpw.scor160.l2 import process_l2
    from microstructure_tpw.scor160.l3 import process_l3

    for path in args.files:
        print(f"\nProcessing {path.name} ...")
        l1, l2_params, l2_ref, l3_params, l3_ref, _l4_ref = read_atomix(path)

        l2_comp = process_l2(l1, l2_params)
        l2_metrics = compare_l2(l2_comp, l2_ref)
        print()
        print(format_l2_report(l2_metrics, filename=path.name))

        l3_comp = process_l3(l2_comp, l1, l3_params)
        l3_metrics = compare_l3(l3_comp, l3_ref)
        print()
        print(format_l3_report(l3_metrics, filename=path.name))


# ---------------------------------------------------------------------------
# l3-l4 (reference L3 -> computed L4)
# ---------------------------------------------------------------------------

def _cmd_l3_l4(args: argparse.Namespace) -> None:
    from microstructure_tpw.scor160.compare import compare_l4, format_l4_report
    from microstructure_tpw.scor160.io import read_atomix
    from microstructure_tpw.scor160.l4 import process_l4

    for path in args.files:
        print(f"\nProcessing {path.name} ...")
        l1, _l2_params, _l2_ref, _l3_params, l3_ref, l4_ref = read_atomix(path)
        print(f"  L3 ref: {l3_ref.n_spectra} spectra, "
              f"{l3_ref.n_shear} shear, {l3_ref.n_wavenumber} wavenumbers")

        l4_comp = process_l4(l3_ref, f_AA=l1.f_AA)
        metrics = compare_l4(l4_comp, l4_ref)
        print()
        print(format_l4_report(metrics, filename=path.name))


# ---------------------------------------------------------------------------
# l2-l4 (reference L2 -> computed L3 -> computed L4)
# ---------------------------------------------------------------------------

def _cmd_l2_l4(args: argparse.Namespace) -> None:
    from microstructure_tpw.scor160.compare import (
        compare_l3,
        compare_l4,
        format_l3_report,
        format_l4_report,
    )
    from microstructure_tpw.scor160.io import read_atomix
    from microstructure_tpw.scor160.l3 import process_l3
    from microstructure_tpw.scor160.l4 import process_l4

    for path in args.files:
        print(f"\nProcessing {path.name} ...")
        l1, _l2_params, l2_ref, l3_params, l3_ref, l4_ref = read_atomix(path)

        l3_comp = process_l3(l2_ref, l1, l3_params)
        l3_metrics = compare_l3(l3_comp, l3_ref)
        print()
        print(format_l3_report(l3_metrics, filename=path.name))

        l4_comp = process_l4(l3_comp, f_AA=l1.f_AA)
        l4_metrics = compare_l4(l4_comp, l4_ref)
        print()
        print(format_l4_report(l4_metrics, filename=path.name))


# ---------------------------------------------------------------------------
# l1-l4 (L1 -> L2 -> L3 -> L4)
# ---------------------------------------------------------------------------

def _cmd_l1_l4(args: argparse.Namespace) -> None:
    from microstructure_tpw.scor160.compare import (
        compare_l2,
        compare_l3,
        compare_l4,
        format_l2_report,
        format_l3_report,
        format_l4_report,
    )
    from microstructure_tpw.scor160.io import read_atomix
    from microstructure_tpw.scor160.l2 import process_l2
    from microstructure_tpw.scor160.l3 import process_l3
    from microstructure_tpw.scor160.l4 import process_l4

    for path in args.files:
        print(f"\nProcessing {path.name} ...")
        l1, l2_params, l2_ref, l3_params, l3_ref, l4_ref = read_atomix(path)

        l2_comp = process_l2(l1, l2_params)
        l2_metrics = compare_l2(l2_comp, l2_ref)
        print()
        print(format_l2_report(l2_metrics, filename=path.name))

        l3_comp = process_l3(l2_comp, l1, l3_params)
        l3_metrics = compare_l3(l3_comp, l3_ref)
        print()
        print(format_l3_report(l3_metrics, filename=path.name))

        l4_comp = process_l4(l3_comp, f_AA=l1.f_AA)
        l4_metrics = compare_l4(l4_comp, l4_ref)
        print()
        print(format_l4_report(l4_metrics, filename=path.name))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_l2(l1, l2_comp, l2_ref, title: str) -> None:
    """Plot L1 vs computed-L2 vs reference-L2 for visual inspection."""
    import matplotlib.pyplot as plt

    fs = l1.fs_fast
    t = np.arange(l1.n_time) / fs

    ref_sec = l2_ref.section_number
    sections = np.unique(ref_sec[ref_sec > 0])
    if len(sections) == 0:
        print("  No sections to plot.")
        return

    sec_mask = ref_sec == sections[0]
    idx = np.where(sec_mask)[0]
    t_sec = t[idx]

    n_sh = min(l1.n_shear, 2)
    n_vib = min(l1.n_vib, 2)
    n_rows = 1 + n_sh + n_vib

    _fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    ax = axes[0]
    ax.plot(t_sec, l2_ref.pspd_rel[idx], "b-", alpha=0.7, label="reference")
    ax.plot(t_sec, l2_comp.pspd_rel[idx], "r--", alpha=0.7, label="computed")
    ax.set_ylabel("W [m/s]")
    ax.legend(loc="upper right")
    ax.set_title(f"{title} -- Section {int(sections[0])}")

    for i in range(n_sh):
        ax = axes[1 + i]
        ax.plot(t_sec, l2_ref.shear[i, idx], "b-", alpha=0.5, lw=0.5, label="reference")
        ax.plot(t_sec, l2_comp.shear[i, idx], "r-", alpha=0.5, lw=0.5, label="computed")
        ax.set_ylabel(f"sh{i + 1} [s-1]")
        ax.legend(loc="upper right")

    for i in range(n_vib):
        ax = axes[1 + n_sh + i]
        ax.plot(t_sec, l2_ref.vib[i, idx], "b-", alpha=0.5, lw=0.5, label="reference")
        ax.plot(t_sec, l2_comp.vib[i, idx], "r-", alpha=0.5, lw=0.5, label="computed")
        label = "Acc" if l1.vib_type == "ACC" else "Vib"
        ax.set_ylabel(f"{label}{i + 1}")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
