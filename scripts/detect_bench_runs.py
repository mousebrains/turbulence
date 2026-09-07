#!/usr/bin/env python3
"""Detect a bench / test-probe run from the DATA, since provenance is unreliable.

    python scripts/detect_bench_runs.py FILE.p [FILE.p ...]

Cast sheets and file names routinely do not record which files are bench runs,
so the classification has to come from the data. Critically it must NOT assume
the readings are correct: a bench run exists to catch bad electronics, so a
test keyed on "the thermistor reads T_0" would reject exactly the file you most
need to find.

Two channel metrics:

FP07 -- "is a thermistor sitting on a fixed resistor?"
    Constancy, not value. A test resistor gives ADC dither only (~1e-5 C); a
    real FP07 drifts ~0.4 C on deck and 1-3 C in water. A minimum duration
    stops a short file being "constant" by brevity. Only ONE thermistor need be
    constant (the other may be open or dead), and nearness to T_0 is reported
    as corroboration, never required.

SHEAR -- "is this channel an open circuit?"
    std expressed in ADC COUNTS, unit-free and instrument-independent:

        counts = std_physical * (2*sqrt(2) * diff_gain * sens) / (adc_fs / 2**bits)

    An open circuit sits at the quantisation floor (sub-LSB). Exactly zero
    counts is a DEAD/railed channel, not an open one -- variance, not value,
    separates those.

Validated separation on labelled ARCTERX-2022 / SUNRISE files:

    shear [ADC counts]      FP07 std [C]     verdict
    ------------------      ------------     -------
    1.06 - 1.14 (10-12)     2e-5 .. 6e-5     BENCH, test probes fitted
    0.00        (1)         3.6e-15          DEAD / railed acquisition
    48 - 148                0.08 .. 0.39     real probes, static on deck
    4550 - 19700            1.5  .. 2.0      in water

A ~50x margin separates an open circuit from the nearest connected case.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from odas_tpw.rsi import PFile
from odas_tpw.rsi.p_file import parse_config, read_config_string

MIN_DURATION_S = 30.0  # below this, constancy is meaningless
THERM_STD_MAX = 0.01  # deg C -- a test resistor is ~1e-5, a real probe >= 0.08
SHEAR_COUNTS_MAX = 2.0  # an open circuit is sub-LSB
T0_TOLERANCE_C = 0.5  # only for the corroborating value check


def analyse(path: str | Path) -> dict[str, Any]:
    """Per-channel constancy (therm) and ADC-count noise (shear) for one file."""
    pf = PFile(str(path))
    chans = {c.get("name"): c for c in parse_config(read_config_string(path))["channels"]}
    out: dict[str, Any] = {
        "file": Path(path).name,
        "duration": float(pf.t_fast[-1] - pf.t_fast[0]),
        "therm": [],
        "shear": [],
    }

    for name in ("T1", "T2"):
        if name not in pf.channels:
            continue
        v = np.asarray(pf.channels[name], dtype=float)
        raw_t0 = chans.get(name, {}).get("t_0")
        t0_c = float(raw_t0) - 273.15 if raw_t0 else np.nan
        median = float(np.nanmedian(v))
        out["therm"].append(
            {
                "name": name,
                "std": float(np.nanstd(v)),
                "median": median,
                "t0": t0_c,
                "deviation": abs(median - t0_c) if np.isfinite(t0_c) else np.nan,
                "n_unique": int(np.unique(v[np.isfinite(v)]).size),
            }
        )

    for name in ("sh1", "sh2"):
        if name not in pf.channels:
            continue
        cfg = chans.get(name, {})
        try:
            diff_gain = float(cfg["diff_gain"])
            sens = float(cfg["sens"])
            adc_fs = float(cfg.get("adc_fs", 4.096))
            adc_bits = float(cfg.get("adc_bits", 16))
        except (KeyError, TypeError, ValueError):
            continue  # cannot convert to counts without the coefficients
        v = np.asarray(pf.channels[name], dtype=float)
        lsb = adc_fs / (2.0**adc_bits)
        out["shear"].append(
            {
                "name": name,
                "std": float(np.nanstd(v)),
                "counts": float(np.nanstd(v)) * (2 * np.sqrt(2) * diff_gain * sens) / lsb,
                "n_unique": int(np.unique(v[np.isfinite(v)]).size),
            }
        )
    return out


def verdict(a: dict[str, Any]) -> str:
    """Classify one file from its per-channel metrics."""
    if a["duration"] < MIN_DURATION_S:
        return "TOO SHORT to judge"

    constant = [t for t in a["therm"] if t["std"] < THERM_STD_MAX]
    open_shear = [s for s in a["shear"] if 0 < s["counts"] < SHEAR_COUNTS_MAX]

    stuck_shear = [s for s in a["shear"] if s["n_unique"] <= 1]
    if (
        a["shear"]
        and len(stuck_shear) == len(a["shear"])
        and all(t["n_unique"] <= 1 for t in a["therm"])
    ):
        return "DEAD / railed record"

    if constant and open_shear:
        near_t0 = [
            t for t in constant
            if np.isfinite(t["deviation"]) and t["deviation"] < T0_TOLERANCE_C
        ]
        if near_t0:
            return "BENCH (test probes)"
        return "BENCH (test probes) [therm constant but NOT near T_0 -> suspect electronics]"
    if constant:
        return "therm constant, shear NOT open"
    if open_shear:
        return "shear open, no constant therm"
    return "not a bench run"


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    warnings.filterwarnings("ignore")
    for path in argv:
        try:
            a = analyse(path)
        except Exception as exc:  # a bad file must not stop the sweep
            print(f"{Path(path).name:26s}  ERROR {type(exc).__name__}: {exc!s:.50}")
            continue
        therm = " ".join(
            f"{t['name']}={t['median']:.3f}(sd{t['std']:.1e},dT0={t['deviation']:.2f})"
            for t in a["therm"]
        )
        shear = " ".join(f"{s['name']}={s['counts']:.2f}cnt(nu{s['n_unique']})" for s in a["shear"])
        print(f"{a['file']:26s} {a['duration']:7.0f}s  {therm}  {shear}   -> {verdict(a)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
