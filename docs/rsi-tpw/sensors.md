# `rsi-tpw sensors` — microstructure sensor inventory & calibration check

Walk a tree of Rockland `.p` files and summarize, per sensor serial number, the
date range of use, the file count, the platform(s), and whether the sensor's
calibration parameters changed. Only each file's 128-byte header and embedded
INI config block are read (never the data records), so the scan is fast over
large trees.

```bash
rsi-tpw sensors VMP/                      # inventory shear + FP07 (default)
rsi-tpw sensors VMP/ --shear --compact    # one line per shear probe
rsi-tpw sensors VMP/ --csv probes.csv     # per-(file,channel) table
```

## Checking shear sensitivities against calibration sheets — `--cal-dir`

Rockland ships a **Shear Probe Calibration Report** (a PDF) with each probe,
giving the probe serial number, its sensitivity (config `sens` / *S*), the
calibration date, and — when the probe has been calibrated before — the
*previous* calibration date and sensitivity. Point `--cal-dir` at a directory of
those PDFs and `sensors` checks every shear probe it finds in the `.p` files
against them, reporting where a file's configured `sens` disagrees with the
calibration that was in effect when the file was recorded.

```bash
rsi-tpw sensors VMP/ --cal-dir /path/to/microstructure_sensors
rsi-tpw sensors VMP/ --cal-dir /path/to/sheets --cal-tol 2   # flag beyond 2%
```

Sheets are matched to probes by serial number. The serial and calibration date
are read from the PDF text (`Probe SN:` / `Calibration Date:`), falling back to
the filename (`M<sn>_<YYYY>_<MM>_<DD>.pdf`) when the text can't be parsed, and
the two are cross-checked when both are present. The **sensitivity** is only ever
taken from the PDF text. Serial matching is case-insensitive.

### Sensitivity model — hold-previous

The sensitivity applied to an observation is that of the **most recent
calibration on or before the observation's date**. So a file recorded between a
2021 and a 2026 calibration is checked against the **2021** value (the
calibration then in effect), not an interpolation. A file recorded before the
earliest known calibration is clamped to that earliest value and marked
`[before earliest cal]`.

Linear interpolation of the drift *between* calibration dates is intentionally
**not** done yet — that convention is still being settled with Rockland. The
lookup carries a `mode`, so interpolation can be added later without changing
how the command is used.

### Output

Only **mismatches** are reported (configured vs in-effect sensitivity differing
by more than `--cal-tol`, default 1%), grouped by probe with the file count,
observation-date span, and percent difference. A coverage line notes any probes
that had no matching sheet, and observations skipped for a missing clock or a
blank configured `sens`.

### The `cal` extra

Reading the PDFs needs [`pypdf`](https://pypi.org/project/pypdf/), an optional
dependency:

```bash
pip install 'microstructure-tpw[cal]'
```

It is imported only when `--cal-dir` is used, so the rest of `sensors` works
without it; using `--cal-dir` without it prints a clear install hint. The
calibration-sheet directory is always an external path you supply — it is never
part of this repository.
