# `rsi-tpw sensors` — microstructure sensor inventory & calibration check

Walk a tree of Rockland `.p` files and summarize, per sensor serial number, the
date range of use, the file count, the platform(s), and whether the sensor's
calibration parameters changed. Only each file's 128-byte header and embedded
INI config block are read (never the data records), so the scan is fast over
large trees.

```bash
rsi-tpw sensors VMP/                      # every kind: shear + FP07 + EM (default)
rsi-tpw sensors VMP/ --shear --compact    # one line per shear probe
rsi-tpw sensors MR/ --em                  # AEM1-G EM speed sensors only
rsi-tpw sensors VMP/ --csv probes.csv     # per-(file,channel) table
```

## Sensor kinds

| flag | kind | tracked parameters |
|---|---|---|
| `--shear` | shear probe | `adc_fs`, `adc_bits`, `diff_gain`, `sens`, `cal_date` |
| `--fp07` | FP07 thermistor | `adc_fs`, `adc_bits`, `a`, `b`, `g`, `e_b`, `beta_1`, `beta_2`, `t_0`, `cal_date` |
| `--em` | JFE AEM1-G EM speed sensor (`U_EM`) | `a`, `b`, `cal_date` |

The EM belongs here for the same reason the shear probes do: `a` and `b` set the
through-water speed, and dissipation goes as **U⁻⁴**, so a coefficient that
changes mid-deployment — or a unit swapped without the config following it —
moves every ε downstream. Scanning both osu685 MicroRider trees at once reports
it plainly: the same glider (`slocum_glider` SN 435) carried **EM 046 in 2023
and EM 066 in 2025**.

Only `aem1g_d` is matched, which was checked against every MicroRider tree to
hand rather than assumed. The neighboring `EM_Cur` / `EMC_Cur` channel is a
plain `voltage` current monitor carrying no calibration, so it is deliberately
not counted as a sensor use.

**AppleDouble sidecars are skipped.** macOS writes `._name.p` next to `name.p`
on filesystems without native fork support — which is every SMB share the
campaign data lives on — and they glob as `.p` files. They are a few hundred
bytes of AppleDouble, so each one would otherwise be reported as a spurious
`invalid header_size=0` error. One Bank Seaspider tree globs 224 files, of which
216 are real.

## Auditing differential gains — `--diff-gain`

`diff_gain` is the pre-emphasis differential gain of the **instrument's own
amplifier chain**, not a property of the probe screwed into it. That makes it
invisible to the sensor inventory above, in two separate ways: the inventory is
keyed on *sensor* serial, so a shear `diff_gain` gets attributed to whichever
probe happened to be installed; and it deliberately drops the `X_dX`
pre-emphasized channels (`T1_dT1`, `T2_dT2`, `P_dP`) because they are the same
physical sensor as their base channel — so the thermistor and pressure gains are
never reported at all.

The stakes are the shear sensitivity's. The conversion is

```
shear = (adc_fs / 2^adc_bits * counts + offset) / (2*sqrt(2)*diff_gain*sens)
```

so **ε goes as `(diff_gain·sens)⁻²` and χ as `diff_gain⁻²`** — a 2% gain error is
4% in both, and an order-of-magnitude error is two orders of magnitude in ε.

```bash
rsi-tpw sensors --diff-gain VMP/                       # audit
rsi-tpw sensors --diff-gain --diff-gain-csv gains.csv VMP/
rsi-tpw sensors --diff-gain --diff-gain-strict VMP/    # exit 4 on an outlier
```

### Outliers

A gain more than **2× from the fleet median for the same channel name** is
flagged, and a gain **≤ 0** is reported outright as invalid — it divides the
shear conversion by zero or flips its sign, and needs no fleet median to be
wrong. The comparison is relative rather than against a hardcoded band because
the plausible range differs by channel class — the differentiator channels sit
near 1, `P_dP` near 20 — and hardcoding either would be a guess.

The median takes **one vote per instrument, not per file**, so a campaign with a
thousand files from one unit cannot drag the fleet. Files carrying no
`instrument_info` are excluded from the vote entirely rather than merged into a
fabricated instrument.

The gate is **per channel**, not per scan: a channel only one or two units carry
cannot be checked even in a large scan. Those are listed under `NOT CHECKED`, so
"no outliers" is never a false all-clear.

This exists because of a real miss. On ARCTERX-2023 one VMP carried
`diff_gain = 0.09` on **both** shear channels where its three siblings in the
same cruise carried 0.92–0.99 — which inflates its ε by roughly 110× — and
nothing in the processing chain noticed:

```
  SN   132 vmp-250-IR       sh1        0.09  vs fleet median 0.941  (0.10x)
  SN   132 vmp-250-IR       sh2        0.09  vs fleet median 0.922  (0.10x)
```

### Changes over time are reported, not flagged

Gains follow the **electronics**, so they legitimately change when an instrument
is rebuilt. The original VMPs (SN < 400) were Persistor CF2 builds; several have
since been upgraded to RDL hardware, and the gains changed with the boards. That
is informative, not an error, so it is shown as a dated transition:

```
SN 142  model vmp-250
    sh1        CHANGED over time:
          0.96   2019-05-28 → 2019-05-30   (12 file(s))
         0.953   2021-06-22 → 2021-07-01   (58 file(s))
      (gains follow the electronics; a change is expected when an instrument is rebuilt)
```

Grouping is on the **serial number alone**, and that is load-bearing: a real
rebuild changes the `model` string too (SN 142 became `VMP250IR_RDL`, SN 479
`VMP250IR_RT`), so keying on `(SN, model)` would split the instrument in two and
suppress exactly the transition this is meant to surface. The models seen are
printed alongside as `model vmp-250 → VMP250IR`.

The `model` string can also **lag** the hardware — SN 142's 2021 files still read
`vmp-250` while already carrying its RDL gains — which is a second reason not to
treat it as a hardware key.

### Exit codes

`--diff-gain-strict` returns **4** when the audit flags an outlier — distinct
from `--cal-strict`'s 3 and from 1 (scan failed), so a script can tell the three
apart.

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
rsi-tpw sensors VMP/ --cal-dir /path/to/sheets --cal-tol 0.0001   # widen the threshold
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

> **Completeness assumption:** the check is only as good as the sheets
> directory. A probe that *was* recalibrated but whose newer sheet is missing
> from `--cal-dir` will be checked (and possibly flagged stale or mismatching)
> against the older calibration. Keep the directory complete, and treat every
> stale annotation as "verify no newer sheet exists" — which is exactly what it
> says.

### Stale calibrations — `--cal-max-age-months`

Each sheet's **"Recommended re-calibration"** date is parsed alongside the
calibration itself. An observation whose *governing* calibration is past that
date at observation time is annotated **stale**:

```
[cal 15 months old at use; recal was recommended by 2025-07-09 — verify no newer sheet exists]
```

When a sheet carries no recommended-recal line, the fallback is a maximum age
of `--cal-max-age-months` (default **12** months — Rockland's actual
recommendation; every parsed sheet's recal date is exactly cal + 12 months).
The flag only changes the fallback; sheets with the line always use its date.
Stale annotations appear on mismatch rows, and the summary reports the stale
count even when there are no mismatches ("No mismatches: ... (M observation(s)
governed by stale calibrations)"). An observation *before* the earliest known
calibration is flagged `[before earliest cal]` instead, never stale.

### Output

Only **mismatches** are reported (configured vs in-effect sensitivity differing
by more than `--cal-tol`), grouped by probe with the file count, observation-date
span, the absolute difference `Δ`, and the percent difference (shown for
context). `--cal-tol` is an **absolute** sensitivity threshold in the same units
as `sens` — not a percentage — because sensitivity is an absolute quantity and
the sheets quote it to four decimals. The default is `0.00005`, half that
4th-decimal resolution, so any difference that would round to a different quoted
value is flagged. A coverage line notes any probes that had no matching sheet,
and observations skipped for a missing clock or a blank configured `sens`.

### Exit codes — `--cal-strict`

By default the calibration check is **report-only**: mismatches are printed but
the exit code stays 0. For CI / scripted gating, pass `--cal-strict` to exit
with code **3** (distinct from the scan-failure code 1) when the check found
mismatches outside `--cal-tol`:

```bash
rsi-tpw sensors VMP/ --cal-dir sheets/ --cal-strict && echo "sens OK"
```

| Exit code | Meaning |
|-----------|---------|
| 0 | Scan (and, with `--cal-dir`, the check) completed; no strict failure |
| 1 | No `.p` files found / unwritable CSV / every file failed to parse / `--cal-strict` without `--cal-dir` |
| 3 | `--cal-strict` and the calibration check found mismatches outside `--cal-tol` |

`--cal-strict` requires `--cal-dir` (it errors otherwise); stale annotations
alone never trip it — only sensitivity mismatches do.

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
