# microstructure_sensors/

Rockland calibration sheets (PDF, gitignored — large/vendor files) and the
tracked **shear-probe sensitivity registry** `shear_sensitivities.csv`.

Holdings as of 2026-10-12: **190 shear-probe sheets for 106 serials**, dated
2004-10-14 to 2026-06-22. (`JAC_CT_SN023.pdf` is a CT certificate, not a shear
sheet, and is skipped by the parser.) The 2026-10-12 batch ("OSU Shear
Calibration Certificates", 129 PDFs, 84 of them new here) added most of the
pre-2020 history, including the ASTRAL and CASPER MicroRider probes requested
in `sheet_requests.md`. Some early calibrations are still known only as a later
sheet's "previous" entry (M1497 and M1501 2016-06-06, M2090 2019-07-02).

## sensor_deployments.csv

Which sensor was fitted where, and when — one row per (sensor, campaign,
instrument, channel) with first/last use, file count and the configured value,
read from the instruments' own `.p` configs. Covers CASPER 2015/2017, ASTRAL
2023 and ARCTERX 2022/2025: 21 distinct FP07 beads and 17 shear probes.

This is the prerequisite for any per-bead calibration history, since a probe
moves between channels and instruments between cruises — see issue #189 (FP07
calibration sheets of our own). Rebuild by scanning a campaign's `.p` configs.

## sheet_requests.md

Sensors found in deployment data for which **no sheet is held here** — the
standing batch request to Rockland. Regenerate/extend it by scanning a
campaign's `.p` configs and differencing against the PDFs in this directory.
As of 2026-09-07 it covers CASPER and ASTRAL: 16 shear probes and 12 FP07
thermistors (we hold no thermistor sheets at all). Fourteen of the sixteen
shear sheets arrived 2026-10-12; Rockland has **no record of M1000 or M1001**.

## shear_sensitivities.csv

One row per (probe, calibration). Columns:

| column | meaning |
|---|---|
| `serial` | probe serial number (e.g. `M1458`) |
| `cal_date` | calibration date, ISO `YYYY-MM-DD` |
| `sens` | sensitivity as printed by Rockland |
| `units` | `V/(m^2 s^-2)` — volts per (m/s)², the sheets' convention |
| `source` | `sheet` (parsed from a Rockland PDF) or `manual` (historical/other records) |
| `sheet` | PDF filename in this directory (empty for `manual` rows) — uniquely identifies the sheet across users |
| `recal_due` | Rockland's "Recommended re-calibration" date, when the sheet carries one |
| `pressure_test_date` | the sheet's pressure-test date; applies to the sheet's own calibration only (empty for previous-calibration and `manual` rows) |
| `notes` | provenance/caveats; `previous-calibration entry on this sheet` marks the sheet's own history row |

**The registry is a record, not a control surface.** `rsi-tpw sensors
--cal-dir` builds its timelines by re-parsing the PDFs; nothing in the package
reads this CSV back into a timeline (`_read_registry` serves only
`update_sensitivity_csv`). So a `notes` caveat — e.g. the `DATE WRONG` flag
on M2245's 2023-06-07 row — is guidance for a human reading the registry, and
does **not** suppress that point in the staleness or mismatch check. If a
doubtful point ever governs a real observation, either exclude its PDF from
the directory or account for it by hand. Excluding a PDF also drops that
sheet's own (valid) calibration, which is why the wrong-date "previous"
entries below are annotated but kept.

Update workflow: drop new Rockland PDFs into this directory and run

```bash
rsi-tpw cal-csv microstructure_sensors
```

Idempotent — existing rows (including hand-added `source=manual` history) are
preserved; a sheet's own entry upgrades a previous-calibration attestation of
the same point; conflicting sensitivities for the same probe+date are kept
side by side and reported. The command exits 1 whenever any conflict exists
or any PDF fails to parse, so in this directory it always does (the M1254 and
M3038 conflicts below, and the non-shear `JAC_CT_SN023.pdf`). Add `manual`
rows by hand with a
provenance note in `notes`. A row must have exactly nine comma-separated
fields, so quote any value containing a comma (`"a, b"`) — an unquoted one is
rejected with the file and line number, leaving the registry untouched rather
than truncating the note. (Registries written before the
`pressure_test_date` column existed are read transparently and upgraded on
the next run.) The `.p`-config cross-check tool is
`rsi-tpw sensors --cal-dir microstructure_sensors --shear`.

### Pressure-test timing

Rockland guidance (via Rockland support, Jul 2026): the pressure test should
occur **after** the calibration, or **~10 days or more before** it. A
calibration performed within ~10 days *after* the pressure test is suspect —
the probe may not have re-settled from the squeeze. In this collection, all
nine probes with both a too-soon calibration and a later compliant
recalibration off the same pressure test show the too-soon one reading high
(+0.4% to +7.0%, median +2.0%). Compare `cal_date` against
`pressure_test_date` when judging a sheet; where a probe has a later,
compliant recalibration, prefer it.

### Sheet layouts

Five layouts are held; the parser handles all of them. They differ in how the
serial, the date, the current sensitivity and the previous calibration are
labeled. (The 2026 sheets also differ from 2024–25 in how they label the
pressure-test date; the parser handles both.)

| era | current sensitivity | previous calibration |
|---|---|---|
| 2004–2019 (tabular) | `Sensitivity (Volts/(m/s)^2)= 0.0805` | operator's free-text `NOTE:` only — **not parsed**, see below |
| 2020 | *(unlabeled)* — see below | none printed |
| through mid-2023 | `sens: 0.0720 V` | prose, wrapped: `Previous calibration on 2021-11-10 with` / `sensitivity 0.0655` |
| mid-2023 | `Sensitivity (sens): 0.1115 V` | — |
| 2024-06 onward | `Sensitivity (sens or S): 0.0777 V` | `Previous Calibration Date:` + `Previous Sensitivity:` |

The **2004–2019 tabular** form is a spreadsheet printout whose page 1 is the
head-angle/voltage table (the 2004 sheet is RGL Consulting's; Rockland kept the
form until 2019-04). The serial and calibration date share a line with no
"Calibration" label, and the date is whatever the operator typed —
`2013-08-08`, `2019_03_29`, `14 Oct 2004` and `01-Mar-13` all occur:

```
Date 2013-08-08 Probe serial # M1021
Total Gain 1 Sensitivity (Volts/(m/s)^2)= 0.0805 Pressure (psi) 8000
```

These sheets print no recommended re-calibration date (staleness falls back to
the 12-month maximum age) and no pressure-test date. Most record a 1-hour,
8000–8700 psi pressure soak as part of the calibration session instead; the
2004 and 2013 M289 sheets leave the pressure blank. The previous calibration
appears only in a hand-typed `NOTE:` (`re-calibration, previous S=0.0791 on
2016-11-16`, `Previous Sens. 0.0805`, ...), and the parser deliberately
ignores it. Many are undated; some are cumulative lists that pypdf wraps onto
the next line (M1133 2016-10-12: `previous sensitivity s=0.0603 ,` /
`0.0596,0.0579`); M289's 2016-04-11 NOTE is a stale copy (0.0524, skipping
2013-12-09's 0.0717); and M1264's 2018-03-20 NOTE cites "0.0411 on
2018-03-20", its own date. Every dated NOTE value that is right is also the
current value of a sheet on file.

On the **2020** sheets the `Serial Number, SN :` and
`Sensitivity [Volts / (m/s)²] :` labels are drawn as graphics, so `pypdf` drops
them and only their *values* survive, glued onto the first extracted line:

```
M2244 0.0678
Operator : S. Yasuda  Probe PN : ...
Calibration Date : 2020-03-10 ...
```

The parser reads that first line as a fallback — only after the labeled
patterns have found nothing, so a properly labeled sheet can never be
overridden. Every other layout opens with a letterhead or a title instead, so
there is no ambiguity. These sheets print no pressure-test date.

Filename dates are not authoritative — several sheets are named for a date
other than the calibration date they carry (e.g. `M2475_2021_11_12.pdf` records
a 2021-11-09 calibration). The registry always takes the date from the sheet
body.

### Known bad sheets

**Some sheets print the right previous sensitivity under the wrong date.**
In each case the value is bit-identical to a real earlier calibration, but the
date is not that calibration's:

| sheet (as filed) | previous entry printed | real previous calibration |
|---|---|---|
| `M2245_2026_04_28.pdf` | 2023-06-07, 0.0682 | 2020-03-12, 0.0682 |
| `M3036_2026_04_28.pdf` | 2023-06-07, 0.1223 | 2024-07-15, 0.1223 |
| `M3037_2026_04_28.pdf` | 2023-06-07, 0.0952 | 2024-07-15, 0.0952 |
| `M1194_2026_06_22.pdf` | 2026-03-10, 0.0591 | 2017-08-24, 0.0591 |
| `M1496_2022_6_20.pdf` | 2021-06-03, 0.0649 | 2016-06-03, 0.0649 (same day and month: likely a mistyped year) |

- Each 2026-04-28 sheet *as first issued* prints 2023-06-07. That is
  impossible for M3036 and M3037, which were pressure-tested 2024-07-05 and
  first calibrated 2024-07-09.
- Rockland generated M1254's and M1458's sheets **twice on 2026-07-03**. The
  first issue (10:34 and 10:38 −07:00) carries 2023-06-07 and 2026-06-02; the
  re-issue ~1.5 h later carries **2021-04-27** for both. The re-issues are the
  copies filed here, and M1254's 2021-04-27 is confirmed by
  `M1254_2021_4_27.pdf`. If a first issue turns up again (the 2026-10-12 batch
  carried both), do not file it. The other 2026 sheets above are original
  issues (generated on their calibration days) and were never corrected.
- M2245's 2023-06-07 row, flagged `DATE DOUBTFUL` since 2026-09, is therefore
  now marked `DATE WRONG`.

The affected rows are annotated in `notes`. Numerically they are harmless —
each repeats the real previous value, so hold-previous returns the same
sensitivity — but they make a stale calibration look fresher than it was: at
ARCTERX-2025 Interior, M2245's observations are governed by the phantom 2023
point and read as ~20 months old, when the real calibration (2020-03-12) was
59 months old. Both ages are past the 12-month limit, so the stale flag still
fires.

**`M1254_2021_4_27.pdf` misquotes the 2015 value.** Its previous entry prints
0.0820 on 2015-02-06; M1254's own 2015-02-06 sheet prints 0.0802, and that
sheet's angle table refits to 0.0801. 0.0820 is a digit transposition. Both
rows are kept, and `cal-csv` reports the conflict. `rsi-tpw sensors` keeps
the first sheet in name order, 0.0802.

**Probably two physical probes printed as M1194; and an M1196-B.**
`M1194_2016_04_07.pdf` reads 0.0793, between M1194 sheets reading **0.0549
(2016-03-14) and 0.0551 (2016-04-18)**: a +44% jump and back within 11 days.
The 04-18 sheet's NOTE cites 2016-03-14 as its previous calibration, skipping
04-07. The 04-07 sheet was distributed to OSU as `M1194_2_...`, in the
VMP-250IR SN 194 build package. Its capacitance (1.087 nF, against 0.846–0.941
nF on M1194's five other sheets) points the same way. But capacitance alone is
weak: it varies by 10–19% within single probes here (M1133, M1201, M289). The
registry keeps the serial as printed, so **M1194's timeline probably mixes two
probes**.

This matters at CASPER-West. The MicroRider's 2018 config patch says sh1
M1194 @ 0.0793 (the 04-07 sheet), while the M1194 lineage was recalibrated to
**0.0591 on 2017-08-24**, two days before that MR's first 2017 file. If the
recalibrated probe is the one that flew, sh1 ε there is 1.80× low. Against
that: sh1/sh2 agree to 0.939 as configured, which a 1.80× sh1-only error would
break unless M1493 had drifted similarly. Unresolved.

`M1196_2016_04_07.pdf`, calibrated the same day, prints the serial
**M1196-B**. Its sensitivity (0.0637) and capacitance (1.090 nF vs 1.194–1.228)
are within M1196's normal range. The case that it is a different probe rests
mainly on the printed suffix.

`M3038_2024_07_15.pdf` is character-for-character identical to
`M3039_2024_07_15.pdf` apart from the serial number — same sensitivity, `c`,
capacitance, calibration temperature, DC offset and data page — and its
"previous" entry contradicts M3038's own `M3038_2024_07_09.pdf` (0.1189 vs
0.1093). It is very likely M3039's report mis-issued under M3038's serial, so
both M3038 rows sourced from it are annotated `SUSPECT` in `notes`. Confirm
with Rockland before using M3038's 2024-07-15 sensitivity. No other sheet in
the directory shares a probe-specific fingerprint with another (re-checked
2026-10-12 over all 190 sheets: angle tables for the tabular form;
capacitance, `c`, temperature and DC offset for the rest).
