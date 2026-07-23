# microstructure_sensors/

Rockland calibration sheets (PDF, gitignored — large/vendor files) and the
tracked **shear-probe sensitivity registry** `shear_sensitivities.csv`.

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
| `notes` | provenance/caveats; `previous-calibration entry on this sheet` marks the sheet's own history row |

Update workflow: drop new Rockland PDFs into this directory and run

```bash
rsi-tpw cal-csv microstructure_sensors
```

Idempotent — existing rows (including hand-added `source=manual` history) are
preserved; a sheet's own entry upgrades a previous-calibration attestation of
the same point; conflicting sensitivities for the same probe+date are kept
side by side and reported (exit 1). Add `manual` rows by hand with a
provenance note in `notes`. The `.p`-config cross-check tool is
`rsi-tpw sensors --cal-dir microstructure_sensors --shear`.

### Sheet layouts

Rockland has issued three layouts; the parser handles all three. They differ
only in how the current sensitivity and the previous calibration are labeled:

| era | current sensitivity | previous calibration |
|---|---|---|
| through mid-2023 | `sens: 0.0720 V` | prose, wrapped: `Previous calibration on 2021-11-10 with` / `sensitivity 0.0655` |
| mid-2023 | `Sensitivity (sens): 0.1115 V` | — |
| 2024-06 onward | `Sensitivity (sens or S): 0.0777 V` | `Previous Calibration Date:` + `Previous Sensitivity:` |

Filename dates are not authoritative — several sheets are named for a date
other than the calibration date they carry (e.g. `M2475_2021_11_12.pdf` records
a 2021-11-09 calibration). The registry always takes the date from the sheet
body.

### Known bad sheet

`M3038_2024_07_15.pdf` is character-for-character identical to
`M3039_2024_07_15.pdf` apart from the serial number — same sensitivity, `c`,
capacitance, calibration temperature, DC offset and data page — and its
"previous" entry contradicts M3038's own `M3038_2024_07_09.pdf` (0.1189 vs
0.1093). It is very likely M3039's report mis-issued under M3038's serial, so
both M3038 rows sourced from it are annotated `SUSPECT` in `notes`. Confirm
with Rockland before using M3038's 2024-07-15 sensitivity. No other sheet in
the directory shares a probe-specific fingerprint with another.
