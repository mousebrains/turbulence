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

Update workflow: add new Rockland PDFs here, re-run the sheet parser
(`odas_tpw.rsi.shear_cal.parse_sheet_text`) to append their rows, or add
`manual` rows by hand with a provenance note. Keep rows sorted by
(serial, cal_date). The `.p`-config cross-check tool is
`rsi-tpw sensors --cal-dir microstructure_sensors --shear`.
