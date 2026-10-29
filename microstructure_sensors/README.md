# microstructure_sensors/

Rockland calibration sheets (PDF, gitignored — large/vendor files), the
tracked **shear-probe sensitivity registry** `shear_sensitivities.csv`, and our
own **FP07 in-situ calibration sheet** `fp07_calibrations.yaml` (below).

Holdings as of 2026-10-26: **200 shear-probe sheets for 116 serials**, dated
2004-10-14 to 2026-06-22, plus eight FP07 production reports (below).
(`JAC_CT_SN023.pdf` is a CT certificate, not a shear sheet, and is skipped by
the parser.) The 2026-10-12 batch ("OSU Shear Calibration Certificates", 129
PDFs, 84 of them new here) added most of the pre-2020 history, including the
ASTRAL and CASPER MicroRider probes requested in `sheet_requests.md`. Some
early calibrations are still known only as a later sheet's "previous" entry
(M1497 and M1501 2016-06-06, M2090 2019-07-02).

The 2026-10-26 batch came from an outside group's 2026 VMP deployments. It
added **10 shear sheets for 10 new serials**: M2662 (2023-04-12), M3244
(2025-04-29) and M3456–M3459 and M3461–M3464 (all 2026-04-07). None of them
prints a previous calibration. All ten were pressure-tested 12–141 days before
calibration, which complies with the timing rule below. None shares a
fingerprint (`c`, capacitance, calibration temperature, DC offset) with any
of the 130 sheets that print those. The same check re-finds the known
M3038/M3039 pair.

### FP07 production reports (`T*.pdf`)

Rockland issues each FP07 a one-page *FP07 Temperature Probe Production
Report*. **It is not a calibration.** It states "This probe is uncalibrated"
and recommends an in-situ fit. It records the serial, part number, the bead's
resistance at one temperature, and its pressure-test and dip-test dates. Eight
are held: T3298, T3299, T3301–T3304, T3306 and T3308, issued 2026-03-13,
pressure-tested 2026-01-09. Their resistances are 2.163–2.499 kΩ at 19.6 °C.
`cal-csv` and `sensors --cal-dir` skip them, since they carry no
sensitivity.

That single resistance is still useful for **bead identity**. When two beads
share the same nominal Steinhart-Hart set, inverting each channel's
temperature gives ln(R/R₀). Their difference at one instant, taken where the
water is near the report temperature, is then ln(R_T2/R_T1), which can be
compared with the reports. On one instrument this matched the configured pair
(T3304/T3303) to 0.0004 (about 0.01 K). It also identified an unrecorded T2
swap to T3301, which the operator later confirmed.

## sensor_deployments.csv

Which sensor was fitted where, and when — one row per (sensor, campaign,
instrument, channel) with first/last use, file count and the configured value,
read from the instruments' own `.p` configs. Covers CASPER 2015/2017, Taiwan
2017, SUNRISE 2019/2021/2022, ASTRAL 2023, ARCTERX 2022/2025, Tanaka's
glider deployments 2015–2022 and an outside user's 2026 VMPs: 30 distinct FP07
beads (plus the placeholder serial `T`) and 51 shear probes.

The Tanaka rows (added 2026-10-18) take the deployment folder under
`/Volumes/SeaChest/Tanaka/gliders/` as their campaign. Their configured FP07
value is `beta_1`, or `beta` where a config has no `beta_1`. MicroRider SN058's FP07s (2015) have no rows, because its
configs record no thermistor serial.

The "Outside user 2026" rows were added 2026-10-26 and describe an outside
group's two VMPs, called A (a VMP-5500) and B (a VMP-250IR) here. They depart
from the configs in two places:

- **A's T2 is attributed by bead, not by config.** Every config says T3304.
  The rows give T3304 through the 2026-08-22 file and T3301 from the next one.
  The beads were identified from the production reports' resistances (*FP07
  production reports* above), and the operator confirmed the swap. T2 was
  failing on the file before 2026-08-22's and broken on 2026-08-22's; those
  two files are counted under T3304, which is an assumption.
- **A bench-test file is left out.**

B's FP07s stay as the placeholder `T`. The resistance evidence makes T2
very likely T3308. T1 is T3306 or T3302, and the evidence cannot separate
them. (T2 drifted +1.4% in resistance during the deployment.)

This is the prerequisite for any per-bead calibration history, since a probe
moves between channels and instruments between cruises — see issue #189 (FP07
calibration sheets of our own). Rebuild by scanning a campaign's `.p` configs.

## fp07_calibrations.yaml — the FP07 calibration sheet

Rockland issues no calibration for an FP07 (its production report says "This
probe is uncalibrated"), so this sheet is our own: per bead, the in-situ
calibrations from `fp07-cal fit` that have been recorded, and a prior for a
fresh deployment.

**One file is the source of truth; two are generated from it.**

| file | what | edit? |
|---|---|---|
| `fp07_calibrations.yaml` | the entries, the transfer uncertainties, the methods | **yes**, by hand |
| `fp07_calibrations.md` | the human-readable sheet: one section per bead, with its deployments and its prior; explains units, L and the uncertainties | **never**, generated |
| `fp07_calibrations.csv` | one row per fit, every value at full precision, for spreadsheets | **never**, generated |

```bash
fp07-cal registry render          # after editing the YAML: rewrite the .md and .csv
fp07-cal registry check           # validate the YAML; exit 1 if invalid or a view is stale
fp07-cal registry show T1592      # one bead's history and its prior
fp07-cal registry show T1592 --instrument 194 --channel T1 --as-of 2023-01-01 --json
fp07-cal registry add fp07cal/coefficients.json --channel T1 --deployment "..." ...  # record a fit
fp07-cal registry response fp07tau/fp07tau_*.json --bead T1592 --deployment "..."  # record its tau
```

`check` and a test (`tests/test_fp07_registry.py`) both fail when a committed
view differs from what the YAML renders, so the three cannot drift apart. The
loader is strict. Each of these raises its own named error (`registry.py`):

- an unknown or missing key;
- a bead serial not of the form `T<number>` without a leading zero (the
  placeholder `T` is not an identity);
- an unquoted instrument or reference serial (YAML would read JAC `0079` as 79).
  Instrument serials are compared as numbers, so `0194` is SN 194;
- first after last, or a reference calibrated after the deployment began;
- a bead in two places at once, or a channel carrying two beads at once;
- an order that disagrees with the `beta` terms or with its held-out errors;
- a second entry for the same bead and deployment, ignoring case and spacing in
  the label;
- coefficients that cannot describe an FP07 over `valid_T_degC`: `t_0` outside
  250-320 K (degC typed for K), `beta_1` <= 0, a range that needs |L| beyond the
  bridge's clip (|Z| <= 0.6, so |L| < 1.386), or T not falling as L rises.

**An entry** is one bead in one deployment, fitted by `fp07-cal fit` pooled over
the deployment. Its `coefficients` are in the `.p` config form (`t_0`,
`beta_1` .. `beta_n`, each `beta` a reciprocal; an absent term is no term).
Its `bridge` holds the six half-bridge constants that define L, because the
coefficients reproduce only on a channel with those constants. Its
`uncertainty` holds the fit's rms, held-out error and block scatter in mK. Its
`reference` names the reference thermometer, its serial and its calibration
date. A null serial or date needs a `note` saying why.

### Adding an entry after `fp07-cal fit`

`fp07-cal registry add` does it, one channel at a time:

```bash
fp07-cal registry add <output_dir>/coefficients.json --channel T1 \
    --deployment "SUNRISE 2021 WaltonSmith" --instrument-model vmp-250 \
    --method vmp-own-jac --reference-kind "JAC CT" --reference-serial 23 \
    --reference-cal-date 2016-02-15 --reference-pole-s 0.169 \
    --flag lag-gate-overridden --notes "..." --dry-run
```

It reads `coefficients.json` and the block table of `<ch>_report.md` beside it
(`--report` to point elsewhere). The options supply only what the record cannot:
the deployment label, the instrument model, the fit `method` (one defined under
`methods`), the reference's kind, serial and calibration date (`none`, with
`--reference-note` saying why), and `pairs.kernel_tau` from the fit's config as
`--reference-pole-s`. Optional: `--bead` (required when the config's serial is
a placeholder), `--instrument-serial`, `--platform`, `--no-robust` (if the fit
ran with `fit.robust: false`), `--flag` (repeatable), `--notes`.

The sheet is never re-dumped. A ruamel round trip (load and dump, nothing
changed) rewrote 2091 of its lines with default settings: it re-wraps the folded
prose. Tuned for width and indentation, it still rewrote every `null`. So the
entry is inserted as text, in the sheet's own layout, after the last
entry of `calibrations:`, leaving every existing line and comment as it was.
Before anything is written, the whole new sheet must pass the strict loader and
the inserted text must read back as exactly the entry built. The views are then
re-rendered. `--dry-run` prints the entry and writes nothing. `add` refuses:

- a second entry for the same bead and deployment (edit the YAML by hand to
  replace one);
- a placeholder bead serial;
- a lag that fp07-cal's trust gate rejected, unless `--flag lag-gate-overridden`;
- a fit that its check against the bead's prior flagged, unless
  `--allow-qc-flag`;
- a fit with no held-out error for its chosen candidate (`fit.order` forced and
  no `fit.prior`), because `held_out_mK` is required.

It keeps the sheet's file mode and line endings (LF or CRLF). An indented
comment after the last entry stays with that entry. An empty `calibrations:`
(or `calibrations: []`) is accepted. Concurrent adds are serialized by a lock
on a sidecar `.fp07_calibrations.yaml.lock`, which git ignores here. A fit
whose held-out-best order was replaced by a plausible one (`fit.prior`) is
recorded with `order_choice: fixed`, because `auto` claims the held-out
minimum.

Where each field comes from, per channel:

| YAML | from |
|---|---|
| `bead` | `coefficients.json` → `channels.<ch>.probe_sn` (if it is a placeholder, identify the bead first) |
| `instrument.serial`, `.channel`, `.model` | `instrument_sn`, the channel, and the `.p` config's `instrument_info.model` |
| `first`, `last` | `channels.<ch>.span.time_start`, `.time_end`: the first sample of the first file that gave the channel a pair its fit kept, and the last of the last (epoch seconds → `YYYY-MM-DDTHH:MM:SSZ`). An `fp07-cal/1` record has no `span`; the run's `time_start`, `time_end` are used, and `registry add` says so |
| `fit.n_files`, `n_profiles`, `n_pairs`, `order`, `lag_s` | `channels.<ch>.span.n_files` (`n_fit_files` from an `fp07-cal/1` record); `channels.<ch>.n_profiles`, `.n_pairs`, `.order`, `.lag_s` |
| `fit.order_choice` | `auto` when `order_selection` is filled, `prior` when the fit kept the prior's shape (`channels.<ch>.prior.chosen_kind` is `offset`), else `fixed` |
| `fit.shape_from` | with `order_choice: prior` only: `channels.<ch>.prior.source.deployment`, the entry whose `beta` terms were held fixed (the loader checks they are that entry's exactly) |
| `fit.held_out_by_order_mK` | `channels.<ch>.order_selection.<n>.held_out_K` × 1000 (the full fits'); required when `order_choice` is `auto`, so the choice can be checked |
| `fit.geometry` | `channels.<ch>.geometry.dz_m`, `.tau_s`, or `false` when it was off |
| `fit.reference_pole_s` | `pairs.kernel_tau` in the fp07-cal config |
| `fit.software_version` | `software_version` |
| `coefficients` | `channels.<ch>.config_equivalent`, verbatim |
| `bridge` | `channels.<ch>.bridge` |
| `uncertainty.fit_rms_mK`, `held_out_mK` | `rms_K` × 1000, the rms over the pairs the robust fit **kept**; the chosen order's `held_out_K` × 1000 (with `order_choice: prior`, the `prior shape` candidate's) |
| `uncertainty.rms_all_mK` | optional, fits made with `fit.prior` only: the chosen candidate's `rms_all_K` × 1000, the rms over **every** pair. A shape that does not fit clips the ends of the range, so its kept-pair rms looks better than it is |
| `uncertainty.block_sd_mK`, `n_blocks` | sample SD and count of the offsets in the block table of `<ch>_report.md` (null below 3 blocks, and only then) |
| `valid_T_degC` | `channels.<ch>.T_range` |

`flags` come from the vocabulary in `registry.py` (`FLAGS`), and `notes` hold
anything a later reader needs: what was screened, episodes, gates overridden.
Commit all three files together. An entry added by hand instead needs
`fp07-cal registry render`, then `check`.

### The FP07 time constant (`response:`)

An entry may carry the bead's FP07 time constant. It is a property of the
bead, carried with its calibration (PI decision, 2026-10-28: the electronics are
much faster than the bead). It is measured, never typed:

```bash
fp07-pair tau 'VMP_results/chi_00/*.nc' --instrument 194 \
    --first 2022-03-21T04:59:26Z --last 2022-03-30T18:45:16Z -o fp07tau/
fp07-cal registry response fp07tau/fp07tau_SN194_T1592-T2005_20220321T0459Z.json \
    --bead T1592 --deployment "ARCTERX-2022 Interior" --dry-run
```

Use the entry's own `first` and `last` (`registry show <bead> --json`), so the
measurement describes the same stretch of the same bead pair as the
calibration. Record it on both beads of the pair: the loader then checks that
the two blocks agree (reciprocal ratios, same counts and halves). `registry
response` inserts the block as text at the end of the entry, like `add`, and
refuses an entry that already has one, a record of another instrument or
channel, a placeholder partner, and a measurement outside the entry's span.
What the number means, and does not, is in the sheet's header comment and in
`fp07_calibrations.md` ("FP07 time constant"); the method and the rule that
decides whether a pair identifies a tau at all are in
`docs/fp07pair/runbook.md` §6.

| YAML (`response.`) | from the `fp07-pair tau` record |
|---|---|
| `method`, `speed_scaling`, `band_Hz`, `software_version` | `method`, `speed_scaling` (the products' tau model), `band_Hz`, `software_version` |
| `partner` | the other bead of `serials` (a placeholder is refused) |
| `first`, `last`, `n_chi_files`, `n_recordings`, `n_windows` | `all.first`, `all.last` (first and last chi file `start_time`, dead files left out), `all.n_files`, `all.n_recordings` (distinct start times, the units the bootstrap resamples), `all.n_windows` |
| `W_ref_m_s` | `W_ref_m_s`, the median speed of the pair's windows (3 decimals) |
| `identified`, `reason` | `all.identified`; `all.reasons` joined, when not identified |
| `tau_ms`, `tau_ms_16_84` | `all.tau_s`, `all.tau_16_84_s` for this bead's channel, x 1000 (0.01 ms); null when not identified |
| `halves.first`, `halves.second` | `halves[0]`, `halves[1]`, the same fields, plus `power_ratio_20Hz` from the half's `ln_ratio` (turned round as below); null when every file shares one start time |
| `power_ratio`, `power_ratio_16_84` | `exp(all.ln_ratio)` at 10 and 20 Hz, turned round for the bead on the first channel, so it is always this bead's power response over its partner's |
| `flags` | the record's `amplitude`, `dead-bead`, `outlier`, `noise-model` and `ratio-shift`, and `drift` when this bead's `drift` is set |
| `rule` | `rule_thresholds`: the thresholds the measurement used (`n_boot` included); the loader refuses an identified block that breaks them, and the sheet names any that differ from the package default |
| `reference_band_Hz` | `[min, max]` of `reference_points_Hz`, where the ratios' reference level was read (2-4 Hz) |
| `noise_floor` | `all.noise_obs_over_model`, this bead's and its partner's (4 significant figures) |
| `notes` | `--notes` |

Tau is stored and rendered at `W_ref_m_s` only. Nothing converts it to another
speed: whether tau ~ W^-1/2 holds across the 0.4-1.3 m/s of gliders and VMPs is
not established, so compare taus at similar `W_ref_m_s`.

### The prior

The prior is for a deployment where a full fit is impossible: too little
reference, or too narrow a temperature range to constrain the shape. Where a
deployment has a decent reference, its own pooled fit beats any carried-over
calibration (compare an entry's block SD with the sheet's transfer offset).

The rule is stated in `fp07_calibrations.md` (`PRIOR_RULE` in `registry.py`)
and implemented in `Registry.prior_for`. In short:

- The source is the bead's widest-range calibration that is not flagged
  `short`, `fast-descent` or `episode`, preferring the target instrument and
  channel when they are given.
- If the bead has no unflagged calibration, the prior falls back to its
  widest-range one, says so, and widens the shape term.
- The prior reports how far the most recent calibration sits from the source.

The uncertainty sizes are not repeated here, so that they cannot go stale.
They live in the YAML's `transfer` section with their evidence, and the
generated sheet renders them. Update `transfer`, with its evidence, when new
comparisons change them.

`prior_for` returns what a `t_0`-only refit with the shape held fixed needs:
the fixed `beta` terms, the bridge constants that define their L, a starting
`t_0` with its uncertainty, and the range where the shape is constrained.
`fp07-cal fit` makes that refit when its config has a `fit.prior` block. It
scores the refit by held-out error against the full fits on the same folds. It
keeps the prior's shape unless the full fit is clearly better (by more than
`tie_fraction`, 5%), and always over a range narrower than 5 K or when no full
order is a plausible curve for the bead. It then checks the result against the
prior over their common range (`docs/fp07cal/runbook.md` §5). A
calibration recorded with the prior's shape (`order_choice: prior`) never
supplies a prior itself, because its shape was not fitted over its own range.

## sheet_requests.md

Sensors found in deployment data for which **no sheet is held here** — the
standing batch request to Rockland. Regenerate/extend it by scanning a
campaign's `.p` configs and differencing against the PDFs in this directory.
As of 2026-09-07 it covers CASPER and ASTRAL: 16 shear probes and 12 FP07
thermistors (we hold no thermistor sheets for those; the only FP07 documents
here are the 2026 production reports above). Fourteen of the sixteen
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
M3038 conflicts below, the non-shear `JAC_CT_SN023.pdf`, and the eight FP07
production reports). Add `manual`
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
