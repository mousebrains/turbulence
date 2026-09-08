# Calibration sheets to request from Rockland — batch list

Sensors that appear in OSU deployment data for which **no calibration sheet is
held in this directory**. Assembled 2026-09-07 by reading the embedded config
of every `.p` file in the CASPER and ASTRAL corpora (901 files, 8 trees), then
differencing against the PDFs here.

Intended to be sent as **one batch**, not piecemeal. Add to it as other
campaigns are inventoried; the ARCTERX corpora are already well covered.

**29 distinct sensors found.** 17 shear probes, of which only M1194 has a
sheet — so **16 shear sheets are requested below**. The 12 FP07 thermistors are
not requestable at all (see below).

Serials and the "configured" values below are read from the instruments' own
embedded configs — they are what the data were processed with, and are exactly
what a sheet would confirm or correct.

---

## Shear probes (16)

| serial | instrument | deployment window | configured sens | why we want it |
|---|---|---|---|---|
| **M1000** | VMP-250IR SN 194 | 2017-09-28 → 10-25 | 0.0716 | **highest priority** — see note 1 |
| **M1001** | VMP-250IR SN 194 | 2017-09-28 → 10-25 | 0.0705 | **highest priority** — see note 1 |
| M1021 | VMP250IR-DL3 SN 92 | 2023-06-17 → 06-19 | 0.0658 | note 2 |
| M1038 | VMP250IR-DL3 SN 92 | 2023-06-20 → 06-21 | 0.0539 | note 2 |
| M1196 | VMP250IR-DL3 SN 92 | 2023-06-22 | 0.0584 | note 2, and see note 3 |
| M1252 | VMP250IR-DL3 SN 92 | 2023-06-22 → 06-24 | 0.0427 | note 2 |
| M1264 | VMP250IR-DL3 SN 92 | 2023-06-19 → 06-20 | 0.0410 | note 2 |
| M1268 | VMP250IR-DL3 SN 92 | 2023-06-21 | 0.0418 | note 2 |
| M1516 | VMP250IR-DL3 SN 92 | 2023-06-17 → 06-18 | 0.0596 | note 2 |
| M1848 | VMP250IR-DL3 SN 92 | 2023-06-21 → 06-24 | 0.0689 | note 2 |
| M1247 | MR_1000_LP SN 134 | 2015-07-27 → 10-14 | 0.0960 | MicroRider, CASPER-East |
| M1249 | MR_1000_LP SN 134 | 2015-07-27 → 10-22 | 0.0748 | MicroRider, CASPER-East |
| M1253 | MR_1000_LP SN 134 | 2015-10-14 → 10-22 | 0.0622 | MicroRider, CASPER-East |
| M1344 | MR_1000_LP SN 134 | 2015-10-23 → 2017-10-23 | 0.0655 | MicroRider, CASPER-East/West |
| M1371 | MR_1000_LP SN 134 | 2015-10-23 → 2017-10-23 | 0.0855 | MicroRider, CASPER-East/West |
| M1493 | MR_1000_LP SN 134 | 2017-08-26 → 10-25 | 0.0805 | MicroRider on glider *doug*, CASPER-West |

Held for comparison: **M1194** (2016-04-07, 0.0793) — MicroRider sh1 on
CASPER-West; its config value matches the sheet exactly, which is the evidence
that this operator populated configs from real sheets.

## FP07 thermistors — NOT requestable

**Rockland does not generate FP07 calibration sheets.** Do not ask for these;
no bead-level calibration exists to be sent. Recorded here so the question is
not re-opened every time a campaign is inventoried.

The 12 distinct FP07s in these corpora (T179, T865, T866, T987, T988, T990,
T991, T1000, T1001, T1030, T1057, T1305) therefore have no traceable
calibration of any kind. Four of them carry Rockland's generic nominal set
(`beta_1 = 3143.55`, `beta_2 = 2.5e5`, `T_0 = 289.301`) and the rest carry no
coefficients at all.

We work around this by fitting Steinhart-Hart coefficients **in situ** against
the instrument's own CT (`fp07-cal`, and perturb's `fp07.calibrate`). Whether
those fits can be promoted into a per-bead calibration record of our own — and
how stable such a record is over time — is tracked in its own issue; see
`docs/` and the repo issue "FP07: can we make our own calibration sheets?".

---

## Notes

**1 — M1000 / M1001 (CASPER-West VMP SN 194) are the most valuable request.**
This instrument's `setup.cfg` was written by RSI on 2015-12-17 and modified
2016-05-17 "with coefficients"; its `cruise_info` still reads the template's
`operator = Dr. ?`, and its FP07 coefficients are the generic nominal set. So
we cannot tell whether `M1000 / M1001` are the real probes fitted in 2017 or
RSI example values left in place — and the two readings differ in what the
absolute epsilon scale means for a 247-file, 2097-profile dataset. A sheet for
either serial settles it outright. **If Rockland has no record of M1000 or
M1001 ever existing, that is equally decisive** and worth reporting back.

**2 — the ASTRAL 2023 probes look anomalously insensitive.** Six of the eight
sit at 0.0410–0.0596, against a fleet range of 0.058–0.113 across every other
sheet we hold. Independent evidence (a least-squares fit across six in-cruise
probe changes) puts M1196's *implied* sensitivity near 0.023, below anything
in the fleet. Either these configs carry wrong values, or these probes were
genuinely out of family — a sheet distinguishes those.

**3 — M1196 is currently flagged unusable** in
`ASTRAL/2023/Data/VMP/calibration/README.md`.

## How to file what comes back

One row per (probe, calibration) in `shear_sensitivities.csv`, with the PDF
dropped in this directory under `<SERIAL>_<YYYY>_<MM>_<DD>.pdf`. If Rockland
supplies a value without a sheet, use `source = manual` and say in `notes`
where it came from and when (see the M1458 2016-04-25 row for the pattern).
