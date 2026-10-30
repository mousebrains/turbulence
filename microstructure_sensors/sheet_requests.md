# Calibration sheets to request from Rockland — batch list

Sensors that appear in OSU deployment data for which **no calibration sheet is
held in this directory**. Assembled 2026-09-07 by reading the embedded config
of every `.p` file in the CASPER and ASTRAL corpora (901 files, 8 trees), then
differencing against the PDFs here.

Intended to be sent as **one batch**, not piecemeal. Add to it as other
campaigns are inventoried; the ARCTERX corpora are already well covered.

**Status: ANSWERED 2026-10-12.** Pat asked Rockland on 2026-09-15. **14 of the
16 shear sheets arrived** (in a 129-PDF batch, 84 of them new here, that also
carried most of these probes' pre-2020 history) and are filed in this directory and in
`shear_sensitivities.csv`. **Rockland has no record of M1000 or M1001** — see
notes 0, 1 and 4. Follow-up questions raised by the batch are under *Next
request* at the end.

**Updated 2026-09-08 with the Taiwan corpora** (`/Volumes/SeaChest/Taiwan`,
377 `.p` files). No *new* serials — Taiwan's four shear probes were all already
on this list — but it moves two deployment windows earlier and adds decisive
evidence to note 1. See note 4.

**29 distinct sensors found.** 17 shear probes, of which only M1194 has a
sheet — so **16 shear sheets are requested below**. The 12 FP07 thermistors are
not requestable at all (see below).

Serials and the "configured" values below are read from the instruments' own
embedded configs — they are what the data were processed with, and are exactly
what a sheet would confirm or correct.

---

## Shear probes (16)

The last two columns were filled in 2026-10-12. "Governing" is the sheet in
effect on the deployment dates; "matches" compares it to the configured value.
The check uses this list's deployment windows, not a per-file scan.

| serial | instrument | deployment window | configured sens | why we want it | received | governing sheet → matches? |
|---|---|---|---|---|---|---|
| **M1000** | VMP-250 SN **142**, then VMP-250IR SN 194 | **2017-02-22 → 02-26**, 2017-09-28 → 10-25 | 0.0716 | **highest priority** — see notes 1, 4 | **Rockland has no record** | — |
| **M1001** | VMP-250 SN **142**, then VMP-250IR SN 194 | **2017-02-22 → 02-26**, 2017-09-28 → 10-25 | 0.0705 | **highest priority** — see notes 1, 4 | **Rockland has no record** | — |
| M1021 | VMP250IR-DL3 SN 92 | 2023-06-17 → 06-19 | 0.0658 | note 2 | 2013, 2016, 2019 | 2019-03-29 0.0658 → yes, 50 mo old |
| M1038 | VMP250IR-DL3 SN 92 | 2023-06-20 → 06-21 | 0.0539 | note 2 | 2013, 2019 | 2019-03-29 0.0539 → yes, 50 mo old |
| M1196 | VMP250IR-DL3 SN 92 | 2023-06-22 | 0.0584 | note 2, and see note 3 | 2014, 2016, 2018, 2019 (+ M1196-B) | 2019-03-18 0.0584 → yes, 51 mo old |
| M1252 | VMP250IR-DL3 SN 92 | 2023-06-22 → 06-24 | 0.0427 | note 2 | 2015–2019 (5) | 2019-03-18 0.0427 → yes, 51 mo old |
| M1264 | VMP250IR-DL3 SN 92 | 2023-06-19 → 06-20 | 0.0410 | note 2 | 2015–2019 (4) | 2019-03-18 0.0410 → yes, 51 mo old |
| M1268 | VMP250IR-DL3 SN 92 | 2023-06-21 | 0.0418 | note 2 | 2015–2019 (4) | 2019-03-18 0.0418 → yes, 51 mo old |
| M1516 | VMP250IR-DL3 SN 92 | 2023-06-17 → 06-18 | 0.0596 | note 2 | 2016, 2018, 2019 | 2019-03-29 0.0596 → yes, 50 mo old |
| M1848 | VMP250IR-DL3 SN 92 | 2023-06-21 → 06-24 | 0.0689 | note 2 | 2018, 2019 | 2019-03-18 0.0689 → yes, 51 mo old |
| M1247 | MR_1000_LP SN 134 | 2015-07-27 → 10-14 | 0.0960 | MicroRider, CASPER-East | 2015-02-05 | 0.0960 → yes |
| M1249 | MR_1000_LP SN 134 | 2015-07-27 → 10-22 | 0.0748 | MicroRider, CASPER-East | 2015-02-06 | 0.0748 → yes |
| M1253 | MR_1000_LP SN 134 | 2015-10-14 → 10-22 | 0.0622 | MicroRider, CASPER-East | 2015-02-06 | 0.0622 → yes |
| M1344 | MR_1000_LP SN 134 | 2015-10-23 → 2017-10-23 | 0.0655 | MicroRider, CASPER-East/West **and Taiwan 2017** (2017-02-16 → 02-28) | 2015-08-11, 2016-04-11 | CASPER-East cruise: 2015-08-11 0.0655 → yes (its stray CAS_170, dated 2016-08-08, would not). **2017: 2016-04-11 0.0615 → no** (note 5) |
| M1371 | MR_1000_LP SN 134 | 2015-10-23 → 2017-10-23 | 0.0855 | MicroRider, CASPER-East/West **and Taiwan 2017** (2017-02-16 → 02-28) | 2015-09-03, 2016-04-11, 2021-04-28 | CASPER-East cruise: 2015-09-03 0.0855 → yes (its stray CAS_170, dated 2016-08-08, would not). **2017: 2016-04-11 0.0729 → no** (note 5) |
| M1493 | MR_1000_LP SN 134 | 2017-08-26 → 10-25 | 0.0805 | MicroRider on glider *doug*, CASPER-West | 2016-05-19 | 0.0805 → yes, 17 mo old |

Held for comparison: **M1194** (2016-04-07, 0.0793) — MicroRider sh1 on
CASPER-West; its config value matches the sheet exactly, which is the evidence
that this operator populated configs from real sheets. **Qualified
2026-10-12:** the batch shows that 2016-04-07 sheet is probably a different
physical probe from the M1194 that Rockland recalibrated to 0.0591 on
2017-08-24. See note 5 and README *Known bad sheets*.

## FP07 thermistors — NOT requestable

**Rockland does not generate FP07 calibration sheets.** Do not ask for these;
no bead-level calibration exists to be sent. Recorded here so the question is
not re-opened every time a campaign is inventoried.

*Added 2026-10-26.* Rockland does issue each FP07 a *production report*,
eight of which are now filed here (README, *FP07 production reports*). Each
report says "This probe is uncalibrated", so the line above stands. A report
gives the bead's resistance at one temperature, which is enough to check bead
identity in situ. Whether Rockland kept reports for the older beads below is
unknown (unverified). If it did, they would be worth asking for on identity
grounds only.

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

**0 — ANSWERED (2026-10-12): Rockland has no record of M1000 or M1001.**
Pat asked for these two sheets with the rest of the batch; Rockland's records
contain neither serial. That does not say *why*: the serials may never have
been issued, or Rockland's records may be incomplete. Either way, no
calibration can be traced to the sensitivities configured for them (0.0716 /
0.0705). The withdrawal below still stands, because the round numbers were
never evidence. What changed is that the sheet request, which was supposed to
settle the question, has now come back empty.

*Earlier text, kept for the record:*

**0 — WITHDRAWN (2026-09-15): "were these serials ever issued?"**
An earlier version of this note asked Rockland whether **M1000, M1001, T1000
and T1001** existed at all, on the premise that a round X1000/X1001 numbering
looked like RSI template text. **That premise is wrong.** Rockland shear-probe
serials `M\d+` and FP07 serials `T\d+` are valid from **M1 / T1** onwards, and
1000 is not a special number (Pat, 2026-09-15). The serial numbering is no
evidence of anything; these four are ordinary serials, and the sheets for the
shear probes are simply requested like the others.

Separately, and still true: both shear probes and FP07 thermistors are
**removable and move between instruments**, so a serial seen on two
instruments (e.g. `T1000`/`T1001` on VMP SN 194 in 2017 and on SN 142 in 2019)
is a probe being moved, the normal case.

**1 — M1000 / M1001 (CASPER-West VMP SN 194) are the most valuable request.**
This instrument's `setup.cfg` was written by RSI on 2015-12-17 and modified
2016-05-17 "with coefficients"; its `cruise_info` still reads the template's
`operator = Dr. ?`, and its FP07 coefficients are the generic nominal set, so
parts of that config were demonstrably never edited. What the data cannot
tell us is whether the configured **sensitivities** 0.0716 / 0.0705 are the
probes' calibrated values — and that decides the absolute epsilon scale for a
247-file, 2097-profile dataset. ~~A sheet for either serial settles it.~~
**No sheet exists (note 0), so the absolute epsilon scale of CASPER-West VMP
SN 194 and Taiwan 2017 VMP SN 142 rests on untraceable sensitivities.** M1001's
0.0705 equal to RSI's example value (note 4) now carries more weight than it
did when a sheet was still expected.

**2 — the ASTRAL 2023 probes look anomalously insensitive.** Six of the eight
sit at 0.0410–0.0596, against a fleet range of 0.058–0.113 across every other
sheet we hold. Independent evidence (a least-squares fit across six in-cruise
probe changes) puts M1196's *implied* sensitivity near 0.023, below anything
in the fleet. Either these configs carry wrong values, or these probes were
genuinely out of family — a sheet distinguishes those.

**ANSWERED 2026-10-12: genuinely out of family, and the configs are
faithful.** All eight configured values equal the probes' **2019-03** Rockland
sheets to four digits. Most of the loss came in one early step and then
held: M1264 0.0555 (2015) → 0.0402 (2016), then 0.0413 and 0.0410; M1268,
M1252 and M1516 the same. Only M1021 (−17%, 2016 → 2019) and M1252 (−6%) kept
falling after that. M1038 went 0.0864 (2013) → 0.0539 (2019) with nothing
between. All eight calibrations were **50–51 months old** at ASTRAL (June
2023), well past Rockland's 12-month recommendation. Whether they moved again
between 2019 and 2023 is unknown. The sheets do **not** explain M1196's implied ~0.023: its
2019-03-18 sheet says 0.0584, in line with its 2014–2018 history
(0.0571–0.0636). That supports the damage reading in note 3.

**3 — M1196 is currently flagged unusable** in
`ASTRAL/2023/Data/VMP/calibration/README.md`.

## How to file what comes back

One row per (probe, calibration) in `shear_sensitivities.csv`, with the PDF
dropped in this directory under `<SERIAL>_<YYYY>_<MM>_<DD>.pdf`. If Rockland
supplies a value without a sheet, use `source = manual` and say in `notes`
where it came from and when. (The M1458 2016-04-25 row was the example of
this pattern until its sheet arrived 2026-10-12; its `notes` still record the
history. No `manual` rows remain.)

---

**4 — Taiwan 2017 moves M1000/M1001 seven months earlier, and sharpens note 1.**

Taiwan 2017 (`Taiwan17/vmp/raw/casts/SN142/` — the raw tree was reorganized
2026-09-15 and the originals now sit under `raw/{casts,bottom_crasher,bench,ctd_rosette}/SN142/`;
VMP-250 **SN 142**, 2017-02-22 → 02-26) carries
**M1000 sens 0.0716 and M1001 sens 0.0705** — the same serials and the same
sensitivities later seen on VMP-250IR SN 194 at CASPER-West. So this is now the
**earliest recorded use** of both serials, and they sit on a *different*
instrument. Two readings (no sheet can now separate them; see the end of this
note):

- they are the fitted probes, moved between two OSU VMPs during 2017 (probes
  are removable, so this is the ordinary case); or
- the same `setup.cfg` was copied from one instrument to the other without its
  probe fields being updated.

Two pieces of evidence found in the Taiwan tree bear on the sensitivities
(not on the serials — see note 0):

- **RSI's own example config is now in hand** — `Taiwan13/.../VMP_002/SETUP.CFG`,
  shipped by RSI 2013-01-23 (`boat = Titanic v2.0`, `captain = Lucky Jim`). Its
  example sensitivities are **sh1 0.0709, sh2 0.0705**. So **M1001's 0.0705
  equals the RSI example value**, while M1000's 0.0716 does not. A coincidence
  to four digits is possible.
- **0.0716 / 0.0705 also appears in 2013**, on VMP **SN 002** (Taiwan 2013,
  `TAI_013_027`–`070`, a Lou St. Laurent instrument) — recovered from the
  cruise's MATLAB output, since the v1 setup file records no shear channel at
  all. **No serial is recorded there**, so it cannot be tied to M1000/M1001, but
  the pair predates both 2017 deployments by four years.

Ask for these two: *"what were the sensitivities and calibration dates of
M1000 and M1001?"* — the same sheet request as every other probe. **Asked;
Rockland has no record of either (note 0).** No sheet will now separate the
two readings above. Rockland's records do reach back to M289 (2004), so a
fitted Rockland probe with no record at all would be odd. On that basis the
copied-config reading now looks the more likely one — a judgment, not a
proof.

**Taiwan adds no new requestable serials.** Taiwan 2013 has none to add:
VMP SN 002's v1 setup file defines no shear channel, and the glider MicroRider
**SN 046** (Lou St. Laurent's, not OSU's) records no probe serials and carries
round nominal sensitivities (0.0700 on 73 files, 0.0800 on 4).

---

**5 — What the 2026-10-12 sheets say about the deployments.** Every shear row
of `sensor_deployments.csv` was checked against the calibration in effect on
its first and last dates (hold-previous, tolerance 0.00005). That is a
deployment-level check; the per-file one is
`rsi-tpw sensors --cal-dir microstructure_sensors --shear` over the raw trees,
which has not been run. The configs that disagree with a now-held sheet:

- **MR SN 134, Taiwan 2017: M1344 0.0655 / M1371 0.0855** are their 2015
  sheets, but Rockland recalibrated both on **2016-04-11, to 0.0615 / 0.0729**.
  This MR's config was never edited between 2015-10 and 2017-10, so it is
  unknown whether these probes were even fitted. If they were, Taiwan 2017 ε is
  low by **1.13× (sh1) and 1.38× (sh2)**. When the same two probes moved to the
  VMPs at SUNRISE 2019, their configs carried the 2016 values. The CASPER-West
  2017 rows for these serials come from the uncorrected duplicate copy, which
  the recipe already excludes.
- **MR SN 134, CASPER-West 2017: sh1 M1194 0.0793**, against **0.0591** from the
  M1194 that Rockland recalibrated on 2017-08-24. The 0.0793 sheet is probably
  a different physical probe that shares the serial (README, *Known bad
  sheets*), so this is an identity question, not yet a scale error. If the
  recalibrated probe flew, sh1 ε is 1.80× low.
- **VMP SN 194, SUNRISE 2022 Point Sur: sh2 M289 0.0725** is its 2016 sheet;
  the 2021-04-27 recalibration says **0.0794** (ε 1.20× high if that is right).
- **VMP SN 142, ARCTERX-2025 Interior: sh2 M1218 0.0629** matches no sheet.
  The only one held is 2014-12-15, 0.0797, so this is most likely a later
  calibration we don't have.
- **The CASPER-East rows for M1344 / M1371 also flag**, only at their last
  date, 2016-08-08. That date comes from `CAS_170`, a stray file whose header
  is dated ten months after the cruise and which yields no data. The cruise
  itself (2015-10-23 → 11-05) is governed by the matching 2015 sheets.
- Not from this batch, still open: VMP SN 479's (ARCTERX-2025 Wake) M2732 /
  M2746 are configured 0.1075 / 0.1130 against their 2023 sheets' 0.0743 /
  0.0607. A lead, unverified: 0.1075 / 0.1130 equal, to four digits, the
  2024-07-18 calibrations of **M3054 / M3055** (the "previous" entries on their
  2024-07-26 sheets). Both channels match, which suggests the probes fitted
  were M3054/M3055 under stale serials.
- **MR SN 435, ARCTERX-2025 Interior: sh1 M3038 and sh2 M3039 are both
  configured 0.1041** (every file from 2025-01-30 04:54Z to 04-10), the value
  of the discounted 2024-07-15 sheets. Rockland gives 0.1093 / 0.1189, so ε is
  **1.10× high (sh1) and 1.30× high (sh2)**. MR SN 433's M3036 0.1223 and
  M3037 0.0952 are their 2024-07-15 sheets, which stand.

## Next request (not yet sent)

1. **M1218** — any calibration after 2014-12-15 (the 0.0629 in the
   ARCTERX-2025 Interior config).
2. **M1194 / M1196-B** — is the 2016-04-07 "M1194" sheet (0.0793, 1.087 nF,
   sent to OSU as `M1194_2_...`) the same physical probe as the M1194 that
   read 0.0549 on 2016-03-14 and 0.0551 on 2016-04-18 (0.846–0.941 nF over
   2014–2026)? What is M1196-B? Which one did Rockland return to OSU after the
   2017-08-24 recalibration?
3. **M1254 2015-02-06** — confirm 0.0802 (its own sheet) rather than the
   0.0820 quoted as "previous" on `M1254_2021_4_27.pdf`.
4. **Wrong "previous calibration" dates** — re-issue M2245, M3036 and M3037
   (2026-04-28), M1194 (2026-06-22) and M1496 (2022-06-20) with the previous
   date corrected, as Rockland already did for M1254 and M1458 on 2026-07-03.
5. ~~**M3038 2024-07-15**~~ — answered 2026-10-30: MR SN 435 used M3038 0.1093
   and M3039 0.1189 (the 2024-07-09 calibrations); both 2024-07-15 sheets are
   discounted (README, *Known bad sheets*).
