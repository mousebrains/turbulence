# Calibration sheets to request from Rockland — batch list

Sensors that appear in OSU deployment data for which **no calibration sheet is
held in this directory**. Assembled 2026-09-07 by reading the embedded config
of every `.p` file in the CASPER and ASTRAL corpora (901 files, 8 trees), then
differencing against the PDFs here.

Intended to be sent as **one batch**, not piecemeal. Add to it as other
campaigns are inventoried; the ARCTERX corpora are already well covered.

**Status: REQUESTED.** Pat asked Rockland for these sheets on 2026-09-15 and
expects them that week. File what comes back as described under *How to file
what comes back*, then mark each row here as received.

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

| serial | instrument | deployment window | configured sens | why we want it |
|---|---|---|---|---|
| **M1000** | VMP-250 SN **142**, then VMP-250IR SN 194 | **2017-02-22 → 02-26**, 2017-09-28 → 10-25 | 0.0716 | **highest priority** — see notes 1, 4 |
| **M1001** | VMP-250 SN **142**, then VMP-250IR SN 194 | **2017-02-22 → 02-26**, 2017-09-28 → 10-25 | 0.0705 | **highest priority** — see notes 1, 4 |
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
| M1344 | MR_1000_LP SN 134 | 2015-10-23 → 2017-10-23 | 0.0655 | MicroRider, CASPER-East/West **and Taiwan 2017** (2017-02-16 → 02-28) |
| M1371 | MR_1000_LP SN 134 | 2015-10-23 → 2017-10-23 | 0.0855 | MicroRider, CASPER-East/West **and Taiwan 2017** (2017-02-16 → 02-28) |
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
247-file, 2097-profile dataset. A sheet for either serial settles it.
(The serial numbers themselves are not in question — see note 0.)

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

---

**4 — Taiwan 2017 moves M1000/M1001 seven months earlier, and sharpens note 1.**

Taiwan 2017 (`Taiwan17/vmp/raw/casts/SN142/` — the raw tree was reorganized
2026-09-15 and the originals now sit under `raw/{casts,bottom_crasher,bench,ctd_rosette}/SN142/`;
VMP-250 **SN 142**, 2017-02-22 → 02-26) carries
**M1000 sens 0.0716 and M1001 sens 0.0705** — the same serials and the same
sensitivities later seen on VMP-250IR SN 194 at CASPER-West. So this is now the
**earliest recorded use** of both serials, and they sit on a *different*
instrument. Two readings, and a sheet still separates them:

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
  to four digits is possible; the sheet will say.
- **0.0716 / 0.0705 also appears in 2013**, on VMP **SN 002** (Taiwan 2013,
  `TAI_013_027`–`070`, a Lou St. Laurent instrument) — recovered from the
  cruise's MATLAB output, since the v1 setup file records no shear channel at
  all. **No serial is recorded there**, so it cannot be tied to M1000/M1001, but
  the pair predates both 2017 deployments by four years.

Ask for these two: *"what were the sensitivities and calibration dates of
M1000 and M1001?"* — the same sheet request as every other probe.

**Taiwan adds no new requestable serials.** Taiwan 2013 has none to add:
VMP SN 002's v1 setup file defines no shear channel, and the glider MicroRider
**SN 046** (Lou St. Laurent's, not OSU's) records no probe serials and carries
round nominal sensitivities (0.0700 on 73 files, 0.0800 on 4).
