# MicroRider SN 134 on glider doug — Taiwan 2017 (SK-II)

Microstructure from the MicroRider carried by Slocum glider `doug` during
R/V Roger Revelle cruise RR1704 (project SK-II), Luzon Strait. **530 profiles
from 72 of 80 files**, 2017-02-19 – 02-28, processed 2026-09-15.

| | |
|---|---|
| `microrider/` | The 80 original `CAS_###.P` files (uppercase). Untouched. |
| `perturb.yaml` | Processing config — read its header first. |
| `Processed/` | `profiles/diss/chi/ctd` (+ `_binned`, `_combo`), generation `_00`. |
| `flight/` | **Through-water speed** (Merckelbach flight model), the MR-vs-glider clock, and the drift-corrected track. **Read `flight/README.md`.** |
| `logs/`, `ma_DOUGfiles/`, `NH_*.mat` | At-sea Iridium logs, mission files and a decimated glider record. |
| `../20170216_doug_Taiwan/` | The Slocum Dinkum binaries (added to SeaChest 2026-09-15). |

## What was done

- **Speed**: no EM on this MR, so U comes from a Merckelbach dynamic flight
  model calibrated per day against the glider's depth rate (`flight/`). Median
  0.28 m/s (5–95%: 0.21–0.41). **No speed was measured on doug, so the absolute
  scale is unvalidated, and ε ∝ U⁻⁴.** The nearest check is indirect and four
  years earlier: an Aquadopp HR on Taiwan 2013's glider jane finds this model's
  angle of attack ~1.6° too small on dives — U ~5–6% high, dive ε ~20–25% low;
  climbs agree to 0.3%. **Not applied here, by decision (Pat, 2026-09-15).**
  doug's own model α is −2.90° diving and +2.89° climbing. Details:
  `flight/README.md` and `docs/aquadopp_flight_model_validation.html` in
  microstructure-tpw.
- **Clock: the MicroRider ran 9 min 21 s fast** (−561.2 s, drifting
  −0.444 s/day). Both the hotel file and the GPS track are written on the MR
  clock, so `time_offset: 0`. Uncorrected, every merged CTD sample would have
  landed tens of meters from where the MR was.
- **FP07**: fitted in situ per file against the glider CTD (the config's
  coefficients are stale — see below).
- **chi by spectral fit** (Method 2), because glider shear is
  vibration-contaminated and makes a poor seed for k_B.
- **8 files produced nothing, correctly**: CAS_001–004, 030, 031, 079, 080 are
  deck/startup records at 0.3–1.3 dbar. perturb refused them rather than
  publish the 0.05 m/s speed floor as a speed.

## Results (1-m bins, 530 profiles)

| | median | spread (MAD) | n |
|---|---|---|---|
| ε (paired) | 2.3e-9 W/kg | 1.31 dex | 88,806 |
| χ | 3.1e-9 K²/s | 1.25 dex | 88,215 |
| N² | 7.0e-5 s⁻² | 0.61 dex | 88,806 |
| Γ | 0.12 | 0.59 dex | 84,456 |
| K_ρ | 7.5e-6 m²/s | 1.50 dex | 88,805 |

**The thermistors agree**: pooled χ(T1)/χ(T2) = **0.982** over 148,981 windows,
so both beads are healthy — unlike Taiwan 2013's MR046, whose T1 died mid-deployment.

## ⚠️ The shear probes disagree by ~3×

Over 155,616 windows the pooled ratio is **sh1/sh2 = 2.94**, and **2.24** where
the signal is strong (sh2 > 1e-8). sh1's noise floor is ~4× higher (1st
percentile 3.3e-11 vs 7.4e-12 W/kg). Unlike Taiwan 2013's MR, here **sh2** is
the channel that fails the spectral-shape test more often (FM > 1.15 in 18.8%
of its windows against 14.0% for sh1).

**An independent check, the ship's VMP SN 142** (`../../vmp/Processed/`), which
profiled the same days, 20–150 dbar:

| day | separation | VMP ε | MR sh1 | MR sh2 | MR geometric mean |
|---|---|---|---|---|---|
| 2017-02-23 | ~24 km | 2.64e-9 | 3.89e-9 | 5.21e-10 | 1.30e-9 |
| 2017-02-25 | ~28 km | 3.20e-9 | 4.43e-9 | 1.14e-9 | 2.12e-9 |
| 2017-02-26 | ~59 km | 5.36e-9 | 6.50e-9 | 1.55e-9 | 2.83e-9 |

Here **sh1 is the channel that tracks the VMP** (within 1.2–1.5×), while sh2
reads 3–5× low and the shipped `epsilonMean` ~2× low. **This is a bound on
gross error, not a validation**: the platforms are 24–59 km apart in different
water and ε varies by decades spatially.

**Taken with Taiwan 2013, which probe to trust is unresolved.** Both
MicroRiders show sh1 above sh2 (6.3× on MR046, 2.9× here), but MR046's spectral
evidence indicts sh1 while doug's indicts sh2, and the VMP comparison favors
the *mean* on MR046 and *sh1* here. Both channels are shipped; decide per use,
and do not treat `epsilonMean` as calibrated.

## Other caveats

- **The probe identities carry no weight.** The config names sh1 M1344 (0.0655)
  and sh2 M1371 (0.0855) and FP07s T865/T866, but **it was never edited between
  2015-10 and 2017-10** — 509 files across CASPER-East, Taiwan 2017 and
  CASPER-West carry the same text, and CASPER-West's patched copies name
  different probes. So the sensitivities are whatever was last set, and
  absolute ε is unverified. See `/Volumes/SeaChest/MICROSTRUCTURE_INVENTORY.md`.
- **ε at this speed cannot resolve the spectral peak at low ε**: 2 s FFT ×
  0.28 m/s = 0.56 m, so the lowest resolved wavenumber is ~1.8 cpm. A platform
  limit, not a setting.
- Positions are the glider's dead-reckoned track with the closing GPS error
  redistributed linearly in time (`flight/make_gps.py`), on the MR clock.

## Reprocess

```bash
cd /Volumes/SeaChest/Taiwan/Taiwan17/gliders/doug
perturb run -c perturb.yaml -j 8        # ~40 min; this MR's config needs no reader fix
```
