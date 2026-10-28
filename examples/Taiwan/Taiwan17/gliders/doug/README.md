# MicroRider SN 134 on glider doug — Taiwan 2017 (SK-II)

Microstructure from the MicroRider carried by Slocum glider `doug` during
R/V Roger Revelle cruise RR1704 (project SK-II), Luzon Strait. **530 profiles
from 72 of 80 files**, 2017-02-19 – 02-28, reprocessed 2026-10-16 on `main`
98b4206. It replaces the 2026-09-15 set; see
[What changed](#what-changed-from-the-2026-09-15-set).

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
  coefficients are stale — see below), **both beads in all 72 files**. That CTD
  reaches perturb through the hotel file, a reference perturb refuses by
  default since #214, so this config sets `fp07.allow_hotel_reference: true`.
- **chi by spectral fit** (Method 2), because glider shear is
  vibration-contaminated and makes a poor seed for k_B.
- **8 files produced nothing, correctly**: CAS_001–004, 030, 031, 079, 080 are
  deck/startup records at 0.3–1.3 dbar. perturb refused them rather than
  publish the 0.05 m/s speed floor as a speed.

## Results (1-m bins, 530 profiles)

| | median | spread (MAD) | n |
|---|---|---|---|
| ε (paired) | 2.3e-9 W/kg | 1.31 dex | 88,806 |
| χ | 3.2e-9 K²/s | 1.22 dex | 87,878 |
| N² (> 0 only) | 8.7e-5 s⁻² | 0.55 dex | 75,929 |
| Γ | 0.096 | 0.58 dex | 75,282 |
| K_ρ | 5.6e-6 m²/s | 1.48 dex | 79,224 |
| K_T | 3.8e-6 m²/s | 1.74 dex | 85,214 |

Median and MAD (×1.4826) of log10 over the bins of `chi_combo_00/combo.nc`
holding a finite, positive value. **14.5% of bins have N² ≤ 0** under the
temperature sort, and the N² row leaves them out. Γ and K_ρ are masked per
window where N² < N2_min and only then binned.

## What changed from the 2026-09-15 set

- **ε: bit-identical.**
- **χ +1.8% per bin (median), concentrated near the noise floor — #215.** The
  raw config writes the legacy key `beta = 3143.55` (no `beta_1`); before
  #215 chi's FP07 noise model ignored it and used 3000. New/old runs from
  +13.5% in the weakest χ decile to +0.2% in the strongest, the same
  signature as Taiwan 2013's MR046, where #215 was bisected.
- **FP07 — #214.** CAS_006 and CAS_078 now have both beads fitted rather
  than factory coefficients.
- **N², Γ, K_ρ, K_T — #219, the temperature sort by default.** Wherever
  temperature and density order agree the bin is unchanged (per-bin medians
  1.000); the 14.5% of bins where salinity reorders density now carry
  N² ≤ 0, and the run notice names 10 files where it scrambles the order.

**The thermistors agree**: pooled χ(T1)/χ(T2) = **0.987** over 147,617 windows
(0.982 in the 2026-09-15 set),
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
reads 3–5× low and the shipped `epsilonMean` ~2× low. (The VMP column is
from the 2026-09-15 VMP set; on the reprocessed VMP the same day medians move
by ≤ 4%, from its regridded windows.) **This is a bound on
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
  absolute ε is unverified. Sheets received 2026-10-12 sharpen this: 0.0655 /
  0.0855 are M1344's and M1371's **2015** values, and Rockland recalibrated
  both on **2016-04-11 to 0.0615 / 0.0729**. If those two probes really flew
  here, ε is low by 1.13× (sh1) and 1.38× (sh2). See `/Volumes/SeaChest/MICROSTRUCTURE_INVENTORY.md`.
- **ε at this speed cannot resolve the spectral peak at low ε**: 2 s FFT ×
  0.28 m/s = 0.56 m, so the lowest resolved wavenumber is ~1.8 cpm. A platform
  limit, not a setting.
- Positions are the glider's dead-reckoned track with the closing GPS error
  redistributed linearly in time (`flight/make_gps.py`), on the MR clock.

## Reprocess

```bash
cd /Volumes/SeaChest/Taiwan/Taiwan17/gliders/doug
perturb run -c perturb.yaml -j 8        # ~8 min; these files were made on main 98b4206
```
