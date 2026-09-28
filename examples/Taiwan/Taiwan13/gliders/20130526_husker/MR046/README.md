# MicroRider MR1000-LP SN 046 on glider husker — Taiwan 2013

Microstructure from the MicroRider carried by Slocum glider `husker` during
RR1306 leg 3, South China Sea. **495 profiles from 58 of 77 files**,
2013-05-02 05:55Z – 05-26 07:32Z, processed 2026-09-15.

| | |
|---|---|
| `DATA/` | The 77 original `.P` files (uppercase) + `046SETUP.CFG`, `CAL046.BAT`, `ODAS5IR.PXE`, `USBL.PXE`. Untouched. |
| `perturb.yaml` | Processing config — read its header first. |
| `Processed/` | `profiles/diss/chi/ctd` (+ `_binned`, `_combo`), generation `_00`. |
| `../flight/` | **The through-water speed** (Merckelbach flight model) and the MR-vs-glider clock. **Read `../flight/README.md`.** |
| `../hotel/` | The CTD-clock hotel file and the drift-corrected track (used for GPS here). |

## Reading it requires a post-v0.4.0 reader

These files use RSI's 2012 config dialect, one named stanza per channel
(`[pitch]`, `[shear1]`, `[therm1]`, …). **v0.4.0 and every earlier release
build zero channels from them.** The fix is PR #196, merged to `main` on
2026-09-16 as **29f3ea9**. These products were made from its branch commit
89f6f79, whose `src/` is identical to 29f3ea9's.

## What was done

- **Speed**: no EM on this MR, so U comes from a Merckelbach dynamic flight
  model calibrated per day against the glider's depth rate (`../flight/`).
  Median 0.30 m/s (5–95%: 0.20–0.41). **No speed was measured on husker, so the
  absolute scale is unvalidated, and ε ∝ U⁻⁴.** The nearest check is indirect:
  an Aquadopp HR on the sister glider jane (2013-04-29 → 05-04) finds the same
  model's angle of attack ~1.6° too small on dives — U ~5–6% high, dive ε
  ~20–25% low; climbs agree to 0.3%. **Not applied here, by decision (Pat,
  2026-09-15).** Details: `../flight/README.md` and
  `docs/aquadopp_flight_model_validation.html` in microstructure-tpw.
- **Clock**: the hotel file is already on the MR clock (the MR ran −2.1 s in
  May 1–3 and −14.3 s drifting +0.37 s/day in May 17–26).
- **FP07**: the config's coefficients are wrong (raw T2 reads 21–39 °C), so
  both beads are fitted **in situ per file** against the glider CTD. It works:
  calibrated T1/T2 span 17.2–27.7 °C against the CTD's 17.3–27.7 °C.
- **chi by spectral fit** (Method 2), because glider shear is
  vibration-contaminated and makes a poor seed for k_B. K_max/k_B median 2.04,
  so the Batchelor rolloff is resolved.
- **19 files produced nothing, correctly**: KTA_001–004 and FSA_001–009 (bench
  and 2013-05-01, before the glider's science record starts 05-02 04:45Z),
  FSB_001–005 and FSB_026. perturb refused them rather than publish the
  0.05 m/s speed floor as a speed.

## Results (1-m bins, 495 profiles)

| | median | spread (MAD) | n |
|---|---|---|---|
| ε (paired) | 1.7e-9 W/kg | 0.99 dex | 76,213 |
| χ | 7.8e-9 K²/s | 1.03 dex | 70,939 |
| N² | 1.4e-4 s⁻² | 0.59 dex | 76,213 |
| Γ | 0.274 | 0.62 dex | 50,497 |
| K_ρ | 2.9e-6 m²/s | 1.20 dex | 76,119 |

## ⚠️ Two instrument faults — read before using ε or χ

### 1. The shear probes disagree by a factor of 4–6

Over 141,189 windows the pooled ratio is **sh1/sh2 = 6.3**, and per file it
runs 2.2–13.4. It is not a calibration error:

- **Where the signal is strong** (sh2 > 1e-7) the ratio is still **4.35**. A 4×
  in ε needs a 2.1× sensitivity error, which would put one probe outside the
  fleet's entire observed range (0.058–0.113). Both are configured at RSI's
  nominal 0.0700, and this config was never edited (below).
- **sh1's noise floor is ~3× higher**: 1st percentile 1.2e-11 vs 7.6e-12 W/kg.
- **sh1 fails the spectral-shape test twice as often**: FM > 1.15 in 22.0% of
  its windows against 10.6% for sh2.
- The two still **track the same signal** (correlation of log ε = 0.77 overall,
  0.89 where sh2 > 1e-8), so sh2 is measuring, not dead.

**Which probe is right is NOT established.** On this MR the spectral evidence
points at sh1 (worse FM, higher floor). The Taiwan 2017 MicroRider (SN 134 on
doug, `Taiwan17/gliders/doug/`) shows the same sign — sh1 above sh2, by 2.9× —
but there **sh2** fails FM more often (18.8% vs 14.0%), and sh1 is the channel
that matches the ship's VMP. So "sh1 is contaminated" fits husker's spectra and
contradicts doug's. Both channels are shipped; `epsilonMean` is their geometric
mean and sits ~2× either side.

**An independent check, the ship's VMP SN 002**, which profiled while husker
flew (`../../../leg3/VMP/Processed/`), 20–150 dbar:

| day | separation | VMP ε | MR sh1 | MR sh2 | MR geometric mean |
|---|---|---|---|---|---|
| 2013-05-20 | ~14 km | 1.55e-9 | 6.8e-9 | 4.9e-10 | 1.8e-9 |
| 2013-05-22 | ~45 km | 2.87e-9 | 4.0e-9 | 3.0e-10 | 1.1e-9 |

The shipped `epsilonMean` lands within 1.2× and 2.6× of the VMP while each
probe alone is 3–10× off in opposite directions. **This is a bound on gross
error, not a validation**: the platforms are 14–45 km apart in different water,
ε varies by decades spatially, and the mean's agreement may simply be sh1 high
times sh2 low. Note the same comparison on doug (Taiwan 2017) favors **sh1**,
not the mean — one more reason not to pick a probe on this evidence.

### 2. T1 dies at FSB_020 — χ from that bead is invalid for 325 of 495 profiles

T1 reads a constant **−16.701 °C** (the bridge rail) from `FSB_020.P` onward;
it is alive and tracking in FSA_010–014 and FSB_006–019. χ for the dead bead is
NaN in those profiles, so `chiMean` falls back to T2 — with one exception:
**in FSB_020 itself the in-situ FP07 fit did not refuse the dead channel.** It
regressed a constant and produced a plausible-looking −10.7 °C, and χ values
came out of it. Treat FSB_020's `chi[probe="T1_dT1"]` as invalid.

perturb has no per-thermistor exclusion (only `exclude_shear_probes`), which is
why this is documented rather than configured away.

## Other caveats

- **The probes have no serial numbers and nominal sensitivities** (sh1 and sh2
  both 0.0700, diff_gain 0.97/0.95). This MR is **not OSU's** — its config
  reads "File made for LSL MR1000-LP SN046", i.e. Lou St. Laurent's. Absolute ε
  rides on a placeholder even before the pair disagreement above.
- **ε at this speed cannot resolve the spectral peak at low ε**: 2 s FFT ×
  0.3 m/s = 0.6 m, so the lowest resolved wavenumber is ~1.7 cpm. A platform
  limit, not a setting.
- Positions come from the drift-corrected dead-reckoned track in
  `../hotel/hotel.nc`, matched within 600 s.

## Reprocess

```bash
cd /Volumes/SeaChest/Taiwan/Taiwan13/gliders/20130526_husker/MR046
perturb run -c perturb.yaml -j 8   # ~35 min; needs microstructure-tpw main >= 29f3ea9
```
