# MicroRider MR-SE SN 240 on SeaExplorer SEA025 — Kuroshio, June 2017

The SeaExplorer of Tanaka et al. (2022, *JTECH*,
doi:10.1175/JTECH-D-21-0074.1). It flew mission 24 along the Kuroshio southeast
of Japan, 2017-06-20 01:14 → 06-26 22:17 UTC, with dives to 700 m. It carried
a Rockland MR-SE SN 240 in the nose: two shear probes, one FP07, RSI's JAC-CT,
an AEM1-G EM sensor and an optode. A Nortek AD2CP looked up from the top.
**Reprocessed 2026-10-17 on main 98b4206** (`microstructure-tpw 0.4.0 (git
v0.4.0-29-g98b42062069c)`, stamped in every product); first processed
2026-10-13 on branch `feat/rsijac-ct`, merged as b1270b5 (#213).

## Verdict

| | status |
|---|---|
| ε | both probes. sh1/sh2 = **0.67 on dives, 0.91 on climbs** (see below) |
| χ | one FP07 (T1415). Wide spread; many windows are below the FP07 noise floor |
| CTD | the MR's own RSI JAC-CT; the SeaExplorer payload had none |
| clock | navigation = MR + **4.7 s**, no drift, 62/64 files, 0.47 s RMS |
| speed | steady-state flight model with the paper's lift/C_D1, reproducing its C_D0, V_GL and c |

## Products: `Processed/`, generation _00

143 profiles (75 dives, 68 climbs) from all 64 files, in 1-m bins.

| | dives | climbs |
|---|---|---|
| speed (model) | 0.278 m/s | 0.395 m/s |
| ε, mean of the two probes | 7.6e-10 W/kg (MAD 0.39 dex) | 1.07e-9 (MAD 0.46 dex) |
| sh1 / sh2, paired | 0.67 (MAD 0.14 dex) | 0.91 (MAD 0.14 dex) |
| FM > 1.15, sh1 / sh2 | 25% / 21% | 45% / 35% |

1-m bins, median (MAD of log10, n): χ 2.3e-9 K²/s (1.37 dex, 82,239), N² (> 0)
2.8e-5 s⁻² (0.58, 77,765), Γ 0.10 (0.69, 71,515), K_ρ 9.2e-6 m²/s (0.82,
79,504), K_T 6.5e-6 m²/s (1.47, 76,273). N² ≤ 0 in 6.9% of bins: salinity from
the JAC-CT scrambles the density order in all 64 files (run notice), so
stratification is temperature-sorted (#219). Γ and K_ρ are masked where
N² < N2_min.

**The probe ratio depends on the leg, and that is unexplained.** A sensitivity
error is a fixed multiplier and cannot change with direction. A flow effect
can: the probes sit in the nose behind a sensor guard. That is a hypothesis,
untested. Climbs also fail the FM shape test about twice as often as dives.
Neither probe has a calibration sheet in `microstructure_sensors/`. The config
header says RSI first wrote nominal values (2017-03-30), and it was patched
2017-06-07, 06-27 and 2021-02-06. **Absolute ε is unverified.**

## What changed from the 2026-10-13 set

The same 143 profiles, 1-m bins.
- **ε: bit-identical** (83,572 bins).
- **χ: paired median ratio 1.000** (82,211 bins), but 4% of windows move by
  more than 1% and 0.15% by more than 0.1 dex. All of it is #214: a one-file
  bisect (DAT_134) has commit b1270b5 reproducing the old products
  bit-for-bit and 53787f0 (#214) making the whole change. #214 moved the T1 fit
  by under 0.1% in gradient gain (0.9997-1.0009 per profile), and T1 still
  agrees with JAC_T to ~1 mK. The windows that move are the more energetic
  ones (K_max/k_B median 2.2 against 3.4); why a 0.1% calibration change moves
  them by more is untested. #215 does not apply: this config uses `beta_1`.
- **Stratification** (#219, temperature sort), paired ratios new/old: N² 0.70,
  |dT/dz| 1.41, Γ 0.44 (median 0.30 → 0.10), K_ρ 1.50, K_T 0.45. N² ≤ 0 in 6.9%
  of bins (0% before).
- **The temperature sort removes a leg bias.** In the same 1-m bins (median
  over dives / median over climbs, then the median over 705 bins), the old
  density-sorted N² was 1.51 times higher on dives than climbs; the new is
  1.03. Γ went from 1.66 to 1.10, K_ρ from 0.60 to 1.03, K_T from 2.90 to 1.16.
  The water is the same on both legs, so a leg-dependent N² is an artifact.
  The likeliest source is the CT's direction-dependent salinity error (thermal
  lag), which the density sort carried; that is inference, not separately
  tested. ε dive/climb is 0.89 in both sets.

## Recipe (run from this directory)

1. **`python make_nav.py` → `nav.nc`.**
   - Reads the 66 `sea025.24.gli.sub.*.gz` files, a 20-s semicolon CSV.
   - `Depth` is the water depth. **`Pa` is internal hull pressure**, 75–82 kPa
     throughout.
   - Positions are NMEA ddmm.mmm. Pitch is positive nose-up.
   - Eleven rows with an unset 2009 clock are dropped.
   - The payload files (`pld1.raw`) are not used. They carry MR telemetry,
     the FLBBCD and AD2CP summaries, and no CTD.
2. **`python mr_clock.py extract`, then `solve`.**
   - Writes `mr_clock_model.json`.
   - The MR pressure is regressed on the navigation `Depth`.
   - **Needs the `rsijac` converters** (in main since b1270b5), because
     extract also caches 1 Hz JAC_T, JAC_C and `U_EMC`.
   - DAT_124 and DAT_172 are too short (< 1800 s in water).
   - The paper says the navigation clock was "about 3 sec behind" the MR.
     The pressure match gives +4.7 s the other way (nav time = MR time + 4.7).
     Either the paper's wording means lagging, or a filter delay in the 20-s
     depth is included; **unresolved**.
3. **`uv run make_flight_speed.py calibrate`, then `run`.**
   - Writes `sea025_flight_hotel.nc`, already on the MR clock.
   - gliderflight 1.2.1 **steady-state** model. Its dynamic model's
     added-mass coefficients are Slocum's and have never been measured for a
     SeaExplorer.
   - Fixed from the paper (Sect. 3.2): a = 4.0 rad⁻¹, C_D1 = 5.0 rad⁻²,
     S = 0.14 m², m = 63.45 kg.
   - Density comes from the JAC-CT.
   - The calibration mask is the paper's quasi-steady criteria: 50–650 m,
     |pitch| > 15°, oil settled, and angular-actuator rate inside its 1–99%.
   - Fitted: **C_D0 0.184, V_GL 62.03 L, c 5.43e-10 Pa⁻¹**, against the
     paper's 0.19 / 62.0 L / 5.3e-10.
   - Modeled AOA −6.8° on dives and +4.4° on climbs. The paper's ADCP means
     are −7.7° and +5.1°, over different selections.
4. **`perturb run -c perturb.yaml -j 8`** (~4.5 min; these files were made on
   main 98b4206).

## The EM sensor (`U_EMC`): not used

Model speed / `U_EMC` is **0.920 on dives** (IQR 0.888–0.950) and **0.934 on
climbs** (0.910–0.957), at 1 Hz over the calibration mask. The paper gives
0.90 against the ADCP. Total-least-squares gain-plus-zero fits on 10-min means
disagree between the legs (dives 1.176·U − 0.025, climbs 1.032·U + 0.018), so
no single gain+zero model is supported here.

The channel name `U_EMC` also keeps perturb's `speed.method: em` from seeing
it at all.

## Not done, and worth doing

- **The AD2CP** (`../post-recovery/adcp/`) recorded both legs in beam
  coordinates. It is the direct check on this speed and on the probe-ratio
  question. The paper's SeaExplorer code is not in its package.
- The optode (`DO`, `DO_T`) is untouched.
