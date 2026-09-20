# MicroRider SN 134 on glider doug — CASPER-West 2017

Microstructure from the Rockland MicroRider carried by Slocum glider `doug`
during the CASPER-West field program (R/V *Sally Ride* **SR1715**), off Point
Mugu / Santa Cruz Island, California. **1,416 profiles from 181 of 188 files**,
2017-09-29 18:34 → 2017-10-23 15:43, 2.5–192.5 m, processed 2026-09-20.

Recipe copied from `Taiwan17/gliders/doug/` — the **same glider and the same
MicroRider**, eight months earlier.

| | |
|---|---|
| `../post-recovery/microrider/corrected_headers/` | the 188 `CPW_*.P`. **Not** `uncorrected_headers/` — see below. |
| `dinkum-flight.yaml` → `flight_inputs.nc` | glider flight + CTD on the flight clock, from 402 dbd + 402 ebd |
| `mr_clock.py` → `mr_clock_model.json`, `mr_clock_offsets.csv` | the MR-vs-glider clock |
| `make_flight_speed.py` → `calibration.json`, **`doug_flight_hotel.nc`** | Merckelbach flight-model speed, already on the MR clock |
| `make_gps.py` → `gps.nc` | drift-corrected dead-reckoned track, MR clock |
| `thermistor_health.py` → `thermistor_health.csv` | per-file FP07 health |
| `perturb.yaml` → `Processed/` | profiles / diss / chi / ctd (+ `_binned`, `_combo`), generation `_00` |

## Results (1-m bins, 1,416 profiles)

| | median | spread (MAD) | n |
|---|---|---|---|
| ε (paired) | 1.65e-9 W/kg | 0.91 dex | 246,946 |
| χ (paired) | 4.2e-10 K²/s | 0.65 dex | 240,807 |
| N² | 7.0e-5 s⁻² | 0.37 dex | 246,946 |
| Γ | **0.030** | 0.65 dex | 207,505 |
| K_ρ | 5.5e-6 m²/s | 0.93 dex | 246,943 |
| K_T | 5.6e-7 m²/s | 0.93 dex | 211,571 |

Through-water speed 0.305 m/s (5–95%: 0.202–0.398). `qc_drop_epsilon` is
0.0000 — nothing was dropped by the QC gate, which as at the VMP is a
statement about the gate, not a clean bill of health.

**Γ = 0.030 is low** — four times below Taiwan17 doug's 0.12 and well under
the canonical 0.2. Read it against the known χ-scale problem on FP07s
(`fp07_tau_scale` was measured at 1.66× on ARCTERX), and against the
thermistor disagreement below. It is not evidence about mixing efficiency
until the χ scale is settled.

## ⚠️ T1 failed, and it was already wrong before it failed

`thermistor_health.csv` has the per-file evidence.

- **T1 rails at 58.458 °C from `CPW_037`, 2017-10-03 10:04, to the end of the
  deployment — 140 of 188 files.** T2 is healthy throughout (σ 3.0–4.7 °C,
  max 21.4–25.4 °C). After the rail T1 contributes **no** usable χ window.
- In the three days when both worked, the paired ratio is
  **χ(T1)/χ(T2) = 1.965** over 34,493 windows in 188 profiles, with a MAD of
  only **0.03 dex**. A 2× offset that stable and that depth-independent is a
  relative *bead response* difference, not turbulence — the same signature as
  this cruise's VMP.

**Use χ from T2.** `chiMean` is the geometric mean of the two and is therefore
pulled high wherever T1 contributed at all (the first three days); `chi_2` is
the product. T1 was degrading before it railed, so even the early T1 windows
should not be trusted.

## The shear probes agree

**sh1/sh2 = 0.939 paired, MAD 0.02 dex over 190 profiles** — 6%, and stable.
That is a sharp contrast with the same MicroRider elsewhere: 6.3× on Taiwan
2013's MR046 and 2.94× on Taiwan 2017's doug. Whatever was wrong there is not
wrong here.

**This constrains RELATIVE sensitivity only.** A common error cancels exactly,
and the probe serials come from a config patched retrospectively in 2018
(§ below), so the absolute ε scale is still unverified.

## Why `corrected_headers/`

The 188 casts exist twice. The uncorrected copies carry the **2015 CASPER-East
config** — the logger's `SETUP.CFG` was never updated for this cruise — naming
sh1 M1344 @ 0.0655 and sh2 M1371 @ 0.0855. Since ε ∝ sens⁻², reading those
would put sh1 **47 % high** and sh2 **11 % low**. Verified across all 188 files
of both sets, not sampled.

The corrected copies were patched 2018-02-18 to **sh1 M1194 @ 0.0793, sh2
M1493 @ 0.0805, T1057/T179**, plus a pressure `coef2 = 7.119e-8`. M1194 @
0.0793 is sheet-exact. But this MR's config was untouched from 2015-10-23 to
2017-10-23 (509 files across three deployments), so the patch is a
**retrospective paper record**: it supports the probe *set* and cannot show it
held for all 188 files.

## The clock: 58.5 minutes fast, and it stepped

`glider_time = MR_time + L`.

| block | files | L at t_ref | drift | residual RMS |
|---|---|---|---|---|
| 09-29 → 10-14 | 111 | −3512.08 s | −0.305 s/day | **0.292 s** |
| 10-15 → 10-23 | 64 | −3528.21 s | −0.316 s/day | **0.321 s** |

A **−12.50 s step** between `CPW_120` (10-14 23:30) and `CPW_121` (10-15
05:31), across a 6 h gap in the MR record. Fitting one line through it gives a
spurious −1.02 s/day drift and 3.34 s RMS; the two blocks' drifts agree to 3 %
although fitted from disjoint data.

**Confirmed independently.** Cross-correlating the MR's *healthy* thermistor
(T2) against the glider CTD — a different sensor pair, nothing to do with the
pressure regression the model was fitted on — gives r = 1.000 on all 7 test
files and a lag matching the model to a consistent **+3.50 s** (spread 1.0 s),
on both sides of the block boundary. CASPER-East gives +3.1 s by the same
test. (The same check against the accelerometer is *uninformative*, |r| ≤ 0.07:
glider pitch is nearly a square wave. It is not evidence either way.)

**That +3.5 s is not corrected and is not a clock error** — it is the glider
CTD's own print lag. At these fall rates it displaces merged T and S by about
**0.45 m vertically**, comparable to the 1 m bin width. Left as measured.

`doug_flight_hotel.nc` and `gps.nc` are both written **on the MR clock**, so
`hotel.time_offset: 0`.

## Speed

No EM flowmeter, so U comes from a Merckelbach dynamic flight model
(`gliderflight` 1.2.1) calibrated against the glider's depth rate.
**Cd₀ = 0.1873** (24 days, 5.3 % spread), Vg stable to 0.04 %,
hull compressibility **8.219e-10 Pa⁻¹**.

Checks the fit did not have to pass:
- **w_water medians to −0.0003 m/s (dive) and −0.0005 (climb)** — the vertical
  water velocity should average to zero, and does.
- **α = −2.84° diving, +3.16° climbing**, against Taiwan17 doug's −2.90° /
  +2.89°: the same glider, eight months and an ocean apart.
- Cd₀ 0.1873 against Taiwan17's 0.1753.

**There is still no independent speed reference, and ε ∝ U⁻⁴.** The nearest
check remains Taiwan 2013's Aquadopp on glider *jane*, which found this model's
angle of attack ~1.6° too small on dives. Not applied, by decision
(Pat, 2026-09-15).

## The seven files that produced nothing

`CPW_001`–`CPW_005` (2017-08-26 → 09-29, before the glider record starts),
`CPW_022` (an 85 kB record with both thermistors frozen) and `CPW_188`
(post-recovery). perturb refused them rather than publish the 0.05 m/s
`speed_cutout` floor as a through-water speed. That is the correct outcome.

## Not yet done

**The VMP co-location.** VMP-250IR SN 194 profiled this same cruise
(`../../../vmp/`, 247 files, 2017-09-28 → 10-25) while doug flew. That is a
far better ε cross-check than Taiwan's 24–59 km separation, and it is the
open question this deployment can actually answer. It needs the VMP rebuilt
into `vmp/analysis/VMP_results/` first, and it must carry the VMP's own
caveats: that instrument's sh2 has a depth-windowed fault from 10-09 and fails
outright on 10-24.
