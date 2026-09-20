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

## The VMP intercomparison — done 2026-09-20

VMP-250IR SN 194 was rebuilt (`../../../vmp/analysis/VMP_results/`, 2,097
profiles from 247 files, 2 errors — the two known empty files). **The ship's
stations were deliberately set up to overlap the MicroRider's virtual
mooring** (Pat), and the geometry confirms it: the ship came within 0.58–2.4 km
of the mooring on **five separate days**.

The two platforms were otherwise **not** co-located — the MR held a
4.7 × 7.9 km box offshore for 24 days while the VMP worked a station grid at a
median 26 km away — so this is a close-approach comparison, restricted to
pairs within a stated distance and time, not a bulk one.

Tool: `colocate_vmp.py` (paired statistic: median over depth bins of
log₁₀(VMP/MR) within a pair, then over pairs).

### ε — consistent, and it bounds gross error

≤ 5 km, ≤ 3 h, by depth band:

| band | n pairs | VMP ε | MR ε | VMP/MR | MAD |
|---|---|---|---|---|---|
| 2.5–35 m | 2,416 | 1.94e-8 | 6.76e-8 | 0.42 | 0.48 dex |
| 35–65 m | 2,370 | 2.96e-9 | 3.98e-9 | 0.88 | 0.47 dex |
| **65–130 m** | **2,353** | **1.19e-9** | **8.26e-10** | **1.75** | **0.37 dex** |

65–130 m is the band to read: it is below the VMP's sh2 fault window and the
VMP's own probes agree there. The ratio is **stable against both thresholds** —
1.19/1.38/1.29 at ±1/3/6 h, and 1.46/1.38/1.14/1.17 at 2/5/10/20 km — so it is
not set by temporal or spatial aliasing.

**But there is no stable offset.** Day by day in that band the ratio runs
**0.26 to 4.10** across 16 days, with tight *within*-day scatter (MAD
0.22–0.45 dex). A real instrumental offset would repeat; this does not. So the
comparison says:

- **No gross error.** A factor-10 scale error — what a wrong `diff_gain` rung
  would give — is excluded. Given that ε rests here on a shear sensitivity
  patched retrospectively in 2018, that is worth having.
- **It cannot resolve a factor of ~2.** Real ε patchiness between a moored
  sampling and a snapshot cast, even 1 km apart, is larger than the question.

### χ — a 9.7× gap that is NOT yet attributable to the instruments

Same pairs, 65–130 m, VMP `chiMean` vs MR `chi_2` (T1 railed, excluded),
excluding 10-24/25 where the VMP probes fail outright:

**VMP/MR = 9.72, and all 16 days lie between 3.30 and 18.54** — same sign
every day, within-day MAD 0.12–0.32 dex. That is the signature of a systematic
offset, not ocean variability, and unlike ε it does not straddle 1. It is also
the right order to explain Γ: 0.165 (VMP) vs 0.030 (MR), a factor 5.5.

**Resolved 2026-09-20: the FP07 beads were degraded. It is not the processing.**

Every processing explanation was tested and eliminated. Runs live beside the
shipped `chi_00` as generations `chi_01`..`chi_05` (changing `chi.*` re-versions
only the chi stage, so each reuses `profiles_00`/`diss_00`).

| candidate | test | result |
|---|---|---|
| chi method 1 vs 2 | `chi_01`, `use_epsilon: true` | closes **1.26×** (9.72 → 7.69) |
| window 2 s → 1 s | `chi_04`, `fft_sec: 1.0` | **0.78× — makes it worse** |
| FP07 tau | `chi_03`, `fp07_tau_scale: 2` | d(log χ)/d(log s) = **0.76**; closing the gap needs **s = 14.7, tau = 266 ms**. A bead is 10–20 ms. **Falsified.** |
| mean dT/dz | co-located 65–130 m | ratio **1.00** (MAD 0.07); T_mean agrees to 0.06 °C. **Falsified.** |
| gradient diff_gain | config audit | MR 0.95/0.95 vs VMP 0.96/0.93. **Falsified.** |
| Batchelor fit + noise model | band-integrated `spec_gradT − spec_noise` | see below. **Falsified.** |
| Goodman | `chi_05`, `goodman: false` | χ ×**0.992**, variance ×1.016. **Falsified.** |

Matching method *and* window together returns to 9.8× — the two nearly cancel.

**The gap is in the measured signal.** Integrating the observed,
noise-subtracted gradient spectrum over the identical **2–100 cpm** band,
co-located in 65–130 m: VMP **1.03e-3** vs MR **1.11e-4** K²/m², a ratio of
**9.28** against a fitted χ ratio of 9.8. The raw integral reproduces the whole
gap, so nothing in the fitting is responsible. The MicroRider's thermistor
records ~**3×** less gradient amplitude (√9.28) in the same water.

**And the same MicroRider was healthy eight months earlier**, at essentially
identical stratification:

| | χ | Γ | N² |
|---|---|---|---|
| Taiwan17 doug, Feb 2017 | 3.07e-9 | **0.12** | 6.98e-5 |
| **CASPER-West doug, Oct 2017** | **4.21e-10** | **0.030** | 7.00e-5 |
| CASPER-West VMP 194, Oct 2017 | 1.38e-8 | 0.165 | 1.14e-4 |

**Reading (hypothesis, consistent with every measurement above):** the FP07
beads on this deployment were **attenuating**, not merely slow. A fouled or
cracked bead loses microscale amplitude at a roughly frequency-independent
factor while still reporting the mean temperature correctly. That explains,
together, why the slow T channel agrees (T_mean 0.06 °C, dT/dz 1.00) while the
microscale variance is 9.3× low; why a tau correction cannot reach it (tau
changes the rolloff *shape*, not a flat gain); why the Method-1/2 tau fit on
this instrument never converges (k_B,fit/k_B,ε sits at 0.50–0.84 across
tau 5–50 ms and never crosses 1, while the VMP's crosses cleanly at
s = 1.40/2.10); and why **T1 on the same instrument railed outright** on
2017-10-03 — the same failure, further along.

**Consequence: CASPER-West MicroRider χ, Γ, K_T and K_ρ are LOW by roughly an
order of magnitude and must not be used as absolute values.** ε is unaffected
(shear probes, agreeing to 6%, and the co-located ε test is consistent).
Relative structure in χ — vertical and temporal patterns — is probably still
informative, since the attenuation looks multiplicative, but that is untested.

## Still open

- **Confirm the bead-attenuation reading** on an instrument where the beads
  are known-good, and check whether the attenuation is really flat in
  wavenumber (fit a constant gain alongside tau, which the single-pole model
  cannot currently express).
- Cross-deployment FP07 tau as a **bead-health diagnostic** (Pat's suggestion):
  tau should not move until the glass fails. 12 beads recur with different
  shear probes; **T1592** is the best (5 shear pairs, 2019–2026) but its legs
  are unprocessed. Start with the control — **ARCTERX-2022 Interior vs Wake**,
  same beads T1592/T2005, same shear M2244/M2245, three weeks apart, both
  already processed. Note χ moves only as s^0.76, so tau is now a health
  check, not a scale correction.
- The 2015 Aquadopp on glider `bob` — a speed-scale check for this fleet.
- The +3.5 s glider-CTD print lag, measured and left uncorrected.
