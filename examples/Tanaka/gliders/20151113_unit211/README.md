# MicroRider SN 058 on Slocum unit211 -- 20151113_unit211

One of T. Tanaka's pre-paper glider deployments: 29.4 N 127.7 E, East China Sea, 2015-11-13 15:41 .. 11-14 22:12
UTC, dives to ~1000 m, 12 MicroRider files. **Reprocessed 2026-10-17 on main
98b4206** (`microstructure-tpw 0.4.0 (git v0.4.0-29-g98b42062069c)`, stamped
in every product); first processed 2026-10-13.

## Verdict

| | status |
|---|---|
| epsilon | **sh2 only** -- sh1 is dead (below). Climbs usable; dives marginal (slow) |
| chi | both thermistors fitted in situ per file in **all 12 files (33 of 33 profiles)**, one calibration; chi1/chi2 = 0.87 paired |
| clock | solved, 7 files, 0.87 s RMS |
| speed | flight model with DEFAULT lift and Cd1 -- untested for this glider |

## Products -- `Processed/`, generation **_00**

33 profiles (20 dives, 13 climbs), 1-m bins. The slow dives are cut into
segments (1504003 holds one dive as five profiles, 592 -> 854 m), so most
profiles start deep. Median, MAD of log10 (dex), and n over finite positive bins:

| | all | dives | climbs |
|---|---|---|---|
| U (m/s) | 0.247 | 0.125 | 0.386 |
| epsilon (sh2), W/kg | 1.1e-9 (0.73, 8,777) | 7.2e-10 (0.85, 4,458) | 1.4e-9 (0.51, 4,319) |
| chi, K2/s | 4.5e-9 (0.79, 8,103) | 5.7e-9 (0.42, 3,909) | 2.1e-9 (1.11, 4,194) |
| N2 (> 0), s-2 | 2.3e-5 (0.73, 7,069) | 1.7e-5 | 3.2e-5 |
| Gamma | 0.29 (0.79, 6,314) | 0.66 | 0.12 |
| K_rho, m2/s | 1.1e-5 (1.02, 7,510) | 1.1e-5 | 1.2e-5 |
| K_T, m2/s | 6.9e-5 (1.86, 7,818) | 1.5e-4 | 3.1e-5 |

N2 <= 0 in 18.8% of bins: the temperature sort (#219) no longer forces it
positive, and salinity scrambles the density order in 4 of 12 files (run
notice). Gamma and K_rho are masked where N2 < N2_min.

**The legs disagree, in opposite directions, and it is unexplained.** In the
same 1-m bins (median over dive profiles / median over climb profiles, then
the median over 749-789 bins): epsilon 0.47, chi **4.5**, Gamma **5.4**. The
same comparison puts N2 at 0.86, so it is the microstructure, not the
stratification. A speed error cannot do it: it moves epsilon (~U^-4) and chi
(~U^-2) the same way. The 50% FM failure rate on climbs is also unexplained;
vibration on the faster climbs is one hypothesis, untested. Treat absolute
epsilon as uncalibrated in any case: no sheet exists in `microstructure_sensors/`
for M1369 or M887.

## Instrument

MR1000-LP **SN 058** -- the config header reads "File made for Yasuda
MR1000-LP SN058, 2011-12-19"; it has no `[instrument_info]` block, so perturb
treats the vehicle as unknown (speed smoothing tau 1.5 s, the generic default).
Probes sh1 **M1369 @ 0.0576**, sh2 **M887 @ 0.0583**, both diff_gain 0.97;
FP07s with the 2011 NOMINAL coefficients, so `fp07.calibrate` fits them per
file against the glider CTD. No EM flowmeter.

**sh1 is dead.** Its raw counts have a standard deviation of ~3 (electronics
noise only) against ~320 on sh2; its epsilon sits at 1e-13 .. 1e-11 W/kg and
fails the FM test on 100% of windows. `perturb.yaml` excludes it under
`instruments: microrider:` (keyed on the directory name, as perturb requires).

## FP07: one calibration, fitted per file

`fp07.calibrate: true` with `fp07.allow_hotel_reference: true` fits both
thermistors per file against the glider CTD (`sci_water_temp`), and it now does
so in all 12 files. The thermistors are healthy, r = 0.9999 with the CTD. On
the 2011 nominal coefficients they read a near-constant **+8.8 C (T1) and
+9.8 C (T2) high**. Fitted, the median T - CTD per profile is within 15 mK on
28 of 33 profiles. On the other five, all partial climbs from depth (ending at
360-670 m), the FP07 reads 0.05-0.13 C warmer than the CTD, and the old set
shows a similar excess on the same segments, so it is in the data, not the fit
(the cause, perhaps a leg-dependent lag, is untested). On the
PR #214 branch, each per-file fit matched the pooled `fp07-cal` fit below:
gradient gain ratio median 0.9999 (range 0.994-1.003), temperature median 3 mK
(max 8 mK).

### The 2026-10-13 set: two calibrations, and the fitted one was worse

That set fitted the thermistors in only 3 of 12 files (1504003, 1504005,
1504012: 19 of 33 profiles) and left the other 9 (14 profiles) on nominal
coefficients ("no profile reached the lag correlation gate"). Why the gate
failed, VERIFIED by re-running with each fix on branch
`fix/fp07-cal-hotel-reference` (PR #214) reverted in turn:
- **A gap silenced whole profiles.** One NaN in a profile's reference (the
  hotel merge blanks gaps wider than `max_gap`) made `_calc_lag` abstain for
  the WHOLE profile, counted as neither accepted nor rejected. Splitting
  profiles at their gaps alone fits T1 in 11 of 12 files and T2 in 10.
- **The low-pass started from zero.** Its output climbed from 0 to the counts'
  level over seconds, and the files whose only profile starts at the top of
  the file (1504004, 1504010) scored that ramp. Starting the filter at rest
  fits them too.
- The CTD's 4.5 s spacing was NOT the cause, as this README first guessed.

**CORRECTED 2026-10-17.** This README said the nominal profiles were the
problem: their gradient gain against the CTD is 0.975-0.990, so chi ~3-4%
high. Both halves of that were wrong:
- **The old per-file fits were wrong on the full-depth profiles.** They agree
  with the new fit to a few mK on the deep, narrow-range segments (4.8-7.7 C)
  and diverge over the full 6-24 C profile. On 1504003 prof001 T1 read 0.47 C
  off the CTD, its gradient ~2.2x too steep, and chi was 3.4x (T1) and 3.1x
  (T2) the new value; on 1504012 prof001 T1 read 2.8 C off and its chi was
  30x the new value (median over windows).
- **Net, chi on the nominal profiles was not high.** The gain error was real,
  but the legacy-`beta` bug (#215: the noise model on beta = 3000, not this
  config's 3143.55) pulled chi down by about as much. Per profile the two
  cancel to 0.977-1.024 (median 0.996 T1, 0.998 T2).

### The `fp07-cal` cross-check

The pre-pipeline `fp07-cal` tool decimates the thermistor onto the CTD's own
samples and gates lag on peak sharpness rather than r. **Its fit was run on
2026-10-13 (`fp07cal/`, FIT ONLY: nothing is patched):**
- all 12 files contribute pairs (16,587 used for T1), 33 profiles, and the CTD
  covers 87% of the record at a median 4.5 s;
- lag +4.7 s on both thermistors, peak width 0.8 s, r 0.94;
- order 2, chosen by held-out error (47 mK held out, 3.5 mK in sample for T1;
  2.6 mK for T2); dive/climb split -1.0 / -0.3 mK.

Against nominal, its gradient gain is 0.986 (T1) and 0.983 (T2), ranging
0.977-0.992 over 4.5-24 C, with offsets of +8.9 / +9.9 C. A 1 Hz regression
of the CTD on nominal FP07 (5-21 C, best lag, six files) gave 0.980-0.987
independently. There are two ways to put the deployment on one calibration;
the products use the second:
- `fp07-cal patch`, then perturb with `fp07.calibrate: false`;
- perturb's per-file fit with `fp07.allow_hotel_reference: true`.

## What changed from the 2026-10-13 set

Old generation `_01` against new `_00`, the same 33 profiles, 1-m bins.
- **epsilon: bit-identical** (8,777 bins).
- **chi: paired median ratio 0.996** (chiMean, 7,851 bins), but per profile it
  ranges 0.03-1.53. On the 14 formerly nominal profiles it is 0.977-1.024; the
  19 formerly fitted profiles carry the large changes, the full-depth ones
  above. Attributed by a one-file bisect on 2015-11-19 (same MR, probes and
  CTD): commit b1270b5 reproduces the old products bit-for-bit; #214 moves chi
  by x0.94-0.96 on a nominal profile and x1.29-1.60 on badly fitted ones; #215
  then adds +4 to +9%; #219 leaves chi alone.
- **N2, Gamma, K_rho, K_T** (#219, temperature sort): where N2 is positive in
  both sets it is unchanged (paired ratio 1.000), but 18.8% of bins now have
  N2 <= 0 (0% before), and Gamma and K_rho are masked there. Medians: N2
  1.7e-5 -> 2.3e-5, Gamma 0.40 -> 0.29, K_rho 1.7e-5 -> 1.1e-5, K_T
  1.3e-4 -> 6.9e-5. In the same depth bins, N2 dive/climb went from 1.38 to
  0.86.
- chi1/chi2 (median over profiles of each profile's median log ratio): 0.94
  -> 0.87 (MAD 0.24 dex, 32 profiles). This README previously quoted 0.92, by a
  method not recorded.

## Recipe (run from this directory)

1. `dinkum-hotel build -c dinkum-flight.yaml` -> `flight_inputs.nc`. Sea-Bird
   CTD on `sci_ctd41cp_timestamp`; buoyancy `m_de_oil_vol`; the cache is a copy
   of `../../cache` (Pat's 2016 sensor-list cache).
2. `python mr_clock.py extract` then `solve` -> `mr_clock_model.json`.
   glider_time = MR_time + L, L = +3.16 s, drift -0.62 s/day,
   7 of 12 files (1504003, 1504005, 1504007, 1504008, 1504009, 1504011, 1504012).
   **One-leg files are refused**: the short files hold a single dive or climb,
   a near-monotonic ramp on which the fitted pressure offset absorbs any time
   shift; before that gate two of them (1504010, 1504013) returned -9.0 and
   -6.25 s against +1.75..+4.25 s from the full yos.
3. `uv run make_flight_speed.py calibrate` then `run` -> `unit211_flight_hotel.nc`
   (already on the MR clock). gliderflight 1.2.1, default lift/Cd1, mass 60 kg
   nominal; global steady fit Cd0 0.219, Vg 58.70 L, compressibility
   6.08e-10 Pa-1; per-day dynamic Cd0 0.231, 0.214.
4. `perturb run -c perturb.yaml -j 8` (~1.5 min; these files were made on main
   98b4206).

## The slow dives

The measured dive rate is only 0.06-0.08 dbar/s against 0.17-0.18 on climbs,
with the full -261 cc of oil in: the glider was barely heavy for this warm
water. The model puts the dive through-water speed at 0.12-0.15 m/s. That is
real, not a model artifact (dP/dt / sin(pitch + AOA) gives the same), and it
makes shear-probe epsilon on the dives marginal.

## Untested, and worth doing

Whether the default lift misfits unit211 the way Tanaka et al. (2022) showed
it misfits unit095 (by up to 1.3 deg AOA). The 2015-11-13 mission carried a
DVL: `../post-recovery/glider/*.pd0` (47 files) could measure the AOA directly,
as the paper did.
