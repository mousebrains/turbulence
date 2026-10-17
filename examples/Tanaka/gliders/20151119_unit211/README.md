# MicroRider SN 058 on Slocum unit211 -- 20151119_unit211

One of T. Tanaka's pre-paper glider deployments: 32.3 N 133.2 E, off Shikoku, 2015-11-19 04:50 .. 11-20 04:51
UTC, dives to ~1000 m, 11 MicroRider files. **Reprocessed 2026-10-17 on main
98b4206** (`microstructure-tpw 0.4.0 (git v0.4.0-29-g98b42062069c)`, stamped
in every product); first processed 2026-10-13.

## Verdict

| | status |
|---|---|
| epsilon | **sh2 only** -- sh1 is dead (below). Climbs usable; dives marginal (slow) |
| chi | both thermistors fitted in situ per file in **all 11 files (19 of 19 profiles)**, one calibration; chi1/chi2 = 1.02 paired |
| clock | solved, 6 files, 0.89 s RMS |
| speed | flight model with DEFAULT lift and Cd1 -- untested for this glider |

## Products -- `Processed/`, generation **_00**

19 profiles (7 dives, 12 climbs), 1-m bins. Median, MAD of log10 (dex), and n
over finite positive bins:

| | all | dives | climbs |
|---|---|---|---|
| U (m/s) | 0.276 | 0.169 | 0.378 |
| epsilon (sh2), W/kg | 1.3e-9 (0.55, 8,928) | 1.0e-9 (0.70, 4,507) | 1.4e-9 (0.41, 4,421) |
| chi, K2/s | 3.1e-9 (0.70, 8,281) | 3.7e-9 (0.46, 4,077) | 1.9e-9 (0.95, 4,204) |
| N2 (> 0), s-2 | 3.3e-5 (0.67, 7,245) | 2.8e-5 | 4.2e-5 |
| Gamma | 0.17 (0.75, 6,711) | 0.27 | 0.071 |
| K_rho, m2/s | 9.7e-6 (0.94, 7,662) | 9.8e-6 | 9.5e-6 |
| K_T, m2/s | 1.8e-5 (1.84, 7,943) | 2.0e-5 | 1.5e-5 |

N2 <= 0 in 18.3% of bins: the temperature sort (#219) no longer forces it
positive, and salinity scrambles the density order in 7 of 11 files (run
notice). Gamma and K_rho are masked where N2 < N2_min.

**The legs disagree, in opposite directions, and it is unexplained.** In the
same 1-m bins (median over dive profiles / median over climb profiles, then
the median over 522-871 bins): epsilon 0.69, chi **2.2**, Gamma **3.2**, while
N2 is 1.01. The same pattern, stronger, is on 2015-11-13 (that README). A speed
error moves epsilon and chi the same way, so it cannot do it alone. The 50% FM
failure rate on climbs is also unexplained; vibration on the faster climbs is
one hypothesis, untested. Treat absolute epsilon as uncalibrated in any case:
no sheet exists in `microstructure_sensors/` for M1369 or M887.

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
so in all 11 files. The thermistors are healthy; on the 2011 nominal
coefficients they read a near-constant **+8.8 C (T1) and +9.8 C (T2) high**.
Fitted, the median T - CTD per profile is within 15 mK on 15 of 19 profiles.
On the other four, all partial climbs from depth (ending at 360-470 m), the
FP07 reads 0.10-0.18 C warmer than the CTD, and the nominal set shows the same
excess on the same segments, so it is in the data, not the fit (the cause,
perhaps a leg-dependent lag, is untested). `fp07-cal` has not been run
on this deployment; on 2015-11-13 (same MR, probes, glider and CTD) the
per-file fits matched it to a gradient gain ratio of 0.9999.

### The 2026-10-13 set: two calibrations

That set fitted the thermistors in only 1 of 11 files (1504029: 3 of 19
profiles) and left the other 10 (16 profiles) on nominal coefficients ("no
profile reached the lag correlation gate"). The causes, found on 2015-11-13
and fixed in PR #214: a gap in the reference silenced the whole profile, and
the low-pass started from zero.

**CORRECTED 2026-10-17.** This README said chi on the nominal profiles was
~3-4% high. Their gradient gain against the new fit is indeed 0.978-0.992 (on
14 of the 16; 1.01-1.07 on the two 1504022 profiles, which span only 1 C), but
the legacy-`beta` bug (#215: the noise model on beta = 3000, not this config's
3143.55) pulled chi down by about as much, and per profile the two cancel
(median 0.998 T1, 1.002 T2). The one fitted file was the worse one: 1504029's
old fit read 0.15-0.57 C off the CTD on its full-depth profiles, with
gradients 10-17% too shallow, and its chi was 1.4-1.6x too low.

## What changed from the 2026-10-13 set

Old generation `_01` against new `_00`, the same 19 profiles, 1-m bins.
- **epsilon: bit-identical** (8,928 bins).
- **chi: paired median ratio 1.011** (chiMean, 8,281 bins). The 16 formerly
  nominal profiles: per profile 0.981-1.275 (median 0.998 T1, 1.002 T2). The 3
  formerly fitted (1504029): 1.02-1.75, median 1.41 (T1) and 1.52 (T2).
  Attributed by a one-file bisect on 1504023 and 1504029: commit b1270b5
  reproduces the old products bit-for-bit; #214 moves chi by x0.94-0.96 on
  1504023 and x1.29-1.60 on 1504029; #215 then adds +4 to +9%; #219 leaves chi
  alone.
- **N2, Gamma, K_rho, K_T** (#219, temperature sort): where N2 is positive in
  both sets it is unchanged (paired ratio 1.000), but 18.3% of bins now have
  N2 <= 0 (0.2% before), and Gamma and K_rho are masked there. Medians: N2
  2.3e-5 -> 3.3e-5, Gamma 0.22 -> 0.17, K_rho 1.5e-5 -> 9.7e-6, K_T
  2.4e-5 -> 1.8e-5. In the same depth bins, N2 dive/climb went from 1.58 to
  1.01.
- chi1/chi2 (median over profiles of each profile's median log ratio): 1.01
  -> 1.02 (MAD 0.13 dex, 19 profiles). This README previously quoted 0.86, by a
  method not recorded.

## Recipe (run from this directory)

1. `dinkum-hotel build -c dinkum-flight.yaml` -> `flight_inputs.nc`. Sea-Bird
   CTD on `sci_ctd41cp_timestamp`; buoyancy `m_de_oil_vol`; the cache is a copy
   of `../../cache` (Pat's 2016 sensor-list cache).
2. `python mr_clock.py extract` then `solve` -> `mr_clock_model.json`.
   glider_time = MR_time + L, L = +1.12 s, drift +0.00 s/day,
   6 of 11 files (1504023, 1504025, 1504027, 1504029, 1504030, 1504031).
   **One-leg files are refused**: the short files hold a single dive or climb,
   a near-monotonic ramp on which the fitted pressure offset absorbs any time
   shift. Here the effect was smaller: before the gate the four one-leg files
   gave -2.5..+0.25 s against -0.25..+2.0 s from the full yos (the -9.0 and
   -6.25 s outliers that motivated the gate were on 2015-11-13).
3. `uv run make_flight_speed.py calibrate` then `run` -> `unit211_flight_hotel.nc`
   (already on the MR clock). gliderflight 1.2.1, default lift/Cd1, mass 60 kg
   nominal; global steady fit Cd0 0.233, Vg 58.71 L, compressibility
   6.70e-10 Pa-1; per-day dynamic Cd0 0.235, 0.227.
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
