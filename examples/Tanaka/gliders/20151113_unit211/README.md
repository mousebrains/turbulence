# MicroRider SN 058 on Slocum unit211 -- 20151113_unit211

One of T. Tanaka's pre-paper glider deployments: 29.4 N 127.7 E, East China Sea, 2015-11-13 15:41 .. 11-14 22:12
UTC, dives to ~1000 m, 12 MicroRider files. Processed 2026-10-13 with
microstructure-tpw (branch `feat/rsijac-ct`, main 0f65101 plus the JAC-CT converters,
which these files do not use).

## Verdict

| | status |
|---|---|
| epsilon | **sh2 only** -- sh1 is dead (below). Climbs usable; dives marginal (slow) |
| chi | both thermistors on TWO calibrations: perturb's per-file in-situ fit in 19 of 33 profiles (files 1504003, 005, 012), NOMINAL in the other 14 (~3-4% high, below); chi1/chi2 = 0.92 paired |
| clock | solved, 7 files, 0.87 s RMS |
| speed | flight model with DEFAULT lift and Cd1 -- untested for this glider |

## Products -- `Processed/`, generation **_01** (generation _00 averaged the dead probe in)

33 (20 dives, 13 climbs) profiles, 1-m bins.

- dives: U 0.123 m/s, epsilon 4.3e-10 W/kg (sh2, MAD 0.52 dex), FM > 1.15 in 20% of windows
- climbs: U 0.374 m/s, epsilon 1.45e-9 W/kg (MAD 0.43 dex), FM > 1.15 in 49%
- chi: 5.5e-9 K2/s (MAD 0.50 dex); chi1/chi2 = 0.92 paired
- Gamma 0.40 (MAD 0.53 dex), K_rho 1.7e-5 m2/s, N2 1.7e-5 s-2

The climb/dive epsilon ratio (2-3x in the same water) and the 50% FM failure
rate on climbs are **unexplained**; vibration on the faster climbs is one
hypothesis, untested. Treat absolute epsilon as uncalibrated in any case: no
sheet exists in `microstructure_sensors/` for M1369 or M887.

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

## FP07: two calibrations in the products, and ~3-4% in chi on one of them

`fp07.calibrate: true` fitted the thermistors per file in 3 of 12 files
(1504003, 1504005, 1504012: 19 of 33 profiles). The other 9 files (14
profiles) kept the 2011 NOMINAL Steinhart coefficients ("no profile reached
the lag correlation gate"). **CORRECTED 2026-10-13:** this README first said
the fit was refused on most files and that chi was nominal throughout. The
products' `calibration` tags say otherwise. The thermistors themselves are
healthy, r = 0.9999 with the glider CTD. On nominal coefficients they read a
near-constant **+8.8 C (T1) and +9.8 C (T2) high**.

Why the gate failed. VERIFIED by re-running this deployment with each fix on
branch `fix/fp07-cal-hotel-reference` (PR #214) reverted in turn:
- **A gap silenced whole profiles.** One NaN in a profile's reference (the
  hotel merge blanks gaps wider than `max_gap`) made `_calc_lag` abstain for
  the WHOLE profile, counted as neither accepted nor rejected. Splitting
  profiles at their gaps alone fits T1 in 11 of 12 files and T2 in 10.
- **The low-pass started from zero.** Its output climbed from 0 to the counts'
  level over seconds, and the files whose only profile starts at the top of
  the file (1504004, 1504010) scored that ramp. Starting the filter at rest
  fits them too.
- With both fixed, all 12 files fit. Each per-file fit matches the pooled
  `fp07-cal` fit below: gradient gain ratio median 0.9999 (range
  0.994-1.003), temperature median 3 mK (max 8 mK).
- The CTD's 4.5 s spacing was NOT the cause, as this README first guessed
  (|r| 0.11-0.63 "against a sparse CTD").

What it costs (VERIFIED, 1 Hz regression of CTD on nominal FP07 over
5-21 C, best lag, six files): dT_ctd/dT_nominal = 0.980-0.987, so chi on the
nominal profiles is **3-4% HIGH** (chi scales as the square). An offset alone
leaves gradients untouched. (An earlier estimate in this session of 10-15%
was wrong.)

The pre-pipeline `fp07-cal` tool decimates the thermistor onto the CTD's own
samples and gates lag on peak sharpness rather than r. **Its fit was run on
2026-10-13 (`fp07cal/`, FIT ONLY: nothing is patched and the products above
are unchanged):**
- all 12 files contribute pairs (16,587 used for T1), 33 profiles, and the CTD
  covers 87% of the record at a median 4.5 s;
- lag +4.7 s on both thermistors, peak width 0.8 s, r 0.94;
- order 2, chosen by held-out error (47 mK held out, 3.5 mK in sample for T1;
  2.6 mK for T2); dive/climb split -1.0 / -0.3 mK.

It confirms the 1 Hz regression independently. Against nominal, the gradient
gain is 0.986 (T1) and 0.983 (T2), ranging 0.977-0.992 over 4.5-24 C, and the
offsets are +8.9 / +9.9 C. So **chi on the nominal profiles is 2.9% (T1) and
3.5% (T2) high.** There are two ways to put the whole deployment on one
calibration:
- `fp07-cal patch`, then perturb with `fp07.calibrate: false`;
- perturb from the fixed branch, with `fp07.allow_hotel_reference: true`.

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
4. `PYTHONPATH=<repo>/src perturb run -c perturb.yaml`.

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
