# MicroRider SN 058 on Slocum unit211 -- 20151119_unit211

One of T. Tanaka's pre-paper glider deployments: 32.3 N 133.2 E, off Shikoku, 2015-11-19 04:50 .. 11-20 04:51
UTC, dives to ~1000 m, 11 MicroRider files. Processed 2026-10-13 with
microstructure-tpw (branch `feat/rsijac-ct`, main 0f65101 plus the JAC-CT converters,
which these files do not use).

## Verdict

| | status |
|---|---|
| epsilon | **sh2 only** -- sh1 is dead (below). Climbs usable; dives marginal (slow) |
| chi | both thermistors, NOMINAL calibration (~3-4% high, below), consistent (chi1/chi2 = 0.86 paired) |
| clock | solved, 6 files, 0.89 s RMS |
| speed | flight model with DEFAULT lift and Cd1 -- untested for this glider |

## Products -- `Processed/`, generation **_01** (generation _00 averaged the dead probe in)

19 (7 dives, 12 climbs) profiles, 1-m bins.

- dives: U 0.148 m/s, epsilon 7.9e-10 W/kg (sh2, MAD 0.48 dex), FM > 1.15 in 20%
- climbs: U 0.364 m/s, epsilon 1.5e-9 W/kg (MAD 0.32 dex), FM > 1.15 in 54%
- chi: 3.0e-9 K2/s (MAD 0.44 dex); chi1/chi2 = 0.86 paired
- Gamma 0.22 (MAD 0.55 dex), K_rho 1.5e-5 m2/s, N2 2.3e-5 s-2

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

## FP07: the in-pipeline calibration did not run, and that costs ~3-4% in chi

`fp07.calibrate: true` was refused on most files ("no profile reached the lag
correlation gate"), so chi uses the 2011 NOMINAL Steinhart coefficients.
The thermistors themselves are healthy -- r = 0.9999 with the glider CTD --
but read a near-constant **+8.8 C (T1) and +9.8 C (T2) high**.

Why the gate failed (VERIFIED by instrumenting `_calc_lag` on 1504003):
- the long dive profile's reference holds 64 s of NaN where the hotel merge
  blanked a gap wider than `max_gap`, and one NaN makes `_calc_lag` abstain
  for the WHOLE profile (counted as neither accepted nor rejected);
- the other profiles score |r| 0.11-0.63 on the derivative cross-correlation
  against a 4.6 s glider CTD interpolated to 64 Hz, with two lags on the
  +-10 s search boundary.

What it costs (VERIFIED, 1 Hz regression of CTD on nominal FP07 over
5-21 C, best lag, six files): dT_ctd/dT_nominal = 0.980-0.987, so chi is
**3-4% HIGH** (chi scales as the square). An offset alone leaves gradients
untouched. (An earlier estimate in this session of 10-15% was wrong.)

The fix is the pre-pipeline `fp07-cal` tool, which decimates the thermistor
onto the CTD's own samples and gates lag on peak sharpness rather than r --
not done here.

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
