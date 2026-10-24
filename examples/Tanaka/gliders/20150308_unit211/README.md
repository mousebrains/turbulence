# MicroRider SN 058 on Slocum unit211 -- 20150308_unit211

One of T. Tanaka's pre-paper glider deployments, off Hokkaido in the Oyashio:
42.46-42.58 N, 144.96-145.21 E, 2015-03-08 10:05 .. 03-14 20:12 UTC. Dives to
about 1000 m, then nine 300-m dives on 03-14. The MicroRider files arrived on
2026-10-18 (see "Intake 2026-10-18" in the SeaChest `Tanaka/README.md`). Before
that this deployment was glider-only. **First processed 2026-10-19 (UTC) on
main 98b4206**
(`microstructure-tpw 0.4.0 (git v0.4.0-29-g98b42062069c)`, stamped in every
product).

## Verdict

| | status |
|---|---|
| epsilon | **both probes**: sh1 here is M881, not the dead M1369 of Nov 2015. e_1/e_2 = 0.81 |
| chi | FP07s fitted in situ per file in 34 of 37 files (88 of 95 profiles, all within 7 mK of the CTD). The other 3 files fall in a 15-h hole in the glider CTD record and keep factory coefficients (chi <= 1% high there) |
| clock | solved, **37 of 37 files**, 0.59 s RMS |
| speed | flight model, default lift and Cd1. **The pump mask differs from Nov 2015** (below) |
| legs | dives and climbs disagree in the same bins (epsilon 0.35, chi 2.5), as in Nov 2015 with the same MR. Unexplained |

## Products -- `Processed/`, generation **_00**

95 profiles (57 dives, 38 climbs) from 37 files, 1-m bins. Median, scaled MAD
of log10 (dex) and n over finite positive bins:

| | all | dives | climbs |
|---|---|---|---|
| U (m/s) | 0.238 | 0.193 | 0.384 |
| epsilon, W/kg | 6.6e-10 (0.60, 60,727) | 3.5e-10 (0.74, 30,355) | 8.7e-10 (0.41, 30,372) |
| chi, K2/s | 5.9e-9 (0.66, 41,988) | 7.6e-9 (0.45, 20,012) | 3.4e-9 (0.70, 21,976) |
| N2 (> 0), s-2 | 8.1e-6 (0.48, 45,602) | 8.1e-6 | 8.1e-6 |
| Gamma | 0.82 (0.56, 26,803) | 1.1 | 0.61 |
| K_rho, m2/s | 1.6e-5 (0.76, 49,313) | 9.8e-6 | 2.4e-5 |
| K_T, m2/s | 2.3e-4 (1.22, 37,680) | 3.7e-4 | 1.5e-4 |

N2 <= 0 in 16.0% of bins (temperature sort, #219). Salinity scrambles the
density order in 28 of 37 files (run notice). Gamma and K_rho are masked where
N2 < N2_min. This is cold (0.2-5 C), weakly stratified water, and Gamma is high
on both legs.

**The legs disagree, as in November 2015.** In the same 1-m bins (median over
dive profiles / median over climb profiles, then the median over ~1,000 bins):
epsilon 0.35, chi 2.50, Gamma 1.85, N2 1.00. The 2015-11-13 deployment, with
the same MR and glider, gave 0.47 / 4.5 / 5.4 / 0.86.
- A speed error cannot do it. epsilon scales roughly as U^-4 and chi as U^-2.
  To match the legs, the dive speed would have to be 0.77x the model for
  epsilon and 1.6x for chi.
- MR SN 156 on the same glider in 2022 shows no such split (epsilon 0.90,
  chi 1.00). So whatever it is goes with SN 058 or its 2015 installation.
  This is an observation, not a diagnosis.

Window QC is poor, on climbs especially.
- The Lueck FM exceeds 1.15 on 43% (sh1) / 47% (sh2) of dive windows and on
  56% / 71% of climb windows. The 2015-11 deployment had 49% on its climbs.
- chi is marginally resolved: K_max/kB medians are 0.73 / 0.55 on dives and
  0.97 / 0.80 on climbs.

## Instrument

MR1000-LP **SN 058**. Its config reads "File made for Yasuda MR1000-LP SN058,
2011-12-19", the same file as in Nov 2015 except for one probe.
- **sh1 is M881 @ 0.0772** (diff_gain 0.97); sh2 is M887 @ 0.0583.
- In water sh1 has a raw standard deviation of 87-134 counts and sh2 71-101.
  Nov 2015's dead M1369 had ~2. So both probes are used and nothing is
  excluded.
- e_1/e_2 = 0.81 binned (MAD 0.22 dex).
- No calibration sheet exists in `microstructure_sensors/` for M881 or M887.
- FP07s carry the 2011 nominal coefficients and no serial numbers. They read
  +8.77 C (T1) and +9.75 C (T2) high, the same offsets as in Nov 2015.
- T1 and T2 each appear twice in the address matrix (rows 4 and 8). PFile
  uses the first occurrence, as for Nov 2015.

## FP07: fitted per file, except in the CTD gap

`fp07.calibrate: true` with `fp07.allow_hotel_reference: true` fits both
thermistors per file against the glider's Sea-Bird CTD (`sci_water_temp`, on
`sci_ctd41cp_timestamp`). Fitted, the median T - CTD per profile is 1.3 mK (T1)
and 1.1 mK (T2), and at most 7 mK, on all 88 fitted profiles.
- **The glider CTD record has a 15-hour hole**, 03-09 19:25 -> 03-10 10:49.
  The flight computer logged 11,918 rows across it, and the science files
  hold only 648.
- MR files `1501041`, `1501042` and `1501043` lie inside it, 7 profiles. They
  have no reference, so the fit refuses them ("no profile segment reached the
  lag gate"). They keep factory coefficients, and they have no N2, Gamma, K_rho
  or K_T.
- Over neighboring fitted profiles, the gradient gain fitted/factory is
  0.997-0.999 (T1) and 0.994-0.996 (T2). So factory chi there is at most ~1%
  high.
- chi_1/chi_2 = 0.71 binned (MAD 0.16 dex).

**The CTD gate is -2.5 C, not 0.5 C.** Oyashio water here reaches 0.18 C, and
1% of the CTD samples are below 0.24 C. The 0.5 C floor of the November
recipes would have cut real water. No row with a valid `sci_ctd41cp_timestamp`
has an exact-zero T or C. The conductivity floor (2 S/m) still rejects 143
surface rows at ~0.04 dbar.

## Speed: the pump mask

`make_flight_speed.py` excludes samples where the buoyancy pump is working.
The November recipe tests this as |measured - commanded oil| > 10 cc. Here the
climb pump stalls short of its +260 cc command at a median of +242 cc. It runs
~200 s at ~2.7 cc/s, overshoots, settles in ~35 s, and then holds to 0.1 cc
(60-s change: 0.02 cc median, 0.1 cc 95th percentile). So the November test
excluded 66.5% of climbs, and **every** climb until 03-09 ~12Z.
- A dives-only day cannot separate Cd0 from Vg. 03-08 calibrated to **Cd0
  0.64**, against 0.23-0.25 on every other day.
- That put 03-08's speed 18% low on dives and 44% low on climbs. With epsilon
  roughly as U^-4, epsilon would have been ~2x and ~10x high.
- Here "pumping" is instead the measured oil volume **moving**: more than
  1 cc of change over a centered 60 s. That excludes 4.0% of climbs and 2.0%
  of dives. The model is fed the measured volume either way. The test compares
  the window's endpoints, so an excursion that returns within 60 s passes. A
  peak-to-peak test would exclude 2.2% of dives instead (climbs unchanged).
  That is negligible, so the products keep the endpoint test.
- In Nov 2015 and 2022 the pump reaches ~254 cc and the November test excludes
  <= 1.4% of climbs, so those recipes are unchanged.
- The rejected calibration and hotel are kept beside the products, renamed
  `*_pumptol10*.rejected`.

Result: gliderflight 1.2.1, default lift and Cd1, mass 60 kg nominal.
- Global steady fit: Cd0 0.239, Vg 58.58 L.
- Per-day dynamic Cd0: 0.231, 0.232, 0.238, 0.241, 0.242, 0.246, 0.237
  (03-08 .. 03-14).
- Hotel medians: dives U 0.189 m/s, AOA -4.5 deg, pitch -23.1 deg; climbs
  0.383 m/s, +3.4 deg, +28.7 deg.
- In the CTD gap the T and C feeding density are interpolated across 15 h.
  The model still tracks the depth rate there: w_water median +1.7 / +0.8
  mm/s on dives / climbs against 0.0 outside, and p90 |w_water| is 29 mm/s
  against 25.

## Clock

`mr_clock.py`:
- glider_time = MR_time + L, with L = +2.35 s at 2015-03-12 06:26Z and a
  drift of -0.46 s/day;
- **all 37 files** (each is a full yo), 0.59 s RMS;
- per-file lags run from +3.75 to -0.25 s;
- **the residuals are not white.** They run -0.8..+0.1 s on 03-08..03-10,
  rise to +0.2..+1.0 s on 03-11..03-13, and fall to -0.16..-1.47 s for the
  nine short 300-m files on 03-14.
- So the drift is curved, not linear, over six days. The linear model is kept;
  its worst misalignment is ~1.5 s, about 0.1-0.3 m of depth on these
  dives.

## Recipe (run from this directory)

1. `dinkum-hotel build -c dinkum-flight.yaml` -> `flight_inputs.nc`. The
   Sea-Bird CTD is on `sci_ctd41cp_timestamp` and buoyancy is
   `m_de_oil_vol`. The cache is a copy of `../../cache`, Pat's 2016
   sensor-list cache, which decodes every 2015-03 file.
2. `python mr_clock.py extract` then `solve` -> `mr_clock_model.json`.
3. `uv run make_flight_speed.py calibrate` then `run` -> `unit211_flight_hotel.nc`,
   already on the MR clock.
4. `perturb run -c perturb.yaml -j 4` (~10 min, run with
   `HDF5_USE_FILE_LOCKING=FALSE` on SeaChest).

## Untested, and worth doing

- The leg asymmetry, now seen in two SN 058 deployments and not in SN 156.
- Whether the default lift misfits unit211. No DVL was delivered for this
  mission.
- What the glider's science computer did between 03-09 19:25 and 03-10 10:49.
