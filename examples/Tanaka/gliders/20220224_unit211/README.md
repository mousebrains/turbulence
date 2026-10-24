# MicroRider SN 156 on Slocum unit211 -- 20220224_unit211

T. Tanaka's 2022 glider deployment, west of Okinawa: 25.60-25.77 N,
126.93-127.00 E, in water 2022-02-24 23:37 .. 02-27 22:48 UTC. Dives to about
1000 m, then four 300-m dives at the end. The data arrived on 2026-10-18 (see
"Intake 2026-10-18" in the SeaChest `Tanaka/README.md`). **First processed
2026-10-19 (UTC) on main 98b4206** (`microstructure-tpw 0.4.0 (git v0.4.0-29-g98b42062069c)`, stamped in
every product).

## Verdict

| | status |
|---|---|
| raw `.P` files | **mis-framed** in 28 of 29 files. Processed from re-framed copies (below) |
| epsilon | both probes alive, but **sh1 reads 0.40x sh2** where the signal dominates. A gain error of ~1.6x in one probe's sensitivity; which one is unresolved |
| chi | FP07s fitted in situ per file in 21 files (46 of 54 profiles, all within 3 mK of the CTD); factory coefficients in 4 short shallow files (8 profiles, chi within ~1.5% (T1) / ~4% low (T2)). **T2 is the slower bead**: chi_1/chi_2 = 1.5 |
| clock | solved, 20 of 29 files, 0.37 s RMS |
| speed | flight model, default lift and Cd1. Dives and climbs agree in the same bins (epsilon 0.90, chi 1.00) |

## Products -- `Processed/`, generation **_00**

54 profiles (29 dives, 25 climbs) from 25 files, 1-m bins. Median, scaled
MAD of log10 (dex) and n over finite positive bins:

| | all | dives | climbs |
|---|---|---|---|
| U (m/s) | 0.310 | 0.222 | 0.480 |
| epsilon, W/kg | 6.7e-10 (1.23, 28,187) | 6.4e-10 (1.11, 14,415) | 7.2e-10 (1.38, 13,772) |
| chi, K2/s | 2.3e-9 (1.21, 27,507) | 2.4e-9 (1.17, 13,879) | 2.3e-9 (1.26, 13,628) |
| N2 (> 0), s-2 | 2.1e-5 (0.61, 24,171) | 2.1e-5 | 2.1e-5 |
| Gamma | 0.39 (0.49, 23,322) | 0.43 | 0.36 |
| K_rho, m2/s | 6.7e-6 (1.33, 25,446) | 7.1e-6 | 6.2e-6 |
| K_T, m2/s | 3.4e-5 (1.57, 26,401) | 4.7e-5 | 2.5e-5 |

N2 <= 0 in 13.6% of bins (temperature sort, #219). Salinity scrambles the
density order in 17 of 25 files (run notice). Gamma and K_rho are masked where
N2 < N2_min.

**The legs agree.** In the same 1-m bins (median over dive profiles / median
over climb profiles, then the median over ~980 bins): epsilon 0.90, chi 1.00,
Gamma 1.22, N2 1.03. The 2015-11-13 deployment of the same glider (another MR)
disagrees by 0.47 / 4.5 / 5.4 on the same comparison, and that is unexplained
there.

Window QC: the Lueck FM exceeds 1.15 on 24% (sh1) / 27% (sh2) of dive windows
and 31% / 15% of climb windows. chi K_max/kB medians are 2.5 / 3.9 on dives and
1.4 / 1.6 on climbs; chiQCFallback is 3-4%.

Four files have no products: perturb refused their speed.
- `211_025` and `211_026` are 11-minute deck records from 02-22, two days
  before the glider's science record starts. That refusal is correct.
- `211_027` and `211_028` are the first 35-m dives. The glider sat at the
  surface for about half of each, where it logs no `m_pitch`, so the flight
  model has no speed there. Speed is 49.9% and 48.3% finite, just under
  perturb's 50% guard on the whole file. Each holds one complete 35-m dive
  with a finite speed, so a lower guard would recover them.

## THE RAW FILES ARE MIS-FRAMED -- `reframe_p.py`, `reframed/`

A v6 `.P` file is record 0 (header + config) followed by fixed-size data
records: a 128-byte header, then 64 address-matrix cycles of 64 words, each
beginning with the special character 0x7FF0. 28 of these 29 files hold **one**
irregular record, always in data records 4-17 (#74 in `211_025`):
- **33 matrices** instead of 64 (10 files): 31 matrices (0.48 s) missing;
- **98 matrices** and the next record number absent (10 files): two records'
  time in 98 matrices, so 30 matrices (0.47 s) missing;
- **67 matrices** and the next record number absent (6 files): 61 matrices
  (0.95 s) missing;
- `211_029` (70 matrices + 56 junk words) and `211_025` (93 + 41).

`211_034` is the only clean file. After the irregular record every file is
regular to its end: 64 matrices per record, consecutive record numbers, header
word 16 = 0. The 2015 SN 058 and 2017 files in this tree are all regular.

PFile, like ODAS `read_odas`, cuts the file at the fixed record size, so every
header after the damage is read as data and vice versa. It warned three times
(bad-buffer flags on ~90% of records, record-number "gaps", a non-integer
record count) and returned mis-assembled channels: `211_029`'s P ran from
-42.8 to -22.1 dbar, and the deep files lost ~15 dbar off their maximum.

`reframe_p.py` walks each file by header signature and matrix runs, drops the
data records up to and including the irregular one, and copies the rest
byte-for-byte to `reframed/microrider/`. Nothing is interpolated or padded.
Record 0's time is advanced by the dropped records' true duration with PFile's
own helper (`_advance_record0_timestamp`, the one `cutp` uses), so every kept
sample stays at its original nominal time.
- The dropped lead-in is 5-17 s at the surface after "acquisition start".
  `211_025`, a deck record, loses 75 s.
- The header times continue at one per second across the damage (`211_040`:
  #13 at 12:04:03, #15 at 12:04:05). So the sampling ran on, and only the
  write lost bytes.
- **Check:** the solved clock lags fall smoothly from +3.25 to -1.25 s over
  the three days (0.37 s RMS about a line). Leaving the correction out would
  scatter files by up to 17 s. A uniform offset would be absorbed into L, so
  the real constraint is the families. The clean `211_034` sits on the line
  (-0.01 s). Files advanced by n records (33 matrices: mean residual -0.13 s,
  7 files) and by n + 1 (98: +0.15 s, 8 files; 67: -0.09 s, 4 files) agree to
  0.3 s.
- **ODAS cross-check:** ODAS 4.5.1 `odas_p2mat` on the re-framed `211_031`
  gives P 1.1-107.4 dbar, as PFile does. On the original it flags 1,519 of
  1,551 records as bad buffers, then stops in `patch_odas` on the non-integer
  record count.
- `reframed/reframe_manifest.csv` lists each file's irregular record, first
  kept record, seconds dropped, old and new start time, and md5 in and out.
  The originals in `../post-recovery/microrider/` are untouched.

Every re-framed file loads with no warning.

## Instrument

MR1000-LP **SN 156**. The config reads "File made for MR1000-LP SN156,
2015-09-30" and has `[instrument_info] vehicle = slocum_glider`. It is not
SN 058, the MR unit211 carried in 2015.
- Probes: sh1 **M1368 @ 0.0634** (diff_gain 0.96), sh2 **M2256 @ 0.0636**
  (0.95). FP07 **T1023** and **T2166** with the nominal beta 3143.55.
- Every `cal_date` in the config is blank. No sheet for any of these probes
  is in `microstructure_sensors/`.
- M1368 is **not** SN 058's dead M1369.
- No EM flowmeter.

## epsilon: the probes disagree by 2.5x

Window by window, with both FM < 1.15, e_1/e_2 depends on the level:

| sqrt(e_1 e_2), W/kg | < 1e-10 | 1e-10..3e-10 | 3e-10..1e-9 | 1e-9..3e-9 | 3e-9..1e-8 | > 1e-8 |
|---|---|---|---|---|---|---|
| e_1/e_2 | 0.88 | 0.84 | 0.62 | 0.50 | 0.41 | 0.39 |

The ratio is near 1 at the noise floor and settles at **0.39-0.41 where the
signal dominates**. A flat ratio there is the signature of a gain error, i.e.
the configured sensitivity of one probe, not noise, vibration or a damaged
probe's spectral shape. The binned products carry e_1 and e_2 separately, and
epsilonMean is their geometric mean.

**Which probe is right is unresolved.** epsilon from the Batchelor fits
(`epsilon_T`) cannot arbitrate. Against shear epsilon it ranges 0.09-1.76
across the Tanaka deployments, with 0.4-0.9 dex of scatter, and here T1's and
T2's own epsilon_T disagree by 2.6x. A calibration sheet for M1368 or M2256
would settle it.

## FP07: fitted per file, and T2 is slower

`fp07.calibrate: true` with `fp07.allow_hotel_reference: true` fits both
thermistors per file against the glider's Sea-Bird CTD (`sci_water_temp`, on
`sci_ctd41cp_timestamp`). It fitted 21 files. On nominal coefficients T1 reads
+5.24 C and T2 -1.6..-1.9 C off the CTD. Fitted, the median T - CTD per profile
is 0.7 mK, and at most 3 mK, on all 46 fitted profiles.

**Factory coefficients stay in 4 files**: `211_029`, `211_031`, `211_043`,
`211_044` (8 profiles, all shallower than 115 m). Their reason is "no profile
segment reached the lag gate". Over the fitted full-depth profiles the gradient
gain fitted/factory is 0.993-1.000 for T1 and 1.019-1.025 for T2. So chi on
those 8 profiles is within ~1.5% (T1) and about 4% low (T2); chi scales as the
gain squared.

**chi_1/chi_2 = 1.5** (binned, MAD 0.15 dex), and it rises with chi: 1.18 near
the floor, 1.83 above 1e-8. The ratio of the two gradient spectra, where both
are > 10x their noise model, says why:

| f (Hz) | 0.5-1 | 1-1.5 | 2-3 | 4-5 | 6-8 | 10-15 | 20-30 |
|---|---|---|---|---|---|---|---|
| S_T1/S_T2 | 1.01 | 1.04 | 1.18 | 1.40 | 1.63 | 1.89 | 1.96 |

The ratio is 1 at low frequency, where the CTD fit fixed both gains, then rises
and plateaus near 2. **T2166 is the slower bead**: as a single pole, its time
constant is ~1.4x T1023's. Both get the same tau model, so chi_2 is
under-corrected against chi_1. The absolute response of either bead is not
identifiable from these data. `fp07-pair fit` refuses the pair: on this slow
glider its usable band ends at 5.6 Hz, below its 30 Hz floor. This was a
diagnostic run; nothing was applied.

## Clock and speed

- **Clock** (`mr_clock.py`, on the re-framed files):
  - glider_time = MR_time + L, with L = +0.74 s at 2022-02-26 16:48Z and a
    drift of -1.61 s/day;
  - 20 of 29 files, 0.37 s RMS;
  - the other 9 have less than 30 min in water.
- **Speed** (`make_flight_speed.py`): gliderflight 1.2.1, default lift and
  Cd1, mass 60 kg nominal.
  - Global steady fit: Cd0 0.154, Vg 58.70 L.
  - Per-day dynamic Cd0: 0.157, 0.156, 0.148 (02-25 .. 02-27). 02-24 had 97
    samples and was skipped.
  - Hotel medians: dives U 0.182 m/s, AOA -2.7 deg, pitch -23.7 deg; climbs
    0.481 m/s, +2.4 deg, +26.6 deg.
  - Cd0 is well below the 0.22-0.24 this glider fitted in 2015. The cause is
    unknown; a different hull configuration is one possibility.
- **Positions** are the dead-reckoned `m_lat`/`m_lon` from the hotel file.
  There are no exact-zero position rows here; see
  `../../20151113_unit211/analysis/README.md`.
  - `flight_inputs.nc` holds 263 pre-deployment rows at about 31.58 N
    130.56 E (02-22 05:24 -> 02-24 23:36).
  - The hotel's first ~43 s therefore interpolate from 31.6 to 25.6 N.
  - No product is affected: the first profile starts 02-25 00:32, and product
    latitudes span 25.60-25.77 N.

## Recipe (run from this directory)

1. `dinkum-hotel build -c dinkum-flight.yaml` -> `flight_inputs.nc`. The
   Sea-Bird CTD is on `sci_ctd41cp_timestamp` and buoyancy is
   `m_de_oil_vol`. The cache is a copy of `../post-recovery/glider/cache`,
   delivered with the data.
2. `python reframe_p.py` -> `reframed/microrider/*.P` + `reframed/reframe_manifest.csv`.
3. `python mr_clock.py extract` then `solve` -> `mr_clock_model.json`. It
   reads the re-framed copies.
4. `uv run make_flight_speed.py calibrate` then `run` -> `unit211_flight_hotel.nc`,
   already on the MR clock.
5. `perturb run -c perturb.yaml -j 4` (~7 min, run with
   `HDF5_USE_FILE_LOCKING=FALSE` on SeaChest).

## Untested, and worth doing

- A calibration sheet, or Rockland's record, for M1368 and M2256. It is the
  only way to settle the 2.5x.
- Whether the default lift misfits unit211. The DVL's 27 PD0 files arrived
  after this run (2026-10-20, `../post-recovery/glider/vb*.pd0`). They could
  test it, as the paper's DVL did on unit095. Not done yet.
- PFile could detect mis-framing (a header signature at each record
  boundary) and refuse or resynchronize, instead of warning and mis-assembling.
  ODAS `read_odas` would mis-assemble these files the same way: it reads the
  data as one block and reshapes it at the fixed record size (`read_odas.m`
  lines 182-193, 314).
