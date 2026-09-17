# Taiwan 2017 (SK-II, RR1704) — VMP-250 SN 142

OSU's Rockland **VMP-250 SN 142** on R/V Roger Revelle cruise **RR1704**
(project SK-II), Luzon Strait / northeastern South China Sea, **22–26 Feb
2017**. 18 raw `.P` files; **12 are profiling casts → 100 profiles**, processed
with `microstructure-tpw` main `7736082` (v0.4.0) on 2026-09-15.

Reorganized 2026-09-15 — see `../README.md`, `../REORG_MANIFEST.csv` and
`/Volumes/SeaChest/REORG_PLAN_Taiwan17.md` for the evidence behind every
statement here.

## Directories

| | what it is |
|---|---|
| `raw/` | **The 18 originals, bytes untouched**, sorted by what the instrument was doing (below). `SETUP.CFG` is byte-identical to the config embedded in every file. |
| `raw/casts/SN142/` | 12 profiling-cast files ← **perturb reads only this** |
| `raw/bottom_crasher/SN142/` | DAT_155–157 — shear contaminated, **not processed** |
| `raw/bench/SN142/` | DAT_158, DAT_161 — deck/bench runs, **not processed** |
| `raw/ctd_rosette/SN142/` | DAT_166 — on the ship's CTD rosette, **not processed** |
| `rosette_crosscheck/` | DAT_166 against the ship's SBE 911plus: thermistor and JAC CT accuracy. Its own scratch `Processed/`; **not a product.** |
| `Processed/` | Current pipeline products: `profiles/diss/chi/ctd` (+ `_binned`, `_combo`, `combo`), generation `_00`. |
| `GPS/` | `gps.nc` — ship track, one-minute medians, from the met stream's unix-epoch clock (`make_gps.py` explains why not its datenum). |
| `legacy_2017/` | The at-sea MATLAB processing: `profiles/` (118 `.p` record cuts + `.p.mat`), `mat/` (ODAS `.mat` of the raw files, 4.9 GB), `gridded/`, `mfiles/`, `odas/` (the ODAS v4.01 library used), `VMPsumm.mat`, `TS_eps.jpg`. **Not an input to anything current.** |
| `process_kipp.m` | **The at-sea cast log.** Comments only; it is what says which file is which. |
| `perturb.yaml` | Processing config (`<CONFIG_DIR>`-relative). |
| `sections.yaml` | 3 plotting sections: two station clusters and the R2N line. |

## What each raw file is

The cast log's claims were each checked against the data.

| file(s) | cast log | measured |
|---|---|---|
| **DAT_155–157** (22 Feb) | *"all the data above here is garbage, because we had the bottom crasher on"* | **Confirmed for shear.** Deep ε floor (p10 below 30 dbar) 1.6–2.9e-8 W/kg in all 17 profiles vs 4–9e-10 in every later file; Lueck FM median 1.7–2.1 (sh1) and 4.6–4.8 (sh2) vs 0.59–0.66 later (ATOMIX rejects > 1.15); accelerometer std ~1.7× higher. The 2017 summary also blanked these casts. JAC T/S are presumably fine; FP07 is untested. |
| **DAT_158, DAT_161** | *"bench test"* | Pressure pinned at 0.0 dbar for 12 and 11 min. |
| **DAT_166** (26 Feb 03:35Z) | *"calibration test on ship's CTD"* | **Strapped to the rosette for ship CTD cast `RR1704_10`** (starts 03:38:27Z; the uCTD calibration cast starts 03:36:44Z — all three in `../uctd/calibration_casts/`). One slow cast to 250 dbar, accelerometer std 15–25× normal. Useless for turbulence. **Compared against the SBE 911plus on 2026-09-17** (`rosette_crosscheck/`): JAC_T +2.1 mK and in-situ-calibrated FP07 +1 mK in uniform water, no gain error distinguishable from zero (calibration moves χ by ≲1%); JAC salinity +0.014 PSU high. |
| **DAT_159, 160, 162–165, 167–172** | casts | 100 profiles, 2017-02-23 00:54Z – 02-26 20:27Z, 21.13–22.21 N, 119.02–119.81 E. |

## Instrument and configuration facts

- **Identity:** `vmp-250`, SN 142, 8×8 matrix (6 fast + 2 slow), 512.03 Hz,
  big-endian. The differential gains — sh1 0.96, sh2 0.97, T1_dT1 0.93,
  T2_dT2 0.95, P_dP 20.08 — match SN 142 in SUNRISE-2019 exactly. They are 2-dp
  **nominal** values; SN 142 received measured 3-dp values only in 2021
  (0.953/0.957 on shear, i.e. ~1% in ε).
- **Probes:** sh1 **M1000** 0.0716, sh2 **M1001** 0.0705, FP07 **T1000**/**T1001**
  with Rockland's generic nominal coefficients (β₁ 3143.55). **No calibration
  sheet exists for any of them**; the identity question is in
  `~/tpw/turbulence/microstructure_sensors/sheet_requests.md` (note 4). The
  absolute ε scale therefore rests on unverified sensitivities.
- **FP07 in-situ calibration is per `.p` FILE**, pooled over that file's
  profiles (median lag, one regression) — so changing which profiles a file
  yields changes every profile's calibration in that file, slightly.
- **`fp07.order: 2`**, measured: per-cast JAC_T span median 8.2 K; order 1
  leaves a systematic 23 mK curvature; order 2 cuts residual MAD 11.6 → 8.8 mK.
  χ differs by only 0.2% between the two.
- **`profiles.min_duration: 30`** removes three 7.5–10.9 s surface soaks that
  also fed the per-file FP07 fit; the shortest real cast is 135 s.

## Results (Processed/, 1-m bins)

| | median | spread (MAD) | n |
|---|---|---|---|
| ε (paired) | 4.8e-9 W/kg | 0.97 dex | 18,894 |
| χ | 1.5e-8 K²/s | 1.09 dex | 18,894 |
| N² | 1.0e-4 s⁻² | 0.27 dex | 18,894 |
| Γ | 0.33 | 0.54 dex | 13,881 |
| K_ρ | 7.4e-6 m²/s | 1.01 dex | 18,884 |

Water depth under the casts (ship multibeam, `../ship_data/met/met.mat` field
`MB`) is **1,500–3,000 m**, shoaling toward the NE end of the line; casts reach
~120–220 m, so **every cast ends ≥ 1,150 m above the seafloor** — no bottom
boundary layer and no bottom strikes. `bottom.enable` stays false.

## Known issues

1. **Shear-probe pair inconsistency, ~24% in ε.** e₁/e₂ plateaus at **~0.76**
   wherever the signal is resolved (0.76/0.74/0.77 from 1e-8.1 to 1e-6 W/kg;
   0.71–0.81 in 11 of 12 files) and approaches 1 only near the shared noise
   floor, so pooled/paired medians (0.82) understate it. That flat plateau is a
   multiplicative gain error of ~13% in sens × diff_gain. **Which probe is
   wrong cannot be determined from these data**; nominal diff_gain explains ~1%.
   `epsilonMean` (a geometric mean) carries ~half of it, sign unknown.
   **It also trends with depth:** within one ε decade the ratio falls 0.81
   (0–40 dbar) → 0.69 (160–250 dbar), opposite to what a floor would do.
   Pressure and temperature cannot be separated here.
   perturb flags 72 profiles for this (`Processed/diss_00/*.log`, not the run log).
2. **Single-cast excursions:** DAT_160 profile 1 (e₁/e₂ 1.80), DAT_165 profiles
   1–2 (1.46, 0.36), DAT_169 profile 6 (1.53), each measured below 30 dbar.
   Cause not established (debris on a probe is one possibility, unverified);
   not masked.
3. **FP07 pair:** χ₁/χ₂ pooled 0.92, moving toward 1 as K_max/k_B increases
   (0.85 → 0.96 by quartile) — a bead **response** asymmetry, not a gain error.
   **No `fp07_tau_scale` is applied, on purpose**: an inter-probe tau fit is
   degenerate in absolute tau, and the absolute χ level is unanchored.
4. **Elevated ε and χ below ~190 dbar at the NE end of the R2N line**
   (DAT_171–172, +0.4 to +1.0 dex over the 10–40 m above). Not a bottom effect
   (see above) and not a generic cast-end artifact (absent in DAT_163–168); it
   is present at 190–210 dbar in casts that continued ≥ 8 m deeper, so it looks
   like a real layer — but on only 45 windows. Not masked.
5. **Concurrent `perturb run`s on SeaChest fail** opening `GPS/gps.nc` (HDF5
   file lock over SMB). Run one at a time, or copy `gps.nc` locally.

## Reprocessing

```bash
cd /Volumes/SeaChest/Taiwan/Taiwan17/vmp
perturb run -c perturb.yaml -j 8        # ~75 s; writes Processed/{stage}_NN
perturb-plot overview --config perturb.yaml --sections sections.yaml --out-dir figs/
```

Generation numbers move whenever any `.py` under `src/odas_tpw/` changes (the
engine fingerprint), so never edit the package while a run is in flight.
