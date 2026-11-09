# CASPER — processing recipes

Recipe only: configs, scripts and the small artifacts that record what was
fitted. **No data.** The trees themselves live on SeaChest at
`/Volumes/SeaChest/CASPER/`, and the reorganization record is
`/Volumes/SeaChest/REORG_PLAN_CASPER.md`.

Two campaigns, one instrument: Rockland `MR_1000_LP` **SN 134** on Slocum
glider `doug`, both times.

| | |
|---|---|
| `2015_East/gliders/doug/` | **CASPER-East**, R/V *Atlantic Explorer*, shelf off Duck NC, Oct–Nov 2015. 50 m water. |
| `2017_West/gliders/doug/` | **CASPER-West**, R/V *Sally Ride* SR1715, off Point Mugu CA, Sep–Oct 2017. |
| `2017_West/vmp/` | VMP-250IR SN 194, the same cruise as CASPER-West |

All three were reprocessed on main 98b4206 on 2026-10-17. CASPER-East doug was
rerun on 2aa9caf on 2026-11-03 (#257), and CASPER-West doug and the VMP on
8748b90 on 2026-11-05. All three were rebuilt on main 92938ae on 2026-11-08
(`git v0.4.0-66-g92938ae9b492`, stamped in every product) for #278: the
per-profile files now store the fast FP07 channels as float64, not float32.
Each glider README says what changed. The VMP's
SeaChest config had been truncated on 2026-09-21 and is restored from
`2017_West/vmp/perturb.yaml` (see its header).

The VMP (2,086 profiles), 8748b90 against 98b4206: ε, N² and dT/dz
bit-identical (115,534 windows); χ ×0.91 (paired per-window median of
`chiMean`, 5–95% 0.59–1.00), the combined `chiMean` median ×0.86
(1.06e-8 → 9.1e-9 K²/s), Γ 0.117 → 0.102. It is #263 (χ no longer uses FFT
bin 1): a rerun with `chi.first_bin: 1` restores the old χ to within ±1%.

The VMP, 92938ae against 8748b90, paired over every window: ε, N², dT/dz
and K_ρ identical; `chiMean` 5–95% 0.9999–1.0001, 0.42% of windows move by
more than 1%. The extremes, ×0.51 and ×2.03, lie outside either probe's own
range (×0.59–×1.32), so they are QC selection flips (`chiMean` changed which
probes it combines). The combined `chiMean` median is unchanged (9.1e-9
K²/s), and so is Γ (0.102).

The recipe is the one proven on this same glider and MicroRider at
`Taiwan/Taiwan17/gliders/doug/`; read that README and Taiwan13's husker
`flight/README.md` for the flight model itself.

```
dinkum-hotel build -c dinkum-flight.yaml     # dbd+ebd  -> flight_inputs.nc
python mr_clock.py extract                   # .P       -> mr1hz/*.npz (resumable)
python mr_clock.py solve                     # -> mr_clock_offsets.csv + mr_clock_model.json
uv run make_flight_speed.py calibrate        # -> calibration.json
uv run make_flight_speed.py run              # -> doug_flight_hotel.nc   (on the MR clock)
python make_gps.py                           # -> gps.nc                 (on the MR clock)
python thermistor_health.py <mr dir> out.csv # per-file FP07 health
perturb run -c perturb.yaml -j 8             # -> Processed/
```

## What differs from the Taiwan copies, and why

- **`mr_clock.py` takes a LIST of block splits**, not one. Both CASPER
  deployments stepped: 2017 once (−12.50 s), 2015 twice (+73.50 s, −12.75 s).
  One line through a step turns it into a fictitious drift — 3.34 s residual
  RMS in 2017, 16.5 s in 2015, against 0.29–0.32 s when split. Each block now
  carries its own `sel_lo_epoch`/`sel_hi_epoch`; the consumers refuse a model
  written by the old single-`split_epoch` code rather than guess.
- **`MIN_YO_DBAR`** is an explicit constant. It was hard-coded at 40 dbar,
  which is right for Taiwan's 200 m yos and discards 115 of 126 good files on
  the 50 m Duck shelf, where every long file has a 25.6–43.6 dbar yo. 2015
  uses 20; 2017 keeps 40.
- **2015 does not fit hull compressibility** (`EPSILON_FIXED`). In 50 m of
  water it is not identifiable: fitting it returns a *negative* value and
  drags Cd₀ to a 74 % day-to-day spread. Held at doug's own CASPER-West value
  (8.219e-10 Pa⁻¹, measured over 0–195 m), the spread falls to 14 %. The
  rejected fit is kept as `calibration.fitted-epsilon-REJECTED.json`.
- **2017 reads `corrected_headers/`.** The sibling `uncorrected_headers/` holds
  the same 188 casts with the stale 2015 config (sh1 M1344 @ 0.0655 vs the
  patched M1194 @ 0.0793); since ε ∝ sens⁻², reading those puts sh1 47 % high.
