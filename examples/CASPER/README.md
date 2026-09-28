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
