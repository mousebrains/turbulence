# Taiwan 2013 — MicroRider MR046 on Slocum husker, and the Aquadopp on jane

The processing recipe for the 2013 glider microstructure from RR1306 leg 3,
South China Sea. **No data lives here.** The data, and the products these files
made, are on SeaChest at `/Volumes/SeaChest/Taiwan/Taiwan13/`. This tree mirrors
that one path for path, so every relative path in these configs (`..`,
`<CONFIG_DIR>/../../gliders/…`) resolves the same way when laid over the data.

Every other file here is a byte-for-byte copy of its SeaChest counterpart as of
2026-09-16. SeaChest is the working copy; this is the record.

## Layout

```
gliders/20130526_husker/
  hotel/     dinkum-hotel.yaml, make_corrected_track.py
             -> hotel.nc (CTD clock) and the drift-corrected track used for positions
  flight/    dinkum-flight.yaml, mr_clock.py, make_flight_speed.py
             -> husker_flight_hotel.nc: through-water speed and angle of attack,
                already on the MicroRider's clock. calibration.json,
                mr_clock_model.json and mr_clock_offsets.csv are the fitted results
                (small, kept so `run` can be repeated without re-calibrating).
  MR046/     perturb.yaml, sections.yaml -> Processed/ (epsilon, chi, 495 profiles)
aquadopp/analysis/
             read_prf.py, beams.py, aoa.py, compare.py, make_flight_speed_jane.py
             -> the independent check on the flight model's angle of attack
```

Run order, from the data tree: `hotel/` → `flight/` → `MR046/`. Each directory's
README has the commands, what was checked, and what went wrong.

## Read before using the products

- **Speed is modeled, not measured.** MR046 had no EM flowmeter; U comes from
  Lucas Merckelbach's `gliderflight` 1.2.1 dynamic model, with drag and volume
  calibrated per UTC day. ε ∝ U⁻⁴.
- **The angle of attack is probably ~1.6° too small on dives.** The Aquadopp on
  the sister glider jane measured it directly: U ~5–6% high, dive ε ~20–25% low,
  climbs agree to 0.3%. **Not applied, by decision (2026-09-15).** Write-up:
  [`docs/aquadopp_flight_model_validation.html`](../../docs/aquadopp_flight_model_validation.html).
- **The shear probes disagree by 4–6×**, and which one is right is not
  established. **T1 fails from file FSB_020 on**, so χ from that thermistor is
  invalid for 325 of 495 profiles. Both faults are in `MR046/README.md`.
- **Reading MR046 needs `main` at or after 29f3ea9** (PR #196). Its config gives
  each channel its own named stanza, and v0.4.0 and earlier build zero channels
  from it.
- `make_flight_speed*.py` are `uv run` scripts with pinned dependencies in their
  PEP 723 headers. They are per-deployment scripts on purpose, not package code.
