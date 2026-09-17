# Taiwan 2017 — MicroRider SN 134 on Slocum doug, and the ship's VMP SN 142

The processing recipe for the microstructure from R/V Roger Revelle RR1704
(project SK-II), Luzon Strait, February 2017: the MicroRider on glider doug, and
the ship's VMP SN 142 with a check of its thermistors against the ship CTD. **No data lives here.** The
data, and the products these files made, are on SeaChest at
`/Volumes/SeaChest/Taiwan/Taiwan17/`. This tree mirrors that one path for path,
so the relative paths in these configs (`../../20170216_doug_Taiwan`,
`<CONFIG_DIR>/flight/…`, `<CONFIG_DIR>/../GPS/gps.nc`) resolve the same way when
laid over the data.

Every other file here is a byte-for-byte copy of its SeaChest counterpart as of
2026-09-17. SeaChest is the working copy; this is the record.

## Layout

```
gliders/doug/
  flight/    dinkum-flight.yaml -> flight_inputs.nc (Slocum binaries, flight clock)
             mr_clock.py        -> MicroRider-vs-glider clock
             make_flight_speed.py -> doug_flight_hotel.nc: speed and angle of attack,
                                   already on the MicroRider's clock
             make_gps.py        -> gps.nc: drift-corrected dead-reckoned track, MR clock
             calibration.json, mr_clock_model.json, mr_clock_offsets.csv are the
             fitted results (small, kept so `run` can be repeated without
             re-calibrating); calibrate.log and extract.log are the run logs.
  perturb.yaml, sections.yaml -> Processed/ (epsilon, chi, 530 profiles)
vmp/
  perturb.yaml, sections.yaml -> Processed/ (epsilon, chi, 100 profiles from the
             12 profiling casts in raw/casts/; README says what every raw file is)
  GPS/       make_gps.py -> gps.nc: ship track from the met stream's epoch clock
             (reads met_nav_export.mat, a MATLAB v7 export of ZD/LA/LO from the
             v7.3 ship_data/met/met.mat, which scipy cannot read)
  rosette_crosscheck/
             DAT_166, strapped to the CTD rosette, against the ship's SBE 911plus:
             perturb_rosette.yaml, crosscheck.py, gain.py, gain_invariance.py, and
             the *.out results. Not a product.
```

Run order, from the data tree:

- **doug:** `flight/` (build, clock, calibrate, run, gps), then `perturb run` in
  `gliders/doug/`.
- **VMP:** `GPS/make_gps.py`, then `perturb run` in `vmp/`; the cross-check in
  `rosette_crosscheck/` is optional.

Each directory's README has the commands, and records what was checked and what
went wrong.

## Read before using the products

### doug

- **Speed is modeled, not measured.** doug's MicroRider had no EM flowmeter; U
  comes from Lucas Merckelbach's `gliderflight` 1.2.1 dynamic model, with drag
  and volume calibrated per UTC day. ε ∝ U⁻⁴. The method, and the `gliderflight`
  pitfalls, are written up once in
  [`Taiwan13/gliders/20130526_husker/flight/README.md`](../Taiwan13/gliders/20130526_husker/flight/README.md).
- **The model's angle of attack may be too small on dives.** An Aquadopp on
  Taiwan 2013's glider jane measured α ~1.6° larger on dives than the same model
  recipe gave for jane: U ~5–6% high, dive ε ~20–25% low. doug itself was never
  compared — different glider, four years later. **Not applied, by decision
  (2026-09-15).** Write-up:
  [`docs/aquadopp_flight_model_validation.html`](../../docs/aquadopp_flight_model_validation.html).
- **The MicroRider clock ran 9 min 21 s fast.** `doug_flight_hotel.nc` and
  `gps.nc` are both written on the MR clock, so `hotel.time_offset` is 0.
- **The shear probes disagree by ~3×**, and which one is right is not
  established. The thermistors agree (χ(T1)/χ(T2) = 0.982). Details in
  `gliders/doug/README.md`.
- **The probe serials in the config carry no weight**: the same config text sat
  unedited across 509 files from three cruises.
- This MicroRider's config uses modern `[channel]` stanzas and reads with any
  version, including v0.4.0.
- `make_flight_speed.py` is a `uv run` script with pinned dependencies in its
  PEP 723 header. It is a per-deployment script on purpose, not package code.

### VMP SN 142

- **Only 12 of the 18 raw files are processed.** The bottom-crasher casts
  (DAT_155–157), the bench runs (DAT_158, 161) and the rosette cast (DAT_166)
  are excluded on evidence from the data, not just the cast log. The glob
  `raw/casts/SN*/*.P` is the selection.
- **The shear-probe pair disagrees by ~24% in ε** (e₁/e₂ ≈ 0.76 wherever the
  signal is resolved), and which probe is wrong cannot be determined: there is
  no calibration sheet for either probe.
- **The FP07s carry Rockland's generic nominal coefficients**, so
  `fp07.calibrate` must stay true. They are calibrated in situ against JAC_T,
  per `.p` file, at order 2.
- **That temperature chain checks out against the ship's SBE 911plus**
  (`rosette_crosscheck/`): JAC_T within ~2 mK and the calibrated FP07s within
  ~1 mK in uniform water, with no gain error distinguishable from zero.
  Calibration moves χ by about 1% or less. JAC salinity reads ~0.014 PSU high.
  The check does **not** test the FP07 time response, which governs χ far more.
