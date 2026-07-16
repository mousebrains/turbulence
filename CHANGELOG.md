# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed
- **Flight-model glide angle now ADDS the angle of attack** (issue #131 M7).
  `speed.method: "flight"` computed `U = |W| / sin(|pitch| − aoa)`; steady-glide
  force balance (Merckelbach et al. 2010) and the ODAS reference
  (`odas_p2mat.m`: `glide_angle = abs(Incl_Y) + aoa`) both make the glide path
  STEEPER than the pitch attitude, `U = |W| / sin(|pitch| + aoa)`. The
  subtraction inflated U by sin(|pitch|+aoa)/sin(|pitch|−aoa) — 1.24× at a
  typical 26° Slocum pitch — and, through epsilon's ~U⁻⁴ speed leverage, biased
  flight-method epsilon ~2.4× LOW (up to ~5× at 15° pitch; ~1.55× on steep 44°
  SL685 climbs). On `MR/AIOP2_SL685_0450.p` the corrected flight speed
  (0.397 m/s) now sits within 10% of the independent `U_EM` flowmeter
  (0.428 m/s). Only `speed.method: "flight"` output is affected; `pressure`,
  `em`, and `constant` are unchanged. A new cross-check warns whenever the
  flight speed and a present `U_EM` channel disagree by more than 20% (median),
  since a bad `aoa_deg`, a mis-picked pitch axis, or a stale EM calibration all
  leak into epsilon the same way. `min_pitch_deg` now gates on the pitch
  attitude itself (`|pitch| < min_pitch_deg` → NaN), not the aoa-shifted path
  angle — the steady-glide model is invalid near dive/climb inflections
  regardless of aoa; the effective default gate moves from |pitch| > 8° to
  |pitch| ≥ 5°. The path angle is clamped at 90° (railed inclinometers).
  Stale `sin(|pitch|−aoa)·cos|roll|` formulas in the perturb template, the
  ARCTERX example config, `--speed-method` help, and
  docs/perturb/configuration.md were corrected to match the code.

### Added
- **Real-glider MicroRider end-to-end fixture + test** (issue #131 m11) —
  `tests/data/MR_SL685_climb.p`, a 150 s / 1.4 MB climb segment (481 → 429 dbar
  at ~0.35 dbar/s, EM speed ~0.5 m/s) cut at record boundaries from
  `MR/AIOP2_SL685_0450.p` (MR1000RDL-EM SN 435 on Slocum osu685, ARCTERX IOP2),
  and `tests/test_mr_e2e.py`: the first test to run the batch machinery
  (PFile → vehicle resolution → glide profile detection → ε → χ Method 2)
  against real MicroRider-on-glider data instead of mocks. Direction, `W_min`,
  and speed (median `|U_EM|`) are passed explicitly so the test is independent
  of vehicle-resolved defaults.
- Docs drift fixes (issue #131 m13): `docs/perturb/cli.md` documents
  `perturb run --force`; `docs/rsi-tpw/cli.md` adds the missing `sensors` row
  to the subcommand table and a `rsi-tpw ml` reference section.
- **Cross-probe consistency diagnostics** (#131) — with two or more shear
  probes (or FP07s), every per-profile epsilon/chi dataset now carries
  per-pair global attrs `probe_ratio_pairs` / `probe_ratio_median` /
  `n_ratio_windows` / `probe_ratio_z` (`chi_`-prefixed on the chi product):
  the median first/second-probe ratio over the windows where both are finite,
  plus the significance z of the median ln-ratio given the Lueck (2022)
  per-window `sigma_ln`. A two-tier `logging` warning flags persistent
  inter-probe disagreement — statistical (z > 3, ≥ 20 windows) and practical
  (median ratio beyond 1.8× either way, ≥ 10 windows) — because per-window QC
  (fom/FM) cannot see a persistent systematic offset (vmp142's pairs disagreed
  1.8×/2.2× with clean per-window QC; CAS_080.P has sh1 ≈ 1000× sh2 with
  fom ≈ 1). Observational only, nothing is auto-dropped; the attrs do **not**
  survive depth binning / combining (attrs are rebuilt from a schema there) —
  read them from the per-profile files. Shared helper in
  `processing/probe_consistency.py`; see docs/rsi-tpw/pipeline.md.
- **Calibration staleness guard** for `rsi-tpw sensors --cal-dir` (#131 m2) —
  each sheet's "Recommended re-calibration" date is parsed onto its
  calibration point, and observations governed by a calibration past that
  date are annotated `[cal N months old at use; recal was recommended by
  YYYY-MM-DD — verify no newer sheet exists]`. Sheets without the line fall
  back to `--cal-max-age-months` (default **12**, Rockland's actual
  recommendation — the flag only sets the fallback). The mismatch summary
  reports the stale count even when no sensitivities mismatch. The check is
  only as good as the sheets directory (completeness assumption documented in
  docs/rsi-tpw/sensors.md).
- **Strict exit codes** (#131 m3) — `rsi-tpw bench` now exits **1** when any
  file failed to evaluate (per-file errors still don't abort the batch;
  checklist FAIL/REVIEW verdicts don't affect the code), and
  `rsi-tpw sensors --cal-strict` exits **3** (distinct from the scan-failure
  code 1) when the `--cal-dir` check finds sensitivity mismatches outside
  `--cal-tol`; default behavior stays report-only. `--cal-strict` without
  `--cal-dir` is an error. Both flags are available on `rsi-tpw sensors` and
  `python -m odas_tpw.rsi.sensor_inventory`.
- **Old-format (2013-2017 CASPER-era) MicroRider config dialects** (#131 m1,
  m5, m6, m7) — pre-2017 configs now inventory and convert faithfully:
  `serial_num` is honored wherever the instrument SN is read (`summary`,
  NetCDF `instrument_sn`, epsilon/chi metadata, `rsi-tpw sensors` platform SN);
  `[cruise info]` — and any unknown section — is kept by `parse_config`
  (section names normalized: lower-cased, internal whitespace folded to `_`;
  `config-patch`/`patch-template` agree on the normalized name); channels
  declared `accel` with `coef0 = 0` / `coef1 = 1` are rewritten to `piezo`
  during config parsing (setupstr.m parity, exact-string trigger), so
  CASPER Ax/Ay convert as piezo counts and route to the `VIB` role instead of
  a fake 9.81·counts "m/s²" ACC; `id_even`/`id_odd` channel sections with no
  `id` synthesize the 2-id (32-bit) join; and a matrix address with no usable
  `[channel]` section now warns (read_odas.m parity; special address 255
  exempt). Note one deliberate side effect: the adapter's vibration routing is
  now type-based, so modern files whose Ax/Ay are declared `type = piezo`
  (e.g. SN479) report `vib_type = "VIB"` instead of `"ACC"` — harmonizing with
  `p_to_netcdf`'s classification. On mixed hardware the vibration stack is the
  name-sorted union of true accelerometers and piezo channels (labeled `ACC`),
  so Goodman noise removal keeps every coherent reference; the label is
  descriptive only and epsilon/chi values are unchanged.
- **Hotel-telemetry speed method** (issue #131 finding M10). New perturb
  `speed.method: "hotel"` (+ `speed.hotel_var`, default `"speed"`) consumes a
  hotel-merged channel as the through-water speed for epsilon/chi — previously
  a merged hotel `speed` channel was written to the outputs but consumed by
  nothing, while the docs claimed it fed the pipeline (finding m10; the
  pipeline/configuration docs now describe the real mechanism, which this
  change makes true). Slow-grid channels are interpolated and
  Butterworth-smoothed onto the fast grid exactly like the em/flight methods;
  fast-grid channels (the default `hotel.fast_channels` placement for
  `"speed"`) get the same NaN-interp/smooth/floor treatment without
  regridding. An explicitly requested hotel speed that is unusable (channel
  missing, matching neither time grid, or < 50% finite) is a per-file
  **error** that aborts the file with a recorded `errors` entry — never a
  silent fall-back to the 0.05 m/s `speed_cutout` floor, and never a silent
  downstream substitution with |dP/dt|. The same abort applies to any
  explicitly selected non-pressure method (em/flight/constant/hotel) whose
  speed stage fails; only the default pressure method keeps the historical
  warn-and-continue (its downstream fallback recomputes the same |dP/dt|).
  And the failures now actually fire: an `em`/`flight` speed with **zero**
  finite samples (dead flowmeter; level-flight/all-inflection pitch) errors
  before the cutout floor is applied — previously the all-NaN fill published
  a constant 0.05 m/s with provenance `"em"`/`"flight"`, indistinguishable
  from a real 0.05 m/s speed — and a non-finite `speed.value` errors for
  `constant`. em/flight deliberately use a zero-finite threshold rather than
  hotel's 50% rule: the flight model legitimately NaNs every sample below
  `min_pitch_deg` at dive/climb inflections, so a fraction cut could
  false-error real casts.
  Product provenance: `speed_source = "hotel:<var>"` on the per-profile
  NetCDFs, carried through to the diss/chi attrs by the precomputed-speed
  mechanism (which now retains upstream source strings that say more than the
  method name, e.g. `"hotel:speed"`, `"constant:0.4"`). And the other half of
  M10: when a hotel merge injected channels literally named
  `speed_fast`/`W_slow` whose values the speed stage recomputes and discards,
  the pipeline now **warns** and names the remedy instead of overwriting
  silently — `W_slow` always (it is always recomputed as smoothed |dP/dt|,
  method-independent), and `speed_fast` unless `speed.method: "hotel"` with
  `speed.hotel_var: "speed_fast"` actually consumes it. The rsi `--speed-method` layer
  keeps `hotel` perturb-only (hotel channels are merged there) and says so
  when asked for it. Note — hash churn: the new `speed.hotel_var` key changes
  the perturb `speed` section hash, so stage directories keyed on it recompute
  into fresh directories on the next run (the key set is part of the
  provenance signature).
- **Selectable reference temperature/conductivity with plausibility QC**
  (issue #131 finding B1). The reference temperature that drives seawater
  properties (viscosity ν for ε; ν and κ_T for χ; the published `T_mean`) is
  no longer hard-coded to `T1`:
  - **rsi**: new `epsilon.temperature` / `epsilon.conductivity` config keys
    (and the same in `chi`), plus `--temperature` / `--conductivity` flags on
    `rsi-tpw eps`, `chi`, and `pipeline`. Default `"auto"` picks the first
    plausible of `T1`..`Tn`, bare `T`, `JAC_T` — a railed/drifting/mostly-NaN
    channel is skipped with a warning (QC evaluates in-water samples,
    P > 0.5 dbar, when pressure is available), and a file with **no**
    plausible reference temperature errors per file instead of publishing
    wrong-viscosity products (on a railed-T1 corpus the old behavior was
    ε ≈ 5× low with all-green QC). An explicitly named channel is honored
    even when it fails QC (loud warning); a **number** is a constant
    reference temperature in °C (ODAS `constant_temp` parity). ODAS
    divergence is deliberate: ODAS uses T1 unchecked and silently
    substitutes 10 °C when temperature is missing.
  - `--salinity` (eps/chi) now also accepts `"measured"`: per-sample
    practical salinity from the resolved conductivity/temperature pair and
    pressure (TEOS-10), preferring the co-located `JAC_C`/`JAC_T` pair and
    falling back to the selected reference temperature (with a warning) when
    JAC_T is implausible; on `pipeline`, `--salinity measured` maps to the
    automatic path (measured JAC salinity was already preferred there).
  - Product provenance: the diss/chi NetCDFs gain `temperature_source`,
    `temperature_qc`, and `conductivity_source` (the resolved conductivity
    channel — consumed only under `salinity: measured`), plus
    `salinity_pair_temperature` when a measured salinity was actually
    computed; the run_pipeline L4 products gain `temperature_source`/
    `temperature_qc`. A constant reference outside the plausible ocean
    range is recorded in `temperature_qc` (and warns) instead of claiming
    "pass".
  - **perturb**: `epsilon.T_source` is now actually implemented (it was
    parsed but stripped before the computation — a dead key). `null`/`"auto"`
    = the QC chain above; a channel name or a number work as in rsi. One knob
    serves both the diss and chi stages. The interactive rsi viewers
    (`rsi-tpw ql`/`dl`/`ml`) reuse the same resolver for their viscosity
    preview.
  - **Note — hash churn**: adding the `temperature`/`conductivity` keys to
    the rsi `epsilon`/`chi` sections (and removing perturb's `T1_norm`/
    `T2_norm`, below) changes the config hashes, so existing `eps_NN`/
    `chi_NN` (and perturb stage) directories will not be reused (epsilon/chi always; prof dirs only when the merged profiles key-set changed — an explicit numeric W_min keeps them) — the next
    run recomputes into fresh directories. Deliberate: the key set is part
    of the provenance signature.
- **Salinity from a hotel file** — `epsilon.salinity`, `chi.salinity`, and the
  new `stratification.salinity` now accept `"hotel"` (or `"hotel:<var>"`) to draw
  practical salinity from a hotel-injected channel (default variable `salinity`),
  in addition to the existing `null` / number / `"measured"` forms. This feeds ε
  and χ **viscosity** and the `N2`/`Γ`/`K_ρ` **mixing** path from an external CTD
  feed, so gliders and MicroRiders with no onboard conductivity get correct
  viscosity and stratification instead of the fixed 35-PSU fallback. Set all
  three keys together (an unset `stratification.salinity` while ε/χ use `"hotel"`
  now warns). `epsilon.salinity` also gains the `"measured"` form for parity with
  `chi`. Note: the practical-/absolute-salinity/density (`SP`/`SA`/`ρ`) CTD and
  profile products still require onboard conductivity and are unaffected. See
  docs/perturb/configuration.md.
- **Shell tab-completion** for the argparse CLIs (`rsi-tpw`, `perturb`,
  `perturb-plot`, `perturb-diag`) via the optional
  [argcomplete](https://github.com/kislyuk/argcomplete) dependency (new
  `completion` extra: `pip install 'microstructure-tpw[completion]'`). Each CLI
  calls a shared `enable_argcomplete()` hook before parsing; it is a no-op unless
  argcomplete is installed and the shell's completion machinery is driving the
  process, so normal runs are unaffected. Enable it per shell with
  `eval "$(register-python-argcomplete rsi-tpw)"`. See docs/rsi-tpw/completion.md.
- **`rsi-tpw config FILE...`** — print a `.p` file's raw embedded configuration
  (INI) record — the `setup.cfg`-style text with the address matrix and every
  channel's calibration coefficients — to stdout. Useful for inspecting a
  coefficient in place (e.g. a suspect pressure `coef2`). Reads only the header
  and config record, so unlike `info` it also works on startup/truncated files
  that carry a config but no data records; backed by the new
  `PFile`-independent `read_config_string()`. See docs/rsi-tpw/cli.md.
- **`rsi-tpw sensors --cal-dir DIR`** — check each shear probe's configured
  `sens` against Rockland shear-probe calibration sheets (PDFs) in `DIR` and
  report mismatches. Parses `Probe SN`, sensitivity, calibration date, and the
  optional previous calibration date/sensitivity; the sensitivity applied to an
  observation is the calibration **in effect** at its date (hold-previous;
  linear interpolation between calibrations is intentionally deferred).
  `--cal-tol` sets the flag threshold as an **absolute** sensitivity difference
  (same units as `sens`, default `0.00005` — half the sheets' 4th-decimal
  resolution), not a percentage. PDF reading uses the new
  optional `cal` extra (`pip install 'microstructure-tpw[cal]'`, i.e. `pypdf`);
  it is imported lazily so the rest of `sensors` works without it. See
  docs/rsi-tpw/sensors.md.
- Zenodo DOI badge + identifiers (concept DOI `10.5281/zenodo.21366142`,
  version DOI `10.5281/zenodo.21366143`) in the README and `CITATION.cff`,
  plus a PyPI version badge — back-filled after v0.3.0 was archived on Zenodo.
- **`--speed-method {pressure,em,flight}` + `--aoa` on `rsi-tpw eps` and `chi`**
  (issue #131 findings M1/M2) — the through-water speed models that `pipeline`
  and perturb already had are now reachable from the modular commands (config
  keys `epsilon.speed_method` / `epsilon.aoa_deg`, same in `chi`; the
  `em`/`flight` methods are mutually exclusive with a fixed `--speed`, and
  the same exclusivity is now enforced by `run_pipeline`). An **explicit**
  `--speed-method pressure` forces the |dP/dt| path even when the input (a
  perturb per-profile file) carries a precomputed `speed_fast` channel;
  leaving the method unset prefers that channel, as before. `rsi-tpw
  pipeline` additionally honors the YAML `epsilon.speed_method`/`aoa_deg`
  (previously CLI-flag-only) and warns when the `[chi]` section sets a
  different `speed_method`/`aoa_deg`/`W_min` (single per-file resolution,
  like `temperature`/`conductivity`).
- **Speed provenance** (issue #131 finding M8): the diss/chi products now carry
  a `speed_source` attr stamped at the speed precedence point —
  `"fixed --speed <v>"` / `"precomputed speed_fast (perturb speed.method=em)"`
  / `"em (U_EM)"` / `"flight (aoa=3)"` / `"pressure |dP/dt|"` — and
  `compute_speed_for_pfile` returns its source string
  (`"pressure" | "em" | "flight" | "constant:<v>"`) as a third value so callers
  stamp what was actually computed. A stale upstream `speed_method` attr is
  cleared when a fixed speed or the pressure path is used, so the attrs can
  never claim e.g. `speed_method="em"` next to a fixed-speed source. The
  perturb pipeline writes `speed_method`/`speed_source` global attrs into the
  per-profile NetCDFs (new `extra_attrs` parameter on `extract_profiles`;
  written whenever the speed channel was computed) and the diss/chi stages
  inherit them.
- **perturb warning capture** (issue #131 finding M8): `warnings.warn(...)`
  (e.g. the glider |dP/dt| speed-bias warning) is now routed into the perturb
  run/worker/stage log files via `logging.captureWarnings(True)`, enabled in
  both `setup_root_logging` and `init_worker_logging` (spawn workers never run
  the former) — never at module import.

### Changed
- **Glider-correct detection defaults** (issue #131 findings M3/M13/M7):
  `W_min` now defaults to **auto** everywhere — 0.3 dbar/s for a free-falling
  profiler, 0.05 for glide/horizontal platforms — in rsi `prof`/`eps`/`chi`/
  `pipeline` (config `profiles.W_min: null`, new `epsilon.W_min`/`chi.W_min`
  keys + `--W-min` flags) and in perturb (`profiles.W_min: null`). In
  `run_pipeline` the resolved value also feeds `L2Params.profile_min_W`, so
  the L2 section selector no longer discards glider data that detection
  accepted. perturb's `profiles.direction` default flips **"down" → "auto"**
  (vehicle-resolved; a Slocum gets `glide` = up + down), and its detection
  site now resolves the same merged defaults that the cache signature hashes.
  Scope qualifier: **auto helps only vehicle-declaring corpora** — files whose
  `instrument_info` carries no `vehicle` (e.g. CASPER-era MicroRiders) still
  resolve to `down`/0.3 and need an explicit `direction: glide`. Note for
  local Rutgers configs: the `Rutgers/perturb.yaml` workaround comment
  ("REQUIRED: batch default 0.3 rejects gliders") is now obsolete — `W_min`
  and `direction` can be left unset there.
- **`rsi-tpw prof` is vehicle-aware** (issue #131 finding M13):
  `extract_profiles` now resolves `vehicle`/`direction`/`W_min`/tau before
  detection — `--vehicle slocum_glider` previously crashed with a
  `TypeError` (unexpected keyword) and `--direction auto` silently behaved
  as `down`. Full-record NetCDF sources resolve the vehicle from their
  `platform_type` attr (falling back to an instrument-model heuristic). VMP
  defaults are bit-identical (`down`, tau 1.5, W_min 0.3).
- **No more silent-empty products** (issue #131 finding M4): when profile
  detection finds nothing, `eps`/`chi` warn with the observed pressure span
  and peak fall rate vs the thresholds (and the flag to relax), print a
  per-file "no profiles detected" plus a final "N of M file(s) produced
  output" summary. **Exit-code change**: `rsi-tpw eps`/`chi` now exit **1**
  when *no* input file produced output (all failed and/or all empty), so a
  `set -e` batch script fails instead of reporting success over an empty
  directory — deliberate.
- **`rsi-tpw info`/`prof` batch robustness** (issue #131 finding M5): a
  startup file (valid config, no data records), 0-byte, or truncated `.p`
  file no longer aborts the batch — per-file errors print `ERROR: ...` and
  processing continues (exit 1 only when every file failed). The `eps`/`chi`
  loops additionally catch `struct.error` (truncated mid-file `.p`), matching
  `nc`.
- **Note — hash churn**: the `profiles`/`epsilon`/`chi` key-set and default
  changes above alter the rsi and perturb config hashes, so existing
  `prof_NN`/`eps_NN`/`chi_NN` and perturb stage directories will not be
  reused — the next run recomputes into fresh directories. Deliberate: the
  key set is part of the provenance signature, and this lands in the same
  release as the reference-temperature keys (one churn, not two).
- **FP07 factory calibration travels with per-profile NetCDFs** (issue #131
  finding m8). `extract_profiles` now writes `diff_gain` and the base
  thermistor's calibration (`e_b`, `b`, `gain`, `beta_1`, `beta_2`, `adc_fs`,
  `adc_bits`, `T_0` — exactly the set `_extract_therm_cal` can produce,
  including `b`, consumed by the electronics-noise model's eta term; the
  noise model's remaining knobs keep their defaults) as float variable attrs
  on every `T*_dT*` gradient channel — and on plain fast `T*` channels for
  instruments without pre-emphasis (the first-difference fallback) — and the
  chi loader reads them back from NetCDF sources. A per-profile file without
  the attrs (written before this change) falls back to the old generic
  defaults (`diff_gain=0.94`, ODAS-default thermistor coefficients) with a
  logged warning naming the file — never silently. The attrs carry the **factory** coefficients from the embedded
  `.p` config string; perturb's FP07 in-situ calibration may rewrite the
  channel *data*, which is compatible — the attrs describe the electronics.
- **`load_channels` carries vibration-sensor types** (`channel_types`: `.p`
  from `pf.channel_info`, NetCDF from the `sensor_type` attr
  `extract_profiles` writes), and the shared L1 builder labels an
  all-piezo-typed vibration stack `VIB` instead of `ACC`, matching the
  adapter's piezo convention (#131 W5-ii, W3 review F5). Label-only: no
  numeric consumer branches on `vib_type` (Goodman treats both alike), but
  displays now say `VIB` for piezo instruments (e.g. SN479's `Ax`/`Ay`).
- **`perturb-plot gamma-scaling` temperature Thorpe route resolves its
  channel** through the QC'd reference-temperature resolver (`T1` first,
  then `T2`…/`T`/`JAC_T`) instead of hard-coding slow `T1`. A profile whose
  every candidate is missing or implausible loses only its temperature route
  (NaN `LT_temp`, counted in the "skipped/degraded" notes) — a figure is
  never crashed by a railed thermistor.

### Removed
- **perturb `epsilon.T1_norm` / `epsilon.T2_norm` config keys** — they were
  never implemented (stripped before the computation, a silent no-op) and the
  template comment ("null = blend T1/T2") described behavior that never
  existed. **Breaking**: `validate_config` is strict, so a config that still
  sets them now fails loudly with an unknown-key error — delete the two lines
  (they never did anything).

### Fixed
- **`rsi-tpw cutp` / `extract_pfile_segment` absolute time** — a segment cut
  with `--start N>0` used to copy record 0's header timestamp verbatim, so the
  output's derived start time read N records too early (embedded provenance
  that would mislead any hotel/GPS join or absolute-time consumer). The tool
  now advances the record-0 timestamp by `N x recsize` (record duration from
  the embedded config, default 1.0 s; datetime arithmetic carries milliseconds
  across minute/hour/day boundaries; a year-0 startup clock is left unchanged
  with a warning). `tests/data/MR_SL685_climb.p` was regenerated with the
  fixed tool — data records byte-identical, only the record-0 timestamp words
  changed — and `test_mr_e2e.py` now pins the fixture's absolute start time.
- **chi `dof_spec` now subtracts the Goodman DOF loss** (#131 m9) — the chi
  product's `dof_spec` attribute is `1.9 * max(num_ffts − n_vib, 1)` when
  Goodman noise removal ran (each coherently-removed vibration signal costs
  one FFT segment of DOF, Lueck 2022b), matching the epsilon convention in
  `dissipation.py` instead of the uncorrected `1.9 * num_ffts`. **Note for
  downstream consumers of `dof_spec`:** values on the chi product are now
  smaller wherever Goodman ran with vibration channels present (e.g. 9.5
  instead of 13.3 for the default 4-segment window with 2 accelerometers);
  chi values themselves are unchanged.
- **Stratification salinity is now scrubbed, with a truthful provenance note**
  (#131 M6). The N²/dT/dz paths (window-scale in the diss/chi products,
  slow-grid in the profile product) previously read a hotel-sourced salinity
  channel raw: an **all-NaN merged channel** (a hotel variable with fewer than
  two finite source samples) silently NaNed out N² and every mixing product
  (K_T/Γ/K_ρ) while the metadata still claimed "salinity from hotel channel",
  and **partial NaNs** (a `"hotel:<var>"` pointing at a derived profile
  variable, e.g. an SP computed from bad conductivity) NaNed the affected
  windows with no note. (The hotel merge itself never produces NaN outside the
  hotel file's time coverage — it boundary-holds and bridges interior gaps —
  so the earlier "outside CTD coverage" description of this failure mode was
  wrong.) Both stratification consumers now scrub the salinity: non-finite
  samples are filled by **interpolation over the finite samples**
  (nearest-finite hold at the edges) — **per cast** on the slow-grid path,
  because interpolating across a cast boundary would blend one cast's deep
  salinity toward the next cast's shallow values and fabricate a wrong-sign
  within-cast dS/dz (collapsing N² near the boundary). A channel — or a cast
  slice — with **fewer than two finite samples** falls through to
  conductivity/35 PSU (the hotel merge's own <2-finite convention; a single
  sample must not constant-fill a profile and beat a valid conductivity
  channel). The N² `comment` records exactly what was used: interpolated vs
  edge-held counts ("N/M non-finite salinity samples interpolated", "N/M edge
  samples held at the nearest finite value; N2 approaches temperature-only
  where held") or the fallback reason ("hotel channel 'x' entirely
  non-finite"). The **conductivity fallback is gated the same way**: a
  conductivity channel that is present but whose derived `SP_from_C` salinity
  has fewer than two finite samples (e.g. an all-NaN `JAC_C`) now falls
  through to the documented 35 PSU fallback with the reason noted ("JAC_C
  present but yielded no finite salinity"), instead of producing all-NaN N²
  under a note claiming C-derived salinity; a partially-NaN derived SP (two
  or more finite samples) keeps the existing per-window masking
  (interp-filling derived SP is deferred). The slow-grid N² comment also no
  longer claims the salinity
  came "from the profile's own C/T/P" when it did not. The viscosity path's
  scrub switches from a whole-profile **median** fill to the same
  interpolation fill: N² is first-order in dS/dz, so a constant fill would
  insert a spurious salinity step at every fill boundary (spurious N² ~
  g·β·ΔS/Δz, orders above thermocline values, and Thorpe sorting keeps it);
  for viscosity the difference is negligible.
- **Old-format MicroRider deconvolution** (#131 M11, M12) — two bugs that
  corrupted or crashed processing of old-MR corpora (e.g. CASPER 2015_East
  CAS_001-006, whose 4×10 matrix samples T1 *and* T1_dT1 as full fast columns
  and P/P_dP twice per scan). **M11**: a natively-fast base channel (T1) kept
  its "fast" classification after deconvolution overwrote it with slow-length
  data, so `is_fast()` lied and `rsi-tpw nc` crashed with a broadcast error in
  the gradT builder; the base is now reclassified slow (the full fast-rate
  deconvolved signal lives in `T1_dT1`, matching the modern-config invariant).
  **M12**: a duplicate-sampled slow pair (P/P_dP, 2× per scan) was deconvolved
  at the 2× matrix-occurrence rate instead of the decimated array's true rate,
  applying the pre-emphasis crossover at twice its design frequency and
  mis-blending P and P_dP — the pressure feeding fall-rate/speed and thus ε/χ.
  The deconvolution rate (and fast/slow branch) is now derived from the array
  actually extracted.
- **Perturb chi no longer computed with generic FP07 calibration** (issue
  #131 finding m8). The perturb pipeline computes chi from per-profile
  NetCDFs, whose loader hard-coded `diff_gain=0.94` and an empty thermistor
  calibration; with the new attrs (see Added) it now uses the instrument's
  real coefficients — on SN479, `diff_gain=0.912`/`0.920` and `b=0.99861`/
  `0.99927`, so the 0.94/1.0 hardcodes were provably wrong there. **Numerics
  note**: chi's FP07 noise floor and bilinear differentiator correction on
  the perturb path change wherever the real coefficients differ from the
  defaults — expected, and in the correct direction (`rsi-tpw chi` on the
  `.p` file and perturb-on-NetCDF now agree; measured on the SN479 fixture
  the residual matched-window difference drops from ≲3.5% to ≲0.15%).
- **Measured-salinity CT pairing QC** (rsi `run_pipeline` adapter): the
  practical salinity computed from `JAC_C` now pairs with `JAC_T` only when
  JAC_T passes the plausibility QC, falling back to the resolved reference
  temperature with a warning — previously a railed JAC_T silently poisoned
  the measured salinity.
- **`rsi-tpw pipeline` batch robustness**: one unreadable/implausible file no
  longer aborts a multi-file run; the pipeline logs the per-file error and
  continues (mirroring the `eps`/`chi` loops).
- **`rsi-tpw patch-template`** now scaffolds *every* per-channel calibration
  field, not just `coef0`/`coef1`. The previous hardcoded whitelist silently
  dropped higher-order polynomial coefficients (a pressure channel's `coef2`,
  `coef3`…) and thermistor Steinhart-Hart terms (`a`, `b`, `beta_1`, `t_0`, …),
  so a coefficient a user needed to patch never appeared in the template. The
  scaffold now emits all fields except the structural identifiers
  (`id`, `name`, `type`, `units`, `sign`), in config-file order.
- **Interactive viewers (`ql`/`dl`/`ml`) — direction-aware `W_min` default.**
  The fall-rate floor for profile detection now defaults to **0.05 dbar/s** when
  the resolved direction is `glide` or `horizontal` (slow glider/AUV motion),
  and stays **0.3 dbar/s** for `down`/`up` (free-falling profilers). Previously a
  fixed 0.3 rejected every cast from a slow platform, so `--direction glide`
  alone still found nothing; now it works without also passing `--W-min`. An
  explicit `--W-min` still overrides. The batch `prof` command is unchanged.
- **Interactive viewers (`ql`/`dl`/`ml`)** now explain *why* no profiles were
  detected instead of the bare "No profiles detected in this file". The message
  reports the observed pressure span and peak fall/rise rate against the
  `P_min`/`W_min` thresholds and suggests the fix — e.g. a slow or glider-style
  cast whose fall rate never reaches the `W_min` default is told to lower
  `--W-min` and/or use `--direction glide`.

## [0.3.0] - 2026-07-14

### Added
- **Release/citation metadata** — `CITATION.cff` (drives GitHub's "Cite this
  repository" button) and `.zenodo.json` (controls the Zenodo archive record),
  PyPI trove classifiers and expanded `[project.urls]`, a package-level
  `odas_tpw.__version__` sourced from the installed distribution, and a
  Citation section in the README. Groundwork for a tagged, DOI-archived
  `v0.3.0` release published to PyPI.
- **`rsi-tpw bench`** — a bench-test diagnostic, ported from ODAS
  `quick_bench.m` and extended to auto-evaluate the *Rockland Bench Test
  Review Checklist* (V3). Produces raw-count time-series and
  counts²·Hz⁻¹ spectra figures (plus a CT/CLTU figure when those
  channels exist) and a checklist report that marks each quantitative
  criterion PASS/FAIL and flags the subjective ones REVIEW. Reads with a
  new `PFile(path, deconvolve=False)` option so the pre-emphasized
  channels (`T1_dT1`, `T2_dT2`, `P_dP`) carry the raw counts the
  checklist thresholds are defined on. See docs/rsi-tpw/bench.md.

### Changed
- **perturb epsilon/chi windows are now duration-based** (`fft_sec`,
  `diss_sec`, `overlap_sec`), resolved to sample counts at each
  instrument's sampling rate, and the **epsilon FFT default changed from
  256 samples to 1 second** (512 samples on a 512-Hz VMP-250; 1-2 kHz
  coastal units scale automatically) per Lueck et al. (2024) best
  practices — see docs/perturb/dissipation_length.md. Sample-count keys
  (`fft_length`/`diss_length`/`overlap`) remain as expert overrides that
  win when set; legacy configs pinning them keep byte-identical stage
  signatures and resolve their existing output directories unchanged.

### Fixed
- Reference library (`papers/`) DOI corrections: fixed three wrong DOIs —
  Rehmann & Hwang (2005) was a dead link (`JPO2676.1` → `JPO-2676.1`) and
  the two Smyth & Moum (2000) companion papers were cross-linked to each
  other / an unrelated paper — plus nine old-AMS DOI hrefs whose literal
  parentheses truncated the clickable link (now percent-encoded), and two
  citation nits (Nash et al. 1999 page range, Shapiro et al. glider-paper
  year). All 53 DOIs now resolve to the correct paper.
- CT alignment (`ct_align`) and the despike pass-count flag: conductivity
  is now shifted with edge-hold instead of `np.roll` (which wrapped one
  end of the record into the other), and the too-many-passes QC flag
  uses each window's own section pass count
  (`PASS_COUNT_SH(probe, section)` semantics) instead of the per-channel
  maximum.
- Method-1 chi: parabolic refinement of the log-grid search removes the
  ~4.7 % grid quantization on chi (synthetic recovery now better
  than 0.5 %).
- No-Goodman shear spectra are detrended parabolically, matching ODAS
  `get_diss_odas.m` (`method='parabolic'` in the no-goodman branch);
  Goodman-cleaned spectra remain linear, as in `clean_shear_spec.m`.
- Fall-rate smoothing no longer reuses the shear high-pass cutoff
  (`HP_cut`) as a pre-smoothing knob when `speed_tau > 0`: the speed is
  smoothed exactly once, at 0.68/tau (the ODAS convention). Benchmark
  speed and epsilon agreement unchanged.
- Documented the FP07 tau/transfer-function pairings (`double_pole`
  pairs with the fixed 3 ms Goto tau, NOT Peterson & Fer's
  speed-dependent tau — and how to reproduce Peterson & Fer exactly)
  and the iterative chi fit's actual limit-refinement behavior.

### Added
- Complete ATOMIX QC flag set for epsilon (`EPSI_FLAGS`): in addition to
  the existing FM (bit 1) and variance-resolved (bit 16) criteria, the
  flags now include despike fraction (bit 2), inter-probe consistency
  (bit 4, `ln(e_i/e_min) > 1.96*sqrt(2)*sigma_ln` with the Lueck 2022a
  variance model — only the larger estimates are flagged), and
  too-many-despike-passes (bit 8). The variance-resolved criterion is
  now applied only to variance-method estimates (not ISR), matching the
  benchmark convention. Per-window `DESPIKE_FRACTION` is computed in
  L2/L3 and carried in `L4Data`. The benchmark comparison
  (`scor160-tpw l1-l4`/`l2-l4`) now reports QC flag agreement using each
  dataset's own thresholds (`read_l4_qc_limits`): 83-100% exact match
  on the VMP250 tidal and MR1000 datasets.
- Derived mixing quantities in the `perturb` chi stage (`chi.mixing`,
  default true): `N2`, `dTdz`, `K_T`, `Gamma`, `K_rho` on the chi window
  grid, with practical salinity from the profile's own C/T/P (TEOS-10)
  so `N2` is fully constrained.
- Derived mixing quantities in the `rsi-tpw pipeline` Method-1 chi output
  (`L4_chi_epsilon.nc`, propagated to `L5_binned.nc`/`L6_combined.nc`):
  window-scale `N2` (TEOS-10) and background `dTdz`, Osborn-Cox heat
  diffusivity `K_T`, measured mixing coefficient `Gamma` (Oakey 1982),
  and Osborn diffusivity `K_rho` (Gamma_0 = 0.2), with
  stratification/gradient validity masking. New module
  `odas_tpw.processing.mixing`.
- Robustness tests for corrupted/truncated `.p` files and for the
  `accel`/`magn` channel converters.
- `perturb` now runs profile detection and CT alignment *before* the CTD
  fork, so the CTD product's salinity/density are computed from
  time-aligned conductivity whenever profiles are detectable (previously
  always unaligned).
- Per-CLI / per-worker / per-stage / per-combo log fan-out for `perturb`. Every
  invocation now writes `<output_root>/logs/run_<stamp>.log`, plus
  `worker_<stamp>_<pid>.log` per parallel worker, `<stem>.log` inside each
  versioned `profiles_NN/`, `diss_NN/`, `chi_NN/`, `ctd_NN/`,
  `*_binned_NN/` dir, and `combo.log` in each combo dir. See
  `docs/perturb/logging.md`.
- `--stdout` and `--log-level` flags on every pipeline-running subcommand.
- YAML configuration file system with `rsi-tpw init` template generation
- Three-way parameter merge: defaults <- config file <- CLI flags
- Cumulative hash-tracked sequential output directories (`eps_00/`, `chi_00/`, ...)
- Configuration validation with clear error messages for unknown sections/keys
- Resolved `config.yaml` written to each output directory for reproducibility
- Comprehensive documentation in `docs/`: configuration reference, CLI reference, Python API, pipeline guide
- `scor160-tpw` CLI reference documentation (`docs/scor160/cli.md`)
- Regression test for parallel Method 1 chi (`test_cmd_chi_method1_parallel`)
- CHANGELOG

### Fixed
- `rsi-tpw chi --epsilon-dir` now works in parallel mode (`-j > 1`); previously `epsilon_dir` was ignored and all workers used Method 2
- Serial chi path now properly closes epsilon datasets via try/finally (prevents resource leak)
- `perturb run` merge stage now feeds merged files to downstream processing (return value was previously discarded)

### Changed
- `perturb` console output is now silent by default — every record goes to the
  per-invocation log file.  Pass `--stdout` to also stream to stderr.  Previously
  `logging.basicConfig` printed everything to stderr unconditionally.
- `*_binned_NN/` and `*_combo*/` output dirs are now created up-front (before
  the bin/combo work runs) so per-input log files can be opened inside them.
  An empty bin/combo step now leaves a dir with just `.params_sha256_*` and the
  log files, instead of no dir at all.
- Output directories now use sequential naming with parameter hash deduplication
- Each output directory includes a `config.yaml` recording the exact parameters used
- `README.md`, `CLAUDE.md`, and `docs/rsi-tpw/python_api.md` now use `run_pipeline()` / `compute_diss_file()` / `compute_chi_file()` instead of deprecated `get_diss()` / `get_chi()`
- `docs/rsi-tpw/pipeline.md` and `docs/rsi-tpw/output_directories.md` updated to reflect actual `{stem}/profile_NNN/` pipeline output layout
- `docs/perturb/pipeline.md` updated: `perturb run` covers the full chain trim→merge→process→bin→combo (`run_pipeline()` now calls `_run_combo`); `perturb combo` remains available to re-run combo assembly on its own
- `docs/perturb/cli.md` now documents `--hotel-file` and `--p-file-root` flags
- Fixed ~30 stale source paths in `docs/chi_mathematics.md`, `docs/epsilon_mathematics.md`, and `docs/rsi-tpw/vectorization.md` (`rsi/` → `chi/` or `scor160/` as appropriate)

## [0.1.0] - 2025-03-08

### Added
- `PFile` class for reading Rockland Scientific `.p` binary files
- Channel conversion functions (raw counts to physical units) for all sensor types
- NetCDF4 conversion with CF-1.13 compliance
- Profile detection from pressure time series
- Epsilon (TKE dissipation rate) calculation from shear probe spectra
  - Nasmyth universal spectrum fitting (Lueck improved coefficients)
  - Iterative despiking of shear signals
  - Goodman coherent noise removal using accelerometer cross-spectra
  - Macoun & Lueck wavenumber correction
  - Figure of merit (fom) and K_max_ratio QC metrics
- Chi (thermal variance dissipation rate) calculation from FP07 thermistor spectra
  - Method 1: from known epsilon with FP07 transfer function correction
  - Method 2a: MLE Batchelor spectrum fitting (Ruddick et al. 2000)
  - Method 2b: Iterative integration (Peterson & Fer 2014)
  - Batchelor and Kraichnan theoretical spectrum models
  - FP07 single-pole and double-pole transfer functions
  - Electronics noise model
- Unified `rsi-tpw` CLI with subcommands: `info`, `nc`, `prof`, `eps`, `chi`, `pipeline`
- Parallel processing support (`-j` flag)
- Seawater property functions via gsw/TEOS-10: viscosity, density, buoyancy frequency
- MATLAB implementation of chi calculation
- CI pipeline: ruff linting, mypy type checking, pytest on Python 3.12/3.13
- Mathematical documentation for epsilon and chi algorithms
