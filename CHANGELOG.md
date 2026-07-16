# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
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

### Removed
- **perturb `epsilon.T1_norm` / `epsilon.T2_norm` config keys** — they were
  never implemented (stripped before the computation, a silent no-op) and the
  template comment ("null = blend T1/T2") described behavior that never
  existed. **Breaking**: `validate_config` is strict, so a config that still
  sets them now fails loudly with an unknown-key error — delete the two lines
  (they never did anything).

### Fixed
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
