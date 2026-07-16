# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **Legacy ODAS header-v1 `.p` support: the `rsi-tpw v1to6` translator**
  (issue #141). Pre-2015 files (header word 11 == 1; record 0 holds a binary
  address matrix, configuration in an external old-dialect setup file) are
  translated to standard v6 files — vendor `patch_setupstr.m` header contract
  (words 11-14 → `0x0600`/config-size/0/0, everything else preserved), data
  records copied byte-for-byte (unlike the vendor tool, per-record headers,
  timestamps, and bad-buffer flags survive), and the embedded INI is
  **synthesized** from the setup file (old `key: values` dialect parser in
  `rsi/setup_v1.py`; auto-detected `setup.txt` > `setup*.txt` > `setup*.cfg`
  siblings with INI-dialect sniffing, cross-candidate consistency warnings,
  and a hard record-0-matrix assertion) with machine-readable provenance keys
  in `[root]`. The **complete** provenance set (`translated_from`,
  `v1_source_file`, `setup_file_source`, `setup_file_md5`, `sens_source`,
  `translator`, `translated_on`) is carried onto every derived product —
  full-record and per-profile NetCDF, standalone epsilon/chi files, and the
  pipeline L4 writers — on both routes (on-disk translated file and direct
  raw-v1 read); the setup-file md5 + sens source are the audit trail for the
  sens⁻² epsilon scaling. `PFile` reads raw v1 files directly via the same
  translation in memory (`setup_file=` kwarg to override discovery;
  `translated_from_v1` / `v1_provenance` attributes), refuses any other
  pre-v6 version loudly (raw + decoded version in the message), and the v6
  path is pinned by golden per-channel regressions of the three committed
  fixtures (exact raw-count hashes; converted channels via stats plus
  order-sensitive shape/finite-mask/decimated-value goldens).
  Shear sens is never defaulted: `--sens sh1=…,sh2=…`, `<name>_sens:` setup
  keys, or `patch-config --add-keys` on translated files (the per-epoch
  workflow); a sens-less shear channel errors at conversion time. FP07
  thermistors stay raw counts on v1 corpora (no authoritative calibration —
  the ~4.2 °C offset of the only on-disk candidate coefficients makes them
  untrustworthy); chi on v1 is deferred. See `docs/rsi-tpw/legacy_v1.md`.
- **Sea-Bird SBE3/SBE4 (`sbt`/`sbc`) converters** (vendor
  `odas_sbt/sbc_internal` parity; coef4-6 carry f0/f_ref/n_periods; verified
  to ≤1e-8 °C / ≤1e-8 mS/cm against the 2013 Taiwan channel-level ground
  truth; zero counts yield NaN instead of a division blow-up). Files whose
  configs declare `sbt`/`sbc`-type channels previously kept raw counts with a
  "no converter" warning — they now convert; files without such channels are
  bit-identical. The `auto` reference-temperature chain gains an `sbt`-typed
  candidate tail (after `JAC_T`, matched by channel TYPE, not name), so a v1
  corpus with counts-valued FP07s lands on the SeaBird automatically, and
  `--conductivity SBC1 --salinity measured` works (mS/cm, SP_from_C-ready).
- **v1 refusal guards** in every other positional `.p` reader — perturb trim
  (which would otherwise write a corrupt "trimmed" copy of every v1 file),
  perturb merge, `rsi-tpw sensors` (silently-empty inventory), patch-config /
  patch-template, `rsi-tpw config` (`read_config_string`), and `cutp`
  (`extract_pfile_segment`) — each names the `rsi-tpw v1to6` remedy.
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
    `chi_NN` (and perturb stage) directories will not be reused — the next
    run recomputes into fresh directories. Deliberate: the key set is part
    of the provenance signature.
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
  `p_to_netcdf`'s classification; the label is descriptive only and epsilon/chi
  values are unchanged.
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

### Changed
- **A shear channel without a usable `sens` is now a hard per-file error**
  (was: a warning plus a fabricated default of 1.0 — plausible-looking shear
  that scales epsilon by sens⁻²). "Usable" means parseable, **finite**, and
  positive: `nan` would bypass a bare sign check (every NaN comparison is
  False → all-NaN shear) and `inf` yields all-zero shear, so both are
  rejected everywhere sens enters — conversion, the `v1to6 --sens` CLI
  parser, and the Python-API overrides. ODAS `convert_odas.m` parity: the
  vendor errors outright too. Modern configs always carry `sens`, so only
  genuinely broken configs and un-patched v1 translations are affected; the
  error names the three remedies (`patch-config --add-keys`, `v1to6 --sens`,
  `<name>_sens:` setup keys). (#141)

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
