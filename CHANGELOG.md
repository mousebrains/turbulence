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

### Fixed
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
