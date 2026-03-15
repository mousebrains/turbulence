# Code Quality Audit

**Date**: 2026-03-14
**Scope**: Full odas_tpw codebase — architecture, duplication, large functions, testing, infrastructure
**Prior audit**: Parallel pipeline (the critical finding last time) has been eliminated. `get_diss()` and `get_chi()` now delegate to the modular scor160 L2→L3→L4 chain. This audit focuses on what remains.

---

## Overall Grade: B (3.0 / 4.0)

The parallel-pipeline anti-pattern is gone, which is a significant improvement. Mathematics and MATLAB agreement remain excellent. However, a thorough line-by-line review reveals substantial remaining issues: **55 functions over 80 lines** (14 over 150), pervasive code duplication in spectral/noise/dataset-builder code, and weak docstring coverage in perturb. The codebase is professionally capable but carries real structural debt.

### Grading Breakdown

| # | Category | Grade | What's Good | What's Lacking |
|---|----------|-------|-------------|----------------|
| 1 | Mathematical correctness | A | Every formula verified against publications | — |
| 2 | MATLAB/ODAS agreement | A | 307/307 cross-validation passing | Known speed difference in scalar spectra |
| 3 | Unit consistency | A | Consistent throughout | — |
| 4 | Architecture & duplication | B- | Single pipeline; clean package layering; config unified | Dataset builders duplicated; noise/spectral/speed code repeated across modules |
| 5 | Large functions | C | Pipeline monoliths gone | 55 functions >80 lines; 14 >150 lines; `p_to_L1` is 374 lines |
| 6 | Dead code | A- | Only 1 TODO marker; no unused functions found | `get_diss()`/`get_chi()` deprecated but still the primary CLI entry points |
| 7 | Test coverage | B+ | 901 tests, 53 files, MATLAB cross-validation, 70% coverage threshold | 26 duplicate config tests; viewers smoke-tested only; 76 silent skip calls |
| 8 | Public API | B | ~50 exports; deprecation warnings on old functions | Two entry points still exported (`get_diss`/`get_chi` + `run_pipeline`) |
| 9 | Type annotations | B- | scor160 and chi well-typed; mypy in CI | 16 modules with mypy overrides; viewers 0%; long return tuples untyped |
| 10 | Docstrings | C+ | Algorithm functions excellent (Nasmyth, Batchelor, despike) | perturb ~50% undocumented; many private functions lack docstrings |
| 11 | Error handling | B | warnings.warn at NaN/clamp sites; narrowed exceptions | Print-based logging in perturb; no structured logging anywhere |
| 12 | Infrastructure | A- | CI matrix (3 OS × 2 Python); ruff + mypy; coverage threshold; release workflow | pytest-xdist installed but unused; MATLAB tests skip silently |

---

## Codebase Metrics

| Metric | Count |
|--------|-------|
| Source files | 61 |
| Source lines | 17,077 |
| Test files | 53 |
| Test lines | 12,648 |
| Test functions | 901 |
| Public exports | ~50 |
| TODO/FIXME markers | 1 |
| Subpackages | 4 (scor160, chi, rsi, perturb) |
| Functions >80 lines | 55 |
| Functions >150 lines | 14 |

### Lines by Subpackage

| Subpackage | Source Lines | Files | Role |
|------------|-------------|-------|------|
| rsi | 7,525 | 20 | I/O, pipeline wrappers, CLI, viewers |
| perturb | 3,713 | 19 | Campaign processing |
| scor160 | 3,515 | 13 | ATOMIX benchmark, core pipeline |
| chi | 2,070 | 7 | Thermal dissipation |
| top-level | 254 | 2 | config_base |

---

## Detailed Findings

### 4. Architecture & Duplication — B-

The parallel-pipeline problem is solved. What remains is scattered duplication within the single pipeline path.

#### 4a. Dataset builders: `_build_diss_dataset` and `_build_chi_dataset` (350 lines total)

`rsi/dissipation.py:318` (175 lines) and `rsi/chi_io.py:343` (175 lines) are structurally identical xarray construction functions — same coordinate setup, same CF-1.13 attribute patterns, same `coords["t"].attrs.update()` call. Only the variable names and units differ. Should be a shared parameterized builder.

#### 4b. FP07 noise: scalar vs batch (290 lines of duplication)

`chi/fp07.py` contains two near-identical implementations:
- `noise_thermchannel()` — 170 lines (L195–364), scalar
- `noise_thermchannel_batch()` — 120 lines (L409–528), vectorized

Both unpack the same `FP07NoiseConfig` object (18 lines duplicated verbatim at L286–303 and L452–469), compute the same F-dependent intermediates, apply the same T-dependent terms. The batch version should call the scalar or share extracted helpers. Similarly `fp07_tau()` (25 lines) and `fp07_tau_batch()` (23 lines) duplicate model-selection logic.

#### 4c. Spectral estimation: three overlapping CSD functions (345 lines total)

`scor160/spectral.py` has three functions that share 60–70% of their logic:
- `csd_odas()` — 98 lines, single-pair
- `csd_matrix()` — 104 lines, multi-channel
- `csd_matrix_batch()` — 143 lines, multi-channel batched

Window/overlap/detrending setup, auto-spectrum normalization, and cross-spectrum computation are repeated across all three. `csd_odas()` should call `csd_matrix()` internally.

#### 4d. Speed computation scattered across 8 call sites

`_smooth_fall_rate()` + `np.interp()` + floor clamp appears in: `rsi/adapter.py`, `rsi/convert.py`, `rsi/helpers.py` (2×), `rsi/pipeline.py`, `rsi/viewer_base.py`, `perturb/pipeline.py`, `perturb/fp07_cal.py`. Each imports `_smooth_fall_rate` from `rsi/profile.py` and repeats the same 5-line pattern. Should be a single `compute_speed_fast()` helper.

#### 4e. L4 chi: `process_l4_chi_epsilon` and `process_l4_chi_fit` (215 lines combined)

`chi/l4_chi.py` — Both functions initialize the same 6 output arrays, loop over `j in range(n_spec), ci in range(n_gradt)` with the same structure, and assemble identical `L4ChiData`. Only the inner fit call differs. Should be unified with a strategy parameter.

#### 4f. Goodman bias correction duplicated

`scor160/goodman.py` — `clean_shear_spec()` (110 lines) and `clean_shear_spec_batch()` (61 lines) duplicate the bias correction calculation verbatim. Extract `_bias_correction()` helper.

#### 4g. Comparison functions share identical patterns

`scor160/compare.py` (422 lines) — All three `compare_l*()` and `format_l*_report()` functions follow the same template: compute metrics, check finite values, compute RMS/correlation, build string report. Log-spectral comparison appears in both L3 and L4. Extract shared helpers.

#### 4h. scor160 CLI: 6 near-identical command handlers

`scor160/cli.py` (290 lines) — `_cmd_l1_l2`, `_cmd_l2_l3`, `_cmd_l1_l3`, `_cmd_l3_l4`, `_cmd_l2_l4`, `_cmd_l1_l4` all follow the same `read → process → compare → format_report` pattern. Should be one parameterized function.

#### 4i. perturb CLI: 3 duplicate analysis commands

`perturb/cli.py` — `_cmd_profiles`, `_cmd_diss`, `_cmd_chi` repeat ~27 lines of identical boilerplate (load config, expand globs, call `run_pipeline()`). Extract a shared helper.

#### 4j. Config test duplication (26 identical test functions)

`tests/test_config.py` (52 tests) and `tests/test_perturb_config.py` (44 tests) share 26 test functions with identical names and near-identical bodies. Both test `ConfigManager` from `config_base.py`. Should be parameterized once.

### 5. Large Functions — C

14 functions exceed 150 lines. These are the worst offenders:

| Function | File | Lines | Issue |
|----------|------|-------|-------|
| `p_to_L1()` | rsi/convert.py | 374 | Monolith: reads binary, classifies channels, writes NetCDF. NetCDF variable creation repeated 12× with near-identical patterns. |
| `run_pipeline()` | rsi/pipeline.py | 288 | Orchestrates 8 steps in one function with nested profile loops. |
| `get_diss()` | rsi/dissipation.py | 248 | Wrapper that builds L1Data, runs L2→L3, then custom epsilon loop. |
| `process_l3_chi()` | chi/l3_chi.py | 246 | Window building, spectral computation, metadata assembly all in one. 11 separate list accumulations for per-window results. |
| `get_chi()` | rsi/chi_io.py | 207 | Similar structure to `get_diss()` but for chi. |
| `process_file()` | perturb/pipeline.py | 180 | Handles 8 distinct stages (hotel, CTD, profiles, FP07, CT align, diss, chi) with try-except per stage. |
| `process_l3()` | scor160/l3.py | 178 | Single monolith for window extraction, spectral computation, wavenumber conversion. Magic constants (0.05, 150) hardcoded. |
| `_build_diss_dataset()` | rsi/dissipation.py | 175 | Repetitive xarray construction. |
| `_build_chi_dataset()` | rsi/chi_io.py | 175 | Same pattern as above. |
| `noise_thermchannel()` | chi/fp07.py | 170 | 19 parameters; duplicated by batch version. |
| `_read()` | rsi/p_file.py | 160 | Binary parsing with complex address matrix logic. |
| `run_pipeline()` | perturb/pipeline.py | 155 | Discover, trim, merge, GPS, hotel, parallel execution, binning. |
| `fp07_calibrate()` | perturb/fp07_cal.py | 150 | Validation, RT/R0 computation, profile lags, Steinhart-Hart fit, calibration. |
| `pfile_to_l1data()` | rsi/adapter.py | 145 | PFile → L1Data conversion with many optional branches. |

Functions in the 80–150 line range (41 additional) include `_estimate_epsilon` (138), `_iterative_fit` (133), `csd_matrix_batch` (143), `_load_from_nc` (127), `noise_thermchannel_batch` (120), and others. Many of these are scientifically complex and would not benefit from splitting (e.g., `_estimate_epsilon` is a well-structured multi-step algorithm). The ones that *should* be split are orchestration functions that mix I/O, data transformation, and computation.

### 6. Dead Code — A-

Only 1 TODO marker (`perturb/pipeline.py:441`). No unused functions detected. `get_diss()` and `get_chi()` emit `DeprecationWarning` but remain the entry points for the `eps` and `chi` CLI subcommands — they're not dead, just deprecated.

### 7. Test Coverage — B+

**901 test functions across 53 files. 1599 passing, 10 skipped.**

| Subpackage | Test Files | Approx Tests | Quality |
|------------|-----------|-------|---------|
| scor160 | ~12 | ~200 | Good — unit + integration + MATLAB |
| rsi | ~18 | ~350 | Good — unit + integration + CLI |
| chi | ~5 | ~150 | Good — unit + integration + MATLAB |
| perturb | ~12 | ~150 | Good — unit + integration |
| cross-validation | 4 | ~60 | Excellent — 30 .p files validated |

**Gaps:**
- **26 duplicate test functions** between `test_config.py` and `test_perturb_config.py`
- **76 `pytest.skip()` calls** — tests skip silently if VMP data is missing, masking coverage loss in CI
- Viewers smoke-tested only (instantiation, not behavior)
- `config_base.py` (254 lines) has no direct unit tests — only tested transitively through rsi/perturb config tests
- No tests for corrupt or truncated `.p` files
- pytest-xdist installed but not used in CI (`-n auto` would parallelize slow MATLAB tests)

### 8. Public API — B

~50 exports across 4 `__init__.py` files. The deprecation situation is awkward: `get_diss()` and `get_chi()` are deprecated yet remain the recommended CLI entry points (via `rsi-tpw eps` and `rsi-tpw chi`). `run_pipeline()` is the replacement but the CLI experience for the non-pipeline subcommands still calls the deprecated functions. Either remove the deprecation warnings or update the CLI to call `run_pipeline()` internally.

### 9. Type Annotations — B-

| Module Group | Coverage | Notes |
|-------------|----------|-------|
| scor160 (all modules) | ~95% | Solid; dataclasses fully typed |
| chi core (batchelor, fp07, l2/l3/l4_chi) | ~90% | Good |
| chi/chi.py | ~80% | Return tuples should use NamedTuple (7-element tuples are fragile) |
| rsi core (p_file, channels, config) | ~90% | Good |
| rsi pipeline (dissipation, helpers, adapter) | ~80% | `dict[str, Any]` returns instead of typed structures |
| rsi viewers (viewer_base, quick_look, diss_look) | ~0% | mypy fully disabled |
| perturb | ~90% | Type hints present but few docstrings |

16 modules have per-module mypy overrides suppressing error codes. The viewer modules have mypy disabled entirely.

### 10. Docstrings — C+

Excellent for core algorithms (Nasmyth, Batchelor, MLE, Goodman, despike — all have Parameters/Returns and publication references). But:

- **perturb**: 50% of functions lack docstrings, including public APIs like `ctd_bin_file()`, `ct_align()`, `detect_bottom_crash()`, `mk_epsilon_mean()`
- **rsi viewers**: Minimal docstrings; methods over 100 lines lack documentation
- **chi/fp07.py**: `noise_thermchannel()` well-documented but helper functions undocumented
- **Private functions**: Inconsistent; some have full NumPy-style docstrings, many have none

### 11. Error Handling — B

- `warnings.warn()` at NaN/clamp sites in chi.py (7 sites)
- `ValueError` raised for invalid inputs
- Narrowed exception types in CLI and convert modules
- **Gaps**: perturb uses `print()` statements for logging (no `logging` module); `ProcessPoolExecutor` errors in perturb/pipeline.py are printed to stderr with no summary or rollback

### 12. Infrastructure — A-

**Strong:**
- CI: ruff lint + mypy + pytest matrix (Python 3.12/3.13 × Ubuntu/macOS/Windows)
- Separate lint, typecheck, test jobs (lint as blocking dependency)
- 70% coverage threshold with codecov integration
- Release workflow (PyPI on `v*` tags)
- Dependabot (pip + GitHub Actions, monthly)

**Gaps:**
- pytest-xdist installed but CI doesn't use `-n auto` — MATLAB tests run serially
- Codecov only reports from Python 3.13 on Ubuntu; if that job fails, coverage is never reported
- No test timeout enforcement (pytest-timeout not installed)
- MATLAB validation tests skip silently in CI rather than failing explicitly

---

## Refactoring Recommendations

### Previously Completed

| Priority | Item | What Changed |
|----------|------|-------------|
| P0 | Eliminate parallel pipelines | Deleted `_compute_profile_diss()` (219 lines), `_compute_profile_chi()` + `_build_chi_dataset()` (485 lines). `get_diss()` and `get_chi()` now delegate to scor160 L2→L3→L4. |
| P1 | Unify config modules | Created `config_base.py` with `ConfigManager`; rsi and perturb delegate to it |
| P1 | Extract shared FP07 transfer | `fp07_tau_batch()`, `fp07_transfer_batch()`, `default_tau_model()` in `chi/fp07.py` |
| P1 | Centralize channel patterns | `SH_PATTERN`, `AC_PATTERN`, `T_PATTERN`, `DT_PATTERN` in `rsi/helpers.py` |
| P2 | Remove dead code | Deleted `compute_nu()`, duplicate `KAPPA_T` |
| P2 | Extract file-write wrapper | `helpers.write_profile_results()` shared by diss and chi |
| P3 | Coverage threshold in CI | `--cov-fail-under=70` |

### Remaining

| Priority | Item | Est. Lines Saved | Notes |
|----------|------|-----------------|-------|
| P1 | Merge dataset builders | ~150 | `_build_diss_dataset` and `_build_chi_dataset` → shared parameterized builder |
| P1 | Consolidate FP07 noise scalar/batch | ~100 | Extract config unpacking and F/T-dependent computation to shared helpers |
| P1 | Split `p_to_L1()` | 0 (clarity) | 374 lines → separate channel classification, NetCDF variable creation loop |
| P2 | Unify L4 chi process functions | ~80 | `process_l4_chi_epsilon` + `process_l4_chi_fit` → one function with strategy param |
| P2 | Consolidate CSD functions | ~80 | `csd_odas()` should call `csd_matrix()` |
| P2 | Extract speed computation helper | ~40 | Replace 8 call sites of `_smooth_fall_rate` + `interp` + clamp |
| P2 | Split `process_l3()` and `process_l3_chi()` | 0 (clarity) | 178 and 246 lines; extract window building and per-section helpers |
| P2 | Parameterize scor160 CLI handlers | ~100 | 6 handlers → 1 parameterized function |
| P2 | Parameterize perturb CLI commands | ~50 | 3 handlers → 1 helper |
| P3 | Deduplicate config tests | ~300 (test) | 26 identical test functions → parameterized fixtures |
| P3 | Extract Goodman bias correction helper | ~15 | Duplicated in scalar and batch versions |
| P3 | Consolidate compare.py helpers | ~60 | Shared log-spectral comparison and report formatting |
| P3 | Add perturb docstrings | 0 | ~50% of public functions undocumented |

---

## Mathematical Correctness — A (unchanged)

All formulas verified against cited publications. No errors found.

- Nasmyth: Lueck improved fit coefficients correct
- Batchelor/Kraichnan: Correct per Dillon & Caldwell / Bogucki et al.
- MLE: Correct for chi-squared spectral estimates
- FP07: Single/double pole transfer functions correct
- Viscosity: Sharqawy coefficients match ODAS exactly
- Goodman: Wiener filter formulation correct with bias correction
- Spectral: Cosine window, FFT normalization, DOF all correct

---

## MATLAB/ODAS Agreement — A (unchanged)

307/307 epsilon + chi cross-validation tests pass across 30 .p files. Known difference: scalar spectra wavenumber vectors differ by ~5-15% due to speed computation method (profile boundaries, filter padding). Chi and epsilon values themselves agree.

---

## References

See [Bibliography](docs/bibliography.md) for full citations. Key references:

- Lueck, R. (2022a,b) -- Nasmyth spectrum improved fit, figure of merit
- Oakey, N. (1982) -- Batchelor constant Q_B = 3.7
- Bogucki, D. et al. (1997) -- Kraichnan constant Q_K = 5.26
- Dillon, T. & Caldwell, D. (1980) -- Batchelor spectrum formulation
- Ruddick, B. et al. (2000) -- MLE spectral fitting
- Peterson, A. & Fer, I. (2014) -- Iterative chi fitting algorithm
- Sharqawy, M. et al. (2010) -- Seawater viscosity
- Mudge, T. & Lueck, R. (1994) -- Deconvolution
- RSI TN-040, TN-042, TN-051, TN-061 -- Instrument technical notes
- ODAS MATLAB Library v4.5.1 -- Reference implementation
