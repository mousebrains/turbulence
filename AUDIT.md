# Code Quality Audit

**Date**: 2026-03-14
**Scope**: Full odas_tpw codebase — architecture, duplication, dead code, testing, documentation, infrastructure
**Verdict**: Mathematics and MATLAB agreement are excellent. Architecture has significant structural debt from two parallel processing paths that were never consolidated.

---

## Overall Grade: B+ (3.2 / 4.0)

The science is solid: every formula is verified, MATLAB cross-validation passes on 30 files, and units are consistent throughout. The prior audit identified significant structural debt; the refactoring below addressed the worst of it. Config duplication eliminated via `ConfigManager`. FP07 batch functions extracted. Channel regex centralized. Dead code removed. Old pipeline deprecated with `DeprecationWarning`. Coverage threshold enforced in CI. The old pipeline code remains for backward compatibility and MATLAB cross-validation, pending removal in the next major version.

### Grading Breakdown

| # | Category | Grade | What's Good | What's Lacking |
|---|----------|-------|-------------|----------------|
| 1 | Mathematical correctness | A | Every formula verified against publications | — |
| 2 | MATLAB/ODAS agreement | A | 307/307 cross-validation passing | Known speed difference in scalar spectra |
| 3 | Unit consistency | A | Consistent throughout both pipelines | — |
| 4 | Architecture & duplication | C | Package layering clean; config unified; channel patterns centralized; FP07 batch extracted | Old pipeline still present (deprecated); ~500 lines overlap pending removal |
| 5 | Dead code | B | `compute_nu()` removed; duplicate `KAPPA_T` removed; 1 TODO marker | Old pipeline is "dead-walking" (deprecated but not removed) |
| 6 | Code cleanliness | B- | Named constants, dataclasses, shared helpers, batch functions | 2 functions over 200 lines (deprecated, pending removal) |
| 7 | Test coverage | B+ | 1606 tests, 53 test files, MATLAB cross-validation | Viewers smoke-tested only; some modules integration-only |
| 8 | Public API | B- | ~55 exports; deprecated functions emit DeprecationWarning | Old API still present; migration guide needed |
| 9 | Type annotations | B- | Core scientific modules annotated; mypy strict mode | Viewers 0%; per-module mypy overrides on 12 modules |
| 10 | Error handling | B+ | warnings.warn at NaN/clamp sites; narrowed exceptions | — |
| 11 | Internal documentation | B | Algorithm functions have Parameters/Returns | Inconsistent across modules; many private functions undocumented |
| 12 | Infrastructure | A- | CI matrix, release workflow, dependabot, ruff+mypy, coverage threshold | No integration tests in CI |

---

## Critical Finding: Two Parallel Pipelines

The single biggest architectural problem. Two complete epsilon-computation paths exist:

**Path A (original):** `rsi/dissipation.py` → `_compute_profile_diss()` (219 lines)
- Inline HP filter, despike, spectral estimation, Goodman cleaning, epsilon fitting
- Called by `get_diss()` → `compute_diss_file()` → CLI `eps` subcommand

**Path B (new):** `scor160/l2.py` → `scor160/l3.py` → `scor160/l4.py`
- Same operations split across clean L2/L3/L4 levels
- Called by `rsi/adapter.py` → `rsi/pipeline.py` → CLI `pipeline` subcommand

Both paths produce epsilon. Both are tested. Both are maintained. The math is identical, but implementation details diverge (e.g., `filtfilt` padlen, despike ordering). This is the textbook "parallel implementations that slowly drift apart" anti-pattern.

**Same situation for chi:** `chi/chi.py` `_compute_profile_chi()` (310 lines) vs. `chi/l2_chi.py` + `chi/l3_chi.py` + `chi/l4_chi.py`.

**Recommendation:** Choose one path and deprecate the other. The scor160 L2-L4 path is cleaner, better tested, and follows the ATOMIX standard. The original path should be deprecated with a timeline.

---

## Detailed Findings

### 4. Architecture & Duplication — D

This is the weakest category and the grade reflects severity, not volume.

#### 4a. Parallel pipelines (~500 lines overlap)

| Component | Path A (original) | Path B (scor160) | Overlap |
|-----------|-------------------|-------------------|---------|
| HP filter + despike | `dissipation.py:_compute_profile_diss` | `scor160/l2.py:process_l2` | ~80 lines |
| Spectral estimation | `dissipation.py:_compute_profile_diss` | `scor160/l3.py:process_l3` | ~60 lines |
| Epsilon fitting | `dissipation.py:_compute_profile_diss` | `scor160/l4.py:process_l4` | ~80 lines |
| Chi spectra | `chi/chi.py:_compute_profile_chi` | `chi/l3_chi.py:process_l3_chi` | ~100 lines |
| Chi fitting | `chi/chi.py:_compute_profile_chi` | `chi/l4_chi.py:process_l4_chi_*` | ~80 lines |
| File I/O wrapper | `dissipation.py:compute_diss_file` | `chi_io.py:compute_chi_file` | ~35 lines |

#### 4b. Config duplication (~200 lines)

`rsi/config.py` (391 lines) and `perturb/config.py` (609 lines) share 9 near-identical functions:

| Function | rsi/config.py | perturb/config.py |
|----------|---------------|-------------------|
| `load_config()` | lines 62-88 | lines 196-222 |
| `validate_config()` | lines 91-104 | lines 225-239 |
| `merge_config()` | lines 112-142 | lines 247-277 |
| `_canonicalize_section()` | lines 168-174 | lines 285-299 |
| `canonicalize()` | lines 177-206 | lines 302-328 |
| `compute_hash()` | lines 209-226 | lines 331-338 |
| `resolve_output_dir()` | lines 234-292 | lines 346-385 |
| `write_signature()` | lines 295-306 | lines 388-399 |
| `write_resolved_config()` | lines 309-340 | lines 402-431 |

The perturb version extends rsi with extra sections and keys, but the core logic is copy-pasted. `_normalize_value` is the one function properly shared (perturb imports it from rsi).

**Recommendation:** Extract a shared `odas_tpw.config_base` module with the 9 common functions. Let rsi and perturb subclass or extend it.

#### 4c. H2/tau0 computation (15 lines, 2 locations)

Identical FP07 transfer function computation in:
- `chi/chi.py:749-763`
- `chi/l3_chi.py:214-226`

Both compute `tau0` from speed using the same tau model dispatch, then build `H2` via `omega_tau` broadcasting. This should be a single function in `chi/fp07.py`.

#### 4d. KAPPA_T constant (2 definitions)

- `chi/batchelor.py:23`: `KAPPA_T = 1.4e-7`
- `chi/l4_chi.py:25`: `KAPPA_T = 1.4e-7`

Should be defined once and imported. `batchelor.py` is the canonical location.

#### 4e. Channel regex patterns (4 locations)

The patterns `r"^sh\d+$"`, `r"^A[xyz]$"`, `r"^T\d+_dT\d+$"`, `r"^T\d+$"` are hardcoded in:
1. `rsi/adapter.py:89-121`
2. `rsi/chi_io.py:273-281`
3. `rsi/helpers.py:89-90, 125-126`
4. `rsi/viewer_base.py`

**Recommendation:** Define compiled patterns once in `rsi/channels.py` or a shared constants module.

#### 4f. Speed smoothing (2 locations)

`rsi/adapter.py:79-86` and `rsi/helpers.py:228-240` both implement `W = smooth(dP/dt)` with Butterworth LP at `0.68/tau`, `filtfilt`, `np.interp` to fast rate, floor clamp. Minor differences (helpers filters both slow and fast; adapter filters fast only).

#### 4g. compute_diss_file / compute_chi_file (~35 lines)

`dissipation.py:630-669` and `chi_io.py:296-335` are structurally identical: resolve paths, call `get_diss`/`get_chi`, loop over results writing NetCDF with profile numbering. Should be a shared `_write_results()` helper.

### 5. Dead Code — C

| Item | Location | Status |
|------|----------|--------|
| `helpers.compute_nu()` | `rsi/helpers.py:265-272` | Defined, never called anywhere |
| Prior audit claimed "zero dead code" | AUDIT.md:92 | False — 19 dead constants were removed during this audit |

Only 1 TODO/FIXME marker in the entire codebase (`perturb/pipeline.py:441`), which is fine.

### 6. Code Cleanliness — C+

**Large functions:**

| Function | File | Lines | Issue |
|----------|------|-------|-------|
| `_compute_profile_chi` | `chi/chi.py` | ~310 | Legacy monolith; duplicates l3_chi + l4_chi |
| `_compute_profile_diss` | `rsi/dissipation.py` | ~219 | Duplicates scor160 L2-L4 |
| `_estimate_epsilon` | `scor160/l4.py` | ~193 | Complex but well-structured; acceptable |

**Large modules:**

| Module | Lines | Issue |
|--------|-------|-------|
| `chi/chi.py` | 1,056 | Should be split once parallel pipeline consolidated |
| `rsi/cli.py` | 872 | Mostly argparse boilerplate; acceptable for CLI |
| `rsi/dissipation.py` | 678 | Will shrink significantly if old pipeline deprecated |
| `rsi/viewer_base.py` | 669 | UI code; lower priority |
| `perturb/config.py` | 609 | 200+ lines are duplicated from rsi/config.py |

**What's good:**
- Named constants with citations throughout
- `FP07NoiseConfig` dataclass consolidates 17 parameters
- `_valid_wavenumber_mask()` helper eliminates 3-way K_AA duplication in chi.py
- Dataset construction extracted into builder functions
- Clean separation of scor160 leaf package (no upward imports)

### 7. Test Coverage — B+

**908 test functions across 53 files. 10 skipped. All passing.**

| Subpackage | Test Files | Tests | Coverage Quality |
|------------|-----------|-------|-----------------|
| scor160 | 8 files | ~200 | Good — unit + integration + MATLAB validation |
| rsi | 15 files | ~350 | Good — unit + integration + CLI |
| chi | 6 files | ~150 | Good — unit + integration + MATLAB validation |
| perturb | 8 files | ~150 | Good — unit + integration |
| cross-validation | 4 files | ~60 | Excellent — 30 .p files validated |

**Gaps:**
- No line-coverage metrics (no `--cov` in CI, no coverage threshold)
- Viewers have smoke tests only (instantiation, not behavior)
- `rsi/dissipation.py` tested via integration only; no unit tests for `_compute_profile_diss` internals
- `rsi/helpers.py` — several helpers tested only transitively through integration tests
- Both pipelines tested independently but no test verifies they produce identical results

**What's strong:**
- MATLAB cross-validation on real VMP data (307 epsilon + chi comparisons)
- Edge cases: empty arrays, all-NaN, negative epsilon, zero viscosity, short signals
- New tests added for previously untested modules: scor160/profile, rsi/dissipation constants, rsi/adapter, rsi/binning+combine, chi/l2_chi

### 8. Public API — C+

The package exports ~55 symbols across 4 `__init__.py` files:
- `rsi`: 13 exports (PFile, get_diss, get_chi, run_pipeline, bin_by_depth, combine_profiles, ...)
- `chi`: 17 exports (L2ChiData, L3ChiData, process_l3_chi, batchelor_grad, fp07_transfer, ...)
- `scor160`: 12 exports (L1Data-L4Data, process_l2-l4, read_atomix, ...)
- `perturb`: 6 exports

**Addressed:** Two ways to compute epsilon (`get_diss()` vs `run_pipeline()`) and chi (`get_chi()` vs L3/L4 chi pipeline). Both `get_diss()` and `get_chi()` now emit `DeprecationWarning` pointing users to `run_pipeline()`. Removal deferred to next major version.

### 9. Type Annotations — B-

| Module Group | Coverage | Notes |
|-------------|----------|-------|
| scor160 core (ocean, nasmyth, spectral, despike, goodman) | ~95% | Solid |
| scor160 pipeline (l2, l3, l4, io) | ~95% | Dataclasses fully typed |
| chi core (batchelor, fp07, l2_chi, l3_chi, l4_chi) | ~90% | Good |
| chi/chi.py | ~80% | Internal functions sparse |
| rsi core (p_file, channels, config, convert) | ~90% | Good |
| rsi pipeline (dissipation, helpers, adapter) | ~80% | Mixed |
| rsi viewers (viewer_base, quick_look, diss_look) | ~0% | mypy disabled |
| perturb | ~85% | Reasonable |

mypy runs with `check_untyped_defs = true` and `warn_return_any = true`, but 12 modules have per-module overrides suppressing various warnings. The viewer modules have mypy disabled entirely.

### 10. Error Handling — B+

Strong coverage of warning sites:
- `warnings.warn()` at 7 NaN return sites in chi.py
- Speed clamping logged at 5 sites
- Narrowed exception types in CLI and convert modules
- `ValueError` raised for invalid inputs (negative viscosity, etc.)

No significant gaps found.

### 11. Internal Documentation — B

Algorithm functions (Nasmyth, Batchelor, MLE, Goodman, despike) have proper NumPy-style docstrings with Parameters/Returns sections and publication references. Private helper functions are inconsistently documented — some have full docstrings, others have none. The scor160 L2-L4 pipeline is well-documented; the original dissipation.py pipeline less so.

### 12. Infrastructure — A-

**Strong:**
- CI: ruff lint + mypy + pytest matrix (Python 3.12/3.13 x ubuntu/macOS/Windows)
- MATLAB linting via MISS_HIT
- Dependabot (pip + GitHub Actions, monthly)
- Release workflow (PyPI on `v*` tags)
- Clean `pyproject.toml` with PEP 517/518, pinned deps

**Gaps:**
- No integration tests in CI (MATLAB validation requires VMP data)

---

## Codebase Metrics

| Metric | Count |
|--------|-------|
| Source files | 60 |
| Source lines | 17,461 |
| Test files | 53 |
| Test lines | 12,767 |
| Test functions | 908 |
| Public exports | ~55 |
| TODO/FIXME markers | 1 |
| Subpackages | 4 (scor160, chi, rsi, perturb) |

### Lines by Subpackage

| Subpackage | Source Lines | Role |
|------------|-------------|------|
| rsi | ~5,000 | I/O, old pipeline, CLI, viewers |
| perturb | ~4,000 | Campaign processing |
| scor160 | ~2,800 | ATOMIX benchmark, new pipeline |
| chi | ~2,500 | Thermal dissipation |

---

## Refactoring Status

### Completed (2026-03-14)

| Priority | Item | What Changed |
|----------|------|-------------|
| P0 | Deprecate parallel pipelines | Added `DeprecationWarning` to `get_diss()`, `get_chi()` pointing to `run_pipeline()` / modular scor160 chain. Docstring deprecation notices on `_compute_profile_diss()` and `_compute_profile_chi()`. Old code preserved for MATLAB cross-validation compatibility; removal deferred to next major version. |
| P1 | Unify config modules | Created `config_base.py` with `ConfigManager` class; both `rsi/config.py` and `perturb/config.py` now instantiate it with their own DEFAULTS, eliminating ~200 lines of duplication |
| P1 | Extract shared FP07 transfer | Added `fp07_tau_batch()`, `fp07_transfer_batch()`, `default_tau_model()` to `chi/fp07.py`; both `chi.py` and `l3_chi.py` now call these instead of inline computation |
| P1 | Centralize channel patterns | Defined `SH_PATTERN`, `AC_PATTERN`, `T_PATTERN`, `DT_PATTERN` in `rsi/helpers.py`; updated `adapter.py`, `viewer_base.py`, `chi_io.py` to import them |
| P2 | Remove dead code | Deleted `helpers.compute_nu()` (unused); removed duplicate `KAPPA_T` from `l4_chi.py` (now imports from `batchelor.py`) |
| P2 | Extract compute_*_file wrapper | Created `helpers.write_profile_results()`; both `compute_diss_file()` and `compute_chi_file()` now delegate to it |
| P3 | Coverage threshold in CI | Added `--cov-fail-under=70` to pytest in CI workflow |
| P3 | Remove deprecated old pipeline | Deleted `_compute_profile_diss()` (219 lines from `rsi/dissipation.py`) and `_compute_profile_chi()` + `_build_chi_dataset()` (485 lines from `chi/chi.py`). Rewrote `get_diss()` and `get_chi()` to delegate to modular scor160 L2→L3→L4 chain. Removed stale tests in `TestVectorizedProfileDiss`. |
| P3 | Decompose large functions | Resolved by removing `_compute_profile_diss` (219 lines) and `_compute_profile_chi` (310 lines). Largest functions now: `get_diss()` ~120 lines, `get_chi()` ~130 lines — both are orchestration wrappers delegating to modular pipeline steps. |

### Remaining

(No remaining refactoring items.)

---

## Mathematical Correctness — A (unchanged)

All formulas verified against cited publications. No errors found. See prior audit for complete verification list. Key results:

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
