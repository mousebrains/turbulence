# Code Quality Audit

**Date**: 2026-03-14 (post-improvement)
**Scope**: Full odas_tpw codebase — architecture, duplication, large functions, testing, infrastructure
**Prior work**: 14-item refactoring (5 phases, `4d8436d`), NaN guard fix (`28b3a08`), and 4-phase audit improvement (`db93190`). All 1599 tests pass. ATOMIX L1→L4 and L2→L4 benchmarks verified. MATLAB cross-validation (307 tests) passing.

---

## Overall Grade: A- (3.7 / 4.0)

Up from B+ (3.3). The 4-phase improvement addressed the major structural issues: deprecated functions removed from the hot path, `p_to_L1` split into a spec-list pattern, TypedDict/NamedTuple for key return types, structured logging in perturb, and pytest-xdist enabled. The NaN guard in `csd_matrix_batch` (the only P0 issue) was fixed in the prior commit. Mathematics and MATLAB agreement remain excellent. The remaining issues are moderate: `print()`-based status output in rsi/pipeline.py and convert.py, visualization code duplication between quick_look and diss_look, and 9 functions still over 150 lines (all scientifically complex or orchestrators).

### Grading Breakdown

| # | Category | Grade | What's Good | What's Lacking |
|---|----------|-------|-------------|----------------|
| 1 | Mathematical correctness | A | Every formula verified against publications | — |
| 2 | MATLAB/ODAS agreement | A | 307/307 cross-validation passing | Known speed difference in scalar spectra |
| 3 | Unit consistency | A | Consistent throughout | — |
| 4 | Architecture & duplication | A- | Single pipeline; shared builders; speed computation centralized via `smooth_speed_interp`; deprecation cycle complete; clean DAG (scor160→chi→rsi→perturb, no cycles) | Visualization code duplicated between quick_look.py and diss_look.py (~100 lines); FP07 noise scalar/batch share ~15 lines of T-dependent code |
| 5 | Large functions | B | 9 over 150 lines (down from 14); `p_to_L1` split (319→144 lines); `get_diss`/`get_chi` are now 5-line wrappers | 9 >150-line functions are scientifically complex (L3, L4, binary parsing, spectral) or orchestrators — further splitting would hurt readability |
| 6 | Dead code | A | 1 TODO marker; no unused functions; deprecated wrappers are thin (5 lines) | — |
| 7 | Test coverage | A- | 871 test functions, 54 files; 1599 passing; MATLAB + ATOMIX cross-validation | 11 silent skip calls; viewers smoke-tested only; no corrupt `.p` file tests |
| 8 | Public API | A- | 50 exports; clean package layering; deprecation cycle complete (CLI paths avoid warning) | Deprecated `get_diss`/`get_chi` still exported for backward compat |
| 9 | Type annotations | B+ | scor160 and chi well-typed; `ChannelsDict` TypedDict; `ChiEpsilonResult`/`ChiFitResult` NamedTuples; ~97% return type coverage | 17 modules suppressing 6 mypy error codes (NumPy false positives); 6 tuple returns in scor160 (despike, goodman, spectral) still untyped; viewers untyped |
| 10 | Docstrings | B+ | Algorithm functions excellent; GPS protocol methods documented; perturb private helpers documented | 4 primary APIs missing docstrings (`diss_look`, `quick_look`, `show`, `main`); ~15 `@property` accessors in io.py lack docstrings |
| 11 | Error handling | B+ | NaN guard in `csd_matrix_batch` fixed (all 6 ATOMIX datasets pass L1→L4); `warnings.warn` at NaN/clamp sites; structured logging in perturb pipeline | `print()`-based status in rsi/pipeline.py, convert.py, profile.py |
| 12 | Infrastructure | A | CI matrix (3 OS × 2 Python); ruff + mypy; pytest-xdist enabled (`-n auto`); 70% coverage threshold; codecov integration | MATLAB tests skip silently |

---

## Codebase Metrics

| Metric | Prior Audit | Current | Delta |
|--------|-------------|---------|-------|
| Source files | 61 | 61 | — |
| Source lines | 16,713 | 16,849 | +136 (new helpers + type defs) |
| Test files | 54 | 54 | — |
| Test lines | 12,518 | 12,518 | — |
| Test functions | 871 | 871 | — |
| Tests passing | 1,599 | 1,599 | — |
| Functions >80 lines | 49 | 50 | +1 (`_l1_variable_specs` replaced inline code) |
| Functions >150 lines | 10 | 9 | -1 |
| TODO/FIXME markers | 1 | 1 | — |
| Circular dependencies | 0 | 0 | — |

### Lines by Subpackage

| Subpackage | Source Lines | Files | Role |
|------------|-------------|-------|------|
| rsi | 7,398 | 20 | I/O, pipeline wrappers, CLI, viewers |
| perturb | 3,648 | 19 | Campaign processing |
| scor160 | 3,539 | 13 | ATOMIX benchmark, core pipeline |
| chi | 2,010 | 7 | Thermal dissipation |
| top-level | 254 | 2 | config_base |

---

## What This Improvement Fixed

| Phase | Items | Impact |
|-------|-------|--------|
| 1 (deprecation) | Extract `_compute_epsilon()`, `_compute_chi()`; thin wrappers for `get_diss`/`get_chi`; update `compute_diss_file`, `compute_chi_file`, and `perturb/pipeline.py` to call internal functions | CLI no longer emits `DeprecationWarning`; `get_diss`/`get_chi` shrunk to 5-line wrappers |
| 2 (p_to_L1 split) | `_create_l1_variables()` + `_l1_variable_specs()` extracted; 18 variable blocks → declarative spec list | `p_to_L1()`: 319→144 lines; spec list is self-documenting |
| 3 (type safety) | `ChannelsDict` TypedDict for `load_channels()`; `ChiEpsilonResult`/`ChiFitResult` NamedTuples in chi.py; GPS `lat()`/`lon()` docstrings | 97% return type coverage; chi fitting returns are named |
| 4A (logging) | `logging.getLogger(__name__)` in perturb/pipeline.py; 27 `print()` → `logger.info/warning/error`; `logging.basicConfig()` in cli.py | Structured logging; message format preserved |
| 4B (CI) | `-n auto` added to pytest command | Parallel test execution in CI |
| 4C (speed helper) | `smooth_speed_interp()` extracted to scor160/profile.py; adapter.py uses it | 6 inline lines → 1 function call; Butterworth filter centralized |

---

## Detailed Findings

### 4. Architecture & Duplication — A-

**Improved from B**. Deprecation cycle complete. Speed computation centralized. No circular dependencies. Module hierarchy is clean: `scor160` (leaf) → `chi` → `rsi` → `perturb`.

#### Remaining issues

**4a. Visualization code duplication (~100 lines)**

`_draw_chi_spectra()` appears in both `rsi/quick_look.py` (137 lines) and `rsi/diss_look.py` (101 lines) with near-identical matplotlib rendering logic. Could be extracted to a shared viewer helper.

**4b. FP07 noise: residual T-dependent duplication (~15 lines)**

`noise_thermchannel()` and `noise_thermchannel_batch()` in fp07.py still duplicate the R_ratio → scale factor computation. Minor — the scalar version has NaN-guarded warnings; the batch version clips silently.

**4c. Butter filter pattern (5+ files)**

`butter(1, f_c / (fs / 2.0))` + `filtfilt()` appears in scor160/l2.py, scor160/profile.py, chi/l2_chi.py. Not true duplication — each site has different parameters and types (HP vs LP, shear vs speed vs temperature) — but a centralized `_butter_filter(data, f_c, fs, btype)` helper could reduce boilerplate.

### 5. Large Functions — B

**Improved from C+**. `p_to_L1` cut from 319→144 lines. `get_diss`/`get_chi` are now 5-line wrappers (down from 248/207). 9 functions remain over 150 lines, all in defensible categories:

| Function | File | Lines | Category |
|----------|------|-------|----------|
| `run_pipeline()` | rsi/pipeline.py | 288 | Orchestrator (8 steps) |
| `_compute_epsilon()` | rsi/dissipation.py | 201 | Scientific pipeline |
| `process_file()` | perturb/pipeline.py | 180 | Orchestrator (8 stages) |
| `compute_eps_window()` | rsi/window.py | 163 | Scientifically dense |
| `process_l3()` | scor160/l3.py | 161 | Spectral pipeline |
| `_read()` | rsi/p_file.py | 160 | Binary parsing |
| `compute_chi_window()` | rsi/window.py | 156 | Scientifically dense |
| `_l1_variable_specs()` | rsi/convert.py | 155 | Declarative spec list |
| `run_pipeline()` | perturb/pipeline.py | 155 | Orchestrator |

The scientific core (L3, epsilon/chi window, binary parsing) and orchestrators are legitimately complex. `_l1_variable_specs()` is long because it's a flat list of 18 variable specs — mechanical but clear.

### 6. Dead Code — A

1 TODO marker (`perturb/pipeline.py:444`). No unused functions. `get_diss()` and `get_chi()` are thin deprecated wrappers — still exported for backward compatibility but not used internally.

### 7. Test Coverage — A-

**871 test functions across 54 files. 1599 passing, 10 skipped.**

| Subpackage | Test Files | Approx Tests | Quality |
|------------|-----------|-------|---------|
| scor160 | ~12 | ~200 | Good — unit + integration + ATOMIX L1→L4 |
| rsi | ~18 | ~350 | Good — unit + integration + CLI |
| chi | ~5 | ~150 | Good — unit + integration + MATLAB |
| perturb | ~12 | ~150 | Good — unit + integration |
| cross-validation | 4 | ~60 | Excellent — 30 .p files validated |
| config (shared) | 1 | ~54 | Parametrized over rsi + perturb |

**Strengths:**
- Comprehensive edge case testing (empty arrays, NaN, zero epsilon, short signals)
- Seeded RNG for reproducibility throughout
- `pytest.raises()` / `pytest.warns()` for error paths
- No flaky patterns (no time.sleep, no random seeds)

**Gaps:**
- 11 `pytest.skip()` calls for missing VMP data (graceful)
- Viewers smoke-tested only (no dedicated quick_look/diss_look tests)
- No tests for corrupt or truncated `.p` files
- 14 modules lack dedicated test files (most covered indirectly via integration tests)

### 8. Public API — A-

50 exports across 4 `__init__.py` files. Deprecation cycle complete: `compute_diss_file()` and `compute_chi_file()` now call `_compute_epsilon()` and `_compute_chi()` directly. `get_diss()`/`get_chi()` remain exported for backward compat but are thin wrappers that warn + delegate.

### 9. Type Annotations — B+

**Improved from B-**. `ChannelsDict` TypedDict replaces `dict[str, Any]` for `load_channels()` and related functions. `ChiEpsilonResult` and `ChiFitResult` NamedTuples replace 6/7-element bare tuples in chi.py. ~97% return type coverage (6 of ~200+ public functions missing annotations — all in viewers).

| Module Group | Coverage | Notes |
|-------------|----------|-------|
| scor160 | ~95% | Dataclasses fully typed |
| chi core | ~95% | Good; NamedTuples for fitting results |
| rsi core | ~95% | Good; TypedDict for channel loading |
| rsi pipeline | ~90% | Good |
| rsi viewers | ~0% | No type hints (6 functions) |
| perturb | ~95% | Good |

**Remaining opportunities:**
- 6 tuple returns in scor160 (despike, goodman, spectral) could be NamedTuples
- 10 dict-returning functions (compare, fp07_cal, etc.) could be TypedDicts
- 17 modules still suppress mypy error codes (driven by NumPy array arithmetic false positives — defensible)

### 10. Docstrings — B+

**Improved from B-**. GPS protocol methods now have 1-line docstrings on all 8 `lat()`/`lon()` implementations.

**Remaining gaps (minor):**
- 4 primary APIs missing docstrings: `diss_look`, `quick_look`, `show`, `scor160/cli.py:main`
- ~15 `@property` accessors in scor160/io.py (n_shear, n_time, etc.) lack docstrings — these are self-evident from naming

### 11. Error Handling — B+

**Improved from B-**. The P0 NaN guard in `csd_matrix_batch` is fixed — all 6 ATOMIX benchmark datasets now pass L1→L4. Structured logging in perturb/pipeline.py replaces `print()` calls.

**Remaining:**
- `print()` status output in rsi/pipeline.py (20 calls), rsi/convert.py (6 calls), rsi/profile.py (2 calls), rsi/helpers.py (1 call) — these are user-facing CLI output, not error handling, but could benefit from logging for consistency
- `ProcessPoolExecutor` errors in perturb/pipeline.py are logged via `logger.error()`

### 12. Infrastructure — A

**Improved from A-**. pytest-xdist enabled with `-n auto` in CI.

- CI: 3 OS × 2 Python versions, ruff lint, mypy typecheck, pytest-xdist, coverage ≥70%, codecov upload
- MATLAB linting with miss_hit
- Concurrency control (cancel-in-progress)
- All runtime + dev dependencies declared in pyproject.toml
- Entry points configured for 3 CLIs

---

## ATOMIX Benchmark Results (post-NaN fix)

### L2→L4 (all 6 datasets pass)

| Dataset | Spectra | L4 ε within factor 2 | Method Agreement |
|---------|---------|---------------------|-----------------|
| Epsilometer | 181 | N/A (ISR) | 98.9–100% |
| MSS Baltic | 61 | N/A (ISR) | 100% |
| MR1000 Minas Passage | 30 | 100% (4 probes) | 100% |
| VMP2000 Faroe Bank | 342 | N/A (ISR) | 100% |
| VMP250 Tidal (cs) | 32 | 96.9–100% | 93.8–96.9% |
| VMP250 Tidal | 32 | 100% | 96.9–100% |

### L1→L4 (all 6 datasets pass)

| Dataset | L2 Overlap | L4 ε within factor 2 | Status |
|---------|-----------|---------------------|--------|
| MR1000 Minas Passage | 100% | 100% (4 probes) | **PASS** |
| VMP2000 Faroe Bank | 100% | N/A (ISR, 100% agreement) | **PASS** |
| VMP250 Tidal (cs) | 100% | 96.9–100% | **PASS** |
| VMP250 Tidal | 99.8% | 100% | **PASS** |
| Epsilometer | 100% | N/A (ISR, 100% agreement) | **PASS** (NaN guard skips 1/131 bad windows) |
| MSS Baltic | 100% | N/A (ISR, 100% agreement) | **PASS** (NaN guard skips 3/33 bad windows) |

---

## Remaining Improvements (optional)

| Priority | Item | Est. Effort | Notes |
|----------|------|-------------|-------|
| P2 | Extract shared viewer helper for `_draw_chi_spectra` | Medium | ~100 lines duplicated between quick_look.py and diss_look.py |
| P2 | NamedTuples for scor160 tuple returns | Medium | 6 functions (despike, goodman, spectral) return 4–6-element tuples |
| P3 | Structured logging in rsi/pipeline.py, convert.py | Medium | ~30 print() calls → logging module |
| P3 | Centralized butter filter helper | Small | 5+ files duplicate `butter(1, f_c / (fs/2))` pattern |
| P3 | Viewer type annotations | Medium | 6 public functions in viewers have no type hints |
| P4 | Corrupt `.p` file test | Small | No tests for truncated/corrupt binary input |

---

## Mathematical Correctness — A (unchanged)

All formulas verified against cited publications. No errors found.

---

## MATLAB/ODAS Agreement — A (unchanged)

307/307 epsilon + chi cross-validation tests pass across 30 .p files. Known difference: scalar spectra wavenumber vectors differ by ~5-15% due to speed computation method (profile boundaries, filter padding).

---

## References

See [Bibliography](docs/bibliography.md) for full citations. Key references:

- Lueck, R. (2022a,b) — Nasmyth spectrum improved fit, figure of merit
- Oakey, N. (1982) — Batchelor constant Q_B = 3.7
- Bogucki, D. et al. (1997) — Kraichnan constant Q_K = 5.26
- Dillon, T. & Caldwell, D. (1980) — Batchelor spectrum formulation
- Ruddick, B. et al. (2000) — MLE spectral fitting
- Peterson, A. & Fer, I. (2014) — Iterative chi fitting algorithm
- Sharqawy, M. et al. (2010) — Seawater viscosity
- Fer, I. et al. (2024) — ATOMIX benchmark datasets
- RSI TN-040, TN-042, TN-051, TN-061 — Instrument technical notes
- ODAS MATLAB Library v4.5.1 — Reference implementation
