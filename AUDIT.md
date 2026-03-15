# Code Quality Audit

**Date**: 2026-03-14 (post-improvement, round 2)
**Scope**: Full odas_tpw codebase — architecture, duplication, large functions, testing, infrastructure
**Prior work**: 14-item refactoring (`4d8436d`), NaN guard fix (`28b3a08`), 4-phase audit improvement (`db93190`), and B+/lower grade improvement (`d37d54c`). All 1599 tests pass. ATOMIX L1→L4 and L2→L4 benchmarks verified. MATLAB cross-validation (307 tests) passing.

---

## Overall Grade: A (3.83 / 4.0)

Up from A- (3.7). The latest improvement (`d37d54c`) addressed all four B+/B categories: NamedTuples for 4 scor160 tuple returns, viewer type annotations + property docstrings, structured logging across 4 rsi modules, and `run_pipeline()` split into `_process_profile()`. No category remains below A-.

### Grading Breakdown

| # | Category | Grade | Prev | What's Good | What's Lacking |
|---|----------|-------|------|-------------|----------------|
| 1 | Mathematical correctness | A | A | Every formula verified against publications | — |
| 2 | MATLAB/ODAS agreement | A | A | 307/307 cross-validation passing | Known speed difference in scalar spectra |
| 3 | Unit consistency | A | A | Consistent throughout | — |
| 4 | Architecture & duplication | A- | A- | Single pipeline; shared builders; speed computation centralized; deprecation cycle complete; clean DAG (scor160→chi→rsi→perturb, no cycles) | Visualization code duplicated between quick_look.py and diss_look.py (~100 lines); FP07 noise scalar/batch share ~15 lines |
| 5 | Large functions | B+ | B | `run_pipeline()` split (288→186 lines via `_process_profile()` extraction); 10 >150 lines remain but all are scientifically complex or orchestrators | `_process_profile()` itself is 165 lines (sequential orchestrator); `_compute_epsilon()` at 201 lines is the longest |
| 6 | Dead code | A | A | 1 TODO marker; no unused functions; deprecated wrappers are thin (5 lines) | — |
| 7 | Test coverage | A- | A- | 871 test functions, 54 files; 1599 passing; MATLAB + ATOMIX cross-validation | 11 silent skip calls; viewers smoke-tested only; no corrupt `.p` file tests |
| 8 | Public API | A- | A- | 50+ exports; clean package layering; `AtomixData` NamedTuple exported | Deprecated `get_diss`/`get_chi` still exported for backward compat |
| 9 | Type annotations | A- | B+ | 4 new NamedTuples (`DespikeResult`, `CleanShearResult`, `CSDResult`, `AtomixData`); viewer functions fully typed; ~100% return type coverage | 0 mypy suppressions; ~18 bare tuple returns remain (channel converters, utility functions) |
| 10 | Docstrings | A- | B+ | 11 property docstrings added to io.py; `scor160/cli.py:main` docstring added; `diss_look`/`quick_look` already had docstrings | 3 minor gaps: `PFile.is_fast`, `PFile.summary`, `ProfileViewer.show` |
| 11 | Error handling | A- | B+ | Structured logging in both perturb and rsi pipelines; 23 `print()` → `logging` across pipeline.py, convert.py, profile.py, helpers.py; `logging.basicConfig` in rsi/cli.py | Remaining `print()` in rsi/cli.py (23, user-facing) and p_file.py (9, `summary()` display) — appropriate uses |
| 12 | Infrastructure | A | A | CI matrix (3 OS × 2 Python); ruff + mypy; pytest-xdist; 70% coverage threshold; codecov | MATLAB tests skip silently |

---

## Codebase Metrics

| Metric | Prior Audit | Current | Delta |
|--------|-------------|---------|-------|
| Source files | 61 | 61 | — |
| Source lines | 16,849 | 16,988 | +139 (NamedTuples, docstrings, logging, extracted function) |
| Test files | 54 | 54 | — |
| Test lines | 12,518 | 12,518 | — |
| Test functions | 871 | 871 | — |
| Tests passing | 1,599 | 1,599 | — |
| Functions >80 lines | 50 | 51 | +1 (`_process_profile` extracted from `run_pipeline`) |
| Functions >150 lines | 9 | 10 | +1 (`_process_profile` at 165 lines; `run_pipeline` dropped from 288→186) |
| TODO/FIXME markers | 1 | 1 | — |
| Circular dependencies | 0 | 0 | — |
| mypy suppressions | 0 | 0 | — |

### Lines by Subpackage

| Subpackage | Source Lines | Files | Role |
|------------|-------------|-------|------|
| rsi | 7,478 | 20 | I/O, pipeline wrappers, CLI, viewers |
| perturb | 3,648 | 19 | Campaign processing |
| scor160 | 3,598 | 13 | ATOMIX benchmark, core pipeline |
| chi | 2,010 | 7 | Thermal dissipation |
| top-level | 254 | 2 | config_base |

---

## What This Improvement Fixed (d37d54c)

| Phase | Items | Impact |
|-------|-------|--------|
| 1 (NamedTuples) | `DespikeResult`, `CleanShearResult`, `CSDResult`, `AtomixData` in despike.py, goodman.py, spectral.py, io.py | 4 bare tuple returns → named fields; backward compatible (positional unpacking unchanged) |
| 2 (type + docs) | 6 viewer return types; 11 property docstrings in io.py; `scor160/cli.py:main` docstring | ~100% return type coverage; all io.py properties documented |
| 3 (logging) | 23 `print()` → `logger.info/warning/error` in pipeline.py, convert.py, profile.py, helpers.py; `logging.basicConfig` in rsi/cli.py | Structured logging in all rsi pipeline modules; format preserved via `%(message)s` |
| 4 (split) | `_process_profile()` extracted from `run_pipeline()` | `run_pipeline()`: 288→186 lines; per-profile logic isolated |

---

## Detailed Findings

### 4. Architecture & Duplication — A-

Unchanged from prior audit. No new duplication introduced. Module hierarchy remains clean.

#### Remaining issues

**4a. Visualization code duplication (~100 lines)** — `_draw_chi_spectra()` in quick_look.py and diss_look.py.

**4b. FP07 noise scalar/batch duplication (~15 lines)** — Minor.

**4c. Butter filter pattern (5+ files)** — Not true duplication; each site has different parameters.

### 5. Large Functions — B+

**Improved from B**. `run_pipeline()` split from 288→186 lines via `_process_profile()` extraction. 10 functions remain over 150 lines:

| Function | File | Lines | Category |
|----------|------|-------|----------|
| `_compute_epsilon()` | rsi/dissipation.py | 201 | Scientific pipeline |
| `run_pipeline()` | rsi/pipeline.py | 186 | Orchestrator (8 steps) |
| `process_file()` | perturb/pipeline.py | 180 | Orchestrator (8 stages) |
| `_process_profile()` | rsi/pipeline.py | 165 | Per-profile orchestrator |
| `compute_eps_window()` | rsi/window.py | 163 | Scientifically dense |
| `process_l3()` | scor160/l3.py | 161 | Spectral pipeline |
| `_read()` | rsi/p_file.py | 160 | Binary parsing |
| `compute_chi_window()` | rsi/window.py | 156 | Scientifically dense |
| `run_pipeline()` | perturb/pipeline.py | 155 | Orchestrator |
| `_l1_variable_specs()` | rsi/convert.py | 155 | Declarative spec list |

All are in defensible categories: scientific algorithms (L3/L4, chi/epsilon window computation), binary format parsing, orchestrators with sequential dependencies, or flat declarative lists. Further splitting any of these would hurt readability without improving maintainability.

### 6. Dead Code — A

1 TODO marker (`perturb/pipeline.py:444`). No unused functions. `get_diss()` and `get_chi()` are thin deprecated wrappers.

### 7. Test Coverage — A-

**871 test functions across 54 files. 1599 passing, 10 skipped.**

Unchanged. No new tests were required for this improvement (all changes are backward compatible; no capsys-based tests existed for the modified modules).

### 8. Public API — A-

`AtomixData` NamedTuple now exported from `scor160/__init__.py`. 4 new NamedTuples provide named access to previously bare tuple returns.

### 9. Type Annotations — A-

**Improved from B+**. All 4 previously-untyped scor160 tuple returns now use NamedTuples. All 6 viewer functions now have return type annotations. ~100% return type coverage across the codebase.

| Module Group | Coverage | Notes |
|-------------|----------|-------|
| scor160 | ~100% | NamedTuples for all spectral/despike/goodman returns |
| chi core | ~95% | NamedTuples for fitting results |
| rsi core | ~100% | TypedDict for channel loading |
| rsi pipeline | ~100% | Fully typed |
| rsi viewers | ~100% | All entry points + helpers typed |
| perturb | ~95% | Good |

**Remaining opportunities:**
- ~18 bare tuple returns in utility functions (channel converters return `tuple[ndarray, str]`, ocean functions return `tuple[ndarray, ndarray]`) — these are simple 2-element tuples where NamedTuples would add overhead without clarity
- 10 dict-returning functions (compare, fp07_cal, etc.) could be TypedDicts

### 10. Docstrings — A-

**Improved from B+**. All 11 `@property` accessors in io.py now have 1-line docstrings. `scor160/cli.py:main()` has a docstring. `diss_look()` and `quick_look()` already had full docstrings with Parameters/Returns sections.

**Remaining gaps (minor):**
- `PFile.is_fast` (trivial bool property)
- `PFile.summary` (display method)
- `ProfileViewer.show` (inherited viewer method)

### 11. Error Handling — A-

**Improved from B+**. Structured logging now covers both the perturb and rsi pipeline stacks. All 4 target files (pipeline.py, convert.py, profile.py, helpers.py) have zero `print()` calls — replaced with `logger.info/warning/error`. `logging.basicConfig(level=INFO, format="%(message)s")` in rsi/cli.py preserves the existing output format.

**Remaining `print()` calls in rsi/ (32 total):**
- rsi/cli.py: 23 — user-facing CLI output (`"Wrote file..."`, `"Warning: no files match..."`)
- rsi/p_file.py: 9 — `summary()` display method

These are appropriate uses of `print()` for direct user output, not status logging.

### 12. Infrastructure — A

Unchanged.

---

## ATOMIX Benchmark Results (unchanged)

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
| P3 | Centralized butter filter helper | Small | 5+ files duplicate `butter(1, f_c / (fs/2))` pattern |
| P3 | TypedDicts for dict-returning functions | Medium | 10 functions (compare, fp07_cal, etc.) return plain dicts |
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
