# Code Quality Audit

**Date**: 2026-03-14 (fresh audit, round 3)
**Scope**: Full odas_tpw codebase — architecture, duplication, large functions, testing, infrastructure
**Prior work**: 14-item refactoring (`4d8436d`), NaN guard fix (`28b3a08`), 4-phase audit improvement (`db93190`), B+/lower grade improvement (`d37d54c`). All 1599 tests pass. ATOMIX L1→L4 and L2→L4 benchmarks verified. MATLAB cross-validation (307 tests) passing.

---

## Overall Grade: A- (3.75 / 4.0)

All major structural improvements are in place: NamedTuples for scor160 tuple returns, viewer type annotations, property docstrings, structured logging across rsi modules, `run_pipeline()` split. No category is below B+. The remaining gaps are moderate: 10 functions over 150 lines (all scientifically complex or orchestrators), 82 mypy errors (mostly in viewer code), 8 `@property` methods in chi/ without docstrings.

### Grading Breakdown

| # | Category | Grade | What's Good | What's Lacking |
|---|----------|-------|-------------|----------------|
| 1 | Mathematical correctness | A | Every formula verified against publications | — |
| 2 | MATLAB/ODAS agreement | A | 307/307 cross-validation passing | Known speed difference in scalar spectra |
| 3 | Unit consistency | A | Consistent throughout | — |
| 4 | Architecture & duplication | A- | Single pipeline; shared builders; speed computation centralized; clean DAG (scor160→chi→rsi→perturb, no cycles) | Visualization code duplicated between quick_look.py and diss_look.py (~100 lines) |
| 5 | Large functions | B+ | `run_pipeline()` split (288→186 lines via `_process_profile()`); 10 >150 lines remain but all are scientifically complex or orchestrators | `_compute_epsilon()` at 201 lines is longest; `_process_profile()` at 165 lines |
| 6 | Dead code | A | 1 TODO marker; no unused functions; deprecated wrappers thin (5 lines) | — |
| 7 | Test coverage | A- | 871 test functions, 54 files; 1599 passing; MATLAB + ATOMIX cross-validation | 10 skipped tests; viewers smoke-tested only; no corrupt `.p` file tests |
| 8 | Public API | A- | 50+ exports; clean package layering; `AtomixData` NamedTuple exported | Deprecated `get_diss`/`get_chi` still exported for backward compat |
| 9 | Type annotations | A- | 4 NamedTuples (`DespikeResult`, `CleanShearResult`, `CSDResult`, `AtomixData`); all public functions have return types; 0 `type: ignore` suppressions | 82 mypy errors in 20 files (mostly viewer code: None indexing, type confusion); 8 bare tuple returns in scor160 (5 private, 3 public) |
| 10 | Docstrings | A- | All io.py properties documented; algorithm functions excellent; `diss_look`/`quick_look` have full docstrings | 8 `@property` methods in chi/ without docstrings; 3 minor gaps: `PFile.is_fast`, `PFile.summary`, `ProfileViewer.show` |
| 11 | Error handling | A- | Structured logging in both perturb and rsi pipelines; 23 `print()` → `logging` across pipeline.py, convert.py, profile.py, helpers.py | Remaining `print()` in rsi/cli.py (23) and p_file.py (9) — appropriate for CLI/display |
| 12 | Infrastructure | A | CI matrix (3 OS × 2 Python); ruff + mypy; pytest-xdist; 70% coverage threshold; codecov | 85 ruff warnings (25 line-length, 22 unused vars, 12 import sort); MATLAB tests skip silently |

---

## Codebase Metrics

| Metric | Value |
|--------|-------|
| Source files | 61 |
| Source lines | 16,988 |
| Test files | 54 |
| Test lines | 12,518 |
| Test functions | 871 |
| Tests passing | 1,599 |
| Tests skipped | 10 |
| Functions >80 lines | 51 |
| Functions >150 lines | 10 |
| TODO/FIXME markers | 1 |
| Circular dependencies | 0 |
| `type: ignore` suppressions | 0 |
| mypy errors | 82 (in 20 files) |
| ruff warnings | 85 |
| `print()` calls in src/ | 46 (37 in CLI modules, 9 in p_file.py) |

### Lines by Subpackage

| Subpackage | Source Lines | Files | Role |
|------------|-------------|-------|------|
| rsi | 7,478 | 20 | I/O, pipeline wrappers, CLI, viewers |
| perturb | 3,648 | 19 | Campaign processing |
| scor160 | 3,598 | 13 | ATOMIX benchmark, core pipeline |
| chi | 2,010 | 7 | Thermal dissipation |
| top-level | 254 | 2 | config_base |

---

## Detailed Findings

### 4. Architecture & Duplication — A-

Module hierarchy clean. No circular dependencies. Dependency graph: `scor160` ← `chi` ← `rsi` ← `perturb`.

#### Remaining issues

- **Visualization duplication (~100 lines)** — `_draw_chi_spectra()` in quick_look.py and diss_look.py.
- **FP07 noise scalar/batch duplication (~15 lines)** — Minor.
- **Butter filter pattern (5+ files)** — Not true duplication; each site has different parameters.

### 5. Large Functions — B+

10 functions over 150 lines:

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

All are in defensible categories: scientific algorithms, binary format parsing, orchestrators with sequential dependencies, or flat declarative lists.

### 6. Dead Code — A

1 TODO marker (`perturb/pipeline.py:444`). No unused functions. `get_diss()` and `get_chi()` are thin deprecated wrappers with `DeprecationWarning`.

### 7. Test Coverage — A-

871 test functions across 54 files. 1599 passing, 10 skipped.

### 8. Public API — A-

`AtomixData` NamedTuple exported from `scor160/__init__.py`. 4 NamedTuples provide named access to previously bare tuple returns. Deprecated `get_diss`/`get_chi` still exported with warnings.

### 9. Type Annotations — A-

All public functions have return type annotations. 0 `type: ignore` suppressions. 4 NamedTuples for major scor160 returns.

**Mypy errors (82 in 20 files):**
- `diss_look.py` (~20 errors): `None` indexing without guards
- `quick_look.py` (~5 errors): variable typed as `int` used as `list`
- `perturb/pipeline.py` (2 errors): `Sequence[str]` vs `list[str]`
- Remaining: NumPy type inference, ndarray attribute resolution

**Bare tuple returns in scor160/ (8 remaining):**
- 5 private: `_single_despike`, `_build_window_arrays`, `_estimate_epsilon`, `_variance_method`, `_inertial_subrange`
- 3 public: `clean_shear_spec_batch`, `buoyancy_freq`, `compute_speed_fast`

### 10. Docstrings — A-

All 11 `@property` accessors in io.py documented. Algorithm functions have excellent docstrings. `diss_look()` and `quick_look()` have full Parameters/Returns sections.

**Gaps:**
- 8 `@property` methods in chi/ data classes without docstrings: `l2_chi.py` (3), `l3_chi.py` (3), `l4_chi.py` (2)
- 3 minor: `PFile.is_fast`, `PFile.summary`, `ProfileViewer.show`

### 11. Error Handling — A-

Structured logging covers both perturb and rsi pipeline stacks. All 4 target files (pipeline.py, convert.py, profile.py, helpers.py) use `logger`. `logging.basicConfig(level=INFO, format="%(message)s")` in rsi/cli.py preserves output format.

**Remaining `print()` calls (46 total):**
- rsi/cli.py: 23 — user-facing CLI output
- perturb/cli.py: 14 — user-facing CLI output
- rsi/p_file.py: 9 — `summary()`/`info()` display methods

These are appropriate uses of `print()` for direct user output.

### 12. Infrastructure — A

CI matrix (3 OS × 2 Python). Ruff + mypy configured. pytest-xdist enabled. 70% coverage threshold. Codecov integration.

**Ruff warnings (85 total):** 25 line-length (E501), 22 unused unpacked vars (RUF059, mostly in tests), 12 import sort (I001), 12 unused locals (F841), 6 unused imports (F401), 4 misc. 21 auto-fixable.

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
| P2 | Fix 82 mypy errors (viewer None guards, type confusion) | Medium | Mostly diss_look.py and quick_look.py |
| P2 | Add docstrings to 8 chi/ `@property` methods | Small | l2_chi.py, l3_chi.py, l4_chi.py |
| P2 | Extract shared viewer helper for `_draw_chi_spectra` | Medium | ~100 lines duplicated |
| P3 | Fix 85 ruff warnings (line length, unused vars, import sort) | Small | 21 auto-fixable |
| P3 | NamedTuples for 3 public bare tuple returns in scor160 | Small | `clean_shear_spec_batch`, `buoyancy_freq`, `compute_speed_fast` |
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
