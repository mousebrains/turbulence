# Code Quality Audit

**Date**: 2026-03-14 (post-refactoring)
**Scope**: Full odas_tpw codebase — architecture, duplication, large functions, testing, infrastructure
**Prior work**: 14-item refactoring plan executed across 5 phases. ~530 net source lines eliminated, 26 duplicate test functions consolidated. All 1599 tests pass. ATOMIX L1→L4 and L2→L4 benchmarks verified unchanged.

---

## Overall Grade: B+ (3.3 / 4.0)

The refactoring addressed the worst duplication (dataset builders, CSD, FP07 noise, CLI handlers, config tests) and improved clarity (l3_chi, l3, goodman, compare). Mathematics and MATLAB agreement remain excellent. The codebase is cleaner but carries real structural debt: 10 functions still exceed 150 lines, `p_to_L1` is still 319 lines, the deprecated `get_diss()`/`get_chi()` remain in the hot path, and L1→L4 crashes on 2 of 6 ATOMIX datasets due to missing NaN guards.

### Grading Breakdown

| # | Category | Grade | What's Good | What's Lacking |
|---|----------|-------|-------------|----------------|
| 1 | Mathematical correctness | A | Every formula verified against publications | — |
| 2 | MATLAB/ODAS agreement | A | 307/307 cross-validation passing | Known speed difference in scalar spectra |
| 3 | Unit consistency | A | Consistent throughout | — |
| 4 | Architecture & duplication | B | Single pipeline; duplication cut ~50%; shared builders; consolidated CSD/FP07/CLI | Speed computation still scattered (6 sites); fp07 scalar/batch still share ~20 lines of T-dependent code; `get_diss`/`get_chi` deprecated but still in CLI hot path |
| 5 | Large functions | C+ | 55→49 over 80 lines; 14→10 over 150 lines; l3_chi cut from 246→110; dataset builders from 175→62 | `p_to_L1` still 319 lines; `run_pipeline` still 288; `get_diss`/`get_chi` still 248/207 |
| 6 | Dead code | A- | 1 TODO marker; no unused functions | Deprecated `get_diss`/`get_chi` still the only path from CLI |
| 7 | Test coverage | A- | 871 test functions, 54 files; 0 duplicate config tests; MATLAB + ATOMIX cross-validation | 11 silent skip calls; viewers smoke-tested only; no corrupt `.p` file tests |
| 8 | Public API | B | 50 exports; clean package layering | Deprecation cycle incomplete: CLI → `compute_diss_file` → `get_diss()` → warns |
| 9 | Type annotations | B- | scor160 and chi well-typed; mypy in CI | 17 modules suppressing 6 error codes; `load_channels()` returns `dict[str, Any]`; viewers untyped |
| 10 | Docstrings | B- | Algorithm functions excellent; all perturb private helpers now documented | 10 GPS protocol methods lack docstrings; `p_to_L1` internal logic underdocumented |
| 11 | Error handling | B- | warnings.warn at NaN/clamp sites; narrowed exceptions | **L1→L4 crashes on 2/6 ATOMIX datasets** (NaN in L2 shear hits `scipy.signal.detrend`); print-based logging in perturb |
| 12 | Infrastructure | A- | CI matrix (3 OS × 2 Python); ruff + mypy; 70% coverage; release workflow | pytest-xdist installed but unused; MATLAB tests skip silently |

---

## Codebase Metrics

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Source files | 61 | 61 | — |
| Source lines | 17,077 | 16,713 | -364 |
| Test files | 53 | 54 | +1 |
| Test lines | 12,648 | 12,518 | -130 |
| Test functions | 901 | 871 | -30 (deduped, parametrized 2×) |
| Tests passing | 1,599 | 1,599 | — |
| Functions >80 lines | 55 | 49 | -6 |
| Functions >150 lines | 14 | 10 | -4 |
| TODO/FIXME markers | 1 | 1 | — |
| Duplicate config tests | 26 | 0 | -26 |

### Lines by Subpackage

| Subpackage | Source Lines | Files | Role |
|------------|-------------|-------|------|
| rsi | 7,333 | 20 | I/O, pipeline wrappers, CLI, viewers |
| perturb | 3,634 | 19 | Campaign processing |
| scor160 | 3,484 | 13 | ATOMIX benchmark, core pipeline |
| chi | 2,008 | 7 | Thermal dissipation |
| top-level | 254 | 2 | config_base |

---

## What the Refactoring Fixed

| Phase | Items | Impact |
|-------|-------|--------|
| 1 (scor160) | Goodman bias helper, CSD delegation, l3 split, compare helpers, CLI parameterization | scor160/cli.py: 290→190 lines; spectral.py: `csd_odas` is now a 15-line wrapper; compare.py: 3 report formatters share helpers |
| 2 (chi) | FP07 noise helpers, L4 chi unification, l3_chi split | fp07.py: eliminated 18-line verbatim duplicate; l3_chi: 246→110 lines with `_SectionResult` dataclass; l4_chi: 2 functions → 1 parameterized + closures |
| 3 (rsi) | Shared dataset builder, `compute_speed_fast`, channel classification | `_build_diss_dataset`: 175→62 lines; `_build_chi_dataset`: 175→64 lines; both delegate to 38-line shared builder |
| 4 (perturb) | CLI `_glob_p_files` + `_run_analysis`, missing docstring | perturb/cli.py: 4 copy-pasted handlers → 4 one-liners calling shared helper |
| 5 (tests) | Config test deduplication | 26 duplicate tests → 27 parametrized tests in `test_config_shared.py` covering both modules; 0 overlap remaining |

---

## Detailed Findings

### 4. Architecture & Duplication — B

**Improved from B-**. The worst duplication is gone: dataset builders, CSD functions, FP07 noise config, CLI handlers, and config tests. What remains is more defensible.

#### Remaining issues

**4a. Speed computation still scattered (6 call sites)**

`smooth_fall_rate()` + interpolate + clamp appears in: `rsi/adapter.py` (manual, due to edge-effect constraint), `rsi/convert.py` (via `compute_speed_fast`), `rsi/helpers.py` (both patterns), `rsi/pipeline.py`, `perturb/pipeline.py`, `perturb/fp07_cal.py`. Three sites use `compute_speed_fast()` as intended; three still call `_smooth_fall_rate` directly because they need only the fall rate, not the full speed pipeline. This is defensible but fragile — the inline Butterworth filter in `adapter.py:75–86` is a maintenance hazard.

**4b. Deprecated functions still in CLI hot path**

`rsi-tpw eps` → `compute_diss_file()` → `get_diss()` (deprecated, warns). Same for `rsi-tpw chi` → `compute_chi_file()` → `get_chi()`. Users see deprecation warnings from the tool's own CLI. Either complete the migration (make the CLI call `run_pipeline` or the scor160 chain directly) or remove the deprecation warnings.

**4c. FP07 noise: residual T-dependent duplication (~15 lines)**

`noise_thermchannel()` (L374–391) and `noise_thermchannel_batch()` (L495–505) still duplicate the R_ratio → scale factor computation. The scalar version has NaN-guarded warnings; the batch version silently clips. Minor, but the scalar version's warning logic should be shared.

### 5. Large Functions — C+

**Improved from C**. 6 fewer functions over 80 lines; 4 fewer over 150.

| Function | File | Lines | Status |
|----------|------|-------|--------|
| `p_to_L1()` | rsi/convert.py | 319 | Channel classification extracted (68 lines out). Still 319 lines — NetCDF variable creation repeated 12× with near-identical patterns. Needs a variable-spec-list approach like the dataset builders. |
| `run_pipeline()` | rsi/pipeline.py | 288 | Orchestrates 8 steps. Could extract per-step functions but risk is low — each step is self-contained. |
| `get_diss()` | rsi/dissipation.py | 248 | Deprecated wrapper. Will shrink when deprecation cycle completes. |
| `get_chi()` | rsi/chi_io.py | 207 | Same. |
| `process_file()` | perturb/pipeline.py | 180 | 8-stage handler. Acceptable for an orchestrator. |
| `compute_eps_window()` | rsi/window.py | 163 | Multi-step epsilon estimation. Scientifically complex; splitting would hurt readability. |
| `process_l3()` | scor160/l3.py | 161 | Down from 178; constants and helpers extracted. Further splitting would fragment the spectral pipeline. |
| `_read()` | rsi/p_file.py | 160 | Binary parsing. Complex but well-structured. |
| `compute_chi_window()` | rsi/window.py | 156 | Same as `compute_eps_window` — scientifically dense. |
| `run_pipeline()` | perturb/pipeline.py | 155 | Orchestrator. Acceptable. |

The scientific core functions (spectral estimation, epsilon/chi window computation, binary parsing) are legitimately complex. The remaining targets for splitting are `p_to_L1()` (mechanical NetCDF construction) and completing the `get_diss`/`get_chi` deprecation.

### 6. Dead Code — A-

1 TODO marker (`perturb/pipeline.py:441`). No unused functions. `get_diss()` and `get_chi()` emit `DeprecationWarning` but are still called by the CLI — they're not dead, but the deprecation cycle is incomplete.

### 7. Test Coverage — A-

**Improved from B+**. Config test duplication eliminated. 27 shared tests now cover both config modules via parametrized fixture.

**871 test functions across 54 files. 1599 passing, 10 skipped.**

| Subpackage | Test Files | Approx Tests | Quality |
|------------|-----------|-------|---------|
| scor160 | ~12 | ~200 | Good — unit + integration + ATOMIX L1→L4 |
| rsi | ~18 | ~350 | Good — unit + integration + CLI |
| chi | ~5 | ~150 | Good — unit + integration + MATLAB |
| perturb | ~12 | ~150 | Good — unit + integration |
| cross-validation | 4 | ~60 | Excellent — 30 .p files validated |
| config (shared) | 1 | ~54 | New — parametrized over rsi + perturb |

**Gaps:**
- **11 `pytest.skip()` calls** — tests skip silently if VMP data missing (down from 76 — fixture-based skips now handled centrally)
- Viewers smoke-tested only
- No tests for corrupt or truncated `.p` files
- pytest-xdist installed but unused in CI

### 8. Public API — B

50 exports across 4 `__init__.py` files. The deprecation cycle is the main issue: `get_diss` and `get_chi` are exported, deprecated, yet called by the CLI's own `compute_diss_file` and `compute_chi_file`. Users running `rsi-tpw eps` see a `DeprecationWarning` from the tool itself.

### 9. Type Annotations — B-

| Module Group | Coverage | Notes |
|-------------|----------|-------|
| scor160 | ~95% | Solid; dataclasses fully typed |
| chi core | ~90% | Good |
| chi/chi.py | ~80% | 7-element return tuples should use NamedTuple |
| rsi core | ~90% | Good |
| rsi pipeline | ~80% | `dict[str, Any]` returns from `load_channels()`, `_channels_from_pfile()` |
| rsi viewers | ~0% | No type hints at all |
| perturb | ~90% | Good |

17 modules have a blanket mypy override suppressing 6 error codes (`operator`, `index`, `arg-type`, `return-value`, `no-any-return`, `assignment`). This is driven by NumPy array arithmetic false positives — defensible but masks real errors.

### 10. Docstrings — B-

**Improved from C+**. Perturb private functions now documented (Phase 4B). The `_get_agg_func` docstring was the only one missing and is now added.

**Remaining gaps:**
- 10 GPS protocol methods (`lat()`/`lon()` on 5 provider classes) lack docstrings — these are 1-line implementations of a protocol, low severity
- `p_to_L1()` internal logic (319 lines of NetCDF construction) has minimal inline documentation
- `rsi/window.py` functions have sparse docstrings for their complexity

### 11. Error Handling — B-

**Downgraded from B** due to the ATOMIX L1→L4 crash.

**Critical: L1→L4 crashes on 2 of 6 ATOMIX benchmark datasets.**

- **Epsilometer**: L2 processing introduces 28 NaN values (HP filter edge effects on data with NaN at boundaries). 1 of 131 spectral windows contains NaN → `scipy.signal.detrend` raises `ValueError: array must not contain infs or NaNs`.
- **MSS Baltic**: L1 data contains 19,246 NaN values (instrument gaps). L2 propagates 19,790 NaN. 3 of 33 windows affected → same crash.
- **Root cause**: `csd_matrix_batch()` calls `_detrend_batch()` → `scipy.signal.detrend()` with no NaN guard. The L2→L4 path works because reference L2 data (from MATLAB) handles NaN differently.
- **Impact**: The `scor160-tpw l1-l4` command crashes on these datasets. The VMP and Nemo datasets (the ones with clean L1 data) work fine. The rsi pipeline (`rsi-tpw eps`) is not affected because it uses a different spectral path.
- **Fix**: Replace NaN values before detrending (interpolate or zero-fill within windows), or skip windows containing NaN and warn.

**Other error handling gaps:**
- perturb uses `print()` for logging — no `logging` module anywhere
- `ProcessPoolExecutor` errors printed to stderr with no summary

### 12. Infrastructure — A-

Unchanged. Strong CI, good linting, appropriate coverage threshold.

---

## Remaining Refactoring Recommendations

### Previously Completed (this session)

| Item | What Changed |
|------|-------------|
| Goodman bias helper | Extracted `_bias_correction()` shared by scalar and batch |
| CSD consolidation | `csd_odas()` now delegates to `csd_matrix()` |
| l3 clarity | Module constants `MACOUN_LUECK_K_MAX/DENOM`, extracted `_build_window_arrays` and `_apply_macoun_lueck` |
| Compare helpers | Extracted `_log_spectral_metrics()` and `_format_report_header()` |
| scor160 CLI parameterization | 6 handlers → `_run_benchmark(args, levels)` |
| FP07 noise consolidation | `_unpack_noise_config()` and `_noise_f_intermediates()` shared; `fp07_tau_batch` is thin wrapper |
| L4 chi unification | `_process_l4_chi()` with strategy closures |
| l3_chi split | `_SectionResult` dataclass + `_process_section_chi()` helper |
| Dataset builder merge | `_build_result_dataset()` in helpers.py, used by both diss and chi |
| `compute_speed_fast` helper | Added to scor160/profile.py, used by convert.py and helpers.py |
| `p_to_L1` channel classification | Extracted `_classify_channels()` (68 lines) |
| perturb CLI parameterization | `_glob_p_files()` + `_run_analysis()` shared handler |
| perturb docstrings | Added missing `_get_agg_func` docstring |
| Config test deduplication | 27 parametrized tests in `test_config_shared.py`; 0 duplicate tests remaining |

### Remaining

| Priority | Item | Est. Effort | Notes |
|----------|------|-------------|-------|
| **P0** | **NaN guard in `csd_matrix_batch`** | Small | Skip or interpolate NaN windows before `detrend`. Crashes L1→L4 on Epsilometer and MSS. |
| P1 | Complete deprecation cycle | Medium | Make `compute_diss_file`/`compute_chi_file` call scor160 chain directly instead of `get_diss`/`get_chi`. Then remove the deprecated functions. |
| P1 | Split `p_to_L1()` NetCDF creation | Medium | 319 lines. Variable creation repeated 12× — use a spec list like the dataset builders. |
| P2 | TypedDict for `load_channels()` return | Small | Replace `dict[str, Any]` with typed structure. 4 call sites. |
| P2 | GPS protocol method docstrings | Trivial | 10 one-line methods on 5 classes. |
| P3 | Structured logging in perturb | Medium | Replace `print()` with `logging` module. |
| P3 | Enable pytest-xdist in CI | Trivial | Add `-n auto` to pytest command. |

---

## ATOMIX Benchmark Results (post-refactoring)

### L2→L4 (all 6 datasets pass)

| Dataset | Spectra | L4 ε within factor 2 | Method Agreement |
|---------|---------|---------------------|-----------------|
| Epsilometer | 181 | N/A (ISR) | 98.9–100% |
| MSS Baltic | 61 | N/A (ISR) | 100% |
| MR1000 Minas Passage | 30 | 100% (4 probes) | 100% |
| VMP2000 Faroe Bank | 342 | N/A (ISR) | 100% |
| VMP250 Tidal (cs) | 32 | 96.9–100% | 93.8–96.9% |
| VMP250 Tidal | 32 | 100% | 96.9–100% |

### L1→L4 (4 of 6 datasets pass)

| Dataset | L2 Overlap | L4 ε within factor 2 | Status |
|---------|-----------|---------------------|--------|
| MR1000 Minas Passage | 100% | 100% (4 probes) | **PASS** |
| VMP2000 Faroe Bank | 100% | N/A (ISR, 100% agreement) | **PASS** |
| VMP250 Tidal (cs) | 100% | 96.9–100% | **PASS** |
| VMP250 Tidal | 99.8% | 100% | **PASS** |
| Epsilometer | 100% (L2 ok) | — | **CRASH** — 28 NaN from HP filter edge; 1/131 windows bad |
| MSS Baltic | 100% (L2 ok) | — | **CRASH** — 19,246 NaN in L1; 3/33 windows bad |

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
