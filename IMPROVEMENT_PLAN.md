# Improvement Plan: All Categories to A- or Better

**Current overall**: B+ (3.2/4.0)
**Target overall**: A- (3.7/4.0)

Categories 1–5 are already A- or better and need no work. The plan below covers the 5 categories that need improvement, organized into phases.

---

## Phase 1: Error Handling (B- → A-)

Estimated scope: ~150 lines changed across 4 files. No algorithm changes.

### 1a. Add warnings to chi.py silent NaN returns

**File**: `src/microstructure_tpw/rsi/chi.py`

Add `import warnings` at top. At each of the 7 early-return NaN sites, add a `warnings.warn()` with the specific reason:

| Line | Condition | Warning message |
|------|-----------|-----------------|
| 92 | `kB < 1` | `"kB={kB:.1f} < 1 cpm; epsilon too low for chi estimation"` |
| 113 | `< 3 valid points` | `"Too few valid wavenumber points for chi integration"` |
| 125 | `chi_trial <= 0` | `"Trial chi <= 0; observed variance too low"` |
| 143 | `V_resolved/V_total <= 0` | `"Batchelor variance non-positive; cannot compute correction"` |
| 204 | `< 3 fit points` | `"Too few valid points for MLE fit"` |
| 223 | `all NLL infinite` | `"All NLL values infinite; MLE fit failed"` |
| 308 | `< 6 valid points` | `"Too few valid points for iterative fit"` |

### 1b. Narrow exception handling in cli.py and convert.py

**Files**: `src/microstructure_tpw/rsi/cli.py`, `src/microstructure_tpw/rsi/convert.py`

Replace `except Exception as e:` with specific exceptions:

```python
# Before
except Exception as e:
    print(f"Error processing {p}: {e}", file=sys.stderr)

# After
except (OSError, ValueError, RuntimeError) as e:
    print(f"Error processing {p}: {e}", file=sys.stderr)
```

8 sites total: `cli.py:271,285,322,336,365,379` and `convert.py:139,150`.

### 1c. Log speed clamping

**Files**: `dissipation.py:576`, `chi.py:807`, `diss_look.py:122,285`, `quick_look.py:291`

Add `import warnings` and warn when clamping:

```python
if W < 0.01:
    warnings.warn(
        f"Speed {W:.4f} m/s below minimum; clamped to 0.01 m/s",
        stacklevel=2,
    )
    W = 0.01
```

### 1d. Fix Goodman short-signal fallback

**File**: `src/microstructure_tpw/rsi/goodman.py:62–80`

Currently returns all-zero matrices despite comment saying "uncleaned spectra". Fix by computing actual auto-spectrum of shear when CSD fails:

```python
except ValueError:
    warnings.warn(...)
    # Compute auto-spectrum only (no cross-spectra for cleaning)
    try:
        UU_diag, F, _, _ = csd_matrix(shear, shear, nfft, rate, overlap=nfft // 2)
        n_ac = accel.shape[1]
        n_freq_fb = len(F)
        AA = np.zeros((n_freq_fb, n_ac, n_ac), dtype=np.complex128)
        UA = np.zeros((n_freq_fb, n_sh, n_ac), dtype=np.complex128)
        return np.real(UU_diag), AA, UU_diag, UA, F
    except ValueError:
        # Signal too short even for auto-spectrum; return zeros
        ...
```

---

## Phase 2: Test Coverage (B → A-)

Estimated scope: ~400 lines of new test code across 2 files.

### 2a. Test deconvolve.py

**File**: `tests/test_deconvolve.py` (new)

Test both functions:
- `deconvolve()`: Apply known pre-emphasis filter, deconvolve, verify recovery. Test edge cases (empty input, single sample, mismatched lengths).
- `_interp_if_required()`: Test passthrough when no interpolation needed, PCHIP interpolation with known function.

### 2b. Test remaining channel converters

**File**: `tests/test_channels.py` (extend)

Add tests for 9 untested converters. For each:
- Construct a known raw-count input
- Verify output against hand-calculated expected value
- Test edge cases (zero counts, max counts, negative counts for two's complement converters)

Converters to test:

| Converter | Key behavior to verify |
|-----------|----------------------|
| `convert_piezo` | Linear scaling from coefficients |
| `convert_voltage` | Linear scaling |
| `convert_inclxy` | Two's complement (14-bit signed from 16-bit), scaling |
| `convert_inclt` | Two's complement (same as inclxy), temperature offset |
| `convert_jac_c` | Unsigned 16-bit wrapping, conductivity formula |
| `convert_jac_t` | Unsigned 16-bit wrapping, temperature formula |
| `convert_aroft_o2` | Unsigned 16-bit wrapping, oxygen formula |
| `convert_aroft_t` | Unsigned 16-bit wrapping, temperature formula |
| `convert_gnd` | Passthrough (alias for convert_raw) |

### 2c. Smoke-test viewers (optional, for full A)

**Files**: `tests/test_quick_look.py`, `tests/test_diss_look.py` (new)

Minimal import + instantiation tests with matplotlib backend set to `Agg`. Don't test interactive behavior, just that the modules load and the core computation helpers (`_compute_windowed_diss`, `_compute_depth_spectra`) produce correct shapes.

---

## Phase 3: Internal Documentation (C+ → A-)

Estimated scope: ~200 lines of docstrings added. No code changes.

### 3a. Algorithm function docstrings

Add full NumPy-style docstrings (Parameters, Returns, Notes with references) to:

| Function | File:Line | Current | Needed |
|----------|-----------|---------|--------|
| `_estimate_epsilon` | `dissipation.py:789` | Brief one-liner | Parameters (11 args), Returns (tuple of 6), Notes (cite Lueck) |
| `_variance_correction` | `dissipation.py:936` | One-liner | Parameters (4 args), Returns, Notes (ODAS lines 623–634) |
| `_chi_from_epsilon` | `chi.py:70` | Returns only | Parameters (12 args) |
| `_mle_fit_kB` | `chi.py:168` | Brief | Parameters (11 args), Returns |
| `_iterative_fit` | `chi.py:279` | Brief + ref | Parameters (12 args), Returns |
| `_safe_float` | `channels.py:13` | None | One-liner + Parameters |
| `_compute_profile_chi` | `chi.py:717` | None? | Brief + Parameters |
| `_compute_profile_diss` | `dissipation.py:470` | None? | Brief + Parameters |

### 3b. Channel converter docstrings

Expand each converter in `channels.py` from one-liners to include:
- What physical quantity it produces and units
- Relevant coefficients from the channel config
- Reference (ODAS function name or RSI TN)

---

## Phase 4: Type Annotations (B- → A-)

Estimated scope: ~100 lines of annotation additions. No logic changes.

### 4a. Core public APIs (highest impact)

| Function | File:Line | Parameters | Return type |
|----------|-----------|------------|-------------|
| `get_chi` | `chi.py:414` | 13 params | `list[xr.Dataset]` |
| `get_diss` | `dissipation.py:304` | 12 params | `list[xr.Dataset]` |
| `load_channels` | `dissipation.py:42` | 5 params | `dict[str, Any]` |
| `compute_chi_file` | `chi.py:1112` | kwargs | `list[Path]` |
| `compute_diss_file` | `dissipation.py:1015` | kwargs | `list[Path]` |

### 4b. Remaining chi.py and dissipation.py functions

Annotate all internal functions in both modules. Target: 100% coverage for these two files.

### 4c. Profile and spectral modules

| Module | Current | Functions to annotate |
|--------|---------|----------------------|
| `profile.py` | 29% (2/7) | `get_profiles`, `extract_profiles`, `_find_breaks`, `_smooth_speed`, `_detect_profiles` |
| `spectral.py` | 60% (3/5) | `csd_matrix`, `_detrend_segment` |
| `despike.py` | 50% (1/2) | `despike` |
| `convert.py` | 67% (2/3) | `convert_all` already typed; check `p_to_netcdf` |
| `nasmyth.py` | 67% (2/3) | `nasmyth_spectrum` or whichever is missing |

### 4d. CLI (lower priority but needed for A-)

Annotate the 25 functions in `cli.py`. Most are `-> None` with `argparse.Namespace` parameters. Straightforward.

---

## Phase 5: Code Cleanliness (B+ → A-)

Estimated scope: ~200 lines refactored across 3 files.

### 5a. Split long functions

**`_compute_profile_chi`** (388 lines → ~4 functions):
- `_preallocate_chi_arrays()` — pre-allocation block
- `_process_chi_window()` — per-window spectrum + fitting logic
- `_build_chi_dataset()` — xarray Dataset construction
- `_compute_profile_chi()` — orchestrator calling the above

**`_compute_profile_diss`** (317 lines → ~4 functions):
- `_preallocate_diss_arrays()` — pre-allocation block
- `_process_diss_window()` — per-window spectrum + epsilon estimation
- `_build_diss_dataset()` — xarray Dataset construction
- `_compute_profile_diss()` — orchestrator

### 5b. Factor out two's complement helpers

**File**: `src/microstructure_tpw/rsi/channels.py`

```python
def _twos_complement_14bit(data: np.ndarray) -> np.ndarray:
    """Convert 16-bit raw to 14-bit signed (right-shift 2, two's complement)."""
    raw = data.astype(np.int32)
    val = (raw >> 2).astype(np.float64)
    val[val >= 2**13] -= 2**14
    return val

def _unsigned_16bit(data: np.ndarray) -> np.ndarray:
    """Convert signed int16 to unsigned by wrapping negatives."""
    d = data.copy()
    d[d < 0] = d[d < 0] + 2**16
    return d
```

Replace 5 call sites with these helpers.

### 5c. Document magic numbers

Add inline comments at point of use:

```python
# goodman.py:105
# Bias correction threshold per ODAS Technical Note 061
if fft_segments <= 1.02 * n_accel:

# goodman.py:115
# Bias correction factor (Goodman 2006, ODAS TN-061 Eq. 3)
R = 1.0 / (1.0 - 1.02 * n_accel / fft_segments)

# dissipation.py:577
# Minimum profiling speed (m/s) to avoid wavenumber singularity
if W < 0.01:
    W = 0.01
```

---

## Phase Order & Dependencies

```
Phase 1 (Error handling)     — no dependencies, do first
Phase 2 (Tests)              — no dependencies, can parallel with Phase 1
Phase 3 (Internal docs)      — no dependencies, can parallel
Phase 4 (Type annotations)   — do after Phase 5a (function splits create new signatures)
Phase 5 (Code cleanliness)   — do before Phase 4
```

Recommended execution order: **1 → 2 → 3 → 5 → 4**, or run 1+2+3 in parallel, then 5, then 4.

## Expected Final Grades

| Category | Before | After | Notes |
|----------|--------|-------|-------|
| Mathematical correctness | A | A | No change needed |
| MATLAB/ODAS agreement | A | A | No change needed |
| Unit consistency | A | A | No change needed |
| Public API documentation | A- | A- | No change needed (PFile docstring in Phase 3) |
| Performance | A- | A- | No change needed |
| Code cleanliness | B+ | A- | Phase 5 (split functions, factor duplication, document magic numbers) |
| Test coverage | B | A- | Phase 2 (deconvolve tests, converter tests, viewer smoke tests) |
| Type annotations | B- | A- | Phase 4 (core APIs, chi, dissipation, profile, CLI) |
| Error handling | B- | A- | Phase 1 (warnings, narrow exceptions, log clamping, fix fallback) |
| Internal documentation | C+ | A- | Phase 3 (algorithm docstrings, converter docstrings) |
| **Overall** | **B+ (3.2)** | **A- (3.7)** | |
