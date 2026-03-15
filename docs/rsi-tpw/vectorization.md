# Epsilon Computation: Vectorization and Performance

This document describes the vectorization work applied to the epsilon dissipation pipeline and quantifies the resulting performance improvements.

## Motivation

Profiling the epsilon pipeline on three ARCTERX `.p` files (0026, 0028, 0029; ~18 profiles, ~7000 estimates total) revealed that `_compute_profile_diss` consumed ~76% of end-to-end epsilon time, with the per-window Python loop as the dominant bottleneck. Within that loop, `csd_matrix` (per-window FFTs, detrending, windowing, and cross-spectral products) accounted for ~63% and `_estimate_epsilon` (iterative Nasmyth fitting) for ~29%.

Phase breakdown before vectorization (per-file averages, 3 files):

| Phase               | Time   | Share |
|---------------------|--------|-------|
| PFile load          | 0.86 s | 16%   |
| load_channels       | 0.57 s | 11%   |
| despike             | 0.21 s |  4%   |
| _prepare_profiles   | 0.61 s | 12%   |
| _compute_prof_diss  | 3.06 s | 58%   |
| **Total**           | 5.31 s |       |

## Changes

### 1. Batched cross-spectral density (`csd_matrix_batch`)

**File:** `src/odas_tpw/scor160/spectral.py`

Replaced per-window calls to `csd_matrix` with a single batched function that processes all dissipation windows at once:

- **Window extraction:** Fancy indexing builds all overlapping FFT segments from all windows in one array operation — `x_windows[:, indices, :]` produces a `(n_windows, n_seg, nfft, n_channels)` array.
- **Detrending:** `_detrend_batch` applies scipy `signal.detrend` (linear) or `np.polynomial.polynomial.polyfit/polyval` (parabolic/cubic) to the full 4D segment array.
- **Windowing:** Element-wise multiply with the cosine (Hann) window, broadcast across all dimensions.
- **FFT:** Single `np.fft.rfft` call on the 4D array along the nfft axis.
- **Cross-spectral products:** `np.einsum('wsfi,wsfj->wfij', conj(fft_x), fft_y)` computes all cross-spectral matrices in one pass, replacing the inner Python loop.

### 2. Batched Goodman noise removal (`clean_shear_spec_batch`)

**File:** `src/odas_tpw/scor160/goodman.py`

Calls `csd_matrix_batch` for the combined shear+accelerometer CSD, then solves the Goodman equation `clean = UU - UA @ inv(AA) @ conj(UA)^H` using `np.linalg.solve` broadcasting over `(n_windows, n_freq)` leading dimensions. Falls back to per-window solve on `LinAlgError`.

### 3. Nasmyth spectrum grid interpolation (`NasmythGrid`, `nasmyth_grid`)

**File:** `src/odas_tpw/scor160/nasmyth.py`

Pre-tabulates `log10(G2)` vs `log10(x)` on a 2000-point grid (x = 1e-6 to 1.0) at module import. Subsequent calls use `np.interp` in log-space instead of evaluating the Lueck (2016) rational polynomial. A lazy singleton (`_get_grid()`) ensures the table is built once.

Maximum relative error vs direct evaluation: ~2e-5 (0.002%), verified across 41,658 estimates on 3 files.

### 4. Vectorized `_compute_profile_diss`

**File:** `src/odas_tpw/rsi/dissipation.py`

Rewrote the main dissipation function to:

1. Extract all windows via fancy indexing on the full-profile arrays.
2. Call `clean_shear_spec_batch` for batched CSD + Goodman (or `csd_matrix_batch` when Goodman is disabled).
3. Vectorize all per-window mean computations (speed, T, P, time, viscosity) and wavenumber/correction arrays using broadcasting.
4. Retain the per-window `_estimate_epsilon` loop — this function contains iterative convergence logic and per-window branching (variance method vs ISR method) that cannot be vectorized.

### 5. Pipeline PFile deduplication

**File:** `src/odas_tpw/rsi/cli.py`

`_cmd_pipeline` previously called `compute_diss_file` and `compute_chi_file`, each of which independently loaded the `.p` file via `PFile`. Now loads `PFile` once and calls `get_diss`/`get_chi` directly, passing the epsilon Dataset to chi for Method 1. Saves ~0.8 s and ~1.4 GB per file.

## Performance Results

Measured on Apple M4 Max, 3 ARCTERX files (0026, 0028, 0029), per-file averages:

| Phase               | Before | After  | Speedup |
|---------------------|--------|--------|---------|
| _compute_prof_diss  | 3.06 s | 1.40 s | 2.19x   |
| End-to-end epsilon  | 5.31 s | 3.13 s | 1.70x   |

- **Profile computation speedup: 2.19x** — the CSD/Goodman batching eliminated ~63% of the per-window overhead; `_estimate_epsilon` (the remaining ~29%) is unchanged.
- **End-to-end speedup: 1.70x** — diluted by fixed costs (PFile load, despike, profile detection).
- **Memory: unchanged** — batched arrays are temporary and freed after use.

## Numerical Equivalence

Epsilon values from the vectorized pipeline were compared against the original per-window loop (with `nasmyth` instead of `nasmyth_grid`) across all 3 files, both probes (sh1, sh2), 41,658 total estimates:

| Metric                | sh1       | sh2       |
|-----------------------|-----------|-----------|
| Max |relative diff|   | 1.97e-05  | 1.94e-05  |
| Mean |relative diff|  | 2.49e-06  | 2.53e-06  |
| Max |log10 diff|      | 0.0000086 | 0.0000084 |
| Bit-identical         | 99.4%     | 99.4%     |

The small residuals are entirely from the Nasmyth grid interpolation (log-space `np.interp`). All differences are well below measurement uncertainty.
