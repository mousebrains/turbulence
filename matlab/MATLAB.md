# MATLAB Directory

This directory contains MATLAB scripts that support the `odas_tpw.rsi` package. The files fall into three categories:

1. **Python-mirrored implementations** — MATLAB functions that implement the same algorithms as modules in `src/odas_tpw/rsi/`. These must stay self-consistent: if the Python implementation changes, the corresponding MATLAB function should be updated (or vice versa) so that both produce equivalent results.

2. **Test data generators** — Scripts that call the ODAS library to produce NetCDF validation files consumed by `tests/test_matlab_*.py`. Re-run these in MATLAB whenever the test expectations change.

3. **Examples and utilities** — Demonstration and visualization scripts not tied to the test suite or Python code.

---

## 1. Python-Mirrored Implementations

These files are standalone MATLAB reimplementations of algorithms in `src/odas_tpw/rsi/`. They share the same formulas, constants, and default parameters. When editing the Python side, check the corresponding MATLAB file for consistency (and vice versa).

| MATLAB file | Python counterpart | Purpose |
|---|---|---|
| `get_chi.m` | `chi.py → get_chi()` | Main entry point: windowed chi computation over a profile. Supports Method 1 and Method 2. |
| `chi_method1.m` | `chi.py → chi_method1()` | Single-window chi from known epsilon (Dillon & Caldwell 1980). Integrates observed gradient spectrum, corrects for FP07 rolloff and unresolved variance. |
| `chi_method2.m` | `chi.py → chi_method2()` | Single-window chi via MLE Batchelor spectrum fit (Ruddick et al. 2000, Peterson & Fer 2014). Grid search over kB, iterative refinement. |
| `batchelor_gradT.m` | `batchelor.py → batchelor_gradT()` | Batchelor temperature gradient spectrum model. |
| `kraichnan_gradT.m` | `batchelor.py → kraichnan_gradT()` | Kraichnan temperature gradient spectrum model. |
| `batchelor_wavenumber.m` | `batchelor.py → batchelor_wavenumber()` | Batchelor wavenumber from epsilon and viscosity. |
| `fp07_transfer.m` | `fp07.py → fp07_transfer()` | FP07 thermistor squared transfer function `|H(f)|²` (single-pole and double-pole models). |
| `fp07_time_constant.m` | `fp07.py → fp07_time_constant()` | Speed-dependent FP07 time constant `τ(v)`. |

### Key shared parameters (defaults must match between MATLAB and Python)

| Parameter | Default | Used in |
|---|---|---|
| `fft_length` | 512 | `get_chi` |
| `diss_length` | 3 × fft_length | `get_chi` |
| `overlap` | diss_length / 2 | `get_chi` |
| `f_AA` | 98 Hz | `get_chi` |
| `diff_gain` | 0.94 s | `get_chi` |
| `spectrum_model` | `"kraichnan"` | `get_chi`, `chi_method1`, `chi_method2` |
| `fp07_model` | `"single_pole"` | `get_chi`, `chi_method1`, `chi_method2` |
| `salinity` | 35 PSU | `get_chi` |
| `q` (Batchelor) | 3.7 | `batchelor_gradT` |
| `q` (Kraichnan) | 5.26 | `kraichnan_gradT` |
| `kappa_T` | 1.4e-7 m²/s | `chi_method1`, `chi_method2`, `batchelor_wavenumber` |

---

## 2. Test Data Generators

These scripts use the ODAS library (`../odas/`) to process the raw `.p` files in `../VMP/` and write NetCDF validation files. The Python test suite loads these NetCDF files and compares Python results against them.

| MATLAB file | Output files | Consumed by test |
|---|---|---|
| `generate_for_tests.m` | *(orchestrator — runs the three scripts below in sequence)* | — |
| `generate_odas_p2mat_nc.m` | `VMP/*_p2mat.nc` | `test_matlab_all_files.py`, `test_matlab_validation.py` |
| `generate_validation_nc.m` | `VMP/*_validation.nc` | `test_matlab_epsilon.py` |
| `generate_scalar_spectra_nc.m` | `VMP/*_scalar_spectra.nc` | `test_matlab_chi.py` |

### What each generator produces

- **`generate_odas_p2mat_nc.m`** — Converts each `.p` file via `odas_p2mat()` and writes channel data (P, T1, T2, sh1, sh2, Ax, Ay, speed, fall rate) at both fast and slow sample rates. Used to validate `PFile` channel conversion in Python.

- **`generate_validation_nc.m`** — Runs `get_diss_odas()` on each profile to compute epsilon, then writes per-profile groups containing epsilon, K_max, figure of merit, MAD, viscosity, speed, pressure/temperature means, wavenumber/frequency vectors, and shear spectra. Config: `fft_length=256, diss_length=512, overlap=256, f_AA=98`. Used to validate `dissipation.py`.

- **`generate_scalar_spectra_nc.m`** — Runs `get_scalar_spectra_odas()` to compute temperature gradient spectra per profile. Writes per-profile groups containing scalar spectra, wavenumber/frequency vectors, speed, and pressure means. Used to validate the spectral pipeline feeding into `chi.py`.

### When to re-run

Re-run `generate_for_tests.m` in MATLAB (with `../odas/` on the path) if you change:
- Processing parameters (fft_length, overlap, f_AA, etc.)
- Profile detection logic that affects which windows are compared
- The set of `.p` files in `VMP/`

---

## 3. Examples and Utilities

These files are not tied to the Python code or test suite. They are for interactive analysis, visualization, and demonstration.

| MATLAB file | Purpose |
|---|---|
| `example00.m` | End-to-end workflow demo: loads a `.p` file, detects profiles, computes epsilon and chi (both methods), extracts salinity from JAC conductivity, and creates a publication-quality figure. Good starting point for new users. |
| `tpw_plot_spectra.m` | Interactive spectrum visualization. Plots shear (epsilon) and temperature (chi) spectra with theoretical model fits (Nasmyth, Batchelor/Kraichnan). Supports profile selection and pressure range filtering. |
| `miss_hit.cfg` | Configuration for MISS_HIT MATLAB linter/formatter (not a script). |

Additionally, `../VMP/p2mat.m` is a small batch-conversion utility that runs `odas_p2mat()` on all `.p` files in the VMP directory.

---

## Requirements

- MATLAB R2021a or newer (uses `arguments` blocks and `name=value` syntax)
- Signal Processing Toolbox (`pwelch`, `hanning`)
- ODAS MATLAB Library v4.5.1 on the path (in `../odas/`)

## References

- Dillon & Caldwell, 1980: The Batchelor spectrum and dissipation in the upper ocean. *J. Geophys. Res.*, 85, 1910-1916.
- Oakey, 1982: Determination of the rate of dissipation of turbulent energy. *J. Phys. Oceanogr.*, 12, 256-271.
- Bogucki, Domaradzki & Yeung, 1997: DNS of passive scalars with Pr>1. *J. Fluid Mech.*, 343, 111-130.
- Ruddick, Anis & Thompson, 2000: Maximum likelihood spectral fitting. *J. Atmos. Oceanic Technol.*, 17, 1541-1555.
- Peterson & Fer, 2014: Dissipation measurements using temperature microstructure from an underwater glider. *Methods in Oceanography*, 10, 44-69.
- RSI Technical Note 040: Noise in Temperature Gradient Measurements.
