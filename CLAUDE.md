# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Processing and analysis code for calculating turbulent kinetic energy (TKE) and chi (thermal dissipation rate) from Rockland Scientific vertical microprofilers and microriders. Instruments use fast temperature sensors (e.g., FP07 thermistors).

## Package: microstructure-tpw

Installable Python package (`pip install -e ".[dev]"`). Source layout: `src/odas_tpw/`.

### Subpackages

- `rsi/` — Rockland Scientific instrument I/O, NetCDF conversion, profiles, epsilon, chi orchestration
- `chi/` — Chi (thermal variance dissipation) calculation, Batchelor/Kraichnan spectra, FP07 transfer function
- `scor160/` — ATOMIX shear-probe benchmark processing (L1–L4), shared physics/spectral modules
- `processing/` — Instrument-agnostic profile-bound cleanup (top_trim prop-wash, bottom-crash detection)
- `perturb/` — Full campaign processing pipeline (trim, merge, calibrate, compute, bin)

### Key Modules (rsi)

- `p_file.py` — `PFile` class: reads Rockland `.p` binary files, parses headers, demultiplexes address matrix, converts to physical units. `parse_config()` parses the embedded INI config string.
- `channels.py` — Conversion functions (raw counts → physical units) for each sensor type. `CONVERTERS` dict maps type names to functions.
- `convert.py` — `p_to_netcdf()` and `convert_all()` for writing NetCDF4 output.
- `profile.py` — Profile detection and per-profile NetCDF extraction.
- `dissipation.py` — Core epsilon calculation with multi-source input, QC metrics (fom, K_max_ratio).
- `chi_io.py` — Chi orchestration: load instrument data and call chi computation.
- `config.py` — YAML configuration loading, merging, template generation.
- `cli.py` — Unified `rsi-tpw` CLI with subcommands.

### Key Modules (chi)

- `chi.py` — Chi (thermal variance dissipation) calculation, Methods 1 and 2, QC metrics.
- `batchelor.py` — Batchelor and Kraichnan temperature gradient spectra.
- `fp07.py` — FP07 thermistor transfer function and electronics noise model.

### Key Modules (scor160)

- `spectral.py` — Cross-spectral density estimation (Welch method, cosine window).
- `goodman.py` — Goodman coherent noise removal using accelerometer spectra.
- `despike.py` — Iterative spike removal for shear probe signals.
- `nasmyth.py` — Nasmyth universal shear spectrum (Lueck improved fit).
- `ocean.py` — Seawater properties: `visc35`, `visc(T,S,P)`, `density(T,S,P)`, `buoyancy_freq(T,S,P)` via gsw (TEOS-10).

### CLI Commands

```bash
rsi-tpw info VMP/*.p                           # print .p file metadata
rsi-tpw nc VMP/*.p -o nc/                      # convert .p to NetCDF
rsi-tpw prof VMP/*.p -o profiles/              # extract profiles
rsi-tpw eps VMP/*.p -o epsilon/                # compute epsilon
rsi-tpw chi VMP/*.p -o chi/                    # compute chi (Method 2)
rsi-tpw chi VMP/*.p --epsilon-dir epsilon/ -o chi/  # chi with epsilon (Method 1)
rsi-tpw pipeline VMP/*.p -o results/           # full pipeline
rsi-tpw eps VMP/*.p --salinity 34.5 -o epsilon/  # custom salinity
```

### Python API

```python
from odas_tpw.rsi import PFile
from odas_tpw.rsi.pipeline import run_pipeline
from odas_tpw.rsi.dissipation import compute_diss_file
from odas_tpw.rsi.chi_io import compute_chi_file
from odas_tpw.scor160.ocean import visc, density, buoyancy_freq
from pathlib import Path

pf = PFile("VMP/ARCTERX_Thompson_2025_SN479_0001.p")
pf.channels["T1"]    # numpy array, physical units (°C)
pf.channels["sh1"]   # shear in s⁻¹
pf.t_fast             # time vector for fast channels
pf.fs_fast            # fast sampling rate (~512 Hz)

# Full pipeline: .p → profiles → epsilon → chi → binning → combine
run_pipeline([Path("VMP/file.p")], Path("results/"))

# Or use modular file-level functions
eps_paths = compute_diss_file("VMP/file.p", "epsilon/")
chi_paths = compute_chi_file("VMP/file.p", "chi/")

# Note: get_diss() and get_chi() still work but are deprecated
```

## Commands

```bash
pip install -e ".[dev]"       # install in editable mode with test deps
python -m pytest              # run all tests
python -m pytest tests/test_p_file.py::test_header  # single test
```

## Domain Context

- **Chi (χ)**: Thermal variance dissipation rate, computed from temperature gradient spectra. Units: K²/s.
- **TKE dissipation (ε)**: Turbulent kinetic energy dissipation rate, computed from shear probe spectra. Units: W/kg.
- **FP07**: Fast-response glass-bead thermistor. Has a known frequency response rolloff that must be corrected.
- **P file format** (RSI TN-051): binary records with 128-byte headers (64 uint16 words). Record 0 = header + ASCII config string. Records 1..N = header + multiplexed int16 data. Endian flag at header word 64 (1=little, 2=big).
- **fom (figure of merit)**: Ratio of observed to attenuated-model variance in the spectral fit range. The model includes Batchelor/Kraichnan spectrum convolved with FP07 transfer function plus noise floor. Values near 1.0 indicate good fit.
- **K_max_ratio**: K_max/K_95 (epsilon) or K_max/kB (chi). Values < 0.5 mean most variance is extrapolated.

## Data

- **Instrument**: VMP-250IR_RT SN 479 (ARCTERX campaign, R/V Thompson, Jan 2025, Saipan)
- **Address matrix**: 8 rows × 10 cols (8 fast + 2 slow). fs_fast ≈ 512 Hz, fs_slow ≈ 64 Hz.
- `VMP/` — 30 raw `.p` files
- `odas/` — Rockland's ODAS MATLAB Library (v4.5.1), reference implementation. Key files: `odas_p2mat.m`, `read_odas.m`, `convert_odas.m`, `setupstr.m`.
