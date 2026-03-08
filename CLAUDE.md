# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Processing and analysis code for calculating turbulent kinetic energy (TKE) and chi (thermal dissipation rate) from Rockland Scientific vertical microprofilers and microriders. Instruments use fast temperature sensors (e.g., FP07 thermistors).

## Package: rsi-python

Installable Python package (`pip install -e ".[dev]"`). Source layout: `src/rsi_python/`.

### Modules

- `p_file.py` — `PFile` class: reads Rockland `.p` binary files, parses headers, demultiplexes address matrix, converts to physical units. `parse_config()` parses the embedded INI config string.
- `channels.py` — Conversion functions (raw counts → physical units) for each sensor type. `CONVERTERS` dict maps type names to functions.
- `convert.py` — `p_to_netcdf()` and `convert_all()` for writing NetCDF4 output.
- `profile.py` — Profile detection and per-profile NetCDF extraction.
- `dissipation.py` — Core epsilon calculation with multi-source input, QC metrics (fom, K_max_ratio).
- `chi.py` — Chi (thermal variance dissipation) calculation, Methods 1 and 2, QC metrics.
- `batchelor.py` — Batchelor and Kraichnan temperature gradient spectra.
- `fp07.py` — FP07 thermistor transfer function and electronics noise model.
- `scalar_spectra.py` — Temperature gradient spectrum computation.
- `spectral.py` — Cross-spectral density estimation (Welch method, cosine window).
- `goodman.py` — Goodman coherent noise removal using accelerometer spectra.
- `despike.py` — Iterative spike removal for shear probe signals.
- `nasmyth.py` — Nasmyth universal shear spectrum (Lueck improved fit).
- `ocean.py` — Seawater properties: `visc35`, `visc(T,S,P)`, `density(T,S,P)`, `buoyancy_freq(T,S,P)` via gsw (TEOS-10).
- `cli.py` — Unified `rsi-tpw` CLI with subcommands.

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
from rsi_python import PFile, get_diss, get_chi
from rsi_python import visc, density, buoyancy_freq

pf = PFile("VMP/ARCTERX_Thompson_2025_SN479_0001.p")
pf.channels["T1"]    # numpy array, physical units (°C)
pf.channels["sh1"]   # shear in s⁻¹
pf.t_fast             # time vector for fast channels
pf.fs_fast            # fast sampling rate (~512 Hz)

# Compute epsilon
eps_results = get_diss("VMP/file.p")
ds = eps_results[0]
ds["epsilon"]       # dissipation rate [W/kg]
ds["fom"]           # figure of merit (obs/Nasmyth variance ratio)
ds["K_max_ratio"]   # K_max/K_95 (spectral resolution)

# Compute with custom salinity
eps_results = get_diss("VMP/file.p", salinity=34.5)
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
- **fom (figure of merit)**: Ratio of observed to model variance in the spectral fit range. Values near 1.0 indicate good fit.
- **K_max_ratio**: K_max/K_95 (epsilon) or K_max/kB (chi). Values < 0.5 mean most variance is extrapolated.

## Data

- **Instrument**: VMP-250IR_RT SN 479 (ARCTERX campaign, R/V Thompson, Jan 2025, Saipan)
- **Address matrix**: 8 rows × 10 cols (8 fast + 2 slow). fs_fast ≈ 512 Hz, fs_slow ≈ 64 Hz.
- `VMP/` — 30 raw `.p` files
- `odas/` — Rockland's ODAS MATLAB Library (v4.5.1), reference implementation. Key files: `odas_p2mat.m`, `read_odas.m`, `convert_odas.m`, `setupstr.m`.
