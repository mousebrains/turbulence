# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Processing and analysis code for calculating turbulent kinetic energy (TKE) and chi (thermal dissipation rate) from Rockland Scientific vertical microprofilers and microriders. Instruments use fast temperature sensors (e.g., FP07 thermistors).

## Package: rsktools

Installable Python package (`pip install -e ".[dev]"`). Source layout: `src/rsktools/`.

### Modules

- `p_file.py` — `PFile` class: reads Rockland `.p` binary files, parses headers, demultiplexes address matrix, converts to physical units. `parse_config()` parses the embedded INI config string.
- `channels.py` — Conversion functions (raw counts → physical units) for each sensor type: therm, shear, poly, voltage, piezo, inclxy, inclt, jac_c, jac_t, raw, aroft_o2, aroft_t. `CONVERTERS` dict maps type names to functions.
- `convert.py` — `p_to_netcdf()` and `convert_all()` for writing NetCDF4 output.
- `cli.py` — CLI entry points `p2nc` and `pinfo`.

### CLI Commands

```bash
# Convert P files to NetCDF
p2nc VMP/*.p -o nc/                           # batch, output directory
p2nc VMP/ARCTERX_Thompson_2025_SN479_0001.p -o out.nc  # single file

# Inspect P file metadata
pinfo VMP/ARCTERX_Thompson_2025_SN479_0001.p
```

### Python API

```python
from rsktools import PFile, p_to_netcdf
pf = PFile("VMP/ARCTERX_Thompson_2025_SN479_0001.p")
pf.channels["T1"]    # numpy array, physical units (°C)
pf.channels["sh1"]   # shear in s⁻¹
pf.t_fast             # time vector for fast channels
pf.fs_fast            # fast sampling rate (~512 Hz)
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

## Data

- **Instrument**: VMP-250IR_RT SN 479 (ARCTERX campaign, R/V Thompson, Jan 2025, Saipan)
- **Address matrix**: 8 rows × 10 cols (8 fast + 2 slow). fs_fast ≈ 512 Hz, fs_slow ≈ 64 Hz.
- `VMP/` — 30 raw `.p` files
- `odas/` — Rockland's ODAS MATLAB Library (v4.5.1), reference implementation. Key files: `odas_p2mat.m`, `read_odas.m`, `convert_odas.m`, `setupstr.m`.
