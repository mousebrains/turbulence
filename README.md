# rsi-python

Python tools for reading Rockland Scientific microprofiler data and computing turbulent dissipation rates from VMP (Vertical Microstructure Profiler) and MicroRider instruments.

## Overview

**rsi-python** provides a complete processing pipeline for ocean turbulence measurements from Rockland Scientific instruments equipped with shear probes and fast thermistors (FP07). The package reads proprietary `.p` binary data files, converts channels to physical units, detects profiles, and computes the rate of dissipation of turbulent kinetic energy (epsilon) following the methods described in the Rockland Scientific ODAS MATLAB Library and associated Technical Notes.

### What it computes

- **Epsilon (TKE dissipation rate)** from shear probe spectra, including:
  - Iterative despiking of shear signals
  - Goodman coherent noise removal using accelerometer cross-spectra
  - Macoun & Lueck wavenumber correction for shear probe spatial response
  - Nasmyth universal spectrum fitting with Lueck's improved coefficients
  - Iterative variance correction using Lueck's resolved-variance model
  - Integration limit detection via polynomial fit to log-log spectra

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python >= 3.10. Dependencies: `numpy`, `netCDF4`, `scipy`, `xarray`.

## Pipeline

The processing pipeline has three stages. Each stage produces NetCDF files, and any later stage can start from any earlier stage's output:

```
.p files ──> p2nc ──> full-record .nc ──> p2prof ──> per-profile .nc ──> p2eps ──> epsilon .nc
```

### Stage 1: Convert `.p` files to NetCDF

```bash
p2nc VMP/*.p -o nc/
```

### Stage 2: Extract profiles

Detects profiling segments from pressure data and writes per-profile NetCDF files:

```bash
p2prof VMP/ARCTERX_Thompson_2025_SN479_0005.p -o profiles/
```

### Stage 3: Compute epsilon

Computes TKE dissipation rate from shear probe spectra:

```bash
# From raw .p files (profiles detected automatically)
p2eps VMP/ARCTERX_Thompson_2025_SN479_0005.p -o epsilon/

# From per-profile .nc files
p2eps profiles/*_prof*.nc -o epsilon/

# Parallel processing
p2eps VMP/*.p -o epsilon/ -j 0
```

### Python API

```python
from rsi_python import PFile, get_diss

# Read a .p file
pf = PFile("VMP/ARCTERX_Thompson_2025_SN479_0005.p")
pf.channels["sh1"]   # shear probe 1 [s⁻¹]
pf.channels["P"]     # pressure [dbar]

# Compute epsilon (returns list of xarray.Datasets, one per profile)
results = get_diss("VMP/ARCTERX_Thompson_2025_SN479_0005.p")
ds = results[0]
ds["epsilon"]         # dissipation rate [W/kg]
ds["spec_shear"]      # shear wavenumber spectra
ds["spec_nasmyth"]    # fitted Nasmyth spectra
```

## Modules

| Module | Description |
|--------|-------------|
| `p_file.py` | `PFile` class: reads `.p` binary files, parses headers, demultiplexes address matrix, converts to physical units |
| `channels.py` | Sensor conversion functions (raw counts to physical units) |
| `convert.py` | Full-record NetCDF export (`p2nc`) |
| `profile.py` | Profile detection and per-profile NetCDF extraction (`p2prof`) |
| `dissipation.py` | Core epsilon calculation with multi-source input (`p2eps`) |
| `spectral.py` | Cross-spectral density estimation (Welch method, cosine window) |
| `goodman.py` | Goodman coherent noise removal using accelerometer spectra |
| `despike.py` | Iterative spike removal for shear probe signals |
| `nasmyth.py` | Nasmyth universal shear spectrum (Lueck improved fit) |
| `ocean.py` | Seawater viscosity at S=35 |

## References

This package is a Python implementation derived from:

- **Rockland Scientific ODAS MATLAB Library** (v4.5.1) — the canonical reference implementation for processing VMP/MicroRider data. Available from [Rockland Scientific](https://rocklandscientific.com/support/software/).

- **Rockland Scientific Technical Notes** — detailed descriptions of the `.p` file format (TN-051), dissipation rate estimation (TN-028), and Goodman noise removal bias correction (TN-061). Available at [rocklandscientific.com/support/technical-notes](https://rocklandscientific.com/support/technical-notes/).

### Epsilon (shear probe) papers

- Lueck, R.G., 2022: [The statistics of oceanic turbulence measurements. Part 1: Shear variance and dissipation rates.](https://doi.org/10.1175/JTECH-D-21-0051.1) *J. Atmos. Oceanic Technol.*, 39, 1259–1276.
- McMillan, J.M., A.E. Hay, R.G. Lueck, and F. Wolk, 2016: [Rates of dissipation of turbulent kinetic energy in a high Reynolds number tidal channel.](https://doi.org/10.1175/JTECH-D-15-0167.1) *J. Atmos. Oceanic Technol.*, 33, 817–837.
- Oakey, N.S., 1982: [Determination of the rate of dissipation of turbulent energy from simultaneous temperature and velocity shear microstructure measurements.](https://doi.org/10.1175/1520-0485(1982)012%3C0256:DOTROD%3E2.0.CO;2) *J. Phys. Oceanogr.*, 12, 256–271.
- Goodman, L., E.R. Levine, and R.G. Lueck, 2006: [On measuring the terms of the turbulent kinetic energy budget from an AUV.](https://doi.org/10.1175/JTECH1889.1) *J. Atmos. Oceanic Technol.*, 23, 977–990.
- Macoun, P. and R.G. Lueck, 2004: [Modeling the spatial response of the airfoil shear probe using different sized probes.](https://doi.org/10.1175/1520-0426(2004)021%3C0284:MTSROT%3E2.0.CO;2) *J. Atmos. Oceanic Technol.*, 21, 284–297.

### Chi (thermal dissipation) papers

- Batchelor, G.K., 1959: [Small-scale variation of convected quantities like temperature in turbulent fluid.](https://doi.org/10.1017/S002211205900009X) *J. Fluid Mech.*, 5, 113–133.
- Dillon, T.M. and D.R. Caldwell, 1980: [The Batchelor spectrum and dissipation in the upper ocean.](https://doi.org/10.1029/JC085iC04p01910) *J. Geophys. Res.*, 85, 1910–1916.
- Bogucki, D., J.A. Domaradzki, and P.K. Yeung, 1997: [Direct numerical simulations of passive scalars with Pr > 1 advected by turbulent flow.](https://doi.org/10.1017/S0022112097005727) *J. Fluid Mech.*, 343, 111–130.
- Ruddick, B., A. Anis, and K. Thompson, 2000: [Maximum likelihood spectral fitting: The Batchelor spectrum.](https://doi.org/10.1175/1520-0426(2000)017%3C1541:MLSFTB%3E2.0.CO;2) *J. Atmos. Oceanic Technol.*, 17, 1541–1555.
- Peterson, A.K. and I. Fer, 2014: [Dissipation measurements using temperature microstructure from an underwater glider.](https://doi.org/10.1016/j.mio.2014.05.002) *Methods in Oceanography*, 10, 44–69.
- Lueck, R.G., O. Hertzman, and T.R. Osborn, 1977: [The spectral response of thermistors.](https://doi.org/10.1016/0146-6291(77)90565-3) *Deep-Sea Res.*, 24, 951–970.
- Nash, J.D., T.B. Caldwell, M.J. Zelman, and J.N. Moum, 1999: [A thermocouple probe for high-speed temperature measurement in the ocean.](https://doi.org/10.1175/1520-0426(1999)016%3C1474:ATPFHS%3E2.0.CO;2) *J. Atmos. Oceanic Technol.*, 16, 1474–1482.
- Gregg, M.C. and T.B. Meagher, 1980: [The dynamic response of glass rod thermistors.](https://doi.org/10.1029/JC085iC05p02779) *J. Geophys. Res.*, 85, 2779–2786.
- Osborn, T.R. and C.S. Cox, 1972: [Oceanic fine structure.](https://doi.org/10.1080/03091927208236085) *Geophys. Fluid Dyn.*, 3, 321–345.

## Testing

```bash
python -m pytest                          # run all tests
python -m pytest tests/test_epsilon.py    # epsilon pipeline tests only
```

## Development

This project was developed in collaboration with [Claude Code](https://claude.ai/code) (Anthropic's Opus 4.6).

## License

GPLv3 — see [LICENSE](LICENSE) for details.
