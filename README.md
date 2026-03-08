# rsi-python

Python tools for reading Rockland Scientific microprofiler data and computing turbulent dissipation rates from VMP (Vertical Microstructure Profiler) and MicroRider instruments.

## Overview

**rsi-python** provides a complete processing pipeline for ocean turbulence measurements from Rockland Scientific instruments equipped with shear probes and fast thermistors (FP07). The package reads proprietary `.p` binary data files, converts channels to physical units, detects profiles, and computes both the rate of dissipation of turbulent kinetic energy (epsilon) and the rate of dissipation of thermal variance (chi), following the methods described in the Rockland Scientific ODAS MATLAB Library and associated Technical Notes.

### What it computes

- **Epsilon (TKE dissipation rate)** from shear probe spectra, including:
  - Iterative despiking of shear signals
  - Goodman coherent noise removal using accelerometer cross-spectra
  - Macoun & Lueck wavenumber correction for shear probe spatial response
  - Nasmyth universal spectrum fitting with Lueck's improved coefficients
  - Iterative variance correction using Lueck's resolved-variance model
  - Integration limit detection via polynomial fit to log-log spectra

- **Chi (thermal variance dissipation rate)** from FP07 thermistor spectra ([detailed mathematics](docs/chi_mathematics.md)), including:
  - Batchelor and Kraichnan theoretical temperature gradient spectra
  - FP07 single-pole and double-pole transfer function correction
  - Electronics noise model (Johnson + amplifier + anti-aliasing + ADC)
  - Method 1: chi from known epsilon (shear probes) with unresolved variance correction
  - Method 2a: MLE Batchelor spectrum fitting (Ruddick et al. 2000)
  - Method 2b: Iterative integration (Peterson & Fer 2014)

## Installation

```bash
# From source (editable, with test dependencies)
pip install -e ".[dev]"

# From source (standard)
pip install .

# With pipx (installs CLI tools in isolated environment)
pipx install .

# From GitHub
pip install git+https://github.com/mousebrains/turbulence.git
pipx install git+https://github.com/mousebrains/turbulence.git
```

Requires Python >= 3.10. Dependencies: `numpy`, `netCDF4`, `scipy`, `xarray`.

## CLI

All commands are available through the `rsi-tpw` command:

```
rsi-tpw <subcommand> [options]
```

| Subcommand | Description |
|------------|-------------|
| `rsi-tpw info`     | Print summary of `.p` file(s) |
| `rsi-tpw nc`       | Convert `.p` files to NetCDF |
| `rsi-tpw prof`     | Extract profiles from `.p` or full-record `.nc` files |
| `rsi-tpw eps`      | Compute epsilon (TKE dissipation) |
| `rsi-tpw chi`      | Compute chi (thermal variance dissipation) |
| `rsi-tpw pipeline` | Run full pipeline (`.p` → epsilon → chi) |

Legacy standalone commands (`p2nc`, `pinfo`, `p2prof`, `p2eps`, `p2chi`) are still available.

## Pipeline

The processing pipeline has four stages. Each stage produces NetCDF files, and any later stage can start from any earlier stage's output:

```
.p files ──> nc ──> full-record .nc ──> prof ──> per-profile .nc ──> eps ──> epsilon .nc
                                                                  ──> chi ──> chi .nc
```

### Full pipeline (recommended)

Run all stages at once from raw `.p` files through epsilon and chi:

```bash
# Process all .p files, output to results/
rsi-tpw pipeline VMP/*.p -o results/

# Writes results/epsilon/ and results/chi/
```

### Stage 1: Convert `.p` files to NetCDF

```bash
rsi-tpw nc VMP/*.p -o nc/
```

### Stage 2: Extract profiles

Detects profiling segments from pressure data and writes per-profile NetCDF files:

```bash
rsi-tpw prof VMP/ARCTERX_Thompson_2025_SN479_0005.p -o profiles/
```

### Stage 3: Compute epsilon

Computes TKE dissipation rate from shear probe spectra:

```bash
# From raw .p files (profiles detected automatically)
rsi-tpw eps VMP/ARCTERX_Thompson_2025_SN479_0005.p -o epsilon/

# From per-profile .nc files
rsi-tpw eps profiles/*_prof*.nc -o epsilon/

# Parallel processing
rsi-tpw eps VMP/*.p -o epsilon/ -j 0
```

### Stage 4: Compute chi

Computes thermal variance dissipation rate from FP07 thermistor spectra:

```bash
# Method 1: chi from known epsilon (uses shear probe results)
rsi-tpw chi VMP/*.p --epsilon-dir epsilon/ -o chi/

# Method 2: chi without epsilon (MLE Batchelor spectrum fitting)
rsi-tpw chi VMP/*.p -o chi/

# Method 2 with Kraichnan spectrum model
rsi-tpw chi VMP/*.p --spectrum-model kraichnan -o chi/
```

### Python API

```python
from rsi_python import PFile, get_diss, get_chi

# Read a .p file
pf = PFile("VMP/ARCTERX_Thompson_2025_SN479_0005.p")
pf.channels["sh1"]   # shear probe 1 [s⁻¹]
pf.channels["P"]     # pressure [dbar]

# Compute epsilon (returns list of xarray.Datasets, one per profile)
eps_results = get_diss("VMP/ARCTERX_Thompson_2025_SN479_0005.p")
ds = eps_results[0]
ds["epsilon"]         # dissipation rate [W/kg]
ds["spec_shear"]      # shear wavenumber spectra
ds["spec_nasmyth"]    # fitted Nasmyth spectra

# Compute chi from known epsilon (Method 1)
chi_results = get_chi("VMP/ARCTERX_Thompson_2025_SN479_0005.p",
                      epsilon_ds=eps_results[0])
ds = chi_results[0]
ds["chi"]             # thermal dissipation rate [K²/s]
ds["spec_gradT"]      # temperature gradient spectra
ds["spec_batch"]      # fitted Batchelor spectra

# Compute chi without epsilon (Method 2: MLE fitting)
chi_results = get_chi("VMP/ARCTERX_Thompson_2025_SN479_0005.p")
ds = chi_results[0]
ds["chi"]             # thermal dissipation rate [K²/s]
ds["epsilon_T"]       # epsilon estimated from temperature
```

## Modules

| Module | Description |
|--------|-------------|
| `p_file.py` | `PFile` class: reads `.p` binary files, parses headers, demultiplexes address matrix, converts to physical units |
| `channels.py` | Sensor conversion functions (raw counts to physical units) |
| `convert.py` | Full-record NetCDF export (`rsi-tpw nc`) |
| `profile.py` | Profile detection and per-profile NetCDF extraction (`rsi-tpw prof`) |
| `dissipation.py` | Core epsilon calculation with multi-source input (`rsi-tpw eps`) |
| `chi.py` | Chi (thermal variance dissipation) calculation, Methods 1 and 2 (`rsi-tpw chi`) |
| `batchelor.py` | Batchelor and Kraichnan temperature gradient spectra |
| `fp07.py` | FP07 thermistor transfer function and electronics noise model |
| `scalar_spectra.py` | Temperature gradient spectrum computation |
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
