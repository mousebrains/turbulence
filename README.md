# rsi-python

[![CI](https://github.com/mousebrains/turbulence/actions/workflows/ci.yml/badge.svg)](https://github.com/mousebrains/turbulence/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![codecov](https://codecov.io/gh/mousebrains/turbulence/graph/badge.svg)](https://codecov.io/gh/mousebrains/turbulence)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Python tools for reading Rockland Scientific microprofiler data and computing turbulent dissipation rates from VMP (Vertical Microstructure Profiler) and MicroRider instruments.

## Overview

**rsi-python** provides a complete processing pipeline for ocean turbulence measurements from Rockland Scientific instruments equipped with shear probes and fast thermistors (FP07). The package reads proprietary `.p` binary data files, converts channels to physical units, detects profiles, and computes both the rate of dissipation of turbulent kinetic energy (epsilon) and the rate of dissipation of thermal variance (chi), following the methods described in the Rockland Scientific ODAS MATLAB Library and associated Technical Notes.

### What it computes

- **Epsilon (TKE dissipation rate)** from shear probe spectra ([detailed mathematics](docs/epsilon_mathematics.md)), including Goodman coherent noise removal, Nasmyth spectrum fitting, and Macoun & Lueck wavenumber correction.

- **Chi (thermal variance dissipation rate)** from FP07 thermistor spectra ([detailed mathematics](docs/chi_mathematics.md)), including Batchelor/Kraichnan spectrum models, FP07 transfer function correction, and MLE spectral fitting.

A **MATLAB implementation** of the chi calculation is also available — see [matlab/MATLAB.md](matlab/MATLAB.md).

## Installation

```bash
pip install -e ".[dev]"    # editable install with dev dependencies
pip install .              # standard install
```

See [docs/installation.md](docs/installation.md) for more options.

## Quick Start

```bash
# Full pipeline: .p files → epsilon → chi
rsi-tpw pipeline VMP/*.p -o results/

# Or run individual stages
rsi-tpw eps VMP/*.p -o epsilon/
rsi-tpw chi VMP/*.p --epsilon-dir epsilon/ -o chi/
```

```python
from rsi_python import PFile, get_diss, get_chi

eps_results = get_diss("VMP/file.p")
chi_results = get_chi("VMP/file.p", epsilon_ds=eps_results[0])
```

## Documentation

| Document | Description |
|----------|-------------|
| [CLI Reference](docs/cli.md) | All `rsi-tpw` subcommands and flags |
| [Configuration](docs/configuration.md) | YAML config file format and all parameter defaults |
| [Pipeline](docs/pipeline.md) | Processing stages and data flow |
| [Python API](docs/python_api.md) | Using rsi-python from Python code |
| [Epsilon Mathematics](docs/epsilon_mathematics.md) | TKE dissipation algorithm details |
| [Chi Mathematics](docs/chi_mathematics.md) | Thermal dissipation algorithm details |
| [Output Directories](docs/output_directories.md) | Sequential hash-tracked output scheme |
| [MATLAB](matlab/MATLAB.md) | MATLAB chi implementation |
| [Changelog](CHANGELOG.md) | Version history |

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
- **CEOAS Ocean Mixing Group** (Oregon State University) — [mixing.ceoas.oregonstate.edu](https://mixing.ceoas.oregonstate.edu/)

## Testing

```bash
python -m pytest                          # run all tests
python -m pytest tests/test_epsilon.py    # epsilon pipeline tests only
```

## Development

This project was developed in collaboration with [Claude Code](https://claude.ai/code) (Anthropic's Opus 4.6).

## License

GPLv3 — see [LICENSE](LICENSE) for details.
