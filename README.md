# rsi-python

[![CI](https://github.com/mousebrains/turbulence/actions/workflows/ci.yml/badge.svg)](https://github.com/mousebrains/turbulence/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![codecov](https://codecov.io/gh/mousebrains/turbulence/graph/badge.svg?token=RwbKxeE7rA)](https://codecov.io/gh/mousebrains/turbulence)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)
[![MATLAB: MISS_HIT](https://img.shields.io/badge/MATLAB-MISS__HIT-blue.svg)](https://misshit.org/)

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
| [Mixing Efficiency](docs/mixing_efficiency.md) | Mathematics of the dissipation ratio |
| [Best Practices](docs/best_practices.md) | Guidance for comparing turbulence measurements |
| [Bibliography](docs/bibliography.md) | Consolidated references |
| [Output Directories](docs/output_directories.md) | Sequential hash-tracked output scheme |
| [MATLAB](matlab/MATLAB.md) | MATLAB chi implementation |
| [Changelog](CHANGELOG.md) | Version history |

## References

This package is a Python implementation derived from the
[Rockland Scientific ODAS MATLAB Library](https://rocklandscientific.com/support/software/) (v4.5.1)
and associated
[Technical Notes](https://rocklandscientific.com/support/technical-notes/)
(TN-028, TN-051, TN-061).

See [docs/bibliography.md](docs/bibliography.md) for the full list of references.

## Testing

```bash
python -m pytest                          # run all tests
python -m pytest tests/test_epsilon.py    # epsilon pipeline tests only
```

## Development

This project was developed in collaboration with [Claude Code](https://claude.ai/code) (Anthropic's Opus 4.6).

## License

GPLv3 — see [LICENSE](LICENSE) for details.
