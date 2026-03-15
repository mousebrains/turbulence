# microstructure-tpw

[![CI](https://github.com/mousebrains/turbulence/actions/workflows/ci.yml/badge.svg)](https://github.com/mousebrains/turbulence/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![codecov](https://codecov.io/gh/mousebrains/turbulence/graph/badge.svg?token=RwbKxeE7rA)](https://codecov.io/gh/mousebrains/turbulence)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)
[![MATLAB: MISS_HIT](https://img.shields.io/badge/MATLAB-MISS__HIT-blue.svg)](https://misshit.org/)

Python tools for reading Rockland Scientific microprofiler data and computing turbulent dissipation rates from VMP (Vertical Microstructure Profiler) and MicroRider instruments.

## Overview

`microstructure-tpw` provides a complete processing pipeline for ocean turbulence measurements from Rockland Scientific instruments equipped with shear probes and fast thermistors (FP07). The package reads proprietary `.p` binary data files, converts channels to physical units, detects profiles, and computes both the rate of dissipation of turbulent kinetic energy (epsilon) and the rate of dissipation of thermal variance (chi), following the methods described in the Rockland Scientific ODAS MATLAB Library and associated Technical Notes.

- **Epsilon (TKE dissipation rate)** from shear probe spectra ([detailed mathematics](docs/epsilon_mathematics.md)), including Goodman coherent noise removal, Nasmyth spectrum fitting, and Macoun & Lueck wavenumber correction.

- **Chi (thermal variance dissipation rate)** from FP07 thermistor spectra ([detailed mathematics](docs/chi_mathematics.md)), including Batchelor/Kraichnan spectrum models, FP07 transfer function correction, and MLE spectral fitting.

A **MATLAB implementation** of the chi calculation is also available — see [matlab/MATLAB.md](matlab/MATLAB.md).

The package is organized into four subpackages under `odas_tpw`:

- **rsi** — Rockland Scientific instrument I/O, NetCDF conversion, profiles, epsilon/chi orchestration
- **chi** — Chi (thermal variance dissipation) calculation
- **scor160** — ATOMIX shear-probe benchmark processing and shared physics modules
- **perturb** — Full campaign processing pipeline (trim, merge, calibrate, compute, bin)

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
from odas_tpw.rsi.pipeline import run_pipeline
from pathlib import Path

# Full pipeline: .p → profiles → epsilon → chi → binning → combine
run_pipeline([Path("VMP/file.p")], Path("results/"))

# Or use the modular API
from odas_tpw.rsi.dissipation import compute_diss_file
from odas_tpw.rsi.chi_io import compute_chi_file

compute_diss_file("VMP/file.p", "epsilon/")
compute_chi_file("VMP/file.p", "chi/")
```

> **Note:** `get_diss()` and `get_chi()` still work for backward compatibility
> but are deprecated in favor of `run_pipeline()` or the modular
> `compute_diss_file()` / `compute_chi_file()` functions.

## Documentation

### rsi-tpw (science library)

| Document | Description |
|----------|-------------|
| [CLI Reference](docs/rsi-tpw/cli.md) | All `rsi-tpw` subcommands and flags |
| [Configuration](docs/rsi-tpw/configuration.md) | YAML config file format and all parameter defaults |
| [Pipeline](docs/rsi-tpw/pipeline.md) | Processing stages and data flow |
| [Python API](docs/rsi-tpw/python_api.md) | Using microstructure-tpw from Python code |
| [Output Directories](docs/rsi-tpw/output_directories.md) | Sequential hash-tracked output scheme |
| [Vectorization](docs/rsi-tpw/vectorization.md) | Vectorized compute internals |

### perturb (batch pipeline)

| Document | Description |
|----------|-------------|
| [Pipeline](docs/perturb/pipeline.md) | Batch processing stages and data flow |
| [CLI Reference](docs/perturb/cli.md) | All `perturb` subcommands and flags |
| [Configuration](docs/perturb/configuration.md) | YAML config file format for perturb |
| [Parallel Scaling](docs/perturb/parallel.md) | Benchmark results for multi-core scaling |
| [Modules](docs/perturb/modules.md) | Module-level reference |
| [pyturb Comparison](docs/perturb/pyturb_comparison.md) | Comparison with oceancascades/pyturb |

### scor160 (ATOMIX benchmark)

| Document | Description |
|----------|-------------|
| [CLI Reference](docs/scor160/cli.md) | All `scor160-tpw` subcommands and flags |

### Shared

| Document | Description |
|----------|-------------|
| [Epsilon Mathematics](docs/epsilon_mathematics.md) | TKE dissipation algorithm details |
| [Chi Mathematics](docs/chi_mathematics.md) | Thermal dissipation algorithm details |
| [Mixing Efficiency](docs/mixing_efficiency.md) | Mathematics of the dissipation ratio |
| [Best Practices](docs/best_practices.md) | Guidance for comparing turbulence measurements |
| [Bibliography](docs/bibliography.md) | Consolidated references |
| [Installation](docs/installation.md) | Installation options |
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
