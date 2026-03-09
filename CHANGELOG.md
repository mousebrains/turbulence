# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- YAML configuration file system with `rsi-tpw init` template generation
- Three-way parameter merge: defaults <- config file <- CLI flags
- Cumulative hash-tracked sequential output directories (`eps_00/`, `chi_00/`, ...)
- Configuration validation with clear error messages for unknown sections/keys
- Resolved `config.yaml` written to each output directory for reproducibility
- Comprehensive documentation in `docs/`: configuration reference, CLI reference, Python API, pipeline guide
- CHANGELOG

### Changed
- Output directories now use sequential naming with parameter hash deduplication
- Each output directory includes a `config.yaml` recording the exact parameters used

## [0.1.0] - 2025-03-08

### Added
- `PFile` class for reading Rockland Scientific `.p` binary files
- Channel conversion functions (raw counts to physical units) for all sensor types
- NetCDF4 conversion with CF-1.13 compliance
- Profile detection from pressure time series
- Epsilon (TKE dissipation rate) calculation from shear probe spectra
  - Nasmyth universal spectrum fitting (Lueck improved coefficients)
  - Iterative despiking of shear signals
  - Goodman coherent noise removal using accelerometer cross-spectra
  - Macoun & Lueck wavenumber correction
  - Figure of merit (fom) and K_max_ratio QC metrics
- Chi (thermal variance dissipation rate) calculation from FP07 thermistor spectra
  - Method 1: from known epsilon with FP07 transfer function correction
  - Method 2a: MLE Batchelor spectrum fitting (Ruddick et al. 2000)
  - Method 2b: Iterative integration (Peterson & Fer 2014)
  - Batchelor and Kraichnan theoretical spectrum models
  - FP07 single-pole and double-pole transfer functions
  - Electronics noise model
- Unified `rsi-tpw` CLI with subcommands: `info`, `nc`, `prof`, `eps`, `chi`, `pipeline`
- Parallel processing support (`-j` flag)
- Seawater property functions via gsw/TEOS-10: viscosity, density, buoyancy frequency
- MATLAB implementation of chi calculation
- CI pipeline: ruff linting, mypy type checking, pytest on Python 3.12/3.13
- Mathematical documentation for epsilon and chi algorithms
