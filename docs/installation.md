# Installation

## Requirements

- Python >= 3.12
- Dependencies: `numpy`, `netCDF4`, `scipy`, `xarray`, `gsw`, `ruamel.yaml`

## Install from Source

```bash
# Editable install with test dependencies (recommended for development)
pip install -e ".[dev]"

# Standard install
pip install .

# With pipx (installs CLI tools in isolated environment)
pipx install .
```

## Install from GitHub

```bash
pip install git+https://github.com/mousebrains/turbulence.git
pipx install git+https://github.com/mousebrains/turbulence.git
```

## Development Setup

```bash
git clone https://github.com/mousebrains/turbulence.git
cd turbulence
pip install -e ".[dev]"

# Run tests
python -m pytest

# Lint
ruff check src/ tests/

# Type check
mypy src/microstructure_tpw/
```
