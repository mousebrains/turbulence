# Installation

## Requirements

- Python >= 3.12
- Dependencies: `numpy`, `netCDF4`, `scipy`, `xarray`, `gsw`, `matplotlib`, `ruamel.yaml`

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

## Which version am I running?

Every command takes `--version` (`rsi-tpw`, `perturb`, `perturb-plot`,
`perturb-diag`, `scor160-tpw`, `fp07-cal`, `fp07-pair`, `mr-clocksync`,
`dinkum-hotel`, `erddap-hotel`). Paste its line into a bug report:

```console
$ perturb --version
perturb 0.4.0 (git v0.4.0-28-g5f6572b9a1e8)
```

The release number alone does not identify the code: `pyproject.toml` keeps the
last release's number for every commit after it. So the line carries the commit
whenever it can be established:

| Installed with | `--version` shows |
|----------------|-------------------|
| `pip install -e .` from a clone, or `PYTHONPATH=<checkout>/src` | `0.4.0 (git v0.4.0-28-g5f6572b9a1e8)`: `git describe` of the checkout the code was imported from, 28 commits after `v0.4.0`. `, uncommitted changes` is added when a file under `src/odas_tpw/` is modified, staged or new (an ignored one too, if Python could import it), or `pyproject.toml` is modified; docs and tests do not count. A shallow clone with no tags gives the bare commit. |
| `pip install git+https://github.com/...` | `0.4.0 (git 5f6572b9a1e8)`, the commit pip recorded at install time. |
| `pip install microstructure-tpw` (PyPI) | `0.4.0 (release, git 2846bf3666dc)`: the release workflow stamps the tagged commit into the wheel it publishes, and refuses a tag that does not match `project.version`. A wheel built anywhere else and installed by name (a local wheelhouse, say) prints `0.4.0 (not a release build, commit unknown)`. |
| `pip install .` (not editable) | `0.4.0 (built from file:///..., commit unknown)`: nothing records which commit was built. Use an editable or `git+` install when you need the commit. |

Whatever cannot be established is printed as `commit unknown` with the reason,
never guessed: for example `running from <dir>, not the installed
microstructure-tpw` when `PYTHONPATH` shadows an installed copy, or `git refused
the checkout: it is owned by another user` for a shared checkout (see git's
`safe.directory`).

`; installed metadata says 0.3.1` at the end means the editable install's
metadata is older than `pyproject.toml` (it is written once, at install time).
Rerun `pip install -e .` to refresh it.

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
mypy src/odas_tpw/
```
