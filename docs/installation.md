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
| `pip install -e .` from a clone, or `PYTHONPATH=<checkout>/src` | `0.4.0 (git v0.4.0-28-g5f6572b9a1e8)`: `git describe` of the checkout the code was imported from, 28 commits after `v0.4.0`. `, uncommitted changes, source 1c0e5a7d9b42` is added when a file under `src/odas_tpw/` is modified, staged or new (an ignored one too, if Python could import it), or `pyproject.toml` is modified; docs and tests do not count. The `source` hash covers the package's `.py` files, so two runs from modified trees can at least be told apart. A shallow clone with no tags gives the bare commit. |
| `pip install git+https://github.com/...` | `0.4.0 (git 5f6572b9a1e8)`, the commit pip recorded at install time. |
| `pip install microstructure-tpw` (PyPI) | `0.4.0 (release, git 2846bf3666dc)`: the release workflow stamps the tagged commit into the wheel it publishes, and refuses a tag that does not match `project.version`. A wheel built anywhere else and installed by name (a local wheelhouse, say) prints `0.4.0 (not a release build, commit unknown)`. |
| `pip install .` (not editable) | `0.4.0 (built from file:///..., commit unknown)`: nothing records which commit was built. Use an editable or `git+` install when you need the commit. |

Whatever cannot be established is printed as `commit unknown` with the reason,
never guessed: for example `running from <dir>, not the installed
microstructure-tpw` when `PYTHONPATH` shadows an installed copy, or `git refused
the checkout: it is owned by another user` for a shared checkout (see git's
`safe.directory`).

`, installed metadata says 0.3.1` inside the parentheses means the editable install's
metadata is older than `pyproject.toml` (it is written once, at install time).
Rerun `pip install -e .` to refresh it.

### Which code made this file?

`--version` names the code installed *now*. Every product records the code
that *wrote it*, in the same words:

- **NetCDF products** (perturb's profiles, diss, chi, CTD, binned and combo
  files; rsi-tpw's L1, profiles, epsilon, chi and L4–L6 files; hotel files;
  `fp07-pair` corrections) carry a `software_version` global attribute and a
  `history` line per processing step, oldest first:

  ```console
  $ ncdump -h diss_00/SN479__SN479_0006_prof001.nc | grep -A1 -E ':software_version|:history'
      :software_version = "microstructure-tpw 0.4.0 (git v0.4.0-28-g5f6572b9a1e8)" ;
      :history = "2026-10-16T23:47:24+00:00: profile 1 extracted by microstructure-tpw 0.4.0 (git v0.4.0-28-g5f6572b9a1e8)\n",
          "2026-10-16T23:47:24+00:00: epsilon by microstructure-tpw 0.4.0 (git v0.4.0-28-g5f6572b9a1e8)" ;
  ```

  A product made from one input keeps that input's history: a diss file's
  first line is the extraction of the profile it was computed from. Binned and
  combo files start a fresh history, because their first input's lineage
  would misdescribe the rest; a `history` set in perturb's `netcdf:` block is
  kept and the combo step appended to it. Instead they list, in
  `input_software_versions`, every code that wrote their inputs (and their
  inputs' inputs). They are rebuilt from cached per-file products, which can
  come from another code state, so more than one line there is worth a look;
  `(none recorded)` means an input predates these stamps.
- **perturb's logs**: the first line of each `logs/run_<timestamp>.log` and a
  line in each worker log (a worker imports the code afresh).
- **Records**: the `rsi-tpw patch-config` / `fp07-cal patch` banner in a
  patched `.p` file, `rsi-tpw v1to6`'s `translator`, `fp07-cal`'s
  `coefficients.json` (`software_version`), and the first line of
  `mr-clocksync`'s `clock_report.txt`.

The version is looked up once per process, and a failed lookup is recorded as
`unknown (version lookup failed: ...)` rather than stopping a write.

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
