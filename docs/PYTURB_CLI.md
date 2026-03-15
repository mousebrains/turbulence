# pyturb-cli — Jesse-compatible VMP processing CLI

`pyturb-cli` provides the same 4-command workflow as
[Jesse's pyturb](https://github.com/oceancascades/pyturb) — `p2nc`, `merge`,
`eps`, `bin` — but backed by the odas_tpw processing engine (SCOR-160 chain,
Macoun & Lueck spatial response correction, optional Goodman coherent noise
removal).

Window parameters (`--fft-len`, `--diss-len`) use the same seconds-to-samples
conversion as pyturb: `n_fft = int(fft_len * fs)` (made even), `n_diss`
rounded to the nearest multiple of `n_fft`, overlap = `n_fft // 2`. Goodman
cleaning is off by default to match pyturb; enable with `--goodman`.

## Installation

```bash
pip install -e ".[dev]"     # from this repository
pyturb-cli --help
```

## Commands

### `pyturb-cli p2nc` — Convert .p files to NetCDF

```bash
pyturb-cli p2nc VMP/*.p -o nc/ --compress --compression-level 4
pyturb-cli p2nc VMP/*.p -o nc/ -n 4        # 4 parallel workers
```

| Flag | Default | Description |
|------|---------|-------------|
| `-o/--output` | `.` | Output directory |
| `--compress/--no-compress` | on | zlib compression |
| `--compression-level` | 4 | 1-9 |
| `-n/--n-workers` | 1 | Parallel workers |
| `--min-file-size` | 100000 | Skip files < N bytes |
| `-w/--overwrite` | off | Overwrite existing files |

### `pyturb-cli merge` — Merge NetCDF files

```bash
pyturb-cli merge nc/*.nc -o merged.nc
pyturb-cli merge nc/*.nc -o merged.nc --dry-run   # preview only
```

| Flag | Default | Description |
|------|---------|-------------|
| `-o/--output` | *required* | Output file |
| `--dry-run` | off | Print summary without writing |
| `-w/--overwrite` | off | Overwrite existing file |

### `pyturb-cli eps` — Compute epsilon and gradT spectra

```bash
pyturb-cli eps VMP/*.p -o eps/
pyturb-cli eps nc/*.nc -o eps/ -d 4.0 -f 1.0
pyturb-cli eps nc/*.nc -o eps/ --direction both --salinity 34.5
pyturb-cli eps VMP/*.p -o eps/ --goodman    # enable Goodman cleaning
```

| Flag | Default | Description |
|------|---------|-------------|
| `-o/--output` | *required* | Output directory |
| `-d/--diss-len` | 4.0 | Dissipation window length [s] |
| `-f/--fft-len` | 1.0 | FFT segment length [s] |
| `-s/--min-speed` | 0.2 | Minimum profiling speed [m/s] |
| `--pressure-smoothing` | 0.25 | Pressure LP filter [s] |
| `-t/--temperature` | `JAC_T` | Temperature variable name |
| `--speed` | `W` | Speed variable name |
| `--direction` | `down` | `down`, `up`, or `both` |
| `--min-profile-pressure` | 0.0 | Minimum pressure [dbar] |
| `--peaks-height` | 25.0 | Peak height [dbar] |
| `--peaks-distance` | 200 | Peak distance [samples] |
| `--peaks-prominence` | 25.0 | Peak prominence [dbar] |
| `--despike-passes` | 6 | Despike iterations |
| `--salinity` | 35.0 | Salinity [PSU] |
| `--goodman/--no-goodman` | off | Goodman coherent noise removal |
| `-a/--aux` | — | Auxiliary CTD NetCDF |
| `-n/--n-workers` | 1 | Parallel workers |
| `-w/--overwrite` | off | Overwrite existing files |

**Output naming**: `{stem}_p{NNNN}.nc` (0-indexed, 4-digit)

**Output variables** (matching pyturb convention):
- `eps_1`, `eps_2` — epsilon per shear probe [W/kg]
- `k_max_1`, `k_max_2` — upper integration wavenumber [cpm]
- `pressure`, `W`, `temperature`, `nu`, `salinity`, `density`
- `S_sh1`, `S_sh2` — cleaned shear wavenumber spectra
- `S_gradT1`, `S_gradT2` — temperature gradient spectra
- `frequency` coordinate [Hz], `k` coordinate [cpm]

### `pyturb-cli bin` — Depth-bin profiles

```bash
pyturb-cli bin eps/*.nc -o binned.nc -b 2.0
pyturb-cli bin eps/*.nc -o binned.nc --dmax 500 --pressure
```

| Flag | Default | Description |
|------|---------|-------------|
| `-o/--output` | `binned_profiles.nc` | Output file |
| `-b/--bin-width` | 2.0 | Bin width [m] |
| `--dmin/--dmax` | 0 / 1000 | Depth range [m] |
| `--lat` | 45.0 | Latitude for P→depth |
| `-p/--pressure` | off | Bin by pressure instead of depth |
| `-v/--vars` | `eps_1,eps_2,...` | Variables to bin |
| `-n/--n-workers` | 1 | Parallel workers |

## Differences from pyturb

| Feature | pyturb | pyturb-cli |
|---------|--------|------------|
| CLI framework | Typer | argparse |
| Shear spectrum | Custom | SCOR-160 (Lueck 2024) |
| Noise removal | — | Goodman (opt-in via `--goodman`) |
| Spatial correction | — | Macoun & Lueck (2004) |
| Window parameters | `int(s*fs)`, even | Same (matching pyturb) |
| Profile detection | profinder | scipy.signal.find_peaks |
| Package | standalone | part of odas_tpw |

## Typical workflow

```bash
# 1. Convert raw .p files to NetCDF
pyturb-cli p2nc VMP/*.p -o nc/ -n 4

# 2. Optionally merge deployments
pyturb-cli merge nc/*.nc -o merged.nc

# 3. Compute epsilon and temperature gradient spectra
pyturb-cli eps nc/*.nc -o eps/ -d 4.0 -f 1.0 --direction down

# 4. Depth-bin the results
pyturb-cli bin eps/*.nc -o binned.nc -b 2.0
```
