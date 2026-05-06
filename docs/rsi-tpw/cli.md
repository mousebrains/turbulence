# CLI Reference

All commands are available through the `rsi-tpw` command:

```
rsi-tpw <subcommand> [options]
```

## Subcommands

| Subcommand | Description |
|------------|-------------|
| `rsi-tpw info`     | Print summary of `.p` file(s) |
| `rsi-tpw nc`       | Convert `.p` files to NetCDF |
| `rsi-tpw prof`     | Extract profiles from `.p` or full-record `.nc` files |
| `rsi-tpw eps`      | Compute epsilon (TKE dissipation) |
| `rsi-tpw chi`      | Compute chi (thermal variance dissipation) |
| `rsi-tpw pipeline` | Run full pipeline (`.p` → epsilon → chi) |
| `rsi-tpw init`     | Generate a template YAML configuration file |
| `rsi-tpw ql`       | Interactive quick-look viewer |
| `rsi-tpw dl`       | Interactive dissipation quality viewer |

## Global Options

| Flag | Description |
|------|-------------|
| `--version` | Show version and exit |
| `-c`, `--config YAML` | Configuration file. See [configuration.md](configuration.md) |

## `rsi-tpw info`

Print summary information about `.p` file(s).

```bash
rsi-tpw info VMP/*.p
```

## `rsi-tpw nc`

Convert `.p` files to NetCDF4 format.

```bash
rsi-tpw nc VMP/*.p -o nc/           # output to directory
rsi-tpw nc VMP/file.p -o file.nc    # single file, explicit name
rsi-tpw nc VMP/*.p -o nc/ -j 4      # parallel with 4 workers
```

| Flag | Description |
|------|-------------|
| `-o`, `--output PATH` | Output file or directory |
| `-j`, `--jobs N` | Parallel workers (0 = all cores, default: 1) |

## `rsi-tpw prof`

Extract profiles from `.p` or full-record `.nc` files.

```bash
rsi-tpw prof VMP/*.p -o profiles/
```

| Flag | Description |
|------|-------------|
| `-o`, `--output DIR` | Output directory (required) |
| `--P-min FLOAT` | Minimum pressure [dbar] (default: 0.5) |
| `--W-min FLOAT` | Minimum fall rate [dbar/s] (default: 0.3) |
| `--direction {auto,up,down,glide,horizontal}` | Profile direction (default: auto, from vehicle) |
| `--min-duration FLOAT` | Minimum profile duration [s] (default: 7) |

## `rsi-tpw eps`

Compute TKE dissipation rate (epsilon) from shear probe spectra.

```bash
rsi-tpw eps VMP/*.p -o epsilon/
rsi-tpw eps profiles/*_prof*.nc -o epsilon/
rsi-tpw eps VMP/*.p -o epsilon/ -j 0          # all cores
rsi-tpw eps VMP/*.p -o epsilon/ --salinity 34.5
```

| Flag | Description |
|------|-------------|
| `-o`, `--output DIR` | Output directory (required) |
| `-j`, `--jobs N` | Parallel workers (0 = all cores, default: 1) |
| `--fft-length N` | FFT segment length [samples] (default: 1024) |
| `--diss-length N` | Dissipation window [samples] (default: 4×fft-length) |
| `--overlap N` | Window overlap [samples] (default: diss-length//2) |
| `--speed FLOAT` | Fixed profiling speed [m/s] (default: from dP/dt) |
| `--direction {auto,up,down,glide,horizontal}` | Profile direction (default: auto, from vehicle) |
| `--no-goodman` | Disable Goodman coherent noise removal |
| `--f-AA FLOAT` | Anti-aliasing filter cutoff [Hz] (default: 98) |
| `--salinity FLOAT` | Salinity [PSU] for viscosity (default: 35, fixed S) |

## `rsi-tpw chi`

Compute thermal variance dissipation rate (chi) from FP07 thermistor spectra.

```bash
# Method 2: spectral fitting (no epsilon needed)
rsi-tpw chi VMP/*.p -o chi/

# Method 1: from known epsilon
rsi-tpw chi VMP/*.p --epsilon-dir epsilon/ -o chi/

# Kraichnan spectrum model
rsi-tpw chi VMP/*.p --spectrum-model kraichnan -o chi/
```

| Flag | Description |
|------|-------------|
| `-o`, `--output DIR` | Output directory (required) |
| `-j`, `--jobs N` | Parallel workers (0 = all cores, default: 1) |
| `--fft-length N` | FFT segment length [samples] (default: 1024) |
| `--diss-length N` | Dissipation window [samples] (default: 4×fft-length) |
| `--overlap N` | Window overlap [samples] (default: diss-length//2) |
| `--speed FLOAT` | Fixed profiling speed [m/s] (default: from dP/dt) |
| `--direction {auto,up,down,glide,horizontal}` | Profile direction (default: auto, from vehicle) |
| `--fp07-model {single_pole,double_pole}` | FP07 transfer function (default: single_pole) |
| `--epsilon-dir DIR` | Directory with epsilon `.nc` files for Method 1 |
| `--no-goodman` | Disable Goodman coherent noise removal |
| `--fit-method {mle,iterative}` | Method 2 fitting algorithm (default: iterative) |
| `--spectrum-model {batchelor,kraichnan}` | Theoretical spectrum model (default: kraichnan) |
| `--f-AA FLOAT` | Anti-aliasing filter cutoff [Hz] (default: 98) |
| `--salinity FLOAT` | Salinity [PSU] for viscosity (default: 35, fixed S) |

## `rsi-tpw pipeline`

Run the full processing pipeline from raw `.p` files through epsilon and chi.

```bash
rsi-tpw pipeline VMP/*.p -o results/
```

| Flag | Description |
|------|-------------|
| `-o`, `--output DIR` | Base output directory (required) |
| `--direction {auto,up,down,glide,horizontal}` | Profile direction (default: auto, from vehicle) |
| `--speed FLOAT` | Fixed profiling speed [m/s] (default: from dP/dt) |
| `--eps-fft-length N` | FFT length for epsilon (default: 1024) |
| `--chi-fft-length N` | FFT length for chi (default: 1024) |
| `--no-goodman` | Disable Goodman noise removal for epsilon and chi |
| `--fp07-model {single_pole,double_pole}` | FP07 transfer function (default: single_pole) |
| `--spectrum-model {batchelor,kraichnan}` | Spectrum model for chi (default: kraichnan) |
| `--f-AA FLOAT` | Anti-aliasing filter cutoff [Hz] (default: 98) |
| `--salinity FLOAT` | Salinity [PSU] for viscosity (default: 35, fixed S) |

## `rsi-tpw init`

Generate a template YAML configuration file with all defaults and comments.

```bash
rsi-tpw init                    # writes config.yaml
rsi-tpw init my_config.yaml     # custom filename
rsi-tpw init --force config.yaml  # overwrite existing
```

| Flag | Description |
|------|-------------|
| `--force` | Overwrite existing file |

## `rsi-tpw ql`

Interactive quick-look viewer with profile navigation. Opens a multi-panel
display showing pressure, shear spectra, epsilon, and chi for each profile.

```bash
rsi-tpw ql VMP/*.p
rsi-tpw ql VMP/*.p --fft-length 512
```

| Flag | Description |
|------|-------------|
| `--fft-length N` | FFT segment length [samples] (default: 1024) |
| `--f-AA FLOAT` | Anti-aliasing filter cutoff [Hz] (default: 98) |
| `--no-goodman` | Disable Goodman coherent noise removal |
| `--direction {auto,up,down,glide,horizontal}` | Profile direction (default: auto, from vehicle) |
| `--spec-P-range P_MIN P_MAX` | Pressure range [dbar] for spectral calculations |
| `--chi-method {1,2}` | Chi method: 1 = from epsilon, 2 = MLE fit (default: 1) |
| `--spectrum-model {batchelor,kraichnan}` | Theoretical spectrum model (default: kraichnan) |

## `rsi-tpw dl`

Interactive dissipation quality viewer. Compares epsilon, chi (Batchelor vs
Kraichnan), and Lueck (2022) figure of merit (FM) with profile navigation.

```bash
rsi-tpw dl VMP/*.p
```

| Flag | Description |
|------|-------------|
| `--fft-length N` | FFT segment length [samples] (default: 1024) |
| `--f-AA FLOAT` | Anti-aliasing filter cutoff [Hz] (default: 98) |
| `--no-goodman` | Disable Goodman coherent noise removal |
| `--direction {auto,up,down,glide,horizontal}` | Profile direction (default: auto, from vehicle) |
| `--spec-P-range P_MIN P_MAX` | Pressure range [dbar] for spectral calculations |
