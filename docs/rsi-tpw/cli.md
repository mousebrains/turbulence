# CLI Reference

All commands are available through the `rsi-tpw` command:

```
rsi-tpw <subcommand> [options]
```

## Subcommands

| Subcommand | Description |
|------------|-------------|
| `rsi-tpw info`     | Print summary of `.p` file(s) |
| `rsi-tpw cutp`     | Copy a short `.p` record range for debugging |
| `rsi-tpw nc`       | Convert `.p` files to NetCDF |
| `rsi-tpw patch-template` | Scaffold a config edit-spec YAML from a `.p` file |
| `rsi-tpw patch-config`   | Edit config fields in `.p` file(s), writing new files |
| `rsi-tpw prof`     | Extract profiles from `.p` or full-record `.nc` files |
| `rsi-tpw eps`      | Compute epsilon (TKE dissipation) |
| `rsi-tpw chi`      | Compute chi (thermal variance dissipation) |
| `rsi-tpw pipeline` | Run full pipeline (`.p` → epsilon → chi) |
| `rsi-tpw init`     | Generate a template YAML configuration file |
| `rsi-tpw ql`       | Interactive quick-look viewer |
| `rsi-tpw dl`       | Interactive dissipation quality viewer |
| `rsi-tpw ml`       | Interactive mixing viewer (N²/dT·dz⁻¹/K_T/Γ/K_ρ) |
| `rsi-tpw bench`    | Bench-test diagnostic (quick_bench figures + checklist) |

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

## `rsi-tpw cutp`

Copy a contiguous range of complete data records from a `.p` file into a new
valid `.p` file. This is byte-level debugging support, not a pressure- or
profile-aware scientific extraction. The header and configuration string are
copied unchanged. Absolute time is correct only for `--start 0`; for `--start
N>0`, the data is shifted relative to the copied timestamp. The header record
count is not authoritative because local readers derive the count from file
size.

```bash
rsi-tpw cutp VMP/file.p -o debug_segment.p --start 300 --n-records 60
```

| Flag | Description |
|------|-------------|
| `-o`, `--output FILE` | Output `.p` file (required) |
| `-s`, `--start N` | First data record to copy, 0-based after the config record (default: 0) |
| `-n`, `--n-records N` | Number of complete data records to copy (default: 60) |
| `-f`, `--force`, `--overwrite` | Overwrite output file if it exists |

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

## `rsi-tpw patch-template`

Scaffold a YAML edit spec from a `.p` file, pre-filled with the file's current
editable values and channel names. Edit it, then apply it with
`rsi-tpw patch-config`.

```bash
rsi-tpw patch-template VMP/file.p                  # print to stdout
rsi-tpw patch-template VMP/file.p -o edits.yaml    # write to a file
```

| Flag | Description |
|------|-------------|
| `-o`, `--output YAML` | Output path (default: print to stdout) |
| `--force` | Overwrite an existing output file |

## `rsi-tpw patch-config`

Edit selected configuration fields — `[instrument_info]`, `[cruise_info]`, and
per-channel calibration — embedded in a `.p` file, driven by a YAML edit spec.
**The original file is never modified**; a patched copy is written into `--out`
with each change annotated inline and the full original configuration embedded
(commented) for recovery. If a file's targeted values already match, that file
is reported and left unwritten.

Acquisition-defining parameters (the `[matrix]` stanza and `[root]`
`rate`/`recsize`/`no-fast`/`no-slow`) cannot be addressed and so can never be
corrupted. Every value in the YAML must be a quoted string, so the text written
to the file is exactly what you typed (e.g. `0.1130` stays `0.1130`).

```bash
rsi-tpw patch-template VMP/file.p -o edits.yaml      # 1. scaffold
#                                                      2. edit edits.yaml
rsi-tpw patch-config VMP/file.p --edits edits.yaml --out patched/   # 3. apply
rsi-tpw patch-config VMP/file.p --edits edits.yaml --out patched/ --dry-run
```

Example `edits.yaml`:

```yaml
note: "Corrected sh1 sensitivity from cal sheet; switched to upward profiling"
author: "Jane Doe"
instrument_info:
  vehicle: "rvmp"
channels:
  sh1:                 # only sh1 is touched; sh2 is untouched
    sens: "0.0812"
    sn: "M2732"
```

| Flag | Description |
|------|-------------|
| `--edits YAML` | YAML edit spec, e.g. from `patch-template` (required) |
| `-o`, `--out DIR` | Output directory for patched `.p` files (required) |
| `--dry-run` | Show the configuration diff; write nothing |
| `--add-keys` | Allow adding keys that do not already exist |
| `--batch-cal` | Allow per-channel calibration edits across multiple files |

## `rsi-tpw prof`

Extract profiles from `.p` or full-record `.nc` files.

```bash
rsi-tpw prof VMP/*.p -o profiles/
```

| Flag | Description |
|------|-------------|
| `-o`, `--output DIR` | Output directory (required) |
| `-j`, `--jobs N` | Parallel workers (0 = all cores, default: 1) |
| `--P-min FLOAT` | Minimum pressure [dbar] (default: 0.5) |
| `--W-min FLOAT` | Minimum fall rate [dbar/s] (default: 0.3) |
| `--direction {auto,up,down,glide,horizontal}` | Profile direction (default: auto, from vehicle) |
| `--vehicle NAME` | Vehicle type override (e.g. slocum_glider, vmp) |
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
| `--vehicle NAME` | Vehicle type override (e.g. slocum_glider, vmp) |
| `--no-goodman` | Disable Goodman coherent noise removal |
| `--f-AA FLOAT` | Anti-aliasing filter cutoff [Hz] (default: 98) |
| `--salinity FLOAT` | Salinity [PSU] for viscosity (default: 35, fixed S) |

The output NetCDF contains two distinct spectral-fit quality metrics:

- `fom` — ratio of observed to Nasmyth-model variance in the fit range;
  values near 1.0 indicate a good fit. This is **not** the ATOMIX FM.
- `FM` — the Lueck (2022) MAD-based figure of merit. Good fits approach 0;
  ATOMIX recommends rejecting estimates with FM > ~1.15.

## `rsi-tpw chi`

Compute thermal variance dissipation rate (chi) from FP07 thermistor spectra.

```bash
# Method 2: spectral fitting (no epsilon needed)
rsi-tpw chi VMP/*.p -o chi/

# Method 1: from known epsilon (eps_NN/ subdirectories are searched automatically)
rsi-tpw chi VMP/*.p --epsilon-dir epsilon/ -o chi/

# Batchelor spectrum model (Kraichnan is the default)
rsi-tpw chi VMP/*.p --spectrum-model batchelor -o chi/
```

> **Note:** `rsi-tpw eps -o epsilon/` writes its output into a hash-tracked
> subdirectory (`epsilon/eps_00/`, see
> [output_directories.md](output_directories.md)). `--epsilon-dir` searches
> the given directory and then its `eps_*` subdirectories (most recently
> modified first), matching both `{stem}_eps.nc` and per-profile
> `{stem}_prof001_eps.nc` names; multiple per-profile files are concatenated
> along time so every chi window pairs with its own profile's epsilon. Only
> when no matching epsilon files exist does chi fall back to Method 2 for
> that file, with a console warning.

| Flag | Description |
|------|-------------|
| `-o`, `--output DIR` | Output directory (required) |
| `-j`, `--jobs N` | Parallel workers (0 = all cores, default: 1) |
| `--fft-length N` | FFT segment length [samples] (default: 1024) |
| `--diss-length N` | Dissipation window [samples] (default: 4×fft-length) |
| `--overlap N` | Window overlap [samples] (default: diss-length//2) |
| `--speed FLOAT` | Fixed profiling speed [m/s] (default: from dP/dt) |
| `--direction {auto,up,down,glide,horizontal}` | Profile direction (default: auto, from vehicle) |
| `--vehicle NAME` | Vehicle type override (e.g. slocum_glider, vmp) |
| `--fp07-model {single_pole,double_pole}` | FP07 transfer function (default: single_pole) |
| `--epsilon-dir DIR` | Directory with epsilon `.nc` files for Method 1 (the `eps_NN/` subdirectory; see warning above). If omitted, or if no matching `{stem}_eps.nc` exists, Method 2 is used |
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
| `--vehicle NAME` | Vehicle type override (e.g. slocum_glider, vmp) |
| `--speed FLOAT` | Fixed profiling speed [m/s] (default: from dP/dt) |
| `--eps-fft-length N` | FFT length for epsilon (default: 1024) |
| `--chi-fft-length N` | FFT length for chi (default: 1024) |
| `--no-goodman` | Disable Goodman noise removal for epsilon and chi |
| `--fp07-model {single_pole,double_pole}` | FP07 transfer function (default: single_pole) |
| `--spectrum-model {batchelor,kraichnan}` | Spectrum model for chi (default: kraichnan) |
| `--f-AA FLOAT` | Anti-aliasing filter cutoff [Hz] (default: 98) |
| `--salinity FLOAT` | Fixed salinity [PSU] fallback for viscosity (default: 35). **Ignored on conductivity-equipped instruments** — the pipeline resolves per-sample salinity from the measured JAC C/T, which takes precedence (see note below). |

> **Salinity precedence (`pipeline` only):** `run_pipeline` prefers measured
> salinity (from the instrument's own JAC conductivity/temperature) over
> `--salinity`, so on a conductivity-equipped instrument (including the
> campaign VMP-250) an explicit `--salinity` does **not** change the epsilon /
> chi / N² viscosity. It applies only when no conductivity channel is present.
> (The `eps` and `chi` subcommands, which have no CTD path, do honor
> `--salinity`.)

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
| `--diss-length N` | Dissipation window [samples] (default: 4×fft-length) |
| `--f-AA FLOAT` | Anti-aliasing filter cutoff [Hz] (default: 98) |
| `--no-goodman` | Disable Goodman coherent noise removal |
| `--direction {auto,up,down,glide,horizontal}` | Profile direction (default: auto, from vehicle) |
| `--vehicle NAME` | Vehicle type override (e.g. slocum_glider, vmp) |
| `--W-min FLOAT` | Minimum fall rate [dbar/s] (default: 0.3) |
| `--spec-P-range P_MIN P_MAX` | Pressure range [dbar] for spectral calculations |
| `--chi-method {1,2}` | Chi method: 1 = from epsilon, 2 = spectral fit (default: 1) |
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
| `--diss-length N` | Dissipation window [samples] (default: 4×fft-length) |
| `--f-AA FLOAT` | Anti-aliasing filter cutoff [Hz] (default: 98) |
| `--no-goodman` | Disable Goodman coherent noise removal |
| `--direction {auto,up,down,glide,horizontal}` | Profile direction (default: auto, from vehicle) |
| `--vehicle NAME` | Vehicle type override (e.g. slocum_glider, vmp) |
| `--W-min FLOAT` | Minimum fall rate [dbar/s] (default: 0.3) |
| `--spec-P-range P_MIN P_MAX` | Pressure range [dbar] for spectral calculations |

## `rsi-tpw bench`

Bench-test diagnostic — a Python port of ODAS `quick_bench.m`, extended with an
automatic evaluation of the **Rockland Bench Test Review Checklist**. Run it on a
short recording taken with the instrument at rest on foam and dummy probes
installed, to catch corroded connections, dead channels and excessive electronic
noise before deployment. See [bench.md](bench.md) for the full checklist mapping.

```bash
rsi-tpw bench VMP/SN479_0001.p                 # writes figures + checklist to ./bench/
rsi-tpw bench VMP/SN479_0001.p -o bench/ --show # save AND open interactive windows
rsi-tpw bench VMP/SN479_0001.p --show          # display only, write nothing
```

Two figures are produced (`_timeseries`, `_spectra`, plus `_ctclu` when JAC /
turbidity / chlorophyll channels exist), all in **raw counts** / **counts²·Hz⁻¹**
so the spectra can be compared directly against the instrument's RSI calibration
report. The checklist text (`_checklist.txt`) reports each quantitative criterion
as PASS/FAIL with its measured value and flags the subjective ones (spectral
shape, "similar to each other", spikes) as REVIEW.

| Flag | Description |
|------|-------------|
| `-o`, `--output DIR` | Output directory for figures + checklist (default: `./bench/`, unless `--show` is given without `-o`, which only displays) |
| `--show` | Open the figures in interactive windows |
| `--sn SN` | Serial number for figure titles/filenames (default: from config) |
| `--fft-sec FLOAT` | FFT segment length for spectra [seconds] (default: 2.0) |
| `--dpi INT` | Figure resolution when saving (default: 150) |
| `--format {png,pdf,both,pdf-bundle}` | Figure format: one file per figure (`png`/`pdf`/`both`), or `pdf-bundle` for a single multi-page PDF (default: png) |
