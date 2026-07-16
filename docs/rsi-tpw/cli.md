# CLI Reference

All commands are available through the `rsi-tpw` command:

```
rsi-tpw <subcommand> [options]
```

## Subcommands

| Subcommand | Description |
|------------|-------------|
| `rsi-tpw info`     | Print summary of `.p` file(s) |
| `rsi-tpw config`   | Print a `.p` file's raw embedded configuration (INI) record |
| `rsi-tpw cutp`     | Copy a short `.p` record range for debugging |
| `rsi-tpw v1to6`    | Translate legacy header-v1 `.p` files to v6 ([legacy_v1.md](legacy_v1.md)) |
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
| `rsi-tpw sensors`  | Inventory shear/FP07 sensors across a `.p` file tree (see [sensors.md](sensors.md)) |

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

## `rsi-tpw config`

Print the raw embedded configuration record — the `setup.cfg`-style INI text
stored in record 0, containing the address `matrix` and every channel's
calibration coefficients — to stdout. This is the authoritative way to inspect a
coefficient in place (e.g. a suspect pressure `coef2`).

```bash
rsi-tpw config VMP/file.p                       # dump the whole config
rsi-tpw config VMP/file.p | grep -A6 'name .*= P$'   # just the pressure channel
```

It reads only the header and config record, so — unlike `info` — it also works
on startup or truncated files that carry a config but no data records. With more
than one file, each config is preceded by a `# ===== <path> =====` banner. To
*change* a coefficient, see [`patch-config`](#rsi-tpw-patch-config).

## `rsi-tpw cutp`

Copy a contiguous range of complete data records from a `.p` file into a new
valid `.p` file. This is byte-level debugging support, not a pressure- or
profile-aware scientific extraction. For `--start N>0` the record-0 header
timestamp is advanced by N record durations (config `recsize`, default 1.0 s)
so the segment's absolute start time matches the copied data; the rest of the
header and the configuration string are copied unchanged. The header record
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

## `rsi-tpw v1to6`

Translate legacy ODAS header-version-1 `.p` files (pre-2015 instruments;
record 0 holds a binary address matrix and the configuration lives in an
external setup file) into standard v6 files that every other tool accepts
unchanged. Data records are copied byte-for-byte; the embedded INI is
synthesized from the setup file with full provenance keys. See
[legacy_v1.md](legacy_v1.md).

```bash
rsi-tpw v1to6 VMP_002/TAI_013_00*.p -o translated/ --sens sh1=0.0893,sh2=0.0558
```

| Flag | Description |
|------|-------------|
| `-o`, `--output DIR` | Output directory for translated files, same basenames (required) |
| `--setup-file PATH` | Setup file to use (default: auto-detect `setup.txt`, `setup*.txt`, `setup*.cfg` next to each `.p` file, then one level up) |
| `--sens NAME=VAL[,...]` | Shear-probe sensitivities (e.g. `sh1=0.0893,sh2=0.0558`); overrides `<name>_sens:` setup keys |
| `-f`, `--force` | Overwrite existing output files |

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
| `--W-min FLOAT` | Minimum fall rate [dbar/s] (default: auto — 0.3 free-fall, 0.05 glide/horizontal) |
| `--direction {auto,up,down,glide,horizontal}` | Profile direction (default: auto, from vehicle) |
| `--vehicle NAME` | Vehicle type override (e.g. slocum_glider, vmp) |
| `--min-duration FLOAT` | Minimum profile duration [s] (default: 7) |

An unreadable file in a batch (e.g. a startup file with no data records, or a
truncated `.p`) prints `ERROR: ...` and processing continues; the exit status
is 1 only when **every** file failed. `rsi-tpw info` behaves the same way.

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
| `--speed FLOAT` | Fixed profiling speed [m/s] (default: from dP/dt). Mutually exclusive with `--speed-method em/flight` |
| `--speed-method {pressure,em,flight}` | Through-water speed model (default: pressure = \|dP/dt\|). `em` uses the `U_EM` flowmeter channel; `flight` uses the inviscid glider flight model \|W\|/sin(\|pitch\|−aoa) from the inclinometers; an explicit `pressure` forces \|dP/dt\| even when the source carries a precomputed `speed_fast` channel (a perturb per-profile file). The choice is recorded in the product attrs (`speed_source`). The `hotel` speed method is perturb-only (hotel channels are merged there); perturb per-profile files carry its result as the precomputed `speed_fast` channel, which is used automatically |
| `--aoa FLOAT` | Angle of attack [deg] for `--speed-method flight` (default: 3.0) |
| `--direction {auto,up,down,glide,horizontal}` | Profile direction (default: auto, from vehicle) |
| `--vehicle NAME` | Vehicle type override (e.g. slocum_glider, vmp) |
| `--W-min FLOAT` | Profile-detection fall-rate floor [dbar/s] (default: auto — 0.3 free-fall, 0.05 glide/horizontal) |
| `--no-goodman` | Disable Goodman coherent noise removal |
| `--f-AA FLOAT` | Anti-aliasing filter cutoff [Hz] (default: 98) |
| `--salinity PSU\|measured` | Salinity for viscosity: a PSU value, or `measured` = computed from the conductivity/temperature channels via TEOS-10 (default: 35, fixed S) |
| `--temperature NAME\|degC` | Reference temperature for viscosity: a channel name (e.g. `T2`, `JAC_T`, or a hotel temperature channel), a fixed value [°C], or `auto` = first plausible of `T1`..`Tn`, `T`, `JAC_T` (default: auto). Implausible channels (railed, drifting, mostly non-finite) are skipped with a warning; an explicitly named channel that fails QC warns but is honored. The selection is recorded in the product attrs (`temperature_source`/`temperature_qc`). |
| `--conductivity NAME` | Conductivity channel for `--salinity measured` (default: auto = `JAC_C` when present) |

`eps` (and `chi`) exit with status 1 when **no** input file produced output —
whether because every file failed or because no profiles were detected — and
print a final `N of M file(s) produced output` summary, so a batch that
silently produced nothing fails a `set -e` script instead of masquerading as
success.

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
| `--speed FLOAT` | Fixed profiling speed [m/s] (default: from dP/dt). Mutually exclusive with `--speed-method em/flight` |
| `--speed-method {pressure,em,flight}` | Through-water speed model (default: pressure = \|dP/dt\|; same semantics as `eps`) |
| `--aoa FLOAT` | Angle of attack [deg] for `--speed-method flight` (default: 3.0) |
| `--direction {auto,up,down,glide,horizontal}` | Profile direction (default: auto, from vehicle) |
| `--vehicle NAME` | Vehicle type override (e.g. slocum_glider, vmp) |
| `--W-min FLOAT` | Profile-detection fall-rate floor [dbar/s] (default: auto — 0.3 free-fall, 0.05 glide/horizontal) |
| `--fp07-model {single_pole,double_pole}` | FP07 transfer function (default: single_pole) |
| `--epsilon-dir DIR` | Directory with epsilon `.nc` files for Method 1 (the `eps_NN/` subdirectory; see warning above). If omitted, or if no matching `{stem}_eps.nc` exists, Method 2 is used |
| `--no-goodman` | Disable Goodman coherent noise removal |
| `--fit-method {mle,iterative}` | Method 2 fitting algorithm (default: iterative) |
| `--spectrum-model {batchelor,kraichnan}` | Theoretical spectrum model (default: kraichnan) |
| `--f-AA FLOAT` | Anti-aliasing filter cutoff [Hz] (default: 98) |
| `--salinity PSU\|measured` | Salinity for viscosity: a PSU value, or `measured` = computed from the conductivity/temperature channels via TEOS-10 (default: 35, fixed S) |
| `--temperature NAME\|degC` | Reference temperature for viscosity/κ_T: a channel name, a fixed value [°C], or `auto` = first plausible of `T1`..`Tn`, `T`, `JAC_T` (default: auto; same semantics as `eps`) |
| `--conductivity NAME` | Conductivity channel for `--salinity measured` (default: auto = `JAC_C` when present) |

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
| `--W-min FLOAT` | Profile-detection fall-rate floor [dbar/s] (default: auto — 0.3 free-fall, 0.05 glide/horizontal; also feeds the L2 section selector) |
| `--speed FLOAT` | Fixed profiling speed [m/s] (default: from dP/dt) |
| `--speed-method {pressure,em,flight}` | Through-water speed model (default: pressure = \|dP/dt\|). Also settable as `epsilon.speed_method` in the YAML config |
| `--aoa FLOAT` | Angle of attack [deg] for `--speed-method flight` (default: 3.0) |
| `--eps-fft-length N` | FFT length for epsilon (default: 1024) |
| `--chi-fft-length N` | FFT length for chi (default: 1024) |
| `--no-goodman` | Disable Goodman noise removal for epsilon and chi |
| `--fp07-model {single_pole,double_pole}` | FP07 transfer function (default: single_pole) |
| `--spectrum-model {batchelor,kraichnan}` | Spectrum model for chi (default: kraichnan) |
| `--f-AA FLOAT` | Anti-aliasing filter cutoff [Hz] (default: 98) |
| `--salinity PSU\|measured` | Fixed salinity [PSU] fallback for viscosity (default: 35). **Ignored on conductivity-equipped instruments** — the pipeline resolves per-sample salinity from the measured JAC C/T, which takes precedence (see note below). `measured` maps to that automatic behavior (never treated as a number). |
| `--temperature NAME\|degC` | Reference temperature for viscosity: a channel name, a fixed value [°C], or `auto` = first plausible of `T1`..`Tn`, `T`, `JAC_T` (default: auto; same semantics as `eps`). Recorded in the L4 products as `temperature_source`/`temperature_qc`. |
| `--conductivity NAME` | Conductivity channel for the measured practical salinity (default: auto = `JAC_C` when present) |

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
| `--W-min FLOAT` | Minimum fall rate [dbar/s] (default: 0.3, or 0.05 for glide/horizontal) |
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
| `--W-min FLOAT` | Minimum fall rate [dbar/s] (default: 0.3, or 0.05 for glide/horizontal) |
| `--spec-P-range P_MIN P_MAX` | Pressure range [dbar] for spectral calculations |

## `rsi-tpw ml`

Interactive mixing viewer. Opens an interactive display of the background
stratification (N², dT/dz) and derived diapycnal-mixing quantities (K_T, Γ,
K_ρ) with profile navigation.

```bash
rsi-tpw ml VMP/*.p
rsi-tpw ml VMP/*.p --salinity 34.5
```

| Flag | Description |
|------|-------------|
| `--fft-length N` | FFT segment length [samples] (default: 1024) |
| `--diss-length N` | Dissipation window [samples] (default: 4×fft-length) |
| `--f-AA FLOAT` | Anti-aliasing filter cutoff [Hz] (default: 98) |
| `--no-goodman` | Disable Goodman coherent noise removal |
| `--direction {auto,up,down,glide,horizontal}` | Profile direction (default: auto, from vehicle) |
| `--vehicle NAME` | Vehicle type override (e.g. slocum_glider, vmp) |
| `--W-min FLOAT` | Minimum fall rate [dbar/s] (default: 0.3, or 0.05 for glide/horizontal) |
| `--spec-P-range P_MIN P_MAX` | Pressure range [dbar] to highlight on the profiles (default: none) |
| `--salinity FLOAT` | Fixed practical salinity [PSU] for stratification (default: measured from JAC C/T, else 35) |

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

A file that cannot be evaluated (unreadable/corrupt `.p`) is reported per-file
without aborting the batch, and the process **exits 1** when any file failed —
so a scripted pre-deployment check can gate on the exit code. (Checklist
FAIL/REVIEW verdicts do not affect the exit code; they are findings, not
errors.)

| Flag | Description |
|------|-------------|
| `-o`, `--output DIR` | Output directory for figures + checklist (default: `./bench/`, unless `--show` is given without `-o`, which only displays) |
| `--show` | Open the figures in interactive windows |
| `--sn SN` | Serial number for figure titles/filenames (default: from config) |
| `--fft-sec FLOAT` | FFT segment length for spectra [seconds] (default: 2.0) |
| `--dpi INT` | Figure resolution when saving (default: 150) |
| `--format {png,pdf,both,pdf-bundle}` | Figure format: one file per figure (`png`/`pdf`/`both`), or `pdf-bundle` for a single multi-page PDF (default: png) |
