# Plan: Python `perturb` Package

## Context

The Matlab [perturb](https://github.com/jessecusack/perturb) package is a batch-processing pipeline for Rockland VMP/MicroRider data. It takes raw `.p` files through discovery → trimming → merging → profile extraction → FP07 calibration → CT alignment → GPS → seawater properties → dissipation → chi → binning → combo assembly → NetCDF.

This project ports perturb to Python, dropping `.mat` output only. Chi is integrated into the dissipation step. Depth/time binning, combined (combo) NetCDF files, and CTD time-binning with GPS are all retained. The package lives in the same repo as `rsi` and reuses its science modules.

## Architecture

**Same repo, second package** under `src/perturb/`. Both packages are discovered by `setuptools.packages.find` from `src/`. One `pip install -e ".[dev]"` installs everything. Separate CLI entry point: `perturb`.

## File Layout

```
src/perturb/
    __init__.py
    cli.py               # `perturb` entry point with subcommands
    config.py             # YAML config schema, defaults, merge, hashing
    discover.py           # Glob .p files
    trim.py               # Trim corrupt final records
    merge.py              # Merge split .p files
    fp07_cal.py           # FP07 in-situ calibration (Steinhart-Hart)
    ct_align.py           # CT sensor cross-correlation alignment
    gps.py                # GPS providers (NaN, fixed, CSV, NetCDF)
    seawater.py           # Full seawater property chain (gsw)
    bottom.py             # Bottom crash detection
    top_trim.py           # Top trimming (initial instabilities)
    epsilon_combine.py    # Multi-probe epsilon combining (mk_epsilon_mean)
    ctd.py                # CTD time-binning with GPS and seawater properties
    binning.py            # Depth/time binning engine for profiles and diss+chi
    combo.py              # Merge binned data across files → combined NetCDF
    pipeline.py           # Per-file and full-pipeline orchestration
    netcdf_schema.py      # CF/ACDD NetCDF schema definitions (replaces Combo.json/CTD.json)
tests/
    test_perturb_*.py     # Tests (flat in existing tests/ dir)
```

## pyproject.toml Changes

Add to `[project.scripts]`:
```toml
perturb = "perturb.cli:main"
```

## Reuse from rsi

| Import | Used By |
|--------|---------|
| `PFile`, `parse_config` | trim, merge, fp07_cal, pipeline |
| `get_profiles`, `_smooth_fall_rate` | pipeline (profile detection) |
| `get_diss` | pipeline (epsilon computation) |
| `get_chi` | pipeline (chi Method 1, using epsilon) |
| `despike` | pipeline (via get_diss) |
| `visc`, `density`, `buoyancy_freq` | seawater, pipeline |
| `config.resolve_output_dir`, `compute_hash` | config (extend pattern) |
| `profile.extract_profiles` | profiles subcommand (pattern reference) |

## CLI Design — Subcommands

```
perturb init config.yaml                         # generate template config
perturb run config.yaml [VMP/*.p]                # full pipeline (all stages)
perturb trim config.yaml [VMP/*.p]               # trim corrupt final records
perturb merge config.yaml [VMP/*.p]              # merge split .p files
perturb profiles config.yaml [VMP/*.p]           # extract profiles (+ FP07 cal, CT align, GPS, seawater)
perturb diss config.yaml [VMP/*.p]               # dissipation + chi (from profile NetCDFs)
perturb ctd config.yaml [VMP/*.p]                # CTD time-binning with GPS
perturb bin config.yaml                          # bin profiles and diss+chi by depth/time
perturb combo config.yaml                        # assemble combined NetCDF across all files
```

All subcommands accept:
- `config.yaml` as first positional arg (required except for `init`)
- Optional `.p` file patterns to override `files.p_file_root` / `files.p_file_pattern`
- `-j/--jobs N` to override `parallel.jobs`
- `--dry-run` to show what would be processed

The `run` subcommand executes: trim → merge → profiles → diss → ctd → bin → combo sequentially.

Individual subcommands can be run independently when re-processing a stage (e.g., re-running `diss` with different parameters without re-extracting profiles).

## Config Schema (`config.yaml`)

```yaml
files:
  p_file_root: "VMP/"
  p_file_pattern: "**/*.p"
  output_root: "results/"
  trim: true
  merge: false

gps:
  source: "nan"          # nan | fixed | csv | netcdf
  lat: null
  lon: null
  file: null
  time_col: "t"
  lat_col: "lat"
  lon_col: "lon"
  max_time_diff: 60      # warning threshold [s]

profiles:
  P_min: 0.5
  W_min: 0.3
  direction: "down"
  min_duration: 7.0

fp07:
  calibrate: true
  order: 2
  max_lag_seconds: 10
  reference: "JAC_T"

ct:
  align: true
  T_name: "JAC_T"
  C_name: "JAC_C"

bottom:
  enable: false
  depth_window: 4.0
  depth_minimum: 10.0
  speed_factor: 0.3

top_trim:
  enable: false
  dz: 0.5
  min_depth: 1.0
  max_depth: 50.0

epsilon:
  fft_length: 256
  diss_length: null
  overlap: null
  goodman: true
  f_AA: 98.0
  fit_order: 3
  despike_thresh: 8
  despike_smooth: 0.5
  salinity: null
  epsilon_minimum: 1e-13

chi:
  fft_length: 512
  diss_length: null
  overlap: null
  fp07_model: "single_pole"
  goodman: true
  f_AA: 98.0
  fit_method: "iterative"
  spectrum_model: "kraichnan"
  salinity: null

ctd:
  enable: true
  bin_width: 60          # time bin width [s]
  T_name: "JAC_T"
  C_name: "JAC_C"

binning:
  method: "depth"        # depth | time
  width: 1.0             # bin width [m or s]
  aggregation: "mean"    # mean | median
  diss_width: null        # override bin width for diss (default: same as width)
  diss_aggregation: null  # override aggregation for diss

netcdf:
  title: null
  institution: null
  creator_name: null
  creator_email: null
  project: null
  references: null
  comment: null

parallel:
  jobs: 1
```

## Output Directory Structure

```
output_root/
  trimmed/                         # trimmed .p files (if trim: true)
  merged/                          # merged .p files (if merge: true)
  profiles_00/                     # per-profile NetCDF
    SN479_0001_prof01.nc           #   calibrated FP07, aligned CT, GPS, seawater
    SN479_0001_prof02.nc
    ...
  diss_00/                         # per-profile dissipation + chi
    SN479_0001_prof01.nc           #   epsilon (per-probe + epsilonMean) + chi
    SN479_0001_prof02.nc
    ...
  ctd_00/                          # per-file CTD time-binned
    SN479_0001.nc                  #   time-binned scalars, seawater props, GPS
    SN479_0002.nc
    ...
  profiles_binned_00/              # per-file binned profiles
    SN479_0001.nc                  #   depth/time-binned profile variables
    SN479_0002.nc
    ...
  diss_binned_00/                  # per-file binned diss+chi
    SN479_0001.nc
    SN479_0002.nc
    ...
  profiles_combo_00/               # combined profiles (all files merged)
    combo.nc                       #   2D: bins × profiles
  diss_combo_00/                   # combined diss+chi (all files merged)
    combo.nc
  ctd_combo_00/                    # combined CTD (all files merged)
    combo.nc
```

Directory versioning follows `rsi.config` pattern: `prefix_NN/` with `.params_sha256_{hash}` signature files. Changing parameters creates a new versioned directory.

### Profile NetCDF contents (`profiles_00/`)
- Dims: `time_fast`, `time_slow`
- Channels: all instrument channels (calibrated FP07, aligned CT)
- Coords: lat, lon (from GPS)
- Seawater vars: SP, SA, CT, sigma0, rho, depth
- Attrs: profile metadata (P_start, P_end, duration, speed, trim_depth, bottom_depth)

### Dissipation+Chi NetCDF contents (`diss_00/`)
- Dim: `time` (dissipation window centers)
- Epsilon vars (per-probe): `epsilon`, `K_max`, `fom`, `FM`, `K_max_ratio`, `method`, `mad`
- Combined: `epsilonMean`, `epsilonLnSigma` (from mk_epsilon_mean)
- Chi vars (per-probe): `chi`, `epsilon_T`, `kB`, `K_max_T`, `fom`, `K_max_ratio`
- Common: `speed`, `nu`, `P_mean`, `T_mean`
- Spectra: `K`, `F`, `spec_shear`, `spec_nasmyth`, `spec_gradT`, `spec_batch`
- Attrs: parameters used, source profile path

### CTD NetCDF contents (`ctd_00/`)
- Dim: `time` (time bin centers)
- Vars: temp, cond, pressure, depth, SP, SA, CT, sigma0, rho, DO, Chlorophyll, Turbidity, lat, lon
- Attrs: bin width, source file, processing level

### Binned NetCDF contents (`profiles_binned_00/`, `diss_binned_00/`)
- Dims: `bin` (depth or time bins), `profile`
- For profiles: all profile variables aggregated into bins
- For diss: epsilon, epsilonMean, chi, QC metrics aggregated into bins
- Info table: one row per profile with metadata (name, sn, t0, t1, lat, lon, depth range)

### Combo NetCDF contents (`*_combo_00/combo.nc`)
- Dims: `bin`, `profile` (all profiles across all files, sorted by time)
- Same variables as binned, but merged across all source .p files
- CF-1.8 / ACDD-1.3 compliant global attributes from config `netcdf` section
- Profile-dimension variables from info table (stime, etime, lat, lon, sn, etc.)

## Module Details

### `config.py` — YAML Config
Extends `rsi.config` patterns: three-way merge (defaults ← file ← CLI), SHA-256 hashing, sequential versioned output dirs.
- `DEFAULTS` dict with all sections and their default values
- `load_config(path) -> dict` — load YAML, validate sections/keys
- `merge_config(section, file_values, cli_overrides) -> dict`
- `resolve_output_dir(base, prefix, section, params, upstream) -> Path`
- `generate_template(path)` — write fully-commented YAML template

### `discover.py` — File Discovery
- `find_p_files(root: Path, pattern: str) -> list[Path]`
- Filters `.p` extension (case-insensitive), excludes `_original.p` and dotfiles
- Returns sorted list

### `trim.py` — Corrupt Record Trimming
- `trim_p_file(source: Path, output_dir: Path) -> Path`
- Reads header via `PFile._parse_header()` to get `record_size`
- If file size has fractional last record: copies only complete records
- Returns path to trimmed file (or original if no trimming needed)
- **Reuses:** `rsi.p_file._detect_endian`, `_parse_header`, header constants

### `merge.py` — Split File Merging
- `find_mergeable_files(p_files: list[Path]) -> list[list[Path]]`
  - Chains by: matching config hash, sequential file numbers, same record_size + endianness
- `merge_p_files(chain: list[Path], output_dir: Path) -> Path`
  - Concatenates data records from chain into single output file
  - First file's header/config preserved; subsequent files contribute only data records
- **Reuses:** `rsi.p_file._detect_endian`, `_parse_header`

### `fp07_cal.py` — FP07 In-Situ Calibration
- `fp07_calibrate(pf: PFile, profiles: list[tuple[int,int]], reference: str, order: int, max_lag_seconds: float) -> dict`
- Steps:
  1. Extract FP07 resistance ratio `ln(R_T/R_0)` from raw counts using channel config (`E_B`, `a`, `b`, `G`)
  2. Low-pass filter FP07 to match reference sensor bandwidth
  3. Per-profile: cross-correlate `diff(filtered_FP07)` vs `diff(reference)` → find lag
  4. Compute median lag across profiles
  5. Steinhart-Hart fit: `1/(T+273.15) = a0 + a1*ln(R) + a2*ln(R)^order`
  6. Apply calibration to both fast and slow FP07 data
- Returns: calibration coefficients, lags, modified channel arrays
- **Reuses:** `rsi.p_file.PFile`, `parse_config`

### `ct_align.py` — CT Sensor Alignment
- `ct_align(T: ndarray, C: ndarray, fs: float, profiles: list[tuple[int,int]]) -> ndarray`
- Per-profile: cross-correlate `diff(T)` vs `diff(C)` (bandpass filtered)
- Weighted median lag, apply shift to conductivity
- Returns shifted C array

### `gps.py` — GPS Providers
- `GPSProvider` protocol with `lat(t: ndarray) -> ndarray`, `lon(t: ndarray) -> ndarray`
- `GPSNaN` — returns NaN
- `GPSFixed(lat, lon)` — constant
- `GPSFromCSV(file, time_col, lat_col, lon_col)` — reads CSV, interpolates
- `GPSFromNetCDF(file, time_var, lat_var, lon_var)` — reads NetCDF, interpolates
- `create_gps(config: dict) -> GPSProvider` — factory dispatching on `source` key

### `seawater.py` — Seawater Properties
- `add_seawater_properties(T, C, P, lat, lon) -> dict[str, ndarray]`
- Returns: SP (practical salinity), SA (absolute salinity), CT (conservative temp), sigma0, rho, depth
- **Uses:** `gsw` functions (gsw.SP_from_C, gsw.SA_from_SP, gsw.CT_from_t, gsw.sigma0, gsw.rho, gsw.z_from_p)

### `bottom.py` — Bottom Crash Detection
- `detect_bottom_crash(depth_fast, Ax, Ay, fs, **params) -> float | None`
- Finds maximum deceleration below `depth_minimum`
- Validates with speed reduction + vibration sensor std
- Returns bottom depth or None

### `top_trim.py` — Top Trimming
- `compute_trim_depths(profiles_data, **params) -> ndarray`
- Per-profile: bin shear/accel/inclinometer/speed by depth, compute std per bin
- Find depth where std drops below median
- Returns per-profile trim depths

### `epsilon_combine.py` — Multi-Probe Combining
- `mk_epsilon_mean(epsilon_ds: xr.Dataset, epsilon_minimum: float = 1e-13) -> xr.Dataset`
- Port of Matlab `mk_epsilon_mean` from `calc_diss_shear.m`:
  1. Floor small values to NaN
  2. Compute Kolmogorov length → physical length → normalized `L_hat`
  3. Compute `var_ln_epsilon = 5.5 / (1 + (L_hat/4)^(7/9))`
  4. 95% CI: `1.96 * sqrt(2) * sigma_ln_epsilon`
  5. Iteratively remove probes outside CI
  6. Geometric mean of surviving probes
- Adds `epsilonMean`, `epsilonLnSigma` to dataset
- **Reuses:** `rsi.dissipation.get_diss` output structure

### `ctd.py` — CTD Time-Binning
- `ctd_bin_file(pf: PFile, gps: GPSProvider, output_dir: Path, **params) -> Path`
- Steps:
  1. Extract slow CT channels (JAC_T, JAC_C) and optional DO, Chlorophyll, Turbidity
  2. Time-bin at configured interval (default 60s)
  3. Compute seawater properties per bin (SP, SA, CT, sigma0, rho, depth)
  4. Interpolate GPS (lat, lon) to bin centers
  5. Write per-file CTD NetCDF
- **Reuses:** `seawater.add_seawater_properties`, `gps.GPSProvider`

### `binning.py` — Depth/Time Binning
- `bin_by_depth(profiles: list[Path], bin_width: float, aggregation: str) -> xr.Dataset`
  - Loads per-profile NetCDFs, bins each by depth, assembles into 2D (bin × profile)
  - For each variable: aggregate within bin using mean or median
- `bin_by_time(profiles: list[Path], bin_width: float, aggregation: str) -> xr.Dataset`
  - Same but bins by time instead of depth
- `bin_diss(diss_files: list[Path], bin_width: float, aggregation: str, method: str) -> xr.Dataset`
  - Bins dissipation+chi estimates by depth or time
  - Handles per-probe variables (e.g., `e_1`, `e_2`) and combined variables (`epsilonMean`)
- Helper: `_bin_array(values, coords, bin_edges, agg_func) -> ndarray`
- Writes per-file binned NetCDF to `profiles_binned_NN/` and `diss_binned_NN/`

### `combo.py` — Combined NetCDF Assembly
- `make_combo(binned_dir: Path, output_dir: Path, schema: dict) -> Path`
  - Loads all binned NetCDFs from a directory
  - Sorts profiles temporally by start time
  - For depth binning: `glue_widthwise` — 2D matrix (bins × all profiles across files)
  - For time binning: `glue_lengthwise` — concatenate all time bins
  - Writes `combo.nc` with CF-1.8 / ACDD-1.3 global attributes
- `make_ctd_combo(ctd_dir: Path, output_dir: Path, schema: dict) -> Path`
  - Concatenates all CTD time-binned files into single combo
- **Schema:** `netcdf_schema.py` provides variable metadata (units, standard_name, long_name) — Python equivalent of Matlab's `Combo.json` and `CTD.json`

### `netcdf_schema.py` — CF/ACDD Schema Definitions
- `COMBO_SCHEMA` dict — variable definitions for profile/diss combo NetCDF
  - Maps table column names → CF standard names, units, long_names
  - e.g., `"t0" → {"nc_name": "stime", "units": "seconds since 1970-01-01", "standard_name": "time"}`
- `CTD_SCHEMA` dict — variable definitions for CTD combo NetCDF
- `apply_schema(ds: xr.Dataset, schema: dict) -> xr.Dataset` — apply CF attributes
- Python replacement for Matlab's `Combo.json` and `CTD.json`

### `pipeline.py` — Orchestration
- `process_file(p_path, config, gps, output_dirs) -> dict`
  - Per-file: PFile → profiles → FP07 cal → CT align → GPS → seawater → bottom → top trim → write profile NetCDFs → per-profile diss+chi → write diss NetCDFs → CTD binning
  - Returns dict of output paths
- Stage runners (called by individual subcommands):
  - `run_trim(config) -> list[Path]`
  - `run_merge(config) -> list[Path]`
  - `run_profiles(config) -> list[Path]`
  - `run_diss(config) -> list[Path]`
  - `run_ctd(config) -> list[Path]`
  - `run_bin(config) -> list[Path]`
  - `run_combo(config) -> list[Path]`
- `run_pipeline(config) -> None`
  - Full: discover → trim → merge → parallel process_file → bin → combo

### `cli.py` — Subcommand CLI
- `main()` — argparse with subcommands: `init`, `run`, `trim`, `merge`, `profiles`, `diss`, `ctd`, `bin`, `combo`
- Each subcommand loads config, extracts CLI overrides, merges, sets up output dirs, calls corresponding `pipeline.run_*`
- Parallel processing via `ProcessPoolExecutor` (at file level)

## Pipeline Data Flow

```
.p files
  → discover (glob from config)
  → trim (optional: fix corrupt final records)
  → merge (optional: concatenate split files)
  → per-file (parallelized):
      → PFile(path)
      → get_profiles() (pressure/speed profile detection)
      → FP07 in-situ calibration (cross-corr lag + Steinhart-Hart)
      → CT alignment (cross-corr shift)
      → GPS assignment (interpolate lat/lon)
      → seawater properties (SP, SA, CT, sigma0, rho, depth)
      → bottom crash detection (optional)
      → top trimming (optional)
      → write per-profile NetCDF → profiles_NN/
      → per-profile:
          → get_diss() (per-probe epsilon)
          → mk_epsilon_mean() (combine probes: 95% CI + geometric mean)
          → get_chi(epsilon_ds=...) (Method 1, using combined epsilon)
          → write per-profile diss+chi NetCDF → diss_NN/
      → CTD time-binning (full file, with GPS + seawater) → ctd_NN/
  → bin profiles by depth/time → profiles_binned_NN/
  → bin diss+chi by depth/time → diss_binned_NN/
  → combo: merge all binned profiles → profiles_combo_NN/combo.nc
  → combo: merge all binned diss+chi → diss_combo_NN/combo.nc
  → combo: merge all CTD → ctd_combo_NN/combo.nc
```

## Implementation Order

Built bottom-up by dependency:

1. **`config.py`** — DEFAULTS dict, YAML load, three-way merge, hashing, output dir management
2. **`discover.py`** — file discovery
3. **`trim.py`** — corrupt record trimming
4. **`merge.py`** — split file merging
5. **`gps.py`** — all four GPS providers
6. **`fp07_cal.py`** — FP07 in-situ calibration
7. **`ct_align.py`** — CT sensor alignment
8. **`seawater.py`** — seawater property chain
9. **`bottom.py`** — bottom crash detection
10. **`top_trim.py`** — top trimming
11. **`epsilon_combine.py`** — mk_epsilon_mean
12. **`ctd.py`** — CTD time-binning with GPS
13. **`binning.py`** — depth/time binning engine
14. **`netcdf_schema.py`** — CF/ACDD schema definitions
15. **`combo.py`** — combined NetCDF assembly
16. **`pipeline.py`** — per-file orchestration + stage runners
17. **`cli.py`** — subcommand CLI
18. **`__init__.py`** — public API
19. **pyproject.toml** — add CLI entry point
20. **Tests** — one test file per module

## Critical Files to Reference

| File | Why |
|------|-----|
| `src/microstructure_tpw/rsi/p_file.py` | PFile class, header parsing, channel reading — reused by trim, merge, fp07_cal |
| `src/microstructure_tpw/rsi/dissipation.py` | `get_diss()` output structure — epsilon_combine must extend it |
| `src/microstructure_tpw/rsi/chi.py` | `get_chi()` with `epsilon_ds` param — Method 1 integration |
| `src/microstructure_tpw/rsi/config.py` | Config patterns to extend: `merge_config`, `compute_hash`, `resolve_output_dir` |
| `src/microstructure_tpw/rsi/cli.py` | CLI patterns: arg parsing, config loading, parallel processing, output dir setup |
| `src/microstructure_tpw/rsi/profile.py` | `get_profiles()`, `extract_profiles()` — profile detection reuse |
| `src/microstructure_tpw/rsi/ocean.py` | `visc()`, `density()` — seawater property base |
| `pyproject.toml` | Add `perturb` CLI entry point |

### Key Matlab source files (https://github.com/jessecusack/perturb)
| File | Maps to |
|------|---------|
| `Code/process_P_files.m` | `pipeline.py` — full orchestration |
| `Code/get_info.m` | `config.py` — parameter defaults and validation |
| `Code/update_paths.m` | `config.py` — output dir versioning with SHA hashing |
| `Code/trim_P_files.m` | `trim.py` |
| `Code/merge_p_files.m` | `merge.py` |
| `Code/mat2profile.m` | `pipeline.py` (profile extraction + enhancement) |
| `Code/fp07_calibration.m` | `fp07_cal.py` |
| `Code/CT_align.m` | `ct_align.py` |
| `Code/calc_diss_shear.m` | `epsilon_combine.py` (mk_epsilon_mean portion) |
| `Code/ctd2binned.m` | `ctd.py` |
| `Code/profile2binned.m` / `diss2binned.m` | `binning.py` |
| `Code/bin_by_real.m` / `bin_by_time.m` | `binning.py` (binning engine) |
| `Code/save2combo.m` | `combo.py` |
| `Code/glue_widthwise.m` / `glue_lengthwise.m` | `combo.py` (merge strategies) |
| `Code/save2NetCDF.m` | `combo.py` + `netcdf_schema.py` |
| `Code/Combo.json` / `CTD.json` | `netcdf_schema.py` |
| `Code/GPS_*.m` | `gps.py` |
| `Code/add_seawater_properties.m` | `seawater.py` |
| `Code/bottom_crash_profile.m` | `bottom.py` |
| `Code/trim_top_profiles.m` | `top_trim.py` |

## Verification

1. `pip install -e ".[dev]"` — both `rsi` and `perturb` install
2. `perturb init test_config.yaml` — generates valid template
3. `perturb run test_config.yaml` — processes VMP/*.p end-to-end
4. Profile NetCDFs contain calibrated FP07, aligned CT, seawater properties, GPS
5. Diss NetCDFs contain per-probe epsilon, epsilonMean, chi (Method 1)
6. CTD NetCDFs contain time-binned scalars with GPS and seawater properties
7. Binned NetCDFs contain depth/time-aggregated data with profile dimension
8. Combo NetCDFs merge all files with CF/ACDD compliance
9. `python -m pytest tests/test_perturb_*.py` — all tests pass
10. `ruff check src/perturb/` and `mypy src/perturb/` — clean
11. Compare epsilon values against `rsi-tpw eps` output for consistency
12. Run individual subcommands (`perturb profiles`, `perturb diss`, `perturb bin`, `perturb combo`) independently — stages work in isolation
