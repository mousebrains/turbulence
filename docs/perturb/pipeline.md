# Perturb Processing Pipeline

The `perturb` pipeline is a batch-processing system for Rockland VMP/MicroRider data. It takes raw `.p` files through a multi-stage chain, producing depth-binned and time-binned NetCDF outputs.

## Pipeline Stages

```
.p files ──> trim ──> merge ──> process ──> bin ──> combo
                                  │
                     ┌────────────┼────────────┐
                     ▼            ▼            ▼
                   CTD        profiles       diss
                (time-bin)   (per-profile)  (epsilon)
                     │            │            │
                     │       FP07 cal     epsilon_mean
                     │       CT align          │
                     │            │          chi (opt)
                     │            ▼            ▼
                     │     profiles_binned  diss_binned
                     │            │         chi_binned
                     ▼            ▼            ▼
                 ctd_combo      combo      diss_combo
```

### Stage 1: Trim

Removes corrupt final records from `.p` files (partial writes from power loss or buffer flush). Output goes to `trimmed/`.

### Stage 2: Merge

Merges split `.p` files that were recorded as sequential segments of the same deployment. Detects mergeable files by matching config strings and record sizes, then concatenates data records.

### Stage 3: Process (per-file)

Each `.p` file is processed through several sub-stages:

1. **Hotel data** (optional) — If a hotel file is configured, external telemetry channels (speed, pitch, roll, heading, CTD from gliders/AUVs) are loaded and interpolated onto the instrument's fast or slow time axes. Channels listed in `fast_channels` are interpolated onto `t_fast`; all others go to `t_slow`. The interpolated data is injected into `pf.channels` before any downstream processing, so hotel-provided channels (e.g., `speed`, `P`) are available to profile detection, dissipation, and chi stages.

2. **CTD time-binning** — Bins slow channels (T, C, P) by time, interpolates GPS, computes seawater properties (SP, SA, CT, sigma0, rho, depth via TEOS-10).

3. **Profile extraction** — Detects down-cast (or up-cast) segments from pressure and fall rate, writes per-profile NetCDF files.

4. **FP07 calibration** — In-situ calibration of FP07 thermistors against a reference sensor (JAC_T) using Steinhart-Hart coefficients and cross-correlation lag estimation.

5. **CT alignment** — Cross-correlation alignment of conductivity and temperature sensors to correct for spatial separation.

6. **Dissipation (epsilon)** — Computes TKE dissipation rate per profile using `rsi.dissipation.get_diss`. Combines multi-probe estimates via `mk_epsilon_mean` (geometric mean with 95% CI filtering).

7. **Chi** (optional) — Computes thermal variance dissipation rate per profile using `rsi.chi.get_chi` with the epsilon dataset (Method 1).

### Stage 4: Bin

Depth-bins (or time-bins) the per-profile and per-diss NetCDF files into 2D arrays (bin x profile). Produces `profiles_binned_NN/`, `diss_binned_NN/`, and `chi_binned_NN/` directories.

### Stage 5: Combo

Assembles binned NetCDF files into combined datasets with CF-1.8/ACDD-1.3 compliant global attributes, geospatial extent, and standardized variable metadata.

## Full Pipeline

```bash
# Process all .p files in VMP/
perturb run -o results/ VMP/*.p

# Explicit file list with 4 parallel workers
perturb run -o results/ -j 4 VMP/*002*.p

# With a configuration file
perturb run -c config.yaml -o results/
```

## Individual Stages

Each stage can be run independently:

```bash
perturb trim -o results/ VMP/*.p
perturb merge -o results/
perturb profiles -o results/ -j 4 VMP/*.p
perturb diss -o results/ -j 4 VMP/*.p
perturb chi -o results/ -j 4 VMP/*.p
perturb ctd -o results/ -j 4 VMP/*.p
perturb bin -c config.yaml -o results/
perturb combo -c config.yaml -o results/
```

## Output Directory Structure

```
results/
  trimmed/                    .p files with corrupt records removed
  profiles_00/                per-profile NetCDFs
    SN479_0002_prof001.nc
    SN479_0002_prof002.nc
    ...
  diss_00/                    per-profile dissipation estimates
    SN479_0002_prof001.nc
    ...
  chi_00/                     per-profile chi estimates (if enabled)
  ctd_00/                     per-file time-binned CTD
    SN479_0002.nc
    ...
  profiles_binned_00/         depth-binned profiles
    binned.nc
  diss_binned_00/             depth-binned dissipation
    binned.nc
  chi_binned_00/              depth-binned chi (if enabled)
    binned.nc
```

Output directories use a sequential, hash-tracked naming scheme (`_00`, `_01`, ...) that automatically deduplicates runs with identical parameters. See [Output Directories](../rsi-tpw/output_directories.md) for the hashing mechanism.

## Parallel Processing

The pipeline supports parallel file processing via the `-j` flag:

```bash
perturb run -j 4 -o results/ VMP/*.p   # 4 workers
perturb run -j 0 -o results/ VMP/*.p   # auto (all cores)
perturb run -j 1 -o results/ VMP/*.p   # serial (default)
```

Parallelism applies to the per-file processing stage (Stage 3). Trimming, merging, binning, and combo assembly run serially.

See [Parallel Scaling](parallel.md) for benchmark results.
