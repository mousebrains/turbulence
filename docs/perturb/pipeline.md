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

6. **Dissipation (epsilon)** — Computes TKE dissipation rate per profile via `rsi.dissipation._compute_epsilon`, which drives the scor160 epsilon estimator (`scor160.l4._estimate_epsilon`). Combines multi-probe estimates via `mk_epsilon_mean` (geometric mean with 95% CI filtering).

7. **Chi** (optional) — Computes thermal variance dissipation rate per profile via `rsi.chi_io._compute_chi`. By default (`chi.use_epsilon: true`) the combined epsilon seeds the calculation (Method 1); set `chi.use_epsilon: false` for Method 2 spectral fitting.

### Stage 4: Bin

Depth-bins (or time-bins) the per-profile and per-diss NetCDF files into 2D arrays (bin x profile). Produces `profiles_binned_NN/`, `diss_binned_NN/`, and `chi_binned_NN/` directories.

### Stage 5: Combo

Assembles binned NetCDF files into combined datasets with CF-1.13/ACDD-1.3 compliant global attributes, geospatial extent, and standardized variable metadata.

`perturb run` covers the full chain trim → merge → process → bin → combo. Combo assembly can also be re-run on its own via `perturb combo`.

## Full Pipeline

```bash
# Process all .p files in VMP/ (trim → merge → process → bin → combo)
perturb run -o results/ VMP/*.p

# Explicit file list with 4 parallel workers
perturb run -o results/ -j 4 VMP/*002*.p

# With a configuration file
perturb run -c config.yaml -o results/

# Re-run combo assembly on its own
perturb combo -c config.yaml -o results/
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

## Incremental re-runs (caching)

Re-running `perturb run` with the same config skips work whose inputs are
unchanged — each `.p` file is reprocessed only when something that could change
its outputs has changed:

- **Trim** (Stage 1): an unchanged source's trim decision is reused from a
  single `stat` (no per-file header read), keyed on the same `size + 2 s-mtime`
  fingerprint, and trusted only while its physical output is still present.
  Only a clean run is cached.
- **Per-file processing** (Stage 3): a file is skipped when an up-to-date cache
  marker (under `<output_root>/.cache/`) matches and its output NetCDFs still
  exist. The marker key folds the input file's identity (size + mtime), the
  hotel/GPS files referenced by path, and the **signature hash** of each output
  dir the file targets. A run that hit a caught failure is not cached (it retries
  next run).
- **Bin/combo** (Stages 4–5): re-bin/re-combine is skipped when the contributing
  per-file outputs are unchanged. This is keyed on the per-file cache keys (an
  `_input_manifest` attribute on `binned.nc`/`combo.nc`), **not** on file mtimes
  — so it is correct on filesystems with coarse mtime granularity (e.g. exFAT
  external drives at 2 s).

**Built for slow/network mounts.** A no-op re-run touches each input once (a
`stat`, not an `open`) and lists each output directory once (membership test, not
a `stat` per NetCDF), so the per-run overhead is a handful of round-trips rather
than thousands — important over an SMB mount where each round-trip is costly. On
a 135-file campaign on an external drive, the trim re-check alone drops from
~18 s to ~1.5 s on the second run.

**Code-aware.** The output-dir hash includes a fingerprint of the processing
code and key numeric dependencies (numpy/scipy/gsw/netCDF4/xarray). So editing
the epsilon/chi/mixing code — or upgrading one of those packages — produces new
`{stage}_NN` dirs and recomputes; it never silently reuses stale results.
(Editing plotting/standalone code does not invalidate the cache.) Plotting still
resolves a dir by config alone, independent of the code version that wrote it.

To pin or override the fingerprint (e.g. to reuse the cache across a code change
you know can't affect numerics, or for reproducible runs), set
`$ODAS_TPW_ENGINE_FINGERPRINT`.

```bash
perturb run --config perturb.yaml          # incremental: skips up-to-date files
perturb run --config perturb.yaml --force  # ignore all caches: re-trim, reprocess,
                                            # re-bin, re-combine everything
```

See [Parallel Scaling](parallel.md) for benchmark results.
