# Processing Pipeline

The processing pipeline has four stages. Each stage produces NetCDF files, and any later stage can start from any earlier stage's output:

```
.p files ──> nc ──> full-record .nc ──> prof ──> per-profile .nc ──> eps ──> epsilon .nc
                                                                  ──> chi ──> chi .nc
```

## Full Pipeline (Recommended)

Run all stages at once from raw `.p` files through epsilon and chi:

```bash
# Process all .p files, output to results/
rsi-tpw pipeline VMP/*.p -o results/

# Writes results/{pfile_stem}/profile_NNN/ directories, each containing:
#   L4_epsilon.nc, L4_chi_epsilon.nc
#   L4_chi_fit.nc (only with compute_chi_fit=True, not exposed via CLI)
#   L5_binned.nc (per-profile depth bins)
# plus results/{pfile_stem}/L6_combined.nc (all profiles combined)
```

`L4_chi_epsilon.nc` (and the binned/combined products) additionally
carry the derived mixing quantities computed on the chi window grid —
`N2`, `dTdz`, `K_T` (Osborn–Cox heat diffusivity), `Gamma` (measured
mixing coefficient, Oakey 1982), and `K_rho` (Osborn diffusivity with
Γ₀ = 0.2).  See [mixing_efficiency.md](../mixing_efficiency.md) for
definitions, masking rules, and the salinity caveat.

## Stage 1: Convert `.p` files to NetCDF

```bash
rsi-tpw nc VMP/*.p -o nc/
```

Reads Rockland `.p` binary files and writes CF-1.13 compliant NetCDF4 files with all channels converted to physical units.

## Stage 2: Extract Profiles

Detects profiling segments from pressure data and writes per-profile NetCDF files:

```bash
rsi-tpw prof VMP/ARCTERX_Thompson_2025_SN479_0005.p -o profiles/
```

Profile detection uses pressure thresholds and fall rate criteria. See [configuration.md](configuration.md) for the `profiles` section parameters.

## Stage 3: Compute Epsilon

Computes TKE dissipation rate from shear probe spectra:

```bash
# From raw .p files (profiles detected automatically)
rsi-tpw eps VMP/ARCTERX_Thompson_2025_SN479_0005.p -o epsilon/

# From per-profile .nc files
rsi-tpw eps profiles/*_prof*.nc -o epsilon/

# Parallel processing
rsi-tpw eps VMP/*.p -o epsilon/ -j 0
```

The epsilon calculation includes iterative despiking, Goodman coherent noise removal, Nasmyth spectrum fitting, and Macoun & Lueck wavenumber correction. See [epsilon_mathematics.md](../epsilon_mathematics.md) for the mathematical details.

## Stage 4: Compute Chi

Computes thermal variance dissipation rate from FP07 thermistor spectra:

```bash
# Method 1: chi from known epsilon (eps_NN/ subdirectories are searched automatically)
rsi-tpw chi VMP/*.p --epsilon-dir epsilon/ -o chi/

# Method 2: chi without epsilon (spectral fitting; defaults are the
# iterative Peterson & Fer 2014 fit with the Kraichnan spectrum model)
rsi-tpw chi VMP/*.p -o chi/

# Method 2 with the Batchelor spectrum model (Kraichnan is the default)
rsi-tpw chi VMP/*.p --spectrum-model batchelor -o chi/
```

> **Note:** `rsi-tpw eps -o epsilon/` writes into a hash-tracked subdirectory
> (`epsilon/eps_00/`). `--epsilon-dir` searches the given directory and its
> `eps_*` subdirectories (most recently modified first), matching both
> `{stem}_eps.nc` and per-profile `{stem}_prof001_eps.nc` names
> (concatenated along time). Only when no matching epsilon files exist does
> chi fall back to Method 2 for that file, with a console warning.

See [chi_mathematics.md](../chi_mathematics.md) for the mathematical details.

## Cross-probe consistency diagnostics

With two or more shear probes (or FP07s), every per-profile epsilon/chi dataset
carries an observational cross-probe consistency diagnostic (issue #131):
per-pair global attributes `probe_ratio_pairs`, `probe_ratio_median` (median
first/second-probe ratio over the windows where both are finite),
`n_ratio_windows`, and `probe_ratio_z` (significance of the median ln-ratio
given the Lueck 2022 per-window `sigma_ln`; the chi product uses a `chi_`
prefix). A two-tier `logging` warning fires on persistent disagreement:

- **statistical** — `z > 3` with at least 20 windows (catches offsets that are
  small but statistically unambiguous, e.g. a 1.2-1.3x systematic pair offset);
- **practical** — median ratio beyond **1.8x** in either direction with at
  least 10 windows (calibration-scale offsets, regardless of formal
  significance — per-window QC like `fom` can look perfect while one probe
  reads 1000x the other).

Nothing is auto-dropped — the metric exists because per-window QC cannot see a
*persistent* systematic offset. On this rsi path it is computed at the dataset
build over **all** finite windows (no fom/FM cut has been applied at that
stage).

> **Note:** these attrs live on the per-profile epsilon/chi files only. Depth
> binning and the combine stage rebuild global attributes from a schema, so the
> diagnostic does **not** survive into binned or combined products — read it
> from the per-profile files (or the log).

## Output Directory Management

The stage subcommands (`prof`, `eps`, `chi`) use a sequential, hash-tracked output naming scheme (`eps_00/`, `eps_01/`, ...) that automatically deduplicates runs with identical parameters. (The `pipeline` subcommand writes its `{pfile_stem}/profile_NNN/` tree directly into the given output directory instead.) Each hash-tracked output directory contains:
- NetCDF data files
- `config.yaml` with the resolved parameters used
- `.params_sha256_<hash>` signature file for parameter tracking

See [output_directories.md](output_directories.md) for details.
