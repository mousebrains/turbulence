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

# Writes results/eps_00/ and results/chi_00/
```

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

The epsilon calculation includes iterative despiking, Goodman coherent noise removal, Nasmyth spectrum fitting, and Macoun & Lueck wavenumber correction. See [epsilon_mathematics.md](epsilon_mathematics.md) for the mathematical details.

## Stage 4: Compute Chi

Computes thermal variance dissipation rate from FP07 thermistor spectra:

```bash
# Method 1: chi from known epsilon (uses shear probe results)
rsi-tpw chi VMP/*.p --epsilon-dir epsilon/ -o chi/

# Method 2: chi without epsilon (MLE Batchelor spectrum fitting)
rsi-tpw chi VMP/*.p -o chi/

# Method 2 with Kraichnan spectrum model
rsi-tpw chi VMP/*.p --spectrum-model kraichnan -o chi/
```

See [chi_mathematics.md](chi_mathematics.md) for the mathematical details.

## Output Directory Management

Output directories use a sequential, hash-tracked naming scheme (`eps_00/`, `eps_01/`, ...) that automatically deduplicates runs with identical parameters. Each output directory contains:
- NetCDF data files
- `config.yaml` with the resolved parameters used
- `.params_sha256_<hash>` signature file for parameter tracking

See [output_directories.md](output_directories.md) for details.
