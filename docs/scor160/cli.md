# scor160-tpw CLI Reference

The `scor160-tpw` CLI processes [ATOMIX benchmark datasets](https://doi.org/10.3389/fmars.2024.1334327) through the SCOR/ATOMIX shear-probe processing levels (L1-L4). Each subcommand runs the specified processing levels and compares computed results against the reference values embedded in the ATOMIX NetCDF files.

```
scor160-tpw <subcommand> <files...>
```

## Subcommands

| Subcommand | Levels | Description |
|------------|--------|-------------|
| `l1-l2` | L1 -> L2 | Section selection, despiking, HP filtering |
| `l2-l3` | L2 -> L3 | Wavenumber spectra (from reference L2) |
| `l1-l3` | L1 -> L2 -> L3 | Full pipeline through spectra |
| `l3-l4` | L3 -> L4 | Dissipation estimation from spectra |
| `l2-l4` | L2 -> L3 -> L4 | Spectra + dissipation |
| `l1-l4` | L1 -> L2 -> L3 -> L4 | Full pipeline |

## Usage

```bash
# Full benchmark: L1 through L4
scor160-tpw l1-l4 AtomixData/*.nc

# L2 only with visual inspection
scor160-tpw l1-l2 AtomixData/*.nc --plot

# L3-L4 only (from reference L3 spectra)
scor160-tpw l3-l4 AtomixData/*.nc
```

## Options

| Flag | Subcommands | Description |
|------|-------------|-------------|
| `--plot` | `l1-l2` | Show L1 vs L2 comparison plots |

## Processing Levels

- **L1**: Raw instrument data (time series of shear, vibration, pressure, temperature)
- **L2**: Cleaned data (section selection by pressure/speed, despiked shear, HP-filtered vibration, smoothed profiling speed)
- **L3**: Wavenumber spectra (Welch method with cosine window, Goodman coherent noise removal, Macoun-Lueck correction)
- **L4**: Dissipation estimates (epsilon via Nasmyth variance method + inertial subrange fitting, QC metrics: fom, MAD, FM, K_max_ratio)

## Input Format

Input files must be ATOMIX-format NetCDF files containing L1 data and reference values for each processing level. The 6 ATOMIX benchmark datasets are available from [Lueck et al. (2024)](https://doi.org/10.3389/fmars.2024.1334327).

## Output

Each subcommand prints a comparison report showing computed vs reference values. For L4, the report includes per-spectrum epsilon comparisons, mean absolute deviation, and figure of merit statistics.
