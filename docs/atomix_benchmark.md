# ATOMIX Benchmark Validation

The [ATOMIX](https://wiki.app.uib.no/atomix/) shear-probe benchmark datasets
(Fer et al., 2024) provide a community-standard test suite for epsilon
estimation algorithms. Five datasets from different instruments and
environments are published in a common ATOMIX NetCDF format with four
processing levels (L1--L4).

## Benchmark Datasets

| Dataset | Instrument | Environment | DOI |
|---------|-----------|-------------|-----|
| Faroe Bank Channel | VMP2000 | Deep overflow, 860 m | [10.5285/05f21d1d...](https://doi.org/10.5285/05f21d1d-bf9c-5549-e063-6c86abc0b846) |
| Haro Strait | VMP250 | Tidal channel, 80 m | [10.5285/0ec16a65...](https://doi.org/10.5285/0ec16a65-abdf-2822-e063-6c86abc06533) |
| Rockall Trough | Epsilometer | Deep boundary layer, 2200 m | [10.5285/0ebffc86...](https://doi.org/10.5285/0ebffc86-ed32-5dde-e063-6c86abc08b3a) |
| Baltic Sea | MSS90-L | Quiescent, 85 m | [10.5285/0e35f96f...](https://doi.org/10.5285/0e35f96f-57e3-540b-e063-6c86abc06660) |
| Minas Passage | MR1000 | Moored horizontal, tidal | [10.5285/0ec17274...](https://doi.org/10.5285/0ec17274-7a64-2b28-e063-6c86abc0ee02) |

The datasets span four orders of magnitude in epsilon (10^-10 to
10^-3 W/kg) and include vertical profilers, a moored horizontal
profiler, and instruments from three manufacturers.

## Comparison Method

The comparison script (`scripts/compare_atomix.py`) reads L3 cleaned
shear spectra (`SH_SPEC_CLEAN`) and wavenumber vectors (`KCYC`) from
the benchmark NetCDF files, then estimates epsilon from each spectrum
using `rsi-tpw`'s `_estimate_epsilon` algorithm (Lueck variance method
for low dissipation, inertial-subrange fitting for high dissipation).
Results are compared against the benchmark L4 epsilon values (`EPSI`).

This tests the spectral fitting / integration step (L3 to L4), not
the full pipeline from raw data.

## Running the Comparison

```bash
# Download benchmark .nc files from BODC (see DOIs above) into benchmarks/atomix/
python scripts/compare_atomix.py --data-dir /path/to/atomix/data --output-dir benchmarks/atomix

# Or open browser tabs for each BODC download page:
python scripts/compare_atomix.py --download

# Process a single dataset:
python scripts/compare_atomix.py --data-dir /path/to/data --datasets TidalChannel
```

## Pass/Fail Criteria

| Metric | Threshold | Description |
|--------|-----------|-------------|
| log10 RMSD | < 0.5 | Root-mean-square deviation in log10(epsilon) |
| Correlation | > 0.8 | Pearson r of log10(epsilon) |
| Within 1 decade | > 90% | Fraction of estimates within 1 order of magnitude |

## Results

All six benchmark files (5 datasets + Haro Strait constant-speed
variant) pass with wide margin:

| Dataset | N spectra | log10 bias | log10 RMSD | Correlation | Within 0.5 dec | Result |
|---------|-----------|------------|------------|-------------|----------------|--------|
| Faroe Bank Channel | 684 | +0.008 | 0.012 | 1.000 | 100% | PASS |
| Haro Strait | 64 | +0.007 | 0.062 | 0.994 | 100% | PASS |
| Haro Strait (const speed) | 64 | +0.003 | 0.059 | 0.995 | 100% | PASS |
| Rockall Trough | 362 | -0.082 | 0.162 | 0.983 | 98.9% | PASS |
| Baltic Sea | 122 | +0.044 | 0.192 | 0.981 | 97.5% | PASS |
| Minas Passage | 120 | +0.055 | 0.076 | 0.985 | 100% | PASS |

Worst-case RMSD is 0.19 decades (Baltic Sea), corresponding to
agreement within a factor of ~1.5. 100% of 1,416 spectra compared
fall within 1 order of magnitude, and >97% fall within half an order
of magnitude.

## Output

The script produces:

- `ATOMIX_COMPARISON_REPORT.md` -- detailed Markdown report with
  per-dataset tables
- `atomix_epsilon_scatter.png` -- scatter plots of rsi-tpw vs ATOMIX
  epsilon
- `atomix_epsilon_profiles.png` -- epsilon vs pressure profiles
- `atomix_kmax_comparison.png` -- integration limit (K_max) comparison
- `atomix_fom_comparison.png` -- figure-of-merit comparison

## References

- Fer, I., Dengler, M., Holtermann, P., Le Boyer, A. & Lueck, R.
  (2024). ATOMIX benchmark datasets for dissipation rate measurements
  using shear probes. *Scientific Data*, 11, 518.
  [doi:10.1038/s41597-024-03323-y](https://doi.org/10.1038/s41597-024-03323-y)
- Lueck, R. et al. (2024). Best practices recommendations for
  estimating dissipation rates from shear probes. *Front. Mar. Sci.*,
  11, 1334327.
  [doi:10.3389/fmars.2024.1334327](https://doi.org/10.3389/fmars.2024.1334327)
- [GitHub: SCOR-ATOMIX/ShearProbes_BenchmarkDescriptor](https://github.com/SCOR-ATOMIX/ShearProbes_BenchmarkDescriptor)

See also [Bibliography](bibliography.md#benchmark-datasets) for
individual dataset citations.
