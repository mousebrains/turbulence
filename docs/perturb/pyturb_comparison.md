# pyturb Comparison

Comparison of [oceancascades/pyturb](https://github.com/oceancascades/pyturb) (Jesse Cusack, Oregon State) with our microstructure-tpw/perturb packages. Both derive from the Rockland ODAS MATLAB library.

## Core Epsilon: Identical

Both packages use the same Nasmyth spectrum (Lueck improved fit), ISR integration, fitting iterations, and variance correction. Same constants throughout (`a = 1.0774e9`, `x_isr = 0.02`, `x_95 = 0.1205`, K_max clamp [7, 150] cpm). Same 3-iteration log-space fitting with flyer removal.

## Where microstructure-tpw Is More Complete

| Feature | microstructure-tpw | pyturb |
|---------|------------|--------|
| Goodman coherent noise removal | Full cross-spectral matrix | Not implemented |
| Chi pipeline | Method 1 (known epsilon) + Method 2 (MLE) | Bare `kraichnan_spectrum()` only |
| Batchelor spectrum | Full implementation | Not implemented |
| FP07 transfer function | Single/double-pole, tau estimation, noise floor | Not implemented |
| QC metrics | fom, K_max_ratio, FM statistic, DOF | k_max only |
| Spectral detrending | Linear detrend per segment (matches ODAS) | None (potential low-freq bias) |
| Viscosity | TEOS-10 (T, S, P dependent) | Millero 1974 (atmospheric P only) |

## pyturb Ideas Worth Harvesting

### Already in our perturb package
- Parallel batch processing (`-j` flag)
- Depth/time binning
- File merging (trim + merge stages)
- Profile detection from pressure

### Could adopt
1. **Speed from pressure** — Estimate fall speed from dP/dt with pitch correction. Useful for MicroRiders on gliders where no speed sensor exists.
2. **Multi-profile detection** — `scipy.signal.find_peaks` on pressure for glider data with many dive cycles per file. Our profile detection works for VMP (single cast per file) but not continuous glider records.
3. **Auxiliary data merging** — Interpolate external time series (glider flight data) onto microstructure time axis.
4. **High-pass filter option** — 1st-order Butterworth HP at 0.5/fft_len Hz before spectral analysis (ODAS recommendation). Our linear detrend may already handle this.

### Not adopting (engineering choices)
- Typer CLI (we use argparse, consistent with rsi-tpw)
- xarray throughout (we use netCDF4-direct for performance)
- uv build system (we use pip/setuptools)

## Spectral Estimation Differences

Both use Welch's method with 50% overlap. pyturb uses scipy Hann window; we use a cosine window with RMS=1 normalization (mathematically identical, matches ODAS output scaling). Critical difference: we detrend each FFT segment, pyturb does not.

## Despiking

Same algorithm: iterative HP/LP envelope threshold detection. Minor differences: pyturb defaults to 6 max passes (ours: 10), uses explicit mirror padding.

## Bottom Line

The core mathematics are the same since both derive from ODAS. microstructure-tpw is more complete (Goodman, chi, FP07, QC, detrending). pyturb has better glider/MicroRider workflow support. The main harvestable ideas are glider-related features (speed from pressure, multi-profile, aux data merge) which would extend our instrument coverage beyond VMP.

## References

- [oceancascades/pyturb](https://github.com/oceancascades/pyturb) — Python package
- [jessecusack/perturb](https://github.com/jessecusack/perturb) — Jesse's MATLAB ODAS wrapper
