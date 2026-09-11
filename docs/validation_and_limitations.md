# Validation and Known Limitations

What this package's outputs have been checked against, what that check does
and does not cover, and what a paper using them should say. Written for
someone deciding whether to trust a number, or reviewing a manuscript that
rests on one.

Short version: **epsilon is validated against the ATOMIX community
benchmark and can be reported as an absolute quantity. Chi should be
reported as a relative field with a stated absolute systematic of order
1.5x** -- the absolute thermal-variance level is not anchored.


## Epsilon (TKE dissipation rate)

### What was checked

The [ATOMIX shear-probe benchmark](https://doi.org/10.1038/s41597-024-03323-y)
(Fer et al. 2024) publishes six dissipation datasets spanning four orders of
magnitude in epsilon, three instrument manufacturers, and both vertical and
moored-horizontal profilers. This package re-processes the benchmark's own
cleaned L3 spectra through its epsilon estimator and compares against the
published L4 dissipation.

Re-run at the v0.4.0 release commit, 1416 spectra pooled across probes:

| Dataset | Spectra | log10 bias | log10 RMSD | r | Within 0.5 decade |
|---------|---------|-----------|-----------|-----|------------------|
| Faroe Bank Channel (VMP2000) | 684 | +0.009 | 0.012 | 1.000 | 100% |
| Haro Strait (VMP250) | 64 | +0.009 | 0.025 | 0.999 | 100% |
| Haro Strait, constant speed | 64 | +0.007 | 0.023 | 0.999 | 100% |
| Rockall Trough (Epsilometer) | 362 | +0.089 | 0.096 | 0.999 | 100% |
| Baltic Sea (MSS90-L) | 122 | +0.006 | 0.008 | 1.000 | 100% |
| Minas Passage (MR1000) | 120 | +0.031 | 0.031 | 1.000 | 100% |

Full per-level detail, including the L1 -> L2 and L1 -> L4 comparisons that
cover the rest of the chain, is in [atomix_benchmark.md](atomix_benchmark.md).

### What that does not establish

- **It is agreement with a reference, not with truth.** The ATOMIX L4 is a
  community consensus product. A bias this package shares with the reference
  implementation would not appear anywhere in the table above.
- **It validates the algorithm, not your calibration.** Every entry uses the
  benchmark's declared shear-probe sensitivities. On your own data the
  dominant epsilon uncertainty is almost always the calibration constants,
  not the estimator -- `sens` enters as `sens^-2`, and `diff_gain` likewise.
  A single transcribed digit is a factor of 100 (see
  [issue #178](https://github.com/mousebrains/turbulence/issues/178), where a
  stale `diff_gain` would inflate epsilon ~110x). Audit them with
  `rsi-tpw sensors --cal-dir` and `--diff-gain` before believing a level.
- **The two VMP250 entries rest on 64 spectra each.** Their third digit is
  not meaningful.
- **Speed is an input, not a validated output.** Epsilon scales steeply with
  the profiling speed used for the frequency-to-wavenumber conversion. On a
  VMP falling freely that speed is well determined from pressure; on a glider
  or MicroRider it comes from a flight model or an EM sensor and carries its
  own systematic (see `docs/glider_dive_geometry_forecast.md`).


## Chi (thermal variance dissipation rate)

### The absolute level is not anchored

There is no chi equivalent of the ATOMIX benchmark, and three separate
attempts to anchor the absolute level from within the data failed
([issue #179](https://github.com/mousebrains/turbulence/issues/179)).

The specific trap: fitting a per-bead FP07 time constant reconciles the two
thermistors on an instrument beautifully -- on ARCTERX-2022 the SN 194 pair
went from 1.77x to 0.98x and SN 428 from 0.73x to 1.06x, with the cross-probe
warning count going from 11 to 0. **That is a relative fix, and it makes the
uncertainty look far smaller than it is.** Both beads share the same fitted
tau and the same `diff_gain`, so the pair spread is now tight *because it was
tuned*, and the pair spread is no longer a valid estimator of chi's systematic
error. Reading +-25% off it would be wrong.

Report an absolute systematic of order **1.5x**.

### `fom` near 1.0 does not mean the spectral fit is good

Chi's figure of merit is the ratio of observed to model variance *integrated*
over the fit range, so a mid-band deficit cancels against excesses at the
band edges. Measured on CASPER-West (VMP-250IR SN 194, Kraichnan model,
2504 windows with `K_max_ratio > 1.2`), the observed-to-model ratio is
structured by roughly a factor of two across the band -- and structured in
the part of the band where the noise model contributes essentially nothing,
so it is not a noise-subtraction artifact. **`fom` for those exact windows is
1.001** (p10 0.987, p90 1.002).
[Issue #191](https://github.com/mousebrains/turbulence/issues/191) has the
per-band numbers. Do not use `fom` alone as evidence that a chi spectrum was
well fit.

### What chi is good for

Relative structure -- vertical profiles, spatial patterns, time series,
ratios between water masses within one deployment -- where the systematic
enters as a common factor and cancels. Cross-deployment absolute comparisons
are the case to avoid, because the FP07 time-constant treatment differs
between them.

Related: the mixing efficiency Gamma derived from chi and epsilon spans 5.5x
across seven campaigns, which is itself evidence that something in the
absolute chain is not yet pinned down. Gamma cannot be used to anchor
absolute chi, and the reverse inference is equally unsafe.


## Suggested wording for a paper

Adapt as needed; the substance is the three claims -- what produced the
numbers, what epsilon was validated against, and what chi's systematic is.

> Dissipation rates were computed with `microstructure-tpw` v0.4.0
> (Welch, 2026; DOI 10.5281/zenodo.22699141). The epsilon estimator was
> validated against the ATOMIX shear-probe benchmark datasets (Fer et al.,
> 2024): over 1416 spectra from six benchmark records spanning 10^-10 to
> 10^-3 W kg^-1, agreement with the published L4 dissipation has a log10
> bias below 0.09 and a log10 RMSD of 0.008-0.096, with 100% of estimates
> within half a decade. Thermal variance dissipation chi is reported as a
> relative field; the absolute level carries an estimated systematic
> uncertainty of order 1.5x arising from the FP07 time-constant correction,
> which is not independently anchored.

Cite the **version** DOI, not the concept DOI, when the numbers matter --
the concept DOI resolves to whatever the latest release happens to be, which
is not what produced your figures. Both are listed in
[`CITATION.cff`](../CITATION.cff).


## Reproducing the validation

```bash
pip install -e ".[dev]"
python -m pytest                # 3609 tests; includes the CI ATOMIX gate

# Full benchmark comparison (needs the ~375 MB ATOMIX corpus):
python scripts/compare_atomix.py --download        # opens the DOI landing pages
python scripts/compare_atomix.py --data-dir AtomixData --output-dir AtomixData
```

The benchmark NetCDFs are not redistributed here; each is separately
published with its own DOI, listed in [atomix_benchmark.md](atomix_benchmark.md).
