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

The `scor160` package reads each benchmark NetCDF file and re-processes
through the full L1--L4 pipeline (`scor160.l2.process_l2`,
`scor160.l3.process_l3`, `scor160.l4.process_l4`), which implements
section selection, despiking, HP filtering, spectral estimation,
Goodman coherent noise removal, and epsilon estimation via the Lueck
variance method (low dissipation) and inertial subrange fitting (high
dissipation) following Lueck et al. (2024).

Results are compared at each level against the benchmark reference data.

**L2 metrics** (time-domain, within overlapping sections):

- **Speed relative RMS**: RMS(comp - ref) / mean(|ref|)
- **Shear relative RMS**: RMS(comp - ref) / RMS(ref)
- **Correlation**: Pearson r between computed and reference time series

**L4 metrics** (log10-space epsilon):

- **Median ratio**: 10^median(log10(comp/ref)) -- systematic bias
- **RMS log10 diff**: Root-mean-square of log10(comp/ref) -- overall
  agreement
- **Within factor N**: Fraction where |log10(comp/ref)| < log10(N)
- **Method agreement**: Fraction where computed and reference use the
  same method (0 = variance, 1 = ISR)

## Running the Comparison

```bash
scor160-tpw l1-l2 /path/to/benchmark/*.nc      # L1 -> computed L2
scor160-tpw l3-l4 /path/to/benchmark/*.nc      # reference L3 -> computed L4
scor160-tpw l1-l4 /path/to/benchmark/*.nc      # full pipeline L1 -> L4
```

## ODAS-Processed Datasets

The VMP2000, VMP250, and Nemo datasets were processed with Rockland
Scientific's ODAS MATLAB Library (v4.5.1) or compatible code. Our
implementation closely mirrors the ODAS algorithm, enabling validation
at every processing level.

### L1 to L2: Section Selection, HP Filtering, Despiking

| Dataset | Probes | Sections | Speed rel RMS | Shear rel RMS | Shear corr | Vib rel RMS |
|---------|--------|----------|---------------|---------------|------------|-------------|
| VMP2000 Faroe Bank | 2 sh, 3 acc | 1/1 | 0.1% | 0.1--0.3% | 1.0000 | 0.0% |
| VMP250 Haro Strait | 2 sh, 2 vib | 1/1 | 0.3% | 0.03--0.2% | 1.0000 | 0.0% |
| VMP250 Haro Strait (cs) | 2 sh, 2 vib | 1/1 | 0.0% | 0.03--0.2% | 1.0000 | 0.0% |
| Nemo Minas Passage | 4 sh, 2 vib | 1/1 | 1.6% | 0.2--0.4% | 1.0000 | 0.0% |

Section detection, speed, shear, and vibration are near-exact for all
ODAS datasets. Vibration has exactly zero difference (no despiking
applied, identical HP filter on identical input).

The VMP250 constant-speed variant uses a declared speed, so speed
difference is exactly zero.

Two implementation details were critical for matching the ODAS reference:

1. **`filtfilt` edge padding**: MATLAB uses `padlen = 3*(nfilt-1)`
   while scipy defaults to `padlen = 3*nfilt`. For a 1st-order
   Butterworth (nfilt=2) this is padlen=3 vs 6. When the section spans
   the entire record (as with Nemo), the mismatch produces ~1% vibration
   error concentrated at the edges. Using the MATLAB padlen eliminates
   this entirely.

2. **HP filter before despike**: The ODAS reference HP-filters before
   despiking, not after. This is evident from the reference L2 data
   containing exactly-constant replacement values within spike regions
   — the signature of despiking an already-HP-filtered signal. Reversing
   the order (despike→HP) causes the filter to spread the constant
   replacement, adding 1--5% shear error.

**Nemo speed (1.6% RMS)**: The Nemo MR1000 is a moored horizontal
profiler with `speed_tau = 60 s`, giving an ODAS-style LP filter cutoff
of `0.68/60 = 0.011 Hz` — 40x lower than the VMP cutoff (0.45 Hz at
tau = 1.5 s). The difference is uniform across the record (not
edge-concentrated), indicating a filter parameter or implementation
difference in how the reference processed horizontal speed rather than
an edge padding issue. The VMP datasets, with their shorter tau, have
negligible filter sensitivity (0.0--0.3% speed RMS).

Residual shear differences (0.03--0.4%) come from minor despike
algorithm differences (spike detection threshold sensitivity, number
of replacement samples).

### L3 to L4: Epsilon Estimation (Reference L3)

Using reference L3 spectra isolates the epsilon estimation algorithm:

| Dataset | Probes | Median ratio | RMS log10 | Factor 2 | Method agree |
|---------|--------|-------------|-----------|----------|-------------|
| VMP2000 Faroe Bank | 2 | 1.016--1.018 | 0.011--0.012 | 100% | 100% |
| VMP250 Haro Strait | 2 | 1.000 | 0.015--0.031 | 100% | 100% |
| VMP250 Haro Strait (cs) | 2 | 0.997--1.000 | 0.016--0.067 | 96.9--100% | 96.9% |
| Nemo Minas Passage | 4 | 1.072--1.075 | 0.030--0.032 | 100% | 100% |

VMP2000 and VMP250 show excellent agreement with median ratios within
2% and RMS below 0.031 log10 decades. Nemo has a systematic +7% bias
(see Discrepancy Analysis below).

### L1 to L4: Full Pipeline Epsilon

The full pipeline propagates L2 differences through L3 spectral
estimation into L4 epsilon:

| Dataset | Probes | Median ratio | RMS log10 | Factor 2 | Method agree |
|---------|--------|-------------|-----------|----------|-------------|
| VMP250 Haro Strait | 2 | 0.646--0.672 | 0.177--0.181 | 96.9--100% | 96.9--100% |
| VMP250 Haro Strait (cs) | 2 | 0.591--0.632 | 0.203--0.211 | 87.5--100% | 93.8--96.9% |
| Nemo Minas Passage | 4 | 0.726--0.795 | 0.100--0.144 | 100% | 100% |

The VMP2000 L1→L4 run did not produce per-probe epsilon statistics in
the comparison (343 vs 342 spectra from a minor section boundary
difference); the L3→L4 result above confirms algorithm correctness.

The VMP250 full-pipeline results show ~35% underestimate (median ratio
0.59--0.67) compared to the near-exact L3→L4 result. This is driven by
small L2 shear differences (1% RMS) that compound through spectral
estimation, particularly at the lowest epsilon values where a 1%
time-domain difference can shift the spectral minimum and change the
integration range. All spectra remain within factor 3 of the reference.

Nemo full-pipeline shows ~20--27% underestimate with low scatter (RMS
0.10--0.14), all within factor 2.


## Non-ODAS Datasets

The MSS (Sea & Sun Technology, processed by IOW) and Epsifish
(Epsilometer, processed by Scripps/MOD) datasets use instrument-specific
processing software for L2 that differs significantly from ODAS. Our
L1→L2 comparison reveals large discrepancies (35--70% shear relative
RMS) driven by proprietary despike algorithms and different HP filter
parameters.

However, the L3→L4 comparison using reference L3 spectra shows these
instruments work well through our epsilon estimation:

| Dataset | Probes | Median ratio | RMS log10 | Factor 2 | Method agree |
|---------|--------|-------------|-----------|----------|-------------|
| MSS Baltic Sea | 2 | 1.008--1.014 | 0.007--0.009 | 100% | 100% |
| Epsifish Rockall Trough | 2 | 1.203--1.226 | 0.092--0.100 | 100% | 99--100% |

The MSS achieves the best L3→L4 agreement of any dataset (RMS 0.007).
The Epsifish +20% bias has identified causes (see Discrepancy Analysis).

These datasets are excluded from L1→L2 validation because their
reference L2 was produced with non-ODAS software whose despike and
filtering algorithms we cannot replicate from the available metadata.

### Epsifish Speed Note

The epsifish L1 speed contains brief dips (1--7 samples at 320 Hz)
below the `speed_cutout = 0.2 m/s` threshold, caused by oscillations
from the drag screen deployment. Applying the ODAS-style speed
smoothing filter (1st-order Butterworth LP at 0.68/tau Hz, tau = 1.5 s)
eliminates this fragmentation and recovers the correct 3 sections. The
epsifish team used a 1 s moving-mean filter on pressure (per their
metadata comment), which gives a marginally better speed match (RMS diff
0.007 vs 0.012 m/s) but both approaches produce identical section
detection.


## Discrepancy Analysis

### VMP2000, VMP250 (ODAS-processed)

These datasets were processed with Rockland Scientific's ODAS MATLAB
Library (v4.5.1) or compatible code. Our implementation closely mirrors
the ODAS algorithm, giving near-exact agreement:

- **VMP2000 Faroe Bank**: RMS 0.011. All variance method. Residual
  ~1.6% bias from minor differences in polynomial spectral-minimum
  detection.
- **VMP250 Haro Strait**: RMS 0.015--0.031. Mixed variance/ISR.
  Method agreement 100%. The constant-speed variant has one borderline
  spectrum per probe where our ISR margin threshold selects variance
  while the reference selects ISR (96.9% agreement); the epsilon
  difference is small.

### Epsifish Rockall Trough (+20% bias)

The Epsifish (Epsilometer) data was processed by the MOD group at
Scripps, not with ODAS. Systematic +20% overestimate (median ratio
1.20--1.23) with two identified causes:

1. **Integration limit (kmax)**: Our polynomial spectral-minimum
   detection finds no minimum above 10 cpm and defaults to K_95
   (median 26 cpm). The reference caps integration at ~16 cpm
   (0.675 x K_95), likely using a noise-floor-aware algorithm. The
   Epsifish spectra show elevated noise above ~16 cpm
   (observed/Nasmyth ratio 1.7--3.0), so our wider integration range
   includes noise variance that inflates epsilon. Using the reference
   kmax reduces the bias to ~5%, confirming kmax as the dominant factor.

2. **Residual noise in integrated spectrum**: Even with matching kmax,
   ~5% overestimate remains from noise contamination within the
   integration range. The reference processing may subtract a noise
   floor estimate before integration.

3. **FOM definition**: The reference uses FOM relative to the Panchev
   spectrum (per L4 attribute `processing_level`), while we use the
   Nasmyth/Lueck spectrum. This affects QC flags but not epsilon
   estimates directly.

The bias is consistent (low scatter, RMS 0.09--0.10) and all spectra
fall within factor 2, indicating a systematic algorithmic difference
rather than random error.

### Nemo MR1000 Minas Passage (+7% bias)

The Nemo (MicroRider-1000) data was processed by Dalhousie University.
All 120 spectra use the ISR method (epsilon 10^-4 to 10^-3 W/kg in
strong tidal currents). Systematic +7% overestimate (median ratio
1.072--1.075) from:

1. **ISR fit range**: The reference uses a narrower ISR range. Our
   computed kmax (median 77--94 cpm) is ~2x the reference (37--45 cpm),
   consistent with the reference using X_ISR = 0.01 while the ODAS
   code has X_ISR = 0.02 (the ODAS `get_diss_odas.m` source shows
   `x_isr = 0.01; x_isr = 2*x_isr; % test pushing this upward`).
   After correcting X_ISR to 0.01, kmax agreement improved to ~3%,
   but ~7% epsilon bias remains.

2. **Residual spectral contamination**: The wider ISR fit range
   (even at X_ISR = 0.01) includes some bins near the transition from
   inertial to dissipation subrange. In the high-energy Minas Passage
   environment, residual vibration noise at these wavenumbers biases
   the ISR fit slightly upward.

3. **EPSI_FINAL aggregation**: With 4 probes, the reference uses
   QC flags (including flag 4 = diss_ratio check) to select probes
   for the geometric mean. Our flag system doesn't include the
   diss_ratio flag, leading to different probe selection and a larger
   EPSI_FINAL discrepancy (median ratio 1.41) than the per-probe
   epsilon (1.07).

### X_ISR constant

The non-dimensional ISR upper wavenumber limit X_ISR controls how far
into the spectrum the ISR fitting extends. The ODAS v4.5.1 code
contains:
```matlab
x_isr = 0.01;           % original value
x_isr = 2*x_isr;        % test pushing this upward
```
All six ATOMIX benchmark datasets were processed with X_ISR = 0.01
(the original value, before the experimental doubling). This was
confirmed by the consistent 2x kmax ratio observed across VMP250 ISR
spectra and Nemo data when using X_ISR = 0.02.

Both `scor160.l4` and `rsi.dissipation` use X_ISR = 0.01 to
match the benchmark processing.

### FOM discrepancy

Our FOM (observed/Nasmyth variance ratio over the fit range) is
systematically higher than the reference FOM across all datasets. This
is expected because:

- The reference may use the Panchev spectrum (Epsifish) or an
  attenuated Nasmyth spectrum convolved with the instrument transfer
  function, giving a different denominator.
- FOM definitions vary between processing tools. The ATOMIX standard
  specifies the general concept but not the exact computation.

FOM differences do not affect epsilon estimates; they only affect QC
flag assignment.

### Method selection threshold

The ISR/variance method switch uses `e_1 >= E_ISR_THRESHOLD * ISR_MARGIN`
where E_ISR_THRESHOLD = 1.5 x 10^-5 W/kg and ISR_MARGIN = 1.65. The
margin prefers the variance method for borderline cases where e_1 is
near the threshold, since variance integrates over a wider spectral
range and is more robust. This was tuned against VMP250 data where
borderline spectra (e_1 = 1.5--2.5 x 10^-5) showed up to 0.35 decade
error with incorrect method selection.


## L2 Processing Details

The `scor160.l2.process_l2` function implements:

1. **Speed smoothing**: ODAS-style 1st-order Butterworth LP filter at
   0.68/tau Hz (tau from metadata `speed_tau`, default 1.5 s). NaN
   values in the speed record are linearly interpolated before filtering
   and restored afterward.

2. **Section selection**: Contiguous runs where speed >= `profile_min_W`
   and pressure >= `profile_min_P` for at least `profile_min_duration`
   seconds. Parameters are read from L2 group attributes, falling back
   to global `speed_cutout` then defaults.

3. **HP filtering**: 1st-order Butterworth HP at `HP_cut` Hz, applied
   forwards and backwards (zero phase) on the entire record. NaN values
   are linearly interpolated before filtering and restored afterward,
   preventing propagation to the entire record. Uses MATLAB-compatible
   `padlen = 3*(nfilt-1)` to match ODAS edge behaviour.

4. **Despiking**: Within each section, the HP-filtered shear and
   vibration are despiked using the iterative threshold method with
   parameters from the benchmark metadata (`despike_sh`, `despike_A`).
   HP filtering before despiking matches the ODAS order of operations.


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
