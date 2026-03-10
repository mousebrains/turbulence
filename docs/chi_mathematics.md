# Mathematics of Chi Estimation

This document describes the mathematical foundations for computing chi, the rate of dissipation of thermal variance, as implemented in `rsi-python`. All equation numbers, constants, and algorithmic details correspond to the actual code in [`batchelor.py`](../src/rsi_python/batchelor.py), [`fp07.py`](../src/rsi_python/fp07.py), and [`chi.py`](../src/rsi_python/chi.py).

## Contents

1. [Definition of Chi](#1-definition-of-chi)
2. [Batchelor Temperature Gradient Spectrum](#2-batchelor-temperature-gradient-spectrum)
3. [Kraichnan Temperature Gradient Spectrum](#3-kraichnan-temperature-gradient-spectrum)
4. [FP07 Thermistor Transfer Function](#4-fp07-thermistor-transfer-function)
5. [Electronics Noise Model](#5-electronics-noise-model)
6. [Method 1: Chi from Known Epsilon](#6-method-1-chi-from-known-epsilon)
7. [Method 2a: Maximum Likelihood Estimation](#7-method-2a-maximum-likelihood-estimation)
8. [Method 2b: Iterative Integration](#8-method-2b-iterative-integration)
9. [Spectral Processing Details](#9-spectral-processing-details)
10. [Constants and Parameters](#10-constants-and-parameters)
11. [References](#11-references)

---

## 1. Definition of Chi

The rate of dissipation of thermal variance, chi, quantifies how quickly temperature microstructure is smoothed out by molecular thermal diffusion. It is defined as ([Dillon & Caldwell 1980](https://doi.org/10.1029/JC085iC04p01910), eq. 2):

```
chi = 6 * kappa_T * integral_0^inf  Phi_{dT/dz}(k)  dk       [K^2/s]
```

where:

- `Phi_{dT/dz}(k)` is the one-dimensional temperature gradient wavenumber spectrum in `[(K/m)^2 / cpm]`
- `kappa_T = 1.4e-7 m^2/s` is the molecular thermal diffusivity of seawater
- `k` is the cyclic wavenumber in cycles per metre (cpm)
- The factor 6 arises from the isotropy assumption: 3 spatial dimensions times a factor of 2 from the definition involving the full gradient tensor

The temperature gradient spectrum is obtained from measurements of fast-response FP07 thermistors sampling at ~512 Hz on a profiling instrument falling at speed `W`. Frequency `f` [Hz] maps to wavenumber via Taylor's frozen-turbulence hypothesis:

```
k = f / W       [cpm]
```

In practice, the integral is truncated at a finite upper wavenumber `k_max` due to the FP07 sensor rolloff and electronic noise, and a correction factor is applied to recover the unresolved variance. The two methods described below differ in how they handle this correction.


## 2. Batchelor Temperature Gradient Spectrum

The [Batchelor (1959)](https://doi.org/10.1017/S002211205900009X) model describes the one-dimensional temperature gradient spectrum in the viscous-convective and viscous-diffusive subranges. Following the formulation of [Dillon & Caldwell (1980)](https://doi.org/10.1029/JC085iC04p01910), eqs. 1 and 3:

### Batchelor wavenumber

The Batchelor wavenumber sets the rolloff scale where molecular diffusion begins to dominate:

```
kB = (1 / (2*pi)) * (epsilon / (nu * kappa_T^2))^(1/4)     [cpm]
```

([`batchelor.py: batchelor_kB`](../src/rsi_python/batchelor.py))

where `epsilon` is the TKE dissipation rate [W/kg] and `nu` is the kinematic viscosity [m^2/s]. The `1/(2*pi)` converts from rad/m to cpm.

### Non-dimensional shape function

```
f(alpha) = alpha * [ exp(-alpha^2 / 2)  -  alpha * sqrt(pi/2) * erfc(alpha / sqrt(2)) ]
```

([`batchelor.py: batchelor_nondim`](../src/rsi_python/batchelor.py))

where `alpha = sqrt(2*q) * k / kB` is the non-dimensional wavenumber and `erfc` is the complementary error function. This function:

- Peaks near `alpha ~ 1` (the viscous-convective to viscous-diffusive transition)
- Rolls off as a **Gaussian** `exp(-alpha^2/2)` at high wavenumbers
- Integrates to exactly `1/3` over `[0, inf)`

### Dimensional gradient spectrum

```
S(k) = sqrt(q/2) * chi / (kB * kappa_T) * f(alpha)     [(K/m)^2 / cpm]
```

([`batchelor.py: batchelor_grad`](../src/rsi_python/batchelor.py))

**Normalization check.** Substituting `u = alpha`, `dk = kB / sqrt(2*q) * du`:

```
integral_0^inf S(k) dk  =  sqrt(q/2) * chi / (kB * kappa_T) * kB / sqrt(2*q) * integral_0^inf f(u) du
                         =  chi / (2 * kappa_T) * (1/3)
                         =  chi / (6 * kappa_T)
```

This is consistent with the definition `chi = 6 * kappa_T * integral S(k) dk`. Verified numerically in [`tests/test_chi.py::TestBatchelorGrad::test_integral_equals_chi_over_6kT`](../tests/test_chi.py).

### Universal constant q

- `q = 3.7 +/- 1.5` ([Oakey 1982](https://doi.org/10.1175/1520-0485(1982)012%3C0256:DOTROD%3E2.0.CO;2), from field observations)
- `q = 3.9 +/- 0.25` ([Bogucki et al. 1997](https://doi.org/10.1017/S0022112097005727), from DNS)

The code uses `Q_BATCHELOR = 3.7` as default.


## 3. Kraichnan Temperature Gradient Spectrum

The [Kraichnan (1968)](https://doi.org/10.1063/1.1692063) model provides an alternative shape for the viscous-diffusive subrange. [Bogucki et al. (1997)](https://doi.org/10.1017/S0022112097005727) showed via DNS that the Kraichnan form fits simulated data significantly better than the Batchelor form, especially at high wavenumbers near and beyond `kB`, reducing epsilon estimation error from ~25% to ~2.5%.

### Dimensional gradient spectrum

The 1D temperature gradient spectrum in cpm, derived from [Bogucki et al. (1997)](https://doi.org/10.1017/S0022112097005727), eq. 11, and normalized to integrate to `chi / (6 * kappa_T)`:

```
S(k) = chi * q / (3 * kappa_T * kB^2) * k * (1 + sqrt(6*q) * y) * exp(-sqrt(6*q) * y)
```

([`batchelor.py: kraichnan_grad`](../src/rsi_python/batchelor.py))

where `y = k / kB` is the non-dimensional wavenumber.

### Key difference from Batchelor

The rolloff is **exponential** `exp(-sqrt(6*q) * y)` rather than **Gaussian** `exp(-alpha^2/2)`. This means:

- The Kraichnan spectrum falls off more gently at high wavenumbers
- More variance resides beyond `kB` (6% unresolved at `k*eta_B = 1` vs 2% for Batchelor)
- The shape better matches DNS data at all Prandtl numbers

### Normalization check

Substituting `u = k/kB`, `dk = kB * du`, and `a = sqrt(6*q)`:

```
integral_0^inf S(k) dk  =  chi * q / (3 * kappa_T) * integral_0^inf u * (1 + a*u) * exp(-a*u) du
```

The integral evaluates to:

```
integral_0^inf u * (1 + a*u) * exp(-a*u) du  =  1/a^2 + 2/a^2  =  3/a^2  =  3/(6*q)  =  1/(2*q)
```

So:

```
integral S dk  =  chi * q / (3 * kappa_T) * 1/(2*q)  =  chi / (6 * kappa_T)
```

Verified numerically in [`tests/test_chi.py::TestKraichnanGrad::test_integral_equals_chi_over_6kT`](../tests/test_chi.py).

### Universal constant q

- `q_K = 5.26 +/- 0.25` ([Bogucki et al. 1997](https://doi.org/10.1017/S0022112097005727), from DNS)

The code uses `Q_KRAICHNAN = 5.26` as default.


## 4. FP07 Thermistor Transfer Function

The FP07 glass-bead thermistor has a finite thermal response time that attenuates high-frequency (high-wavenumber) temperature fluctuations. This must be corrected when computing chi.

### Single-pole model ([Lueck et al. 1977](https://doi.org/10.1016/0146-6291(77)90565-3))

```
|H(f)|^2 = 1 / (1 + (2*pi*f*tau_0)^2)
```

([`fp07.py: fp07_transfer`](../src/rsi_python/fp07.py))

### Double-pole model ([Gregg & Meagher 1980](https://doi.org/10.1029/JC085iC05p02779))

Accounts for both the glass bead thermal mass and the thermal boundary layer:

```
|H(f)|^2 = 1 / (1 + (2*pi*f*tau_0)^2)^2
```

([`fp07.py: fp07_double_pole`](../src/rsi_python/fp07.py))

### Speed-dependent time constant

The FP07 time constant depends on the flow speed past the sensor, which controls the thermal boundary layer thickness:

| Model | Formula | Reference |
|-------|---------|-----------|
| Lueck | `tau_0 = 0.01 * (1/W)^0.5` | [Lueck et al. 1977](https://doi.org/10.1016/0146-6291(77)90565-3) |
| Peterson | `tau_0 = 0.012 * W^(-0.32)` | [Peterson & Fer 2014](https://doi.org/10.1016/j.mio.2014.05.002) |
| Goto | `tau_0 = 0.003` (fixed) | [Goto et al. 2016](https://doi.org/10.1175/JTECH-D-15-0220.1) |

([`fp07.py: fp07_tau`](../src/rsi_python/fp07.py))

At a typical profiling speed of `W = 0.7 m/s`, the Lueck model gives `tau_0 ~ 0.012 s`, corresponding to a half-power frequency of `f_{3dB} = 1/(2*pi*tau_0) ~ 13 Hz` or roughly `k_{3dB} ~ 19 cpm`. This means the FP07 attenuates the observed spectrum significantly in the viscous-diffusive subrange where the Batchelor spectrum rolls off — the correction for this attenuation is a central part of chi estimation.


## 5. Electronics Noise Model

The noise model determines the frequency-dependent noise floor of the temperature gradient measurement, which sets the upper integration limit for chi. It is ported from the ODAS MATLAB functions `noise_thermchannel.m` and `gradT_noise_odas.m` ([RSI Technical Note 040](https://rocklandscientific.com/support/technical-notes/)).

([`fp07.py: noise_thermchannel`](../src/rsi_python/fp07.py))

The noise propagates through four stages of the signal chain:

### Stage 1: First amplifier + thermistor Johnson noise

```
V_1(f) = 2 * E_n^2 * sqrt(1 + (f/f_c)^2) / (f/f_c)       [V^2/Hz]
phi_R  = 4 * K_B * R_0 * T_K                                [V^2/Hz]  (Johnson noise)
Noise_1 = G_1^2 * (V_1 + phi_R)
```

where `E_n = 4e-9 V/sqrt(Hz)` is the amplifier input voltage noise, `f_c = 18.7 Hz` is the flicker-noise knee frequency, `K_B` is Boltzmann's constant, `R_0` is the thermistor resistance, `T_K` is the operating temperature in Kelvin, and `G_1 = 6` is the first-stage gain.

### Stage 2: Pre-emphasis differentiator + second amplifier

```
G_2(f) = 1 + (2*pi*G_D*f)^2                                (pre-emphasis gain)
V_2(f) = 2 * E_n2^2 * sqrt(1 + (f/f_{c2})^2) / (f/f_{c2})
Noise_2 = G_2 * (Noise_1 + V_2)
```

where `G_D = 0.94 s` is the pre-emphasis differentiator gain, `E_n2 = 8e-9 V/sqrt(Hz)` is the second-stage amplifier noise, and `f_{c2} = 42 Hz` is its flicker-noise knee.

### Stage 3: Anti-aliasing filter

Two cascaded 4th-order Butterworth filters with cutoff `f_AA = 110 Hz`:

```
G_AA(f) = 1 / (1 + (f/f_AA)^8)^2
Noise_3 = Noise_2 * G_AA
```

### Stage 4: ADC sampling

```
delta = V_FS / 2^B                     (ADC step size)
Noise_4 = Noise_3 + gamma * delta^2 / (12 * f_N)
```

where `V_FS = 4.096 V`, `B = 16` bits, `gamma = 3` (RSI sampler excess noise factor), and `f_N = f_s/2` is the Nyquist frequency.

### Conversion to physical units

The noise in ADC counts is converted to temperature gradient units `[(K/m)^2 / Hz]` via:

1. Convert to counts: `Noise_counts = Noise_4 / delta^2`
2. Apply the high-pass transfer function from pre-emphasis deconvolution:

```
G_HP(f) = (1/G_D)^2 * (2*pi*G_D*f)^2 / (1 + (2*pi*G_D*f)^2)
```

3. Convert to physical gradient units using the Steinhart-Hart scale factor:

```
eta = (b/2) * 2^B * G_1 * E_b / V_FS
scale = T_K^2 * (1 + R/R_0)^2 / (2 * eta * beta_1 * R/R_0)
noise_physical = Noise_counts * G_HP * scale^2
```

The conversion from frequency spectrum to wavenumber spectrum is: `Phi_noise(k) = Phi_noise(f) * W`, where `W` is the profiling speed.


## 6. Method 1: Chi from Known Epsilon

When shear probes provide an independent estimate of epsilon (from [`get_diss`](../src/rsi_python/dissipation.py)), the Batchelor wavenumber is fully determined and chi can be computed by integrating the observed temperature gradient spectrum with corrections for sensor rolloff and unresolved variance.

([`chi.py: _chi_from_epsilon`](../src/rsi_python/chi.py))

### Algorithm

For each dissipation window:

**Step 1.** Compute the Batchelor wavenumber from the known epsilon:

```
kB = (1/(2*pi)) * (epsilon / (nu * kappa_T^2))^(1/4)     [cpm]
```

**Step 2.** Compute the FP07 transfer function `|H(k)|^2` using the speed-dependent time constant. The tau model is auto-selected based on `fp07_model`: `single_pole` uses the Lueck model; `double_pole` uses the Goto model (see Section 4).

```
|H(f)|^2 = 1 / (1 + (2*pi*f*tau_0)^2)       (single_pole)
|H(f)|^2 = 1 / (1 + (2*pi*f*tau_0)^2)^2     (double_pole)
```

where `f = k * W`.

**Step 3.** Compute the electronics noise spectrum `Phi_noise(k)` (Section 5).

**Step 4.** Determine the upper integration limit `k_max`: the highest wavenumber where `Phi_obs(k) > 2 * Phi_noise(k)` and `k <= f_AA / W`. Falls back to the anti-aliasing limit if fewer than 3 wavenumber bins are above the noise.

**Step 5.** Integrate the observed spectrum:

```
V_obs = integral_0^{k_max}  Phi_obs(k)  dk
```

**Step 6.** Compute the correction factor for FP07 rolloff and unresolved variance. Using a trial `chi_trial = 6 * kappa_T * V_obs`:

```
V_total    = integral_0^{5*kB}  Phi_Batchelor(k; kB, chi_trial)  dk

V_resolved = integral_0^{k_max}  Phi_Batchelor(k; kB, chi_trial) * |H(k)|^2  dk

C = V_total / V_resolved
```

The correction `C` accounts for two effects simultaneously:
- The FP07 sensor attenuates the observed spectrum (`|H|^2 < 1` at high k)
- The integration is truncated at `k_max < inf`, missing the tail of the spectrum

**Step 7.** Compute chi:

```
chi = 6 * kappa_T * V_obs * C       [K^2/s]
```

Note that `chi_trial` cancels in the ratio `C = V_total / V_resolved` because both the numerator and denominator scale linearly with chi. The correction depends only on the spectral *shape* (determined by `kB` and `q`), not on the amplitude.


## 7. Method 2a: Maximum Likelihood Estimation

When no independent epsilon is available (e.g., MicroRider deployments without shear probes), the Batchelor wavenumber `kB` is treated as a free parameter and estimated by fitting the theoretical spectrum to the observed spectrum. This simultaneously yields both chi and epsilon from temperature data alone.

([`chi.py: _mle_fit_kB`](../src/rsi_python/chi.py))

### Theory ([Ruddick et al. 2000](https://doi.org/10.1175/1520-0426(2000)017%3C1541:MLSFTB%3E2.0.CO;2))

Spectral estimates from Welch's method are chi-squared distributed. For `d` degrees of freedom and a model spectrum `Phi_model(k_i)`, the negative log-likelihood is:

```
-ln L = sum_i [ ln Phi_model(k_i)  +  Phi_obs(k_i) / Phi_model(k_i) ]       (for d = 2)
```

The model spectrum at each wavenumber is the theoretical gradient spectrum convolved with the FP07 transfer function, plus the noise floor:

```
Phi_model(k; kB) = Phi_Batchelor(k; kB, chi_obs) * |H(k)|^2  +  Phi_noise(k)
```

The **only free parameter is `kB`**. Given an initial estimate of chi from integrating the observed spectrum (`chi_obs`), the Batchelor/Kraichnan spectrum shape is completely determined by `kB` and `q`.

### Grid search algorithm

The MLE is solved by grid search rather than gradient optimization, following [Ruddick et al. (2000)](https://doi.org/10.1175/1520-0426(2000)017%3C1541:MLSFTB%3E2.0.CO;2). Two passes:

1. **Coarse search:** 100 log-spaced `kB` values over `[1, 10^4.5]` cpm. Evaluate `-ln L` at each and find the minimum.

2. **Fine search:** 100 linearly-spaced `kB` values over `[0.5 * kB_coarse, 2 * kB_coarse]`. Find the refined minimum.

Total: 200 function evaluations.

### Fitting range

- **Low-k cutoff:** first wavenumber bin above zero
- **High-k cutoff:** highest `k` where `Phi_obs(k) > 2 * Phi_noise(k)` and `k <= f_AA / W`
- Minimum 6 wavenumber points required in the fit range

### Recovering epsilon and chi

From the best-fit `kB`:

```
epsilon = (2*pi * kB)^4 * nu * kappa_T^2       [W/kg]
```

(This is the inverse of the `batchelor_kB` formula.)

Chi is computed by integrating the fitted Batchelor spectrum over its full extent:

```
chi = 6 * kappa_T * integral_0^{5*kB}  Phi_Batchelor(k; kB_fit, chi_obs)  dk
```

### Properties

- MLE bias is < 0.25% (essentially unbiased per Monte Carlo tests in [Ruddick et al. 2000](https://doi.org/10.1175/1520-0426(2000)017%3C1541:MLSFTB%3E2.0.CO;2))
- Error bars can be obtained from the region where `ln L` drops by 0.5 from the maximum (1-sigma confidence)


## 8. Method 2b: Iterative Integration

[Peterson & Fer (2014)](https://doi.org/10.1016/j.mio.2014.05.002) extended the MLE approach with iterative refinement of the integration limits and correction for unresolved variance at both low and high wavenumbers.

([`chi.py: _iterative_fit`](../src/rsi_python/chi.py))

### Algorithm (3 iterations)

**Initialization:**

```
chi_obs = 6 * kappa_T * integral_{k_1}^{k_u}  max(Phi_obs(k) - Phi_noise(k), 0)  dk
```

where `k_1` and `k_u` are the initial low and high cutoffs (from the noise intersection criterion).

**Each iteration:**

1. Perform MLE grid search (Section 7) to find `kB` given `chi_obs`.

2. Compute the characteristic wavenumber `k*` separating the inertial-convective and viscous-convective subranges:

```
k* = 0.04 * kB * sqrt(kappa_T / nu)
```

3. Refine the lower integration limit:

```
k_l = max(k_1, 3 * k*)
```

This ensures the fit range lies within the viscous-convective and viscous-diffusive subranges where the Batchelor/Kraichnan model is valid, excluding the inertial-convective subrange where the spectrum follows a different power law.

4. Recompute `chi_obs` with the refined limits:

```
chi_obs = 6 * kappa_T * integral_{k_l}^{k_u}  max(Phi_obs - Phi_noise, 0)  dk
```

5. Compute unresolved variance from the Batchelor model:

```
chi_low  = 6 * kappa_T * integral_0^{k_l}      Phi_Batchelor(k; kB, chi_obs)  dk
chi_high = 6 * kappa_T * integral_{k_u}^{5*kB}  Phi_Batchelor(k; kB, chi_obs)  dk
```

6. Update total chi:

```
chi = chi_low + chi_obs + chi_high
```

7. Re-fit `kB` by MLE with the updated chi.

### Quality control ([Peterson & Fer 2014](https://doi.org/10.1016/j.mio.2014.05.002))

The following checks (not yet enforced as hard filters in the code, but available for post-processing):

- At least 6 wavenumber points in the fit range
- `chi_low + chi_high <= chi_obs` (model corrections should not dominate the observed variance)
- Mean absolute deviation `MAD < 2 * sqrt(2/d)` where `d` is the spectral degrees of freedom (goodness of fit)
- For FP07 thermistors: `epsilon <= ~2e-7 W/kg` for reliable estimates (at higher dissipation rates the Batchelor rolloff moves beyond the sensor's resolved bandwidth)


## 9. Spectral Processing Details

### Welch's method

Temperature gradient spectra are computed using [`csd_odas`](../src/rsi_python/spectral.py) (ported from ODAS `csd_odas.m`):

- Window: Hann (cosine) window normalized to RMS = 1
- Overlap: 50% (default `fft_length // 2`)
- Detrending: linear
- Normalization: one-sided power spectral density in `[units^2 / Hz]`

### Frequency to wavenumber conversion

```
Phi(k) = Phi(f) * W       [units^2/Hz * m/s = units^2/cpm]
k = f / W                 [cpm]
```

where `W` is the mean profiling speed for the dissipation window.

### First-difference correction

When the temperature gradient is obtained by first-differencing the deconvolved temperature signal (`T[n+1] - T[n]`), the resulting spectrum must be corrected for the frequency response of the first-difference operator:

```
C_FD(f) = ( pi*f / (f_s * sin(pi*f/f_s)) )^2       for f > 0
```

([`chi.py: _compute_profile_chi`](../src/rsi_python/chi.py))

This correction approaches 1 at low frequencies and diverges at the Nyquist frequency. The corrected spectrum is:

```
Phi_corrected(k) = Phi_raw(k) * C_FD(f)
```

### Dissipation windows

The profile is divided into overlapping windows of length `diss_length` samples (default `3 * fft_length = 1536` samples at 512 Hz = 3 seconds of data). Adjacent windows overlap by `overlap` samples (default `diss_length // 2`). Within each window:

- Mean pressure, temperature, speed, and time are computed
- Kinematic viscosity `nu` is computed from [`visc35(T_mean)`](../src/rsi_python/ocean.py)
- The spectral estimate uses `2 * (diss_length // fft_length) - 1` overlapping FFT segments
- Degrees of freedom: `d = 1.9 * num_ffts` (Nuttall 1971)


## 10. Constants and Parameters

### Physical constants

| Symbol | Value | Description | Source |
|--------|-------|-------------|--------|
| `kappa_T` | `1.4e-7 m^2/s` | Molecular thermal diffusivity of seawater | Standard value |
| `q_B` | `3.7` | Batchelor universal constant | [Oakey 1982](https://doi.org/10.1175/1520-0485(1982)012%3C0256:DOTROD%3E2.0.CO;2) |
| `q_K` | `5.26` | Kraichnan universal constant | [Bogucki et al. 1997](https://doi.org/10.1017/S0022112097005727) |

### Default instrument parameters

| Symbol | Value | Description |
|--------|-------|-------------|
| `f_s` | `512 Hz` | Fast sampling rate |
| `f_AA` | `98 Hz` | Anti-aliasing filter effective cutoff (0.9 * 110 Hz) |
| `G_D` | `0.94 s` | Pre-emphasis differentiator gain |
| `G_1` | `6` | First-stage amplifier gain |
| `R_0` | `3000 Ohm` | Nominal thermistor resistance |
| `E_n` | `4e-9 V/sqrt(Hz)` | First-stage amplifier voltage noise |
| `f_c` | `18.7 Hz` | First-stage flicker noise knee |
| `E_n2` | `8e-9 V/sqrt(Hz)` | Second-stage amplifier voltage noise |
| `f_{c2}` | `42 Hz` | Second-stage flicker noise knee |
| `V_FS` | `4.096 V` | ADC full-scale voltage |
| `B` | `16` | ADC bits |
| `gamma` | `3` | RSI sampler excess noise factor |

### Default processing parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fft_length` | `512` | FFT segment length [samples] |
| `diss_length` | `1536` | Dissipation window length (3 * fft_length) [samples] |
| `overlap` | `768` | Window overlap (diss_length // 2) [samples] |
| `fp07_model` | `single_pole` | FP07 transfer function model |
| `spectrum_model` | `kraichnan` | Theoretical spectrum (batchelor or kraichnan) |
| `fit_method` | `iterative` | Method 2 fitting algorithm (mle or iterative) |
| `goodman` | `True` | Goodman coherent noise removal using accelerometers |


## 11. References

### Batchelor spectrum and chi theory

- Batchelor, G.K., 1959: [Small-scale variation of convected quantities like temperature in turbulent fluid.](https://doi.org/10.1017/S002211205900009X) *J. Fluid Mech.*, **5**, 113-133.
- Kraichnan, R.H., 1968: [Small-scale structure of a scalar field convected by turbulence.](https://doi.org/10.1063/1.1692063) *Phys. Fluids*, **11**, 945-953.
- Dillon, T.M. and D.R. Caldwell, 1980: [The Batchelor spectrum and dissipation in the upper ocean.](https://doi.org/10.1029/JC085iC04p01910) *J. Geophys. Res.*, **85**, 1910-1916.
- Oakey, N.S., 1982: [Determination of the rate of dissipation of turbulent energy from simultaneous temperature and velocity shear microstructure measurements.](https://doi.org/10.1175/1520-0485(1982)012%3C0256:DOTROD%3E2.0.CO;2) *J. Phys. Oceanogr.*, **12**, 256-271.

### Chi estimation methods

- Ruddick, B., A. Anis, and K. Thompson, 2000: [Maximum likelihood spectral fitting: The Batchelor spectrum.](https://doi.org/10.1175/1520-0426(2000)017%3C1541:MLSFTB%3E2.0.CO;2) *J. Atmos. Oceanic Technol.*, **17**, 1541-1555.
- Peterson, A.K. and I. Fer, 2014: [Dissipation measurements using temperature microstructure from an underwater glider.](https://doi.org/10.1016/j.mio.2014.05.002) *Methods in Oceanography*, **10**, 44-69.
- Osborn, T.R. and C.S. Cox, 1972: [Oceanic fine structure.](https://doi.org/10.1080/03091927208236085) *Geophys. Fluid Dyn.*, **3**, 321-345.

### FP07 thermistor response

- Lueck, R.G., O. Hertzman, and T.R. Osborn, 1977: [The spectral response of thermistors.](https://doi.org/10.1016/0146-6291(77)90565-3) *Deep-Sea Res.*, **24**, 951-970.
- Gregg, M.C. and T.B. Meagher, 1980: [The dynamic response of glass rod thermistors.](https://doi.org/10.1029/JC085iC05p02779) *J. Geophys. Res.*, **85**, 2779-2786.
- Goto, Y., I. Yasuda, and M. Nagasawa, 2016: [Comparison of turbulence intensity from CTD-attached and free-fall microstructure profilers.](https://doi.org/10.1175/JTECH-D-15-0220.1) *J. Atmos. Oceanic Technol.*, **33**, 1065-1081.
- Nash, J.D., T.B. Caldwell, M.J. Zelman, and J.N. Moum, 1999: [A thermocouple probe for high-speed temperature measurement in the ocean.](https://doi.org/10.1175/1520-0426(1999)016%3C1474:ATPFHS%3E2.0.CO;2) *J. Atmos. Oceanic Technol.*, **16**, 1474-1482.

### Noise model

- Rockland Scientific, [Technical Note 040](https://rocklandscientific.com/support/technical-notes/): Noise in Temperature Gradient Measurements.

### DNS and empirical validation

- Bogucki, D., J.A. Domaradzki, and P.K. Yeung, 1997: [Direct numerical simulations of passive scalars with Pr > 1 advected by turbulent flow.](https://doi.org/10.1017/S0022112097005727) *J. Fluid Mech.*, **343**, 111-130.

### Software

- Rockland Scientific [ODAS MATLAB Library](https://rocklandscientific.com/support/software/) (v4.5.1).
