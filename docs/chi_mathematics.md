# Mathematics of Chi Estimation

This document describes the mathematical foundations for computing chi, the rate of dissipation of thermal variance, as implemented in `microstructure-tpw`. All equation numbers, constants, and algorithmic details correspond to the actual code in [`batchelor.py`](../src/odas_tpw/chi/batchelor.py), [`fp07.py`](../src/odas_tpw/chi/fp07.py), and [`chi.py`](../src/odas_tpw/chi/chi.py).

## Contents

1. [Definition of Chi](#1-definition-of-chi)
2. [Batchelor Temperature Gradient Spectrum](#2-batchelor-temperature-gradient-spectrum)
3. [Kraichnan Temperature Gradient Spectrum](#3-kraichnan-temperature-gradient-spectrum)
4. [FP07 Thermistor Transfer Function](#4-fp07-thermistor-transfer-function)
5. [Electronics Noise Model](#5-electronics-noise-model)
6. [Method 1: Chi from Known Epsilon](#6-method-1-chi-from-known-epsilon)
7. [Method 2a: Maximum Likelihood Estimation](#7-method-2a-maximum-likelihood-estimation)
8. [Method 2b: Iterative Integration](#8-method-2b-iterative-integration)
9. [Quality Control Metrics](#9-quality-control-metrics)
10. [Spectral Processing Details](#10-spectral-processing-details)
11. [Constants and Parameters](#11-constants-and-parameters)
12. [References](#12-references)

---

## 1. Definition of Chi

The rate of dissipation of thermal variance, chi, quantifies how quickly temperature microstructure is smoothed out by molecular thermal diffusion. It is defined as ([Dillon & Caldwell 1980](https://doi.org/10.1029/JC085iC04p01910), eq. 2):

```
chi = 6 * kappa_T * integral_0^inf  Phi_{dT/dz}(k)  dk       [K^2/s]
```

where:

- `Phi_{dT/dz}(k)` is the one-dimensional temperature gradient wavenumber spectrum in `[(K/m)^2 / cpm]`
- `kappa_T` is the molecular thermal diffusivity of seawater, evaluated
  **per window** at the window-mean temperature/salinity/pressure via
  [`scor160.ocean.kappa_T`](../src/odas_tpw/scor160/ocean.py) (Sharqawy 2010 /
  Jamieson–Tudhope 1970 thermal conductivity ÷ gsw density·specific-heat). It
  ranges from ~`1.39e-7` at −1 °C to ~`1.51e-7 m^2/s` at 32 °C. The fixed
  `1.4e-7` (`chi.batchelor.KAPPA_T`) is now only the backward-compatible
  fallback default — see §11.
- `k` is the cyclic wavenumber in cycles per meter (cpm)
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

([`batchelor.py: batchelor_kB`](../src/odas_tpw/chi/batchelor.py))

where `epsilon` is the TKE dissipation rate [W/kg] and `nu` is the kinematic viscosity [m^2/s]. The `1/(2*pi)` converts from rad/m to cpm.

### Non-dimensional shape function

```
f(alpha) = alpha * [ exp(-alpha^2 / 2)  -  alpha * sqrt(pi/2) * erfc(alpha / sqrt(2)) ]
```

([`batchelor.py: batchelor_nondim`](../src/odas_tpw/chi/batchelor.py))

where `alpha = sqrt(2*q) * k / kB` is the non-dimensional wavenumber and `erfc` is the complementary error function. This function:

- Peaks near `alpha ~ 1` (the viscous-convective to viscous-diffusive transition)
- Rolls off as a **Gaussian** `exp(-alpha^2/2)` at high wavenumbers
- Integrates to exactly `1/3` over `[0, inf)`

### Dimensional gradient spectrum

```
S(k) = sqrt(q/2) * chi / (kB * kappa_T) * f(alpha)     [(K/m)^2 / cpm]
```

([`batchelor.py: batchelor_grad`](../src/odas_tpw/chi/batchelor.py))

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

The one-dimensional (along-path) temperature gradient spectrum in cpm:

```
S(k) = q * chi / (kappa_T * kB^2) * k * exp(-sqrt(6*q) * y)
```

([`batchelor.py: kraichnan_grad`](../src/odas_tpw/chi/batchelor.py))

where `y = k / kB` is the non-dimensional wavenumber.

This form is obtained by applying the isotropic transform

```
G1(k1) = k1^2 * integral_{k1}^inf  E(k) / k  dk
```

to the three-dimensional Kraichnan scalar spectrum of [Bogucki et al. (1997)](https://doi.org/10.1017/S0022112097005727), eq. 11. For the Kraichnan form the transform evaluates in closed form to the simple exponential above — the same expression as [Peterson & Fer (2014)](https://doi.org/10.1016/j.mio.2014.05.002), eq. 8. The spectrum peaks at `k = kB / sqrt(6*q)`.

### Key difference from Batchelor

The rolloff is **exponential** `exp(-sqrt(6*q) * y)` rather than **Gaussian** `exp(-alpha^2/2)`. This means:

- The Kraichnan spectrum falls off more gently at high wavenumbers
- More variance resides beyond `kB` (6% unresolved at `k*eta_B = 1` vs 2% for Batchelor)
- The shape better matches DNS data at all Prandtl numbers

### Normalization check

Substituting `u = k/kB`, `dk = kB * du`, and `a = sqrt(6*q)`:

```
integral_0^inf S(k) dk  =  chi * q / kappa_T * integral_0^inf u * exp(-a*u) du
```

The integral evaluates to:

```
integral_0^inf u * exp(-a*u) du  =  1/a^2  =  1/(6*q)
```

So:

```
integral S dk  =  chi * q / kappa_T * 1/(6*q)  =  chi / (6 * kappa_T)
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

([`fp07.py: fp07_transfer`](../src/odas_tpw/chi/fp07.py))

### Double-pole model ([Gregg & Meagher 1980](https://doi.org/10.1029/JC085iC05p02779))

Accounts for both the glass bead thermal mass and the thermal boundary layer:

```
|H(f)|^2 = 1 / (1 + (2*pi*f*tau_0)^2)^2
```

([`fp07.py: fp07_double_pole`](../src/odas_tpw/chi/fp07.py))

### Speed-dependent time constant

The FP07 time constant depends on the flow speed past the sensor, which controls the thermal boundary layer thickness. The `W^(-1/2)` speed scaling was established by Vachon & Lueck (1984, *Proc. 1984 STD Conference and Workshop*, Marine Technology Society); [Peterson & Fer (2014)](https://doi.org/10.1016/j.mio.2014.05.002) fitted a slightly weaker exponent for their glider data:

| Model | Formula | Reference |
|-------|---------|-----------|
| Lueck | `tau_0 = 0.01 * (1/W)^0.5` | [Lueck et al. 1977](https://doi.org/10.1016/0146-6291(77)90565-3) |
| Peterson | `tau_0 = 0.012 * W^(-0.32)` | [Peterson & Fer 2014](https://doi.org/10.1016/j.mio.2014.05.002) |
| Goto | `tau_0 = 0.003` (fixed) | [Goto et al. 2016](https://doi.org/10.1175/JTECH-D-15-0220.1) |

([`fp07.py: fp07_tau`](../src/odas_tpw/chi/fp07.py))

At a typical profiling speed of `W = 0.7 m/s`, the Lueck model gives `tau_0 ~ 0.012 s`, corresponding to a half-power frequency of `f_{3dB} = 1/(2*pi*tau_0) ~ 13 Hz` or roughly `k_{3dB} ~ 19 cpm`. This means the FP07 attenuates the observed spectrum significantly in the viscous-diffusive subrange where the Batchelor spectrum rolls off — the correction for this attenuation is a central part of chi estimation.


## 5. Electronics Noise Model

The noise model determines the frequency-dependent noise floor of the temperature gradient measurement, which sets the upper integration limit for chi. It is ported from the ODAS MATLAB functions `noise_thermchannel.m` and `gradT_noise_odas.m` ([RSI Technical Note 040](https://rocklandscientific.com/support/technical-notes/)).

([`fp07.py: noise_thermchannel`](../src/odas_tpw/chi/fp07.py))

The noise propagates through four stages of the signal chain:

### Stage 1: First amplifier + thermistor Johnson noise

```
V_1(f) = 2 * E_n^2 * sqrt(1 + (f/f_c)^2) / (f/f_c)       [V^2/Hz]
phi_R  = 4 * K_B * R_actual * T_K                          [V^2/Hz]  (Johnson noise)
Noise_1 = G_1^2 * (V_1 + phi_R)
```

where `E_n = 4e-9 V/sqrt(Hz)` is the amplifier input voltage noise, `f_c = 18.7 Hz` is the flicker-noise knee frequency, `K_B` is Boltzmann's constant, `R_actual = R_ratio * R_0` is the temperature-corrected thermistor resistance (the code uses the in-situ value, not the nominal `R_0`), `T_K` is the operating temperature in Kelvin, and `G_1 = 6` is the first-stage gain.

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

The noise in ADC counts is converted to temperature *time-derivative* units `[(K/s)^2 / Hz]` via:

1. Convert to counts: `Noise_counts = Noise_4 / delta^2`
2. Apply the high-pass transfer function from pre-emphasis deconvolution:

```
G_HP(f) = (1/G_D)^2 * (2*pi*G_D*f)^2 / (1 + (2*pi*G_D*f)^2)
```

3. Convert to physical gradient units using the Steinhart-Hart scale factor:

```
eta = (b/2) * 2^B * G_1 * E_b / V_FS
scale = T_in_situ^2 * (1 + R/R_0)^2 / (2 * eta * beta_1 * R/R_0)
noise_physical = Noise_counts * G_HP * scale^2
```

where `T_in_situ` is the in-situ temperature in Kelvin (`T_mean + 273.15`), distinct from the fixed `T_K ≈ 295 K` operating temperature used in the Johnson-noise term above.

The conversion from frequency spectrum to wavenumber spectrum is:

```
Phi_noise(k) = Phi_noise(f) / W
```

where `W` is the profiling speed ([`fp07.py: gradT_noise`](../src/odas_tpw/chi/fp07.py), matching ODAS `gradT_noise_odas.m`). The electronics noise model produces a *time-derivative* noise spectrum in `(K/s)^2/Hz`; dividing by `W^2` converts it to a spatial-gradient spectrum in `(K/m)^2/Hz`, and multiplying by `W` converts per-Hz to per-cpm. The net factor is `1/W`.


## 6. Method 1: Chi from Known Epsilon

When shear probes provide an independent estimate of epsilon (from [`get_diss`](../src/odas_tpw/rsi/dissipation.py)), the Batchelor wavenumber is fully determined and chi can be computed by integrating the observed temperature gradient spectrum with corrections for sensor rolloff and unresolved variance.

([`chi.py: _chi_from_epsilon`](../src/odas_tpw/chi/chi.py))

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

**Step 5.** Compute an initial (variance-corrected) chi estimate. First integrate the observed spectrum:

```
V_obs = integral_0^{k_max}  Phi_obs(k)  dk
```

Then compute the correction factor for FP07 rolloff and unresolved variance:

```
V_total    = integral over all k  of  Phi_model(k; kB, chi=1)

V_resolved = integral_0^{k_max}   of  Phi_model(k; kB, chi=1) * |H(k)|^2

C = V_total / V_resolved
```

The correction `C` accounts for two effects simultaneously:
- The FP07 sensor attenuates the observed spectrum (`|H|^2 < 1` at high k)
- The integration is truncated at `k_max < inf`, missing the tail of the spectrum

`C` is evaluated at unit chi because both the numerator and denominator scale linearly with chi, so the amplitude cancels; the correction depends only on the spectral *shape* (determined by `kB` and `q`). The variance-corrected estimate is:

```
chi_vc = 6 * kappa_T * V_obs * C       [K^2/s]
```

**Step 6.** Refine chi by a log-space least-squares fit. `chi_vc` is used only to center a grid search: 200 log-spaced chi values spanning 4 decades, `[0.01 * chi_vc, 100 * chi_vc]`. For each candidate chi the model spectrum is:

```
Phi_model(k; chi) = chi * S_unit(k; kB) * |H(k)|^2  +  Phi_noise(k)
```

where `S_unit` is the Batchelor/Kraichnan gradient spectrum at unit chi with `kB` fixed from epsilon. The cost function is the sum of squared log residuals over the valid wavenumber range:

```
cost(chi) = sum_i [ ln Phi_model(k_i; chi) - ln Phi_obs(k_i) ]^2
```

**Step 7.** Report the grid-minimizing chi:

```
chi = argmin_chi  cost(chi)       [K^2/s]
```

Minimizing in log space penalizes over- and under-estimation symmetrically on the log-log plot, and is more robust than the pure variance-correction estimate when the epsilon-derived `kB` does not perfectly match the temperature spectrum. The reported chi is not the raw grid argmin: the grid minimum is refined by fitting a parabola through the minimum and its two neighbors in `(log10 chi, cost)` space, removing the ~half-grid-step (~2.3%) quantization. If the fit fails (all costs non-finite), `chi_vc` is used.


## 7. Method 2a: Maximum Likelihood Estimation

When no independent epsilon is available (e.g., MicroRider deployments without shear probes), the Batchelor wavenumber `kB` is treated as a free parameter and estimated by fitting the theoretical spectrum to the observed spectrum. This simultaneously yields both chi and epsilon from temperature data alone.

([`chi.py: _mle_fit_kB`](../src/odas_tpw/chi/chi.py))

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

2. **Fine search:** 100 linearly-spaced `kB` values over `[max(0.5 * kB_coarse, 1.0), 2 * kB_coarse]`. The lower bound is clamped to 1 cpm. Find the refined minimum.

Total: 200 function evaluations.

### Fitting range

- **Low-k cutoff:** first wavenumber bin above the `2 * Phi_noise` criterion (fallback to all `k` in `(0, K_AA]` when fewer than 6 bins qualify)
- **High-k cutoff:** highest `k` where `Phi_obs(k) > 2 * Phi_noise(k)` and `k <= f_AA / W`
- Minimum 6 wavenumber points required in the fit range

### Recovering epsilon and chi

From the best-fit `kB`:

```
epsilon = (2*pi * kB)^4 * nu * kappa_T^2       [W/kg]
```

(This is the inverse of the `batchelor_kB` formula.)

Chi is re-estimated from the *observed* spectrum with an unresolved-variance correction (the same construction as Method 1, Step 5):

```
chi = 6 * kappa_T * integral_{fit range}  max(Phi_obs(k) - Phi_noise(k), 0)  dk  *  (V_total / V_resolved)
```

where the correction ratio is computed from the fitted model at unit chi:

```
V_total    = integral over all k    of  Phi_model(k; kB_fit, chi=1)
V_resolved = integral_{K_fit_low}^{K_max_fit} of  Phi_model(k; kB_fit, chi=1) * |H(k)|^2
```

`V_resolved` includes the FP07 attenuation `|H|^2`, so the ratio simultaneously corrects for the unresolved band edges and for in-band sensor response. If the correction is not finite, the initial integrated estimate `chi_obs` is reported.

### Properties

- MLE bias is < 0.25% (essentially unbiased per Monte Carlo tests in [Ruddick et al. 2000](https://doi.org/10.1175/1520-0426(2000)017%3C1541:MLSFTB%3E2.0.CO;2))
- Error bars can be obtained from the region where `ln L` drops by 0.5 from the maximum (1-sigma confidence)


## 8. Method 2b: Iterative Integration

[Peterson & Fer (2014)](https://doi.org/10.1016/j.mio.2014.05.002) extended the MLE approach with iterative refinement of the integration limits and correction for unresolved variance at both low and high wavenumbers.

([`chi.py: _iterative_fit`](../src/odas_tpw/chi/chi.py))

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
k_l = max(K[1], 3 * k*)
```

where `K[1]` is the first nonzero wavenumber bin.

This ensures the fit range lies within the viscous-convective and viscous-diffusive subranges where the Batchelor/Kraichnan model is valid, excluding the inertial-convective subrange where the spectrum follows a different power law.

4. Recompute the band-limited chi with the refined limits:

```
chi_band = 6 * kappa_T * integral_{k_l}^{k_u}  max(Phi_obs - Phi_noise, 0)  dk
```

5. Apply a model-based variance-ratio correction for the variance outside `[k_l, k_u]` *and* for in-band FP07 attenuation, computed from the fitted model at unit chi (amplitude cancels in the ratio):

```
V_total    = integral over all k   of  Phi_model(k; kB, chi=1)
V_resolved = integral_{k_l}^{k_u}  of  Phi_model(k; kB, chi=1) * |H(k)|^2

chi_obs = chi_band * (V_total / V_resolved)
```

The observed spectrum is never divided by `|H|^2`, so `V_resolved` carries the `|H|^2` factor instead. [Peterson & Fer (2014)](https://doi.org/10.1016/j.mio.2014.05.002) equivalently boost the measured spectrum by the response function before integrating.

6. Re-fit `kB` by MLE with the updated `chi_obs`. The loop stops early when `kB` changes by less than 1% between iterations.

### Quality control ([Peterson & Fer 2014](https://doi.org/10.1016/j.mio.2014.05.002))

The following checks (not yet enforced as hard filters in the code, but available for post-processing):

- At least 6 wavenumber points in the fit range
- The model-based correction should not dominate the observed variance: the variance restored by the correction should not exceed the directly observed band-limited variance (correction factor `V_total / V_resolved <~ 2`; Peterson & Fer express this as `chi_low + chi_high <= chi_obs` for their additive out-of-band tail corrections)
- Mean absolute deviation `MAD < 2 * sqrt(2/d)` where `d` is the spectral degrees of freedom (goodness of fit)
- For FP07 thermistors: `epsilon <= ~2e-7 W/kg` for reliable estimates (at higher dissipation rates the Batchelor rolloff moves beyond the sensor's resolved bandwidth)


## 9. Quality Control Metrics

All three chi methods report the same two QC metrics ([`chi.py`](../src/odas_tpw/chi/chi.py)):

### Figure of merit (fom)

The ratio of observed variance to *attenuated-model* variance over the fit range:

```
fom = integral Phi_obs(k) dk  /  integral [ Phi_model(k; kB, chi) * |H(k)|^2 + Phi_noise(k) ] dk
```

The denominator is the fitted Batchelor/Kraichnan spectrum convolved with the FP07 transfer function plus the electronics noise floor — i.e., what the sensor *should* observe given the fitted (chi, kB). Values near 1.0 indicate a good fit; values far from 1.0 indicate the model does not explain the observed variance.

Note: this is a simple variance ratio, **not** the MAD-based FM statistic of Lueck (2022a,b) used for the ATOMIX shear-probe QC (see [epsilon_mathematics.md](epsilon_mathematics.md#10-quality-control-metrics)).

### K_max ratio

```
K_max_ratio = K_max / kB
```

where `K_max` is the upper limit of the fit/integration range and `kB` is the Batchelor wavenumber. Values < 0.5 mean the resolved band ends well below the spectral rolloff, so most of the variance is extrapolated through the model-based variance correction and the chi estimate should be treated with caution.

### FP07 validity caveat

As noted in Section 8, [Peterson & Fer (2014)](https://doi.org/10.1016/j.mio.2014.05.002) recommend trusting FP07-derived estimates only for `epsilon <= ~2e-7 W/kg`; at higher dissipation rates the Batchelor rolloff moves beyond the sensor's resolved bandwidth (low `K_max_ratio`).


## 10. Spectral Processing Details

### Welch's method

Temperature gradient spectra are computed using [`csd_odas`](../src/odas_tpw/scor160/spectral.py) (ported from ODAS `csd_odas.m`):

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

([`l3_chi.py: process_l3_chi`](../src/odas_tpw/chi/l3_chi.py), see the `fd_correction` array built around L254)

This correction approaches 1 at low frequencies and diverges at the Nyquist frequency. The corrected spectrum is:

```
Phi_corrected(k) = Phi_raw(k) * C_FD(f)
```

### Dissipation windows

The profile is divided into overlapping windows of length `diss_length` samples (default `4 * fft_length = 4096` samples at 512 Hz = 8 seconds of data). Adjacent windows overlap by `overlap` samples (default `diss_length // 2 = 2048`). Within each window:

- Mean pressure, temperature, speed, and time are computed
- Kinematic viscosity `nu` is computed from [`visc(T_mean, S, P_mean)`](../src/odas_tpw/scor160/ocean.py) when salinity is available (measured JAC C/T on the pipeline path, or user-supplied), falling back to `visc35(T_mean)` otherwise
- The spectral estimate uses `2 * (diss_length // fft_length) - 1` overlapping FFT segments
- Degrees of freedom: `d = 1.9 * num_ffts` (Nuttall 1971)


## 11. Constants and Parameters

### Physical constants

| Symbol | Value | Description | Source |
|--------|-------|-------------|--------|
| `kappa_T(T,S,P)` | `1.39e-7`–`1.51e-7 m^2/s` | Molecular thermal diffusivity of seawater, **per-window** temperature-dependent | [`scor160.ocean.kappa_T`](../src/odas_tpw/scor160/ocean.py); Sharqawy 2010 / Jamieson–Tudhope 1970 + gsw |
| `KAPPA_T` | `1.4e-7 m^2/s` | Fixed ~15 °C fallback default only (used when no per-window value is supplied) | Standard value |
| `q_B` | `3.7` | Batchelor universal constant | [Oakey 1982](https://doi.org/10.1175/1520-0485(1982)012%3C0256:DOTROD%3E2.0.CO;2) |
| `q_K` | `5.26` | Kraichnan universal constant | [Bogucki et al. 1997](https://doi.org/10.1017/S0022112097005727) |

> **Temperature-dependent `kappa_T` (audit 2026-07-01, fixed 2026-07-03).** The
> molecular thermal diffusivity `kappa_T = k/(ρ·c_p)` is temperature-dependent:
> ≈`1.39e-7` at −1 °C, ≈`1.45e-7` at 15 °C, ≈`1.50e-7` at 28 °C, ≈`1.51e-7` at
> 32 °C. It is now evaluated **per window** at the window-mean T/S/P (the same
> T/S/P already used for `ν`) in the L3 stage and threaded through every chi
> integration in `chi.py`/`l4_chi.py`. Because `chi = 6·kappa_T·∫…` and
> `epsilon_T = (2π·kB)⁴·ν·kappa_T²`, this removed a chi bias that ran from
> **≈ −1 % at −1 °C to ≈ +8 % at 32 °C** (epsilon_T twice that in log-space)
> against the old fixed `1.4e-7` — e.g. ARCTERX warm (~28 °C) surface chi rises
> ≈ +7 %. The fixed `KAPPA_T` constant survives only as the fallback default for
> the `chi.py` functions' `kappa_T=` argument. Verified in
> [`tests/test_audit_2026_07_robustness.py::TestChiKappaTemperatureDependence`](../tests/test_audit_2026_07_robustness.py).

### Default instrument parameters

| Symbol | Value | Description |
|--------|-------|-------------|
| `f_s` | `512 Hz` | Fast sampling rate |
| `f_AA` | `98 Hz` | Anti-aliasing filter cutoff (input). Chi integration uses the effective `0.9 * f_AA = 88.2 Hz`; the electronics-noise model uses a separate `110 Hz` Butterworth cutoff |
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
| `fft_length` | `1024` | FFT segment length [samples] |
| `diss_length` | `4096` | Dissipation window length (4 * fft_length) [samples] |
| `overlap` | `2048` | Window overlap (diss_length // 2) [samples] |
| `fp07_model` | `single_pole` | FP07 transfer function model |
| `spectrum_model` | `kraichnan` | Theoretical spectrum (batchelor or kraichnan) |
| `fit_method` | `iterative` | Method 2 fitting algorithm (mle or iterative) |
| `goodman` | `True` | Goodman coherent noise removal using accelerometers |

These are the `rsi-tpw` defaults ([`rsi/chi_io.py`](../src/odas_tpw/rsi/chi_io.py), [`rsi/config.py`](../src/odas_tpw/rsi/config.py)). The `perturb` campaign pipeline uses different defaults — `fft_length = 512` for chi (and 256 for epsilon), see [`perturb/config.py`](../src/odas_tpw/perturb/config.py).


## 12. References

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
- Goto, Y., I. Yasuda, and M. Nagasawa, 2016: [Turbulence estimation using fast-response thermistors attached to a free-fall vertical microstructure profiler.](https://doi.org/10.1175/JTECH-D-15-0220.1) *J. Atmos. Oceanic Technol.*, **33**(10), 2065-2078.
- Vachon, P. and R.G. Lueck, 1984: A small combined temperature-conductivity probe. *Proceedings of the 1984 STD Conference and Workshop*, San Diego, Marine Technology Society.
- Nash, J.D., T.B. Caldwell, M.J. Zelman, and J.N. Moum, 1999: [A thermocouple probe for high-speed temperature measurement in the ocean.](https://doi.org/10.1175/1520-0426(1999)016%3C1474:ATPFHS%3E2.0.CO;2) *J. Atmos. Oceanic Technol.*, **16**, 1474-1482.

### Noise model

- Rockland Scientific, [Technical Note 040](https://rocklandscientific.com/support/technical-notes/): Noise in Temperature Gradient Measurements.

### DNS and empirical validation

- Bogucki, D., J.A. Domaradzki, and P.K. Yeung, 1997: [Direct numerical simulations of passive scalars with Pr > 1 advected by turbulent flow.](https://doi.org/10.1017/S0022112097005727) *J. Fluid Mech.*, **343**, 111-130.

### Software

- Rockland Scientific [ODAS MATLAB Library](https://rocklandscientific.com/support/software/) (v4.5.1).
