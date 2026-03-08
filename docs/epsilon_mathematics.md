# Mathematics of Epsilon Estimation

This document describes the mathematical foundations for computing epsilon, the rate of dissipation of turbulent kinetic energy, as implemented in `rsi-python`. All equation numbers, constants, and algorithmic details correspond to the actual code in [`dissipation.py`](../src/rsi_python/dissipation.py), [`spectral.py`](../src/rsi_python/spectral.py), [`goodman.py`](../src/rsi_python/goodman.py), [`despike.py`](../src/rsi_python/despike.py), [`nasmyth.py`](../src/rsi_python/nasmyth.py), and [`ocean.py`](../src/rsi_python/ocean.py).

## Contents

1. [Definition of Epsilon](#1-definition-of-epsilon)
2. [Nasmyth Universal Shear Spectrum](#2-nasmyth-universal-shear-spectrum)
3. [Spectral Estimation](#3-spectral-estimation)
4. [Shear Probe Despiking](#4-shear-probe-despiking)
5. [Goodman Coherent Noise Removal](#5-goodman-coherent-noise-removal)
6. [Macoun-Lueck Wavenumber Correction](#6-macoun-lueck-wavenumber-correction)
7. [Epsilon Estimation: Variance Method](#7-epsilon-estimation-variance-method)
8. [Epsilon Estimation: Inertial Subrange Method](#8-epsilon-estimation-inertial-subrange-method)
9. [Iterative Variance Correction](#9-iterative-variance-correction)
10. [Seawater Viscosity](#10-seawater-viscosity)
11. [Constants and Parameters](#11-constants-and-parameters)
12. [References](#12-references)

---

## 1. Definition of Epsilon

The rate of dissipation of turbulent kinetic energy (epsilon) quantifies how quickly turbulent motions are converted to heat by viscous friction. Under the assumption of local isotropy, epsilon is related to the one-dimensional shear wavenumber spectrum by ([Oakey 1982](https://doi.org/10.1175/1520-0485(1982)012%3C0256:DOTROD%3E2.0.CO;2)):

```
epsilon = 7.5 * nu * integral_0^inf  Phi_shear(k)  dk       [W/kg]
```

where:

- `Phi_shear(k)` is the one-dimensional shear wavenumber spectrum in `[s^-2 / cpm]`
- `nu` is the kinematic viscosity of seawater `[m^2/s]`
- `k` is the cyclic wavenumber in cycles per metre (cpm)
- The factor 7.5 arises from the isotropy assumption: `epsilon = 15/2 * nu * <(du/dz)^2>`

The shear spectrum is measured by airfoil shear probes sampling at ~512 Hz on a profiling instrument falling at speed `W`. Frequency `f` [Hz] maps to wavenumber via Taylor's frozen-turbulence hypothesis:

```
k = f / W       [cpm]
```

([`dissipation.py: _compute_profile_diss`](../src/rsi_python/dissipation.py))

In practice, the integral is truncated at a finite upper wavenumber `K_max` determined by instrument noise, spectral shape, or the anti-aliasing filter. A correction factor accounts for the unresolved variance beyond `K_max`.


## 2. Nasmyth Universal Shear Spectrum

The Nasmyth (1970) spectrum describes the universal shape of the one-dimensional shear spectrum across inertial and viscous subranges. The implementation uses Lueck's improved empirical fit ([McMillan et al. 2016](https://doi.org/10.1175/JTECH-D-15-0167.1)):

### Non-dimensional form

```
G2(x) = 8.05 * x^(1/3) / (1 + (20.6 * x)^3.715)
```

([`nasmyth.py: _nasmyth_g2`](../src/rsi_python/nasmyth.py))

where `x = k / k_s` is the non-dimensional wavenumber and `k_s` is the Kolmogorov wavenumber:

```
k_s = (epsilon / nu^3)^(1/4)       [cpm]
```

### Dimensional form

The dimensional shear spectrum in `[s^-2 / cpm]` is:

```
Phi_nasmyth(k) = epsilon^(3/4) * nu^(-1/4) * G2(k / k_s)
```

([`nasmyth.py: nasmyth`](../src/rsi_python/nasmyth.py))

### Key properties

- At low wavenumbers (inertial subrange, `x << 1`): `G2 ~ 8.05 * x^(1/3)`, giving the `k^(1/3)` spectral slope
- At high wavenumbers (viscous subrange, `x >> 1`): exponential rolloff as `x^(1/3) / x^3.715`
- The spectrum contains 95% of total variance below `x = X_95 = 0.1205`, i.e., at wavenumber `K_95 = 0.1205 * k_s`

### LUECK_A constant

The relationship between dissipation integrated to 10 cpm (`e_10`) and total dissipation is:

```
e_total = e_10 * sqrt(1 + LUECK_A * e_10)
```

where `LUECK_A = 1.0774e9` is derived from the non-dimensional integral of the Nasmyth spectrum ([Lueck 2022](https://doi.org/10.1175/JTECH-D-21-0051.1)).

([`nasmyth.py`](../src/rsi_python/nasmyth.py), [`dissipation.py: _estimate_epsilon`](../src/rsi_python/dissipation.py))


## 3. Spectral Estimation

Cross-spectral density is computed using Welch's averaged periodogram method with a cosine (Hann) window ([Welch 1967](https://doi.org/10.1109/TAU.1967.1161901)), porting `csd_odas.m` from the ODAS MATLAB library.

### Cosine window

```
w[i] = 1 + cos(pi * (-1 + 2*i/N))     for i = 0, 1, ..., N-1
w = w / sqrt(mean(w^2))
```

([`spectral.py: _cosine_window`](../src/rsi_python/spectral.py))

The RMS normalization ensures that the window preserves spectral power.

### Welch's method

Given a signal of length `L`, FFT length `N_fft`, and overlap `N_ov` (default `N_fft / 2`):

1. **Segment the signal:**
   ```
   step = N_fft - N_ov
   n_seg = floor((L - N_ov) / step)
   ```

2. **For each segment `i`:**
   ```
   s = i * step
   seg = detrend(x[s : s + N_fft]) * w
   X = rfft(seg)
   ```
   Detrending options: none, constant (mean removal), linear, parabolic, or cubic.

3. **Accumulate auto-spectrum:**
   ```
   C_xx += |X|^2
   ```

4. **Normalize:**
   ```
   C_xx = C_xx / (n_seg * N_fft * f_s / 2)
   C_xx[0] /= 2       (DC component)
   C_xx[-1] /= 2       (Nyquist component)
   ```

5. **Frequency vector:**
   ```
   F[j] = j * f_s / N_fft      for j = 0, 1, ..., N_fft/2
   ```

([`spectral.py: csd_odas`](../src/rsi_python/spectral.py))

### Degrees of freedom

For `n_seg` FFT segments with 50% overlap and a Hann window (Nuttall 1971):

```
dof = 1.9 * n_seg
```

This is used for spectral confidence intervals and the Goodman bias correction.

### Multi-channel spectral matrix

For multi-channel signals `x = (x_1, ..., x_p)` and `y = (y_1, ..., y_q)`, the cross-spectral matrix is:

```
C_xy[f, i, j] = <Y_j(f) * conj(X_i(f))>
```

where `< >` denotes the average over FFT segments.

([`spectral.py: csd_matrix`](../src/rsi_python/spectral.py))


## 4. Shear Probe Despiking

Shear signals are contaminated by intermittent spikes from collisions with particles or biological matter. The iterative despiking algorithm (porting `despike.m` from ODAS) detects and replaces these spikes.

### Algorithm

For each iteration (up to `max_passes = 10`):

1. **High-pass filter and rectify:**
   ```
   dv_hp = |filtfilt(butter_hp(1, 0.5 Hz), signal)|
   ```
   A 1st-order Butterworth high-pass filter at 0.5 Hz isolates the high-frequency content where spikes appear. The signal is zero-padded with reflections to reduce edge effects.

2. **Smooth the rectified envelope:**
   ```
   dv_lp = filtfilt(butter_lp(1, f_smooth), dv_hp)
   ```
   A 1st-order Butterworth low-pass filter at `f_smooth` (default 0.5 Hz) produces a slowly-varying envelope estimate.

3. **Detect spikes by threshold exceedance:**
   ```
   spike_indices = {i : dv_hp[i] / dv_lp[i] > thresh}
   ```
   Default threshold `thresh = 8`. Points where the instantaneous high-passed amplitude exceeds 8 times the local envelope are flagged.

4. **Mark affected regions:**
   ```
   For each spike at index s:
     bad_region = [s - N/2, s + 2*N]
   ```
   where `N = round(0.04 * f_s)` (approximately 40 ms at 512 Hz, default ~20 samples). The asymmetric window (shorter before, longer after) accounts for the typical spike shape.

5. **Replace with local mean:**
   ```
   R = round(f_s / (4 * f_smooth))
   replacement = (mean(good_data_before) + mean(good_data_after)) / 2
   ```
   The averaging region spans `R` samples on each side of the bad region, using only non-flagged data points.

6. **Iterate** until no new spikes are detected or `max_passes` reached.

([`despike.py: despike, _single_despike`](../src/rsi_python/despike.py))


## 5. Goodman Coherent Noise Removal

Platform vibrations contaminate shear spectra with energy coherent between shear probes and accelerometers. The [Goodman et al. (2006)](https://doi.org/10.1175/JTECH1889.1) method removes this coherent component, with a bias correction from [RSI Technical Note 61](https://rocklandscientific.com/support/technical-notes/).

### Mathematical basis

The observed shear spectrum contains a signal component and a vibration-coherent noise component:

```
Phi_observed(f) = Phi_signal(f) + Phi_noise(f)
```

The noise component is estimated from the transfer function between accelerometers and shear:

```
Phi_noise(f) = H(f) * Phi_accel(f) * conj(H(f))
```

where `H(f) = C_ua(f) / C_aa(f)` is the frequency-dependent transfer function from accelerometers to shear-coherent vibration.

### Algorithm

1. **Compute the full spectral matrix** from `n_sh` shear channels and `n_ac` accelerometer channels:
   ```
   C_ua(f): (n_sh x n_ac) — shear-accel cross-spectra
   C_uu(f): (n_sh x n_sh) — shear auto/cross-spectra
   C_aa(f): (n_ac x n_ac) — accel auto/cross-spectra
   ```

2. **For each frequency bin, remove the coherent component:**
   ```
   C_uu_clean(f) = C_uu(f) - C_ua(f) * inv(C_aa(f)) * C_ua(f)^H
   ```
   where `^H` denotes the conjugate transpose. If `C_aa(f)` is singular, the original spectrum is retained.

3. **Apply bias correction** ([RSI TN-61](https://rocklandscientific.com/support/technical-notes/)):
   ```
   n_fft_segments = 2 * N / N_fft - 1
   R = 1 / (1 - 1.02 * n_ac / n_fft_segments)
   C_uu_clean *= R
   ```
   This corrects for the degrees-of-freedom lost in estimating the transfer function.

([`goodman.py: clean_shear_spec`](../src/rsi_python/goodman.py))


## 6. Macoun-Lueck Wavenumber Correction

The airfoil shear probe has a finite spatial extent that attenuates the measured shear at low wavenumbers. The [Macoun & Lueck (2004)](https://doi.org/10.1175/1520-0426(2004)021%3C0284:MTSROT%3E2.0.CO;2) correction compensates for this:

```
correction(k) = 1 + (k / 48)^2       for k <= 150 cpm
correction(k) = 1                      for k > 150 cpm
```

The correction is applied multiplicatively to the wavenumber spectrum:

```
Phi_corrected(k) = Phi_observed(k) * correction(k)
```

([`dissipation.py: _compute_profile_diss`](../src/rsi_python/dissipation.py))

The denominator wavenumber 48 cpm corresponds to the probe's half-power point. At `k = 48` cpm, the correction is a factor of 2. The correction is only applied below 150 cpm because the empirical fit is not validated at higher wavenumbers.

### Frequency-to-wavenumber conversion

Before applying the correction, the frequency spectrum is converted to a wavenumber spectrum:

```
Phi(k) = Phi(f) * W
k = f / W
```

where `W` is the profiling speed [m/s]. The factor `W` accounts for the Jacobian of the `f → k` transformation.


## 7. Epsilon Estimation: Variance Method

The variance method integrates the shear spectrum directly. It is used when dissipation is moderate (`epsilon < 1.5e-5 W/kg`), where the spectrum is well-resolved.

### Step 1: Initial estimate from low wavenumbers

Integrate the shear spectrum from the lowest resolved wavenumber to 10 cpm:

```
e_10 = 7.5 * nu * integral_{K_min}^{10} Phi_shear(k) dk
```

This initial estimate uses only the well-resolved, low-wavenumber portion of the spectrum. The total dissipation is then estimated using the LUECK_A correction:

```
e_1 = e_10 * sqrt(1 + LUECK_A * e_10)
```

where `LUECK_A = 1.0774e9` accounts for the variance beyond 10 cpm.

### Step 2: Inertial subrange refinement (optional)

If enough spectral points are available in the inertial subrange (`k < 0.02 * k_s` and at least 20 points), refine the estimate by fitting to the Nasmyth spectrum shape:

```
k_isr = 0.02 * (e_1 / nu^3)^(1/4)
```

If this yields at least 20 wavenumber bins, apply the inertial subrange fitting method (Section 8) to get a refined `e_2`. Otherwise, `e_2 = e_1`.

### Step 3: Determine integration limit

The integration limit `K_max` is determined from the intersection of two criteria:

**a) 95% variance wavenumber:**
```
K_95 = X_95 * (e_2 / nu^3)^(1/4)       X_95 = 0.1205
```

**b) Anti-aliasing filter cutoff:**
```
K_AA = 0.9 * f_AA / W
```

The candidate limit is:
```
K_limit = min(K_AA, K_95)
K_limit = clip(K_limit, 7, 150)       [cpm]
```

### Step 4: Polynomial spectral minimum

To find the wavenumber where the observed spectrum departs from the turbulence signal (due to noise or other contamination), fit a polynomial to the log-log spectrum and find its first minimum:

1. Transform to log-log space:
   ```
   x = log_10(k),  y = log_10(Phi_shear(k))
   ```

2. Fit polynomial of order `fit_order` (default 3, range 3-8):
   ```
   y = p_0 + p_1*x + p_2*x^2 + ... + p_n*x^n
   ```

3. Find roots of the first derivative:
   ```
   dy/dx = p_1 + 2*p_2*x + ... + n*p_n*x^(n-1) = 0
   ```

4. Select the first real root where the second derivative is positive (a minimum) and `k >= 10` cpm.

5. The integration limit is:
   ```
   K_max = min(K_limit, K_polymin)
   ```

If no suitable minimum is found (polynomial fit order is decreased until one is found, down to order 3), `K_max = K_limit`.

([`dissipation.py: _estimate_epsilon`](../src/rsi_python/dissipation.py))

### Step 5: Final integration

```
e_3 = 7.5 * nu * integral_{K_min}^{K_max} Phi_shear(k) dk
```

### Step 6: Variance correction

Apply the iterative variance correction (Section 9) to account for unresolved variance beyond `K_max`:

```
e_4 = _variance_correction(e_3, K_max, nu)
```

### Step 7: Bottom-end correction

Estimate the missing variance at the lowest wavenumbers using the Nasmyth spectrum:

```
Phi_low = nasmyth(e_4, nu, K[1:3])
e_4 += 0.25 * 7.5 * nu * K[1] * Phi_low[0]
```

If this correction exceeds 10% (`e_corrected / e_4 > 1.1`), apply the variance correction again.


## 8. Epsilon Estimation: Inertial Subrange Method

The inertial subrange (ISR) method fits the observed spectrum to the Nasmyth shape in the inertial subrange. It is used when dissipation is high (`epsilon >= 1.5e-5 W/kg`), where the spectral rolloff may not be resolved.

### Fitting range

The fit is restricted to the inertial subrange:

```
k_isr = min(0.02 * (epsilon / nu^3)^(1/4), K_limit)
fit_range = {k : k <= k_isr}
```

The constant 0.02 ensures the fit stays well below the viscous rolloff (`x << 1`).

### Three-pass iterative fitting

Starting from the initial epsilon estimate:

```
For pass = 1, 2, 3:
  Phi_nasmyth = nasmyth(epsilon, nu, K[fit_range])
  ratio = Phi_observed / Phi_nasmyth       (excluding DC)
  fit_error = mean(log_10(ratio))
  epsilon = epsilon * 10^(3/2 * fit_error)
```

The correction factor `10^(3/2 * fit_error)` arises because the Nasmyth spectrum scales as `epsilon^(3/4)`, so a factor-of-10 error in the spectral ratio corresponds to `10^(1/(3/4)) = 10^(4/3)` in epsilon. The factor `3/2` is `1 / (2/3)` reflecting the log-space geometry of the fit.

### Flyer removal

After the initial fit, outlier spectral points ("flyers") are identified and removed:

```
fit_error_vec = log_10(Phi_observed / Phi_nasmyth)
flyers = {i : |fit_error_vec[i]| > 0.5}
```

At most 20% of flagged points are removed from the fit range. Two additional fitting passes are performed on the cleaned spectrum.

### Output

The method returns:
- `epsilon`: fitted dissipation rate
- `K_max`: highest wavenumber in the fit range

([`dissipation.py: _inertial_subrange`](../src/rsi_python/dissipation.py))


## 9. Iterative Variance Correction

The observed shear spectrum is truncated at `K_max`, so some fraction of the total variance is unresolved. The [Lueck (2022)](https://doi.org/10.1175/JTECH-D-21-0051.1) resolved-variance model provides the correction.

### Resolved variance fraction

The fraction of total Nasmyth variance resolved below a non-dimensional wavenumber `x` is:

```
x = K_max * (nu^3 / epsilon)^(1/4)
x_43 = x^(4/3)
V_resolved(x) = tanh(48 * x_43) - 2.9 * x_43 * exp(-22.3 * x_43)
```

This empirical fit ([Lueck 2022](https://doi.org/10.1175/JTECH-D-21-0051.1)) captures the shape of the cumulative Nasmyth spectrum integral. At `x = X_95 = 0.1205`, `V_resolved = 0.95` (95% of variance).

### Iterative correction

Since `V_resolved` depends on epsilon, which in turn depends on the correction, an iterative approach is used:

```
e_new = e_3       (initial uncorrected estimate)

For iteration = 1, 2, ..., max_iter (default 50):
  x = K_max * (nu^3 / e_new)^(1/4)
  x_43 = x^(4/3)
  V = tanh(48 * x_43) - 2.9 * x_43 * exp(-22.3 * x_43)

  if V <= 0: break

  e_old = e_new
  e_new = e_3 / V

  if e_new / e_old < 1.02: break       (converged)
```

The iteration typically converges in 3-5 steps. The correction is larger when `K_max` is low relative to the Kolmogorov wavenumber.

([`dissipation.py: _variance_correction`](../src/rsi_python/dissipation.py))


## 10. Seawater Viscosity

Kinematic viscosity at salinity S = 35 PSU and atmospheric pressure is computed from a 3rd-order polynomial fit (porting `visc35.m` from ODAS):

```
nu(T) = p_0 + p_1*T + p_2*T^2 + p_3*T^3       [m^2/s]
```

where T is temperature in degrees Celsius and:

```
p_0 =  1.828297985908266e-06
p_1 = -5.864346822839289e-08
p_2 =  1.199552027472192e-09
p_3 = -1.131311019739306e-11
```

([`ocean.py: visc35`](../src/rsi_python/ocean.py))

Valid for 0 <= T <= 20 degrees C. Typical values: ~1.3e-6 m^2/s at 10 degrees C, ~1.0e-6 m^2/s at 20 degrees C.


## 11. Constants and Parameters

### Physical constants

| Constant | Value | Description | Source |
|----------|-------|-------------|--------|
| `LUECK_A` | `1.0774e9` | e/e_10 model constant | [Lueck 2022](https://doi.org/10.1175/JTECH-D-21-0051.1) |
| `X_95` | `0.1205` | Non-dimensional wavenumber for 95% variance | [Lueck 2022](https://doi.org/10.1175/JTECH-D-21-0051.1) |
| Isotropy factor | `7.5` | `15/2 * nu` relates shear variance to epsilon | [Oakey 1982](https://doi.org/10.1175/1520-0485(1982)012%3C0256:DOTROD%3E2.0.CO;2) |

### Default processing parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fft_length` | 256 | FFT segment length [samples] |
| `diss_length` | `2 * fft_length` | Dissipation window length [samples] |
| `overlap` | `diss_length / 2` | Window overlap [samples] |
| `f_AA` | 98.0 | Anti-aliasing filter cutoff [Hz] |
| `fit_order` | 3 | Polynomial order for spectral minimum |
| `despike_thresh` | 8 | Spike detection threshold (ratio) |
| `despike_smooth` | 0.5 | Despike envelope smoothing frequency [Hz] |
| `e_isr_threshold` | `1.5e-5` | Epsilon threshold for ISR method [W/kg] |

### Macoun-Lueck correction

| Parameter | Value | Description |
|-----------|-------|-------------|
| Half-power wavenumber | 48 cpm | Probe spatial response rolloff |
| Maximum correction wavenumber | 150 cpm | Upper limit of empirical fit validity |

### Goodman bias correction

| Parameter | Value | Description |
|-----------|-------|-------------|
| Bias factor | `1.02 * n_accel / n_segments` | Degrees-of-freedom correction ([TN-61](https://rocklandscientific.com/support/technical-notes/)) |


## 12. References

### Epsilon estimation

- Lueck, R.G., 2022: [The statistics of oceanic turbulence measurements. Part 1: Shear variance and dissipation rates.](https://doi.org/10.1175/JTECH-D-21-0051.1) *J. Atmos. Oceanic Technol.*, 39, 1259-1276.
- McMillan, J.M., A.E. Hay, R.G. Lueck, and F. Wolk, 2016: [Rates of dissipation of turbulent kinetic energy in a high Reynolds number tidal channel.](https://doi.org/10.1175/JTECH-D-15-0167.1) *J. Atmos. Oceanic Technol.*, 33, 817-837.
- Oakey, N.S., 1982: [Determination of the rate of dissipation of turbulent energy from simultaneous temperature and velocity shear microstructure measurements.](https://doi.org/10.1175/1520-0485(1982)012%3C0256:DOTROD%3E2.0.CO;2) *J. Phys. Oceanogr.*, 12, 256-271.

### Spectral methods

- Welch, P.D., 1967: [The use of fast Fourier transform for the estimation of power spectra.](https://doi.org/10.1109/TAU.1967.1161901) *IEEE Trans. Audio Electroacoust.*, AU-15, 70-73.
- Nuttall, A.H., 1971: Spectral estimation by means of overlapped fast Fourier transform processing of windowed data. Naval Underwater Systems Center Report No. 4169.

### Coherent noise removal

- Goodman, L., E.R. Levine, and R.G. Lueck, 2006: [On measuring the terms of the turbulent kinetic energy budget from an AUV.](https://doi.org/10.1175/JTECH1889.1) *J. Atmos. Oceanic Technol.*, 23, 977-990.
- Rockland Scientific, [Technical Note 061](https://rocklandscientific.com/support/technical-notes/): Bias correction for Goodman coherent noise removal.

### Shear probe response

- Macoun, P. and R.G. Lueck, 2004: [Modeling the spatial response of the airfoil shear probe using different sized probes.](https://doi.org/10.1175/1520-0426(2004)021%3C0284:MTSROT%3E2.0.CO;2) *J. Atmos. Oceanic Technol.*, 21, 284-297.

### Nasmyth spectrum

- Nasmyth, P.W., 1970: Oceanic turbulence. Ph.D. dissertation, University of British Columbia, 69 pp.

### Rockland Scientific Technical Notes

- [**TN-028:**](https://rocklandscientific.com/support/technical-notes/) Dissipation rate estimation
- [**TN-051:**](https://rocklandscientific.com/support/technical-notes/) P file format
- [**TN-061:**](https://rocklandscientific.com/support/technical-notes/) Goodman noise removal bias correction
