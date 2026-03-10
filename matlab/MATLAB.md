# MATLAB Chi Calculation

MATLAB functions for computing chi (thermal variance dissipation rate) from Rockland Scientific microprofiler data. Implements both Method 1 (chi from known epsilon) and Method 2 (MLE Batchelor spectrum fitting without epsilon).

These are standalone implementations that mirror the algorithms in the Python `rsi_python.chi` module.

## Requirements

- MATLAB R2021a or newer (uses `arguments` blocks and `name=value` syntax)
- Signal Processing Toolbox (for `pwelch`, `hanning`)
- ODAS MATLAB Library on the path (for `odas_p2mat`, `make_gradT_odas`, `get_diss_odas`)

## Quick Start — Full Workflow from a .p File

```matlab
%% 1. Convert .p file to MATLAB workspace
addpath('/path/to/odas');    % ODAS library
addpath('/path/to/matlab');  % this directory

fname = 'VMP/ARCTERX_Thompson_2025_SN479_0006.p';
d = odas_p2mat(fname);

%% 2. Detect profiles
W = gradient(d.P_slow, 1/d.fs_slow);   % dP/dt from slow pressure
profiles = get_profile(d.P_slow, W, 0.5, 0.3, 'down', 7, d.fs_slow);
fprintf('Found %d profiles\n', size(profiles, 2));

%% 3. Extract first profile
ratio = round(d.fs_fast / d.fs_slow);
s_slow = profiles(1,1);
e_slow = profiles(2,1);
s_fast = (s_slow - 1) * ratio + 1;
e_fast = e_slow * ratio;

%% 4. Compute temperature gradient using ODAS
obj = setupstr(d.setupfilestr);
gradT1 = make_gradT_odas(d.T1_dT1(s_fast:e_fast), ...
    'obj', obj, 'fs', d.fs_fast, 'speed', d.speed_fast(s_fast:e_fast), ...
    'name_with_pre_emphasis', 'T1_dT1', ...
    'name_without_pre_emphasis', 'T1');

%% 5. Compute chi — Method 2 (no epsilon needed)
results = get_chi(gradT1, d.P(s_fast:e_fast), ...
    d.T1(s_slow:e_slow), d.speed_fast(s_fast:e_fast), d.fs_fast);

%% 6. Plot results
figure;
semilogy(results.P_mean, results.chi, 'b-', LineWidth=1.5);
xlabel('Pressure [dbar]');
ylabel('\chi [K^2/s]');
title('Thermal Dissipation Rate — Method 2');
grid on;
set(gca, YDir="reverse");
```

### Method 1 (with epsilon from shear probes)

```matlab
%% Compute epsilon first using ODAS
diss = get_diss_odas(d.sh1(s_fast:e_fast), ...
    [d.Ax(s_fast:e_fast), d.Ay(s_fast:e_fast)], ...
    struct('fft_length', 256, 'diss_length', 768, ...
           'overlap', 384, 'fs_fast', d.fs_fast, ...
           'fs_slow', d.fs_slow, 'speed', d.speed_fast(s_fast:e_fast), ...
           'T', d.T1(s_fast:e_fast), 'P', d.P(s_fast:e_fast), ...
           'goodman', true));

%% Compute chi using epsilon — Method 1
results_m1 = get_chi(gradT1, d.P(s_fast:e_fast), ...
    d.T1(s_slow:e_slow), d.speed_fast(s_fast:e_fast), d.fs_fast, ...
    method=1, epsilon=diss.e);
```

## Function Reference

### `get_chi` — Main chi computation

```matlab
results = get_chi(gradT, P_fast, T_slow, speed, fs_fast, Name=Value)
```

**Inputs:**
| Argument | Type | Description |
|----------|------|-------------|
| `gradT` | matrix | Temperature gradient [K/m], one column per probe |
| `P_fast` | vector | Pressure at fast rate [dbar] |
| `T_slow` | vector | Temperature at slow rate [deg C] |
| `speed` | scalar/vector | Profiling speed [m/s] |
| `fs_fast` | scalar | Fast sampling rate [Hz] |

**Name-Value Options:**
| Name | Default | Description |
|------|---------|-------------|
| `method` | 2 | 1 = from epsilon, 2 = MLE fit |
| `epsilon` | [] | Epsilon per window [W/kg], required for method 1 |
| `fft_length` | 512 | FFT segment length [samples] |
| `diss_length` | 3×fft_length | Analysis window [samples] |
| `overlap` | diss_length/2 | Window overlap [samples] |
| `f_AA` | 98 | Anti-aliasing frequency [Hz] |
| `diff_gain` | 0.94 | Differentiator gain [s] |
| `spectrum_model` | "kraichnan" | "batchelor" or "kraichnan" |
| `fp07_model` | "single_pole" | "single_pole" or "double_pole" |
| `salinity` | 35 | Salinity [PSU] |

**Output struct fields:**
| Field | Size | Description |
|-------|------|-------------|
| `chi` | n_probes × n_est | Thermal dissipation rate [K²/s] |
| `epsilon_T` | n_probes × n_est | Epsilon from temperature [W/kg] |
| `kB` | n_probes × n_est | Batchelor wavenumber [cpm] |
| `K_max` | n_probes × n_est | Max integration wavenumber [cpm] |
| `fom` | n_probes × n_est | Figure of merit |
| `K_max_ratio` | n_probes × n_est | K_max / kB |
| `P_mean` | 1 × n_est | Mean pressure [dbar] |
| `T_mean` | 1 × n_est | Mean temperature [°C] |
| `spec_gradT` | n_freq × n_probes × n_est | Observed spectra |
| `spec_batch` | n_freq × n_probes × n_est | Fitted Batchelor spectra |

---

### `chi_method1` — Chi from known epsilon

```matlab
result = chi_method1(spec_obs, K, epsilon, nu, speed, Name=Value)
```

Computes chi for a single window given the observed spectrum and epsilon from shear probes. Uses the Batchelor spectrum to correct for FP07 rolloff and unresolved variance.

---

### `chi_method2` — Chi via MLE spectral fitting

```matlab
result = chi_method2(spec_obs, K, nu, speed, Name=Value)
```

Estimates kB by maximum likelihood grid search, then recovers chi and epsilon. Uses the iterative refinement approach of Peterson & Fer (2014).

---

### Supporting Functions

| Function | Description |
|----------|-------------|
| `batchelor_gradT(k, kB, chi)` | Batchelor temperature gradient spectrum |
| `kraichnan_gradT(k, kB, chi)` | Kraichnan temperature gradient spectrum |
| `batchelor_wavenumber(epsilon, nu)` | Batchelor wavenumber from epsilon |
| `fp07_time_constant(speed)` | Speed-dependent FP07 time constant |
| `fp07_transfer(f, tau0)` | FP07 squared transfer function \|H(f)\|² |

## Methods

### Method 1: Chi from Epsilon (Dillon & Caldwell 1980)

Given epsilon from shear probes:
1. Compute kB from epsilon: `kB = (1/2π)(ε/(ν·κ_T²))^{1/4}`
2. Integrate observed gradient spectrum up to noise floor
3. Correct for FP07 rolloff and unresolved high-wavenumber variance
4. `χ = 6·κ_T · ∫S_obs(k)dk · V_total/V_resolved`

### Method 2: MLE Batchelor Fitting (Ruddick et al. 2000)

Without epsilon:
1. Estimate initial chi from spectrum minus noise
2. Grid search over kB, minimizing `NLL = Σ[log(S_model) + S_obs/S_model]`
3. Iteratively refine integration limits (Peterson & Fer 2014)
4. Recover epsilon from fitted kB: `ε = (2π·kB)⁴·ν·κ_T²`

## References

- Dillon & Caldwell, 1980: The Batchelor spectrum and dissipation in the upper ocean. *J. Geophys. Res.*, 85, 1910-1916.
- Oakey, 1982: Determination of the rate of dissipation of turbulent energy. *J. Phys. Oceanogr.*, 12, 256-271.
- Bogucki, Domaradzki & Yeung, 1997: DNS of passive scalars with Pr>1. *J. Fluid Mech.*, 343, 111-130.
- Ruddick, Anis & Thompson, 2000: Maximum likelihood spectral fitting. *J. Atmos. Oceanic Technol.*, 17, 1541-1555.
- Peterson & Fer, 2014: Dissipation measurements using temperature microstructure from an underwater glider. *Methods in Oceanography*, 10, 44-69.
- RSI Technical Note 040: Noise in Temperature Gradient Measurements.
