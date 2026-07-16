# Configuration Reference

Processing parameters can be set via a YAML configuration file, CLI flags, or both. The three-way merge order is:

```
defaults ← config file ← CLI flags
```

CLI flags always take precedence over the config file, which takes precedence over built-in defaults.

## Quick Start

Generate a template configuration file with all defaults:

```bash
rsi-tpw init                        # writes config.yaml in current directory
rsi-tpw init my_settings.yaml       # custom filename
rsi-tpw init --force config.yaml    # overwrite existing file
```

Use a configuration file:

```bash
rsi-tpw eps VMP/*.p -c config.yaml -o results/
rsi-tpw eps VMP/*.p -c config.yaml -o results/ --fft-length 512  # CLI overrides config
```

## Configuration Sections

The configuration file has three sections: `profiles`, `epsilon`, and `chi`. Each parameter is optional — unset values fall back to defaults.

---

### `profiles` — Profile Detection

Controls how profiling segments are identified from pressure data.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `P_min` | float | `0.5` | Minimum pressure threshold [dbar]. Data below this pressure is excluded from profiles. |
| `W_min` | float | `0.3` | Minimum fall rate [dbar/s]. Segments with fall rate below this are not considered part of a profile. |
| `direction` | string | `"auto"` | Profile direction: `"auto"`, `"up"`, `"down"`, `"glide"`, or `"horizontal"`. `"auto"` infers from the vehicle. |
| `vehicle` | string | `null` | Vehicle type override (e.g. `"slocum_glider"`, `"vmp"`). If `null`, the vehicle is read from the `.p` file. |
| `min_duration` | float | `7.0` | Minimum profile duration [seconds]. Profiles shorter than this are discarded. |

---

### `epsilon` — TKE Dissipation Rate

Controls computation of the turbulent kinetic energy dissipation rate (epsilon) from shear probe spectra.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fft_length` | int | `1024` | FFT segment length in samples. At 512 Hz sampling, 1024 samples = 2.0 s. Determines the spectral resolution (df = fs / fft_length). |
| `diss_length` | int | `null` | Dissipation estimate window length in samples. Each epsilon estimate uses this many samples. Default: `4 * fft_length`. |
| `overlap` | int | `null` | Overlap between successive dissipation windows in samples. Default: `diss_length // 2` (50% overlap). |
| `speed` | float | `null` | Fixed profiling speed [m/s]. If `null`, speed is computed from dP/dt (pressure rate of change). Use a fixed value when pressure data is unreliable. |
| `direction` | string | `"auto"` | Profile direction: `"auto"`, `"up"`, `"down"`, `"glide"`, or `"horizontal"`. |
| `vehicle` | string | `null` | Vehicle type override (e.g. `"slocum_glider"`, `"vmp"`). If `null`, the vehicle is read from the `.p` file. |
| `goodman` | bool | `true` | Enable Goodman coherent noise removal using accelerometer cross-spectra. Removes vibration-coherent contamination from shear spectra. Disable with `--no-goodman` on the CLI. |
| `f_AA` | float | `98.0` | Anti-aliasing filter cutoff frequency [Hz]. Spectral estimates above this frequency are excluded from variance integration. |
| `f_limit` | float | `null` | Upper frequency limit [Hz] for spectral integration. If `null`, uses `f_AA`. Set lower to exclude high-frequency noise. |
| `fit_order` | int | `3` | Polynomial order for the log-log fit used to determine the upper integration limit in the Nasmyth spectrum fitting. |
| `despike_thresh` | float | `8` | Spike detection threshold. Ratio of the instantaneous rectified high-pass signal to its smoothed envelope. Higher values are less aggressive. |
| `despike_smooth` | float | `0.5` | Low-pass cutoff frequency [Hz] for the spike detection envelope smoother. |
| `salinity` | float \| `"measured"` | `null` | Salinity for computing kinematic viscosity via TEOS-10 (gsw): a fixed PSU value, or `"measured"` = per-sample practical salinity from the resolved conductivity/temperature channels and pressure (falls back to fixed S with a warning when no conductivity exists). If `null`, uses a simplified viscosity formula assuming S = 35. |
| `temperature` | string \| float | `"auto"` | Reference-temperature source for viscosity and `T_mean`. `"auto"` = first plausible of `T1`..`Tn`, `T`, `JAC_T` — implausible channels (railed, drifting, mostly non-finite) are skipped with a warning, and QC evaluates in-water samples (P > 0.5 dbar) when pressure is available; a channel name is honored even when it fails QC (with a warning); a number = constant reference temperature [°C] (ODAS `constant_temp` parity). Recorded in the products as `temperature_source`/`temperature_qc`. (ODAS divergence, deliberate: ODAS uses T1 with no plausibility check and silently substitutes 10 °C when temperature is missing; here a file with no plausible reference temperature errors instead of publishing wrong-viscosity products.) |
| `conductivity` | string | `"auto"` | Conductivity channel behind `salinity: "measured"`. `"auto"` = `JAC_C` when present (no error otherwise); an explicit name must exist as a slow channel. Recorded as `conductivity_source`. |

---

### `chi` — Thermal Variance Dissipation Rate

Controls computation of the thermal variance dissipation rate (chi) from FP07 thermistor spectra.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fft_length` | int | `1024` | FFT segment length in samples. At 512 Hz, 1024 samples = 2.0 s. |
| `diss_length` | int | `null` | Dissipation estimate window length in samples. Default: `4 * fft_length`. |
| `overlap` | int | `null` | Overlap between successive dissipation windows in samples. Default: `diss_length // 2`. |
| `speed` | float | `null` | Fixed profiling speed [m/s]. If `null`, computed from dP/dt. |
| `direction` | string | `"auto"` | Profile direction: `"auto"`, `"up"`, `"down"`, `"glide"`, or `"horizontal"`. |
| `vehicle` | string | `null` | Vehicle type override (e.g. `"slocum_glider"`, `"vmp"`). If `null`, the vehicle is read from the `.p` file. |
| `fp07_model` | string | `"single_pole"` | FP07 thermistor transfer function model. `"single_pole"`: Lueck et al. 1977. `"double_pole"`: Gregg & Meagher 1980 (accounts for thermal boundary layer). |
| `goodman` | bool | `true` | Enable Goodman coherent noise removal for temperature spectra. Disable with `--no-goodman` on the CLI. |
| `f_AA` | float | `98.0` | Anti-aliasing filter cutoff frequency [Hz]. |
| `fit_method` | string | `"iterative"` | Method 2 spectral fitting algorithm. `"iterative"`: Iterative integration (Peterson & Fer 2014). `"mle"`: Maximum Likelihood Estimation (Ruddick et al. 2000). Only used when epsilon is not provided. |
| `spectrum_model` | string | `"kraichnan"` | Theoretical temperature gradient spectrum model. `"batchelor"`: Gaussian rolloff (Batchelor 1959). `"kraichnan"`: Exponential rolloff (Bogucki et al. 1997). |
| `salinity` | float \| `"measured"` | `null` | Salinity for computing kinematic viscosity: a fixed PSU value, or `"measured"` (see the `epsilon` section). If `null`, uses simplified S = 35 formula. |
| `temperature` | string \| float | `"auto"` | Reference-temperature source for viscosity/κ_T and `T_mean` (same semantics as the `epsilon` section). |
| `conductivity` | string | `"auto"` | Conductivity channel behind `salinity: "measured"` (same semantics as the `epsilon` section). |

---

## Example Configuration File

```yaml
# rsi-tpw configuration
# Values shown are the defaults. Uncomment and edit to customize.
# CLI flags override values in this file.

profiles:
  P_min: 0.5            # minimum pressure [dbar]
  W_min: 0.3            # minimum fall rate [dbar/s]
  direction: auto       # profile direction: auto, up, down, glide, horizontal
  vehicle: null         # vehicle override (null = from .p file)
  min_duration: 7.0     # minimum profile duration [s]

epsilon:
  fft_length: 1024      # FFT segment length [samples]
  diss_length: null     # dissipation window [samples] (null = 4 * fft_length)
  overlap: null         # window overlap [samples] (null = diss_length // 2)
  speed: null           # profiling speed [m/s] (null = from dP/dt)
  direction: auto       # profile direction: auto, up, down, glide, horizontal
  vehicle: null         # vehicle override (null = from .p file)
  goodman: true         # Goodman coherent noise removal
  f_AA: 98.0            # anti-aliasing filter cutoff [Hz]
  f_limit: null         # upper frequency limit [Hz] (null = f_AA)
  fit_order: 3          # polynomial fit order for Nasmyth integration
  despike_thresh: 8     # despike threshold: ratio of rectified HP signal to its LP envelope
  despike_smooth: 0.5   # despike envelope low-pass cutoff [Hz]
  salinity: null        # PSU value, or "measured" (JAC/selected C+T) (null = 35, fixed S)
  temperature: auto     # reference temperature for viscosity: auto = first
                        # plausible of T1..Tn, T, JAC_T; a channel name (QC
                        # failure warns but proceeds); or a number = constant
                        # reference temperature [degC]
  conductivity: auto    # conductivity channel for salinity "measured"
                        # (auto = JAC_C when present)

chi:
  fft_length: 1024      # FFT segment length [samples]
  diss_length: null     # dissipation window [samples] (null = 4 * fft_length)
  overlap: null         # window overlap [samples] (null = diss_length // 2)
  speed: null           # profiling speed [m/s] (null = from dP/dt)
  direction: auto       # profile direction: auto, up, down, glide, horizontal
  vehicle: null         # vehicle override (null = from .p file)
  fp07_model: single_pole  # FP07 transfer function: single_pole or double_pole
  goodman: true         # Goodman coherent noise removal
  f_AA: 98.0            # anti-aliasing filter cutoff [Hz]
  fit_method: iterative # Method 2 fitting: iterative or mle
  spectrum_model: kraichnan  # theoretical spectrum: batchelor or kraichnan
  salinity: null        # PSU value, or "measured" (JAC/selected C+T) (null = 35, fixed S)
  temperature: auto     # reference temperature for viscosity/kappa_T: auto =
                        # first plausible of T1..Tn, T, JAC_T; a channel name
                        # (QC failure warns but proceeds); or a number =
                        # constant reference temperature [degC]
  conductivity: auto    # conductivity channel for salinity "measured"
                        # (auto = JAC_C when present)
```

## `null` Values

In YAML, `null` means "not set." When a parameter is `null`:
- **`diss_length`**: Computed as `4 * fft_length` for both epsilon and chi.
- **`overlap`**: Computed as `diss_length // 2`.
- **`speed`**: Derived from the pressure time series (dP/dt).
- **`f_limit`**: Uses `f_AA` as the upper frequency limit.
- **`salinity`**: Uses a simplified viscosity formula assuming S = 35 PSU (visc35). Set an explicit salinity — or `"measured"` to derive it per sample from the conductivity/temperature channels — to use the full TEOS-10 equation of state via gsw.

## Chi Computation Methods

The chi command supports two methods, selected automatically based on whether epsilon data is provided:

**Method 1** (with `--epsilon-dir`): Uses shear-probe epsilon to compute the Batchelor wavenumber kB, then integrates the corrected temperature gradient spectrum up to kB to obtain chi. This is the preferred method when shear probe data is available.

> **Note:** `rsi-tpw eps -o epsilon/` writes into a hash-tracked
> subdirectory (`epsilon/eps_00/`); `--epsilon-dir` searches the given
> directory and its `eps_*` subdirectories (most recently modified first),
> matching both `{stem}_eps.nc` and per-profile `{stem}_prof001_eps.nc`
> names (concatenated along time). Only when no matching epsilon files
> exist does chi fall back to Method 2 for that file, with a console
> warning.

**Method 2** (without `--epsilon-dir`): Fits a theoretical spectrum (Batchelor or Kraichnan) to the observed temperature gradient spectrum to simultaneously estimate kB and chi. Two fitting algorithms are available:
- `iterative` (default): Iterative integration with progressive refinement of integration limits (Peterson & Fer 2014).
- `mle`: Maximum Likelihood Estimation grid search over kB (Ruddick et al. 2000).

## Output Directory Management

Output directories use a sequential, hash-tracked naming scheme. See [output_directories.md](output_directories.md) for details.
