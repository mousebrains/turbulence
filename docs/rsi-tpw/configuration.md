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
| `W_min` | float | `null` | Minimum fall rate [dbar/s]. Segments with fall rate below this are not considered part of a profile. `null` = auto: 0.3 for a free-falling profiler, 0.05 for glide/horizontal platforms (a glider's vertical rate never reaches the VMP-tuned 0.3). |
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
| `speed` | float | `null` | Fixed profiling speed [m/s]. If `null`, speed follows `speed_method`. Use a fixed value when pressure data is unreliable. Mutually exclusive with `speed_method: em`/`flight`. |
| `speed_method` | string | `null` | Through-water speed model: `null`/`"pressure"` = \|dP/dt\| (the historical default; correct for a free-falling profiler, biased for gliders); `"em"` = the `U_EM` flowmeter channel; `"flight"` = the inviscid glider flight model \|W\|/sin(\|pitch\|−aoa) from the inclinometers. An explicit `"pressure"` forces \|dP/dt\| even when the source carries a precomputed `speed_fast` channel; `null` prefers that channel. The resolved provenance is recorded in the products as `speed_source`. (The `hotel` speed method is perturb-only — its result arrives here as the precomputed `speed_fast` channel.) |
| `aoa_deg` | float | `3.0` | Angle of attack [deg] for `speed_method: flight`. |
| `direction` | string | `"auto"` | Profile direction: `"auto"`, `"up"`, `"down"`, `"glide"`, or `"horizontal"`. |
| `vehicle` | string | `null` | Vehicle type override (e.g. `"slocum_glider"`, `"vmp"`). If `null`, the vehicle is read from the `.p` file. |
| `W_min` | float | `null` | Profile-detection fall-rate floor [dbar/s]. `null` = auto (0.3 free-fall, 0.05 glide/horizontal). |
| `goodman` | bool | `true` | Enable Goodman coherent noise removal using accelerometer cross-spectra. Removes vibration-coherent contamination from shear spectra. Disable with `--no-goodman` on the CLI. |
| `f_AA` | float | `98.0` | Anti-aliasing filter cutoff frequency [Hz]. Spectral estimates above this frequency are excluded from variance integration. |
| `f_limit` | float | `null` | Upper frequency limit [Hz] for spectral integration. If `null`, uses `f_AA`. Set lower to exclude high-frequency noise. |
| `fit_order` | int | `3` | Polynomial order for the log-log fit used to determine the upper integration limit in the Nasmyth spectrum fitting. |
| `despike_thresh` | float | `8` | Spike detection threshold. Ratio of the instantaneous rectified high-pass signal to its smoothed envelope. Higher values are less aggressive. |
| `despike_smooth` | float | `0.5` | Low-pass cutoff frequency [Hz] for the spike detection envelope smoother. |
| `despike_half_width` | float | `0.04` | Shear spike removal half-width [s], in (0, 0.5]. The despiker replaces half of it before each detected spike and all of it after (as ODAS `despike.m`), so `0.04` (the ODAS default) replaces ~60 ms per spike (31 samples at 512 Hz) and `0.01` ~14 ms (7 samples). A narrow electrical pulse needs less than a probe strike: for a ~16 ms pulse injected at 20× the record rms into clean shear, `0.04` biased epsilon by −0.05 dex, `0.02` by −0.01 to −0.02 and `0.01` by +0.02 to +0.05 (median over windows; two pulse shapes and host files). No half-width helps a pulse below `despike_thresh`, which is never detected (see `record_locked_shear`). The accelerometers are not despiked. |
| `record_locked_shear` | bool | `false` | Subtract shear interference that sits at fixed sample positions in every `.p` data record (electrical pickup from the instrument's own data system), which Goodman cannot remove when the accelerometers are clean and the despiker misses below `despike_thresh`. Per profile and shear channel: the mean of the high-passed shear over the profile's quiet complete records (rms ≤ 3× the median, at least 60) decides **where** -- the samples departing from its median by ≥ 5 standard errors, grown through ≥ 2 and dilated 2 samples -- and each record's leave-one-out mean over the 16 records on either side decides **what** is subtracted there, because the interference drifts within a cast (several-fold, on the instrument it was built for); this repeats, up to 3 passes, while anything stays significant. Every other sample is left bit-identical; on clean shear nothing is corrected. A sample's place in its record comes from its index and the header's samples per record, never from time. Needs a `.p` file or a per-profile NetCDF written by `rsi-tpw prof` (which records the framing, as `rsi-tpw nc` files now do too); other sources raise. Each product records `record_locked_<channel>_records/_samples/_peak/_snr/_peak_range/_phase_power_ratio/_phase_power_ratio_loud` and `record_locked_comment`. What it cannot remove: a part that varies in shape from record to record (the phase power ratios, above 1, measure it -- those windows stay biased high), and a slow tail below significance at every sample, which still shows at the record's ~1 Hz fundamental. Every estimate is per profile, so the corrected samples follow the profile bounds (one cast, sh1/sh2: 365/80 samples per record here, 258/120 on perturb's top-trimmed profile). `rsi-tpw pipeline` refuses it. See `odas_tpw.rsi.record_locked`. |
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
| `speed` | float | `null` | Fixed profiling speed [m/s]. If `null`, speed follows `speed_method`. Mutually exclusive with `speed_method: em`/`flight`. |
| `speed_method` | string | `null` | Through-water speed model (same semantics as the `epsilon` section). |
| `aoa_deg` | float | `3.0` | Angle of attack [deg] for `speed_method: flight`. |
| `direction` | string | `"auto"` | Profile direction: `"auto"`, `"up"`, `"down"`, `"glide"`, or `"horizontal"`. |
| `vehicle` | string | `null` | Vehicle type override (e.g. `"slocum_glider"`, `"vmp"`). If `null`, the vehicle is read from the `.p` file. |
| `W_min` | float | `null` | Profile-detection fall-rate floor [dbar/s]. `null` = auto (0.3 free-fall, 0.05 glide/horizontal). |
| `fp07_model` | string | `"single_pole"` | FP07 thermistor transfer function model. `"single_pole"`: Lueck et al. 1977. `"double_pole"`: Gregg & Meagher 1980 (accounts for thermal boundary layer). |
| `goodman` | bool | `true` | Enable Goodman coherent noise removal for temperature spectra. Disable with `--no-goodman` on the CLI. |
| `f_AA` | float | `98.0` | Anti-aliasing filter cutoff frequency [Hz]. |
| `fit_method` | string | `"iterative"` | Method 2 spectral fitting algorithm. `"iterative"`: Iterative integration (Peterson & Fer 2014). `"mle"`: Maximum Likelihood Estimation (Ruddick et al. 2000). Only used when epsilon is not provided. |
| `spectrum_model` | string | `"kraichnan"` | Theoretical temperature gradient spectrum model. `"batchelor"`: Gaussian rolloff (Batchelor 1959). `"kraichnan"`: Exponential rolloff (Bogucki et al. 1997). |
| `first_bin` | int | `2` | Lowest FFT bin any chi estimate uses (DC = 0; must be >= 1). The per-segment linear detrend leaves bin 1 off the spectrum at its own wavenumber, ~30% low on the red spectra real data carry (issue #191; [chi_mathematics.md](../chi_mathematics.md#lowest-usable-bin-first_bin)). `1` reproduces a chi product made before bin 1 was dropped and keeps that run's output-directory hash; any other value is hashed and recorded in the chi file's `first_bin` attribute. Also honored by `rsi-tpw pipeline`. |
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
  W_min: null           # minimum fall rate [dbar/s] (null = auto: 0.3
                        # free-fall, 0.05 glide/horizontal)
  direction: auto       # profile direction: auto, up, down, glide, horizontal
  vehicle: null         # vehicle override (null = from .p file)
  min_duration: 7.0     # minimum profile duration [s]

epsilon:
  fft_length: 1024      # FFT segment length [samples]
  diss_length: null     # dissipation window [samples] (null = 4 * fft_length)
  overlap: null         # window overlap [samples] (null = diss_length // 2)
  speed: null           # fixed profiling speed [m/s] (null = speed_method)
  speed_method: null    # through-water speed model: null = pressure |dP/dt|;
                        # em (U_EM flowmeter); flight (|W|/sin(|pitch|-aoa))
  aoa_deg: 3.0          # angle of attack [deg] for speed_method: flight
  direction: auto       # profile direction: auto, up, down, glide, horizontal
  vehicle: null         # vehicle override (null = from .p file)
  W_min: null           # minimum fall rate [dbar/s] (null = auto: 0.3
                        # free-fall, 0.05 glide/horizontal)
  goodman: true         # Goodman coherent noise removal
  f_AA: 98.0            # anti-aliasing filter cutoff [Hz]
  f_limit: null         # upper frequency limit [Hz] (null = f_AA)
  fit_order: 3          # polynomial fit order for Nasmyth integration
  despike_thresh: 8     # despike threshold: ratio of rectified HP signal to its LP envelope
  despike_smooth: 0.5   # despike envelope low-pass cutoff [Hz]
  despike_half_width: 0.04  # despike removal half-width [s]: blanks half of it
                        # before each spike and all of it after (0.04 = ~60 ms
                        # per spike, 0.01 = ~14 ms); in (0, 0.5]
  record_locked_shear: false  # subtract shear interference that sits at fixed
                        # positions in every .p data record (an electrical
                        # pulse the despiker misses below its threshold);
                        # corrects only samples where the per-profile record
                        # mean is significant. Needs a .p file or a profile
                        # NetCDF from rsi-tpw prof
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
  speed: null           # fixed profiling speed [m/s] (null = speed_method)
  speed_method: null    # through-water speed model: null = pressure |dP/dt|;
                        # em (U_EM flowmeter); flight (|W|/sin(|pitch|-aoa))
  aoa_deg: 3.0          # angle of attack [deg] for speed_method: flight
  direction: auto       # profile direction: auto, up, down, glide, horizontal
  vehicle: null         # vehicle override (null = from .p file)
  W_min: null           # minimum fall rate [dbar/s] (null = auto: 0.3
                        # free-fall, 0.05 glide/horizontal)
  fp07_model: single_pole  # FP07 transfer function: single_pole or double_pole
  goodman: true         # Goodman coherent noise removal
  f_AA: 98.0            # anti-aliasing filter cutoff [Hz]
  fit_method: iterative # Method 2 fitting: iterative or mle
  spectrum_model: kraichnan  # theoretical spectrum: batchelor or kraichnan
  first_bin: 2          # lowest FFT bin chi uses (DC = 0). The per-segment
                        # linear detrend leaves bin 1 ~30% low on a red
                        # spectrum (issue #191); 1 reproduces a product
                        # made before bin 1 was dropped
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
- **`speed`**: Derived per `speed_method` (default: the pressure time series, dP/dt).
- **`speed_method`**: The pressure method (\|dP/dt\|).
- **`W_min`**: Resolved from the vehicle direction: 0.3 dbar/s for a free-falling profiler, 0.05 for glide/horizontal platforms.
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
