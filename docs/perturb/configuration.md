# Perturb Configuration Reference

Processing parameters can be set via a YAML configuration file, CLI flags, or both. The merge order is:

```
defaults <- config file <- CLI flags
```

## Quick Start

```bash
perturb init                        # writes config.yaml with all defaults
perturb init my_settings.yaml       # custom filename
perturb run -c config.yaml -o results/ VMP/*.p
```

## Configuration Sections

The perturb configuration has 14 sections. Each parameter is optional — unset values fall back to defaults.

---

### `files` — File Discovery

Controls where `.p` files are found and where output goes.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `p_file_root` | string | `"VMP/"` | Root directory for .p file discovery |
| `p_file_pattern` | string | `"**/*.p"` | Glob pattern for .p files |
| `output_root` | string | `"results/"` | Base output directory |
| `trim` | bool | `true` | Enable trimming of corrupt final records |
| `merge` | bool | `false` | Enable merging of split .p files |

---

### `gps` — GPS Providers

Controls how GPS positions are assigned to measurements.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | string | `"nan"` | GPS source: `"nan"`, `"fixed"`, `"csv"`, `"netcdf"` |
| `lat` | float | `null` | Fixed latitude (for source="fixed") |
| `lon` | float | `null` | Fixed longitude (for source="fixed") |
| `file` | string | `null` | Path to CSV or NetCDF GPS file |
| `time_col` | string | `"t"` | Time column/variable name |
| `lat_col` | string | `"lat"` | Latitude column/variable name |
| `lon_col` | string | `"lon"` | Longitude column/variable name |
| `max_time_diff` | float | `60` | Max time difference [s] for interpolation |

---

### `hotel` — Hotel File (External Telemetry)

Injects external vehicle telemetry (speed, pitch, roll, heading, CTD) from gliders, AUVs, or Remus into the instrument channels. Data is interpolated onto the instrument's fast or slow time axes.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable` | bool | `false` | Enable hotel file loading |
| `file` | string | `null` | Path to hotel file (CSV, NetCDF, or .mat) |
| `time_column` | string | `"time"` | Time column/variable name in hotel file |
| `time_format` | string | `"auto"` | Time format: `"auto"`, `"seconds"`, `"epoch"`, `"iso"` |
| `channels` | dict | `{}` | Column name mapping (hotel → output). Empty = load all |
| `fast_channels` | list | `["speed", "P"]` | Channels interpolated onto the fast time axis |
| `interpolation` | string | `"pchip"` | Interpolation method: `"pchip"` or `"linear"` |

---

### `profiles` — Profile Detection

Controls how profiling segments are identified from pressure data.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `P_min` | float | `0.5` | Minimum pressure threshold [dbar] |
| `W_min` | float | `0.3` | Minimum fall rate [dbar/s] |
| `direction` | string | `"down"` | Profile direction: `"up"` or `"down"` |
| `min_duration` | float | `7.0` | Minimum profile duration [seconds] |
| `diagnostics` | bool | `false` | Include diagnostic variables in output |

---

### `fp07` — FP07 In-Situ Calibration

Controls calibration of FP07 thermistors against a reference sensor.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `calibrate` | bool | `true` | Enable FP07 calibration |
| `order` | int | `2` | Steinhart-Hart polynomial order |
| `max_lag_seconds` | float | `10` | Max cross-correlation lag [seconds] |
| `reference` | string | `"JAC_T"` | Reference temperature channel name |
| `must_be_negative` | bool | `true` | Restrict lag to negative values (FP07 leads reference) |

---

### `ct` — CT Sensor Alignment

Controls cross-correlation alignment of conductivity and temperature sensors.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `align` | bool | `true` | Enable CT alignment |
| `T_name` | string | `"JAC_T"` | Temperature channel name |
| `C_name` | string | `"JAC_C"` | Conductivity channel name |

---

### `bottom` — Bottom Crash Detection

Controls detection and removal of bottom-crash contaminated data.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable` | bool | `false` | Enable bottom crash detection |
| `depth_window` | float | `4.0` | Depth window for detection [m] |
| `depth_minimum` | float | `10.0` | Minimum depth for detection [m] |
| `speed_factor` | float | `0.3` | Speed threshold factor |
| `median_factor` | float | `1.0` | Median deviation factor |
| `vibration_frequency` | int | `16` | Vibration detection frequency [Hz] |
| `vibration_factor` | float | `4.0` | Vibration amplitude factor |

---

### `top_trim` — Surface Trimming

Controls removal of initial surface instabilities from profiles.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable` | bool | `false` | Enable top trimming |
| `dz` | float | `0.5` | Depth bin width for variance calculation [m] |
| `min_depth` | float | `1.0` | Minimum trim depth [m] |
| `max_depth` | float | `50.0` | Maximum trim depth [m] |
| `quantile` | float | `0.6` | Variance quantile threshold |

---

### `epsilon` — TKE Dissipation Rate

Controls computation of epsilon from shear probe spectra. Parameters are passed through to `rsi_python.dissipation.get_diss`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fft_length` | int | `256` | FFT segment length [samples] |
| `diss_length` | int | `null` | Dissipation window [samples] (null = 2 x fft_length) |
| `overlap` | int | `null` | Window overlap [samples] (null = diss_length // 2) |
| `goodman` | bool | `true` | Enable Goodman coherent noise removal |
| `f_AA` | float | `98.0` | Anti-aliasing filter cutoff [Hz] |
| `f_limit` | float | `null` | Upper frequency limit [Hz] (null = f_AA) |
| `fit_order` | int | `3` | Polynomial fit order for Nasmyth integration |
| `despike_thresh` | float | `8` | Despike threshold [MAD] |
| `despike_smooth` | float | `0.5` | Despike smoothing window [s] |
| `salinity` | float | `null` | Salinity [PSU] for viscosity (null = S=35 approx) |
| `epsilon_minimum` | float | `1e-13` | Floor: values below this are set to NaN |
| `T_source` | string | `null` | Temperature source for viscosity |
| `T1_norm` | float | `1.0` | Shear probe 1 normalization factor |
| `T2_norm` | float | `1.0` | Shear probe 2 normalization factor |
| `diagnostics` | bool | `false` | Include diagnostic variables |

---

### `chi` — Thermal Variance Dissipation Rate

Controls computation of chi from FP07 thermistor spectra.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable` | bool | `false` | Enable chi computation |
| `fft_length` | int | `512` | FFT segment length [samples] |
| `diss_length` | int | `null` | Dissipation window [samples] (null = 3 x fft_length) |
| `overlap` | int | `null` | Window overlap [samples] (null = diss_length // 2) |
| `fp07_model` | string | `"single_pole"` | FP07 transfer function: `single_pole` or `double_pole` |
| `goodman` | bool | `true` | Enable Goodman coherent noise removal |
| `f_AA` | float | `98.0` | Anti-aliasing filter cutoff [Hz] |
| `fit_method` | string | `"iterative"` | Method 2 fitting: `iterative` or `mle` |
| `spectrum_model` | string | `"kraichnan"` | Theoretical spectrum: `batchelor` or `kraichnan` |
| `salinity` | float | `null` | Salinity [PSU] for viscosity |
| `diagnostics` | bool | `false` | Include diagnostic variables |

---

### `ctd` — CTD Time-Binning

Controls time-binning of CTD channels per file.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable` | bool | `true` | Enable CTD binning |
| `bin_width` | float | `0.5` | Time bin width [seconds] |
| `T_name` | string | `"JAC_T"` | Temperature channel name |
| `C_name` | string | `"JAC_C"` | Conductivity channel name |
| `variables` | list | `null` | Explicit list of channels to bin (null = auto-detect) |
| `method` | string | `"mean"` | Aggregation method: `"mean"` or `"median"` |
| `diagnostics` | bool | `false` | Include diagnostic variables (n_samples, *_std) |

---

### `binning` — Depth/Time Binning

Controls binning of per-profile and per-diss NetCDFs.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | string | `"depth"` | Binning method: `"depth"` or `"time"` |
| `width` | float | `1.0` | Bin width [m for depth, s for time] |
| `aggregation` | string | `"mean"` | Aggregation: `"mean"` or `"median"` |
| `diss_width` | float | `null` | Override bin width for diss (null = use `width`) |
| `diss_aggregation` | string | `null` | Override aggregation for diss |
| `chi_width` | float | `null` | Override bin width for chi |
| `chi_aggregation` | string | `null` | Override aggregation for chi |
| `diagnostics` | bool | `false` | Include diagnostic variables |

---

### `netcdf` — NetCDF Global Attributes

CF-1.8 / ACDD-1.3 global attributes applied to combo output files. All default to `null` (not set) unless specified.

| Parameter | Type | Description |
|-----------|------|-------------|
| `title` | string | Dataset title |
| `summary` | string | Dataset summary |
| `institution` | string | Data-producing institution |
| `creator_name` | string | Creator name |
| `creator_email` | string | Creator email |
| `project` | string | Project name |
| `Conventions` | string | Default: `"CF-1.8, ACDD-1.3"` |

See [CF Conventions](https://cfconventions.org/) and [ACDD](https://wiki.esipfed.org/Attribute_Convention_for_Data_Discovery_1-3) for the full attribute list.

---

### `parallel` — Parallel Processing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `jobs` | int | `1` | Number of parallel workers (0 = auto-detect cores) |

---

## Diagnostics

Several sections include a `diagnostics` flag. When set to `true`, additional diagnostic variables (standard deviations, sample counts, etc.) are included in the output. Toggling diagnostics does **not** create a new output directory — the hash excludes the `diagnostics` key.
