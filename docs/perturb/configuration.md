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

The perturb configuration has 18 sections (`files`, `gps`, `hotel`, `profiles`, `fp07`, `ct`, `bottom`, `top_trim`, `epsilon`, `chi`, `ctd`, `speed`, `qc`, `binning`, `netcdf`, `stratification`, `parallel`, `instruments`). Each parameter is optional — unset values fall back to defaults.

---

### `files` — File Discovery

Controls where `.p` files are found and where output goes.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `p_file_root` | string | `"VMP/"` | Root directory for .p file discovery |
| `p_file_pattern` | string | `"**/*.p"` | Glob pattern for .p files |
| `output_root` | string | `"results/"` | Base output directory |
| `trim` | bool | `true` | Enable trimming of corrupt final records (complete files are referenced in place, not copied) |
| `force_trim` | bool | `false` | Re-trim even when an up-to-date trimmed output already exists |
| `merge` | bool | `false` | Enable merging of split .p files |

#### Config-relative paths (`<CONFIG_DIR>`)

`p_file_root`, `output_root`, `gps.file`, and `hotel.file` may begin with the
token `<CONFIG_DIR>`, which expands to the **directory of the config file
itself** at the moment a path is used. This lets a config and its data tree live
together and be run from any working directory (or mounted at a different point
on another machine):

```yaml
files:
  p_file_root: <CONFIG_DIR>/VMP      # the VMP/ folder next to this YAML
  output_root: <CONFIG_DIR>/results
```

Paths **without** the token keep their existing meaning — relative to the
current working directory. Crucially, the `<CONFIG_DIR>` *token* (not the
resolved absolute path) is what feeds the stage-directory cache signatures, so
moving or remounting the config + data tree does **not** invalidate previously
computed outputs. `<CONFIG_DIR>` is only meaningful for a config loaded from a
file; using it in a config assembled in memory raises an error.

This applies to the versioned `{stage}_NN` output *directories*. The finer-grained
per-file skip markers additionally key on each input's size and modification time,
so a copy/remount that does **not** preserve mtimes will re-process the individual
`.p` files — into the same, correctly matched output directory (no orphans), just
not for free.

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
| `max_time_diff` | float | `60` | Warn when positions are extrapolated more than this [s] outside GPS coverage |

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
| `speed_factor` | float | `0.3` | Currently unused (reserved; tuning has no effect) |
| `median_factor` | float | `1.0` | Currently unused (reserved; tuning has no effect) |
| `vibration_frequency` | int | `16` | Currently unused (reserved; tuning has no effect) |
| `vibration_factor` | float | `4.0` | Vibration standard-deviation acceptance factor |

---

### `top_trim` — Surface Trimming

Controls removal of initial surface instabilities from profiles.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable` | bool | `false` | Enable top trimming |
| `dz` | float | `0.5` | Depth bin width for variance calculation [m] |
| `min_depth` | float | `1.0` | Minimum trim depth [m] |
| `max_depth` | float | `50.0` | Maximum trim depth [m] |
| `quantile` | float | `0.6` | Quantile of per-bin std taken as the settled background; trimming is robust while the prop wash spans less than `1 − quantile` of the search range |
| `noise_factor` | float | `2.0` | A bin is still in the prop wash when its std exceeds `noise_factor` × background; trimming clears the deepest such bin |

> Top trim is driven by the **accelerometers** (Ax, Ay) only — they mark the instrument's mechanical settling. Shear probes, inclinometers, and fall rate respond to the ocean turbulence the instrument falls through and would over-trim. VMP only; MicroRiders use a separate operation.

---

### `epsilon` — TKE Dissipation Rate

Controls computation of epsilon from shear probe spectra.

Note that perturb and `rsi-tpw` use different spectral defaults (perturb: `fft_sec` 1.0 — one second, resolved per instrument sampling rate — for both epsilon and chi; `rsi-tpw`: 1024 samples with a 4096-sample dissipation window), so their outputs can differ in vertical resolution and noise behavior.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fft_sec` | float | `1.0` | FFT segment duration [s], converted per instrument via its sampling rate (512 Hz VMP-250 → 512 samples; 1–2 kHz coastal units scale automatically) |
| `diss_sec` | float | `null` | Dissipation window [s] (null = 4 × fft_sec). See [dissipation_length.md](dissipation_length.md) for how to choose these (the optimum is ε-dependent) |
| `overlap_sec` | float | `null` | Window overlap [s] (null = half the window) |
| `fft_length` | int | `null` | EXPERT override [samples]; wins over `fft_sec` (legacy configs keep bit-identical behavior and signatures) |
| `diss_length` | int | `null` | EXPERT override [samples]; wins over `diss_sec` (null = 4 x fft) |
| `overlap` | int | `null` | Window overlap [samples] (null = diss_length // 2) |
| `goodman` | bool | `true` | Enable Goodman coherent noise removal |
| `f_AA` | float | `98.0` | Anti-aliasing filter cutoff [Hz] |
| `f_limit` | float | `null` | Upper frequency limit [Hz] (null = f_AA) |
| `fit_order` | int | `3` | Polynomial fit order for Nasmyth integration |
| `despike_thresh` | float | `8` | Despike threshold: ratio of the rectified high-passed signal to its low-passed envelope |
| `despike_smooth` | float | `0.5` | Low-pass cutoff [Hz] for the despike envelope smoother |
| `salinity` | float \| `"measured"` \| `"hotel"` \| `null` | `null` | Salinity [PSU] for viscosity. `null` = fixed 35; a number = that fixed value; `"measured"` = per-profile from C/T/P (TEOS-10, needs conductivity); `"hotel"` (or `"hotel:<var>"`) = a [hotel](#hotel--hotel-file-external-telemetry)-injected salinity channel (default variable `salinity`) — for gliders/MicroRiders without onboard conductivity |
| `epsilon_minimum` | float | `1e-13` | Floor: values below this are set to NaN |
| `T_source` | string | `null` | Temperature source for viscosity |
| `T1_norm` | float | `1.0` | Shear probe 1 normalization factor |
| `T2_norm` | float | `1.0` | Shear probe 2 normalization factor |
| `fom_max` | float | `null` | Per-probe figure-of-merit cut (null = no cut). E.g. `2.0` NaNs each per-probe cell (`e_N`, `epsilon[probe,:]`) whose `fom[probe,seg]` >= `fom_max`, applied **before** `mk_epsilon_mean` so bad probes drop out of the geometric mean individually |
| `diagnostics` | bool | `false` | Include diagnostic variables |

---

### `chi` — Thermal Variance Dissipation Rate

Controls computation of chi from FP07 thermistor spectra.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable` | bool | `false` | Enable chi computation |
| `fft_sec` | float | `1.0` | FFT segment duration [s] (same duration-first interface as `[epsilon]`) |
| `diss_sec` | float | `null` | Dissipation window [s] (null = 4 x fft_sec) |
| `overlap_sec` | float | `null` | Window overlap [s] (null = half the window) |
| `fft_length` | int | `null` | EXPERT override [samples]; wins over `fft_sec` |
| `diss_length` | int | `null` | EXPERT override [samples]; wins over `diss_sec` (null = 4 x fft) |
| `overlap` | int | `null` | Window overlap [samples] (null = diss_length // 2) |
| `fp07_model` | string | `"single_pole"` | FP07 transfer function: `single_pole` or `double_pole` |
| `goodman` | bool | `true` | Enable Goodman coherent noise removal |
| `f_AA` | float | `98.0` | Anti-aliasing filter cutoff [Hz] |
| `use_epsilon` | bool | `true` | Method selector. `true` = Method 1 (chi from shear-probe epsilon); `false` = Method 2 spectral fit (uses `fit_method`). Set `false` for instruments where shear epsilon is unreliable, e.g. a MicroRider on a vibrating glider |
| `fit_method` | string | `"iterative"` | Method 2 fitting: `iterative` or `mle` (ignored when `use_epsilon: true`) |
| `spectrum_model` | string | `"kraichnan"` | Theoretical spectrum: `batchelor` or `kraichnan` |
| `salinity` | float \| `"measured"` \| `"hotel"` \| `null` | `null` | Salinity [PSU] for the viscosity in the chi spectral fit. `null` = fixed 35; a number = that fixed value; `"measured"` = per-profile practical salinity from the profile's own `JAC_C`/`JAC_T`/`P` (TEOS-10); `"hotel"` (or `"hotel:<var>"`) = a [hotel](#hotel--hotel-file-external-telemetry)-injected salinity channel |
| `mixing` | bool | `true` | Append derived mixing quantities (`N2`, `dTdz`, `K_T`, `Gamma`, `K_rho`, plus the paired `epsilon_paired` for traceability) to the chi NetCDFs, on the chi window grid. The `N2` salinity follows the `stratification.salinity` setting (conductivity by default, or a hotel channel); see [mixing_efficiency.md](../mixing_efficiency.md) for definitions and masking |
| `chi_minimum` | float | `1e-13` | Floor for `mk_chi_mean`: values <= this go to NaN |
| `fom_max` | float | `null` | Per-probe figure-of-merit cut (null = no cut). Same mechanism as `epsilon.fom_max` but on the chi NetCDFs: NaNs `chi[probe,seg]` / `chi_N` where `fom[probe,seg]` >= `fom_max` |
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

### `speed` — Through-Water Speed Source

Controls how the through-water (profiling) speed is computed. The speed is computed after the hotel merge, so all methods have access to both `.p`-file and hotel channels.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | string | `"pressure"` | Speed source: `"pressure"` (ODAS smoothed \|dP/dt\|; correct for VMP), `"em"` (the `U_EM` channel from a MicroRider EM flowmeter; errors out if missing), `"flight"` (glider flight model: \|W\| / sin(\|pitch\|+aoa), the ODAS convention — the glide path is steeper than pitch by the angle of attack; roll does not enter; pitch axis auto-picked from `Incl_X`/`Incl_Y` by amplitude), or `"constant"` (the scalar in `value`) |
| `value` | float | `null` | Fixed speed [m/s], only for `method: constant` |
| `aoa_deg` | float | `3.0` | Angle of attack [deg], only for `method: flight` |
| `min_pitch_deg` | float | `5.0` | Flight method: drop samples with \|pitch\| below this [deg] (steady-glide flight is invalid near dive/climb inflections) |
| `speed_cutout` | float | `0.05` | Floor [m/s] applied to the fast-rate speed |
| `tau` | float | `null` | Smoothing time constant [s]; null = vehicle default (vmp/xmp 1.5, slocum_glider 3.0, ...) |
| `amplitude_quantile` | list | `[1.0, 99.0]` | Flight method: percentile spread used to auto-pick the pitch axis from `Incl_X`/`Incl_Y`; 1..99 strips outliers (surface tumbles, sensor saturation spikes) |

---

### `qc` — Per-Segment QC Gate

Flags (and optionally NaNs) dissipation/chi segments based on QC channels. Each `*_drop_from` entry names a hotel-injected channel (uint8 bitfield or boolean) sampled by time over the segment's window; if any sample is nonzero, the segment is flagged. The `qc_drop_epsilon` / `qc_drop_chi` variables are always written to the diss / chi NetCDFs.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable` | bool | `true` | Enable the QC gate |
| `drop_action` | string | `"nan"` | `"nan"` NaNs `e_*`/`epsilonMean` (and `chi_*`/`chiMean`) for flagged segments; `"flag_only"` leaves the values untouched (the `qc_drop_*` bitfield is still written) |
| `epsilon_drop_from` | list | `[]` | Channel names OR'd over each diss segment's time window, e.g. `["q_drop_epsilon"]` |
| `chi_drop_from` | list | `[]` | Same, for chi segments |
| `rules` | dict | `{}` | Internal QC rules. Each named entry produces a synthetic uint8 channel that can be referenced by `*_drop_from`. See below |

Each `rules` entry has a `type` (default `range`) and a `bit` to set in the synthetic channel:

- **`range`** — flags samples where a channel is out of range. Keys: `channel` (a `pf.channels` name, or pseudo-names `pitch`/`roll` auto-picked from `Incl_X`/`Incl_Y`), and any of `min`, `max`, `abs_max`.
- **`pitch_w_consistency`** — flags samples where pitch direction and dP/dt sign disagree (e.g. a stalled glider pitched up while sinking). Keys: `pitch_min_deg`, `W_min_dbar_per_s` (dead bands around level/stationary), and `pitch_positive` resolving the inclinometer mounting polarity: `"auto"` (default; infer from the deployment-wide majority sign of pitch·W), `"nose_down"` (positive pitch = nose-down), or `"nose_up"` (positive pitch = nose-up).

See `odas_tpw.perturb.qc_rules` for the full per-entry schema.

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

CF-1.13 / ACDD-1.3 global attributes applied to combo output files. All default to `null` (not set) unless specified.

| Parameter | Type | Description |
|-----------|------|-------------|
| `title` | string | Dataset title |
| `summary` | string | Dataset summary |
| `institution` | string | Data-producing institution |
| `creator_name` | string | Creator name |
| `creator_email` | string | Creator email |
| `project` | string | Project name |
| `Conventions` | string | Default: `"CF-1.13, ACDD-1.3"` |

See [CF Conventions](https://cfconventions.org/) and [ACDD](https://wiki.esipfed.org/Attribute_Convention_for_Data_Discovery_1-3) for the full attribute list.

---

### `stratification` — Background N²/dT/dz

Background buoyancy frequency (`N2`) and temperature gradient (`dTdz`), computed
with the Thorpe-sorted (adiabatically leveled) method and written to the
profile and dissipation products independent of epsilon/chi (the chi
product's `N2`/`dTdz` are governed by `chi.mixing`). The profile product uses
the configurable background `window`; the diss product uses its own dissipation
window. These are **profile-only** (down-cast) quantities and are **not** written
to the CTD product, which spans the whole up/down trajectory.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable` | bool | `true` | Write `N2`/`dTdz` to the profile and diss products (not the CTD product) |
| `window` | float | `2.0` | Background vertical window [dbar] for the profile product |
| `salinity` | float \| `"measured"` \| `"hotel"` \| `null` | `null` | Salinity source for `N2`. `null` = the profile's own conductivity via TEOS-10 (else 35 PSU); a number = that fixed PSU; `"measured"` = C/T/P (TEOS-10); `"hotel"` (or `"hotel:<var>"`) = a [hotel](#hotel--hotel-file-external-telemetry)-injected salinity channel (default variable `salinity`). Use `"hotel"` for gliders/MicroRiders with no onboard conductivity but an external CTD feed |

---

### `parallel` — Parallel Processing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `jobs` | int | `1` | Number of parallel workers (0 = auto-detect cores) |

---

### `instruments` — Per-Instrument Overrides

Overrides keyed by serial-number identifier, matched against the parent directory of each `.p` file (e.g. `ARCTERX/VMP/SN465` → `SN465`). Default: `{}` (no overrides).

| Inner key | Type | Description |
|-----------|------|-------------|
| `exclude_shear_probes` | list of strings | Probe names (e.g. `["sh2"]`) to suppress for this instrument. The named probe is NaN'd before `mk_epsilon_mean`, so it is excluded from the multi-probe `epsilonMean` and from chi Method 1 (which uses `epsilonMean`) |

```yaml
instruments:
  SN465:
    exclude_shear_probes: ["sh2"]
```

---

## Diagnostics

Several sections include a `diagnostics` flag. When set to `true`, additional diagnostic variables (standard deviations, sample counts, etc.) are included in the output. Toggling diagnostics does **not** create a new output directory — the hash excludes the `diagnostics` key.
