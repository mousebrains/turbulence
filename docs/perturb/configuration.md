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

Merged channels become plain instrument channels: they are written into the per-profile NetCDFs, can drive [QC rules](#qc--per-segment-qc-gate), and feed salinity (`epsilon`/`chi`/`stratification` `salinity: "hotel[:<var>]"`). A merged **speed** channel drives the through-water speed only when [`speed.method: "hotel"`](#speed--through-water-speed-source) selects it — merging alone does not change the speed used by epsilon/chi.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable` | bool | `false` | Enable hotel file loading |
| `file` | string | `null` | Path to hotel file (CSV, NetCDF, or .mat) |
| `time_column` | string | `"time"` | Time column/variable name in hotel file |
| `time_format` | string | `"auto"` | Time format: `"auto"`, `"seconds"`, `"epoch"`, `"iso"` |
| `channels` | dict | `{}` | Column name mapping (hotel → output). Empty = load all |
| `fast_channels` | list | `["speed", "P"]` | Channels interpolated onto the fast time axis |
| `interpolation` | string | `"pchip"` | Interpolation method: `"pchip"` or `"linear"` |
| `max_gap` | float or `"unlimited"` | — (**required**) | [s] NaN the merged channel where the two bracketing source samples are farther apart than this, instead of ruling a straight line across the hole. No default: the right limit is the sensor's own rate (~30 s for a 1 Hz CTD, minutes for a flight-state variable), so omitting it raises. `"unlimited"` deliberately keeps the old interpolate-across-anything behaviour. Per-channel override wins; a per-channel `null` inherits this value |
| `extrapolate` | bool | `false` | Hold the first/last source value outside the source's own time range instead of NaN. Per-channel override wins; a per-channel `null` inherits this value |
| `time_offset` | float | `0.0` | [s] **Added** to the hotel file's timestamps to put them on the instrument's clock. A MicroRider takes its time from the science computer once, at file open, and then free-runs, so two unsynchronised clocks is the normal case rather than the exception. `fp07-cal fit` measures the offset from pressure against pressure — `clock_offset_s` in `coefficients.json` — with no thermal physics involved. **Sign:** `clock_offset_s` is positive when the hotel timestamps run *ahead* of the instrument's, so set `time_offset = -clock_offset_s`. Per-channel override wins (the CTD and the flight computer are different clocks). Leaving it `0.0` asserts the two clocks agree: on osu684 the offset was +5.06 s, which at a 0.27 dbar/s climb misregisters every merged CTD sample by 1.4 dbar — more than the 1 m bin it lands in |

---

### `profiles` — Profile Detection

Controls how profiling segments are identified from pressure data.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `P_min` | float | `0.5` | Minimum pressure threshold [dbar] |
| `W_min` | float | `null` | Minimum fall rate [dbar/s]. `null` = auto: 0.3 for a free-falling profiler, 0.05 for glide/horizontal platforms (the VMP-tuned 0.3 rejects every glider cast) |
| `direction` | string | `"auto"` | Profile direction: `"auto"`, `"up"`, `"down"`, `"glide"` (up + down), or `"horizontal"`. `"auto"` resolves from the instrument's `vehicle` (e.g. `slocum_glider` → `glide`); instruments without a `vehicle` in their config default to `down` — set `direction: glide` explicitly for such glider corpora |
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
| `T_source` | string \| float \| `null` | `null` | Reference temperature for seawater properties (viscosity for ε; viscosity and κ_T for χ — one knob serves both stages). `null`/`"auto"` = first plausible of `T1`, `T2`, …, `T`, `JAC_T` (implausible channels — railed, drifting, mostly non-finite — are skipped with a warning; QC evaluates in-water samples, P > 0.5 dbar, when pressure is available); a channel name (e.g. `"T2"`, `"JAC_T"`, or a hotel temperature channel) = use that channel (a QC failure warns but proceeds); a number = constant reference temperature [°C] (ODAS `constant_temp` parity). The resolved source is recorded in the diss/chi products as `temperature_source`/`temperature_qc` attributes. |
| `spectral_qc` | bool | `false` | ATOMIX-style per-probe cut (bits 1 and 16 only — see below). Default `false` while `chi.spectral_qc` defaults `true`; deliberate, documented asymmetry |
| `FM_max` | float | `1.15` | Bit 1: cut where `FM > FM_max`. The **MAD-based `FM`**, not the variance-ratio `fom` |
| `var_resolved_min` | float | `0.5` | Bit 16: cut where `var_resolved < min` **and `method == 0`**. Skipped entirely when no `method` variable exists |
| `pair_policy` | string | `"keep_both"` | Bit 4, windows with **exactly two finite probes**: `keep_both` \| `drop_high` \| `flag_only` |
| `pair_limit` | float | `2.7718…` | Full coefficient of mean(σ_ln); equals `scor160.l4.DEFAULT_DISS_RATIO_LIMIT` (1.96·√2) exactly, so a value moves between the two settings unchanged |
| `fom_max` | float | `null` | Per-probe figure-of-merit cut (null = no cut). E.g. `2.0` NaNs each per-probe cell (`e_N`, `epsilon[probe,:]`) whose `fom[probe,seg]` >= `fom_max`, applied **before** `mk_epsilon_mean` so bad probes drop out of the geometric mean individually. **Caveat:** on a corpus containing ISR estimates a low value cuts them *as a class* — the variance-ratio `fom` compares observed variance against a model integral an ISR fit never uses, so a poor ratio there is expected, not diagnostic. On ARCTERX-2022 every window with `fom >= 1.15` was an ISR estimate. Prefer `spectral_qc` |
| `diagnostics` | bool | `false` | Include diagnostic variables |


#### `epsilon.spectral_qc` — what it does and does not implement

`scor160.l4._compute_flags` (the rsi path) has five criteria. perturb's diss
product carries the inputs for two of them:

| bit | criterion | here |
|---|---|---|
| 1 | `FM > limit` | **implemented** (`FM_max`) |
| 2 | despike fraction > limit | not implementable — no `despike_fraction` in the product |
| 4 | inter-probe consistency | `pair_policy`, two-probe windows only (see below) |
| 8 | despike passes > limit | not implementable — no `despike_passes` |
| 16 | `var_resolved < limit`, `method == 0` only | **implemented** (`var_resolved_min`) |

That is why it is called `spectral_qc`, mirroring the shipped
`chi.spectral_qc`, and **not** `atomix_qc`: it cannot be the full flag set, and
naming it after the standard would claim a conformance it does not have.

**The `method` gate on bit 16 is not optional.** An ISR fit never integrates the
dissipation range, so a low resolved fraction is expected rather than
diagnostic, and ATOMIX exempts those estimates. Ungated on ARCTERX-2022 the
criterion rejects 3.51% of probe-windows instead of 0.18%. If the dataset has no
`method` variable the criterion is **skipped with a warning**, never guessed.

**Failing every probe drops the window.** With no clean probe the window becomes
NaN, matching rsi's `_compute_epsi_final` and deliberately unlike
`chi.spectral_qc`, which falls back to all probes. The reason is the Method-1
coupling: a finite-but-wrong epsilon rescales chi roughly linearly while the chi
`fom` stays ≈1, so nothing downstream can reject it, whereas NaN self-excludes.

#### `epsilon.pair_policy` — the two-probe case the other rules decline

`mk_epsilon_mean`'s outlier rule needs `n_probes >= 3`. With two probes neither
is identifiable as the outlier, and always dropping the maximum would
systematically retain the lower probe and bias `epsilonMean` low, so it keeps
both. ATOMIX bit 4 *does* act on a pair, and keeps the minimum. Both positions
are defensible, so the choice is exposed rather than made:

- **`keep_both`** (default) — perturb's existing behaviour, bit-identical.
- **`drop_high`** — ATOMIX's action. Opt in knowing it makes a low junk probe
  authoritative: on a three-probe fixture that costs a factor of 50.
- **`flag_only`** — count the disagreement, mask nothing.

Windows with three or more **finite** probes are left to `mk_epsilon_mean`, so
the two rules are exhaustive and never contest the same window. The gate is on
the per-window finite count, not the instrument's probe count: `mk_epsilon_mean`
declines `finite_count > 2`, so a three-probe instrument with one probe NaN in a
window falls into exactly the same gap as a two-probe one. The threshold and σ_ln come from
`processing.probe_consistency`, shared with the cross-probe consistency log, so
the gate and the diagnostic always describe the same statistic.

`tests/test_interprobe_consistency_forms.py` pins both rules and their
disagreement. On ARCTERX-2022, 6.40% of epsilon windows exceed the threshold, so
the policy is not academic.

#### Provenance

An applied `spectral_qc` writes onto the diss product: `spectral_qc_applied`,
`spectral_qc_FM_max`, `spectral_qc_var_resolved_min`, `spectral_qc_pair_policy`,
`spectral_qc_pair_limit`, the per-criterion counts `spectral_qc_n_cut_FM` /
`_n_cut_var_resolved` / `_n_cut_pair`, and `spectral_qc_rejected_fraction`
(over finite cells). The cut is **not** epsilon-neutral — on ARCTERX-2022 it
rejects ~20% of probe-windows, 47% of the top epsilon decile, and shifts the
median epsilon by 1.14× — so the size of it belongs in the file, not only in
this document.

---

### `chi` — Thermal Variance Dissipation Rate

Controls computation of chi from FP07 thermistor spectra. The reference
temperature for the χ viscosity/κ_T comes from `epsilon.T_source` — one knob
serves both stages (there is no `chi.T_source`).

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

Controls how the through-water (profiling) speed is computed. Speed is computed after the [hotel merge](#hotel--hotel-file-external-telemetry): `method: "hotel"` consumes a hotel-merged channel (named by `hotel_var`), while the other methods read the instrument's own channels. The selected source is recorded on the products as `speed_source` (`"pressure"`, `"em"`, `"flight"`, `"constant:<v>"`, or `"hotel:<var>"`). If an explicitly selected non-pressure method fails (an unusable hotel channel; a missing or all-NaN `U_EM`; a flight model with zero finite samples — pitch never clearing `min_pitch_deg`; a non-finite `value`), the file is **aborted with a recorded error** — it never silently falls back to \|dP/dt\| (which has ~U⁴ leverage on ε) or publishes the `speed_cutout` floor as data.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | string | `"pressure"` | Speed source: `"pressure"` (ODAS smoothed \|dP/dt\|; correct for VMP), `"em"` (the `U_EM` channel from a MicroRider EM flowmeter; errors out if missing), `"flight"` (glider flight model: \|W\| / sin(\|pitch\|+aoa), the ODAS convention — the glide path is steeper than pitch by the angle of attack; roll does not enter; pitch axis auto-picked from `Incl_X`/`Incl_Y` by amplitude), `"constant"` (the scalar in `value`), or `"hotel"` (the hotel-merged channel named by `hotel_var`; errors out when the channel is missing, matches neither time grid, or is less than 50% finite — the file is aborted with a recorded error, never silently floored to `speed_cutout` or substituted with \|dP/dt\|) |
| `value` | float | `null` | Fixed speed [m/s], only for `method: constant` |
| `hotel_var` | string | `"speed"` | Merged channel name, only for `method: hotel`. Map a hotel source variable onto it via `hotel.channels` (e.g. `m_speed: "speed"`); the default `hotel.fast_channels` puts `"speed"` on the fast grid, and slow-grid channels are interpolated/smoothed to fast rate like the other methods |
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

**Each key versions only the stages it reaches.** The block feeds the stage
signature so a change re-versions the affected output, but `fp07_tau_scale` is
consumed only by the chi computation, so it versions **chi** and not **diss** —
setting it does not force an epsilon recompute that could not change a value
(verified bit-identical over 34 820 `epsilon`/`FM`/`fom`/`var_resolved` values).
`exclude_shear_probes` reaches diss directly and chi transitively through
Method 1's `epsilonMean`, so it versions both. An instrument whose settings are
all irrelevant to a stage is dropped from that stage's hash entirely, so a
config carrying only a chi-side knob hashes identically to one with no
`instruments` at all.

| Inner key | Type | Description |
|-----------|------|-------------|
| `exclude_shear_probes` | list of strings | Probe names (e.g. `["sh2"]`) to suppress for this instrument. The named probe is NaN'd before `mk_epsilon_mean`, so it is excluded from the multi-probe `epsilonMean` and from chi Method 1 (which uses `epsilonMean`) |
| `fp07_tau_scale` | dict | Per-thermistor multiplier on the FP07 time constant, e.g. `{T1: 0.30, T2: 0.75}`. Keys accept the bead name (`T1`) or the gradient channel (`T1_dT1`); values must be finite and > 0. Default `{}` = every probe on the tau model, bit-identical to before |

```yaml
instruments:
  SN465:
    exclude_shear_probes: ["sh2"]
  SN194:
    fp07_tau_scale: {T1: 0.30, T2: 0.75}
```

#### `fp07_tau_scale` — when two beads do not agree

Chi is computed by fitting a Batchelor/Kraichnan spectrum, attenuated by the
FP07 transfer function `|H(f)|² = 1/(1 + (2πf·τ)²)`, to the observed gradient
spectrum. Until now `τ` came from a single model
([`fp07_tau`](../../src/odas_tpw/chi/fp07.py) — `lueck` for `single_pole`,
`goto` for `double_pole`) and was shared by every thermistor on the
instrument. Two beads on one probe head can have materially different
response, and that assumption then biases them apart.

The symptom is a **large, persistent chi disagreement between the two
thermistors that varies with `K_max_ratio`** (= `K_max/kB`, how much of the
Batchelor rolloff is resolved rather than extrapolated). On ARCTERX-2022:

| unit | `chi(T1)/chi(T2)` | across `K_max/kB` quartiles |
|---|---|---|
| SN 194 | 1.77x | 2.27 → 1.83 → 1.67 → 1.54 |
| SN 428 | 0.72x | 0.59 → 0.71 → 0.77 → 0.80 |

Both trend toward 1 as more of the spectrum is resolved. That slope is the
diagnostic: a **flat gain** error (wrong `diff_gain`, bridge gain, `E_B`) is
`K_max_ratio`-independent, whereas a **response** error is amplified exactly
where the extrapolation carries the most weight. Note also that chi goes as the
gradient *squared*, so a 1.77x chi ratio is only a 1.33x response difference —
easy to under-rate.

Fitting one τ per bead collapsed both units to 1.03x with the quartile spread
falling from 1.47x/1.37x to 1.13x/1.11x.

The applied multipliers are recorded on the chi product as
`fp07_tau_scale_<channel>` (only for probes actually scaled, so an untouched
file carries no misleading `1.0`) alongside `fp07_tau_model`, so a chi file is
traceable to the τ that produced it without the config.

The value is a **multiplier on the model**, not an absolute τ, so the model's
speed dependence survives — a scale factor on `τ(U)` is what the data
constrain. A typo'd probe name **raises** rather than being ignored: a
silently-uncorrected probe would bias chi by the square of the response error
with no downstream symptom.

Caveats before reaching for it:

- **`fom` will not tell you which bead to trust.** On ARCTERX-2022 it was
  ~1.000 for *both* beads on both units — the fit is formally excellent for
  each, they simply fit to chi values a factor apart. `chi.fom_max` is blind
  to this.
- **It makes the beads agree with each other; it does not make either
  right.** Anchor the absolute level separately — e.g. restrict to
  `K_max_ratio > 1.2`, where extrapolation is smallest, and test the
  Osborn-Cox balance against the shear-probe epsilon.
- **Fit it per method.** The τ that unifies the beads under Method 1
  (`use_epsilon: true`, kB fixed by the shear epsilon) is not identical to the
  Method 2 value, because the two methods weight the spectrum differently.
- **Check it is stable in time.** A τ that drifts over a deployment is
  fouling or damage, not a probe constant, and a single value is then wrong.

---

## Diagnostics

Several sections include a `diagnostics` flag. When set to `true`, additional diagnostic variables (standard deviations, sample counts, etc.) are included in the output. Toggling diagnostics does **not** create a new output directory — the hash excludes the `diagnostics` key.
