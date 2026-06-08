# VMP Data Processing Pipeline

Processing levels for turbulent kinetic energy (TKE) dissipation and thermal
variance dissipation (chi) from Rockland Scientific vertical microprofilers.
Levels L1 through L4_dissipation follow the ATOMIX / SCOR-160 shear-probe
benchmark (Lueck et al., 2024, doi:10.3389/fmars.2024.1334327).
Chi levels and binning are extensions to the SCOR-160 framework.

![Data flow diagram](data_flow.png)

---

## CLI commands

| Command | Input | Output | Purpose |
|---------|-------|--------|---------|
| `rsi-tpw info` | `.p` files | stdout | Print .p file metadata |
| `rsi-tpw cutp` | `.p` file | Short `.p` file | Copy record ranges for debugging |
| `rsi-tpw nc` | `.p` files | Full-record `.nc` | Convert .p to NetCDF (L1) |
| `rsi-tpw prof` | `.p` or `.nc` | Per-profile `.nc` | Extract profiles |
| `rsi-tpw eps` | `.p`, `.nc`, or per-profile `.nc` | `*_eps.nc` per profile | Compute epsilon |
| `rsi-tpw chi` | `.p`, `.nc`, or per-profile `.nc` | `*_chi.nc` per profile | Compute chi |
| `rsi-tpw pipeline` | `.p` files | Multi-level directory tree | Full L1→L6 pipeline |
| `rsi-tpw ql` | `.p` files | Interactive viewer | Quick-look exploration |
| `rsi-tpw dl` | `.p` files | Interactive viewer | Dissipation quality inspection |
| `rsi-tpw init` | — | `config.yaml` | Generate template config |

---

## Full pipeline

`rsi-tpw pipeline VMP/*.p -o results/`

For each .P file, profiles are detected automatically, then each profile is
processed through L1→L6.  Levels L1 through L3 are in-memory processing steps;
only L4, L5, and L6 are written to disk.

### L1 — Converted (in-memory)

**Source:** .P binary file, one profile slice
**Module:** `rsi/adapter.py` → `pfile_to_l1data()`

Reads the .P binary, parses the header and configuration string,
demultiplexes the address matrix, converts all channels to physical units,
and slices to the detected profile window.  Pressure is interpolated from
slow to fast rate.  Profiling speed is computed from dP/dt with Butterworth
smoothing.

**Contents (L1Data):**
- Shear probes (sh1, sh2) normalised by speed: du/dz in s⁻¹
- Piezo accelerometers / vibration channels
- Fast thermistor temperature (T1, T2)
- Pressure interpolated to fast rate
- Profiling speed (from dP/dt)
- Time vectors, sampling rates

---

### L2 — Cleaned (in-memory)

**Source:** L1Data
**Module:** `scor160/l2.py` → `process_l2()`

**Processing:**
- High-pass filter shear and vibration (default HP_cut = 0.25 Hz)
- Despike shear channels (iterative median filter, default threshold = 8 MAD)
- Section selection (minimum speed, depth, duration)

---

### L3 — Shear spectra (in-memory)

**Source:** L2Data + L1Data
**Module:** `scor160/l3.py` → `process_l3()`

**Processing:**
- Divide sections into overlapping dissipation windows
- FFT + Welch method (cosine window, 50% overlap within each window)
- Goodman coherent noise removal using accelerometer spectra
- Macoun–Lueck spatial response correction
- Convert frequency → wavenumber using mean segment speed

**Output (L3Data):**
- Shear wavenumber spectra per probe per window (raw and cleaned)
- Vibration/accelerometer spectra
- Wavenumber vector, mid-window pressure/temperature/speed

---

### L4 — Epsilon (written to disk)

**Source:** L3Data
**Module:** `scor160/l4.py` → `process_l4()`
**Output file:** `profile_NNN/L4_epsilon.nc`

TKE dissipation rate ε estimated by fitting the Nasmyth spectrum.

**Processing:**
- Per probe, per window: initial ε estimate via spectral integration
- Variance method (low ε) or inertial subrange (ISR) method (high ε),
  selected by the e_ISR_threshold criterion
- QC metrics: figure of merit (FOM), mean absolute deviation (MAD),
  Lueck figure of merit (FM), K_max_ratio, fraction of variance resolved
- Final ε: geometric mean across passing probes

**Variables:**

| Variable | Dimensions | Description |
|----------|-----------|-------------|
| epsilon | (probe, time) | ε per shear probe |
| epsilon_final | (time,) | Selected best ε |
| fom | (probe, time) | Figure of merit |
| mad | (probe, time) | Mean absolute deviation |
| kmax | (probe, time) | Upper integration wavenumber |
| var_resolved | (probe, time) | Fraction of variance resolved |
| sea_water_pressure | (time,) | Mean pressure per window |
| pspd_rel | (time,) | Profiling speed |

---

### Chi processing chain

From L2, the chi chain runs its own L2_chi and L3_chi cleaning and spectral
computation, then branches into two methods at L4.

#### L2_chi — Temperature cleaning (in-memory)

**Source:** L1Data + L2Data
**Module:** `chi/l2_chi.py` → `process_l2_chi()`

**Processing:**
- Despike fast thermistor temperature (default threshold = 10 MAD)
- Compute temperature gradient: dT/dz = fs·diff(T) / speed
- HP-filter vibration for Goodman

#### L3_chi — Temperature gradient spectra (in-memory)

**Source:** L2ChiData
**Module:** `chi/l3_chi.py` → `process_l3_chi()`

**Processing:**
- Same windowing as epsilon L3
- FFT + Welch on temperature gradient
- Goodman coherent noise removal (if accelerometers present)
- FP07 transfer function correction (single-pole or double-pole)
- First-difference and bilinear corrections

#### L4_chi_epsilon — Chi from epsilon (Method 1)

**Source:** L3ChiData + L4Data
**Module:** `chi/l4_chi.py` → `process_l4_chi_epsilon()`
**Output file:** `profile_NNN/L4_chi_epsilon.nc`

Thermal variance dissipation rate χ computed using ε from L4.
The Batchelor wavenumber is fixed by the shear-derived ε, and the
Batchelor/Kraichnan spectrum is fit to the observed temperature gradient
spectrum.

#### L4_chi_fit — Chi from spectral fit (Method 2, optional)

**Source:** L3ChiData
**Module:** `chi/l4_chi.py` → `process_l4_chi_fit()`
**Output file:** `profile_NNN/L4_chi_fit.nc`

Thermal variance dissipation rate χ computed without ε.  A Kraichnan
model is fit directly to the temperature gradient spectrum by jointly
estimating χ and the Batchelor wavenumber (MLE or iterative method).

**L4_chi variables (both methods):**

| Variable | Dimensions | Description |
|----------|-----------|-------------|
| chi | (probe, time) | χ per thermistor |
| chi_final | (time,) | Selected best χ |
| epsilon_T | (probe, time) | ε implied by chi |
| kB | (probe, time) | Batchelor wavenumber |
| K_max | (probe, time) | Upper integration wavenumber |
| fom | (probe, time) | Figure of merit |
| K_max_ratio | (probe, time) | K_max / kB ratio |
| sea_water_pressure | (time,) | Mean pressure per window |
| pspd_rel | (time,) | Profiling speed |

---

### L5 — Depth-binned (written to disk)

**Source:** L4 epsilon + L4 chi
**Module:** `rsi/binning.py` → `bin_by_depth()`
**Output file:** `profile_NNN/L5_binned.nc`

Epsilon and chi estimates from L4 are depth-binned together into a single
dataset.  Log-normal variables (epsilon, chi) use geometric mean; others
use arithmetic mean.

**Dimensions:** `(depth_bin,)` — bin centers in dbar (default 1 dbar bins)

---

### L6 — Combined (written to disk)

**Source:** L5 datasets from all profiles across all .P files
**Module:** `rsi/combine.py` → `combine_profiles()`
**Output file:** `<p_file_stem>/L6_combined.nc`

Aligns all profiles to a common depth grid (union of bin centers) and
stacks into a multi-profile dataset.

**Dimensions:** `(profile, depth_bin)`

---

## Modular commands

The modular commands (`nc`, `prof`, `eps`, `chi`) can be run independently
and accept input at any pipeline stage (.p files, full-record .nc, or
per-profile .nc).

### `rsi-tpw cutp` — Copy debug records

```
.P file → byte-level record slice → shorter .P file
```

Copies the original header/config block and a contiguous range of complete
data records. This is useful for small debugging fixtures and reproductions,
not for pressure- or profile-aware science processing. Header metadata is
preserved unchanged, so absolute time is correct only for `--start 0`; for
`--start N>0`, data is shifted relative to the copied timestamp. The header
record count is not authoritative because local readers derive the count from
file size.

### `rsi-tpw nc` — Convert to NetCDF

```
.P file → p_to_L1() → full-record .nc
```

Produces one NetCDF file per .P file containing all channels in physical
units.  This is the starting point for downstream modular commands.

### `rsi-tpw prof` — Extract profiles

```
.P or .nc → get_profiles() → per-profile .nc files
```

Detects profiles by pressure/fall-rate criteria and writes one NetCDF per
profile.

### `rsi-tpw eps` — Compute epsilon

```
.P, .nc, or per-profile .nc → L1→L2→L3→L4 → *_eps.nc per profile
```

Runs the SCOR-160 shear spectral processing chain and writes per-profile
epsilon files.  Supports parallel processing (`-j N`).

### `rsi-tpw chi` — Compute chi

```
.P, .nc, or per-profile .nc → L2_chi→L3_chi→L4_chi → *_chi.nc per profile
```

Optionally reads epsilon files from `--epsilon-dir` for Method 1.
Without epsilon, uses Method 2 (spectral fitting).

---

## Interactive viewers

### `rsi-tpw ql` — Quick Look

Loads a .P file and opens a 2×4 panel interactive matplotlib viewer.
For each profile, computes spectra at a single dissipation window near
the midpoint pressure.  Panels: profile overview, shear spectra with
Nasmyth fits, chi spectra with Method 1 and Method 2 fits, shear
channels, temperature, and fall rate.

### `rsi-tpw dl` — Dissipation Look

Loads a .P file and opens a 2×4 panel interactive matplotlib viewer.
Pre-computes windowed dissipation for all windows in the profile.
Panels: epsilon vs depth, chi (Batchelor and Kraichnan) vs depth,
Lueck figure of merit (FM), shear/chi spectra at midpoint.

Both viewers support Prev/Next profile navigation via buttons.

---

## Output directory structure

### Pipeline output

```
results/
└── <p_file_stem>/
    ├── L6_combined.nc ............. All profiles combined (profile × depth_bin)
    └── profile_001/
    │   ├── L4_epsilon.nc .......... Per-window epsilon estimates
    │   ├── L4_chi_epsilon.nc ...... Chi Method 1 (if temp channels present)
    │   ├── L4_chi_fit.nc .......... Chi Method 2 (if enabled)
    │   └── L5_binned.nc ........... Depth-binned epsilon + chi
    └── profile_002/
        ├── ...
```

### Modular command output

```
# rsi-tpw nc
nc/<p_file_stem>.nc                     Full-record NetCDF

# rsi-tpw prof
profiles/<p_file_stem>_prof001.nc       Per-profile NetCDF
profiles/<p_file_stem>_prof002.nc

# rsi-tpw eps
epsilon/<p_file_stem>_prof001_eps.nc    Per-profile epsilon
epsilon/<p_file_stem>_prof002_eps.nc

# rsi-tpw chi
chi/<p_file_stem>_prof001_chi.nc        Per-profile chi
chi/<p_file_stem>_prof002_chi.nc
```

---

## Configuration

The `rsi-tpw` CLI supports YAML configuration files with three sections:
`profiles`, `epsilon`, and `chi`.  Generate a template with `rsi-tpw init`.

**Merge priority:** defaults ← config file ← CLI flags

```yaml
profiles:
  P_min: 0.5            # Minimum pressure [dbar]
  W_min: 0.3            # Minimum fall rate [dbar/s]
  direction: down        # Profile direction
  min_duration: 7.0      # Minimum duration [s]

epsilon:
  fft_length: 1024       # FFT segment length [samples]
  diss_length: null      # Dissipation window (default: 4×fft_length)
  goodman: true          # Goodman coherent noise removal
  f_AA: 98.0             # Anti-aliasing filter cutoff [Hz]
  fit_order: 3           # Polynomial fit order
  salinity: null         # Fixed salinity (null → S=35)

chi:
  fft_length: 1024
  fp07_model: single_pole  # FP07 transfer function model
  spectrum_model: kraichnan
  fit_method: iterative    # Method 2: mle or iterative
  goodman: true
  f_AA: 98.0
  salinity: null
```

---

## Library architecture

### Package structure

```
src/odas_tpw/
├── rsi/          RSI instrument I/O, pipeline orchestration, CLI, viewers
├── chi/          Chi computation library (instrument-agnostic)
├── scor160/      SCOR-160 shear spectral processing (instrument-agnostic)
└── perturb/      Campaign-level processing pipeline (separate entry point)
```

### Module roles

The `scor160/` and `chi/` subpackages provide pure spectral processing
functions with no instrument dependencies.  The `rsi/` subpackage bridges
RSI-specific I/O (PFile, address matrix, channel conversion) to the
generic processing chain:

| Layer | Modules | Role |
|-------|---------|------|
| I/O | `rsi/p_file.py`, `rsi/channels.py` | Read .P binary, convert to physical units |
| Adapter | `rsi/adapter.py` | PFile → L1Data bridge |
| Processing | `scor160/l2.py`, `scor160/l3.py`, `scor160/l4.py` | SCOR-160 shear chain (L2→L3→L4) |
| Processing | `chi/l2_chi.py`, `chi/l3_chi.py`, `chi/l4_chi.py` | Chi chain (L2_chi→L3_chi→L4_chi) |
| Spectral | `scor160/spectral.py`, `scor160/goodman.py` | CSD estimation, coherent noise removal |
| Models | `scor160/nasmyth.py`, `chi/batchelor.py`, `chi/fp07.py` | Theoretical spectra, transfer functions |
| Physics | `scor160/ocean.py` | Seawater viscosity, density, buoyancy frequency |
| Orchestration | `rsi/pipeline.py`, `rsi/dissipation.py`, `rsi/chi_io.py` | Pipeline, modular file-level entry points |
| Post-processing | `rsi/binning.py`, `rsi/combine.py` | L5 depth binning, L6 profile combining |
| Viewers | `rsi/quick_look.py`, `rsi/diss_look.py`, `rsi/viewer_base.py` | Interactive matplotlib viewers |
| CLI | `rsi/cli.py`, `rsi/config.py` | Argument parsing, config merge |

### Perturb subpackage

`src/odas_tpw/perturb/` provides a separate campaign-level pipeline with
additional steps (GPS injection, hotel data, FP07 calibration, CT alignment)
not present in the standard `rsi-tpw pipeline`.  It has its own CLI and
configuration system.

---

## References

- Lueck, R. G., et al. (2024). "Recommendations for shear-probe benchmark
  datasets." *Frontiers in Marine Science*, 11, 1334327.
  doi:[10.3389/fmars.2024.1334327](https://doi.org/10.3389/fmars.2024.1334327)

- Bluteau, C. E., Wain, D., Mullarney, J. C., & Stevens, C. L. (2025).
  "Best practices for estimating turbulent dissipation from oceanic
  single-point velocity timeseries observations." *EGUsphere* (preprint).
  doi:[10.5194/egusphere-2025-4433](https://doi.org/10.5194/egusphere-2025-4433)
