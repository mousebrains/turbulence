# Python API

## Reading `.p` Files

```python
from microstructure_tpw.rsi import PFile

pf = PFile("VMP/ARCTERX_Thompson_2025_SN479_0005.p")

# Channel data (physical units)
pf.channels["sh1"]   # shear probe 1 [s⁻¹]
pf.channels["T1"]    # temperature [°C]
pf.channels["P"]     # pressure [dbar]

# Time vectors
pf.t_fast             # time vector for fast channels [s]
pf.t_slow             # time vector for slow channels [s]

# Sampling rates
pf.fs_fast            # fast sampling rate (~512 Hz)
pf.fs_slow            # slow sampling rate (~64 Hz)

# Metadata
pf.start_time         # datetime object
pf.config             # parsed INI configuration dict
pf.channel_info       # dict of channel metadata (type, units, etc.)
```

## Computing Epsilon

```python
from microstructure_tpw.rsi import get_diss

# Compute epsilon (returns list of xarray.Datasets, one per profile)
eps_results = get_diss("VMP/ARCTERX_Thompson_2025_SN479_0005.p")
ds = eps_results[0]

# Output variables
ds["epsilon"]         # dissipation rate [W/kg]
ds["spec_shear"]      # shear wavenumber spectra
ds["spec_nasmyth"]    # fitted Nasmyth spectra
ds["fom"]             # figure of merit (obs/Nasmyth variance ratio)
ds["K_max_ratio"]     # K_max/K_95 (spectral resolution)

# With options
eps_results = get_diss("VMP/file.p",
    fft_length=512,
    goodman=True,
    salinity=34.5,
    speed=0.6,
)
```

## Computing Chi

```python
from microstructure_tpw.rsi import get_chi

# Method 1: chi from known epsilon (preferred)
chi_results = get_chi("VMP/ARCTERX_Thompson_2025_SN479_0005.p",
                      epsilon_ds=eps_results[0])
ds = chi_results[0]
ds["chi"]             # thermal dissipation rate [K²/s]
ds["spec_gradT"]      # temperature gradient spectra
ds["spec_batch"]      # fitted Batchelor spectra

# Method 2: chi without epsilon (MLE fitting)
chi_results = get_chi("VMP/ARCTERX_Thompson_2025_SN479_0005.p")
ds = chi_results[0]
ds["chi"]             # thermal dissipation rate [K²/s]
ds["epsilon_T"]       # epsilon estimated from temperature
```

## Seawater Properties

```python
from microstructure_tpw.rsi import visc, density, buoyancy_freq

# Kinematic viscosity [m²/s]
nu = visc(10.0, 35.0, 100.0)  # T=10°C, S=35, P=100 dbar

# Density [kg/m³]
rho = density(10.0, 35.0, 100.0)

# Buoyancy frequency squared [s⁻²]
import numpy as np
T = np.array([20.0, 15.0, 10.0, 5.0])
S = np.full(4, 35.0)
P = np.array([0.0, 100.0, 200.0, 300.0])
N2, p_mid = buoyancy_freq(T, S, P)
```

## Processing Pipeline

The pipeline can also be driven from Python:

```python
from microstructure_tpw.rsi.convert import p_to_netcdf
from microstructure_tpw.rsi.profile import extract_profiles
from microstructure_tpw.rsi.dissipation import compute_diss_file
from microstructure_tpw.rsi.chi import compute_chi_file

# Stage 1: Convert to NetCDF
pf, nc_path = p_to_netcdf("VMP/file.p", "output/file.nc")

# Stage 2: Extract profiles
prof_paths = extract_profiles("VMP/file.p", "profiles/")

# Stage 3: Compute epsilon
eps_paths = compute_diss_file("VMP/file.p", "epsilon/", fft_length=256)

# Stage 4: Compute chi with epsilon (Method 1)
import xarray as xr
eps_ds = xr.open_dataset(eps_paths[0])
chi_paths = compute_chi_file("VMP/file.p", "chi/", epsilon_ds=eps_ds)
eps_ds.close()
```

## Modules

| Module | Description |
|--------|-------------|
| `p_file.py` | `PFile` class: reads `.p` binary files, parses headers, demultiplexes address matrix, converts to physical units |
| `channels.py` | Sensor conversion functions (raw counts to physical units) |
| `convert.py` | Full-record NetCDF export |
| `profile.py` | Profile detection and per-profile NetCDF extraction |
| `dissipation.py` | Core epsilon calculation with multi-source input |
| `chi.py` | Chi calculation, Methods 1 and 2 |
| `batchelor.py` | Batchelor and Kraichnan temperature gradient spectra |
| `fp07.py` | FP07 thermistor transfer function and electronics noise model |
| `spectral.py` | Cross-spectral density estimation (Welch method, cosine window) |
| `goodman.py` | Goodman coherent noise removal using accelerometer spectra |
| `despike.py` | Iterative spike removal for shear probe signals |
| `nasmyth.py` | Nasmyth universal shear spectrum (Lueck improved fit) |
| `ocean.py` | Seawater properties: viscosity, density, buoyancy frequency (gsw/TEOS-10) |
| `config.py` | YAML configuration file support |
