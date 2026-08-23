# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Processing and analysis code for calculating turbulent kinetic energy (TKE) and chi (thermal dissipation rate) from Rockland Scientific vertical microprofilers and microriders. Instruments use fast temperature sensors (e.g., FP07 thermistors).

## Package: microstructure-tpw

Installable Python package (`pip install -e ".[dev]"`). Source layout: `src/odas_tpw/`.

### Subpackages

- `rsi/` — Rockland Scientific instrument I/O, NetCDF conversion, profiles, epsilon, chi orchestration
- `chi/` — Chi (thermal variance dissipation) calculation, Batchelor/Kraichnan spectra, FP07 transfer function
- `scor160/` — ATOMIX shear-probe benchmark processing (L1–L4), shared physics/spectral modules
- `processing/` — Instrument-agnostic profile processing (top_trim, bottom-crash, ct_align, mk_epsilon_mean, mk_chi_mean, mixing N²/dT/dz/Γ/K_T/K_ρ)
- `perturb/` — Full campaign processing pipeline (trim, merge, calibrate, compute, bin)
- `pyturb/` — Jesse's standalone analysis code (hosted here; `pyturb-cli` entry point)
- `fp07cal/` — FP07 in-situ calibration against a CTD (`fp07-cal` entry point). A **pre-pipeline** step: fits one Steinhart-Hart set per deployment and patches it into the `.p` files. See `docs/fp07cal/runbook.md`.

### Key Modules (rsi)

- `p_file.py` — `PFile` class: reads Rockland `.p` binary files, parses headers, demultiplexes address matrix, converts to physical units. `parse_config()` parses the embedded INI config string.
- `channels.py` — Conversion functions (raw counts → physical units) for each sensor type. `CONVERTERS` dict maps type names to functions.
- `convert.py` — `p_to_netcdf()` and `convert_all()` for writing NetCDF4 output.
- `profile.py` — Profile detection and per-profile NetCDF extraction.
- `dissipation.py` — Core epsilon calculation with multi-source input, QC metrics (fom, K_max_ratio).
- `chi_io.py` — Chi orchestration: load instrument data and call chi computation.
- `config.py` — YAML configuration loading, merging, template generation.
- `cli.py` — Unified `rsi-tpw` CLI with subcommands.

### Key Modules (chi)

- `chi.py` — Chi (thermal variance dissipation) calculation, Methods 1 and 2, QC metrics.
- `batchelor.py` — Batchelor and Kraichnan temperature gradient spectra.
- `fp07.py` — FP07 thermistor transfer function and electronics noise model.

### Key Modules (scor160)

- `spectral.py` — Cross-spectral density estimation (Welch method, cosine window).
- `goodman.py` — Goodman coherent noise removal using accelerometer spectra.
- `despike.py` — Iterative spike removal for shear probe signals.
- `nasmyth.py` — Nasmyth universal shear spectrum (Lueck improved fit).
- `ocean.py` — Seawater properties: `visc35`, `visc(T,S,P)`, `density(T,S,P)`, `buoyancy_freq(T,S,P)`, `kappa_T(T,S,P)` (thermal diffusivity for chi) via gsw (TEOS-10).

### CLI Commands

```bash
rsi-tpw info VMP/*.p                           # print .p file metadata
rsi-tpw nc VMP/*.p -o nc/                      # convert .p to NetCDF
rsi-tpw prof VMP/*.p -o profiles/              # extract profiles
rsi-tpw eps VMP/*.p -o epsilon/                # compute epsilon
rsi-tpw chi VMP/*.p -o chi/                    # compute chi (Method 2)
rsi-tpw chi VMP/*.p --epsilon-dir epsilon/ -o chi/  # chi with epsilon (Method 1)
rsi-tpw pipeline VMP/*.p -o results/           # full pipeline
rsi-tpw eps VMP/*.p --salinity 34.5 -o epsilon/  # custom salinity

fp07-cal demo                                  # exercise the chain on synthetic data
fp07-cal init -o fp07-cal.yaml                 # commented template config
fp07-cal coverage -c fp07-cal.yaml             # what CTD reference do we actually have?
fp07-cal fit -c fp07-cal.yaml                  # coefficients + stability diagnostic
fp07-cal patch -c fp07-cal.yaml                # write them into calibrated .p copies
```

### FP07 in-situ calibration (`fp07-cal`)

A **pre-pipeline** tool, run once per deployment against a CTD reference;
perturb then reads the patched files with `fp07.calibrate: false` and needs no
changes. Runbook: `docs/fp07cal/runbook.md`. Worked 72-day result:
`docs/fp07cal/osu685_stability.md`. Design and adversarial review:
`docs/fp07_insitu_calibration_PLAN.md`.

Facts that are easy to get wrong:

- The reference is read **directly from the CTD NetCDF on the CTD's own
  clock**, never through `perturb/hotel.py`. That merge interpolates across
  arbitrary gaps and edge-holds outside coverage, so on a CT that ran only some
  yos it hands the fit a fabricated ramp. Confirmed empirically: NaN-marking a
  gap produces **byte-identical** output from the perturb loader, so gap
  control applied when building a hotel file does not survive the merge.
- The thermistor is **decimated down onto real CTD sample times**, not the CTD
  interpolated up. That invents no reference, bandwidth-matches the regressor
  (so `beta_1` is not attenuated by errors-in-variables), and makes a sparse
  reference a non-event: no CTD sample, no pair.
- **Lag is gated on peak sharpness, never on `r`.** A glider dive is a
  monotonic ramp, and a shifted straight line is the same line plus a constant,
  which every correlation removes: on real data raw pressure scored
  `r = 1.000000` at *every* lag over ±30 s. Scores are computed on high-passed
  series, and a peak at the search boundary is refused.
- **`beta_2 = 0` does not delete a quadratic term** — the config value is a
  reciprocal, so zero means an infinite term and `convert_therm` raises
  `ZeroDivisionError`. `beta_2 = 1e30` is bit-identical to omitting the key.
- Sensor geometry (`dz·dT/dz`, from the FP07 and CTD sitting at different
  depths at the same instant) is fitted **jointly** with the coefficients.
  Estimating it from post-fit residuals lets the calibration absorb it into
  `t_0` — 25 cm injected came back as 0.4 cm.
- Polynomial order is chosen by **held-out** error split on temperature, not by
  in-sample fit or a t-test: `beta_3` can carry a t-statistic of 10 while
  making extrapolation four times worse.

### Python API

```python
from odas_tpw.rsi import PFile
from odas_tpw.rsi.pipeline import run_pipeline
from odas_tpw.rsi.dissipation import compute_diss_file
from odas_tpw.rsi.chi_io import compute_chi_file
from odas_tpw.scor160.ocean import visc, density, buoyancy_freq
from pathlib import Path

pf = PFile("VMP/ARCTERX_Thompson_2025_SN479_0001.p")
pf.channels["T1"]    # numpy array, physical units (°C)
pf.channels["sh1"]   # un-normalized shear intermediate (needs /speed²; s⁻¹ only after)
pf.t_fast             # time vector for fast channels
pf.fs_fast            # fast sampling rate (~512 Hz)

# Full pipeline: .p → profiles → epsilon → chi → binning → combine
run_pipeline([Path("VMP/file.p")], Path("results/"))

# Or use modular file-level functions
eps_paths = compute_diss_file("VMP/file.p", "epsilon/")
chi_paths = compute_chi_file("VMP/file.p", "chi/")

# Note: get_diss() and get_chi() still work but are deprecated
```

## Commands

```bash
pip install -e ".[dev]"       # install in editable mode with test deps
python -m pytest              # run all tests
python -m pytest tests/test_p_file.py::test_header  # single test
```

## Domain Context

- **Chi (χ)**: Thermal variance dissipation rate, computed from temperature gradient spectra. Units: K²/s.
- **TKE dissipation (ε)**: Turbulent kinetic energy dissipation rate, computed from shear probe spectra. Units: W/kg.
- **FP07**: Fast-response glass-bead thermistor. Has a known frequency response rolloff that must be corrected.
- **P file format** (RSI TN-051, rev. 2026-01-12): binary records with 128-byte headers (64 uint16 words). Record 0 = header + ASCII config string. Records 1..N = header + multiplexed int16 data. Endian flag at header word 64 (0=unknown, 1=little, 2=big). Versions 6.0–6.3 differ in timekeeping and bad-buffer handling — see `docs/rsi-tpw/odas_file_format.md`. `fs_fast` from words 21/22 is correct as read (511.9454 Hz on a 10-column VMP is the by-design rate, confirmed against ODAS MATLAB output); the §2.4.3 count-by-one bug affects **per-record synthesized timestamps**, which we never read. One open item for Rockland: the note's 38.4 MHz clock figure, which our files contradict (all rates are exact divisors of n×24 MHz).
- **fom (figure of merit)**: Ratio of observed to model variance in the spectral fit range; values near 1.0 indicate a good fit. For **epsilon** the model is the **Nasmyth** spectrum (observed/Nasmyth variance ratio; see `scor160/l4.py`). For **chi** the model is the **Batchelor/Kraichnan** spectrum convolved with the FP07 transfer function plus the noise floor (`chi/chi.py`). (Distinct from the Lueck-2022 `FM` reject statistic on the epsilon side.)
- **K_max_ratio**: K_max/K_95 (epsilon) or K_max/kB (chi). Values < 0.5 mean most variance is extrapolated.
- **Inclinometers**: on a VMP, `Incl_Y` is capped at 90° with **+90° = pointing
  straight down** (so a falling VMP reads Incl_Y ≈ +90). On an MR, `Incl_Y` is
  approximately pitch and `Incl_X` mostly roll.

## Data

- **ARCTERX campaign structure**: a three-year effort with at least six field
  campaigns (~2/year; each year split into a **Wake** effort and an
  **Interior** effort). Ships: R/V Revelle (2022), R/V Thompson (2023, 2025).
  Instruments differ per effort — e.g. ARCTERX-2025-**Interior** used VMP SNs
  **142 and 194** (`/Volumes/SeaChest/ARCTERX/2025/Interior/`), while SN
  **479** was a **Wake** effort (believed 2025). Do not assume one serial
  number spans the campaign.
- **Instrument (repo-local `VMP/` data)**: VMP-250IR_RT SN 479 (ARCTERX Wake,
  R/V Thompson, Jan 2025, Saipan)
- **Address matrix**: 8 rows × 10 cols (8 fast + 2 slow). fs_fast ≈ 512 Hz, fs_slow ≈ 64 Hz.
- `VMP/` — 30 raw `.p` files
- `odas/` — Rockland's ODAS MATLAB Library (v4.5.1), reference implementation. Key files: `odas_p2mat.m`, `read_odas.m`, `convert_odas.m`, `setupstr.m`.

## MATLAB parallel (removed)

A hand-maintained MATLAB port of the chi calculation and its comparison tests
used to live in `matlab/`. It drifted ~4 months behind the Python side and no
CI test enforced parity, so it was **removed on purpose** (commit on branch
`chore/remove-matlab-parallel`, July 2026) pending a re-sync once the Python
code base settles. This is not the vendor `odas/` library above — that stays.
The removed files remain in git history; a working copy was archived outside the
repo at `/Users/pat/tpw/turbulence-matlab-reference/` (see its `ARCHIVE_NOTE.md`
for the source commit). Revisit re-porting after the algorithms stabilize.
