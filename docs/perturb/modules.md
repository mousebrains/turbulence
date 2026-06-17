# Perturb Modules

The `perturb` package (`src/odas_tpw/perturb/`) provides batch processing on top of the `rsi` science library.

| Module | Description |
|--------|-------------|
| `cli.py` | `perturb` CLI entry point with subcommands |
| `config.py` | YAML config schema (18 sections), defaults, three-way merge, deterministic hashing |
| `pipeline.py` | Pipeline orchestration: `process_file`, `run_trim`, `run_merge`, `run_pipeline` |
| `discover.py` | Glob-based `.p` file discovery with filtering |
| `trim.py` | Trim corrupt final records from `.p` files (complete files referenced in place) |
| `merge.py` | Detect and merge split `.p` files by matching config/record size |
| `fp07_cal.py` | FP07 in-situ calibration (Steinhart-Hart fit, cross-correlation lag) |
| `speed.py` | Through-water speed source (pressure-rate, EM, flight model, constant) |
| `gps.py` | GPS providers: `GPSNaN`, `GPSFixed`, `GPSFromCSV`, `GPSFromNetCDF`, `create_gps` factory |
| `hotel.py` | Hotel file support — external vehicle telemetry (speed, pitch, roll, heading, CTD from gliders/AUVs/Remus) interpolated onto instrument time axes as new channels |
| `seawater.py` | Full seawater property chain via gsw/TEOS-10 (SP, SA, CT, sigma0, rho, depth) |
| `ctd.py` | CTD time-binning with GPS interpolation and seawater properties |
| `qc_gate.py` | Per-segment QC gate — flag/NaN epsilon/chi from hotel bitfield channels |
| `qc_rules.py` | Declarative internal QC rules synthesizing uint8 flag channels |
| `binning.py` | Depth and time binning of per-profile NetCDFs |
| `combo.py` | Combined NetCDF assembly with CF/ACDD attributes |
| `netcdf_schema.py` | Variable schemas (units, long_name) for combo, chi, and CTD outputs |
| `plot/` | `perturb-plot` CLI: epsilon/chi profile and section figures |
| `logging_setup.py` | Per-stage logging configuration (per-file logs, per-run summary) |

The instrument-agnostic profile-processing helpers used by perturb live one
package over, in `src/odas_tpw/processing/`:

| Module | Description |
|--------|-------------|
| `processing/bottom.py` | Bottom-crash detection from pressure/speed/vibration |
| `processing/top_trim.py` | Surface instability trimming (variance-based) |
| `processing/ct_align.py` | CT sensor cross-correlation alignment |
| `processing/epsilon_combine.py` | Multi-probe epsilon combining (`mk_epsilon_mean`) with 95% CI filtering |
| `processing/chi_combine.py` | Multi-probe chi combining (`mk_chi_mean`) |
| `processing/mixing.py` | Background stratification (`N2`, `dTdz`, sorted/window methods) and derived mixing quantities (`K_T`, `Gamma`, `K_rho`) |
