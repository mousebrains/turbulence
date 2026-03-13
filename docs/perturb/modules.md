# Perturb Modules

The `perturb` package (`src/perturb/`) provides batch processing on top of the `rsi_python` science library.

| Module | Description |
|--------|-------------|
| `cli.py` | `perturb` CLI entry point with subcommands |
| `config.py` | YAML config schema (13 sections), defaults, three-way merge, deterministic hashing |
| `pipeline.py` | Pipeline orchestration: `process_file`, `run_trim`, `run_merge`, `run_pipeline` |
| `discover.py` | Glob-based `.p` file discovery with filtering |
| `trim.py` | Trim corrupt final records from `.p` files |
| `merge.py` | Detect and merge split `.p` files by matching config/record size |
| `fp07_cal.py` | FP07 in-situ calibration (Steinhart-Hart fit, cross-correlation lag) |
| `ct_align.py` | CT sensor cross-correlation alignment |
| `gps.py` | GPS providers: `GPSNaN`, `GPSFixed`, `GPSFromCSV`, `GPSFromNetCDF`, `create_gps` factory |
| `seawater.py` | Full seawater property chain via gsw/TEOS-10 (SP, SA, CT, sigma0, rho, depth) |
| `bottom.py` | Bottom crash detection from pressure/speed/vibration |
| `top_trim.py` | Surface instability trimming (variance-based) |
| `epsilon_combine.py` | Multi-probe epsilon combining (`mk_epsilon_mean`) with 95% CI filtering |
| `ctd.py` | CTD time-binning with GPS interpolation and seawater properties |
| `binning.py` | Depth and time binning of per-profile NetCDFs |
| `combo.py` | Combined NetCDF assembly with CF/ACDD attributes |
| `netcdf_schema.py` | Variable schemas (units, long_name) for combo, chi, and CTD outputs |
