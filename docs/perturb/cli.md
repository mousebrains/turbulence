# Perturb CLI Reference

All commands are available through the `perturb` command:

```
perturb <subcommand> [options]
```

## Subcommands

| Subcommand | Description |
|------------|-------------|
| `perturb init`     | Generate a template YAML configuration file |
| `perturb run`      | Run the full pipeline (trim -> merge -> process -> bin -> combo) |
| `perturb trim`     | Trim corrupt final records from .p files |
| `perturb merge`    | Merge split .p files |
| `perturb profiles` | Extract per-profile NetCDFs |
| `perturb diss`     | Compute epsilon (TKE dissipation) per profile |
| `perturb chi`      | Compute chi (thermal variance dissipation) per profile |
| `perturb ctd`      | Time-bin CTD channels per file |
| `perturb bin`      | Depth/time bin profiles, diss, and chi |
| `perturb combo`    | Assemble combo NetCDFs from binned data |
| `perturb sections` | Auto-generate a plotting `sections.yaml` by splitting casts on time gaps |

## Common Options

Most subcommands share these flags:

| Flag | Description |
|------|-------------|
| `-c`, `--config YAML` | Configuration file (default: none) |
| `-o`, `--output DIR` | Output root directory |
| `-j`, `--jobs N` | Parallel workers (0=auto, 1=serial, default=config or 1) |
| `--stdout` | Also stream log records to stderr (default: log to file only) |
| `--log-level LEVEL` | DEBUG, INFO, WARNING, ERROR (default: INFO) |

> **Note** Every pipeline-running subcommand writes a log file to
> `<output_root>/logs/run_<timestamp>.log`. Console output is silent by
> default — pass `--stdout` to also see records on stderr while the run
> progresses. See [logging.md](logging.md) for the full layout
> (worker, per-stage, per-combo logs).

File-accepting subcommands also take:

| Flag | Description |
|------|-------------|
| `FILE ...` | Positional .p file paths or glob patterns |
| `--p-file-root DIR` | Root directory for .p file discovery |

## `perturb init`

Generate a template YAML configuration file with all defaults and comments.

```bash
perturb init                    # writes config.yaml
perturb init my_config.yaml     # custom filename
perturb init --force config.yaml  # overwrite existing
```

| Flag | Description |
|------|-------------|
| `--force` | Overwrite existing file |

## `perturb run`

Run the full pipeline from raw `.p` files through binning and combo assembly.

```bash
perturb run -o results/ VMP/*.p
perturb run -c config.yaml -o results/ -j 4 VMP/*002*.p
perturb run -o results/ --p-file-root /data/VMP/
```

| Flag | Description |
|------|-------------|
| `-c`, `--config YAML` | Configuration file |
| `-o`, `--output DIR` | Output root directory |
| `-j`, `--jobs N` | Parallel workers (0=auto, 1=serial) |
| `--p-file-root DIR` | Root directory for .p file discovery |
| `--hotel-file FILE` | Hotel data file (external telemetry: speed, pitch, roll, heading, CTD) |
| `FILE ...` | Explicit .p file paths or globs |

## `perturb trim`

Trim corrupt final records from `.p` files. Writes trimmed copies to `{output}/trimmed/`.

```bash
perturb trim -o results/
perturb trim -o results/ --p-file-root /data/VMP/
```

| Flag | Description |
|------|-------------|
| `--p-file-root DIR` | Root directory for .p file discovery |

## `perturb merge`

Merge split `.p` files into single files. Detects mergeable sequences automatically by matching config strings and record sizes.

```bash
perturb merge -o results/
perturb merge -o results/ --p-file-root /data/VMP/
```

| Flag | Description |
|------|-------------|
| `--p-file-root DIR` | Root directory for .p file discovery |

## `perturb profiles`

Extract per-profile NetCDFs. Runs the process stage without trim or merge.

```bash
perturb profiles -o results/ -j 4 VMP/*.p
```

## `perturb diss`

Compute epsilon (TKE dissipation) per profile. Runs the process stage without trim or merge.

```bash
perturb diss -o results/ -j 4 VMP/*.p
```

## `perturb chi`

Compute chi (thermal variance dissipation) per profile. Enables chi processing and runs the process stage.

```bash
perturb chi -o results/ -j 4 VMP/*.p
```

## `perturb ctd`

Time-bin CTD channels per file. Enables CTD processing and runs the process stage.

```bash
perturb ctd -o results/ -j 4 VMP/*.p
```

## `perturb bin`

Depth/time bin profiles, diss, and chi from previously computed output directories.

```bash
perturb bin -c config.yaml -o results/
```

Does not accept `-j` (binning runs serially).

## `perturb combo`

Assemble combo NetCDFs from binned data directories.

```bash
perturb combo -c config.yaml -o results/
```

Does not accept `-j` (combo assembly runs serially).

## `perturb sections`

Auto-generate a plotting `sections.yaml` from a completed run's profiles. Casts
usually arrive in station batches separated by transits; this reads the run's
per-profile start times from the `combo` product and starts a new section
wherever the gap between consecutive casts exceeds `--gap`. The emitted file is
the same schema `perturb-plot`/`perturb-diag` consume via `--sections` (see
[plotting.md](plotting.md) for the x-axis methods) and is validated against that
loader before it is written.

```bash
perturb sections -c perturb.yaml                       # preview on stdout (1h gap)
perturb sections -c perturb.yaml -o sections.yaml       # write the file
perturb sections -c perturb.yaml --gap 2h -o sections.yaml   # coarser batching
perturb sections -c perturb.yaml --xaxis signed_distance --units km -o sections.yaml
```

It prints the profile count, resulting section count, and the largest
inter-cast gaps to stderr so `--gap` is easy to tune. The output is a starting
point — edit the windows, rename sections, or change any section's `xaxis`
by hand afterward. Unlike the pipeline subcommands, `-o`/`--output` names a
**file** (not a directory) and there is no `-j`.

| Flag | Description |
|------|-------------|
| `-c`, `--config YAML` | Perturb config; locates the run's combo output (required) |
| `-o`, `--output FILE` | Write the sections YAML here (default: stdout) |
| `--gap DUR` | New section when the inter-cast gap exceeds this (`90m`, `1.5h`, `3600s`, `2d`; default `1h`) |
| `--xaxis METHOD` | x-axis for every section: `time` (default), `latitude`, `longitude`, `signed_distance` |
| `--units U` | Distance units for a spatial `--xaxis` (`m`/`km`/`nm`; default `km`) |
| `--pad SEC` | Pad each section's time window by this many seconds (default 30) |
| `--product P` | Combo product to read per-profile times from (default `combo`) |
| `-f`, `--force` | Overwrite `--output` if it already exists |
