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

## Common Options

Most subcommands share these flags:

| Flag | Description |
|------|-------------|
| `-c`, `--config YAML` | Configuration file (default: none) |
| `-o`, `--output DIR` | Output root directory |
| `-j`, `--jobs N` | Parallel workers (0=auto, 1=serial, default=config or 1) |

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
| `FILE ...` | Explicit .p file paths or globs |

## `perturb trim`

Trim corrupt final records from `.p` files. Writes trimmed copies to `{output}/trimmed/`.

```bash
perturb trim -o results/
perturb trim -o results/ --p-file-root /data/VMP/
```

## `perturb merge`

Merge split `.p` files into single files. Detects mergeable sequences automatically by matching config strings and record sizes.

```bash
perturb merge -o results/
```

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
