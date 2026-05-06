# Output Directory Structure

The `rsi-tpw` CLI uses a sequential, hash-tracked output directory scheme for the `eps`, `chi`, `prof`, and `pipeline` subcommands. This ensures **reproducibility** — every output directory records the exact parameters used to produce it — and **deduplication** — re-running with the same parameters reuses the existing directory rather than creating a new one.

## Directory layout

```
results/
  eps_00/
    .params_sha256_<64-char-hex>      # parameter signature file
    config.yaml                        # resolved configuration (human-readable)
    SN479_0001_eps.nc
    SN479_0002_eps.nc
    ...
  eps_01/                              # different epsilon params → new directory
    .params_sha256_<different-hex>
    config.yaml
    ...
  chi_00/
    .params_sha256_<64-char-hex>
    config.yaml
    SN479_0001_chi.nc
    ...
  prof_00/
    .params_sha256_<64-char-hex>
    config.yaml
    SN479_0005_prof_01.nc
    ...
```

The `-o/--output` flag specifies the **base directory**. Within it, sequential subdirectories are created with the pattern `{prefix}_{NN}`, where:

- **prefix** is `eps`, `chi`, or `prof` (matching the subcommand)
- **NN** is a zero-padded two-digit sequence number starting at `00`

## How directories are matched

Each output directory contains a hidden **signature file** named `.params_sha256_<hash>`, where `<hash>` is the full 64-character SHA-256 hex digest of the canonicalized parameters.

When you run a command:

1. The resolved parameters (defaults ← config file ← CLI flags) are canonicalized into a deterministic JSON string
2. The SHA-256 hash of that string is computed
3. Existing `{prefix}_NN` directories under the base are scanned for a matching `.params_sha256_<hash>` file
4. **If found** → that directory is reused (new output files are written there)
5. **If not found** → the next sequential directory is created with a new signature file

This means:

```bash
# First run: creates results/eps_00/
rsi-tpw eps VMP/*.p -o results/

# Same params: reuses results/eps_00/
rsi-tpw eps VMP/*.p -o results/

# Different params: creates results/eps_01/
rsi-tpw eps VMP/*.p -o results/ --fft-length 512

# Original params again: reuses results/eps_00/
rsi-tpw eps VMP/*.p -o results/ --fft-length 1024
```

## Touchfile contents

The signature file is not just a marker — it contains the **canonical JSON** representation of the full parameter set. This makes it possible to inspect exactly which parameters produced a given directory:

```bash
cat results/eps_00/.params_sha256_*
# {"despike_smooth":0.5,"despike_thresh":8,"direction":"auto","diss_length":null,...}
```

## Resolved config.yaml

Each output directory also contains a `config.yaml` file with the resolved parameters in human-readable YAML format. This is the configuration as actually used, after merging defaults, the config file, and CLI flags.

## Parameter merge precedence

Parameters are resolved in three layers, from lowest to highest priority:

```
function defaults  ←  config.yaml values  ←  CLI flags
```

- **Function defaults** are the built-in values (e.g., `fft_length: 1024` for epsilon)
- **Config file values** (`-c config.yaml`) override defaults for any key that is set to a non-null value
- **CLI flags** (e.g., `--fft-length 512`) override both defaults and config values

A `null` value in the config file means "use the default" — it does not mask the default.

## Hash algorithm

SHA-256, using the full 64-character hex digest. The canonicalization process:

1. Start with section defaults (e.g., all epsilon parameters with their default values)
2. Overlay any non-null user-specified parameters
3. Normalize types: booleans stay booleans, integer-valued floats become ints, floats are rounded to 10 decimal places
4. Encode as compact sorted JSON: `json.dumps(d, sort_keys=True, separators=(',',':'))`
5. Hash: `hashlib.sha256(json_string.encode()).hexdigest()`

Hashes are **cumulative** — downstream steps include all upstream parameters. For example, in the `pipeline` subcommand, the chi hash includes both chi parameters and the epsilon parameters used to compute the upstream epsilon. This means if you change an epsilon parameter, both the `eps_NN/` and `chi_NN/` directories will change, correctly reflecting that the chi results depend on different upstream epsilon values.

For standalone commands (`rsi-tpw eps`, `rsi-tpw chi` without `--epsilon-dir`), the hash includes only that section's parameters since there is no upstream dependency tracked by the tool.

## Pipeline subcommand

The `pipeline` subcommand creates a per-file, per-profile directory structure:

```bash
rsi-tpw pipeline VMP/*.p -o results/
# Creates:
#   results/{pfile_stem}/
#     profile_001/
#       L4_epsilon.nc       # epsilon estimates
#       L4_chi_epsilon.nc   # chi from known epsilon (Method 1)
#       L4_chi_fit.nc       # chi from spectral fit (Method 2)
#     profile_002/
#       ...
#     L5_binned.nc          # depth-binned profiles
#     L6_combined.nc        # all profiles combined
```

The standalone `eps` and `chi` subcommands use the sequential hash-tracked scheme (`eps_00/`, `chi_00/` etc.) described above.

## Generating a template config

Use `rsi-tpw init` to generate a fully-commented template configuration file:

```bash
rsi-tpw init                    # writes config.yaml in current directory
rsi-tpw init my_config.yaml     # writes to specified path
rsi-tpw init --force            # overwrite existing file
```
