# Perturb plotting (`perturb-plot`)

Figures from a perturb run's aggregated (combo) outputs. Each plot is a
subcommand of the `perturb-plot` console script:

```
perturb-plot <subcommand> [options]
```

| Subcommand | Description |
|------------|-------------|
| `figure`   | Render many figures from one YAML spec (presets `scalar`/`profiles`/`epsilon`/`chi`/`mixing`/`eps-chi`), resolving directories from a perturb config. |
| `eps-chi`  | Pcolor of log10(epsilon), log10(chi), log10(chi/epsilon) vs depth and cast number (reads `diss_combo`/`chi_combo`). |
| `scalar`   | Depth-vs-x scalar sections (T / S / density / ...) from the CTD combo, with selectable x-axis. |
| `profiles` | Depth-vs-x sections of the binned slow channels (`T1`/`T2`/`N2`/`dTdz`) from `combo_NN`, one column per cast. |
| `epsilon`  | Depth-vs-x sections of binned epsilon (`epsilonMean`) from `diss_combo_NN`. |
| `chi`      | Depth-vs-x sections of binned chi (`chiMean`) from `chi_combo_NN`. |
| `mixing`   | Depth-vs-x sections of binned mixing (`K_T`/`Gamma`/`K_rho`) from `chi_combo_NN`. |

`profiles`, `epsilon`, `chi`, and `mixing` share one engine (they differ only in
which `(bin, profile)` product and default variables they read), so every
section / x-axis / color / QC option below applies identically to all four.

Each subcommand finds its input directory either by **`--config perturb.yaml`**
(the directory whose stored config-hash signature matches that config) or, with
**`--root DIR`**, the newest versioned `{stage}_NN` under it. `--config` is
exact — it never silently picks a different config's product. On a non-exact
match it errors (or, with a single candidate, warns and uses it); `--strict`
forces the error, `--latest` forces the newest.

## `perturb-plot figure`

One YAML lists figures, each naming a **preset** (the subcommands above) plus
that subcommand's own options; the driver compiles each entry into the chosen
subcommand and runs it (every kernel and behavior is reused). With no output
destination the figures are **shown on screen** when a display is available
(falling back to a PNG tree in the cwd when it is not — headless / no tty / a
non-GUI backend), mirroring the subcommands. Set a destination to force files:
one PNG tree (`output_dir`, one subdirectory per figure) **or** one combined
multipage PDF (`output_pdf`, one page per figure the preset produces).

```yaml
source:                          # exactly one of config / root
  config: perturb.yaml           # resolve directories from this config
  # output_root: ~/Desktop/VMP_results   # optional: override where to look
output_dir: figs/                # one PNG tree (XOR output_pdf)
# output_pdf: report.pdf         # ...or one combined multipage PDF
dpi: 150                         # optional default for any figure without its own
sections:                        # optional; file ref or an inline list
  file: sections.yaml
figures:
  - {name: ts,       preset: scalar,  vars: [JAC_T, SP, sigma0], depth_max: 150,
     figsize: [11, 9], title: "T/S overview", dpi: 200}
  - {name: mixing,   preset: mixing,  vars: [K_T, Gamma, K_rho]}
  - {name: overview, preset: eps-chi, gap_seconds: 600}
```

- A figure's keys are the chosen preset's options (`vars` is sugar for repeated
  `--var`; `clim` is a `{VAR: [min, max]}` map). Each binned product is its own
  preset (`profiles`/`epsilon`/`chi`/`mixing`) — there is no `product:` key.
  `section` selects from the `sections` block by name, a list, or `"*"` (all).
  `eps-chi` has no x-axis, so `section`/`vars`/`clim` are rejected for it.
- **Output controls** every preset accepts: `figsize: [w, h]` (inches), `title`,
  and `dpi` (raster resolution; the top-level `dpi` is the default, a figure's
  own `dpi` wins). `title` replaces the auto suptitle for the section presets;
  for `eps-chi` it sets the title *prefix* (the spectrum/method/QC summary is
  still appended). Set exactly one of top-level `output_dir` / `output_pdf`.
- Values are validated exactly as the CLI would parse them. Boolean keys take
  `true`/`false` (YAML 1.2; not `yes`/`no`), and a fixed-count option such as
  `figsize: [w, h]` or `point: [lat, lon]` must be a list of that exact length.
- `--figure NAME` renders only the named figure(s) from the spec's `figures:`
  list (repeatable). Separately, `--sections SECTIONS.YAML` overrides the spec's
  `sections:` block and `--select NAME` renders only those section(s) — narrowing
  every figure's own `section:`, exactly as `perturb-plot scalar --select` does
  (`eps-chi` has no x-axis and ignores it). `--strict`/`--latest` pass through to
  config resolution.
- **Compare runs without editing the spec.** The command line overrides the
  spec's `source` and output destination: `--config PERTURB.YAML` (or `--root
  DIR`) re-points the whole spec at another perturb run, and `--output-dir DIR`
  / `--output-pdf PDF` redirect where the figures land. `--config`/`--root` are
  mutually exclusive, as are `--output-dir`/`--output-pdf`.
- `perturb-plot figure --list-presets` and `--dump-preset NAME` print copyable
  example specs.

```bash
perturb-plot figure --spec figure.yaml
perturb-plot figure --dump-preset scalar > my_figure.yaml   # copy and edit

# Same spec, two perturb runs, side-by-side outputs:
perturb-plot figure --spec figure.yaml --config perturb.1.yaml --output-dir figs/run1
perturb-plot figure --spec figure.yaml --config perturb.2.yaml --output-dir figs/run2
```

---

## `perturb-plot scalar`

Renders depth (y, inverted) against a chosen x-axis with a CTD scalar in
color, from the CTD trajectory product `ctd_combo_NN/combo.nc`.

By default the figures are shown on screen. They are written to PNG files
instead when `--out-dir` is given, or when no interactive display is available
(no controlling tty, or a non-GUI matplotlib backend such as Agg) — in which
case they go to `--root`.

That product is a **continuous down/up sawtooth** on a `time` axis (the full
file, both cast directions, not profile-segmented). The section is built by
binning the scattered `(x, depth, value)` samples onto a regular grid and
**averaging each cell** — empty cells stay blank (light gray). Nothing is
interpolated across unsampled gaps, so the figure never paints values into
water the vehicle did not sample.

### Sections vs rendering

A **section** is only a way of *chopping the trajectory and choosing the
x-axis*: a name, an optional UTC time window, and an `xaxis` method with its
parameters. *Rendering* choices — which variables, depth/x bin sizes, color
limits — are separate CLI flags that apply to every section in the run.

Provide sections either from a YAML file (`--sections`) or as a single ad-hoc
section built from CLI flags.

### x-axis methods

| `method` | x-axis | Parameters |
|----------|--------|------------|
| `time` | sample time (UTC) | — |
| `latitude` | latitude (degrees N) | — |
| `longitude` | longitude (degrees E) | — |
| `distance_from_point` | great-circle distance from a fixed point | `point: [lat, lon]`, `units` |
| `along_line` | distance projected onto a waypoint polyline | `waypoints: [[lat, lon], ...]` (>= 2), `units` |
| `signed_distance` | signed distance from the track midpoint along the points' principal axis (earliest negative, latest positive); the axis label reports the midpoint lat/lon and the section orientation ± circular std relative to true north | `units` |

`units` is `m`, `km` (default), or `nm`. Distances use a TEOS-10-consistent
great-circle (haversine) from the fixed point; `along_line` projects each
sample onto the polyline in a local equirectangular plane (accurate to well
under 1% over a few-hundred-km transect).

> **Honesty note.** `time` is the only x-axis with no aliasing: it is strictly
> monotonic and every column is one moment. The spatial axes
> (`latitude`/`longitude`/`distance_from_point`/`along_line`) assume a single
> monotonic transect — if a track re-occupies the same x more than once,
> those occupations are averaged into the same column. For a normal cast
> sequence the only re-occupation is a cast's own down- and up-legs meeting at
> the bottom turn, which is the intended averaging.

### sections.yaml

```yaml
sections:
  - name: north_transect
    start: "2025-01-20T00:00:00Z"   # optional; UTC (trailing Z or none)
    stop:  "2025-01-22T00:00:00Z"   # optional
    xaxis:
      method: along_line
      units: km
      waypoints: [[18.5, 130.0], [20.0, 132.5]]   # [lat, lon]

  - name: full_timeseries
    xaxis: {method: time}

  - name: by_distance
    xaxis: {method: distance_from_point, units: km, point: [15.2, 145.7]}
```

This is a standalone plot config — it is intentionally **not** the perturb
pipeline configuration and is not validated against it.

### Options

| Flag | Description |
|------|-------------|
| `--root DIR` | perturb output root (contains `ctd_combo_NN/`). Required. |
| `--ctd-combo PATH` | Explicit combo dir or `combo.nc` (default: latest under `--root`). |
| `--sections YAML` | Sections file. If omitted, one ad-hoc section is built from the flags below. |
| `--select NAME` | Plot only the named section(s) from `--sections`, by their `name:` in the YAML (repeatable, or comma-separated). Default: every section. An unknown name is an error. |
| `--out-dir DIR` | Write `scalar_<name>.png` here instead of showing on screen. Omit to display interactively (figures fall back to `--root` when no display is available). |
| `--var NAME` | Scalar variable to panel (repeatable). Default: `JAC_T`, `SP`, `sigma0`, `rho`, plus `DO`/`Chlorophyll`/`Turbidity` when present, arranged in a 2-column grid (`--ncols` overrides). |
| `--z-bin M` | Depth bin width in meters (default 1.0). |
| `--x-bin U` | x bin width in x-axis units (default: ~300 columns). |
| `--depth-max M` | Clip the depth axis at this value. |
| `--vmin` / `--vmax` | Color-scale limits. Apply **only with a single `--var`** (error otherwise); for multiple variables use `--clim`. Default: inner 1/99 percentile. |
| `--clim VAR MIN MAX` | Per-variable color limits (repeatable), e.g. `--clim JAC_T 18 28 --clim SP 34.5 34.9`. Wins over `--vmin`/`--vmax` for that variable. |
| `--xaxis METHOD` | X-axis method. Without `--sections`, builds one ad-hoc section (default `time`). With `--sections`, an explicit `--xaxis` **overrides every section's x-axis** (keeping each section's name + window); spatial methods then take `--point`/`--waypoints`/`--units` from the CLI. |
| `--start` / `--stop` | Ad-hoc UTC window. |
| `--point LAT LON` | Reference point for ad-hoc `distance_from_point`. |
| `--waypoints "lat,lon;lat,lon;..."` | Polyline for ad-hoc `along_line`. |
| `--units {m,km,nm}` | Distance units for spatial x-axes (default `km`). |

Default density panels are `sigma0` (potential density **anomaly**) and `rho`
(in-situ density − 1000 kg/m³, stored and labeled as such). Color limits are the
inner 1/99 percentile (sign-aware, so a near-surface negative `sigma0` is not
clipped). The salinity (`SP`) and density (`sigma0`) colorbars run min-at-top
to max-at-bottom, mirroring the depth axis (both increase with depth).

### Examples

```bash
# Quick time section of the default scalars, shown on screen
perturb-plot scalar --root ~/Desktop/vmp_results

# A latitude section of temperature only, top 150 m
perturb-plot scalar --root RESULTS --xaxis latitude --var JAC_T --depth-max 150

# Multiple named sections from a file, written to PNGs
perturb-plot scalar --root RESULTS --sections sections.yaml --out-dir figs/

# Just two named sections from the file
perturb-plot scalar --root RESULTS --sections sections.yaml \
    --select north_transect --select along_NE_line
```

---

## `perturb-plot profiles` / `epsilon` / `chi` / `mixing`

Depth-vs-x sections from the binned **`(bin, profile)`** products — one column
per cast. Unlike `scalar` (which grids a continuous trajectory), these products
are already binned by depth with one `lat`/`lon`/`stime` per cast, so each
profile is a single column placed at its x-position. Columns are sorted by x
and drawn one mesh per x-cluster, leaving **blank gaps** where sampling is
sparse (never stretched across unsampled water/time).

These four subcommands share one engine, one per product. Each defaults to a
multi-panel overview in a 3-column grid (`--ncols` overrides):

| Subcommand | Reads | Default variables (3-column grid) | Scale |
|------------|-------|-------------------|-------|
| `profiles` | `combo_NN` | `JAC_T`, `T1`, `T2`, `SP`, `rho`, `sigma0`, `W_slow`, `dTdz`, `N2` | T linear, N2 symlog, dTdz diverging |
| `epsilon`  | `diss_combo_NN` | `speed`, `nu`, `T_mean`, `e_1`, `e_2`, `epsilonMean`, `N2`, `dTdz` | epsilon log |
| `chi`      | `chi_combo_NN` | `speed`, `nu`, `T_mean`, `chi_1`, `chi_2`, `chiMean`, `N2`, `dTdz`, `qc_drop_chi` | chi log |
| `mixing`   | `chi_combo_NN` **+** `diss_combo_NN` | `e_1`, `e_2`, `epsilonMean`, `chi_1`, `chi_2`, `chiMean`, `K_T`, `K_rho`, `Gamma` | log |

`mixing` merges the two combos on their shared `(bin, profile)` grid — `e_*`
come from the diss combo, chi/K/Gamma from the chi combo. If the diss combo is
absent (a chi-only run) the `e_*` panels are simply dropped. Its QC is the
**union** of `qc_drop_epsilon` and `qc_drop_chi` (a cell is masked if either
flags it).

Any combo variable can be panelled with `--var` (e.g. `--var e_1 --var e_2`).
The section / `--sections` / `--select` / `--xaxis`-override / `--clim` /
display options are identical to `scalar`. Additional options (all four
subcommands):

| Flag | Description |
|------|-------------|
| `--p-max M` | Clip the depth axis at this value [m]. |
| `--gap-factor N` | Split casts into clusters when the x-gap exceeds N× the median cast spacing (default 4). |
| `--apply-qc` / `--no-qc` | NaN cells flagged by the product's `qc_drop_*` field (default on). |

The vertical axis is **depth (m)** — the binned products' `bin` coordinate is
converted to depth at write time via `gsw.z_from_p` (using each profile's own
latitude, or a mid-latitude default when absent), so `bin:units="m"` is correct.
Dissipation/chi/diffusivity/`N2` panels use a log scale; `dTdz` and
inclinometers are diverging; temperatures are linear.

### Diagnostic pseudo-variables

Shear / vibration / temperature-gradient **variance** is not stored in any
product — it lives in the raw fast channels of the per-profile files
(`profiles_NN/*_prof*.nc`). Request it with `--var <channel>_var`:

| Pseudo-variable | Raw channel | Quantity |
|-----------------|-------------|----------|
| `sh1_var`, `sh2_var` | `sh1`/`sh2` | shear variance [s⁻²] |
| `Ax_var`, `Ay_var` | `Ax`/`Ay` | vibration variance [counts²] |
| `T1_dT1_var`, `T2_dT2_var` | `T1_dT1`/`T2_dT2` | gradient-channel variance [K²] |

These are computed **at plot time**: each cast is matched to its raw file by
`stime`, the channel is high-pass filtered and despiked exactly as the epsilon
path does (`--hp-cut`, `--despike-thresh`, `--despike-smooth`), and the
time-variance is taken in each depth bin (log-scaled). They are
**contamination / activity diagnostics, not turbulence quantities** — `Ax`/`Ay`
are raw piezo *counts* (instrument-relative), and shear variance is related to
but not equal to ε. Reading the raw files makes these panels slower than the
stored variables; restrict the section window to keep it quick.

```bash
# Epsilon beside shear/vibration/T-gradient variance, one cluster
perturb-plot epsilon --root RESULTS \
    --start 2025-02-04T00:00:00Z --stop 2025-02-06T00:00:00Z \
    --var epsilonMean --var sh1_var --var Ax_var --var T1_dT1_var
```

### Examples

```bash
# Epsilon vs cast/time, on screen
perturb-plot epsilon --root RESULTS

# Mixing (K_T, Gamma, K_rho) along a latitude transect, written to PNGs
perturb-plot mixing --root RESULTS --xaxis latitude --out-dir figs/

# Chi by signed distance, top 150 m, custom color limits
perturb-plot chi --root RESULTS --xaxis signed_distance \
    --p-max 150 --clim chiMean 1e-10 1e-6
```
