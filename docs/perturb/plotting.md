# Perturb plotting (`perturb-plot`)

Figures from a perturb run's aggregated (combo) outputs. Each plot is a
subcommand of the `perturb-plot` console script:

```
perturb-plot <subcommand> [options]
```

| Subcommand | Description |
|------------|-------------|
| `eps-chi`  | Pcolor of log10(epsilon), log10(chi), log10(chi/epsilon) vs depth and cast number (reads `diss_combo`/`chi_combo`). |
| `scalar`   | Depth-vs-x scalar sections (T / S / density / ...) from the CTD combo, with selectable x-axis. |

Each subcommand discovers the latest versioned stage directory under `--root`
(e.g. `ctd_combo_01/` in preference to `ctd_combo_00/`).

---

## `perturb-plot scalar`

Renders depth (y, inverted) against a chosen x-axis with a CTD scalar in
colour, from the CTD trajectory product `ctd_combo_NN/combo.nc`.

By default the figures are shown on screen. They are written to PNG files
instead when `--out-dir` is given, or when no interactive display is available
(no controlling tty, or a non-GUI matplotlib backend such as Agg) — in which
case they go to `--root`.

That product is a **continuous down/up sawtooth** on a `time` axis (the full
file, both cast directions, not profile-segmented). The section is built by
binning the scattered `(x, depth, value)` samples onto a regular grid and
**averaging each cell** — empty cells stay blank (light grey). Nothing is
interpolated across unsampled gaps, so the figure never paints values into
water the vehicle did not sample.

### Sections vs rendering

A **section** is only a way of *chopping the trajectory and choosing the
x-axis*: a name, an optional UTC time window, and an `xaxis` method with its
parameters. *Rendering* choices — which variables, depth/x bin sizes, colour
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
| `--out-dir DIR` | Write `scalar_<name>.png` here instead of showing on screen. Omit to display interactively (figures fall back to `--root` when no display is available). |
| `--var NAME` | Scalar variable to panel (repeatable). Default: `JAC_T`, `SP`, `sigma0`, plus `DO`/`Chlorophyll`/`Turbidity` when present. |
| `--z-bin M` | Depth bin width in metres (default 1.0). |
| `--x-bin U` | x bin width in x-axis units (default: ~300 columns). |
| `--depth-max M` | Clip the depth axis at this value. |
| `--vmin` / `--vmax` | Override colour-scale limits (default: inner 1/99 percentile). |
| `--xaxis METHOD` | Ad-hoc x-axis method (default `time`). Ignored when `--sections` is given. |
| `--start` / `--stop` | Ad-hoc UTC window. |
| `--point LAT LON` | Reference point for ad-hoc `distance_from_point`. |
| `--waypoints "lat,lon;lat,lon;..."` | Polyline for ad-hoc `along_line`. |
| `--units {m,km,nm}` | Distance units for spatial x-axes (default `km`). |

Default density panel is `sigma0` (potential density **anomaly**); `rho` is
stored as in-situ density − 1000 and labelled as such. Colour limits are the
inner 1/99 percentile (sign-aware, so a near-surface negative `sigma0` is not
clipped). The salinity (`SP`) and density (`sigma0`) colormaps are reversed
relative to the cmocean defaults.

### Examples

```bash
# Quick time section of the default scalars, shown on screen
perturb-plot scalar --root ~/Desktop/vmp_results

# A latitude section of temperature only, top 150 m
perturb-plot scalar --root RESULTS --xaxis latitude --var JAC_T --depth-max 150

# Multiple named sections from a file, written to PNGs
perturb-plot scalar --root RESULTS --sections sections.yaml --out-dir figs/
```
