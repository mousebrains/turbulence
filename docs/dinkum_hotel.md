# `dinkum-hotel` — Slocum Dinkum files → perturb hotel file

Turns a glider's Dinkum Binary Data files (`*.dbd`/`*.ebd`, or their
LZ4-compressed `*.dcd`/`*.ecd`) into a hotel NetCDF that
[`perturb`](perturb/configuration.md#hotel--hotel-file-external-telemetry)
merges onto an instrument's clock — most often a MicroRider riding the same
glider.

```bash
dinkum-hotel backends                       # which readers are available
dinkum-hotel sensors *.ecd -C ./cache       # what do these files actually carry?
dinkum-hotel init -o dinkum-hotel.yaml      # commented template
dinkum-hotel build -c dinkum-hotel.yaml     # write the hotel file
```

## Why a build step exists

A Slocum record carries only the sensors that reported on that cycle;
everything else is absent. And there is no single clock — three matter:

| Time sensor | What it stamps |
|---|---|
| `m_present_time` | the flight computer's cycle |
| `sci_m_present_time` | the science computer's cycle |
| `sci_ctd41cp_timestamp` | when the CTD's print **arrived** at the science computer |

So a sensor's samples are the rows where *it* is finite, timed by *its* clock.
Putting several side by side means projecting each from its own irregular
sample times onto one shared basis. That projection, plus the sanity rules
below, is what this tool does.

The raw converters (`dbd2netCDF`, `xarray-dbd`) stop one step short: they give
you every sensor on a record index, with real NaNs for absent values, but with
timestamps that are **not** monotonic — Slocum repeats the last CTD stamp on
rows the CTD did not refresh, and writes `0.0` where a field was never set. Fed
straight to an interpolator, duplicates make `pchip` raise and `linear` produce
infinite slopes.

## What the build does

1. **Sanitize the base time sensor.** Drop non-finite, drop outside the valid
   range, sort, dedupe. The result is the output time vector — the base
   sensor's own native sample times, strictly increasing. No resampling: gaps
   stay gaps.
2. **Per sensor**, pair its finite values with its own time sensor's valid
   stamps, drop out-of-range values (*before* interpolating, so a spike is
   removed rather than smeared), collapse duplicate timestamps, then
   interpolate onto the output vector.
3. Apply `scale`/`offset`, NaN across gaps wider than `max_gap`, and write —
   with a per-variable and global record of what each rule discarded.

The output time variable **keeps the base sensor's name**, so the perturb side
reads `time_column: "sci_ctd41cp_timestamp"` and means exactly that.

## The time sanity range

`time.min_value` / `time.max_value` bound *every* time sensor — the base and any
per-sensor override. Each accepts epoch seconds or an ISO-8601 date:

```yaml
time:
  min_value: 100                      # or "2025-01-15T00:00:00Z"
  max_value: "2025-06-01T00:00:00Z"
```

Left `null` they fall back to **100 s** and **now + 365 days**. The floor
rejects the `0.0` Slocum writes for "never set"; the ceiling rejects a clock
that ran away forward (an unset RTC after a battery swap). A bare ISO date with
no zone is read as UTC — guessing local time would shift the window by the
operator's offset and silently reject good data.

Note the now-relative ceiling moves between runs. Pin `max_value` (or pass
`--now`) when a rerun has to be reproducible.

## Projection methods

`projection.method` sets the default; a per-sensor `method` wins. The vocabulary
is deliberately the same as perturb's hotel loader: `linear`, `pchip`,
`nearest`, `previous`, `next`, `zero`, `slinear`, `quadratic`, `cubic`.

Use `previous` (zero-order hold) for flight state and discrete flags, where
interpolating between states invents values that never held. `linear` is right
for the CTD.

Duplicate timestamps are collapsed per `time.dedupe` (`mean` by default), but a
sensor projected with a step-like method (`previous`, `next`, `nearest`,
`zero`) defaults to `last` — the value actually in force — because a mean of
two states is a state the sensor never reported. A per-sensor `dedupe` always
wins.

`max_gap` NaNs the output where the bracketing source samples are farther apart
than the limit, rather than ruling a straight line across a dropout. A
per-sensor `max_gap` overrides the global; `max_gap: null` on a sensor inherits
the global rather than disabling it. Output
times that land **exactly** on a source sample are measurements, not
interpolations, and are kept regardless — this matters because every sensor
riding the base clock lands on the output times exactly.

## Readers

| `files.reader` | Needs | Compressed `*.?c?` |
|---|---|---|
| `xarray-dbd` | `pip install "microstructure-tpw[dbd]"` | yes (via `lz4`) |
| `dbd2netcdf` | the `dbd2netCDF` binary on `PATH` | yes, natively |
| `netcdf` | nothing — inputs are already NetCDF | n/a |
| `auto` (default) | NetCDF inputs → `netcdf`; else `xarray-dbd` if importable; else `dbd2netcdf` | |

**The sensor-list cache is not optional.** Slocum files reference their sensor
list by hash rather than carrying it, so a file whose hash is missing from
`files.cache` cannot be decoded. `dinkum-hotel` treats a skipped file as an
error: it compares the number of files decoded with the number requested and
refuses to build a partial hotel file (a mission whose science-file hashes are
uncached would otherwise quietly become flight-only). The error names the
skipped files and points at `files.cache`.

`files.skip_first_record` is honoured by both DBD backends (xarray-dbd's
`skip_first_record`, dbd2netCDF's `--skipAll`); neither reader skips the first
record on its own. Integer-coded sensors (status/flag enums) have their NetCDF
fill values masked to NaN on every backend.

## Worked example

`examples/slocum_glider_hotel/` holds both halves of the chain:

- `dinkum-hotel.yaml` — EBD → `hotel.nc`
- `perturb.yaml` — MicroRider `.p` + `hotel.nc` → ε, χ, CTD

```bash
dinkum-hotel build -c examples/slocum_glider_hotel/dinkum-hotel.yaml
perturb run -c examples/slocum_glider_hotel/perturb.yaml
```

### Units are converted once, in the builder

`sensors.*.scale` is where unit conversion belongs, because that is where the
source units are known:

| Sensor | Slocum unit | Wanted | `scale` |
|---|---|---|---|
| `sci_water_cond` | S/m | mS/cm | `10.0` |
| `sci_water_pressure` | bar | dbar | `10.0` |
| `sci_water_temp` | °C | °C | — |

The hotel file is then correct and self-describing, so the perturb side must
**not** re-apply those scales — doing so multiplies by 10 twice and puts
salinity into fantasy. The shipped `perturb.yaml` carries no `scale:` on these
channels for exactly this reason.

### Coordinates need `transform`, not `scale`

Slocum reports **every** geographic coordinate in NMEA `ddmm.mmmm`.
`m_lat = 2015.61159` is 20° 15.61159′ = **20.260193 °N** — not 20.1561.
No scale factor converts it:

| Sensor | Raw | `scale: 0.01` | `transform: nmea_degrees` | Error |
|---|---|---|---|---|
| `m_lat` | `2015.61159` | 20.1561 | **20.260193** | 11.6 km |
| `m_lon` | `-12949.49159` | −129.4949 | **−129.824860** | 34 km |

and the discrepancy is a sawtooth in the minutes field, so an `offset` cannot
absorb it either. Set `transform: nmea_degrees` on `m_lat`/`m_lon`,
`m_gps_lat`/`m_gps_lon`, `c_wpt_lat`/`c_wpt_lon`:

```yaml
sensors:
  m_lat:
    name: "lat"
    time_sensor: "m_present_time"
    transform: nmea_degrees
    units: "degrees_north"
    valid_min: -90.0          # degrees — the check sees the TRANSFORMED value
    valid_max: 90.0
```

Where the conversion happens is not a detail. `scale`/`offset` is affine, so it
commutes with interpolation and is applied once, on the output. A `transform`
is **not** affine and runs on the *source* samples, before the range check and
before projection — because the raw form steps by 40.02 across a whole minute
(`2059.9994` → `2100.0006`) where the vehicle moved 2 × 10⁻⁵ °. Interpolating
that raw form and converting afterwards puts the midpoint at 21.3333 ° instead
of 21.0000 °: a 37 km error manufactured once per minute of latitude.

Because the transform runs first, `valid_min`/`valid_max` are expressed in the
**transformed** units (degrees), and `scale`/`offset` — if you also set one —
applies afterwards to the output.

### `valid_min`/`valid_max` are in SOURCE units, not `units:`

The range check runs on the source samples, before interpolation (so a spike is
removed rather than smeared) and therefore **before `scale`**. So the bounds are
in the sensor's raw units — post-`transform`, pre-`scale` — even though the
neighbouring `units:` key names the *output* unit:

```yaml
  sci_water_cond:
    scale: 10.0             # S/m -> mS/cm
    units: "mS/cm"          # the OUTPUT unit
    valid_min: 0.0          # ... but these are S/m
    valid_max: 7.0          #     7 S/m = 70 mS/cm.  `70` here would be
                            #     700 mS/cm and would never reject anything.
  sci_water_pressure:
    scale: 10.0             # bar -> dbar
    units: "dbar"
    valid_max: 200.0        # bar = 2000 dbar
```

This ordering is deliberate: `erddap-hotel` fills these from the server's
declared `valid_min`/`valid_max` attributes, which are in the served units. The
cost is that a bound written in output units is not an error — it is a check
that silently never fires. After a build, confirm each `qc_valid_*` attribute
against the variable's actual range.

## Provenance

Every build records what it discarded, so a surprising hotel file can be
audited without rerunning:

```
dinkum_time_base                     = sci_ctd41cp_timestamp
dinkum_time_min / dinkum_time_max    = resolved epoch bounds
dinkum_time_base_rejected_nonfinite  = 2001
dinkum_time_base_rejected_range      = 2
dinkum_time_base_duplicates          = 997
dinkum_provenance                    = per-sensor counts, one entry each
```

and per variable: `dinkum_sensor`, `dinkum_time_sensor`, `projection_method`,
`source_samples`, `qc_valid_min`/`qc_valid_max`, `gap_blanked_samples`.
