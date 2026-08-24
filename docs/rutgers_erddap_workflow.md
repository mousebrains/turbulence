# MicroRider on a Slocum, CTD over ERDDAP — the full workflow

Every command, in order, for processing a MicroRider deployment whose CTD
reference comes from an ERDDAP server rather than from local Slocum files.

Worked against Rutgers' `slocum-data.marine.rutgers.edu`. The configs are in
`examples/rutgers_erddap/`; copy that directory and edit two lines.

```
0.  MicroRider .p files ──────────────────┐   off the instrument, by hand
1.  erddap-hotel  ──────────►  hotel.nc   │   the CTD reference
2.  perturb trim  ──────────►  trimmed .p ◄┘
3.  fp07-cal      ──────────►  patched .p     one calibration per deployment
4.  perturb run   ──────────►  results/
```

Steps 2 and 3 are why this is not simply "hotel, then pipeline": the FP07
calibration is a **pre-pipeline** step that needs both halves — trimmed `.p`
files *and* the hotel file — and it hands perturb `.p` copies that already
carry the right coefficients.

---

## Step 0 — get the `.p` files

Not over ERDDAP. `tabledap` serves rows and `griddap` serves grids; neither
serves an RSI binary, so "MicroRider over ERDDAP" would mean already-converted
data — and converted values **cannot be recalibrated at all**, because
`L = ln(R_T/R_0)` needs raw counts and the bridge constants. Recalibration and
epsilon both become impossible. See `docs/erddap_access_DESIGN.md` §7.

ERDDAP replaces only the *reference*. Put the `.p` files wherever
`files.p_file_root` points.

---

## Step 1 — build the hotel file

```bash
cp -r examples/rutgers_erddap  /path/to/deployment
cd /path/to/deployment
```

Edit `server.dataset_id` and the two time windows in `erddap-hotel.yaml`, then:

```bash
erddap-hotel info  -c erddap-hotel.yaml     # look before you download
```

Run this first, the way `dinkum-hotel sensors` comes before a build. It probes
`.das`/`.dds` and prints each requested variable's units, valid range and
`_FillValue`, and it exits non-zero naming anything absent — ERDDAP 400s the
*whole* request when one variable is misspelled, so catching it here saves an
opaque failure later.

**Read the units it prints.** They are what `sensors.*.scale` has to match.
On Rutgers' raw datasets:

| variable | served as | wanted | scale |
|---|---|---|---|
| `sci_water_temp` | `degrees_C` | °C | 1.0 |
| `sci_water_cond` | **`S m-1`** | mS/cm | **10.0** |
| `sci_water_pressure` | **`bar`** | dbar | **10.0** |

```bash
erddap-hotel fetch -c erddap-hotel.yaml --dry-run   # print the URLs, fetch nothing
erddap-hotel fetch -c erddap-hotel.yaml             # populate the cache
erddap-hotel build -c erddap-hotel.yaml             # cache -> hotel.nc
```

`fetch` is separable so a long download happens once; `build` can then be
re-run offline while you iterate on the QC settings. `build` will fetch
anything missing, so running `fetch` first is optional.

Three things worth knowing:

- **`--offline`** never touches the network. Use it to rebuild from the cache.
- **`refresh: incremental`** (the default) picks up whatever landed since the
  last run — an active mission's dataset is appended to. It also refetches
  `overlap_chunks` of the tail, because a delayed-mode dataset can *revise*
  recent rows, not only extend them.
- A window with no rows is **not an error**. ERDDAP answers 404 for it, the
  same status as a wrong dataset ID; a gap in a deployment is data.

Check the result before building on it:

```bash
ncdump -h hotel.nc | head -40
```

Conductivity should read ~40 mS/cm, not ~4. Pressure should read ~1000 at
1000 m, not ~100. If either is off by ten, the scale is applied twice or not at
all.

---

## Step 2 — trim the `.p` files

```bash
perturb trim -c perturb.erddap.yaml
```

Writes trimmed copies into `<output_root>/trimmed/`, flattening the
`<root>/<SN>/<file>.p` layout. Point step 3's `files.p_file_root` there.

---

## Step 3 — calibrate the FP07s

A **pre-pipeline** step, run once per deployment. Copy `fp07-cal.yaml` from
`examples/slocum_glider_hotel/` — it needs no ERDDAP-specific change, because
it reads the same `hotel.nc`.

Set in it:

- `files.p_file_root` → the `trimmed/` directory from step 2
- `reference.file` → `hotel.nc` from step 1
- `reference.pressure_scale: 1.0` — **the hotel file already applied the ×10**.
  Use `10.0` only when pointing straight at a raw converted `ebd.nc`. Getting
  this wrong does not affect the lags (they are correlation-based and
  scale-invariant) but does corrupt the sensor geometry term.

```bash
fp07-cal coverage -c fp07-cal.yaml     # which yos actually carried a reference?
fp07-cal fit      -c fp07-cal.yaml     # coefficients + the stability diagnostic
fp07-cal patch    -c fp07-cal.yaml     # write them into .p COPIES
```

`coverage` before `fit` is the same discipline as `info` before `build`.

`fit` reports whether the calibration is stable in time and depth, and gives
the lag decomposition. Read it — on a 72-day deployment the in-situ fit was
worth −2.66 K and −1.62 K against the factory nominal, and both probes had
shipped with *identical* nominal coefficients, so the in-situ fit was the only
real calibration those channels had.

`patch` writes to `<output_dir>/patched` by default, **not in place**. Use
`--dry-run` first, and `-o` to send them elsewhere.

One value not to refit: `pairs.kernel_tau` is the CTD thermistor's response
pole, ~0.7 s for a Seabird GPCTD, and it is a property of the CTD *model*
rather than the unit or the mission. What *does* need checking is which CTD the
glider carries — TWR's masterdata names `SBE41CP` for both the unpumped CTDs
and the GPCTD, and an unpumped CTD is slower and flow-dependent.

---

## Step 4 — run the pipeline

Two edits to `perturb.erddap.yaml` first:

- `files.p_file_root` → the `patched/` directory from step 3
- `fp07.calibrate: false` — otherwise perturb refits per file *on top of* the
  patch

```bash
perturb run -c perturb.erddap.yaml -j 0
```

`-j 0` auto-sizes the worker pool. Or override paths without editing the
config:

```bash
perturb run -c perturb.erddap.yaml --p-file-root fp07cal/patched --hotel-file hotel.nc
```

---

## Re-running on a live mission

```bash
erddap-hotel verify -c erddap-hotel.yaml    # has the dataset changed?
erddap-hotel build  -c erddap-hotel.yaml    # picks up new data (incremental)
perturb run         -c perturb.erddap.yaml
```

`verify` re-fetches only the `.das` — a few tens of kB, no data transfer — and
compares its digest against the one recorded in `hotel.nc`. Exit code 2 means
the dataset was revised upstream; rebuild with `refresh: always`.

**Expect a re-run to reprocess a lot.** New data changes `hotel.nc`, which
changes its fingerprint, which correctly invalidates perturb's per-file
markers. That is the intended behaviour — a fabricated "nothing changed" is the
failure mode this design exists to avoid — but on a multi-month deployment it
is the difference between a quick re-run and a full reprocess. If you only want
new files processed, keep the hotel file fixed (`refresh: never`) until you
deliberately want to pick up new CTD data.

Whether to redo step 3 is a judgement call: the coefficients were fitted on the
data available at the time, and more data makes them better. Re-running `fit`
and `patch` after a substantial extension is reasonable; doing it every run is
not, because it changes the calibration underneath already-processed profiles.

---

## Troubleshooting

| symptom | cause |
|---|---|
| `400 Unrecognized variable=…` | a typo in `fetch.variables`. Run `erddap-hotel info`. |
| `no data in N window(s)` | `fetch.time_min`/`time_max` miss the deployment. `info` prints its real coverage. |
| the build seems to hang before any chunk lands | it is issuing row-count requests. Each one downloads the window's rows as CSV, so a 7-day window on a busy dataset is ~400 000 lines. The window is clamped to the dataset's declared coverage, so this should be a handful of requests -- if it is not, check `--dry-run`, which prints the clamped plan. |
| `row count mismatch … probably truncated` | should not happen; the count query projects the same columns as the data request. If it recurs the download really is being cut. |
| salinity absurd | the ×10 applied twice — a `scale` on **both** the builder and the perturb side. It belongs only in the builder. |
| `verify` always says CHANGED | should not happen; the `.das` digest ignores ERDDAP's per-request stamps. If it recurs, the dataset really is being reprocessed. |
| every channel NaN after the merge | `hotel.time_column` naming a clock the file does not carry, or `max_gap` shorter than the CTD's real sampling interval. |
| `--offline` refuses | the cache is keyed on the request URL *and* the dataset revision; changing a time window or the variable list is a new key. Fetch once online. |

## See also

- `docs/erddap_access_DESIGN.md` — why it is shaped this way, and §13 for where
  the build differed from the design
- `docs/dinkum_hotel.md` — the local-file twin, when you have EBD files
- `docs/fp07cal/runbook.md` — step 3 in full
- `examples/rutgers_erddap/` — the two configs, annotated
