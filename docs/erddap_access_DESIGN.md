# ERDDAP as a data source

**Status:** design, not implemented.
**Branch/worktree:** `design/erddap-access` @ `.claude/worktrees/erddap-design`
**Date:** 2026-08-23
**Reference implementation:** `ru33_microrider_preparation.ipynb` (Rutgers)

---

## 0. What this is for

Today every external data feed the pipeline consumes is a **local file**: a
hotel `.csv`/`.nc`/`.mat`, a GPS track, a Slocum binary. Rutgers publish their
glider data through **ERDDAP** instead, and their notebook already does the
whole job — fetch a variable subset from `tabledap`, sanitise it, write a
NetCDF that is hotel-shaped in all but name.

The question this document answers is not "can we call an ERDDAP URL" — the
notebook proves we can, in six lines. It is **where in the pipeline the network
belongs**, and what has to be true for a processing run that depends on a remote
server to still be reproducible a year later.

### Scope

In scope: ERDDAP as a source for the **hotel/CTD feed**, which is what Rutgers
publish and what the FP07 calibration and half of perturb consume.

**Explicitly out of scope: the MicroRider data itself.** `.p` files always
arrive as files. This was considered and dropped; §7 records why, so it does not
get re-proposed.

Also out of scope for now, but the design should not preclude them: GPS tracks
(`gps.source`), ADCP, and publishing our *outputs* to ERDDAP.

### A note on the current tree

This branch is off `main`, which does **not** yet contain three pieces of
in-flight work this design leans on:

| piece | PR | why it matters here |
|---|---|---|
| `fp07cal/` | #149 | reads a CTD reference directly, on the CTD's own clock |
| `dinkum/` | #149 | the existing "build a hotel file from a raw source" tool — the pattern to copy |
| hotel gap gating | #150 | `max_gap` required; changes what a merged channel means |

Where they matter, they are called out.

---

## 1. What the notebook actually does

Six steps, and every one of them is a design decision worth keeping or arguing
with.

```python
ERDDAP_BASE = "https://slocum-data.marine.rutgers.edu/erddap/tabledap"
DATASET_ID  = "ru33-20211001T1841-trajectory-raw-delayed"
VARIABLES   = ["sci_ctd41cp_timestamp", "sci_water_cond",
               "sci_water_pressure", "sci_water_temp"]

query = urllib.parse.quote(",".join(VARIABLES), safe="")
url   = f"{ERDDAP_BASE}/{DATASET_ID}.nc?{query}"
urllib.request.urlretrieve(url, RAW_PATH)
```

1. **Fetch** a variable subset as NetCDF3 (`.nc`), server-side. Metadata and
   variable selection come free; no client-side filtering.
2. **Open undecoded** — `decode_times=False, mask_and_scale=False` — so QC
   operates on stored values rather than on xarray's interpretation of them.
3. **Replace `0.0` with NaN** across *all four* variables: on this glider `0.0`
   is the "CTD did not sample this cycle" sentinel.
4. **Drop rows** with a NaN `sci_ctd41cp_timestamp`.
5. **Stable-sort** by that timestamp, then **drop duplicates keeping the
   first**.
6. **Write** a compressed NetCDF, appending to `history` / `processing_level`
   rather than clobbering, via a temp file and atomic replace. (ERDDAP has
   already written a `history` line of its own into the served file — the full
   request URL and the UTC fetch time — so "append" is the right verb.)

The output is a strictly-increasing, sanitised CTD table. That is exactly the
contract `perturb`'s `hotel:` block wants, and exactly what
`fp07cal.load_hotel_reference` (#149) wants.

**The notebook is a good reference implementation and the design should absorb
it rather than reinvent it.** What follows is mostly about the things a
notebook run by hand does not have to worry about and a batch pipeline does.

---

## 2. Where files enter today

| consumer | source | loader |
|---|---|---|
| hotel/CTD | `.csv` / `.nc` / `.mat` | `perturb/hotel.py::load_hotel` (dispatches on suffix) |
| CTD reference for FP07 | `.nc` | `fp07cal/series.py::load_hotel_reference` (#149) |
| Slocum raw | `*.dbd`/`*.ebd`/`*.dcd`/`*.ecd` | `dinkum/` → writes `hotel.nc` (#149) |
| GPS | `nan` / `fixed` / `csv` / `netcdf` | `perturb/gps.py`, dispatches on `gps.source` |

Two different shapes are already in the tree:

- **`load_hotel` dispatches on file extension.** Adding a URL here means the
  "path" is no longer a path.
- **`gps.py` dispatches on an explicit `source:` key.** That is the better
  precedent: the source is *named*, not inferred.
- **`dinkum-hotel` is a separate build step** that turns a foreign format into
  the canonical artifact, and nothing downstream knows it exists.

That third pattern is the one this design follows.

---

## 3. What ERDDAP gives you, and what it does not

Worth stating plainly, because several of the failure modes below follow from
it.

**`tabledap`** serves row-oriented data (a glider trajectory). **`griddap`**
serves gridded data. Rutgers' glider datasets are `tabledap`. The response
format is chosen by file extension on the dataset ID: `.nc`, `.ncCF`, `.csv`,
`.json`, `.das`, `.dds`. The query string after `?` is a variable list followed
by optional constraints:

```
…/tabledap/ru33-….nc?sci_water_temp%2Csci_ctd41cp_timestamp&time%3E%3D2021-10-01T00%3A00%3A00Z&time%3C%3D2021-10-08T00%3A00%3A00Z
```

The constraints are **percent-encoded**. A literal `>=` in the request target is
rejected by Tomcat before ERDDAP sees it (`400`, an HTML "Invalid character
found in the request target" page); the same query with `%3E%3D` returns `200
text/csv`. Both probed against Rutgers' server.

Things that matter for us:

- **Server-side subsetting is the whole point.** The notebook fetches the
  *entire* dataset with no time constraint. For a 6-month trajectory at 1 Hz
  that is a large download to obtain the week someone actually wants. Any real
  implementation must push a time window to the server.
- **`.das` / `.dds` are cheap metadata probes.** They answer "does this dataset
  exist, what variables does it have, what is its time range" without
  transferring data. That is how a `coverage`-style command should work.
- **ERDDAP errors are non-2xx, with the detail in a plain-text body.** Probed
  live (ERDDAP 2.30.0 behind nginx/Tomcat):

  | request | status | body |
  |---|---|---|
  | unknown variable | `400 text/plain` | `Error {code=400; message="Bad Request: Query error: Unrecognized variable=\"nosuchvar\"."}` |
  | raw `&time>=…` (unencoded) | `400 text/html` | Tomcat "HTTP Status 400 – Bad Request" |
  | unknown datasetID | `404 text/plain` | `Error {code=404; message="Not Found: Currently unknown datasetID=…"}` |
  | **valid query, empty time window** | `404 text/plain` | `Error {code=404; message="Not Found: Your query produced no matching results. (time>=… is outside of the variable's actual_range: …)"}` |
  | valid, encoded | `200 application/x-netcdf` | NetCDF, `Transfer-Encoding: chunked`, **no `Content-Length`** |

  So `urlretrieve` *does* raise `HTTPError` on every error case above; the
  magic-byte check in F1 is defensive, not the primary guard. Two consequences
  that are not obvious: **an empty window is a 404, not a request bug** — the
  fetcher must recognise "produced no matching results" and treat it as an
  empty chunk (no retry, no failure; §8 F6/F8) — and **a truncated `.nc`
  cannot be detected by length**, because there is none (§8 F3).
- **Datasets are revised.** A `-raw-delayed` dataset gets reprocessed. The same
  URL a month later is not guaranteed to be the same bytes. This is the central
  reproducibility problem.
- **There is no pagination.** A too-large request is refused, or times out, or
  succeeds slowly. Chunking is the client's job.

---

## 4. Design options

### Option A — URL-aware loaders

Let `hotel.file` accept an `https://` URL and have `load_hotel` fetch it.

*For:* smallest diff; nothing new to learn.

*Against:* not the fetch count — `load_hotel` runs **once, in the parent,
before the pool** (`perturb/pipeline.py:3062-3070`) and the `HotelData` object
is handed to every worker (`:3130`, `:3156`); `process_file` only calls
`merge_hotel_into_pfile`. A URL here would be fetched once per run. The defect
is the **skip-cache**: `_external_input_fingerprints` (`pipeline.py:352-366`)
keys the hotel by path/size/mtime, and for a URL `Path(...).exists()` is False,
so the fingerprint is the constant `{"missing": True}`. `_marker_is_current`
would then keep *hitting* forever — a dataset revised on the server would never
trigger reprocessing, and nothing would say so. And the artifact that a result
was computed from no longer exists anywhere.

**Rejected.** Not because fetching is hard, but because the run's own
change-detection is blind to the input, which destroys reproducibility in the
quietest way possible, and every downstream consumer inherits a network failure
mode.

### Option B — A fetch tool that writes the canonical artifact

`erddap-hotel build -c erddap-hotel.yaml` → `hotel.nc`, a sibling of
`dinkum-hotel`. Everything downstream is unchanged and does not know ERDDAP
exists.

*For:* the artifact is a file, so reproducibility, caching, provenance and
offline reruns all come free. It mirrors a pattern already in the tree. The
network touches the pipeline exactly once, before it starts.

*Against:* an explicit step to run. Two commands instead of one.

### Option C — B, plus a thin read-through cache

As B, but the fetch is content-addressed by `(server, dataset, query,
date_modified)` — the last taken from a `.das` probe before each fetch — and
cached under a directory, so re-running is free and an interactive user can
point straight at a URL without thinking about it.

### Recommendation: **Option B, with C's caching as the fetch layer.**

The deciding argument is not convenience. It is that **a processing result must
be reproducible from artifacts that exist locally.** ERDDAP datasets are
revised; servers go down; a `-raw-delayed` dataset by construction becomes a
different dataset later. If the only record of what a calibration was fitted
against is a URL, the calibration is not reproducible. Writing the fetched
subset to a file, with the query and fetch time recorded in its `history`, makes
it so.

This is the same argument that made `fp07-cal` a pre-pipeline step rather than
a pipeline stage, and it lands in the same place.

---

## 5. The design

### 5.1 Shape

```
ERDDAP ──► erddap-hotel fetch ──► raw subset .nc  (cached, immutable)
                                        │
                                        ▼
                              erddap-hotel build ──► hotel.nc
                                        │
                                        ▼
              perturb (hotel:) / fp07-cal (reference:) — unchanged
```

`fetch` and `build` are split for the same reason `fp07-cal` splits `coverage`
from `fit`: the network step is slow and should happen once, while the QC step
is the part that gets iterated on. A cached raw subset also means the QC rules
can be revised and re-applied without re-downloading.

The `build` half is **not** a third sanitiser. `dinkum/build.py` already
exposes `resolve_time_bounds`, `time_validity`, `sanitize_time` (with its
`_dedupe`) and `project_sensor` as pure numpy functions (`build.py:82-296`);
they import standalone today with no change to #149. `erddap-hotel build`
imports them. The only constraint that imposes is merge order: this lands
after #149.

**Output clock.** The served `time` axis is itself not unique — a 1-h sample
carried repeated `00:02:24Z` stamps with NaN temperature — so deduplicating on
`sci_ctd41cp_timestamp` alone does not yield a unique `time`, and vice versa.
Exactly as `dinkum-hotel` does, each variable is attributed to its own clock
(`time_sensor`) and projected with `project_sensor` onto a single declared
`time.base`; for a CTD-only feed that base is `sci_ctd41cp_timestamp`.

### 5.2 Config

```yaml
server:
  base_url: "https://slocum-data.marine.rutgers.edu/erddap"
  protocol: "tabledap"          # tabledap | griddap
  dataset_id: "ru33-20211001T1841-trajectory-raw-delayed"
  timeout_s: 120
  retries: 3                    # on connection error / 5xx only, never on 4xx
  cache: "<CONFIG_DIR>/erddap_cache"

fetch:
  variables:                    # pushed to the server; nothing is fetched that
    - sci_ctd41cp_timestamp     # is not listed
    - sci_water_temp
    - sci_water_cond
    - sci_water_pressure
  time_variable: "time"         # the variable the constraint applies to --
                                # NOT necessarily the one you sort by
  time_min: "2021-10-01T00:00:00Z"   # server-side constraint. null = whole
  time_max: null                     # dataset, which for a long deployment is
                                     # a large download
  chunk_days: 7                 # split the window into requests this long.
                                # Bounds each request, lets a partial failure
                                # retry cheaply, and keeps cache entries small
  constraints: []               # extra server-side clauses, e.g. "distinct()";
                                # percent-encoded by the builder, never by hand
  format: "nc"                  # nc | ncCF -- what is downloaded; see 5.4

qc:
  # See section 6. These are the rules the notebook applies, made explicit and
  # per-variable rather than blanket.
  time_base: "sci_ctd41cp_timestamp"       # the output clock, as dinkum's
                                           # time.base; every variable is
                                           # projected onto it
  drop_zero_as_fill: ["sci_ctd41cp_timestamp"]   # NOT every variable
  valid_range:
    sci_water_temp: [-5.0, 45.0]
    sci_water_cond: [0.0, 70.0]
    sci_water_pressure: [-2.0, 12000.0]
  dedupe: "mean"                # mean | first | last -- matches dinkum-hotel
  require_increasing: true

output:
  file: "<CONFIG_DIR>/hotel.nc"
  scale:                        # unit normalisation happens ONCE, here
    sci_water_cond: 10.0        # S/m -> mS/cm
    sci_water_pressure: 10.0    # bar -> dbar
```

Two things deliberately mirror `dinkum-hotel`: `dedupe` semantics and the fact
that **unit conversion happens once in the builder**. A hotel file arriving at
perturb should already be in dbar and mS/cm; the perturb side must not re-apply
a scale. That has bitten before.

### 5.3 Provenance is the deliverable

The output NetCDF must record enough to reconstruct itself:

```
history:      <ERDDAP's own lines, preserved verbatim: the upstream processing
               chain, then "<UTC>: <full request URL>" per chunk>
              2026-08-23T…Z: erddap-hotel build: <the QC rules applied, with counts>
source:       ERDDAP tabledap <base_url>/<dataset_id>
erddap_date_modified: 2021-11-10T20:11:47Z   # from .das at fetch time
erddap_das_checksum: <sha256 of the .das response at fetch time>
```

ERDDAP already records the request URL and fetch time in `history` — verified
on a served `.nc`, whose `history` ends `2026-08-23T19:…Z: https://…/tabledap/
ru33-….nc?…` — so the builder does **not** duplicate that in its own
attributes; it preserves the line and appends only QC provenance. The `.das`
carries `date_modified` (here `"2021-11-10T20:11:47Z"`) for free; together with
the `.das` sha256 it is the cheap way to notice a dataset has been revised, and
both go into the cache key (§4 C). `erddap-hotel verify` re-fetches just the
`.das` and tells you whether the dataset still matches what you built from —
without downloading the data.

### 5.4 CLI

```bash
erddap-hotel datasets -s <base_url>          # search/list dataset IDs
erddap-hotel info -c erddap-hotel.yaml       # .das/.dds probe: variables,
                                             # time range; row count from
                                             # a .csv?time&<constraints>
erddap-hotel fetch -c erddap-hotel.yaml      # populate the cache only
erddap-hotel build -c erddap-hotel.yaml      # fetch (cached) + QC -> hotel.nc
erddap-hotel verify -c erddap-hotel.yaml     # has the dataset changed?
```

`info` before `build` is the same discipline as `fp07-cal coverage` before
`fit`: look at what you actually have before computing on it.

Two details of the row count. `.dds` is a type schema (`Dataset { Sequence
{…} }`) and `.das` has `actual_range` — neither carries a count, and
`.ncHeader` returned a `500 unknown DataType == uint` on this dataset. The
reliable count is `.csv?time&<constraints>`: lines minus the two header rows
(a 1-day window: 88 730 lines, 88 728 rows, matching the `.nc` exactly).

**`.nc` versus `.ncCF`.** `perturb/hotel.py::_load_netcdf` reads whatever
`time_column` names, so either flavour works downstream. The default is `.nc`
(flat `row` dimension, the shape the notebook and `load_hotel` both expect);
the fetcher falls back to `.ncCF` for a chunk whose `.nc` fails to open
(§8 F3). `&distinct()` as a server-side constraint removes exact-duplicate
rows before download and is worth enabling by default; it halves the
duplicate problem, it does not remove it (rows sharing a stamp but differing
in a NaN are not duplicates to the server).

### 5.5 Dependencies

`erddapy` is the standard Python ERDDAP client, but **neither it nor `requests`
is currently installed**, and the notebook shows the whole fetch is six lines of
`urllib`. **Decision: stdlib `urllib`, no new required dependency.**

`urllib` is sufficient, with three things it does not give you for free and
which the fetcher must therefore do itself:

- **A timeout.** `urlopen` defaults to no timeout, so a hung server hangs the
  build forever. Always pass one.
- **Retry and backoff.** No built-in retry; the loop is ours (and see F8 — 5xx
  and connection errors only, never 4xx).
- **Connection reuse.** No pooling, so each chunk is a fresh TCP+TLS handshake.
  Irrelevant at `chunk_days` granularity — tens of requests, not thousands.

None of those argues for a dependency. `erddapy` stays an optional extra, and
only if `datasets`-style discovery turns out to be wanted.

---

## 6. QC: three sanitisers, three different answers

This is the part most likely to cause quiet damage. Once #149 merges, the tree
contains **three independent implementations of "clean up a Slocum CTD table"**,
plus a fourth thing that silently drops data without being one. They do not
agree.

| | **1.** notebook | **2.** `dinkum/build.py` | **3.** `fp07cal/series.py` |
|---|---|---|---|
| where | Rutgers, external | `sanitize_time`, `_dedupe`, `time_validity` | `sanitize_reference` |
| bad values | `== 0.0` → NaN, **every variable** | per-sensor `valid_min`/`valid_max` | non-finite, then per-variable `valid_min`/`valid_max` |
| bad times | drop NaN only | finite ∧ within `time.min_value`…`max_value` | finite ∧ `100 ≤ t ≤ 4e9` |
| duplicates | keep **first** | `mean` \| `first` \| `last`, one global `time.dedupe` | **mean**, not configurable |
| ordering | stable sort | sort | stable sort |
| reports what it dropped | yes, into `history` | yes, a stats tally | **no** |

And the fourth:

**4. `perturb/hotel.py::_interp_one`** — not a sanitiser, but it drops every
non-finite sample and then interpolates across the hole, which un-does whatever
the three above decided to mark bad. That is what PR #150 gates behind
`max_gap`; before it, a builder that carefully NaN-marked a dropout produced
byte-identical output to one that did not.

### What to do about it

Short term, for this design: **the ERDDAP builder should use rule set 2's
shape** — per-variable ranges, configurable dedupe, and a rejection tally —
because it is the most complete of the three and already has a config schema
that expresses it.

Concretely: `erddap-hotel` **imports** rule set 2 rather than copying its
shape (§5.1) — `time_validity` for the clock, `sanitize_time` for the
sort/dedupe/tally, `project_sensor` for the output base — and adds only what
ERDDAP needs on top (`drop_zero_as_fill`, `_FillValue` masking). That keeps the
count at three implementations rather than four. Longer term, three is still
two too many; the natural convergence is `fp07cal.sanitize_reference` becoming
a thin caller of the same functions. Not proposed here.

**The `0.0`-as-fill rule is the dangerous one.** Applied to
`sci_ctd41cp_timestamp` it is exactly right — `0.0` is the never-set sentinel.
Applied to `sci_water_temp` it silently destroys a genuine 0 °C reading, and to
`sci_water_pressure` a genuine surface sample. For a mid-latitude glider that
may never bite; on a polar deployment it certainly would. **The design should
apply the zero-sentinel rule only to the variables where it is a sentinel
(`drop_zero_as_fill`), and use `valid_range` for everything else** — which is
what the other two implementations already do.

**Duplicate handling should default to `mean`**, matching the other two.
Keeping the first is defensible only if duplicates are exact repeats; if the
CTD reported twice within a timestamp's resolution, the mean is the better
estimate and dropping is a silent bias toward whichever row sorted first.

**`mask_and_scale=False` deserves a note.** The notebook opens undecoded so QC
sees stored values. That is right, but it also means a declared `_FillValue` is
*not* masked — so the sanitiser must apply `_FillValue` itself rather than
assuming xarray did. On this dataset that is not hypothetical: the `.das`
declares `_FillValue 9.96921e+36` on every float variable, while
`sci_ctd41cp_timestamp` has **no** `_FillValue` attribute at all, and the two
conventions coexist in the same file. In a 1-day `.nc`, `sci_water_temp` has
49 147 fill-stored rows and 8 literal zeros; `sci_ctd41cp_timestamp` has the
same 49 147 NaN and 8 zeros. So the builder masks `_FillValue` where declared
**and** applies `drop_zero_as_fill` where configured; neither subsumes the
other.

**Interaction with #150.** Once `hotel.max_gap` is required, a sparsely-sampled
ERDDAP feed will produce NaN over its gaps rather than a fabricated ramp —
which is correct, and means the ERDDAP path inherits that protection for free.
The builder should *not* also gap-fill; its job is to deliver real samples and
let the merge decide.

---

## 7. Decided: the MicroRider data does not come over ERDDAP

Considered, and dropped. Recorded here because the reasoning is not obvious and
the idea is otherwise attractive.

**ERDDAP cannot serve a `.p` file.** `tabledap` serves rows, `griddap` serves
grids; neither serves the RSI binary. So "MR over ERDDAP" necessarily means
*already-converted* data — and conversion is lossy in exactly the way that
matters:

- **Recalibration becomes impossible.** The FP07 in-situ fit computes
  `L = ln(R_T/R_0)` from **raw counts** plus the bridge constants
  (`a`, `b`, `g`, `e_b`, `adc_fs`, `adc_bits`). Given temperature in degC there
  is no way back to `L`, so Steinhart-Hart cannot be refitted at all — only an
  offset applied in temperature space, which is a strictly weaker and different
  correction. The same argument kills epsilon: it needs raw shear counts and the
  probe sensitivity.
- **The calibration metadata lives in the file.** The config string in record 0
  carries the bridge constants, the coefficients and the probe serials. A table
  of physical values carries none of it unless someone deliberately attaches it.
- **The volume is not serviceable anyway.** For an osu685-scale deployment,
  512 Hz × 8 fast channels × 72 days ≈ **2.5 × 10¹⁰ samples**. That is not a
  `tabledap` request, chunked or otherwise.

So the pipeline's input path stays: **`.p` files as files, hotel/CTD feed over
ERDDAP.** That also keeps `fp07-cal patch` intact — its whole design is to write
corrected coefficients into a `.p` file so perturb needs no changes (#149 §D7),
and there is nothing to patch in a remote dataset.

**What remains genuinely interesting, and is a separate piece of work:**
publishing *our own* derived products — binned profiles, ε, χ, mixing — to
ERDDAP for downstream consumers. That is an output path, not an input path, and
shares only the QC machinery with this design. Not proposed here.

---

## 8. Failure modes

Ordered by how quietly they corrupt a result.

| # | failure | why it is quiet | mitigation |
|---|---|---|---|
| F1 | a non-NetCDF body lands in the cache | **not** the common case — every probed error was non-2xx and `urlretrieve` raises `HTTPError` (§3) — but a proxy/captive-portal page, or a future server, could answer 200 | catch `HTTPError` and surface ERDDAP's `Error {…}` message; then sniff the magic bytes (`CDF\x01` / `\x89HDF`) and **open with netCDF4** before caching; refuse anything that fails |
| F2 | dataset revised between runs | same URL, different data, no error | `.das` checksum recorded at build; `verify` re-checks it |
| F3 | partial/truncated response | `.nc` is served chunked with **no `Content-Length`**, so a TCP-level cut is invisible by length; one unreproduced probe saw a 200 `CDF\x01` body cut mid-header | open with netCDF4 after download; compare `row` against the line count of `.csv?time&<same constraints>`; on mismatch or open failure re-fetch, then fall back to `.ncCF` |
| F4 | silent time-zone or epoch mismatch | ERDDAP `time` is seconds since 1970 UTC, but `sci_ctd41cp_timestamp` is a *separate* variable with its own units | never assume; read `units` from the response and convert explicitly |
| F5 | requesting the whole dataset | works, slowly, until it does not | `chunk_days` and a required `time_min` for large datasets |
| F6 | a variable is absent from the dataset | ERDDAP `400`s the whole request (`Unrecognized variable=…`) | `info` first; name the missing variable |
| F6b | a chunk's time window holds no rows | ERDDAP answers **`404` "Your query produced no matching results"** — the same status as a wrong dataset ID | match that message; treat as an empty chunk, log it, **do not retry and do not fail** — a gap in a deployment is data, not a bug |
| F7 | server down mid-run | a batch job dies hours in | fetch is a separate step; the cache means `build` never needs the network |
| F8 | rate limiting / throttling | intermittent, looks like a network error | bounded retries with backoff, on connection errors, `429` and 5xx **only** — never retry a `400`/`404` (a request bug, or F6b's empty window) |

F3 is the one I would fix first: it is the one where the run succeeds and the
file is simply shorter than it should be.

---

## 9. Testing without a server

We have no ERDDAP to test against, and the design should not need one.

1. **Recorded fixtures.** Already captured from Rutgers' server on
   2026-08-23: `ru33-20211001T1841-trajectory-raw-delayed` `.das` (38 kB), a
   1-day `.nc` (2.5 MB, 88 728 rows) with its matching `.csv?time&…`, and the
   four error bodies from the §3 table. Commit the `.das`, the error bodies and
   a 1-h `.csv`; the 1-day `.nc` is too large for the tree and lives with the
   opt-in live test. Serve them from a `http.server` on localhost in tests.
   This exercises the real URL construction and the real parser against real
   bytes.
2. **A fake server for the failure modes.** A tiny handler that can return: a
   `404 Error{…no matching results}`, a `400 Unrecognized variable`, a
   chunked body cut mid-header, a 429, a 500, a slow response. Each of F1–F8
   becomes a test.
3. **Contract tests on the QC layer**, which needs no network at all — it takes
   an `xr.Dataset` and returns one. This is where most of the logic lives and
   all of it is testable today.
4. **A `--dry-run` that prints the URLs** it would fetch, so the query builder
   can be tested by string comparison without any I/O.
5. **One opt-in live test** (`ERDDAP_LIVE=1`) that hits the real server, skipped
   by default, for when we do have access.

The split matters: (3) and (4) cover the parts most likely to be wrong, and
neither needs a server. Getting a test server is not on the critical path.

---

## 10. Open questions

1. **Which dataset variant?** `-raw-delayed` versus a QC'd/science product. The
   raw one is what the notebook uses and what the Slocum-native path expects,
   but it is also the one most likely to be reprocessed.
2. **Which glider datasets does Rutgers publish, and under what IDs?** Only
   the science/CTD feed matters now that MR data is out of scope, but the
   dataset naming convention decides how much of the config can be shared
   between deployments.
3. **Should `erddap-hotel` and `dinkum-hotel` converge further?** They share
   the projection/QC functions from day one (§5.1). What they do not share is
   the config schema and the NetCDF writer; a `hotel_builder` core with two
   front ends is the natural next step once both have merged.
4. **Cache invalidation policy.** Decided: the key includes `date_modified`
   (and the `.das` sha256) from a probe before each fetch, so a server-side
   reprocess misses the cache by construction. The cost is one small `.das`
   request per `build`; `--offline` skips the probe and uses the last key.
5. **Credentials.** Rutgers' server is public. If a future one is not, no token
   should ever live in a config file that gets committed — environment variable
   or a keychain, and never echoed into `history`.

---

## 11. Adversarial review

Attacks on this design, and what changed.

| attack | outcome |
|---|---|
| "Just let `hotel.file` take a URL — it is four lines." | **Rejected, for a different reason than first written.** The fetch would run once (`load_hotel` is called in the parent, `pipeline.py:3062`), not per file. The defect is that `_external_input_fingerprints` (`:352-366`) returns the constant `{"missing": True}` for a non-path, so the skip-cache hits forever and a revised dataset never reprocesses. Forced Option B. |
| "Then cache it and the URL approach is fine." | **Partly accepted.** The cache is real and is in the design, but it does not fix reproducibility on its own — a cache can be cleared, and it does not record *what* was fetched. The artifact plus its `history` does. |
| "The notebook works; just port it." | **Mostly accepted, with three changes.** Its blanket `0.0 → NaN` would destroy a genuine 0 °C sample; it ignores the declared `_FillValue` that coexists with the zeros; and it fetches the entire dataset with no time constraint. |
| "Dedupe by keeping the first, like the notebook." | **Rejected as the default.** The other two sanitisers in the tree use `mean`, and keeping the first is a silent bias toward sort order when two rows share a timestamp. Kept as an option. |
| "Check the HTTP status and you are safe." | **Mostly accepted.** Every probed ERDDAP error was non-2xx and `urlretrieve` raises on it (§3). The status is the primary guard; magic bytes plus a netCDF4 open stay as defence against proxies and truncation, which the status cannot see (no `Content-Length`). |
| "Retry on failure." | **Qualified.** Retry connection errors, 429 and 5xx; never retry a 400/404 — a malformed variable list is a bug, and an empty time window is a 404 that is not an error at all (F6b). |
| "We cannot design this without a server to test against." | **Rejected.** The QC layer and the query builder are the parts most likely to be wrong and neither needs a network. Recorded fixtures plus a fake server cover the rest. |
| "Fetch the MicroRider data over ERDDAP too." | **Considered and dropped** (§7). ERDDAP serves tables, not `.p` binaries, so it means already-converted data — and converted values cannot be recalibrated at all: `L = ln(R_T/R_0)` needs raw counts and the bridge constants. Recalibration and epsilon both become impossible, and 512 Hz x 8 channels x 72 days (~2.5e10 samples) is not a tabledap request regardless. |
| "Use erddapy — it is the standard client." | **Rejected as a required dependency.** Neither it nor `requests` is installed, the fetch is six lines of stdlib `urllib`, and the three things urllib lacks (timeout, retry, pooling) are either trivial to add or irrelevant at tens-of-requests scale. Optional extra only. |
| "Make `erddap-hotel` share code with `dinkum-hotel` now." | **Accepted.** The overlap is importable today: `resolve_time_bounds`, `time_validity`, `sanitize_time`, `project_sensor` are pure numpy functions in `dinkum/build.py` and need no change to #149. The only constraint is merge order. Sharing the config schema and writer is the follow-up. |

---

## 12. What I would build first

In order, each independently useful:

1. **The QC layer** — takes an `xr.Dataset`, masks `_FillValue` and
   `drop_zero_as_fill`, then calls `dinkum.build.time_validity` /
   `sanitize_time` / `project_sensor` onto `time_base`, and returns a sanitised
   dataset with a `history` entry. No network. Testable today against the
   1-day fixture, with the notebook's own output as a golden case.
2. **The query builder** — config → URL list, with `chunk_days`. Encoding rule:
   the variable list and every constraint are passed through
   `urllib.parse.quote(…, safe="")` — the notebook already does this for the
   variables — so `>=` becomes `%3E%3D`, `,` becomes `%2C`, and `:` in the
   timestamps becomes `%3A`; only the `?` and the joining `&` are literal. Pure
   string construction, testable by comparison.
3. **The fetcher** — `urllib`, with `HTTPError` mapped to ERDDAP's message,
   the F6b empty-window case, the F1/F3 validation (magic bytes, netCDF4 open,
   row count against `.csv?time`), retries, and the
   `(server, dataset, query, date_modified)` cache.
4. **`info` / `verify`** on `.das`/`.dds`.
5. **`build`** wiring the above together, plus the example config and docs.

Steps 1 and 2 are the bulk of the logic and can be written, tested and reviewed
before anyone has ERDDAP access.
