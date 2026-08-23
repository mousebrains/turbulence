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
   rather than clobbering, via a temp file and atomic replace.

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
…/tabledap/ru33-….nc?sci_water_temp,sci_ctd41cp_timestamp&time>=2021-10-01T00:00:00Z&time<=2021-10-08T00:00:00Z
```

Things that matter for us:

- **Server-side subsetting is the whole point.** The notebook fetches the
  *entire* dataset with no time constraint. For a 6-month trajectory at 1 Hz
  that is a large download to obtain the week someone actually wants. Any real
  implementation must push a time window to the server.
- **`.das` / `.dds` are cheap metadata probes.** They answer "does this dataset
  exist, what variables does it have, what is its time range" without
  transferring data. That is how a `coverage`-style command should work.
- **ERDDAP reports errors in the body, not always in the status code.** A
  request for a non-existent variable can come back `404` with an HTML page, or
  `200` with a plain-text `Error {...}` block. **A downloader that only checks
  the HTTP status will happily write an HTML error page to `hotel.nc`.** The
  notebook's `urlretrieve` does exactly this.
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

*Against:* it puts the network **inside the processing loop**. `process_file`
runs per `.p` file, in a `ProcessPoolExecutor` — this would issue one fetch per
worker per file. A run over 1200 files becomes 1200 requests, or needs a cache
bolted on anyway. A dataset revised mid-run yields a run that used two different
references. And the artifact that a result was computed from no longer exists
anywhere.

**Rejected.** Not because fetching is hard, but because it destroys
reproducibility and makes every downstream consumer inherit a network failure
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

As B, but the fetch is content-addressed by `(server, dataset, query)` and
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

`fetch` and `build` are split for the same reason `fp07-cal` splits `pairs`
from `fit`: the network step is slow and should happen once, while the QC step
is the part that gets iterated on. A cached raw subset also means the QC rules
can be revised and re-applied without re-downloading.

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
  constraints: []               # extra server-side clauses, verbatim

qc:
  # See section 6. These are the rules the notebook applies, made explicit and
  # per-variable rather than blanket.
  time_variable: "sci_ctd41cp_timestamp"   # the sort/dedupe key
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
history:      2026-08-23T…Z: fetched <full URL> (3 chunks, 1.2M rows);
              <the QC rules applied, with counts>
source:       ERDDAP tabledap <base_url>/<dataset_id>
erddap_query: sci_ctd41cp_timestamp,sci_water_temp,…&time>=…&time<=…
erddap_fetched_at: 2026-08-23T…Z
erddap_das_checksum: <sha256 of the .das response at fetch time>
```

The `.das` checksum is the cheap way to notice a dataset has been revised: it
covers the metadata block, so a reprocessing that changes attributes or the
time range shows up. `erddap-hotel verify` can re-fetch just the `.das` and
tell you whether the dataset still matches what you built from — without
downloading the data.

### 5.4 CLI

```bash
erddap-hotel datasets -s <base_url>          # search/list dataset IDs
erddap-hotel info -c erddap-hotel.yaml       # .das/.dds probe: variables,
                                             # time range, row count estimate
erddap-hotel fetch -c erddap-hotel.yaml      # populate the cache only
erddap-hotel build -c erddap-hotel.yaml      # fetch (cached) + QC -> hotel.nc
erddap-hotel verify -c erddap-hotel.yaml     # has the dataset changed?
```

`info` before `build` is the same discipline as `fp07-cal coverage` before
`fit`: look at what you actually have before computing on it.

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
| bad values | `== 0.0` → NaN, **every variable** | per-sensor `valid_min`/`valid_max` | per-variable `valid_min`/`valid_max` |
| bad times | drop NaN only | finite ∧ within `time.min_value`…`max_value` | finite ∧ `100 ≤ t ≤ 4e9` |
| duplicates | keep **first** | `mean` \| `first` \| `last`, configurable | **mean**, not configurable |
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

Longer term, three implementations is two too many. The natural convergence is
a single `sanitize_table(ds, rules) -> (ds, stats)` that all of
`dinkum-hotel`, `erddap-hotel` and `fp07cal` call, with `sanitize_reference`
becoming a thin caller. Deliberately **not** proposed as part of this work: it
is a refactor across #149 before it has merged.

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
sees stored values. That is right, but it also means a declared `_FillValue`
(say `-9999`) is *not* masked — so the sanitiser must apply `_FillValue` itself
rather than assuming xarray did. This is easy to miss precisely because the
notebook's dataset uses `0.0` instead.

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
| F1 | ERDDAP returns an HTML/text error with a 200 | `urlretrieve` writes it to `hotel.nc`; the failure surfaces later as "not a valid NetCDF", or worse, not at all | check `Content-Type`, sniff the magic bytes (`CDF\x01` / `\x89HDF`), and refuse to cache a non-NetCDF response |
| F2 | dataset revised between runs | same URL, different data, no error | `.das` checksum recorded at build; `verify` re-checks it |
| F3 | partial/truncated response | a short file parses fine and is simply missing the tail | compare row count against the `.dds`/`info` estimate; record both |
| F4 | silent time-zone or epoch mismatch | ERDDAP `time` is seconds since 1970 UTC, but `sci_ctd41cp_timestamp` is a *separate* variable with its own units | never assume; read `units` from the response and convert explicitly |
| F5 | requesting the whole dataset | works, slowly, until it does not | `chunk_days` and a required `time_min` for large datasets |
| F6 | a variable is absent from the dataset | ERDDAP 404s the whole request | `info` first; name the missing variable |
| F7 | server down mid-run | a batch job dies hours in | fetch is a separate step; the cache means `build` never needs the network |
| F8 | rate limiting / throttling | intermittent, looks like a network error | bounded retries with backoff, on connection errors and 5xx **only** — never retry a 4xx, which is a request bug |

F1 is the one I would fix first. It is the difference between "the run failed"
and "the run succeeded against an HTML page".

---

## 9. Testing without a server

We have no ERDDAP to test against, and the design should not need one.

1. **Recorded fixtures.** Capture one real `.das`, one `.dds` and one small
   `.nc` response from Rutgers' server *once*, commit them as test fixtures, and
   serve them from a `http.server` on localhost in tests. This exercises the
   real URL construction and the real parser against real bytes.
2. **A fake server for the failure modes.** A tiny handler that can return: an
   HTML error with a 200, a truncated body, a 429, a 500, a slow response. Each
   of F1–F8 becomes a test.
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
3. **Should `erddap-hotel` and `dinkum-hotel` converge?** They differ only in
   where the rows come from; the projection/QC/write half is nearly identical.
   A shared `hotel_builder` core with two front ends is tempting but is a
   refactor of #149's code before it has merged. Revisit after.
4. **Cache invalidation policy.** Never (content-addressed, immutable), on a
   TTL, or on `.das` change? Immutable plus an explicit `verify` is the
   simplest thing that is honest.
5. **Credentials.** Rutgers' server is public. If a future one is not, no token
   should ever live in a config file that gets committed — environment variable
   or a keychain, and never echoed into `history`.

---

## 11. Adversarial review

Attacks on this design, and what changed.

| attack | outcome |
|---|---|
| "Just let `hotel.file` take a URL — it is four lines." | **Rejected.** Puts the network inside `process_file`, which runs per file in a process pool: 1200 files, 1200 fetches, and a dataset revised mid-run gives one result computed against two references. Forced Option B. |
| "Then cache it and the URL approach is fine." | **Partly accepted.** The cache is real and is in the design, but it does not fix reproducibility on its own — a cache can be cleared, and it does not record *what* was fetched. The artifact plus its `history` does. |
| "The notebook works; just port it." | **Mostly accepted, with three changes.** Its blanket `0.0 → NaN` would destroy a genuine 0 °C sample; its `urlretrieve` will write an HTML error page to a `.nc` without noticing; and it fetches the entire dataset with no time constraint. |
| "Dedupe by keeping the first, like the notebook." | **Rejected as the default.** The other two sanitisers in the tree use `mean`, and keeping the first is a silent bias toward sort order when two rows share a timestamp. Kept as an option. |
| "Check the HTTP status and you are safe." | **Rejected.** ERDDAP reports errors in the body, sometimes with a 200. Content-type plus magic-byte sniffing is the check that actually works. |
| "Retry on failure." | **Qualified.** Retry connection errors and 5xx; never retry a 4xx — a malformed variable list is a bug, and retrying it three times just makes it slower to diagnose. |
| "We cannot design this without a server to test against." | **Rejected.** The QC layer and the query builder are the parts most likely to be wrong and neither needs a network. Recorded fixtures plus a fake server cover the rest. |
| "Fetch the MicroRider data over ERDDAP too." | **Considered and dropped** (§7). ERDDAP serves tables, not `.p` binaries, so it means already-converted data — and converted values cannot be recalibrated at all: `L = ln(R_T/R_0)` needs raw counts and the bridge constants. Recalibration and epsilon both become impossible, and 512 Hz x 8 channels x 72 days (~2.5e10 samples) is not a tabledap request regardless. |
| "Use erddapy — it is the standard client." | **Rejected as a required dependency.** Neither it nor `requests` is installed, the fetch is six lines of stdlib `urllib`, and the three things urllib lacks (timeout, retry, pooling) are either trivial to add or irrelevant at tens-of-requests scale. Optional extra only. |
| "Make `erddap-hotel` share code with `dinkum-hotel` now." | **Deferred.** They genuinely overlap, but #149 has not merged and refactoring across an open PR is how you get a painful rebase. Noted as a follow-up. |

---

## 12. What I would build first

In order, each independently useful:

1. **The QC layer** — takes an `xr.Dataset`, applies `drop_zero_as_fill` /
   `valid_range` / sort / dedupe, returns a sanitised dataset with a `history`
   entry. No network. Testable today, against the notebook's own output as a
   golden case.
2. **The query builder** — config → URL list, with `chunk_days`. Pure string
   construction, testable by comparison.
3. **The fetcher** — `urllib`, with the F1 response validation, retries, and the
   content-addressed cache.
4. **`info` / `verify`** on `.das`/`.dds`.
5. **`build`** wiring the above together, plus the example config and docs.

Steps 1 and 2 are the bulk of the logic and can be written, tested and reviewed
before anyone has ERDDAP access.
