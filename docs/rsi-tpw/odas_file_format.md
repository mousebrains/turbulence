# ODAS `.p` file format — what we implement, and where we differ

Reference: Rockland Technical Note **TN-051, "Rockland ODAS v6 Data File
Format", rev. 2026-01-12**
([public PDF](https://rocklandscientific.com/wp-content/uploads/2026/01/TN_051_Rockland_ODAS_v6_Data_File_Format.pdf)).
That revision renamed the note (it was "Rockland Data File Anatomy"), added
the Bad Buffer material in section 3, and extended the version table to 6.3.

The reader lives in `src/odas_tpw/rsi/p_file.py`; the vendor reference is
`odas/read_odas.m` / `odas_p2mat.m` (ODAS MATLAB Library v4.5.1, 2015), which
predates this revision and therefore implements none of the v6.1+ material
below.

## Versions we handle

| Version | Source | Status here |
|---|---|---|
| 1.0 | pre-2015 | Translated to v6 on read (`rsi-tpw v1to6`, see [legacy_v1.md](legacy_v1.md)) |
| 6.0 | ODAS-RT, ODAS5IR (CF2 Persistor) | Read; whole-record bad-buffer flag honored |
| 6.1 | RDL OS ≤ 4.11 | Read; per-sample bad buffers detected |
| 6.2 | RDL OS 4.12–4.16 | Read; per-record timestamps drift (§2.4.3), unused by us |
| 6.3 | RDL OS 4.17+ | Read; nothing outstanding |
| 6.4 | not yet public | Reads as-is — no record-layout change |

Dispatch is on `major >= 6`, so 6.4 reads without a code change. Per Rockland
(via Pat, who co-drove these changes), **6.4 changes no record layout**: it adds
a subtle timing correction and **an explicit flag for stitching files
together**. When it ships, that flag should replace the timestamp-gap heuristic
described under [Time](#time) — it answers directly the question that heuristic
can only guess at. No v6.4 file appears in the 6665-file ARCTERX census.

## Bad buffers

The mechanism changed at v6.1 and the two are mutually exclusive.

**v6.0** (TN-051 §3.1) writes special character `32752` (0x7FF0) from channel
255 in every record; a failed check sets header word 16 and condemns the whole
record. We warn on a non-zero word 16 (we do not repair, unlike ODAS
`fix_bad_buffers.m`).

**v6.1 and newer** (§3.2) removed channel 255 and instead substitute `-32753`
(0x800F) for the individual missing or erroneous *sample*. Header word 16 can
no longer fire — confirmed empirically: it is zero in every record of all 29
SN479 v6.1 and 10 MR v6.3 files on hand. `PFile._check_bad_buffer_markers`
scans for the sentinel and reports findings in `PFile.bad_buffer_report`.

Detection keys on **run length**, because the sentinel is an unlikely value,
not an impossible one. Measured over the SN479 v6.1 set (2.9 × 10⁸ samples),
the two populations are cleanly separated:

| | channels | per file | run lengths |
|---|---|---|---|
| Real dropouts | `DO_T` (addr 152), 14 of 29 files | 1–4 runs | **63 or 64** |
| Coincidences | `JAC_T`, `JAC_C`, `Turbidity` | 1–24 hits | **1**, rarely 2 |

The coincidences sit on channels whose signal rides the rails — `JAC_T` is
`sign = unsigned`, so the sentinel's bit pattern (32783 unsigned) is ordinary
mid-range data for it. The real dropouts enter from values as far away as
+32741 and hold exactly `-32753` for a full 63–64-sample buffer. The threshold
is 4, sitting in the empty gap with 16× margin either side.

Affected samples are **reported, never modified by the reader** — `PFile`'s
contract is detection only, and the repair happens at the load boundary
instead (see [below](#repairing-and-masking-ε-and-χ--resolved)), so the raw
reader and the archival per-profile NetCDF both keep the file as it was. The
report's `spans` give `(start, length)` into the channel's own **extracted**
samples, and each entry declares the `rate` (`fast`/`slow`) those indices are
on. The rate has to be recorded at detection time: the scan runs before
deconvolution, which demotes a base channel sampled as a full fast column to a
slow-length view and reclassifies it (`_apply_deconvolution`, "both branches
leave the base holding a slow-length view, no matter how it was sampled").
Afterwards neither the stored length nor `is_fast()` reports the axis the scan
measured runs on.

Extraction and detection do not cover the same cells, and `spans` follow
extraction: `_read` keeps a fast channel's whole column but only the **first
occurrence** of a slow address, and joins the two words of a 32-bit (2-id)
channel. So `spans` is the union over the channel's first-occurrence series,
and it can be empty for a confirmed channel whose only long run sits in a
decimated occurrence — dirty in the file, clean in the data we use.

### Which channels are affected, and does it matter

On the SN479 VMP files the dropouts land only on `DO_T` — an auxiliary channel
feeding neither ε nor χ, so those products are unaffected there.

**That does not generalize.** A 76-file sample across the archive also found
dropouts on **`U_EM`** in glider MicroRider files (`A685_0241.p`,
`A685_0250.p`: 1–3 runs of up to 7 samples). `U_EM` is a *direct* ε input
whenever `speed.method = "em"` — `rsi/speed.py` takes it as the through-water
speed that normalizes shear and converts frequency to wavenumber.

A sentinel sample converts to **3.52 m/s** through the `aem1g_d` calibration,
against a real glider speed of ~0.43 m/s (observed range −0.27 to 0.53). That
is a 6.6× outlier which nothing downstream currently rejects: `_slow_to_fast`
low-pass smooths it rather than removing it, so it contaminates a window wider
than the dropout itself, and ε carries roughly a U⁻⁴ sensitivity to the result.
At 64 Hz slow rate, a 7-sample run is ~0.11 s — a small fraction of a
dissipation window, but a large excursion within it. Under the treatment below
that run is short enough to interpolate away; a full 64-sample `U_EM` buffer
loss (1.0 s) is not, and rejects its windows.

### Repairing and masking ε and χ — RESOLVED

`odas_tpw.rsi.bad_buffer` grades every confirmed run by **duration** and treats
the two regimes differently:

| gap | treatment |
|---|---|
| ≤ 0.25 s (`MAX_INTERP_S`) | **linearly interpolated** in the channel data, at the load boundary, before anything consumes it |
| > 0.25 s | **rejected** — every ε/χ estimate whose window overlaps it is set NaN |

Interpolating across a gap removes roughly its own fraction of the variance, so
for a gap that is a small part of an FFT segment the bias is far inside a single
dissipation estimate's own uncertainty; a long contiguous gap has no information
to interpolate across at any scale. The threshold is where the observed dropouts
actually separate: the RDL always loses one fixed 64-sample buffer, which is
**0.125 s on a fast channel** (sh, T_dT at 512 Hz) but **1.0 s on a slow one**
(U_EM, P, DO_T at 64 Hz). A window is rejected anyway once more than
`MAX_INTERP_FRACTION` (5%) of it has been interpolated, bounding the accumulated
variance loss.

Both are published per probe × time — `bad_buffer_fraction` (rejected) and
`interpolated_fraction` (repaired) — so the treatment is auditable, and
`mask_bad_buffers=False` keeps the estimates for anyone who wants to filter
differently. On the χ side the rejection happens inside L4 *before* `chi_final`
is formed, so a contaminated probe cannot be averaged back into the reported χ.

The handling is **dependency-scoped** — a dropout only affects what that channel
actually feeds:

| channel | affects |
|---|---|
| `sh{i}` | ε for probe *i* only |
| `T{i}_dT{i}` | χ for thermistor *i* only (and its `T{i}` base — deconvolution couples them) |
| accelerometer / piezo | ε for **every** probe, but only when Goodman is on: it mixes the vibration reference into every shear spectrum |
| `P` / `P_dP` | both, always (depth), plus speed on the pressure and flight paths |
| `U_EM` | both, **only when the speed actually came from the EM flowmeter** |
| `Incl_X` / `Incl_Y` | both, only when the speed came from the flight model |
| reference `T`, `C` | both, via viscosity / κ_T |

The `U_EM` row is the point of the design. Under a **flight model** `U_EM` is
not a speed input — `speed.py` reads it only to raise a disagreement warning —
so a `U_EM` dropout is neither repaired nor rejected there, even though it is
exactly the channel the archive scan found dropouts on. Conversely the flight
model *does* read `W_slow`, hence pressure, so a `P` dropout still counts. Getting this from the
method name alone is not safe (a precomputed `speed_fast` reads no channel of
this file at all), so `prepare_profiles` — the branch that knows which speed
path won — stamps `metadata["speed_channels"]` with what it consumed, and the
mask builder reads that.

Smears are applied to the **rejected** grade only, because an interpolated
sample carries a plausible value before any filter sees it: slow-channel masks
expand to every fast sample whose `np.interp` stencil touches the bad sample;
speed inputs widen by the Butterworth smoothing constant; and a pre-emphasized
channel widens forward by 3 × `diff_gain`, the e-folding reach of the
deconvolution filter.

Masks ride through the per-profile NetCDF as a `bad_buffer_spans` attribute on
each affected variable — `start:length:grade`, profile-local indices on that
variable's own axis — so the `prof → eps/chi` and perturb routes behave like the
direct `.p` route rather than silently losing the dropouts at the file boundary.
The grade travels with the span because it cannot be recomputed downstream: it
depends on the run's duration on its original axis, and a profile slice can cut
a run short.

Verified as a null test on the SN479 set: because every dropout there is on
`DO_T`, which feeds nothing, ε, χ, `fom` and `epsilon_T` come out **bit-identical**
to the pre-masking code across 29 profiles.

## Sampling rate and the §2.4.3 count-by-one error — RESOLVED

**We are not affected, and `fs_fast` is correct as read.** `fs_fast` is always
`(word21 + word22/1000) / n_cols`, matching `read_odas.m:170`.

TN-051 §2.4.3 notes that "a firmware bug caused a count-by-one error in the
data acquisition clock" in v6.1/v6.2, fixed in v6.3, and that Zissou corrects
for it. Read in context that paragraph sits inside **"Version 6.2 Time"**,
describing how records get synthesized date-times "derived from the previous
record time and the expected duration of a record"; the version table calls
6.3's fix "the **timing error** in version 6.2 data files". So the bug is in
the **record-duration accumulation used to synthesize per-record timestamps**,
not in the sampling rate.

We never read per-record timestamps — the absolute start comes from record 0
only and every later time is derived from `fs_fast` (see [Time](#time)) — so
the bug cannot reach our time base. Anyone who *does* consume per-record
header date-times from a v6.1/v6.2 file should expect them to drift.

Two checks confirm the rate itself is right, correcting an earlier reading of
this document that treated 511.9454 Hz as a symptom:

- **Vendor agreement.** ODAS MATLAB's `_allch.nc` for a v6.1 SN479 file gives
  `fs = 511.945400 Hz` — identical to our header-derived value to 0.00 ppm.
  The vendor library applies no rate correction.
- **Operator experience.** 511.9454 Hz is the rate this instrument class has
  reported historically; it is the by-design rate, not a fault.

For the record, the coincidence that prompted the earlier misreading: of the
four archive configurations, three take `count = floor(clock / nominal)`
(truncation, not rounding — 10417 would be *nearer* 512 Hz than the 10416
used), while the 10-column @512 Hz config, whose division is exact at 9375,
uses 9376. That is numerology on an unrelated coincidence, and nothing in the
file states a requested rate anyway — the config carries no rate or frequency
key at all (`[root]` holds only `prefix`), so the requested rate lives in the
RDL OS settings, outside the data file.

## The documented 38.4 MHz clock is wrong — OPEN, report to Rockland

Independent of the above, and still worth reporting. TN-051 note 5 states a
38.4 MHz data acquisition clock. A header-only scan of 5360 v6+ files finds
exactly **four** distinct reported clock values, none consistent with it:

| f_clock | cols | fs_fast | files | versions | 48 MHz count | back-calc err | 38.4 MHz count |
|---|---|---|---|---|---|---|---|
| 4096.262 | 8 | 512.0327 | 292 | 6.0×15, 6.1×264, 6.2×13 | **11718** | 0.0002 Hz | 9374.**400** |
| 4608.295 | 9 | 512.0328 | 3049 | 6.0×5, 6.1×1550, 6.3×1494 | **10416** | 0.0001 Hz | 8332.**800** |
| 5119.454 | 10 | 511.9454 | 1138 | 6.0×90, 6.1×1001, 6.2×47 | **9376** | 0.0001 Hz | 7500.**800** |
| 9216.590 | 9 | 1024.0656 | 881 | 6.0×647, 6.1×18, 6.2×216 | **5208** | 0.0001 Hz | 4166.**400** |

All four are integers under 48 MHz to within 5 × 10⁻⁴, and back-calculating
`48e6/count` reproduces the stored value to ≤ 0.0002 Hz — inside the header's
own 0.001 Hz resolution. Under 38.4 MHz none is an integer and the back-calc
errors are 0.11–0.88 Hz, 100–900× that resolution.

The failure is structural: **38.4/48 = 0.8 exactly**, so the 38.4 MHz "counts"
are precisely 0.8× the 48 MHz ones — every one landing on .400 or .800. A base
that is 4/5 of the true base yields an integer only when the true count is a
multiple of 5, and none of these are.

A search over 19 candidate bases leaves only **24, 48, 72 and 96 MHz** viable
— integer multiples of 24 MHz; 38.4 and 40 MHz are excluded. The data cannot
distinguish 24 from 48 MHz (all four counts are even, which either favors
24 MHz or indicates a divide-by-2 prescaler), so the defensible claim is *a
multiple of 24 MHz, and definitely not 38.4*. Do not hard-code 38.4 MHz.

This is a documentation-accuracy issue only: we read the frequency from words
21/22 and never use a base clock, so nothing in our processing depends on it.

*(The 90 files claiming v6.0 while reporting 5119.454 Hz were checked: all 90
carry a `; <date> patched configuration string` first line — the vendor
`patch_setupstr.m` signature — and none has channel 255 in its matrix or a
single 32752 special character, which TN-051 §3.1 requires of a genuine v6.0
acquisition. They are v6.1 files whose version word was rewritten downstream,
not v6.0 acquisitions.)*

## Header fields

All 20 fields we read match TN-051 Table 3. Deliberate choices:

- **Words 2 and 20 (record number, records written)** are running counters in
  practice, not file totals: record 0 carries 0 and data record *k* carries
  *k*, in both. The record count still comes from the file size, as in
  `read_odas.m`, but word 2 is now checked for **forward jumps**, which mean
  records went missing in transfer — otherwise a silent time shift, since
  timestamps come from the sample index. A counter that *resets* is treated as
  a splice boundary rather than damage: that is what `perturb merge` output
  and a uint16 wrap at 65536 records both look like. Files we emit therefore
  carry counters that are correct within each spliced segment (`trim` and
  `patch-config` preserve them exactly; `merge` restarts at each boundary and
  `cutp` keeps the source's origin offset).
- **Word 13 (product ID)** is read but unused; §2.1.1 note 2 says it is
  permanently 1 from v6.1 on.
- **Word 63 (profile flag)** is read but unused; note 7 says it is obsolete.
- **Word 64 (endian)** drives byte order, with `0 → little` matching
  `fopen_odas.m`. Where the flag is present but contradictory we fall back to
  testing `header_size == 128` in both orders, which is more permissive than
  the vendor's hard error.
- **Words 29/30/31** are the address-matrix dimensions and are cross-checked
  against the config `[matrix]`; a mismatch raises, since the data block is
  reshaped on the header while channel columns come from the config, and
  disagreement would silently read the wrong multiplexer slots.

## Time

Per §2.4.1 we take the absolute start from record 0 only and derive every
subsequent timestamp from the sampling rate; per-record header stamps are not
used. ODAS defines the data start as the header time **minus** one record
duration (`odas_p2mat.m:413`, `recsize` defaulting to 1.0 s), which we match.

§2.4.5 (contiguous files) interacts with `perturb merge`: for v6.0/v6.1 the
header stamp is when a record was *written*, and the RDL may buffer minutes of
records first, so a timestamp gap is not evidence of a data gap. Sequential
files whose gap falls between the 5 s chaining tolerance and 5 minutes are
therefore **not** merged — splicing a real gap into one time base is worse
than leaving files apart — but the ambiguity is warned about so an operator
can check the acquisition log. v6.2+ stamps are calculated and trusted as-is.

**This heuristic has a known expiry date.** v6.4 adds an explicit
stitch-together flag (see the version table), which states outright what the
gap test can only infer. Replace `_warn_if_gap_may_be_a_timestamp_artifact`
and the `_MERGE_GAP_AMBIGUOUS_S` band with that flag once v6.4 files appear.

### Clock drift between instruments

The instrument clock drifts enough to matter when aligning *across*
instruments over long deployments, though not within one. Measured by Pat over
a month-long deployment: **~10 s per month, i.e. ~3.9 ppm** — above the
~1.5 ppm accuracy and 0.5 ppm temperature stability TN-051 note 5 quotes. Two
MicroRiders were self-consistent and a CTD + Signature1000 were self-consistent
but the two groups diverged; calibrating against wave phase shifts and
arbitrating with pressure sensors and tidal models showed **the MicroRiders
were the ones drifting**.

Irrelevant to dissipation — 3.9 ppm on `fs_fast` does nothing to a spectrum —
and this package applies no drift correction. It matters only when MR data is
time-matched against another instrument over weeks (ADCP shear, CTD casts),
where ~10 s/month of relative offset accumulates.
