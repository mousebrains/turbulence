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
| 6.1 | RDL OS ≤ 4.11 | Read; per-sample bad buffers detected; clock caveat below |
| 6.2 | RDL OS 4.12–4.16 | Read; same clock caveat |
| 6.3 | RDL OS 4.17+ | Read; nothing outstanding |

Dispatch is on `major >= 6`, so a future 6.4 reads without a code change.

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

Affected samples are **reported, never modified** — the right repair (mask,
interpolate, drop the profile) depends on the channel and the analysis. The
report's `spans` give `(start, length)` into the channel's own samples.

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
dissipation window, but a large excursion within it.

**Open action:** decide whether the ε/χ path should mask confirmed dropouts
before deriving speed. The detector deliberately does not modify samples, and
`PFile.bad_buffer_report[...]['spans']` gives the indices needed to do it.

## Sampling rate: the v6.1/v6.2 count-by-one error — UNRESOLVED

TN-051 §2.4.3–2.4.4 record that a firmware bug "caused a count-by-one error in
the data acquisition clock" in v6.1 and v6.2 files, fixed in v6.3, and that
Zissou corrects for it. **The note does not publish the correction**, so we
apply none: `fs_fast` is always `(word21 + word22/1000) / n_cols`, matching
`read_odas.m:170`.

Two findings from the local files, both from `diagnose_daq_clock()`:

**1. The clock is 48 MHz, not the 38.4 MHz of note 5.** Every observed rate is
an exact integer divisor of 48 MHz and of neither 38.4 nor 40 MHz:

| Instrument | Version | Reported | 48 MHz count | 38.4 MHz count |
|---|---|---|---|---|
| VMP-250IR SN479 | 6.1 | 5119.454 Hz | **9376** ✓ | 7500.80 ✗ |
| MR1000 SN410 | 6.0 | 4608.295 Hz | **10416** ✓ | 8332.80 ✗ |
| MR1000RDL SN435 | 6.3 | 4608.295 Hz | **10416** ✓ | 8332.80 ✗ |

Both counts land within 2 × 10⁻⁴ of an integer and round back to the stored
millihertz exactly. Two unrelated instrument families agreeing rules out
coincidence. Do not hard-code 38.4 MHz anywhere.

**2. The SN479 v6.1 files look affected.** Correct-firmware files take
`floor(clock / requested_rate)`: the MR's 4608 Hz is not an exact divisor
(48e6/4608 = 10416.67) and both its v6.0 and v6.3 files use 10416. SN479's
5120 Hz *is* exact — count 9375 — yet its v6.1 files use 9376, one too high,
putting the reported rate **107 ppm low** (511.9454 Hz where 512.0000 was
asked for).

Magnitude if the inference is right: 1.5 s of drift across a full 14400 s
file, and ~0.02 % in ε — negligible for dissipation, marginal for tight time
matching against shipboard data.

**What is unverified:** the *direction*. That the header reports a rate the
instrument did not run at (rather than faithfully reporting an off-nominal
rate it did) is inferred from Zissou "correcting" the value, not stated. The
~20 s restart gaps between the SN479 files are far too coarse to settle 1.5 s
empirically. **Confirm with `support@rocklandscientific.com` before applying
any correction.** Until then `PFile.clock_diagnosis` reports it, `rsi-tpw
info` prints a note, and nothing is changed.

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
