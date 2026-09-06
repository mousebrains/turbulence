# `mr-clocksync` runbook

Solve a **per-file** clock offset (and drift rate) for a MicroRider — or any
`.p`-recording instrument — against a reference pressure record, using the wave
band.

The MR clock jumps between `.p` files and runs at the wrong rate, so a single
per-deployment offset is wrong by construction. When the MR is deployed
alongside an instrument whose clock is trusted, a shared surface-wave signal in
pressure recovers the offset for each file to a few tens of milliseconds.

---

## 1. Run it

```bash
mr-clocksync init    -o clocksync.yaml     # commented template
$EDITOR clocksync.yaml                     # reference.file, targets:
mr-clocksync probe   -c clocksync.yaml     # is there a wave? ALWAYS run this first
mr-clocksync extract -c clocksync.yaml     # .p trees -> cache (resumable, ~1.7 s/file)
mr-clocksync solve   -c clocksync.yaml     # -> clock_offsets.csv + clock_report.txt
```

`probe` costs one file per target and answers the only question that decides
whether the method applies: **is there band-passed wave signal well above the
5 mm gate?** On Bank Seaspider it reports 94–130 mm. If it reports single-digit
millimeters, stop — there is no wave to synchronize on and no amount of tuning
will invent one.

`extract` is the only slow step (~12 min for 431 files / 67 GB) and it is
resumable: kill it and rerun, and it picks up where it stopped. Each `.p` file
becomes a ~700 kB `.npz`, so every later `solve` runs in seconds.

## 2. Set `max_lag` first, and set it wide

This is the setting that will waste your afternoon. The search is `±max_lag`
seconds; if the true offset lies outside, the file fails with **"envelope peak
at the ±N s search boundary"** and you get no answer rather than a wrong one.

On Bank Seaspider the MR clocks were **43 to 149 seconds** out. A 60 s search
solved 6 of 40 files; a 900 s search solved 40 of 40.

Procedure: start at 900 s, run `solve`, look at the reported offset range, then
narrow to a little beyond it. A wider search is more chances to lock onto the
wrong wave cycle, so do not leave it wide once you know the answer.

## 3. Reading the report

```
MR330  --  20/20 files solved
  offset   median  -111.271 s   MAD  30.868 s   range [-149.315, -79.362] s
  sigma    median      22.3 ms  worst 29.0 ms
  drift    median   +33.194 s/day   MAD 0.427 s/day
  step     |offset[i]-offset[i-1]| median 25.858 s   max 56.147 s
  windows  226/240 used   median coherence 0.771
```

- **sigma** is a real error bar, from the cross-spectral coherence — not the
  scatter of the windows. Tens of milliseconds is normal with a good swell.
- **windows used** well below total means many windows were rejected; the
  reasons are summarized per file in the CSV. A file solved from 3 of 12
  windows deserves less trust than the sigma alone suggests.
- **chi2** (per file, in the CSV) is the goodness of the linear clock model
  inside that file. ~1–3 is healthy. Large means the clock did something other
  than drift linearly, and it is flagged.

## 4. What Bank Seaspider's clocks actually do

Two findings from the first 20-file subset, both worth expecting elsewhere:

**These files carry the ODAS timing bug, and this dataset is where it was
found.** Both MRs record **header version 6.0**. Pat diagnosed the clocking
problem from this deployment and Rockland fixed it in later software; he
co-drove the v6.3/6.4 timing work with William at Rockland and is the authority
on `.p` timing — not TN-051, which still documents it incompletely. So the
numbers below are not a puzzle to be re-solved; they are a **known defect,
reproduced independently by this tool**, and they are the reason a v6.0
deployment cannot be trusted to its own timestamps.

**Within a file, both MRs drift at +33 s/day** — MR330 +33.19 ± 0.43,
MR429 +32.97 ± 0.38 (median ± MAD over 20 files each). Two independent
instruments agreeing to under 1% is not two bad crystals; it is systematic.
33 s/day is 3.84e-4 fractional, which against `fs_slow` = 128.0082 Hz would
imply a true rate near 127.959 Hz.

Both files report `f_clock` = **9216.590 Hz**, with `fs_fast` = clock/9 and
`fs_slow` = clock/72. The header stores that as two words, `clock_hz` = 9216
and `clock_frac` = 590, and

    f_clock = clock_hz + clock_frac / 1000

— **thousandths of a Hz, not /65536**. Two independent checks pin the /1000
form: it is what reproduces the `fs_slow` = 128.0082 Hz the reader actually
derives (the /65536 form gives 128.000125 Hz), and it makes
`f_clock` = 24 MHz / 2604 exactly, to within the header's own 0.001 Hz
resolution. Every clock value in the archive is an exact integer divisor of
24 MHz — see `docs/rsi-tpw/odas_file_format.md`, where Rockland has since
confirmed 24 MHz as the base for these devices.

That matters here because patching the rate means writing this field back
(`clock_frac` in thousandths gives 1.1e-7 fractional resolution, ~1.2 ms
across a 3 h file), and a wrong divisor would put the correction out by a
factor of 65.

**Confirmed on the full record.** The subset numbers above were reproduced
over all 431 files (Bank Seaspider EM analysis, 2026-08-26): MR330 solved
207/216, offset median −111.68 s over [−154.3, −71.0], sigma 24.9 ms, drift
**+33.068 s/day**; MR429 solved 207/215, offset median −68.78 s over
[−107.8, −28.2], sigma 17.8 ms, drift **+33.062 s/day**. The two instruments
agree on drift to 0.02%. The 17 unsolved files are all outside the reference's
coverage — pre-deployment, or post-recovery in air — and are refused rather
than guessed.

**Between files the offset ratchets and resets.** It walks steadily one way
(~−25 s per 3 h file), then jumps back (~+48 s), then walks again:

```
-89.2 -> -113.2 -> -140.0 -> -91.7 -> -111.9 -> -129.1 -> ... -> -135.0 -> -85.5
```

Range across just 20 files: **−43 to −149 s**. This ratchet-then-reset shape
matches the earlier MATLAB analysis of the same deployment, which found the
spread widening to 120 s or more before narrowing. Two consequences:

- `max_lag` must cover the whole ratchet, not the typical offset.
- The continuity check may flag the resets. **A reset is not a cycle slip.**
  Read a continuity flag as "look at this", never as "discard this".

Within files there are no cycle slips at all: 11–12 windows per file agree on a
single line with chi2 of 1–3, and a slipped window would sit ~8 s (≈400 sigma)
off it.

## 5. Why the estimator is built the way it is

Detail in the `lag.py` module docstring; the short version is three traps.

**Do not correlate raw pressure.** A tide is a ramp, and a shifted straight
line is the same line plus a constant, which every correlation removes.
`fp07cal` measured this on real data: raw pressure scored `r = 1.000000` at
*every* lag over ±30 s. Everything here is band-passed first, and the
acceptance gate is peak **sharpness**, never `r`.

**Do not `argmax` a narrowband correlation.** An 8 s swell puts a side lobe
every 8 s, and picking the largest of 36 near-equal lobes is a coin flip. The
analytic envelope has no carrier, so its peak identifies the right lobe; the
cross-spectral phase then refines inside it.

**Do not trust a lag with no wave behind it.** Band-passing a tide-only window
leaves numerical ringing that is identical in both records — coherence 1.00, a
sharp peak, and a confident lag derived from filter transients. The
`min_amplitude` gate (5 mm of band-passed water) is what stops that, and it is
the gate that fires on calm periods.

Precision, verified by injection over ±12 s of shift: **0.6 ms, with a sigma
that matches.** The MATLAB predecessor picked `max(xcorr)` at 16 Hz and was
quantised to 62.5 ms per window with no error bar at all.

## 6. Conventions that are load-bearing

- **`lag` is the number of seconds to ADD to the target's timestamps** to align
  them with the reference. Positive means the target clock is slow. Pinned by
  `tests/test_clocksync.py::test_sign_and_accuracy`, because a sign error is
  invisible in every summary statistic.
- **The reference is never resampled onto the target.** The target is decimated
  down onto the reference's own grid — the `fp07cal` convention: match
  bandwidth to the thing you trust, and invent nothing.
- **Time is float64 everywhere.** Epoch seconds are ~1.7e9 and float32 resolves
  that to ~128 s, which silently destroys every time difference. Pressure is
  float32 in the cache; time never is.
- **Gaps stay gaps.** Holes longer than `max_gap` are NaN, and a window that is
  more than 20% empty is skipped rather than filled.
- ODAS MATLAB's `Milli` is **fractional seconds** (`d.Milli = y(6) - d.Second`),
  despite the name; the `.p` header word 9 is **integer milliseconds**. Reading
  one as the other is a factor-of-1000 error in exactly the sub-second term
  this tool exists to measure.

## 7. Verifying the reference clock

The tool assumes the reference is right. Verify rather than assume: sync a
*third* pressure record to the same reference and check the result is
consistent, and sync the two targets to each other directly. On Bank Seaspider
the RBR wave gauge (`kind: netcdf`, 2 Hz, `pressure_var: pdbar`) is the
independent third record, and MR330-against-MR429 (`kind: pfile`) closes the
triangle. Three pairwise solutions that do not close are telling you the
trusted clock is not.
