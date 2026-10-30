# `fp07-cal` runbook

FP07 in-situ calibration for a MicroRider carried by a Slocum, against the
glider's own glider CTD, or for a VMP against its own JAC CT (§8). It is a
**pre-pipeline** step: it runs once per deployment and hands perturb `.p`
files that already carry the right coefficients, so perturb itself needs no
changes.

```
raw *.p ───► perturb trim ─► trimmed *.p ─┬─► fp07-cal fit ─► coefficients.json
                                          │                    + reports
*.ebd ─────► (any converter) ─► hotel.nc ─┘        │
                                                    ▼
                              trimmed *.p ─► fp07-cal patch ─► calibrated *.p
                                                                     │
                                                                     ▼
                                          perturb run  (fp07.calibrate: false)
```

## 0. Check the install

```bash
fp07-cal demo --yos 40 --ct-every-n 3
```

Builds a synthetic deployment from known coefficients and recovers them. It
prints truth alongside the recovered values; `t_0` should come back within a
few mK and `beta_1` within ~100 ppm. If this fails, nothing below will work.

## 1. Write a config

```bash
fp07-cal init -o fp07-cal.yaml
```

The template is commented. The three settings that actually matter:

| setting | why |
|---|---|
| `reference.pressure_scale` | A Slocum reports `sci_water_pressure` in **bar**. Use `1.0` if the ×10 was applied when the hotel file was built; **`10.0`** when pointing straight at a converted `ebd.nc`, which is what was validated. Getting this wrong does not affect the lags (correlation-based, scale-invariant) but does corrupt the sensor-geometry term. Sanity check: pressure should read ~1000 at 1000 m. |
| `pairs.max_gap` | Defines where the reference *exists*. Not a smoothing knob. CTD samples further apart than this are treated as separate coverage spans, and nothing is interpolated across the hole. |
| `files.max_fit_files` | How many files are held in memory for the lag and fit stage. Everything else streams. 100 is plenty; a full deployment at once is several GB. |

Leave `fit.order` at `"auto"` — see §4.

## 2. Look before you fit

```bash
fp07-cal coverage -c fp07-cal.yaml
```

Reports how much reference you actually have: sample count, continuous spans,
duty cycle, temperature range, and pairs contributed per file. Files
contributing zero pairs are **normal** for a deployment that ran CT on only
some yos — they still get calibrated, from the pooled fit.

## 3. Fit

```bash
fp07-cal fit -c fp07-cal.yaml
```

Writes `coefficients.json`, a per-channel `*_report.md`, and diagnostics PNGs.
**Read the report before using the numbers.** Specifically:

- **`lag ... — ok` vs `NOT TRUSTWORTHY`.** A glider dive is a monotonic ramp,
  and a shifted straight line is the same line plus a constant, so a raw
  correlation sits at r ≈ 1.0 at *every* lag. The gate is on peak sharpness,
  never on r: the peak must be interior and must fall by at least 0.10 across
  the search. If the lag is not trustworthy, the coefficients are not either,
  and `patch` will refuse them. A search too narrow to show that fall (a VMP's
  ±1 s) is judged on the peak's curvature and on two halves of the profiles
  finding it again; see §8.
- **`REFUSED`.** No coefficients are exported for a channel whose fit
  mispredicts its own reference by more than 1 K anywhere, or whose curve is
  not an FP07's (T must fall as L rises over the fitted range). The 1 K gate
  is judged over the pairs the fit used: pairs its robust loop already
  rejected as outliers are reported in a `WARNING` (count, worst error, time,
  pressure, file) and in the record's `outliers_beyond_gate`, but no longer
  refuse the fit. The loop judges each residual about the median of the pairs
  it keeps, so a minority population (part of the reference a few K off, say)
  is rejected whole instead of dragging the fit off the majority (issue #246).
  It is rejected whole whatever its offset, 50 mK as well as 5 K, and is then
  left out of the stability blocks too, so a `WARNING` and the record's
  `robust_rejection` say when the loop rejected a population rather than a
  tail: more than 15% of the pairs, or at least half of the rejections in
  profiles that each lost more than half their pairs, when those hold at least
  0.5% of the pairs (11 real deployments: at most 7.5%, 12.7% and 0.44%). Look
  at those times before trusting the fit or its stability. With the geometry
  term the loop also judges the pairs it cannot fit for want of a reference
  gradient (profile ends, short profiles), by the calibration's residual, so a
  population is flagged there too. Nor do rejected pairs set the fitted range (the report's "do NOT
  extrapolate" line, the record's `T_range`, `patch`'s extrapolation warning)
  or the warm/cold folds that choose the order: pairs every order rejects are
  left out of those. A fit that is not an FP07 curve is refused with
  `error: "not an FP07 curve"`: typically water with too little temperature
  range, where a fit to noise can take either sign (an isothermal synthetic
  gave β₁ = −1.06×10⁵). The same refusal, with `beta_range_K` in the record,
  meets a curve of the right sign whose slope no FP07 has: a local beta
  (`1/(d(1/T_K)/dL)`, the slope with the `T_K²` every thermistor shares
  divided out) outside 2723-3416 K anywhere over the fitted range. Every
  calibration in the sheet lies within 2995-3105 K (`prior.FP07_BETA_K`; a
  test fails if one leaves it), and the bound adds 10% each way; isothermal
  noise fits of the right sign came out 14-42 times an FP07's (issue #247). A
  `fit.prior` block (§5) lends such a deployment the bead's shape.
- **`## Notes` in the report.** A requested geometry term that could not be
  fitted (no profile spans the ~30 dbar the reference gradient needs; the fit
  is then made without it, and the record has `geometry_skipped`), and
  non-finite thermistor counts (the record's `thermistor_gaps`; see the
  gotchas).
- **the dive/climb split.** Should be ~0. Nonzero means an unremoved lag being
  absorbed into `t_0`. Unavailable if you record only one leg.
- **the geometry collinearity note.** If it fires, `dz` and the residual lag
  are not separately resolved — only their combination is measured. Recording
  some dives fixes this.
- **the fitted temperature range.** The coefficients are not valid far outside
  it.

## 4. Order is chosen for you

`fit.order: "auto"` picks by **held-out** error, splitting on temperature (fit
the warm half, predict the cold half). In-sample fit improves monotonically
with order, so it can only ever say "more"; what matters is extrapolating onto
profiles outside the fitted range.

On osu685 (24 °C of range) this picks order 2: it halves the in-sample residual
*and* improves extrapolation 2–5×. Order 3 gains 0.013 mK in sample and makes
extrapolation **four times worse** — while carrying a t-statistic of 10, which
a significance test would have waved through. Over a narrow range (<8 °C) the
same test will pick order 1.

## 5. Reusing a bead: fit with its prior

A bead that has been calibrated before has a **prior** in the FP07 calibration
sheet (`microstructure_sensors/fp07_calibrations.yaml`): the shape (`beta`
terms) and `t_0` of its widest-range unflagged calibration. Across deployments
a bead's shape carries to a few mK, its `t_0` only to ~15 mK (issue #189). So
when a deployment cannot pin the shape itself, usually because its temperature
range is narrow, the prior's shape with a freshly fitted `t_0` extrapolates
better than the deployment's own full fit. Add a `fit.prior` block to the
config (`fp07-cal init` writes it commented out):

```yaml
fit:
  order: "auto"
  prior:
    sheet: "<CONFIG_DIR>/../microstructure_sensors/fp07_calibrations.yaml"  # required
    mode: auto          # auto | offset | full
    beads: {T1: auto, T2: auto}   # optional; auto, a serial (T1592), or none
    tie_fraction: 0.05  # optional; see step 4
```

`sheet` is required: the sheet lives in the repository, not in the installed
package. Without the block, `fit` writes exactly what it did before, byte for
byte.

What `fit` then does, per channel:

1. **Identifies the bead** from the `.p` configs' serial. It refuses a
   placeholder (`T`) and a serial that differs between the files of one run;
   name the bead with `beads.T1: T1592`, or turn the prior off for a channel
   with `none`. Every sheet entry of the bead whose dates overlap the span
   being fitted is excluded from its prior. A bead is in one place at a time,
   so that is the deployment's own entry when the sheet already has one.
2. **Compares the bridge constants** of this channel with the prior source's.
   They define L, the variable the shape is a function of. When they differ,
   or the prior comes from another instrument or channel, the prior's `t_0`
   uncertainty includes the cross-instrument term (±18 mK instead of ±15).
3. **Fits two ways**: the full fit (orders 1–3, as in §4), and an
   offset-only fit with the prior's shape held fixed and only `t_0` fitted,
   with the same robust iteration. With `geometry: true` the offset-only fit
   is joint too, so it solves for `t_0`, `dz` and `tau`.
4. **Chooses.** Every candidate is scored on the same warm-half/cold-half
   folds that pick the order; a fold that any candidate cannot be fitted on
   is dropped for all of them. In `auto`, the prior's shape is kept:
   - when no full order is a **plausible** curve for this bead: T must fall as
     L rises, with dT/dL within ±10% of the prior's at the same L. Across the
     sheet's 23 calibrations of 12 beads, dT/dL at one temperature varies by
     at most 1.65%. In isothermal water a flat fit to noise wins on held-out
     error with a slope 20 or more times wrong, sometimes of the right sign
     (a synthetic exported β₁ = 6.8×10⁴, 7.6 K wrong at 20 °C, before this
     bound). If only the held-out-best order is implausible, the next
     plausible order stands in for it;
   - when the deployment spans less than the sheet's shape span (5 K). Held-out
     error over a 2 K range extrapolates only ~1 K and cannot test a shape that
     will be used across 10 K;
   - when its held-out error is within `tie_fraction` (default 5%) of the full
     fit's. The full fit has to be clearly better: on a wide synthetic range a
     prior shape wrong by the sheet's own ±5 mK lost by only 4%. A tie goes to
     the prior.

   `mode: offset` forces the prior's shape; `mode: full` keeps the full fit
   (today's order, implausible or not) and uses the prior only for the check
   below.
5. **Checks the result against the prior**, whichever candidate won. It
   compares the two curves at the same L over the overlap of the prior's
   fitted range and this deployment's. The limit at the center is twice the
   prior's `t_0` uncertainty. Anywhere else in the overlap it is twice
   √(σ_t0² + σ_shape²), because the shape is known only to σ_shape away from
   where the curves are pinned. When the ranges do not overlap, the comparison
   runs over this deployment's range, where the prior is extrapolated, with
   the wider limit everywhere, and the record says it is an extrapolation.
   Beyond either limit the check prints a `WARNING` and heads the report: a
   possible unrecorded bead swap, reference problem or failing bead, or a
   wrong serial in the config. An exported fit that is not plausible (step 4)
   is flagged too.

`coefficients.json` carries a `prior` record per channel:

- the source entry, and the sheet's path and SHA-256;
- the bridge comparison;
- every candidate's held-out error, its rms over **all** pairs (the same set
  for every candidate) and over the pairs its own robust loop kept, the
  fraction dropped, and its plausibility;
- the margin, the choice and why;
- the comparison with the prior over the overlap, and the verdict.

A `WARNING` also fires when the offset-only fit's robust loop drops 2
percentage points more of the pairs than the full fit's. A shape that does not
fit clips the ends of the range, and looks better over the pairs it kept than
it is. `<ch>_report.md` has the same as a "Prior" section. The exported
coefficients stay in `.p` config form (`t_0` and the prior's `beta` terms,
verbatim), so `patch` (§6) needs nothing new.

Worked example, bead T1592 on VMP SN194 against its own JAC CT. The prior is
ARCTERX-2022 Interior (order 3, 18.0–28.35 °C); in each case the deployment's
own entry was excluded:

| deployment | range [°C] | held-out, full fit | held-out, prior's shape | margin | kept |
|---|---|---|---|---|---|
| SUNRISE 2021 WaltonSmith | 27.2–29.2 | 5.49 mK (order 1) | 5.20 mK | −5.4% | prior's shape (narrow) |
| ARCTERX-2022 Wake | 18.4–28.2 | 5.32 mK (order 2) | 5.87 mK | +10.3% | full fit |
| ARCTERX-2022 Wake, geometry on | 18.4–28.2 | 5.53 mK (order 2) | 5.77 mK | +4.2% | prior's shape (tie band) |
| SUNRISE 2022 PointSur (fast descents) | 22.5–32.1 | 24.11 mK (order 1) | 23.91 mK | −0.8% | prior's shape (tie band) |

On WaltonSmith the two candidates agree to −0.5..+1.1 mK over the fitted range.
Extrapolated to 18 °C they part:

- **Against ARCTERX-2022 Wake** (a separate fit of separate data, a week after
  the prior's source, through the same JAC), the prior's shape stays within
  4–7 mK over 18–28 °C, while the full order-1 fit is 30 mK off at 18 °C.
  This is the independent comparison.
- **Against ARCTERX-2022 Interior** the prior's shape differs by a constant
  −3.9 mK only because it IS that curve with another `t_0`: circular, so it
  shows nothing about the shape.

With T2005's prior deliberately applied to T1592's channel, the check reads
−2500 mK against a 36 mK limit and flags it. The held-out choice cannot catch
this, because the two beads' shapes are alike: identity shows in `t_0`.

## 6. Patch

```bash
fp07-cal patch -c fp07-cal.yaml --dry-run    # inspect the edits first
fp07-cal patch -c fp07-cal.yaml
```

Writes corrected `.p` copies with the original configuration retained
commented-out and a provenance banner. It **refuses** on: an instrument serial
mismatch, any bridge-parameter mismatch, a lag that failed the sharpness gate,
or an input that already carries an **fp07-cal** banner (stacking two in-situ
calibrations would make the provenance ambiguous). A file patched by
`rsi-tpw patch-config` for some other reason — a bridge-parameter fix, say —
is fine: `config_patch` keeps a single frozen original-config block across
passes.

Then point perturb at the patched files with:

```yaml
fp07:
  calibrate: false      # the files already carry the in-situ coefficients
```

## 7. Record the fit in the calibration sheet

`coefficients.json` lives with the deployment's processing. The bead's history
lives in `microstructure_sensors/fp07_calibrations.yaml`. Record one entry per
channel, one bead in one deployment:

```bash
fp07-cal registry add fp07cal/coefficients.json --channel T1 \
    --deployment "SUNRISE 2021 WaltonSmith" --instrument-model vmp-250 \
    --method vmp-own-jac --reference-kind "JAC CT" --reference-serial 23 \
    --reference-cal-date 2016-02-15 --reference-pole-s 0.169 \
    --flag lag-gate-overridden --notes "..." --dry-run
```

It builds the entry from `coefficients.json` and the block table of
`<ch>_report.md` beside it, and takes from the options only what the record
cannot supply. The entry's `first`, `last` and `fit.n_files` are the
time the bead was ALIVE (issue #249): the record (`schema: fp07-cal/2`)
carries a `span` per exported channel, the files of the run in which its
thermistor recorded live data, from the first sample of the first to the last
of the last. A file is dead for a bead when half or more of its 4 s windows
have counts that do not move or touch the ADC rail (fp07-pair's dead-bead
rule, with `clipped_counts`), or when the whole file spans less than ~10 mK
(a test probe reading its `T_0`). `fp07-cal fit` judges every file of the run
in its streaming pass; with `--no-stream` on a `max_fit_files` subsample the
span's `from` says so, and `registry add` warns that a dying bead's end can be
a file stride off. A bead that died before the run ended gets its own end:
SUNRISE 2021 Pelican's T1593, on the ADC rail through SR1P_516, ends
2021-06-28T02:58:43Z in 11 files, not at the run's 2021-06-29T10:33Z in 12.
`first_kept_pair`/`last_kept_pair` are the extent of the pairs the fit kept,
the calibration's support, which can be much shorter (a glider with CT on
some yos). A bead swinging between the rail and nonsense in fewer than half
its windows still counts as alive: screen such files out (`files.exclude`). An
`fp07-cal/1` record (before #249) has only the run's span; it is still read,
by `patch` too, and `registry add` takes the run's span and file count from it
and says so. It adds the entry to the YAML as text, leaving every existing
line and comment untouched, then validates the whole sheet and re-renders the
`.md` and `.csv`. `--dry-run` prints the entry and writes nothing. It refuses:

- a second entry for the same bead and deployment;
- a placeholder serial (give `--bead`);
- a lag the trust gate rejected, unless `--flag lag-gate-overridden`;
- a fit the prior check flagged, unless `--allow-qc-flag`.

A fit made with the prior's shape is recorded with `order_choice: prior` and
`fit.shape_from` naming the source entry. The loader checks that its `beta`
terms are that entry's exactly. Such an entry never becomes the source of a
prior itself, because its shape was not fitted over its own range. The
field-by-field mapping is in `microstructure_sensors/README.md`. Commit the
three sheet files together; never edit the `.md` or `.csv` by hand.

An entry can also carry the bead's FP07 time constant, measured from the same
deployment's chi products with `fp07-pair tau` over the entry's own span
(`docs/fp07pair/runbook.md` §6) and recorded on both beads of the pair:

```bash
fp07-cal registry response fp07tau/fp07tau_SN194_T1592-T2005_20220321T0459Z.json \
    --bead T1592 --deployment "ARCTERX-2022 Interior" --dry-run
```

It inserts a `response:` block at the end of that entry, as text, and refuses a
record of another instrument or channel, a measurement outside the entry's
span, and an entry that already has one. It is a single-pole-equivalent tau
whose absolute level nothing anchors; the sheet's header and the generated
`.md` say what it does and does not mean.

The sheet also prints a bead's prior directly, for a deployment with no
reference at all:

```bash
fp07-cal registry show T2811 --instrument 435 --channel T1
```

This prints the coefficients of the bead's widest-range unflagged calibration as
config lines, with their uncertainties and how far the most recent calibration
sits from them. The rule and the uncertainty sizes, with their evidence, are in
`microstructure_sensors/fp07_calibrations.md`. When pasting the lines into a
`.p` config, remove any legacy `beta` key from that channel: the reader uses
`beta` before `beta_1`.

## 8. A VMP against its own JAC CT

A VMP carries its reference: the JAC CT on the same logger, sampled on the
same slow clock as the thermistors. The same commands apply, with a reference
built from the instrument's own samples and a few settings changed.

**The reference.** Every n-th slow sample of `JAC_T` and `P`, on the
instrument's own time stamps. A subset is fine; nothing may be interpolated,
so samples PFile reports as bad buffers (or rebuilt after a framing repair)
are blanked, and the NaN drops them. `bad_buffer_mask` is chi's mask, as
`fp07-cal` uses it for the thermistors: `P` with its pre-emphasised `P_dP`
folded in and widened over the deconvolution's 3 `diff_gain`. A marker run on
`P` or `P_dP` moves the deconvolved `P` over the whole file, as on a
thermistor (issue #259), so `smeared_file_wide` drops that file's `P` whole:

```python
from pathlib import Path

import numpy as np
import xarray as xr

from odas_tpw.fp07cal.series import bad_buffer_mask, smeared_file_wide
from odas_tpw.rsi.p_file import PFile

t, T, P = [], [], []
for path in sorted(Path("VMP").glob("*.p")):
    pf = PFile(path)
    ts = pf.start_time.timestamp() + np.asarray(pf.t_slow, dtype=np.float64)

    def real(name):
        x = np.asarray(pf.channels[name], dtype=np.float64)[: ts.size].copy()
        x[bad_buffer_mask(pf, name)[: ts.size]] = np.nan  # not measurements
        if smeared_file_wide(pf, name):  # a marker run, through the deconvolution
            x[:] = np.nan
        return x[::8]

    t.append(ts[::8])
    T.append(real("JAC_T"))
    P.append(real("P"))
ds = xr.Dataset({"JAC_T": ("obs", np.concatenate(T)), "P": ("obs", np.concatenate(P))},
                coords={"time": ("obs", np.concatenate(t))})
ds["time"].attrs["units"] = "seconds since 1970-01-01"
ds.to_netcdf("jac_reference.nc")
```

**The config** differs from the glider template in these blocks:

```yaml
reference:
  file: "<CONFIG_DIR>/jac_reference.nc"
  time_var: "time"
  value_var: "JAC_T"
  pressure_var: "P"
  same_clock: auto        # detected; see below
pairs:
  max_gap: 1.0            # [s] the reference is continuous at 8 Hz
  kernel_width: 0.047     # [s] three slow samples
  kernel_tau: 0.17        # [s] the JAC T response, 1/(2 pi fc) with the vendor
                          # relation perturb uses, fc = 0.73 sqrt(W/0.62) Hz,
                          # at the median fall rate W: 0.17 s at ~1 dbar/s
  min_speed: 0.0
profiles:
  speed_var: null
  W_min: 0.3              # [dbar/s] a VMP falls at 0.6-1
  P_min: 3.0
  min_duration: 20.0
  direction: down         # the hauled-in ascent is in the body's wake
lag:
  max_lag: 1.0            # [s] the lag is ~0.1 s
  step: 0.015625          # [s] one slow sample
  detrend_s: 10.0
```

What then differs from a glider:

- **The clock offset is zero by construction**, and the ±40 s pressure search
  is skipped. `same_clock: auto` (the default) detects it: within the files'
  spans, the reference's samples are the instrument's own, each time stamp
  within 1 µs of a sample time and its pressure within 1 mm of the
  instrument's `P` there. A separate CTD cannot pass that by accident, and a
  CTD resampled onto the instrument's grid fails the pressure half. The record
  says `clock_source`; the coefficients are unaffected either way.
  `same_clock: true` asserts the zero without checking: `clock_source` then
  says it was asserted, not measured, and that `clock_offset_s` is NOT a value
  for perturb's `hotel.time_offset`. On SUNRISE
  2022 PointSur (77 files) the search took 232 s of a 894 s run, to return
  −1.3 µs; the temperature lag search (650 s) is the rest.
- **The lag is trusted on curvature and halves.** The search is ±1 s, where
  even a perfectly identified peak falls by only 0.002–0.10, below the 0.10
  the glider gate asks across ±20 s. So a peak that falls by less is trusted
  when (a) its curvature, read from the contiguous width of the peak, implies
  a fall of at least 0.10 across ±20 s, and (b) the peak located again on two
  interleaved halves of the profiles lands within two lag steps of each
  other. The peak must also be one the grid measures: at least two steps
  wide, the only top within 1% of the fall anywhere in the search (two lobes
  are an ambiguity), and no wider than a quarter of the search (a flat top).
  The record's `lag_resolution` carries the evidence and, when refused, the
  `problem`. No one test suffices: a glider's mean-removed ramp replicates on
  both halves (fall 5e-4), a one-step bump on the 1/64 s grid passes the
  curvature bound, and a flat top or two lobes pass both unless the width is
  the peak's own. On eight VMP deployments (SUNRISE
  2019-2022, ARCTERX-2022 and 2025) the falls are 2.3-49 and the halves agree
  to 0.1-8 ms, while the old rule refused every one of them. A search of
  ±20 s or more is judged on its fall alone, as before.
- **The geometry term needs profiles spanning ~30 dbar.** Shorter descents
  have no reference gradient (on SUNRISE 2021 no descent's pairs span more
  than 28.3 dbar). `fit.geometry: true` then fits without the term and says
  so in the report's Notes and the record's `geometry_skipped`, and a
  `fit.prior` block is consulted as usual.
- **Reference glitches.** A JAC_T spike is a rejected outlier, reported but no
  longer refusing the fit (§3). An instrument-wide transient is more than
  that: on ARCTERX-2025 Interior, JAC_T jumped to 5 °C and P by 3 dbar within
  0.1 s, with both FP07 counts ramping together. Left in, its pairs are
  rejected outliers: they refuse nothing, set neither the fitted range nor
  the order folds, and the curve (order 2, as blanked) stays within 1.7 mK of
  the one fitted with the event blanked. But the event is in both series the
  lag is scored on, so it dominates the high-passed score (r 0.99997 →
  0.986) and moves the lag from +0.105 to +0.131 s; the two halves then
  disagree by 52 ms and the lag is refused, so `patch` refuses the record.
  Blank the reference around such an event before fitting.

## Gotchas worth knowing

- **A non-finite thermistor count costs only the pairs near it.** The single
  pole that slows the thermistor to the reference restarts after a gap of
  non-finite counts (a channel shorter than the slow grid, say), with the
  first five time constants after the restart blanked while it settles. Every
  pair whose kernel the gap or that blank touches at all is dropped, so no
  pair is averaged over a shorter window than the rest. Before issue #238 one
  NaN turned the rest of the file to NaN. The report's Notes and the record's
  `thermistor_gaps` say what was lost.
- **Bad-buffer thermistor counts are blanked, not paired** (issue #248).
  PFile reports RDL bad-buffer runs and the record slots a framing repair
  rebuilt by interpolation, and leaves the values in place; `fp07-cal` sets
  those thermistor counts to NaN, so the gap path above drops every pair they
  touch. The mask is chi's per-thermistor one (`T1` and `T1_dT1`, widened over
  the 3 `diff_gain` the deconvolution smears a dropout forward), without the
  pressure channels, which do not enter a pair's value. The report's Notes,
  the record's `bad_buffer_samples` and the streaming pass's `per_profile`
  count them. On ARCTERX-2022 Wake, whose ARC1A046 has one rebuilt record,
  250 and 244 counts were blanked and the fit is the same to round-off: the
  slot lies before the first descent. A rebuilt slot stays local because
  PFile interpolates it BEFORE deconvolving; an RDL marker run (the -32753
  sentinel, left in place) on `T1` or `T1_dT1` instead moves the
  deconvolution's two whole-file regressions (issue #259): 64 sentinels
  injected into SN479_0006 moved `T1` by -8..+57 mK (in `T1_dT1`) and
  -234..-64 mK (in `T1`) more than 20 s away, linearly in T. So that channel
  is left out of that file whole (`bad_buffer_dropped`; the record's
  `n_files_dropped`/`files_dropped`). None of 2248 glider and ARCTERX-2022
  Wake files has such a run (osu685 2023 has U_EM runs in 22). A v6.0 file
  flags bad records in header word 16 instead, which PFile only warns about
  and does not report: those records are still paired.
- **The T1 − T2 check uses this run's fitted coefficients**, one set per
  channel. Each file's embedded set was used before, and where files had been
  patched one by one those differ by several K at fixed L, so `T1 − T2`
  stepped at every change: on ARCTERX-2022 Interior (13 sets in 30 files) by
  up to 2.5 K between files, a per-profile scatter of 1.09 K and a slope of
  +0.32 K/day; with the fitted sets, 2.5 mK, 0.9 mK and +1.3×10⁻⁴ K/day.
  With identical nominal coefficients on both beads (osu684, osu685) the old
  check also read the change in water temperature as `T1 − T2` drift; with
  fitted ones its slope is 5× (osu685 2025) and 6.5× (osu684 2023) smaller.
  Both channels' coefficients come from one source, named in the report
  line: this run's fit when it fitted both, else the files' embedded sets
  when each is one set across the files, else the check is unavailable with
  the reason. A fitted T1 is never compared with an embedded T2.
- **The factory coefficients may be nominal.** On osu685 both probes carried
  identical values despite being physically different beads, so the in-situ fit
  is the *only* calibration they have — worth 2.7 K.
- **`beta` shadows `beta_1`.** `convert_therm` checks the legacy `beta` first,
  so a file carrying it needs that key patched. The tool detects this and
  writes to whichever key the reader will actually use.
- **Never set `beta_2 = 0` to remove a quadratic term.** The config value is a
  *reciprocal*, so zero means an infinitely large term and the reader refuses
  the channel with a `ValueError` naming the coefficient (before issue #180 it
  was a bare `ZeroDivisionError`). The tool writes `1e30`, which is
  bit-identical to omitting the key.
- **Record some dives if you can.** Climb-only costs you the dive/climb residual
  split, leaves the sensor-geometry decomposition unresolved, and confounds
  depth with elapsed-time-since-file-start.
- **Don't route the reference through perturb's hotel merge for this.** Its
  gap handling is a configuration matter (`hotel.max_gap` / `extrapolate`,
  PR #150), and a calibration must not depend on how a downstream merge is
  configured. `fp07-cal` reads the CTD on its own clock — real samples only —
  for exactly this reason. Any NetCDF carrying the CTD and its own timestamps
  will do — a hotel file, or a converted `ebd.nc`.
