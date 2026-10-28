# `fp07-cal` runbook

FP07 in-situ calibration for a MicroRider carried by a Slocum, against the
glider's own glider CTD. It is a **pre-pipeline** step: it runs once per
deployment and hands perturb `.p` files that already carry the right
coefficients, so perturb itself needs no changes.

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
  never on r. If the lag is not trustworthy, the coefficients are not either,
  and `patch` will refuse them.
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
cannot supply. It adds the entry to the YAML as text, leaving every existing
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

## Gotchas worth knowing

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
