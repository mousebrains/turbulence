# `fp07-cal` runbook

FP07 in-situ calibration for a MicroRider carried by a Slocum, against the
glider's own SBE41cp. It is a **pre-pipeline** step: it runs once per
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

## 5. Patch

```bash
fp07-cal patch -c fp07-cal.yaml --dry-run    # inspect the edits first
fp07-cal patch -c fp07-cal.yaml
```

Writes corrected `.p` copies with the original configuration retained
commented-out and a provenance banner. It **refuses** on: an instrument serial
mismatch, any bridge-parameter mismatch, a lag that failed the sharpness gate,
or an input that is already patched (a second pass would nest banners and
destroy the original-config block — always patch from the originals).

Then point perturb at the patched files with:

```yaml
fp07:
  calibrate: false      # the files already carry the in-situ coefficients
```

## Gotchas worth knowing

- **The factory coefficients may be nominal.** On osu685 both probes carried
  identical values despite being physically different beads, so the in-situ fit
  is the *only* calibration they have — worth 2.7 K.
- **`beta` shadows `beta_1`.** `convert_therm` checks the legacy `beta` first,
  so a file carrying it needs that key patched. The tool detects this and
  writes to whichever key the reader will actually use.
- **Never set `beta_2 = 0` to remove a quadratic term.** The config value is a
  *reciprocal*, so zero means an infinitely large term and the reader raises
  `ZeroDivisionError`. The tool writes `1e30`, which is bit-identical to
  omitting the key.
- **Record some dives if you can.** Climb-only costs you the dive/climb residual
  split, leaves the sensor-geometry decomposition unresolved, and confounds
  depth with elapsed-time-since-file-start.
- **Don't route the reference through perturb's hotel merge for this.** It
  interpolates across arbitrary gaps and edge-holds outside coverage, so on a
  sparsely-sampled CT it would hand the fit a fabricated ramp. `fp07-cal` reads
  the CTD on its own clock for exactly this reason. Any NetCDF carrying the CTD
  and its own timestamps will do — a hotel file, or a converted `ebd.nc`.
