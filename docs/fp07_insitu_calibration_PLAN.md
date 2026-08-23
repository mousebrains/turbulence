# FP07 in-situ calibration — extracting reusable coefficients

**Status:** plan, not yet implemented.
**Branch/worktree:** `worktree-fp07-insitu-cal` @ `.claude/worktrees/fp07-insitu-cal`
**Date:** 2026-08-22

---

## 0. The problem

A user has a MicroRider on a Slocum glider. The FP07 thermistors are to be
calibrated in situ against the glider's SBE41cp, but **the CT was only sampled
every *n*th yo**. We want to come out the far side with *coefficients* — numbers
that can replace or amend the FP07 Steinhart–Hart coefficients in the `.p`
config — rather than with a one-shot in-memory temperature array.

Two things follow immediately, and they set the whole shape of this plan:

1. **The fit and the application must be separated, and both belong *before*
   the pipeline.** The reference exists on a small subset of yos; the
   calibration must be applied to *all* of them. A per-file, in-pipeline fit
   (what exists today) cannot do this. The calibration is deployment-scoped, is
   run once, and hands perturb `.p` files that are already correct.
2. **How far a coefficient set travels is an open empirical question.** It is
   plausibly specific to the instrument, the channel (port/bridge), and the
   individual probe. The deliverable therefore is not "the coefficients" but a
   **keyed, provenance-bearing coefficient record that declares its own domain
   of validity and refuses to be applied outside it.** Establishing how far it
   actually generalizes is Phase 5 — a measurement, not an assumption.

---

## 1. What exists today (verified)

| Piece | Location | What it does |
|---|---|---|
| In-situ fit | `src/odas_tpw/perturb/fp07_cal.py` | `fp07_calibrate(pf, profiles, reference, order, …)`. Per-`.p`-file. Computes `L = ln(R_T/R_0)`, lag-aligns a reference channel, least-squares fits `1/T_K = Σ aᵢLⁱ`, returns calibrated arrays **and** coefficients. |
| Call site | `perturb/pipeline.py:2142` | Inside `process_file`. Consumes only `cal_result["channels"]` and `["fast_channels"]`. |
| Reader | `rsi/channels.py:111` `convert_therm` | `1/T_K = 1/t_0 + (1/beta_1)·L + (1/beta_2)·L² + (1/beta_3)·L³` |
| Config rewriter | `rsi/config_patch.py` | Rewrites config values in record 0 of a `.p` file into a **new** file, with a provenance banner and the original config retained commented-out. Supports adding keys (`add_keys`). **No delete.** |
| Reference plumbing | `perturb/hotel.py`, `odas_tpw/dinkum/` | Slocum `dbd`/`ebd` → `hotel.nc` → merged onto the `.p` time grid as ordinary channels. |

### 1.1 The coefficient mapping is exact

`fp07_cal._compute_RT_R0` and `channels.convert_therm` compute **the same** `L`
(same bridge algebra, same `np.clip(Z, ±0.6)`). So:

```
a0 = 1/t_0     a1 = 1/beta_1     a2 = 1/beta_2     a3 = 1/beta_3
```

is an exact, lossless round trip. This is the single most important enabling
fact in the plan: we do not need a new calibration *form*, only a better
*estimate* and a path to disk.

### 1.2 Three things block the goal today

- **The coefficients are thrown away.** `pipeline.py:2160-2175` reads only the
  calibrated arrays. Nothing persists `cal_result["coefficients"]`.
- **The scope is one `.p` file.** Files are processed in a
  `ProcessPoolExecutor`; workers cannot pool a fit across the deployment.
- **The reference the fit sees is largely fabricated.** See §2.1. This is the
  one that would silently produce plausible, wrong numbers.

---

## 2. Adversarial findings against the existing code

These were found by reading the code against *this* use case. They are ordered
by how badly they would corrupt the answer. Each is a requirement on the new
work, not a digression.

### A1 — The hotel merge invents the reference. **Fatal.**

`hotel.py::_interp_one` (lines 437–462) drops non-finite samples and then
interpolates straight across the hole; outside coverage it edge-holds
(`fill_value=(data[0], data[-1])`). There is **no `max_gap`** on the perturb
side.

For CT-every-*n*th-yo that means: on the (n−1)/n of the record with no CT,
`pf.channels["sci_water_temp"]` is a smooth, entirely fictitious ramp between
the last real sample and the next one — hours away. `fp07_calibrate` cannot
tell the difference and will happily regress against it.

Worse, `dinkum-hotel`'s `projection.max_gap` (which *would* NaN those spans)
is **defeated** by this loader: the builder writes NaN, the loader drops the
NaN and interpolates across it anyway. Setting `max_gap` in the builder gives
false comfort.

> **Requirement R1.** The calibration pass reads the reference from the CTD
> source on the CTD's own clock. It never consumes an interpolated reference,
> and it never reads `pf.channels[reference]`.

> **Note (out of scope, must be reported).** The same fabricated reference is
> being consumed by `ct`, `ctd`, `stratification`, `salinity: "measured"` and
> `epsilon.T_source` in the example config. That is a separate bug with a wider
> blast radius. This plan will file it; it will not fix it.

### A2 — No bandwidth matching against a 1 Hz CTD

`_lowpass_filter` (fp07_cal.py:83–110) branches on
`reference.upper().startswith("JAC")`. For `sci_water_temp` it takes the else
branch: `fc = fs_slow/3 ≈ 21 Hz`. That is *no* filtering relative to an SBE41cp
with τ ≈ 0.5 s sampled at 1 Hz.

The regressor `L` then carries ~20 Hz of bandwidth the reference does not have.
This is textbook errors-in-variables: the fitted slope is attenuated toward
zero by roughly σ²_signal/(σ²_signal + σ²_noise), which lands directly on
`beta_1`.

> **Requirement R2.** Bandwidth-match by construction — see D2.

### A3 — Numerical conditioning

The fit builds a raw Vandermonde in `L` (`np.column_stack([RT_R0**i …])`) and
calls `lstsq` with no centering. Over a glider deployment's ~3 °C range, `L`
occupies a very short interval far from zero; the design matrix is badly
conditioned and `a₂`, `a₃` become numerically meaningless long before they
become statistically meaningless.

> **Requirement R3.** Fit in a centered/scaled variable, then transform the
> polynomial back exactly. Report the condition number.

### A4 — `_compute_RT_R0` and `convert_therm` disagree on defaults

| key | `fp07_cal` default | `convert_therm` default |
|---|---|---|
| `adc_fs` | 5 | 4.096 |
| `g` | 1 | 6.0 |
| `e_b` | **0** | 0.68 |

If any of these is absent from the config the fit's `L` differs from the
reader's `L`, and the emitted coefficients are silently wrong. `e_b = 0` is
especially nasty: `G·E_B == 0` short-circuits the scaling entirely.

> **Requirement R4.** One shared `log_R` implementation, missing bridge keys a
> hard error in the calibration path, and the bridge parameters recorded in the
> coefficient record (§4) so a mismatch at apply time is detectable.

### A5 — `beta` shadows `beta_1`

`convert_therm:132-137` checks `beta` **first**. If the factory config carries
the legacy `beta`, patching `beta_1` is a **silent no-op** — the file looks
recalibrated and is not.

> **Requirement R5.** The emitter detects which key is live and patches that
> one, or refuses.

### A6 — Order downgrade leaves stale higher terms

`convert_therm` applies `beta_2`/`beta_3` whenever present. Emitting an order-1
fit into a config that already carries `beta_2` leaves the quadratic term
active against new linear coefficients. `config_patch` can add keys but
**cannot delete** them.

> **Requirement R6.** Emit the *complete* set at the fitted order. For a
> downgrade, either add a delete capability to `config_patch` or neutralize the
> term (`1/beta_k → 0` needs `beta_k → ∞`, which is not expressible — so
> deletion is the honest fix). Decide in Phase 3; flag as an open question.

### A7 — Silent `Z` clipping

`np.clip(Z, ±0.6)` bounds the bridge output. Clipped samples have a wrong `L`
and are indistinguishable from good ones downstream.

> **Requirement R7.** Count clipped samples; exclude them from the fit; fail
> loudly above a threshold.

### A8 — Probe serial numbers are often placeholders

`sensor_inventory.py:26-27` records that FP07 configs frequently use a shared
placeholder (`sn = T` on both T1 and T2) or leave it blank.

**This lands squarely on the generalizability question.** A coefficient store
keyed on probe SN would happily apply T1's calibration to T2. The record must
carry a `probe_sn_trusted` flag and, when untrusted, key on
`(instrument_sn, channel_name, time_range)` instead — never on probe SN alone.

### A9 — Two different lags are being conflated

The measured FP07-vs-CTD lag is the sum of

- the **MR-vs-glider clock offset** (constant-ish, possibly drifting, sign
  arbitrary), and
- the **SBE41cp response + plumbing transit** (speed-dependent, always
  positive: the CTD lags the water).

`fp07_calibrate` estimates one number for both. And with a 1 Hz reference,
sub-sample lag precision from a 64 Hz cross-correlation is illusory.

> **Requirement R8.** Estimate the clock offset **independently** from
> `sci_water_pressure` (CTD clock) against the MR's native `P` — a pure timing
> measurement with no thermal physics in it. Then estimate the residual thermal
> lag from T-vs-T. Report both. The example config already flags the P
> difference as "worth watching"; this makes it load-bearing.

### A10 — `must_be_negative` and `shift_edge_hold`

`must_be_negative=True` (the VMP free-fall assumption) is wrong here; the
example config already sets `False`. And `shift_edge_hold` applied to a
reference that is mostly edge-held filler is meaningless. Both are symptoms of
a VMP-shaped design; the new estimator should not inherit either.

### A12 — No correlation-quality gate on the lag. **New.**

`_calc_lag` returns `(lag, corr)` and `fp07_calibrate` collects `corr` into
`info["median_corr"]` — but **nothing ever thresholds it**. Measured against a
fabricated mid-gap ramp (§3.1):

```
lag = -7.05 s,  corr = 0.021
```

A correlation of 0.02 is pure noise, and it is accepted as a confident lag and
folded into the median. The one guard that exists (`norm <= 0` → NaN) only
catches the *perfectly constant* reference.

> **Requirement R9.** A minimum correlation, a maximum inter-profile lag
> scatter, and a minimum contributing-profile count. Below any of them, the
> channel abstains rather than emitting coefficients.

### A11 — No test data in the repo

`VMP/` is absent from the current checkout. There is no glider `.p` file, no
`hotel.nc`, no `.ebd`. **A synthetic generator is not optional** — it is the
only way to write a test that proves coefficient recovery.

---

## 3. `.p` files with no hotel CT

This is not an edge case — with CT on every *n*th yo it is the **majority** of
the files. It deserves its own section.

### 3.1 What happens today — three regimes, measured

The reference channel essentially always *exists* after the hotel merge
(`_interp_one` returns a full array whenever the hotel file has ≥2 finite
samples anywhere), so `fp07_calibrate`'s "reference not found" path never
fires. What varies is what that array *contains*. Measured against the real
code:

| File's position vs CT coverage | Merged reference | `_calc_lag` | Result |
|---|---|---|---|
| Entirely outside coverage | constant (edge-hold, e.g. `22.59`) | `(nan, nan)` — guard fires | **No calibration.** Factory coefficients, silently. |
| Inside a CT gap | fabricated linear ramp (`19.16 → 19.40` across a 1 h hole) | `(-7.05 s, corr 0.021)` — **guard does not fire** | **Calibrated against fiction.** |
| Real CT coverage | genuine | genuine | Genuine fit. |

So a single deployment today would come out the far side with **three different
calibration regimes distributed across adjacent yos**, and the only trace is a
`channel_info["calibration"]` tag that nothing surfaces. That is a worse outcome
than never calibrating at all, because the discontinuities are of order the
correction itself.

The gap-ramp row is A1 and A12 acting together: A1 invents the signal, A12
declines to notice that nothing correlates with it.

> Also confirmed empirically: marking the gap with NaN in the hotel file — what
> `dinkum-hotel`'s `projection.max_gap` does — produces **byte-identical**
> output from `_interp_one`. The builder-side gap control is entirely defeated
> by the loader.

### 3.2 What happens under this plan

A file with no CT is a **non-event**:

1. `fp07-cal pairs` emits zero pairs for it. Expected, not an error.
2. `fp07-cal fit` pools pairs from the covered yos only — one coefficient set
   for the deployment.
3. `fp07-cal patch` writes that set into **every** `.p` file, covered or not.

The uncovered yos get the calibration precisely because the coefficients
describe *the probe*, not *the yo*. This is the payoff for separating fit from
apply, and for the patch-the-file sink: there is no code path in which some
files are calibrated and others are not.

### 3.3 The sub-cases that do need handling

**(a) File outside the fit's *time* range.** Not merely "no CT this yo" but
outside `validity.time_start … time_end` — the science computer was off, or the
`.ebd` set is incomplete. Patching it extrapolates in time.
*Decision: patch anyway, warn loudly, and record the extrapolation in the
provenance banner.* Leaving it on factory coefficients would re-create exactly
the discontinuity §3.1 condemns. Consistency beats a false show of caution, as
long as it is visible.

**(b) File outside the fit's *temperature* range.** More insidious. The
polynomial is constrained only over the T range the CT-covered yos happened to
sample; an uncovered yo that went deeper and colder is extrapolated in the state
variable.

> **This is an independent and stronger argument for order 1** than the
> conditioning argument in A3. A straight line in `L` extrapolates gracefully
> and stays physically sensible (it *is* the two-parameter Steinhart–Hart
> model, which is a good description of a glass-bead thermistor over ~10 °C).
> An order-2 fit over a 3 °C range, extrapolated 2 °C beyond it, can go
> anywhere. Combined with A3 and D5.5, order 1 is the default and order ≥2
> requires the operator to have looked at the coverage report and chosen it.

Report per-file how far outside the fitted range it went; NaN or flag beyond a
configurable margin.

**(c) Zero pairs across the whole deployment.** CT never on, or every pair
gated out. **Hard error** with the rejection-reason histogram — never an empty
coefficient file that a later step treats as "no correction needed".

**(d) Too few pairs, or too narrow a T range.** Gated by R9. Note that the
binding constraint here is **systematic, not random**: with an SBE41cp
reference, a few thousand pairs over 3 °C pins the slope far tighter than the
dive/climb asymmetry, residual thermal lag, and MR-vs-CTD mounting separation
do. The report must therefore present a **systematic error budget**, not a
formal standard error — a tight confidence interval on a biased fit is the most
misleading thing this tool could print.

### 3.4 The coverage manifest

`fp07-cal coverage` (Phase 0) and `patch` both emit a per-file table — the
operator's direct answer to "what happened to my files with no CT?":

| file | n_pairs | CT coverage | in time range | T range vs fit | action |
|---|---|---|---|---|---|
| `…_0007.p` | 412 | 96% | yes | inside | patched (contributed) |
| `…_0008.p` | 0 | 0% | yes | inside | patched (extrapolated in T? no) |
| `…_0031.p` | 0 | 0% | **no** | −1.8 °C below | **patched, WARNED ×2** |

---

## 4. Design

### D1 — A pre-pipeline step, not a pipeline stage

The calibration runs to completion **before perturb is invoked at all**, and
hands perturb `.p` files that already carry the corrected coefficients. Perturb
then needs no knowledge of any of this and no code change.

`perturb trim` is the precedent: it is already a standalone subcommand
(`perturb/cli.py:533`) that reads `.p` files and writes corrected `.p` files
for the pipeline to consume. `fp07-cal` is the same shape.

```
   raw *.p ──┐
             ├──► perturb trim ──► trimmed *.p ──┬──► fp07-cal fit ──► coefficients.yaml
   *.ebd ────┴──► dinkum-hotel ──► hotel.nc  ────┘         │              + report
                                                            ▼
                                      trimmed *.p ──► fp07-cal patch ──► calibrated *.p
                                                                              │
                                                                              ▼
                                                        perturb run  (fp07.calibrate: false)
```

New top-level tool, run **once per deployment per channel**:

```
fp07-cal coverage -c fp07-cal.yaml     # what reference do we actually have?
fp07-cal pairs    -c fp07-cal.yaml     # build (L, T_ref) pairs -> pairs.nc
fp07-cal fit      -c fp07-cal.yaml     # pairs.nc -> coefficients.yaml + report
fp07-cal patch    -c fp07-cal.yaml     # coefficients.yaml -> calibrated *.p
```

Splitting `pairs` from `fit` matters: pair-building is the expensive I/O pass
over every `.p` file, and the fit is the part we will iterate on (order,
outlier rules, segmentation). One slow pass, many fast fits.

**Ordering constraint.** `fp07-cal` needs profile detection, so it wants
already-trimmed files; and `config_patch` copies data records byte-for-byte, so
patching last is safe. Hence **trim → fit → patch → run**, with the pipeline
config then set to `files.trim: false` (the files are already trimmed) and
`fp07.calibrate: false` (they are already calibrated). Running trim *after*
patch would need trim to preserve the patched config string — untested, and
avoidable by ordering.

### D2 — Regress on the CTD's own sample times (the key inversion)

Instead of interpolating the 1 Hz reference **up** to 64 Hz, decimate the FP07
**down** onto real CTD samples.

For each valid CTD sample at time `t_k`, form

```
L_k = <L(t)>  averaged over the CTD's effective sampling kernel, centred at t_k − lag
```

This single change resolves three findings at once:

- **A1** — no reference value is ever invented. Sparsity is handled by
  construction: no CT sample, no pair.
- **A2** — bandwidth matching is automatic; the FP07 is averaged over exactly
  the interval the CTD integrated.
- honest degrees of freedom — `N` is the number of *real* CTD samples, so the
  reported uncertainty is not inflated by a 64× interpolation.

The kernel needs a defensible shape. Start with a boxcar of width = median CTD
sample interval, convolved with a single-pole τ for the SBE41cp thermistor;
make both configurable and test sensitivity to them (see V4).

### D3 — Pair selection gates

A CTD sample contributes a pair only if all hold:

| Gate | Rationale |
|---|---|
| Inside a detected profile (dive or climb) | Excludes surface/apogee loiter |
| Speed ≥ threshold | FP07 flushing fails at low flow |
| CTD sample spacing to both neighbours ≤ `max_gap` | The kernel must not span a dropout |
| No clipped `Z` in the kernel window (A7) | |
| `|dT/dt|` above a floor, *or* explicitly kept | Isothermal water constrains the slope not at all but does constrain the offset — keep it, but track the two populations separately |
| FP07 and CTD finite over the whole kernel | |

Every rejection is **counted by reason** and reported. A calibration that
silently kept 3% of its data is worse than one that refuses to run.

### D4 — The fit

- Fit `1/T_K = Σ aᵢ·((L − L̄)/s)ⁱ`, transform back exactly (R3).
- **Order chosen by data, not by config default.** Gate on observed T range
  (the existing 8 °C heuristic is a reasonable floor for order ≥ 2) *and* on
  cross-validated residual. Default order 1 for a glider deployment.
- **Report the errors-in-variables bracket:** OLS of `y|x` and of `x|y` bracket
  the true slope. If the bracket is wide, the coefficients are not trustworthy
  and the report must say so rather than emitting a point estimate with a
  confident face. Consider Deming regression once the noise ratio is known.
- **Robustness:** iterate with an outlier rejection (e.g. drop |residual| > 4σ,
  refit, ≤3 passes). Report how many were dropped.

### D5 — Residual diagnostics (the real QC)

The fit will always return numbers. These decide whether to believe them:

1. **Dive vs climb residual split.** The single best diagnostic. Correct lag and
   correct bandwidth match ⇒ no dive/climb asymmetry. Any systematic split is
   an unremoved lag or a thermal-mass artifact and **must not be absorbed into
   the coefficients**.
2. **Residual vs pressure.** Depth-dependent bias — MR and CTD are at different
   points on the vehicle and sample different water.
3. **Residual vs time.** Drift over the deployment. Reported as a rate.
4. **T1 − T2 over the *whole* deployment.** Needs no reference at all, so it
   covers the yos with no CT. **This is what licenses extrapolating a
   CT-subset calibration onto the uncovered yos.** If T1−T2 is stable
   everywhere, a static fit is defensible. If it wanders on the uncovered yos,
   it is not, and the report must say so.
5. **Selection bias check.** "Every *n*th yo" may not be random — if CT was
   enabled on a schedule correlated with time of day or water mass, the fit's T
   range is biased. Compare the FP07-only T distribution on covered vs
   uncovered yos.

### D6 — Deliverable: a keyed coefficient record

This is the direct answer to "I don't know how generalizable these are."
The record does not claim generality; it **states its domain and enforces it**.

```yaml
schema: fp07-cal/1
instrument:
  sn: "MR2041"              # from [instrument_info]
  vehicle: "slocum_glider"
channel: "T1"               # bridge/port identity
probe:
  sn: "T"                   # as read from the config
  sn_trusted: false         # placeholder detected -- see A8
bridge:                     # the L-definition this fit is bound to (R4)
  a: ...   b: ...   g: ...   e_b: ...   adc_fs: ...   adc_bits: ...
validity:
  time_start: "2025-03-01T00:00:00Z"
  time_end:   "2025-03-28T00:00:00Z"
  T_min: 12.41              # calibrated range -- do NOT extrapolate
  T_max: 21.83
  P_max: 195.0
reference:
  source: "hotel.nc:sci_water_temp"
  instrument: "SBE41cp"
  n_pairs: 18432
  n_profiles: 47
  n_profiles_total: 412     # <- the every-nth-yo ratio, stated plainly
fit:
  form: "inv_T_kelvin_vs_lnR"
  order: 1
  coefficients: [a0, a1]
  config_equivalent: {t_0: 288.94, beta_1: 3123.7}
  live_beta_key: "beta_1"   # or "beta" -- see A5
  clock_offset_s: -1.83     # from P-vs-P (R8)
  thermal_lag_s: 0.42       # residual, from T-vs-T
  rms_residual_K: 0.0031
  slope_bracket: [3118.2, 3129.4]   # errors-in-variables (D4)
  condition_number: 4.1e3
diagnostics: {dive_climb_split_K: 0.0004, drift_K_per_day: 1.2e-4, ...}
```

**Apply-time refusal.** Applying a record is checked against: instrument SN,
channel name, every bridge parameter, time range, and observed T range. Any
mismatch is an error unless explicitly forced. A record whose
`probe_sn_trusted` is false may never be applied to a different channel.

### D7 — The sink: patched `.p` files

`fp07-cal patch` drives `rsi/config_patch.py` to write a new `.p` file per
input with the corrected `t_0`/`beta_*`, the original config retained
commented-out, and a provenance banner naming the coefficient record.

Why this is the right sink for a pre-pipeline step:

- **Perturb needs no code change.** Set `fp07.calibrate: false` and the
  factory-vs-in-situ distinction disappears into the file, where it belongs.
- **Every reader benefits**, including ODAS MATLAB and `rsi-tpw` — not just
  perturb.
- **Provenance travels with the data.** `config_patch` already embeds the
  original config and a banner; the patched file is self-describing forever.
- **It works identically on yos with and without CT**, which is the whole
  point: the coefficients are a property of the probe, not of whether that
  particular yo happened to have the CT enabled.

Costs and constraints, stated plainly:

- **Disk.** One full copy of the `.p` set. `perturb trim` already establishes
  that this is acceptable in this pipeline, but for a long glider deployment it
  is worth checking before committing.
- **Re-fitting means re-patching.** Patch from the *trimmed originals* every
  time, never patch a patched file — `config_patch` would then bury one banner
  inside another and the "original config" block would no longer be original.
  The tool must detect an already-patched file and refuse (the banner marker is
  already parsed — `_find_self_original_marker`, config_patch.py:319).

**Deliberately out of scope:** an in-pipeline `fp07.mode: coefficients` that
reads the record at run time. It would duplicate the apply logic in a second
place, and the user's framing — pre-pipeline — makes it unnecessary. Revisit
only if the disk cost above turns out to be prohibitive.

---

## 5. Phases

| # | Deliverable | Notes |
|---|---|---|
| 0 | `fp07-cal coverage` + fix A4 | Read-only. Tells us how many yos actually have CT, the gap structure, T range, dive/climb balance, clipping rate. **Do this before writing the estimator** — it may change the design. A4 is a standalone safety fix. |
| 1 | Synthetic generator + `fp07-cal pairs` | Generator emits a `.p` + `hotel.nc` from known coefficients, known lag, known noise, with configurable reference sparsity. |
| 2 | `fp07-cal fit` + diagnostics + **stability report** | D4, D5, D8 (§8). Cheap given the pairs/fit split; answers the drift question from this deployment's own data. |
| 3 | Coefficient record + `fp07-cal patch` | **DONE.** D6, D7, A5, A6. The delete-key question resolved without touching `config_patch`: `beta_k = 0` raises `ZeroDivisionError` (the value is a reciprocal), while `beta_k = 1e30` is bit-identical to omitting the key. |
| 4 | Worked example + docs | **DONE.** `examples/slocum_glider_hotel/fp07-cal.yaml`, `docs/fp07cal/runbook.md`, a pointer from the example `perturb.yaml`, and a CLAUDE.md section. No perturb code change — V11 confirms it. |
| 5 | **Transferability study** | The empirical answer to the generalizability question. See §6. |

Phases 0–2 are the load-bearing ones; 3 is plumbing onto machinery that already
exists (`config_patch`), and 4 is documentation.

---

## 6. Validation

| # | Test |
|---|---|
| V1 | **Round trip, dense.** Synthesize from known `t_0`/`beta_1`(/`beta_2`), recover within tolerance. |
| V2 | **Round trip, sparse.** Decimate the reference to every *n*th yo. Recovery must match V1; the number of pairs must equal the number of real CTD samples. |
| V3 | **Fabrication guard.** A reference with a multi-hour gap must produce *zero* pairs in that span. Regression test against A1. |
| V4 | **Kernel sensitivity.** Vary the CTD kernel width/τ over a plausible range; the coefficient shift must be small relative to the reported uncertainty, or the report must widen. |
| V5 | **Config round trip.** Patch a `.p` with the emitted coefficients, re-read with `PFile`, assert `pf.channels["T1"]` equals the fit's calibrated array to floating-point tolerance. Proves the a↔beta mapping and the live-key logic (A5). |
| V6 | **Order-downgrade guard.** Emitting order 1 into a config carrying `beta_2` must not silently leave the quadratic active (A6). |
| V7 | **Lag recovery.** Known injected clock offset recovered from P-vs-P independently of an injected thermal lag (R8). |
| V8 | **VMP non-regression.** The existing dense-`JAC_T` path must be unchanged. Golden-compare against ODAS `cal_FP07_in_situ.m` if VMP data is restored. |
| V9 | **Refusal.** Applying a record with a mismatched bridge parameter / SN / out-of-range T must raise, not warn. |
| V10 | **No double-patch.** Running `fp07-cal patch` on an already-patched `.p` must refuse (D7). |
| V12 | **No-CT files are still patched.** A synthetic set where only every 3rd file has CT: all files must come out patched with identical coefficients, and the no-CT files must contribute zero pairs (§3.2). |
| V13 | **Gap-ramp rejection.** The measured §3.1 failure — a fabricated mid-gap ramp yielding `lag -7.05 s, corr 0.021` — must produce zero pairs and trip R9, not a fit. |
| V14 | **Drift recovery.** Inject a known `t_0` ramp with `beta_1` fixed; the blocked estimator (D8) must recover the rate, and must report "no significant drift" on a drift-free control. |
| V11 | **Pipeline is untouched.** ✅ **Verified on osu685.** `perturb profiles` over patched files with `fp07.calibrate: false` produced profile products shifted by −2.658 K (T1) / −1.617 K (T2) against the same run on the originals — matching the reader-level −2.659 / −1.620 exactly, with zero changes to `perturb/`. Provenance travels too: the profile NetCDF's `configuration_string` global attribute carries the full patched config including the `; PATCHED by fp07-cal` banner and the original values commented out. |

---

## 7. Generalizability across probes and instruments (Phase 5)

We should not guess. The record's keying (D6) makes the question *testable*,
because every fit is labelled with what it was derived from. Once there are two
or more fits in hand:

- **Same probe, same channel, different deployments** → is the drift within the
  reported residual? Answers "does a calibration survive a redeployment?"
- **Same probe moved between channels** (T1 ↔ T2) → separates *probe* from
  *bridge*. This is the decisive experiment, and `sensor_inventory.py` already
  tracks probes across ports — but only where the SN is real (A8).
- **Different probes, same channel** → how much of the correction is the port
  and how much is the glass bead?
- **Factory vs in-situ delta** → if the in-situ correction is consistently a
  small offset in `t_0` with `beta_1` essentially unchanged, that is a very
  different (and much more transferable) story than both moving.

Until that is measured, the operating assumption is the conservative one:
**a coefficient set is valid for one instrument, one channel, one probe, one
deployment, over the T range it was fitted on.** The tooling enforces exactly
that, so a future relaxation is a deliberate config change rather than a silent
drift in behavior.

---

## 8. Temporal stability of the coefficients

Section 7 asks how far a coefficient set travels across *probes*. This one asks
how far it travels across *time* — and it is the question the sparse-CT dataset
is unusually well suited to answer, because CT-on-every-*n*th-yo gives a
**time series of independent calibration opportunities spread across the whole
deployment**. What looked like the central difficulty is also the opportunity.

### 8.1 The two coefficients are not equally stable, and that is the design lever

They have different physical origins:

| | origin | expected behavior |
|---|---|---|
| `beta_1` | the bead's material B-value | intrinsic to the glass; expected **stable** |
| `t_0` | probe `R_0` × bridge (`a`, `b`, `G`, `E_B`) electronics offset | expected to **drift**: aging, bridge offset, biofouling |

They also have different *estimation* requirements, and the two facts point in
opposite directions:

- `beta_1` is a **slope** — it needs the widest possible temperature range, so
  it wants **all the pairs, pooled over the whole deployment**.
- `t_0` is an **offset** — it needs almost no temperature range at all, so it
  can be estimated from **one block of yos at a time**.

> **D8 — the stability estimator.** Fit `beta_1` **globally**, then fit `t_0`
> **per temporal block** with `beta_1` held fixed. This separates the parameter
> that needs range from the parameter that needs time resolution, and it is
> strictly better than the naive "refit everything per block", where each
> block's narrow T range makes `beta_1` wander wildly and drags `t_0` with it
> through their strong covariance.

This is nearly free: `pairs` and `fit` are already split (D1) so that one slow
I/O pass feeds many fast fits. Blocked refitting is a loop over the fits.

### 8.2 Per-file `t_0` is free — because we patch per file

If `t_0` drift turns out to be real and resolvable, the pre-pipeline patch sink
can express it at **no additional cost**: `fp07-cal patch` already writes one
file at a time, so each `.p` file can receive `t_0` evaluated at its own
midpoint time, with a single global `beta_1` throughout.

A slowly-drifting calibration therefore costs nothing structurally. This is a
second, unplanned payoff from the patch-the-file sink — it is not expressible
at all in a single static coefficient record.

Guard rails, because this is exactly the feature that would let us fit noise
and call it science:

- **Off by default.** Static `t_0` unless drift clears the gates below.
- **Significance.** Block-to-block `t_0` variation must exceed its own
  uncertainty, and the trend must survive a permutation test against
  block-order shuffling.
- **No extrapolation.** Hold `t_0` constant outside the CT-covered time span —
  never continue the trend into files at either end (§3.3a).
- **Corroboration required.** §8.4.
- **Report both.** The static fit and the drifting fit, side by side, with the
  temperature difference between them. If that difference is below the
  systematic budget (§3.3d), take the static one.

### 8.3 What the in-situ correction actually absorbs

An honest caveat before believing any drift number. The in-situ fit absorbs
*everything* between true water temperature and reported FP07 temperature:

- genuine thermistor drift (what we want),
- **the reference's own drift** — we measure FP07 *relative to* the SBE41cp,
- unremoved lag, and the MR-vs-CTD mounting separation,
- biofouling on either sensor.

So "FP07 drift" is unattributable in principle with one reference. In practice
the attribution is defensible in one direction only: Sea-Bird's quoted
*temperature* stability for the SBE41/41CP is far better than any plausible
FP07 drift, so charging the difference to the FP07 is reasonable. (Confirm the
figure against the current spec sheet before it goes in a paper — and note this
holds for temperature specifically; SBE conductivity drift is a different and
much larger story, which is why the reference here is `sci_water_temp` and not
a derived quantity.)

The existence of `cal_FP07_in_situ.m` in Rockland's own ODAS library, framed as
a per-deployment step, is itself an implicit vendor statement that they do not
expect these coefficients to survive between deployments.

### 8.4 Reference-free corroboration: the uncovered yos

The blind spot in §8.1 is that it can only see drift **where CT was on**. Two
checks cover the gaps, and neither needs a reference:

1. **`T1 − T2` over the whole deployment** (D5.4). Continuous, reference-free,
   and directly sensitive to differential probe drift. It is also a
   *discriminator*: if `t_0` drift appears in the blocked fit **and** in
   `T1 − T2`, it is probe-specific and real. If it appears in the blocked fit
   but **not** in `T1 − T2`, then either both probes drifted together — which
   points at the bridge electronics or the reference, not the beads — or the
   apparent drift is an artifact of the CT-covered subset (§D5.5 selection
   bias). Those demand different responses, and this is the only cheap way to
   tell them apart.
2. **Repeat-water-mass check.** Where the glider repeatedly samples the same
   deep, weakly-varying isotherm, FP07-at-depth over time is a drift indicator
   with no reference at all. Assumption-laden — it cannot separate sensor drift
   from a genuine water-mass shift — so it is corroborating evidence only,
   never primary.

### 8.5 What this adds to the plan

Phase 2 gains a **stability report** (blocked `t_0`, fixed `beta_1`, with the
`T1 − T2` overlay). It is cheap given the `pairs`/`fit` split, and it answers
the question with this deployment's own data rather than from priors.

Concretely, the answer to "how stable are they?" should come back as a plot and
three numbers: `d(t_0)/dt` in K/day with its uncertainty, the static-vs-drifting
temperature difference, and whether `T1 − T2` corroborates. **Recommendation:
build the diagnostic in Phase 2, but ship static coefficients until it says
otherwise.**

---

## 8b. What a climb-only deployment costs (osu685, confirmed)

The MicroRider on osu685 was powered at apogee and recorded the ascent only:
every `.p` file starts at 507–1016 dbar and ends at the surface. **1225 files,
zero dives.** That is not a quirk of profile detection — it is how the vehicle
was flown, and it removes three things the plan had been leaning on.

**1. The dive/climb residual split (D5.1) is unavailable.** It was the sharpest
diagnostic for an unremoved lag, because a timing error displaces the two legs
in opposite directions. With one leg there is nothing to compare.

**2. `w` never changes sign, so the sensor-geometry decomposition does not
separate.** The residual carries `dz·dT/dz + τ·w·dT/dz`; the two terms are told
apart only by variation in `w`, and with `w` one-signed they stay ~0.92
collinear. Measured on osu685: `dz = +79 cm` (T1) / `+82 cm` (T2) with a
residual lag of ~2.5 s, but the **split between them is not trustworthy** — only
the combination is. Dives would break this cleanly and cheaply.

**3. Depth and elapsed-time-since-file-start are the same axis.** The glider
always runs deep→shallow, so anything that grows through a file — thermistor
settling, a clock ramp — is indistinguishable from a genuine depth dependence.

> **The one lever that survives.** The glider alternated ~500 m and ~1000 m
> climbs. At a matched depth the deep climbs have been running roughly twice as
> long, so a genuine *depth* dependence cancels in `deep − shallow` while
> anything tracking *elapsed time* does not. This is the only way to answer the
> depth-stability question on this deployment, and it is why the analysis bins
> the two populations separately rather than pooling them.

### 8b.1 The lag, decomposed

Two independent estimators agree to 0.01 s (time-domain on high-passed `L` vs
`1/T_ref`: **+7.37 s**; `dT/dz` matched in depth space: **+7.36 s**), and both
track a deliberately injected shift exactly.

| term | value | how it was obtained |
|---|---|---|
| pressure offset (clock skew + geometric transit) | **+4.5 s** | high-passed P vs P |
| total temperature lag | **+7.4 s** | two independent methods |
| **CTD sensor response** | **+2.7 s** | the difference |

T1 and T2 agree on the response to 0.05 s, which is a real cross-check: the two
probes are independent but share the reference.

The FP07 leads the CTD by ~1 m along the axis (build-dependent — an extended
energy bay changes it) and sits ~17 cm above it perpendicular. **The
longitudinal separation is not recoverable from timing on this deployment** —
per-band offsets on a monotonic climb have a residual rms of 17–20 s, and `1/U`
is 70–90% collinear with elapsed time; the fit returned −18.6 m or +9.3 m
depending on the nuisance term. Nor does pitch separate the two geometric
terms: climb pitch is nearly constant (−19° to −49°, clustered at −44°), giving
`corr(sinθ, cosθ) = −0.964` and a three-term fit that returns −10 m.

What *is* well measured is the combination, by three independent routes that
agree:

| route | result |
|---|---|
| joint calibration+geometry fit on the residual | 79–82 cm |
| direct `P_ctd − P_mr` | 91 cm |
| stated build at the observed 44° pitch: `1.0·sin44 + 0.17·cos44` | 81 cm |

---

## 9. Open questions

1. **Delete-key support in `config_patch`** (A6) — add it, or forbid order
   downgrades? Adding deletion is small and honest; forbidding is smaller.
2. **CTD kernel** — is the SBE41cp value in the `.ebd` an instantaneous sample
   or an internal average? This sets the kernel in D2 and is worth confirming
   against the SBE41cp manual rather than assuming.
3. **Where does `fp07-cal` live** — new `src/odas_tpw/fp07cal/` subpackage, or
   under `perturb/`? Leaning new subpackage: it is a pre-pipeline,
   deployment-scoped tool, whereas `perturb/` is per-file campaign processing.
   (`rsi/` is the other candidate, since the sink is `rsi/config_patch.py` —
   but the *source* of the reference is a hotel file, which is a perturb-side
   concept. A separate subpackage avoids picking a side.)
4. **Data to work against** — no `.p`, no `hotel.nc`, no `.ebd` in the repo
   (A11). Needed before Phase 0 produces anything real.
5. **Does the user's deployment have two thermistors?** D5.4 (the T1−T2 check)
   is what licenses applying the calibration to the uncovered yos. With one
   probe we lose the best cross-check and the report must be correspondingly
   more cautious.
6. **Block size for the stability estimator** (§8.1) — needs the Phase 0
   coverage report first: it is set by how the CT-covered yos are actually
   distributed in time, which we do not yet know.
7. **Confirm the SBE41/41CP temperature stability figure** against the current
   Sea-Bird spec sheet before any drift number is attributed to the FP07
   (§8.3).
8. **Disk budget for patched `.p` files** (D7) — how large is the deployment?
   If a full copy is unacceptable we would have to reconsider the in-pipeline
   apply that D7 currently rules out.

---

## Appendix — adversarial review log

Two passes were made over the draft. Findings that **changed** the plan:

| Attack | Outcome |
|---|---|
| "Just call the existing `fp07_calibrate` with `reference: sci_water_temp` and pool the results." | **Rejected.** A1 — it regresses against an invented reference on the uncovered yos. Pooling bad pairs does not help. Forced R1 and D2. |
| "Interpolate the reference onto the fast grid and fit there — more points, better statistics." | **Rejected.** More points but no more information, and it maximizes the errors-in-variables attenuation (A2). Inverted to D2: decimate the FP07 down instead. |
| "Fit per-file and median the coefficients across files." | **Rejected.** Files with partial coverage give badly-constrained fits; the median is then dominated by the noisiest members. Pool *pairs*, not *coefficients*. |
| "Use order 2 for a better fit." | **Rejected as a default.** A3 + narrow glider T range: order 2 over 3 °C is fitting noise with a badly-conditioned basis. Order is data-driven (D4), default 1. |
| "The lag is one number." | **Rejected.** A9 — clock offset and sensor response are physically distinct and separately measurable. Forced R8 and the P-vs-P estimator. |
| "Key the coefficients on probe SN." | **Rejected.** A8 — FP07 SNs are often placeholders. Forced `probe_sn_trusted` and SN-mismatch refusal. |
| "Emit `beta_1` and be done." | **Rejected.** A5 (`beta` shadows it) and A6 (stale `beta_2`). Forced the live-key logic and V5/V6. |
| "A good RMS residual means the calibration is good." | **Rejected.** A fit with an unremoved lag has a small RMS and a large dive/climb asymmetry. Forced D5.1 as a first-class gate. |
| "The CT-covered yos are representative." | **Not assumed.** Every-*n*th-yo may correlate with time of day or water mass. Forced D5.5. |
| "One static calibration for the deployment." | **Kept, but conditionally.** Only licensed by the T1−T2 stability check (D5.4), which is the one diagnostic that covers the uncovered yos. |
| "Apply the coefficients inside the pipeline via a new `fp07.mode`." | **Rejected** (user direction: pre-pipeline). It puts apply logic in a second place and leaves the raw files lying about their own calibration. Patched `.p` files instead (D7) — no perturb code change, and every reader benefits. |
| "Patch first, then trim." | **Rejected.** Trim rewrites the file and its preservation of a patched config string is untested. Ordering (trim → fit → patch) makes the question moot. |
| "A file with no CT reference is harmless — the code already guards for a missing reference." | **Rejected, and measured.** The reference is never *missing* after the hotel merge, only *fabricated*. The guard fires only for a perfectly constant reference; the gap-ramp case sails through at corr 0.021 (§3.1, A12). Forced R9 and §3. |
| "Refit per temporal block to measure drift." | **Refined, not accepted as stated.** Each block's narrow T range makes `beta_1` wander and drag `t_0` with it through their covariance. Split into global `beta_1` + blocked `t_0` (D8). |
| "Emit a drifting `t_0` since we patch per file anyway." | **Kept but gated off by default.** Free structurally (§8.2), and exactly the feature that would let us fit noise and call it drift. Requires significance, a permutation test, no extrapolation, and `T1 − T2` corroboration. |
| "Measured drift is FP07 drift." | **Rejected as stated.** With one reference it is FP07 *relative to* the SBE41cp, and it also absorbs lag and mounting effects (§8.3). Attribution is defensible but must be stated, not assumed. |
| "Patching is idempotent, just re-run it." | **Rejected.** A second patch would nest banners and destroy the "original config" block. Forced the refuse-if-already-patched check and V10. |

Findings that did **not** change the plan but are recorded:

- The fabricated-reference problem (A1) also affects `ct`, `ctd`,
  `stratification`, `salinity: "measured"` and `epsilon.T_source`. Wider blast
  radius, separate fix, out of scope here.
- Using the same CTD as both the calibration reference and the `T_source` for
  viscosity/κ_T correlates those errors. Second-order; noted, not addressed.
