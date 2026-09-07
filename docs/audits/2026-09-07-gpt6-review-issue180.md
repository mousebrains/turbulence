# GPT-6 whole-codebase review (issue #180): validation, plan, and red team

Reviewed commit: `a7a28a857d3b4077f7368bdd6ae0e2fb44132240`.
Validation transcript and per-finding verdicts:
[issue #180 comment](https://github.com/mousebrains/turbulence/issues/180#issuecomment-5573010223).

**All 21 reproducers run bit-for-bit at the pinned commit.** No finding was
denied. Five needed a scope or severity correction; two live in
collaborator-owned `pyturb/` and are excluded from this remediation by Pat's
decision (2026-09-07).

This document is the remediation plan and the adversarial review of that plan.
Section 3 is the red team: it records the attacks that *changed* the plan, which
is the only part of a self-review worth writing down.

---

## 1. What is being fixed, and what is not

| Finding | Action | Numeric effect on existing products |
| --- | --- | --- |
| F01 bad-buffer QC dropped at L4 | mask in `process_l4` like the other two routes | `rsi-tpw pipeline` only; drops contaminated windows |
| F02 failed probe sets the ratio-test floor | eligible-probe candidate set | recovers windows that were dropped |
| F03a noise-only detection gate | DOF/bandwidth-aware floor | rejects more marginal windows |
| F03b bridged integration | contiguous-band integral | chi falls on ~5% of windows (see §2.3) |
| F04 all-fail chi restored unflagged | export the flag, drop from mixing | K_T/Gamma lose all-failed windows |
| F05 QC gaps widen pairing tolerance | tolerance from the unfiltered cadence | fewer, better epsilon/chi pairings |
| F06 malformed gain silently 1.0 | fail closed | none on valid config |
| F07 one EM sample fabricates a record | coverage + gap guard, provenance | none on well-covered records |
| F08 geometry reappears as bead drift | thread geometry into stability | changes fp07cal drift verdicts |
| F09 correlated pairs counted as replication | profile-clustered SEs | widens dz/tau error bars |
| F10 asynchronous ADCP masks | common-support differencing | more NaN, less fake shear |
| F13 min-span overrides anti-alias ceiling | hard bound + refusal | none at `f_AA = 98` |
| F14 catastrophic cancellation | two-pass centered variance | diagnostics only |
| F15 equal length taken as alignment | require the depth dimension | none on real profiles |
| F16 odd FFT lengths | parity-aware fold + shared segment count | none at even `nfft` |
| F17 window fill undoes `max_gap` | max consecutive hole check | rejects fabricated windows |
| F18 outlier absolves itself | uncertainty floor + leave-one-out scale | more continuity flags |
| F19 T2 drift corroborated with T1's sign | channel-aware sign | fixes inverted verdicts |
| F20 rank failure is not a gate | identifiability gate on the full design | refuses unidentifiable fits |
| F21 constant pressure yields no bins | guarantee one bin | none on real profiles |

**Not fixed here:** F11 (no `time` coordinate) and F12 (per-window `k` grids
collapsed to a mean), both in `src/odas_tpw/pyturb/_compat.py`. That subpackage
is Jesse's project, hosted here for his convenience and outside this pipeline.
Both findings are correct and are documented in the issue comment for him to
act on. Editing them without his sign-off would be the wrong call regardless of
the defect.

---

## 2. The plan

Ordered so that each group can be tested before the next lands. Every finding
gets a regression test built from the reviewer's own counterexample — their
"required closure evidence" item 1 — because a fix without the negative control
that provoked it is not closed, it is unobserved.

### 2.1 Group A — fail closed on bad input (no change to valid data)

**A1 (F06).** `rsi/channels.py`: add `_require_finite_float`, which raises on a
key that is *present but unparseable, non-finite, or non-physical*, while
keeping the existing warn-and-default behaviour for a *missing* key (that
asymmetry is deliberate and documented). Apply to `diff_gain` in
`convert_shear`, and to the ADC scaling (`adc_fs`, `adc_bits`) that multiplies
every converted channel. The thermistor `beta_*` reciprocals get the same
treatment: `beta_2 = 0` is already documented as an infinite term, but
`beta_2 = "0,1"` currently becomes `0.0` and raises `ZeroDivisionError` from
inside a numeric expression rather than a named calibration error.

**A2 (F13).** `scor160/l4.py::_variance_method`: re-apply `min(..., K_AA)`
*after* the `K_LIMIT_MIN` clip so the anti-alias ceiling is a hard bound, and
when fewer than three trusted bins remain below `K_AA`, return a non-estimate
rather than integrating into excluded frequencies. Warn with the offending
`f_AA` and speed.

**A3 (F16).** `scor160/spectral.py`: halve the last one-sided bin only when
`nfft` is even (it is Nyquist only then), in both `csd_matrix` and
`csd_matrix_batch`. Add `n_segments(n_samples, nfft, overlap)` next to the
window helpers and use it in both estimators *and* in
`goodman._bias_correction`, replacing `2*N//nfft - 1`. Proved by exhaustive
search that the two agree for every even `nfft`, so this is a no-op on every
production configuration and a correction for odd ones.

**A4 (F21).** `rsi/binning.py::bin_by_depth`: when `ceil(max) == floor(min)`,
extend the range by one `bin_size` so constant-pressure data lands in a bin
instead of vanishing.

**A5 (F15).** `perturb/binning.py::_load_profile_snapshot`: accept a 1-D
variable only when its dimension *is* the depth variable's dimension. Length
equality is not alignment.

### 2.2 Group B — QC propagation and provenance

**B1 (F01).** Give `process_l4` a `mask_bad_buffers: bool = True` parameter and
apply `bad_fraction > 0 | interp_fraction > BAD_MAX_INTERP_FRACTION` exactly as
`rsi/dissipation.py:360-375` and `chi/l4_chi.py:163-166` already do, sharing the
one `BAD_MAX_INTERP_FRACTION` constant. Carry `bad_fraction` /
`interp_fraction` onto `L4Data` so the *reason* survives — a NaN epsilon with
flag 255 says "invalid", not "the RDL dropped buffers here".

**B2 (F02).** In `_compute_flags`, form the dissipation-ratio candidate set from
probes that have not already failed FOM or despiking; if that empties the set,
fall back to all finite probes so the test degrades to today's behaviour rather
than silently switching off. Document in the docstring that the ATOMIX reference
files disagree with each other on this point (`MSS_Baltic` flags a passing probe
against a FOM-failing minimum and flags the minimum too; `epsifish` does not),
so the choice rests on Lueck et al. (2024) §3.4.5 and on the principle that a
probe already declared untrustworthy must not be the yardstick.

**B3 (F04).** `_compute_chi_final` and `mk_chi_mean` return a per-window
`chi_qc_fallback` boolean alongside the value: true where no probe passed the
fom band / `K_max_ratio` floor and the value is the all-failed geometric mean.
Export it (`chi_qc_fallback` variable in the rsi chi product and the perturb
`chi_NN` product) and exclude flagged windows from the K_T / Gamma / K_rho
inputs by default, with a config escape (`chi.mixing_use_qc_fallback: false`).
The reported chi stays finite and backward compatible; the derived mixing
products stop being built from inputs that failed every test.

**B4 (F05).** `processing/mixing.py::pair_nearest`: derive the default `max_dt`
from the *unfiltered* `src_times` cadence, before the finite mask is applied, and
cap it so a sparse source can never pair across an arbitrary span.

**B5 (F07).** `rsi/speed.py`: for `method='em'`, require the same 50% finite
coverage the `hotel` route already requires, and reject an interior gap longer
than a configurable `max_gap_s`. For `method='flight'` keep the "any finite
sample" rule — the model legitimately NaNs every sample below `min_pitch_deg`,
so a coverage fraction would false-reject real casts — but apply the same
interior-gap ceiling, which those inflection NaNs never breach. Return the
measured coverage and the largest interior gap in the provenance so window QC
can see imputation rather than inferring it.

### 2.3 Group C — chi integration (changes chi numbers)

**C1 (F03b).** `chi/chi.py`: integrate on the original wavenumber grid over the
contiguous band `[K[valid][0], K[valid][-1]]` instead of over the subsetted
abscissa, at all five sites (`:374`, `:384`, `:392` in Method 1;
`:558`, `:670`, `:720`, `:760` in Method 2). This is not a matter of taste: the
`_variance_correction` factor already treats `[K_min, K_max]` as a *continuous*
interval, so the current subsetted integral and its own correction disagree
about what band was measured. Measured effect on the repo's `VMP/` corpus
(29 files, 2944 probe-windows): median ratio 1.0000 (MAD 0.0000), 5.50% of
windows fall by more than 5%, 1.19% by more than 50%, worst case 3.79×.

**C2 (F03a).** Make the non-detection gate aware of how many bins were searched
and of the window's spectral degrees of freedom. Plumb `num_ffts` onto
`L3ChiData` (already computable from `L3Params`), and require the count of bins
above 2× noise to exceed the null expectation `n_bins · P(chi²_d/d > 2)` plus a
one-sided margin, with the current `min_points = 3` retained as a floor. When
`num_ffts` is unavailable (older products, direct calls) the behaviour is
exactly today's. This is a calibrated floor, **not** the full window-level null
test the review asks for; the residual is recorded in §4.

### 2.4 Group D — calibration and clock inference

**D1 (F08).** `fp07cal/stability.py::blocked_offsets` takes the `GeometryFit`
and subtracts `dz·g + tau·w·g` from the reference before blocking, so a changing
gradient across the deployment cannot masquerade as bead drift. `cli.py` passes
the `geo` it already has.

**D2 (F09).** Profile-clustered covariance for `dz` / `tau` in `joint_fit` and
`geometry_fit` (cluster-robust sandwich on `profile_uid`), reported alongside
the conditional IID SE rather than replacing it, and a warning when the cluster
count is too small for the sandwich to be trustworthy. Duplicating every
observation must leave the SE unchanged.

**D3 (F20).** Identifiability gate: rank-revealing SVD on the *full scaled*
design, report the smallest singular value and the numerical rank, and refuse to
export coefficients when the fit fails a physical prediction check
(`max |T̂(L) − T_ref|` over the fitted range above a generous limit) or when the
design is rank-deficient. Extend `separately_resolved` so it means "separately
resolved from the calibration polynomial as well as from each other", not just
the dz/tau correlation.

**D4 (F19).** `corroborates` reads `stab.channel` and flips the expected sign for
the second channel of the differential.

**D5 (F17).** `clocksync/fit.py::window_pairs` checks the longest *consecutive*
missing run against `max_gap` after common-grid alignment and skips the window
when it is exceeded, instead of asserting in a comment that no such run exists.

**D6 (F18).** `clocksync/qc.py::check_continuity` floors the robust scale with
the fits' own `offset_sigma` and computes each residual's scale leaving that
residual out, so an extreme slip cannot supply the scale that absolves it.

### 2.5 Group E — ADCP and binning

**E1 (F10).** `perturb/adcp.py::window_shear` averages each adjacent cell pair
over the ensembles where *both* cells and both depths are finite, so a
first-difference is never taken between two different ensembles. Export the
per-pair support count. Velocity-noise suppression (the reason the means are
taken first) is preserved.

**E2 (F14).** `perturb/binning.py::_bin_std` uses a centered two-pass
accumulation. Offset invariance, constant input, and tiny-variance cases become
tests.

---

## 3. Red team on this plan

Attacks that changed a decision. Attacks that failed are listed at the end,
because a plan that survived nothing was not attacked.

### 3.1 "B1 masks the wrong thing, and destroys the evidence"

Setting `epsi = NaN` makes `_compute_flags` write 255 = *invalid estimate*. A
user reading the L4 product then cannot distinguish "the RDL dropped buffers" from
"the spectrum was unusable" — and the whole point of F01 is provenance.
**Changed:** `L4Data` gains `bad_fraction` / `interp_fraction`, and the pipeline
writes them into the NetCDF, matching what `rsi/dissipation.py` already
publishes. Masking without publishing the reason would have half-fixed it.

### 3.2 "B2 could regress the ATOMIX benchmark"

It cannot, and I checked rather than assumed: `tests/test_atomix_l3l4_gate.py`
and `scripts/compare_atomix.py` compare `_estimate_epsilon` output against
reference `EPSI`; neither compares `EPSI_FLAGS`. But the deeper attack landed:
I originally justified B2 by "the reference excludes poor-FOM probes". It does
not — `MSS_Baltic` win 37 does the opposite. **Changed:** the justification now
rests on the paper and on the principle, and the docstring records the
benchmark's internal disagreement so nobody re-derives the wrong warrant later.

### 3.3 "C1 changes published chi; you cannot land that on an assertion"

Correct, and this was the plan's weakest point. The original wording was "fixes
a bridging bug". **Changed:** C1 now carries (a) the self-consistency argument —
`_variance_correction` already assumes a contiguous `[K_min, K_max]`, so the
subsetted integral contradicts its own correction factor — and (b) a measured
before/after on the whole repo corpus with the distribution, not a mean. The PR
must publish the same distribution after the change, not just "tests pass".

### 3.4 "C2 as first written is a research project disguised as a patch"

The review asks to "calibrate a window-level noise-null test for bandwidth/DOF
and correlation". Welch bins from overlapping segments are *not* independent, so
a clean analytic false-positive rate does not exist, and a binomial threshold
that pretends they are will be miscalibrated in the direction of accepting
noise. **Changed:** C2 is explicitly scoped down to a DOF-aware *floor* that
strictly dominates the current fixed `min_points = 3`, keeps today's behaviour
when DOF is unknown, and is pinned by a Monte-Carlo test that records the
achieved false-positive rate rather than asserting one. The unclosed remainder
is written into §4 instead of being quietly declared fixed.

### 3.5 "B5 will break osu684/osu685, where U_EM is known bad"

Real risk: those deployments have documented EM problems, and a hard error on
low coverage could stop a working analysis. **Changed:** the `em` threshold
matches the existing `hotel` rule (50%) rather than inventing a stricter one,
the gap ceiling is configurable, and coverage/gap are *reported* on every run
regardless of whether they trip. A partly-dead flowmeter that still covers most
of the cast keeps working and now says so.

### 3.6 "D3 could refuse a fit that has been used in production"

`osu685` has a published 72-day stability result. A condition-number gate would
be an arbitrary line that a healthy but poorly scaled design could cross.
**Changed:** the export gate keys on a *physical* prediction failure
(`max |T̂ − T_ref|` far outside anything a working calibration produces) plus
outright rank deficiency, with the singular values reported either way. A
conditioning number alone is a diagnostic, not a verdict.

### 3.7 "D2's cluster-robust SE is unreliable with six profiles"

Sandwich estimators need many clusters; six is not many, and replacing a
too-narrow SE with a noisy one is not obviously progress. **Changed:** both SEs
are reported, the clustered one is labelled as the one to quote, and a warning
fires below a documented cluster count so the number is never read as
authoritative when it cannot be.

### 3.8 "E1 trades fake shear for missing shear without saying so"

Common-support differencing will produce NaN where patchy coverage previously
produced a number. That is correct but it silently changes yield. **Changed:**
the per-pair support count is exported so a drop in yield is attributable rather
than mysterious.

### 3.9 "One PR for nineteen findings across twelve modules is unreviewable"

Genuine, and I have no clean answer: the findings interact (F01 → F02 → F05 all
touch which windows survive to mixing), and splitting them means landing a
pipeline in an inconsistent intermediate state. **Decision, not a change:** one
PR, but grouped commits (A/B/C/D/E) that are individually reviewable and
individually revertible, and the numeric-effect table in §1 up front so a
reviewer knows which commits can change a published number.

### Attacks that failed

* *"A3 changes the benchmark"* — exhaustive check over every even `nfft` from 2
  to 4096 shows the old and new segment counts are identical; the endpoint fold
  changes only the odd case.
* *"A2 changes epsilon"* — needs `K_AA < 7 cpm`, i.e. `W > 12.6 m/s` at the
  default `f_AA`. Unreachable; the guard is for a misconfigured `f_AA`.
* *"F14 affects products"* — `_bin_std` runs only under
  `binning.diagnostics: true`. The fix is still right, the severity is not.
* *"B4 will change every pairing"* — only pairings whose source series had
  QC gaps, and only in the direction of refusing distant matches.

---

## 4. What this does not close

Honest residue, so the next reviewer does not have to rediscover it:

1. **C2 is a floor, not a null test.** Overlapping Welch bins are correlated;
   the achieved false-positive rate is measured by a Monte-Carlo test, not
   derived. A defensible detection limit needs a calibration against real
   noise-only records from each instrument.
2. **Absolute chi scale (#179) and the SN132 gain (#178) are untouched.** They
   are measurement problems, not code problems.
3. **`X_ISR` / `DEFAULT_ISR_MARGIN` circularity stands.** Both were tuned on the
   ATOMIX files later used to report agreement. Only a genuinely held-out
   deployment retires this, and none of the fixes here changes that.
4. **No field reprocessing.** The review's closure item 4 asks for a re-run of
   affected products with pinned inputs and a quantified delta. This PR
   quantifies the C1 delta on the repo corpus only. Campaign-level impact
   remains unmeasured.
5. **F11 / F12 remain open** in `pyturb/`, pending the owner.
