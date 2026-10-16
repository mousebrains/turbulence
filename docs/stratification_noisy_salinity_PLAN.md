# Background stratification with a noisy salinity: PLAN

Status: draft, 2026-10-15. Prompted by two VMP-5500 casts from an outside
user (casts A and B, 2026; unpumped SBE3/SBE4 CTD). The casts are not ours to
publish, so they and the analysis scripts are outside the repo; the numbers
below are from them.

## 1. Problem

`processing.mixing._stable_window` sorts every window's parcels by potential
density before it fits dT/dz and differences N². That is right when the
salinity is good. On these casts the unpumped SBE4 salinity is not:

| cast A, 150-250 dbar, 2-dbar windows | median dT/dz [K/m] | median N² [s⁻²] |
|---|---|---|
| 100-dbar linear fits (reference) | -0.0701 | 7.50e-5 |
| density sort, measured S (today) | -0.0302 | 7.37e-4 (10x) |
| density sort, S = 35 | -0.0650 | 1.34e-4 |
| temperature sort, measured S | -0.0663 | 2.96e-4 |

cast B is worse (N² 18x high, dT/dz -0.019 against -0.064).

Facts measured on the two casts (n = 2 casts, 100 windows each at 150-250
dbar; details outside the repo):

1. The salinity error is structured, not white: rms Thorpe displacement from
   density is 20-28 m against ~1 m from temperature alone. No C shift within
   ±6 s removes it, and it scales with dT/dz.
2. The density sort carries that error into **dT/dz** as well as N²: sorting
   by a density dominated by salinity noise scrambles the temperature order.
3. The SBE3 (SBT1) trails the FP07s by ~4 s (3.5-4.4 s in 2-minute blocks).
   Even shifted by 4.2 s, SBT1's 2-dbar dT/dz differs from the FP07's by more
   than 2x in 66-78% of windows (high-passed above 0.1 Hz, the two correlate
   at r = 0.3-0.5), while T1 and T2 agree within 2x in 96-100% (r = 1.00).
   The CT temperature is not a good estimate of the gradient χ worked against.
4. **No salinity smoothing by depth fixes N² here.** With a temperature sort
   and S replaced by a running linear fit over L dbar (assigned by position):

   | L [dbar] | cast A median N², frac ≤ 0 | cast B median N², frac ≤ 0 |
   |---|---|---|
   | raw | 2.5e-5, 0.46 | 3.9e-4, 0.43 |
   | 10 | 7.7e-5, 0.35 | 2.6e-5, 0.49 |
   | 50 | 7.1e-5, 0.29 | 8.5e-5, 0.34 |
   | 100 | 5.6e-5, 0.28 | 6.5e-5, 0.26 |

   The medians come right, but a quarter or more of the windows stay
   negative. That is physics, not noise: in a staircase the layers are mixed
   in **both** T and S, so imposing a smooth dS/dz on a layer with dT/dz ≈ 0
   gives N² < 0. Salinity has to follow temperature at the step scale,
   S = f(θ). The casts' own T-S fit worked on cast A and failed on cast B
   (dT/dz -0.034), so f has to come from a better sensor (a ship CTD).

### 1b. It is not only this instrument

The same density sort biases the products of instruments with good, pumped
JAC CTDs, at a smaller level. Measured on stored perturb products,
30-40 casts per campaign sampled evenly, temperature-stratified windows only
(|unsorted dCT/dz| ≥ 1e-3 K/m), medians per cast then across casts (MAD):

| campaign (window) | density-sorted dT/dz ÷ unsorted | T-sorted ÷ unsorted | FP07 ÷ JAC, unsorted |
|---|---|---|---|
| CASPER-West 2017 (2 dbar) | 0.881 (0.036) | 1.002 (0.002) | 0.990 (0.028) |
| ARCTERX-2025 Interior (2 dbar) | 0.887 (0.040) | 1.038 (0.031) | 0.991 (0.017) |
| Taiwan17 (2 dbar) | 0.881 (0.029) | 1.008 (0.004) | 0.995 (0.013) |
| ASTRAL 2023 (2 dbar) | 0.920 (0.014) | 1.003 (0.001) | 0.994 (0.013) |

By the rearrangement inequality the monotonic (temperature) order gives the
steepest least-squares slope of any ordering, so a sort that respects
temperature can't come out shallower than the measured order. The density
order does: salinity noise (the JAC T/C response mismatch spikes) is
reordering parcels. FP07 and JAC agree within 1% before sorting, so the
FP07 path is not a different gradient.

The **stored chi-window dT/dz**, the one K_T uses, against a temperature-only
recompute at the same window times (n = 27-30 casts each):

| campaign | chi window | stored ÷ T-only | implied K_T factor |
|---|---|---|---|
| ARCTERX-2025 Interior | 2 s | 0.700 (0.070) | 2.04x |
| Taiwan17 | 2 s | 0.718 (0.040) | 1.89x |
| CASPER-West 2017 | 4 s | 0.933 (0.029) | 1.15x |
| ASTRAL 2023 | 4 s | 0.922 (0.022) | 1.18x |
| ARCTERX-2023 Wake | 4 s | 0.972 (0.034) | 1.06x |

Shorter windows are worse, as sample-scale salinity noise predicts and
m-scale compensated intrusions do not. N² shows the matching bias: in the same
2-dbar windows the density-sorted median is 1.12-1.48x the unsorted median,
while a temperature sort that carries each parcel's salinity gives 0.95-1.08x
with no more N² ≤ 0 windows than unsorted (2-7%). Γ ∝ N²/dTdz² takes both.

Theory points the same way for dT/dz. χ is the dissipation of *temperature*
variance, and the background gradient it works against is that of the sorted
temperature field (Winters & D'Asaro 1996; the χpod processing uses it,
Warner 2021). Sorting temperature by density is the right reorder only when
density and temperature orders coincide.

## 2. Scope

**This PR** (three changes; `dTdz_source: fp07` is opt-in, and the default
sort became `temperature`, section 10):

- **A. `stratification.sort`**: `temperature` (default since 2026-10-16;
  `density` before) | `density` | `none`.
- **B. `stratification.dTdz_source`**: `ct` (default, unchanged) | `fp07`.
  dT/dz from the in-situ-calibrated FP07s.
- **C. Sort-agreement diagnostic**: with a varying salinity, under any sort
  (section 10; first written for `sort: density` only), compare each window's
  density-sorted dT/dz with its temperature-sorted dT/dz; record the median
  ratio κ per cast/profile and warn when κ < 0.8.

**Default.** Section 1b argues for `sort: temperature` as the default. That
changes K_T and Γ in every campaign's products by up to 2x, so it was Pat's
decision. **Decided 2026-10-16: flip it** ("T is typically a more stable
observation"); section 10 has what that changed.

**Follow-up PR** (not here): `stratification.salinity: "ts:<path>"`, S from
a user-supplied θ-S relation. Section 1 item 4 says this, not depth
smoothing, is the "smooth salinity" that works. It needs a file fingerprint in
the skip-cache (a path string alone would hash identically when the file
changes; same trap as a URL in `hotel.file`), so it is its own change. Today
the same result is reachable through `salinity: "hotel"` with S = f(SBT1)
written on the VMP clock.

Out of scope: rsi-tpw's `run_pipeline` options (it gets the diagnostic only);
`mixing_look`; aligning the CT channels to the FP07 clock with the fp07-cal lag
(a separate idea, noted in section 8).

## 3. Design

### A. Sort order (`processing/mixing.py`)

`_stable_window` gains `sort` and returns `(N2, dTdz, dTdz_T)`, where `dTdz_T` is the
temperature-sorted slope under `sort="density"` (NaN otherwise; section 3C).
(As built after the default flip, section 10: `(N2, dTdz, dTdz_rho, dTdz_T)`,
with both κ slopes under every sort.)

| `sort` | parcel order assigned to the depth-sorted positions |
|---|---|
| `density` | ascending `sigma0` (today's code, bit-identical) |
| `temperature` | CT at the window's median salinity, monotonic in the direction of a **quadratic fit over five windows** centered on this one; the measured order where that fit's extremum lies inside the window or its central gradient is under 1e-4 K/m (revised after review, section 9) |
| `none` | position order (no sort) |

N² is `gsw.Nsquared` on the shallow/deep-half means of (SA, CT) in that
order, as today.

Why the trend direction: a temperature sort that always puts warm on top
would flip the gradient's sign in a diffusive-regime window (T rising with
depth) and report a stable N² there. Following the trend removes inversions
*relative to the local gradient* in either regime. The trade-off is stated in
the docs: a temperature sort also removes real density-compensated
inversions (intrusions), so it is for water where temperature carries the
density signal, or where the salinity cannot be trusted.

`sorted_stratification`, `profile_stratification` take `sort=` (default
`"density"`). `StratificationResult` gets a third field `dTdz_T`.
`profile_stratification` keeps its 3-tuple return; a new
`profile_stratification_full` returns a NamedTuple
`(target_P, N2, dTdz, dTdz_T)` and `profile_stratification` wraps it. A
library function `sort_kappa(dTdz, dTdz_T, min_abs=1e-3, min_windows=5)`
reduces the pair to κ. (Since section 10: default `sort="temperature"`, a
`dTdz_rho` field on both result types, and `sort_kappa(dTdz_rho, dTdz_T, ...)`.)
Callers use attribute access today (checked: rsi/pipeline.py,
perturb/pipeline.py, mixing_look.py, plot/gamma_scaling.py, tests).

`sort` is validated against `SORT_MODES = ("density", "temperature", "none")`
in the library (ValueError) and at config load.

### B. dT/dz from the FP07s

New library function, the temperature-only twin of the two stratification
functions:

```python
window_temperature_gradient(win_times, win_half_width, t, P, T, S=None,
                            lat=0, lon=0, sort="temperature",
                            min_samples=4, min_dp=DEFAULT_MIN_DP) -> ndarray
profile_temperature_gradient(P, T, S=None, lat=0, lon=0, window=2.0,
                             sort="temperature", ...) -> (target_P, dTdz)
```

- CT from the FP07 temperature with the window's **median** salinity (scalar
  per window; 35 PSU if none). Salinity is not co-registered with the FP07,
  and ∂CT/∂SP ≈ -0.013 to -0.031 K/psu at fixed t (measured with gsw at
  20/15/5 °C), so a 0.5-psu salinity spike would put 15 mK into a parcel's CT.
  A per-window scalar removes that path entirely.
- `sort="density"` is not available here (no co-registered density), so the
  caller maps it to `temperature`, and the variable comment says so.
- Per probe, then `nanmean` across probes. Not the mean temperature: two beads
  on nominal coefficients sit 0.6 K apart (casts A/B: +3.1 and +3.7 K), so a
  per-sample mean jumps wherever one probe drops a sample.

Pipeline wiring (perturb, `dTdz_source: fp07`):

- **Channels**: `_find_fp07_channels` (`^T\d+$`, type therm). Each is gated by
  `_gate_reference_temperature` (rails, #166). A fast channel is box-car
  averaged over `fs_fast/fs_slow` samples and interpolated onto `t_slow`, so
  both products use slow-rate FP07 on the same grid as P. Sorting a noisy
  512 Hz series would bias the slope steep where the gradient is weak.
- **Profile product** (`_compute_slow_stratification` runs before the FP07
  calibration, in the main flow): after `fp07_calibrate` in the profile fork,
  recompute `dTdz` from the calibrated FP07 on the **same pre-trim casts** N2
  used. (Superseded after review: the whole block moves to after the
  calibration instead; section 9.)
- **Diss and chi products** (`_window_stratification_for_profile`): read the
  FP07 channels and their `calibration` attrs from the profile NetCDF.
- **No FP07 channel** (or none with two finite samples): warn and fall back to
  the CT gradient, with the fallback in the comment. Never a silent switch.
- **Provenance** in the `dTdz` comment: which channels, each one's
  calibration tag (in-situ vs which reference, or "embedded coefficients"),
  the sort, "mean of per-probe slopes".

Why this is the better gradient where it applies (besides fact 3):
K_T = χ / (2 dT/dz²) and Γ = N² χ / (2 ε dT/dz²). With χ and dT/dz from the
same bead, a bead gain error g enters χ as g² and dT/dz² as g², so **K_T and
Γ are independent of the FP07 gain**, whether or not the in-situ fit
succeeded. (Exact per probe for a *static* gain; for chiMean, which drops
probes per window, only to first order in the probe-gain difference, about
±4% on SN479's factory pair. Dynamic errors such as a wrong diff_gain or bead
attenuation do not cancel.) The dTdz product itself carries the
gain, so the comment names the calibration.

N² stays on the CT pair: density needs T and S measured at the same time,
which the FP07 and a lagged SBE4 are not.

### C. Diagnostic and warning

`_stable_window` also returns the temperature-sorted dT/dz when
`sort="density"` (one extra sort and fit). Then

    kappa = median over windows of dTdz_density / dTdz_temperature,
            over windows with |dTdz_temperature| >= 1e-3 K/m and at least 5 of them

κ ≈ 1 means salinity does not reorder the parcels. κ < 1 means it does:
salinity noise, or density-compensated structure. The warning says both and
names the remedy for the first.

Computed with a per-sample salinity only. With a fixed salinity the two
orders coincide in stable windows and κ is 1 or meaningless. (First written
for `sort: density` only, on the grounds that under `temperature`/`none` there
is nothing to diagnose. Reversed with the default flip, section 10: N² still
carries the salinity, so κ is computed, and gives the same value, under every
sort.)

Why not the sorted/unsorted N² ratio first proposed (and promised in the
reply draft): on the fleet it false-alarms. Shallow, mixed-layer-dominated
casts (max P 9-64 dbar) gave ratios of 5-49 with good CTDs, and the winter
ARCTERX-2025 Interior casts had 55% of casts above 2, from real
weak stratification rather than salinity. κ looks at the thing that goes
wrong, the order.

Threshold `KAPPA_WARN = 0.8` (K_T off by more than 1/0.8² = 1.56x). Fleet
κ at 2 dbar (median per campaign): CASPER 0.87, ASTRAL 0.85, ARCTERX-2023
Wake 0.88, ARCTERX-2022 Interior 0.76, Taiwan17 0.79, ARCTERX-2025 Interior
0.67; the VMP-5500 casts 0.46 and 0.28; husker (smooth hotel salinity) 1.00. So
it would fire on three of six good-CTD campaigns, correctly by section 1b.

- **Profile product**: κ per cast; one warning per file naming the casts
  below 0.8 and the median; the N2 and dTdz comments record the median κ.
- **Diss/chi**: `sort_kappa` (and the worst 50-dbar band) as attributes on `N2`, the variable the density sort always shapes, and a warning per
  profile below 0.8.
- **rsi-tpw `run_pipeline`**: the same attribute and warning.

### D. Config (perturb)

```yaml
stratification:
  sort: density        # density | temperature | none
  dTdz_source: ct      # ct | fp07
```

- Validated at load (`validate_config`); unknown value = ValueError.
- Canonical form: a key is dropped from the hashed section view when it
  selects main's behavior (`sort: density`, `dTdz_source: ct`), so such a
  config keeps its signature (the #211 `anchor` precedent). Since the default
  flip, an unset or null `sort` resolves to `temperature` and is **kept**, so
  density-sorted products from before the flip are recomputed, not reused
  (section 10). Note the engine fingerprint changes with any source edit
  anyway; this keeps a pinned-fingerprint run stable.
- Template comments and `docs/perturb/configuration.md` updated;
  `docs/mixing_efficiency.md` gets the gain-cancellation note.

## 4. Behavior that must not change

- `sort: density`: every N2/dTdz value bit-identical to main (the existing
  code; a test pins it against a stored result from main). Until the default
  flip (section 10) this was the default config; now the default changes the
  products, on purpose.
- `window_stratification` untouched.
- Existing comment text changes in one place: today's dTdz comment says
  "in-situ temperature" but the code fits conservative temperature since audit
  r1-1. Fixed.

## 5. Validation

1. Unit tests on synthetics (section 6).
2. The two VMP-5500 casts through perturb outside the repo (not committed;
   not our data): `sort: temperature` and `dTdz_source: fp07` should bring median dT/dz
   to ≈ -0.065 to -0.070 at 150-250 dbar; κ should be ≈ 0.3-0.5 on both.
   (Wrong for the median in a staircase; see section 9, end-to-end run.)
3. **Fleet check (done, section 1b and 3C)**: the sorted/unsorted N² ratio
   failed it; κ separates. Re-run the scripts against the implementation so the
   library's κ matches the prototype's per campaign.
4. **FP07 vs CT dT/dz (done, section 1b)**: 0.990-0.995 unsorted on four
   JAC campaigns. The exception is informative: husker (MR on a Slocum, hotel
   CTD) gives 0.51, with the hotel slope 1.7-2.4x the FP07's at 60 and 100
   dbar and equal at 30 and 150, as a clock offset between the MR and the glider
   CTD would make on a curved thermocline (the whole-cast T1-vs-hotel slope is
   1.0015, so the calibration is fine). On a glider the FP07 is the gradient
   that is co-registered with P.

## 6. Tests

- `sort="density"` reproduces main bit-for-bit (stored fixture).
- Clean linear profile: all three sorts agree; κ ≈ 1.
- Structured salinity noise: density sort inflates N² and flattens dT/dz;
  temperature sort recovers dT/dz within 5%; κ < 0.8 and the warning fires.
- Diffusive window (T rising with depth): temperature sort keeps the positive
  slope; a warm-on-top sort would not.
- `none` equals `window_stratification`'s N²/dT/dz on the same window.
- FP07 gradient: +3 K offset leaves it unchanged; gain g scales it by g;
  salinity spikes don't move it; fast-channel decimation path.
- Bad `sort` / `dTdz_source` values raise at config load.
- Canonical form unchanged for a config without the new keys.
- `_window_stratification_for_profile` with `fp07`: synthetic profile NetCDF
  with T1/T2 (one in-situ tagged), comment names both; no FP07 → CT fallback
  with the fallback in the comment.
- Profile product: `dTdz` replaced after calibration, N2 unchanged.
- Diagnostic: attribute written, warning text, NaN with < 5 usable windows,
  not computed for fixed salinity (computed under every sort since
  section 10).

## 7. Reply to the user

Update the reply's "what the code should do" to what lands: the two options,
the κ attribute/warning (not the sorted-vs-linear N² warning the draft
promised, which false-alarms; `sort: none` gives the unsorted N² on demand),
the θ-S follow-up, and section 1b: the question exposed a bias in everyone's
products.

## 8. Not doing, and why

- **Position-smoothed salinity**: section 1 item 4.
- **Writing a second N² variable**: `sort: none` produces it.
- **Default `dTdz_source: fp07`**: plausibly better for every instrument
  (gain cancellation), but it changes every campaign's products. Decide after
  the fleet comparison in section 5 item 4.
- **Shifting the CT pair by the fp07-cal lag**: fp07-cal already measures the
  reference's lag against each FP07 (-4 s here). Applying it to the CT
  channels would fix the N² misregistration too, but it touches salinity,
  the CTD product, and fp07-cal's own fit. Separate change.

## 9. Adversarial review (2026-10-15) and what changed

One reviewer, reading the plan against the code with simulations. Findings and
dispositions (all checked; "fixed" means in the implementation and tested):

1. **Trend-following temperature sort fails** (major, verified by sim): an
   overturn larger than half the window reverses the window's own trend, so
   the sort kept it unstable (N² < 0, masked in the most energetic windows); at
   a temperature extremum any monotonic sort invents a gradient. **Fixed**:
   direction from a quadratic fit over a span 5x the window; measured order
   where its vertex lies inside the window or its central gradient is under
   1e-4 K/m. Tests: a 3-m overturn in a 2-dbar window is stabilized and its
   dT/dz matches the density sort within 2%; a salinity-held temperature
   minimum keeps the measured order.
2. **Cast-wide diagnostic dilutes a band-limited failure, false-alarms in mixed
   layers** (major): moot for the ratio-of-medians, already replaced by κ, but
   the dilution point holds for κ too. **Fixed**: κ is also taken per 50-dbar
   band (≥ 10 voting windows); the warning goes on the worst band. Mixed layers
   don't vote (|dT/dz_T| < 1e-3 K/m).
3. **Provenance lost in binned/combo products** (major): **fixed** with a short
   `stratification` attribute ("sort=...; dTdz_source=...") on N2/dTdz, added
   to binning's preserved attributes (first-file-wins caveat documented), plus
   run-summary notices.
4. **A failure in the FP07 path could drop diss and, with it, chi** (major):
   **fixed**: the reader and the gradient are each wrapped; any exception falls
   back to the CT gradient with the reason in the comment. Tested by injecting
   exceptions.
5. **No check that a bead is alive** (major): a test plug reads a constant in
   range; its zero slope would halve a two-probe mean (K_T 4x high).
   **Fixed**: per probe, over the casts, slope 0.5-2 and r ≥ 0.9 against
   `ct.T_name` (or a range ≥ 0.01 K without one); dropped probes are named.
   Per-probe K_T (χ_i / 2 dTdz_i²) would remove the residual chiMean issue;
   deferred.
6. **Gain cancellation overstated** (minor): wording narrowed (section 3B and
   docs/mixing_efficiency.md).
7. **FP07-vs-JAC agreement is partly by construction** (major for the
   validation claim): yes, fp07-cal fits the FP07 to JAC_T, so 0.99 checks the
   FP07 path's mechanics (decimation, sort, conversion), not its gain. Also,
   median-salinity CT vs per-sample CT differ by the haline term dCT/dSA x
   dSA/dz (0.2% in the open ocean, up to tens of percent in a strong
   halocline). For χ of in-situ temperature the median-salinity (θ-like)
   gradient is arguably the commensurate one; today's CT path includes the
   haline term. Left as a follow-up; the temperature sort uses median-salinity
   CT for dT/dz, the density/none paths keep per-sample CT (unchanged).
8. **`ct.T_name: T1` already exists; profile-product stratification ran before
   the FP07 calibration** (minor, latent bug): **fixed** by moving the whole
   block to after `fp07_calibrate`, on a copy of the detected (pre-trim)
   casts. Bit-identical for every config whose `ct.T_name` is not an FP07
   (nothing between the old and new positions touches P, T or C). The stale
   comment claiming the CTD product bins N2 went with it.
9. **Config validation** (minor): **fixed**: also validated per file next to
   the salinity normalization (configs built in code skip `load_config`); a
   test pins the default canonical form against main's.
10. **API** (minor): `StratificationResult` gained `dTdz_T`; the pipeline
    helpers keep their 3-tuple returns and take an optional `diag` dict.
    (Section 10 added `dTdz_rho`.)
11. **Nits**: the "in-situ temperature" comments fixed in all four places; the
    N2 "Thorpe-sorted" labels in the schema and the profile plot made neutral;
    ∂CT/∂SP range corrected; the temperature sort keys on median-salinity CT.
12. **Fleet null test design** (conjecture): the JAC is unpumped, so "good CTD"
    is an assumption; the VMP-250 fleet doesn't sample deep weak
    stratification. Noted; the synthetic tests include a positive control.

### End-to-end run on the VMP-5500 casts (perturb, worktree code)

Both files through `perturb run` twice: defaults (A) and `sort: temperature`,
`dTdz_source: fp07` (B). Both rc = 0, no FP07 fallback in B.

| 150-250 dbar, median | A dTdz | B dTdz | A K_T | B K_T | A Γ | B Γ |
|---|---|---|---|---|---|---|
| cast A chi windows | -0.026 | -0.051 | 1.6e-5 | 5.1e-6 | 2.2 | 2.4 |
| cast B chi windows | -0.016 | -0.051 | 6.8e-6 | 4.1e-6 | 0.79 | 1.15 |

(Final code, after the code review; re-run 2026-10-15. Defaults identical to
origin/main in all 22 compared product arrays.)

- κ (A, chi windows): 0.58 and 0.48 over the casts; worst bands -0.27 at
  1000-1050 dbar and -0.82 at 850-900 dbar. The run summary named both files.
- B's median dT/dz is -0.05, not the -0.065 to -0.070 section 5 expected. That
  expectation was wrong: in a staircase the 2-dbar FP07 gradients are near zero
  in the layers (p90 ≈ 0) and steep at the interfaces, so the **median** sits
  below the **mean**. The FP07 window mean is -0.066 / -0.062 against the
  100-dbar fits of -0.069 / -0.059. The temperature-sorted SBT1 has a mean
  19-25% *above* the fit and a p90 of -0.03 / -0.017: it smears the steps, and
  sorting its noise inflates weak gradients. So for an SBE3-class CT the
  remedy is `dTdz_source: fp07`, not `sort: temperature` alone.
- N² in B still carries the salinity (that is what the warning says), so Γ is
  not fixed; it needs the θ-S follow-up or a hotel salinity.
- The in-pipeline FP07 calibration kept **both** beads on their embedded
  (nominal) coefficients in both files: "no profile segment reached the lag
  correlation gate (|r| >= 0.5)" against SBT1. That is what the earlier
  measurement predicts (high-passed above 0.1 Hz, T1 and SBT1 correlate at only
  r = 0.3-0.5 at the best lag). So B's dT/dz is on nominal gains (1.00 and
  0.97 against SBT1), and its K_T and Γ are gain-immune by construction; the
  advice to check for "vs SBT1" in the calibration attribute has to
  change (the standalone fp07-cal gates on peak sharpness instead and may fit).

### Code review (2026-10-15) and what changed

A second reviewer read the implementation and ran it (default path verified
bit-identical to cc5107d on 12 ASTRAL casts and end to end on SN479_0006).
Findings and dispositions:

1. **κ fired on clean data** (major, verified): votes were gated on the
   temperature-*sorted* slope, which sorting noise inflates (ASTRAL mixed-layer
   windows: unsorted 4.4e-4, sorted 5.2e-3 K/m), so mixed layers voted, and the
   worst-band rule then warned on 37/40 ASTRAL casts. **Fixed**: a window votes
   only where its measured-order gradient is ≥ 1e-3 K/m (the prototype's gate),
   and the worst band warns only below 0.6 (the cast below 0.8). Re-calibrated
   on stored products with the implementation, chi windows, 30 casts each:
   ASTRAL 0, CASPER-West 1, ARCTERX-2023 Wake 3, ARCTERX-2025 Interior 13,
   Taiwan17 20 warnings; the cast-wide κ (0.95, 0.93, 0.97, 0.83, 0.80) tracks
   the stored-vs-temperature-sorted dT/dz ratio (0.92, 0.93, 0.97, 0.70, 0.72),
   reading high on the two 2-s-window campaigns. Null test added (S = f(T), mixed
   layer, sensor noise: no warning in 10/10).
2. **Fast-channel rails gated after the box-car** (major, verified): a 58 C
   sample smeared into an in-range spike. **Fixed**: gate first. Test added.
3. **Temperature sort inflates noise-dominated weak gradients** (major,
   opt-in): **fixed**: the measured order is kept where the fitted change across
   the window is under 2x the MAD scatter about the five-window fit. Test added.
4. Ungated reference in the profile-product alive check: **fixed**.
5. A bad `window` value escaped the try: **fixed** (back inside it).
6. `sort_kappa` placement: on `N2` everywhere now (perturb and rsi-tpw, with the
   worst-band attributes); config template corrected.
7. Bit-identity regression test: **added** (stored cc5107d result on
   SN479_0006, rtol 1e-10), plus a `process_file` end-to-end test on the same
   file for both the defaults and the new options.
8-9. Nits: "pumped" corrected; warning names the band only when the band
   triggered; negative pressures go to the top band; odd centered box-car,
   clamped to the series length; the profile-path provenance tag is short; the
   comments say "as measured where the trend is unresolved".
10. Cost: the default path no longer computes the temperature order with a
    fixed salinity (κ is undefined there). (Superseded by section 10: every
    window now computes both orders; the cost is one argsort and one
    `gsw.CT_from_t` per window.)

Known edge (pre-existing, not changed): `_inject_seawater_properties` writes
pf.channels["SP"] (and SA/CT/sigma0/rho), so a hotel channel *named* SP is
overwritten by the JAC-derived value before stratification reads it. Diss/chi
already read the overwritten one on main; the profile product now agrees with
them. Follow-up: refuse or rename a hotel channel that collides with those names.

## 10. Default flipped to the temperature sort (2026-10-16)

Pat's decision, on section 1b's evidence: "T is typically a more stable
observation." What changed with it:

- **Library**: `processing.mixing.DEFAULT_SORT = "temperature"`, the default of
  `sorted_stratification`, `profile_stratification(_full)` and `kappa_warning`.
  perturb's `DEFAULTS["stratification"]["sort"]` matches it (a test ties them).
  rsi-tpw's `run_pipeline` and `mixing_look` follow the library default; the
  rsi-tpw N2/dTdz comments now say temperature-sorted.
- **Canonical form**: products made before the flip are density-sorted, so
  the canonical form drops `sort` only when it is `density` (main's form), not
  when it is the default. A config that leaves `sort` unset or null resolves to
  `temperature` and gets a new signature; `sort: density` keeps main's. Tests
  pin both. This governs only runs with a pinned
  `ODAS_TPW_ENGINE_FINGERPRINT` and plot-time stage resolution: otherwise the
  engine fingerprint changes with this code, so every config is recomputed on
  the next run anyway (the review below verified main's and this branch's diss
  hashes differ for `sort: density`). First written as if the canonical form
  alone forced the recompute; corrected.
- **κ under every sort** (reverses section 3C's "only under density"). Flipping
  the default with κ density-only would have dropped the diagnostic from every
  default product, and with it the third change Pat approved. `_stable_window`
  now returns the density-order and temperature-order slopes of the per-sample
  CT (`dTdz_rho`, `dTdz_T`) whenever the salinity is per-sample, whatever the
  sort; κ is identical under all three sorts (a test pins it), and under density
  `dTdz_rho` is the product's own dT/dz, so density-path κ is unchanged.
  `StratificationResult` and `ProfileStratification` gained `dTdz_rho`.
- **Warning text follows the sort**: under density as before ("dT/dz is biased
  low and N2 high"); under temperature/none, "a density-sorted dT/dz would be
  ...; dT/dz here is temperature-sorted, so the salinity does not reorder it,
  but N2 carries the salinity, and Gamma and K_rho with it". Same for the run
  summary line.
- **Behavior a density sort hid**: a column warm-under-cold beyond five windows
  (the whole fit span), with a constant salinity, stays unstable under the
  temperature sort (N² < 0, which the Osborn step masks); the density sort
  flipped it stable. A salinity-stabilized inversion is kept with its own sign
  and N² > 0 under both. Tests document both.
- **Pinned to density**: `perturb/plot/gamma_scaling.py` (its Ri_g N², so the
  figure keeps its earlier numbers; see the flip review, item 6), and every test that is about the Thorpe sort itself,
  including the stored-main bit-identity test (now `TestDensityPathIdentity`,
  which also checks the default differs from it on SN479).
- **Expected effect on existing campaigns** (section 1b's fleet numbers, not
  re-run on the implementation): chi-window dT/dz steeper by up to 1/0.70, K_T
  lower by ~2x on the 2-s-window campaigns (ARCTERX-2025 Interior, Taiwan17)
  and by 1.06-1.18x on the 4-s ones; profile N² from 1.12-1.48x the unsorted
  value to 0.95-1.08x. Products change only when a campaign is rerun.

### Review of the flip (2026-10-16) and what changed

One reviewer, reading and running the flip (density path bit-equal to main in
~150 module-level cases and end to end through `process_file`; κ identical
across sorts end to end; no stale 3-tuple callers). Findings:

1. **Negative N² under the new default** (major, verified independently): on
   SN479_0006, 23 of 263 2-dbar profile windows have N² ≤ 0 under the
   temperature sort (density 0, unsorted 34), throughout the thermocline (one
   at 145 dbar with dT/dz = -0.15 K/m). Invariance test (window size): at
   60-400 dbar the **mean** 2-dbar N² under the temperature sort is within 10%
   of a 10- and 20-dbar unsorted linear fit of potential density and of a
   temperature-only N² (1.44e-4 vs 1.33-1.49e-4; 1.74e-4 vs 1.68-1.78e-4;
   1.65e-4 vs 1.56-1.84e-4), while the density sort's mean is ~1.3x high
   (1.92e-4, 2.25e-4, 2.20e-4); medians sit lower than means for both
   (skewed per-window distribution, as in section 9's staircase). On five raw
   SN479 casts read without conductivity alignment, the density sort's median
   N² ran 10-47x the large-window fit and the temperature sort's tracked it.
   So the negatives are the noise tail of an estimate that is unbiased in the
   mean, which is the trade the flip makes. **Disposition**: documented
   (configuration reference, N2 comments in perturb and rsi-tpw);
   `mixing_look`'s "removing overturns" wording removed. Chi windows on the
   same cast: 1 of 99 negative. (n = one cast with aligned salinity; spread
   by depth band as quoted.)
2. **Cache claim** (minor): corrected above (engine fingerprint).
3. **A NaN in a per-sample salinity drops the sample from dT/dz too**
   (minor, pre-existing on main, verified by the reviewer: SP NaN over 50-80
   dbar removed every window there). The temperature-sort dT/dz needs only
   the window's median salinity. **Follow-up**: two masks (T/P for dT/dz,
   T/P/S for N² and κ).
4. **Warning text keyed on `sort`, not the dT/dz used** (minor): with
   `sort: density` and `dTdz_source: fp07` it called the (temperature-sorted)
   FP07 gradient biased and advised fp07. **Fixed**: the warnings and the run
   summary use the order the written dT/dz used; test added. The circular
   "use a fixed or hotel salinity" (circular under `salinity: hotel`) now
   reads "a fixed salinity or one from a T-S relation".
5. `stratification.enable: false` still hashes `sort` (minor): not changed;
   the section's other keys already behave that way, and only pinned-engine
   runs and plot-time resolution see it.
6. `gamma_scaling`'s pin comment claimed it reproduced the paper (nit): the
   paper's N² is temperature-only. **Fixed** the comment: the pin keeps the
   figure's earlier numbers.
7. "(none order)" in the run summary (nit): **fixed**.
8. The bit-identity fixture was untracked (nit): added in the commit.
9. Stale design text (nit): annotated in sections 3A and 9.
10. **Pre-existing hazard, not changed here**: `perturb bin` bins every
    `profiles_NN`/`diss_NN`/`chi_NN` directory into the one binned directory
    resolved from the current config, so after a rerun the old and new
    versions are binned in turn (the last, highest-numbered, wins) and their
    logs mix. **Follow-up** issue.
